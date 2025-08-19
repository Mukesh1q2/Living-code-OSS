"""
LLM Integration Service for Vidya Quantum Interface.

This module provides local Hugging Face model integration with fallback mechanisms,
model management, and text embedding generation for semantic analysis.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import os
import requests

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        pipeline, Pipeline
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None
    AutoModelForCausalLM = None
    pipeline = None
    Pipeline = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models supported by the LLM integration."""
    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"
    QUESTION_ANSWERING = "question-answering"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    FEATURE_EXTRACTION = "feature-extraction"


@dataclass
class ModelConfig:
    """Configuration for a Hugging Face model."""
    name: str
    model_id: str
    model_type: ModelType
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    local_files_only: bool = False
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    config: ModelConfig
    loaded: bool = False
    load_time: float = 0.0
    memory_usage: int = 0
    last_used: float = 0.0
    error_message: Optional[str] = None


@dataclass
class InferenceRequest:
    """Request for model inference."""
    text: str
    model_name: str = "default"
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    return_full_text: bool = False


@dataclass
class InferenceResponse:
    """Response from model inference."""
    success: bool
    text: Optional[str] = None
    embeddings: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    processing_time: float = 0.0
    model_used: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMIntegrationService:
    """
    Service for integrating Hugging Face models with local inference capabilities.
    
    Features:
    - Local model loading and management
    - Text generation and embedding
    - Fallback mechanisms for unavailable models
    - Memory management and model caching
    - Async inference support
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the LLM integration service."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vidya_llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelInfo] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.embedding_models: Dict[str, SentenceTransformer] = {}
        
        self.device = self._detect_device()
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        # OpenAI-compatible endpoint (e.g., LM Studio) support
        # If set, we will use this HTTP endpoint for generation/embeddings
        self.openai_base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("LMSTUDIO_BASE_URL")
            or None
        )
        # Optional API key (LM Studio usually accepts any token)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        # Model overrides for text and embedding
        self.openai_text_model = os.getenv("OPENAI_TEXT_MODEL", os.getenv("LMSTUDIO_TEXT_MODEL", "gpt-oss-20b"))
        self.openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-3-small"))
        
        # Default model configurations
        self.default_configs = self._get_default_model_configs()
        
        logger.info(f"LLM Integration Service initialized")
        logger.info(f"Transformers available: {self.transformers_available}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")
        if self.openai_base_url:
            logger.info(f"OpenAI-compatible endpoint enabled: {self.openai_base_url}")
    
    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        if not TRANSFORMERS_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_default_model_configs(self) -> Dict[str, ModelConfig]:
        """Get default model configurations."""
        return {
            "default": ModelConfig(
                name="default",
                model_id="microsoft/DialoGPT-small",
                model_type=ModelType.TEXT_GENERATION,
                device=self.device,
                max_length=256,
                local_files_only=False
            ),
            "embeddings": ModelConfig(
                name="embeddings",
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                model_type=ModelType.TEXT_EMBEDDING,
                device=self.device,
                local_files_only=False
            ),
            "sanskrit-aware": ModelConfig(
                name="sanskrit-aware",
                model_id="microsoft/DialoGPT-medium",
                model_type=ModelType.TEXT_GENERATION,
                device=self.device,
                max_length=512,
                temperature=0.8,
                local_files_only=False
            )
        }
    
    async def initialize_models(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Initialize specified models or all default models.
        
        Args:
            model_names: List of model names to initialize. If None, initializes default models.
            
        Returns:
            Dictionary mapping model names to success status.
        """
        if not self.transformers_available:
            logger.warning("Transformers not available - models will use fallback mode")
            return {}
        
        if model_names is None:
            model_names = ["default", "embeddings"]
        
        results = {}
        
        for model_name in model_names:
            try:
                success = await self._load_model(model_name)
                results[model_name] = success
                logger.info(f"Model '{model_name}' initialization: {'success' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Failed to initialize model '{model_name}': {e}")
                results[model_name] = False
        
        return results
    
    async def _load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        if model_name in self.models and self.models[model_name].loaded:
            logger.info(f"Model '{model_name}' already loaded")
            return True
        
        config = self.default_configs.get(model_name)
        if not config:
            logger.error(f"No configuration found for model '{model_name}'")
            return False
        
        start_time = time.time()
        
        try:
            if config.model_type == ModelType.TEXT_EMBEDDING:
                # Load embedding model
                model = SentenceTransformer(
                    config.model_id,
                    cache_folder=str(self.cache_dir),
                    device=config.device
                )
                self.embedding_models[model_name] = model
                
            else:
                # Load text generation or other pipeline models
                pipe = pipeline(
                    config.model_type.value,
                    model=config.model_id,
                    tokenizer=config.model_id,
                    device=0 if config.device == "cuda" else -1,
                    model_kwargs={
                        "cache_dir": str(self.cache_dir),
                        "local_files_only": config.local_files_only,
                        "trust_remote_code": config.trust_remote_code,
                        "torch_dtype": getattr(torch, config.torch_dtype) if config.torch_dtype != "auto" else "auto"
                    }
                )
                self.pipelines[model_name] = pipe
            
            load_time = time.time() - start_time
            
            # Update model info
            self.models[model_name] = ModelInfo(
                config=config,
                loaded=True,
                load_time=load_time,
                last_used=time.time()
            )
            
            logger.info(f"Successfully loaded model '{model_name}' in {load_time:.2f}s")
            return True
            
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = f"Failed to load model '{model_name}': {e}"
            
            self.models[model_name] = ModelInfo(
                config=config,
                loaded=False,
                load_time=load_time,
                error_message=error_msg
            )
            
            logger.error(error_msg)
            return False
    
    async def generate_text(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using a language model.
        
        Args:
            request: Inference request with text and parameters.
            
        Returns:
            Inference response with generated text or error.
        """
        start_time = time.time()
        
        # If an OpenAI-compatible endpoint is configured, use it first
        if self.openai_base_url:
            try:
                generated_text = self._openai_chat_completion(request)
                processing_time = time.time() - start_time
                return InferenceResponse(
                    success=True,
                    text=generated_text,
                    processing_time=processing_time,
                    model_used=request.model_name or "openai-compatible",
                    metadata={"backend": "openai-compatible", "base_url": self.openai_base_url}
                )
            except Exception as e:
                logger.error(f"OpenAI-compatible generation failed: {e}")
                # fall through to transformers/fallback
        
        # Check if transformers are available
        if not self.transformers_available:
            return self._fallback_text_generation(request, start_time)
        
        # Ensure model is loaded
        if request.model_name not in self.models or not self.models[request.model_name].loaded:
            success = await self._load_model(request.model_name)
            if not success:
                return self._fallback_text_generation(request, start_time)
        
        try:
            pipe = self.pipelines.get(request.model_name)
            if not pipe:
                return self._fallback_text_generation(request, start_time)
            
            # Prepare generation parameters
            config = self.models[request.model_name].config
            generation_params = {
                "max_length": request.max_length or config.max_length,
                "temperature": request.temperature or config.temperature,
                "top_p": request.top_p or config.top_p,
                "do_sample": request.do_sample if request.do_sample is not None else config.do_sample,
                "return_full_text": request.return_full_text,
                "pad_token_id": pipe.tokenizer.eos_token_id
            }
            
            # Generate text
            result = pipe(request.text, **generation_params)
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = str(result)
            
            # Update model usage
            self.models[request.model_name].last_used = time.time()
            
            processing_time = time.time() - start_time
            
            return InferenceResponse(
                success=True,
                text=generated_text,
                processing_time=processing_time,
                model_used=request.model_name,
                metadata={
                    "generation_params": generation_params,
                    "input_length": len(request.text),
                    "output_length": len(generated_text),
                    "backend": "transformers"
                }
            )
            
        except Exception as e:
            logger.error(f"Text generation failed for model '{request.model_name}': {e}")
            return self._fallback_text_generation(request, start_time)
    
    async def generate_embeddings(self, text: str, model_name: str = "embeddings") -> InferenceResponse:
        """
        Generate text embeddings for semantic analysis.
        
        Args:
            text: Input text to embed.
            model_name: Name of the embedding model to use.
            
        Returns:
            Inference response with embeddings or error.
        """
        start_time = time.time()
        
        # If an OpenAI-compatible endpoint is configured, use it first
        if self.openai_base_url:
            try:
                vec = self._openai_embeddings(text)
                processing_time = time.time() - start_time
                return InferenceResponse(
                    success=True,
                    embeddings=vec,
                    processing_time=processing_time,
                    model_used=model_name or "openai-compatible",
                    metadata={"backend": "openai-compatible", "base_url": self.openai_base_url, "embedding_dimension": len(vec)}
                )
            except Exception as e:
                logger.error(f"OpenAI-compatible embeddings failed: {e}")
                # fall through to transformers/fallback
        
        # Check if transformers are available
        if not self.transformers_available:
            return self._fallback_embeddings(text, model_name, start_time)
        
        # Ensure model is loaded
        if model_name not in self.models or not self.models[model_name].loaded:
            success = await self._load_model(model_name)
            if not success:
                return self._fallback_embeddings(text, model_name, start_time)
        
        try:
            embedding_model = self.embedding_models.get(model_name)
            if not embedding_model:
                return self._fallback_embeddings(text, model_name, start_time)
            
            # Generate embeddings
            embeddings = embedding_model.encode([text])
            embedding_vector = embeddings[0].tolist()
            
            # Update model usage
            self.models[model_name].last_used = time.time()
            
            processing_time = time.time() - start_time
            
            return InferenceResponse(
                success=True,
                embeddings=embedding_vector,
                processing_time=processing_time,
                model_used=model_name,
                metadata={
                    "embedding_dimension": len(embedding_vector),
                    "input_length": len(text),
                    "backend": "transformers"
                }
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed for model '{model_name}': {e}")
            return self._fallback_embeddings(text, model_name, start_time)
    
    async def stream_text_generation(self, request: InferenceRequest) -> AsyncIterator[str]:
        """
        Stream text generation for real-time responses.
        
        Args:
            request: Inference request with text and parameters.
            
        Yields:
            Generated text chunks.
        """
        # If OpenAI-compatible endpoint is available, emulate streaming by chunking the full response
        if self.openai_base_url:
            try:
                resp = await self.generate_text(request)
                if resp.success and resp.text:
                    text = resp.text
                    # yield in ~20 char chunks
                    chunk_size = 20
                    for i in range(0, len(text), chunk_size):
                        yield text[i:i+chunk_size]
                    return
            except Exception as e:
                logger.error(f"OpenAI-compatible streaming failed: {e}")
        
        # For now, implement as a simple async wrapper around batch generation
        # In a full implementation, this would use streaming generation
        response = await self.generate_text(request)
        
        if response.success and response.text:
            # Simulate streaming by yielding chunks
            text = response.text
            chunk_size = max(1, len(text) // 10)  # 10 chunks
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.1)  # Simulate streaming delay
        else:
            yield f"Error: {response.error_message or 'Text generation failed'}"
    
    def _fallback_text_generation(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Fallback text generation when models are unavailable."""
        processing_time = time.time() - start_time
        
        # Simple rule-based fallback
        fallback_responses = [
            f"I understand you're asking about: {request.text[:50]}...",
            f"Based on your input '{request.text[:30]}...', I can provide some insights.",
            f"Your query about '{request.text[:40]}...' is interesting. Let me help.",
            "I'm processing your request using fallback mode. Full AI models are not available."
        ]
        
        # Choose response based on input length
        response_idx = min(len(request.text) // 20, len(fallback_responses) - 1)
        fallback_text = fallback_responses[response_idx]
        
        return InferenceResponse(
            success=True,
            text=fallback_text,
            processing_time=processing_time,
            model_used="fallback",
            metadata={
                "fallback_mode": True,
                "reason": "Transformers not available or model failed to load"
            }
        )
    
    def _fallback_embeddings(self, text: str, model_name: str, start_time: float) -> InferenceResponse:
        """Fallback embedding generation when models are unavailable."""
        processing_time = time.time() - start_time
        
        # Simple hash-based fallback embeddings
        import hashlib
        
        # Create a deterministic but varied embedding based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to 384-dimensional vector (similar to sentence transformers)
        embedding_dim = 384
        embeddings = []
        
        for i in range(embedding_dim):
            # Use different parts of the hash to create varied values
            hash_part = text_hash[(i * 2) % len(text_hash):(i * 2 + 2) % len(text_hash)]
            if len(hash_part) < 2:
                hash_part = text_hash[:2]
            
            # Convert to float between -1 and 1
            value = (int(hash_part, 16) / 255.0) * 2 - 1
            embeddings.append(value)
        
        return InferenceResponse(
            success=True,
            embeddings=embeddings,
            processing_time=processing_time,
            model_used="fallback",
            metadata={
                "fallback_mode": True,
                "embedding_dimension": embedding_dim,
                "reason": "Transformers not available or model failed to load"
            }
        )
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Args:
            model_name: Specific model name, or None for all models.
            
        Returns:
            Model information dictionary.
        """
        if model_name:
            if model_name in self.models:
                model_info = self.models[model_name]
                return {
                    "name": model_name,
                    "loaded": model_info.loaded,
                    "config": model_info.config.__dict__,
                    "load_time": model_info.load_time,
                    "last_used": model_info.last_used,
                    "error_message": model_info.error_message
                }
            else:
                return {"error": f"Model '{model_name}' not found"}
        
        # Return info for all models
        return {
            "transformers_available": self.transformers_available,
            "device": self.device,
            "cache_dir": str(self.cache_dir),
            "models": {
                name: {
                    "loaded": info.loaded,
                    "config": info.config.__dict__,
                    "load_time": info.load_time,
                    "last_used": info.last_used,
                    "error_message": info.error_message
                }
                for name, info in self.models.items()
            }
        }
    
    # -------------------- OpenAI-compatible helpers (LM Studio, etc.) --------------------
    def _openai_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }
    
    def _resolve_text_model(self, request: InferenceRequest) -> str:
        # Allow per-request model override via config mapping if present
        # Otherwise use env override, then default
        config = self.default_configs.get(request.model_name)
        if config and config.model_id:
            return os.getenv("OPENAI_MODEL_" + request.model_name.upper(), self.openai_text_model)
        return self.openai_text_model
    
    def _openai_chat_completion(self, request: InferenceRequest) -> str:
        if not self.openai_base_url:
            raise RuntimeError("OpenAI-compatible base URL not configured")
        url = self.openai_base_url.rstrip("/") + "/chat/completions"
        model = self._resolve_text_model(request)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.text},
            ],
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 0.9,
            "max_tokens": request.max_length if request.max_length is not None else 256,
            "stream": False,
        }
        resp = requests.post(url, headers=self._openai_headers(), json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        # Fallback to raw text
        return json.dumps(data)
    
    def _openai_embeddings(self, text: str) -> List[float]:
        if not self.openai_base_url:
            raise RuntimeError("OpenAI-compatible base URL not configured")
        url = self.openai_base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": self.openai_embed_model,
            "input": text,
        }
        resp = requests.post(url, headers=self._openai_headers(), json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data[0].get("embedding") is not None:
            return data["data"][0]["embedding"]
        # If embeddings not available, raise to trigger fallback
        raise RuntimeError("No embeddings in response")
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model configurations."""
        return [
            {
                "name": name,
                "model_id": config.model_id,
                "type": config.model_type.value,
                "device": config.device,
                "loaded": name in self.models and self.models[name].loaded
            }
            for name, config in self.default_configs.items()
        ]
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free memory.
        
        Args:
            model_name: Name of the model to unload.
            
        Returns:
            True if successfully unloaded, False otherwise.
        """
        try:
            if model_name in self.pipelines:
                del self.pipelines[model_name]
            
            if model_name in self.embedding_models:
                del self.embedding_models[model_name]
            
            if model_name in self.models:
                self.models[model_name].loaded = False
            
            # Force garbage collection
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully unloaded model '{model_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model '{model_name}': {e}")
            return False
    
    async def cleanup(self):
        """Clean up all loaded models and free resources."""
        logger.info("Cleaning up LLM Integration Service")
        
        for model_name in list(self.models.keys()):
            await self.unload_model(model_name)
        
        self.models.clear()
        self.pipelines.clear()
        self.embedding_models.clear()
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LLM Integration Service cleanup complete")


# Global service instance
_llm_service: Optional[LLMIntegrationService] = None


def get_llm_service() -> LLMIntegrationService:
    """Get the global LLM integration service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMIntegrationService()
    return _llm_service


async def initialize_llm_service(model_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """Initialize the global LLM service with specified models."""
    service = get_llm_service()
    return await service.initialize_models(model_names)