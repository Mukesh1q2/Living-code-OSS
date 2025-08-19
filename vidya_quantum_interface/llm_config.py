"""
Configuration for LLM Integration in Vidya Quantum Interface.

This module provides configuration management for Hugging Face models,
including model selection, device management, and fallback strategies.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

try:
    from .llm_integration import ModelConfig, ModelType
except ImportError:
    # Handle case where this is imported before llm_integration
    from enum import Enum
    from dataclasses import dataclass
    
    class ModelType(Enum):
        TEXT_GENERATION = "text-generation"
        TEXT_EMBEDDING = "text-embedding"
        QUESTION_ANSWERING = "question-answering"
        SENTIMENT_ANALYSIS = "sentiment-analysis"
        FEATURE_EXTRACTION = "feature-extraction"
    
    @dataclass
    class ModelConfig:
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

logger = logging.getLogger(__name__)


@dataclass
class LLMServiceConfig:
    """Configuration for the LLM integration service."""
    cache_dir: Optional[str] = None
    default_device: str = "auto"
    enable_fallbacks: bool = True
    max_memory_usage: int = 4096  # MB
    model_timeout: int = 300  # seconds
    enable_streaming: bool = True
    log_level: str = "INFO"


class LLMConfigManager:
    """
    Manager for LLM configuration and model definitions.
    
    Handles loading configuration from files, environment variables,
    and provides model recommendations based on system capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager."""
        self.config_path = Path(config_path) if config_path else Path("llm_config.json")
        self.service_config = LLMServiceConfig()
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Load configuration
        self._load_config()
        self._setup_default_models()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update service config
                service_config_data = config_data.get("service", {})
                for key, value in service_config_data.items():
                    if hasattr(self.service_config, key):
                        setattr(self.service_config, key, value)
                
                # Load model configs
                models_data = config_data.get("models", {})
                for name, model_data in models_data.items():
                    self.model_configs[name] = ModelConfig(
                        name=name,
                        model_id=model_data["model_id"],
                        model_type=ModelType(model_data["model_type"]),
                        device=model_data.get("device", "auto"),
                        max_length=model_data.get("max_length", 512),
                        temperature=model_data.get("temperature", 0.7),
                        top_p=model_data.get("top_p", 0.9),
                        do_sample=model_data.get("do_sample", True),
                        local_files_only=model_data.get("local_files_only", False),
                        trust_remote_code=model_data.get("trust_remote_code", False),
                        torch_dtype=model_data.get("torch_dtype", "auto")
                    )
                
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # Override with environment variables
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "VIDYA_LLM_CACHE_DIR": ("cache_dir", str),
            "VIDYA_LLM_DEVICE": ("default_device", str),
            "VIDYA_LLM_ENABLE_FALLBACKS": ("enable_fallbacks", lambda x: x.lower() == "true"),
            "VIDYA_LLM_MAX_MEMORY": ("max_memory_usage", int),
            "VIDYA_LLM_TIMEOUT": ("model_timeout", int),
            "VIDYA_LLM_LOG_LEVEL": ("log_level", str)
        }
        
        for env_var, (attr_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    setattr(self.service_config, attr_name, converted_value)
                    logger.info(f"Set {attr_name} from environment: {converted_value}")
                except Exception as e:
                    logger.warning(f"Failed to convert {env_var}={value}: {e}")
    
    def _setup_default_models(self):
        """Set up default model configurations if not already defined."""
        default_models = {
            "default": {
                "model_id": "microsoft/DialoGPT-small",
                "model_type": "text-generation",
                "max_length": 256,
                "temperature": 0.7,
                "description": "Small conversational model for general text generation"
            },
            "embeddings": {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "model_type": "text-embedding",
                "description": "Lightweight embedding model for semantic analysis"
            },
            "sanskrit-aware": {
                "model_id": "microsoft/DialoGPT-medium",
                "model_type": "text-generation",
                "max_length": 512,
                "temperature": 0.8,
                "description": "Medium-sized model with better context understanding"
            },
            "large-generation": {
                "model_id": "microsoft/DialoGPT-large",
                "model_type": "text-generation",
                "max_length": 1024,
                "temperature": 0.7,
                "description": "Large model for high-quality text generation (requires more memory)"
            },
            "embeddings-large": {
                "model_id": "sentence-transformers/all-mpnet-base-v2",
                "model_type": "text-embedding",
                "description": "High-quality embedding model (larger, slower)"
            }
        }
        
        for name, config_data in default_models.items():
            if name not in self.model_configs:
                self.model_configs[name] = ModelConfig(
                    name=name,
                    model_id=config_data["model_id"],
                    model_type=ModelType(config_data["model_type"]),
                    device=self.service_config.default_device,
                    max_length=config_data.get("max_length", 512),
                    temperature=config_data.get("temperature", 0.7),
                    top_p=0.9,
                    do_sample=True,
                    local_files_only=False,
                    trust_remote_code=False,
                    torch_dtype="auto"
                )
    
    def get_service_config(self) -> LLMServiceConfig:
        """Get the service configuration."""
        return self.service_config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.model_configs.get(model_name)
    
    def get_all_model_configs(self) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        return self.model_configs.copy()
    
    def add_model_config(self, config: ModelConfig):
        """Add a new model configuration."""
        self.model_configs[config.name] = config
        logger.info(f"Added model configuration: {config.name}")
    
    def remove_model_config(self, model_name: str) -> bool:
        """Remove a model configuration."""
        if model_name in self.model_configs:
            del self.model_configs[model_name]
            logger.info(f"Removed model configuration: {model_name}")
            return True
        return False
    
    def get_recommended_models(self, system_info: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get recommended models based on system capabilities.
        
        Args:
            system_info: Dictionary with system information (memory, GPU, etc.)
            
        Returns:
            List of recommended model names.
        """
        if system_info is None:
            system_info = self._detect_system_capabilities()
        
        recommended = []
        
        # Always recommend basic models
        recommended.extend(["default", "embeddings"])
        
        # Recommend based on available memory
        available_memory = system_info.get("memory_gb", 4)
        
        if available_memory >= 8:
            recommended.append("sanskrit-aware")
        
        if available_memory >= 16:
            recommended.extend(["large-generation", "embeddings-large"])
        
        # Filter to only include configured models
        return [name for name in recommended if name in self.model_configs]
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities for model recommendations."""
        capabilities = {
            "memory_gb": 4,  # Default assumption
            "has_gpu": False,
            "gpu_memory_gb": 0
        }
        
        try:
            import psutil
            # Get system memory in GB
            memory_bytes = psutil.virtual_memory().total
            capabilities["memory_gb"] = memory_bytes / (1024**3)
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                capabilities["has_gpu"] = True
                # Get GPU memory
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                capabilities["gpu_memory_gb"] = gpu_memory_bytes / (1024**3)
        except ImportError:
            pass
        
        return capabilities
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        config_data = {
            "service": asdict(self.service_config),
            "models": {
                name: {
                    "model_id": config.model_id,
                    "model_type": config.model_type.value,
                    "device": config.device,
                    "max_length": config.max_length,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "do_sample": config.do_sample,
                    "local_files_only": config.local_files_only,
                    "trust_remote_code": config.trust_remote_code,
                    "torch_dtype": config.torch_dtype
                }
                for name, config in self.model_configs.items()
            }
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation warnings/errors.
        """
        issues = []
        
        # Check service config
        if self.service_config.max_memory_usage < 1024:
            issues.append("Max memory usage is very low (< 1GB)")
        
        if self.service_config.model_timeout < 60:
            issues.append("Model timeout is very short (< 60s)")
        
        # Check model configs
        for name, config in self.model_configs.items():
            if not config.model_id:
                issues.append(f"Model '{name}' has no model_id")
            
            if config.max_length > 2048:
                issues.append(f"Model '{name}' has very high max_length ({config.max_length})")
            
            if config.temperature < 0 or config.temperature > 2:
                issues.append(f"Model '{name}' has unusual temperature ({config.temperature})")
        
        return issues


# Global configuration manager instance
_config_manager: Optional[LLMConfigManager] = None


def get_config_manager() -> LLMConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = LLMConfigManager()
    return _config_manager


def load_config(config_path: Optional[str] = None) -> LLMConfigManager:
    """Load configuration from a specific path."""
    global _config_manager
    _config_manager = LLMConfigManager(config_path)
    return _config_manager