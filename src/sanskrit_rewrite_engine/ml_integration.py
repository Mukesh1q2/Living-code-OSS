"""
Machine Learning integration hooks for the Sanskrit Rewrite Engine.

This module provides interfaces and utilities for integrating machine learning
components into the Sanskrit processing pipeline, preparing for future
enhancements with neural models and advanced linguistic analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

from .interfaces import (
    MLIntegrationInterface, ProcessingContext, ProcessingStage,
    LinguisticFeature, AdvancedToken, AdvancedRule
)


logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """Types of ML models that can be integrated."""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CNN = "cnn"
    CLASSICAL = "classical"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class MLTask(Enum):
    """ML tasks for Sanskrit processing."""
    TOKENIZATION = "tokenization"
    POS_TAGGING = "pos_tagging"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"
    DEPENDENCY_PARSING = "dependency_parsing"
    SEMANTIC_ROLE_LABELING = "semantic_role_labeling"
    SANDHI_SPLITTING = "sandhi_splitting"
    COMPOUND_ANALYSIS = "compound_analysis"
    METER_ANALYSIS = "meter_analysis"
    TRANSLATION = "translation"


@dataclass
class MLPrediction:
    """Represents a prediction from an ML model."""
    prediction: Any
    confidence: float
    model_name: str
    task: MLTask
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None


@dataclass
class MLModelInfo:
    """Information about an ML model."""
    name: str
    version: str
    model_type: MLModelType
    supported_tasks: List[MLTask]
    input_format: str
    output_format: str
    description: str = ""
    author: str = ""
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


class MLModelAdapter(ABC):
    """Abstract adapter for ML models."""
    
    @abstractmethod
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """Load the ML model."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, context: ProcessingContext) -> MLPrediction:
        """Make a prediction."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> MLModelInfo:
        """Get model information."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass


class SanskritMLIntegration(MLIntegrationInterface):
    """Main ML integration class for Sanskrit processing."""
    
    def __init__(self):
        self._models: Dict[str, MLModelAdapter] = {}
        self._task_models: Dict[MLTask, List[str]] = {}
        self._default_models: Dict[MLTask, str] = {}
        self._feature_extractors: Dict[str, Callable] = {}
        self._prediction_cache: Dict[str, MLPrediction] = {}
        self._cache_enabled = True
        self._cache_size_limit = 1000
    
    def register_model(self, name: str, adapter: MLModelAdapter, 
                      tasks: List[MLTask], is_default: bool = False) -> None:
        """Register an ML model adapter.
        
        Args:
            name: Name of the model
            adapter: Model adapter instance
            tasks: Tasks this model can perform
            is_default: Whether this is the default model for these tasks
        """
        self._models[name] = adapter
        
        for task in tasks:
            if task not in self._task_models:
                self._task_models[task] = []
            self._task_models[task].append(name)
            
            if is_default or task not in self._default_models:
                self._default_models[task] = name
        
        logger.info(f"Registered ML model: {name} for tasks: {[t.value for t in tasks]}")
    
    def unregister_model(self, name: str) -> None:
        """Unregister an ML model.
        
        Args:
            name: Name of the model to unregister
        """
        if name in self._models:
            # Unload the model
            self._models[name].unload_model()
            del self._models[name]
            
            # Remove from task mappings
            for task, models in self._task_models.items():
                if name in models:
                    models.remove(name)
            
            # Update default models
            for task, default_model in list(self._default_models.items()):
                if default_model == name:
                    # Find another model for this task
                    if self._task_models.get(task):
                        self._default_models[task] = self._task_models[task][0]
                    else:
                        del self._default_models[task]
            
            logger.info(f"Unregistered ML model: {name}")
    
    def predict(self, input_data: Any, context: ProcessingContext) -> Any:
        """Make predictions using appropriate ML models.
        
        Args:
            input_data: Input data for prediction
            context: Processing context
            
        Returns:
            Prediction results
        """
        # Determine task from context
        task = self._infer_task_from_context(context)
        if not task:
            logger.warning("Could not infer ML task from context")
            return input_data
        
        # Get appropriate model
        model_name = self._get_model_for_task(task)
        if not model_name:
            logger.warning(f"No model available for task: {task.value}")
            return input_data
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key(input_data, model_name, context)
            if cache_key in self._prediction_cache:
                logger.debug(f"Cache hit for prediction: {cache_key}")
                return self._prediction_cache[cache_key]
        
        # Make prediction
        try:
            model = self._models[model_name]
            prediction = model.predict(input_data, context)
            
            # Cache result
            if self._cache_enabled:
                self._cache_prediction(cache_key, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"ML prediction failed with model {model_name}: {e}")
            return input_data
    
    def train(self, training_data: List[Tuple[Any, Any]]) -> None:
        """Train models (if they support training).
        
        Args:
            training_data: List of (input, expected_output) tuples
        """
        logger.info("Training not implemented for current models")
        # This would be implemented for trainable models
        pass
    
    def get_confidence(self, prediction: Any) -> float:
        """Get confidence score for a prediction.
        
        Args:
            prediction: Prediction object
            
        Returns:
            Confidence score between 0 and 1
        """
        if isinstance(prediction, MLPrediction):
            return prediction.confidence
        elif isinstance(prediction, dict) and 'confidence' in prediction:
            return prediction['confidence']
        else:
            return 1.0  # Default confidence
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all registered models.
        
        Returns:
            Dictionary of model information
        """
        model_info = {}
        for name, adapter in self._models.items():
            try:
                model_info[name] = adapter.get_model_info()
            except Exception as e:
                logger.error(f"Failed to get info for model {name}: {e}")
                model_info[name] = {"error": str(e)}
        
        return {
            "registered_models": model_info,
            "task_mappings": {task.value: models for task, models in self._task_models.items()},
            "default_models": {task.value: model for task, model in self._default_models.items()},
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._prediction_cache)
        }
    
    def register_feature_extractor(self, name: str, extractor: Callable) -> None:
        """Register a feature extractor function.
        
        Args:
            name: Name of the feature extractor
            extractor: Function that extracts features from input
        """
        self._feature_extractors[name] = extractor
        logger.info(f"Registered feature extractor: {name}")
    
    def extract_features(self, input_data: Any, extractor_names: List[str]) -> Dict[str, Any]:
        """Extract features using registered extractors.
        
        Args:
            input_data: Input data
            extractor_names: Names of extractors to use
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        for name in extractor_names:
            if name in self._feature_extractors:
                try:
                    features[name] = self._feature_extractors[name](input_data)
                except Exception as e:
                    logger.error(f"Feature extraction failed for {name}: {e}")
                    features[name] = None
        
        return features
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable prediction caching.
        
        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
        if not enabled:
            self._prediction_cache.clear()
    
    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._prediction_cache.clear()
        logger.info("Cleared ML prediction cache")
    
    def _infer_task_from_context(self, context: ProcessingContext) -> Optional[MLTask]:
        """Infer ML task from processing context.
        
        Args:
            context: Processing context
            
        Returns:
            Inferred ML task or None
        """
        stage_task_map = {
            ProcessingStage.TOKENIZATION: MLTask.TOKENIZATION,
            ProcessingStage.MORPHOLOGICAL_ANALYSIS: MLTask.MORPHOLOGICAL_ANALYSIS,
            ProcessingStage.SYNTACTIC_ANALYSIS: MLTask.DEPENDENCY_PARSING,
            ProcessingStage.SEMANTIC_ANALYSIS: MLTask.SEMANTIC_ROLE_LABELING
        }
        
        return stage_task_map.get(context.stage)
    
    def _get_model_for_task(self, task: MLTask) -> Optional[str]:
        """Get the best model for a task.
        
        Args:
            task: ML task
            
        Returns:
            Model name or None
        """
        # Return default model for task
        return self._default_models.get(task)
    
    def _generate_cache_key(self, input_data: Any, model_name: str, context: ProcessingContext) -> str:
        """Generate cache key for prediction.
        
        Args:
            input_data: Input data
            model_name: Name of the model
            context: Processing context
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a string representation of the input
        input_str = str(input_data)
        context_str = f"{context.stage.value}_{context.analysis_level.value}"
        
        # Generate hash
        key_str = f"{model_name}_{input_str}_{context_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_prediction(self, cache_key: str, prediction: MLPrediction) -> None:
        """Cache a prediction result.
        
        Args:
            cache_key: Cache key
            prediction: Prediction to cache
        """
        # Implement LRU-style cache management
        if len(self._prediction_cache) >= self._cache_size_limit:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        
        self._prediction_cache[cache_key] = prediction


# Concrete ML model adapters for common scenarios

class DummyMLAdapter(MLModelAdapter):
    """Dummy ML adapter for testing and demonstration."""
    
    def __init__(self, name: str, tasks: List[MLTask]):
        self.name = name
        self.tasks = tasks
        self._loaded = False
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """Load dummy model."""
        self._loaded = True
        logger.info(f"Loaded dummy model: {self.name}")
    
    def predict(self, input_data: Any, context: ProcessingContext) -> MLPrediction:
        """Make dummy prediction."""
        return MLPrediction(
            prediction=f"dummy_prediction_for_{input_data}",
            confidence=0.85,
            model_name=self.name,
            task=self.tasks[0] if self.tasks else MLTask.TOKENIZATION,
            features_used=["dummy_feature"],
            metadata={"dummy": True}
        )
    
    def get_model_info(self) -> MLModelInfo:
        """Get dummy model info."""
        return MLModelInfo(
            name=self.name,
            version="1.0.0",
            model_type=MLModelType.CUSTOM,
            supported_tasks=self.tasks,
            input_format="text",
            output_format="prediction",
            description="Dummy model for testing"
        )
    
    def is_loaded(self) -> bool:
        """Check if loaded."""
        return self._loaded
    
    def unload_model(self) -> None:
        """Unload dummy model."""
        self._loaded = False
        logger.info(f"Unloaded dummy model: {self.name}")


# Feature extractors for Sanskrit text

def extract_character_features(text: str) -> Dict[str, Any]:
    """Extract character-level features from Sanskrit text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of character features
    """
    if not text:
        return {}
    
    # Basic character statistics
    features = {
        'length': len(text),
        'unique_chars': len(set(text)),
        'char_diversity': len(set(text)) / len(text) if text else 0
    }
    
    # Sanskrit-specific character analysis
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    
    features.update({
        'devanagari_ratio': devanagari_chars / len(text) if text else 0,
        'latin_ratio': latin_chars / len(text) if text else 0,
        'has_diacritics': any(c in 'āīūṛṝḷḹṅñṭḍṇśṣṃḥ' for c in text)
    })
    
    return features


def extract_morphological_features(token: Any) -> Dict[str, Any]:
    """Extract morphological features from a token.
    
    Args:
        token: Input token
        
    Returns:
        Dictionary of morphological features
    """
    features = {}
    
    if hasattr(token, 'text'):
        text = token.text
        features.update({
            'token_length': len(text),
            'starts_with_vowel': text[0] in 'aeiouāīūṛṝḷḹeo' if text else False,
            'ends_with_consonant': text[-1] not in 'aeiouāīūṛṝḷḹeo' if text else False
        })
    
    if hasattr(token, 'morphological_features'):
        features.update(token.morphological_features)
    
    return features


def extract_phonological_features(text: str) -> Dict[str, Any]:
    """Extract phonological features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of phonological features
    """
    if not text:
        return {}
    
    # Vowel and consonant analysis
    vowels = 'aeiouāīūṛṝḷḹeoai'
    consonants = 'kgcjṭḍtdpbmnrlvśṣsh'
    
    vowel_count = sum(1 for c in text.lower() if c in vowels)
    consonant_count = sum(1 for c in text.lower() if c in consonants)
    
    features = {
        'vowel_count': vowel_count,
        'consonant_count': consonant_count,
        'vowel_consonant_ratio': vowel_count / consonant_count if consonant_count > 0 else 0,
        'syllable_estimate': max(vowel_count, 1)  # Rough syllable count
    }
    
    return features


# Utility functions for ML integration

def create_ml_integration() -> SanskritMLIntegration:
    """Create and configure ML integration with default models.
    
    Returns:
        Configured ML integration instance
    """
    ml_integration = SanskritMLIntegration()
    
    # Register dummy models for demonstration
    tokenizer_model = DummyMLAdapter("dummy_tokenizer", [MLTask.TOKENIZATION])
    ml_integration.register_model("dummy_tokenizer", tokenizer_model, [MLTask.TOKENIZATION], is_default=True)
    
    morphology_model = DummyMLAdapter("dummy_morphology", [MLTask.MORPHOLOGICAL_ANALYSIS])
    ml_integration.register_model("dummy_morphology", morphology_model, [MLTask.MORPHOLOGICAL_ANALYSIS], is_default=True)
    
    # Register feature extractors
    ml_integration.register_feature_extractor("character_features", extract_character_features)
    ml_integration.register_feature_extractor("morphological_features", extract_morphological_features)
    ml_integration.register_feature_extractor("phonological_features", extract_phonological_features)
    
    return ml_integration


def setup_ml_hooks(engine: Any) -> None:
    """Set up ML integration hooks in the engine.
    
    Args:
        engine: Sanskrit rewrite engine instance
    """
    # This would integrate ML components into the main engine
    # Implementation depends on engine architecture
    logger.info("ML hooks setup completed")