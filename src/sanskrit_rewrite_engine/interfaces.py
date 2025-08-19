"""
Future architecture interfaces for the Sanskrit Rewrite Engine.

This module defines interfaces and abstract base classes that support future
enhancements including token-based processing, Pāṇini sūtra encoding,
plugin architecture, and machine learning integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable, 
    Protocol, runtime_checkable, TypeVar, Generic
)
import uuid
from datetime import datetime


# Type variables for generic interfaces
T = TypeVar('T')
R = TypeVar('R')


class ProcessingStage(Enum):
    """Stages in the Sanskrit processing pipeline."""
    TOKENIZATION = "tokenization"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"
    SYNTACTIC_ANALYSIS = "syntactic_analysis"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    RULE_APPLICATION = "rule_application"
    POST_PROCESSING = "post_processing"


class RuleType(Enum):
    """Types of transformation rules."""
    PHONOLOGICAL = "phonological"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    SANDHI = "sandhi"
    COMPOUND = "compound"
    SUTRA = "sutra"  # For future Pāṇini sūtra encoding


class TokenType(Enum):
    """Enhanced token types for future processing."""
    PHONEME = "phoneme"
    MORPHEME = "morpheme"
    SYLLABLE = "syllable"
    WORD = "word"
    COMPOUND = "compound"
    PHRASE = "phrase"
    CLAUSE = "clause"


class AnalysisLevel(Enum):
    """Levels of linguistic analysis."""
    SURFACE = "surface"
    PHONOLOGICAL = "phonological"
    MORPHOLOGICAL = "morphological"
    LEXICAL = "lexical"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"


# Core interfaces for future architecture

@dataclass
class LinguisticFeature:
    """Represents a linguistic feature with metadata."""
    name: str
    value: Any
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingContext:
    """Context information for processing operations."""
    stage: ProcessingStage
    analysis_level: AnalysisLevel
    features: Dict[str, LinguisticFeature] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@runtime_checkable
class TokenProcessor(Protocol):
    """Protocol for token-based processing components."""
    
    def process_token(self, token: Any, context: ProcessingContext) -> Any:
        """Process a single token with context."""
        ...
    
    def can_process(self, token: Any, context: ProcessingContext) -> bool:
        """Check if this processor can handle the given token."""
        ...


@runtime_checkable
class RuleProcessor(Protocol):
    """Protocol for rule processing components."""
    
    def apply_rule(self, rule: Any, target: Any, context: ProcessingContext) -> Any:
        """Apply a rule to a target with context."""
        ...
    
    def validate_rule(self, rule: Any) -> bool:
        """Validate that a rule is well-formed."""
        ...


class AdvancedToken(ABC):
    """Abstract base class for advanced token representations."""
    
    @property
    @abstractmethod
    def surface_form(self) -> str:
        """The surface representation of the token."""
        pass
    
    @property
    @abstractmethod
    def underlying_form(self) -> str:
        """The underlying/canonical form of the token."""
        pass
    
    @property
    @abstractmethod
    def features(self) -> Dict[str, LinguisticFeature]:
        """Linguistic features associated with the token."""
        pass
    
    @abstractmethod
    def get_feature(self, name: str) -> Optional[LinguisticFeature]:
        """Get a specific linguistic feature."""
        pass
    
    @abstractmethod
    def set_feature(self, name: str, feature: LinguisticFeature) -> None:
        """Set a linguistic feature."""
        pass


class AdvancedRule(ABC):
    """Abstract base class for advanced rule representations."""
    
    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for the rule."""
        pass
    
    @property
    @abstractmethod
    def rule_type(self) -> RuleType:
        """Type of the rule."""
        pass
    
    @property
    @abstractmethod
    def sutra_reference(self) -> Optional[str]:
        """Reference to Pāṇini sūtra (for future use)."""
        pass
    
    @abstractmethod
    def matches(self, target: Any, context: ProcessingContext) -> bool:
        """Check if rule matches the target in given context."""
        pass
    
    @abstractmethod
    def apply(self, target: Any, context: ProcessingContext) -> Any:
        """Apply the rule to the target."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get rule metadata including sūtra references."""
        pass


class ProcessingPipeline(ABC):
    """Abstract base class for processing pipelines."""
    
    @abstractmethod
    def add_stage(self, stage: ProcessingStage, processor: TokenProcessor) -> None:
        """Add a processing stage to the pipeline."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any, context: ProcessingContext) -> Any:
        """Process input through the pipeline."""
        pass
    
    @abstractmethod
    def get_stage_processors(self, stage: ProcessingStage) -> List[TokenProcessor]:
        """Get processors for a specific stage."""
        pass


class PluginInterface(ABC):
    """Interface for plugin components."""
    
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Name of the plugin."""
        pass
    
    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Version of the plugin."""
        pass
    
    @property
    @abstractmethod
    def supported_stages(self) -> List[ProcessingStage]:
        """Processing stages this plugin supports."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class MLIntegrationInterface(ABC):
    """Interface for machine learning integration."""
    
    @abstractmethod
    def predict(self, input_data: Any, context: ProcessingContext) -> Any:
        """Make predictions using ML model."""
        pass
    
    @abstractmethod
    def train(self, training_data: List[Tuple[Any, Any]]) -> None:
        """Train the ML model (if applicable)."""
        pass
    
    @abstractmethod
    def get_confidence(self, prediction: Any) -> float:
        """Get confidence score for a prediction."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ML model."""
        pass


# Concrete implementations for immediate use

@dataclass
class EnhancedToken(AdvancedToken):
    """Enhanced token implementation with linguistic features."""
    
    _surface_form: str
    _underlying_form: str
    _features: Dict[str, LinguisticFeature] = field(default_factory=dict)
    _token_type: TokenType = TokenType.WORD
    _position: int = 0
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def surface_form(self) -> str:
        return self._surface_form
    
    @property
    def underlying_form(self) -> str:
        return self._underlying_form
    
    @property
    def features(self) -> Dict[str, LinguisticFeature]:
        return self._features.copy()
    
    def get_feature(self, name: str) -> Optional[LinguisticFeature]:
        return self._features.get(name)
    
    def set_feature(self, name: str, feature: LinguisticFeature) -> None:
        self._features[name] = feature
    
    @property
    def token_type(self) -> TokenType:
        return self._token_type
    
    @property
    def position(self) -> int:
        return self._position


@dataclass
class SutraRule(AdvancedRule):
    """Rule implementation with sūtra encoding support."""
    
    _rule_id: str
    _rule_type: RuleType
    _pattern: str
    _replacement: str
    _sutra_reference: Optional[str] = None
    _conditions: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _priority: int = 1
    _enabled: bool = True
    
    @property
    def rule_id(self) -> str:
        return self._rule_id
    
    @property
    def rule_type(self) -> RuleType:
        return self._rule_type
    
    @property
    def sutra_reference(self) -> Optional[str]:
        return self._sutra_reference
    
    def matches(self, target: Any, context: ProcessingContext) -> bool:
        """Check if rule matches target in context."""
        # Basic pattern matching - can be enhanced
        if hasattr(target, 'surface_form'):
            import re
            return bool(re.search(self._pattern, target.surface_form))
        return False
    
    def apply(self, target: Any, context: ProcessingContext) -> Any:
        """Apply rule to target."""
        if hasattr(target, 'surface_form'):
            import re
            new_surface = re.sub(self._pattern, self._replacement, target.surface_form)
            if hasattr(target, '_surface_form'):
                target._surface_form = new_surface
        return target
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = self._metadata.copy()
        if self._sutra_reference:
            metadata['sutra_reference'] = self._sutra_reference
        metadata['conditions'] = self._conditions
        metadata['priority'] = self._priority
        metadata['enabled'] = self._enabled
        return metadata


class ExtensiblePipeline(ProcessingPipeline):
    """Extensible processing pipeline implementation."""
    
    def __init__(self):
        self._stages: Dict[ProcessingStage, List[TokenProcessor]] = {}
        self._plugins: List[PluginInterface] = []
        self._ml_components: List[MLIntegrationInterface] = []
    
    def add_stage(self, stage: ProcessingStage, processor: TokenProcessor) -> None:
        if stage not in self._stages:
            self._stages[stage] = []
        self._stages[stage].append(processor)
    
    def process(self, input_data: Any, context: ProcessingContext) -> Any:
        """Process input through all pipeline stages."""
        current_data = input_data
        
        # Process through each stage in order
        for stage in ProcessingStage:
            if stage in self._stages:
                context.stage = stage
                for processor in self._stages[stage]:
                    if processor.can_process(current_data, context):
                        current_data = processor.process_token(current_data, context)
        
        return current_data
    
    def get_stage_processors(self, stage: ProcessingStage) -> List[TokenProcessor]:
        return self._stages.get(stage, []).copy()
    
    def add_plugin(self, plugin: PluginInterface) -> None:
        """Add a plugin to the pipeline."""
        self._plugins.append(plugin)
        # Auto-register plugin processors for supported stages
        # This would be implemented based on plugin capabilities
    
    def add_ml_component(self, component: MLIntegrationInterface) -> None:
        """Add an ML component to the pipeline."""
        self._ml_components.append(component)


# Factory classes for creating advanced components

class TokenFactory:
    """Factory for creating enhanced tokens."""
    
    @staticmethod
    def create_token(surface_form: str, underlying_form: Optional[str] = None,
                    token_type: TokenType = TokenType.WORD) -> EnhancedToken:
        """Create an enhanced token."""
        return EnhancedToken(
            _surface_form=surface_form,
            _underlying_form=underlying_form or surface_form,
            _token_type=token_type
        )
    
    @staticmethod
    def from_basic_token(basic_token: Any) -> EnhancedToken:
        """Convert a basic token to enhanced token."""
        surface = getattr(basic_token, 'text', str(basic_token))
        return TokenFactory.create_token(surface)


class RuleFactory:
    """Factory for creating advanced rules."""
    
    @staticmethod
    def create_sutra_rule(rule_id: str, pattern: str, replacement: str,
                         rule_type: RuleType = RuleType.PHONOLOGICAL,
                         sutra_reference: Optional[str] = None) -> SutraRule:
        """Create a sūtra-based rule."""
        return SutraRule(
            _rule_id=rule_id,
            _rule_type=rule_type,
            _pattern=pattern,
            _replacement=replacement,
            _sutra_reference=sutra_reference
        )
    
    @staticmethod
    def from_basic_rule(basic_rule: Any) -> SutraRule:
        """Convert a basic rule to sūtra rule."""
        rule_id = getattr(basic_rule, 'id', 'unknown')
        pattern = getattr(basic_rule, 'pattern', '')
        replacement = getattr(basic_rule, 'replacement', '')
        
        # Determine rule type from metadata
        metadata = getattr(basic_rule, 'metadata', {})
        category = metadata.get('category', 'phonological')
        
        rule_type_map = {
            'vowel_sandhi': RuleType.SANDHI,
            'consonant_sandhi': RuleType.SANDHI,
            'compound_formation': RuleType.COMPOUND,
            'morphological': RuleType.MORPHOLOGICAL,
            'phonological': RuleType.PHONOLOGICAL
        }
        
        rule_type = rule_type_map.get(category, RuleType.PHONOLOGICAL)
        sutra_ref = metadata.get('sutra_ref')
        
        return RuleFactory.create_sutra_rule(
            rule_id, pattern, replacement, rule_type, sutra_ref
        )


# Plugin registry for managing plugins

class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, plugin: PluginInterface, config: Optional[Dict[str, Any]] = None) -> None:
        """Register a plugin."""
        self._plugins[plugin.plugin_name] = plugin
        if config:
            self._plugin_configs[plugin.plugin_name] = config
        plugin.initialize(config or {})
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].cleanup()
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_configs:
                del self._plugin_configs[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def get_plugins_for_stage(self, stage: ProcessingStage) -> List[PluginInterface]:
        """Get all plugins that support a specific stage."""
        return [plugin for plugin in self._plugins.values() 
                if stage in plugin.supported_stages]


# ML integration registry

class MLRegistry:
    """Registry for ML components."""
    
    def __init__(self):
        self._components: Dict[str, MLIntegrationInterface] = {}
    
    def register_component(self, name: str, component: MLIntegrationInterface) -> None:
        """Register an ML component."""
        self._components[name] = component
    
    def get_component(self, name: str) -> Optional[MLIntegrationInterface]:
        """Get an ML component by name."""
        return self._components.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered ML component names."""
        return list(self._components.keys())


# Utility functions for architecture migration

def migrate_basic_to_advanced_token(basic_token: Any) -> EnhancedToken:
    """Migrate a basic token to the advanced token format."""
    return TokenFactory.from_basic_token(basic_token)


def migrate_basic_to_advanced_rule(basic_rule: Any) -> SutraRule:
    """Migrate a basic rule to the advanced rule format."""
    return RuleFactory.from_basic_rule(basic_rule)


def create_processing_context(stage: ProcessingStage = ProcessingStage.TOKENIZATION,
                            analysis_level: AnalysisLevel = AnalysisLevel.SURFACE) -> ProcessingContext:
    """Create a processing context for operations."""
    return ProcessingContext(stage=stage, analysis_level=analysis_level)


# Hook system for extensibility

class ProcessingHook(ABC):
    """Abstract base class for processing hooks."""
    
    @abstractmethod
    def before_processing(self, data: Any, context: ProcessingContext) -> Any:
        """Called before processing starts."""
        pass
    
    @abstractmethod
    def after_processing(self, data: Any, result: Any, context: ProcessingContext) -> Any:
        """Called after processing completes."""
        pass
    
    @abstractmethod
    def on_error(self, error: Exception, data: Any, context: ProcessingContext) -> None:
        """Called when an error occurs during processing."""
        pass


class HookRegistry:
    """Registry for processing hooks."""
    
    def __init__(self):
        self._hooks: Dict[ProcessingStage, List[ProcessingHook]] = {}
    
    def register_hook(self, stage: ProcessingStage, hook: ProcessingHook) -> None:
        """Register a hook for a specific stage."""
        if stage not in self._hooks:
            self._hooks[stage] = []
        self._hooks[stage].append(hook)
    
    def get_hooks(self, stage: ProcessingStage) -> List[ProcessingHook]:
        """Get hooks for a specific stage."""
        return self._hooks.get(stage, [])
    
    def execute_before_hooks(self, stage: ProcessingStage, data: Any, context: ProcessingContext) -> Any:
        """Execute before-processing hooks."""
        current_data = data
        for hook in self.get_hooks(stage):
            current_data = hook.before_processing(current_data, context)
        return current_data
    
    def execute_after_hooks(self, stage: ProcessingStage, data: Any, result: Any, context: ProcessingContext) -> Any:
        """Execute after-processing hooks."""
        current_result = result
        for hook in self.get_hooks(stage):
            current_result = hook.after_processing(data, current_result, context)
        return current_result
    
    def execute_error_hooks(self, stage: ProcessingStage, error: Exception, data: Any, context: ProcessingContext) -> None:
        """Execute error hooks."""
        for hook in self.get_hooks(stage):
            hook.on_error(error, data, context)