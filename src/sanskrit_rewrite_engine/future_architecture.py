"""
Future architecture preparation for the Sanskrit Rewrite Engine.

This module implements the architectural foundations needed for future enhancements
including advanced token-based processing, Pāṇini sūtra encoding, extensible rule
formats, plugin architecture, and machine learning integration hooks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Protocol
import json
import importlib
import inspect
from pathlib import Path

from .interfaces import (
    ProcessingStage, RuleType, TokenType, AnalysisLevel,
    AdvancedToken, AdvancedRule, ProcessingPipeline, PluginInterface,
    MLIntegrationInterface, ProcessingContext, LinguisticFeature,
    EnhancedToken, SutraRule, ExtensiblePipeline, TokenFactory,
    RuleFactory, PluginRegistry, MLRegistry, ProcessingHook, HookRegistry
)


logger = logging.getLogger(__name__)


class SutraCategory(Enum):
    """Categories of Pāṇini sūtras for future encoding."""
    SANDHI = "sandhi"  # Euphonic combination rules
    PRATYAYA = "pratyaya"  # Suffix rules
    DHATU = "dhatu"  # Verbal root rules
    SUBANTA = "subanta"  # Nominal declension rules
    TINGANTA = "tinganta"  # Verbal conjugation rules
    TADDHITA = "taddhita"  # Secondary derivative rules
    KRIT = "krit"  # Primary derivative rules
    SAMASA = "samasa"  # Compound formation rules
    STRI = "stri"  # Feminine formation rules
    ACCENT = "accent"  # Accentuation rules


@dataclass
class SutraReference:
    """Reference to a Pāṇini sūtra with metadata."""
    sutra_number: str  # e.g., "6.1.77"
    sutra_text: str  # Sanskrit text of the sūtra
    category: SutraCategory
    translation: Optional[str] = None
    commentary: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other sūtra numbers
    scope: Optional[str] = None  # Scope of application
    exceptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenMetadata:
    """Enhanced metadata for token-based processing."""
    phonological_features: Dict[str, Any] = field(default_factory=dict)
    morphological_features: Dict[str, Any] = field(default_factory=dict)
    syntactic_features: Dict[str, Any] = field(default_factory=dict)
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    prosodic_features: Dict[str, Any] = field(default_factory=dict)
    derivational_history: List[str] = field(default_factory=list)
    sutra_applications: List[SutraReference] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    alternative_analyses: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedSanskritToken(EnhancedToken):
    """Advanced token implementation with comprehensive Sanskrit linguistic features."""
    
    def __init__(self, surface_form: str, underlying_form: Optional[str] = None):
        super().__init__(surface_form, underlying_form or surface_form)
        self.token_metadata = TokenMetadata()
        self._phonetic_representation: Optional[str] = None
        self._morpheme_boundaries: List[int] = []
        self._syllable_boundaries: List[int] = []
        self._stress_pattern: Optional[str] = None
        self._meter_weight: Optional[str] = None
    
    @property
    def phonetic_representation(self) -> Optional[str]:
        """Phonetic representation of the token."""
        return self._phonetic_representation
    
    @phonetic_representation.setter
    def phonetic_representation(self, value: str) -> None:
        self._phonetic_representation = value
    
    def add_sutra_application(self, sutra_ref: SutraReference) -> None:
        """Add a sūtra application to the token's history."""
        self.token_metadata.sutra_applications.append(sutra_ref)
    
    def get_sutra_applications(self) -> List[SutraReference]:
        """Get all sūtra applications for this token."""
        return self.token_metadata.sutra_applications.copy()
    
    def set_morpheme_boundaries(self, boundaries: List[int]) -> None:
        """Set morpheme boundary positions."""
        self._morpheme_boundaries = boundaries.copy()
    
    def get_morpheme_boundaries(self) -> List[int]:
        """Get morpheme boundary positions."""
        return self._morpheme_boundaries.copy()
    
    def set_syllable_boundaries(self, boundaries: List[int]) -> None:
        """Set syllable boundary positions."""
        self._syllable_boundaries = boundaries.copy()
    
    def get_syllable_boundaries(self) -> List[int]:
        """Get syllable boundary positions."""
        return self._syllable_boundaries.copy()
    
    def add_alternative_analysis(self, analysis: Dict[str, Any]) -> None:
        """Add an alternative linguistic analysis."""
        self.token_metadata.alternative_analyses.append(analysis)
    
    def get_best_analysis(self) -> Dict[str, Any]:
        """Get the best analysis based on confidence scores."""
        if not self.token_metadata.alternative_analyses:
            return {}
        
        # Return analysis with highest average confidence
        best_analysis = None
        best_score = 0.0
        
        for analysis in self.token_metadata.alternative_analyses:
            scores = analysis.get('confidence_scores', {})
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_analysis = analysis
        
        return best_analysis or self.token_metadata.alternative_analyses[0]


class ExtensibleSutraRule(SutraRule):
    """Extensible rule implementation with comprehensive sūtra encoding."""
    
    def __init__(self, rule_id: str, rule_type: RuleType, pattern: str, replacement: str):
        super().__init__(rule_id, rule_type, pattern, replacement)
        self.sutra_reference_obj: Optional[SutraReference] = None
        self._preconditions: List[Callable[[Any, ProcessingContext], bool]] = []
        self._postconditions: List[Callable[[Any, Any, ProcessingContext], bool]] = []
        self._side_effects: List[Callable[[Any, Any, ProcessingContext], None]] = []
        self._rule_interactions: Dict[str, str] = {}  # rule_id -> interaction_type
        self._application_contexts: List[str] = []
        self._blocking_contexts: List[str] = []
    
    def set_sutra_reference(self, sutra_ref: SutraReference) -> None:
        """Set the sūtra reference for this rule."""
        self.sutra_reference_obj = sutra_ref
        self._sutra_reference = sutra_ref.sutra_number
    
    def add_precondition(self, condition: Callable[[Any, ProcessingContext], bool]) -> None:
        """Add a precondition that must be met for rule application."""
        self._preconditions.append(condition)
    
    def add_postcondition(self, condition: Callable[[Any, Any, ProcessingContext], bool]) -> None:
        """Add a postcondition that must be met after rule application."""
        self._postconditions.append(condition)
    
    def add_side_effect(self, effect: Callable[[Any, Any, ProcessingContext], None]) -> None:
        """Add a side effect to execute after rule application."""
        self._side_effects.append(effect)
    
    def set_rule_interaction(self, other_rule_id: str, interaction_type: str) -> None:
        """Set interaction with another rule (blocks, requires, etc.)."""
        self._rule_interactions[other_rule_id] = interaction_type
    
    def matches(self, target: Any, context: ProcessingContext) -> bool:
        """Enhanced matching with preconditions and context checking."""
        # Check basic pattern match
        if not super().matches(target, context):
            return False
        
        # Check application contexts
        if self._application_contexts:
            current_context = context.metadata.get('rule_context', '')
            if current_context not in self._application_contexts:
                return False
        
        # Check blocking contexts
        if self._blocking_contexts:
            current_context = context.metadata.get('rule_context', '')
            if current_context in self._blocking_contexts:
                return False
        
        # Check preconditions
        for precondition in self._preconditions:
            if not precondition(target, context):
                return False
        
        return True
    
    def apply(self, target: Any, context: ProcessingContext) -> Any:
        """Enhanced application with postconditions and side effects."""
        # Apply the basic rule
        result = super().apply(target, context)
        
        # Check postconditions
        for postcondition in self._postconditions:
            if not postcondition(target, result, context):
                logger.warning(f"Postcondition failed for rule {self.rule_id}")
                return target  # Revert if postcondition fails
        
        # Execute side effects
        for side_effect in self._side_effects:
            side_effect(target, result, context)
        
        # Record sūtra application if available
        if isinstance(result, AdvancedSanskritToken) and self.sutra_reference_obj:
            result.add_sutra_application(self.sutra_reference_obj)
        
        return result
    
    def get_rule_interactions(self) -> Dict[str, str]:
        """Get rule interactions."""
        return self._rule_interactions.copy()
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize rule to JSON format."""
        rule_data = {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type.value,
            'pattern': self._pattern,
            'replacement': self._replacement,
            'priority': self._priority,
            'enabled': self._enabled,
            'metadata': self.get_metadata()
        }
        
        if self.sutra_reference_obj:
            rule_data['sutra_reference'] = {
                'sutra_number': self.sutra_reference_obj.sutra_number,
                'sutra_text': self.sutra_reference_obj.sutra_text,
                'category': self.sutra_reference_obj.category.value,
                'translation': self.sutra_reference_obj.translation,
                'examples': self.sutra_reference_obj.examples
            }
        
        if self._application_contexts:
            rule_data['application_contexts'] = self._application_contexts
        
        if self._blocking_contexts:
            rule_data['blocking_contexts'] = self._blocking_contexts
        
        if self._rule_interactions:
            rule_data['rule_interactions'] = self._rule_interactions
        
        return rule_data
    
    @classmethod
    def from_json(cls, rule_data: Dict[str, Any]) -> 'ExtensibleSutraRule':
        """Create rule from JSON data."""
        rule = cls(
            rule_id=rule_data['rule_id'],
            rule_type=RuleType(rule_data['rule_type']),
            pattern=rule_data['pattern'],
            replacement=rule_data['replacement']
        )
        
        rule._priority = rule_data.get('priority', 1)
        rule._enabled = rule_data.get('enabled', True)
        rule._metadata.update(rule_data.get('metadata', {}))
        
        # Load sūtra reference if present
        if 'sutra_reference' in rule_data:
            sutra_data = rule_data['sutra_reference']
            sutra_ref = SutraReference(
                sutra_number=sutra_data['sutra_number'],
                sutra_text=sutra_data['sutra_text'],
                category=SutraCategory(sutra_data['category']),
                translation=sutra_data.get('translation'),
                examples=sutra_data.get('examples', [])
            )
            rule.set_sutra_reference(sutra_ref)
        
        # Load contexts and interactions
        rule._application_contexts = rule_data.get('application_contexts', [])
        rule._blocking_contexts = rule_data.get('blocking_contexts', [])
        rule._rule_interactions = rule_data.get('rule_interactions', {})
        
        return rule


class LinguisticPlugin(PluginInterface):
    """Base class for linguistic processing plugins."""
    
    def __init__(self, name: str, version: str, supported_stages: List[ProcessingStage]):
        self._name = name
        self._version = version
        self._supported_stages = supported_stages
        self._config: Dict[str, Any] = {}
        self._initialized = False
    
    @property
    def plugin_name(self) -> str:
        return self._name
    
    @property
    def plugin_version(self) -> str:
        return self._version
    
    @property
    def supported_stages(self) -> List[ProcessingStage]:
        return self._supported_stages.copy()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        self._config = config.copy()
        self._initialized = True
        logger.info(f"Initialized plugin: {self.plugin_name} v{self.plugin_version}")
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
        logger.info(f"Cleaned up plugin: {self.plugin_name}")
    
    @abstractmethod
    def process(self, data: Any, context: ProcessingContext) -> Any:
        """Process data in the plugin."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self._config.copy()


class MLIntegrationHook(ProcessingHook):
    """Hook for integrating ML components into processing pipeline."""
    
    def __init__(self, ml_component: MLIntegrationInterface, 
                 stages: List[ProcessingStage]):
        self.ml_component = ml_component
        self.stages = stages
        self._predictions_cache: Dict[str, Any] = {}
    
    def before_processing(self, data: Any, context: ProcessingContext) -> Any:
        """Apply ML preprocessing if applicable."""
        if context.stage in self.stages:
            try:
                # Use ML component for preprocessing
                enhanced_data = self.ml_component.predict(data, context)
                return enhanced_data
            except Exception as e:
                logger.error(f"ML preprocessing failed: {e}")
                return data
        return data
    
    def after_processing(self, data: Any, result: Any, context: ProcessingContext) -> Any:
        """Apply ML postprocessing if applicable."""
        if context.stage in self.stages:
            try:
                # Use ML component for result enhancement
                enhanced_result = self.ml_component.predict(result, context)
                return enhanced_result
            except Exception as e:
                logger.error(f"ML postprocessing failed: {e}")
                return result
        return result
    
    def on_error(self, error: Exception, data: Any, context: ProcessingContext) -> None:
        """Handle errors in ML processing."""
        logger.error(f"ML processing error in stage {context.stage}: {error}")


class ArchitectureMigrationManager:
    """Manager for migrating from current to future architecture."""
    
    def __init__(self):
        self.plugin_registry = PluginRegistry()
        self.ml_registry = MLRegistry()
        self.hook_registry = HookRegistry()
        self.pipeline = ExtensiblePipeline()
        self._migration_strategies: Dict[str, Callable] = {}
    
    def register_migration_strategy(self, component_type: str, 
                                  strategy: Callable[[Any], Any]) -> None:
        """Register a migration strategy for a component type."""
        self._migration_strategies[component_type] = strategy
    
    def migrate_token(self, basic_token: Any) -> AdvancedSanskritToken:
        """Migrate a basic token to advanced Sanskrit token."""
        surface_form = getattr(basic_token, 'text', str(basic_token))
        advanced_token = AdvancedSanskritToken(surface_form)
        
        # Copy basic attributes
        if hasattr(basic_token, 'start_pos'):
            advanced_token._position = basic_token.start_pos
        
        if hasattr(basic_token, 'metadata'):
            for key, value in basic_token.metadata.items():
                if key.startswith('phonological_'):
                    advanced_token.token_metadata.phonological_features[key] = value
                elif key.startswith('morphological_'):
                    advanced_token.token_metadata.morphological_features[key] = value
                elif key.startswith('syntactic_'):
                    advanced_token.token_metadata.syntactic_features[key] = value
        
        return advanced_token
    
    def migrate_rule(self, basic_rule: Any) -> ExtensibleSutraRule:
        """Migrate a basic rule to extensible sūtra rule."""
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
        
        advanced_rule = ExtensibleSutraRule(rule_id, rule_type, pattern, replacement)
        advanced_rule._priority = getattr(basic_rule, 'priority', 1)
        advanced_rule._enabled = getattr(basic_rule, 'enabled', True)
        
        # Add sūtra reference if available
        sutra_ref = metadata.get('sutra_ref')
        if sutra_ref:
            # Create basic sūtra reference
            sutra_category_map = {
                'sandhi': SutraCategory.SANDHI,
                'compound': SutraCategory.SAMASA,
                'morphological': SutraCategory.PRATYAYA
            }
            
            sutra_category = sutra_category_map.get(category, SutraCategory.SANDHI)
            sutra_reference = SutraReference(
                sutra_number=sutra_ref,
                sutra_text=f"Sūtra {sutra_ref}",  # Placeholder
                category=sutra_category
            )
            advanced_rule.set_sutra_reference(sutra_reference)
        
        return advanced_rule
    
    def setup_ml_integration(self, ml_component: MLIntegrationInterface,
                           stages: List[ProcessingStage]) -> None:
        """Set up ML integration for specified stages."""
        # Register ML component
        component_name = f"ml_component_{len(self.ml_registry.list_components())}"
        self.ml_registry.register_component(component_name, ml_component)
        
        # Create and register ML hook
        ml_hook = MLIntegrationHook(ml_component, stages)
        for stage in stages:
            self.hook_registry.register_hook(stage, ml_hook)
        
        logger.info(f"Set up ML integration for stages: {[s.value for s in stages]}")
    
    def load_plugin_from_file(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Load a plugin from a Python file."""
        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(plugin_module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    
                    # Instantiate and register plugin
                    plugin_instance = obj()
                    self.plugin_registry.register_plugin(plugin_instance, config)
                    logger.info(f"Loaded plugin: {plugin_instance.plugin_name}")
                    break
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
    
    def create_enhanced_pipeline(self) -> ExtensiblePipeline:
        """Create an enhanced processing pipeline with all registered components."""
        enhanced_pipeline = ExtensiblePipeline()
        
        # Add registered plugins to appropriate stages
        for plugin_name in self.plugin_registry.list_plugins():
            plugin = self.plugin_registry.get_plugin(plugin_name)
            if plugin:
                for stage in plugin.supported_stages:
                    # Create a processor wrapper for the plugin
                    processor = PluginProcessor(plugin)
                    enhanced_pipeline.add_stage(stage, processor)
        
        # Add ML components
        for component_name in self.ml_registry.list_components():
            component = self.ml_registry.get_component(component_name)
            if component:
                enhanced_pipeline.add_ml_component(component)
        
        return enhanced_pipeline
    
    def export_configuration(self, output_path: str) -> None:
        """Export current architecture configuration to JSON."""
        config = {
            'plugins': [
                {
                    'name': name,
                    'config': self.plugin_registry._plugin_configs.get(name, {})
                }
                for name in self.plugin_registry.list_plugins()
            ],
            'ml_components': self.ml_registry.list_components(),
            'migration_strategies': list(self._migration_strategies.keys())
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported architecture configuration to {output_path}")


class PluginProcessor:
    """Wrapper to make plugins compatible with pipeline processors."""
    
    def __init__(self, plugin: PluginInterface):
        self.plugin = plugin
    
    def process_token(self, token: Any, context: ProcessingContext) -> Any:
        """Process token using the plugin."""
        if hasattr(self.plugin, 'process'):
            return self.plugin.process(token, context)
        return token
    
    def can_process(self, token: Any, context: ProcessingContext) -> bool:
        """Check if plugin can process the token."""
        return context.stage in self.plugin.supported_stages


# Factory functions for creating architecture components

def create_sutra_reference(sutra_number: str, sutra_text: str, 
                          category: SutraCategory, **kwargs) -> SutraReference:
    """Create a sūtra reference with metadata."""
    return SutraReference(
        sutra_number=sutra_number,
        sutra_text=sutra_text,
        category=category,
        **kwargs
    )


def create_advanced_token(surface_form: str, **kwargs) -> AdvancedSanskritToken:
    """Create an advanced Sanskrit token."""
    token = AdvancedSanskritToken(surface_form)
    
    # Set optional attributes
    if 'phonetic_representation' in kwargs:
        token.phonetic_representation = kwargs['phonetic_representation']
    
    if 'morpheme_boundaries' in kwargs:
        token.set_morpheme_boundaries(kwargs['morpheme_boundaries'])
    
    if 'syllable_boundaries' in kwargs:
        token.set_syllable_boundaries(kwargs['syllable_boundaries'])
    
    return token


def create_extensible_rule(rule_id: str, pattern: str, replacement: str,
                          rule_type: RuleType = RuleType.PHONOLOGICAL,
                          **kwargs) -> ExtensibleSutraRule:
    """Create an extensible sūtra rule."""
    rule = ExtensibleSutraRule(rule_id, rule_type, pattern, replacement)
    
    # Set optional attributes
    if 'sutra_reference' in kwargs:
        rule.set_sutra_reference(kwargs['sutra_reference'])
    
    if 'preconditions' in kwargs:
        for condition in kwargs['preconditions']:
            rule.add_precondition(condition)
    
    if 'postconditions' in kwargs:
        for condition in kwargs['postconditions']:
            rule.add_postcondition(condition)
    
    return rule


def setup_future_architecture(engine: Any) -> ArchitectureMigrationManager:
    """Set up future architecture components for an engine."""
    migration_manager = ArchitectureMigrationManager()
    
    # Register default migration strategies
    migration_manager.register_migration_strategy('token', migration_manager.migrate_token)
    migration_manager.register_migration_strategy('rule', migration_manager.migrate_rule)
    
    # Add architecture preparation to engine if possible
    if hasattr(engine, '_future_architecture'):
        engine._future_architecture = migration_manager
    
    logger.info("Future architecture setup completed")
    return migration_manager