"""
Architecture integration module for the Sanskrit Rewrite Engine.

This module integrates all future architecture components and provides
a unified interface for using advanced features while maintaining
backward compatibility with the existing system.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

from .interfaces import (
    ProcessingStage, ProcessingContext, AdvancedToken, AdvancedRule,
    ExtensiblePipeline, PluginRegistry, MLRegistry, HookRegistry
)
from .future_architecture import (
    ArchitectureMigrationManager, AdvancedSanskritToken, ExtensibleSutraRule,
    SutraReference, SutraCategory, MLIntegrationHook
)
from .plugin_system import (
    PluginLoader, SandhiAnalysisPlugin, CompoundAnalysisPlugin,
    MeterAnalysisPlugin, get_example_plugins
)
from .enhanced_rules import (
    ComplexLinguisticRule, RuleSetManager, RuleCondition, RuleAction,
    create_sandhi_rule, create_morphological_rule, create_compound_rule
)
from .ml_integration import SanskritMLIntegration, create_ml_integration


logger = logging.getLogger(__name__)


class FutureArchitectureManager:
    """Main manager for future architecture components."""
    
    def __init__(self, engine: Any):
        self.engine = engine
        self.migration_manager = ArchitectureMigrationManager()
        self.plugin_loader: Optional[PluginLoader] = None
        self.rule_set_manager = RuleSetManager()
        self.ml_integration = create_ml_integration()
        self._initialized = False
        self._config: Dict[str, Any] = {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the future architecture system."""
        if self._initialized:
            logger.warning("Future architecture already initialized")
            return
        
        self._config = config or {}
        
        # Initialize plugin system
        plugin_dirs = self._config.get('plugin_directories', ['plugins'])
        self.plugin_loader = PluginLoader(plugin_dirs)
        
        # Load example plugins if requested
        if self._config.get('load_example_plugins', True):
            self._load_example_plugins()
        
        # Set up ML integration
        if self._config.get('enable_ml_integration', True):
            self._setup_ml_integration()
        
        # Load rule sets
        rule_files = self._config.get('rule_files', [])
        for rule_file in rule_files:
            self.rule_set_manager.load_from_json(rule_file)
        
        # Create default rule sets if none exist
        if not self.rule_set_manager.rule_sets and self._config.get('create_default_rules', True):
            self._create_default_rules()
        
        self._initialized = True
        logger.info("Future architecture initialized successfully")
    
    def migrate_token(self, basic_token: Any) -> AdvancedSanskritToken:
        """Migrate a basic token to advanced format."""
        return self.migration_manager.migrate_token(basic_token)
    
    def migrate_rule(self, basic_rule: Any) -> ExtensibleSutraRule:
        """Migrate a basic rule to advanced format."""
        return self.migration_manager.migrate_rule(basic_rule)
    
    def process_with_plugins(self, data: Any, stage: ProcessingStage) -> Any:
        """Process data using loaded plugins for a specific stage."""
        if not self.plugin_loader:
            return data
        
        context = ProcessingContext(stage=stage)
        current_data = data
        
        # Get plugins for this stage
        loaded_plugins = self.plugin_loader.get_loaded_plugins()
        for plugin_name, plugin in loaded_plugins.items():
            if stage in plugin.supported_stages:
                try:
                    if hasattr(plugin, 'process'):
                        current_data = plugin.process(current_data, context)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin_name}: {e}")
        
        return current_data
    
    def apply_complex_rules(self, tokens: List[Any], rule_set_name: Optional[str] = None) -> List[Any]:
        """Apply complex linguistic rules to tokens."""
        if rule_set_name:
            rules = self.rule_set_manager.get_rule_set(rule_set_name)
        else:
            rules = list(self.rule_set_manager.rules.values())
        
        if not rules:
            return tokens
        
        # Get optimal application order
        rule_ids = [rule.rule_id for rule in rules]
        ordered_ids = self.rule_set_manager.get_application_order(rule_ids)
        ordered_rules = [self.rule_set_manager.rules[rule_id] for rule_id in ordered_ids]
        
        # Apply rules to each token
        result_tokens = []
        for token in tokens:
            current_token = token
            context = ProcessingContext(stage=ProcessingStage.RULE_APPLICATION)
            
            for rule in ordered_rules:
                if rule.matches(current_token, context):
                    current_token = rule.apply(current_token, context)
            
            result_tokens.append(current_token)
        
        return result_tokens
    
    def enhance_processing_pipeline(self) -> ExtensiblePipeline:
        """Create an enhanced processing pipeline with all components."""
        return self.migration_manager.create_enhanced_pipeline()
    
    def get_sutra_references(self, text: str) -> List[SutraReference]:
        """Get relevant sūtra references for text processing."""
        references = []
        
        # Check applied rules for sūtra references
        for rule in self.rule_set_manager.rules.values():
            if rule.sutra_reference_obj and rule.matches(text, ProcessingContext()):
                references.append(rule.sutra_reference_obj)
        
        return references
    
    def validate_architecture(self) -> Dict[str, List[str]]:
        """Validate the current architecture setup."""
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Validate rule dependencies
        rule_issues = self.rule_set_manager.validate_dependencies()
        validation_results['errors'].extend(rule_issues)
        
        # Validate plugins
        if self.plugin_loader:
            loaded_plugins = self.plugin_loader.get_loaded_plugins()
            if not loaded_plugins:
                validation_results['warnings'].append("No plugins loaded")
            else:
                validation_results['info'].append(f"Loaded {len(loaded_plugins)} plugins")
        
        # Validate ML integration
        ml_info = self.ml_integration.get_model_info()
        if not ml_info.get('registered_models'):
            validation_results['warnings'].append("No ML models registered")
        else:
            model_count = len(ml_info['registered_models'])
            validation_results['info'].append(f"Registered {model_count} ML models")
        
        return validation_results
    
    def export_configuration(self, output_path: str) -> None:
        """Export current configuration for future use."""
        config = {
            'plugin_directories': self._config.get('plugin_directories', []),
            'rule_files': self._config.get('rule_files', []),
            'ml_integration_enabled': self._config.get('enable_ml_integration', True),
            'loaded_plugins': list(self.plugin_loader.get_loaded_plugins().keys()) if self.plugin_loader else [],
            'rule_sets': list(self.rule_set_manager.rule_sets.keys()),
            'ml_models': self.ml_integration.get_model_info()
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration exported to {output_path}")
    
    def _load_example_plugins(self) -> None:
        """Load example plugins for demonstration."""
        example_plugins = get_example_plugins()
        
        for plugin_class in example_plugins:
            try:
                plugin_instance = plugin_class()
                plugin_name = plugin_instance.plugin_name
                
                # Register with plugin loader
                if self.plugin_loader:
                    self.plugin_loader._loaded_plugins[plugin_name] = plugin_instance
                
                # Register with migration manager
                self.migration_manager.plugin_registry.register_plugin(plugin_instance)
                
                logger.info(f"Loaded example plugin: {plugin_name}")
                
            except Exception as e:
                logger.error(f"Failed to load example plugin {plugin_class.__name__}: {e}")
    
    def _setup_ml_integration(self) -> None:
        """Set up ML integration hooks."""
        # Set up ML integration for key stages
        stages = [
            ProcessingStage.TOKENIZATION,
            ProcessingStage.MORPHOLOGICAL_ANALYSIS,
            ProcessingStage.SYNTACTIC_ANALYSIS
        ]
        
        self.migration_manager.setup_ml_integration(self.ml_integration, stages)
        logger.info("ML integration hooks set up")
    
    def _create_default_rules(self) -> None:
        """Create default rule sets for demonstration."""
        # Create basic sandhi rules
        sandhi_rules = [
            create_sandhi_rule(
                "vowel_sandhi_a_i",
                r"a\s*\+\s*i",
                "e",
                [{'field': 'vowel_type', 'value': 'simple', 'operator': 'equals'}]
            ),
            create_sandhi_rule(
                "vowel_sandhi_a_u",
                r"a\s*\+\s*u",
                "o",
                [{'field': 'vowel_type', 'value': 'simple', 'operator': 'equals'}]
            )
        ]
        
        # Create morphological rules
        morphological_rules = [
            create_morphological_rule(
                "add_case_marker",
                r"(\w+)_(\w+)",
                r"\1\2",
                {'case_applied': True, 'case_type': 'instrumental'}
            )
        ]
        
        # Create compound rules
        compound_rules = [
            create_compound_rule(
                "tatpurusha_compound",
                r"(\w+)\+(\w+)",
                r"\1\2",
                "tatpurusha"
            )
        ]
        
        # Add rules to manager
        all_rules = sandhi_rules + morphological_rules + compound_rules
        for rule in all_rules:
            self.rule_set_manager.add_rule(rule)
        
        # Create rule sets
        self.rule_set_manager.create_rule_set(
            "basic_sandhi",
            [rule.rule_id for rule in sandhi_rules]
        )
        self.rule_set_manager.create_rule_set(
            "morphological_processing",
            [rule.rule_id for rule in morphological_rules]
        )
        self.rule_set_manager.create_rule_set(
            "compound_formation",
            [rule.rule_id for rule in compound_rules]
        )
        
        logger.info("Created default rule sets")


class BackwardCompatibilityLayer:
    """Layer to maintain backward compatibility while using future architecture."""
    
    def __init__(self, engine: Any, future_manager: FutureArchitectureManager):
        self.engine = engine
        self.future_manager = future_manager
    
    def process_with_future_features(self, text: str, **kwargs) -> Any:
        """Process text using future architecture while maintaining compatibility."""
        # Get original result
        original_result = self.engine.process(text, **kwargs)
        
        # If future architecture is not initialized, return original result
        if not self.future_manager._initialized:
            return original_result
        
        try:
            # Migrate tokens to advanced format
            if hasattr(original_result, 'transformations_applied'):
                # This looks like a TransformationResult
                enhanced_result = original_result
                
                # Apply plugins if available
                enhanced_text = self.future_manager.process_with_plugins(
                    text, ProcessingStage.TOKENIZATION
                )
                
                # Apply complex rules if available
                if self.future_manager.rule_set_manager.rules:
                    # Create tokens for rule application
                    tokens = [self.future_manager.migrate_token({'text': enhanced_text})]
                    enhanced_tokens = self.future_manager.apply_complex_rules(tokens)
                    
                    if enhanced_tokens:
                        enhanced_result.output_text = enhanced_tokens[0].surface_form
                
                return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in future architecture processing: {e}")
            # Fall back to original result
            return original_result
        
        return original_result
    
    def get_enhanced_metadata(self, result: Any) -> Dict[str, Any]:
        """Get enhanced metadata from processing result."""
        metadata = {}
        
        if hasattr(result, 'trace'):
            metadata['original_trace'] = result.trace
        
        # Add sūtra references if available
        if hasattr(result, 'output_text'):
            sutra_refs = self.future_manager.get_sutra_references(result.output_text)
            if sutra_refs:
                metadata['sutra_references'] = [
                    {
                        'sutra_number': ref.sutra_number,
                        'sutra_text': ref.sutra_text,
                        'category': ref.category.value
                    }
                    for ref in sutra_refs
                ]
        
        # Add plugin information
        if self.future_manager.plugin_loader:
            loaded_plugins = self.future_manager.plugin_loader.get_loaded_plugins()
            metadata['active_plugins'] = list(loaded_plugins.keys())
        
        return metadata


# Factory function for setting up future architecture

def setup_future_architecture(engine: Any, config: Optional[Dict[str, Any]] = None) -> FutureArchitectureManager:
    """Set up future architecture for an engine instance."""
    future_manager = FutureArchitectureManager(engine)
    future_manager.initialize(config)
    
    # Add future architecture to engine
    if hasattr(engine, '_future_architecture'):
        engine._future_architecture = future_manager
    
    # Create backward compatibility layer
    compat_layer = BackwardCompatibilityLayer(engine, future_manager)
    if hasattr(engine, '_compat_layer'):
        engine._compat_layer = compat_layer
    
    return future_manager


def create_enhanced_engine(base_engine_class: type, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create an enhanced engine with future architecture features."""
    # Create base engine
    engine = base_engine_class(config)
    
    # Set up future architecture
    future_manager = setup_future_architecture(engine, config)
    
    # Add enhanced processing method
    def enhanced_process(text: str, **kwargs):
        if hasattr(engine, '_compat_layer'):
            return engine._compat_layer.process_with_future_features(text, **kwargs)
        else:
            return engine.process(text, **kwargs)
    
    # Replace process method
    engine.enhanced_process = enhanced_process
    
    return engine


# Utility functions for architecture preparation

def validate_future_architecture(engine: Any) -> Dict[str, Any]:
    """Validate future architecture setup for an engine."""
    validation_result = {
        'has_future_architecture': False,
        'components': {},
        'issues': []
    }
    
    if hasattr(engine, '_future_architecture'):
        future_manager = engine._future_architecture
        validation_result['has_future_architecture'] = True
        
        # Validate components
        arch_validation = future_manager.validate_architecture()
        validation_result['components'] = arch_validation
        
        # Check initialization
        if not future_manager._initialized:
            validation_result['issues'].append("Future architecture not initialized")
    else:
        validation_result['issues'].append("Future architecture not set up")
    
    return validation_result


def get_architecture_info(engine: Any) -> Dict[str, Any]:
    """Get information about the engine's architecture setup."""
    info = {
        'has_future_architecture': False,
        'plugins': [],
        'rule_sets': [],
        'ml_models': [],
        'sutra_references': 0
    }
    
    if hasattr(engine, '_future_architecture'):
        future_manager = engine._future_architecture
        info['has_future_architecture'] = True
        
        # Plugin information
        if future_manager.plugin_loader:
            info['plugins'] = list(future_manager.plugin_loader.get_loaded_plugins().keys())
        
        # Rule set information
        info['rule_sets'] = list(future_manager.rule_set_manager.rule_sets.keys())
        
        # ML model information
        ml_info = future_manager.ml_integration.get_model_info()
        info['ml_models'] = list(ml_info.get('registered_models', {}).keys())
        
        # Count sūtra references
        sutra_count = 0
        for rule in future_manager.rule_set_manager.rules.values():
            if rule.sutra_reference_obj:
                sutra_count += 1
        info['sutra_references'] = sutra_count
    
    return info


def migrate_existing_rules(engine: Any, output_path: Optional[str] = None) -> List[ExtensibleSutraRule]:
    """Migrate existing rules to the new format."""
    migrated_rules = []
    
    if hasattr(engine, 'rule_registry'):
        rule_registry = engine.rule_registry
        
        if hasattr(engine, '_future_architecture'):
            future_manager = engine._future_architecture
            
            # Migrate each rule
            for rule in rule_registry._rules:
                try:
                    migrated_rule = future_manager.migrate_rule(rule)
                    migrated_rules.append(migrated_rule)
                    
                    # Add to rule set manager
                    future_manager.rule_set_manager.add_rule(migrated_rule)
                    
                except Exception as e:
                    logger.error(f"Failed to migrate rule {getattr(rule, 'id', 'unknown')}: {e}")
            
            # Save migrated rules if output path provided
            if output_path and migrated_rules:
                future_manager.rule_set_manager.save_to_json(output_path)
                logger.info(f"Migrated {len(migrated_rules)} rules to {output_path}")
    
    return migrated_rules