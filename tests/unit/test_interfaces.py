"""
Tests for the future architecture interfaces.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.sanskrit_rewrite_engine.interfaces import (
    ProcessingStage, RuleType, TokenType, AnalysisLevel,
    LinguisticFeature, ProcessingContext, EnhancedToken, SutraRule,
    ExtensiblePipeline, TokenFactory, RuleFactory, PluginRegistry,
    MLRegistry, HookRegistry, ProcessingHook,
    migrate_basic_to_advanced_token, migrate_basic_to_advanced_rule,
    create_processing_context
)


class TestLinguisticFeature:
    """Test LinguisticFeature dataclass."""
    
    def test_create_feature(self):
        feature = LinguisticFeature(
            name="case",
            value="nominative",
            confidence=0.95,
            source="morphological_analyzer"
        )
        
        assert feature.name == "case"
        assert feature.value == "nominative"
        assert feature.confidence == 0.95
        assert feature.source == "morphological_analyzer"
        assert isinstance(feature.metadata, dict)


class TestProcessingContext:
    """Test ProcessingContext dataclass."""
    
    def test_create_context(self):
        context = ProcessingContext(
            stage=ProcessingStage.TOKENIZATION,
            analysis_level=AnalysisLevel.MORPHOLOGICAL
        )
        
        assert context.stage == ProcessingStage.TOKENIZATION
        assert context.analysis_level == AnalysisLevel.MORPHOLOGICAL
        assert isinstance(context.features, dict)
        assert isinstance(context.metadata, dict)
        assert context.processing_id is not None
    
    def test_create_context_utility(self):
        context = create_processing_context(
            ProcessingStage.RULE_APPLICATION,
            AnalysisLevel.SYNTACTIC
        )
        
        assert context.stage == ProcessingStage.RULE_APPLICATION
        assert context.analysis_level == AnalysisLevel.SYNTACTIC


class TestEnhancedToken:
    """Test EnhancedToken implementation."""
    
    def test_create_token(self):
        token = EnhancedToken(
            _surface_form="rāma",
            _underlying_form="rāma",
            _token_type=TokenType.WORD
        )
        
        assert token.surface_form == "rāma"
        assert token.underlying_form == "rāma"
        assert token.token_type == TokenType.WORD
        assert isinstance(token.features, dict)
    
    def test_token_features(self):
        token = EnhancedToken("test", "test")
        
        feature = LinguisticFeature("case", "nominative", 0.9)
        token.set_feature("case", feature)
        
        retrieved_feature = token.get_feature("case")
        assert retrieved_feature is not None
        assert retrieved_feature.value == "nominative"
        assert retrieved_feature.confidence == 0.9
        
        # Test non-existent feature
        assert token.get_feature("nonexistent") is None


class TestSutraRule:
    """Test SutraRule implementation."""
    
    def test_create_rule(self):
        rule = SutraRule(
            _rule_id="test_rule",
            _rule_type=RuleType.SANDHI,
            _pattern="a\\s*\\+\\s*i",
            _replacement="e",
            _sutra_reference="6.1.87"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.rule_type == RuleType.SANDHI
        assert rule.sutra_reference == "6.1.87"
    
    def test_rule_matching(self):
        rule = SutraRule(
            _rule_id="vowel_sandhi",
            _rule_type=RuleType.SANDHI,
            _pattern="a.*i",
            _replacement="e"
        )
        
        # Create mock token
        token = Mock()
        token.surface_form = "a + i"
        
        context = create_processing_context()
        
        assert rule.matches(token, context) is True
        
        # Test non-matching
        token.surface_form = "u + a"
        assert rule.matches(token, context) is False
    
    def test_rule_application(self):
        rule = SutraRule(
            _rule_id="simple_replace",
            _rule_type=RuleType.PHONOLOGICAL,
            _pattern="old",
            _replacement="new"
        )
        
        token = Mock()
        token.surface_form = "old_text"
        token._surface_form = "old_text"
        
        context = create_processing_context()
        
        result = rule.apply(token, context)
        assert result._surface_form == "new_text"
    
    def test_rule_metadata(self):
        rule = SutraRule(
            _rule_id="test_rule",
            _rule_type=RuleType.MORPHOLOGICAL,
            _pattern="test",
            _replacement="result",
            _sutra_reference="1.1.1"
        )
        
        metadata = rule.get_metadata()
        
        assert metadata['sutra_reference'] == "1.1.1"
        assert metadata['conditions'] == {}
        assert metadata['priority'] == 1
        assert metadata['enabled'] is True


class TestExtensiblePipeline:
    """Test ExtensiblePipeline implementation."""
    
    def test_create_pipeline(self):
        pipeline = ExtensiblePipeline()
        assert isinstance(pipeline._stages, dict)
        assert isinstance(pipeline._plugins, list)
        assert isinstance(pipeline._ml_components, list)
    
    def test_add_stage_processor(self):
        pipeline = ExtensiblePipeline()
        
        # Create mock processor
        processor = Mock()
        processor.can_process.return_value = True
        processor.process_token.return_value = "processed"
        
        pipeline.add_stage(ProcessingStage.TOKENIZATION, processor)
        
        processors = pipeline.get_stage_processors(ProcessingStage.TOKENIZATION)
        assert len(processors) == 1
        assert processors[0] == processor
    
    def test_pipeline_processing(self):
        pipeline = ExtensiblePipeline()
        
        # Create mock processor
        processor = Mock()
        processor.can_process.return_value = True
        processor.process_token.return_value = "processed_data"
        
        pipeline.add_stage(ProcessingStage.TOKENIZATION, processor)
        
        context = create_processing_context(ProcessingStage.TOKENIZATION)
        result = pipeline.process("input_data", context)
        
        assert result == "processed_data"
        processor.can_process.assert_called_once()
        processor.process_token.assert_called_once()


class TestTokenFactory:
    """Test TokenFactory utility."""
    
    def test_create_token(self):
        token = TokenFactory.create_token(
            "rāma",
            "rāma",
            TokenType.WORD
        )
        
        assert isinstance(token, EnhancedToken)
        assert token.surface_form == "rāma"
        assert token.underlying_form == "rāma"
        assert token.token_type == TokenType.WORD
    
    def test_from_basic_token(self):
        # Create mock basic token
        basic_token = Mock()
        basic_token.text = "test_text"
        
        enhanced_token = TokenFactory.from_basic_token(basic_token)
        
        assert isinstance(enhanced_token, EnhancedToken)
        assert enhanced_token.surface_form == "test_text"
        assert enhanced_token.underlying_form == "test_text"


class TestRuleFactory:
    """Test RuleFactory utility."""
    
    def test_create_sutra_rule(self):
        rule = RuleFactory.create_sutra_rule(
            "test_rule",
            "pattern",
            "replacement",
            RuleType.SANDHI,
            "6.1.87"
        )
        
        assert isinstance(rule, SutraRule)
        assert rule.rule_id == "test_rule"
        assert rule.rule_type == RuleType.SANDHI
        assert rule.sutra_reference == "6.1.87"
    
    def test_from_basic_rule(self):
        # Create mock basic rule
        basic_rule = Mock()
        basic_rule.id = "basic_rule"
        basic_rule.pattern = "test_pattern"
        basic_rule.replacement = "test_replacement"
        basic_rule.metadata = {
            'category': 'vowel_sandhi',
            'sutra_ref': '6.1.87'
        }
        
        sutra_rule = RuleFactory.from_basic_rule(basic_rule)
        
        assert isinstance(sutra_rule, SutraRule)
        assert sutra_rule.rule_id == "basic_rule"
        assert sutra_rule.rule_type == RuleType.SANDHI
        assert sutra_rule.sutra_reference == "6.1.87"


class TestPluginRegistry:
    """Test PluginRegistry functionality."""
    
    def test_create_registry(self):
        registry = PluginRegistry()
        assert isinstance(registry._plugins, dict)
        assert isinstance(registry._plugin_configs, dict)
    
    def test_register_plugin(self):
        registry = PluginRegistry()
        
        # Create mock plugin
        plugin = Mock()
        plugin.plugin_name = "test_plugin"
        plugin.supported_stages = [ProcessingStage.TOKENIZATION]
        plugin.initialize = Mock()
        plugin.cleanup = Mock()
        
        config = {"setting": "value"}
        registry.register_plugin(plugin, config)
        
        assert registry.get_plugin("test_plugin") == plugin
        plugin.initialize.assert_called_once_with(config)
    
    def test_unregister_plugin(self):
        registry = PluginRegistry()
        
        plugin = Mock()
        plugin.plugin_name = "test_plugin"
        plugin.supported_stages = []
        plugin.initialize = Mock()
        plugin.cleanup = Mock()
        
        registry.register_plugin(plugin)
        registry.unregister_plugin("test_plugin")
        
        assert registry.get_plugin("test_plugin") is None
        plugin.cleanup.assert_called_once()
    
    def test_get_plugins_for_stage(self):
        registry = PluginRegistry()
        
        plugin1 = Mock()
        plugin1.plugin_name = "plugin1"
        plugin1.supported_stages = [ProcessingStage.TOKENIZATION]
        plugin1.initialize = Mock()
        plugin1.cleanup = Mock()
        
        plugin2 = Mock()
        plugin2.plugin_name = "plugin2"
        plugin2.supported_stages = [ProcessingStage.RULE_APPLICATION]
        plugin2.initialize = Mock()
        plugin2.cleanup = Mock()
        
        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)
        
        tokenization_plugins = registry.get_plugins_for_stage(ProcessingStage.TOKENIZATION)
        assert len(tokenization_plugins) == 1
        assert tokenization_plugins[0] == plugin1


class TestMLRegistry:
    """Test MLRegistry functionality."""
    
    def test_create_registry(self):
        registry = MLRegistry()
        assert isinstance(registry._components, dict)
    
    def test_register_component(self):
        registry = MLRegistry()
        
        component = Mock()
        registry.register_component("test_ml", component)
        
        assert registry.get_component("test_ml") == component
        assert "test_ml" in registry.list_components()


class TestHookRegistry:
    """Test HookRegistry functionality."""
    
    def test_create_registry(self):
        registry = HookRegistry()
        assert isinstance(registry._hooks, dict)
    
    def test_register_hook(self):
        registry = HookRegistry()
        
        hook = Mock()
        hook.before_processing.return_value = "before_result"
        hook.after_processing.return_value = "after_result"
        
        registry.register_hook(ProcessingStage.TOKENIZATION, hook)
        
        hooks = registry.get_hooks(ProcessingStage.TOKENIZATION)
        assert len(hooks) == 1
        assert hooks[0] == hook
    
    def test_execute_hooks(self):
        registry = HookRegistry()
        
        hook = Mock()
        hook.before_processing.return_value = "modified_data"
        hook.after_processing.return_value = "modified_result"
        hook.on_error = Mock()
        
        registry.register_hook(ProcessingStage.TOKENIZATION, hook)
        
        context = create_processing_context(ProcessingStage.TOKENIZATION)
        
        # Test before hooks
        result = registry.execute_before_hooks(ProcessingStage.TOKENIZATION, "data", context)
        assert result == "modified_data"
        hook.before_processing.assert_called_once_with("data", context)
        
        # Test after hooks
        result = registry.execute_after_hooks(ProcessingStage.TOKENIZATION, "data", "result", context)
        assert result == "modified_result"
        hook.after_processing.assert_called_once_with("data", "result", context)
        
        # Test error hooks
        error = Exception("test error")
        registry.execute_error_hooks(ProcessingStage.TOKENIZATION, error, "data", context)
        hook.on_error.assert_called_once_with(error, "data", context)


class TestMigrationUtilities:
    """Test migration utilities."""
    
    def test_migrate_basic_to_advanced_token(self):
        basic_token = Mock()
        basic_token.text = "test_token"
        
        advanced_token = migrate_basic_to_advanced_token(basic_token)
        
        assert isinstance(advanced_token, EnhancedToken)
        assert advanced_token.surface_form == "test_token"
    
    def test_migrate_basic_to_advanced_rule(self):
        basic_rule = Mock()
        basic_rule.id = "test_rule"
        basic_rule.pattern = "test_pattern"
        basic_rule.replacement = "test_replacement"
        basic_rule.metadata = {'category': 'phonological'}
        
        advanced_rule = migrate_basic_to_advanced_rule(basic_rule)
        
        assert isinstance(advanced_rule, SutraRule)
        assert advanced_rule.rule_id == "test_rule"
        assert advanced_rule.rule_type == RuleType.PHONOLOGICAL


class MockProcessingHook(ProcessingHook):
    """Mock processing hook for testing."""
    
    def __init__(self):
        self.before_called = False
        self.after_called = False
        self.error_called = False
    
    def before_processing(self, data, context):
        self.before_called = True
        return f"before_{data}"
    
    def after_processing(self, data, result, context):
        self.after_called = True
        return f"after_{result}"
    
    def on_error(self, error, data, context):
        self.error_called = True


class TestProcessingHook:
    """Test ProcessingHook abstract base class."""
    
    def test_hook_implementation(self):
        hook = MockProcessingHook()
        context = create_processing_context()
        
        # Test before processing
        result = hook.before_processing("data", context)
        assert result == "before_data"
        assert hook.before_called is True
        
        # Test after processing
        result = hook.after_processing("data", "result", context)
        assert result == "after_result"
        assert hook.after_called is True
        
        # Test error handling
        error = Exception("test")
        hook.on_error(error, "data", context)
        assert hook.error_called is True