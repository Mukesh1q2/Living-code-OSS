"""
Tests for future architecture components.

This module tests the integration and functionality of all future architecture
components including token-based processing, sūtra references, extensible rules,
plugin architecture, and ML integration hooks.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanskrit_rewrite_engine.interfaces import (
    ProcessingStage, RuleType, ProcessingContext, AnalysisLevel
)
from sanskrit_rewrite_engine.future_architecture import (
    AdvancedSanskritToken, ExtensibleSutraRule, SutraReference, SutraCategory,
    ArchitectureMigrationManager, create_advanced_token, create_extensible_rule
)
from sanskrit_rewrite_engine.plugin_system import (
    PluginLoader, SandhiAnalysisPlugin, CompoundAnalysisPlugin,
    MeterAnalysisPlugin, PluginTemplate, PluginValidator
)
from sanskrit_rewrite_engine.enhanced_rules import (
    ComplexLinguisticRule, RuleSetManager, RuleCondition, RuleAction,
    RuleConditionType, RuleOperator, create_sandhi_rule
)
from sanskrit_rewrite_engine.ml_integration import (
    SanskritMLIntegration, DummyMLAdapter, MLTask
)
from sanskrit_rewrite_engine.architecture_integration import (
    FutureArchitectureManager, setup_future_architecture,
    create_enhanced_engine, validate_future_architecture
)


class TestAdvancedSanskritToken:
    """Test advanced Sanskrit token functionality."""
    
    def test_token_creation(self):
        """Test creating advanced tokens."""
        token = AdvancedSanskritToken("rāma")
        assert token.surface_form == "rāma"
        assert token.underlying_form == "rāma"
        assert token.token_metadata is not None
    
    def test_sutra_application(self):
        """Test adding sūtra applications to tokens."""
        token = AdvancedSanskritToken("rāma")
        sutra_ref = SutraReference(
            sutra_number="6.1.77",
            sutra_text="iko yaṇ aci",
            category=SutraCategory.SANDHI
        )
        
        token.add_sutra_application(sutra_ref)
        applications = token.get_sutra_applications()
        
        assert len(applications) == 1
        assert applications[0].sutra_number == "6.1.77"
    
    def test_morpheme_boundaries(self):
        """Test morpheme boundary functionality."""
        token = AdvancedSanskritToken("rāmasya")
        token.set_morpheme_boundaries([4])  # rāma|sya
        
        boundaries = token.get_morpheme_boundaries()
        assert boundaries == [4]
    
    def test_alternative_analyses(self):
        """Test alternative analysis functionality."""
        token = AdvancedSanskritToken("rāma")
        
        analysis1 = {
            'analysis_type': 'morphological',
            'features': {'case': 'nominative', 'number': 'singular'},
            'confidence_scores': {'morphology': 0.9}
        }
        
        analysis2 = {
            'analysis_type': 'morphological', 
            'features': {'case': 'vocative', 'number': 'singular'},
            'confidence_scores': {'morphology': 0.7}
        }
        
        token.add_alternative_analysis(analysis1)
        token.add_alternative_analysis(analysis2)
        
        best = token.get_best_analysis()
        assert best['confidence_scores']['morphology'] == 0.9


class TestExtensibleSutraRule:
    """Test extensible sūtra rule functionality."""
    
    def test_rule_creation(self):
        """Test creating extensible rules."""
        rule = ExtensibleSutraRule("test_rule", RuleType.SANDHI, "a\\+i", "e")
        assert rule.rule_id == "test_rule"
        assert rule.rule_type == RuleType.SANDHI
    
    def test_sutra_reference(self):
        """Test sūtra reference functionality."""
        rule = ExtensibleSutraRule("vowel_sandhi", RuleType.SANDHI, "a\\+i", "e")
        sutra_ref = SutraReference(
            sutra_number="6.1.87",
            sutra_text="ād guṇaḥ",
            category=SutraCategory.SANDHI
        )
        
        rule.set_sutra_reference(sutra_ref)
        assert rule.sutra_reference == "6.1.87"
        assert rule.sutra_reference_obj.sutra_text == "ād guṇaḥ"
    
    def test_preconditions(self):
        """Test rule preconditions."""
        rule = ExtensibleSutraRule("test_rule", RuleType.SANDHI, "a\\+i", "e")
        
        def test_condition(token, context):
            return hasattr(token, 'surface_form')
        
        rule.add_precondition(test_condition)
        
        token = AdvancedSanskritToken("a+i")
        context = ProcessingContext(ProcessingStage.RULE_APPLICATION)
        
        assert rule.matches(token, context)
    
    def test_json_serialization(self):
        """Test JSON serialization of rules."""
        rule = ExtensibleSutraRule("test_rule", RuleType.SANDHI, "a\\+i", "e")
        sutra_ref = SutraReference(
            sutra_number="6.1.87",
            sutra_text="ād guṇaḥ",
            category=SutraCategory.SANDHI
        )
        rule.set_sutra_reference(sutra_ref)
        
        json_data = rule.to_json()
        assert json_data['rule_id'] == "test_rule"
        assert json_data['sutra_reference']['sutra_number'] == "6.1.87"
        
        # Test deserialization
        restored_rule = ExtensibleSutraRule.from_json(json_data)
        assert restored_rule.rule_id == "test_rule"
        assert restored_rule.sutra_reference == "6.1.87"


class TestPluginSystem:
    """Test plugin system functionality."""
    
    def test_plugin_creation(self):
        """Test creating plugins."""
        plugin = SandhiAnalysisPlugin()
        assert plugin.plugin_name == "sandhi_analysis"
        assert ProcessingStage.MORPHOLOGICAL_ANALYSIS in plugin.supported_stages
    
    def test_plugin_processing(self):
        """Test plugin processing."""
        plugin = SandhiAnalysisPlugin()
        plugin.initialize({})
        
        token = AdvancedSanskritToken("rāmeti")
        context = ProcessingContext(ProcessingStage.MORPHOLOGICAL_ANALYSIS)
        
        result = plugin.process(token, context)
        assert isinstance(result, AdvancedSanskritToken)
    
    def test_plugin_loader(self):
        """Test plugin loader functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = PluginLoader([temp_dir])
            
            # Create a simple plugin file
            plugin_code = '''
from sanskrit_rewrite_engine.plugin_system import LinguisticPlugin
from sanskrit_rewrite_engine.interfaces import ProcessingStage

class TestPlugin(LinguisticPlugin):
    PLUGIN_NAME = "test_plugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "Test"
    PLUGIN_DESCRIPTION = "Test plugin"
    
    def __init__(self):
        super().__init__(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            supported_stages=[ProcessingStage.TOKENIZATION]
        )
    
    def process(self, data, context):
        return data
'''
            
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_file.write_text(plugin_code)
            
            # Discover plugins
            discovered = loader.discover_plugins()
            assert len(discovered) == 1
            assert discovered[0].name == "test_plugin"
    
    def test_plugin_template_generation(self):
        """Test plugin template generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            PluginTemplate.generate_plugin_template(
                "my_plugin", "Test Author", 
                [ProcessingStage.TOKENIZATION], temp_dir
            )
            
            plugin_dir = Path(temp_dir) / "my_plugin"
            assert plugin_dir.exists()
            assert (plugin_dir / "main.py").exists()
            assert (plugin_dir / "plugin.json").exists()
    
    def test_plugin_validation(self):
        """Test plugin validation."""
        plugin = SandhiAnalysisPlugin()
        issues = PluginValidator.validate_plugin(plugin)
        assert len(issues) == 0  # Should be valid


class TestEnhancedRules:
    """Test enhanced rule system."""
    
    def test_rule_conditions(self):
        """Test rule conditions."""
        condition = RuleCondition(
            condition_type=RuleConditionType.MORPHOLOGICAL,
            operator=RuleOperator.EQUALS,
            target_field="case",
            expected_value="nominative"
        )
        
        token = AdvancedSanskritToken("rāma")
        token.token_metadata.morphological_features["case"] = "nominative"
        context = ProcessingContext(ProcessingStage.RULE_APPLICATION)
        
        assert condition.evaluate(token, context)
    
    def test_rule_actions(self):
        """Test rule actions."""
        action = RuleAction(
            action_type="add_feature",
            parameters={
                'feature_name': 'processed',
                'feature_value': True,
                'feature_type': 'morphological'
            }
        )
        
        token = AdvancedSanskritToken("rāma")
        context = ProcessingContext(ProcessingStage.RULE_APPLICATION)
        
        result = action.execute(token, context)
        assert result.token_metadata.morphological_features.get('processed') is True
    
    def test_complex_rule(self):
        """Test complex linguistic rules."""
        rule = ComplexLinguisticRule("complex_test", RuleType.SANDHI, "a\\+i", "e")
        
        # Add condition
        condition = RuleCondition(
            condition_type=RuleConditionType.PHONOLOGICAL,
            operator=RuleOperator.CONTAINS,
            target_field="surface_form",
            expected_value="+"
        )
        rule.add_condition(condition)
        
        # Add action
        action = RuleAction(
            action_type="add_feature",
            parameters={
                'feature_name': 'sandhi_applied',
                'feature_value': True,
                'feature_type': 'morphological'
            }
        )
        rule.add_action(action)
        
        token = AdvancedSanskritToken("a+i")
        context = ProcessingContext(ProcessingStage.RULE_APPLICATION)
        
        assert rule.matches(token, context)
        result = rule.apply(token, context)
        assert result.token_metadata.morphological_features.get('sandhi_applied') is True
    
    def test_rule_set_manager(self):
        """Test rule set management."""
        manager = RuleSetManager()
        
        rule1 = create_sandhi_rule("rule1", "a\\+i", "e")
        rule2 = create_sandhi_rule("rule2", "a\\+u", "o")
        
        manager.add_rule(rule1)
        manager.add_rule(rule2)
        
        assert len(manager.rules) == 2
        
        # Create rule set
        manager.create_rule_set("vowel_sandhi", ["rule1", "rule2"])
        rule_set = manager.get_rule_set("vowel_sandhi")
        
        assert len(rule_set) == 2
    
    def test_rule_dependencies(self):
        """Test rule dependency validation."""
        manager = RuleSetManager()
        
        rule1 = ComplexLinguisticRule("rule1", RuleType.SANDHI, "a", "b")
        rule2 = ComplexLinguisticRule("rule2", RuleType.SANDHI, "b", "c")
        rule2.set_dependencies(["rule1"])
        
        manager.add_rule(rule1)
        manager.add_rule(rule2)
        
        issues = manager.validate_dependencies()
        assert len(issues) == 0  # No issues
        
        # Test application order
        order = manager.get_application_order(["rule1", "rule2"])
        assert order == ["rule1", "rule2"]


class TestMLIntegration:
    """Test ML integration functionality."""
    
    def test_ml_integration_creation(self):
        """Test creating ML integration."""
        ml_integration = SanskritMLIntegration()
        assert ml_integration is not None
    
    def test_model_registration(self):
        """Test ML model registration."""
        ml_integration = SanskritMLIntegration()
        
        dummy_model = DummyMLAdapter("test_model", [MLTask.TOKENIZATION])
        ml_integration.register_model(
            "test_model", dummy_model, [MLTask.TOKENIZATION], is_default=True
        )
        
        model_info = ml_integration.get_model_info()
        assert "test_model" in model_info["registered_models"]
    
    def test_prediction(self):
        """Test ML prediction."""
        ml_integration = SanskritMLIntegration()
        
        dummy_model = DummyMLAdapter("test_model", [MLTask.TOKENIZATION])
        ml_integration.register_model(
            "test_model", dummy_model, [MLTask.TOKENIZATION], is_default=True
        )
        
        context = ProcessingContext(ProcessingStage.TOKENIZATION)
        result = ml_integration.predict("test input", context)
        
        assert result is not None


class TestArchitectureIntegration:
    """Test architecture integration functionality."""
    
    def test_migration_manager(self):
        """Test architecture migration manager."""
        manager = ArchitectureMigrationManager()
        
        # Test token migration
        basic_token = Mock()
        basic_token.text = "rāma"
        basic_token.start_pos = 0
        basic_token.metadata = {'test': 'value'}
        
        advanced_token = manager.migrate_token(basic_token)
        assert isinstance(advanced_token, AdvancedSanskritToken)
        assert advanced_token.surface_form == "rāma"
    
    def test_future_architecture_manager(self):
        """Test future architecture manager."""
        mock_engine = Mock()
        manager = FutureArchitectureManager(mock_engine)
        
        config = {
            'plugin_directories': [],
            'load_example_plugins': True,
            'enable_ml_integration': True,
            'create_default_rules': True
        }
        
        manager.initialize(config)
        assert manager._initialized
    
    def test_enhanced_engine_creation(self):
        """Test creating enhanced engine."""
        # Mock base engine class
        class MockEngine:
            def __init__(self, config=None):
                self.config = config
            
            def process(self, text, **kwargs):
                return Mock(output_text=text, transformations_applied=[])
        
        enhanced_engine = create_enhanced_engine(MockEngine)
        assert hasattr(enhanced_engine, 'enhanced_process')
        assert hasattr(enhanced_engine, '_future_architecture')
    
    def test_architecture_validation(self):
        """Test architecture validation."""
        mock_engine = Mock()
        mock_engine._future_architecture = Mock()
        mock_engine._future_architecture._initialized = True
        mock_engine._future_architecture.validate_architecture.return_value = {
            'errors': [],
            'warnings': [],
            'info': ['Test info']
        }
        
        result = validate_future_architecture(mock_engine)
        assert result['has_future_architecture'] is True


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_complete_processing_pipeline(self):
        """Test complete processing with all components."""
        # Create mock engine
        mock_engine = Mock()
        mock_engine.process.return_value = Mock(
            output_text="rāmeti",
            transformations_applied=["sandhi_rule"],
            trace=[]
        )
        
        # Set up future architecture
        future_manager = FutureArchitectureManager(mock_engine)
        config = {
            'load_example_plugins': True,
            'enable_ml_integration': True,
            'create_default_rules': True
        }
        future_manager.initialize(config)
        
        # Test token migration and processing
        basic_token = Mock()
        basic_token.text = "rāma+iti"
        
        advanced_token = future_manager.migrate_token(basic_token)
        assert isinstance(advanced_token, AdvancedSanskritToken)
        
        # Test plugin processing
        processed = future_manager.process_with_plugins(
            advanced_token, ProcessingStage.MORPHOLOGICAL_ANALYSIS
        )
        assert processed is not None
        
        # Test rule application
        tokens = [advanced_token]
        rule_result = future_manager.apply_complex_rules(tokens, "basic_sandhi")
        assert len(rule_result) == 1
    
    def test_backward_compatibility(self):
        """Test backward compatibility layer."""
        from sanskrit_rewrite_engine.architecture_integration import BackwardCompatibilityLayer
        
        # Mock engine and future manager
        mock_engine = Mock()
        mock_engine.process.return_value = Mock(
            output_text="test",
            transformations_applied=[]
        )
        
        mock_future_manager = Mock()
        mock_future_manager._initialized = True
        mock_future_manager.process_with_plugins.return_value = "enhanced_test"
        mock_future_manager.rule_set_manager.rules = {}
        
        compat_layer = BackwardCompatibilityLayer(mock_engine, mock_future_manager)
        result = compat_layer.process_with_future_features("test")
        
        assert result is not None
    
    def test_configuration_export_import(self):
        """Test configuration export and import."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            
            mock_engine = Mock()
            future_manager = FutureArchitectureManager(mock_engine)
            future_manager.initialize({'create_default_rules': True})
            
            # Export configuration
            future_manager.export_configuration(str(config_file))
            assert config_file.exists()
            
            # Verify configuration content
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            assert 'rule_sets' in config_data
            assert 'ml_models' in config_data


if __name__ == "__main__":
    pytest.main([__file__])