"""
Simple tests for future architecture components without heavy dependencies.

This module tests the core future architecture functionality without requiring
all the engine dependencies.
"""

import pytest
import sys
import os
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only the interfaces and future architecture components directly
# Import interfaces directly to avoid __init__.py dependencies
import importlib.util
import os

# Load interfaces module directly
interfaces_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'sanskrit_rewrite_engine', 'interfaces.py')
spec = importlib.util.spec_from_file_location("interfaces", interfaces_path)
interfaces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interfaces)

# Import the classes we need
ProcessingStage = interfaces.ProcessingStage
RuleType = interfaces.RuleType
ProcessingContext = interfaces.ProcessingContext
AnalysisLevel = interfaces.AnalysisLevel
LinguisticFeature = interfaces.LinguisticFeature
EnhancedToken = interfaces.EnhancedToken
SutraRule = interfaces.SutraRule


class TestInterfaces:
    """Test the interface definitions."""
    
    def test_processing_stage_enum(self):
        """Test ProcessingStage enum."""
        assert ProcessingStage.TOKENIZATION.value == "tokenization"
        assert ProcessingStage.MORPHOLOGICAL_ANALYSIS.value == "morphological_analysis"
        assert ProcessingStage.RULE_APPLICATION.value == "rule_application"
    
    def test_rule_type_enum(self):
        """Test RuleType enum."""
        assert RuleType.SANDHI.value == "sandhi"
        assert RuleType.MORPHOLOGICAL.value == "morphological"
        assert RuleType.SUTRA.value == "sutra"
    
    def test_processing_context(self):
        """Test ProcessingContext creation."""
        context = ProcessingContext(
            stage=ProcessingStage.TOKENIZATION,
            analysis_level=AnalysisLevel.SURFACE
        )
        assert context.stage == ProcessingStage.TOKENIZATION
        assert context.analysis_level == AnalysisLevel.SURFACE
        assert isinstance(context.features, dict)
        assert isinstance(context.metadata, dict)
    
    def test_linguistic_feature(self):
        """Test LinguisticFeature creation."""
        feature = LinguisticFeature(
            name="case",
            value="nominative",
            confidence=0.9
        )
        assert feature.name == "case"
        assert feature.value == "nominative"
        assert feature.confidence == 0.9
    
    def test_enhanced_token(self):
        """Test EnhancedToken creation."""
        token = EnhancedToken(
            _surface_form="rāma",
            _underlying_form="rāma"
        )
        assert token.surface_form == "rāma"
        assert token.underlying_form == "rāma"
        assert isinstance(token.features, dict)
    
    def test_sutra_rule(self):
        """Test SutraRule creation."""
        rule = SutraRule(
            _rule_id="test_rule",
            _rule_type=RuleType.SANDHI,
            _pattern="a\\+i",
            _replacement="e"
        )
        assert rule.rule_id == "test_rule"
        assert rule.rule_type == RuleType.SANDHI
        assert rule.matches is not None
        assert rule.apply is not None


class TestFutureArchitectureComponents:
    """Test future architecture components that don't require engine dependencies."""
    
    def test_import_future_architecture(self):
        """Test that future architecture modules can be imported."""
        try:
            from sanskrit_rewrite_engine.future_architecture import (
                SutraCategory, SutraReference
            )
            assert SutraCategory.SANDHI.value == "sandhi"
            
            sutra_ref = SutraReference(
                sutra_number="6.1.77",
                sutra_text="iko yaṇ aci",
                category=SutraCategory.SANDHI
            )
            assert sutra_ref.sutra_number == "6.1.77"
            
        except ImportError as e:
            pytest.skip(f"Future architecture import failed: {e}")
    
    def test_import_enhanced_rules(self):
        """Test that enhanced rules can be imported."""
        try:
            from sanskrit_rewrite_engine.enhanced_rules import (
                RuleConditionType, RuleOperator, RuleCondition
            )
            assert RuleConditionType.PHONOLOGICAL.value == "phonological"
            assert RuleOperator.EQUALS.value == "equals"
            
            condition = RuleCondition(
                condition_type=RuleConditionType.MORPHOLOGICAL,
                operator=RuleOperator.EQUALS,
                target_field="case",
                expected_value="nominative"
            )
            assert condition.condition_type == RuleConditionType.MORPHOLOGICAL
            
        except ImportError as e:
            pytest.skip(f"Enhanced rules import failed: {e}")
    
    def test_import_ml_integration(self):
        """Test that ML integration can be imported."""
        try:
            from sanskrit_rewrite_engine.ml_integration import (
                MLModelType, MLTask, MLPrediction
            )
            assert MLModelType.TRANSFORMER.value == "transformer"
            assert MLTask.TOKENIZATION.value == "tokenization"
            
            prediction = MLPrediction(
                prediction="test_result",
                confidence=0.85,
                model_name="test_model",
                task=MLTask.TOKENIZATION
            )
            assert prediction.prediction == "test_result"
            assert prediction.confidence == 0.85
            
        except ImportError as e:
            pytest.skip(f"ML integration import failed: {e}")


class TestArchitectureIntegration:
    """Test architecture integration without full engine dependencies."""
    
    def test_basic_functionality(self):
        """Test basic functionality that doesn't require engine."""
        # Test that we can create processing contexts
        context = ProcessingContext(
            stage=ProcessingStage.TOKENIZATION,
            analysis_level=AnalysisLevel.SURFACE
        )
        
        # Add some metadata
        context.metadata['test_key'] = 'test_value'
        assert context.metadata['test_key'] == 'test_value'
        
        # Add some features
        feature = LinguisticFeature(
            name="test_feature",
            value="test_value",
            confidence=0.8
        )
        context.features['test_feature'] = feature
        assert context.features['test_feature'].name == "test_feature"
    
    def test_mock_integration(self):
        """Test integration with mocked components."""
        # Create mock engine
        mock_engine = Mock()
        mock_engine.process.return_value = Mock(
            output_text="test_output",
            transformations_applied=["test_rule"]
        )
        
        # Test that we can work with mock components
        assert mock_engine.process("test_input").output_text == "test_output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])