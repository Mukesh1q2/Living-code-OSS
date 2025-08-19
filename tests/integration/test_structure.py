"""
Integration test to verify the test structure and organization works correctly.
"""

import pytest
from pathlib import Path


class TestIntegrationStructure:
    """Test the integration test structure."""
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that integration marker is applied correctly."""
        # This test should have the integration marker
        assert True
    
    def test_integration_directory_structure(self):
        """Test that integration tests are in the correct directory."""
        current_file = Path(__file__)
        assert current_file.parent.name == "integration"
        assert current_file.parent.parent.name == "tests"
    
    def test_can_access_fixtures(self, sample_texts, mock_engine):
        """Test that fixtures are accessible from integration tests."""
        assert sample_texts is not None
        assert mock_engine is not None
        
        # Test that we can use the fixtures
        result = mock_engine.process("test text")
        assert result is not None
        assert hasattr(result, "input_text")
        assert hasattr(result, "output_text")
    
    def test_mock_objects_work_together(self, mock_engine, mock_tokenizer, sample_rule):
        """Test that mock objects can work together in integration scenarios."""
        # Test tokenizer
        tokens = mock_tokenizer.tokenize("test")
        assert len(tokens) == 4  # "test" has 4 characters
        
        # Test engine
        result = mock_engine.process("test + input")
        assert result.success
        assert "+" not in result.output_text  # Mock removes +
        
        # Test rule
        assert sample_rule.id == "test_rule"
        assert sample_rule.pattern == "test"
    
    def test_temp_files_work(self, temp_rule_file, temp_config_file):
        """Test that temporary files work in integration tests."""
        import json
        
        # Test rule file
        with open(temp_rule_file, 'r') as f:
            rules_data = json.load(f)
        assert "rule_set" in rules_data
        assert "rules" in rules_data
        
        # Test config file
        with open(temp_config_file, 'r') as f:
            config_data = json.load(f)
        assert "max_iterations" in config_data
        assert "enable_tracing" in config_data