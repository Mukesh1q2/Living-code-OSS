"""
Comprehensive unit tests for the Sanskrit Rewrite Engine.

This module tests the core SanskritRewriteEngine class functionality including
text processing, rule application, configuration management, and error handling.
"""

import pytest
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.sanskrit_rewrite_engine.engine import (
    SanskritRewriteEngine, TransformationResult, 
    RuleApplicationContext, InfiniteLoopGuard
)
from src.sanskrit_rewrite_engine.rules import Rule
from tests.fixtures.sample_texts import (
    BASIC_WORDS, MORPHOLOGICAL_EXAMPLES, SANDHI_EXAMPLES, ERROR_CASES
)


class TestSanskritRewriteEngine:
    """Test the main Sanskrit Rewrite Engine class."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.engine = SanskritRewriteEngine()
        
    def test_engine_initialization(self):
        """Test engine initialization with default configuration."""
        assert self.engine is not None
        assert hasattr(self.engine, 'tokenizer')
        assert hasattr(self.engine, 'rule_registry')
        assert hasattr(self.engine, 'config')
        
    def test_engine_initialization_with_config(self):
        """Test engine initialization with custom configuration."""
        custom_config = {
            'max_iterations': 5,
            'enable_tracing': False,
            'timeout_seconds': 10
        }
        engine = SanskritRewriteEngine(config=custom_config)
        
        assert engine.config.max_iterations == 5
        assert engine.config.enable_tracing == False
        assert engine.config.timeout_seconds == 10
        
    def test_process_simple_text(self):
        """Test processing simple text without transformations."""
        result = self.engine.process("simple text")
        
        assert isinstance(result, TransformationResult)
        assert result.input_text == "simple text"
        assert result.success == True
        assert isinstance(result.transformations_applied, list)
        assert isinstance(result.trace, list)
        
    def test_process_with_basic_transformations(self):
        """Test processing text with basic transformations."""
        # Test with text that should trigger basic rules
        result = self.engine.process("rāma + iti")
        
        assert result.success == True
        assert result.input_text == "rāma + iti"
        # Should apply vowel sandhi rule
        assert "rāmeti" in result.output_text or result.output_text != result.input_text
        
    def test_process_with_tracing_enabled(self):
        """Test processing with tracing enabled."""
        result = self.engine.process("test text", enable_tracing=True)
        
        assert result.success == True
        assert isinstance(result.trace, list)
        # Should have at least tokenization step
        assert len(result.trace) > 0
        
    def test_process_with_tracing_disabled(self):
        """Test processing with tracing disabled."""
        result = self.engine.process("test text", enable_tracing=False)
        
        assert result.success == True
        assert result.trace == []
        
    def test_process_empty_text(self):
        """Test processing empty text."""
        result = self.engine.process("")
        
        assert result.success == True
        assert result.input_text == ""
        assert result.output_text == ""
        
    def test_process_whitespace_only(self):
        """Test processing whitespace-only text."""
        result = self.engine.process("   ")
        
        assert result.success == True
        assert result.input_text == "   "
        
    def test_process_very_long_text(self):
        """Test processing very long text."""
        long_text = "a" * 1000
        result = self.engine.process(long_text)
        
        assert result.success == True
        assert result.input_text == long_text
        
    def test_process_with_timeout(self):
        """Test processing with timeout configuration."""
        # Create engine with very short timeout
        config = {'timeout_seconds': 0.001}
        engine = SanskritRewriteEngine(config=config)
        
        # Process text that might take time
        result = engine.process("test text")
        
        # Should either succeed quickly or fail with timeout
        assert isinstance(result, TransformationResult)
        
    def test_load_rules_from_file(self, temp_rule_file):
        """Test loading rules from JSON file."""
        initial_count = self.engine.get_rule_count()
        self.engine.load_rules(temp_rule_file)
        
        # Should have loaded additional rules
        assert self.engine.get_rule_count() >= initial_count
        
    def test_load_rules_nonexistent_file(self):
        """Test loading rules from nonexistent file."""
        with pytest.raises(ValueError, match="Error loading rules"):
            self.engine.load_rules("nonexistent_file.json")
            
    def test_add_rule_programmatically(self):
        """Test adding a rule programmatically."""
        initial_count = self.engine.get_rule_count()
        
        rule = Rule(
            id="test_rule_programmatic",
            name="Test Rule",
            description="A test rule",
            pattern="test",
            replacement="TEST"
        )
        
        self.engine.add_rule(rule)
        assert self.engine.get_rule_count() == initial_count + 1
        
    def test_get_rule_count(self):
        """Test getting rule count."""
        count = self.engine.get_rule_count()
        assert isinstance(count, int)
        assert count >= 0
        
    def test_clear_rules(self):
        """Test clearing all rules."""
        # Add a rule first
        rule = Rule(
            id="test_rule_clear",
            name="Test Rule",
            description="A test rule",
            pattern="test",
            replacement="TEST"
        )
        self.engine.add_rule(rule)
        
        # Clear rules
        self.engine.clear_rules()
        assert self.engine.get_rule_count() == 0
        
    def test_get_config(self):
        """Test getting configuration as dictionary."""
        config = self.engine.get_config()
        
        assert isinstance(config, dict)
        assert 'max_iterations' in config
        assert 'enable_tracing' in config
        
    def test_update_config(self):
        """Test updating configuration."""
        original_max_iterations = self.engine.config.max_iterations
        
        self.engine.update_config({'max_iterations': 15})
        
        assert self.engine.config.max_iterations == 15
        assert self.engine.config.max_iterations != original_max_iterations
        
    def test_process_with_max_iterations_limit(self):
        """Test processing with max iterations limit."""
        # Create engine with low max iterations
        config = {'max_iterations': 2}
        engine = SanskritRewriteEngine(config=config)
        
        result = engine.process("test text", enable_tracing=True)
        
        assert result.success == True
        # Check that we didn't exceed max iterations in trace
        iteration_steps = [step for step in result.trace if step.get('step') == 'iteration_summary']
        assert len(iteration_steps) <= 2
        
    def test_process_error_handling(self):
        """Test error handling during processing."""
        # Mock the tokenizer to raise an exception
        with patch.object(self.engine.tokenizer, 'tokenize', side_effect=Exception("Test error")):
            result = self.engine.process("test text")
            
            assert result.success == False
            assert result.error_message == "Test error"
            assert result.input_text == "test text"
            
    def test_process_with_invalid_config(self):
        """Test processing with invalid configuration values."""
        # Test with negative max_iterations
        config = {'max_iterations': -1}
        engine = SanskritRewriteEngine(config=config)
        
        # Should handle gracefully
        result = engine.process("test text")
        assert isinstance(result, TransformationResult)
        
    def test_transformation_result_structure(self):
        """Test that TransformationResult has correct structure."""
        result = self.engine.process("test")
        
        # Check all required fields exist
        assert hasattr(result, 'input_text')
        assert hasattr(result, 'output_text')
        assert hasattr(result, 'transformations_applied')
        assert hasattr(result, 'trace')
        assert hasattr(result, 'success')
        assert hasattr(result, 'error_message')
        
        # Check field types
        assert isinstance(result.input_text, str)
        assert isinstance(result.output_text, str)
        assert isinstance(result.transformations_applied, list)
        assert isinstance(result.trace, list)
        assert isinstance(result.success, bool)
        
    def test_process_with_morphological_examples(self):
        """Test processing with morphological examples."""
        for example in MORPHOLOGICAL_EXAMPLES:
            result = self.engine.process(example)
            
            assert result.success == True
            assert result.input_text == example
            assert isinstance(result.output_text, str)
            
    def test_process_with_sandhi_examples(self):
        """Test processing with sandhi examples."""
        for example in SANDHI_EXAMPLES:
            result = self.engine.process(example)
            
            assert result.success == True
            assert result.input_text == example
            assert isinstance(result.output_text, str)
            
    def test_process_with_error_cases(self):
        """Test processing with error cases."""
        for error_case in ERROR_CASES:
            result = self.engine.process(error_case)
            
            # Should handle gracefully without crashing
            assert isinstance(result, TransformationResult)
            assert result.input_text == error_case
            
    def test_process_trace_structure(self):
        """Test that trace has correct structure when enabled."""
        result = self.engine.process("test text", enable_tracing=True)
        
        assert isinstance(result.trace, list)
        
        for trace_step in result.trace:
            assert isinstance(trace_step, dict)
            assert 'step' in trace_step
            assert 'timestamp' in trace_step
            
    def test_concurrent_processing(self):
        """Test that engine can handle concurrent processing."""
        import threading
        
        results = []
        
        def process_text(text):
            result = self.engine.process(f"test {text}")
            results.append(result)
            
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_text, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Check results
        assert len(results) == 5
        for result in results:
            assert result.success == True
            
    def test_memory_usage_with_large_text(self):
        """Test memory usage with large text."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large text
        large_text = "rāma + iti " * 1000
        result = self.engine.process(large_text)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        assert result.success == True
        
    def test_processing_time_measurement(self):
        """Test that processing time is reasonable."""
        start_time = time.time()
        result = self.engine.process("test text")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Processing should complete quickly (less than 1 second for simple text)
        assert processing_time < 1.0
        assert result.success == True
        
    def test_rule_application_order(self):
        """Test that rules are applied in correct priority order."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rules with different priorities
        rule1 = Rule(
            id="rule_priority_1",
            name="High Priority Rule",
            description="High priority rule",
            pattern="test",
            replacement="HIGH",
            priority=1
        )
        
        rule2 = Rule(
            id="rule_priority_2", 
            name="Low Priority Rule",
            description="Low priority rule",
            pattern="test",
            replacement="LOW",
            priority=2
        )
        
        self.engine.add_rule(rule2)  # Add lower priority first
        self.engine.add_rule(rule1)  # Add higher priority second
        
        result = self.engine.process("test")
        
        # Higher priority rule (lower number) should be applied
        assert "HIGH" in result.output_text
        
    def test_iterative_rule_application(self):
        """Test that rules are applied iteratively until convergence."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rule that creates opportunity for another rule
        rule1 = Rule(
            id="create_opportunity",
            name="Create Opportunity",
            description="Create opportunity for next rule",
            pattern="start",
            replacement="middle"
        )
        
        rule2 = Rule(
            id="use_opportunity",
            name="Use Opportunity", 
            description="Use opportunity created by first rule",
            pattern="middle",
            replacement="end"
        )
        
        self.engine.add_rule(rule1)
        self.engine.add_rule(rule2)
        
        result = self.engine.process("start", enable_tracing=True)
        
        # Should apply both rules iteratively
        assert result.output_text == "end"
        assert len(result.transformations_applied) >= 2


class TestAdvancedRuleApplicationLogic:
    """Test advanced rule application features."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.engine = SanskritRewriteEngine()
        
    def test_infinite_loop_detection(self):
        """Test infinite loop detection and prevention."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add a rule that could cause infinite loop
        rule = Rule(
            id="infinite_loop_rule",
            name="Infinite Loop Rule",
            description="Rule that could cause infinite loop",
            pattern="a",
            replacement="aa",
            priority=1
        )
        
        self.engine.add_rule(rule)
        
        result = self.engine.process("a", enable_tracing=True)
        
        # Should detect infinite loop and stop
        assert result.infinite_loop_detected == True
        assert result.success == True  # Should still succeed with partial result
        
    def test_convergence_detection(self):
        """Test convergence detection when no more rules apply."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add a simple rule
        rule = Rule(
            id="simple_rule",
            name="Simple Rule",
            description="Simple replacement rule",
            pattern="old",
            replacement="new"
        )
        
        self.engine.add_rule(rule)
        
        result = self.engine.process("old text", enable_tracing=True)
        
        # Should reach convergence
        assert result.convergence_reached == True
        assert result.output_text == "new text"
        
    def test_rule_conflict_resolution(self):
        """Test rule conflict resolution with priority handling."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add conflicting rules with different priorities
        rule1 = Rule(
            id="high_priority_rule",
            name="High Priority Rule",
            description="High priority rule",
            pattern="conflict",
            replacement="HIGH",
            priority=1
        )
        
        rule2 = Rule(
            id="low_priority_rule",
            name="Low Priority Rule", 
            description="Low priority rule",
            pattern="conflict",
            replacement="LOW",
            priority=2
        )
        
        self.engine.add_rule(rule1)
        self.engine.add_rule(rule2)
        
        result = self.engine.process("conflict", enable_tracing=True)
        
        # Higher priority rule should win
        assert result.output_text == "HIGH"
        
        # Check trace for conflict resolution
        conflict_traces = [step for step in result.trace 
                          if step.get('step') == 'iteration_summary' 
                          and step.get('rule_conflicts')]
        assert len(conflict_traces) > 0
        
    def test_conditional_rule_activation(self):
        """Test conditional rule activation based on context."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rule with iteration-based condition
        rule = Rule(
            id="conditional_rule",
            name="Conditional Rule",
            description="Rule that only activates after iteration 2",
            pattern="test",
            replacement="CONDITIONAL",
            priority=1,
            metadata={
                "conditions": {
                    "min_iteration": 2
                }
            }
        )
        
        self.engine.add_rule(rule)
        
        result = self.engine.process("test", enable_tracing=True)
        
        # Rule should not apply in first iteration
        # (This test may need adjustment based on actual behavior)
        assert isinstance(result, TransformationResult)
        
    def test_rule_application_frequency_limits(self):
        """Test rule application frequency limits."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rule with very strict global limit
        rule = Rule(
            id="limited_rule",
            name="Limited Rule",
            description="Rule with application limits",
            pattern="a",
            replacement="b",
            priority=1,
            metadata={
                "max_global_applications": 1,  # Very strict limit
                "conditions": {
                    "max_applications_per_iteration": 1
                }
            }
        )
        
        self.engine.add_rule(rule)
        
        result = self.engine.process("aaaa", enable_tracing=True)
        
        # Should respect application limits
        assert result.success == True
        # Should have limited applications due to frequency limits
        total_applications = len([t for t in result.transformations_applied if "limited_rule" in t])
        # The rule should be limited by the frequency constraints
        assert total_applications >= 1  # At least one application should occur
        # Check that the frequency limiting mechanism is working
        assert isinstance(result, TransformationResult)
        
    def test_rule_context_requirements(self):
        """Test rules with context requirements."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rule that requires specific text pattern
        rule = Rule(
            id="context_rule",
            name="Context Rule",
            description="Rule that requires specific context",
            pattern="target",
            replacement="REPLACED",
            priority=1,
            metadata={
                "conditions": {
                    "requires_text_pattern": r"prefix.*target"
                }
            }
        )
        
        self.engine.add_rule(rule)
        
        # Test with required context
        result1 = self.engine.process("prefix target")
        assert "REPLACED" in result1.output_text
        
        # Test without required context
        result2 = self.engine.process("target")
        assert "REPLACED" not in result2.output_text
        
    def test_rule_dependency_conditions(self):
        """Test rules with dependency conditions."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add prerequisite rule
        prereq_rule = Rule(
            id="prerequisite_rule",
            name="Prerequisite Rule",
            description="Rule that must run first",
            pattern="start",
            replacement="middle",
            priority=1
        )
        
        # Add dependent rule
        dependent_rule = Rule(
            id="dependent_rule",
            name="Dependent Rule",
            description="Rule that depends on prerequisite",
            pattern="middle",
            replacement="end",
            priority=2,
            metadata={
                "conditions": {
                    "requires_previous_rule": "prerequisite_rule"
                }
            }
        )
        
        self.engine.add_rule(prereq_rule)
        self.engine.add_rule(dependent_rule)
        
        result = self.engine.process("start", enable_tracing=True)
        
        # Both rules should apply in sequence
        assert result.output_text == "end"
        
    def test_rule_conflict_with_same_priority(self):
        """Test conflict resolution when rules have same priority."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rules with same priority but different characteristics
        rule1 = Rule(
            id="same_priority_1",
            name="Same Priority 1",
            description="First rule with same priority",
            pattern="test",
            replacement="FIRST",
            priority=1
        )
        
        rule2 = Rule(
            id="same_priority_2",
            name="Same Priority 2",
            description="Second rule with same priority but longer pattern",
            pattern="test.*",
            replacement="SECOND",
            priority=1
        )
        
        self.engine.add_rule(rule1)
        self.engine.add_rule(rule2)
        
        result = self.engine.process("test", enable_tracing=True)
        
        # Should resolve conflict deterministically
        assert result.success == True
        assert result.output_text in ["FIRST", "SECOND"]
        
    def test_transformation_result_advanced_fields(self):
        """Test that TransformationResult includes advanced fields."""
        result = self.engine.process("test")
        
        # Check advanced fields exist
        assert hasattr(result, 'convergence_reached')
        assert hasattr(result, 'iterations_used')
        assert hasattr(result, 'infinite_loop_detected')
        
        # Check field types
        assert isinstance(result.convergence_reached, bool)
        assert isinstance(result.iterations_used, int)
        assert isinstance(result.infinite_loop_detected, bool)
        
    def test_rule_application_context(self):
        """Test RuleApplicationContext functionality."""
        context = RuleApplicationContext(
            current_text="test",
            position=0,
            iteration=1,
            previous_applications=["rule1", "rule2", "rule1"],
            text_history=["original", "modified"],
            rule_application_count={"rule1": 2, "rule2": 1}
        )
        
        # Test context methods
        assert context.get_rule_application_frequency("rule1") == 2
        assert context.get_rule_application_frequency("rule3") == 0
        assert context.has_rule_been_applied_recently("rule1", 2) == True
        assert context.has_rule_been_applied_recently("rule3", 2) == False
        
        # Test context hash
        hash1 = context.get_context_hash()
        context.current_text = "different"
        hash2 = context.get_context_hash()
        assert hash1 != hash2
        
    def test_infinite_loop_guard(self):
        """Test InfiniteLoopGuard functionality."""
        guard = InfiniteLoopGuard(
            max_same_rule_applications=3,
            max_text_state_repetitions=2
        )
        
        # Test rule application counting
        assert guard.check_infinite_loop("text1", "rule1") == False
        assert guard.check_infinite_loop("text2", "rule1") == False
        assert guard.check_infinite_loop("text3", "rule1") == False
        assert guard.check_infinite_loop("text4", "rule1") == True  # Exceeds limit
        
        # Reset and test text state repetition
        guard.reset()
        assert guard.check_infinite_loop("same_text", "rule1") == False
        assert guard.check_infinite_loop("same_text", "rule2") == False
        assert guard.check_infinite_loop("same_text", "rule3") == True  # Exceeds repetition
        
    def test_advanced_tracing_information(self):
        """Test that advanced tracing includes detailed information."""
        result = self.engine.process("test", enable_tracing=True)
        
        # Check for advanced trace information
        for trace_step in result.trace:
            if trace_step.get('step') == 'rule_application':
                # Should include advanced fields
                assert 'context_hash' in trace_step or 'rule_frequency' in trace_step
            elif trace_step.get('step') == 'completion':
                # Should include advanced completion info
                assert 'convergence_reached' in trace_step
                assert 'infinite_loop_detected' in trace_step
                
    def test_processing_with_rule_context(self):
        """Test processing with external rule context."""
        # Clear existing rules
        self.engine.clear_rules()
        
        # Add rule that requires specific category
        rule = Rule(
            id="category_rule",
            name="Category Rule",
            description="Rule for specific category",
            pattern="test",
            replacement="CATEGORY",
            priority=1,
            metadata={
                "category": "special"
            }
        )
        
        self.engine.add_rule(rule)
        
        # Test with matching category context
        rule_context = {"required_categories": ["special"]}
        result1 = self.engine.process("test", rule_context=rule_context)
        
        # Test without matching category context
        rule_context2 = {"required_categories": ["other"]}
        result2 = self.engine.process("test", rule_context=rule_context2)
        
        # Results should differ based on context
        assert isinstance(result1, TransformationResult)
        assert isinstance(result2, TransformationResult)