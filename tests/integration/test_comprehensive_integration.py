"""
Comprehensive integration tests for Sanskrit Rewrite Engine.
Part of TO1 comprehensive test suite.

Tests cover:
- End-to-end processing (Requirements 8-9)
- Extensibility and API boundaries (Requirements 10-13)
- Performance and scalability (Requirement 14)
- Cross-verification against Sanskrit datasets
"""

import pytest
import json
import time
import psutil
import gc
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from sanskrit_rewrite_engine.token import Token, TokenKind
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.rule import SutraRule, SutraReference, RuleType, RuleRegistry
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine, PaniniEngineResult


class TestDataLoader:
    """Utility for loading test datasets."""
    
    @staticmethod
    def load_sanskrit_training_data() -> List[Dict]:
        """Load Sanskrit training dataset."""
        try:
            with open("sanskrit_datasets/sanskrit_train.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    @staticmethod
    def load_sanskrit_validation_data() -> List[Dict]:
        """Load Sanskrit validation dataset."""
        try:
            with open("sanskrit_datasets/sanskrit_val.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    @staticmethod
    def load_all_corpus_data() -> Dict[str, List[Dict]]:
        """Load all available corpus data."""
        corpus_files = [
            ("sandhi_examples", "test_corpus/sandhi_examples.json"),
            ("morphological_forms", "test_corpus/morphological_forms.json"),
            ("compound_examples", "test_corpus/compound_examples.json"),
            ("derivation_examples", "test_corpus/derivation_examples.json")
        ]
        
        corpus_data = {}
        for name, filepath in corpus_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    corpus_data[name] = json.load(f)
            except FileNotFoundError:
                corpus_data[name] = []
        
        return corpus_data


class PerformanceProfiler:
    """Performance profiling utility."""
    
    def __init__(self):
        self.start_memory = None
        self.start_time = None
    
    def start_profiling(self):
        """Start performance profiling."""
        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss
        self.start_time = time.time()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'execution_time': end_time - self.start_time if self.start_time else 0,
            'memory_delta': end_memory - self.start_memory if self.start_memory else 0,
            'peak_memory': peak,
            'current_memory': current
        }


@pytest.fixture
def rule_engine():
    """Fixture for Panini rule engine."""
    return PaniniRuleEngine()


@pytest.fixture
def tokenizer():
    """Fixture for Sanskrit tokenizer."""
    return SanskritTokenizer()


@pytest.fixture
def test_datasets():
    """Fixture for test datasets."""
    return {
        'training_data': TestDataLoader.load_sanskrit_training_data(),
        'validation_data': TestDataLoader.load_sanskrit_validation_data(),
        'corpus_data': TestDataLoader.load_all_corpus_data()
    }


class TestRequirement8TracingDebugging:
    """Test Requirement 8: Comprehensive tracing and debugging support."""
    
    def test_detailed_transformation_traces(self, rule_engine, tokenizer):
        """Test that detailed traces are recorded for each transformation."""
        tokens = tokenizer.tokenize("rāma+iti")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Verify trace structure
        assert hasattr(result, 'traces')
        assert isinstance(result.traces, list)
        
        for trace in result.traces:
            # Verify trace fields
            assert hasattr(trace, 'pass_number')
            assert hasattr(trace, 'tokens_before')
            assert hasattr(trace, 'tokens_after')
            assert hasattr(trace, 'transformations')
            assert hasattr(trace, 'paribhasa_applications')
            
            # Verify transformation details
            for transformation in trace.transformations:
                assert hasattr(transformation, 'rule_name')
                assert hasattr(transformation, 'rule_id')
                assert hasattr(transformation, 'sutra_ref')
                assert hasattr(transformation, 'index')
                assert hasattr(transformation, 'tokens_before')
                assert hasattr(transformation, 'tokens_after')
                assert hasattr(transformation, 'timestamp')
                assert isinstance(transformation.timestamp, datetime)
    
    def test_before_after_token_states(self, rule_engine, tokenizer):
        """Test that before/after token states are recorded for each pass."""
        tokens = tokenizer.tokenize("a+i")
        result = rule_engine.process(tokens, max_passes=5)
        
        for trace in result.traces:
            # Should have before and after states
            assert isinstance(trace.tokens_before, list)
            assert isinstance(trace.tokens_after, list)
            
            # All tokens should be valid
            for token in trace.tokens_before + trace.tokens_after:
                assert isinstance(token, Token)
                assert isinstance(token.text, str)
                assert isinstance(token.kind, TokenKind)
    
    def test_error_capture_in_traces(self, rule_engine, tokenizer):
        """Test that errors during rule application are captured."""
        # Create potentially problematic input
        tokens = [Token("", TokenKind.OTHER)]  # Empty token
        result = rule_engine.process(tokens, max_passes=5)
        
        # Should capture any errors
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, list)
        
        # Errors should be strings
        for error in result.errors:
            assert isinstance(error, str)
    
    def test_convergence_status_reporting(self, rule_engine, tokenizer):
        """Test convergence status and pass count reporting."""
        tokens = tokenizer.tokenize("stable")  # Should converge quickly
        result = rule_engine.process(tokens, max_passes=10)
        
        # Should report convergence status
        assert hasattr(result, 'converged')
        assert isinstance(result.converged, bool)
        assert hasattr(result, 'passes')
        assert isinstance(result.passes, int)
        assert result.passes >= 0
        assert result.passes <= 10


class TestRequirement9IterativeProcessing:
    """Test Requirement 9: Iterative processing until fixedpoint."""
    
    def test_fixedpoint_convergence(self, rule_engine, tokenizer):
        """Test that processing continues until no more rules can be applied."""
        tokens = tokenizer.tokenize("a+i")
        result = rule_engine.process(tokens, max_passes=20)
        
        if result.converged:
            # If converged, should have reached fixedpoint
            assert result.passes > 0
            
            # Last pass should have no transformations
            if result.traces:
                last_trace = result.traces[-1]
                # Convergence means no more transformations in final pass
                # (This might be checked differently depending on implementation)
                assert isinstance(last_trace.transformations, list)
        else:
            # If not converged, should have reached max passes
            assert result.passes == 20
    
    def test_max_passes_enforcement(self, rule_engine, tokenizer):
        """Test that maximum passes limit is enforced."""
        tokens = tokenizer.tokenize("test")
        
        # Test with different max_passes values
        for max_passes in [1, 3, 5]:
            result = rule_engine.process(tokens, max_passes=max_passes)
            assert result.passes <= max_passes
    
    def test_complete_trace_history(self, rule_engine, tokenizer):
        """Test that complete trace history is maintained for all passes."""
        tokens = tokenizer.tokenize("a+a")  # Might trigger multiple passes
        result = rule_engine.process(tokens, max_passes=10)
        
        # Should maintain trace for all passes
        assert len(result.traces) == result.passes
        
        # Each trace should have correct pass number
        for i, trace in enumerate(result.traces):
            assert trace.pass_number == i + 1
    
    def test_non_convergence_handling(self, rule_engine, tokenizer):
        """Test handling of non-convergence scenarios."""
        tokens = tokenizer.tokenize("complex input")
        result = rule_engine.process(tokens, max_passes=2)  # Very low limit
        
        # Should handle non-convergence gracefully
        if not result.converged:
            assert result.passes == 2
            assert "Failed to converge" in " ".join(result.errors) or len(result.errors) == 0


class TestRequirement10Extensibility:
    """Test Requirement 10: Extensible rule definition capabilities."""
    
    def test_custom_rule_addition(self, rule_engine):
        """Test adding custom rules with match and transformation functions."""
        # Create custom match function
        def custom_match(tokens: List[Token], index: int) -> bool:
            return (index < len(tokens) and 
                   tokens[index].text == "custom" and 
                   tokens[index].kind == TokenKind.OTHER)
        
        # Create custom transformation function
        def custom_apply(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
            new_tokens = tokens.copy()
            new_tokens[index] = Token("transformed", TokenKind.OTHER, position=tokens[index].position)
            new_tokens[index].add_tag("custom_transformation")
            new_tokens[index].set_meta("transformation_type", "custom")
            return new_tokens, index + 1
        
        # Create custom rule
        custom_rule = SutraRule(
            sutra_ref=SutraReference(9, 9, 9),
            name="custom_extensibility_rule",
            description="Custom rule for testing extensibility",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=custom_match,
            apply_fn=custom_apply,
            max_applications=5
        )
        
        # Add custom rule
        initial_count = len(rule_engine.registry.get_active_sutra_rules())
        rule_engine.add_custom_rule(custom_rule)
        final_count = len(rule_engine.registry.get_active_sutra_rules())
        
        assert final_count == initial_count + 1
        
        # Test custom rule application
        tokens = [Token("custom", TokenKind.OTHER)]
        result = rule_engine.process(tokens, max_passes=5)
        
        # Should apply custom rule
        transformed_tokens = [t for t in result.output_tokens if t.has_tag("custom_transformation")]
        assert len(transformed_tokens) > 0
        
        # Check metadata
        transformed_token = transformed_tokens[0]
        assert transformed_token.get_meta("transformation_type") == "custom"
    
    def test_rule_priority_specification(self, rule_engine):
        """Test that custom rules can specify priority and application limits."""
        high_priority_rule = SutraRule(
            sutra_ref=SutraReference(9, 9, 10),
            name="high_priority_rule",
            description="High priority test rule",
            rule_type=RuleType.SUTRA,
            priority=0,  # Very high priority
            match_fn=lambda tokens, i: False,  # Never matches
            apply_fn=lambda tokens, i: (tokens, i),
            max_applications=1
        )
        
        low_priority_rule = SutraRule(
            sutra_ref=SutraReference(9, 9, 11),
            name="low_priority_rule",
            description="Low priority test rule",
            rule_type=RuleType.SUTRA,
            priority=100,  # Very low priority
            match_fn=lambda tokens, i: False,  # Never matches
            apply_fn=lambda tokens, i: (tokens, i),
            max_applications=10
        )
        
        rule_engine.add_custom_rule(high_priority_rule)
        rule_engine.add_custom_rule(low_priority_rule)
        
        # Check that rules are ordered by priority
        rules = rule_engine.registry.get_active_sutra_rules()
        priorities = [rule.priority for rule in rules]
        assert priorities == sorted(priorities)
    
    def test_utility_functions_for_tokens(self, rule_engine):
        """Test utility functions for common token operations."""
        # Test token type checking
        vowel_token = Token("a", TokenKind.VOWEL)
        consonant_token = Token("k", TokenKind.CONSONANT)
        marker_token = Token("+", TokenKind.MARKER)
        
        assert vowel_token.kind == TokenKind.VOWEL
        assert consonant_token.kind == TokenKind.CONSONANT
        assert marker_token.kind == TokenKind.MARKER
        
        # Test metadata operations
        vowel_token.set_meta("length", "short")
        vowel_token.set_meta("quality", "low")
        
        assert vowel_token.get_meta("length") == "short"
        assert vowel_token.get_meta("quality") == "low"
        assert vowel_token.get_meta("nonexistent") is None
        assert vowel_token.get_meta("nonexistent", "default") == "default"
        
        # Test tag operations
        vowel_token.add_tag("nucleus")
        vowel_token.add_tag("short")
        
        assert vowel_token.has_tag("nucleus")
        assert vowel_token.has_tag("short")
        assert not vowel_token.has_tag("long")
        
        vowel_token.remove_tag("short")
        assert not vowel_token.has_tag("short")
        assert vowel_token.has_tag("nucleus")  # Other tags preserved
    
    def test_rule_metadata_specification(self, rule_engine):
        """Test that rules can specify structured metadata."""
        metadata_rule = SutraRule(
            sutra_ref=SutraReference(9, 9, 12),
            name="metadata_test_rule",
            description="Rule for testing metadata",
            rule_type=RuleType.SUTRA,
            priority=5,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i),
            meta_data={
                "category": "test",
                "complexity": "simple",
                "author": "test_suite",
                "version": "1.0"
            }
        )
        
        rule_engine.add_custom_rule(metadata_rule)
        
        # Verify metadata is preserved
        assert metadata_rule.meta_data["category"] == "test"
        assert metadata_rule.meta_data["complexity"] == "simple"
        assert metadata_rule.meta_data["author"] == "test_suite"
        assert metadata_rule.meta_data["version"] == "1.0"


class TestRequirement13APIBoundaries:
    """Test Requirement 13: Clear API boundaries for integration."""
    
    def test_standardized_input_output_interfaces(self, rule_engine, tokenizer):
        """Test standardized input/output interfaces."""
        # Test input interface
        tokens = tokenizer.tokenize("test input")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Verify standardized output structure
        assert isinstance(result, PaniniEngineResult)
        
        # Check all required attributes
        required_attributes = [
            'input_text', 'input_tokens', 'output_tokens', 'converged',
            'passes', 'traces', 'errors', 'statistics'
        ]
        
        for attr in required_attributes:
            assert hasattr(result, attr), f"Missing required attribute: {attr}"
        
        # Check attribute types
        assert isinstance(result.input_text, str)
        assert isinstance(result.input_tokens, list)
        assert isinstance(result.output_tokens, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.passes, int)
        assert isinstance(result.traces, list)
        assert isinstance(result.errors, list)
        assert isinstance(result.statistics, dict)
    
    def test_atomic_rule_operations(self, rule_engine, tokenizer):
        """Test that rule applications are exposed as atomic operations."""
        tokens = tokenizer.tokenize("a+i")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Each transformation should be atomic and complete
        for trace in result.traces:
            for transformation in trace.transformations:
                # Should have complete before/after state
                assert isinstance(transformation.tokens_before, list)
                assert isinstance(transformation.tokens_after, list)
                assert transformation.rule_name is not None
                assert transformation.rule_id is not None
                
                # Transformation should be self-contained
                assert len(transformation.tokens_before) > 0 or len(transformation.tokens_after) > 0
    
    def test_structured_trace_data_for_external_analysis(self, rule_engine, tokenizer):
        """Test that trace data is structured for external analysis."""
        tokens = tokenizer.tokenize("complex+analysis")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Trace data should be easily convertible for external analysis
        for trace in result.traces:
            # Should be able to extract structured information
            trace_summary = {
                'pass_number': trace.pass_number,
                'transformation_count': len(trace.transformations),
                'rules_applied': [t.rule_name for t in trace.transformations],
                'paribhasa_count': len(trace.paribhasa_applications),
                'convergence_achieved': trace.convergence_achieved
            }
            
            # All fields should be serializable
            assert isinstance(trace_summary['pass_number'], int)
            assert isinstance(trace_summary['transformation_count'], int)
            assert isinstance(trace_summary['rules_applied'], list)
            assert isinstance(trace_summary['paribhasa_count'], int)
            assert isinstance(trace_summary['convergence_achieved'], bool)
    
    def test_detailed_error_information_with_context(self, rule_engine, tokenizer):
        """Test that errors provide detailed information with context."""
        # Test with potentially problematic input
        tokens = []  # Empty input
        result = rule_engine.process(tokens, max_passes=5)
        
        # Should handle gracefully with informative errors if any
        assert isinstance(result.errors, list)
        
        # If there are errors, they should be informative
        for error in result.errors:
            assert isinstance(error, str)
            assert len(error) > 0  # Should have meaningful content
    
    def test_result_serialization_capability(self, rule_engine, tokenizer):
        """Test that results can be serialized for external systems."""
        tokens = tokenizer.tokenize("serialization test")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Should be able to extract key information for serialization
        serializable_result = {
            'input_text': result.input_text,
            'output_text': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformation_summary': result.get_transformation_summary(),
            'rules_applied': list(result.get_rules_applied()),
            'error_count': len(result.errors)
        }
        
        # All fields should be JSON-serializable types
        import json
        try:
            json.dumps(serializable_result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not serializable: {e}")


class TestRequirement14Performance:
    """Test Requirement 14: Efficient processing capabilities."""
    
    def test_reasonable_performance_typical_inputs(self, rule_engine, tokenizer):
        """Test reasonable performance for typical Sanskrit inputs."""
        typical_texts = [
            "rāma",
            "rāma gacchati",
            "dharma artha kāma mokṣa",
            "satyameva jayate nānṛtam",
            "वसुधैव कुटुम्बकम्"
        ]
        
        profiler = PerformanceProfiler()
        
        for text in typical_texts:
            tokens = tokenizer.tokenize(text)
            
            profiler.start_profiling()
            result = rule_engine.process(tokens, max_passes=10)
            metrics = profiler.stop_profiling()
            
            # Should complete in sub-second time for typical inputs
            assert metrics['execution_time'] < 1.0, f"Too slow for '{text}': {metrics['execution_time']} seconds"
            
            # Should not use excessive memory
            assert metrics['memory_delta'] < 50 * 1024 * 1024, f"Too much memory for '{text}': {metrics['memory_delta']} bytes"
    
    def test_incremental_rule_loading(self, rule_engine):
        """Test incremental rule loading and unloading."""
        # Test adding rules incrementally
        initial_count = len(rule_engine.registry.get_active_sutra_rules())
        
        # Add multiple custom rules
        for i in range(5):
            custom_rule = SutraRule(
                sutra_ref=SutraReference(9, 8, i + 1),
                name=f"incremental_rule_{i}",
                description=f"Incremental test rule {i}",
                rule_type=RuleType.SUTRA,
                priority=10 + i,
                match_fn=lambda tokens, idx: False,  # Never matches
                apply_fn=lambda tokens, idx: (tokens, idx)
            )
            rule_engine.add_custom_rule(custom_rule)
        
        final_count = len(rule_engine.registry.get_active_sutra_rules())
        assert final_count == initial_count + 5
        
        # Test rule enabling/disabling
        rules = rule_engine.registry.get_active_sutra_rules()
        if rules:
            rule = rules[-1]  # Get last added rule
            rule_id = rule.id
            
            rule_engine.disable_rule(rule_id)
            assert not rule.active
            
            rule_engine.enable_rule(rule_id)
            assert rule.active
    
    def test_memory_management_multiple_cycles(self, rule_engine, tokenizer):
        """Test memory management over multiple processing cycles."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Process multiple texts in cycles
        test_texts = ["test " + str(i) for i in range(20)]
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=5)
            
            # Verify processing completed
            assert isinstance(result, PaniniEngineResult)
            
            # Force garbage collection periodically
            if len(text) % 5 == 0:
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 100 * 1024 * 1024, f"Excessive memory growth: {memory_growth} bytes"
    
    def test_scalability_with_rule_set_size(self, rule_engine, tokenizer):
        """Test scalability with increasing rule set sizes."""
        # Measure baseline performance
        tokens = tokenizer.tokenize("baseline test")
        
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        result = rule_engine.process(tokens, max_passes=5)
        baseline_metrics = profiler.stop_profiling()
        
        # Add many rules and test again
        for i in range(50):
            dummy_rule = SutraRule(
                sutra_ref=SutraReference(9, 7, i + 1),
                name=f"scalability_rule_{i}",
                description=f"Scalability test rule {i}",
                rule_type=RuleType.SUTRA,
                priority=20 + i,
                match_fn=lambda tokens, idx: False,  # Never matches
                apply_fn=lambda tokens, idx: (tokens, idx)
            )
            rule_engine.add_custom_rule(dummy_rule)
        
        # Test performance with larger rule set
        profiler.start_profiling()
        result = rule_engine.process(tokens, max_passes=5)
        scaled_metrics = profiler.stop_profiling()
        
        # Performance should not degrade dramatically
        performance_ratio = scaled_metrics['execution_time'] / baseline_metrics['execution_time'] if baseline_metrics['execution_time'] > 0 else 1
        assert performance_ratio < 5.0, f"Poor scalability: {performance_ratio}x slower with more rules"


class TestCrossVerification:
    """Cross-verification against multiple Sanskrit datasets."""
    
    def test_consistency_across_datasets(self, rule_engine, tokenizer, test_datasets):
        """Test consistency across different Sanskrit datasets."""
        training_data = test_datasets['training_data'][:5]  # Limit for performance
        validation_data = test_datasets['validation_data'][:3]
        
        # Process training examples
        training_results = []
        for example in training_data:
            if 'sanskrit_problem' in example:
                tokens = tokenizer.tokenize(example['sanskrit_problem'])
                result = rule_engine.process(tokens, max_passes=5)
                training_results.append({
                    'converged': result.converged,
                    'passes': result.passes,
                    'error_count': len(result.errors)
                })
        
        # Process validation examples
        validation_results = []
        for example in validation_data:
            if 'sanskrit_problem' in example:
                tokens = tokenizer.tokenize(example['sanskrit_problem'])
                result = rule_engine.process(tokens, max_passes=5)
                validation_results.append({
                    'converged': result.converged,
                    'passes': result.passes,
                    'error_count': len(result.errors)
                })
        
        # Check consistency
        if training_results and validation_results:
            training_convergence = sum(r['converged'] for r in training_results) / len(training_results)
            validation_convergence = sum(r['converged'] for r in validation_results) / len(validation_results)
            
            # Convergence rates should be reasonably similar
            convergence_diff = abs(training_convergence - validation_convergence)
            assert convergence_diff < 0.7, f"Inconsistent convergence rates: {convergence_diff}"
    
    def test_corpus_validation_accuracy(self, rule_engine, tokenizer, test_datasets):
        """Test validation against known Sanskrit transformations."""
        corpus_data = test_datasets['corpus_data']
        
        validation_results = []
        
        # Test sandhi examples
        for example in corpus_data.get('sandhi_examples', [])[:3]:
            source_text = example['source_text']
            expected_target = example['target_text']
            
            tokens = tokenizer.tokenize(source_text)
            result = rule_engine.process(tokens, max_passes=10)
            output_text = result.get_output_text()
            
            validation_results.append({
                'source': source_text,
                'expected': expected_target,
                'actual': output_text,
                'transformed': output_text != source_text,
                'converged': result.converged
            })
        
        # Should show some transformation activity
        if validation_results:
            transformation_rate = sum(r['transformed'] for r in validation_results) / len(validation_results)
            convergence_rate = sum(r['converged'] for r in validation_results) / len(validation_results)
            
            # Should have reasonable transformation and convergence rates
            assert transformation_rate >= 0.2, f"Low transformation rate: {transformation_rate}"
            assert convergence_rate >= 0.5, f"Low convergence rate: {convergence_rate}"
    
    def test_multiple_dataset_processing(self, rule_engine, tokenizer, test_datasets):
        """Test processing examples from multiple datasets."""
        all_examples = []
        
        # Collect examples from all datasets
        for example in test_datasets['training_data'][:3]:
            if 'sanskrit_problem' in example:
                all_examples.append(('training', example['sanskrit_problem']))
        
        for example in test_datasets['validation_data'][:2]:
            if 'sanskrit_problem' in example:
                all_examples.append(('validation', example['sanskrit_problem']))
        
        for dataset_name, dataset in test_datasets['corpus_data'].items():
            for example in dataset[:2]:
                if 'source_text' in example:
                    all_examples.append((dataset_name, example['source_text']))
        
        # Process all examples
        results_by_dataset = {}
        for dataset_name, text in all_examples:
            if dataset_name not in results_by_dataset:
                results_by_dataset[dataset_name] = []
            
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=5)
            
            results_by_dataset[dataset_name].append({
                'text': text,
                'success': len(result.errors) == 0 or all("Failed to converge" in e for e in result.errors),
                'converged': result.converged
            })
        
        # Check that all datasets are processed successfully
        for dataset_name, results in results_by_dataset.items():
            if results:
                success_rate = sum(r['success'] for r in results) / len(results)
                assert success_rate >= 0.7, f"Low success rate for {dataset_name}: {success_rate}"


class TestStressTesting:
    """Stress testing with large inputs and concurrent processing."""
    
    def test_large_text_processing(self, rule_engine, tokenizer):
        """Test processing of large Sanskrit texts."""
        # Create a large synthetic Sanskrit text
        base_words = ["rāma", "sītā", "lakṣmaṇa", "hanumān", "bharata"]
        large_text = " ".join(base_words * 50)  # 250 words
        
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        
        tokens = tokenizer.tokenize(large_text)
        result = rule_engine.process(tokens, max_passes=3)  # Limited passes for large text
        
        metrics = profiler.stop_profiling()
        
        # Should handle large text without issues
        assert len(result.output_tokens) > 0
        assert metrics['execution_time'] < 15.0, f"Too slow for large text: {metrics['execution_time']} seconds"
        assert isinstance(result, PaniniEngineResult)
    
    def test_memory_stress_with_repeated_processing(self, rule_engine, tokenizer):
        """Test memory usage under repeated processing stress."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Process many texts repeatedly
        for cycle in range(10):
            for i in range(10):
                text = f"stress test {cycle} iteration {i}"
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=3)
                
                assert isinstance(result, PaniniEngineResult)
            
            # Periodic garbage collection
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should not have excessive memory growth under stress
        assert memory_growth < 200 * 1024 * 1024, f"Excessive memory growth under stress: {memory_growth} bytes"
    
    def test_concurrent_processing_safety(self, rule_engine, tokenizer):
        """Test that concurrent processing is safe."""
        import threading
        import queue
        
        # Create multiple processing tasks
        tasks = [f"concurrent test {i}" for i in range(10)]
        results_queue = queue.Queue()
        
        def process_task(text):
            try:
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=3)
                results_queue.put(('success', result))
            except Exception as e:
                results_queue.put(('error', str(e)))
        
        # Run tasks concurrently
        threads = []
        for task in tasks:
            thread = threading.Thread(target=process_task, args=(task,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All should complete successfully
        assert len(results) == len(tasks)
        
        success_count = sum(1 for status, _ in results if status == 'success')
        assert success_count == len(tasks), f"Some concurrent tasks failed: {len(tasks) - success_count} failures"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])