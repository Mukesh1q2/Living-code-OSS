"""
Regression test suite for continuous integration.
Part of TO1 comprehensive test suite.

Tests cover:
- Regression tests for continuous integration
- API stability and backward compatibility
- Core functionality preservation
- Performance regression detection
"""

import pytest
import json
import time
import hashlib
from typing import List, Dict, Any, Set
from pathlib import Path

from sanskrit_rewrite_engine.token import Token, TokenKind
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.rule import SutraRule, SutraReference, RuleType
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine, PaniniEngineResult


class RegressionTestData:
    """Manages regression test data and expected results."""
    
    @staticmethod
    def get_core_functionality_tests() -> List[Dict[str, Any]]:
        """Get core functionality test cases with expected results."""
        return [
            {
                'name': 'basic_tokenization',
                'input': 'rama',
                'expected_token_count': 4,
                'expected_kinds': ['CONSONANT', 'VOWEL', 'CONSONANT', 'VOWEL']
            },
            {
                'name': 'compound_vowel_tokenization',
                'input': 'ai',
                'expected_token_count': 1,
                'expected_tags': ['compound']
            },
            {
                'name': 'marker_preservation',
                'input': 'word+marker',
                'expected_marker_count': 1,
                'expected_marker_text': '+'
            },
            {
                'name': 'devanagari_tokenization',
                'input': 'राम',
                'expected_token_count': 2,  # र + आम (approximate)
                'min_token_count': 1
            },
            {
                'name': 'empty_input_handling',
                'input': '',
                'expected_token_count': 0
            },
            {
                'name': 'whitespace_handling',
                'input': '   ',
                'expected_token_count': 0
            }
        ]
    
    @staticmethod
    def get_processing_tests() -> List[Dict[str, Any]]:
        """Get processing test cases with expected behavior."""
        return [
            {
                'name': 'simple_vowel_sequence',
                'input': 'a+i',
                'max_passes': 5,
                'should_converge': True,
                'max_time': 1.0
            },
            {
                'name': 'identical_vowels',
                'input': 'a+a',
                'max_passes': 5,
                'should_converge': True,
                'max_time': 1.0
            },
            {
                'name': 'complex_sequence',
                'input': 'rāma+iti',
                'max_passes': 10,
                'should_converge': True,
                'max_time': 2.0
            },
            {
                'name': 'morphological_marker',
                'input': 'word:GEN',
                'max_passes': 5,
                'should_converge': True,
                'max_time': 1.0
            },
            {
                'name': 'no_transformation_needed',
                'input': 'stable',
                'max_passes': 3,
                'should_converge': True,
                'max_time': 0.5
            }
        ]
    
    @staticmethod
    def get_api_stability_tests() -> List[Dict[str, Any]]:
        """Get API stability test cases."""
        return [
            {
                'name': 'tokenizer_interface',
                'methods': ['tokenize', 'get_token_statistics'],
                'properties': ['vowels', 'consonants', 'markers']
            },
            {
                'name': 'rule_engine_interface',
                'methods': ['process', 'add_custom_rule', 'enable_rule', 'disable_rule', 'reset_engine'],
                'properties': ['registry', 'guard_system']
            },
            {
                'name': 'result_interface',
                'attributes': ['input_text', 'input_tokens', 'output_tokens', 'converged', 
                             'passes', 'traces', 'errors', 'statistics'],
                'methods': ['get_output_text', 'get_transformation_summary', 'get_rules_applied']
            },
            {
                'name': 'token_interface',
                'attributes': ['text', 'kind', 'tags', 'meta', 'position'],
                'methods': ['has_tag', 'add_tag', 'remove_tag', 'get_meta', 'set_meta']
            }
        ]


@pytest.fixture
def tokenizer():
    """Fixture for Sanskrit tokenizer."""
    return SanskritTokenizer()


@pytest.fixture
def rule_engine():
    """Fixture for Panini rule engine."""
    return PaniniRuleEngine()


class TestCoreFunctionalityRegression:
    """Test that core functionality doesn't regress."""
    
    def test_tokenization_regression(self, tokenizer):
        """Test that tokenization behavior remains consistent."""
        test_cases = RegressionTestData.get_core_functionality_tests()
        
        for test_case in test_cases:
            input_text = test_case['input']
            tokens = tokenizer.tokenize(input_text)
            
            # Check token count
            if 'expected_token_count' in test_case:
                assert len(tokens) == test_case['expected_token_count'], \
                    f"Token count regression in {test_case['name']}: got {len(tokens)}, expected {test_case['expected_token_count']}"
            
            if 'min_token_count' in test_case:
                assert len(tokens) >= test_case['min_token_count'], \
                    f"Token count below minimum in {test_case['name']}: got {len(tokens)}, min {test_case['min_token_count']}"
            
            # Check token kinds
            if 'expected_kinds' in test_case and tokens:
                actual_kinds = [token.kind.value for token in tokens]
                expected_kinds = test_case['expected_kinds']
                assert actual_kinds == expected_kinds, \
                    f"Token kinds regression in {test_case['name']}: got {actual_kinds}, expected {expected_kinds}"
            
            # Check for expected tags
            if 'expected_tags' in test_case:
                expected_tags = set(test_case['expected_tags'])
                actual_tags = set()
                for token in tokens:
                    actual_tags.update(token.tags)
                
                assert expected_tags.issubset(actual_tags), \
                    f"Missing expected tags in {test_case['name']}: expected {expected_tags}, got {actual_tags}"
            
            # Check marker preservation
            if 'expected_marker_count' in test_case:
                marker_tokens = [t for t in tokens if t.kind == TokenKind.MARKER]
                assert len(marker_tokens) == test_case['expected_marker_count'], \
                    f"Marker count regression in {test_case['name']}: got {len(marker_tokens)}, expected {test_case['expected_marker_count']}"
                
                if 'expected_marker_text' in test_case and marker_tokens:
                    marker_texts = {t.text for t in marker_tokens}
                    assert test_case['expected_marker_text'] in marker_texts, \
                        f"Expected marker text missing in {test_case['name']}: {test_case['expected_marker_text']} not in {marker_texts}"
    
    def test_processing_regression(self, rule_engine, tokenizer):
        """Test that processing behavior remains consistent."""
        test_cases = RegressionTestData.get_processing_tests()
        
        for test_case in test_cases:
            input_text = test_case['input']
            max_passes = test_case['max_passes']
            
            # Measure processing time
            start_time = time.time()
            tokens = tokenizer.tokenize(input_text)
            result = rule_engine.process(tokens, max_passes=max_passes)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Check convergence behavior
            if 'should_converge' in test_case:
                if test_case['should_converge']:
                    assert result.converged or result.passes < max_passes, \
                        f"Convergence regression in {test_case['name']}: failed to converge in {max_passes} passes"
            
            # Check processing time
            if 'max_time' in test_case:
                assert processing_time < test_case['max_time'], \
                    f"Performance regression in {test_case['name']}: took {processing_time:.3f}s, max {test_case['max_time']}s"
            
            # Verify result structure
            assert isinstance(result, PaniniEngineResult), \
                f"Result type regression in {test_case['name']}: got {type(result)}"
            
            assert isinstance(result.output_tokens, list), \
                f"Output tokens type regression in {test_case['name']}: got {type(result.output_tokens)}"
            
            assert isinstance(result.traces, list), \
                f"Traces type regression in {test_case['name']}: got {type(result.traces)}"
    
    def test_token_behavior_regression(self, tokenizer):
        """Test that token behavior remains consistent."""
        # Test basic token creation and manipulation
        token = Token("test", TokenKind.VOWEL, position=0)
        
        # Test tag operations
        token.add_tag("test_tag")
        assert token.has_tag("test_tag"), "Tag addition regression"
        
        token.remove_tag("test_tag")
        assert not token.has_tag("test_tag"), "Tag removal regression"
        
        # Test metadata operations
        token.set_meta("test_key", "test_value")
        assert token.get_meta("test_key") == "test_value", "Metadata setting regression"
        
        assert token.get_meta("nonexistent") is None, "Metadata default regression"
        assert token.get_meta("nonexistent", "default") == "default", "Metadata default value regression"
        
        # Test string representations
        str_repr = str(token)
        assert "test" in str_repr and "VOWEL" in str_repr, "String representation regression"
        
        repr_str = repr(token)
        assert "Token" in repr_str and "test" in repr_str, "Repr representation regression"
    
    def test_statistics_regression(self, tokenizer):
        """Test that statistics generation remains consistent."""
        text = "rāma+iti गच्छति"
        tokens = tokenizer.tokenize(text)
        stats = tokenizer.get_token_statistics(tokens)
        
        # Check required statistics fields
        required_fields = ['total_tokens', 'vowel_count', 'consonant_count', 'marker_count', 'other_count']
        for field in required_fields:
            assert field in stats, f"Missing statistics field: {field}"
            assert isinstance(stats[field], int), f"Statistics field {field} should be integer"
            assert stats[field] >= 0, f"Statistics field {field} should be non-negative"
        
        # Check that total equals sum of parts
        calculated_total = stats['vowel_count'] + stats['consonant_count'] + stats['marker_count'] + stats['other_count']
        assert stats['total_tokens'] == calculated_total, "Statistics total calculation regression"


class TestAPIStabilityRegression:
    """Test that API interfaces remain stable."""
    
    def test_tokenizer_api_stability(self, tokenizer):
        """Test that tokenizer API remains stable."""
        api_tests = RegressionTestData.get_api_stability_tests()
        tokenizer_test = next(test for test in api_tests if test['name'] == 'tokenizer_interface')
        
        # Check required methods exist
        for method_name in tokenizer_test['methods']:
            assert hasattr(tokenizer, method_name), f"Missing tokenizer method: {method_name}"
            assert callable(getattr(tokenizer, method_name)), f"Tokenizer method {method_name} not callable"
        
        # Check required properties exist
        for property_name in tokenizer_test['properties']:
            assert hasattr(tokenizer, property_name), f"Missing tokenizer property: {property_name}"
        
        # Test method signatures (basic smoke test)
        tokens = tokenizer.tokenize("test")
        assert isinstance(tokens, list), "tokenize() should return list"
        
        stats = tokenizer.get_token_statistics(tokens)
        assert isinstance(stats, dict), "get_token_statistics() should return dict"
    
    def test_rule_engine_api_stability(self, rule_engine):
        """Test that rule engine API remains stable."""
        api_tests = RegressionTestData.get_api_stability_tests()
        engine_test = next(test for test in api_tests if test['name'] == 'rule_engine_interface')
        
        # Check required methods exist
        for method_name in engine_test['methods']:
            assert hasattr(rule_engine, method_name), f"Missing rule engine method: {method_name}"
            assert callable(getattr(rule_engine, method_name)), f"Rule engine method {method_name} not callable"
        
        # Check required properties exist
        for property_name in engine_test['properties']:
            assert hasattr(rule_engine, property_name), f"Missing rule engine property: {property_name}"
        
        # Test method signatures (basic smoke test)
        tokens = [Token("test", TokenKind.OTHER)]
        result = rule_engine.process(tokens, max_passes=1)
        assert isinstance(result, PaniniEngineResult), "process() should return PaniniEngineResult"
    
    def test_result_api_stability(self, rule_engine, tokenizer):
        """Test that result object API remains stable."""
        api_tests = RegressionTestData.get_api_stability_tests()
        result_test = next(test for test in api_tests if test['name'] == 'result_interface')
        
        # Create a result object
        tokens = tokenizer.tokenize("test")
        result = rule_engine.process(tokens, max_passes=1)
        
        # Check required attributes exist
        for attr_name in result_test['attributes']:
            assert hasattr(result, attr_name), f"Missing result attribute: {attr_name}"
        
        # Check required methods exist
        for method_name in result_test['methods']:
            assert hasattr(result, method_name), f"Missing result method: {method_name}"
            assert callable(getattr(result, method_name)), f"Result method {method_name} not callable"
        
        # Test method return types
        output_text = result.get_output_text()
        assert isinstance(output_text, str), "get_output_text() should return string"
        
        transformation_summary = result.get_transformation_summary()
        assert isinstance(transformation_summary, dict), "get_transformation_summary() should return dict"
        
        rules_applied = result.get_rules_applied()
        assert isinstance(rules_applied, set), "get_rules_applied() should return set"
    
    def test_token_api_stability(self):
        """Test that token API remains stable."""
        api_tests = RegressionTestData.get_api_stability_tests()
        token_test = next(test for test in api_tests if test['name'] == 'token_interface')
        
        # Create a token
        token = Token("test", TokenKind.VOWEL, position=0)
        
        # Check required attributes exist
        for attr_name in token_test['attributes']:
            assert hasattr(token, attr_name), f"Missing token attribute: {attr_name}"
        
        # Check required methods exist
        for method_name in token_test['methods']:
            assert hasattr(token, method_name), f"Missing token method: {method_name}"
            assert callable(getattr(token, method_name)), f"Token method {method_name} not callable"
        
        # Test method behavior
        token.add_tag("test")
        assert token.has_tag("test"), "has_tag() behavior regression"
        
        token.set_meta("key", "value")
        assert token.get_meta("key") == "value", "get_meta() behavior regression"


class TestBackwardCompatibilityRegression:
    """Test backward compatibility with previous versions."""
    
    def test_token_kind_enum_stability(self):
        """Test that TokenKind enum values remain stable."""
        # These values should never change to maintain compatibility
        expected_values = {
            TokenKind.VOWEL: "VOWEL",
            TokenKind.CONSONANT: "CONSONANT", 
            TokenKind.MARKER: "MARKER",
            TokenKind.OTHER: "OTHER"
        }
        
        for kind, expected_value in expected_values.items():
            assert kind.value == expected_value, f"TokenKind enum value changed: {kind} = {kind.value}, expected {expected_value}"
    
    def test_rule_type_enum_stability(self):
        """Test that RuleType enum values remain stable."""
        expected_values = {
            RuleType.SUTRA: "SUTRA",
            RuleType.PARIBHASA: "PARIBHASA",
            RuleType.ADHIKARA: "ADHIKARA",
            RuleType.ANUVRTI: "ANUVRTI"
        }
        
        for rule_type, expected_value in expected_values.items():
            assert rule_type.value == expected_value, f"RuleType enum value changed: {rule_type} = {rule_type.value}, expected {expected_value}"
    
    def test_sutra_reference_format_stability(self):
        """Test that SutraReference format remains stable."""
        ref = SutraReference(1, 2, 3)
        
        # String format should remain stable
        assert str(ref) == "1.2.3", f"SutraReference format changed: {str(ref)}"
        
        # Comparison should work consistently
        ref2 = SutraReference(1, 2, 4)
        assert ref < ref2, "SutraReference comparison regression"
        
        # Hash should be consistent
        assert hash(ref) == hash(SutraReference(1, 2, 3)), "SutraReference hash regression"
    
    def test_processing_result_structure_stability(self, rule_engine, tokenizer):
        """Test that processing result structure remains stable."""
        tokens = tokenizer.tokenize("test")
        result = rule_engine.process(tokens, max_passes=1)
        
        # Core attributes should exist and have correct types
        assert isinstance(result.input_text, str), "input_text type regression"
        assert isinstance(result.input_tokens, list), "input_tokens type regression"
        assert isinstance(result.output_tokens, list), "output_tokens type regression"
        assert isinstance(result.converged, bool), "converged type regression"
        assert isinstance(result.passes, int), "passes type regression"
        assert isinstance(result.traces, list), "traces type regression"
        assert isinstance(result.errors, list), "errors type regression"
        assert isinstance(result.statistics, dict), "statistics type regression"
        
        # Methods should return expected types
        assert isinstance(result.get_output_text(), str), "get_output_text() return type regression"
        assert isinstance(result.get_transformation_summary(), dict), "get_transformation_summary() return type regression"
        assert isinstance(result.get_rules_applied(), set), "get_rules_applied() return type regression"


class TestPerformanceRegression:
    """Test that performance doesn't regress significantly."""
    
    def test_tokenization_performance_regression(self, tokenizer):
        """Test that tokenization performance doesn't regress."""
        # Define performance baselines (in seconds)
        baselines = {
            'simple_text': 0.001,      # 1ms for simple text
            'complex_text': 0.01,      # 10ms for complex text
            'large_text': 0.1          # 100ms for large text
        }
        
        test_cases = [
            ('simple_text', 'rama'),
            ('complex_text', 'क्ष्म्य्व्र्त्न्प्फ्ब्भ्म्य्र्ल्व्श्ष्स्ह्'),
            ('large_text', 'रामायणमहाकाव्यस्य आदिकविः वाल्मीकिः ' * 50)
        ]
        
        for test_name, text in test_cases:
            # Warm up
            for _ in range(5):
                tokenizer.tokenize(text)
            
            # Measure performance
            iterations = 10 if test_name != 'large_text' else 1
            start_time = time.time()
            for _ in range(iterations):
                tokens = tokenizer.tokenize(text)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            baseline = baselines[test_name]
            
            assert avg_time < baseline * 2, \
                f"Tokenization performance regression for {test_name}: {avg_time:.4f}s > {baseline * 2:.4f}s (2x baseline)"
            
            # Verify tokens were created
            assert len(tokens) > 0, f"No tokens created for {test_name}"
    
    def test_processing_performance_regression(self, rule_engine, tokenizer):
        """Test that processing performance doesn't regress."""
        # Define performance baselines (in seconds)
        baselines = {
            'simple_processing': 0.1,    # 100ms for simple processing
            'medium_processing': 0.5,    # 500ms for medium processing
            'complex_processing': 2.0    # 2s for complex processing
        }
        
        test_cases = [
            ('simple_processing', 'a+i', 3),
            ('medium_processing', 'rāma+iti+gacchati', 5),
            ('complex_processing', 'dharma+artha+kāma+mokṣa:GEN:PL:ACC', 10)
        ]
        
        for test_name, text, max_passes in test_cases:
            tokens = tokenizer.tokenize(text)
            
            # Warm up
            for _ in range(2):
                rule_engine.process(tokens.copy(), max_passes=max_passes)
            
            # Measure performance
            start_time = time.time()
            result = rule_engine.process(tokens.copy(), max_passes=max_passes)
            end_time = time.time()
            
            processing_time = end_time - start_time
            baseline = baselines[test_name]
            
            assert processing_time < baseline * 2, \
                f"Processing performance regression for {test_name}: {processing_time:.4f}s > {baseline * 2:.4f}s (2x baseline)"
            
            # Verify processing completed
            assert isinstance(result, PaniniEngineResult), f"Invalid result type for {test_name}"
    
    def test_memory_usage_regression(self, rule_engine, tokenizer):
        """Test that memory usage doesn't regress significantly."""
        import psutil
        import gc
        
        # Measure baseline memory
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform operations that should not cause memory leaks
        for i in range(50):
            text = f"memory test {i} with various tokens"
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=3)
            
            # Verify operations completed
            assert len(tokens) > 0
            assert isinstance(result, PaniniEngineResult)
        
        # Force garbage collection
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        memory_growth = final_memory - initial_memory
        max_acceptable_growth = 50 * 1024 * 1024  # 50MB
        
        assert memory_growth < max_acceptable_growth, \
            f"Memory usage regression: {memory_growth} bytes growth > {max_acceptable_growth} bytes"


class TestDataIntegrityRegression:
    """Test that data processing integrity is maintained."""
    
    def test_deterministic_output_regression(self, rule_engine, tokenizer):
        """Test that processing produces deterministic outputs."""
        test_inputs = [
            "rama",
            "a+i", 
            "rāma+iti",
            "test:GEN"
        ]
        
        for input_text in test_inputs:
            tokens = tokenizer.tokenize(input_text)
            
            # Process multiple times
            results = []
            for _ in range(5):
                # Create fresh token copies
                token_copies = [Token(t.text, t.kind, position=t.position) for t in tokens]
                result = rule_engine.process(token_copies, max_passes=5)
                results.append(result.get_output_text())
            
            # All results should be identical
            unique_results = set(results)
            assert len(unique_results) == 1, \
                f"Non-deterministic output regression for '{input_text}': {unique_results}"
    
    def test_input_preservation_regression(self, rule_engine, tokenizer):
        """Test that input tokens are not modified during processing."""
        input_text = "rāma+iti"
        original_tokens = tokenizer.tokenize(input_text)
        
        # Create checksums of original tokens
        original_checksums = []
        for token in original_tokens:
            token_data = f"{token.text}|{token.kind.value}|{token.position}"
            checksum = hashlib.md5(token_data.encode()).hexdigest()
            original_checksums.append(checksum)
        
        # Process tokens
        result = rule_engine.process(original_tokens, max_passes=5)
        
        # Verify original tokens are unchanged
        for i, token in enumerate(original_tokens):
            token_data = f"{token.text}|{token.kind.value}|{token.position}"
            checksum = hashlib.md5(token_data.encode()).hexdigest()
            assert checksum == original_checksums[i], \
                f"Input token {i} was modified during processing"
        
        # Verify result has input tokens preserved
        assert len(result.input_tokens) == len(original_tokens), \
            "Input tokens count changed in result"
    
    def test_trace_completeness_regression(self, rule_engine, tokenizer):
        """Test that trace information remains complete."""
        tokens = tokenizer.tokenize("a+i")
        result = rule_engine.process(tokens, max_passes=5)
        
        # Verify trace structure
        assert isinstance(result.traces, list), "Traces should be a list"
        
        for i, trace in enumerate(result.traces):
            # Each trace should have required fields
            assert hasattr(trace, 'pass_number'), f"Trace {i} missing pass_number"
            assert hasattr(trace, 'tokens_before'), f"Trace {i} missing tokens_before"
            assert hasattr(trace, 'tokens_after'), f"Trace {i} missing tokens_after"
            assert hasattr(trace, 'transformations'), f"Trace {i} missing transformations"
            
            # Pass numbers should be sequential
            assert trace.pass_number == i + 1, f"Trace {i} has incorrect pass_number: {trace.pass_number}"
            
            # Tokens should be valid
            assert isinstance(trace.tokens_before, list), f"Trace {i} tokens_before not a list"
            assert isinstance(trace.tokens_after, list), f"Trace {i} tokens_after not a list"
            
            # Transformations should be valid
            assert isinstance(trace.transformations, list), f"Trace {i} transformations not a list"
            
            for j, transformation in enumerate(trace.transformations):
                assert hasattr(transformation, 'rule_name'), f"Transformation {i}.{j} missing rule_name"
                assert hasattr(transformation, 'rule_id'), f"Transformation {i}.{j} missing rule_id"
                assert hasattr(transformation, 'timestamp'), f"Transformation {i}.{j} missing timestamp"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])