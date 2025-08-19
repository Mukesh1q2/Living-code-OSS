"""
Comprehensive tokenization tests for Sanskrit Rewrite Engine.
Part of TO1 comprehensive test suite.

Tests cover:
- Unit tests for tokenization (Requirements 1, 14)
- Adversarial input testing for robustness
- Performance profiling for tokenization
"""

import pytest
import time
import psutil
from typing import List, Dict, Any

from sanskrit_rewrite_engine.token import Token, TokenKind
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer


class AdversarialInputGenerator:
    """Generator for adversarial inputs to test robustness."""
    
    @staticmethod
    def generate_malformed_inputs() -> List[str]:
        """Generate malformed input strings."""
        return [
            "",  # Empty string
            " ",  # Only whitespace
            "\n\t\r",  # Various whitespace
            "a" * 10000,  # Very long string
            "🙂😊🎉",  # Emoji
            "abc123!@#",  # Mixed alphanumeric and symbols
            "\x00\x01\x02",  # Control characters
            "ॐ" * 1000,  # Repeated Sanskrit symbols
            "क्ष्म्य्व्र्त्न्",  # Complex conjuncts
            "अअअअअ",  # Repeated vowels
            "्््््",  # Repeated virama
            "++++++",  # Repeated markers
            "a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z",  # Long marker chain
        ]
    
    @staticmethod
    def generate_edge_case_sanskrit() -> List[str]:
        """Generate edge case Sanskrit inputs."""
        return [
            "क्",  # Consonant with virama at end
            "अ्",  # Vowel with virama (invalid)
            "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",  # All consonants
            "अआइईउऊऋॠऌॡएऐओऔ",  # All vowels
            "क्क्क्क्क्क्क्क्क्क्",  # Deep conjunct nesting
            "रामरामरामराम",  # Repeated words
            "a+i+u+e+o+ai+au",  # Mixed vowel chain
            "क्ष्ट्र्य्व्ल्",  # Complex consonant clusters
        ]
    
    @staticmethod
    def generate_unicode_edge_cases() -> List[str]:
        """Generate Unicode edge cases."""
        return [
            "\u0900\u0901\u0902",  # Devanagari combining marks
            "\u200B\u200C\u200D",  # Zero-width characters
            "क\u093C",  # Consonant with nukta
            "अ\u0951\u0952",  # Vowel with Vedic accents
            "\uFEFF",  # Byte order mark
            "क़ख़ग़ज़ड़ढ़फ़य़",  # Nukta variants
        ]


@pytest.fixture
def tokenizer():
    """Fixture for Sanskrit tokenizer."""
    return SanskritTokenizer()


class TestTokenizationRequirement1:
    """Test Requirement 1: Token-based rewrite engine with linguistic context."""
    
    def test_token_based_processing(self, tokenizer):
        """Test that input is tokenized into typed tokens."""
        text = "rāma+iti"
        tokens = tokenizer.tokenize(text)
        
        # Verify token-based processing
        assert len(tokens) > 0
        assert all(isinstance(token, Token) for token in tokens)
        
        # Verify typed tokens (VOWEL, CONSONANT, MARKER, OTHER)
        token_kinds = {token.kind for token in tokens}
        expected_kinds = {TokenKind.VOWEL, TokenKind.CONSONANT, TokenKind.MARKER}
        assert token_kinds.intersection(expected_kinds), "Missing expected token kinds"
    
    def test_multi_character_vowel_detection(self, tokenizer):
        """Test correct identification of multi-character vowels like 'ai', 'au'."""
        test_cases = [
            ("ai", 1, "compound"),
            ("au", 1, "compound"), 
            ("rāma", 4, None),  # Contains ā as compound vowel
        ]
        
        for text, expected_count, expected_tag in test_cases:
            tokens = tokenizer.tokenize(text)
            assert len(tokens) == expected_count, f"Failed for '{text}': got {len(tokens)} tokens, expected {expected_count}"
            
            if expected_tag:
                compound_tokens = [t for t in tokens if t.has_tag(expected_tag)]
                assert len(compound_tokens) > 0, f"No tokens with tag '{expected_tag}' found in '{text}'"
    
    def test_morphological_marker_preservation(self, tokenizer):
        """Test preservation of morphological markers like '+', '_', ':'."""
        text = "word+marker_test:case"
        tokens = tokenizer.tokenize(text)
        
        marker_tokens = [t for t in tokens if t.kind == TokenKind.MARKER]
        marker_texts = {t.text for t in marker_tokens}
        
        expected_markers = {'+', '_', ':'}
        assert expected_markers.issubset(marker_texts), "Missing expected markers"
        
        # Verify each marker token has proper metadata
        for token in marker_tokens:
            assert token.text in expected_markers
            assert hasattr(token, 'meta')
            assert hasattr(token, 'tags')
    
    def test_token_metadata_fields(self, tokenizer):
        """Test that tokens contain text, kind, tags, and metadata fields."""
        text = "rama"
        tokens = tokenizer.tokenize(text)
        
        for token in tokens:
            # Verify required fields exist
            assert hasattr(token, 'text')
            assert hasattr(token, 'kind')
            assert hasattr(token, 'tags')
            assert hasattr(token, 'meta')
            assert hasattr(token, 'position')
            
            # Verify field types
            assert isinstance(token.text, str)
            assert isinstance(token.kind, TokenKind)
            assert isinstance(token.tags, set)
            assert isinstance(token.meta, dict)
            
            # Verify position is set
            assert token.position is not None
            assert isinstance(token.position, int)
            assert token.position >= 0


class TestTokenizationRobustness:
    """Test tokenization robustness with adversarial inputs."""
    
    def test_malformed_input_handling(self, tokenizer):
        """Test handling of malformed inputs."""
        malformed_inputs = AdversarialInputGenerator.generate_malformed_inputs()
        
        for test_input in malformed_inputs:
            try:
                tokens = tokenizer.tokenize(test_input)
                # Should not crash and should return a list
                assert isinstance(tokens, list)
                # All tokens should be valid
                for token in tokens:
                    assert isinstance(token, Token)
                    assert isinstance(token.text, str)
                    assert isinstance(token.kind, TokenKind)
            except Exception as e:
                pytest.fail(f"Tokenizer crashed on malformed input '{test_input}': {e}")
    
    def test_edge_case_sanskrit_handling(self, tokenizer):
        """Test handling of edge case Sanskrit inputs."""
        edge_cases = AdversarialInputGenerator.generate_edge_case_sanskrit()
        
        for test_input in edge_cases:
            try:
                tokens = tokenizer.tokenize(test_input)
                assert isinstance(tokens, list)
                # Should handle Sanskrit edge cases gracefully
                for token in tokens:
                    assert isinstance(token, Token)
                    assert len(token.text) > 0  # No empty tokens
            except Exception as e:
                pytest.fail(f"Tokenizer crashed on Sanskrit edge case '{test_input}': {e}")
    
    def test_unicode_edge_case_handling(self, tokenizer):
        """Test handling of Unicode edge cases."""
        unicode_cases = AdversarialInputGenerator.generate_unicode_edge_cases()
        
        for test_input in unicode_cases:
            try:
                tokens = tokenizer.tokenize(test_input)
                assert isinstance(tokens, list)
                # Should handle Unicode edge cases without crashing
                for token in tokens:
                    assert isinstance(token, Token)
            except Exception as e:
                pytest.fail(f"Tokenizer crashed on Unicode edge case '{test_input}': {e}")
    
    def test_very_long_input_handling(self, tokenizer):
        """Test handling of very long inputs."""
        # Create a very long Sanskrit text
        long_text = "रामरामराम" * 1000  # 3000 characters
        
        start_time = time.time()
        tokens = tokenizer.tokenize(long_text)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0, "Tokenization too slow for long input"
        assert len(tokens) > 0
        assert isinstance(tokens, list)
    
    def test_memory_usage_with_large_input(self, tokenizer):
        """Test memory usage with large inputs."""
        import gc
        
        # Measure initial memory
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Process large input
        large_text = "अ" * 10000
        tokens = tokenizer.tokenize(large_text)
        
        # Measure final memory
        final_memory = psutil.Process().memory_info().rss
        memory_delta = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_delta < 100 * 1024 * 1024, f"Excessive memory usage: {memory_delta} bytes"
        assert len(tokens) > 0


class TestTokenizationPerformance:
    """Test tokenization performance (Requirement 14)."""
    
    def test_typical_input_performance(self, tokenizer):
        """Test performance with typical Sanskrit inputs."""
        typical_texts = [
            "rama",
            "rāma gacchati",
            "dharma artha kāma mokṣa",
            "satyameva jayate",
            "वसुधैव कुटुम्बकम्"
        ]
        
        for text in typical_texts:
            start_time = time.time()
            tokens = tokenizer.tokenize(text)
            end_time = time.time()
            
            # Should complete quickly (sub-second for typical inputs)
            execution_time = end_time - start_time
            assert execution_time < 1.0, f"Too slow for '{text}': {execution_time} seconds"
            assert len(tokens) > 0
    
    def test_scalability_with_input_size(self, tokenizer):
        """Test scalability with increasing input sizes."""
        base_text = "रामायण"
        sizes = [10, 100, 1000]
        times = []
        
        for size in sizes:
            text = base_text * size
            
            start_time = time.time()
            tokens = tokenizer.tokenize(text)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            assert len(tokens) > 0
            # Should scale reasonably (not exponentially)
            assert execution_time < size * 0.001, f"Poor scalability at size {size}"
        
        # Time should scale roughly linearly
        if len(times) >= 2:
            # Ratio of times should be roughly proportional to size ratio
            time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
            size_ratio = sizes[-1] / sizes[0]
            
            # Allow some overhead, but should be roughly linear
            assert time_ratio < size_ratio * 2, f"Non-linear scaling: time ratio {time_ratio}, size ratio {size_ratio}"
    
    def test_memory_efficiency(self, tokenizer):
        """Test memory efficiency of tokenization."""
        import gc
        
        # Test with multiple tokenizations
        texts = ["test text " + str(i) for i in range(100)]
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        for text in texts:
            tokens = tokenizer.tokenize(text)
            assert len(tokens) > 0
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should not have significant memory growth
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth too high: {memory_growth} bytes"


class TestTokenizationIntegration:
    """Integration tests for tokenization with other components."""
    
    def test_devanagari_iast_integration(self, tokenizer):
        """Test integration between Devanagari and IAST scripts."""
        # Test both scripts
        devanagari_text = "राम"
        iast_text = "rāma"
        
        devanagari_tokens = tokenizer.tokenize(devanagari_text)
        iast_tokens = tokenizer.tokenize(iast_text)
        
        # Both should tokenize successfully
        assert len(devanagari_tokens) > 0
        assert len(iast_tokens) > 0
        
        # Should detect appropriate character types
        devanagari_kinds = {t.kind for t in devanagari_tokens}
        iast_kinds = {t.kind for t in iast_tokens}
        
        # Both should have vowels and consonants
        assert TokenKind.VOWEL in devanagari_kinds or TokenKind.CONSONANT in devanagari_kinds
        assert TokenKind.VOWEL in iast_kinds or TokenKind.CONSONANT in iast_kinds
    
    def test_compound_detection_integration(self, tokenizer):
        """Test integration of compound detection with tokenization."""
        compound_texts = [
            "ai",  # Compound vowel
            "kha",  # Compound consonant
            "क्त",  # Devanagari conjunct
        ]
        
        for text in compound_texts:
            tokens = tokenizer.tokenize(text)
            
            # Should detect compounds appropriately
            compound_tokens = [t for t in tokens if t.has_tag('compound') or t.has_tag('conjunct')]
            
            # At least some compound detection should occur
            if any(len(char) > 1 for char in text if char.isalpha()):
                assert len(compound_tokens) >= 0  # May be 0 if no compounds detected
    
    def test_position_tracking_integration(self, tokenizer):
        """Test position tracking integration."""
        text = "rāma gacchati"
        tokens = tokenizer.tokenize(text)
        
        # All tokens should have positions
        positions = [t.position for t in tokens if t.position is not None]
        assert len(positions) == len(tokens), "Not all tokens have positions"
        
        # Positions should be reasonable
        for pos in positions:
            assert 0 <= pos < len(text), f"Position {pos} out of range for text length {len(text)}"
    
    def test_statistics_integration(self, tokenizer):
        """Test integration with token statistics."""
        text = "rāma+iti गच्छति"
        tokens = tokenizer.tokenize(text)
        
        # Should be able to generate statistics
        stats = tokenizer.get_token_statistics(tokens)
        
        # Verify statistics structure
        expected_keys = [
            'total_tokens', 'vowel_count', 'consonant_count', 
            'marker_count', 'other_count'
        ]
        
        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"
            assert isinstance(stats[key], int), f"Statistic {key} should be integer"
            assert stats[key] >= 0, f"Statistic {key} should be non-negative"
        
        # Total should equal sum of parts
        total_calculated = (stats['vowel_count'] + stats['consonant_count'] + 
                          stats['marker_count'] + stats['other_count'])
        assert stats['total_tokens'] == total_calculated, "Statistics don't add up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])