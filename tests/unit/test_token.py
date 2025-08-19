"""
Tests for the Token class and TokenKind enum.
"""

import pytest
from sanskrit_rewrite_engine.token import Token, TokenKind


class TestTokenKind:
    
    def test_token_kind_values(self):
        """Test that TokenKind enum has expected values."""
        assert TokenKind.VOWEL.value == "VOWEL"
        assert TokenKind.CONSONANT.value == "CONSONANT"
        assert TokenKind.MARKER.value == "MARKER"
        assert TokenKind.OTHER.value == "OTHER"
    
    def test_token_kind_membership(self):
        """Test TokenKind enum membership."""
        all_kinds = list(TokenKind)
        assert len(all_kinds) == 4
        assert TokenKind.VOWEL in all_kinds
        assert TokenKind.CONSONANT in all_kinds
        assert TokenKind.MARKER in all_kinds
        assert TokenKind.OTHER in all_kinds


class TestToken:
    
    def test_token_creation(self):
        """Test basic token creation."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        
        assert token.text == "a"
        assert token.kind == TokenKind.VOWEL
        assert isinstance(token.tags, set)
        assert isinstance(token.meta, dict)
        assert token.position is None
    
    def test_token_with_position(self):
        """Test token creation with position."""
        token = Token(text="k", kind=TokenKind.CONSONANT, position=5)
        
        assert token.position == 5
    
    def test_token_with_initial_tags_and_meta(self):
        """Test token creation with initial tags and metadata."""
        initial_tags = {"long", "vowel"}
        initial_meta = {"length": "long", "quality": "back"}
        
        token = Token(
            text="ā",
            kind=TokenKind.VOWEL,
            tags=initial_tags,
            meta=initial_meta,
            position=0
        )
        
        assert token.tags == initial_tags
        assert token.meta == initial_meta
    
    def test_has_tag(self):
        """Test has_tag method."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        
        # Initially no tags
        assert not token.has_tag("long")
        
        # Add a tag
        token.add_tag("long")
        assert token.has_tag("long")
        assert not token.has_tag("short")
    
    def test_add_tag(self):
        """Test add_tag method."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        
        # Add single tag
        token.add_tag("short")
        assert "short" in token.tags
        
        # Add multiple tags
        token.add_tag("front")
        token.add_tag("low")
        assert "short" in token.tags
        assert "front" in token.tags
        assert "low" in token.tags
        
        # Adding same tag twice should not duplicate
        token.add_tag("short")
        # Sets automatically handle duplicates
        assert len([tag for tag in token.tags if tag == "short"]) == 1
    
    def test_remove_tag(self):
        """Test remove_tag method."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        
        # Add tags
        token.add_tag("short")
        token.add_tag("front")
        assert token.has_tag("short")
        assert token.has_tag("front")
        
        # Remove one tag
        token.remove_tag("short")
        assert not token.has_tag("short")
        assert token.has_tag("front")
        
        # Remove non-existent tag should not raise error
        token.remove_tag("nonexistent")
        assert token.has_tag("front")  # Other tags unaffected
    
    def test_get_meta(self):
        """Test get_meta method."""
        token = Token(text="k", kind=TokenKind.CONSONANT)
        
        # Get non-existent key with default
        assert token.get_meta("place") is None
        assert token.get_meta("place", "unknown") == "unknown"
        
        # Set and get metadata
        token.set_meta("place", "velar")
        assert token.get_meta("place") == "velar"
        assert token.get_meta("place", "unknown") == "velar"  # Default ignored when key exists
    
    def test_set_meta(self):
        """Test set_meta method."""
        token = Token(text="k", kind=TokenKind.CONSONANT)
        
        # Set metadata
        token.set_meta("place", "velar")
        token.set_meta("manner", "stop")
        token.set_meta("voice", "voiceless")
        
        assert token.meta["place"] == "velar"
        assert token.meta["manner"] == "stop"
        assert token.meta["voice"] == "voiceless"
        
        # Overwrite existing metadata
        token.set_meta("place", "palatal")
        assert token.meta["place"] == "palatal"
        assert token.meta["manner"] == "stop"  # Other values unchanged
    
    def test_str_representation(self):
        """Test string representation of token."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        str_repr = str(token)
        
        assert "a" in str_repr
        assert "VOWEL" in str_repr
        assert str_repr.startswith("Token(")
    
    def test_repr_representation(self):
        """Test repr representation of token."""
        token = Token(text="k", kind=TokenKind.CONSONANT, position=3)
        token.add_tag("velar")
        
        repr_str = repr(token)
        
        assert "k" in repr_str
        assert "CONSONANT" in repr_str
        assert "3" in repr_str
        assert "velar" in repr_str
        assert repr_str.startswith("Token(")
    
    def test_token_equality(self):
        """Test token equality comparison."""
        token1 = Token(text="a", kind=TokenKind.VOWEL, position=0)
        token2 = Token(text="a", kind=TokenKind.VOWEL, position=0)
        token3 = Token(text="b", kind=TokenKind.CONSONANT, position=0)
        
        # Note: dataclass equality is based on all fields
        assert token1 == token2
        assert token1 != token3
    
    def test_token_with_complex_metadata(self):
        """Test token with complex metadata structures."""
        token = Token(text="ai", kind=TokenKind.VOWEL)
        
        # Set complex metadata
        token.set_meta("components", ["a", "i"])
        token.set_meta("phonetic_features", {
            "height": "mid",
            "backness": "front",
            "roundness": "unrounded"
        })
        token.set_meta("frequency", 0.15)
        
        # Retrieve complex metadata
        components = token.get_meta("components")
        assert components == ["a", "i"]
        
        features = token.get_meta("phonetic_features")
        assert features["height"] == "mid"
        assert features["backness"] == "front"
        
        frequency = token.get_meta("frequency")
        assert frequency == 0.15
    
    def test_token_immutable_text_and_kind(self):
        """Test that text and kind are set at creation."""
        token = Token(text="original", kind=TokenKind.VOWEL)
        
        # These should be set and accessible
        assert token.text == "original"
        assert token.kind == TokenKind.VOWEL
        
        # Note: Since we're using dataclass, fields are mutable by default
        # If immutability is needed, we'd need frozen=True in the dataclass decorator
    
    def test_token_tags_are_set(self):
        """Test that tags are properly managed as a set."""
        token = Token(text="a", kind=TokenKind.VOWEL)
        
        # Add duplicate tags
        token.add_tag("short")
        token.add_tag("front")
        token.add_tag("short")  # Duplicate
        
        # Should only have unique tags
        assert len(token.tags) == 2
        assert "short" in token.tags
        assert "front" in token.tags
    
    def test_token_metadata_types(self):
        """Test that metadata can store various types."""
        token = Token(text="test", kind=TokenKind.OTHER)
        
        # Store different types
        token.set_meta("string_value", "hello")
        token.set_meta("int_value", 42)
        token.set_meta("float_value", 3.14)
        token.set_meta("bool_value", True)
        token.set_meta("list_value", [1, 2, 3])
        token.set_meta("dict_value", {"key": "value"})
        token.set_meta("none_value", None)
        
        # Retrieve and verify types
        assert isinstance(token.get_meta("string_value"), str)
        assert isinstance(token.get_meta("int_value"), int)
        assert isinstance(token.get_meta("float_value"), float)
        assert isinstance(token.get_meta("bool_value"), bool)
        assert isinstance(token.get_meta("list_value"), list)
        assert isinstance(token.get_meta("dict_value"), dict)
        assert token.get_meta("none_value") is None