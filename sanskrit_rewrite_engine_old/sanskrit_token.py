"""
Token system for Sanskrit text processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Dict, Any, Optional


class TokenKind(Enum):
    """Types of tokens in Sanskrit text processing."""
    VOWEL = "VOWEL"
    CONSONANT = "CONSONANT"
    MARKER = "MARKER"
    OTHER = "OTHER"


@dataclass
class Token:
    """
    A token representing a unit of Sanskrit text with linguistic metadata.
    
    Attributes:
        text: The actual text content of the token
        kind: The linguistic type of the token
        tags: Set of descriptive tags for the token
        meta: Dictionary of metadata associated with the token
        position: Original position in the input text for tracing
    """
    text: str
    kind: TokenKind
    tags: Set[str] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)
    position: Optional[int] = None
    
    def has_tag(self, tag: str) -> bool:
        """Check if the token has a specific tag."""
        return tag in self.tags
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the token."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the token."""
        self.tags.discard(tag)
    
    def get_meta(self, key: str, default=None) -> Any:
        """Get metadata value by key."""
        return self.meta.get(key, default)
    
    def set_meta(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        self.meta[key] = value
    
    def __str__(self) -> str:
        return f"Token({self.text}, {self.kind.value})"
    
    def __repr__(self) -> str:
        return f"Token(text='{self.text}', kind={self.kind}, tags={self.tags}, position={self.position})"