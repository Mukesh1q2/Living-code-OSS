"""
Sanskrit tokenizer with support for Devanagari and IAST scripts.
"""

from typing import List, Set, Dict, Any, Optional, Tuple
import re
from .sanskrit_token import Token, TokenKind
from .transliterator import DevanagariIASTTransliterator


class SanskritTokenizer:
    """
    Advanced tokenizer for Sanskrit text supporting both Devanagari and IAST scripts.
    
    Features:
    - Multi-character vowel detection (ai, au, etc.)
    - Compound consonant handling
    - Morphological marker preservation
    - Position tracking for derivation steps
    - Sandhi boundary identification
    - Vedic variant support
    """
    
    def __init__(self, vowels: Optional[Set[str]] = None, consonants: Optional[Set[str]] = None):
        """
        Initialize the tokenizer with character sets.
        
        Args:
            vowels: Set of vowel characters (defaults to Sanskrit vowels)
            consonants: Set of consonant characters (defaults to Sanskrit consonants)
        """
        self.transliterator = DevanagariIASTTransliterator()
        
        # Default Sanskrit vowels (IAST)
        self.vowels = vowels or {
            'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ',
            'e', 'ai', 'o', 'au',
            # Devanagari vowels
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ',
            'ए', 'ऐ', 'ओ', 'औ',
            # Vowel marks
            'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ'
        }
        
        # Default Sanskrit consonants (IAST)
        self.consonants = consonants or {
            'k', 'kh', 'g', 'gh', 'ṅ',
            'c', 'ch', 'j', 'jh', 'ñ',
            'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
            't', 'th', 'd', 'dh', 'n',
            'p', 'ph', 'b', 'bh', 'm',
            'y', 'r', 'l', 'v',
            'ś', 'ṣ', 's', 'h',
            # Devanagari consonants
            'क', 'ख', 'ग', 'घ', 'ङ',
            'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण',
            'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म',
            'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह'
        }
        
        # Morphological markers
        self.markers = {'+', '_', ':', '-', '=', '।', '॥', '|', '||'}
        
        # Special characters
        self.special_chars = {'ं', 'ः', '्', 'ॐ', 'ṃ', 'ḥ'}
        
        # Compound vowel patterns (longest first for greedy matching)
        self.compound_vowels = ['ai', 'au', 'ā', 'ī', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ']
        
        # Compound consonant patterns
        self.compound_consonants = ['kh', 'gh', 'ch', 'jh', 'ṭh', 'ḍh', 'th', 'dh', 'ph', 'bh']
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient tokenization."""
        # Pattern for whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Pattern for numbers
        self.number_pattern = re.compile(r'[0-9०-९]+')
        
        # Pattern for punctuation
        self.punct_pattern = re.compile(r'[।॥|,.;:!?()[\]{}"\'-]')
        
        # Pattern for Devanagari conjuncts
        self.conjunct_pattern = re.compile(r'[क-ह]्[क-ह]')
        
        # Pattern for morphological markers
        marker_chars = ''.join(re.escape(m) for m in self.markers)
        self.marker_pattern = re.compile(f'[{marker_chars}]')
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize Sanskrit text into typed tokens.
        
        Args:
            text: Input Sanskrit text (Devanagari or IAST)
            
        Returns:
            List of Token objects with linguistic metadata
        """
        if not text.strip():
            return []
        
        # Detect script and normalize if needed
        script = self.transliterator.detect_script(text)
        
        # Pre-process text to handle compounds and boundaries
        preprocessed_text, position_map = self._preprocess_text(text)
        
        tokens = []
        i = 0
        original_position = 0
        
        while i < len(preprocessed_text):
            char = preprocessed_text[i]
            
            # Skip whitespace but track position
            if self.whitespace_pattern.match(char):
                i += 1
                original_position += 1
                continue
            
            # Try to match compound patterns first (greedy)
            token, consumed = self._match_compound_token(preprocessed_text, i, original_position)
            
            if token:
                tokens.append(token)
                i += consumed
                original_position += consumed
            else:
                # Single character token
                token = self._create_single_char_token(char, original_position)
                tokens.append(token)
                i += 1
                original_position += 1
        
        # Post-process tokens for additional metadata
        self._add_contextual_metadata(tokens, text)
        
        return tokens
    
    def _preprocess_text(self, text: str) -> Tuple[str, Dict[int, int]]:
        """
        Preprocess text to handle special cases and create position mapping.
        
        Returns:
            Tuple of (preprocessed_text, position_mapping)
        """
        # For now, return text as-is with identity mapping
        # This can be extended for more complex preprocessing
        position_map = {i: i for i in range(len(text))}
        return text, position_map
    
    def _match_compound_token(self, text: str, start: int, position: int) -> Tuple[Optional[Token], int]:
        """
        Try to match compound tokens (multi-character vowels, consonants, etc.).
        
        Returns:
            Tuple of (Token or None, characters_consumed)
        """
        # Try compound vowels first (longest match)
        for compound in self.compound_vowels:
            if text[start:start + len(compound)] == compound:
                token = Token(
                    text=compound,
                    kind=TokenKind.VOWEL,
                    position=position
                )
                token.add_tag('compound')
                token.set_meta('compound_type', 'vowel')
                
                # Add length and vocalic tags for compound vowels
                if compound in ['ā', 'ī', 'ū', 'ṝ', 'ḹ']:
                    token.add_tag('long')
                elif compound in ['a', 'i', 'u']:
                    token.add_tag('short')
                    
                if compound in ['ṛ', 'ṝ', 'ḷ', 'ḹ']:
                    token.add_tag('vocalic')
                    
                return token, len(compound)
        
        # Try compound consonants
        for compound in self.compound_consonants:
            if text[start:start + len(compound)] == compound:
                token = Token(
                    text=compound,
                    kind=TokenKind.CONSONANT,
                    position=position
                )
                token.add_tag('compound')
                token.set_meta('compound_type', 'consonant')
                return token, len(compound)
        
        # Try Devanagari conjuncts
        conjunct_match = self.conjunct_pattern.match(text, start)
        if conjunct_match:
            conjunct_text = conjunct_match.group()
            token = Token(
                text=conjunct_text,
                kind=TokenKind.CONSONANT,
                position=position
            )
            token.add_tag('conjunct')
            token.set_meta('conjunct_type', 'devanagari')
            return token, len(conjunct_text)
        
        # Try morphological markers
        if start < len(text) and text[start] in self.markers:
            token = Token(
                text=text[start],
                kind=TokenKind.MARKER,
                position=position
            )
            token.add_tag('morphological')
            return token, 1
        
        return None, 0
    
    def _create_single_char_token(self, char: str, position: int) -> Token:
        """Create a token for a single character."""
        kind = self._identify_token_kind(char)
        
        token = Token(
            text=char,
            kind=kind,
            position=position
        )
        
        # Add specific tags based on character type
        if kind == TokenKind.VOWEL:
            if char in ['ā', 'ī', 'ū', 'ṝ', 'ḹ', 'आ', 'ई', 'ऊ', 'ॠ', 'ॡ']:
                token.add_tag('long')
            elif char in ['a', 'i', 'u', 'अ', 'इ', 'उ']:
                token.add_tag('short')
                
            if char in ['ṛ', 'ṝ', 'ḷ', 'ḹ', 'ऋ', 'ॠ', 'ऌ', 'ॡ']:
                token.add_tag('vocalic')
        
        elif kind == TokenKind.CONSONANT:
            # Add phonetic classification tags
            if char in ['k', 'kh', 'g', 'gh', 'ṅ', 'क', 'ख', 'ग', 'घ', 'ङ']:
                token.add_tag('velar')
            elif char in ['c', 'ch', 'j', 'jh', 'ñ', 'च', 'छ', 'ज', 'झ', 'ञ']:
                token.add_tag('palatal')
            elif char in ['ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ', 'ट', 'ठ', 'ड', 'ढ', 'ण']:
                token.add_tag('retroflex')
            elif char in ['t', 'th', 'd', 'dh', 'n', 'त', 'थ', 'द', 'ध', 'न']:
                token.add_tag('dental')
            elif char in ['p', 'ph', 'b', 'bh', 'm', 'प', 'फ', 'ब', 'भ', 'म']:
                token.add_tag('labial')
            elif char in ['y', 'r', 'l', 'v', 'य', 'र', 'ल', 'व']:
                token.add_tag('semivowel')
            elif char in ['ś', 'ṣ', 's', 'h', 'श', 'ष', 'स', 'ह']:
                token.add_tag('sibilant')
        
        elif kind == TokenKind.MARKER:
            if char in ['+', '_', ':']:
                token.add_tag('morphological')
            elif char in ['।', '॥', '|', '||']:
                token.add_tag('punctuation')
        
        return token
    
    def _identify_token_kind(self, text: str) -> TokenKind:
        """Identify the linguistic kind of a token."""
        if not text:
            return TokenKind.OTHER
        
        # Check if it's a vowel
        if any(v in text for v in self.vowels):
            return TokenKind.VOWEL
        
        # Check if it's a consonant
        if any(c in text for c in self.consonants):
            return TokenKind.CONSONANT
        
        # Check if it's a marker
        if any(m in text for m in self.markers):
            return TokenKind.MARKER
        
        # Check special characters
        if any(s in text for s in self.special_chars):
            if text in ['ं', 'ः', 'ṃ', 'ḥ']:
                return TokenKind.CONSONANT  # Anusvāra and Visarga are consonantal
            return TokenKind.OTHER
        
        # Numbers and punctuation
        if self.number_pattern.match(text) or self.punct_pattern.match(text):
            return TokenKind.OTHER
        
        return TokenKind.OTHER
    
    def _add_contextual_metadata(self, tokens: List[Token], original_text: str):
        """Add contextual metadata to tokens based on surrounding context."""
        # Identify sandhi boundaries
        sandhi_boundaries = self.transliterator.identify_sandhi_boundaries(original_text)
        
        for i, token in enumerate(tokens):
            # Mark tokens near sandhi boundaries
            if token.position and token.position in sandhi_boundaries:
                token.add_tag('sandhi_boundary')
            
            # Mark Vedic variants
            if self.transliterator.is_vedic_variant(token.text):
                token.add_tag('vedic')
            
            # Add positional context
            if i == 0:
                token.add_tag('word_initial')
            elif i == len(tokens) - 1:
                token.add_tag('word_final')
            else:
                token.add_tag('word_medial')
            
            # Add syllable information
            if token.kind == TokenKind.VOWEL:
                token.add_tag('syllable_nucleus')
            elif token.kind == TokenKind.CONSONANT:
                # Check if it's followed by a vowel
                if i + 1 < len(tokens) and tokens[i + 1].kind == TokenKind.VOWEL:
                    token.add_tag('syllable_onset')
                else:
                    token.add_tag('syllable_coda')
    
    def _handle_compound_vowels(self, text: str) -> List[Token]:
        """Handle compound vowels like 'ai', 'au'."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Try compound vowels first
            found_compound = False
            for compound in self.compound_vowels:
                if text[i:i + len(compound)] == compound:
                    token = Token(
                        text=compound,
                        kind=TokenKind.VOWEL,
                        position=i
                    )
                    token.add_tag('compound')
                    tokens.append(token)
                    i += len(compound)
                    found_compound = True
                    break
            
            if not found_compound:
                # Single character
                if text[i] in self.vowels:
                    token = Token(
                        text=text[i],
                        kind=TokenKind.VOWEL,
                        position=i
                    )
                    tokens.append(token)
                i += 1
        
        return tokens
    
    def _preserve_markers(self, text: str) -> List[str]:
        """Preserve morphological markers during tokenization."""
        # Split on markers but keep them
        parts = []
        current = ""
        
        for char in text:
            if char in self.markers:
                if current:
                    parts.append(current)
                    current = ""
                parts.append(char)
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def get_token_statistics(self, tokens: List[Token]) -> Dict[str, Any]:
        """Get statistics about the tokenized text."""
        stats = {
            'total_tokens': len(tokens),
            'vowel_count': sum(1 for t in tokens if t.kind == TokenKind.VOWEL),
            'consonant_count': sum(1 for t in tokens if t.kind == TokenKind.CONSONANT),
            'marker_count': sum(1 for t in tokens if t.kind == TokenKind.MARKER),
            'other_count': sum(1 for t in tokens if t.kind == TokenKind.OTHER),
            'compound_count': sum(1 for t in tokens if t.has_tag('compound')),
            'conjunct_count': sum(1 for t in tokens if t.has_tag('conjunct')),
            'vedic_count': sum(1 for t in tokens if t.has_tag('vedic')),
            'sandhi_boundaries': sum(1 for t in tokens if t.has_tag('sandhi_boundary'))
        }
        
        return stats