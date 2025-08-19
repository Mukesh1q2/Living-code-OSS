"""
Sanskrit text tokenization functionality with enhanced linguistic awareness.

This module provides advanced tokenization capabilities for Sanskrit text processing,
including support for multiple input formats (IAST, Devanagari), compound vowel
handling, morphological boundary detection, and context-aware tokenization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Set, Optional, Tuple, Union
import re
import unicodedata


class TokenKind(Enum):
    """Types of tokens in Sanskrit text processing."""
    VOWEL = "VOWEL"
    CONSONANT = "CONSONANT"
    MARKER = "MARKER"
    COMPOUND_VOWEL = "COMPOUND_VOWEL"
    CONJUNCT = "CONJUNCT"
    SYLLABLE = "SYLLABLE"
    MORPHEME = "MORPHEME"
    OTHER = "OTHER"


class InputFormat(Enum):
    """Supported Sanskrit input formats."""
    IAST = "IAST"  # International Alphabet of Sanskrit Transliteration
    DEVANAGARI = "DEVANAGARI"
    HARVARD_KYOTO = "HARVARD_KYOTO"
    VELTHUIS = "VELTHUIS"
    AUTO = "AUTO"  # Auto-detect format


@dataclass
class Token:
    """
    A token representing a unit of Sanskrit text with enhanced linguistic metadata.
    
    Attributes:
        text: The actual text content of the token
        start_pos: Starting position in original text
        end_pos: Ending position in original text
        token_type: The linguistic type of the token (for backward compatibility)
        kind: The linguistic type of the token (enum)
        tags: Set of descriptive tags for the token
        metadata: Dictionary of metadata associated with the token
        position: Position index in token sequence
        input_format: The detected input format of the token
        normalized_form: Normalized representation of the token
        phonetic_features: Phonetic and phonological features
        morphological_features: Morphological analysis features
        syllable_info: Syllable structure information
    """
    text: str
    start_pos: int
    end_pos: int
    token_type: str = "WORD"  # WORD, PUNCTUATION, WHITESPACE, MARKER
    kind: Optional[TokenKind] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Optional[int] = None
    input_format: Optional[InputFormat] = None
    normalized_form: Optional[str] = None
    phonetic_features: Dict[str, Any] = field(default_factory=dict)
    morphological_features: Dict[str, Any] = field(default_factory=dict)
    syllable_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set kind based on token_type if not provided."""
        if self.kind is None:
            if self.token_type == "MARKER":
                self.kind = TokenKind.MARKER
            else:
                self.kind = TokenKind.OTHER
    
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
        return self.metadata.get(key, default)
    
    def set_meta(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        self.metadata[key] = value
    
    # Alias for backward compatibility with tests
    @property
    def meta(self) -> Dict[str, Any]:
        """Alias for metadata property."""
        return self.metadata


class BasicSanskritTokenizer:
    """Enhanced tokenizer for Sanskrit text with linguistic awareness and multi-format support."""
    
    def __init__(self, input_format: InputFormat = InputFormat.AUTO):
        """Initialize the tokenizer with Sanskrit character sets and format support.
        
        Args:
            input_format: Expected input format (AUTO for auto-detection)
        """
        self.input_format = input_format
        
        # Enhanced Devanagari character set with additional characters
        self.sanskrit_chars: Set[str] = set(
            "अआइईउऊऋॠऌॡएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
            "ँंःऽ्॒॑॓॔ॕॖॗ॰।॥"  # Additional marks and punctuation
        )
        
        # Morphological markers used in Sanskrit processing
        self.markers: Set[str] = set("+-_:=|~")
        
        # Enhanced vowel sets with compound vowel support
        self.simple_vowels: Set[str] = {
            'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'e', 'o',
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ए', 'ओ'
        }
        
        # Compound vowels (diphthongs)
        self.compound_vowels: Set[str] = {
            'ai', 'au', 'ऐ', 'औ'
        }
        
        # All vowels (simple + compound)
        self.vowels: Set[str] = self.simple_vowels | self.compound_vowels | {
            # Vowel marks (matras)
            'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ'
        }
        
        # Vowel length classification
        self.short_vowels: Set[str] = {
            'a', 'i', 'u', 'ṛ', 'ḷ', 'e', 'o',
            'अ', 'इ', 'उ', 'ऋ', 'ऌ', 'ए', 'ओ'
        }
        
        self.long_vowels: Set[str] = {
            'ā', 'ī', 'ū', 'ṝ', 'ḹ', 'ai', 'au',
            'आ', 'ई', 'ऊ', 'ॠ', 'ॡ', 'ऐ', 'औ',
            'ा', 'ी', 'ू', 'ॄ', 'ॣ', 'ै', 'ौ'
        }
        
        # Vocalic consonants (syllabic liquids)
        self.vocalic_consonants: Set[str] = {'ṛ', 'ṝ', 'ḷ', 'ḹ', 'ऋ', 'ॠ', 'ऌ', 'ॡ'}
        
        # Enhanced consonant sets
        self.consonants: Set[str] = {
            # IAST consonants
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
        
        # Compound consonants (aspirated and others)
        self.compound_consonants: Set[str] = {
            'kh', 'gh', 'ch', 'jh', 'ṭh', 'ḍh', 'th', 'dh', 'ph', 'bh',
            'ख', 'घ', 'छ', 'झ', 'ठ', 'ढ', 'थ', 'ध', 'फ', 'भ'
        }
        
        # Phonetic classification by place of articulation
        self.velar_consonants = {'k', 'kh', 'g', 'gh', 'ṅ', 'क', 'ख', 'ग', 'घ', 'ङ'}
        self.palatal_consonants = {'c', 'ch', 'j', 'jh', 'ñ', 'च', 'छ', 'ज', 'झ', 'ञ'}
        self.retroflex_consonants = {'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ', 'ट', 'ठ', 'ड', 'ढ', 'ण'}
        self.dental_consonants = {'t', 'th', 'd', 'dh', 'n', 't', 'th', 'd', 'dh', 'n'}
        self.labial_consonants = {'p', 'ph', 'b', 'bh', 'm', 'प', 'फ', 'ब', 'भ', 'म'}
        self.semivowels = {'y', 'r', 'l', 'v', 'य', 'र', 'ल', 'व'}
        self.sibilants = {'ś', 'ṣ', 's', 'श', 'ष', 'स'}
        
        # Phonetic classification by manner of articulation
        self.stops = {'k', 'kh', 'g', 'gh', 'c', 'ch', 'j', 'jh', 'ṭ', 'ṭh', 'ḍ', 'ḍh', 
                     't', 'th', 'd', 'dh', 'p', 'ph', 'b', 'bh'}
        self.nasals = {'ṅ', 'ñ', 'ṇ', 'n', 'm', 'ङ', 'ञ', 'ण', 'न', 'म'}
        self.fricatives = {'ś', 'ṣ', 's', 'h', 'श', 'ष', 'स', 'ह'}
        self.liquids = {'r', 'l', 'र', 'ल'}
        self.glides = {'y', 'v', 'य', 'व'}
        
        # Voicing classification
        self.voiced_consonants = {'g', 'gh', 'j', 'jh', 'ḍ', 'ḍh', 'd', 'dh', 'b', 'bh',
                                 'ग', 'घ', 'ज', 'झ', 'ड', 'ढ', 'द', 'ध', 'ब', 'भ'}
        self.voiceless_consonants = {'k', 'kh', 'c', 'ch', 'ṭ', 'ṭh', 't', 'th', 'p', 'ph',
                                    'क', 'ख', 'च', 'छ', 'ट', 'ठ', 'त', 'थ', 'प', 'फ'}
        
        # Aspiration classification
        self.aspirated_consonants = {'kh', 'gh', 'ch', 'jh', 'ṭh', 'ḍh', 'th', 'dh', 'ph', 'bh',
                                    'ख', 'घ', 'छ', 'झ', 'ठ', 'ढ', 'थ', 'ध', 'फ', 'भ'}
        
        # Devanagari-specific elements
        self.devanagari_punctuation = {'।', '॥', '॰'}
        self.halant = '्'  # Virama
        self.anusvara = 'ं'
        self.visarga = 'ः'
        self.candrabindu = 'ँ'
        self.avagraha = 'ऽ'
        
        # Vedic accent marks
        self.vedic_accents = {'॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ'}
        
        # Format-specific character mappings
        self._init_format_mappings()
        
        # Morphological boundary patterns
        self.morpheme_boundary_patterns = [
            (r'([aeiouāīūṛṝḷḹeoai]+)([+-])', 'vowel_morpheme'),  # Vowel + morpheme boundary
            (r'([kgcjṭḍtdpbmnrlvśṣsh]+)([+-])', 'consonant_morpheme'),  # Consonant + morpheme boundary
            (r'([aeiouāīūṛṝḷḹeoai]+)([_:])', 'vowel_case'),  # Vowel + case/stem marker
            (r'([kgcjṭḍtdpbmnrlvśṣsh]+)([_:])', 'consonant_case'),  # Consonant + case/stem marker
        ]
        
        # Sandhi context patterns for boundary detection
        self.sandhi_patterns = [
            (r'([aā])([iī])', 'vowel_sandhi'),  # a/ā + i/ī → e
            (r'([aā])([uū])', 'vowel_sandhi'),  # a/ā + u/ū → o
            (r'([aā])([eo])', 'vowel_sandhi'),  # a/ā + e/o → ai/au
            (r'([iī])([aāuūeo])', 'vowel_sandhi'),  # i/ī + vowel
            (r'([uū])([aāiīeo])', 'vowel_sandhi'),  # u/ū + vowel
            (r'([eo])([aāiīuū])', 'vowel_sandhi'),  # e/o + vowel
            (r'([mnṅñṇ])([kgcjṭḍtdpb])', 'nasal_assimilation'),  # Nasal + stop
            (r'([td])([śṣs])', 'consonant_sandhi'),  # Dental + sibilant
            (r'([n])([rlv])', 'consonant_sandhi'),  # n + liquid/glide
        ]
    
    def _init_format_mappings(self) -> None:
        """Initialize character mappings for different input formats."""
        # IAST to Devanagari mapping
        self.iast_to_devanagari = {
            # Vowels
            'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई', 'u': 'उ', 'ū': 'ऊ',
            'ṛ': 'ऋ', 'ṝ': 'ॠ', 'ḷ': 'ऌ', 'ḹ': 'ॡ', 'e': 'ए', 'ai': 'ऐ',
            'o': 'ओ', 'au': 'औ',
            # Consonants
            'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ṅ': 'ङ',
            'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
            'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण',
            't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
            'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
            'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व',
            'ś': 'श', 'ṣ': 'ष', 's': 'स', 'h': 'ह',
            # Special characters
            'ṃ': 'ं', 'ḥ': 'ः', '~': 'ँ'
        }
        
        # Reverse mapping
        self.devanagari_to_iast = {v: k for k, v in self.iast_to_devanagari.items()}
        
        # Harvard-Kyoto mapping (simplified)
        self.harvard_kyoto_to_iast = {
            'A': 'ā', 'I': 'ī', 'U': 'ū', 'R': 'ṛ', 'RR': 'ṝ',
            'lR': 'ḷ', 'lRR': 'ḹ', 'G': 'ṅ', 'J': 'ñ', 'T': 'ṭ',
            'Th': 'ṭh', 'D': 'ḍ', 'Dh': 'ḍh', 'N': 'ṇ', 'z': 'ś',
            'S': 'ṣ', 'M': 'ṃ', 'H': 'ḥ'
        }
        
        # Velthuis mapping (simplified)
        self.velthuis_to_iast = {
            'aa': 'ā', 'ii': 'ī', 'uu': 'ū', '.r': 'ṛ', '.rr': 'ṝ',
            '.l': 'ḷ', '.ll': 'ḹ', '~n': 'ṅ', '~m': 'ṃ', '.t': 'ṭ',
            '.th': 'ṭh', '.d': 'ḍ', '.dh': 'ḍh', '.n': 'ṇ', '"s': 'ś',
            '.s': 'ṣ', '.h': 'ḥ'
        }
        
    def detect_input_format(self, text: str) -> InputFormat:
        """Detect the input format of Sanskrit text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected InputFormat
        """
        if not text:
            return InputFormat.AUTO
        
        # Check for Devanagari characters
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # Check for IAST diacritics
        iast_diacritics = {'ā', 'ī', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ', 'ṅ', 'ñ', 'ṭ', 'ḍ', 'ṇ', 'ś', 'ṣ', 'ṃ', 'ḥ'}
        iast_count = sum(1 for char in text if char in iast_diacritics)
        
        # Check for Harvard-Kyoto patterns
        hk_patterns = ['A', 'I', 'U', 'R', 'G', 'J', 'T', 'D', 'N', 'S', 'M', 'H']
        hk_count = sum(1 for pattern in hk_patterns if pattern in text)
        
        # Check for Velthuis patterns
        velthuis_patterns = ['aa', 'ii', 'uu', '.r', '.t', '.d', '.n', '.s', '"s']
        velthuis_count = sum(1 for pattern in velthuis_patterns if pattern in text)
        
        # Determine format based on character counts
        total_chars = len(text)
        if total_chars == 0:
            return InputFormat.AUTO
        
        devanagari_ratio = devanagari_count / total_chars
        iast_ratio = iast_count / total_chars
        
        if devanagari_ratio > 0.1:  # 10% Devanagari characters
            return InputFormat.DEVANAGARI
        elif iast_ratio > 0.05:  # 5% IAST diacritics
            return InputFormat.IAST
        elif hk_count > 0 and velthuis_count == 0:
            return InputFormat.HARVARD_KYOTO
        elif velthuis_count > 0:
            return InputFormat.VELTHUIS
        else:
            # Default to IAST for Roman script
            return InputFormat.IAST
    
    def normalize_text(self, text: str, target_format: InputFormat = InputFormat.IAST) -> str:
        """Normalize text to a target format.
        
        Args:
            text: Input text
            target_format: Target format for normalization
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        detected_format = self.detect_input_format(text)
        
        if detected_format == target_format:
            return text
        
        # Convert to IAST as intermediate format
        if detected_format == InputFormat.DEVANAGARI:
            # Convert Devanagari to IAST
            normalized = self._devanagari_to_iast(text)
        elif detected_format == InputFormat.HARVARD_KYOTO:
            normalized = self._harvard_kyoto_to_iast(text)
        elif detected_format == InputFormat.VELTHUIS:
            normalized = self._velthuis_to_iast(text)
        else:
            normalized = text
        
        # Convert from IAST to target format if needed
        if target_format == InputFormat.DEVANAGARI:
            return self._iast_to_devanagari(normalized)
        else:
            return normalized
    
    def _devanagari_to_iast(self, text: str) -> str:
        """Convert Devanagari text to IAST."""
        result = []
        for char in text:
            if char in self.devanagari_to_iast:
                result.append(self.devanagari_to_iast[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def _iast_to_devanagari(self, text: str) -> str:
        """Convert IAST text to Devanagari."""
        # Sort by length (longest first) to handle compound characters
        sorted_mappings = sorted(self.iast_to_devanagari.items(), key=lambda x: len(x[0]), reverse=True)
        
        result = text
        for iast, devanagari in sorted_mappings:
            result = result.replace(iast, devanagari)
        
        return result
    
    def _harvard_kyoto_to_iast(self, text: str) -> str:
        """Convert Harvard-Kyoto text to IAST."""
        result = text
        for hk, iast in self.harvard_kyoto_to_iast.items():
            result = result.replace(hk, iast)
        return result
    
    def _velthuis_to_iast(self, text: str) -> str:
        """Convert Velthuis text to IAST."""
        result = text
        # Sort by length (longest first) to handle compound characters
        sorted_mappings = sorted(self.velthuis_to_iast.items(), key=lambda x: len(x[0]), reverse=True)
        for velthuis, iast in sorted_mappings:
            result = result.replace(velthuis, iast)
        return result

    def tokenize(self, text: str, preserve_format: bool = True) -> List[Token]:
        """Tokenize Sanskrit text into linguistic units with enhanced processing.
        
        Args:
            text: Input Sanskrit text
            preserve_format: Whether to preserve original format in tokens
            
        Returns:
            List of Token objects with enhanced linguistic metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        # Remove leading/trailing whitespace but preserve internal structure
        original_text = text
        text = text.strip()
        if not text:
            return []
        
        # Detect input format
        detected_format = self.detect_input_format(text) if self.input_format == InputFormat.AUTO else self.input_format
        
        # Normalize text for processing (but preserve original in tokens if requested)
        normalized_text = self.normalize_text(text, InputFormat.IAST)
        
        # For simple single words without markers, use enhanced word tokenization
        if not any(marker in text for marker in self.markers) and ' ' not in text:
            return self._tokenize_enhanced_word(text, normalized_text, detected_format, preserve_format)
        
        tokens = []
        position = 0
        
        # First pass: detect morphological boundaries and split accordingly
        segments = self._segment_with_morphological_awareness(text, normalized_text)
        
        current_pos = 0
        for segment_info in segments:
            segment = segment_info['text']
            segment_type = segment_info['type']
            normalized_segment = segment_info.get('normalized', segment)
            
            if not segment:
                continue
                
            # Find actual position in original text
            start_pos = text.find(segment, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(segment)
            
            if segment_type == 'marker':
                # Create enhanced marker token
                token = self._create_enhanced_marker_token(
                    segment, start_pos, end_pos, position, detected_format
                )
                tokens.append(token)
                position += 1
            elif segment_type == 'morpheme_boundary':
                # Create morpheme boundary token
                token = self._create_morpheme_boundary_token(
                    segment, start_pos, end_pos, position, detected_format
                )
                tokens.append(token)
                position += 1
            else:
                # Process linguistic segments with enhanced analysis
                word_tokens = self._tokenize_linguistic_segment(
                    segment, normalized_segment, start_pos, position, detected_format, preserve_format
                )
                tokens.extend(word_tokens)
                position += len(word_tokens)
            
            current_pos = end_pos
        
        # Enhanced post-processing with linguistic analysis
        if tokens:
            self._add_enhanced_linguistic_features(tokens)
            self._detect_compound_vowel_sequences(tokens)
            self._analyze_syllable_structure(tokens)
            self._detect_morphological_boundaries(tokens)
            self._analyze_sandhi_contexts(tokens)
            self._add_phonological_features(tokens)
        
        return tokens
    
    def _segment_with_morphological_awareness(self, text: str, normalized_text: str) -> List[Dict[str, Any]]:
        """Segment text with morphological boundary awareness.
        
        Args:
            text: Original text
            normalized_text: Normalized text for analysis
            
        Returns:
            List of segment dictionaries with type and metadata
        """
        segments = []
        
        # First, identify morphological markers
        marker_positions = []
        for i, char in enumerate(text):
            if char in self.markers:
                marker_positions.append(i)
        
        # Split text around markers while preserving them
        current_pos = 0
        for marker_pos in marker_positions:
            # Add text before marker
            if marker_pos > current_pos:
                segment_text = text[current_pos:marker_pos].strip()
                if segment_text:
                    segments.append({
                        'text': segment_text,
                        'type': 'word',
                        'normalized': normalized_text[current_pos:marker_pos].strip()
                    })
            
            # Add marker
            segments.append({
                'text': text[marker_pos],
                'type': 'marker',
                'normalized': text[marker_pos]
            })
            current_pos = marker_pos + 1
        
        # Add remaining text
        if current_pos < len(text):
            segment_text = text[current_pos:].strip()
            if segment_text:
                segments.append({
                    'text': segment_text,
                    'type': 'word',
                    'normalized': normalized_text[current_pos:].strip()
                })
        
        # If no markers found, treat as single segment
        if not segments:
            segments.append({
                'text': text,
                'type': 'word',
                'normalized': normalized_text
            })
        
        # Further segment words by whitespace
        final_segments = []
        for segment in segments:
            if segment['type'] == 'word' and ' ' in segment['text']:
                words = segment['text'].split()
                normalized_words = segment['normalized'].split()
                for i, word in enumerate(words):
                    normalized_word = normalized_words[i] if i < len(normalized_words) else word
                    final_segments.append({
                        'text': word,
                        'type': 'word',
                        'normalized': normalized_word
                    })
            else:
                final_segments.append(segment)
        
        return final_segments
    
    def _create_enhanced_marker_token(self, text: str, start_pos: int, end_pos: int, 
                                    position: int, input_format: InputFormat) -> Token:
        """Create an enhanced marker token with linguistic metadata."""
        token = Token(
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            token_type="MARKER",
            kind=TokenKind.MARKER,
            position=position,
            input_format=input_format,
            normalized_form=text
        )
        
        # Add marker-specific tags
        token.add_tag('morphological')
        
        # Classify marker type
        if text in ['+', '-']:
            token.add_tag('compound_marker')
            token.morphological_features['boundary_type'] = 'compound'
        elif text in ['_', ':']:
            token.add_tag('case_marker')
            token.morphological_features['boundary_type'] = 'inflection'
        elif text in ['=']:
            token.add_tag('clitic_marker')
            token.morphological_features['boundary_type'] = 'clitic'
        elif text in ['|', '~']:
            token.add_tag('prosodic_marker')
            token.morphological_features['boundary_type'] = 'prosodic'
        
        return token
    
    def _create_morpheme_boundary_token(self, text: str, start_pos: int, end_pos: int,
                                      position: int, input_format: InputFormat) -> Token:
        """Create a morpheme boundary token."""
        token = Token(
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            token_type="MARKER",
            kind=TokenKind.MORPHEME,
            position=position,
            input_format=input_format,
            normalized_form=text
        )
        
        token.add_tag('morpheme_boundary')
        return token
    
    def _tokenize_enhanced_word(self, word: str, normalized_word: str, input_format: InputFormat,
                              preserve_format: bool) -> List[Token]:
        """Tokenize a single word with enhanced linguistic analysis."""
        if not word.strip():
            return []
        
        # For compound vowels and simple cases, use enhanced single token approach
        if self._is_simple_linguistic_unit(normalized_word):
            token = self._create_enhanced_linguistic_token(
                word, normalized_word, 0, len(word), 0, input_format, preserve_format
            )
            self._analyze_single_token_features(token)
            return [token]
        
        # Use character-level tokenization with linguistic awareness
        return self._tokenize_linguistic_segment(word, normalized_word, 0, 0, input_format, preserve_format)
    
    def _is_simple_linguistic_unit(self, normalized_word: str) -> bool:
        """Check if a word is a simple linguistic unit that shouldn't be further segmented."""
        # Compound vowels
        if normalized_word in self.compound_vowels:
            return True
        
        # Short words (2 characters or less) that don't contain compound elements
        if len(normalized_word) <= 2:
            # But check if it's a compound consonant itself
            if normalized_word in self.compound_consonants:
                return False  # Should be segmented to show compound structure
            return True
        
        # Check for mixed content (letters + numbers + punctuation)
        has_letters = any(c.isalpha() or c in self.sanskrit_chars for c in normalized_word)
        has_numbers = any(c.isdigit() for c in normalized_word)
        has_punctuation = any(c in self.devanagari_punctuation or not c.isalnum() and c not in self.vowels and c not in self.consonants for c in normalized_word)
        
        # If mixed content, don't treat as simple unit
        if (has_letters and has_numbers) or (has_letters and has_punctuation) or (has_numbers and has_punctuation):
            return False
        
        # Check for conjunct consonants
        if self.halant in normalized_word:
            return False
        
        # Check for compound consonants - if present, should be segmented
        if any(compound in normalized_word for compound in self.compound_consonants):
            return False
        
        # For simple Sanskrit words without complex structures, treat as single units
        # This preserves the original behavior for basic words like "rama"
        return True
    
    def _tokenize_linguistic_segment(self, segment: str, normalized_segment: str, start_offset: int,
                                   position_offset: int, input_format: InputFormat, 
                                   preserve_format: bool) -> List[Token]:
        """Tokenize a linguistic segment with enhanced analysis."""
        if not segment.strip():
            return []
        
        tokens = []
        position = position_offset
        char_pos = 0
        normalized_pos = 0
        
        while char_pos < len(segment) and normalized_pos < len(normalized_segment):
            # Check for compound vowels first
            if self._check_compound_vowel_at_position(normalized_segment, normalized_pos):
                vowel_info = self._extract_compound_vowel(segment, normalized_segment, char_pos, normalized_pos)
                token = self._create_compound_vowel_token(
                    vowel_info, start_offset, position, input_format, preserve_format
                )
                tokens.append(token)
                char_pos += vowel_info['char_length']
                normalized_pos += vowel_info['normalized_length']
                position += 1
                continue
            
            # Check for conjunct consonants
            if self._check_conjunct_at_position(normalized_segment, normalized_pos):
                conjunct_info = self._extract_conjunct(segment, normalized_segment, char_pos, normalized_pos)
                token = self._create_conjunct_token(
                    conjunct_info, start_offset, position, input_format, preserve_format
                )
                tokens.append(token)
                char_pos += conjunct_info['char_length']
                normalized_pos += conjunct_info['normalized_length']
                position += 1
                continue
            
            # Check for compound consonants
            if self._check_compound_consonant_at_position(normalized_segment, normalized_pos):
                compound_info = self._extract_compound_consonant(segment, normalized_segment, char_pos, normalized_pos)
                token = self._create_compound_consonant_token(
                    compound_info, start_offset, position, input_format, preserve_format
                )
                tokens.append(token)
                char_pos += compound_info['char_length']
                normalized_pos += compound_info['normalized_length']
                position += 1
                continue
            
            # Single character/unit
            char_info = self._extract_single_unit(segment, normalized_segment, char_pos, normalized_pos)
            if char_info['text'].strip():  # Skip whitespace
                token = self._create_enhanced_linguistic_token(
                    char_info['text'], char_info['normalized'], 
                    start_offset + char_pos, start_offset + char_pos + char_info['char_length'],
                    position, input_format, preserve_format
                )
                tokens.append(token)
                position += 1
            
            char_pos += char_info['char_length']
            normalized_pos += char_info['normalized_length']
        
        return tokens
    
    def _check_compound_vowel_at_position(self, text: str, pos: int) -> bool:
        """Check if there's a compound vowel at the given position."""
        for vowel in sorted(self.compound_vowels, key=len, reverse=True):
            if text[pos:pos+len(vowel)] == vowel:
                return True
        return False
    
    def _extract_compound_vowel(self, original: str, normalized: str, char_pos: int, norm_pos: int) -> Dict[str, Any]:
        """Extract compound vowel information."""
        for vowel in sorted(self.compound_vowels, key=len, reverse=True):
            if normalized[norm_pos:norm_pos+len(vowel)] == vowel:
                # Find corresponding characters in original text
                char_length = len(vowel)  # Default assumption
                if original != normalized:
                    # Handle format differences
                    if vowel == 'ai' and char_pos < len(original) and original[char_pos] == 'ऐ':
                        char_length = 1
                    elif vowel == 'au' and char_pos < len(original) and original[char_pos] == 'औ':
                        char_length = 1
                
                return {
                    'text': original[char_pos:char_pos+char_length],
                    'normalized': vowel,
                    'char_length': char_length,
                    'normalized_length': len(vowel),
                    'vowel_type': 'compound'
                }
        return {}
    
    def _check_conjunct_at_position(self, text: str, pos: int) -> bool:
        """Check if there's a conjunct consonant at the given position."""
        if pos + 2 < len(text):
            if (text[pos] in self.consonants and 
                text[pos + 1] == self.halant and 
                text[pos + 2] in self.consonants):
                return True
        return False
    
    def _extract_conjunct(self, original: str, normalized: str, char_pos: int, norm_pos: int) -> Dict[str, Any]:
        """Extract conjunct consonant information."""
        if norm_pos + 2 < len(normalized):
            conjunct = normalized[norm_pos:norm_pos+3]
            return {
                'text': original[char_pos:char_pos+3] if char_pos + 2 < len(original) else conjunct,
                'normalized': conjunct,
                'char_length': 3,
                'normalized_length': 3,
                'consonant_type': 'conjunct'
            }
        return {}
    
    def _check_compound_consonant_at_position(self, text: str, pos: int) -> bool:
        """Check if there's a compound consonant at the given position."""
        for consonant in sorted(self.compound_consonants, key=len, reverse=True):
            if text[pos:pos+len(consonant)] == consonant:
                return True
        return False
    
    def _extract_compound_consonant(self, original: str, normalized: str, char_pos: int, norm_pos: int) -> Dict[str, Any]:
        """Extract compound consonant information."""
        for consonant in sorted(self.compound_consonants, key=len, reverse=True):
            if normalized[norm_pos:norm_pos+len(consonant)] == consonant:
                char_length = len(consonant)
                if original != normalized:
                    # Handle Devanagari single characters for compound consonants
                    if char_pos < len(original) and original[char_pos] in self.compound_consonants:
                        char_length = 1
                
                return {
                    'text': original[char_pos:char_pos+char_length],
                    'normalized': consonant,
                    'char_length': char_length,
                    'normalized_length': len(consonant),
                    'consonant_type': 'compound'
                }
        return {}
    
    def _extract_single_unit(self, original: str, normalized: str, char_pos: int, norm_pos: int) -> Dict[str, Any]:
        """Extract a single linguistic unit."""
        char_length = 1
        norm_length = 1
        
        # Handle multi-byte characters
        if char_pos < len(original):
            char = original[char_pos]
            norm_char = normalized[norm_pos] if norm_pos < len(normalized) else char
            
            return {
                'text': char,
                'normalized': norm_char,
                'char_length': char_length,
                'normalized_length': norm_length
            }
        
        return {'text': '', 'normalized': '', 'char_length': 0, 'normalized_length': 0}
    
    def _create_compound_vowel_token(self, vowel_info: Dict[str, Any], start_offset: int,
                                   position: int, input_format: InputFormat, preserve_format: bool) -> Token:
        """Create a token for compound vowels."""
        token = Token(
            text=vowel_info['text'],
            start_pos=start_offset,
            end_pos=start_offset + vowel_info['char_length'],
            token_type="WORD",
            kind=TokenKind.COMPOUND_VOWEL,
            position=position,
            input_format=input_format,
            normalized_form=vowel_info['normalized']
        )
        
        token.add_tag('compound_vowel')
        token.add_tag('long')  # Compound vowels are typically long
        token.phonetic_features['vowel_type'] = 'compound'
        token.phonetic_features['length'] = 'long'
        
        # Add specific compound vowel features
        if vowel_info['normalized'] in ['ai', 'ऐ']:
            token.phonetic_features['components'] = ['a', 'i']
            token.add_tag('palatal_diphthong')
        elif vowel_info['normalized'] in ['au', 'औ']:
            token.phonetic_features['components'] = ['a', 'u']
            token.add_tag('labial_diphthong')
        
        return token
    
    def _create_conjunct_token(self, conjunct_info: Dict[str, Any], start_offset: int,
                             position: int, input_format: InputFormat, preserve_format: bool) -> Token:
        """Create a token for conjunct consonants."""
        token = Token(
            text=conjunct_info['text'],
            start_pos=start_offset,
            end_pos=start_offset + conjunct_info['char_length'],
            token_type="WORD",
            kind=TokenKind.CONJUNCT,
            position=position,
            input_format=input_format,
            normalized_form=conjunct_info['normalized']
        )
        
        token.add_tag('conjunct')
        token.add_tag('consonant_cluster')
        token.phonetic_features['consonant_type'] = 'conjunct'
        token.morphological_features['structure'] = 'C+virama+C'
        
        return token
    
    def _create_compound_consonant_token(self, consonant_info: Dict[str, Any], start_offset: int,
                                       position: int, input_format: InputFormat, preserve_format: bool) -> Token:
        """Create a token for compound consonants."""
        token = Token(
            text=consonant_info['text'],
            start_pos=start_offset,
            end_pos=start_offset + consonant_info['char_length'],
            token_type="WORD",
            kind=TokenKind.CONSONANT,
            position=position,
            input_format=input_format,
            normalized_form=consonant_info['normalized']
        )
        
        token.add_tag('compound')  # Add the tag the test expects
        token.add_tag('compound_consonant')
        token.add_tag('aspirated')  # Most compound consonants are aspirated
        token.phonetic_features['consonant_type'] = 'compound'
        token.phonetic_features['aspiration'] = 'aspirated'
        token.set_meta('compound_type', 'consonant')  # Add metadata the test expects
        
        return token
    
    def _create_enhanced_linguistic_token(self, text: str, normalized: str, start_pos: int, end_pos: int,
                                        position: int, input_format: InputFormat, preserve_format: bool) -> Token:
        """Create an enhanced linguistic token with comprehensive metadata."""
        token_type = self._classify_token(text)
        kind = self._get_enhanced_token_kind(normalized)
        
        token = Token(
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            token_type=token_type,
            kind=kind,
            position=position,
            input_format=input_format,
            normalized_form=normalized
        )
        
        # Add comprehensive linguistic features
        self._add_enhanced_linguistic_tags(token)
        self._add_phonetic_features(token)
        self._add_morphological_features(token)
        
        return token
    
    def _add_enhanced_linguistic_tags(self, token: Token) -> None:
        """Add enhanced linguistic tags to a token."""
        if token.kind == TokenKind.MARKER:
            token.add_tag('morphological')
        elif token.kind == TokenKind.COMPOUND_VOWEL:
            token.add_tag('compound_vowel')
            token.add_tag('diphthong')
        elif token.kind == TokenKind.CONJUNCT:
            token.add_tag('conjunct')
            token.add_tag('consonant_cluster')
        elif token.kind == TokenKind.VOWEL:
            self._add_vowel_specific_tags(token)
        elif token.kind == TokenKind.CONSONANT:
            self._add_consonant_specific_tags(token)
    
    def _add_vowel_specific_tags(self, token: Token) -> None:
        """Add vowel-specific tags."""
        vowel = token.normalized_form
        
        if vowel in self.short_vowels:
            token.add_tag('short')
        elif vowel in self.long_vowels:
            token.add_tag('long')
        
        if vowel in self.vocalic_consonants:
            token.add_tag('vocalic')
    
    def _add_consonant_specific_tags(self, token: Token) -> None:
        """Add consonant-specific tags."""
        char = token.normalized_form[0] if token.normalized_form else ''
        
        # Add phonetic classification tags
        if char in self.velar_consonants:
            token.add_tag('velar')
        elif char in self.palatal_consonants:
            token.add_tag('palatal')
        elif char in self.retroflex_consonants:
            token.add_tag('retroflex')
        elif char in self.dental_consonants:
            token.add_tag('dental')
        elif char in self.labial_consonants:
            token.add_tag('labial')
        elif char in self.semivowels:
            token.add_tag('semivowel')
        elif char in self.sibilants:
            token.add_tag('sibilant')
        
        # Add compound consonant tags
        if token.normalized_form in self.compound_consonants:
            token.add_tag('compound')
            token.add_tag('aspirated')
    
    def _add_phonetic_features(self, token: Token) -> None:
        """Add phonetic features to token (wrapper for compatibility)."""
        # This method is called from _create_enhanced_linguistic_token
        # The actual phonetic features are added in _add_phonological_features
        pass
    
    def _add_morphological_features(self, token: Token) -> None:
        """Add morphological features to token."""
        # Basic morphological classification
        if token.kind == TokenKind.MARKER:
            token.morphological_features['type'] = 'boundary_marker'
        elif token.kind in [TokenKind.VOWEL, TokenKind.COMPOUND_VOWEL]:
            token.morphological_features['type'] = 'vowel'
        elif token.kind in [TokenKind.CONSONANT, TokenKind.CONJUNCT]:
            token.morphological_features['type'] = 'consonant'
    
    def _analyze_single_token_features(self, token: Token) -> None:
        """Analyze features for a single token (used in simple word tokenization)."""
        self._add_enhanced_linguistic_tags(token)
        self._add_phonetic_features(token)
        self._add_morphological_features(token)
        
        # Add word-level features
        token.add_tag('word_initial')
        token.add_tag('word_final')
        token.syllable_info['position'] = 'complete_word'
        token.syllable_info['total_syllables'] = 1
        
        # For single-token words, analyze internal structure
        self._analyze_word_internal_structure(token)
    
    def _analyze_word_internal_structure(self, token: Token) -> None:
        """Analyze internal structure of single-token words."""
        word = token.normalized_form or token.text
        
        # Check for compound consonants within the word
        for compound in self.compound_consonants:
            if compound in word:
                token.add_tag('compound')
                token.set_meta('compound_type', 'consonant')
                # Also add to morphological features for consistency
                token.morphological_features['contains_compound'] = compound
                break
        
        # Check for conjunct consonants
        if self.halant in word:
            token.add_tag('conjunct')
            token.set_meta('conjunct_type', 'devanagari')
        
        # Add syllable structure tags for single words
        # Simple heuristic: if it contains vowels, it has syllable nuclei
        if any(char in self.vowels for char in word):
            token.add_tag('syllable_nucleus')
        
        # If it contains consonants, add syllable onset tag
        if any(char in self.consonants for char in word):
            token.add_tag('syllable_onset')
    
    def _get_enhanced_token_kind(self, normalized_text: str) -> TokenKind:
        """Get enhanced TokenKind with better classification."""
        if not normalized_text:
            return TokenKind.OTHER
        
        if normalized_text in self.markers or normalized_text in self.devanagari_punctuation:
            return TokenKind.MARKER
        elif normalized_text in self.compound_vowels:
            return TokenKind.COMPOUND_VOWEL
        elif self.halant in normalized_text and len(normalized_text) > 1:
            return TokenKind.CONJUNCT
        elif any(compound in normalized_text for compound in self.compound_consonants):
            return TokenKind.CONSONANT
        elif any(char in self.consonants for char in normalized_text):
            return TokenKind.CONSONANT
        elif any(char in self.vowels for char in normalized_text):
            return TokenKind.VOWEL
        else:
            return TokenKind.OTHER
    
    def _tokenize_simple_word(self, word: str) -> List[Token]:
        """Tokenize a simple word without markers or whitespace.
        
        Args:
            word: Simple word to tokenize
            
        Returns:
            List of tokens
        """
        # Check if word contains compound consonants - if so, use character-level tokenization
        has_compound_consonants = any(compound in word for compound in self.compound_consonants)
        has_conjuncts = self.halant in word
        
        if has_compound_consonants or has_conjuncts:
            # Use character-level tokenization for compound analysis
            tokens = self._tokenize_word(word, 0, 0)
            if tokens:
                self._add_positional_tags(tokens)
                self._add_syllable_structure_tags(tokens)
                self._detect_compound_words(tokens)
                self._detect_sandhi_boundaries(tokens)
            return tokens
        
        # For basic words, return as single token with linguistic analysis
        token_type = self._classify_token(word)
        kind = self._get_token_kind(word)
        
        token = Token(
            text=word,
            start_pos=0,
            end_pos=len(word),
            token_type=token_type,
            kind=kind,
            position=0
        )
        
        # Add comprehensive linguistic tags
        self._add_basic_tags(token)
        self._add_word_level_tags(token)
        token.add_tag('word_initial')
        token.add_tag('word_final')
        
        return [token]
    
    def _add_word_level_tags(self, token: Token) -> None:
        """Add word-level linguistic tags to a token.
        
        Args:
            token: Token to analyze and tag
        """
        word = token.text
        
        # Check for compound consonants in the word
        for compound in self.compound_consonants:
            if compound in word:
                token.add_tag('compound')
                token.set_meta('compound_type', 'consonant')
                break
        
        # Check for conjunct consonants (Devanagari)
        if self.halant in word:
            token.add_tag('conjunct')
            token.set_meta('conjunct_type', 'devanagari')
        
        # Add phonetic classification based on first consonant
        for char in word:
            if char in self.consonants:
                if char in self.velar_consonants:
                    token.add_tag('velar')
                elif char in self.palatal_consonants:
                    token.add_tag('palatal')
                elif char in self.retroflex_consonants:
                    token.add_tag('retroflex')
                elif char in self.dental_consonants:
                    token.add_tag('dental')
                elif char in self.labial_consonants:
                    token.add_tag('labial')
                elif char in self.semivowels:
                    token.add_tag('semivowel')
                elif char in self.sibilants:
                    token.add_tag('sibilant')
                break
        
        # Add syllable structure tags for words containing vowels
        if any(char in self.vowels for char in word):
            token.add_tag('syllable_nucleus')
        
        # Add syllable position tags for consonant words
        if token.kind == TokenKind.CONSONANT:
            token.add_tag('syllable_onset')  # Default for consonant words
        
        # Add sandhi boundary detection for vowel endings
        if word and word[-1] in self.vowels:
            token.add_tag('sandhi_boundary')
        
    def is_sanskrit_char(self, char: str) -> bool:
        """Check if character is Sanskrit.
        
        Args:
            char: Character to check
            
        Returns:
            True if character is Sanskrit
        """
        return char in self.sanskrit_chars
        
    def preserve_markers(self, text: str) -> List[str]:
        """Split text while preserving morphological markers.
        
        Args:
            text: Input text with markers
            
        Returns:
            List of text segments with preserved markers
        """
        if not text:
            return []
        
        # Split on markers but keep them
        parts = []
        current = ""
        
        for char in text:
            if char in self.markers:
                if current.strip():  # Only add non-empty, non-whitespace segments
                    parts.append(current.strip())
                    current = ""
                parts.append(char)
            else:
                current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return [part for part in parts if part]  # Remove empty parts
        
    def _classify_token(self, text: str) -> str:
        """Classify a token by type.
        
        Args:
            text: Token text
            
        Returns:
            Token type string
        """
        if not text:
            return "OTHER"
        
        if text in self.markers:
            return "MARKER"
        elif text in self.devanagari_punctuation:
            return "MARKER"  # Treat Devanagari punctuation as markers
        elif any(char in self.vowels for char in text):
            return "WORD"
        elif any(char in self.consonants for char in text):
            return "WORD"
        elif any(self.is_sanskrit_char(char) for char in text):
            return "WORD"
        elif text.isspace():
            return "WHITESPACE"
        elif any(char.isalpha() for char in text):
            # Roman script Sanskrit words
            return "WORD"
        elif text.isdigit():
            return "OTHER"
        else:
            return "PUNCTUATION"
            
    def _get_token_kind(self, text: str) -> TokenKind:
        """Get the TokenKind enum for a token.
        
        Args:
            text: Token text
            
        Returns:
            TokenKind enum value
        """
        if not text:
            return TokenKind.OTHER
        
        if text in self.markers or text in self.devanagari_punctuation:
            return TokenKind.MARKER
        
        # Check for compound consonants first (prioritize consonant classification)
        if any(compound in text for compound in self.compound_consonants):
            return TokenKind.CONSONANT
        
        # Check for conjunct consonants (Devanagari)
        if self.halant in text:
            return TokenKind.CONSONANT
        
        # Check if text contains numbers or punctuation - classify as OTHER
        if any(char.isdigit() or not char.isalpha() for char in text if char not in self.vowels and char not in self.consonants and char not in self.markers):
            return TokenKind.OTHER
        elif any(char in self.consonants for char in text):
            return TokenKind.CONSONANT
        elif any(char in self.vowels for char in text):
            return TokenKind.VOWEL
        else:
            return TokenKind.OTHER
            
    def _tokenize_word(self, word: str, start_offset: int, position_offset: int) -> List[Token]:
        """Tokenize a single word into linguistic units.
        
        Args:
            word: Word to tokenize
            start_offset: Starting position in original text
            position_offset: Starting position in token sequence
            
        Returns:
            List of linguistic tokens
        """
        if not word.strip():
            return []
        
        # For simple cases like compound vowels, treat as single tokens
        if word in ['ai', 'au'] or len(word) <= 2:
            token = self._create_character_token(
                word, start_offset, start_offset + len(word), position_offset
            )
            return [token]
        
        tokens = []
        position = position_offset
        char_pos = 0
        
        while char_pos < len(word):
            # Check for compound vowels first (2 characters)
            if char_pos < len(word) - 1:
                two_char = word[char_pos:char_pos + 2]
                if two_char in ['ai', 'au', 'ā', 'ī', 'ū']:
                    token = self._create_character_token(
                        two_char, start_offset + char_pos, start_offset + char_pos + 2, position
                    )
                    tokens.append(token)
                    char_pos += 2
                    position += 1
                    continue
                elif two_char in self.compound_consonants:
                    token = self._create_character_token(
                        two_char, start_offset + char_pos, start_offset + char_pos + 2, position
                    )
                    token.add_tag('compound')
                    token.set_meta('compound_type', 'consonant')
                    tokens.append(token)
                    char_pos += 2
                    position += 1
                    continue
            
            # Check for conjunct consonants (consonant + halant + consonant)
            if (char_pos < len(word) - 2 and 
                word[char_pos + 1] == self.halant and 
                word[char_pos] in self.consonants and 
                word[char_pos + 2] in self.consonants):
                
                conjunct = word[char_pos:char_pos + 3]
                token = self._create_character_token(
                    conjunct, start_offset + char_pos, start_offset + char_pos + 3, position
                )
                token.add_tag('conjunct')
                token.set_meta('conjunct_type', 'devanagari')
                tokens.append(token)
                char_pos += 3
                position += 1
                continue
            
            # Single character
            char = word[char_pos]
            if not char.isspace():  # Skip whitespace characters
                token = self._create_character_token(
                    char, start_offset + char_pos, start_offset + char_pos + 1, position
                )
                tokens.append(token)
                position += 1
            char_pos += 1
        
        return tokens
    
    def _create_character_token(self, char: str, start_pos: int, end_pos: int, position: int) -> Token:
        """Create a token for a character or character sequence.
        
        Args:
            char: Character or character sequence
            start_pos: Starting position in original text
            end_pos: Ending position in original text
            position: Position in token sequence
            
        Returns:
            Token object with appropriate classification
        """
        token_type = self._classify_token(char)
        kind = self._get_token_kind(char)
        
        token = Token(
            text=char,
            start_pos=start_pos,
            end_pos=end_pos,
            token_type=token_type,
            kind=kind,
            position=position
        )
        
        # Add basic linguistic tags
        self._add_basic_tags(token)
        
        return token
    
    def _add_basic_tags(self, token: Token) -> None:
        """Add basic tags to a token based on its content.
        
        Args:
            token: Token to add tags to
        """
        if token.kind == TokenKind.MARKER:
            token.add_tag('morphological')
        elif token.kind == TokenKind.VOWEL:
            # Check for long vowels
            if any(char in self.long_vowels for char in token.text):
                token.add_tag('long')
            elif any(char in self.short_vowels for char in token.text):
                token.add_tag('short')
            
            # Check for vocalic consonants
            if any(char in self.vocalic_consonants for char in token.text):
                token.add_tag('vocalic')
        elif token.kind == TokenKind.CONSONANT:
            # Add basic consonant classification
            if any(char in self.sanskrit_chars for char in token.text):
                token.add_tag('sanskrit')
            
            # Check for compound consonants
            if token.text in self.compound_consonants:
                token.add_tag('compound')
                token.set_meta('compound_type', 'consonant')
            
            # Add phonetic classification
            self._add_phonetic_tags(token)
    
    def _add_phonetic_tags(self, token: Token) -> None:
        """Add phonetic classification tags to consonant tokens.
        
        Args:
            token: Consonant token to classify
        """
        char = token.text[0] if token.text else ''
        
        if char in self.velar_consonants:
            token.add_tag('velar')
        elif char in self.palatal_consonants:
            token.add_tag('palatal')
        elif char in self.retroflex_consonants:
            token.add_tag('retroflex')
        elif char in self.dental_consonants:
            token.add_tag('dental')
        elif char in self.labial_consonants:
            token.add_tag('labial')
        elif char in self.semivowels:
            token.add_tag('semivowel')
        elif char in self.sibilants:
            token.add_tag('sibilant')
    
    def _add_positional_tags(self, tokens: List[Token]) -> None:
        """Add positional context tags to tokens.
        
        Args:
            tokens: List of tokens to tag
        """
        if not tokens:
            return
        
        # Tag first and last tokens
        tokens[0].add_tag('word_initial')
        if len(tokens) > 1:
            tokens[-1].add_tag('word_final')
        
        # Tag middle tokens
        for token in tokens[1:-1]:
            token.add_tag('word_medial')
    
    def _add_syllable_structure_tags(self, tokens: List[Token]) -> None:
        """Add syllable structure tags to tokens.
        
        Args:
            tokens: List of tokens to analyze
        """
        for i, token in enumerate(tokens):
            if token.kind == TokenKind.VOWEL:
                token.add_tag('syllable_nucleus')
            elif token.kind == TokenKind.CONSONANT:
                # Simple heuristic: consonants before vowels are onsets,
                # consonants after vowels are codas
                if i < len(tokens) - 1 and tokens[i + 1].kind == TokenKind.VOWEL:
                    token.add_tag('syllable_onset')
                elif i > 0 and tokens[i - 1].kind == TokenKind.VOWEL:
                    token.add_tag('syllable_coda')
                else:
                    token.add_tag('syllable_onset')  # Default for isolated consonants
            elif token.kind == TokenKind.OTHER and any(char in self.vowels for char in token.text):
                # Handle cases where whole words are classified as OTHER but contain vowels
                token.add_tag('syllable_nucleus')
    
    def _detect_compound_words(self, tokens: List[Token]) -> None:
        """Detect compound word boundaries in token sequence.
        
        Args:
            tokens: List of tokens to analyze
        """
        # Look for patterns that suggest compound boundaries
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            
            # Simple heuristic: marker followed by consonant suggests compound boundary
            if current.kind == TokenKind.MARKER and next_token.kind == TokenKind.CONSONANT:
                next_token.add_tag('compound_boundary')
    
    def _detect_sandhi_boundaries(self, tokens: List[Token]) -> None:
        """Detect potential sandhi boundaries in token sequence.
        
        Args:
            tokens: List of tokens to analyze
        """
        # Look for vowel-vowel or vowel-consonant boundaries that might involve sandhi
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            
            # Vowel followed by vowel or certain consonants
            if (current.kind == TokenKind.VOWEL and 
                (next_token.kind == TokenKind.VOWEL or 
                 (next_token.kind == TokenKind.CONSONANT and next_token.text in ['y', 'r', 'v', 'h']))):
                current.add_tag('sandhi_boundary')
    
    def _add_enhanced_linguistic_features(self, tokens: List[Token]) -> None:
        """Add enhanced linguistic features to all tokens."""
        for i, token in enumerate(tokens):
            # Add positional context
            self._add_positional_context(token, i, len(tokens))
            
            # Add syllable information
            self._add_syllable_information(token, tokens, i)
            
            # Add morphological context
            self._add_morphological_context(token, tokens, i)
    
    def _add_positional_context(self, token: Token, index: int, total_tokens: int) -> None:
        """Add positional context information to token."""
        if index == 0:
            token.add_tag('word_initial')
            token.syllable_info['position'] = 'initial'
        elif index == total_tokens - 1:
            token.add_tag('word_final')
            token.syllable_info['position'] = 'final'
        else:
            token.add_tag('word_medial')
            token.syllable_info['position'] = 'medial'
        
        # Add relative position information
        token.syllable_info['index'] = index
        token.syllable_info['total_tokens'] = total_tokens
    
    def _add_syllable_information(self, token: Token, tokens: List[Token], index: int) -> None:
        """Add syllable structure information to token."""
        if token.kind == TokenKind.VOWEL or token.kind == TokenKind.COMPOUND_VOWEL:
            token.add_tag('syllable_nucleus')
            token.syllable_info['role'] = 'nucleus'
            
            # Check for syllable boundaries
            if index > 0 and tokens[index-1].kind in [TokenKind.CONSONANT, TokenKind.CONJUNCT]:
                token.syllable_info['has_onset'] = True
            if index < len(tokens) - 1 and tokens[index+1].kind == TokenKind.CONSONANT:
                token.syllable_info['has_coda'] = True
                
        elif token.kind in [TokenKind.CONSONANT, TokenKind.CONJUNCT]:
            # Determine consonant role in syllable
            if index < len(tokens) - 1 and tokens[index+1].kind in [TokenKind.VOWEL, TokenKind.COMPOUND_VOWEL]:
                token.add_tag('syllable_onset')
                token.syllable_info['role'] = 'onset'
            elif index > 0 and tokens[index-1].kind in [TokenKind.VOWEL, TokenKind.COMPOUND_VOWEL]:
                token.add_tag('syllable_coda')
                token.syllable_info['role'] = 'coda'
            else:
                token.add_tag('syllable_onset')  # Default
                token.syllable_info['role'] = 'onset'
    
    def _add_morphological_context(self, token: Token, tokens: List[Token], index: int) -> None:
        """Add morphological context information to token."""
        # Check for morpheme boundaries
        if index > 0 and tokens[index-1].kind == TokenKind.MARKER:
            token.add_tag('morpheme_initial')
            token.morphological_features['boundary_before'] = True
        
        if index < len(tokens) - 1 and tokens[index+1].kind == TokenKind.MARKER:
            token.add_tag('morpheme_final')
            token.morphological_features['boundary_after'] = True
        
        # Analyze morphological role
        if token.kind == TokenKind.MARKER:
            self._analyze_morphological_marker(token, tokens, index)
    
    def _analyze_morphological_marker(self, token: Token, tokens: List[Token], index: int) -> None:
        """Analyze the role of morphological markers."""
        marker = token.text
        
        # Analyze context before and after marker
        before_context = tokens[index-1] if index > 0 else None
        after_context = tokens[index+1] if index < len(tokens) - 1 else None
        
        if marker == '+':
            token.morphological_features['function'] = 'compound_junction'
            if before_context and after_context:
                token.morphological_features['joins'] = [before_context.text, after_context.text]
        elif marker == '-':
            token.morphological_features['function'] = 'hyphenation'
        elif marker == '_':
            token.morphological_features['function'] = 'stem_case_boundary'
        elif marker == ':':
            token.morphological_features['function'] = 'grammatical_boundary'
    
    def _detect_compound_vowel_sequences(self, tokens: List[Token]) -> None:
        """Detect and mark compound vowel sequences."""
        for i, token in enumerate(tokens):
            if token.kind == TokenKind.COMPOUND_VOWEL:
                # Mark surrounding context
                if i > 0:
                    tokens[i-1].add_tag('pre_compound_vowel')
                if i < len(tokens) - 1:
                    tokens[i+1].add_tag('post_compound_vowel')
                
                # Analyze compound vowel context for sandhi
                self._analyze_compound_vowel_sandhi(token, tokens, i)
    
    def _analyze_compound_vowel_sandhi(self, token: Token, tokens: List[Token], index: int) -> None:
        """Analyze sandhi context for compound vowels."""
        if token.normalized_form in ['ai', 'ऐ']:
            # Check if this could be result of a + i sandhi
            token.phonetic_features['possible_sandhi'] = 'a_plus_i'
            token.add_tag('sandhi_result')
        elif token.normalized_form in ['au', 'औ']:
            # Check if this could be result of a + u sandhi
            token.phonetic_features['possible_sandhi'] = 'a_plus_u'
            token.add_tag('sandhi_result')
    
    def _analyze_syllable_structure(self, tokens: List[Token]) -> None:
        """Analyze overall syllable structure of the token sequence."""
        syllable_count = 0
        current_syllable = []
        
        for token in tokens:
            if token.kind in [TokenKind.VOWEL, TokenKind.COMPOUND_VOWEL]:
                syllable_count += 1
                current_syllable.append(token)
                
                # Mark syllable completion
                for syl_token in current_syllable:
                    syl_token.syllable_info['syllable_number'] = syllable_count
                
                current_syllable = []
            elif token.kind in [TokenKind.CONSONANT, TokenKind.CONJUNCT]:
                current_syllable.append(token)
        
        # Add syllable count to all tokens
        for token in tokens:
            token.syllable_info['total_syllables'] = syllable_count
    
    def _detect_morphological_boundaries(self, tokens: List[Token]) -> None:
        """Detect morphological boundaries using enhanced patterns."""
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            
            # Apply morphological boundary patterns
            for pattern, boundary_type in self.morpheme_boundary_patterns:
                text_sequence = current.normalized_form + next_token.normalized_form
                if re.search(pattern, text_sequence):
                    current.add_tag('morpheme_boundary')
                    current.morphological_features['boundary_type'] = boundary_type
                    next_token.add_tag('morpheme_start')
    
    def _analyze_sandhi_contexts(self, tokens: List[Token]) -> None:
        """Analyze sandhi contexts using enhanced patterns."""
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            
            # Apply sandhi patterns
            for pattern, sandhi_type in self.sandhi_patterns:
                text_sequence = current.normalized_form + next_token.normalized_form
                if re.search(pattern, text_sequence):
                    current.add_tag('sandhi_boundary')
                    current.phonetic_features['sandhi_type'] = sandhi_type
                    current.phonetic_features['sandhi_context'] = text_sequence
                    
                    # Mark the next token as potentially affected
                    next_token.add_tag('sandhi_affected')
    
    def _add_phonological_features(self, tokens: List[Token]) -> None:
        """Add phonological features to tokens."""
        for token in tokens:
            if token.kind in [TokenKind.CONSONANT, TokenKind.CONJUNCT]:
                self._add_consonant_phonological_features(token)
            elif token.kind in [TokenKind.VOWEL, TokenKind.COMPOUND_VOWEL]:
                self._add_vowel_phonological_features(token)
    
    def _add_consonant_phonological_features(self, token: Token) -> None:
        """Add phonological features specific to consonants."""
        char = token.normalized_form[0] if token.normalized_form else ''
        
        # Place of articulation
        if char in self.velar_consonants:
            token.phonetic_features['place'] = 'velar'
        elif char in self.palatal_consonants:
            token.phonetic_features['place'] = 'palatal'
        elif char in self.retroflex_consonants:
            token.phonetic_features['place'] = 'retroflex'
        elif char in self.dental_consonants:
            token.phonetic_features['place'] = 'dental'
        elif char in self.labial_consonants:
            token.phonetic_features['place'] = 'labial'
        
        # Manner of articulation
        if char in self.stops:
            token.phonetic_features['manner'] = 'stop'
        elif char in self.nasals:
            token.phonetic_features['manner'] = 'nasal'
        elif char in self.fricatives:
            token.phonetic_features['manner'] = 'fricative'
        elif char in self.liquids:
            token.phonetic_features['manner'] = 'liquid'
        elif char in self.glides:
            token.phonetic_features['manner'] = 'glide'
        
        # Voicing
        if char in self.voiced_consonants:
            token.phonetic_features['voicing'] = 'voiced'
        elif char in self.voiceless_consonants:
            token.phonetic_features['voicing'] = 'voiceless'
        
        # Aspiration
        if char in self.aspirated_consonants:
            token.phonetic_features['aspiration'] = 'aspirated'
        else:
            token.phonetic_features['aspiration'] = 'unaspirated'
    
    def _add_vowel_phonological_features(self, token: Token) -> None:
        """Add phonological features specific to vowels."""
        vowel = token.normalized_form
        
        # Length
        if vowel in self.short_vowels:
            token.phonetic_features['length'] = 'short'
        elif vowel in self.long_vowels:
            token.phonetic_features['length'] = 'long'
        
        # Height and backness (simplified)
        if vowel in ['i', 'ī', 'इ', 'ई']:
            token.phonetic_features['height'] = 'high'
            token.phonetic_features['backness'] = 'front'
        elif vowel in ['u', 'ū', 'उ', 'ऊ']:
            token.phonetic_features['height'] = 'high'
            token.phonetic_features['backness'] = 'back'
        elif vowel in ['e', 'ए']:
            token.phonetic_features['height'] = 'mid'
            token.phonetic_features['backness'] = 'front'
        elif vowel in ['o', 'ओ']:
            token.phonetic_features['height'] = 'mid'
            token.phonetic_features['backness'] = 'back'
        elif vowel in ['a', 'अ']:
            token.phonetic_features['height'] = 'low'
            token.phonetic_features['backness'] = 'central'
        elif vowel in ['ā', 'आ']:
            token.phonetic_features['height'] = 'low'
            token.phonetic_features['backness'] = 'back'
        
        # Vocalic consonants
        if vowel in self.vocalic_consonants:
            token.add_tag('vocalic_consonant')
            token.phonetic_features['type'] = 'vocalic_consonant'

    def get_token_statistics(self, tokens: List[Token]) -> Dict[str, int]:
        """Get statistics about a list of tokens.
        
        Args:
            tokens: List of tokens to analyze
            
        Returns:
            Dictionary with token statistics
        """
        stats = {
            'total_tokens': len(tokens),
            'vowel_count': 0,
            'consonant_count': 0,
            'marker_count': 0,
            'other_count': 0
        }
        
        for token in tokens:
            if token.kind == TokenKind.VOWEL:
                stats['vowel_count'] += 1
            elif token.kind == TokenKind.CONSONANT:
                stats['consonant_count'] += 1
            elif token.kind == TokenKind.MARKER:
                stats['marker_count'] += 1
            else:
                stats['other_count'] += 1
        
        return stats