"""
Devanagari ↔ IAST transliteration with lossless conversion support.
"""

from typing import Dict, Set, Tuple, Optional
import re


class DevanagariIASTTransliterator:
    """
    Bidirectional transliterator between Devanagari and IAST (International Alphabet of Sanskrit Transliteration).
    
    Supports:
    - Lossless conversion between scripts
    - Vedic variants and archaic forms
    - Compound character handling
    - Position tracking for derivation steps
    """
    
    def __init__(self):
        # Devanagari to IAST mapping
        self.devanagari_to_iast = {
            # Vowels
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
            'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ॡ': 'ḹ',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
            
            # Consonants
            'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa',
            'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña',
            'ट': 'ṭa', 'ठ': 'ṭha', 'ड': 'ḍa', 'ढ': 'ḍha', 'ण': 'ṇa',
            'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
            'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
            'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
            'श': 'śa', 'ष': 'ṣa', 'स': 'sa', 'ह': 'ha',
            
            # Vowel marks (mātrās)
            'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū',
            'ृ': 'ṛ', 'ॄ': 'ṝ', 'ॢ': 'ḷ', 'ॣ': 'ḹ',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
            
            # Special characters
            'ं': 'ṃ',  # Anusvāra
            'ः': 'ḥ',  # Visarga
            '्': '',   # Virāma (halanta)
            'ॐ': 'oṃ', # Om
            '।': '|',  # Daṇḍa
            '॥': '||', # Double daṇḍa
            
            # Vedic accents
            '॑': '́',   # Udātta
            '॒': '̀',   # Anudātta
            '॓': '̂',   # Svarita
            
            # Numbers
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
        }
        
        # Create reverse mapping for IAST to Devanagari
        # Prioritize independent vowels over vowel marks
        self.iast_to_devanagari = {}
        
        # First pass: vowel marks and consonants
        for dev, iast in self.devanagari_to_iast.items():
            if iast:  # Skip empty mappings like virāma
                self.iast_to_devanagari[iast] = dev
        
        # Second pass: override with independent vowels (higher priority)
        independent_vowels = {
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
            'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ॡ': 'ḹ',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        }
        for dev, iast in independent_vowels.items():
            self.iast_to_devanagari[iast] = dev
        
        # Vedic and archaic forms
        self.vedic_variants = {
            # Vedic ḷ vowels
            'ऌ': 'ḷ', 'ॡ': 'ḹ', 'ॢ': 'ḷ', 'ॣ': 'ḹ',
            # Archaic forms
            'ॲ': 'ê', 'ॳ': 'ô',  # Candra vowels
            'क़': 'qa', 'ख़': 'kha', 'ग़': 'ġa', 'ज़': 'za',
            'ड़': 'ṛa', 'ढ़': 'ṛha', 'फ़': 'fa', 'य़': 'ẏa',
        }
        
        # Compound vowel patterns
        self.compound_vowels = {
            'ai': ['a', 'i'],
            'au': ['a', 'u'],
            'ā': ['a'],  # Long vowel
            'ī': ['i'],  # Long vowel
            'ū': ['u'],  # Long vowel
            'ṛ': ['r'],  # Vocalic r
            'ṝ': ['r'],  # Long vocalic r
            'ḷ': ['l'],  # Vocalic l
            'ḹ': ['l'],  # Long vocalic l
        }
        
        # Sandhi boundary markers
        self.sandhi_markers = {'+', '_', ':', '-', '='}
        
        # Compile regex patterns for efficient processing
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient transliteration."""
        # Pattern for Devanagari characters (including conjuncts)
        devanagari_chars = ''.join(self.devanagari_to_iast.keys())
        self.devanagari_pattern = re.compile(f'[{re.escape(devanagari_chars)}]+')
        
        # Pattern for IAST characters (sorted by length, longest first)
        iast_chars = sorted(self.iast_to_devanagari.keys(), key=len, reverse=True)
        iast_pattern = '|'.join(re.escape(char) for char in iast_chars)
        self.iast_pattern = re.compile(f'({iast_pattern})')
        
        # Pattern for compound detection
        self.compound_pattern = re.compile(r'([kgcjṭḍtdpb])([hy])')
        
        # Pattern for sandhi boundaries
        sandhi_chars = ''.join(re.escape(c) for c in self.sandhi_markers)
        self.sandhi_pattern = re.compile(f'[{sandhi_chars}]')
    
    def devanagari_to_iast_text(self, text: str, preserve_positions: bool = True) -> Tuple[str, Dict[int, int]]:
        """
        Convert Devanagari text to IAST.
        
        Args:
            text: Input Devanagari text
            preserve_positions: Whether to track character position mappings
            
        Returns:
            Tuple of (converted_text, position_mapping)
        """
        result = []
        position_map = {}
        original_pos = 0
        result_pos = 0
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Handle multi-character sequences (conjuncts, etc.)
            if char in self.devanagari_to_iast:
                # Check for conjunct consonants
                if i + 1 < len(text) and text[i + 1] == '्':
                    # This is a conjunct, handle specially
                    conjunct_chars = [char]
                    j = i + 1
                    while j < len(text) and text[j] == '्':
                        conjunct_chars.append(text[j])
                        j += 1
                        if j < len(text):
                            conjunct_chars.append(text[j])
                            j += 1
                    
                    # Convert conjunct
                    conjunct_result = self._convert_conjunct(conjunct_chars)
                    result.append(conjunct_result)
                    
                    if preserve_positions:
                        for k in range(len(conjunct_chars)):
                            position_map[original_pos + k] = result_pos
                        result_pos += len(conjunct_result)
                    
                    i = j
                    original_pos = i
                else:
                    # Regular character
                    converted = self.devanagari_to_iast[char]
                    result.append(converted)
                    
                    if preserve_positions:
                        position_map[original_pos] = result_pos
                        result_pos += len(converted)
                    
                    i += 1
                    original_pos = i
            else:
                # Non-Devanagari character, preserve as-is
                result.append(char)
                if preserve_positions:
                    position_map[original_pos] = result_pos
                    result_pos += 1
                i += 1
                original_pos = i
        
        return ''.join(result), position_map
    
    def iast_to_devanagari_text(self, text: str, preserve_positions: bool = True) -> Tuple[str, Dict[int, int]]:
        """
        Convert IAST text to Devanagari.
        
        Args:
            text: Input IAST text
            preserve_positions: Whether to track character position mappings
            
        Returns:
            Tuple of (converted_text, position_mapping)
        """
        result = []
        position_map = {}
        
        # Use regex to find IAST sequences
        last_end = 0
        result_pos = 0
        
        for match in self.iast_pattern.finditer(text):
            start, end = match.span()
            
            # Add any non-IAST text before this match
            if start > last_end:
                non_iast = text[last_end:start]
                result.append(non_iast)
                if preserve_positions:
                    for i, char in enumerate(non_iast):
                        position_map[last_end + i] = result_pos + i
                    result_pos += len(non_iast)
            
            # Convert the IAST sequence
            iast_char = match.group(1)
            if iast_char in self.iast_to_devanagari:
                devanagari_char = self.iast_to_devanagari[iast_char]
                result.append(devanagari_char)
                
                if preserve_positions:
                    for i in range(start, end):
                        position_map[i] = result_pos
                    result_pos += len(devanagari_char)
            else:
                # Fallback: preserve original
                result.append(iast_char)
                if preserve_positions:
                    for i in range(start, end):
                        position_map[i] = result_pos + (i - start)
                    result_pos += len(iast_char)
            
            last_end = end
        
        # Add any remaining text
        if last_end < len(text):
            remaining = text[last_end:]
            result.append(remaining)
            if preserve_positions:
                for i, char in enumerate(remaining):
                    position_map[last_end + i] = result_pos + i
        
        return ''.join(result), position_map
    
    def _convert_conjunct(self, chars: list) -> str:
        """Convert a conjunct consonant sequence to IAST."""
        result = []
        i = 0
        while i < len(chars):
            char = chars[i]
            if char == '्':
                # Virāma - don't add 'a' to previous consonant
                i += 1
                continue
            elif char in self.devanagari_to_iast:
                converted = self.devanagari_to_iast[char]
                if converted.endswith('a') and i + 1 < len(chars) and chars[i + 1] == '्':
                    # Remove inherent 'a' before virāma
                    result.append(converted[:-1])
                else:
                    result.append(converted)
            else:
                result.append(char)
            i += 1
        
        return ''.join(result)
    
    def detect_script(self, text: str) -> str:
        """
        Detect whether text is primarily Devanagari or IAST.
        
        Returns:
            'devanagari', 'iast', or 'mixed'
        """
        devanagari_count = 0
        iast_count = 0
        
        for char in text:
            if char in self.devanagari_to_iast:
                devanagari_count += 1
            elif char in self.iast_to_devanagari:
                iast_count += 1
        
        if devanagari_count > iast_count * 2:
            return 'devanagari'
        elif iast_count > devanagari_count * 2:
            return 'iast'
        else:
            return 'mixed'
    
    def identify_compounds(self, text: str) -> list:
        """
        Identify compound characters and conjuncts in text.
        
        Returns:
            List of (start_pos, end_pos, compound_type) tuples
        """
        compounds = []
        
        # Find conjunct consonants
        for match in re.finditer(r'[क-ह]्[क-ह]', text):
            compounds.append((match.start(), match.end(), 'conjunct'))
        
        # Find compound vowels in IAST
        for match in re.finditer(r'(ai|au|ā|ī|ū|ṛ|ṝ|ḷ|ḹ)', text):
            compounds.append((match.start(), match.end(), 'compound_vowel'))
        
        return compounds
    
    def identify_sandhi_boundaries(self, text: str) -> list:
        """
        Identify potential sandhi boundaries in text.
        
        Returns:
            List of positions where sandhi might occur
        """
        boundaries = []
        
        # Explicit markers
        for match in self.sandhi_pattern.finditer(text):
            boundaries.append(match.start())
        
        # Word boundaries (spaces)
        for match in re.finditer(r'\s+', text):
            boundaries.append(match.start())
            boundaries.append(match.end())
        
        # Vowel-vowel boundaries (potential sandhi)
        vowel_pattern = r'[aeiouāīūṛṝḷḹ][aeiouāīūṛṝḷḹ]'
        for match in re.finditer(vowel_pattern, text):
            boundaries.append(match.start() + 1)
        
        return sorted(set(boundaries))
    
    def is_vedic_variant(self, char: str) -> bool:
        """Check if a character is a Vedic or archaic variant."""
        return char in self.vedic_variants
    
    def normalize_vedic(self, text: str) -> str:
        """Normalize Vedic variants to standard forms."""
        result = text
        for vedic, standard in self.vedic_variants.items():
            result = result.replace(vedic, standard)
        return result