"""
Morphological analyzer for Sanskrit text processing.

This module implements sophisticated morphological analysis including:
- Word segmentation into dhātu (roots), pratyaya (suffixes), and prefixes
- Grammatical category tagging: vibhakti (case), kāraka, tense, etc.
- Compound analysis (samāsa) with constituent identification
- Context-based morphological disambiguation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import re
from .token import Token, TokenKind


class MorphemeType(Enum):
    """Types of morphemes in Sanskrit morphological analysis."""
    ROOT = "ROOT"             # Root (alias for DHATU)
    DHATU = "DHATU"           # Root
    SUFFIX = "SUFFIX"         # Suffix (alias for PRATYAYA)
    PRATYAYA = "PRATYAYA"     # Suffix
    INFLECTION = "INFLECTION" # Inflectional suffix
    UPASARGA = "UPASARGA"     # Prefix
    NIPATA = "NIPATA"         # Particle
    SAMASA = "SAMASA"         # Compound


class GrammaticalCategory(Enum):
    """Sanskrit grammatical categories."""
    # Vibhakti (Cases)
    PRATHAMA = "PRATHAMA"     # Nominative
    DVITIYA = "DVITIYA"       # Accusative
    TRITIYA = "TRITIYA"       # Instrumental
    CHATURTHI = "CHATURTHI"   # Dative
    PANCHAMI = "PANCHAMI"     # Ablative
    SHASHTI = "SHASHTI"       # Genitive
    SAPTAMI = "SAPTAMI"       # Locative
    SAMBODHAN = "SAMBODHAN"   # Vocative
    
    # Vacana (Number)
    EKAVACANA = "EKAVACANA"   # Singular
    DVIVACANA = "DVIVACANA"   # Dual
    BAHUVACANA = "BAHUVACANA" # Plural
    
    # Linga (Gender)
    PUMLINGA = "PUMLINGA"     # Masculine
    STRILINGA = "STRILINGA"   # Feminine
    NAPUMSAKALINGA = "NAPUMSAKALINGA"  # Neuter
    
    # Kala (Tense)
    VARTAMANA = "VARTAMANA"   # Present
    BHUTA = "BHUTA"           # Past
    BHAVISHYAT = "BHAVISHYAT" # Future
    
    # Purusha (Person)
    PRATHAMA_PURUSHA = "PRATHAMA_PURUSHA"  # Third person
    MADHYAMA_PURUSHA = "MADHYAMA_PURUSHA"  # Second person
    UTTAMA_PURUSHA = "UTTAMA_PURUSHA"      # First person


class SamasaType(Enum):
    """Types of Sanskrit compounds (samāsa)."""
    AVYAYIBHAVA = "AVYAYIBHAVA"       # Adverbial compound
    TATPURUSHA = "TATPURUSHA"         # Determinative compound
    KARMADHARAYA = "KARMADHARAYA"     # Descriptive compound
    DVIGU = "DVIGU"                   # Numerical compound
    DVANDVA = "DVANDVA"               # Copulative compound
    BAHUVRIHI = "BAHUVRIHI"           # Possessive compound


@dataclass
class Morpheme:
    """
    A morpheme with its linguistic properties.
    
    Attributes:
        text: The morpheme text
        type: Type of morpheme (root, suffix, etc.)
        meaning: Semantic meaning or function
        grammatical_info: Grammatical properties
        position: Position in the original word
        confidence: Confidence score for this analysis
    """
    text: str
    type: MorphemeType
    meaning: Optional[str] = None
    grammatical_info: Dict[str, Any] = field(default_factory=dict)
    position: Optional[int] = None
    confidence: float = 1.0
    
    def add_grammatical_feature(self, category: str, value: Any) -> None:
        """Add a grammatical feature to this morpheme."""
        self.grammatical_info[category] = value
    
    def has_grammatical_feature(self, category: str) -> bool:
        """Check if morpheme has a specific grammatical feature."""
        return category in self.grammatical_info
    
    def get_grammatical_feature(self, category: str, default=None) -> Any:
        """Get a grammatical feature value."""
        return self.grammatical_info.get(category, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert morpheme to dictionary for serialization."""
        return {
            'text': self.text,
            'type': self.type.value,
            'meaning': self.meaning,
            'grammatical_info': self.grammatical_info,
            'position': self.position,
            'confidence': self.confidence
        }


@dataclass
class CompoundAnalysis:
    """
    Analysis of a Sanskrit compound (samāsa).
    
    Attributes:
        compound_text: The full compound text
        type: Type of compound
        constituents: List of constituent morphemes
        head: The head of the compound (if applicable)
        modifier: The modifier of the compound (if applicable)
        semantic_relation: Semantic relationship between constituents
        confidence: Confidence score for this analysis
    """
    compound_text: str
    type: SamasaType
    constituents: List[Morpheme]
    head: Optional[Morpheme] = None
    modifier: Optional[Morpheme] = None
    semantic_relation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class MorphologicalAnalysis:
    """
    Complete morphological analysis of a word.
    
    Attributes:
        word: The analyzed word
        morphemes: List of identified morphemes
        grammatical_categories: Set of grammatical categories
        compound_analysis: Compound analysis if applicable
        alternative_analyses: Alternative possible analyses
        confidence: Overall confidence score
        context_info: Contextual information used in analysis
    """
    word: str
    morphemes: List[Morpheme]
    grammatical_categories: Set[GrammaticalCategory] = field(default_factory=set)
    compound_analysis: Optional[CompoundAnalysis] = None
    alternative_analyses: List['MorphologicalAnalysis'] = field(default_factory=list)
    confidence: float = 1.0
    context_info: Dict[str, Any] = field(default_factory=dict)
    
    def add_grammatical_category(self, category: GrammaticalCategory) -> None:
        """Add a grammatical category to this analysis."""
        self.grammatical_categories.add(category)
    
    def has_category(self, category: GrammaticalCategory) -> bool:
        """Check if analysis has a specific grammatical category."""
        return category in self.grammatical_categories
    
    def get_root_morphemes(self) -> List[Morpheme]:
        """Get all root morphemes (dhātu)."""
        return [m for m in self.morphemes if m.type == MorphemeType.DHATU]
    
    def get_suffix_morphemes(self) -> List[Morpheme]:
        """Get all suffix morphemes (pratyaya)."""
        return [m for m in self.morphemes if m.type == MorphemeType.PRATYAYA]
    
    def get_prefix_morphemes(self) -> List[Morpheme]:
        """Get all prefix morphemes (upasarga)."""
        return [m for m in self.morphemes if m.type == MorphemeType.UPASARGA]


class MorphologicalDatabase:
    """
    Database of Sanskrit morphological information.
    
    Contains roots, suffixes, prefixes, and their properties.
    """
    
    def __init__(self):
        self.dhatu_db = self._initialize_dhatu_database()
        self.pratyaya_db = self._initialize_pratyaya_database()
        self.upasarga_db = self._initialize_upasarga_database()
        self.compound_patterns = self._initialize_compound_patterns()
        self.inflection_patterns = self._initialize_inflection_patterns()
    
    def _initialize_dhatu_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of Sanskrit roots (dhātu)."""
        return {
            # Common roots with their meanings and classes
            "gam": {
                "meaning": "to go",
                "class": "bhvadi",
                "number": 1,
                "parasmaipada": True,
                "atmanepada": False
            },
            "kar": {
                "meaning": "to do, make",
                "class": "tanadi", 
                "number": 8,
                "parasmaipada": True,
                "atmanepada": True
            },
            "bhū": {
                "meaning": "to be, become",
                "class": "bhvadi",
                "number": 1,
                "parasmaipada": True,
                "atmanepada": False
            },
            "as": {
                "meaning": "to be",
                "class": "adadi",
                "number": 2,
                "parasmaipada": True,
                "atmanepada": False
            },
            "dā": {
                "meaning": "to give",
                "class": "juhotyadi",
                "number": 3,
                "parasmaipada": True,
                "atmanepada": False
            },
            "sthā": {
                "meaning": "to stand",
                "class": "bhvadi",
                "number": 1,
                "parasmaipada": True,
                "atmanepada": False
            },
            "paṭh": {
                "meaning": "to read",
                "class": "bhvadi",
                "number": 1,
                "parasmaipada": True,
                "atmanepada": False
            },
            "liKh": {
                "meaning": "to write",
                "class": "bhvadi",
                "number": 1,
                "parasmaipada": True,
                "atmanepada": False
            }
        }
    
    def _initialize_pratyaya_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of Sanskrit suffixes (pratyaya)."""
        return {
            # Nominal suffixes
            "a": {
                "type": "nominal",
                "function": "stem_formation",
                "gender": ["masculine", "neuter"],
                "examples": ["rāma", "phala"]
            },
            "ā": {
                "type": "nominal", 
                "function": "stem_formation",
                "gender": ["feminine"],
                "examples": ["ramā", "latā"]
            },
            "i": {
                "type": "nominal",
                "function": "stem_formation", 
                "gender": ["masculine", "feminine"],
                "examples": ["kavi", "mati"]
            },
            "ī": {
                "type": "nominal",
                "function": "stem_formation",
                "gender": ["feminine"],
                "examples": ["nadī", "devī"]
            },
            
            # Case endings
            "s": {
                "type": "case_ending",
                "vibhakti": "prathama",
                "vacana": "ekavacana",
                "linga": ["pumlinga"]
            },
            "am": {
                "type": "case_ending", 
                "vibhakti": "prathama",
                "vacana": "ekavacana",
                "linga": ["napumsakalinga"]
            },
            "au": {
                "type": "case_ending",
                "vibhakti": "prathama", 
                "vacana": "dvivacana",
                "linga": ["pumlinga", "napumsakalinga"]
            },
            "āḥ": {
                "type": "case_ending",
                "vibhakti": "prathama",
                "vacana": "bahuvacana", 
                "linga": ["pumlinga"]
            },
            "āni": {
                "type": "case_ending",
                "vibhakti": "prathama",
                "vacana": "bahuvacana",
                "linga": ["napumsakalinga"]
            },
            "asya": {
                "type": "case_ending",
                "vibhakti": "shashti",
                "vacana": "ekavacana",
                "linga": ["pumlinga", "napumsakalinga"]
            },
            
            # Verbal suffixes
            "ti": {
                "type": "verbal_ending",
                "purusha": "prathama",
                "vacana": "ekavacana",
                "pada": "parasmaipada"
            },
            "anti": {
                "type": "verbal_ending", 
                "purusha": "prathama",
                "vacana": "bahuvacana",
                "pada": "parasmaipada"
            },
            "si": {
                "type": "verbal_ending",
                "purusha": "madhyama", 
                "vacana": "ekavacana",
                "pada": "parasmaipada"
            },
            "mi": {
                "type": "verbal_ending",
                "purusha": "uttama",
                "vacana": "ekavacana", 
                "pada": "parasmaipada"
            }
        }
    
    def _initialize_upasarga_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of Sanskrit prefixes (upasarga)."""
        return {
            "pra": {
                "meaning": "forward, forth",
                "type": "directional",
                "examples": ["prayāti", "prakāśa"]
            },
            "vi": {
                "meaning": "apart, away",
                "type": "separative", 
                "examples": ["viyāti", "vikāśa"]
            },
            "sam": {
                "meaning": "together, with",
                "type": "collective",
                "examples": ["saṃgacchati", "saṃskāra"]
            },
            "upa": {
                "meaning": "near, towards",
                "type": "directional",
                "examples": ["upagacchati", "upakāra"]
            },
            "anu": {
                "meaning": "after, along",
                "type": "sequential",
                "examples": ["anugacchati", "anukāra"]
            },
            "ā": {
                "meaning": "towards, until",
                "type": "directional", 
                "examples": ["āgacchati", "ākāra"]
            },
            "ni": {
                "meaning": "down, into",
                "type": "directional",
                "examples": ["nigacchati", "nikāra"]
            },
            "ud": {
                "meaning": "up, out",
                "type": "directional",
                "examples": ["udgacchati", "utkāra"]
            }
        }
    
    def _initialize_compound_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for compound analysis."""
        return {
            "tatpurusha": {
                "pattern": r"(.+)(.+)",
                "head_position": "right",
                "relation": "dependency",
                "examples": ["rājaputra", "grāmavāsa"]
            },
            "karmadharaya": {
                "pattern": r"(.+)(.+)", 
                "head_position": "right",
                "relation": "qualification",
                "examples": ["nīlotpala", "mahārāja"]
            },
            "dvandva": {
                "pattern": r"(.+)(.+)",
                "head_position": "both",
                "relation": "coordination", 
                "examples": ["rāmalakṣmaṇa", "pitāmātā"]
            },
            "bahuvrihi": {
                "pattern": r"(.+)(.+)",
                "head_position": "external",
                "relation": "possession",
                "examples": ["cakrapāṇi", "gajavaktra"]
            }
        }
    
    def _initialize_inflection_patterns(self) -> Dict[str, List[str]]:
        """Initialize common inflection patterns."""
        return {
            "a_stem_masculine": [
                "s", "am", "ena", "āya", "āt", "asya", "e", "a",  # Singular
                "au", "au", "ābhyām", "ābhyām", "ābhyām", "ayoḥ", "ayoḥ", "au",  # Dual
                "āḥ", "ān", "aiḥ", "ebhyaḥ", "ebhyaḥ", "ānām", "eṣu", "āḥ"  # Plural
            ],
            "ā_stem_feminine": [
                "ā", "ām", "ayā", "āyai", "āyāḥ", "āyāḥ", "āyām", "e",  # Singular
                "e", "e", "ābhyām", "ābhyām", "ābhyām", "ayoḥ", "ayoḥ", "e",  # Dual
                "āḥ", "āḥ", "ābhiḥ", "ābhyaḥ", "ābhyaḥ", "ānām", "āsu", "āḥ"  # Plural
            ]
        }
    
    def lookup_dhatu(self, root: str) -> Optional[Dict[str, Any]]:
        """Look up a root in the dhātu database."""
        return self.dhatu_db.get(root)
    
    def lookup_pratyaya(self, suffix: str) -> Optional[Dict[str, Any]]:
        """Look up a suffix in the pratyaya database."""
        return self.pratyaya_db.get(suffix)
    
    def lookup_upasarga(self, prefix: str) -> Optional[Dict[str, Any]]:
        """Look up a prefix in the upasarga database."""
        return self.upasarga_db.get(prefix)


class SanskritMorphologicalAnalyzer:
    """
    Advanced morphological analyzer for Sanskrit text.
    
    Provides comprehensive morphological analysis including:
    - Root, suffix, and prefix identification
    - Grammatical category tagging
    - Compound analysis
    - Context-based disambiguation
    """
    
    def __init__(self, database: Optional[MorphologicalDatabase] = None):
        """
        Initialize the morphological analyzer.
        
        Args:
            database: Morphological database (creates default if None)
        """
        self.database = database or MorphologicalDatabase()
        self.compound_analyzer = CompoundAnalyzer(self.database)
        self.context_disambiguator = ContextDisambiguator()
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for morphological analysis."""
        # Pattern for identifying potential roots
        self.root_pattern = re.compile(r'[a-zA-Zāīūṛṝḷḹēōṃḥśṣṅñṇṭḍ]+')
        
        # Pattern for case endings
        self.case_ending_pattern = re.compile(r'(s|am|ena|āya|āt|asya|e|au|ābhyām|ayoḥ|āḥ|ān|aiḥ|ebhyaḥ|ānām|eṣu)$')
        
        # Pattern for verbal endings
        self.verbal_ending_pattern = re.compile(r'(ti|anti|si|tha|mi|maḥ|te|ante|se|dhve|e|mahe)$')
        
        # Pattern for common prefixes
        self.prefix_pattern = re.compile(r'^(pra|vi|sam|upa|anu|ā|ni|ud|abhi|adhi|ati)')
    
    def analyze_word(self, word: str, context: Optional[List[str]] = None) -> MorphologicalAnalysis:
        """
        Perform complete morphological analysis of a word.
        
        Args:
            word: The word to analyze
            context: Optional context words for disambiguation
            
        Returns:
            Complete morphological analysis
        """
        # Clean and normalize the word
        normalized_word = self._normalize_word(word)
        
        # Generate possible morphological segmentations
        segmentations = self._segment_word(normalized_word)
        
        # Analyze each segmentation
        analyses = []
        for segmentation in segmentations:
            analysis = self._analyze_segmentation(normalized_word, segmentation)
            if analysis:
                analyses.append(analysis)
        
        # Select best analysis or create default
        if analyses:
            best_analysis = self._select_best_analysis(analyses, context)
        else:
            best_analysis = self._create_default_analysis(normalized_word)
        
        # Add alternative analyses
        best_analysis.alternative_analyses = [a for a in analyses if a != best_analysis]
        
        # Perform compound analysis if applicable
        compound_analysis = self.compound_analyzer.analyze_compound(normalized_word)
        if compound_analysis:
            best_analysis.compound_analysis = compound_analysis
        
        # Apply context-based disambiguation
        if context:
            best_analysis = self.context_disambiguator.disambiguate(best_analysis, context)
        
        return best_analysis
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word for analysis."""
        # Remove extra whitespace and convert to lowercase
        normalized = word.strip().lower()
        
        # Handle common transliteration variations
        replacements = {
            'sh': 'ś',
            # Keep aspirated consonants as-is
            # 'ch' should remain 'ch' in Sanskrit
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _segment_word(self, word: str) -> List[List[str]]:
        """
        Generate possible morphological segmentations of a word.
        
        Returns:
            List of possible segmentations, each as a list of morphemes
        """
        segmentations = []
        
        # Try different segmentation strategies
        
        # 1. Prefix + Root + Suffix segmentation
        prefix_segmentations = self._segment_with_prefixes(word)
        segmentations.extend(prefix_segmentations)
        
        # 2. Root + Suffix segmentation (no prefix)
        suffix_segmentations = self._segment_with_suffixes(word)
        segmentations.extend(suffix_segmentations)
        
        # 3. Compound segmentation
        compound_segmentations = self._segment_compounds(word)
        segmentations.extend(compound_segmentations)
        
        # 4. Single morpheme (unsegmented)
        if not segmentations:
            segmentations.append([word])
        
        return segmentations
    
    def _segment_with_prefixes(self, word: str) -> List[List[str]]:
        """Segment word considering prefixes."""
        segmentations = []
        
        # Check for known prefixes
        for prefix in self.database.upasarga_db.keys():
            if word.startswith(prefix):
                remainder = word[len(prefix):]
                if remainder:
                    # Recursively segment the remainder
                    remainder_segmentations = self._segment_with_suffixes(remainder)
                    for seg in remainder_segmentations:
                        segmentations.append([prefix] + seg)
        
        return segmentations
    
    def _segment_with_suffixes(self, word: str) -> List[List[str]]:
        """Segment word considering suffixes."""
        segmentations = []
        
        # Try different suffix lengths
        for suffix_len in range(1, min(len(word), 6)):  # Max suffix length of 5
            potential_suffix = word[-suffix_len:]
            potential_root = word[:-suffix_len]
            
            if potential_root and self.database.lookup_pratyaya(potential_suffix):
                # Check if the root is valid
                if self._is_valid_root(potential_root):
                    segmentations.append([potential_root, potential_suffix])
        
        # Also try the word as a single root
        if self._is_valid_root(word):
            segmentations.append([word])
        
        return segmentations
    
    def _segment_compounds(self, word: str) -> List[List[str]]:
        """Segment potential compounds."""
        segmentations = []
        
        # Try different split points for binary compounds
        for i in range(2, len(word) - 1):
            first_part = word[:i]
            second_part = word[i:]
            
            # Check if both parts could be valid words
            if (self._is_potential_word(first_part) and 
                self._is_potential_word(second_part)):
                segmentations.append([first_part, second_part])
        
        return segmentations
    
    def _is_valid_root(self, root: str) -> bool:
        """Check if a string could be a valid Sanskrit root."""
        # Check database first
        if self.database.lookup_dhatu(root):
            return True
        
        # Apply heuristics for unknown roots
        if len(root) < 2:
            return False
        
        # Should contain at least one vowel
        vowels = set('aāiīuūṛṝḷḹeēoō')
        if not any(c in vowels for c in root):
            return False
        
        # Should not end with certain consonant clusters
        invalid_endings = ['kṣ', 'jñ', 'tr', 'pr']
        if any(root.endswith(ending) for ending in invalid_endings):
            return False
        
        return True
    
    def _is_potential_word(self, word: str) -> bool:
        """Check if a string could be a potential Sanskrit word."""
        if len(word) < 2:
            return False
        
        # Basic phonotactic constraints
        vowels = set('aāiīuūṛṝḷḹeēoō')
        consonants = set('kgṅcjñṭḍṇtdnpbmyrlvśṣsh')
        
        # Should contain at least one vowel
        if not any(c in vowels for c in word):
            return False
        
        # Check if word contains only Sanskrit characters
        sanskrit_chars = vowels | consonants | set('hḥṃ')
        if not all(c in sanskrit_chars for c in word):
            return False
        
        # Should not have invalid consonant clusters
        # This is a simplified check
        prev_char = ''
        for char in word:
            if prev_char in consonants and char in consonants:
                # Check for valid consonant clusters
                cluster = prev_char + char
                valid_clusters = ['kṣ', 'jñ', 'tr', 'pr', 'kr', 'gr', 'dr', 'br', 'cc', 'tt', 'nn', 'mm', 'll']
                # Allow geminated consonants (double consonants) which are common in Sanskrit
                if cluster not in valid_clusters and prev_char != char:
                    # Only reject if it's not a geminated consonant and not in valid clusters
                    pass  # For now, be permissive
            prev_char = char
        
        return True    

    def _analyze_segmentation(self, word: str, segmentation: List[str]) -> Optional[MorphologicalAnalysis]:
        """Analyze a specific morphological segmentation."""
        morphemes = []
        grammatical_categories = set()
        confidence = 1.0
        
        for i, segment in enumerate(segmentation):
            morpheme = self._analyze_morpheme(segment, i, len(segmentation))
            if morpheme:
                morphemes.append(morpheme)
                
                # Adjust confidence based on morpheme confidence
                confidence *= morpheme.confidence
                
                # Extract grammatical categories from morpheme
                if morpheme.type == MorphemeType.PRATYAYA:
                    categories = self._extract_grammatical_categories(morpheme)
                    grammatical_categories.update(categories)
            else:
                # Unknown morpheme, reduce confidence
                confidence *= 0.3
                # Create a default morpheme
                morpheme = Morpheme(
                    text=segment,
                    type=MorphemeType.DHATU if i == 0 else MorphemeType.PRATYAYA,
                    confidence=0.3
                )
                morphemes.append(morpheme)
        
        if not morphemes:
            return None
        
        analysis = MorphologicalAnalysis(
            word=word,
            morphemes=morphemes,
            grammatical_categories=grammatical_categories,
            confidence=confidence
        )
        
        return analysis
    
    def _analyze_morpheme(self, segment: str, position: int, total_segments: int) -> Optional[Morpheme]:
        """Analyze a single morpheme segment."""
        # Check if it's a known root
        dhatu_info = self.database.lookup_dhatu(segment)
        if dhatu_info:
            morpheme = Morpheme(
                text=segment,
                type=MorphemeType.DHATU,
                meaning=dhatu_info.get('meaning'),
                position=position,
                confidence=0.9
            )
            morpheme.add_grammatical_feature('class', dhatu_info.get('class'))
            morpheme.add_grammatical_feature('number', dhatu_info.get('number'))
            return morpheme
        
        # Check if it's a known suffix
        pratyaya_info = self.database.lookup_pratyaya(segment)
        if pratyaya_info:
            morpheme = Morpheme(
                text=segment,
                type=MorphemeType.PRATYAYA,
                position=position,
                confidence=0.9
            )
            for key, value in pratyaya_info.items():
                morpheme.add_grammatical_feature(key, value)
            return morpheme
        
        # Check if it's a known prefix
        upasarga_info = self.database.lookup_upasarga(segment)
        if upasarga_info:
            morpheme = Morpheme(
                text=segment,
                type=MorphemeType.UPASARGA,
                meaning=upasarga_info.get('meaning'),
                position=position,
                confidence=0.9
            )
            morpheme.add_grammatical_feature('type', upasarga_info.get('type'))
            return morpheme
        
        # Determine type based on position and patterns
        if position == 0 and total_segments > 1:
            # First segment is likely a prefix or root
            if self.prefix_pattern.match(segment):
                return Morpheme(text=segment, type=MorphemeType.UPASARGA, position=position, confidence=0.6)
            else:
                return Morpheme(text=segment, type=MorphemeType.DHATU, position=position, confidence=0.6)
        elif position == total_segments - 1 and total_segments > 1:
            # Last segment is likely a suffix
            return Morpheme(text=segment, type=MorphemeType.PRATYAYA, position=position, confidence=0.6)
        else:
            # Middle segment or single segment - likely a root
            return Morpheme(text=segment, type=MorphemeType.DHATU, position=position, confidence=0.5)
    
    def _extract_grammatical_categories(self, morpheme: Morpheme) -> Set[GrammaticalCategory]:
        """Extract grammatical categories from a morpheme."""
        categories = set()
        
        # Map morpheme features to grammatical categories
        feature_mappings = {
            'vibhakti': {
                'prathama': GrammaticalCategory.PRATHAMA,
                'dvitiya': GrammaticalCategory.DVITIYA,
                'tritiya': GrammaticalCategory.TRITIYA,
                'chaturthi': GrammaticalCategory.CHATURTHI,
                'panchami': GrammaticalCategory.PANCHAMI,
                'shashti': GrammaticalCategory.SHASHTI,
                'saptami': GrammaticalCategory.SAPTAMI,
                'sambodhan': GrammaticalCategory.SAMBODHAN
            },
            'vacana': {
                'ekavacana': GrammaticalCategory.EKAVACANA,
                'dvivacana': GrammaticalCategory.DVIVACANA,
                'bahuvacana': GrammaticalCategory.BAHUVACANA
            },
            'linga': {
                'pumlinga': GrammaticalCategory.PUMLINGA,
                'strilinga': GrammaticalCategory.STRILINGA,
                'napumsakalinga': GrammaticalCategory.NAPUMSAKALINGA
            },
            'purusha': {
                'prathama': GrammaticalCategory.PRATHAMA_PURUSHA,
                'madhyama': GrammaticalCategory.MADHYAMA_PURUSHA,
                'uttama': GrammaticalCategory.UTTAMA_PURUSHA
            }
        }
        
        for feature_type, mapping in feature_mappings.items():
            feature_value = morpheme.get_grammatical_feature(feature_type)
            if feature_value:
                # Handle both single values and lists
                if isinstance(feature_value, list):
                    for value in feature_value:
                        if value in mapping:
                            categories.add(mapping[value])
                elif feature_value in mapping:
                    categories.add(mapping[feature_value])
        
        return categories
    
    def _select_best_analysis(self, analyses: List[MorphologicalAnalysis], 
                            context: Optional[List[str]] = None) -> MorphologicalAnalysis:
        """Select the best analysis from multiple candidates."""
        if not analyses:
            raise ValueError("No analyses provided")
        
        if len(analyses) == 1:
            return analyses[0]
        
        # Score analyses based on various factors
        scored_analyses = []
        for analysis in analyses:
            score = self._score_analysis(analysis, context)
            scored_analyses.append((score, analysis))
        
        # Return the highest scoring analysis
        scored_analyses.sort(key=lambda x: x[0], reverse=True)
        return scored_analyses[0][1]
    
    def _score_analysis(self, analysis: MorphologicalAnalysis, 
                       context: Optional[List[str]] = None) -> float:
        """Score an analysis based on various factors."""
        score = analysis.confidence
        
        # Prefer analyses with known morphemes
        known_morphemes = sum(1 for m in analysis.morphemes if m.confidence > 0.8)
        score += known_morphemes * 0.2
        
        # Prefer analyses with more grammatical information
        score += len(analysis.grammatical_categories) * 0.1
        
        # Prefer analyses with compound information
        if analysis.compound_analysis:
            score += 0.3
        
        # Context-based scoring (simplified)
        if context:
            # This would be more sophisticated in a real implementation
            score += 0.1
        
        return score
    
    def _create_default_analysis(self, word: str) -> MorphologicalAnalysis:
        """Create a default analysis for unknown words."""
        # Assume it's a single root morpheme
        morpheme = Morpheme(
            text=word,
            type=MorphemeType.DHATU,
            confidence=0.3
        )
        
        analysis = MorphologicalAnalysis(
            word=word,
            morphemes=[morpheme],
            confidence=0.3
        )
        # Set the overall confidence to match the morpheme confidence
        analysis.confidence = morpheme.confidence
        return analysis
    
    def analyze_sentence(self, words: List[str]) -> List[MorphologicalAnalysis]:
        """
        Analyze morphology of all words in a sentence with context.
        
        Args:
            words: List of words to analyze
            
        Returns:
            List of morphological analyses with contextual information
        """
        analyses = []
        
        for i, word in enumerate(words):
            # Get context words (previous and next)
            context = []
            if i > 0:
                context.append(words[i-1])
            if i < len(words) - 1:
                context.append(words[i+1])
            
            analysis = self.analyze_word(word, context)
            analysis.context_info['sentence_position'] = i
            analysis.context_info['sentence_length'] = len(words)
            
            analyses.append(analysis)
        
        # Post-process for sentence-level consistency
        self._ensure_sentence_consistency(analyses)
        
        return analyses
    
    def _ensure_sentence_consistency(self, analyses: List[MorphologicalAnalysis]) -> None:
        """Ensure morphological analyses are consistent across the sentence."""
        # This is a placeholder for more sophisticated consistency checking
        # In a full implementation, this would check for:
        # - Agreement between adjectives and nouns
        # - Proper case relationships
        # - Verb-argument structure consistency
        pass


class CompoundAnalyzer:
    """Specialized analyzer for Sanskrit compounds (samāsa)."""
    
    def __init__(self, database: MorphologicalDatabase):
        self.database = database
    
    def analyze_compound(self, word: str) -> Optional[CompoundAnalysis]:
        """
        Analyze a word as a potential compound.
        
        Args:
            word: The word to analyze as a compound
            
        Returns:
            Compound analysis if the word appears to be a compound, None otherwise
        """
        # Try different compound types
        for compound_type in SamasaType:
            analysis = self._analyze_compound_type(word, compound_type)
            if analysis and analysis.confidence > 0.5:
                return analysis
        
        return None
    
    def _analyze_compound_type(self, word: str, compound_type: SamasaType) -> Optional[CompoundAnalysis]:
        """Analyze word as a specific type of compound."""
        if compound_type == SamasaType.TATPURUSHA:
            return self._analyze_tatpurusha(word)
        elif compound_type == SamasaType.KARMADHARAYA:
            return self._analyze_karmadharaya(word)
        elif compound_type == SamasaType.DVANDVA:
            return self._analyze_dvandva(word)
        elif compound_type == SamasaType.BAHUVRIHI:
            return self._analyze_bahuvrihi(word)
        else:
            return None
    
    def _analyze_tatpurusha(self, word: str) -> Optional[CompoundAnalysis]:
        """Analyze as tatpurusha compound."""
        # Try different split points
        for i in range(2, len(word) - 1):
            first_part = word[:i]
            second_part = word[i:]
            
            # Check if both parts are valid
            if self._is_valid_compound_member(first_part) and self._is_valid_compound_member(second_part):
                # Create morphemes for each part
                first_morpheme = Morpheme(text=first_part, type=MorphemeType.SAMASA, position=0)
                second_morpheme = Morpheme(text=second_part, type=MorphemeType.SAMASA, position=1)
                
                analysis = CompoundAnalysis(
                    compound_text=word,
                    type=SamasaType.TATPURUSHA,
                    constituents=[first_morpheme, second_morpheme],
                    head=second_morpheme,  # Head is typically the second member
                    modifier=first_morpheme,
                    semantic_relation="dependency",
                    confidence=0.7
                )
                
                return analysis
        
        return None
    
    def _analyze_karmadharaya(self, word: str) -> Optional[CompoundAnalysis]:
        """Analyze as karmadharaya compound."""
        # Similar to tatpurusha but with qualification relation
        for i in range(2, len(word) - 1):
            first_part = word[:i]
            second_part = word[i:]
            
            if self._is_valid_compound_member(first_part) and self._is_valid_compound_member(second_part):
                first_morpheme = Morpheme(text=first_part, type=MorphemeType.SAMASA, position=0)
                second_morpheme = Morpheme(text=second_part, type=MorphemeType.SAMASA, position=1)
                
                analysis = CompoundAnalysis(
                    compound_text=word,
                    type=SamasaType.KARMADHARAYA,
                    constituents=[first_morpheme, second_morpheme],
                    head=second_morpheme,
                    modifier=first_morpheme,
                    semantic_relation="qualification",
                    confidence=0.6
                )
                
                return analysis
        
        return None
    
    def _analyze_dvandva(self, word: str) -> Optional[CompoundAnalysis]:
        """Analyze as dvandva compound."""
        for i in range(2, len(word) - 1):
            first_part = word[:i]
            second_part = word[i:]
            
            if self._is_valid_compound_member(first_part) and self._is_valid_compound_member(second_part):
                first_morpheme = Morpheme(text=first_part, type=MorphemeType.SAMASA, position=0)
                second_morpheme = Morpheme(text=second_part, type=MorphemeType.SAMASA, position=1)
                
                analysis = CompoundAnalysis(
                    compound_text=word,
                    type=SamasaType.DVANDVA,
                    constituents=[first_morpheme, second_morpheme],
                    semantic_relation="coordination",
                    confidence=0.5  # Lower confidence as dvandva is less common
                )
                
                return analysis
        
        return None
    
    def _analyze_bahuvrihi(self, word: str) -> Optional[CompoundAnalysis]:
        """Analyze as bahuvrihi compound."""
        for i in range(2, len(word) - 1):
            first_part = word[:i]
            second_part = word[i:]
            
            if self._is_valid_compound_member(first_part) and self._is_valid_compound_member(second_part):
                first_morpheme = Morpheme(text=first_part, type=MorphemeType.SAMASA, position=0)
                second_morpheme = Morpheme(text=second_part, type=MorphemeType.SAMASA, position=1)
                
                analysis = CompoundAnalysis(
                    compound_text=word,
                    type=SamasaType.BAHUVRIHI,
                    constituents=[first_morpheme, second_morpheme],
                    semantic_relation="possession",
                    confidence=0.5
                )
                
                return analysis
        
        return None
    
    def _is_valid_compound_member(self, part: str) -> bool:
        """Check if a string could be a valid compound member."""
        if len(part) < 2:
            return False
        
        # Check if it's a known word or root
        if (self.database.lookup_dhatu(part) or 
            self.database.lookup_upasarga(part) or
            part in ['rāma', 'kṛṣṇa', 'deva', 'putra', 'kāra', 'vāsa']):  # Common compound members
            return True
        
        # Apply basic phonotactic rules
        vowels = set('aāiīuūṛṝḷḹeēoō')
        if not any(c in vowels for c in part):
            return False
        
        return True


class ContextDisambiguator:
    """Disambiguates morphological analyses using contextual information."""
    
    def disambiguate(self, analysis: MorphologicalAnalysis, 
                    context: List[str]) -> MorphologicalAnalysis:
        """
        Disambiguate morphological analysis using context.
        
        Args:
            analysis: The morphological analysis to disambiguate
            context: Context words for disambiguation
            
        Returns:
            Disambiguated analysis (may be the same as input)
        """
        # This is a simplified implementation
        # In a full system, this would use more sophisticated NLP techniques
        
        # If there are alternative analyses, try to select the best one based on context
        if analysis.alternative_analyses:
            best_analysis = self._select_contextually_appropriate_analysis(
                [analysis] + analysis.alternative_analyses, 
                context
            )
            if best_analysis != analysis:
                # Move the selected analysis to primary and others to alternatives
                best_analysis.alternative_analyses = [a for a in [analysis] + analysis.alternative_analyses 
                                                   if a != best_analysis]
                return best_analysis
        
        # Add contextual information to the analysis
        analysis.context_info['context_words'] = context
        analysis.context_info['context_based_confidence_boost'] = 0.1
        analysis.confidence = min(1.0, analysis.confidence + 0.1)
        
        return analysis
    
    def _select_contextually_appropriate_analysis(self, analyses: List[MorphologicalAnalysis], 
                                                context: List[str]) -> MorphologicalAnalysis:
        """Select the most contextually appropriate analysis."""
        # Simple heuristic: prefer analyses with more grammatical categories
        # that might agree with context words
        
        scored_analyses = []
        for analysis in analyses:
            score = analysis.confidence
            
            # Boost score based on grammatical richness
            score += len(analysis.grammatical_categories) * 0.05
            
            # Boost score if compound analysis is present and context suggests compounds
            if analysis.compound_analysis and len(context) > 0:
                score += 0.1
            
            scored_analyses.append((score, analysis))
        
        scored_analyses.sort(key=lambda x: x[0], reverse=True)
        return scored_analyses[0][1]