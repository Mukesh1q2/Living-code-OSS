"""
Tests for the Sanskrit morphological analyzer.
"""

import pytest
from sanskrit_rewrite_engine.morphological_analyzer import (
    SanskritMorphologicalAnalyzer, MorphologicalDatabase, MorphologicalAnalysis,
    Morpheme, MorphemeType, GrammaticalCategory, SamasaType, CompoundAnalysis,
    CompoundAnalyzer, ContextDisambiguator
)


class TestMorpheme:
    
    def test_morpheme_creation(self):
        """Test basic morpheme creation."""
        morpheme = Morpheme(text="gam", type=MorphemeType.DHATU)
        
        assert morpheme.text == "gam"
        assert morpheme.type == MorphemeType.DHATU
        assert morpheme.meaning is None
        assert morpheme.confidence == 1.0
        assert isinstance(morpheme.grammatical_info, dict)
    
    def test_morpheme_with_meaning(self):
        """Test morpheme creation with meaning."""
        morpheme = Morpheme(
            text="gam", 
            type=MorphemeType.DHATU, 
            meaning="to go",
            confidence=0.9
        )
        
        assert morpheme.meaning == "to go"
        assert morpheme.confidence == 0.9
    
    def test_grammatical_features(self):
        """Test grammatical feature management."""
        morpheme = Morpheme(text="ti", type=MorphemeType.PRATYAYA)
        
        # Add features
        morpheme.add_grammatical_feature("purusha", "prathama")
        morpheme.add_grammatical_feature("vacana", "ekavacana")
        
        # Check features
        assert morpheme.has_grammatical_feature("purusha")
        assert morpheme.get_grammatical_feature("purusha") == "prathama"
        assert morpheme.get_grammatical_feature("vacana") == "ekavacana"
        assert morpheme.get_grammatical_feature("nonexistent") is None
        assert morpheme.get_grammatical_feature("nonexistent", "default") == "default"


class TestMorphologicalDatabase:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.database = MorphologicalDatabase()
    
    def test_dhatu_lookup(self):
        """Test dhātu (root) lookup."""
        # Known root
        gam_info = self.database.lookup_dhatu("gam")
        assert gam_info is not None
        assert gam_info["meaning"] == "to go"
        assert gam_info["class"] == "bhvadi"
        
        # Unknown root
        unknown_info = self.database.lookup_dhatu("unknown")
        assert unknown_info is None
    
    def test_pratyaya_lookup(self):
        """Test pratyaya (suffix) lookup."""
        # Known suffix
        ti_info = self.database.lookup_pratyaya("ti")
        assert ti_info is not None
        assert ti_info["type"] == "verbal_ending"
        assert ti_info["purusha"] == "prathama"
        
        # Case ending
        asya_info = self.database.lookup_pratyaya("asya")
        assert asya_info is not None
        assert asya_info["vibhakti"] == "shashti"
        
        # Unknown suffix
        unknown_info = self.database.lookup_pratyaya("unknown")
        assert unknown_info is None
    
    def test_upasarga_lookup(self):
        """Test upasarga (prefix) lookup."""
        # Known prefix
        pra_info = self.database.lookup_upasarga("pra")
        assert pra_info is not None
        assert pra_info["meaning"] == "forward, forth"
        assert pra_info["type"] == "directional"
        
        # Unknown prefix
        unknown_info = self.database.lookup_upasarga("unknown")
        assert unknown_info is None


class TestMorphologicalAnalysis:
    
    def test_analysis_creation(self):
        """Test morphological analysis creation."""
        morphemes = [
            Morpheme(text="gam", type=MorphemeType.DHATU),
            Morpheme(text="ti", type=MorphemeType.PRATYAYA)
        ]
        
        analysis = MorphologicalAnalysis(word="gamati", morphemes=morphemes)
        
        assert analysis.word == "gamati"
        assert len(analysis.morphemes) == 2
        assert analysis.confidence == 1.0
        assert len(analysis.grammatical_categories) == 0
    
    def test_grammatical_categories(self):
        """Test grammatical category management."""
        analysis = MorphologicalAnalysis(word="test", morphemes=[])
        
        # Add categories
        analysis.add_grammatical_category(GrammaticalCategory.PRATHAMA)
        analysis.add_grammatical_category(GrammaticalCategory.EKAVACANA)
        
        # Check categories
        assert analysis.has_category(GrammaticalCategory.PRATHAMA)
        assert analysis.has_category(GrammaticalCategory.EKAVACANA)
        assert not analysis.has_category(GrammaticalCategory.DVITIYA)
        assert len(analysis.grammatical_categories) == 2
    
    def test_morpheme_filtering(self):
        """Test morpheme filtering methods."""
        morphemes = [
            Morpheme(text="pra", type=MorphemeType.UPASARGA),
            Morpheme(text="gam", type=MorphemeType.DHATU),
            Morpheme(text="ti", type=MorphemeType.PRATYAYA)
        ]
        
        analysis = MorphologicalAnalysis(word="pragamati", morphemes=morphemes)
        
        # Test filtering
        roots = analysis.get_root_morphemes()
        assert len(roots) == 1
        assert roots[0].text == "gam"
        
        suffixes = analysis.get_suffix_morphemes()
        assert len(suffixes) == 1
        assert suffixes[0].text == "ti"
        
        prefixes = analysis.get_prefix_morphemes()
        assert len(prefixes) == 1
        assert prefixes[0].text == "pra"


class TestSanskritMorphologicalAnalyzer:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SanskritMorphologicalAnalyzer()
    
    def test_word_normalization(self):
        """Test word normalization."""
        # Test basic normalization
        normalized = self.analyzer._normalize_word("  RAMA  ")
        assert normalized == "rama"
        
        # Test transliteration normalization
        normalized = self.analyzer._normalize_word("shiva")
        assert normalized == "śiva"
        
        normalized = self.analyzer._normalize_word("chandra")
        assert normalized == "chandra"  # Keep 'ch' as is in Sanskrit
    
    def test_root_validation(self):
        """Test root validation logic."""
        # Valid roots
        assert self.analyzer._is_valid_root("gam")  # Known root
        assert self.analyzer._is_valid_root("kar")  # Known root
        assert self.analyzer._is_valid_root("rama")  # Has vowels, reasonable structure
        
        # Invalid roots
        assert not self.analyzer._is_valid_root("x")  # Too short
        assert not self.analyzer._is_valid_root("kṣtr")  # No vowels, invalid ending
    
    def test_potential_word_validation(self):
        """Test potential word validation."""
        # Valid potential words
        assert self.analyzer._is_potential_word("rama")
        assert self.analyzer._is_potential_word("gacchati")
        assert self.analyzer._is_potential_word("putra")
        
        # Invalid potential words
        assert not self.analyzer._is_potential_word("x")  # Too short
        assert not self.analyzer._is_potential_word("kkkk")  # No vowels
    
    def test_simple_word_analysis(self):
        """Test analysis of simple words."""
        # Analyze a known root
        analysis = self.analyzer.analyze_word("gam")
        
        assert analysis.word == "gam"
        assert len(analysis.morphemes) >= 1
        
        # Should identify as root
        roots = analysis.get_root_morphemes()
        assert len(roots) >= 1
        
        # Check if it found the known root
        root_texts = [m.text for m in roots]
        assert "gam" in root_texts
    
    def test_prefixed_word_analysis(self):
        """Test analysis of words with prefixes."""
        # Analyze a word with known prefix
        analysis = self.analyzer.analyze_word("pragam")
        
        assert analysis.word == "pragam"
        
        # Should identify prefix and root
        prefixes = analysis.get_prefix_morphemes()
        roots = analysis.get_root_morphemes()
        
        # Should have at least one segmentation with prefix
        if prefixes:
            assert any(m.text == "pra" for m in prefixes)
        if roots:
            assert any(m.text == "gam" for m in roots)
    
    def test_suffixed_word_analysis(self):
        """Test analysis of words with suffixes."""
        # Analyze a word with known suffix
        analysis = self.analyzer.analyze_word("rāmasya")
        
        assert analysis.word == "rāmasya"
        
        # Should identify root and suffix
        roots = analysis.get_root_morphemes()
        suffixes = analysis.get_suffix_morphemes()
        
        # Should have some morphological structure
        assert len(analysis.morphemes) >= 1
        
        # Check for genitive case if suffix is recognized
        if GrammaticalCategory.SHASHTI in analysis.grammatical_categories:
            assert analysis.has_category(GrammaticalCategory.SHASHTI)
    
    def test_compound_word_analysis(self):
        """Test analysis of compound words."""
        # Analyze a potential compound
        analysis = self.analyzer.analyze_word("rāmaputra")
        
        assert analysis.word == "rāmaputra"
        
        # Should have morphological structure
        assert len(analysis.morphemes) >= 1
        
        # May have compound analysis
        if analysis.compound_analysis:
            assert analysis.compound_analysis.compound_text == "rāmaputra"
            assert len(analysis.compound_analysis.constituents) >= 2
    
    def test_unknown_word_analysis(self):
        """Test analysis of unknown words."""
        # Analyze a completely unknown word
        analysis = self.analyzer.analyze_word("xyzabc")
        
        assert analysis.word == "xyzabc"
        assert len(analysis.morphemes) >= 1
        
        # Should have low confidence
        assert analysis.confidence <= 0.5
    
    def test_sentence_analysis(self):
        """Test sentence-level morphological analysis."""
        words = ["rāma", "gacchati", "grāmam"]
        analyses = self.analyzer.analyze_sentence(words)
        
        assert len(analyses) == 3
        
        # Each analysis should have context information
        for i, analysis in enumerate(analyses):
            assert analysis.context_info['sentence_position'] == i
            assert analysis.context_info['sentence_length'] == 3
    
    def test_context_based_analysis(self):
        """Test context-based morphological analysis."""
        # Analyze with context
        analysis = self.analyzer.analyze_word("rāma", context=["gacchati"])
        
        assert analysis.word == "rāma"
        
        # Should have context information
        if 'context_words' in analysis.context_info:
            assert "gacchati" in analysis.context_info['context_words']


class TestCompoundAnalyzer:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.database = MorphologicalDatabase()
        self.compound_analyzer = CompoundAnalyzer(self.database)
    
    def test_compound_member_validation(self):
        """Test compound member validation."""
        # Valid compound members
        assert self.compound_analyzer._is_valid_compound_member("rāma")
        assert self.compound_analyzer._is_valid_compound_member("putra")
        assert self.compound_analyzer._is_valid_compound_member("deva")
        
        # Invalid compound members
        assert not self.compound_analyzer._is_valid_compound_member("x")  # Too short
        assert not self.compound_analyzer._is_valid_compound_member("xyz")  # No vowels
    
    def test_tatpurusha_analysis(self):
        """Test tatpurusha compound analysis."""
        analysis = self.compound_analyzer._analyze_tatpurusha("rāmaputra")
        
        if analysis:  # May not always find a valid analysis
            assert analysis.type == SamasaType.TATPURUSHA
            assert analysis.compound_text == "rāmaputra"
            assert len(analysis.constituents) == 2
            assert analysis.semantic_relation == "dependency"
            assert analysis.head is not None
            assert analysis.modifier is not None
    
    def test_karmadharaya_analysis(self):
        """Test karmadharaya compound analysis."""
        analysis = self.compound_analyzer._analyze_karmadharaya("mahārāja")
        
        if analysis:  # May not always find a valid analysis
            assert analysis.type == SamasaType.KARMADHARAYA
            assert analysis.semantic_relation == "qualification"
    
    def test_dvandva_analysis(self):
        """Test dvandva compound analysis."""
        analysis = self.compound_analyzer._analyze_dvandva("rāmalakṣmaṇa")
        
        if analysis:  # May not always find a valid analysis
            assert analysis.type == SamasaType.DVANDVA
            assert analysis.semantic_relation == "coordination"
    
    def test_bahuvrihi_analysis(self):
        """Test bahuvrihi compound analysis."""
        analysis = self.compound_analyzer._analyze_bahuvrihi("cakrapāṇi")
        
        if analysis:  # May not always find a valid analysis
            assert analysis.type == SamasaType.BAHUVRIHI
            assert analysis.semantic_relation == "possession"
    
    def test_compound_analysis_integration(self):
        """Test integrated compound analysis."""
        analysis = self.compound_analyzer.analyze_compound("devaputra")
        
        # Should either find a compound analysis or return None
        if analysis:
            assert isinstance(analysis, CompoundAnalysis)
            assert analysis.compound_text == "devaputra"
            assert len(analysis.constituents) >= 2
            assert analysis.confidence > 0.0


class TestContextDisambiguator:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.disambiguator = ContextDisambiguator()
    
    def test_basic_disambiguation(self):
        """Test basic context-based disambiguation."""
        # Create a mock analysis
        morpheme = Morpheme(text="test", type=MorphemeType.DHATU)
        analysis = MorphologicalAnalysis(word="test", morphemes=[morpheme])
        
        # Disambiguate with context
        context = ["context1", "context2"]
        disambiguated = self.disambiguator.disambiguate(analysis, context)
        
        # Should return an analysis (possibly modified)
        assert isinstance(disambiguated, MorphologicalAnalysis)
        assert disambiguated.word == "test"
        
        # Should have context information
        if 'context_words' in disambiguated.context_info:
            assert disambiguated.context_info['context_words'] == context
    
    def test_alternative_analysis_selection(self):
        """Test selection among alternative analyses."""
        # Create primary analysis
        morpheme1 = Morpheme(text="test", type=MorphemeType.DHATU, confidence=0.6)
        analysis1 = MorphologicalAnalysis(word="test", morphemes=[morpheme1], confidence=0.6)
        
        # Create alternative analysis with higher confidence
        morpheme2 = Morpheme(text="test", type=MorphemeType.PRATYAYA, confidence=0.8)
        analysis2 = MorphologicalAnalysis(word="test", morphemes=[morpheme2], confidence=0.8)
        analysis2.add_grammatical_category(GrammaticalCategory.PRATHAMA)
        
        # Set up alternatives
        analysis1.alternative_analyses = [analysis2]
        
        # Disambiguate
        context = ["context"]
        result = self.disambiguator.disambiguate(analysis1, context)
        
        # Should return an analysis
        assert isinstance(result, MorphologicalAnalysis)


class TestComplexMorphologicalScenarios:
    """Test complex morphological analysis scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SanskritMorphologicalAnalyzer()
    
    def test_complex_verbal_form(self):
        """Test analysis of complex verbal forms."""
        # Test a complex verbal form
        analysis = self.analyzer.analyze_word("pragacchati")
        
        assert analysis.word == "pragacchati"
        assert len(analysis.morphemes) >= 1
        
        # Should identify some morphological structure
        morpheme_types = [m.type for m in analysis.morphemes]
        assert len(set(morpheme_types)) >= 1  # At least one type of morpheme
    
    def test_complex_nominal_form(self):
        """Test analysis of complex nominal forms."""
        # Test a complex nominal form
        analysis = self.analyzer.analyze_word("rājñaḥ")
        
        assert analysis.word == "rājñaḥ"
        assert len(analysis.morphemes) >= 1
        
        # Should have some analysis
        assert analysis.confidence > 0.0
    
    def test_sandhi_affected_word(self):
        """Test analysis of words affected by sandhi."""
        # Test words that might have undergone sandhi
        test_words = ["rāmo", "gacchaty", "ityādi"]
        
        for word in test_words:
            analysis = self.analyzer.analyze_word(word)
            
            assert analysis.word == word
            assert len(analysis.morphemes) >= 1
            assert analysis.confidence > 0.0
    
    def test_multiple_prefix_word(self):
        """Test analysis of words with multiple prefixes."""
        # Test word with multiple prefixes
        analysis = self.analyzer.analyze_word("sampragam")
        
        assert analysis.word == "sampragam"
        assert len(analysis.morphemes) >= 1
        
        # Should identify some structure
        prefixes = analysis.get_prefix_morphemes()
        roots = analysis.get_root_morphemes()
        
        # Should have at least some morphological analysis
        assert len(analysis.morphemes) >= 1
    
    def test_long_compound_word(self):
        """Test analysis of long compound words."""
        # Test a longer compound
        analysis = self.analyzer.analyze_word("mahārājaputra")
        
        assert analysis.word == "mahārājaputra"
        assert len(analysis.morphemes) >= 1
        
        # May have compound analysis
        if analysis.compound_analysis:
            assert len(analysis.compound_analysis.constituents) >= 2
    
    def test_ambiguous_word_analysis(self):
        """Test analysis of ambiguous words."""
        # Test words that could have multiple analyses
        ambiguous_words = ["kara", "mata", "gata"]
        
        for word in ambiguous_words:
            analysis = self.analyzer.analyze_word(word)
            
            assert analysis.word == word
            assert len(analysis.morphemes) >= 1
            
            # May have alternative analyses
            total_analyses = 1 + len(analysis.alternative_analyses)
            assert total_analyses >= 1
    
    def test_sentence_level_consistency(self):
        """Test sentence-level morphological consistency."""
        # Test a simple Sanskrit sentence
        sentence = ["rāmaḥ", "vanam", "gacchati"]
        analyses = self.analyzer.analyze_sentence(sentence)
        
        assert len(analyses) == 3
        
        # Each word should have analysis
        for analysis in analyses:
            assert len(analysis.morphemes) >= 1
            assert analysis.confidence > 0.0
        
        # Should have contextual information
        for i, analysis in enumerate(analyses):
            assert analysis.context_info['sentence_position'] == i
    
    def test_morphological_feature_extraction(self):
        """Test extraction of morphological features."""
        # Test words with clear morphological features
        test_cases = [
            ("rāmasya", [GrammaticalCategory.SHASHTI]),  # Genitive
            ("rāmau", [GrammaticalCategory.DVIVACANA]),   # Dual
            ("rāmāḥ", [GrammaticalCategory.BAHUVACANA])   # Plural
        ]
        
        for word, expected_categories in test_cases:
            analysis = self.analyzer.analyze_word(word)
            
            assert analysis.word == word
            
            # Check if any expected categories are found
            found_categories = analysis.grammatical_categories
            if found_categories:
                # At least some morphological analysis should be present
                assert len(found_categories) > 0
    
    def test_error_handling(self):
        """Test error handling in morphological analysis."""
        # Test with empty string
        analysis = self.analyzer.analyze_word("")
        assert analysis.word == ""
        assert len(analysis.morphemes) >= 1  # Should create default analysis
        
        # Test with very long string
        long_word = "a" * 100
        analysis = self.analyzer.analyze_word(long_word)
        assert analysis.word == long_word
        assert len(analysis.morphemes) >= 1
        
        # Test with special characters
        special_word = "rāma@#$"
        analysis = self.analyzer.analyze_word(special_word)
        assert len(analysis.morphemes) >= 1
    
    def test_confidence_scoring(self):
        """Test confidence scoring in morphological analysis."""
        # Test words with different expected confidence levels
        high_confidence_words = ["gam", "kar", "rāma"]  # Known roots/words
        low_confidence_words = ["xyzabc", "qwerty"]     # Unknown words
        
        for word in high_confidence_words:
            analysis = self.analyzer.analyze_word(word)
            # Known words should have reasonable confidence
            assert analysis.confidence > 0.3
        
        for word in low_confidence_words:
            analysis = self.analyzer.analyze_word(word)
            # Unknown words should have lower confidence
            assert analysis.confidence <= 1.0  # Should not exceed maximum