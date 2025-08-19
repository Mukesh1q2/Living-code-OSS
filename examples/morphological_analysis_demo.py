#!/usr/bin/env python3
"""
Demonstration of the Sanskrit Morphological Analyzer.

This script shows how to use the morphological analyzer to:
1. Analyze individual words into morphemes
2. Identify grammatical categories
3. Analyze compounds
4. Process sentences with context
"""

from sanskrit_rewrite_engine.morphological_analyzer import (
    SanskritMorphologicalAnalyzer, MorphemeType, GrammaticalCategory, SamasaType
)


def print_analysis(analysis, word_title="Word"):
    """Print a detailed morphological analysis."""
    print(f"\n{word_title}: {analysis.word}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    print("\nMorphemes:")
    for i, morpheme in enumerate(analysis.morphemes):
        print(f"  {i+1}. '{morpheme.text}' ({morpheme.type.value})")
        if morpheme.meaning:
            print(f"     Meaning: {morpheme.meaning}")
        if morpheme.grammatical_info:
            print(f"     Grammar: {morpheme.grammatical_info}")
        print(f"     Confidence: {morpheme.confidence:.2f}")
    
    if analysis.grammatical_categories:
        print(f"\nGrammatical Categories:")
        for category in analysis.grammatical_categories:
            print(f"  - {category.value}")
    
    if analysis.compound_analysis:
        compound = analysis.compound_analysis
        print(f"\nCompound Analysis:")
        print(f"  Type: {compound.type.value}")
        print(f"  Relation: {compound.semantic_relation}")
        print(f"  Constituents: {[c.text for c in compound.constituents]}")
        if compound.head:
            print(f"  Head: {compound.head.text}")
        if compound.modifier:
            print(f"  Modifier: {compound.modifier.text}")
    
    if analysis.alternative_analyses:
        print(f"\nAlternative Analyses: {len(analysis.alternative_analyses)}")
        for i, alt in enumerate(analysis.alternative_analyses):
            print(f"  {i+1}. Confidence: {alt.confidence:.2f}, Morphemes: {len(alt.morphemes)}")


def demo_basic_analysis():
    """Demonstrate basic morphological analysis."""
    print("=" * 60)
    print("BASIC MORPHOLOGICAL ANALYSIS")
    print("=" * 60)
    
    analyzer = SanskritMorphologicalAnalyzer()
    
    # Test words with different morphological structures
    test_words = [
        "gam",           # Simple root
        "gacchati",      # Root + suffix
        "pragacchati",   # Prefix + root + suffix
        "rāmasya",       # Stem + genitive ending
        "rāmaputra",     # Potential compound
    ]
    
    for word in test_words:
        analysis = analyzer.analyze_word(word)
        print_analysis(analysis)


def demo_compound_analysis():
    """Demonstrate compound analysis."""
    print("\n" + "=" * 60)
    print("COMPOUND ANALYSIS")
    print("=" * 60)
    
    analyzer = SanskritMorphologicalAnalyzer()
    
    # Test potential compounds
    compounds = [
        "devaputra",     # deva + putra (god's son)
        "mahārāja",      # mahā + rāja (great king)
        "rāmalakṣmaṇa",  # rāma + lakṣmaṇa (Rama and Lakshmana)
        "cakrapāṇi",     # cakra + pāṇi (wheel-handed, i.e., Vishnu)
    ]
    
    for compound in compounds:
        analysis = analyzer.analyze_word(compound)
        print_analysis(analysis, f"Compound")


def demo_sentence_analysis():
    """Demonstrate sentence-level analysis with context."""
    print("\n" + "=" * 60)
    print("SENTENCE-LEVEL ANALYSIS")
    print("=" * 60)
    
    analyzer = SanskritMorphologicalAnalyzer()
    
    # Simple Sanskrit sentence
    sentence = ["rāmaḥ", "vanam", "gacchati"]
    print(f"Sentence: {' '.join(sentence)}")
    print("(Rama goes to the forest)")
    
    analyses = analyzer.analyze_sentence(sentence)
    
    for i, analysis in enumerate(analyses):
        print_analysis(analysis, f"Word {i+1}")
        
        # Show context information
        if analysis.context_info:
            print(f"Context Info:")
            for key, value in analysis.context_info.items():
                print(f"  {key}: {value}")


def demo_morphological_features():
    """Demonstrate extraction of morphological features."""
    print("\n" + "=" * 60)
    print("MORPHOLOGICAL FEATURES")
    print("=" * 60)
    
    analyzer = SanskritMorphologicalAnalyzer()
    
    # Words with clear morphological features
    feature_words = [
        ("rāmasya", "Genitive case"),
        ("rāmau", "Dual number"),
        ("rāmāḥ", "Plural number"),
        ("gacchati", "3rd person singular verb"),
    ]
    
    for word, description in feature_words:
        print(f"\n{description}: {word}")
        analysis = analyzer.analyze_word(word)
        
        print(f"Morphemes: {[m.text for m in analysis.morphemes]}")
        
        if analysis.grammatical_categories:
            print(f"Categories: {[c.value for c in analysis.grammatical_categories]}")
        else:
            print("Categories: None detected")


def demo_database_lookup():
    """Demonstrate database lookup functionality."""
    print("\n" + "=" * 60)
    print("DATABASE LOOKUP")
    print("=" * 60)
    
    analyzer = SanskritMorphologicalAnalyzer()
    db = analyzer.database
    
    print("Sample Dhātu (Roots):")
    for root in ["gam", "kar", "bhū", "dā"]:
        info = db.lookup_dhatu(root)
        if info:
            print(f"  {root}: {info['meaning']} (class: {info['class']})")
    
    print("\nSample Pratyaya (Suffixes):")
    for suffix in ["ti", "asya", "am", "āḥ"]:
        info = db.lookup_pratyaya(suffix)
        if info:
            print(f"  {suffix}: {info.get('type', 'unknown type')}")
    
    print("\nSample Upasarga (Prefixes):")
    for prefix in ["pra", "vi", "sam", "upa"]:
        info = db.lookup_upasarga(prefix)
        if info:
            print(f"  {prefix}: {info['meaning']} ({info['type']})")


def main():
    """Run all demonstrations."""
    print("Sanskrit Morphological Analyzer Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_analysis()
        demo_compound_analysis()
        demo_sentence_analysis()
        demo_morphological_features()
        demo_database_lookup()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()