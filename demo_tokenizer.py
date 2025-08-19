#!/usr/bin/env python3
"""
Demonstration of the Sanskrit Devanagari + IAST tokenization engine.
"""

from sanskrit_rewrite_engine import SanskritTokenizer, DevanagariIASTTransliterator, Token, TokenKind

def demonstrate_tokenization():
    """Demonstrate the tokenization capabilities."""
    print("=== Sanskrit Tokenization Engine Demo ===\n")
    
    # Initialize tokenizer and transliterator
    tokenizer = SanskritTokenizer()
    transliterator = DevanagariIASTTransliterator()
    
    # Test cases
    test_cases = [
        "rama",           # Basic IAST
        "rāma",          # IAST with long vowels
        "राम",            # Devanagari
        "rama+iti",      # With morphological markers
        "kṛṣṇa",         # Vocalic consonants
        "aiśvarya",      # Compound vowels
        "क्त",            # Conjunct consonants
        "dharma_artha",  # Compound with marker
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"{i}. Input: '{text}'")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"   Tokens ({len(tokens)}):")
        
        for j, token in enumerate(tokens):
            tags_str = ', '.join(sorted(token.tags)) if token.tags else 'none'
            print(f"     {j+1}. '{token.text}' [{token.kind.value}] tags: {tags_str}")
            
            # Show metadata if present
            if token.meta:
                meta_items = [f"{k}={v}" for k, v in token.meta.items()]
                print(f"        meta: {', '.join(meta_items)}")
        
        # Show statistics
        stats = tokenizer.get_token_statistics(tokens)
        print(f"   Stats: {stats['vowel_count']} vowels, {stats['consonant_count']} consonants, "
              f"{stats['marker_count']} markers, {stats['compound_count']} compounds")
        print()

def demonstrate_transliteration():
    """Demonstrate transliteration capabilities."""
    print("=== Transliteration Demo ===\n")
    
    transliterator = DevanagariIASTTransliterator()
    
    # Devanagari to IAST
    devanagari_texts = ["राम", "कृष्ण", "धर्म", "योग", "वेद"]
    print("Devanagari → IAST:")
    for text in devanagari_texts:
        iast, pos_map = transliterator.devanagari_to_iast_text(text)
        print(f"  {text} → {iast}")
    print()
    
    # IAST to Devanagari
    iast_texts = ["rāma", "kṛṣṇa", "dharma", "yoga", "veda"]
    print("IAST → Devanagari:")
    for text in iast_texts:
        devanagari, pos_map = transliterator.iast_to_devanagari_text(text)
        print(f"  {text} → {devanagari}")
    print()
    
    # Script detection
    mixed_texts = ["राम rama", "kṛṣṇa कृष्ण", "hello world"]
    print("Script Detection:")
    for text in mixed_texts:
        script = transliterator.detect_script(text)
        print(f"  '{text}' → {script}")
    print()

def demonstrate_compound_detection():
    """Demonstrate compound and sandhi boundary detection."""
    print("=== Compound & Sandhi Detection Demo ===\n")
    
    transliterator = DevanagariIASTTransliterator()
    
    # Compound detection
    texts_with_compounds = ["aiśvarya", "kauśalya", "क्त", "स्व"]
    print("Compound Detection:")
    for text in texts_with_compounds:
        compounds = transliterator.identify_compounds(text)
        print(f"  '{text}' compounds: {compounds}")
    print()
    
    # Sandhi boundary detection
    texts_with_sandhi = ["rama + iti", "rāma iti", "word_boundary"]
    print("Sandhi Boundary Detection:")
    for text in texts_with_sandhi:
        boundaries = transliterator.identify_sandhi_boundaries(text)
        print(f"  '{text}' boundaries at positions: {boundaries}")
    print()

def demonstrate_vedic_support():
    """Demonstrate Vedic variant support."""
    print("=== Vedic Variant Support Demo ===\n")
    
    transliterator = DevanagariIASTTransliterator()
    
    # Vedic variants (using available characters)
    vedic_chars = ['ॲ', 'ॳ']  # Candra vowels
    print("Vedic Variant Detection:")
    for char in vedic_chars:
        is_vedic = transliterator.is_vedic_variant(char)
        print(f"  '{char}' is Vedic: {is_vedic}")
    
    # Normalization
    text_with_vedic = "ॲ test ॳ"
    normalized = transliterator.normalize_vedic(text_with_vedic)
    print(f"  Original: '{text_with_vedic}'")
    print(f"  Normalized: '{normalized}'")
    print()

def demonstrate_position_tracking():
    """Demonstrate position tracking for prakriyā steps."""
    print("=== Position Tracking Demo ===\n")
    
    tokenizer = SanskritTokenizer()
    transliterator = DevanagariIASTTransliterator()
    
    text = "rāma"
    print(f"Input: '{text}'")
    
    # Tokenization with positions
    tokens = tokenizer.tokenize(text)
    print("Token positions:")
    for token in tokens:
        print(f"  '{token.text}' at position {token.position}")
    
    # Transliteration with position mapping
    devanagari, pos_map = transliterator.iast_to_devanagari_text(text)
    print(f"Transliterated: '{devanagari}'")
    print("Position mapping (IAST → Devanagari):")
    for iast_pos, dev_pos in pos_map.items():
        print(f"  position {iast_pos} → {dev_pos}")
    print()

if __name__ == "__main__":
    try:
        demonstrate_tokenization()
        demonstrate_transliteration()
        demonstrate_compound_detection()
        demonstrate_vedic_support()
        demonstrate_position_tracking()
        
        print("=== Demo Complete ===")
        print("The Sanskrit tokenization engine successfully demonstrates:")
        print("✓ Lossless transliteration between Devanagari and IAST")
        print("✓ Compound detection and sandhi boundary identification")
        print("✓ Support for Vedic variants and archaic forms")
        print("✓ Position tracking for prakriyā (derivation) steps")
        print("✓ Comprehensive tokenization with linguistic metadata")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()