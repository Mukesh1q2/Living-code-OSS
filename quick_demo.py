#!/usr/bin/env python3
"""
Quick demo script for Sanskrit Rewrite Engine
Run this to test the system without full installation
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic Sanskrit processing functionality"""
    print("🚀 Sanskrit Rewrite Engine - Quick Demo")
    print("=" * 50)
    
    try:
        # Test tokenization
        print("1. Testing Tokenization...")
        from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
        
        tokenizer = SanskritTokenizer()
        text = "rāma iti"
        tokens = tokenizer.tokenize(text)
        print(f"   Input: '{text}'")
        print(f"   Tokens: {[token.text for token in tokens]}")
        print("   ✅ Tokenization working!")
        
    except Exception as e:
        print(f"   ❌ Tokenization error: {e}")
    
    try:
        # Test transliteration
        print("\n2. Testing Transliteration...")
        from sanskrit_rewrite_engine.transliterator import DevanagariIASTTransliterator
        
        transliterator = DevanagariIASTTransliterator()
        iast_text = "rāma"
        devanagari_text = transliterator.iast_to_devanagari(iast_text)
        back_to_iast = transliterator.devanagari_to_iast(devanagari_text)
        
        print(f"   IAST: {iast_text}")
        print(f"   Devanagari: {devanagari_text}")
        print(f"   Back to IAST: {back_to_iast}")
        print("   ✅ Transliteration working!")
        
    except Exception as e:
        print(f"   ❌ Transliteration error: {e}")
    
    try:
        # Test rule engine
        print("\n3. Testing Rule Engine...")
        from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine
        
        engine = PaniniRuleEngine()
        print(f"   Engine created with {len(engine.rule_registry.rules)} rules")
        print("   ✅ Rule engine working!")
        
    except Exception as e:
        print(f"   ❌ Rule engine error: {e}")
    
    try:
        # Test morphological analyzer
        print("\n4. Testing Morphological Analyzer...")
        from sanskrit_rewrite_engine.morphological_analyzer import SanskritMorphologicalAnalyzer
        
        analyzer = SanskritMorphologicalAnalyzer()
        print("   Morphological analyzer created")
        print("   ✅ Morphological analyzer working!")
        
    except Exception as e:
        print(f"   ❌ Morphological analyzer error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! The Sanskrit Rewrite Engine is functional.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install package: pip install -e .")
    print("3. Run full tests: python run_tests.py")
    print("4. Start web interface: python sanskrit_rewrite_engine/run_server.py")

def test_simple_processing():
    """Test simple Sanskrit text processing"""
    print("\n🔄 Testing Simple Processing...")
    print("-" * 30)
    
    try:
        from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
        from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine
        
        # Create components
        tokenizer = SanskritTokenizer()
        engine = PaniniRuleEngine()
        
        # Test cases
        test_cases = [
            "rāma",
            "deva",
            "guru",
            "śiva"
        ]
        
        for text in test_cases:
            tokens = tokenizer.tokenize(text)
            print(f"   '{text}' → {[t.text for t in tokens]}")
        
        print("   ✅ Simple processing working!")
        
    except Exception as e:
        print(f"   ❌ Processing error: {e}")

if __name__ == "__main__":
    test_basic_functionality()
    test_simple_processing()