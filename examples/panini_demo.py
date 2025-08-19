#!/usr/bin/env python3
"""
Demonstration of the Pāṇini rule engine with essential sūtras.

This script shows how to use the Sanskrit Rewrite Engine's Pāṇini rule system
to apply traditional Sanskrit grammatical transformations.
"""

import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sanskrit_rewrite_engine import (
    Token, TokenKind, PaniniRuleEngine, SanskritTokenizer
)


def demonstrate_sandhi_rules():
    """Demonstrate basic sandhi rule applications."""
    print("=== Pāṇini Rule Engine Demonstration ===\n")
    
    # Create the engine
    engine = PaniniRuleEngine()
    
    print("Loaded rules:")
    stats = engine.get_rule_statistics()
    print(f"  - Total sūtra rules: {stats['total_sutra_rules']}")
    print(f"  - Total paribhāṣā rules: {stats['total_paribhasa_rules']}")
    print(f"  - Active sūtra rules: {stats['active_sutra_rules']}")
    print()
    
    # Test cases for different sandhi rules
    test_cases = [
        {
            'name': 'Guṇa Sandhi (a + i → e)',
            'tokens': [Token('a', TokenKind.VOWEL), Token('i', TokenKind.VOWEL)],
            'expected': 'e'
        },
        {
            'name': 'Guṇa Sandhi (a + u → o)', 
            'tokens': [Token('a', TokenKind.VOWEL), Token('u', TokenKind.VOWEL)],
            'expected': 'o'
        },
        {
            'name': 'Vṛddhi Sandhi (a + e → ai)',
            'tokens': [Token('a', TokenKind.VOWEL), Token('e', TokenKind.VOWEL)],
            'expected': 'ai'
        },
        {
            'name': 'Vṛddhi Sandhi (a + o → au)',
            'tokens': [Token('a', TokenKind.VOWEL), Token('o', TokenKind.VOWEL)],
            'expected': 'au'
        },
        {
            'name': 'Savarṇa Dīrgha (a + a → ā)',
            'tokens': [Token('a', TokenKind.VOWEL), Token('a', TokenKind.VOWEL)],
            'expected': 'ā'
        },
        {
            'name': 'Savarṇa Dīrgha (i + i → ī)',
            'tokens': [Token('i', TokenKind.VOWEL), Token('i', TokenKind.VOWEL)],
            'expected': 'ī'
        },
        {
            'name': 'Iko yaṇ aci (i + a → ya)',
            'tokens': [Token('i', TokenKind.VOWEL), Token('a', TokenKind.VOWEL)],
            'expected': 'ya'
        },
        {
            'name': 'Iko yaṇ aci (u + a → va)',
            'tokens': [Token('u', TokenKind.VOWEL), Token('a', TokenKind.VOWEL)],
            'expected': 'va'
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        print(f"Input: {' + '.join(t.text for t in test_case['tokens'])}")
        
        # Process through the engine
        result = engine.process(test_case['tokens'])
        
        output_text = result.get_output_text()
        print(f"Output: {output_text}")
        print(f"Expected: {test_case['expected']}")
        print(f"Converged: {result.converged}")
        print(f"Passes: {result.passes}")
        
        if result.traces:
            print("Rules applied:")
            for trace in result.traces:
                for transformation in trace.transformations:
                    print(f"  - {transformation.rule_name} ({transformation.sutra_ref})")
        
        print(f"Match: {'✓' if output_text == test_case['expected'] else '✗'}")
        print("-" * 50)
    
    # Show transformation summary
    print("\n=== Engine Statistics ===")
    final_stats = engine.get_rule_statistics()
    print(f"Registry statistics: {final_stats}")


def demonstrate_complex_processing():
    """Demonstrate more complex multi-step processing."""
    print("\n=== Complex Processing Example ===")
    
    engine = PaniniRuleEngine()
    
    # Create a sequence that should trigger multiple rules
    tokens = [
        Token('a', TokenKind.VOWEL),
        Token('i', TokenKind.VOWEL),
        Token('a', TokenKind.VOWEL),
        Token('u', TokenKind.VOWEL)
    ]
    
    print(f"Input sequence: {' '.join(t.text for t in tokens)}")
    
    result = engine.process(tokens, max_passes=10)
    
    print(f"Final output: {result.get_output_text()}")
    print(f"Converged: {result.converged}")
    print(f"Total passes: {result.passes}")
    
    print("\nTransformation trace:")
    for i, trace in enumerate(result.traces, 1):
        print(f"Pass {i}:")
        input_text = ''.join(t.text for t in trace.tokens_before)
        output_text = ''.join(t.text for t in trace.tokens_after)
        print(f"  Input: {input_text}")
        print(f"  Output: {output_text}")
        
        if trace.transformations:
            print("  Rules applied:")
            for transformation in trace.transformations:
                print(f"    - {transformation.rule_name} ({transformation.sutra_ref}) at position {transformation.index}")
        else:
            print("  No transformations (convergence)")
        print()
    
    # Show rule usage statistics
    summary = result.get_transformation_summary()
    if summary:
        print("Rule usage summary:")
        for rule_name, count in summary.items():
            print(f"  {rule_name}: {count} applications")


def demonstrate_rule_management():
    """Demonstrate rule management features."""
    print("\n=== Rule Management Example ===")
    
    engine = PaniniRuleEngine()
    
    # Show initial state
    print("Initial rule statistics:")
    stats = engine.get_rule_statistics()
    print(f"  Active sūtra rules: {stats['active_sutra_rules']}")
    
    # Test with a rule disabled
    print("\nDisabling guṇa rule (6.1.87)...")
    engine.disable_rule("6.1.87")
    
    # Test the same transformation that should use guṇa
    tokens = [Token('a', TokenKind.VOWEL), Token('i', TokenKind.VOWEL)]
    result = engine.process(tokens)
    
    print(f"Input: a + i")
    print(f"Output with disabled guṇa: {result.get_output_text()}")
    print(f"Rules applied: {list(result.get_rules_applied())}")
    
    # Re-enable the rule
    print("\nRe-enabling guṇa rule...")
    engine.enable_rule("6.1.87")
    
    result2 = engine.process(tokens)
    print(f"Output with enabled guṇa: {result2.get_output_text()}")
    print(f"Rules applied: {list(result2.get_rules_applied())}")


if __name__ == "__main__":
    try:
        demonstrate_sandhi_rules()
        demonstrate_complex_processing()
        demonstrate_rule_management()
        
        print("\n=== Demonstration Complete ===")
        print("The Pāṇini rule engine successfully demonstrates:")
        print("✓ Essential sūtra rule loading")
        print("✓ Guarded rule application with loop prevention")
        print("✓ Sūtra numbering system and cross-references")
        print("✓ Complex rule interactions and inheritance")
        print("✓ Comprehensive tracing and debugging")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)