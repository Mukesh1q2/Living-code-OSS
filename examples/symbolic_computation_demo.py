#!/usr/bin/env python3
"""
Demonstration of Sanskrit Symbolic Computation Integration.

This script demonstrates the integration of SymPy for algebraic computations,
Sanskrit sūtra to mathematical formula translation, and mathematical proof
verification using sūtra logic.
"""

import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from sanskrit_rewrite_engine.symbolic_computation import (
        create_symbolic_computation_engine,
        apply_vedic_math,
        verify_mathematical_identity,
        MathematicalDomain
    )
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Demo not available: {e}")
    print("Please install SymPy: pip install sympy")
    DEMO_AVAILABLE = False


def demonstrate_vedic_mathematics():
    """Demonstrate Vedic Mathematics sūtras."""
    print("=" * 60)
    print("VEDIC MATHEMATICS DEMONSTRATION")
    print("=" * 60)
    
    # Ekadhikena Purvena (squaring numbers ending in 5)
    print("\n1. Ekadhikena Purvena (एकाधिकेन पूर्वेण)")
    print("   'One more than the previous' - for squaring numbers ending in 5")
    print("-" * 50)
    
    for number in [15, 25, 35, 45, 65, 85]:
        result = apply_vedic_math("ekadhikena_purvena", number)
        print(f"\n{number}² = {result['result']}")
        print("Steps:")
        for i, step in enumerate(result['steps'], 1):
            print(f"  {i}. {step}")
    
    # Nikhilam (multiplication near base)
    print("\n\n2. Nikhilam Navatashcaramam Dashatah (निखिलं नवतश्चरमं दशतः)")
    print("   'All from 9 and last from 10' - for multiplication near powers of 10")
    print("-" * 50)
    
    test_cases = [(97, 96, 100), (93, 87, 100), (998, 997, 1000)]
    for a, b, base in test_cases:
        result = apply_vedic_math("nikhilam", a, b, base=base)
        print(f"\n{a} × {b} = {result['result']}")
        print("Steps:")
        for i, step in enumerate(result['steps'], 1):
            print(f"  {i}. {step}")
    
    # Urdhva-Tiryagbyham (general multiplication)
    print("\n\n3. Urdhva-Tiryagbyham (ऊर्ध्वतिर्यग्भ्याम्)")
    print("   'Vertically and crosswise' - for general multiplication")
    print("-" * 50)
    
    test_cases = [(23, 47), (12, 34), (123, 456)]
    for a, b in test_cases:
        result = apply_vedic_math("urdhva_tiryagbyham", a, b)
        print(f"\n{a} × {b} = {result['result']}")
        print("Steps (showing cross-multiplication pattern):")
        for i, step in enumerate(result['steps'][:5], 1):  # Show first 5 steps
            print(f"  {i}. {step}")
        if len(result['steps']) > 5:
            print(f"  ... ({len(result['steps']) - 5} more steps)")
    
    # Paravartya Yojayet (solving linear equations)
    print("\n\n4. Paravartya Yojayet (परावर्त्य योजयेत्)")
    print("   'Transpose and apply' - for solving linear equations")
    print("-" * 50)
    
    # System: 2x + 3y = 7, 4x + 5y = 13
    equations = [([2, 3], 7), ([4, 5], 13)]
    result = apply_vedic_math("paravartya_yojayet", equations)
    
    print("\nSolving system:")
    print("  2x + 3y = 7")
    print("  4x + 5y = 13")
    print(f"\nSolution: x = {result['result'][0]}, y = {result['result'][1]}")
    print("Steps:")
    for i, step in enumerate(result['steps'], 1):
        print(f"  {i}. {step}")


def demonstrate_symbolic_algebra():
    """Demonstrate symbolic algebraic manipulation."""
    print("\n\n" + "=" * 60)
    print("SYMBOLIC ALGEBRA DEMONSTRATION")
    print("=" * 60)
    
    engine = create_symbolic_computation_engine()
    
    # Algebraic simplification
    print("\n1. Algebraic Simplification (Sanskrit Sandhi-like)")
    print("-" * 50)
    
    expressions = [
        "x + x + x",
        "2*x + 3*x - x",
        "sin(x)**2 + cos(x)**2",
        "(x + 1)**2 - (x**2 + 2*x + 1)"
    ]
    
    for expr in expressions:
        result = engine.process_mathematical_query(f"simplify {expr}")
        print(f"\nSimplify: {expr}")
        print(f"Result: {result['result']}")
        if result.get('steps'):
            print("Steps:")
            for step in result['steps']:
                print(f"  • {step}")
    
    # Algebraic expansion
    print("\n\n2. Algebraic Expansion (Sanskrit Compound-like)")
    print("-" * 50)
    
    expressions = [
        "(x + 1)*(x + 2)",
        "(x + y)**2",
        "(a + b)*(c + d)",
        "(x - 1)**3"
    ]
    
    for expr in expressions:
        result = engine.process_mathematical_query(f"expand {expr}")
        print(f"\nExpand: {expr}")
        print(f"Result: {result['result']}")
        if result.get('steps'):
            print("Steps:")
            for step in result['steps']:
                print(f"  • {step}")
    
    # Algebraic factoring
    print("\n\n3. Algebraic Factoring (Sanskrit Morphological-like)")
    print("-" * 50)
    
    expressions = [
        "x**2 + 3*x + 2",
        "x**2 - 4",
        "x**3 - 1",
        "6*x**2 + 11*x + 3"
    ]
    
    for expr in expressions:
        result = engine.process_mathematical_query(f"factor {expr}")
        print(f"\nFactor: {expr}")
        print(f"Result: {result['result']}")
        if result.get('steps'):
            print("Steps:")
            for step in result['steps']:
                print(f"  • {step}")


def demonstrate_equation_solving():
    """Demonstrate equation solving capabilities."""
    print("\n\n" + "=" * 60)
    print("EQUATION SOLVING DEMONSTRATION")
    print("=" * 60)
    
    engine = create_symbolic_computation_engine()
    
    equations = [
        "x + 5 = 12",
        "2*x - 3 = 7",
        "x**2 - 5*x + 6 = 0",
        "x**3 - 8 = 0",
        "sin(x) = 1/2"
    ]
    
    for equation in equations:
        result = engine.process_mathematical_query(f"solve {equation}")
        print(f"\nSolve: {equation}")
        
        if 'result' in result and 'solutions' in result['result']:
            solutions = result['result']['solutions']
            if solutions:
                print(f"Solutions: {', '.join(solutions)}")
            else:
                print("No solutions found")
        else:
            print(f"Result: {result.get('result', 'Error processing equation')}")


def demonstrate_proof_verification():
    """Demonstrate mathematical proof verification."""
    print("\n\n" + "=" * 60)
    print("PROOF VERIFICATION DEMONSTRATION")
    print("=" * 60)
    
    engine = create_symbolic_computation_engine()
    
    # Test various mathematical identities
    identities = [
        "(x + 1)**2 = x**2 + 2*x + 1",
        "(a + b)**2 = a**2 + 2*a*b + b**2",
        "sin(x)**2 + cos(x)**2 = 1",
        "x**2 - 1 = (x - 1)*(x + 1)",
        "x + 1 = x + 2",  # This should be false
        "log(a*b) = log(a) + log(b)"
    ]
    
    for identity in identities:
        result = engine.process_mathematical_query(f"prove {identity}")
        print(f"\nProve: {identity}")
        
        if 'result' in result and 'valid' in result['result']:
            is_valid = result['result']['valid']
            print(f"Valid: {'✓' if is_valid else '✗'}")
            
            if 'verification_steps' in result['result']:
                print("Verification steps:")
                for step in result['result']['verification_steps']:
                    print(f"  • {step}")
        else:
            print(f"Result: {result.get('result', 'Error processing proof')}")
    
    # Direct verification using convenience function
    print("\n\nDirect Identity Verification:")
    print("-" * 30)
    
    test_cases = [
        ("(x + 1)**2", "x**2 + 2*x + 1"),
        ("x**2 - 4", "(x - 2)*(x + 2)"),
        ("x + 1", "x + 2")  # Should be false
    ]
    
    for left, right in test_cases:
        is_valid = verify_mathematical_identity(left, right)
        print(f"{left} ≡ {right}: {'✓' if is_valid else '✗'}")


def demonstrate_integration_with_reasoning():
    """Demonstrate integration with Sanskrit reasoning core."""
    print("\n\n" + "=" * 60)
    print("INTEGRATION WITH SANSKRIT REASONING")
    print("=" * 60)
    
    engine = create_symbolic_computation_engine()
    
    # Show statistics
    print("\n1. System Statistics")
    print("-" * 30)
    
    # Process some queries first
    engine.process_mathematical_query("square 25 using vedic method")
    engine.process_mathematical_query("solve x**2 - 4 = 0")
    engine.process_mathematical_query("prove (x + 1)**2 = x**2 + 2*x + 1")
    engine.process_mathematical_query("simplify x + x + x")
    
    stats = engine.get_computation_statistics()
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    print(f"Available Vedic sūtras: {len(stats['available_sutras'])}")
    
    print("\nQuery distribution:")
    for approach, count in stats['approach_distribution'].items():
        print(f"  {approach}: {count}")
    
    print("\nAvailable Vedic Mathematics sūtras:")
    for sutra in stats['available_sutras']:
        print(f"  • {sutra}")
    
    # Show mathematical domains
    print("\n\n2. Mathematical Domains")
    print("-" * 30)
    
    domains = list(MathematicalDomain)
    for domain in domains:
        print(f"  • {domain.value}")


def main():
    """Main demonstration function."""
    if not DEMO_AVAILABLE:
        return
    
    print("Sanskrit Symbolic Computation Integration Demo")
    print("=" * 60)
    print("This demo showcases the integration of SymPy with Sanskrit")
    print("grammatical principles for mathematical computation.")
    
    try:
        demonstrate_vedic_mathematics()
        demonstrate_symbolic_algebra()
        demonstrate_equation_solving()
        demonstrate_proof_verification()
        demonstrate_integration_with_reasoning()
        
        print("\n\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe symbolic computation system demonstrates:")
        print("• Integration of Vedic Mathematics sūtras with modern computation")
        print("• Sanskrit-inspired algebraic transformations")
        print("• Mathematical proof verification using sūtra logic")
        print("• Seamless integration with the Sanskrit reasoning core")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()