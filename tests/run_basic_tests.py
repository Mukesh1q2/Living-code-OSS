#!/usr/bin/env python3
"""
Basic test runner for Sanskrit Rewrite Engine.

This script runs the core tests that are currently working with the existing implementation.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_test_suite(test_path, description):
    """Run a test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {duration:.2f} seconds")
        
        if result.stdout:
            print(f"\nOutput:\n{result.stdout}")
        
        if result.stderr and result.returncode != 0:
            print(f"\nErrors:\n{result.stderr}")
            
        return result.returncode == 0, duration
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, 0


def main():
    """Run basic test suite."""
    print("Sanskrit Rewrite Engine - Basic Test Suite")
    print(f"Python version: {sys.version}")
    
    # Test suites that should work
    test_suites = [
        ("tests/unit/test_engine.py", "Engine Unit Tests"),
        ("tests/unit/test_rules.py", "Rules Unit Tests"),
        ("tests/unit/test_tokenizer.py::TestSanskritTokenizer::test_basic_tokenization", "Basic Tokenization Test"),
        ("tests/unit/test_tokenizer.py::TestSanskritTokenizer::test_empty_and_whitespace_handling", "Tokenizer Edge Cases"),
    ]
    
    results = []
    total_time = 0
    
    for test_path, description in test_suites:
        success, duration = run_test_suite(test_path, description)
        results.append((description, success, duration))
        total_time += duration
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Total test suites: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Total time: {total_time:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for description, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {description} ({duration:.2f}s)")
    
    if passed == total:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())