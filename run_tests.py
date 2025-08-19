#!/usr/bin/env python3
"""
Test runner for Sanskrit Rewrite Engine.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests using pytest."""
    try:
        # Try to run with pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("pytest not found, trying to run tests manually...")
        return run_tests_manually()

def run_tests_manually():
    """Run tests manually without pytest."""
    import importlib.util
    import traceback
    
    test_files = [
        "tests/test_token.py",
        "tests/test_transliterator.py", 
        "tests/test_tokenizer.py"
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"Test file {test_file} not found")
            continue
            
        print(f"\nRunning {test_file}...")
        
        try:
            # Load the test module
            spec = importlib.util.spec_from_file_location("test_module", test_file)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Find test classes
            for name in dir(test_module):
                obj = getattr(test_module, name)
                if isinstance(obj, type) and name.startswith('Test'):
                    test_class = obj()
                    
                    # Run setup if it exists
                    if hasattr(test_class, 'setup_method'):
                        test_class.setup_method()
                    
                    # Find and run test methods
                    for method_name in dir(test_class):
                        if method_name.startswith('test_'):
                            total_tests += 1
                            try:
                                method = getattr(test_class, method_name)
                                method()
                                print(f"  ✓ {method_name}")
                                passed_tests += 1
                            except Exception as e:
                                print(f"  ✗ {method_name}: {e}")
                                traceback.print_exc()
                                
        except Exception as e:
            print(f"Error loading {test_file}: {e}")
            traceback.print_exc()
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)