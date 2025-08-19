#!/usr/bin/env python3
"""
Test runner script for Sanskrit Rewrite Engine.
Demonstrates proper test discovery and organization.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Run various test scenarios to verify organization."""
    print("Sanskrit Rewrite Engine Test Organization Verification")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Discover all tests
    total_tests += 1
    if run_command([
        sys.executable, "-m", "pytest", 
        "--collect-only", "-q"
    ], "Test Discovery - All Tests"):
        success_count += 1
    
    # Test 2: Run unit tests only
    total_tests += 1
    if run_command([
        sys.executable, "-m", "pytest", 
        "tests/unit/", "-v", "--tb=short"
    ], "Unit Tests Only"):
        success_count += 1
    
    # Test 3: Run integration tests only
    total_tests += 1
    if run_command([
        sys.executable, "-m", "pytest", 
        "tests/integration/", "-v", "--tb=short"
    ], "Integration Tests Only"):
        success_count += 1
    
    # Test 4: Test marker filtering
    total_tests += 1
    if run_command([
        sys.executable, "-m", "pytest", 
        "-m", "unit", "--collect-only", "-q"
    ], "Unit Marker Filtering"):
        success_count += 1
    
    # Test 5: Test marker filtering for integration
    total_tests += 1
    if run_command([
        sys.executable, "-m", "pytest", 
        "-m", "integration", "--collect-only", "-q"
    ], "Integration Marker Filtering"):
        success_count += 1
    
    # Test 6: Verify fixtures work
    total_tests += 1
    if run_command([
        sys.executable, "-c", 
        "import pytest; from tests.conftest import sample_texts; print('Fixtures imported successfully')"
    ], "Fixture Import Test"):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST ORGANIZATION VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_tests}")
    print(f"Success rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("✅ All test organization checks passed!")
        return 0
    else:
        print("❌ Some test organization checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())