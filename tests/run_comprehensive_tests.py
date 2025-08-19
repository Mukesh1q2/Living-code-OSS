#!/usr/bin/env python3
"""
Comprehensive test runner for Sanskrit Rewrite Engine.

This script runs all test suites including unit tests, integration tests,
performance tests, and generates coverage reports.
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any


def run_command(cmd: List[str], description: str, timeout: int = 300) -> Dict[str, Any]:
    """Run a command and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {duration:.2f} seconds")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
            
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration,
            'description': description
        }
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'duration': timeout,
            'description': description
        }
    except Exception as e:
        print(f"Error running command: {e}")
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': 0,
            'description': description
        }


def run_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run unit tests."""
    cmd = [sys.executable, '-m', 'pytest', 'tests/unit/', '-v']
    
    if verbose:
        cmd.extend(['--tb=short', '--show-capture=all'])
    else:
        cmd.extend(['--tb=line'])
        
    cmd.extend([
        '--durations=10',  # Show 10 slowest tests
        '--strict-markers',  # Strict marker checking
        '-x'  # Stop on first failure
    ])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run integration tests."""
    cmd = [sys.executable, '-m', 'pytest', 'tests/integration/', '-v']
    
    if verbose:
        cmd.extend(['--tb=short', '--show-capture=all'])
    else:
        cmd.extend(['--tb=line'])
        
    cmd.extend([
        '--durations=10',
        '--strict-markers',
        '-x'
    ])
    
    return run_command(cmd, "Integration Tests", timeout=600)  # Longer timeout


def run_performance_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run performance tests."""
    cmd = [sys.executable, '-m', 'pytest', 'tests/integration/test_performance.py', '-v']
    
    if verbose:
        cmd.extend(['--tb=short', '--show-capture=all'])
    else:
        cmd.extend(['--tb=line'])
        
    cmd.extend([
        '-m', 'performance',  # Only run performance marked tests
        '--durations=0',  # Show all test durations
        '--strict-markers'
    ])
    
    return run_command(cmd, "Performance Tests", timeout=900)  # Even longer timeout


def run_coverage_tests() -> Dict[str, Any]:
    """Run tests with coverage reporting."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--cov=src/sanskrit_rewrite_engine',
        '--cov-report=html:htmlcov',
        '--cov-report=term-missing',
        '--cov-report=xml:coverage.xml',
        '--cov-fail-under=80',  # Require 80% coverage
        '-v'
    ]
    
    return run_command(cmd, "Coverage Tests", timeout=900)


def run_linting() -> Dict[str, Any]:
    """Run code linting."""
    results = []
    
    # Run flake8
    cmd = [sys.executable, '-m', 'flake8', 'src/', 'tests/', '--max-line-length=100']
    results.append(run_command(cmd, "Flake8 Linting"))
    
    # Run black check
    cmd = [sys.executable, '-m', 'black', '--check', '--diff', 'src/', 'tests/']
    results.append(run_command(cmd, "Black Formatting Check"))
    
    # Run isort check
    cmd = [sys.executable, '-m', 'isort', '--check-only', '--diff', 'src/', 'tests/']
    results.append(run_command(cmd, "Import Sorting Check"))
    
    # Combine results
    all_success = all(r['success'] for r in results)
    total_duration = sum(r['duration'] for r in results)
    
    return {
        'success': all_success,
        'returncode': 0 if all_success else 1,
        'stdout': '\n'.join(r['stdout'] for r in results),
        'stderr': '\n'.join(r['stderr'] for r in results),
        'duration': total_duration,
        'description': 'Code Linting'
    }


def run_type_checking() -> Dict[str, Any]:
    """Run type checking with mypy."""
    cmd = [
        sys.executable, '-m', 'mypy',
        'src/sanskrit_rewrite_engine/',
        '--ignore-missing-imports',
        '--strict-optional',
        '--warn-redundant-casts',
        '--warn-unused-ignores'
    ]
    
    return run_command(cmd, "Type Checking (MyPy)")


def run_security_checks() -> Dict[str, Any]:
    """Run security checks."""
    cmd = [sys.executable, '-m', 'bandit', '-r', 'src/', '-f', 'json']
    
    return run_command(cmd, "Security Checks (Bandit)")


def generate_test_report(results: List[Dict[str, Any]]) -> None:
    """Generate a comprehensive test report."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    total_duration = sum(r['duration'] for r in results)
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"\nSUMMARY:")
    print(f"  Total test suites: {len(results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Failed: {len(failed_tests)}")
    print(f"  Total duration: {total_duration:.2f} seconds")
    
    if successful_tests:
        print(f"\n✅ SUCCESSFUL TEST SUITES:")
        for result in successful_tests:
            print(f"  • {result['description']} ({result['duration']:.2f}s)")
    
    if failed_tests:
        print(f"\n❌ FAILED TEST SUITES:")
        for result in failed_tests:
            print(f"  • {result['description']} ({result['duration']:.2f}s)")
            if result['stderr']:
                print(f"    Error: {result['stderr'][:200]}...")
    
    # Performance summary
    performance_results = [r for r in results if 'Performance' in r['description']]
    if performance_results:
        print(f"\n📊 PERFORMANCE SUMMARY:")
        for result in performance_results:
            print(f"  • {result['description']}: {result['duration']:.2f}s")
    
    # Coverage summary
    coverage_results = [r for r in results if 'Coverage' in r['description']]
    if coverage_results:
        print(f"\n📈 COVERAGE SUMMARY:")
        for result in coverage_results:
            if 'coverage' in result['stdout'].lower():
                # Extract coverage percentage from output
                lines = result['stdout'].split('\n')
                for line in lines:
                    if 'TOTAL' in line and '%' in line:
                        print(f"  • {line.strip()}")
    
    print(f"\n{'='*80}")
    
    # Overall result
    if failed_tests:
        print("❌ OVERALL RESULT: FAILED")
        sys.exit(1)
    else:
        print("✅ OVERALL RESULT: PASSED")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner for Sanskrit Rewrite Engine")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--coverage', action='store_true', help='Run coverage tests only')
    parser.add_argument('--lint', action='store_true', help='Run linting only')
    parser.add_argument('--type-check', action='store_true', help='Run type checking only')
    parser.add_argument('--security', action='store_true', help='Run security checks only')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (skip performance)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage reporting')
    parser.add_argument('--no-lint', action='store_true', help='Skip linting')
    
    args = parser.parse_args()
    
    print("Sanskrit Rewrite Engine - Comprehensive Test Suite")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    results = []
    
    # Determine which tests to run
    run_all = not any([
        args.unit, args.integration, args.performance, args.coverage,
        args.lint, args.type_check, args.security
    ])
    
    # Run selected test suites
    if args.unit or run_all:
        results.append(run_unit_tests(args.verbose))
    
    if args.integration or run_all:
        results.append(run_integration_tests(args.verbose))
    
    if (args.performance or run_all) and not args.fast:
        results.append(run_performance_tests(args.verbose))
    
    if args.coverage or (run_all and not args.no_coverage):
        results.append(run_coverage_tests())
    
    if args.lint or (run_all and not args.no_lint):
        results.append(run_linting())
    
    if args.type_check or run_all:
        results.append(run_type_checking())
    
    if args.security or run_all:
        results.append(run_security_checks())
    
    # Generate comprehensive report
    generate_test_report(results)


if __name__ == '__main__':
    main()