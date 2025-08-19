"""
Comprehensive test runner for Sanskrit Rewrite Engine.
Executes all tests from TO1 comprehensive test suite.

Usage:
    python tests/run_comprehensive_tests.py [options]

Options:
    --fast          Run only fast tests (skip stress tests)
    --performance   Run only performance tests
    --regression    Run only regression tests
    --verbose       Verbose output
    --report        Generate detailed test report
"""

import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any


class ComprehensiveTestRunner:
    """Runner for comprehensive test suite."""
    
    def __init__(self):
        self.test_modules = {
            'tokenization': 'tests/test_comprehensive_tokenization.py',
            'rules': 'tests/test_comprehensive_rules.py', 
            'integration': 'tests/test_comprehensive_integration.py',
            'performance': 'tests/test_performance_profiling.py',
            'regression': 'tests/test_regression_suite.py'
        }
        
        self.test_categories = {
            'fast': ['tokenization', 'rules'],
            'performance': ['performance'],
            'regression': ['regression'],
            'integration': ['integration'],
            'all': list(self.test_modules.keys())
        }
    
    def run_test_module(self, module_name: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a single test module and return results."""
        module_path = self.test_modules[module_name]
        
        print(f"\n{'='*60}")
        print(f"Running {module_name} tests: {module_path}")
        print(f"{'='*60}")
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', module_path]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.append('-q')
        
        # Add additional options
        cmd.extend([
            '--tb=short',           # Short traceback format
            '--durations=10',       # Show 10 slowest tests
            '--junit-xml=test_results.xml',  # Generate XML report
            '--cov=sanskrit_rewrite_engine',  # Coverage if available
            '--cov-report=term-missing'
        ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'module': module_name,
                'success': result.returncode == 0,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'module': module_name,
                'success': False,
                'duration': 300,
                'stdout': '',
                'stderr': 'Test module timed out after 5 minutes',
                'returncode': -1
            }
        except Exception as e:
            return {
                'module': module_name,
                'success': False,
                'duration': 0,
                'stdout': '',
                'stderr': f'Error running tests: {str(e)}',
                'returncode': -2
            }
    
    def run_test_category(self, category: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """Run all tests in a category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        modules = self.test_categories[category]
        results = []
        
        print(f"\nRunning {category} test category ({len(modules)} modules)")
        print(f"Modules: {', '.join(modules)}")
        
        for module in modules:
            if module in self.test_modules:
                result = self.run_test_module(module, verbose)
                results.append(result)
                
                # Print immediate feedback
                status = "✓ PASSED" if result['success'] else "✗ FAILED"
                print(f"{status} {module} ({result['duration']:.1f}s)")
                
                if not result['success'] and result['stderr']:
                    print(f"  Error: {result['stderr'][:200]}...")
            else:
                print(f"Warning: Module {module} not found")
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - passed_tests
        total_duration = sum(r['duration'] for r in results)
        
        report = {
            'summary': {
                'total_modules': total_tests,
                'passed_modules': passed_tests,
                'failed_modules': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'average_duration': total_duration / total_tests if total_tests > 0 else 0
            },
            'modules': results,
            'failed_modules': [r for r in results if not r['success']],
            'slowest_modules': sorted(results, key=lambda x: x['duration'], reverse=True)[:5]
        }
        
        return report
    
    def print_summary_report(self, report: Dict[str, Any]):
        """Print summary report to console."""
        summary = report['summary']
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total modules:     {summary['total_modules']}")
        print(f"Passed modules:    {summary['passed_modules']}")
        print(f"Failed modules:    {summary['failed_modules']}")
        print(f"Success rate:      {summary['success_rate']:.1%}")
        print(f"Total duration:    {summary['total_duration']:.1f}s")
        print(f"Average duration:  {summary['average_duration']:.1f}s")
        
        if report['failed_modules']:
            print(f"\nFAILED MODULES:")
            for result in report['failed_modules']:
                print(f"  ✗ {result['module']} ({result['duration']:.1f}s)")
                if result['stderr']:
                    print(f"    Error: {result['stderr'][:100]}...")
        
        if report['slowest_modules']:
            print(f"\nSLOWEST MODULES:")
            for result in report['slowest_modules'][:3]:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {result['module']} ({result['duration']:.1f}s)")
        
        # Overall result
        if summary['failed_modules'] == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n❌ {summary['failed_modules']} MODULE(S) FAILED")
    
    def save_detailed_report(self, report: Dict[str, Any], filename: str = "test_report.json"):
        """Save detailed report to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to: {filename}")
        except Exception as e:
            print(f"Warning: Could not save report to {filename}: {e}")
    
    def check_requirements(self) -> bool:
        """Check if all required dependencies are available."""
        required_packages = ['pytest', 'psutil']
        optional_packages = ['pytest-cov', 'pytest-xdist']
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_required.append(package)
        
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_optional.append(package)
        
        if missing_required:
            print(f"Error: Missing required packages: {', '.join(missing_required)}")
            print("Install with: pip install " + " ".join(missing_required))
            return False
        
        if missing_optional:
            print(f"Warning: Missing optional packages: {', '.join(missing_optional)}")
            print("Install with: pip install " + " ".join(missing_optional))
        
        return True


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive test suite for Sanskrit Rewrite Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_comprehensive_tests.py --fast
    python tests/run_comprehensive_tests.py --performance --verbose
    python tests/run_comprehensive_tests.py --regression --report
    python tests/run_comprehensive_tests.py  # Run all tests
        """
    )
    
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (skip stress tests)')
    parser.add_argument('--performance', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--regression', action='store_true',
                       help='Run only regression tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed test report')
    parser.add_argument('--output', '-o', default='test_report.json',
                       help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner()
    
    # Check requirements
    if not runner.check_requirements():
        return 1
    
    # Determine which tests to run
    if args.fast:
        category = 'fast'
    elif args.performance:
        category = 'performance'
    elif args.regression:
        category = 'regression'
    elif args.integration:
        category = 'integration'
    else:
        category = 'all'
    
    print(f"Sanskrit Rewrite Engine - Comprehensive Test Suite")
    print(f"Running {category} tests...")
    
    # Run tests
    start_time = time.time()
    results = runner.run_test_category(category, args.verbose)
    end_time = time.time()
    
    # Generate report
    report = runner.generate_report(results)
    report['summary']['total_suite_duration'] = end_time - start_time
    
    # Print summary
    runner.print_summary_report(report)
    
    # Save detailed report if requested
    if args.report:
        runner.save_detailed_report(report, args.output)
    
    # Return appropriate exit code
    failed_count = report['summary']['failed_modules']
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())