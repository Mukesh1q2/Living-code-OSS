#!/usr/bin/env python3
"""
Performance Optimization Demo for Sanskrit Rewrite Engine.

This demo showcases the comprehensive performance optimization features
including parallel processing, caching, profiling, and benchmarking.
"""

import time
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine
from sanskrit_rewrite_engine.performance_optimization import (
    PerformanceOptimizer, PerformanceConfig,
    get_high_performance_config, get_memory_optimized_config
)
from sanskrit_rewrite_engine.performance_benchmarks import (
    PerformanceBenchmark, run_full_benchmark_suite
)
from sanskrit_rewrite_engine.performance_profiler import (
    AdvancedProfiler, PerformanceRegression
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_optimization():
    """Demonstrate basic performance optimization features."""
    print("=" * 60)
    print("BASIC PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Create sample Sanskrit text
    sanskrit_text = """
    अथ योगानुशासनम् ॥१॥
    योगश्चित्तवृत्तिनिरोधः ॥२॥
    तदा द्रष्टुः स्वरूपेऽवस्थानम् ॥३॥
    वृत्तिसारूप्यमितरत्र ॥४॥
    वृत्तयः पञ्चतय्यः क्लिष्टाक्लिष्टाः ॥५॥
    """
    
    # Tokenize text
    tokenizer = SanskritTokenizer()
    tokens = tokenizer.tokenize(sanskrit_text)
    
    print(f"Processing {len(tokens)} tokens...")
    
    # Test different configurations
    configurations = {
        'default': PerformanceConfig(),
        'high_performance': get_high_performance_config(),
        'memory_optimized': get_memory_optimized_config()
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\nTesting {config_name} configuration...")
        
        # Create optimizer
        optimizer = PerformanceOptimizer(config)
        
        # Create engine
        engine = PaniniRuleEngine(tokenizer)
        
        # Measure processing time
        start_time = time.time()
        
        # Process tokens multiple times to test caching
        for i in range(3):
            result = engine.process(tokens)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get performance report
        report = optimizer.get_performance_report()
        
        results[config_name] = {
            'processing_time': processing_time,
            'cache_hit_rate': report['cache_statistics']['hit_rate'],
            'memory_usage_mb': report['memory_report']['memory_stats']['used_memory_gb'] * 1024,
            'suggestions': report['optimization_suggestions']
        }
        
        print(f"  Processing time: {processing_time:.4f}s")
        print(f"  Cache hit rate: {report['cache_statistics']['hit_rate']:.2%}")
        print(f"  Memory usage: {report['memory_report']['memory_stats']['used_memory_gb'] * 1024:.1f}MB")
        
        if report['optimization_suggestions']:
            print("  Optimization suggestions:")
            for suggestion in report['optimization_suggestions'][:3]:
                print(f"    - {suggestion}")
    
    # Compare configurations
    print("\n" + "=" * 40)
    print("CONFIGURATION COMPARISON")
    print("=" * 40)
    
    fastest_config = min(results.items(), key=lambda x: x[1]['processing_time'])
    best_cache = max(results.items(), key=lambda x: x[1]['cache_hit_rate'])
    lowest_memory = min(results.items(), key=lambda x: x[1]['memory_usage_mb'])
    
    print(f"Fastest configuration: {fastest_config[0]} ({fastest_config[1]['processing_time']:.4f}s)")
    print(f"Best cache performance: {best_cache[0]} ({best_cache[1]['cache_hit_rate']:.2%})")
    print(f"Lowest memory usage: {lowest_memory[0]} ({lowest_memory[1]['memory_usage_mb']:.1f}MB)")


def demo_advanced_profiling():
    """Demonstrate advanced profiling capabilities."""
    print("\n" + "=" * 60)
    print("ADVANCED PROFILING DEMO")
    print("=" * 60)
    
    # Create profiler
    profiler = AdvancedProfiler(
        enable_line_profiling=False,  # Disable for demo
        enable_memory_profiling=True
    )
    
    # Define functions to profile
    @profiler.profile_function("tokenization", include_line_profile=False)
    def profile_tokenization(text):
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    @profiler.profile_function("rule_processing", include_line_profile=False)
    def profile_rule_processing(tokens):
        engine = PaniniRuleEngine()
        return engine.process(tokens)
    
    @profiler.profile_function("expensive_operation", include_line_profile=False)
    def expensive_operation(n):
        # Simulate expensive computation
        result = 0
        for i in range(n):
            result += i ** 2
        time.sleep(0.01)  # Simulate I/O
        return result
    
    # Sample texts of different sizes
    texts = {
        'small': "राम गच्छति वनम्",
        'medium': "अथ योगानुशासनम् । योगश्चित्तवृत्तिनिरोधः । तदा द्रष्टुः स्वरूपेऽवस्थानम् ।",
        'large': """
        श्रीमद्भगवद्गीता अध्याय १
        धृतराष्ट्र उवाच ।
        धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।
        मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥१॥
        सञ्जय उवाच ।
        दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा ।
        आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् ॥२॥
        """
    }
    
    print("Running profiled operations...")
    
    # Profile operations
    for size, text in texts.items():
        print(f"\nProcessing {size} text...")
        
        # Tokenization
        tokens = profile_tokenization(text)
        print(f"  Tokenized into {len(tokens)} tokens")
        
        # Rule processing
        result = profile_rule_processing(tokens)
        print(f"  Applied rules in {result.passes} passes")
        
        # Expensive operation (for demonstration)
        expensive_result = expensive_operation(len(tokens) * 100)
        print(f"  Expensive operation result: {expensive_result}")
    
    # Create profiling session
    session = profiler.create_session("demo_session")
    
    print(f"\nProfiling Results:")
    print(f"Session ID: {session.session_id}")
    print(f"Total profiles: {len(session.profiles)}")
    
    # Display profile data
    for name, profile in session.profiles.items():
        print(f"\n{name}:")
        print(f"  Total time: {profile.total_time:.4f}s")
        print(f"  Call count: {profile.call_count}")
        print(f"  Average time: {profile.average_time:.4f}s")
        print(f"  Memory usage: {profile.memory_usage_mb:.2f}MB")
        print(f"  Bottleneck score: {profile.bottleneck_score:.2f}")
    
    # Show bottlenecks
    if session.bottlenecks:
        print(f"\nBottlenecks identified:")
        for bottleneck in session.bottlenecks:
            print(f"  - {bottleneck}")
    
    # Show recommendations
    if session.recommendations:
        print(f"\nOptimization recommendations:")
        for recommendation in session.recommendations:
            print(f"  - {recommendation}")
    
    # Save session
    output_file = "demo_profiling_session.json"
    profiler.save_session(session, output_file)
    print(f"\nProfiling session saved to: {output_file}")
    
    # Cleanup
    profiler.cleanup()


def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING DEMO")
    print("=" * 60)
    
    # Create benchmark
    benchmark_dir = "./demo_benchmark_results"
    benchmark = PerformanceBenchmark(benchmark_dir)
    
    print(f"Benchmark results will be saved to: {benchmark_dir}")
    
    # Generate test data
    print("\nGenerating test data...")
    test_cases = {
        'small_text': benchmark._generate_small_text(),
        'medium_text': benchmark._generate_medium_text(),
        'complex_sandhi': benchmark._generate_complex_sandhi(),
        'repetitive_patterns': benchmark._generate_repetitive_patterns()
    }
    
    for name, tokens in test_cases.items():
        print(f"  {name}: {len(tokens)} tokens")
    
    # Run configuration comparison
    print("\nRunning configuration comparison benchmark...")
    comparison_tokens = test_cases['medium_text']
    comparison_suite = benchmark.run_configuration_comparison(comparison_tokens)
    
    # Save results
    comparison_file = benchmark.save_benchmark_results(
        comparison_suite, "demo_configuration_comparison.json"
    )
    
    # Generate report
    report = benchmark.generate_performance_report(comparison_suite)
    
    print("\nBenchmark Report:")
    print("-" * 40)
    print(report)
    
    # Save report
    report_file = Path(benchmark_dir) / "demo_benchmark_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Run cache effectiveness test
    print("\nRunning cache effectiveness benchmark...")
    cache_tokens = test_cases['repetitive_patterns']
    cache_suite = benchmark.run_cache_effectiveness_benchmark(cache_tokens)
    
    # Analyze cache performance
    with_cache_results = cache_suite.get_results_by_config("with_cache")
    without_cache_results = cache_suite.get_results_by_config("without_cache")
    
    if with_cache_results and without_cache_results:
        avg_with_cache = sum(r.processing_time for r in with_cache_results) / len(with_cache_results)
        avg_without_cache = sum(r.processing_time for r in without_cache_results) / len(without_cache_results)
        
        speedup = avg_without_cache / avg_with_cache if avg_with_cache > 0 else 1.0
        
        print(f"\nCache Effectiveness Results:")
        print(f"  Average time with cache: {avg_with_cache:.4f}s")
        print(f"  Average time without cache: {avg_without_cache:.4f}s")
        print(f"  Speedup from caching: {speedup:.2f}x")
        
        # Show cache hit rate progression
        cache_hit_rates = [r.cache_hit_rate for r in with_cache_results]
        print(f"  Cache hit rate progression: {cache_hit_rates}")


def demo_regression_detection():
    """Demonstrate performance regression detection."""
    print("\n" + "=" * 60)
    print("PERFORMANCE REGRESSION DETECTION DEMO")
    print("=" * 60)
    
    # Create regression detector
    baseline_file = "demo_baseline.json"
    regression = PerformanceRegression(baseline_file)
    
    # Create profiler for baseline
    profiler = AdvancedProfiler(enable_memory_profiling=True)
    
    @profiler.profile_function("baseline_function")
    def baseline_function(n):
        # Simulate baseline performance
        time.sleep(0.01)  # 10ms baseline
        return sum(range(n))
    
    # Run baseline
    print("Establishing baseline performance...")
    for i in range(5):
        baseline_function(100)
    
    baseline_session = profiler.create_session("baseline")
    regression.save_baseline(baseline_session)
    print(f"Baseline saved to: {baseline_file}")
    
    # Simulate performance regression
    profiler.cleanup()
    profiler = AdvancedProfiler(enable_memory_profiling=True)
    
    @profiler.profile_function("baseline_function")  # Same name for comparison
    def regressed_function(n):
        # Simulate performance regression
        time.sleep(0.02)  # 20ms (100% slower)
        # Also use more memory
        dummy_data = [i for i in range(n * 10)]  # More memory usage
        return sum(range(n))
    
    # Run regressed version
    print("\nRunning potentially regressed version...")
    for i in range(5):
        regressed_function(100)
    
    current_session = profiler.create_session("current")
    
    # Detect regressions
    regressions = regression.detect_regressions(current_session, threshold=0.1)
    
    if regressions:
        print(f"\nPerformance regressions detected:")
        for regression_msg in regressions:
            print(f"  - {regression_msg}")
    else:
        print(f"\nNo significant performance regressions detected.")
    
    # Cleanup
    profiler.cleanup()
    
    # Clean up demo files
    for file_path in [baseline_file]:
        if Path(file_path).exists():
            Path(file_path).unlink()


def demo_optimization_recommendations():
    """Demonstrate automatic optimization recommendations."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS DEMO")
    print("=" * 60)
    
    # Create different scenarios to trigger recommendations
    scenarios = [
        {
            'name': 'Low Cache Hit Rate',
            'config': PerformanceConfig(
                enable_rule_caching=True,
                cache_size_mb=1  # Very small cache
            )
        },
        {
            'name': 'High Memory Usage',
            'config': PerformanceConfig(
                enable_rule_caching=True,
                cache_size_mb=2048,  # Large cache
                enable_compression=False
            )
        },
        {
            'name': 'No Parallel Processing',
            'config': PerformanceConfig(
                enable_parallel_processing=False,
                enable_rule_caching=True
            )
        }
    ]
    
    # Sample text for testing
    tokenizer = SanskritTokenizer()
    tokens = tokenizer.tokenize("राम गच्छति वनम् । सीता पठति पुस्तकम् ।" * 50)  # Repeat for larger dataset
    
    print(f"Testing with {len(tokens)} tokens...")
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)
        
        # Create optimizer with scenario config
        optimizer = PerformanceOptimizer(scenario['config'])
        
        # Simulate processing
        engine = PaniniRuleEngine(tokenizer)
        
        # Process multiple times to generate metrics
        for i in range(5):
            result = engine.process(tokens)
        
        # Get recommendations
        suggestions = optimizer.generate_optimization_suggestions()
        
        if suggestions:
            print("Recommendations:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
        else:
            print("No specific recommendations for this scenario.")
        
        # Show key metrics
        report = optimizer.get_performance_report()
        cache_stats = report['cache_statistics']
        memory_stats = report['memory_report']['memory_stats']
        
        print(f"Key metrics:")
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Memory utilization: {memory_stats.get('utilization_percent', 0):.1f}%")
        print(f"  Parallel processing: {'Enabled' if scenario['config'].enable_parallel_processing else 'Disabled'}")


def main():
    """Run all performance optimization demos."""
    print("Sanskrit Rewrite Engine - Performance Optimization Demo")
    print("=" * 60)
    
    try:
        # Run individual demos
        demo_basic_optimization()
        demo_advanced_profiling()
        demo_performance_benchmarking()
        demo_regression_detection()
        demo_optimization_recommendations()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - demo_profiling_session.json")
        print("  - ./demo_benchmark_results/ (directory)")
        print("  - Various benchmark and report files")
        
        print("\nNext steps:")
        print("  1. Review the generated benchmark reports")
        print("  2. Experiment with different configurations")
        print("  3. Run full benchmark suite with: run_full_benchmark_suite()")
        print("  4. Integrate profiling into your development workflow")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())