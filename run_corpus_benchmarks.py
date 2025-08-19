#!/usr/bin/env python3
"""
Comprehensive corpus benchmarking runner for the Sanskrit Rewrite Engine.

This script runs all benchmarking suites and generates comprehensive reports:
- Sanskrit corpus evaluation with accuracy metrics
- Mathematical and programming benchmarks
- Performance and scalability analysis
- Rule coverage analysis
- Semantic consistency validation
- Optimization recommendations
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

from sanskrit_rewrite_engine.corpus_benchmarking import (
    create_corpus_benchmarker, benchmark_all_test_corpora
)
from sanskrit_rewrite_engine.math_programming_benchmarks import (
    run_math_programming_benchmarks, MathProgrammingBenchmarker
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'benchmark_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def run_corpus_benchmarks(corpus_paths: list = None, output_dir: str = "benchmark_results") -> dict:
    """
    Run comprehensive corpus benchmarks.
    
    Args:
        corpus_paths: List of corpus file paths (uses all test corpora if None)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting corpus benchmarking...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create benchmarker
    benchmarker = create_corpus_benchmarker()
    
    results = {}
    
    if corpus_paths:
        # Benchmark specific corpora
        logger.info(f"Benchmarking {len(corpus_paths)} specified corpora...")
        benchmark_results = benchmarker.benchmark_multiple_corpora(corpus_paths)
    else:
        # Benchmark all test corpora
        logger.info("Benchmarking all available test corpora...")
        benchmark_results = benchmark_all_test_corpora()
    
    if not benchmark_results:
        logger.warning("No corpus benchmark results obtained")
        return {}
    
    # Store results
    benchmarker.benchmark_results = benchmark_results
    results['corpus_benchmarks'] = benchmark_results
    
    # Generate comprehensive report
    logger.info("Generating corpus benchmark report...")
    report = benchmarker.generate_comprehensive_report()
    
    # Save report
    report_file = output_path / f"corpus_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Corpus benchmark report saved to: {report_file}")
    
    # Save detailed results
    results_file = benchmarker.save_results_to_json(str(output_path / "corpus_benchmark_results.json"))
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CORPUS BENCHMARK SUMMARY")
    print("="*60)
    print(report)
    
    return results


def run_math_programming_benchmarks_suite(output_dir: str = "benchmark_results") -> dict:
    """
    Run mathematical and programming benchmarks.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting mathematical and programming benchmarks...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run benchmarks
    result = run_math_programming_benchmarks()
    
    # Generate report
    benchmarker = MathProgrammingBenchmarker()
    report = benchmarker.generate_report(result)
    
    # Save report
    report_file = output_path / f"math_programming_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Math/programming report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("MATHEMATICAL & PROGRAMMING BENCHMARK SUMMARY")
    print("="*60)
    print(report)
    
    return {'math_programming_benchmarks': result}


def run_scalability_analysis(base_corpus: str = None, output_dir: str = "benchmark_results") -> dict:
    """
    Run scalability analysis.
    
    Args:
        base_corpus: Path to base corpus for scalability testing
        output_dir: Directory to save results
        
    Returns:
        Dictionary with scalability results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting scalability analysis...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find base corpus if not specified
    if not base_corpus:
        corpus_dir = Path("test_corpus")
        if corpus_dir.exists():
            corpus_files = list(corpus_dir.glob("*.json"))
            if corpus_files:
                base_corpus = str(corpus_files[0])  # Use first available corpus
                logger.info(f"Using {base_corpus} for scalability analysis")
    
    if not base_corpus or not Path(base_corpus).exists():
        logger.warning("No suitable corpus found for scalability analysis")
        return {}
    
    # Create benchmarker and run scalability test
    benchmarker = create_corpus_benchmarker()
    
    # Use smaller scale factors for reasonable test time
    scale_factors = [1, 2, 3, 5]
    scalability_results = benchmarker.benchmark_scalability(base_corpus, scale_factors)
    
    # Generate scalability report
    report_lines = []
    report_lines.append("SCALABILITY ANALYSIS REPORT")
    report_lines.append("=" * 30)
    report_lines.append(f"Base Corpus: {base_corpus}")
    report_lines.append(f"Scale Factors: {scale_factors}")
    report_lines.append("")
    
    # Processing time analysis
    if scalability_results.get('processing_times'):
        report_lines.append("Processing Time Analysis:")
        for i, (factor, time_val) in enumerate(zip(scale_factors, scalability_results['processing_times'])):
            report_lines.append(f"  Scale {factor}x: {time_val:.3f}s")
        report_lines.append("")
    
    # Throughput analysis
    if scalability_results.get('throughput'):
        report_lines.append("Throughput Analysis:")
        for i, (factor, throughput) in enumerate(zip(scale_factors, scalability_results['throughput'])):
            report_lines.append(f"  Scale {factor}x: {throughput:.1f} tokens/s")
        report_lines.append("")
    
    # Memory usage analysis
    if scalability_results.get('memory_usage'):
        report_lines.append("Memory Usage Analysis:")
        for i, (factor, memory) in enumerate(zip(scale_factors, scalability_results['memory_usage'])):
            report_lines.append(f"  Scale {factor}x: {memory:.2f} MB")
        report_lines.append("")
    
    # Accuracy degradation analysis
    if scalability_results.get('accuracy_degradation'):
        report_lines.append("Accuracy Degradation Analysis:")
        for i, (factor, degradation) in enumerate(zip(scale_factors, scalability_results['accuracy_degradation'])):
            report_lines.append(f"  Scale {factor}x: {degradation:+.2f}%")
        report_lines.append("")
    
    scalability_report = "\n".join(report_lines)
    
    # Save report
    report_file = output_path / f"scalability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(scalability_report)
    
    logger.info(f"Scalability report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS SUMMARY")
    print("="*60)
    print(scalability_report)
    
    return {'scalability_analysis': scalability_results}


def generate_optimization_recommendations(all_results: dict, output_dir: str = "benchmark_results") -> None:
    """
    Generate comprehensive optimization recommendations.
    
    Args:
        all_results: Combined results from all benchmarks
        output_dir: Directory to save recommendations
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating optimization recommendations...")
    
    output_path = Path(output_dir)
    recommendations = []
    
    # Analyze corpus benchmark results
    if 'corpus_benchmarks' in all_results:
        corpus_results = all_results['corpus_benchmarks']
        if corpus_results:
            avg_accuracy = sum(r.accuracy_metrics.accuracy for r in corpus_results) / len(corpus_results)
            avg_performance = sum(r.performance_metrics.mean_processing_time for r in corpus_results) / len(corpus_results)
            
            if avg_accuracy < 0.8:
                recommendations.append(
                    f"CRITICAL: Low average corpus accuracy ({avg_accuracy:.2%}). "
                    f"Review rule implementations and add more comprehensive test cases."
                )
            
            if avg_performance > 0.5:
                recommendations.append(
                    f"PERFORMANCE: High average processing time ({avg_performance:.3f}s). "
                    f"Consider optimizing rule matching algorithms and adding indexing."
                )
    
    # Analyze math/programming results
    if 'math_programming_benchmarks' in all_results:
        math_prog_result = all_results['math_programming_benchmarks']
        
        if math_prog_result.math_accuracy.accuracy < 0.7:
            recommendations.append(
                f"MATHEMATICAL: Low math translation accuracy ({math_prog_result.math_accuracy.accuracy:.2%}). "
                f"Expand symbolic computation engine and Vedic mathematics implementations."
            )
        
        if math_prog_result.programming_accuracy.accuracy < 0.6:
            recommendations.append(
                f"PROGRAMMING: Low code generation accuracy ({math_prog_result.programming_accuracy.accuracy:.2%}). "
                f"Enhance domain mapping with more programming patterns."
            )
    
    # Analyze scalability results
    if 'scalability_analysis' in all_results:
        scalability = all_results['scalability_analysis']
        
        if scalability.get('processing_times'):
            times = scalability['processing_times']
            if len(times) >= 2:
                # Check if processing time grows super-linearly
                time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
                scale_ratio = scalability['scale_factors'][-1] / scalability['scale_factors'][0]
                
                if time_ratio > scale_ratio * 1.5:  # More than 50% overhead
                    recommendations.append(
                        f"SCALABILITY: Processing time grows super-linearly (ratio: {time_ratio:.2f}). "
                        f"Implement parallel processing or optimize algorithms for large inputs."
                    )
    
    # Generate final recommendations report
    report_lines = []
    report_lines.append("OPTIMIZATION RECOMMENDATIONS REPORT")
    report_lines.append("=" * 40)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    if recommendations:
        report_lines.append("PRIORITY RECOMMENDATIONS:")
        report_lines.append("-" * 25)
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")
    else:
        report_lines.append("No critical issues identified. System performance is within acceptable ranges.")
        report_lines.append("")
    
    # Add general recommendations
    report_lines.append("GENERAL RECOMMENDATIONS:")
    report_lines.append("-" * 22)
    report_lines.append("1. Regularly run benchmarks to monitor performance trends")
    report_lines.append("2. Expand test corpora to cover more linguistic phenomena")
    report_lines.append("3. Implement continuous integration for benchmark regression testing")
    report_lines.append("4. Consider adding more Vedic mathematics sutras for mathematical accuracy")
    report_lines.append("5. Enhance cross-domain mapping capabilities for better semantic preservation")
    
    recommendations_report = "\n".join(report_lines)
    
    # Save recommendations
    rec_file = output_path / f"optimization_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(rec_file, 'w', encoding='utf-8') as f:
        f.write(recommendations_report)
    
    logger.info(f"Optimization recommendations saved to: {rec_file}")
    
    # Print recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    print(recommendations_report)


def main():
    """Main benchmarking runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive corpus benchmarks for Sanskrit Rewrite Engine"
    )
    
    parser.add_argument(
        '--corpus-paths', 
        nargs='*', 
        help='Specific corpus files to benchmark (default: all test corpora)'
    )
    parser.add_argument(
        '--output-dir', 
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    parser.add_argument(
        '--skip-corpus', 
        action='store_true',
        help='Skip corpus benchmarking'
    )
    parser.add_argument(
        '--skip-math-prog', 
        action='store_true',
        help='Skip mathematical and programming benchmarks'
    )
    parser.add_argument(
        '--skip-scalability', 
        action='store_true',
        help='Skip scalability analysis'
    )
    parser.add_argument(
        '--base-corpus', 
        help='Base corpus for scalability testing'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Start benchmarking
    start_time = time.time()
    logger.info("Starting comprehensive benchmarking suite...")
    
    all_results = {}
    
    try:
        # Run corpus benchmarks
        if not args.skip_corpus:
            corpus_results = run_corpus_benchmarks(args.corpus_paths, args.output_dir)
            all_results.update(corpus_results)
        
        # Run mathematical and programming benchmarks
        if not args.skip_math_prog:
            math_prog_results = run_math_programming_benchmarks_suite(args.output_dir)
            all_results.update(math_prog_results)
        
        # Run scalability analysis
        if not args.skip_scalability:
            scalability_results = run_scalability_analysis(args.base_corpus, args.output_dir)
            all_results.update(scalability_results)
        
        # Generate optimization recommendations
        if all_results:
            generate_optimization_recommendations(all_results, args.output_dir)
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print("BENCHMARKING COMPLETE")
        print("="*60)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Results saved to: {args.output_dir}")
        
        if all_results:
            print(f"Benchmark suites completed: {len(all_results)}")
            for suite_name in all_results.keys():
                print(f"  - {suite_name}")
        
        logger.info("Benchmarking suite completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        print(f"ERROR: Benchmarking failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()