"""
Performance benchmarking suite for Sanskrit Rewrite Engine.

This module provides comprehensive benchmarking capabilities to measure
and analyze performance across different configurations and workloads.
"""

import time
import json
import statistics
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import gc
import sys
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .token import Token, TokenKind
from .rule import SutraRule, RuleRegistry, GuardSystem
from .tokenizer import SanskritTokenizer
from .panini_engine import PaniniRuleEngine
from .performance_optimization import (
    PerformanceOptimizer, PerformanceConfig,
    get_high_performance_config, get_memory_optimized_config
)
from .memory_optimization import MemoryOptimizer, get_rx6800m_memory_config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    config_name: str
    processing_time: float
    tokens_per_second: float
    cache_hit_rate: float
    memory_usage_mb: float
    rules_applied: int
    tokens_processed: int
    parallel_chunks: int
    bottlenecks: List[str]
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
    
    def get_results_by_config(self, config_name: str) -> List[BenchmarkResult]:
        """Get results for a specific configuration."""
        return [r for r in self.results if r.config_name == config_name]
    
    def get_average_performance(self, config_name: str) -> Dict[str, float]:
        """Get average performance metrics for a configuration."""
        results = self.get_results_by_config(config_name)
        if not results:
            return {}
        
        return {
            'avg_processing_time': statistics.mean(r.processing_time for r in results),
            'avg_tokens_per_second': statistics.mean(r.tokens_per_second for r in results),
            'avg_cache_hit_rate': statistics.mean(r.cache_hit_rate for r in results),
            'avg_memory_usage_mb': statistics.mean(r.memory_usage_mb for r in results),
            'total_tokens_processed': sum(r.tokens_processed for r in results),
            'total_rules_applied': sum(r.rules_applied for r in results)
        }
    
    def compare_configurations(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across configurations."""
        config_names = set(r.config_name for r in self.results)
        comparison = {}
        
        for config_name in config_names:
            comparison[config_name] = self.get_average_performance(config_name)
        
        return comparison


class PerformanceBenchmark:
    """Main performance benchmarking class."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configurations
        self.configurations = {
            'high_performance': get_high_performance_config(),
            'memory_optimized': get_memory_optimized_config(),
            'default': PerformanceConfig(),
            'parallel_disabled': PerformanceConfig(enable_parallel_processing=False),
            'cache_disabled': PerformanceConfig(enable_rule_caching=False),
            'minimal': PerformanceConfig(
                enable_parallel_processing=False,
                enable_rule_caching=False,
                enable_lazy_evaluation=False,
                enable_memoization=False
            )
        }
        
        # Test data generators
        self.test_data_generators = {
            'small_text': self._generate_small_text,
            'medium_text': self._generate_medium_text,
            'large_text': self._generate_large_text,
            'complex_sandhi': self._generate_complex_sandhi,
            'repetitive_patterns': self._generate_repetitive_patterns,
            'mixed_content': self._generate_mixed_content
        }
        
        logger.info(f"Performance benchmark initialized, output: {self.output_dir}")
    
    def _generate_small_text(self) -> List[Token]:
        """Generate small text sample (10-50 tokens)."""
        text = "राम गच्छति वनम्"
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    def _generate_medium_text(self) -> List[Token]:
        """Generate medium text sample (100-500 tokens)."""
        text = """
        अथ योगानुशासनम् ॥१॥
        योगश्चित्तवृत्तिनिरोधः ॥२॥
        तदा द्रष्टुः स्वरूपेऽवस्थानम् ॥३॥
        वृत्तिसारूप्यमितरत्र ॥४॥
        वृत्तयः पञ्चतय्यः क्लिष्टाक्लिष्टाः ॥५॥
        प्रमाणविपर्ययविकल्पनिद्रास्मृतयः ॥६॥
        प्रत्यक्षानुमानागमाः प्रमाणानि ॥७॥
        विपर्ययो मिथ्याज्ञानमतद्रूपप्रतिष्ठम् ॥८॥
        शब्दज्ञानानुपाती वस्तुशून्यो विकल्पः ॥९॥
        अभावप्रत्ययालम्बना वृत्तिर्निद्रा ॥१०॥
        """
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    def _generate_large_text(self) -> List[Token]:
        """Generate large text sample (1000+ tokens)."""
        # Repeat medium text multiple times
        medium_tokens = self._generate_medium_text()
        return medium_tokens * 10  # ~5000 tokens
    
    def _generate_complex_sandhi(self) -> List[Token]:
        """Generate text with complex sandhi patterns."""
        text = """
        तत् + अस्ति = तदस्ति
        राम + आगच्छति = रामागच्छति
        देव + इन्द्र = देवेन्द्र
        सूर्य + उदय = सूर्योदय
        गुरु + उपदेश = गुरूपदेश
        """
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    def _generate_repetitive_patterns(self) -> List[Token]:
        """Generate text with repetitive patterns for cache testing."""
        patterns = [
            "राम गच्छति",
            "सीता पठति",
            "गुरु शिक्षयति",
            "छात्र लिखति"
        ]
        
        text = " । ".join(patterns * 25)  # Repeat patterns
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    def _generate_mixed_content(self) -> List[Token]:
        """Generate mixed content with various linguistic features."""
        text = """
        श्रीमद्भगवद्गीता अध्याय १
        धृतराष्ट्र उवाच ।
        धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।
        मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥१॥
        
        सञ्जय उवाच ।
        दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा ।
        आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् ॥२॥
        """
        tokenizer = SanskritTokenizer()
        return tokenizer.tokenize(text)
    
    def run_single_benchmark(self, test_name: str, config_name: str, 
                           tokens: List[Token], runs: int = 3) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.info(f"Running benchmark: {test_name} with {config_name}")
        
        config = self.configurations[config_name]
        processing_times = []
        
        try:
            # Warm up
            optimizer = PerformanceOptimizer(config)
            engine = PaniniRuleEngine()
            
            # Run multiple times for statistical accuracy
            for run in range(runs):
                # Reset state
                gc.collect()
                
                start_time = time.time()
                
                # Process tokens
                result = engine.process(tokens)
                
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
            
            # Calculate average metrics
            avg_processing_time = statistics.mean(processing_times)
            tokens_per_second = len(tokens) / avg_processing_time if avg_processing_time > 0 else 0
            
            # Get performance report
            report = optimizer.get_performance_report()
            
            return BenchmarkResult(
                name=test_name,
                config_name=config_name,
                processing_time=avg_processing_time,
                tokens_per_second=tokens_per_second,
                cache_hit_rate=report['cache_statistics']['hit_rate'],
                memory_usage_mb=report['memory_report']['memory_stats']['used_memory_gb'] * 1024,
                rules_applied=report['performance_metrics']['rules_applied'],
                tokens_processed=len(tokens),
                parallel_chunks=report['performance_metrics']['parallel_chunks_processed'],
                bottlenecks=report['bottlenecks']
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                name=test_name,
                config_name=config_name,
                processing_time=0.0,
                tokens_per_second=0.0,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,
                rules_applied=0,
                tokens_processed=len(tokens),
                parallel_chunks=0,
                bottlenecks=[],
                error=str(e)
            )
    
    def run_comprehensive_benchmark(self, runs_per_test: int = 3) -> BenchmarkSuite:
        """Run comprehensive benchmark across all configurations and test cases."""
        logger.info("Starting comprehensive performance benchmark")
        
        suite = BenchmarkSuite(
            name="comprehensive_benchmark",
            metadata={
                'timestamp': time.time(),
                'runs_per_test': runs_per_test,
                'cpu_count': multiprocessing.cpu_count(),
                'python_version': sys.version
            }
        )
        
        total_tests = len(self.test_data_generators) * len(self.configurations)
        current_test = 0
        
        for test_name, generator in self.test_data_generators.items():
            logger.info(f"Generating test data: {test_name}")
            tokens = generator()
            
            for config_name in self.configurations:
                current_test += 1
                logger.info(f"Progress: {current_test}/{total_tests}")
                
                result = self.run_single_benchmark(
                    test_name, config_name, tokens, runs_per_test
                )
                suite.add_result(result)
        
        logger.info("Comprehensive benchmark completed")
        return suite
    
    def run_scalability_benchmark(self, base_tokens: List[Token], 
                                scale_factors: List[int] = None) -> BenchmarkSuite:
        """Run scalability benchmark with increasing token counts."""
        if scale_factors is None:
            scale_factors = [1, 2, 5, 10, 20, 50]
        
        logger.info("Starting scalability benchmark")
        
        suite = BenchmarkSuite(
            name="scalability_benchmark",
            metadata={
                'base_token_count': len(base_tokens),
                'scale_factors': scale_factors
            }
        )
        
        for scale_factor in scale_factors:
            scaled_tokens = base_tokens * scale_factor
            test_name = f"scale_{scale_factor}x_{len(scaled_tokens)}_tokens"
            
            # Test with high-performance configuration
            result = self.run_single_benchmark(
                test_name, 'high_performance', scaled_tokens
            )
            suite.add_result(result)
        
        return suite
    
    def run_configuration_comparison(self, tokens: List[Token]) -> BenchmarkSuite:
        """Compare all configurations on the same dataset."""
        logger.info("Starting configuration comparison benchmark")
        
        suite = BenchmarkSuite(
            name="configuration_comparison",
            metadata={'token_count': len(tokens)}
        )
        
        for config_name in self.configurations:
            result = self.run_single_benchmark(
                "comparison_test", config_name, tokens, runs=5
            )
            suite.add_result(result)
        
        return suite
    
    def run_cache_effectiveness_benchmark(self, tokens: List[Token]) -> BenchmarkSuite:
        """Benchmark cache effectiveness with repeated processing."""
        logger.info("Starting cache effectiveness benchmark")
        
        suite = BenchmarkSuite(
            name="cache_effectiveness",
            metadata={'token_count': len(tokens)}
        )
        
        # Test with cache enabled
        config_with_cache = PerformanceConfig(enable_rule_caching=True, cache_size_mb=512)
        optimizer_with_cache = PerformanceOptimizer(config_with_cache)
        
        # Test without cache
        config_without_cache = PerformanceConfig(enable_rule_caching=False)
        optimizer_without_cache = PerformanceOptimizer(config_without_cache)
        
        # Run multiple iterations to test cache warming
        for iteration in range(10):
            # With cache
            start_time = time.time()
            engine = PaniniRuleEngine()
            result_with_cache = engine.process(tokens)
            time_with_cache = time.time() - start_time
            
            report_with_cache = optimizer_with_cache.get_performance_report()
            
            suite.add_result(BenchmarkResult(
                name=f"iteration_{iteration}",
                config_name="with_cache",
                processing_time=time_with_cache,
                tokens_per_second=len(tokens) / time_with_cache,
                cache_hit_rate=report_with_cache['cache_statistics']['hit_rate'],
                memory_usage_mb=report_with_cache['memory_report']['memory_stats']['used_memory_gb'] * 1024,
                rules_applied=report_with_cache['performance_metrics']['rules_applied'],
                tokens_processed=len(tokens),
                parallel_chunks=0,
                bottlenecks=[]
            ))
            
            # Without cache
            start_time = time.time()
            engine = PaniniRuleEngine()
            result_without_cache = engine.process(tokens)
            time_without_cache = time.time() - start_time
            
            suite.add_result(BenchmarkResult(
                name=f"iteration_{iteration}",
                config_name="without_cache",
                processing_time=time_without_cache,
                tokens_per_second=len(tokens) / time_without_cache,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,
                rules_applied=0,
                tokens_processed=len(tokens),
                parallel_chunks=0,
                bottlenecks=[]
            ))
        
        return suite
    
    def save_benchmark_results(self, suite: BenchmarkSuite, filename: str = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"{suite.name}_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = {
            'name': suite.name,
            'metadata': suite.metadata,
            'results': [
                {
                    'name': r.name,
                    'config_name': r.config_name,
                    'processing_time': r.processing_time,
                    'tokens_per_second': r.tokens_per_second,
                    'cache_hit_rate': r.cache_hit_rate,
                    'memory_usage_mb': r.memory_usage_mb,
                    'rules_applied': r.rules_applied,
                    'tokens_processed': r.tokens_processed,
                    'parallel_chunks': r.parallel_chunks,
                    'bottlenecks': r.bottlenecks,
                    'error': r.error
                }
                for r in suite.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved: {output_file}")
        return str(output_file)
    
    def load_benchmark_results(self, filename: str) -> BenchmarkSuite:
        """Load benchmark results from JSON file."""
        input_file = self.output_dir / filename
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        suite = BenchmarkSuite(
            name=data['name'],
            metadata=data['metadata']
        )
        
        for result_data in data['results']:
            result = BenchmarkResult(**result_data)
            suite.add_result(result)
        
        return suite
    
    def generate_performance_report(self, suite: BenchmarkSuite) -> str:
        """Generate a comprehensive performance report."""
        report_lines = [
            f"Performance Benchmark Report: {suite.name}",
            "=" * 50,
            f"Generated at: {time.ctime(suite.metadata.get('timestamp', time.time()))}",
            f"Total tests: {len(suite.results)}",
            ""
        ]
        
        # Configuration comparison
        comparison = suite.compare_configurations()
        if comparison:
            report_lines.extend([
                "Configuration Performance Comparison:",
                "-" * 40
            ])
            
            for config_name, metrics in comparison.items():
                report_lines.extend([
                    f"\n{config_name.upper()}:",
                    f"  Average processing time: {metrics.get('avg_processing_time', 0):.4f}s",
                    f"  Average tokens/second: {metrics.get('avg_tokens_per_second', 0):.1f}",
                    f"  Average cache hit rate: {metrics.get('avg_cache_hit_rate', 0):.2%}",
                    f"  Average memory usage: {metrics.get('avg_memory_usage_mb', 0):.1f}MB",
                    f"  Total tokens processed: {metrics.get('total_tokens_processed', 0)}",
                    f"  Total rules applied: {metrics.get('total_rules_applied', 0)}"
                ])
        
        # Performance rankings
        if suite.results:
            report_lines.extend([
                "\n\nPerformance Rankings (by tokens/second):",
                "-" * 40
            ])
            
            # Sort by tokens per second
            sorted_results = sorted(
                suite.results, 
                key=lambda r: r.tokens_per_second, 
                reverse=True
            )
            
            for i, result in enumerate(sorted_results[:10], 1):
                report_lines.append(
                    f"{i:2d}. {result.name} ({result.config_name}): "
                    f"{result.tokens_per_second:.1f} tokens/sec"
                )
        
        # Bottleneck analysis
        all_bottlenecks = []
        for result in suite.results:
            all_bottlenecks.extend(result.bottlenecks)
        
        if all_bottlenecks:
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            report_lines.extend([
                "\n\nCommon Bottlenecks:",
                "-" * 20
            ])
            
            for bottleneck, count in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {bottleneck} (occurred {count} times)")
        
        # Recommendations
        report_lines.extend([
            "\n\nRecommendations:",
            "-" * 15
        ])
        
        if comparison:
            best_config = max(comparison.items(), key=lambda x: x[1].get('avg_tokens_per_second', 0))
            report_lines.append(f"  • Best overall configuration: {best_config[0]}")
            
            best_cache = max(comparison.items(), key=lambda x: x[1].get('avg_cache_hit_rate', 0))
            report_lines.append(f"  • Best cache performance: {best_cache[0]}")
            
            lowest_memory = min(comparison.items(), key=lambda x: x[1].get('avg_memory_usage_mb', float('inf')))
            report_lines.append(f"  • Lowest memory usage: {lowest_memory[0]}")
        
        return "\n".join(report_lines)
    
    def plot_performance_comparison(self, suite: BenchmarkSuite, 
                                  output_file: str = None) -> Optional[str]:
        """Plot performance comparison charts."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plots")
            return None
        
        comparison = suite.compare_configurations()
        if not comparison:
            logger.warning("No comparison data available for plotting")
            return None
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Benchmark: {suite.name}', fontsize=16)
        
        configs = list(comparison.keys())
        
        # Processing time comparison
        processing_times = [comparison[config].get('avg_processing_time', 0) for config in configs]
        ax1.bar(configs, processing_times)
        ax1.set_title('Average Processing Time')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Tokens per second comparison
        tokens_per_sec = [comparison[config].get('avg_tokens_per_second', 0) for config in configs]
        ax2.bar(configs, tokens_per_sec)
        ax2.set_title('Average Tokens per Second')
        ax2.set_ylabel('Tokens/sec')
        ax2.tick_params(axis='x', rotation=45)
        
        # Cache hit rate comparison
        cache_rates = [comparison[config].get('avg_cache_hit_rate', 0) * 100 for config in configs]
        ax3.bar(configs, cache_rates)
        ax3.set_title('Average Cache Hit Rate')
        ax3.set_ylabel('Hit Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory_usage = [comparison[config].get('avg_memory_usage_mb', 0) for config in configs]
        ax4.bar(configs, memory_usage)
        ax4.set_title('Average Memory Usage')
        ax4.set_ylabel('Memory (MB)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"performance_comparison_{timestamp}.png"
        
        plot_path = self.output_dir / output_file
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved: {plot_path}")
        return str(plot_path)


def run_full_benchmark_suite(output_dir: str = "./benchmark_results") -> Dict[str, str]:
    """Run the complete benchmark suite and generate reports."""
    benchmark = PerformanceBenchmark(output_dir)
    
    results = {}
    
    # Comprehensive benchmark
    logger.info("Running comprehensive benchmark...")
    comprehensive_suite = benchmark.run_comprehensive_benchmark(runs_per_test=3)
    results['comprehensive'] = benchmark.save_benchmark_results(
        comprehensive_suite, "comprehensive_benchmark.json"
    )
    
    # Scalability benchmark
    logger.info("Running scalability benchmark...")
    base_tokens = benchmark._generate_medium_text()
    scalability_suite = benchmark.run_scalability_benchmark(base_tokens)
    results['scalability'] = benchmark.save_benchmark_results(
        scalability_suite, "scalability_benchmark.json"
    )
    
    # Configuration comparison
    logger.info("Running configuration comparison...")
    comparison_tokens = benchmark._generate_large_text()
    comparison_suite = benchmark.run_configuration_comparison(comparison_tokens)
    results['comparison'] = benchmark.save_benchmark_results(
        comparison_suite, "configuration_comparison.json"
    )
    
    # Cache effectiveness
    logger.info("Running cache effectiveness benchmark...")
    cache_tokens = benchmark._generate_repetitive_patterns()
    cache_suite = benchmark.run_cache_effectiveness_benchmark(cache_tokens)
    results['cache'] = benchmark.save_benchmark_results(
        cache_suite, "cache_effectiveness.json"
    )
    
    # Generate reports
    for suite_name, suite in [
        ('comprehensive', comprehensive_suite),
        ('scalability', scalability_suite),
        ('comparison', comparison_suite),
        ('cache', cache_suite)
    ]:
        report = benchmark.generate_performance_report(suite)
        report_file = benchmark.output_dir / f"{suite_name}_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        results[f'{suite_name}_report'] = str(report_file)
        
        # Generate plots
        plot_file = benchmark.plot_performance_comparison(suite, f"{suite_name}_plots.png")
        if plot_file:
            results[f'{suite_name}_plots'] = plot_file
    
    logger.info(f"Full benchmark suite completed. Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    # Run benchmark suite if executed directly
    results = run_full_benchmark_suite()
    
    print("Benchmark Suite Completed!")
    print("Generated files:")
    for name, path in results.items():
        print(f"  {name}: {path}")