"""
Comprehensive corpus benchmarking system for the Sanskrit Rewrite Engine.

This module provides extensive benchmarking capabilities including:
- Sanskrit corpus evaluation and accuracy metrics
- Performance benchmarking for scalability testing
- Rule coverage analysis for completeness validation
- Semantic consistency validation across transformations
- Automated benchmarking reports and optimization recommendations
"""

import json
import time
import statistics
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .tokenizer import SanskritTokenizer
from .panini_engine import PaniniRuleEngine, PaniniEngineResult
from .morphological_analyzer import SanskritMorphologicalAnalyzer
from .semantic_pipeline import SemanticProcessor, process_sanskrit_text
from .rule import RuleRegistry
from .essential_sutras import create_essential_sutras


@dataclass
class AccuracyMetrics:
    """Metrics for grammatical analysis accuracy."""
    total_tests: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        return self.correct_predictions / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        tp_fp = self.correct_predictions + self.false_positives
        return self.correct_predictions / tp_fp if tp_fp > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        tp_fn = self.correct_predictions + self.false_negatives
        return self.correct_predictions / tp_fn if tp_fn > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for scalability testing."""
    processing_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    rule_applications: List[int] = field(default_factory=list)
    
    @property
    def mean_processing_time(self) -> float:
        """Average processing time."""
        return statistics.mean(self.processing_times) if self.processing_times else 0.0
    
    @property
    def throughput_tokens_per_second(self) -> float:
        """Tokens processed per second."""
        total_tokens = sum(self.token_counts)
        total_time = sum(self.processing_times)
        return total_tokens / total_time if total_time > 0 else 0.0
    
    @property
    def mean_memory_usage(self) -> float:
        """Average memory usage in MB."""
        return statistics.mean(self.memory_usage) if self.memory_usage else 0.0


@dataclass
class RuleCoverageMetrics:
    """Metrics for rule coverage analysis."""
    total_rules: int = 0
    applied_rules: Set[str] = field(default_factory=set)
    rule_application_counts: Dict[str, int] = field(default_factory=dict)
    unused_rules: Set[str] = field(default_factory=set)
    
    @property
    def coverage_percentage(self) -> float:
        """Percentage of rules that were applied."""
        return len(self.applied_rules) / self.total_rules * 100 if self.total_rules > 0 else 0.0
    
    @property
    def most_used_rules(self) -> List[Tuple[str, int]]:
        """Top 10 most frequently used rules."""
        return sorted(self.rule_application_counts.items(), key=lambda x: x[1], reverse=True)[:10]


@dataclass
class SemanticConsistencyMetrics:
    """Metrics for semantic consistency validation."""
    total_transformations: int = 0
    consistent_transformations: int = 0
    semantic_errors: List[str] = field(default_factory=list)
    cross_language_consistency: Dict[str, float] = field(default_factory=dict)
    
    @property
    def consistency_rate(self) -> float:
        """Rate of semantically consistent transformations."""
        return self.consistent_transformations / self.total_transformations if self.total_transformations > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    timestamp: datetime
    corpus_name: str
    accuracy_metrics: AccuracyMetrics
    performance_metrics: PerformanceMetrics
    rule_coverage_metrics: RuleCoverageMetrics
    semantic_consistency_metrics: SemanticConsistencyMetrics
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CorpusBenchmarker:
    """
    Comprehensive benchmarking system for Sanskrit corpus evaluation.
    
    Features:
    - Multi-corpus evaluation with accuracy metrics
    - Performance profiling and scalability analysis
    - Rule coverage analysis and optimization recommendations
    - Semantic consistency validation across transformations
    - Automated report generation with actionable insights
    """
    
    def __init__(self, 
                 tokenizer: Optional[SanskritTokenizer] = None,
                 engine: Optional[PaniniRuleEngine] = None,
                 morphological_analyzer: Optional[SanskritMorphologicalAnalyzer] = None,
                 semantic_processor: Optional[SemanticProcessor] = None):
        """
        Initialize the corpus benchmarker.
        
        Args:
            tokenizer: Sanskrit tokenizer instance
            engine: Panini rule engine instance
            morphological_analyzer: Morphological analyzer instance
            semantic_processor: Semantic processor instance
        """
        self.tokenizer = tokenizer or SanskritTokenizer()
        self.engine = engine or PaniniRuleEngine(self.tokenizer)
        self.morphological_analyzer = morphological_analyzer or SanskritMorphologicalAnalyzer()
        self.semantic_processor = semantic_processor or SemanticProcessor()
        
        self.logger = logging.getLogger(__name__)
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Initialize memory tracking
        try:
            import psutil
            import os
            self.process = psutil.Process(os.getpid())
            self.memory_tracking_available = True
        except ImportError:
            self.memory_tracking_available = False
            self.logger.warning("psutil not available. Memory tracking disabled.")
    
    def benchmark_corpus(self, corpus_path: str, corpus_name: str = None) -> BenchmarkResult:
        """
        Benchmark a complete Sanskrit corpus.
        
        Args:
            corpus_path: Path to the corpus file (JSON format)
            corpus_name: Name of the corpus for reporting
            
        Returns:
            Complete benchmark result
        """
        corpus_name = corpus_name or Path(corpus_path).stem
        self.logger.info(f"Starting benchmark for corpus: {corpus_name}")
        
        # Load corpus data
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load corpus {corpus_path}: {e}")
            return self._create_error_result(corpus_name, f"Failed to load corpus: {e}")
        
        # Initialize metrics
        accuracy_metrics = AccuracyMetrics()
        performance_metrics = PerformanceMetrics()
        rule_coverage_metrics = RuleCoverageMetrics()
        semantic_consistency_metrics = SemanticConsistencyMetrics()
        errors = []
        
        # Get total rule count for coverage analysis
        rule_coverage_metrics.total_rules = len(self.engine.registry.get_active_sutra_rules())
        
        # Process each item in the corpus
        for i, item in enumerate(corpus_data):
            try:
                self.logger.debug(f"Processing item {i+1}/{len(corpus_data)}: {item.get('id', 'unknown')}")
                
                # Extract test data
                source_text = item.get('source_text', '')
                expected_target = item.get('target_text', '')
                expected_analysis = item.get('grammatical_analysis', {})
                
                if not source_text:
                    continue
                
                # Measure performance
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                # Process with engine
                tokens = self.tokenizer.tokenize(source_text)
                result = self.engine.process(tokens)
                
                # Morphological analysis
                try:
                    output_text = result.get_output_text()
                    words = output_text.split()
                    morph_analysis = self.morphological_analyzer.analyze_sentence(words) if words else []
                except (AttributeError, Exception):
                    # Fallback if analysis fails
                    morph_analysis = []
                
                # Semantic processing
                semantic_result = process_sanskrit_text(source_text)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                
                # Update performance metrics
                processing_time = end_time - start_time
                performance_metrics.processing_times.append(processing_time)
                performance_metrics.token_counts.append(len(tokens))
                performance_metrics.rule_applications.append(len(result.get_rules_applied()))
                
                if self.memory_tracking_available:
                    memory_delta = end_memory - start_memory
                    performance_metrics.memory_usage.append(memory_delta)
                
                # Update rule coverage metrics
                applied_rules = result.get_rules_applied()
                rule_coverage_metrics.applied_rules.update(applied_rules)
                
                for rule_name in applied_rules:
                    rule_coverage_metrics.rule_application_counts[rule_name] = \
                        rule_coverage_metrics.rule_application_counts.get(rule_name, 0) + 1
                
                # Evaluate accuracy
                self._evaluate_accuracy(result, expected_target, expected_analysis, accuracy_metrics)
                
                # Evaluate semantic consistency
                self._evaluate_semantic_consistency(
                    source_text, result, semantic_result, semantic_consistency_metrics
                )
                
            except Exception as e:
                error_msg = f"Error processing item {i+1}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        # Calculate unused rules
        all_rules = {rule.name for rule in self.engine.registry.get_active_sutra_rules()}
        rule_coverage_metrics.unused_rules = all_rules - rule_coverage_metrics.applied_rules
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy_metrics, performance_metrics, rule_coverage_metrics, semantic_consistency_metrics
        )
        
        # Create final result
        result = BenchmarkResult(
            timestamp=datetime.now(),
            corpus_name=corpus_name,
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            rule_coverage_metrics=rule_coverage_metrics,
            semantic_consistency_metrics=semantic_consistency_metrics,
            errors=errors,
            recommendations=recommendations
        )
        
        self.benchmark_results.append(result)
        self.logger.info(f"Completed benchmark for corpus: {corpus_name}")
        
        return result
    
    def benchmark_multiple_corpora(self, corpus_paths: List[str]) -> List[BenchmarkResult]:
        """
        Benchmark multiple corpora and return aggregated results.
        
        Args:
            corpus_paths: List of paths to corpus files
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for corpus_path in corpus_paths:
            try:
                result = self.benchmark_corpus(corpus_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to benchmark corpus {corpus_path}: {e}")
                error_result = self._create_error_result(
                    Path(corpus_path).stem, f"Benchmark failed: {e}"
                )
                results.append(error_result)
        
        return results
    
    def benchmark_scalability(self, 
                            base_corpus_path: str, 
                            scale_factors: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark scalability by testing with different corpus sizes.
        
        Args:
            base_corpus_path: Path to base corpus file
            scale_factors: List of scale factors to test (default: [1, 2, 5, 10])
            
        Returns:
            Scalability analysis results
        """
        scale_factors = scale_factors or [1, 2, 5, 10]
        
        self.logger.info(f"Starting scalability benchmark with factors: {scale_factors}")
        
        # Load base corpus
        with open(base_corpus_path, 'r', encoding='utf-8') as f:
            base_corpus = json.load(f)
        
        scalability_results = {
            'scale_factors': scale_factors,
            'processing_times': [],
            'memory_usage': [],
            'throughput': [],
            'accuracy_degradation': [],
            'rule_coverage_change': []
        }
        
        baseline_accuracy = None
        baseline_coverage = None
        
        for scale_factor in scale_factors:
            # Create scaled corpus
            scaled_corpus = base_corpus * scale_factor
            
            # Save temporary corpus
            temp_path = f"temp_corpus_scale_{scale_factor}.json"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(scaled_corpus, f)
            
            try:
                # Benchmark scaled corpus
                result = self.benchmark_corpus(temp_path, f"scaled_{scale_factor}x")
                
                # Record scalability metrics
                scalability_results['processing_times'].append(
                    result.performance_metrics.mean_processing_time
                )
                scalability_results['memory_usage'].append(
                    result.performance_metrics.mean_memory_usage
                )
                scalability_results['throughput'].append(
                    result.performance_metrics.throughput_tokens_per_second
                )
                
                # Track accuracy and coverage changes
                current_accuracy = result.accuracy_metrics.accuracy
                current_coverage = result.rule_coverage_metrics.coverage_percentage
                
                if baseline_accuracy is None:
                    baseline_accuracy = current_accuracy
                    baseline_coverage = current_coverage
                    scalability_results['accuracy_degradation'].append(0.0)
                    scalability_results['rule_coverage_change'].append(0.0)
                else:
                    accuracy_change = ((current_accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0.0
                    coverage_change = ((current_coverage - baseline_coverage) / baseline_coverage * 100) if baseline_coverage > 0 else 0.0
                    scalability_results['accuracy_degradation'].append(accuracy_change)
                    scalability_results['rule_coverage_change'].append(coverage_change)
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
        
        return scalability_results
    
    def _evaluate_accuracy(self, 
                          result: PaniniEngineResult, 
                          expected_target: str, 
                          expected_analysis: Dict[str, Any], 
                          metrics: AccuracyMetrics) -> None:
        """Evaluate accuracy of the transformation result."""
        metrics.total_tests += 1
        
        # Get actual output
        actual_output = result.get_output_text()
        
        # Simple string comparison for now
        # TODO: Implement more sophisticated comparison considering variants
        if actual_output.strip() == expected_target.strip():
            metrics.correct_predictions += 1
        else:
            # Determine if it's a false positive or false negative
            # This is a simplified heuristic - could be improved
            if len(actual_output) > len(expected_target):
                metrics.false_positives += 1
            else:
                metrics.false_negatives += 1
    
    def _evaluate_semantic_consistency(self, 
                                     source_text: str, 
                                     engine_result: PaniniEngineResult, 
                                     semantic_result: Any, 
                                     metrics: SemanticConsistencyMetrics) -> None:
        """Evaluate semantic consistency of transformations."""
        metrics.total_transformations += 1
        
        try:
            # Check if transformations preserve semantic meaning
            # This is a placeholder for more sophisticated semantic analysis
            
            # Basic consistency check: ensure no semantic information is lost
            source_concepts = len(source_text.split())  # Simplified
            target_concepts = len(engine_result.get_output_text().split())
            
            # Allow for reasonable variation in concept count
            if abs(source_concepts - target_concepts) <= 1:
                metrics.consistent_transformations += 1
            else:
                error_msg = f"Semantic inconsistency: {source_concepts} -> {target_concepts} concepts"
                metrics.semantic_errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Semantic evaluation error: {e}"
            metrics.semantic_errors.append(error_msg)
    
    def _generate_recommendations(self, 
                                accuracy: AccuracyMetrics, 
                                performance: PerformanceMetrics, 
                                coverage: RuleCoverageMetrics, 
                                semantic: SemanticConsistencyMetrics) -> List[str]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        
        # Accuracy recommendations
        if accuracy.accuracy < 0.8:
            recommendations.append(
                f"Low accuracy ({accuracy.accuracy:.2%}). Consider reviewing rule priorities "
                f"and adding more specific transformation rules."
            )
        
        if accuracy.precision < 0.7:
            recommendations.append(
                f"Low precision ({accuracy.precision:.2%}). Reduce false positives by "
                f"adding more restrictive rule conditions."
            )
        
        if accuracy.recall < 0.7:
            recommendations.append(
                f"Low recall ({accuracy.recall:.2%}). Add more comprehensive rules to "
                f"catch missed transformations."
            )
        
        # Performance recommendations
        if performance.mean_processing_time > 1.0:
            recommendations.append(
                f"High processing time ({performance.mean_processing_time:.3f}s). "
                f"Consider optimizing rule matching algorithms or adding rule indexing."
            )
        
        if performance.throughput_tokens_per_second < 100:
            recommendations.append(
                f"Low throughput ({performance.throughput_tokens_per_second:.1f} tokens/s). "
                f"Consider batch processing or parallel rule application."
            )
        
        # Rule coverage recommendations
        if coverage.coverage_percentage < 50:
            recommendations.append(
                f"Low rule coverage ({coverage.coverage_percentage:.1f}%). "
                f"Many rules are unused - consider removing redundant rules or "
                f"expanding test corpus to exercise more rules."
            )
        
        if len(coverage.unused_rules) > coverage.total_rules * 0.3:
            recommendations.append(
                f"High number of unused rules ({len(coverage.unused_rules)}). "
                f"Review rule necessity and corpus completeness."
            )
        
        # Semantic consistency recommendations
        if semantic.consistency_rate < 0.9:
            recommendations.append(
                f"Low semantic consistency ({semantic.consistency_rate:.2%}). "
                f"Review transformations for semantic preservation."
            )
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.memory_tracking_available:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def _create_error_result(self, corpus_name: str, error_message: str) -> BenchmarkResult:
        """Create a benchmark result for error cases."""
        return BenchmarkResult(
            timestamp=datetime.now(),
            corpus_name=corpus_name,
            accuracy_metrics=AccuracyMetrics(),
            performance_metrics=PerformanceMetrics(),
            rule_coverage_metrics=RuleCoverageMetrics(),
            semantic_consistency_metrics=SemanticConsistencyMetrics(),
            errors=[error_message],
            recommendations=["Fix corpus loading or processing errors before benchmarking."]
        )
    
    def generate_comprehensive_report(self, results: List[BenchmarkResult] = None) -> str:
        """
        Generate a comprehensive benchmarking report.
        
        Args:
            results: List of benchmark results (uses all stored results if None)
            
        Returns:
            Formatted report string
        """
        results = results or self.benchmark_results
        
        if not results:
            return "No benchmark results available."
        
        report_lines = []
        report_lines.append("SANSKRIT REWRITE ENGINE - CORPUS BENCHMARKING REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Corpora Evaluated: {len(results)}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        
        avg_accuracy = statistics.mean([r.accuracy_metrics.accuracy for r in results])
        avg_performance = statistics.mean([r.performance_metrics.mean_processing_time for r in results])
        avg_coverage = statistics.mean([r.rule_coverage_metrics.coverage_percentage for r in results])
        avg_semantic = statistics.mean([r.semantic_consistency_metrics.consistency_rate for r in results])
        
        report_lines.append(f"Average Accuracy: {avg_accuracy:.2%}")
        report_lines.append(f"Average Processing Time: {avg_performance:.3f}s")
        report_lines.append(f"Average Rule Coverage: {avg_coverage:.1f}%")
        report_lines.append(f"Average Semantic Consistency: {avg_semantic:.2%}")
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("DETAILED RESULTS BY CORPUS")
        report_lines.append("-" * 30)
        
        for result in results:
            report_lines.append(f"\nCorpus: {result.corpus_name}")
            report_lines.append(f"  Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Accuracy Metrics
            acc = result.accuracy_metrics
            report_lines.append(f"  Accuracy: {acc.accuracy:.2%} ({acc.correct_predictions}/{acc.total_tests})")
            report_lines.append(f"  Precision: {acc.precision:.2%}")
            report_lines.append(f"  Recall: {acc.recall:.2%}")
            report_lines.append(f"  F1 Score: {acc.f1_score:.2%}")
            
            # Performance Metrics
            perf = result.performance_metrics
            report_lines.append(f"  Avg Processing Time: {perf.mean_processing_time:.3f}s")
            report_lines.append(f"  Throughput: {perf.throughput_tokens_per_second:.1f} tokens/s")
            if perf.memory_usage:
                report_lines.append(f"  Avg Memory Usage: {perf.mean_memory_usage:.2f} MB")
            
            # Rule Coverage
            cov = result.rule_coverage_metrics
            report_lines.append(f"  Rule Coverage: {cov.coverage_percentage:.1f}% ({len(cov.applied_rules)}/{cov.total_rules})")
            report_lines.append(f"  Unused Rules: {len(cov.unused_rules)}")
            
            # Top Rules
            if cov.most_used_rules:
                top_rule = cov.most_used_rules[0]
                report_lines.append(f"  Most Used Rule: {top_rule[0]} ({top_rule[1]} applications)")
            
            # Semantic Consistency
            sem = result.semantic_consistency_metrics
            report_lines.append(f"  Semantic Consistency: {sem.consistency_rate:.2%}")
            
            # Errors
            if result.errors:
                report_lines.append(f"  Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    report_lines.append(f"    - {error}")
                if len(result.errors) > 3:
                    report_lines.append(f"    ... and {len(result.errors) - 3} more")
        
        # Recommendations
        report_lines.append("\nOPTIMIZATION RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Deduplicate and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        for i, rec in enumerate(unique_recommendations[:10], 1):  # Top 10
            report_lines.append(f"{i}. {rec}")
        
        # Performance Analysis
        report_lines.append("\nPERFORMANCE ANALYSIS")
        report_lines.append("-" * 20)
        
        processing_times = [r.performance_metrics.mean_processing_time for r in results]
        throughputs = [r.performance_metrics.throughput_tokens_per_second for r in results]
        
        if len(processing_times) > 1:
            report_lines.append(f"Processing Time - Min: {min(processing_times):.3f}s, "
                              f"Max: {max(processing_times):.3f}s, "
                              f"Std Dev: {statistics.stdev(processing_times):.3f}s")
        else:
            report_lines.append(f"Processing Time - Single measurement: {processing_times[0]:.3f}s")
        
        if len(throughputs) > 1:
            report_lines.append(f"Throughput - Min: {min(throughputs):.1f}, "
                              f"Max: {max(throughputs):.1f}, "
                              f"Std Dev: {statistics.stdev(throughputs):.1f} tokens/s")
        else:
            report_lines.append(f"Throughput - Single measurement: {throughputs[0]:.1f} tokens/s")
        
        return "\n".join(report_lines)
    
    def save_results_to_json(self, filename: str = None) -> str:
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"corpus_benchmark_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.benchmark_results:
            serializable_result = {
                'timestamp': result.timestamp.isoformat(),
                'corpus_name': result.corpus_name,
                'accuracy_metrics': {
                    'total_tests': result.accuracy_metrics.total_tests,
                    'correct_predictions': result.accuracy_metrics.correct_predictions,
                    'false_positives': result.accuracy_metrics.false_positives,
                    'false_negatives': result.accuracy_metrics.false_negatives,
                    'accuracy': result.accuracy_metrics.accuracy,
                    'precision': result.accuracy_metrics.precision,
                    'recall': result.accuracy_metrics.recall,
                    'f1_score': result.accuracy_metrics.f1_score
                },
                'performance_metrics': {
                    'processing_times': result.performance_metrics.processing_times,
                    'memory_usage': result.performance_metrics.memory_usage,
                    'token_counts': result.performance_metrics.token_counts,
                    'rule_applications': result.performance_metrics.rule_applications,
                    'mean_processing_time': result.performance_metrics.mean_processing_time,
                    'throughput_tokens_per_second': result.performance_metrics.throughput_tokens_per_second,
                    'mean_memory_usage': result.performance_metrics.mean_memory_usage
                },
                'rule_coverage_metrics': {
                    'total_rules': result.rule_coverage_metrics.total_rules,
                    'applied_rules': list(result.rule_coverage_metrics.applied_rules),
                    'rule_application_counts': result.rule_coverage_metrics.rule_application_counts,
                    'unused_rules': list(result.rule_coverage_metrics.unused_rules),
                    'coverage_percentage': result.rule_coverage_metrics.coverage_percentage,
                    'most_used_rules': result.rule_coverage_metrics.most_used_rules
                },
                'semantic_consistency_metrics': {
                    'total_transformations': result.semantic_consistency_metrics.total_transformations,
                    'consistent_transformations': result.semantic_consistency_metrics.consistent_transformations,
                    'semantic_errors': result.semantic_consistency_metrics.semantic_errors,
                    'cross_language_consistency': result.semantic_consistency_metrics.cross_language_consistency,
                    'consistency_rate': result.semantic_consistency_metrics.consistency_rate
                },
                'errors': result.errors,
                'recommendations': result.recommendations
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Benchmark results saved to {filename}")
        return filename


def create_corpus_benchmarker() -> CorpusBenchmarker:
    """Create a fully configured corpus benchmarker."""
    tokenizer = SanskritTokenizer()
    engine = PaniniRuleEngine(tokenizer)
    morphological_analyzer = SanskritMorphologicalAnalyzer()
    semantic_processor = SemanticProcessor()
    
    return CorpusBenchmarker(
        tokenizer=tokenizer,
        engine=engine,
        morphological_analyzer=morphological_analyzer,
        semantic_processor=semantic_processor
    )


def benchmark_all_test_corpora() -> List[BenchmarkResult]:
    """
    Benchmark all available test corpora.
    
    Returns:
        List of benchmark results
    """
    benchmarker = create_corpus_benchmarker()
    
    # Find all corpus files
    corpus_dir = Path("test_corpus")
    corpus_files = list(corpus_dir.glob("*.json"))
    
    if not corpus_files:
        logging.warning("No corpus files found in test_corpus directory")
        return []
    
    # Benchmark each corpus
    results = []
    for corpus_file in corpus_files:
        try:
            result = benchmarker.benchmark_corpus(str(corpus_file))
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to benchmark {corpus_file}: {e}")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks on all test corpora
    results = benchmark_all_test_corpora()
    
    if results:
        # Create benchmarker for report generation
        benchmarker = create_corpus_benchmarker()
        benchmarker.benchmark_results = results
        
        # Generate and print report
        report = benchmarker.generate_comprehensive_report()
        print(report)
        
        # Save results
        output_file = benchmarker.save_results_to_json()
        print(f"\nResults saved to: {output_file}")
    else:
        print("No benchmark results to report.")