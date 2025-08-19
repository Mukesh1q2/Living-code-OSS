"""
Comprehensive test suite for corpus benchmarking functionality.

This module tests all aspects of the corpus benchmarking system including:
- Sanskrit corpus evaluation
- Mathematical and programming benchmarks
- Performance and scalability testing
- Rule coverage analysis
- Semantic consistency validation
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from sanskrit_rewrite_engine.corpus_benchmarking import (
    CorpusBenchmarker, AccuracyMetrics, PerformanceMetrics, 
    RuleCoverageMetrics, SemanticConsistencyMetrics,
    create_corpus_benchmarker, benchmark_all_test_corpora
)
from sanskrit_rewrite_engine.math_programming_benchmarks import (
    MathProgrammingBenchmarker, run_math_programming_benchmarks
)
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine


class TestAccuracyMetrics(unittest.TestCase):
    """Test accuracy metrics calculations."""
    
    def test_accuracy_calculation(self):
        """Test accuracy metric calculations."""
        metrics = AccuracyMetrics()
        metrics.total_tests = 100
        metrics.correct_predictions = 80
        metrics.false_positives = 10
        metrics.false_negatives = 10
        
        self.assertEqual(metrics.accuracy, 0.8)
        self.assertEqual(metrics.precision, 80/90)  # TP/(TP+FP)
        self.assertEqual(metrics.recall, 80/90)     # TP/(TP+FN)
        
        # Test F1 score
        expected_f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        self.assertAlmostEqual(metrics.f1_score, expected_f1, places=5)
    
    def test_edge_cases(self):
        """Test edge cases for metrics."""
        metrics = AccuracyMetrics()
        
        # No tests
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1_score, 0.0)
        
        # Perfect accuracy
        metrics.total_tests = 10
        metrics.correct_predictions = 10
        self.assertEqual(metrics.accuracy, 1.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculations."""
    
    def test_performance_calculations(self):
        """Test performance metric calculations."""
        metrics = PerformanceMetrics()
        metrics.processing_times = [0.1, 0.2, 0.3]
        metrics.token_counts = [10, 20, 30]
        metrics.memory_usage = [1.0, 2.0, 3.0]
        
        self.assertEqual(metrics.mean_processing_time, 0.2)
        self.assertAlmostEqual(metrics.throughput_tokens_per_second, 100.0, places=1)  # 100 tokens/s
        self.assertEqual(metrics.mean_memory_usage, 2.0)
    
    def test_empty_metrics(self):
        """Test empty performance metrics."""
        metrics = PerformanceMetrics()
        
        self.assertEqual(metrics.mean_processing_time, 0.0)
        self.assertEqual(metrics.throughput_tokens_per_second, 0.0)
        self.assertEqual(metrics.mean_memory_usage, 0.0)


class TestRuleCoverageMetrics(unittest.TestCase):
    """Test rule coverage metrics."""
    
    def test_coverage_calculation(self):
        """Test rule coverage calculations."""
        metrics = RuleCoverageMetrics()
        metrics.total_rules = 100
        metrics.applied_rules = {"rule1", "rule2", "rule3"}
        metrics.rule_application_counts = {"rule1": 5, "rule2": 3, "rule3": 1}
        
        self.assertEqual(metrics.coverage_percentage, 3.0)
        
        most_used = metrics.most_used_rules
        self.assertEqual(most_used[0], ("rule1", 5))
        self.assertEqual(most_used[1], ("rule2", 3))
        self.assertEqual(most_used[2], ("rule3", 1))


class TestCorpusBenchmarker(unittest.TestCase):
    """Test the main corpus benchmarker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmarker = create_corpus_benchmarker()
        
        # Create temporary test corpus
        self.test_corpus = [
            {
                "id": "test_001",
                "source_text": "rāma + iti",
                "target_text": "rāmeti",
                "grammatical_analysis": {
                    "rule_type": "vowel_sandhi",
                    "transformation": "a + i → e"
                },
                "sutra_references": ["6.1.87"],
                "metadata": {"difficulty": "BEGINNER"}
            },
            {
                "id": "test_002",
                "source_text": "dharma + artha",
                "target_text": "dharmārtha",
                "grammatical_analysis": {
                    "rule_type": "vowel_sandhi",
                    "transformation": "a + a → ā"
                },
                "sutra_references": ["6.1.101"],
                "metadata": {"difficulty": "BEGINNER"}
            }
        ]
    
    def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        self.assertIsInstance(self.benchmarker.tokenizer, SanskritTokenizer)
        self.assertIsInstance(self.benchmarker.engine, PaniniRuleEngine)
        self.assertEqual(len(self.benchmarker.benchmark_results), 0)
    
    def test_corpus_benchmarking(self):
        """Test corpus benchmarking functionality."""
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_corpus, f)
            temp_path = f.name
        
        try:
            # Run benchmark
            result = self.benchmarker.benchmark_corpus(temp_path, "test_corpus")
            
            # Verify result structure
            self.assertEqual(result.corpus_name, "test_corpus")
            self.assertIsInstance(result.accuracy_metrics, AccuracyMetrics)
            self.assertIsInstance(result.performance_metrics, PerformanceMetrics)
            self.assertIsInstance(result.rule_coverage_metrics, RuleCoverageMetrics)
            self.assertIsInstance(result.semantic_consistency_metrics, SemanticConsistencyMetrics)
            
            # Verify metrics were populated
            self.assertEqual(result.accuracy_metrics.total_tests, 2)
            self.assertGreater(len(result.performance_metrics.processing_times), 0)
            
        finally:
            os.unlink(temp_path)
    
    def test_error_handling(self):
        """Test error handling in benchmarking."""
        # Test with non-existent file
        result = self.benchmarker.benchmark_corpus("nonexistent.json", "error_test")
        
        self.assertEqual(result.corpus_name, "error_test")
        self.assertGreater(len(result.errors), 0)
        self.assertIn("Failed to load corpus", result.errors[0])
    
    def test_report_generation(self):
        """Test report generation."""
        # Create a mock result
        from sanskrit_rewrite_engine.corpus_benchmarking import BenchmarkResult
        from datetime import datetime
        
        mock_result = BenchmarkResult(
            timestamp=datetime.now(),
            corpus_name="test",
            accuracy_metrics=AccuracyMetrics(),
            performance_metrics=PerformanceMetrics(),
            rule_coverage_metrics=RuleCoverageMetrics(),
            semantic_consistency_metrics=SemanticConsistencyMetrics(),
            recommendations=["Test recommendation"]
        )
        
        self.benchmarker.benchmark_results = [mock_result]
        report = self.benchmarker.generate_comprehensive_report()
        
        self.assertIn("SANSKRIT REWRITE ENGINE", report)
        self.assertIn("test", report)
        self.assertIn("Test recommendation", report)
    
    def test_scalability_benchmarking(self):
        """Test scalability benchmarking."""
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_corpus, f)
            temp_path = f.name
        
        try:
            # Run scalability benchmark with small scale factors
            result = self.benchmarker.benchmark_scalability(temp_path, [1, 2])
            
            # Verify result structure
            self.assertIn('scale_factors', result)
            self.assertIn('processing_times', result)
            self.assertIn('throughput', result)
            self.assertEqual(len(result['scale_factors']), 2)
            
        finally:
            os.unlink(temp_path)


class TestMathProgrammingBenchmarker(unittest.TestCase):
    """Test mathematical and programming benchmarker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmarker = MathProgrammingBenchmarker()
    
    def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        self.assertIsNotNone(self.benchmarker.symbolic_engine)
        self.assertIsNotNone(self.benchmarker.domain_mapper)
        self.assertGreater(len(self.benchmarker.math_benchmarks), 0)
        self.assertGreater(len(self.benchmarker.programming_benchmarks), 0)
        self.assertGreater(len(self.benchmarker.cross_domain_benchmarks), 0)
    
    def test_mathematical_benchmarking(self):
        """Test mathematical translation benchmarking."""
        accuracy, vedic_coverage = self.benchmarker.benchmark_mathematical_translation()
        
        self.assertIsInstance(accuracy, AccuracyMetrics)
        self.assertGreater(accuracy.total_tests, 0)
        self.assertIsInstance(vedic_coverage, dict)
    
    def test_programming_benchmarking(self):
        """Test programming generation benchmarking."""
        accuracy, lang_coverage = self.benchmarker.benchmark_programming_generation()
        
        self.assertIsInstance(accuracy, AccuracyMetrics)
        self.assertGreater(accuracy.total_tests, 0)
        self.assertIsInstance(lang_coverage, dict)
    
    def test_cross_domain_benchmarking(self):
        """Test cross-domain mapping benchmarking."""
        accuracy, semantic_rate = self.benchmarker.benchmark_cross_domain_mapping()
        
        self.assertIsInstance(accuracy, AccuracyMetrics)
        self.assertGreater(accuracy.total_tests, 0)
        self.assertIsInstance(semantic_rate, float)
        self.assertGreaterEqual(semantic_rate, 0.0)
        self.assertLessEqual(semantic_rate, 1.0)
    
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmarking."""
        result = self.benchmarker.run_comprehensive_benchmark()
        
        # Verify result structure
        self.assertIsNotNone(result.timestamp)
        self.assertIsInstance(result.math_accuracy, AccuracyMetrics)
        self.assertIsInstance(result.programming_accuracy, AccuracyMetrics)
        self.assertIsInstance(result.cross_domain_accuracy, AccuracyMetrics)
        self.assertIsInstance(result.performance_metrics, PerformanceMetrics)
        self.assertIsInstance(result.vedic_math_coverage, dict)
        self.assertIsInstance(result.programming_language_coverage, dict)
        self.assertIsInstance(result.semantic_preservation_rate, float)
    
    def test_report_generation(self):
        """Test report generation for math/programming benchmarks."""
        result = self.benchmarker.run_comprehensive_benchmark()
        report = self.benchmarker.generate_report(result)
        
        self.assertIn("MATHEMATICAL & PROGRAMMING BENCHMARK REPORT", report)
        self.assertIn("MATHEMATICAL TRANSLATION", report)
        self.assertIn("PROGRAMMING GENERATION", report)
        self.assertIn("CROSS-DOMAIN MAPPING", report)


class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration of all benchmarking components."""
    
    def test_corpus_benchmark_integration(self):
        """Test integration with existing test corpus."""
        # Check if test corpus directory exists
        corpus_dir = Path("test_corpus")
        if corpus_dir.exists():
            corpus_files = list(corpus_dir.glob("*.json"))
            if corpus_files:
                # Run benchmark on first available corpus
                benchmarker = create_corpus_benchmarker()
                result = benchmarker.benchmark_corpus(str(corpus_files[0]))
                
                self.assertIsNotNone(result)
                self.assertGreater(len(result.corpus_name), 0)
    
    def test_math_programming_integration(self):
        """Test math/programming benchmark integration."""
        result = run_math_programming_benchmarks()
        
        self.assertIsNotNone(result)
        self.assertGreater(result.math_accuracy.total_tests, 0)
        self.assertGreater(result.programming_accuracy.total_tests, 0)
        self.assertGreater(result.cross_domain_accuracy.total_tests, 0)
    
    @patch('sanskrit_rewrite_engine.corpus_benchmarking.Path')
    def test_benchmark_all_corpora_no_files(self, mock_path):
        """Test benchmark_all_test_corpora with no files."""
        mock_path.return_value.glob.return_value = []
        
        results = benchmark_all_test_corpora()
        self.assertEqual(len(results), 0)


class TestBenchmarkUtilities(unittest.TestCase):
    """Test utility functions for benchmarking."""
    
    def test_comparison_functions(self):
        """Test various comparison functions."""
        benchmarker = MathProgrammingBenchmarker()
        
        # Test mathematical expression comparison
        self.assertTrue(benchmarker._compare_mathematical_expressions("x + y", "x+y"))
        self.assertTrue(benchmarker._compare_mathematical_expressions("2*x", "2*X"))
        self.assertFalse(benchmarker._compare_mathematical_expressions("x + y", "x - y"))
        
        # Test code semantic comparison
        self.assertTrue(benchmarker._compare_code_semantics("if x: y", "if x:y"))
        self.assertTrue(benchmarker._check_code_equivalence("def f(): return x", "def g(): return y"))
        
        # Test syntax validation
        self.assertTrue(benchmarker._is_syntactically_valid("x = 1", "python"))
        self.assertFalse(benchmarker._is_syntactically_valid("x = ", "python"))
    
    def test_concept_extraction(self):
        """Test concept extraction from text."""
        benchmarker = MathProgrammingBenchmarker()
        
        # Test Sanskrit grammar concepts
        concepts = benchmarker._extract_key_concepts("sandhi yoga", "sanskrit_grammar")
        self.assertIn("sandhi", concepts)
        
        # Test mathematical concepts
        concepts = benchmarker._extract_key_concepts("square formula", "mathematics")
        self.assertIn("square", concepts)
        self.assertIn("formula", concepts)
        
        # Test programming concepts
        concepts = benchmarker._extract_key_concepts("function loop", "programming")
        self.assertIn("function", concepts)
        self.assertIn("loop", concepts)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run all tests
    unittest.main(verbosity=2)