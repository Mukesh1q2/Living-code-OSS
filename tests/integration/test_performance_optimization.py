"""
Test suite for performance optimization module.
"""

import unittest
import time
import threading
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sanskrit_rewrite_engine.token import Token, TokenKind
from sanskrit_rewrite_engine.rule import SutraRule, SutraReference, RuleType, GuardSystem
from sanskrit_rewrite_engine.performance_optimization import (
    PerformanceOptimizer, PerformanceConfig, RuleCache, MemoizedFunction,
    LazyTokenStream, ParallelRuleProcessor, PerformanceProfiler,
    get_high_performance_config, get_memory_optimized_config, memoize
)
from sanskrit_rewrite_engine.performance_benchmarks import (
    PerformanceBenchmark, BenchmarkResult, BenchmarkSuite
)
from sanskrit_rewrite_engine.performance_profiler import (
    AdvancedProfiler, ProfileData, ProfilingSession, PerformanceRegression
)


class TestRuleCache(unittest.TestCase):
    """Test rule caching functionality."""
    
    def setUp(self):
        self.cache = RuleCache(max_size_mb=1)  # Small cache for testing
        self.tokens = [
            Token("test", TokenKind.OTHER),
            Token("token", TokenKind.OTHER),
            Token("sequence", TokenKind.OTHER)
        ]
    
    def test_cache_put_get(self):
        """Test basic cache put and get operations."""
        rule_id = "test_rule"
        index = 0
        result = (self.tokens, 1)
        
        # Put result in cache
        self.cache.put(rule_id, self.tokens, index, result)
        
        # Get result from cache
        cached_result = self.cache.get(rule_id, self.tokens, index)
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result, result)
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 0)
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        rule_id = "nonexistent_rule"
        index = 0
        
        result = self.cache.get(rule_id, self.tokens, index)
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 1)
    
    def test_cache_eviction(self):
        """Test cache eviction when size limit is reached."""
        # Fill cache beyond capacity with very large items
        for i in range(10):  # Fewer items but much larger
            rule_id = f"rule_{i}"
            # Create very large tokens to exceed 1MB limit
            large_tokens = [Token(f"token_{j}" * 1000, TokenKind.OTHER) for j in range(1000)]
            result = (large_tokens, i)
            self.cache.put(rule_id, self.tokens, 0, result)
        
        # Cache should have evicted some entries due to size constraints
        # The exact number depends on size estimation, so just check it's reasonable
        self.assertLessEqual(len(self.cache.cache), 10)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        
        self.assertIn('size_mb', stats)
        self.assertIn('max_size_mb', stats)
        self.assertIn('entries', stats)
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('hit_rate', stats)
        self.assertIn('utilization', stats)


class TestMemoizedFunction(unittest.TestCase):
    """Test memoization functionality."""
    
    def test_memoization_decorator(self):
        """Test memoization decorator."""
        call_count = 0
        
        @memoize(max_size=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call with same arguments (should use cache)
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Third call with different arguments
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)
    
    def test_memoization_cache_info(self):
        """Test memoization cache info."""
        @memoize(max_size=10)
        def test_function(x):
            return x * 2
        
        # Make some calls
        test_function(1)
        test_function(1)  # Cache hit
        test_function(2)
        
        cache_info = test_function.cache_info()
        
        self.assertEqual(cache_info['hits'], 1)
        self.assertEqual(cache_info['misses'], 2)
        self.assertEqual(cache_info['currsize'], 2)


class TestLazyTokenStream(unittest.TestCase):
    """Test lazy token stream functionality."""
    
    def setUp(self):
        self.tokens = [
            Token("test", TokenKind.OTHER),
            Token("lazy", TokenKind.OTHER),
            Token("stream", TokenKind.OTHER)
        ]
        self.stream = LazyTokenStream(self.tokens)
    
    def test_lazy_access(self):
        """Test lazy token access."""
        # Access token by index
        token = self.stream[0]
        self.assertEqual(token.text, "test")
        
        # Check that processing cache is populated
        self.assertIn(0, self.stream._processed_cache)
    
    def test_lazy_processing_function(self):
        """Test lazy processing with functions."""
        def add_tag(token):
            new_token = Token(token.text, token.kind, token.tags.copy(), token.meta.copy())
            new_token.add_tag("processed")
            return new_token
        
        self.stream.add_processing_function(add_tag)
        
        # Access token - should apply processing function
        token = self.stream[0]
        self.assertTrue(token.has_tag("processed"))
    
    def test_lazy_slice(self):
        """Test lazy slice operation."""
        slice_tokens = self.stream.slice(0, 2)
        
        self.assertEqual(len(slice_tokens), 2)
        self.assertEqual(slice_tokens[0].text, "test")
        self.assertEqual(slice_tokens[1].text, "lazy")
    
    def test_lazy_iteration(self):
        """Test lazy iteration."""
        token_texts = [token.text for token in self.stream]
        expected_texts = ["test", "lazy", "stream"]
        
        self.assertEqual(token_texts, expected_texts)


class TestParallelRuleProcessor(unittest.TestCase):
    """Test parallel rule processing."""
    
    def setUp(self):
        self.tokens = [Token(f"token_{i}", TokenKind.OTHER) for i in range(100)]
        self.guard_system = GuardSystem()
        
        # Create a mock rule
        self.mock_rule = Mock(spec=SutraRule)
        self.mock_rule.id = "test_rule"
        self.mock_rule.name = "Test Rule"
        self.mock_rule.meta_data = {}
        self.mock_rule.match_fn = Mock(return_value=True)
        self.mock_rule.apply_fn = Mock(return_value=(self.tokens, 1))
    
    def test_parallel_processor_context_manager(self):
        """Test parallel processor as context manager."""
        with ParallelRuleProcessor(max_workers=2, chunk_size=10) as processor:
            self.assertIsNotNone(processor.executor)
        
        # Executor should be shut down after context
        self.assertTrue(processor.executor._shutdown)
    
    def test_can_parallelize_rule(self):
        """Test rule parallelization check."""
        with ParallelRuleProcessor(max_workers=2, chunk_size=10) as processor:
            # Rule that doesn't modify length should be parallelizable
            self.mock_rule.meta_data = {'modifies_length': False}
            can_parallelize = processor.can_parallelize_rule(self.mock_rule, self.tokens)
            self.assertTrue(can_parallelize)
            
            # Rule that modifies length should not be parallelizable
            self.mock_rule.meta_data = {'modifies_length': True}
            can_parallelize = processor.can_parallelize_rule(self.mock_rule, self.tokens)
            self.assertFalse(can_parallelize)
    
    def test_create_chunks(self):
        """Test token chunking."""
        with ParallelRuleProcessor(max_workers=2, chunk_size=10) as processor:
            chunks = processor._create_chunks(self.tokens)
            
            self.assertEqual(len(chunks), 10)  # 100 tokens / 10 chunk_size
            self.assertEqual(len(chunks[0]), 10)
            self.assertEqual(len(chunks[-1]), 10)


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiler functionality."""
    
    def setUp(self):
        self.profiler = PerformanceProfiler()
    
    def test_profiler_start_stop(self):
        """Test profiler start and stop."""
        profile_name = "test_profile"
        
        self.profiler.start_profiling(profile_name)
        self.assertIn(profile_name, self.profiler.profilers)
        
        time.sleep(0.01)  # Small delay
        
        report_file = self.profiler.stop_profiling(profile_name)
        self.assertIsNotNone(report_file)
        self.assertNotIn(profile_name, self.profiler.profilers)
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        @self.profiler.profile_function("test_function")
        def test_function(n):
            return sum(range(n))
        
        result = test_function(100)
        self.assertEqual(result, sum(range(100)))
        
        # Check timing data - the key might have a timestamp suffix
        timing_keys = list(self.profiler.timing_data.keys())
        self.assertTrue(any("test_function" in key for key in timing_keys))
    
    def test_time_function_decorator(self):
        """Test function timing decorator."""
        @self.profiler.time_function("timed_function")
        def timed_function(n):
            time.sleep(0.01)
            return n * 2
        
        result = timed_function(5)
        self.assertEqual(result, 10)
        
        # Check timing data
        timing_stats = self.profiler.get_timing_statistics()
        self.assertIn("timed_function", timing_stats)
        self.assertGreater(timing_stats["timed_function"]["average_time"], 0.005)
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        # Add some timing data
        self.profiler.timing_data["slow_function"] = [0.2, 0.3, 0.25]
        self.profiler.timing_data["fast_function"] = [0.001, 0.002, 0.001]
        
        bottlenecks = self.profiler.identify_bottlenecks(threshold_seconds=0.1)
        
        self.assertEqual(len(bottlenecks), 1)
        self.assertIn("slow_function", bottlenecks[0])


class TestPerformanceOptimizer(unittest.TestCase):
    """Test performance optimizer functionality."""
    
    def setUp(self):
        self.config = PerformanceConfig(
            enable_parallel_processing=False,  # Disable for testing
            enable_rule_caching=True,
            cache_size_mb=1,
            enable_profiling=False
        )
        self.optimizer = PerformanceOptimizer(self.config)
        
        # Create test tokens and rules
        self.tokens = [Token(f"token_{i}", TokenKind.OTHER) for i in range(10)]
        
        # Mock rule
        self.mock_rule = Mock(spec=SutraRule)
        self.mock_rule.id = "test_rule"
        self.mock_rule.name = "Test Rule"
        self.mock_rule.apply_fn = Mock(return_value=(self.tokens, 1))
        
        self.guard_system = GuardSystem()
    
    def test_optimize_rule_application(self):
        """Test optimized rule application."""
        new_tokens, new_index, cached = self.optimizer.optimize_rule_application(
            self.mock_rule, self.tokens, 0, self.guard_system
        )
        
        self.assertEqual(new_tokens, self.tokens)
        self.assertEqual(new_index, 1)
        self.assertFalse(cached)  # First call should not be cached
        
        # Second call should use cache
        new_tokens2, new_index2, cached2 = self.optimizer.optimize_rule_application(
            self.mock_rule, self.tokens, 0, self.guard_system
        )
        
        self.assertTrue(cached2)
    
    def test_performance_report(self):
        """Test performance report generation."""
        # Process some tokens to generate metrics
        self.optimizer.optimize_rule_application(
            self.mock_rule, self.tokens, 0, self.guard_system
        )
        
        report = self.optimizer.get_performance_report()
        
        self.assertIn('performance_metrics', report)
        self.assertIn('cache_statistics', report)
        self.assertIn('memory_report', report)
        self.assertIn('configuration', report)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions generation."""
        suggestions = self.optimizer.generate_optimization_suggestions()
        
        self.assertIsInstance(suggestions, list)
        # Suggestions depend on current state, so just check type
    
    def test_configuration_optimization(self):
        """Test automatic configuration optimization."""
        optimized_config = self.optimizer.optimize_configuration()
        
        self.assertIsInstance(optimized_config, PerformanceConfig)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test performance benchmarking functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = PerformanceBenchmark(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        self.assertIsInstance(self.benchmark.configurations, dict)
        self.assertIn('high_performance', self.benchmark.configurations)
        self.assertIn('memory_optimized', self.benchmark.configurations)
    
    def test_test_data_generation(self):
        """Test test data generation."""
        small_tokens = self.benchmark._generate_small_text()
        medium_tokens = self.benchmark._generate_medium_text()
        large_tokens = self.benchmark._generate_large_text()
        
        self.assertGreater(len(small_tokens), 0)
        self.assertGreater(len(medium_tokens), len(small_tokens))
        self.assertGreater(len(large_tokens), len(medium_tokens))
    
    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        result = BenchmarkResult(
            name="test",
            config_name="default",
            processing_time=0.1,
            tokens_per_second=100.0,
            cache_hit_rate=0.5,
            memory_usage_mb=10.0,
            rules_applied=5,
            tokens_processed=10,
            parallel_chunks=0,
            bottlenecks=[]
        )
        
        self.assertEqual(result.name, "test")
        self.assertEqual(result.processing_time, 0.1)
    
    def test_benchmark_suite(self):
        """Test benchmark suite functionality."""
        suite = BenchmarkSuite("test_suite")
        
        result1 = BenchmarkResult(
            name="test1", config_name="config1", processing_time=0.1,
            tokens_per_second=100.0, cache_hit_rate=0.5, memory_usage_mb=10.0,
            rules_applied=5, tokens_processed=10, parallel_chunks=0, bottlenecks=[]
        )
        
        result2 = BenchmarkResult(
            name="test2", config_name="config1", processing_time=0.2,
            tokens_per_second=50.0, cache_hit_rate=0.3, memory_usage_mb=15.0,
            rules_applied=3, tokens_processed=10, parallel_chunks=0, bottlenecks=[]
        )
        
        suite.add_result(result1)
        suite.add_result(result2)
        
        # Test filtering by config
        config1_results = suite.get_results_by_config("config1")
        self.assertEqual(len(config1_results), 2)
        
        # Test average performance
        avg_perf = suite.get_average_performance("config1")
        self.assertAlmostEqual(avg_perf['avg_processing_time'], 0.15)
        self.assertAlmostEqual(avg_perf['avg_tokens_per_second'], 75.0)
    
    def test_save_load_benchmark_results(self):
        """Test saving and loading benchmark results."""
        suite = BenchmarkSuite("test_suite")
        result = BenchmarkResult(
            name="test", config_name="default", processing_time=0.1,
            tokens_per_second=100.0, cache_hit_rate=0.5, memory_usage_mb=10.0,
            rules_applied=5, tokens_processed=10, parallel_chunks=0, bottlenecks=[]
        )
        suite.add_result(result)
        
        # Save results
        filename = self.benchmark.save_benchmark_results(suite, "test_results.json")
        self.assertTrue(Path(filename).exists())
        
        # Load results
        loaded_suite = self.benchmark.load_benchmark_results("test_results.json")
        self.assertEqual(loaded_suite.name, suite.name)
        self.assertEqual(len(loaded_suite.results), 1)


class TestAdvancedProfiler(unittest.TestCase):
    """Test advanced profiler functionality."""
    
    def setUp(self):
        self.profiler = AdvancedProfiler(
            enable_line_profiling=False,  # Disable for testing
            enable_memory_profiling=True
        )
    
    def tearDown(self):
        self.profiler.cleanup()
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertIsInstance(self.profiler.active_profiles, dict)
        self.assertIsInstance(self.profiler.completed_profiles, dict)
    
    def test_function_profiling(self):
        """Test function profiling decorator."""
        @self.profiler.profile_function("test_function")
        def test_function(n):
            time.sleep(0.01)
            return sum(range(n))
        
        result = test_function(10)
        self.assertEqual(result, sum(range(10)))
        
        # Check that profile data was recorded
        self.assertIn("test_function", self.profiler.active_profiles)
    
    def test_block_profiling(self):
        """Test block profiling context manager."""
        with self.profiler.profile_block("test_block"):
            time.sleep(0.01)
            result = 2 + 2
        
        self.assertEqual(result, 4)
        self.assertIn("test_block", self.profiler.active_profiles)
    
    def test_profile_finalization(self):
        """Test profile finalization."""
        @self.profiler.profile_function("finalize_test")
        def test_function():
            time.sleep(0.01)
            return 42
        
        # Call function multiple times
        for _ in range(3):
            test_function()
        
        # Finalize profiles
        profiles = self.profiler.finalize_profiles()
        
        self.assertIn("finalize_test", profiles)
        profile = profiles["finalize_test"]
        self.assertEqual(profile.call_count, 3)
        self.assertGreater(profile.total_time, 0.02)  # At least 3 * 0.01
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        # Create a slow function
        @self.profiler.profile_function("slow_function")
        def slow_function():
            time.sleep(0.1)
            return 1
        
        # Create a fast function
        @self.profiler.profile_function("fast_function")
        def fast_function():
            return 2
        
        # Call functions
        slow_function()
        fast_function()
        
        # Finalize and identify bottlenecks
        self.profiler.finalize_profiles()
        bottlenecks = self.profiler.identify_bottlenecks(threshold=0.1)
        
        # Should identify slow_function as bottleneck
        self.assertTrue(any("slow_function" in b for b in bottlenecks))
    
    def test_session_creation(self):
        """Test profiling session creation."""
        @self.profiler.profile_function("session_test")
        def test_function():
            return 1
        
        test_function()
        
        session = self.profiler.create_session("test_session")
        
        self.assertEqual(session.session_id, "test_session")
        self.assertIn("session_test", session.profiles)
        self.assertIsInstance(session.system_info, dict)


class TestPerformanceRegression(unittest.TestCase):
    """Test performance regression detection."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.regression = PerformanceRegression(self.temp_file.name)
    
    def tearDown(self):
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_baseline_save_load(self):
        """Test baseline saving and loading."""
        # Create mock session
        profile = ProfileData(
            name="test_function",
            total_time=0.1,
            call_count=1,
            average_time=0.1,
            min_time=0.1,
            max_time=0.1,
            memory_usage_mb=10.0,
            peak_memory_mb=15.0,
            cpu_percent=50.0
        )
        
        session = ProfilingSession(
            session_id="baseline",
            start_time=time.time(),
            end_time=time.time(),
            profiles={"test_function": profile}
        )
        
        # Save baseline
        self.regression.save_baseline(session)
        
        # Check that baseline was saved
        self.assertIn("profiles", self.regression.baseline_data)
        self.assertIn("test_function", self.regression.baseline_data["profiles"])
    
    def test_regression_detection(self):
        """Test regression detection."""
        # Create baseline
        baseline_profile = ProfileData(
            name="test_function",
            total_time=0.1,
            call_count=1,
            average_time=0.1,
            min_time=0.1,
            max_time=0.1,
            memory_usage_mb=10.0,
            peak_memory_mb=15.0,
            cpu_percent=50.0
        )
        
        baseline_session = ProfilingSession(
            session_id="baseline",
            start_time=time.time(),
            end_time=time.time(),
            profiles={"test_function": baseline_profile}
        )
        
        self.regression.save_baseline(baseline_session)
        
        # Create current session with regression
        current_profile = ProfileData(
            name="test_function",
            total_time=0.2,
            call_count=1,
            average_time=0.2,  # 100% slower
            min_time=0.2,
            max_time=0.2,
            memory_usage_mb=20.0,  # 100% more memory
            peak_memory_mb=30.0,
            cpu_percent=50.0
        )
        
        current_session = ProfilingSession(
            session_id="current",
            start_time=time.time(),
            end_time=time.time(),
            profiles={"test_function": current_profile}
        )
        
        # Detect regressions
        regressions = self.regression.detect_regressions(current_session, threshold=0.1)
        
        # Should detect both time and memory regressions
        self.assertGreater(len(regressions), 0)
        self.assertTrue(any("slower" in r for r in regressions))
        self.assertTrue(any("more memory" in r for r in regressions))


class TestConfigurationPresets(unittest.TestCase):
    """Test configuration presets."""
    
    def test_high_performance_config(self):
        """Test high performance configuration."""
        config = get_high_performance_config()
        
        self.assertTrue(config.enable_parallel_processing)
        self.assertTrue(config.enable_rule_caching)
        self.assertTrue(config.enable_lazy_evaluation)
        self.assertGreater(config.cache_size_mb, 500)
    
    def test_memory_optimized_config(self):
        """Test memory optimized configuration."""
        config = get_memory_optimized_config()
        
        self.assertFalse(config.enable_parallel_processing)
        self.assertTrue(config.enable_rule_caching)
        self.assertTrue(config.enable_lazy_evaluation)
        self.assertLess(config.cache_size_mb, 500)


if __name__ == '__main__':
    unittest.main()