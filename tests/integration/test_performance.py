"""
Performance tests for the Sanskrit Rewrite Engine.

This module tests processing speed, memory usage, and scalability
of the Sanskrit text processing system.
"""

import pytest
import time
import psutil
import os
import threading
import concurrent.futures
from typing import List, Dict, Any
from unittest.mock import patch

from src.sanskrit_rewrite_engine.engine import SanskritRewriteEngine, TransformationResult
from src.sanskrit_rewrite_engine.rules import Rule, RuleRegistry
from src.sanskrit_rewrite_engine.tokenizer import BasicSanskritTokenizer
from tests.fixtures.sample_texts import (
    BASIC_WORDS, MORPHOLOGICAL_EXAMPLES, SANDHI_EXAMPLES, COMPLEX_SENTENCES
)


@pytest.mark.performance
class TestEnginePerformance:
    """Test engine performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SanskritRewriteEngine()
        
    def test_simple_text_processing_speed(self):
        """Test processing speed for simple text."""
        text = "rāma + iti"
        
        start_time = time.time()
        result = self.engine.process(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result.success == True
        # Should process simple text very quickly (< 100ms)
        assert processing_time < 0.1
        
    def test_medium_text_processing_speed(self):
        """Test processing speed for medium-length text."""
        text = " ".join(MORPHOLOGICAL_EXAMPLES * 10)  # ~50 words
        
        start_time = time.time()
        result = self.engine.process(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result.success == True
        # Should process medium text reasonably quickly (< 1s)
        assert processing_time < 1.0
        
    def test_large_text_processing_speed(self):
        """Test processing speed for large text."""
        text = " ".join(COMPLEX_SENTENCES * 100)  # ~400 words
        
        start_time = time.time()
        result = self.engine.process(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result.success == True
        # Should process large text within reasonable time (< 5s)
        assert processing_time < 5.0
        
    def test_repeated_processing_performance(self):
        """Test performance of repeated processing."""
        text = "rāma + iti"
        iterations = 100
        
        start_time = time.time()
        
        for _ in range(iterations):
            result = self.engine.process(text)
            assert result.success == True
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Average processing time should be very fast
        assert avg_time < 0.01  # < 10ms per iteration
        
    def test_memory_usage_simple_text(self):
        """Test memory usage for simple text processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process text multiple times
        for _ in range(50):
            result = self.engine.process("rāma + iti")
            assert result.success == True
            
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 10MB)
        assert memory_increase < 10 * 1024 * 1024
        
    def test_memory_usage_large_text(self):
        """Test memory usage for large text processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large text
        large_text = "rāma + iti " * 1000
        result = self.engine.process(large_text)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.success == True
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024
        
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent load."""
        def process_text(text_id):
            text = f"rāma + iti {text_id}"
            start_time = time.time()
            result = self.engine.process(text)
            end_time = time.time()
            return result.success, end_time - start_time
            
        # Test with multiple threads
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_text, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        # All should succeed
        successes = [success for success, _ in results]
        times = [time_taken for _, time_taken in results]
        
        assert all(successes)
        # Even under concurrent load, individual requests should be fast
        assert max(times) < 1.0
        assert sum(times) / len(times) < 0.5  # Average < 500ms
        
    def test_scaling_with_text_length(self):
        """Test how processing time scales with text length."""
        base_text = "rāma + iti "
        lengths = [10, 50, 100, 500, 1000]
        times = []
        
        for length in lengths:
            text = base_text * length
            
            start_time = time.time()
            result = self.engine.process(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            assert result.success == True
            
        # Processing time should scale reasonably (not exponentially)
        # Check that 10x text doesn't take more than 20x time
        ratio_1000_to_10 = times[-1] / times[0]
        assert ratio_1000_to_10 < 200  # Should be much less than 200x
        
    def test_rule_count_impact_on_performance(self):
        """Test impact of rule count on processing performance."""
        # Test with minimal rules
        minimal_engine = SanskritRewriteEngine()
        minimal_engine.clear_rules()
        
        start_time = time.time()
        result1 = minimal_engine.process("rāma + iti")
        time_minimal = time.time() - start_time
        
        # Test with many rules
        many_rules_engine = SanskritRewriteEngine()
        
        # Add many dummy rules
        for i in range(100):
            rule = Rule(
                id=f"dummy_rule_{i}",
                name=f"Dummy Rule {i}",
                description="A dummy rule for testing",
                pattern=f"dummy{i}",
                replacement=f"DUMMY{i}",
                priority=i + 10
            )
            many_rules_engine.add_rule(rule)
            
        start_time = time.time()
        result2 = many_rules_engine.process("rāma + iti")
        time_many_rules = time.time() - start_time
        
        assert result1.success == True
        assert result2.success == True
        
        # Many rules should not dramatically slow down processing
        # (since most won't match the input text)
        slowdown_factor = time_many_rules / time_minimal if time_minimal > 0 else 1
        assert slowdown_factor < 10  # Should not be more than 10x slower
        
    def test_tracing_performance_impact(self):
        """Test performance impact of tracing."""
        text = "rāma + iti dharma + artha"
        
        # Test without tracing
        start_time = time.time()
        result_no_trace = self.engine.process(text, enable_tracing=False)
        time_no_trace = time.time() - start_time
        
        # Test with tracing
        start_time = time.time()
        result_with_trace = self.engine.process(text, enable_tracing=True)
        time_with_trace = time.time() - start_time
        
        assert result_no_trace.success == True
        assert result_with_trace.success == True
        
        # Tracing should not dramatically slow down processing
        if time_no_trace > 0:
            slowdown_factor = time_with_trace / time_no_trace
            assert slowdown_factor < 5  # Should not be more than 5x slower
            
    def test_iteration_limit_performance(self):
        """Test performance with different iteration limits."""
        text = "rāma + iti"
        iteration_limits = [1, 5, 10, 20, 50]
        times = []
        
        for limit in iteration_limits:
            config = {'max_iterations': limit}
            engine = SanskritRewriteEngine(config=config)
            
            start_time = time.time()
            result = engine.process(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            assert result.success == True
            
        # Higher iteration limits should not dramatically increase time
        # for simple text that converges quickly
        max_time = max(times)
        min_time = min(times)
        
        if min_time > 0:
            ratio = max_time / min_time
            assert ratio < 10  # Should not be more than 10x difference


@pytest.mark.performance
class TestTokenizerPerformance:
    """Test tokenizer performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = BasicSanskritTokenizer()
        
    def test_tokenization_speed(self):
        """Test tokenization speed."""
        text = " ".join(BASIC_WORDS * 100)  # ~800 characters
        
        start_time = time.time()
        tokens = self.tokenizer.tokenize(text)
        end_time = time.time()
        
        tokenization_time = end_time - start_time
        
        assert len(tokens) > 0
        # Should tokenize quickly (< 100ms)
        assert tokenization_time < 0.1
        
    def test_tokenization_scaling(self):
        """Test how tokenization scales with text length."""
        base_text = " ".join(BASIC_WORDS)
        lengths = [1, 10, 50, 100, 500]
        times = []
        
        for length in lengths:
            text = (base_text + " ") * length
            
            start_time = time.time()
            tokens = self.tokenizer.tokenize(text)
            end_time = time.time()
            
            tokenization_time = end_time - start_time
            times.append(tokenization_time)
            
            assert len(tokens) > 0
            
        # Should scale roughly linearly
        ratio_500_to_1 = times[-1] / times[0] if times[0] > 0 else 1
        assert ratio_500_to_1 < 1000  # Should be much less than 1000x
        
    def test_tokenization_memory_usage(self):
        """Test tokenization memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Tokenize large text
        large_text = " ".join(BASIC_WORDS * 1000)
        tokens = self.tokenizer.tokenize(large_text)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert len(tokens) > 0
        # Memory increase should be reasonable (< 20MB)
        assert memory_increase < 20 * 1024 * 1024


@pytest.mark.performance
class TestRuleRegistryPerformance:
    """Test rule registry performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = RuleRegistry()
        
    def test_rule_loading_performance(self):
        """Test rule loading performance."""
        # Create a large rule set
        rules_data = {
            "rule_set": "performance_test",
            "version": "1.0",
            "rules": []
        }
        
        # Add many rules
        for i in range(1000):
            rules_data["rules"].append({
                "id": f"rule_{i}",
                "name": f"Rule {i}",
                "description": f"Test rule {i}",
                "pattern": f"pattern{i}",
                "replacement": f"replacement{i}",
                "priority": i % 10,
                "enabled": True
            })
            
        # Write to temporary file
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rules_data, f)
            rule_file = f.name
            
        try:
            start_time = time.time()
            self.registry.load_from_json(rule_file)
            end_time = time.time()
            
            loading_time = end_time - start_time
            
            assert self.registry.get_rule_count() == 1000
            # Should load rules quickly (< 1s)
            assert loading_time < 1.0
            
        finally:
            os.unlink(rule_file)
            
    def test_rule_matching_performance(self):
        """Test rule matching performance."""
        # Add many rules
        for i in range(100):
            rule = Rule(
                id=f"rule_{i}",
                name=f"Rule {i}",
                description=f"Test rule {i}",
                pattern=f"test{i}",
                replacement=f"TEST{i}",
                priority=i
            )
            self.registry.add_rule(rule)
            
        text = "test50 some other text"
        
        start_time = time.time()
        applicable_rules = self.registry.get_applicable_rules(text, 0)
        end_time = time.time()
        
        matching_time = end_time - start_time
        
        assert len(applicable_rules) > 0
        # Should find applicable rules quickly (< 10ms)
        assert matching_time < 0.01
        
    def test_rule_priority_sorting_performance(self):
        """Test rule priority sorting performance."""
        # Add many rules with random priorities
        import random
        
        for i in range(1000):
            rule = Rule(
                id=f"rule_{i}",
                name=f"Rule {i}",
                description=f"Test rule {i}",
                pattern=f"test{i}",
                replacement=f"TEST{i}",
                priority=random.randint(1, 100)
            )
            self.registry.add_rule(rule)
            
        start_time = time.time()
        sorted_rules = self.registry.get_rules_by_priority()
        end_time = time.time()
        
        sorting_time = end_time - start_time
        
        assert len(sorted_rules) == 1000
        # Should sort rules quickly (< 100ms)
        assert sorting_time < 0.1
        
        # Verify sorting is correct
        priorities = [rule.priority for rule in sorted_rules]
        assert priorities == sorted(priorities)


@pytest.mark.performance
@pytest.mark.slow
class TestSystemPerformance:
    """Test overall system performance characteristics."""
    
    def test_end_to_end_performance(self):
        """Test end-to-end system performance."""
        engine = SanskritRewriteEngine()
        
        # Test with various realistic workloads
        workloads = [
            ("Simple", "rāma + iti"),
            ("Medium", " ".join(MORPHOLOGICAL_EXAMPLES)),
            ("Complex", " ".join(COMPLEX_SENTENCES)),
            ("Large", " ".join(SANDHI_EXAMPLES * 50))
        ]
        
        for workload_name, text in workloads:
            start_time = time.time()
            result = engine.process(text, enable_tracing=True)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert result.success == True
            
            # Performance targets based on workload
            if workload_name == "Simple":
                assert processing_time < 0.1  # 100ms
            elif workload_name == "Medium":
                assert processing_time < 0.5  # 500ms
            elif workload_name == "Complex":
                assert processing_time < 1.0  # 1s
            elif workload_name == "Large":
                assert processing_time < 5.0  # 5s
                
    def test_stress_test(self):
        """Test system under stress conditions."""
        engine = SanskritRewriteEngine()
        
        # Process many texts concurrently
        def stress_worker(worker_id):
            results = []
            for i in range(10):
                text = f"rāma + iti {worker_id}_{i}"
                start_time = time.time()
                result = engine.process(text)
                end_time = time.time()
                
                results.append({
                    'success': result.success,
                    'time': end_time - start_time,
                    'worker_id': worker_id,
                    'iteration': i
                })
            return results
            
        # Run stress test with multiple workers
        num_workers = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
                
        # Analyze results
        successes = [r['success'] for r in all_results]
        times = [r['time'] for r in all_results]
        
        # All should succeed
        assert all(successes)
        
        # Performance should remain reasonable under load
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 1.0  # Average < 1s
        assert max_time < 5.0   # Max < 5s
        
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        engine = SanskritRewriteEngine()
        process = psutil.Process(os.getpid())
        
        # Record initial memory
        initial_memory = process.memory_info().rss
        
        # Process many texts
        for i in range(100):
            text = f"rāma + iti iteration {i}"
            result = engine.process(text)
            assert result.success == True
            
            # Check memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow excessively
                # Allow some growth but not more than 100MB
                assert memory_increase < 100 * 1024 * 1024
                
        # Final memory check
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_memory_increase < 50 * 1024 * 1024  # < 50MB
        
    def test_performance_regression(self):
        """Test for performance regression."""
        engine = SanskritRewriteEngine()
        
        # Baseline performance test
        baseline_text = "rāma + iti dharma + artha"
        baseline_times = []
        
        # Run multiple times to get stable baseline
        for _ in range(10):
            start_time = time.time()
            result = engine.process(baseline_text)
            end_time = time.time()
            
            assert result.success == True
            baseline_times.append(end_time - start_time)
            
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        
        # Performance should be consistently good
        # (This is a placeholder - in real scenarios, you'd compare against
        # historical performance data)
        assert avg_baseline_time < 0.5  # Should average < 500ms
        
        # Check for consistency (low variance)
        max_time = max(baseline_times)
        min_time = min(baseline_times)
        
        if min_time > 0:
            variance_ratio = max_time / min_time
            assert variance_ratio < 10  # Should not vary by more than 10x


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for comparison and monitoring."""
    
    def test_throughput_benchmark(self):
        """Benchmark processing throughput."""
        engine = SanskritRewriteEngine()
        
        # Test texts of different sizes
        test_cases = [
            ("tiny", "rāma"),
            ("small", "rāma + iti"),
            ("medium", " ".join(MORPHOLOGICAL_EXAMPLES[:5])),
            ("large", " ".join(COMPLEX_SENTENCES))
        ]
        
        results = {}
        
        for size_name, text in test_cases:
            # Process multiple times to get stable measurements
            times = []
            for _ in range(20):
                start_time = time.time()
                result = engine.process(text)
                end_time = time.time()
                
                assert result.success == True
                times.append(end_time - start_time)
                
            avg_time = sum(times) / len(times)
            chars_per_second = len(text) / avg_time if avg_time > 0 else 0
            
            results[size_name] = {
                'avg_time': avg_time,
                'chars_per_second': chars_per_second,
                'text_length': len(text)
            }
            
        # Print benchmark results for monitoring
        print("\nPerformance Benchmark Results:")
        print("-" * 50)
        for size_name, metrics in results.items():
            print(f"{size_name:8s}: {metrics['avg_time']:.4f}s avg, "
                  f"{metrics['chars_per_second']:.0f} chars/sec, "
                  f"{metrics['text_length']} chars")
            
        # Basic performance assertions
        assert results['tiny']['chars_per_second'] > 1000  # > 1000 chars/sec
        assert results['small']['chars_per_second'] > 500   # > 500 chars/sec
        
    def test_scalability_benchmark(self):
        """Benchmark system scalability."""
        engine = SanskritRewriteEngine()
        
        # Test with increasing load
        load_levels = [1, 2, 5, 10]
        results = {}
        
        for num_concurrent in load_levels:
            def worker():
                start_time = time.time()
                result = engine.process("rāma + iti dharma + artha")
                end_time = time.time()
                return result.success, end_time - start_time
                
            # Run concurrent workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(worker) for _ in range(num_concurrent)]
                worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
            successes = [success for success, _ in worker_results]
            times = [time_taken for _, time_taken in worker_results]
            
            results[num_concurrent] = {
                'success_rate': sum(successes) / len(successes),
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
                'throughput': num_concurrent / sum(times) if sum(times) > 0 else 0
            }
            
        # Print scalability results
        print("\nScalability Benchmark Results:")
        print("-" * 60)
        print("Concurrent | Success Rate | Avg Time | Max Time | Throughput")
        print("-" * 60)
        for num_concurrent, metrics in results.items():
            print(f"{num_concurrent:10d} | {metrics['success_rate']:11.2%} | "
                  f"{metrics['avg_time']:8.3f}s | {metrics['max_time']:8.3f}s | "
                  f"{metrics['throughput']:10.1f}")
            
        # Scalability assertions
        for metrics in results.values():
            assert metrics['success_rate'] == 1.0  # 100% success rate
            assert metrics['avg_time'] < 2.0       # < 2s average
            assert metrics['max_time'] < 5.0       # < 5s maximum