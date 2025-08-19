"""
Performance profiling and memory leak detection tests.
Part of TO1 comprehensive test suite.

Tests cover:
- Performance profiling and memory leak detection (Requirement 14)
- Stress testing with large Sanskrit corpora
- Memory usage monitoring and optimization
- CPU profiling for bottleneck identification
"""

import pytest
import time
import psutil
import gc
import tracemalloc
import threading
import cProfile
import pstats
from io import StringIO
from typing import List, Dict, Any
import weakref

from sanskrit_rewrite_engine.token import Token, TokenKind
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine


class MemoryProfiler:
    """Advanced memory profiling utility."""
    
    def __init__(self):
        self.snapshots = []
        self.weak_refs = []
    
    def start_monitoring(self):
        """Start memory monitoring."""
        tracemalloc.start()
        gc.collect()
        self.snapshots = []
        self.weak_refs = []
    
    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        memory_info = psutil.Process().memory_info()
        
        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'timestamp': time.time()
        })
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return analysis."""
        if not self.snapshots:
            return {}
        
        tracemalloc.stop()
        
        # Analyze memory growth
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        rss_growth = last_snapshot['rss'] - first_snapshot['rss']
        vms_growth = last_snapshot['vms'] - first_snapshot['vms']
        
        # Get top memory consumers
        top_stats = last_snapshot['snapshot'].statistics('lineno')[:10]
        
        return {
            'rss_growth': rss_growth,
            'vms_growth': vms_growth,
            'peak_rss': max(s['rss'] for s in self.snapshots),
            'peak_vms': max(s['vms'] for s in self.snapshots),
            'snapshot_count': len(self.snapshots),
            'top_memory_consumers': [
                {
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size': stat.size,
                    'count': stat.count
                }
                for stat in top_stats
            ]
        }
    
    def add_weak_reference(self, obj, name: str):
        """Add weak reference to track object lifecycle."""
        self.weak_refs.append({
            'name': name,
            'ref': weakref.ref(obj),
            'created_at': time.time()
        })
    
    def check_weak_references(self) -> List[str]:
        """Check which objects are still alive."""
        alive_objects = []
        for ref_info in self.weak_refs:
            if ref_info['ref']() is not None:
                alive_objects.append(ref_info['name'])
        return alive_objects


class CPUProfiler:
    """CPU profiling utility."""
    
    def __init__(self):
        self.profiler = None
        self.start_time = None
    
    def start_profiling(self):
        """Start CPU profiling."""
        self.profiler = cProfile.Profile()
        self.start_time = time.time()
        self.profiler.enable()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis."""
        if not self.profiler:
            return {}
        
        self.profiler.disable()
        end_time = time.time()
        
        # Get profiling statistics
        stats_stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        # Get function call statistics
        stats.sort_stats('calls')
        calls_stream = StringIO()
        calls_stats = pstats.Stats(self.profiler, stream=calls_stream)
        calls_stats.print_stats(10)
        
        return {
            'total_time': end_time - self.start_time if self.start_time else 0,
            'profile_stats': stats_stream.getvalue(),
            'call_stats': calls_stream.getvalue(),
            'function_count': len(stats.stats) if hasattr(stats, 'stats') else 0
        }


@pytest.fixture
def tokenizer():
    """Fixture for Sanskrit tokenizer."""
    return SanskritTokenizer()


@pytest.fixture
def rule_engine():
    """Fixture for Panini rule engine."""
    return PaniniRuleEngine()


class TestMemoryLeakDetection:
    """Test for memory leaks in the Sanskrit rewrite engine."""
    
    def test_tokenizer_memory_leaks(self, tokenizer):
        """Test tokenizer for memory leaks during repeated use."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("initial")
        
        # Perform many tokenization operations
        test_texts = [
            "rāma gacchati vanaṃ",
            "सत्यमेव जयते",
            "धर्म अर्थ काम मोक्ष",
            "वसुधैव कुटुम्बकम्",
            "अहिंसा परमो धर्मः"
        ]
        
        for cycle in range(20):  # 20 cycles
            for text in test_texts:
                tokens = tokenizer.tokenize(text)
                
                # Add weak references to track token lifecycle
                for i, token in enumerate(tokens):
                    profiler.add_weak_reference(token, f"token_{cycle}_{i}")
                
                # Verify tokens are created properly
                assert len(tokens) > 0
                assert all(isinstance(t, Token) for t in tokens)
            
            # Take periodic snapshots
            if cycle % 5 == 0:
                profiler.take_snapshot(f"cycle_{cycle}")
        
        # Force garbage collection
        gc.collect()
        profiler.take_snapshot("final")
        
        # Analyze memory usage
        analysis = profiler.stop_monitoring()
        
        # Check for memory leaks
        assert analysis['rss_growth'] < 50 * 1024 * 1024, f"Potential memory leak: {analysis['rss_growth']} bytes growth"
        
        # Check that most tokens are garbage collected
        alive_objects = profiler.check_weak_references()
        leak_rate = len(alive_objects) / len(profiler.weak_refs) if profiler.weak_refs else 0
        assert leak_rate < 0.1, f"High object retention rate: {leak_rate}"
    
    def test_rule_engine_memory_leaks(self, rule_engine, tokenizer):
        """Test rule engine for memory leaks during repeated processing."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("initial")
        
        # Perform many rule processing operations
        test_cases = [
            "a+i",
            "rāma+iti",
            "dharma+artha",
            "guru+upadeśa",
            "test:GEN"
        ]
        
        for cycle in range(15):  # 15 cycles
            for text in test_cases:
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=5)
                
                # Add weak references to track result lifecycle
                profiler.add_weak_reference(result, f"result_{cycle}_{text}")
                
                # Verify processing completed
                assert isinstance(result.output_tokens, list)
                assert isinstance(result.traces, list)
            
            # Reset engine state periodically
            if cycle % 5 == 0:
                rule_engine.reset_engine()
                profiler.take_snapshot(f"cycle_{cycle}")
        
        # Force cleanup
        gc.collect()
        profiler.take_snapshot("final")
        
        # Analyze memory usage
        analysis = profiler.stop_monitoring()
        
        # Check for memory leaks
        assert analysis['rss_growth'] < 100 * 1024 * 1024, f"Potential memory leak: {analysis['rss_growth']} bytes growth"
        
        # Check object retention
        alive_objects = profiler.check_weak_references()
        leak_rate = len(alive_objects) / len(profiler.weak_refs) if profiler.weak_refs else 0
        assert leak_rate < 0.2, f"High result retention rate: {leak_rate}"
    
    def test_trace_memory_management(self, rule_engine, tokenizer):
        """Test that trace data doesn't accumulate excessively."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("initial")
        
        # Process texts that generate traces
        for i in range(50):
            text = f"trace test {i} with multiple tokens"
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=10)
            
            # Track trace objects
            for j, trace in enumerate(result.traces):
                profiler.add_weak_reference(trace, f"trace_{i}_{j}")
            
            # Periodically clear references and check memory
            if i % 10 == 0:
                gc.collect()
                profiler.take_snapshot(f"iteration_{i}")
        
        # Final cleanup
        gc.collect()
        profiler.take_snapshot("final")
        
        analysis = profiler.stop_monitoring()
        
        # Trace data should not cause excessive memory growth
        assert analysis['rss_growth'] < 75 * 1024 * 1024, f"Trace memory leak: {analysis['rss_growth']} bytes growth"
    
    def test_custom_rule_memory_management(self, rule_engine):
        """Test memory management when adding/removing custom rules."""
        from sanskrit_rewrite_engine.rule import SutraRule, SutraReference, RuleType
        
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("initial")
        
        # Add many custom rules
        custom_rules = []
        for i in range(100):
            rule = SutraRule(
                sutra_ref=SutraReference(9, 8, i + 1),
                name=f"memory_test_rule_{i}",
                description=f"Memory test rule {i}",
                rule_type=RuleType.SUTRA,
                priority=100 + i,
                match_fn=lambda tokens, idx: False,
                apply_fn=lambda tokens, idx: (tokens, idx)
            )
            
            rule_engine.add_custom_rule(rule)
            custom_rules.append(rule)
            profiler.add_weak_reference(rule, f"rule_{i}")
            
            if i % 20 == 0:
                profiler.take_snapshot(f"added_{i}_rules")
        
        # Clear references to rules
        custom_rules.clear()
        
        # Force garbage collection
        gc.collect()
        profiler.take_snapshot("final")
        
        analysis = profiler.stop_monitoring()
        
        # Rule addition should not cause excessive memory growth
        assert analysis['rss_growth'] < 30 * 1024 * 1024, f"Rule memory leak: {analysis['rss_growth']} bytes growth"


class TestPerformanceProfiling:
    """Test performance characteristics and identify bottlenecks."""
    
    def test_tokenization_performance_profiling(self, tokenizer):
        """Profile tokenization performance to identify bottlenecks."""
        profiler = CPUProfiler()
        
        # Create test data of varying complexity
        simple_texts = ["rama", "gacchati", "vanaṃ"]
        complex_texts = [
            "रामायणमहाकाव्यस्य आदिकविः वाल्मीकिः",
            "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः",
            "क्ष्म्य्व्र्त्न्प्फ्ब्भ्म्य्र्ल्व्श्ष्स्ह्"  # Complex conjuncts
        ]
        
        profiler.start_profiling()
        
        # Profile simple tokenization
        for _ in range(100):
            for text in simple_texts:
                tokens = tokenizer.tokenize(text)
                assert len(tokens) > 0
        
        # Profile complex tokenization
        for _ in range(50):
            for text in complex_texts:
                tokens = tokenizer.tokenize(text)
                assert len(tokens) > 0
        
        analysis = profiler.stop_profiling()
        
        # Should complete in reasonable time
        assert analysis['total_time'] < 5.0, f"Tokenization too slow: {analysis['total_time']} seconds"
        
        # Print profiling information for analysis
        print(f"\nTokenization Performance Profile:")
        print(f"Total time: {analysis['total_time']:.3f} seconds")
        print(f"Functions profiled: {analysis['function_count']}")
    
    def test_rule_processing_performance_profiling(self, rule_engine, tokenizer):
        """Profile rule processing performance to identify bottlenecks."""
        profiler = CPUProfiler()
        
        # Create test cases with different complexity levels
        test_cases = [
            # Simple cases
            ("a", 1),
            ("rama", 3),
            
            # Medium complexity
            ("a+i", 5),
            ("rāma+iti", 10),
            
            # Complex cases
            ("dharma+artha+kāma+mokṣa", 15),
            ("test:GEN:PL:ACC", 20)
        ]
        
        profiler.start_profiling()
        
        for text, max_passes in test_cases:
            # Run multiple iterations for statistical significance
            for _ in range(10):
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=max_passes)
                assert isinstance(result.output_tokens, list)
        
        analysis = profiler.stop_profiling()
        
        # Should complete in reasonable time
        assert analysis['total_time'] < 10.0, f"Rule processing too slow: {analysis['total_time']} seconds"
        
        print(f"\nRule Processing Performance Profile:")
        print(f"Total time: {analysis['total_time']:.3f} seconds")
        print(f"Functions profiled: {analysis['function_count']}")
    
    def test_scalability_profiling(self, rule_engine, tokenizer):
        """Profile scalability with increasing input sizes."""
        profiler = CPUProfiler()
        
        # Test with increasing input sizes
        base_text = "rāma gacchati vanaṃ"
        sizes = [1, 5, 10, 20, 50]
        
        performance_data = []
        
        for size in sizes:
            text = (base_text + " ") * size
            
            profiler.start_profiling()
            
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=5)
            
            analysis = profiler.stop_profiling()
            
            performance_data.append({
                'size': size,
                'token_count': len(tokens),
                'time': analysis['total_time'],
                'converged': result.converged
            })
            
            assert analysis['total_time'] < size * 0.5, f"Poor scalability at size {size}"
        
        # Check that performance scales reasonably
        if len(performance_data) >= 2:
            first_time = performance_data[0]['time']
            last_time = performance_data[-1]['time']
            first_size = performance_data[0]['size']
            last_size = performance_data[-1]['size']
            
            if first_time > 0:
                time_ratio = last_time / first_time
                size_ratio = last_size / first_size
                
                # Should scale sub-quadratically
                assert time_ratio < size_ratio ** 1.5, f"Poor scalability: time ratio {time_ratio}, size ratio {size_ratio}"
        
        print(f"\nScalability Profile:")
        for data in performance_data:
            print(f"Size {data['size']}: {data['time']:.3f}s, {data['token_count']} tokens")
    
    def test_memory_efficiency_profiling(self, rule_engine, tokenizer):
        """Profile memory efficiency during processing."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("start")
        
        # Process texts of increasing complexity
        test_texts = [
            "a",
            "rama",
            "rāma gacchati",
            "dharma artha kāma mokṣa",
            "सत्यमेव जयते नानृतम्",
            "वसुधैव कुटुम्बकम् सर्वे भवन्तु सुखिनः"
        ]
        
        memory_usage = []
        
        for i, text in enumerate(test_texts):
            tokens = tokenizer.tokenize(text)
            result = rule_engine.process(tokens, max_passes=10)
            
            profiler.take_snapshot(f"text_{i}")
            
            # Record memory usage
            current_memory = psutil.Process().memory_info().rss
            memory_usage.append({
                'text_length': len(text),
                'token_count': len(tokens),
                'memory_rss': current_memory,
                'trace_count': len(result.traces)
            })
        
        analysis = profiler.stop_monitoring()
        
        # Memory usage should scale reasonably with input size
        if len(memory_usage) >= 2:
            first_memory = memory_usage[0]['memory_rss']
            last_memory = memory_usage[-1]['memory_rss']
            memory_growth = last_memory - first_memory
            
            # Should not use excessive memory for small inputs
            assert memory_growth < 50 * 1024 * 1024, f"Excessive memory growth: {memory_growth} bytes"
        
        print(f"\nMemory Efficiency Profile:")
        print(f"Total memory growth: {analysis['rss_growth']} bytes")
        print(f"Peak RSS: {analysis['peak_rss']} bytes")


class TestStressTestingLargeCorpora:
    """Stress testing with large Sanskrit corpora simulation."""
    
    def test_mahabharat_scale_processing(self, rule_engine, tokenizer):
        """Test processing at Mahābhārata scale (simulated)."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("start")
        
        # Simulate large Sanskrit text processing
        # Mahābhārata has ~100,000 verses, we'll simulate a smaller portion
        base_verses = [
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
            "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय",
            "दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा",
            "आचार्यमुपसंगम्य राजा वचनमब्रवीत्"
        ]
        
        # Process 1000 verses (simulated)
        processed_count = 0
        start_time = time.time()
        
        for cycle in range(250):  # 250 cycles × 4 verses = 1000 verses
            for verse in base_verses:
                tokens = tokenizer.tokenize(verse)
                result = rule_engine.process(tokens, max_passes=3)  # Limited passes for performance
                
                processed_count += 1
                assert isinstance(result.output_tokens, list)
                
                # Periodic memory check
                if processed_count % 100 == 0:
                    profiler.take_snapshot(f"verse_{processed_count}")
                    
                    # Check memory usage doesn't grow excessively
                    current_memory = psutil.Process().memory_info().rss
                    if processed_count > 100:
                        # Memory should not grow linearly with processed count
                        memory_per_verse = current_memory / processed_count
                        assert memory_per_verse < 1024 * 1024, f"Excessive memory per verse: {memory_per_verse} bytes"
        
        end_time = time.time()
        total_time = end_time - start_time
        
        profiler.take_snapshot("end")
        analysis = profiler.stop_monitoring()
        
        # Performance assertions
        assert total_time < 60.0, f"Too slow for large corpus: {total_time} seconds"
        assert analysis['rss_growth'] < 200 * 1024 * 1024, f"Excessive memory growth: {analysis['rss_growth']} bytes"
        
        # Calculate throughput
        verses_per_second = processed_count / total_time
        assert verses_per_second > 10, f"Low throughput: {verses_per_second} verses/second"
        
        print(f"\nLarge Corpus Processing Results:")
        print(f"Processed {processed_count} verses in {total_time:.2f} seconds")
        print(f"Throughput: {verses_per_second:.2f} verses/second")
        print(f"Memory growth: {analysis['rss_growth']} bytes")
    
    def test_concurrent_large_text_processing(self, rule_engine, tokenizer):
        """Test concurrent processing of large texts."""
        import threading
        import queue
        
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("start")
        
        # Create large text chunks
        base_text = "रामो राजमणिः सदा विजयते रामं रमेशं भजे"
        large_chunks = [base_text * 50 for _ in range(10)]  # 10 large chunks
        
        results_queue = queue.Queue()
        start_time = time.time()
        
        def process_chunk(chunk_id, text):
            try:
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=2)
                results_queue.put(('success', chunk_id, len(result.output_tokens)))
            except Exception as e:
                results_queue.put(('error', chunk_id, str(e)))
        
        # Start concurrent processing
        threads = []
        for i, chunk in enumerate(large_chunks):
            thread = threading.Thread(target=process_chunk, args=(i, chunk))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        profiler.take_snapshot("end")
        analysis = profiler.stop_monitoring()
        
        # Verify all chunks processed successfully
        success_count = sum(1 for status, _, _ in results if status == 'success')
        assert success_count == len(large_chunks), f"Some chunks failed: {len(large_chunks) - success_count} failures"
        
        # Performance should be reasonable even with concurrency
        assert total_time < 30.0, f"Concurrent processing too slow: {total_time} seconds"
        assert analysis['rss_growth'] < 300 * 1024 * 1024, f"Excessive memory in concurrent processing: {analysis['rss_growth']} bytes"
        
        print(f"\nConcurrent Processing Results:")
        print(f"Processed {len(large_chunks)} chunks concurrently in {total_time:.2f} seconds")
        print(f"Memory growth: {analysis['rss_growth']} bytes")
    
    def test_memory_pressure_handling(self, rule_engine, tokenizer):
        """Test behavior under memory pressure conditions."""
        profiler = MemoryProfiler()
        profiler.start_monitoring()
        profiler.take_snapshot("start")
        
        # Create memory pressure by processing many large texts
        large_texts = []
        for i in range(20):
            # Create progressively larger texts
            text = f"मेमोरी प्रेशर टेस्ट {i} " * (100 + i * 50)
            large_texts.append(text)
        
        processed_successfully = 0
        
        for i, text in enumerate(large_texts):
            try:
                tokens = tokenizer.tokenize(text)
                result = rule_engine.process(tokens, max_passes=2)
                
                processed_successfully += 1
                
                # Check memory usage
                current_memory = psutil.Process().memory_info().rss
                if current_memory > 500 * 1024 * 1024:  # 500MB threshold
                    # Force garbage collection under pressure
                    gc.collect()
                
                profiler.take_snapshot(f"text_{i}")
                
            except MemoryError:
                # Should handle memory errors gracefully
                gc.collect()
                continue
        
        profiler.take_snapshot("end")
        analysis = profiler.stop_monitoring()
        
        # Should process most texts successfully
        success_rate = processed_successfully / len(large_texts)
        assert success_rate > 0.7, f"Low success rate under memory pressure: {success_rate}"
        
        # Memory growth should be bounded
        assert analysis['rss_growth'] < 400 * 1024 * 1024, f"Unbounded memory growth: {analysis['rss_growth']} bytes"
        
        print(f"\nMemory Pressure Test Results:")
        print(f"Successfully processed {processed_successfully}/{len(large_texts)} texts")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Peak memory: {analysis['peak_rss']} bytes")


class TestRegressionPerformance:
    """Regression tests to ensure performance doesn't degrade."""
    
    def test_baseline_performance_regression(self, rule_engine, tokenizer):
        """Test that performance doesn't regress from baseline."""
        # Define baseline performance expectations
        baseline_expectations = {
            'simple_tokenization': 0.001,  # 1ms for simple text
            'complex_tokenization': 0.01,   # 10ms for complex text
            'simple_processing': 0.1,       # 100ms for simple processing
            'complex_processing': 1.0       # 1s for complex processing
        }
        
        # Test simple tokenization
        start_time = time.time()
        for _ in range(100):
            tokens = tokenizer.tokenize("rama")
        simple_tokenization_time = (time.time() - start_time) / 100
        
        assert simple_tokenization_time < baseline_expectations['simple_tokenization'], \
            f"Simple tokenization regression: {simple_tokenization_time:.4f}s > {baseline_expectations['simple_tokenization']}s"
        
        # Test complex tokenization
        complex_text = "क्ष्म्य्व्र्त्न्प्फ्ब्भ्म्य्र्ल्व्श्ष्स्ह्"
        start_time = time.time()
        for _ in range(50):
            tokens = tokenizer.tokenize(complex_text)
        complex_tokenization_time = (time.time() - start_time) / 50
        
        assert complex_tokenization_time < baseline_expectations['complex_tokenization'], \
            f"Complex tokenization regression: {complex_tokenization_time:.4f}s > {baseline_expectations['complex_tokenization']}s"
        
        # Test simple processing
        simple_tokens = tokenizer.tokenize("a+i")
        start_time = time.time()
        for _ in range(10):
            result = rule_engine.process(simple_tokens, max_passes=5)
        simple_processing_time = (time.time() - start_time) / 10
        
        assert simple_processing_time < baseline_expectations['simple_processing'], \
            f"Simple processing regression: {simple_processing_time:.4f}s > {baseline_expectations['simple_processing']}s"
        
        # Test complex processing
        complex_tokens = tokenizer.tokenize("dharma+artha+kāma+mokṣa:GEN:PL")
        start_time = time.time()
        result = rule_engine.process(complex_tokens, max_passes=10)
        complex_processing_time = time.time() - start_time
        
        assert complex_processing_time < baseline_expectations['complex_processing'], \
            f"Complex processing regression: {complex_processing_time:.4f}s > {baseline_expectations['complex_processing']}s"
        
        print(f"\nPerformance Regression Test Results:")
        print(f"Simple tokenization: {simple_tokenization_time:.4f}s (baseline: {baseline_expectations['simple_tokenization']}s)")
        print(f"Complex tokenization: {complex_tokenization_time:.4f}s (baseline: {baseline_expectations['complex_tokenization']}s)")
        print(f"Simple processing: {simple_processing_time:.4f}s (baseline: {baseline_expectations['simple_processing']}s)")
        print(f"Complex processing: {complex_processing_time:.4f}s (baseline: {baseline_expectations['complex_processing']}s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to show print statements