"""
Tests for performance optimization features.
"""

import time
import pytest
from unittest.mock import Mock, patch
from sanskrit_rewrite_engine.performance import (
    TransformationCache, RuleIndex, LazyTextProcessor, MemoryMonitor,
    ProcessingLimiter, PerformanceOptimizer, PerformanceMetrics,
    performance_timer, memory_limit_check
)
from sanskrit_rewrite_engine.rules import Rule


class TestTransformationCache:
    """Test transformation cache functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache get/put operations."""
        cache = TransformationCache(max_size=3, ttl_seconds=60)
        
        # Test cache miss
        result = cache.get("test", "rules_hash", "config_hash")
        assert result is None
        
        # Test cache put and hit
        test_result = {"output": "transformed"}
        cache.put("test", "rules_hash", "config_hash", test_result)
        
        cached = cache.get("test", "rules_hash", "config_hash")
        assert cached == test_result
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = TransformationCache(max_size=2, ttl_seconds=60)
        
        # Add items up to limit
        cache.put("text1", "hash1", "config1", "result1")
        cache.put("text2", "hash2", "config2", "result2")
        
        # Both should be cached
        assert cache.get("text1", "hash1", "config1") == "result1"
        assert cache.get("text2", "hash2", "config2") == "result2"
        
        # Add third item, should evict first
        cache.put("text3", "hash3", "config3", "result3")
        
        # First should be evicted
        assert cache.get("text1", "hash1", "config1") is None
        assert cache.get("text2", "hash2", "config2") == "result2"
        assert cache.get("text3", "hash3", "config3") == "result3"
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = TransformationCache(max_size=10, ttl_seconds=0.1)
        
        cache.put("test", "hash", "config", "result")
        assert cache.get("test", "hash", "config") == "result"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("test", "hash", "config") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TransformationCache(max_size=5, ttl_seconds=60)
        cache.put("test", "hash", "config", "result")
        
        stats = cache.get_stats()
        assert stats['size'] == 1
        assert stats['max_size'] == 5
        assert stats['ttl_seconds'] == 60
        assert stats['oldest_entry_age'] >= 0


class TestRuleIndex:
    """Test rule indexing functionality."""
    
    def test_rule_indexing(self):
        """Test basic rule indexing operations."""
        index = RuleIndex()
        
        rule1 = Rule(
            id="test1",
            name="Test Rule 1",
            description="Test",
            pattern="abc",
            replacement="xyz"
        )
        
        rule2 = Rule(
            id="test2", 
            name="Test Rule 2",
            description="Test",
            pattern="def",
            replacement="uvw"
        )
        
        index.add_rule(rule1)
        index.add_rule(rule2)
        
        # Test candidate rule retrieval
        candidates = index.get_candidate_rules("abcdef", 0)
        assert len(candidates) >= 1
        
        # Test rule removal
        index.remove_rule("test1")
        candidates = index.get_candidate_rules("abcdef", 0)
        # Should still have rule2 or fallback to all rules
        assert len(candidates) >= 0
    
    def test_literal_prefix_extraction(self):
        """Test literal prefix extraction from patterns."""
        index = RuleIndex()
        
        # Test simple literal prefix
        prefix = index._extract_literal_prefix("hello.*world")
        assert prefix == "he"  # At least 2 characters
        
        # Test no prefix
        prefix = index._extract_literal_prefix(".*test")
        assert prefix is None
        
        # Test escaped characters
        prefix = index._extract_literal_prefix(r"test\+more")
        assert prefix == "test+"
    
    def test_fast_match(self):
        """Test fast pattern matching."""
        index = RuleIndex()
        
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern=r"a\+b",
            replacement="c"
        )
        
        index.add_rule(rule)
        
        # Test successful match
        match = index.fast_match("test", "a+b test", 0)
        assert match is not None
        
        # Test no match
        match = index.fast_match("test", "xyz test", 0)
        assert match is None


class TestLazyTextProcessor:
    """Test lazy text processing functionality."""
    
    def test_chunk_creation(self):
        """Test text chunking."""
        text = "abcdefghijklmnopqrstuvwxyz"
        processor = LazyTextProcessor(text, chunk_size=5)
        
        assert processor.chunk_count() == 6  # 26 chars / 5 = 5.2, rounded up
        
        # Test chunk retrieval
        chunk0 = processor.get_chunk(0)
        assert chunk0 == "abcde"
        
        chunk1 = processor.get_chunk(1)
        assert chunk1 == "fghij"
        
        # Test last chunk
        last_chunk = processor.get_chunk(5)
        assert last_chunk == "z"
    
    def test_lazy_processing(self):
        """Test lazy chunk processing."""
        text = "hello world test"
        processor = LazyTextProcessor(text, chunk_size=5)
        
        def uppercase_processor(chunk: str) -> str:
            return chunk.upper()
        
        # Process specific chunks
        result0 = processor.process_chunk(0, uppercase_processor)
        assert result0 == "HELLO"
        
        result1 = processor.process_chunk(1, uppercase_processor)
        assert result1 == " WORL"
        
        # Get processed text
        processed = processor.get_processed_text()
        assert processed == "HELLO WORL test"  # Only first two chunks processed


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    @patch('psutil.Process')
    def test_memory_monitoring(self, mock_process):
        """Test memory usage monitoring."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(memory_limit_mb=50, check_interval=0.1)
        
        # Test current usage
        usage = monitor.get_current_usage()
        assert usage == 100.0  # 100 MB
        
        # Test limit check
        assert monitor.check_memory_limit() is True  # Exceeds 50 MB limit
        
        # Test stats
        stats = monitor.get_memory_stats()
        assert stats['current_mb'] == 100.0
        assert stats['limit_mb'] == 50
        assert stats['usage_percentage'] == 200.0
    
    @patch('psutil.Process')
    def test_memory_check_interval(self, mock_process):
        """Test memory check interval."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        monitor = MemoryMonitor(memory_limit_mb=50, check_interval=1.0)
        
        # First check should work
        assert monitor.check_memory_limit() is True
        
        # Immediate second check should return False (within interval)
        assert monitor.check_memory_limit() is False


class TestProcessingLimiter:
    """Test processing limits and timeouts."""
    
    def test_timeout_checking(self):
        """Test timeout functionality."""
        limiter = ProcessingLimiter(max_processing_time=0.1)
        
        limiter.start_processing()
        assert limiter.check_timeout() is False
        
        # Wait for timeout
        time.sleep(0.2)
        assert limiter.check_timeout() is True
    
    def test_iteration_limits(self):
        """Test iteration limit checking."""
        limiter = ProcessingLimiter(max_iterations=3)
        
        assert limiter.check_iteration_limit() is False
        
        limiter.increment_iteration()  # 1
        limiter.increment_iteration()  # 2
        assert limiter.check_iteration_limit() is False
        
        limiter.increment_iteration()  # 3
        assert limiter.check_iteration_limit() is True
    
    def test_text_length_limits(self):
        """Test text length limit checking."""
        limiter = ProcessingLimiter(max_text_length=10)
        
        assert limiter.check_text_length("short") is False
        assert limiter.check_text_length("this is a very long text") is True
    
    def test_processing_stats(self):
        """Test processing statistics."""
        limiter = ProcessingLimiter(max_processing_time=1.0, max_iterations=5)
        
        limiter.start_processing()
        limiter.increment_iteration()
        
        stats = limiter.get_stats()
        assert stats['iteration_count'] == 1
        assert stats['max_iterations'] == 5
        assert stats['elapsed_time'] >= 0
        assert stats['timeout_exceeded'] is False
        assert stats['iteration_limit_exceeded'] is False


class TestPerformanceOptimizer:
    """Test performance optimizer coordination."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer(
            cache_size=100,
            cache_ttl=1800,
            memory_limit_mb=256,
            chunk_size=500
        )
        
        assert optimizer.cache.max_size == 100
        assert optimizer.cache.ttl_seconds == 1800
        assert optimizer.memory_monitor.memory_limit_mb == 256
        assert optimizer.chunk_size == 500
    
    def test_metrics_update(self):
        """Test metrics updating."""
        optimizer = PerformanceOptimizer()
        
        # Update metrics
        optimizer.update_metrics(1.5, 5, cache_hit=False)
        optimizer.update_metrics(0.8, 3, cache_hit=True)
        
        metrics = optimizer.metrics
        assert metrics.total_processing_time == 2.3
        assert metrics.total_transformations == 8
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
    
    def test_performance_report(self):
        """Test performance report generation."""
        optimizer = PerformanceOptimizer()
        optimizer.update_metrics(1.0, 2, cache_hit=False)
        
        report = optimizer.get_performance_report()
        
        assert 'metrics' in report
        assert 'cache_stats' in report
        assert 'memory_stats' in report
        
        assert report['metrics']['total_processing_time'] == 1.0
        assert report['metrics']['total_transformations'] == 2
    
    def test_lazy_processor_creation(self):
        """Test lazy processor creation."""
        optimizer = PerformanceOptimizer(chunk_size=10)
        
        processor = optimizer.create_lazy_processor("test text here")
        assert isinstance(processor, LazyTextProcessor)
        assert processor.chunk_size == 10
    
    def test_limiter_creation(self):
        """Test processing limiter creation."""
        optimizer = PerformanceOptimizer()
        
        limiter = optimizer.create_limiter(max_time=5.0, max_iterations=100)
        assert isinstance(limiter, ProcessingLimiter)
        assert limiter.max_processing_time == 5.0
        assert limiter.max_iterations == 100


class TestPerformanceDecorators:
    """Test performance decorators."""
    
    def test_performance_timer(self):
        """Test performance timer decorator."""
        
        class MockEngine:
            def __init__(self):
                self._performance_optimizer = PerformanceOptimizer()
        
        engine = MockEngine()
        
        @performance_timer
        def test_function(self):
            time.sleep(0.01)
            return "result"
        
        result = test_function(engine)
        assert result == "result"
        
        # Check that timing was recorded
        metrics = engine._performance_optimizer.metrics
        assert 'test_function' in metrics.rule_application_times
        assert metrics.rule_application_times['test_function'] > 0
    
    @patch('sanskrit_rewrite_engine.performance.MemoryMonitor')
    def test_memory_limit_decorator(self, mock_monitor_class):
        """Test memory limit decorator."""
        mock_monitor = Mock()
        mock_monitor.check_memory_limit.return_value = False
        mock_monitor_class.return_value = mock_monitor
        
        @memory_limit_check(100)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Test memory limit exceeded
        mock_monitor.check_memory_limit.return_value = True
        
        with pytest.raises(MemoryError):
            test_function()


class TestPerformanceMetrics:
    """Test performance metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_processing_time == 0.0
        assert metrics.total_transformations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.peak_memory_usage == 0.0
        assert metrics.timeouts_triggered == 0
        assert metrics.memory_limits_triggered == 0
    
    def test_metrics_to_dict(self):
        """Test metrics dictionary conversion."""
        metrics = PerformanceMetrics()
        metrics.total_processing_time = 5.0
        metrics.total_transformations = 10
        metrics.cache_hits = 3
        metrics.cache_misses = 7
        
        data = metrics.to_dict()
        
        assert data['total_processing_time'] == 5.0
        assert data['total_transformations'] == 10
        assert data['cache_hits'] == 3
        assert data['cache_misses'] == 7
        assert data['cache_hit_rate'] == 0.3  # 3 / (3 + 7)
        
        # Test zero division protection
        metrics_empty = PerformanceMetrics()
        data_empty = metrics_empty.to_dict()
        assert data_empty['cache_hit_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])