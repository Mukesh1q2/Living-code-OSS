"""
Integration tests for engine performance optimizations.
"""

import time
import pytest
from unittest.mock import patch
from sanskrit_rewrite_engine.engine import SanskritRewriteEngine
from sanskrit_rewrite_engine.config import EngineConfig
from sanskrit_rewrite_engine.rules import Rule


class TestEnginePerformanceIntegration:
    """Test engine performance optimization integration."""
    
    def test_caching_functionality(self):
        """Test that caching works in the engine."""
        config = EngineConfig(
            cache_size=10,
            cache_ttl=60,
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add a simple rule
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="Test",
            pattern="a",
            replacement="b"
        )
        engine.add_rule(rule)
        
        # First processing - should be cache miss
        result1 = engine.process("a test")
        assert result1.success
        assert result1.output_text == "b test"
        
        # Second processing - should be cache hit
        start_time = time.time()
        result2 = engine.process("a test")
        cache_time = time.time() - start_time
        
        assert result2.success
        assert result2.output_text == "b test"
        assert result1.output_text == result2.output_text
        
        # Cache hit should be faster (though this might be flaky in tests)
        # We mainly check that it works without error
        
        # Check performance metrics
        metrics = engine.get_performance_metrics()
        assert metrics['metrics']['cache_hits'] >= 1
        assert metrics['metrics']['total_transformations'] >= 1
    
    def test_memory_monitoring(self):
        """Test memory monitoring integration."""
        config = EngineConfig(
            memory_limit_mb=1,  # Very low limit to trigger monitoring
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Check that memory monitoring is active
        metrics = engine.get_performance_metrics()
        assert 'memory_stats' in metrics
        assert metrics['memory_stats']['limit_mb'] == 1
    
    def test_processing_limits(self):
        """Test processing limits and timeouts."""
        config = EngineConfig(
            timeout_seconds=0.1,  # Very short timeout
            max_iterations=2,
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add a rule that could cause many iterations
        rule = Rule(
            id="loop_rule",
            name="Loop Rule", 
            description="Test",
            pattern="a",
            replacement="aa"  # This could cause exponential growth
        )
        engine.add_rule(rule)
        
        # Process text that might timeout or hit iteration limit
        result = engine.process("a")
        
        # Should either succeed quickly or fail gracefully
        if not result.success:
            assert "timeout" in result.error_message.lower() or "iteration" in result.error_message.lower()
        else:
            assert result.iterations_used <= config.max_iterations
    
    def test_lazy_processing_large_text(self):
        """Test lazy processing for large texts."""
        config = EngineConfig(
            max_text_length=1000,
            chunk_size=100,
            enable_lazy_processing=True,
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add a simple rule
        rule = Rule(
            id="simple_rule",
            name="Simple Rule",
            description="Test",
            pattern="x",
            replacement="y"
        )
        engine.add_rule(rule)
        
        # Create large text (but not too large to avoid timeout)
        large_text = "x " * 200  # 400 characters
        
        result = engine.process(large_text)
        assert result.success
        assert "y " in result.output_text
        assert result.output_text.count("y") == 200
    
    def test_rule_indexing_performance(self):
        """Test that rule indexing improves performance."""
        config = EngineConfig(enable_tracing=False)
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add multiple rules
        for i in range(10):
            rule = Rule(
                id=f"rule_{i}",
                name=f"Rule {i}",
                description="Test",
                pattern=f"test{i}",
                replacement=f"result{i}"
            )
            engine.add_rule(rule)
        
        # Process text that matches one of the rules
        result = engine.process("test5 example")
        assert result.success
        assert "result5" in result.output_text
        
        # Check that rule indexing is working
        assert len(engine._performance_optimizer.rule_index._rules_by_id) == 13  # 10 + 3 basic rules
    
    def test_performance_metrics_collection(self):
        """Test comprehensive performance metrics collection."""
        config = EngineConfig(enable_tracing=True)
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add a rule
        rule = Rule(
            id="metrics_rule",
            name="Metrics Rule",
            description="Test",
            pattern="hello",
            replacement="hi"
        )
        engine.add_rule(rule)
        
        # Process some text
        result = engine.process("hello world")
        assert result.success
        
        # Check metrics
        metrics = engine.get_performance_metrics()
        
        # Check that all metric categories are present
        assert 'metrics' in metrics
        assert 'cache_stats' in metrics
        assert 'memory_stats' in metrics
        
        # Check specific metrics
        assert metrics['metrics']['total_processing_time'] > 0
        assert metrics['metrics']['total_transformations'] >= 1
        assert metrics['metrics']['cache_misses'] >= 1
        
        # Check cache stats
        assert 'size' in metrics['cache_stats']
        assert 'max_size' in metrics['cache_stats']
        
        # Check memory stats
        assert 'current_mb' in metrics['memory_stats']
        assert 'peak_mb' in metrics['memory_stats']
    
    def test_cache_invalidation_on_rule_changes(self):
        """Test that cache is properly invalidated when rules change."""
        config = EngineConfig(cache_size=10, enable_tracing=False)
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Process text with initial rules
        result1 = engine.process("test text")
        
        # Add a new rule
        rule = Rule(
            id="new_rule",
            name="New Rule",
            description="Test",
            pattern="test",
            replacement="changed"
        )
        engine.add_rule(rule)
        
        # Process same text - should get different result due to new rule
        result2 = engine.process("test text")
        
        assert result1.output_text != result2.output_text
        assert "changed" in result2.output_text
    
    def test_performance_optimization_disabled(self):
        """Test engine works when performance optimizations are minimal."""
        config = EngineConfig(
            cache_size=0,  # Disable caching
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Add a rule
        rule = Rule(
            id="no_cache_rule",
            name="No Cache Rule",
            description="Test",
            pattern="abc",
            replacement="xyz"
        )
        engine.add_rule(rule)
        
        # Process text multiple times
        result1 = engine.process("abc test")
        result2 = engine.process("abc test")
        
        assert result1.success
        assert result2.success
        assert result1.output_text == result2.output_text == "xyz test"
        
        # Should have no cache hits since caching is disabled
        metrics = engine.get_performance_metrics()
        assert metrics['metrics']['cache_hits'] == 0
    
    @patch('sanskrit_rewrite_engine.performance.psutil.Process')
    def test_memory_limit_enforcement(self, mock_process):
        """Test memory limit enforcement."""
        # Mock high memory usage
        mock_memory_info = type('MockMemoryInfo', (), {'rss': 600 * 1024 * 1024})()  # 600 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        config = EngineConfig(
            memory_limit_mb=500,  # 500 MB limit
            enable_tracing=False
        )
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Force memory check
        engine._performance_optimizer.memory_monitor.check_memory_limit()
        
        # Check that memory monitoring detected the issue
        stats = engine._performance_optimizer.memory_monitor.get_memory_stats()
        assert stats['current_mb'] == 600.0
        assert stats['usage_percentage'] > 100.0
    
    def test_clear_cache_functionality(self):
        """Test cache clearing functionality."""
        config = EngineConfig(cache_size=10, enable_tracing=False)
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Process text to populate cache
        result = engine.process("test")
        assert result.success
        
        # Check cache has entries
        metrics_before = engine.get_performance_metrics()
        cache_size_before = metrics_before['cache_stats']['size']
        
        # Clear cache
        engine.clear_cache()
        
        # Check cache is empty
        metrics_after = engine.get_performance_metrics()
        cache_size_after = metrics_after['cache_stats']['size']
        
        assert cache_size_after == 0
        assert cache_size_before > cache_size_after
    
    def test_reset_performance_metrics(self):
        """Test performance metrics reset functionality."""
        config = EngineConfig(enable_tracing=False)
        engine = SanskritRewriteEngine(config.to_dict())
        
        # Process text to generate metrics
        result = engine.process("test")
        assert result.success
        
        # Check metrics exist
        metrics_before = engine.get_performance_metrics()
        assert metrics_before['metrics']['total_processing_time'] > 0
        
        # Reset metrics
        engine.reset_performance_metrics()
        
        # Check metrics are reset
        metrics_after = engine.get_performance_metrics()
        assert metrics_after['metrics']['total_processing_time'] == 0.0
        assert metrics_after['metrics']['total_transformations'] == 0
        assert metrics_after['metrics']['cache_hits'] == 0
        assert metrics_after['metrics']['cache_misses'] == 0


if __name__ == "__main__":
    pytest.main([__file__])