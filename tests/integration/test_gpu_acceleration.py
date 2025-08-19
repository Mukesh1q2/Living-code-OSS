"""
Test suite for GPU acceleration components.

Tests GPU acceleration, memory optimization, and performance benchmarking
for the Sanskrit Rewrite Engine.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
from sanskrit_rewrite_engine.gpu_acceleration import (
    GPUConfig, GPUDeviceManager, MixedPrecisionManager, 
    VRAMAwareBatcher, AutoScalingManager, GPUAcceleratedInference,
    create_gpu_inference_engine, get_optimal_gpu_config
)
from sanskrit_rewrite_engine.memory_optimization import (
    MemoryConfig, LRUCache, ChunkedDataLoader, MemoryMappedStorage,
    CompressedStorage, MemoryOptimizer, create_memory_optimizer,
    get_rx6800m_memory_config
)
from sanskrit_rewrite_engine.gpu_benchmarks import (
    BenchmarkConfig, GPUBenchmarkRunner, SanskritBenchmarkData,
    run_gpu_benchmarks
)


class TestGPUConfig:
    """Test GPU configuration."""
    
    def test_default_config(self):
        """Test default GPU configuration."""
        config = GPUConfig()
        
        assert config.device == "auto"
        assert config.mixed_precision is True
        assert config.batch_size == 32
        assert config.max_sequence_length == 512
        assert config.memory_fraction == 0.8
        assert config.rocm_optimization is True
        assert config.auto_scaling is True
    
    def test_rx6800m_config(self):
        """Test RX6800M specific configuration."""
        config = GPUConfig()
        
        assert config.rx6800m_vram_gb == 12.0
        assert config.rx6800m_compute_units == 60
        assert config.rx6800m_base_clock == 2105
        assert config.rx6800m_memory_clock == 2000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GPUConfig(
            device="cuda",
            batch_size=64,
            mixed_precision=False,
            auto_scaling=False
        )
        
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.mixed_precision is False
        assert config.auto_scaling is False


class TestGPUDeviceManager:
    """Test GPU device manager."""
    
    def test_cpu_fallback(self):
        """Test CPU fallback when GPU not available."""
        config = GPUConfig(device="cpu")
        manager = GPUDeviceManager(config)
        
        assert manager.device_type == "cpu"
        assert manager.device is not None
    
    @patch('sanskrit_rewrite_engine.gpu_acceleration.torch')
    def test_cuda_detection(self, mock_torch):
        """Test CUDA GPU detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "11.8"
        mock_torch.device.return_value = Mock()
        
        config = GPUConfig(device="auto")
        manager = GPUDeviceManager(config)
        
        # Should attempt CUDA detection
        mock_torch.cuda.is_available.assert_called()
    
    @patch('sanskrit_rewrite_engine.gpu_acceleration.torch')
    def test_rocm_detection(self, mock_torch):
        """Test ROCm GPU detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "HIP"  # ROCm indicator
        mock_torch.device.return_value = Mock()
        
        config = GPUConfig(device="auto")
        manager = GPUDeviceManager(config)
        
        # Should detect ROCm
        mock_torch.cuda.is_available.assert_called()
    
    def test_memory_stats_cpu(self):
        """Test memory statistics on CPU."""
        config = GPUConfig(device="cpu")
        manager = GPUDeviceManager(config)
        
        stats = manager.get_memory_stats()
        
        # CPU should have zero GPU memory stats
        assert stats.total_memory == 0.0
        assert stats.allocated_memory == 0.0
        assert stats.utilization_percent == 0.0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        config = GPUConfig(device="cpu")
        manager = GPUDeviceManager(config)
        
        # Should not raise exception
        manager.clear_cache()


class TestMixedPrecisionManager:
    """Test mixed precision manager."""
    
    def test_initialization_cpu(self):
        """Test initialization on CPU."""
        config = GPUConfig(device="cpu", mixed_precision=True)
        device_manager = GPUDeviceManager(config)
        mp_manager = MixedPrecisionManager(config, device_manager)
        
        # Mixed precision should be disabled on CPU
        assert mp_manager.enabled is False
    
    def test_autocast_context_cpu(self):
        """Test autocast context on CPU."""
        config = GPUConfig(device="cpu", mixed_precision=True)
        device_manager = GPUDeviceManager(config)
        mp_manager = MixedPrecisionManager(config, device_manager)
        
        # Should return null context for CPU
        with mp_manager.autocast_context():
            pass  # Should not raise exception
    
    def test_disabled_mixed_precision(self):
        """Test disabled mixed precision."""
        config = GPUConfig(mixed_precision=False)
        device_manager = GPUDeviceManager(config)
        mp_manager = MixedPrecisionManager(config, device_manager)
        
        assert mp_manager.enabled is False


class TestVRAMAwareBatcher:
    """Test VRAM-aware batching."""
    
    def test_initialization(self):
        """Test batcher initialization."""
        config = GPUConfig(batch_size=32)
        device_manager = GPUDeviceManager(config)
        batcher = VRAMAwareBatcher(config, device_manager)
        
        assert batcher.current_batch_size == 32
        assert batcher.min_batch_size == 1
        assert batcher.max_batch_size == 64
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        config = GPUConfig(batch_size=32)
        device_manager = GPUDeviceManager(config)
        batcher = VRAMAwareBatcher(config, device_manager)
        
        # Test with small memory estimate
        batch_size = batcher.get_optimal_batch_size(100, 0.01)  # 10MB per item
        assert batch_size >= batcher.min_batch_size
        assert batch_size <= min(batcher.max_batch_size, 100)
    
    def test_create_batches(self):
        """Test batch creation."""
        config = GPUConfig(batch_size=10)
        device_manager = GPUDeviceManager(config)
        batcher = VRAMAwareBatcher(config, device_manager)
        
        data = list(range(25))  # 25 items
        batches = batcher.create_batches(data, 0.01)
        
        assert len(batches) > 0
        assert sum(len(batch) for batch in batches) == len(data)
    
    def test_adaptive_batch_resize(self):
        """Test adaptive batch resizing."""
        config = GPUConfig(batch_size=32)
        device_manager = GPUDeviceManager(config)
        batcher = VRAMAwareBatcher(config, device_manager)
        
        # High memory usage should reduce batch size
        new_size = batcher.adaptive_batch_resize(0.95)
        assert new_size < 32
        
        # Low memory usage should increase batch size
        batcher.current_batch_size = 16
        new_size = batcher.adaptive_batch_resize(0.5)
        assert new_size > 16


class TestAutoScalingManager:
    """Test auto-scaling manager."""
    
    def test_initialization(self):
        """Test auto-scaling manager initialization."""
        config = GPUConfig(auto_scaling=True)
        manager = AutoScalingManager(config)
        
        assert manager.enabled is True
        assert manager.scaling_factor == 1.0
        assert manager.min_scaling == 0.5
        assert manager.max_scaling == 2.0
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        config = GPUConfig(auto_scaling=True)
        manager = AutoScalingManager(config)
        
        # Simple data
        simple_data = ["a", "b", "c"]
        complexity = manager.analyze_complexity(simple_data)
        assert complexity > 0
        
        # Complex data
        complex_data = [{"key": "value" * 100} for _ in range(10)]
        complex_complexity = manager.analyze_complexity(complex_data)
        assert complex_complexity > complexity
    
    def test_scaling_factor_calculation(self):
        """Test scaling factor calculation."""
        config = GPUConfig(auto_scaling=True)
        manager = AutoScalingManager(config)
        
        # High complexity should increase scaling
        scaling = manager.calculate_scaling_factor(8.0, 100.0)
        assert scaling > 1.0
        
        # Low complexity should decrease scaling
        scaling = manager.calculate_scaling_factor(1.0, 100.0)
        assert scaling < 1.0
    
    def test_apply_scaling(self):
        """Test scaling application."""
        config = GPUConfig(auto_scaling=True)
        manager = AutoScalingManager(config)
        manager.scaling_factor = 1.5
        
        base_config = {
            'batch_size': 32,
            'num_workers': 4,
            'memory_fraction': 0.8
        }
        
        scaled_config = manager.apply_scaling(base_config)
        
        assert scaled_config['batch_size'] == int(32 * 1.5)
        assert scaled_config['num_workers'] >= 1
        assert scaled_config['memory_fraction'] <= 0.95


class TestGPUAcceleratedInference:
    """Test GPU accelerated inference engine."""
    
    def test_initialization(self):
        """Test inference engine initialization."""
        config = GPUConfig(device="cpu")  # Use CPU for testing
        engine = GPUAcceleratedInference(config)
        
        assert engine.config == config
        assert engine.device_manager is not None
        assert engine.mixed_precision is not None
        assert engine.batcher is not None
        assert engine.auto_scaler is not None
    
    def test_process_batch(self):
        """Test batch processing."""
        config = GPUConfig(device="cpu")
        engine = GPUAcceleratedInference(config)
        
        def mock_processing_function(data):
            return [item.upper() if isinstance(item, str) else str(item) for item in data]
        
        test_data = ["hello", "world", "test"]
        result = engine.process_batch(test_data, mock_processing_function)
        
        assert result.outputs == ["HELLO", "WORLD", "TEST"]
        assert result.processing_time > 0
        assert result.throughput > 0
        assert len(result.errors) == 0
    
    def test_process_large_dataset(self):
        """Test large dataset processing."""
        config = GPUConfig(device="cpu", batch_size=5)
        engine = GPUAcceleratedInference(config)
        
        def mock_processing_function(data):
            return [f"processed_{item}" for item in data]
        
        test_data = list(range(12))  # 12 items, batch size 5
        results = engine.process_large_dataset(test_data, mock_processing_function)
        
        assert len(results) >= 2  # At least 2 batches
        assert all(len(result.errors) == 0 for result in results)
        
        # Check total outputs
        total_outputs = []
        for result in results:
            total_outputs.extend(result.outputs)
        assert len(total_outputs) == len(test_data)
    
    def test_get_performance_stats(self):
        """Test performance statistics."""
        config = GPUConfig(device="cpu")
        engine = GPUAcceleratedInference(config)
        
        stats = engine.get_performance_stats()
        
        assert 'device_info' in stats
        assert 'memory_stats' in stats
        assert 'inference_stats' in stats
        assert 'config' in stats
        
        assert stats['device_info']['device_type'] == 'cpu'
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        config = GPUConfig(device="cpu")
        engine = GPUAcceleratedInference(config)
        
        def mock_processing_function(data):
            time.sleep(0.01)  # Small delay
            return [f"result_{item}" for item in data]
        
        test_data = ["test1", "test2", "test3"]
        benchmark_results = engine.benchmark_performance(
            test_data, mock_processing_function, iterations=2
        )
        
        assert benchmark_results['iterations'] == 2
        assert benchmark_results['data_size'] == 3
        assert len(benchmark_results['results']) == 2
        assert benchmark_results['average_throughput'] > 0


class TestMemoryOptimization:
    """Test memory optimization components."""
    
    def test_memory_config(self):
        """Test memory configuration."""
        config = MemoryConfig()
        
        assert config.max_memory_gb == 12.0
        assert config.cache_size_mb == 2048
        assert config.enable_memory_mapping is True
        assert config.enable_compression is True
    
    def test_rx6800m_memory_config(self):
        """Test RX6800M specific memory configuration."""
        config = get_rx6800m_memory_config()
        
        assert config.max_memory_gb == 12.0
        assert config.gc_threshold == 0.75
        assert config.chunk_size == 500
        assert config.swap_threshold == 0.85


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size_mb=1)  # 1MB cache
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_cache_eviction(self):
        """Test cache eviction."""
        cache = LRUCache(max_size_mb=1)  # Small cache
        
        # Fill cache with large items
        large_data = "x" * (1024 * 512)  # 512KB
        cache.put("key1", large_data)
        cache.put("key2", large_data)  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == large_data  # Still there
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size_mb=1)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.current_size == 0


class TestChunkedDataLoader:
    """Test chunked data loader."""
    
    def test_json_data_loading(self):
        """Test loading JSON data in chunks."""
        # Create temporary JSON file
        test_data = [{"id": i, "value": f"item_{i}"} for i in range(25)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            loader = ChunkedDataLoader(temp_file, chunk_size=10)
            
            # Check chunk info
            assert len(loader.chunks_info) == 3  # 25 items, chunk size 10
            
            # Test chunk loading
            chunk = loader.get_chunk(0)
            assert len(chunk) == 10
            assert chunk[0]["id"] == 0
            
            # Test iteration
            chunks = list(loader)
            assert len(chunks) == 3
            assert sum(len(chunk) for chunk in chunks) == 25
            
        finally:
            Path(temp_file).unlink()
    
    def test_prefetching(self):
        """Test prefetching functionality."""
        test_data = [f"item_{i}" for i in range(20)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            loader = ChunkedDataLoader(temp_file, chunk_size=5, prefetch_size=2)
            
            # Start prefetching
            loader.start_prefetching()
            time.sleep(0.1)  # Allow prefetching to start
            
            # Stop prefetching
            loader.stop_prefetching()
            
        finally:
            Path(temp_file).unlink()


class TestMemoryMappedStorage:
    """Test memory-mapped storage."""
    
    def test_storage_operations(self):
        """Test memory-mapped storage operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MemoryMappedStorage(temp_dir, max_size_gb=1.0)
            
            # Test data storage
            test_data = {"key": "value", "numbers": [1, 2, 3, 4, 5]}
            success = storage.store_data("test_key", test_data)
            assert success is True
            
            # Test data loading
            loaded_data = storage.load_data("test_key")
            assert loaded_data == test_data
            
            # Test data removal
            success = storage.remove_data("test_key")
            assert success is True
            
            # Test loading removed data
            loaded_data = storage.load_data("test_key")
            assert loaded_data is None
    
    def test_cleanup(self):
        """Test storage cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MemoryMappedStorage(temp_dir)
            
            # Store some data
            storage.store_data("key1", "value1")
            storage.store_data("key2", "value2")
            
            # Cleanup
            storage.cleanup()
            
            # Data should be removed
            assert storage.load_data("key1") is None
            assert storage.load_data("key2") is None


class TestCompressedStorage:
    """Test compressed storage."""
    
    def test_compression_operations(self):
        """Test compression and decompression."""
        storage = CompressedStorage(compression_level=6)
        
        if not storage.compression_available:
            pytest.skip("Compression not available")
        
        # Test data compression
        test_data = {"message": "hello world" * 100}  # Compressible data
        success = storage.compress_data("test_key", test_data)
        assert success is True
        
        # Test data decompression
        decompressed_data = storage.decompress_data("test_key")
        assert decompressed_data == test_data
        
        # Test compression stats
        stats = storage.get_compression_stats()
        assert stats['items'] == 1
        assert stats['compression_ratio'] < 1.0  # Should be compressed


class TestMemoryOptimizer:
    """Test memory optimizer."""
    
    def test_initialization(self):
        """Test memory optimizer initialization."""
        config = MemoryConfig(enable_memory_mapping=False)  # Disable for testing
        optimizer = MemoryOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.cache is not None
        assert optimizer.compressed_storage is not None
    
    def test_large_dataset_storage(self):
        """Test large dataset storage."""
        config = MemoryConfig(
            enable_memory_mapping=False,
            enable_compression=True,
            max_memory_gb=1.0
        )
        optimizer = MemoryOptimizer(config)
        
        # Small dataset - should use cache
        small_data = ["item1", "item2", "item3"]
        success = optimizer.store_large_dataset("small_key", small_data)
        assert success is True
        
        loaded_data = optimizer.load_large_dataset("small_key")
        assert loaded_data == small_data
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        config = MemoryConfig(enable_memory_mapping=False)
        optimizer = MemoryOptimizer(config)
        
        # Store some data
        optimizer.cache.put("key1", "value1")
        optimizer.cache.put("key2", "value2")
        
        # Optimize memory
        results = optimizer.optimize_memory_usage()
        
        assert 'actions_taken' in results
        assert 'gc_triggered' in results
        assert results['gc_triggered'] is True
    
    def test_memory_report(self):
        """Test memory report generation."""
        config = MemoryConfig(enable_memory_mapping=False)
        optimizer = MemoryOptimizer(config)
        
        report = optimizer.get_memory_report()
        
        assert 'memory_stats' in report
        assert 'cache_stats' in report
        assert 'compression_stats' in report
        assert 'config' in report
    
    def test_cleanup(self):
        """Test memory optimizer cleanup."""
        config = MemoryConfig(enable_memory_mapping=False)
        optimizer = MemoryOptimizer(config)
        
        # Should not raise exception
        optimizer.cleanup()


class TestSanskritBenchmarkData:
    """Test Sanskrit benchmark data generation."""
    
    def test_sanskrit_text_generation(self):
        """Test Sanskrit text generation."""
        generator = SanskritBenchmarkData()
        
        texts = generator.generate_sanskrit_texts(5, avg_length=100)
        
        assert len(texts) == 5
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)
    
    def test_rule_application_data(self):
        """Test rule application data generation."""
        generator = SanskritBenchmarkData()
        
        data = generator.generate_rule_application_data(10)
        
        assert len(data) == 10
        assert all('input_text' in item for item in data)
        assert all('expected_output' in item for item in data)
        assert all('rule_type' in item for item in data)
    
    def test_mixed_complexity_data(self):
        """Test mixed complexity data generation."""
        generator = SanskritBenchmarkData()
        
        data = generator.generate_mixed_complexity_data(8)
        
        assert len(data) == 8
        assert all('complexity' in item for item in data)
        assert all(1 <= item['complexity'] <= 4 for item in data)


class TestGPUBenchmarkRunner:
    """Test GPU benchmark runner."""
    
    def test_initialization(self):
        """Test benchmark runner initialization."""
        config = BenchmarkConfig(
            test_data_sizes=[10, 20],
            iterations_per_test=1,
            save_results=False
        )
        runner = GPUBenchmarkRunner(config)
        
        assert runner.config == config
        assert runner.data_generator is not None
    
    @patch('sanskrit_rewrite_engine.gpu_benchmarks.GPUAcceleratedInference')
    def test_benchmark_execution(self, mock_gpu_engine):
        """Test benchmark execution."""
        # Mock GPU engine
        mock_engine_instance = Mock()
        mock_engine_instance.process_large_dataset.return_value = [
            Mock(throughput=100.0, memory_peak=1.0, gpu_utilization=50.0, errors=[])
        ]
        mock_engine_instance.device_manager.get_memory_stats.return_value = Mock(
            total_memory=12.0, allocated_memory=2.0, utilization_percent=16.7
        )
        mock_gpu_engine.return_value = mock_engine_instance
        
        config = BenchmarkConfig(
            test_data_sizes=[5],
            batch_sizes=[2],
            iterations_per_test=1,
            throughput_test=True,
            latency_test=False,
            memory_stress_test=False,
            scaling_test=False,
            save_results=False
        )
        
        runner = GPUBenchmarkRunner(config)
        suite = runner.run_full_benchmark_suite()
        
        assert suite.suite_name == "Sanskrit GPU Acceleration Benchmark"
        assert len(suite.results) > 0
        assert 'total_tests' in suite.summary


# Integration tests
class TestIntegration:
    """Integration tests for GPU acceleration components."""
    
    def test_create_gpu_inference_engine(self):
        """Test GPU inference engine creation."""
        engine = create_gpu_inference_engine()
        
        assert engine is not None
        assert hasattr(engine, 'config')
        assert hasattr(engine, 'device_manager')
    
    def test_get_optimal_gpu_config(self):
        """Test optimal GPU configuration."""
        config = get_optimal_gpu_config()
        
        assert isinstance(config, GPUConfig)
        assert config.batch_size > 0
        assert config.memory_fraction > 0
    
    def test_create_memory_optimizer(self):
        """Test memory optimizer creation."""
        optimizer = create_memory_optimizer()
        
        assert optimizer is not None
        assert hasattr(optimizer, 'config')
        assert hasattr(optimizer, 'cache')
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with GPU acceleration."""
        # Create components
        gpu_config = GPUConfig(device="cpu")  # Use CPU for testing
        memory_config = MemoryConfig(enable_memory_mapping=False)
        
        gpu_engine = GPUAcceleratedInference(gpu_config)
        memory_optimizer = MemoryOptimizer(memory_config)
        
        # Test data
        test_data = ["sanskrit text 1", "sanskrit text 2", "sanskrit text 3"]
        
        # Store in memory optimizer
        success = memory_optimizer.store_large_dataset("test_data", test_data)
        assert success is True
        
        # Load from memory optimizer
        loaded_data = memory_optimizer.load_large_dataset("test_data")
        assert loaded_data == test_data
        
        # Process with GPU engine
        def mock_processing(data):
            return [f"processed: {item}" for item in data]
        
        result = gpu_engine.process_batch(loaded_data, mock_processing)
        
        assert len(result.outputs) == len(test_data)
        assert all("processed:" in output for output in result.outputs)
        assert result.throughput > 0
        
        # Cleanup
        memory_optimizer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])