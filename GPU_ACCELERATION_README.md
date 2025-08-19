# Sanskrit Rewrite Engine - GPU Acceleration

This document provides comprehensive information about the GPU acceleration capabilities of the Sanskrit Rewrite Engine, optimized for RX6800M GPU with VRAM-aware batching and mixed-precision inference.

## Overview

The GPU acceleration system provides:

- **RX6800M Optimization**: Specialized optimizations for AMD RX6800M GPU with 12GB VRAM
- **Mixed-Precision Inference**: Speed optimization using FP16/FP32 mixed precision
- **VRAM-Aware Batching**: Dynamic batch sizing based on available GPU memory
- **Memory Optimization**: Efficient handling of large rule sets and Sanskrit corpora
- **Auto-Scaling**: Automatic resource scaling based on processing complexity
- **Containerized Deployment**: Docker support with ROCm for AMD GPUs
- **Performance Benchmarking**: Comprehensive benchmarking and profiling tools

## System Requirements

### Hardware Requirements

- **GPU**: AMD RX6800M (12GB VRAM) or compatible
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for models and cache

### Software Requirements

- **OS**: Ubuntu 22.04+ or compatible Linux distribution
- **Python**: 3.10+
- **ROCm**: 6.0+ for AMD GPU support
- **Docker**: 20.10+ (for containerized deployment)

## Installation

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y git wget curl build-essential

# Install ROCm (for AMD GPUs)
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm
```

### 2. Install Python Dependencies

```bash
# Clone repository
git clone <repository-url>
cd sanskrit-rewrite-engine

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test GPU availability
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Run quick demo
python -m sanskrit_rewrite_engine --demo --quick
```

## Usage

### Command Line Interface

The GPU acceleration system provides a comprehensive CLI:

```bash
# Run demonstration with RX6800M optimizations
python -m sanskrit_rewrite_engine --demo --rx6800m --data-size 1000

# Run performance benchmarks
python -m sanskrit_rewrite_engine --benchmark --output ./results

# Run quick benchmark
python -m sanskrit_rewrite_engine --benchmark --quick

# Run memory stress test
python -m sanskrit_rewrite_engine --benchmark --stress

# Test memory optimization
python -m sanskrit_rewrite_engine --memory-test --data-size 5000

# Start service mode
python -m sanskrit_rewrite_engine --serve
```

### Python API

```python
from sanskrit_rewrite_engine.gpu_acceleration import (
    GPUAcceleratedInference, GPUConfig, create_gpu_inference_engine
)
from sanskrit_rewrite_engine.memory_optimization import (
    MemoryOptimizer, get_rx6800m_memory_config
)

# Create GPU-accelerated inference engine
gpu_config = GPUConfig(
    device="auto",
    mixed_precision=True,
    batch_size=64,
    rocm_optimization=True,
    auto_scaling=True
)

engine = create_gpu_inference_engine(gpu_config)

# Create memory optimizer
memory_config = get_rx6800m_memory_config()
optimizer = MemoryOptimizer(memory_config)

# Process data
def sanskrit_processing_function(data):
    # Your Sanskrit processing logic here
    return [f"processed: {item}" for item in data]

# Process with GPU acceleration
test_data = ["sanskrit text 1", "sanskrit text 2", "sanskrit text 3"]
result = engine.process_batch(test_data, sanskrit_processing_function)

print(f"Throughput: {result.throughput:.2f} items/sec")
print(f"GPU Utilization: {result.gpu_utilization:.1f}%")
```

## Docker Deployment

### Build and Run with ROCm Support

```bash
# Build Docker image
docker build -f docker/Dockerfile.rocm -t sanskrit-engine-rocm .

# Run with GPU support
docker-compose -f docker/docker-compose.rocm.yml up -d

# Check GPU access in container
docker exec sanskrit-rewrite-engine-rocm python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### Docker Services

The Docker setup includes:

- **sanskrit-engine-rocm**: Main processing service
- **sanskrit-monitor**: Monitoring dashboard
- **redis**: Caching service (optional)

Access points:
- Jupyter: http://localhost:8888
- API: http://localhost:8000
- Monitoring: http://localhost:3000

## Configuration

### GPU Configuration

```python
from sanskrit_rewrite_engine.gpu_acceleration import GPUConfig

config = GPUConfig(
    device="auto",              # "auto", "cuda", "rocm", "cpu"
    mixed_precision=True,       # Enable mixed precision
    batch_size=64,             # Batch size (auto-adjusted for RX6800M)
    max_sequence_length=512,   # Maximum sequence length
    memory_fraction=0.8,       # GPU memory fraction to use
    rocm_optimization=True,    # RX6800M specific optimizations
    auto_scaling=True,         # Enable auto-scaling
    benchmark_mode=False       # Enable for benchmarking
)
```

### Memory Configuration

```python
from sanskrit_rewrite_engine.memory_optimization import MemoryConfig

config = MemoryConfig(
    max_memory_gb=12.0,        # RX6800M VRAM limit
    cache_size_mb=2048,        # 2GB cache
    enable_memory_mapping=True, # Memory mapping for large datasets
    enable_compression=True,    # Data compression
    gc_threshold=0.75,         # Garbage collection threshold
    chunk_size=500,            # Items per chunk
    prefetch_size=2,           # Prefetch chunks
    swap_threshold=0.85        # Swap to disk threshold
)
```

## Performance Optimization

### RX6800M Specific Optimizations

The system includes several RX6800M-specific optimizations:

1. **VRAM Management**: Optimized for 12GB VRAM with intelligent batching
2. **ROCm Optimizations**: Kernel optimizations for AMD architecture
3. **Memory Allocation**: Efficient memory allocation patterns
4. **Batch Sizing**: Dynamic batch sizing based on available VRAM
5. **Cache Management**: Optimized caching strategies

### Performance Tuning Tips

1. **Batch Size**: Start with 64 and adjust based on memory usage
2. **Mixed Precision**: Enable for 2x speed improvement
3. **Memory Fraction**: Use 0.8 for RX6800M to leave headroom
4. **Auto-Scaling**: Enable for dynamic resource management
5. **Prefetching**: Use 2-3 chunks for optimal I/O overlap

## Benchmarking

### Running Benchmarks

```bash
# Full benchmark suite
python -m sanskrit_rewrite_engine --benchmark

# Quick benchmark (reduced test cases)
python -m sanskrit_rewrite_engine --benchmark --quick

# Memory stress test
python -m sanskrit_rewrite_engine --benchmark --stress

# Custom benchmark
python -c "
from sanskrit_rewrite_engine.gpu_benchmarks import run_gpu_benchmarks, BenchmarkConfig
config = BenchmarkConfig(test_data_sizes=[100, 500], iterations_per_test=3)
results = run_gpu_benchmarks(config)
print(f'Average throughput: {results.summary[\"throughput_stats\"][\"mean\"]:.2f} items/sec')
"
```

### Benchmark Results Interpretation

Typical RX6800M performance metrics:

- **Throughput**: 500-2000 items/sec (depending on complexity)
- **Memory Usage**: 8-10GB VRAM utilization
- **GPU Utilization**: 70-90% during processing
- **Latency**: 10-50ms per batch

### Performance Monitoring

```python
# Get real-time performance stats
engine = create_gpu_inference_engine()
stats = engine.get_performance_stats()

print(f"Device: {stats['device_info']['device']}")
print(f"Memory utilization: {stats['memory_stats']['utilization_percent']:.1f}%")
print(f"Average throughput: {stats['inference_stats']['average_throughput']:.2f} items/sec")
```

## Memory Management

### Large Dataset Handling

```python
from sanskrit_rewrite_engine.memory_optimization import MemoryOptimizer

optimizer = MemoryOptimizer()

# Store large dataset efficiently
large_dataset = load_large_sanskrit_corpus()  # Your data loading
success = optimizer.store_large_dataset("corpus", large_dataset)

# Load with automatic chunking
for chunk in optimizer.load_large_dataset("corpus"):
    # Process chunk
    results = process_sanskrit_chunk(chunk)
```

### Memory Optimization Strategies

1. **Chunked Loading**: Automatic chunking for large datasets
2. **Compression**: Transparent data compression
3. **Memory Mapping**: Efficient file-based storage
4. **LRU Caching**: Intelligent caching with eviction
5. **Garbage Collection**: Automatic memory cleanup

## Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check ROCm installation
rocm-smi

# Verify PyTorch ROCm support
python -c "import torch; print(torch.version.cuda)"

# Check device visibility
export HIP_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.device_count())"
```

#### Out of Memory Errors

```python
# Reduce batch size
config = GPUConfig(batch_size=16)  # Reduce from default 32

# Enable memory optimization
config.memory_fraction = 0.7  # Reduce from default 0.8

# Enable aggressive garbage collection
memory_config = MemoryConfig(gc_threshold=0.6)
```

#### Performance Issues

```bash
# Enable ROCm optimizations
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1

# Check GPU utilization
rocm-smi -d 0 --showuse

# Run benchmark to identify bottlenecks
python -m sanskrit_rewrite_engine --benchmark --quick
```

### Debug Mode

```bash
# Enable debug logging
python -m sanskrit_rewrite_engine --demo --debug

# Profile memory usage
python -c "
from sanskrit_rewrite_engine.memory_optimization import MemoryOptimizer
optimizer = MemoryOptimizer()
optimizer.start_monitoring()
# ... your code ...
report = optimizer.get_memory_report()
print(report)
"
```

## API Reference

### Core Classes

- **GPUAcceleratedInference**: Main inference engine
- **GPUConfig**: GPU configuration
- **MemoryOptimizer**: Memory management
- **VRAMAwareBatcher**: Intelligent batching
- **AutoScalingManager**: Resource scaling
- **GPUBenchmarkRunner**: Performance benchmarking

### Key Methods

```python
# GPU Inference
engine.process_batch(data, processing_function)
engine.process_large_dataset(data, processing_function)
engine.get_performance_stats()
engine.benchmark_performance(test_data, processing_function)

# Memory Management
optimizer.store_large_dataset(key, data)
optimizer.load_large_dataset(key)
optimizer.optimize_memory_usage()
optimizer.get_memory_report()

# Benchmarking
runner.run_full_benchmark_suite()
runner.get_performance_stats()
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests
python -m pytest tests/test_gpu_acceleration.py -v

# Run benchmarks
python -m sanskrit_rewrite_engine --benchmark --quick

# Format code
black sanskrit_rewrite_engine/
flake8 sanskrit_rewrite_engine/
```

### Adding New Optimizations

1. Extend `GPUConfig` for new configuration options
2. Implement optimization in `GPUDeviceManager`
3. Add tests in `test_gpu_acceleration.py`
4. Update benchmarks in `gpu_benchmarks.py`
5. Document in this README

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check this README and troubleshooting section
2. Run diagnostics: `python -m sanskrit_rewrite_engine --demo --debug`
3. Check GPU status: `rocm-smi`
4. Review logs in `./r_zero_storage/logs/`

## Changelog

### Version 1.0.0

- Initial GPU acceleration implementation
- RX6800M optimization support
- Mixed-precision inference
- VRAM-aware batching
- Memory optimization system
- Containerized deployment
- Comprehensive benchmarking
- Auto-scaling capabilities