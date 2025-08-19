# Sanskrit Rewrite Engine Performance Tuning Guide

## Overview

This guide provides comprehensive performance tuning strategies for the Sanskrit Rewrite Engine, covering optimization techniques, configuration guidelines, and best practices for achieving optimal performance across different use cases.

## Performance Architecture

### Core Components

1. **Rule Cache System**: High-performance LRU cache for rule application results
2. **Parallel Processing**: Multi-threaded rule application for large token streams
3. **Lazy Evaluation**: Deferred token processing for memory efficiency
4. **Memory Optimization**: Advanced memory management with compression and mapping
5. **Performance Profiling**: Bottleneck identification and optimization guidance

### Performance Metrics

- **Processing Time**: Total time for rule application and token processing
- **Cache Hit Rate**: Percentage of rule applications served from cache
- **Memory Usage**: RAM and VRAM utilization patterns
- **Parallel Efficiency**: Effectiveness of multi-threaded processing
- **Bottleneck Analysis**: Identification of performance-limiting components

## Configuration Optimization

### High-Performance Configuration

For maximum processing speed with adequate memory:

```python
from sanskrit_rewrite_engine.performance_optimization import get_high_performance_config

config = get_high_performance_config()
# Enables:
# - Parallel processing with all CPU cores
# - 1GB rule cache
# - Aggressive lazy evaluation
# - Comprehensive profiling
# - Large memoization cache
```

### Memory-Optimized Configuration

For memory-constrained environments:

```python
from sanskrit_rewrite_engine.performance_optimization import get_memory_optimized_config

config = get_memory_optimized_config()
# Optimizes for:
# - Minimal memory footprint
# - Reduced parallel overhead
# - Smaller caches
# - Conservative resource usage
```

### Custom Configuration

```python
from sanskrit_rewrite_engine.performance_optimization import PerformanceConfig

config = PerformanceConfig(
    enable_parallel_processing=True,
    max_worker_threads=4,
    enable_rule_caching=True,
    cache_size_mb=512,
    enable_lazy_evaluation=True,
    enable_profiling=True,
    parallel_threshold=100,
    chunk_size=50,
    cache_persistence=True
)
```

## Optimization Strategies

### 1. Rule Caching

**Purpose**: Avoid recomputing identical rule applications

**Configuration**:
```python
config.enable_rule_caching = True
config.cache_size_mb = 1024  # Adjust based on available memory
config.cache_persistence = True  # Save cache between sessions
```

**Best Practices**:
- Monitor cache hit rate (target: >70%)
- Increase cache size if hit rate is low and memory allows
- Enable persistence for repeated processing of similar texts
- Clear cache periodically to prevent stale entries

### 2. Parallel Processing

**Purpose**: Utilize multiple CPU cores for rule application

**Configuration**:
```python
config.enable_parallel_processing = True
config.max_worker_threads = multiprocessing.cpu_count()
config.parallel_threshold = 50  # Minimum tokens for parallelization
config.chunk_size = 25  # Tokens per parallel chunk
```

**Best Practices**:
- Use parallel processing for token streams >100 tokens
- Adjust chunk size based on rule complexity:
  - Simple rules: larger chunks (50-100 tokens)
  - Complex rules: smaller chunks (10-25 tokens)
- Monitor parallel efficiency (target: >60%)
- Disable for memory-constrained environments

### 3. Lazy Evaluation

**Purpose**: Defer expensive computations until needed

**Configuration**:
```python
config.enable_lazy_evaluation = True
```

**Best Practices**:
- Enable for large token streams (>1000 tokens)
- Combine with caching for maximum benefit
- Monitor memory usage to ensure effectiveness
- Use for streaming or incremental processing

### 4. Memory Optimization

**Purpose**: Efficient memory usage and garbage collection

**Configuration**:
```python
from sanskrit_rewrite_engine.memory_optimization import get_rx6800m_memory_config

memory_config = get_rx6800m_memory_config()
# Optimized for RX6800M GPU with 12GB VRAM
```

**Best Practices**:
- Monitor memory utilization (keep <80%)
- Enable compression for large datasets
- Use memory mapping for persistent data
- Implement regular garbage collection
- Configure swap thresholds appropriately

### 5. Memoization

**Purpose**: Cache expensive function results

**Configuration**:
```python
config.enable_memoization = True
config.memoization_max_size = 10000
```

**Best Practices**:
- Use for computationally expensive functions
- Monitor cache hit rates
- Adjust cache size based on function call patterns
- Clear caches periodically to prevent memory bloat

## Performance Monitoring

### Real-time Monitoring

```python
from sanskrit_rewrite_engine.performance_optimization import PerformanceOptimizer

optimizer = PerformanceOptimizer(config)

# Process tokens with monitoring
processed_tokens = optimizer.optimize_token_processing(tokens, rules, guard_system)

# Get performance report
report = optimizer.get_performance_report()
print(f"Cache hit rate: {report['cache_statistics']['hit_rate']:.2%}")
print(f"Processing time: {report['performance_metrics']['total_processing_time']:.4f}s")
```

### Profiling and Bottleneck Analysis

```python
# Enable profiling
config.enable_profiling = True

# Process with profiling
optimizer = PerformanceOptimizer(config)
result = optimizer.profile_performance("rule_processing", process_function, tokens)

# Analyze bottlenecks
bottlenecks = optimizer.profiler.identify_bottlenecks(threshold_seconds=0.1)
for bottleneck in bottlenecks:
    print(f"Bottleneck: {bottleneck}")
```

### Performance Metrics Dashboard

```python
def print_performance_dashboard(optimizer):
    report = optimizer.get_performance_report()
    
    print("=== Performance Dashboard ===")
    print(f"Total Processing Time: {report['performance_metrics']['total_processing_time']:.4f}s")
    print(f"Cache Hit Rate: {report['cache_statistics']['hit_rate']:.2%}")
    print(f"Memory Usage: {report['memory_report']['memory_stats']['utilization_percent']:.1f}%")
    print(f"Rules Applied: {report['performance_metrics']['rules_applied']}")
    print(f"Tokens Processed: {report['performance_metrics']['tokens_processed']}")
    
    if report['bottlenecks']:
        print("\nBottlenecks:")
        for bottleneck in report['bottlenecks']:
            print(f"  - {bottleneck}")
    
    if report['optimization_suggestions']:
        print("\nOptimization Suggestions:")
        for suggestion in report['optimization_suggestions']:
            print(f"  - {suggestion}")
```

## Use Case Optimization

### 1. Large Corpus Processing

**Scenario**: Processing extensive Sanskrit texts (>10MB)

**Optimizations**:
```python
config = PerformanceConfig(
    enable_parallel_processing=True,
    max_worker_threads=8,
    enable_rule_caching=True,
    cache_size_mb=2048,  # Large cache
    enable_lazy_evaluation=True,
    parallel_threshold=50,  # Lower threshold
    chunk_size=100,  # Larger chunks
    cache_persistence=True
)
```

**Additional Strategies**:
- Use chunked data loading
- Enable memory mapping for persistent storage
- Implement streaming processing
- Monitor memory usage closely

### 2. Real-time Processing

**Scenario**: Interactive Sanskrit analysis with low latency requirements

**Optimizations**:
```python
config = PerformanceConfig(
    enable_parallel_processing=False,  # Reduce overhead
    enable_rule_caching=True,
    cache_size_mb=512,
    enable_lazy_evaluation=False,  # Immediate processing
    enable_memoization=True,
    memoization_max_size=20000,  # Large memoization
    cache_persistence=True  # Pre-warmed cache
)
```

**Additional Strategies**:
- Pre-warm caches with common patterns
- Use smaller, focused rule sets
- Implement result caching at application level
- Optimize rule matching functions

### 3. Memory-Constrained Environments

**Scenario**: Limited RAM/VRAM (e.g., embedded systems)

**Optimizations**:
```python
config = PerformanceConfig(
    enable_parallel_processing=False,
    enable_rule_caching=True,
    cache_size_mb=128,  # Small cache
    enable_lazy_evaluation=True,
    enable_compression=True,
    parallel_threshold=500,  # High threshold
    chunk_size=200,  # Large chunks
    cache_persistence=False  # Reduce I/O
)
```

**Additional Strategies**:
- Enable aggressive compression
- Use memory mapping for large data
- Implement frequent garbage collection
- Process in smaller batches

### 4. GPU-Accelerated Processing

**Scenario**: Leveraging GPU acceleration (RX6800M)

**Optimizations**:
```python
from sanskrit_rewrite_engine.memory_optimization import get_rx6800m_memory_config

memory_config = get_rx6800m_memory_config()
config = PerformanceConfig(
    enable_parallel_processing=True,
    max_worker_threads=6,  # Leave cores for GPU coordination
    enable_rule_caching=True,
    cache_size_mb=1024,
    enable_lazy_evaluation=True,
    parallel_threshold=100,
    chunk_size=50
)
```

**Additional Strategies**:
- Monitor VRAM usage carefully
- Use mixed precision when possible
- Implement VRAM-aware batching
- Enable memory optimization features

## Troubleshooting Performance Issues

### Low Cache Hit Rate (<50%)

**Symptoms**:
- High processing times
- Repeated rule computations
- Low cache utilization

**Solutions**:
1. Increase cache size
2. Review rule patterns for cachability
3. Enable cache persistence
4. Optimize cache key generation

### High Memory Usage (>80%)

**Symptoms**:
- Frequent garbage collection
- System slowdown
- Out of memory errors

**Solutions**:
1. Enable compression
2. Reduce cache sizes
3. Use lazy evaluation
4. Implement memory mapping
5. Process in smaller batches

### Poor Parallel Performance

**Symptoms**:
- No speedup from parallelization
- High thread overhead
- Resource contention

**Solutions**:
1. Increase parallel threshold
2. Optimize chunk size
3. Review rule parallelization compatibility
4. Reduce thread count
5. Check for thread-unsafe operations

### Slow Rule Application

**Symptoms**:
- High rule application time
- Bottlenecks in rule matching
- Poor overall performance

**Solutions**:
1. Optimize rule matching functions
2. Use memoization for expensive operations
3. Review rule complexity
4. Enable profiling to identify specific bottlenecks
5. Consider rule set optimization

## Advanced Optimization Techniques

### 1. Adaptive Configuration

```python
def adaptive_optimization(optimizer, tokens):
    # Analyze workload characteristics
    token_count = len(tokens)
    
    if token_count > 10000:
        # Large corpus optimization
        config = get_high_performance_config()
    elif token_count < 100:
        # Small text optimization
        config = get_memory_optimized_config()
        config.enable_parallel_processing = False
    else:
        # Balanced optimization
        config = PerformanceConfig()
    
    # Apply adaptive configuration
    optimizer.config = config
    return optimizer.optimize_token_processing(tokens, rules, guard_system)
```

### 2. Dynamic Cache Management

```python
def dynamic_cache_management(optimizer):
    cache_stats = optimizer.rule_cache.get_stats()
    
    if cache_stats['hit_rate'] < 0.3:
        # Increase cache size
        optimizer.rule_cache.max_size_bytes *= 2
    elif cache_stats['utilization'] < 0.5:
        # Decrease cache size
        optimizer.rule_cache.max_size_bytes //= 2
    
    # Clear stale entries periodically
    if cache_stats['entries'] > 50000:
        optimizer.rule_cache.clear()
```

### 3. Workload-Specific Optimization

```python
def optimize_for_workload(workload_type):
    if workload_type == "batch_processing":
        return PerformanceConfig(
            enable_parallel_processing=True,
            max_worker_threads=multiprocessing.cpu_count(),
            cache_size_mb=2048,
            parallel_threshold=50
        )
    elif workload_type == "interactive":
        return PerformanceConfig(
            enable_parallel_processing=False,
            cache_size_mb=512,
            enable_memoization=True,
            memoization_max_size=50000
        )
    elif workload_type == "streaming":
        return PerformanceConfig(
            enable_lazy_evaluation=True,
            cache_size_mb=256,
            enable_compression=True,
            chunk_size=25
        )
```

## Performance Testing and Benchmarking

### Benchmark Suite

```python
def run_performance_benchmark(optimizer, test_cases):
    results = {}
    
    for name, tokens in test_cases.items():
        start_time = time.time()
        
        # Process tokens
        processed = optimizer.optimize_token_processing(tokens, rules, guard_system)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Collect metrics
        report = optimizer.get_performance_report()
        
        results[name] = {
            'processing_time': processing_time,
            'tokens_per_second': len(tokens) / processing_time,
            'cache_hit_rate': report['cache_statistics']['hit_rate'],
            'memory_usage_mb': report['memory_report']['memory_stats']['used_memory_gb'] * 1024
        }
    
    return results
```

### Performance Regression Testing

```python
def performance_regression_test(baseline_results, current_results, tolerance=0.1):
    regressions = []
    
    for test_name in baseline_results:
        if test_name not in current_results:
            continue
        
        baseline_time = baseline_results[test_name]['processing_time']
        current_time = current_results[test_name]['processing_time']
        
        if current_time > baseline_time * (1 + tolerance):
            regression_percent = ((current_time - baseline_time) / baseline_time) * 100
            regressions.append(f"{test_name}: {regression_percent:.1f}% slower")
    
    return regressions
```

## Best Practices Summary

### Configuration
1. **Start with preset configurations** for common use cases
2. **Monitor performance metrics** continuously
3. **Adjust based on workload characteristics**
4. **Use adaptive optimization** for varying workloads

### Caching
1. **Enable rule caching** for repeated processing
2. **Monitor cache hit rates** (target >70%)
3. **Use cache persistence** for similar workloads
4. **Clear caches periodically** to prevent staleness

### Parallel Processing
1. **Use for large token streams** (>100 tokens)
2. **Adjust chunk size** based on rule complexity
3. **Monitor parallel efficiency** (target >60%)
4. **Consider memory overhead** in constrained environments

### Memory Management
1. **Monitor memory utilization** (keep <80%)
2. **Enable compression** for large datasets
3. **Use lazy evaluation** for streaming workloads
4. **Implement regular garbage collection**

### Profiling
1. **Enable profiling** during development
2. **Identify bottlenecks** regularly
3. **Optimize based on profiling data**
4. **Test performance regressions**

## Conclusion

Effective performance tuning of the Sanskrit Rewrite Engine requires understanding the workload characteristics, monitoring key metrics, and applying appropriate optimization strategies. Use this guide as a reference for achieving optimal performance across different use cases and environments.

For specific performance issues or advanced optimization needs, consult the detailed API documentation and consider implementing custom optimization strategies based on your specific requirements.