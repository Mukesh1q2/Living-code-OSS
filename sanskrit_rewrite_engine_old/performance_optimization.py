"""
Performance optimization module for Sanskrit Rewrite Engine.

This module implements comprehensive performance optimizations including:
- Parallel rule application where possible
- Memory usage monitoring and optimization  
- Rule caching and memoization strategies
- Token stream processing with lazy evaluation
- Performance profiling with bottleneck identification
- Performance tuning guides and best practices
"""

import time
import threading
import multiprocessing
import concurrent.futures
import functools
import weakref
import cProfile
import pstats
import io
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Iterator, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from pathlib import Path
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
import gc
import sys
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .token import Token, TokenKind
from .rule import SutraRule, RuleRegistry, GuardSystem
from .memory_optimization import MemoryOptimizer, MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    enable_rule_caching: bool = True
    cache_size_mb: int = 512
    enable_lazy_evaluation: bool = True
    enable_profiling: bool = False
    profiling_output_dir: str = "./performance_profiles"
    enable_memoization: bool = True
    memoization_max_size: int = 10000
    parallel_threshold: int = 100  # Minimum tokens for parallel processing
    chunk_size: int = 50  # Tokens per chunk for parallel processing
    enable_optimization_hints: bool = True
    cache_persistence: bool = True
    cache_file: str = "./rule_cache.pkl"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    total_processing_time: float = 0.0
    tokenization_time: float = 0.0
    rule_application_time: float = 0.0
    parallel_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    rules_applied: int = 0
    tokens_processed: int = 0
    parallel_chunks_processed: int = 0
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class RuleCache:
    """High-performance cache for rule applications."""
    
    def __init__(self, max_size_mb: int = 512, persistence_file: Optional[str] = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.persistence_file = persistence_file
        self._lock = threading.RLock()
        
        # Load persistent cache if available
        self._load_persistent_cache()
    
    def _generate_cache_key(self, rule_id: str, tokens: List[Token], index: int) -> str:
        """Generate cache key for rule application."""
        # Create context window around the index
        context_start = max(0, index - 2)
        context_end = min(len(tokens), index + 3)
        context_tokens = tokens[context_start:context_end]
        
        # Create hash from rule and context
        context_str = ''.join(f"{t.text}:{t.kind.value}" for t in context_tokens)
        cache_key = f"{rule_id}:{index-context_start}:{context_str}"
        
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def get(self, rule_id: str, tokens: List[Token], index: int) -> Optional[Tuple[List[Token], int]]:
        """Get cached rule application result."""
        cache_key = self._generate_cache_key(rule_id, tokens, index)
        
        with self._lock:
            if cache_key in self.cache:
                result = self.cache.pop(cache_key)
                self.cache[cache_key] = result  # Move to end
                self.hits += 1
                return result
            else:
                self.misses += 1
                return None
    
    def put(self, rule_id: str, tokens: List[Token], index: int, 
            result: Tuple[List[Token], int]) -> None:
        """Cache rule application result."""
        cache_key = self._generate_cache_key(rule_id, tokens, index)
        
        with self._lock:
            # Estimate size
            result_size = self._estimate_result_size(result)
            
            # Remove existing entry if present
            if cache_key in self.cache:
                old_size = self._estimate_result_size(self.cache[cache_key])
                self.current_size -= old_size
                del self.cache[cache_key]
            
            # Evict entries if necessary
            while (self.current_size + result_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                oldest_key, oldest_value = self.cache.popitem(last=False)
                self.current_size -= self._estimate_result_size(oldest_value)
            
            # Add new entry
            if result_size <= self.max_size_bytes:
                self.cache[cache_key] = result
                self.current_size += result_size
    
    def _estimate_result_size(self, result: Tuple[List[Token], int]) -> int:
        """Estimate size of cached result."""
        tokens, index = result
        return sum(sys.getsizeof(token) for token in tokens) + sys.getsizeof(index)
    
    def _load_persistent_cache(self) -> None:
        """Load cache from persistent storage."""
        if not self.persistence_file or not Path(self.persistence_file).exists():
            return
        
        try:
            with open(self.persistence_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.cache = cached_data.get('cache', OrderedDict())
                self.current_size = cached_data.get('size', 0)
                self.hits = cached_data.get('hits', 0)
                self.misses = cached_data.get('misses', 0)
            
            logger.info(f"Loaded {len(self.cache)} cached entries from {self.persistence_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    def save_persistent_cache(self) -> None:
        """Save cache to persistent storage."""
        if not self.persistence_file:
            return
        
        try:
            cached_data = {
                'cache': self.cache,
                'size': self.current_size,
                'hits': self.hits,
                'misses': self.misses
            }
            
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.info(f"Saved {len(self.cache)} cached entries to {self.persistence_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entries': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'utilization': self.current_size / self.max_size_bytes
            }
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0


class MemoizedFunction:
    """Memoization decorator for expensive functions."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = self._make_key(args, kwargs)
            
            with self._lock:
                if key in self.cache:
                    result = self.cache.pop(key)
                    self.cache[key] = result  # Move to end
                    self.hits += 1
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                self.misses += 1
                
                # Cache result
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove oldest
                
                self.cache[key] = result
                return result
        
        wrapper.cache_info = lambda: {
            'hits': self.hits,
            'misses': self.misses,
            'maxsize': self.max_size,
            'currsize': len(self.cache)
        }
        wrapper.cache_clear = lambda: self.cache.clear()
        
        return wrapper
    
    def _make_key(self, args: Tuple, kwargs: Dict) -> str:
        """Create hashable key from arguments."""
        try:
            # Convert args and kwargs to string representation
            args_str = str(args)
            kwargs_str = str(sorted(kwargs.items()))
            key_str = f"{args_str}:{kwargs_str}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            # Fallback to id-based key
            return str(id(args)) + str(id(kwargs))


def memoize(max_size: int = 1000):
    """Memoization decorator."""
    return MemoizedFunction(max_size)


class LazyTokenStream:
    """Lazy evaluation token stream for memory efficiency."""
    
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._processed_cache = {}
        self._processing_functions = []
    
    def add_processing_function(self, func: Callable[[Token], Token]) -> 'LazyTokenStream':
        """Add a processing function to be applied lazily."""
        self._processing_functions.append(func)
        return self
    
    def __getitem__(self, index: int) -> Token:
        """Get token at index with lazy processing."""
        if index in self._processed_cache:
            return self._processed_cache[index]
        
        if index < 0 or index >= len(self._tokens):
            raise IndexError(f"Token index {index} out of range")
        
        # Apply processing functions lazily
        token = self._tokens[index]
        for func in self._processing_functions:
            token = func(token)
        
        # Cache processed token
        self._processed_cache[index] = token
        return token
    
    def __len__(self) -> int:
        return len(self._tokens)
    
    def __iter__(self) -> Iterator[Token]:
        for i in range(len(self._tokens)):
            yield self[i]
    
    def slice(self, start: int, end: int) -> List[Token]:
        """Get slice of tokens with lazy processing."""
        return [self[i] for i in range(start, end)]
    
    def clear_cache(self) -> None:
        """Clear processing cache."""
        self._processed_cache.clear()


class ParallelRuleProcessor:
    """Parallel processor for rule applications."""
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 50):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.executor = None
    
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def can_parallelize_rule(self, rule: SutraRule, tokens: List[Token]) -> bool:
        """Check if rule can be safely parallelized."""
        # Rules that modify token sequence length cannot be easily parallelized
        # For now, only allow rules that don't change token count
        return (
            len(tokens) >= self.chunk_size * 2 and  # Minimum size for parallelization
            not rule.meta_data.get('modifies_length', True)  # Rule doesn't change token count
        )
    
    def process_chunks_parallel(self, tokens: List[Token], rule: SutraRule, 
                              guard_system: GuardSystem) -> Tuple[List[Token], List[int]]:
        """Process token chunks in parallel."""
        if not self.executor:
            raise RuntimeError("ParallelRuleProcessor not initialized")
        
        # Split tokens into chunks
        chunks = self._create_chunks(tokens)
        
        # Process chunks in parallel
        futures = []
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(
                self._process_chunk,
                chunk, rule, guard_system, i * self.chunk_size
            )
            futures.append(future)
        
        # Collect results
        processed_chunks = []
        applied_positions = []
        
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_result, positions = future.result()
                processed_chunks.append(chunk_result)
                applied_positions.extend(positions)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")
                # Fallback to original chunk
                processed_chunks.append(chunks[len(processed_chunks)])
        
        # Merge chunks back together
        merged_tokens = []
        for chunk in processed_chunks:
            merged_tokens.extend(chunk)
        
        return merged_tokens, applied_positions
    
    def _create_chunks(self, tokens: List[Token]) -> List[List[Token]]:
        """Create token chunks for parallel processing."""
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _process_chunk(self, chunk: List[Token], rule: SutraRule, 
                      guard_system: GuardSystem, offset: int) -> Tuple[List[Token], List[int]]:
        """Process a single chunk."""
        processed_chunk = chunk.copy()
        applied_positions = []
        
        for i in range(len(chunk)):
            absolute_position = offset + i
            
            if guard_system.can_apply_rule(rule, processed_chunk, i):
                try:
                    if rule.match_fn(processed_chunk, i):
                        new_tokens, new_index = rule.apply_fn(processed_chunk, i)
                        processed_chunk = new_tokens
                        applied_positions.append(absolute_position)
                except Exception as e:
                    logger.warning(f"Rule application error in chunk: {e}")
        
        return processed_chunk, applied_positions


class PerformanceProfiler:
    """Performance profiler for bottleneck identification."""
    
    def __init__(self, output_dir: str = "./performance_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profilers = {}
        self.timing_data = defaultdict(list)
        self.active_profiles = {}
    
    def start_profiling(self, profile_name: str) -> None:
        """Start profiling a section."""
        profiler = cProfile.Profile()
        profiler.enable()
        self.profilers[profile_name] = profiler
        self.active_profiles[profile_name] = time.time()
    
    def stop_profiling(self, profile_name: str) -> Optional[str]:
        """Stop profiling and save results."""
        if profile_name not in self.profilers:
            return None
        
        profiler = self.profilers[profile_name]
        profiler.disable()
        
        # Calculate total time
        if profile_name in self.active_profiles:
            total_time = time.time() - self.active_profiles[profile_name]
            self.timing_data[profile_name].append(total_time)
            del self.active_profiles[profile_name]
        
        # Save profile
        output_file = self.output_dir / f"{profile_name}_{int(time.time())}.prof"
        profiler.dump_stats(str(output_file))
        
        # Generate text report
        report_file = self.output_dir / f"{profile_name}_{int(time.time())}.txt"
        with open(report_file, 'w') as f:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            f.write(s.getvalue())
        
        del self.profilers[profile_name]
        
        logger.info(f"Profile saved: {output_file}")
        return str(report_file)
    
    def profile_function(self, func_name: str):
        """Decorator for profiling functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = f"{func_name}_{int(time.time())}"
                self.start_profiling(profile_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop_profiling(profile_name)
            return wrapper
        return decorator
    
    def time_function(self, func_name: str):
        """Decorator for timing functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    self.timing_data[func_name].append(execution_time)
                    logger.debug(f"{func_name} executed in {execution_time:.4f}s")
            return wrapper
        return decorator
    
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all profiled functions."""
        stats = {}
        
        for func_name, times in self.timing_data.items():
            if times:
                stats[func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return stats
    
    def identify_bottlenecks(self, threshold_seconds: float = 0.1) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        stats = self.get_timing_statistics()
        
        for func_name, func_stats in stats.items():
            if func_stats['average_time'] > threshold_seconds:
                bottlenecks.append(
                    f"{func_name}: avg {func_stats['average_time']:.4f}s, "
                    f"max {func_stats['max_time']:.4f}s"
                )
        
        return bottlenecks


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.metrics = PerformanceMetrics()
        
        # Initialize components
        self.rule_cache = RuleCache(
            self.config.cache_size_mb,
            self.config.cache_file if self.config.cache_persistence else None
        )
        
        self.profiler = PerformanceProfiler(self.config.profiling_output_dir)
        self.memory_optimizer = MemoryOptimizer()
        
        # Parallel processor
        self.parallel_processor = None
        
        logger.info("Performance optimizer initialized")
    
    def optimize_rule_application(self, rule: SutraRule, tokens: List[Token], 
                                index: int, guard_system: GuardSystem) -> Tuple[List[Token], int, bool]:
        """Optimized rule application with caching and profiling."""
        start_time = time.time()
        
        # Check cache first
        if self.config.enable_rule_caching:
            cached_result = self.rule_cache.get(rule.id, tokens, index)
            if cached_result is not None:
                self.metrics.cache_hit_rate += 1
                return cached_result[0], cached_result[1], True
        
        # Apply rule
        try:
            new_tokens, new_index = rule.apply_fn(tokens, index)
            
            # Cache result
            if self.config.enable_rule_caching:
                self.rule_cache.put(rule.id, tokens, index, (new_tokens, new_index))
            
            # Update metrics
            self.metrics.rules_applied += 1
            self.metrics.rule_application_time += time.time() - start_time
            
            return new_tokens, new_index, False
            
        except Exception as e:
            logger.error(f"Rule application failed: {e}")
            return tokens, index + 1, False
    
    def optimize_token_processing(self, tokens: List[Token], 
                                rules: List[SutraRule], 
                                guard_system: GuardSystem) -> List[Token]:
        """Optimized token processing with parallel execution."""
        if not tokens:
            return tokens
        
        start_time = time.time()
        
        # Use lazy evaluation for large token streams
        if self.config.enable_lazy_evaluation and len(tokens) > 1000:
            token_stream = LazyTokenStream(tokens)
        else:
            token_stream = tokens
        
        # Determine if parallel processing is beneficial
        if (self.config.enable_parallel_processing and 
            len(tokens) >= self.config.parallel_threshold):
            
            processed_tokens = self._process_tokens_parallel(
                tokens, rules, guard_system
            )
        else:
            processed_tokens = self._process_tokens_sequential(
                tokens, rules, guard_system
            )
        
        # Update metrics
        self.metrics.total_processing_time += time.time() - start_time
        self.metrics.tokens_processed += len(tokens)
        
        return processed_tokens
    
    def _process_tokens_parallel(self, tokens: List[Token], 
                               rules: List[SutraRule], 
                               guard_system: GuardSystem) -> List[Token]:
        """Process tokens using parallel execution."""
        start_time = time.time()
        
        with ParallelRuleProcessor(
            self.config.max_worker_threads, 
            self.config.chunk_size
        ) as processor:
            
            processed_tokens = tokens.copy()
            
            for rule in rules:
                if processor.can_parallelize_rule(rule, processed_tokens):
                    try:
                        processed_tokens, positions = processor.process_chunks_parallel(
                            processed_tokens, rule, guard_system
                        )
                        self.metrics.parallel_chunks_processed += len(positions)
                    except Exception as e:
                        logger.warning(f"Parallel processing failed for rule {rule.name}: {e}")
                        # Fallback to sequential processing
                        processed_tokens = self._apply_rule_sequential(
                            processed_tokens, rule, guard_system
                        )
                else:
                    # Sequential processing for non-parallelizable rules
                    processed_tokens = self._apply_rule_sequential(
                        processed_tokens, rule, guard_system
                    )
        
        self.metrics.parallel_processing_time += time.time() - start_time
        return processed_tokens
    
    def _process_tokens_sequential(self, tokens: List[Token], 
                                 rules: List[SutraRule], 
                                 guard_system: GuardSystem) -> List[Token]:
        """Process tokens sequentially."""
        processed_tokens = tokens.copy()
        
        for rule in rules:
            processed_tokens = self._apply_rule_sequential(
                processed_tokens, rule, guard_system
            )
        
        return processed_tokens
    
    def _apply_rule_sequential(self, tokens: List[Token], 
                             rule: SutraRule, 
                             guard_system: GuardSystem) -> List[Token]:
        """Apply rule sequentially to tokens."""
        current_tokens = tokens
        index = 0
        
        while index < len(current_tokens):
            if guard_system.can_apply_rule(rule, current_tokens, index):
                try:
                    if rule.match_fn(current_tokens, index):
                        new_tokens, new_index, cached = self.optimize_rule_application(
                            rule, current_tokens, index, guard_system
                        )
                        current_tokens = new_tokens
                        index = new_index
                        continue
                except Exception as e:
                    logger.warning(f"Rule application error: {e}")
            
            index += 1
        
        return current_tokens
    
    def profile_performance(self, func_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile function performance."""
        if not self.config.enable_profiling:
            return func(*args, **kwargs)
        
        self.profiler.start_profiling(func_name)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.profiler.stop_profiling(func_name)
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage."""
        resources = {}
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU usage
                resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                resources['memory_percent'] = memory.percent
                resources['memory_available_gb'] = memory.available / (1024**3)
                
                # Process-specific metrics
                process = psutil.Process()
                resources['process_memory_mb'] = process.memory_info().rss / (1024**2)
                resources['process_cpu_percent'] = process.cpu_percent()
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
        
        return resources
    
    def generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on metrics."""
        suggestions = []
        
        # Cache performance
        cache_stats = self.rule_cache.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            suggestions.append(
                f"Low cache hit rate ({cache_stats['hit_rate']:.2%}). "
                "Consider increasing cache size or reviewing rule patterns."
            )
        
        # Memory usage
        memory_report = self.memory_optimizer.get_memory_report()
        memory_util = memory_report['memory_stats']['utilization_percent']
        if memory_util > 80:
            suggestions.append(
                f"High memory utilization ({memory_util:.1f}%). "
                "Consider enabling compression or reducing batch sizes."
            )
        
        # Processing time
        if self.metrics.rule_application_time > self.metrics.total_processing_time * 0.8:
            suggestions.append(
                "Rule application is the main bottleneck. "
                "Consider optimizing rule matching functions or enabling parallel processing."
            )
        
        # Parallel processing
        if (self.config.enable_parallel_processing and 
            self.metrics.parallel_chunks_processed == 0 and 
            self.metrics.tokens_processed > self.config.parallel_threshold):
            suggestions.append(
                "No parallel processing occurred despite large token count. "
                "Review rule parallelization compatibility."
            )
        
        return suggestions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Update metrics
        self.metrics.cache_hit_rate = self.rule_cache.get_stats()['hit_rate']
        
        # System resources
        system_resources = self.monitor_system_resources()
        
        # Bottlenecks
        bottlenecks = self.profiler.identify_bottlenecks()
        
        # Optimization suggestions
        suggestions = self.generate_optimization_suggestions()
        
        report = {
            'performance_metrics': {
                'total_processing_time': self.metrics.total_processing_time,
                'tokenization_time': self.metrics.tokenization_time,
                'rule_application_time': self.metrics.rule_application_time,
                'parallel_processing_time': self.metrics.parallel_processing_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'rules_applied': self.metrics.rules_applied,
                'tokens_processed': self.metrics.tokens_processed,
                'parallel_chunks_processed': self.metrics.parallel_chunks_processed
            },
            'cache_statistics': self.rule_cache.get_stats(),
            'memory_report': self.memory_optimizer.get_memory_report(),
            'system_resources': system_resources,
            'timing_statistics': self.profiler.get_timing_statistics(),
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions,
            'configuration': {
                'parallel_processing_enabled': self.config.enable_parallel_processing,
                'max_worker_threads': self.config.max_worker_threads,
                'rule_caching_enabled': self.config.enable_rule_caching,
                'cache_size_mb': self.config.cache_size_mb,
                'lazy_evaluation_enabled': self.config.enable_lazy_evaluation,
                'profiling_enabled': self.config.enable_profiling
            }
        }
        
        return report
    
    def optimize_configuration(self) -> PerformanceConfig:
        """Automatically optimize configuration based on performance data."""
        optimized_config = PerformanceConfig()
        
        # Analyze current performance
        cache_stats = self.rule_cache.get_stats()
        memory_report = self.memory_optimizer.get_memory_report()
        
        # Optimize cache size
        if cache_stats['hit_rate'] < 0.3 and cache_stats['utilization'] > 0.9:
            optimized_config.cache_size_mb = min(
                self.config.cache_size_mb * 2, 
                2048  # Max 2GB cache
            )
        
        # Optimize parallel processing
        if self.metrics.parallel_processing_time > 0:
            parallel_efficiency = (
                self.metrics.parallel_chunks_processed / 
                max(1, self.metrics.tokens_processed / self.config.chunk_size)
            )
            
            if parallel_efficiency < 0.5:
                optimized_config.max_worker_threads = max(
                    2, self.config.max_worker_threads - 1
                )
            elif parallel_efficiency > 0.8:
                optimized_config.max_worker_threads = min(
                    multiprocessing.cpu_count(), 
                    self.config.max_worker_threads + 1
                )
        
        # Optimize chunk size
        if self.metrics.parallel_chunks_processed > 0:
            avg_chunk_time = (
                self.metrics.parallel_processing_time / 
                self.metrics.parallel_chunks_processed
            )
            
            if avg_chunk_time < 0.001:  # Too small chunks
                optimized_config.chunk_size = min(100, self.config.chunk_size * 2)
            elif avg_chunk_time > 0.1:  # Too large chunks
                optimized_config.chunk_size = max(10, self.config.chunk_size // 2)
        
        logger.info("Configuration optimized based on performance data")
        return optimized_config
    
    def cleanup(self) -> None:
        """Cleanup performance optimizer."""
        logger.info("Cleaning up performance optimizer")
        
        # Save persistent cache
        if self.config.cache_persistence:
            self.rule_cache.save_persistent_cache()
        
        # Cleanup memory optimizer
        self.memory_optimizer.cleanup()
        
        # Final garbage collection
        gc.collect()


def create_performance_optimizer(config: Optional[PerformanceConfig] = None) -> PerformanceOptimizer:
    """Create performance optimizer with configuration."""
    return PerformanceOptimizer(config)


def get_high_performance_config() -> PerformanceConfig:
    """Get high-performance configuration."""
    return PerformanceConfig(
        enable_parallel_processing=True,
        max_worker_threads=multiprocessing.cpu_count(),
        enable_rule_caching=True,
        cache_size_mb=1024,  # 1GB cache
        enable_lazy_evaluation=True,
        enable_profiling=True,
        enable_memoization=True,
        memoization_max_size=50000,
        parallel_threshold=50,  # Lower threshold for more parallelization
        chunk_size=25,  # Smaller chunks for better load balancing
        cache_persistence=True
    )


def get_memory_optimized_config() -> PerformanceConfig:
    """Get memory-optimized configuration."""
    return PerformanceConfig(
        enable_parallel_processing=False,  # Reduce memory overhead
        max_worker_threads=2,
        enable_rule_caching=True,
        cache_size_mb=256,  # Smaller cache
        enable_lazy_evaluation=True,
        enable_profiling=False,  # Reduce memory overhead
        enable_memoization=True,
        memoization_max_size=5000,
        parallel_threshold=200,  # Higher threshold
        chunk_size=100,  # Larger chunks
        cache_persistence=False  # Reduce I/O
    )