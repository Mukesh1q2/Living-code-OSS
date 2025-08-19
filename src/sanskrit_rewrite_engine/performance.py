"""
Performance optimization utilities for the Sanskrit Rewrite Engine.

This module provides caching, lazy evaluation, rule indexing, memory monitoring,
and processing limits to optimize engine performance.
"""

import time
import threading

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Iterator
from functools import lru_cache, wraps
import hashlib
import re
from weakref import WeakValueDictionary


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring engine performance."""
    
    # Processing metrics
    total_processing_time: float = 0.0
    total_transformations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Memory metrics
    peak_memory_usage: float = 0.0  # MB
    current_memory_usage: float = 0.0  # MB
    
    # Rule application metrics
    rule_application_times: Dict[str, float] = field(default_factory=dict)
    rule_match_times: Dict[str, float] = field(default_factory=dict)
    
    # Timeout and limit metrics
    timeouts_triggered: int = 0
    memory_limits_triggered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_processing_time': self.total_processing_time,
            'total_transformations': self.total_transformations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'peak_memory_usage_mb': self.peak_memory_usage,
            'current_memory_usage_mb': self.current_memory_usage,
            'rule_application_times': dict(self.rule_application_times),
            'rule_match_times': dict(self.rule_match_times),
            'timeouts_triggered': self.timeouts_triggered,
            'memory_limits_triggered': self.memory_limits_triggered
        }


class TransformationCache:
    """LRU cache for transformation results with size and TTL limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize transformation cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, text: str, rules_hash: str, config_hash: str) -> str:
        """Generate cache key from input parameters."""
        combined = f"{text}:{rules_hash}:{config_hash}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
    
    def get(self, text: str, rules_hash: str, config_hash: str) -> Optional[Any]:
        """Get cached transformation result.
        
        Args:
            text: Input text
            rules_hash: Hash of active rules
            config_hash: Hash of configuration
            
        Returns:
            Cached result or None if not found/expired
        """
        with self._lock:
            key = self._generate_key(text, rules_hash, config_hash)
            
            if key not in self._cache or self._is_expired(key):
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def put(self, text: str, rules_hash: str, config_hash: str, result: Any):
        """Store transformation result in cache.
        
        Args:
            text: Input text
            rules_hash: Hash of active rules
            config_hash: Hash of configuration
            result: Transformation result to cache
        """
        with self._lock:
            key = self._generate_key(text, rules_hash, config_hash)
            
            # Remove expired entries
            self._evict_expired()
            
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
                self._timestamps.pop(oldest_key, None)
            
            # Add new entry
            self._cache[key] = result
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'oldest_entry_age': time.time() - min(self._timestamps.values()) if self._timestamps else 0
            }


class RuleIndex:
    """Index for fast rule pattern matching."""
    
    def __init__(self):
        """Initialize rule index."""
        self._pattern_index: Dict[str, List[Any]] = defaultdict(list)
        self._prefix_index: Dict[str, List[Any]] = defaultdict(list)
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._rules_by_id: Dict[str, Any] = {}
    
    def add_rule(self, rule: Any):
        """Add rule to index.
        
        Args:
            rule: Rule to add to index
        """
        self._rules_by_id[rule.id] = rule
        
        # Index by pattern
        self._pattern_index[rule.pattern].append(rule)
        
        # Compile and cache regex pattern
        try:
            self._compiled_patterns[rule.id] = re.compile(rule.pattern)
        except re.error:
            # Invalid regex, will fall back to string matching
            pass
        
        # Index by pattern prefix for fast lookup
        if len(rule.pattern) > 0:
            # Extract literal prefix from pattern
            prefix = self._extract_literal_prefix(rule.pattern)
            if prefix:
                self._prefix_index[prefix].append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove rule from index.
        
        Args:
            rule_id: ID of rule to remove
        """
        if rule_id not in self._rules_by_id:
            return
        
        rule = self._rules_by_id[rule_id]
        
        # Remove from pattern index
        if rule.pattern in self._pattern_index:
            self._pattern_index[rule.pattern] = [
                r for r in self._pattern_index[rule.pattern] if r.id != rule_id
            ]
            if not self._pattern_index[rule.pattern]:
                del self._pattern_index[rule.pattern]
        
        # Remove from prefix index
        prefix = self._extract_literal_prefix(rule.pattern)
        if prefix and prefix in self._prefix_index:
            self._prefix_index[prefix] = [
                r for r in self._prefix_index[prefix] if r.id != rule_id
            ]
            if not self._prefix_index[prefix]:
                del self._prefix_index[prefix]
        
        # Remove compiled pattern
        self._compiled_patterns.pop(rule_id, None)
        
        # Remove from rules
        del self._rules_by_id[rule_id]
    
    def _extract_literal_prefix(self, pattern: str) -> Optional[str]:
        """Extract literal prefix from regex pattern.
        
        Args:
            pattern: Regex pattern
            
        Returns:
            Literal prefix or None
        """
        # Simple heuristic: extract characters before first regex metacharacter
        metacharacters = set(r'.*+?^${}[]|()\/')
        prefix = ""
        
        i = 0
        while i < len(pattern):
            char = pattern[i]
            if char == '\\' and i + 1 < len(pattern):
                # Escaped character
                prefix += pattern[i + 1]
                i += 2
            elif char in metacharacters:
                break
            else:
                prefix += char
                i += 1
        
        return prefix if len(prefix) >= 2 else None
    
    def get_candidate_rules(self, text: str, position: int) -> List[Any]:
        """Get candidate rules that might match at position.
        
        Args:
            text: Text to match against
            position: Position in text
            
        Returns:
            List of candidate rules
        """
        candidates = set()
        remaining_text = text[position:]
        
        # Check prefix index for fast matching
        for prefix_len in range(min(10, len(remaining_text)), 1, -1):
            prefix = remaining_text[:prefix_len]
            if prefix in self._prefix_index:
                candidates.update(self._prefix_index[prefix])
        
        # If no prefix matches found, return all rules (fallback)
        if not candidates:
            candidates.update(self._rules_by_id.values())
        
        return list(candidates)
    
    def fast_match(self, rule_id: str, text: str, position: int) -> Optional[re.Match]:
        """Fast pattern matching using compiled regex.
        
        Args:
            rule_id: Rule ID
            text: Text to match
            position: Position in text
            
        Returns:
            Match object or None
        """
        if rule_id not in self._compiled_patterns:
            return None
        
        pattern = self._compiled_patterns[rule_id]
        return pattern.match(text, position)
    
    def clear(self):
        """Clear all indexed rules."""
        self._pattern_index.clear()
        self._prefix_index.clear()
        self._compiled_patterns.clear()
        self._rules_by_id.clear()


class LazyTextProcessor:
    """Lazy evaluation processor for large text processing."""
    
    def __init__(self, text: str, chunk_size: int = 1000):
        """Initialize lazy text processor.
        
        Args:
            text: Text to process
            chunk_size: Size of text chunks for processing
        """
        self.text = text
        self.chunk_size = chunk_size
        self._chunks: Optional[List[str]] = None
        self._processed_chunks: Dict[int, str] = {}
    
    def _create_chunks(self) -> List[str]:
        """Create text chunks for processing."""
        if self._chunks is not None:
            return self._chunks
        
        chunks = []
        for i in range(0, len(self.text), self.chunk_size):
            chunk = self.text[i:i + self.chunk_size]
            chunks.append(chunk)
        
        self._chunks = chunks
        return chunks
    
    def get_chunk(self, chunk_index: int) -> str:
        """Get text chunk by index.
        
        Args:
            chunk_index: Index of chunk to retrieve
            
        Returns:
            Text chunk
        """
        chunks = self._create_chunks()
        if 0 <= chunk_index < len(chunks):
            return chunks[chunk_index]
        return ""
    
    def process_chunk(self, chunk_index: int, processor: Callable[[str], str]) -> str:
        """Process a specific chunk lazily.
        
        Args:
            chunk_index: Index of chunk to process
            processor: Function to process the chunk
            
        Returns:
            Processed chunk text
        """
        if chunk_index in self._processed_chunks:
            return self._processed_chunks[chunk_index]
        
        chunk = self.get_chunk(chunk_index)
        if chunk:
            processed = processor(chunk)
            self._processed_chunks[chunk_index] = processed
            return processed
        
        return ""
    
    def get_processed_text(self) -> str:
        """Get fully processed text by combining all processed chunks.
        
        Returns:
            Complete processed text
        """
        chunks = self._create_chunks()
        result_parts = []
        
        for i in range(len(chunks)):
            if i in self._processed_chunks:
                result_parts.append(self._processed_chunks[i])
            else:
                result_parts.append(chunks[i])
        
        return "".join(result_parts)
    
    def chunk_count(self) -> int:
        """Get number of chunks.
        
        Returns:
            Number of text chunks
        """
        return len(self._create_chunks())


class MemoryMonitor:
    """Memory usage monitor with limits and alerts."""
    
    def __init__(self, memory_limit_mb: int = 500, check_interval: float = 1.0):
        """Initialize memory monitor.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
            check_interval: Check interval in seconds
        """
        self.memory_limit_mb = memory_limit_mb
        self.check_interval = check_interval
        self.peak_usage = 0.0
        self._last_check = 0.0
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in megabytes
        """
        if not PSUTIL_AVAILABLE or self._process is None:
            return 0.0
            
        try:
            memory_info = self._process.memory_info()
            usage_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
            self.peak_usage = max(self.peak_usage, usage_mb)
            return usage_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit.
        
        Returns:
            True if memory limit is exceeded
        """
        current_time = time.time()
        if current_time - self._last_check < self.check_interval:
            return False
        
        self._last_check = current_time
        current_usage = self.get_current_usage()
        return current_usage > self.memory_limit_mb
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        current = self.get_current_usage()
        return {
            'current_mb': current,
            'peak_mb': self.peak_usage,
            'limit_mb': self.memory_limit_mb,
            'usage_percentage': (current / self.memory_limit_mb) * 100 if self.memory_limit_mb > 0 else 0
        }


class ProcessingLimiter:
    """Processing limits and timeout manager."""
    
    def __init__(self, 
                 max_processing_time: float = 30.0,
                 max_iterations: int = 1000,
                 max_text_length: int = 100000):
        """Initialize processing limiter.
        
        Args:
            max_processing_time: Maximum processing time in seconds
            max_iterations: Maximum number of iterations
            max_text_length: Maximum text length to process
        """
        self.max_processing_time = max_processing_time
        self.max_iterations = max_iterations
        self.max_text_length = max_text_length
        self.start_time: Optional[float] = None
        self.iteration_count = 0
    
    def start_processing(self):
        """Start processing timer."""
        self.start_time = time.time()
        self.iteration_count = 0
    
    def check_timeout(self) -> bool:
        """Check if processing has timed out.
        
        Returns:
            True if timeout exceeded
        """
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed > self.max_processing_time
    
    def check_iteration_limit(self) -> bool:
        """Check if iteration limit is exceeded.
        
        Returns:
            True if iteration limit exceeded
        """
        return self.iteration_count >= self.max_iterations
    
    def check_text_length(self, text: str) -> bool:
        """Check if text length exceeds limit.
        
        Args:
            text: Text to check
            
        Returns:
            True if text length exceeds limit
        """
        return len(text) > self.max_text_length
    
    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration_count += 1
    
    def get_elapsed_time(self) -> float:
        """Get elapsed processing time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'elapsed_time': self.get_elapsed_time(),
            'max_processing_time': self.max_processing_time,
            'iteration_count': self.iteration_count,
            'max_iterations': self.max_iterations,
            'timeout_exceeded': self.check_timeout(),
            'iteration_limit_exceeded': self.check_iteration_limit()
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, 
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 memory_limit_mb: int = 500,
                 chunk_size: int = 1000):
        """Initialize performance optimizer.
        
        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache TTL in seconds
            memory_limit_mb: Memory limit in megabytes
            chunk_size: Text chunk size for lazy processing
        """
        self.cache = TransformationCache(cache_size, cache_ttl)
        self.rule_index = RuleIndex()
        self.memory_monitor = MemoryMonitor(memory_limit_mb)
        self.metrics = PerformanceMetrics()
        self.chunk_size = chunk_size
    
    def create_lazy_processor(self, text: str) -> LazyTextProcessor:
        """Create lazy text processor.
        
        Args:
            text: Text to process lazily
            
        Returns:
            LazyTextProcessor instance
        """
        return LazyTextProcessor(text, self.chunk_size)
    
    def create_limiter(self, max_time: float = 30.0, max_iterations: int = 1000, 
                      max_text_length: int = 100000) -> ProcessingLimiter:
        """Create processing limiter.
        
        Args:
            max_time: Maximum processing time
            max_iterations: Maximum iterations
            max_text_length: Maximum text length
            
        Returns:
            ProcessingLimiter instance
        """
        return ProcessingLimiter(max_time, max_iterations, max_text_length)
    
    def update_metrics(self, processing_time: float, transformations: int, 
                      cache_hit: bool = False):
        """Update performance metrics.
        
        Args:
            processing_time: Time taken for processing
            transformations: Number of transformations applied
            cache_hit: Whether result came from cache
        """
        self.metrics.total_processing_time += processing_time
        self.metrics.total_transformations += transformations
        
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        # Update memory metrics
        self.metrics.current_memory_usage = self.memory_monitor.get_current_usage()
        self.metrics.peak_memory_usage = max(
            self.metrics.peak_memory_usage, 
            self.metrics.current_memory_usage
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        return {
            'metrics': self.metrics.to_dict(),
            'cache_stats': self.cache.get_stats(),
            'memory_stats': self.memory_monitor.get_memory_stats()
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()


def performance_timer(func: Callable) -> Callable:
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            # Store timing information if optimizer is available
            if hasattr(args[0], '_performance_optimizer'):
                optimizer = args[0]._performance_optimizer
                func_name = func.__name__
                if func_name not in optimizer.metrics.rule_application_times:
                    optimizer.metrics.rule_application_times[func_name] = 0.0
                optimizer.metrics.rule_application_times[func_name] += elapsed
    
    return wrapper


def memory_limit_check(limit_mb: int = 500):
    """Decorator to check memory limits.
    
    Args:
        limit_mb: Memory limit in megabytes
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            monitor = MemoryMonitor(limit_mb)
            if monitor.check_memory_limit():
                raise MemoryError(f"Memory usage exceeds limit of {limit_mb}MB")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator