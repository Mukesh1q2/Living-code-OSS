"""
Memory optimization module for Sanskrit Rewrite Engine.

This module provides memory optimization strategies for handling large rule sets
and Sanskrit corpora efficiently, with special focus on RX6800M VRAM constraints.
"""

import os
import sys
import gc
import mmap
import pickle
import logging
import threading
import weakref
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict, defaultdict
import time
import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    max_memory_gb: float = 12.0  # RX6800M VRAM limit
    cache_size_mb: int = 2048  # 2GB cache
    enable_memory_mapping: bool = True
    enable_lazy_loading: bool = True
    enable_compression: bool = True
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    chunk_size: int = 1000  # Items per chunk
    prefetch_size: int = 3  # Number of chunks to prefetch
    enable_swap: bool = True
    swap_threshold: float = 0.9  # Swap to disk at 90% memory
    compression_level: int = 6  # Compression level (1-9)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    cached_memory_gb: float = 0.0
    swap_memory_gb: float = 0.0
    fragmentation_ratio: float = 0.0
    gc_collections: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    disk_reads: int = 0
    disk_writes: int = 0


class LRUCache:
    """Least Recently Used cache with memory awareness."""
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            # Estimate size
            item_size = self._estimate_size(value)
            
            # Remove existing key if present
            if key in self.cache:
                old_size = self._estimate_size(self.cache[key])
                self.current_size -= old_size
                del self.cache[key]
            
            # Evict items if necessary
            while (self.current_size + item_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                oldest_key, oldest_value = self.cache.popitem(last=False)
                self.current_size -= self._estimate_size(oldest_value)
            
            # Add new item
            if item_size <= self.max_size_bytes:
                self.cache[key] = value
                self.current_size += item_size
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            elif isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in obj.items())
            else:
                return sys.getsizeof(obj)
        except:
            return 1024  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'items': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'utilization': self.current_size / self.max_size_bytes
            }


class ChunkedDataLoader:
    """Chunked data loader for large datasets."""
    
    def __init__(self, data_path: str, chunk_size: int = 1000, 
                 prefetch_size: int = 3):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.prefetch_size = prefetch_size
        self.chunks_info = []
        self.current_chunk = 0
        self.prefetch_queue = Queue(maxsize=prefetch_size)
        self.prefetch_thread = None
        self.prefetching = False
        
        self._analyze_data()
    
    def _analyze_data(self) -> None:
        """Analyze data file and create chunk information."""
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return
        
        try:
            # For JSON files
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    total_items = len(data)
                    num_chunks = (total_items + self.chunk_size - 1) // self.chunk_size
                    
                    for i in range(num_chunks):
                        start_idx = i * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, total_items)
                        
                        self.chunks_info.append({
                            'chunk_id': i,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'size': end_idx - start_idx
                        })
            
            logger.info(f"Created {len(self.chunks_info)} chunks from {self.data_path}")
            
        except Exception as e:
            logger.error(f"Failed to analyze data file: {e}")
    
    def start_prefetching(self) -> None:
        """Start prefetching chunks."""
        if self.prefetching:
            return
        
        self.prefetching = True
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()
    
    def stop_prefetching(self) -> None:
        """Stop prefetching."""
        self.prefetching = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self) -> None:
        """Prefetch worker thread."""
        while self.prefetching:
            try:
                # Check if queue has space
                if self.prefetch_queue.qsize() < self.prefetch_size:
                    # Calculate next chunk to prefetch
                    next_chunk_id = (self.current_chunk + 
                                   self.prefetch_queue.qsize()) % len(self.chunks_info)
                    
                    # Load chunk
                    chunk_data = self._load_chunk(next_chunk_id)
                    if chunk_data:
                        self.prefetch_queue.put((next_chunk_id, chunk_data), timeout=1.0)
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                logger.warning(f"Prefetch error: {e}")
                time.sleep(1.0)
    
    def _load_chunk(self, chunk_id: int) -> Optional[List[Any]]:
        """Load a specific chunk."""
        if chunk_id >= len(self.chunks_info):
            return None
        
        chunk_info = self.chunks_info[chunk_id]
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data[chunk_info['start_idx']:chunk_info['end_idx']]
            
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_id}: {e}")
        
        return None
    
    def get_chunk(self, chunk_id: int) -> Optional[List[Any]]:
        """Get chunk by ID."""
        # Try prefetch queue first
        try:
            while not self.prefetch_queue.empty():
                queued_id, queued_data = self.prefetch_queue.get_nowait()
                if queued_id == chunk_id:
                    return queued_data
        except Empty:
            pass
        
        # Load directly
        return self._load_chunk(chunk_id)
    
    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over chunks."""
        self.start_prefetching()
        
        try:
            for chunk_id in range(len(self.chunks_info)):
                chunk_data = self.get_chunk(chunk_id)
                if chunk_data:
                    yield chunk_data
                self.current_chunk = chunk_id
        finally:
            self.stop_prefetching()


class MemoryMappedStorage:
    """Memory-mapped storage for large data structures."""
    
    def __init__(self, storage_path: str, max_size_gb: float = 4.0):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.mapped_files = {}
        self.file_handles = {}
    
    def store_data(self, key: str, data: Any) -> bool:
        """Store data with memory mapping."""
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)
            
            if data_size > self.max_size_bytes:
                logger.warning(f"Data too large for memory mapping: {data_size} bytes")
                return False
            
            # Create file
            file_path = self.storage_path / f"{key}.mmap"
            
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            # Create memory mapping
            file_handle = open(file_path, 'rb')
            mapped_data = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
            self.file_handles[key] = file_handle
            self.mapped_files[key] = mapped_data
            
            logger.debug(f"Stored {data_size} bytes for key '{key}' with memory mapping")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data with memory mapping: {e}")
            return False
    
    def load_data(self, key: str) -> Optional[Any]:
        """Load data from memory mapping."""
        try:
            if key not in self.mapped_files:
                return None
            
            mapped_data = self.mapped_files[key]
            data = pickle.loads(mapped_data[:])
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from memory mapping: {e}")
            return None
    
    def remove_data(self, key: str) -> bool:
        """Remove data from memory mapping."""
        try:
            if key in self.mapped_files:
                self.mapped_files[key].close()
                del self.mapped_files[key]
            
            if key in self.file_handles:
                self.file_handles[key].close()
                del self.file_handles[key]
            
            # Remove file
            file_path = self.storage_path / f"{key}.mmap"
            if file_path.exists():
                file_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove memory mapped data: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup all memory mappings."""
        for key in list(self.mapped_files.keys()):
            self.remove_data(key)


class CompressedStorage:
    """Compressed storage for memory optimization."""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.compressed_data = {}
        self.original_sizes = {}
        
        # Try to import compression libraries
        self.compression_available = False
        try:
            import zlib
            self.compressor = zlib
            self.compression_available = True
        except ImportError:
            logger.warning("Compression not available")
    
    def compress_data(self, key: str, data: Any) -> bool:
        """Compress and store data."""
        if not self.compression_available:
            return False
        
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            original_size = len(serialized_data)
            
            # Compress
            compressed_data = self.compressor.compress(
                serialized_data, 
                self.compression_level
            )
            compressed_size = len(compressed_data)
            
            # Store
            self.compressed_data[key] = compressed_data
            self.original_sizes[key] = original_size
            
            compression_ratio = compressed_size / original_size
            logger.debug(f"Compressed '{key}': {original_size} -> {compressed_size} bytes "
                        f"(ratio: {compression_ratio:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            return False
    
    def decompress_data(self, key: str) -> Optional[Any]:
        """Decompress and return data."""
        if not self.compression_available or key not in self.compressed_data:
            return None
        
        try:
            # Decompress
            compressed_data = self.compressed_data[key]
            decompressed_data = self.compressor.decompress(compressed_data)
            
            # Deserialize
            data = pickle.loads(decompressed_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            return None
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compressed_data:
            return {}
        
        total_original = sum(self.original_sizes.values())
        total_compressed = sum(len(data) for data in self.compressed_data.values())
        
        return {
            'items': len(self.compressed_data),
            'original_size_mb': total_original / (1024 * 1024),
            'compressed_size_mb': total_compressed / (1024 * 1024),
            'compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'space_saved_mb': (total_original - total_compressed) / (1024 * 1024)
        }


class MemoryOptimizer:
    """Main memory optimization manager."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.cache = LRUCache(self.config.cache_size_mb)
        self.compressed_storage = CompressedStorage(self.config.compression_level)
        self.memory_mapped_storage = None
        self.chunked_loaders = {}
        
        # Memory monitoring
        self.memory_stats = MemoryStats()
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize memory optimization components."""
        if self.config.enable_memory_mapping:
            storage_path = "./memory_mapped_storage"
            self.memory_mapped_storage = MemoryMappedStorage(
                storage_path, 
                self.config.max_memory_gb / 2  # Use half for memory mapping
            )
        
        logger.info("Memory optimizer initialized")
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self) -> None:
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                self._update_memory_stats()
                
                # Check if GC is needed
                if self.memory_stats.used_memory_gb / self.config.max_memory_gb > self.config.gc_threshold:
                    self._trigger_garbage_collection()
                
                # Check if swap is needed
                if (self.memory_stats.used_memory_gb / self.config.max_memory_gb > 
                    self.config.swap_threshold):
                    self._trigger_swap_to_disk()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(10.0)
    
    def _update_memory_stats(self) -> None:
        """Update memory statistics."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                self.memory_stats.total_memory_gb = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
                self.memory_stats.used_memory_gb = (
                    torch.cuda.memory_allocated(0) / 1e9
                )
                self.memory_stats.cached_memory_gb = (
                    torch.cuda.memory_reserved(0) / 1e9
                )
            except Exception:
                pass
        
        # Update cache stats
        cache_stats = self.cache.get_stats()
        self.memory_stats.cache_hits = cache_stats['hits']
        self.memory_stats.cache_misses = cache_stats['misses']
    
    def _trigger_garbage_collection(self) -> None:
        """Trigger garbage collection."""
        logger.debug("Triggering garbage collection")
        
        # Python GC
        collected = gc.collect()
        self.memory_stats.gc_collections += 1
        
        # PyTorch cache cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug(f"Garbage collection completed, collected {collected} objects")
    
    def _trigger_swap_to_disk(self) -> None:
        """Trigger swap to disk for memory relief."""
        if not self.config.enable_swap:
            return
        
        logger.debug("Triggering swap to disk")
        
        # Clear cache to free memory
        self.cache.clear()
        
        # Force garbage collection
        self._trigger_garbage_collection()
    
    def store_large_dataset(self, key: str, data: List[Any]) -> bool:
        """Store large dataset with optimization."""
        try:
            # Estimate memory usage
            estimated_size = self._estimate_data_size(data)
            
            if estimated_size > self.config.max_memory_gb * 0.5:
                # Use chunked storage for very large datasets
                return self._store_as_chunks(key, data)
            elif self.config.enable_compression:
                # Use compression for medium datasets
                return self.compressed_storage.compress_data(key, data)
            elif self.config.enable_memory_mapping:
                # Use memory mapping
                return self.memory_mapped_storage.store_data(key, data)
            else:
                # Store in cache
                self.cache.put(key, data)
                return True
                
        except Exception as e:
            logger.error(f"Failed to store large dataset: {e}")
            return False
    
    def load_large_dataset(self, key: str) -> Optional[Any]:
        """Load large dataset with optimization."""
        try:
            # Try cache first
            data = self.cache.get(key)
            if data is not None:
                return data
            
            # Try compressed storage
            data = self.compressed_storage.decompress_data(key)
            if data is not None:
                return data
            
            # Try memory mapped storage
            if self.memory_mapped_storage:
                data = self.memory_mapped_storage.load_data(key)
                if data is not None:
                    return data
            
            # Try chunked loader
            if key in self.chunked_loaders:
                return self.chunked_loaders[key]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load large dataset: {e}")
            return None
    
    def _store_as_chunks(self, key: str, data: List[Any]) -> bool:
        """Store data as chunks."""
        try:
            # Create temporary file
            temp_file = Path(f"./temp_{key}.json")
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            # Create chunked loader
            loader = ChunkedDataLoader(
                str(temp_file),
                self.config.chunk_size,
                self.config.prefetch_size
            )
            
            self.chunked_loaders[key] = loader
            
            logger.info(f"Stored {len(data)} items as chunks for key '{key}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store as chunks: {e}")
            return False
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in GB."""
        try:
            if isinstance(data, list):
                # Sample estimation
                if len(data) > 100:
                    sample_size = sys.getsizeof(data[:100])
                    estimated_total = sample_size * len(data) / 100
                else:
                    estimated_total = sys.getsizeof(data)
            else:
                estimated_total = sys.getsizeof(data)
            
            return estimated_total / 1e9  # Convert to GB
            
        except Exception:
            return 1.0  # Default estimate
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize current memory usage."""
        logger.info("Optimizing memory usage")
        
        optimization_results = {
            'actions_taken': [],
            'memory_freed_gb': 0.0,
            'cache_cleared': False,
            'gc_triggered': False
        }
        
        initial_memory = self.memory_stats.used_memory_gb
        
        # Clear cache if memory usage is high
        if self.memory_stats.used_memory_gb / self.config.max_memory_gb > 0.8:
            self.cache.clear()
            optimization_results['cache_cleared'] = True
            optimization_results['actions_taken'].append('cache_cleared')
        
        # Trigger garbage collection
        self._trigger_garbage_collection()
        optimization_results['gc_triggered'] = True
        optimization_results['actions_taken'].append('garbage_collection')
        
        # Update stats
        self._update_memory_stats()
        final_memory = self.memory_stats.used_memory_gb
        optimization_results['memory_freed_gb'] = initial_memory - final_memory
        
        logger.info(f"Memory optimization completed, freed {optimization_results['memory_freed_gb']:.2f} GB")
        return optimization_results
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        self._update_memory_stats()
        
        report = {
            'memory_stats': {
                'total_memory_gb': self.memory_stats.total_memory_gb,
                'used_memory_gb': self.memory_stats.used_memory_gb,
                'cached_memory_gb': self.memory_stats.cached_memory_gb,
                'utilization_percent': (
                    self.memory_stats.used_memory_gb / self.memory_stats.total_memory_gb * 100
                    if self.memory_stats.total_memory_gb > 0 else 0
                )
            },
            'cache_stats': self.cache.get_stats(),
            'compression_stats': self.compressed_storage.get_compression_stats(),
            'chunked_loaders': len(self.chunked_loaders),
            'gc_collections': self.memory_stats.gc_collections,
            'config': {
                'max_memory_gb': self.config.max_memory_gb,
                'cache_size_mb': self.config.cache_size_mb,
                'compression_enabled': self.config.enable_compression,
                'memory_mapping_enabled': self.config.enable_memory_mapping
            }
        }
        
        return report
    
    def cleanup(self) -> None:
        """Cleanup memory optimizer."""
        logger.info("Cleaning up memory optimizer")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clear cache
        self.cache.clear()
        
        # Cleanup memory mapped storage
        if self.memory_mapped_storage:
            self.memory_mapped_storage.cleanup()
        
        # Stop chunked loaders
        for loader in self.chunked_loaders.values():
            loader.stop_prefetching()
        
        # Final garbage collection
        self._trigger_garbage_collection()


def create_memory_optimizer(config: Optional[MemoryConfig] = None) -> MemoryOptimizer:
    """Create memory optimizer with configuration."""
    return MemoryOptimizer(config)


def get_rx6800m_memory_config() -> MemoryConfig:
    """Get optimized memory configuration for RX6800M."""
    return MemoryConfig(
        max_memory_gb=12.0,  # RX6800M VRAM
        cache_size_mb=2048,  # 2GB cache
        enable_memory_mapping=True,
        enable_lazy_loading=True,
        enable_compression=True,
        gc_threshold=0.75,   # More aggressive GC for limited VRAM
        chunk_size=500,      # Smaller chunks for VRAM efficiency
        prefetch_size=2,     # Limited prefetch for VRAM
        enable_swap=True,
        swap_threshold=0.85, # Swap earlier for VRAM
        compression_level=6
    )