"""
GPU acceleration module for Sanskrit Rewrite Engine.

This module provides GPU acceleration capabilities optimized for RX6800M GPU
with VRAM-aware batching, mixed-precision inference, and memory optimization
for large rule sets and Sanskrit corpora.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import psutil

# GPU and ML imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
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
class GPUConfig:
    """Configuration for GPU acceleration."""
    device: str = "auto"  # "auto", "cuda", "rocm", "cpu"
    mixed_precision: bool = True
    batch_size: int = 32
    max_sequence_length: int = 512
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    enable_memory_mapping: bool = True
    enable_gradient_checkpointing: bool = True
    optimization_level: str = "O2"  # Mixed precision optimization level
    rocm_optimization: bool = True  # RX6800M specific optimizations
    auto_scaling: bool = True
    benchmark_mode: bool = False
    
    # RX6800M specific settings
    rx6800m_vram_gb: float = 12.0
    rx6800m_compute_units: int = 60
    rx6800m_base_clock: int = 2105  # MHz
    rx6800m_memory_clock: int = 2000  # MHz


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    total_memory: float = 0.0
    allocated_memory: float = 0.0
    cached_memory: float = 0.0
    free_memory: float = 0.0
    utilization_percent: float = 0.0
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


@dataclass
class BatchProcessingResult:
    """Result of batch processing."""
    outputs: List[Any]
    processing_time: float
    memory_peak: float
    throughput: float  # items per second
    gpu_utilization: float
    errors: List[str] = field(default_factory=list)


class GPUDeviceManager:
    """Manages GPU device detection and configuration."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = None
        self.device_type = None
        self.device_properties = {}
        self._initialize_device()
    
    def _initialize_device(self) -> None:
        """Initialize GPU device."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to CPU")
            self.device = torch.device("cpu") if torch else None
            self.device_type = "cpu"
            return
        
        if self.config.device == "auto":
            self.device, self.device_type = self._auto_detect_device()
        else:
            self.device = torch.device(self.config.device)
            self.device_type = self.config.device
        
        if self.device and self.device.type != "cpu":
            self._configure_gpu_device()
        
        logger.info(f"Initialized device: {self.device} (type: {self.device_type})")
    
    def _auto_detect_device(self) -> Tuple[torch.device, str]:
        """Auto-detect best available device."""
        # Check for ROCm (AMD GPU)
        if torch.cuda.is_available() and "HIP" in str(torch.version.cuda):
            device = torch.device("cuda:0")
            device_type = "rocm"
            logger.info("Detected ROCm (AMD GPU) device")
            return device, device_type
        
        # Check for CUDA (NVIDIA GPU)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_type = "cuda"
            logger.info("Detected CUDA (NVIDIA GPU) device")
            return device, device_type
        
        # Fallback to CPU
        else:
            device = torch.device("cpu")
            device_type = "cpu"
            logger.info("No GPU detected, using CPU")
            return device, device_type
    
    def _configure_gpu_device(self) -> None:
        """Configure GPU device settings."""
        if not torch.cuda.is_available():
            return
        
        # Set memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        # Get device properties
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            self.device_properties = {
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            }
            
            # RX6800M specific optimizations
            if "RX 6800M" in props.name or self.config.rocm_optimization:
                self._apply_rx6800m_optimizations()
    
    def _apply_rx6800m_optimizations(self) -> None:
        """Apply RX6800M specific optimizations."""
        logger.info("Applying RX6800M specific optimizations")
        
        # Set optimal batch sizes for RX6800M
        if self.config.batch_size > 64:
            self.config.batch_size = 64
            logger.info("Adjusted batch size to 64 for RX6800M")
        
        # Enable ROCm specific optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Set memory management for VRAM efficiency
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    def get_memory_stats(self) -> GPUMemoryStats:
        """Get current GPU memory statistics."""
        stats = GPUMemoryStats()
        
        if self.device and self.device.type != "cpu" and torch.cuda.is_available():
            try:
                stats.total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                stats.allocated_memory = torch.cuda.memory_allocated(self.device) / 1e9
                stats.cached_memory = torch.cuda.memory_reserved(self.device) / 1e9
                stats.free_memory = stats.total_memory - stats.allocated_memory
                stats.utilization_percent = (stats.allocated_memory / stats.total_memory) * 100
            except Exception as e:
                logger.warning(f"Failed to get GPU memory stats: {e}")
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device and self.device.type != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")


class MixedPrecisionManager:
    """Manages mixed precision inference for speed optimization."""
    
    def __init__(self, config: GPUConfig, device_manager: GPUDeviceManager):
        self.config = config
        self.device_manager = device_manager
        self.scaler = None
        self.enabled = config.mixed_precision and TORCH_AVAILABLE
        
        if self.enabled:
            self._initialize_mixed_precision()
    
    def _initialize_mixed_precision(self) -> None:
        """Initialize mixed precision components."""
        if torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info(f"Mixed precision enabled with optimization level {self.config.optimization_level}")
        else:
            self.enabled = False
            logger.warning("Mixed precision disabled - CUDA not available")
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.enabled and torch.cuda.is_available():
            return autocast()
        else:
            # Return dummy context manager for CPU
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss):
        """Scale loss for mixed precision training."""
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer):
        """Step optimizer with mixed precision."""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class VRAMAwareBatcher:
    """VRAM-aware batching system for efficient memory usage."""
    
    def __init__(self, config: GPUConfig, device_manager: GPUDeviceManager):
        self.config = config
        self.device_manager = device_manager
        self.current_batch_size = config.batch_size
        self.min_batch_size = 1
        self.max_batch_size = config.batch_size * 2
        self.memory_threshold = 0.9  # 90% memory usage threshold
        
    def get_optimal_batch_size(self, data_size: int, item_memory_estimate: float) -> int:
        """Calculate optimal batch size based on available VRAM."""
        memory_stats = self.device_manager.get_memory_stats()
        available_memory = memory_stats.free_memory
        
        if available_memory <= 0:
            return self.min_batch_size
        
        # Estimate how many items can fit in available memory
        max_items = int(available_memory / item_memory_estimate * 0.8)  # 80% safety margin
        
        # Adjust batch size
        optimal_batch_size = min(max_items, self.max_batch_size, data_size)
        optimal_batch_size = max(optimal_batch_size, self.min_batch_size)
        
        logger.debug(f"Optimal batch size: {optimal_batch_size} (available memory: {available_memory:.2f}GB)")
        return optimal_batch_size
    
    def create_batches(self, data: List[Any], item_memory_estimate: float = 0.1) -> List[List[Any]]:
        """Create memory-aware batches from data."""
        if not data:
            return []
        
        batch_size = self.get_optimal_batch_size(len(data), item_memory_estimate)
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches with size {batch_size}")
        return batches
    
    def adaptive_batch_resize(self, current_memory_usage: float) -> int:
        """Adaptively resize batch based on current memory usage."""
        if current_memory_usage > self.memory_threshold:
            # Reduce batch size
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            logger.info(f"Reduced batch size to {self.current_batch_size} due to high memory usage")
        elif current_memory_usage < 0.6:
            # Increase batch size
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            logger.debug(f"Increased batch size to {self.current_batch_size}")
        
        return self.current_batch_size


class AutoScalingManager:
    """Auto-scaling manager based on processing complexity."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.enabled = config.auto_scaling
        self.complexity_history = []
        self.performance_history = []
        self.scaling_factor = 1.0
        self.min_scaling = 0.5
        self.max_scaling = 2.0
    
    def analyze_complexity(self, data: List[Any]) -> float:
        """Analyze processing complexity of data."""
        if not data:
            return 0.0
        
        # Simple complexity estimation based on data characteristics
        total_complexity = 0.0
        
        for item in data:
            item_complexity = 1.0  # Base complexity
            
            # Adjust based on item characteristics
            if isinstance(item, str):
                item_complexity *= len(item) / 100.0  # Length factor
            elif isinstance(item, dict):
                item_complexity *= len(item) / 10.0  # Dictionary size factor
            elif isinstance(item, list):
                item_complexity *= len(item) / 20.0  # List size factor
            
            total_complexity += item_complexity
        
        avg_complexity = total_complexity / len(data)
        return min(avg_complexity, 10.0)  # Cap at 10.0
    
    def calculate_scaling_factor(self, complexity: float, current_performance: float) -> float:
        """Calculate scaling factor based on complexity and performance."""
        if not self.enabled:
            return 1.0
        
        # Store history
        self.complexity_history.append(complexity)
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.complexity_history) > 10:
            self.complexity_history = self.complexity_history[-10:]
            self.performance_history = self.performance_history[-10:]
        
        # Calculate scaling based on complexity and performance trends
        if complexity > 5.0:
            # High complexity - scale up resources
            target_scaling = min(self.max_scaling, 1.0 + (complexity - 5.0) / 10.0)
        elif complexity < 2.0:
            # Low complexity - scale down resources
            target_scaling = max(self.min_scaling, 1.0 - (2.0 - complexity) / 4.0)
        else:
            # Medium complexity - maintain current scaling
            target_scaling = self.scaling_factor
        
        # Smooth scaling changes
        self.scaling_factor = 0.7 * self.scaling_factor + 0.3 * target_scaling
        
        logger.debug(f"Complexity: {complexity:.2f}, Scaling factor: {self.scaling_factor:.2f}")
        return self.scaling_factor
    
    def apply_scaling(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scaling to configuration."""
        if not self.enabled:
            return base_config
        
        scaled_config = base_config.copy()
        
        # Scale batch size
        if 'batch_size' in scaled_config:
            scaled_config['batch_size'] = int(scaled_config['batch_size'] * self.scaling_factor)
        
        # Scale worker count
        if 'num_workers' in scaled_config:
            scaled_config['num_workers'] = max(1, int(scaled_config['num_workers'] * self.scaling_factor))
        
        # Scale memory allocation
        if 'memory_fraction' in scaled_config:
            scaled_config['memory_fraction'] = min(0.95, scaled_config['memory_fraction'] * self.scaling_factor)
        
        return scaled_config


class GPUAcceleratedInference:
    """Main GPU-accelerated inference engine."""
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.device_manager = GPUDeviceManager(self.config)
        self.mixed_precision = MixedPrecisionManager(self.config, self.device_manager)
        self.batcher = VRAMAwareBatcher(self.config, self.device_manager)
        self.auto_scaler = AutoScalingManager(self.config)
        
        # Performance tracking
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'average_throughput': 0.0,
            'memory_peak': 0.0
        }
        
        logger.info("GPU accelerated inference engine initialized")
    
    def process_batch(self, batch_data: List[Any], 
                     processing_function: callable,
                     **kwargs) -> BatchProcessingResult:
        """Process a batch of data with GPU acceleration."""
        start_time = time.time()
        initial_memory = self.device_manager.get_memory_stats()
        
        try:
            # Analyze complexity for auto-scaling
            complexity = self.auto_scaler.analyze_complexity(batch_data)
            
            # Apply auto-scaling
            scaled_config = self.auto_scaler.apply_scaling({
                'batch_size': len(batch_data),
                'memory_fraction': self.config.memory_fraction
            })
            
            # Process with mixed precision if enabled
            with self.mixed_precision.autocast_context():
                outputs = processing_function(batch_data, **kwargs)
            
            # Calculate performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            final_memory = self.device_manager.get_memory_stats()
            memory_peak = max(initial_memory.allocated_memory, final_memory.allocated_memory)
            throughput = len(batch_data) / processing_time if processing_time > 0 else 0
            
            # Update auto-scaler with performance
            self.auto_scaler.calculate_scaling_factor(complexity, throughput)
            
            # Update stats
            self.inference_stats['total_inferences'] += len(batch_data)
            self.inference_stats['total_time'] += processing_time
            self.inference_stats['memory_peak'] = max(self.inference_stats['memory_peak'], memory_peak)
            
            if self.inference_stats['total_time'] > 0:
                self.inference_stats['average_throughput'] = (
                    self.inference_stats['total_inferences'] / self.inference_stats['total_time']
                )
            
            return BatchProcessingResult(
                outputs=outputs,
                processing_time=processing_time,
                memory_peak=memory_peak,
                throughput=throughput,
                gpu_utilization=final_memory.utilization_percent
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return BatchProcessingResult(
                outputs=[],
                processing_time=time.time() - start_time,
                memory_peak=0.0,
                throughput=0.0,
                gpu_utilization=0.0,
                errors=[str(e)]
            )
        finally:
            # Clear cache to free memory
            self.device_manager.clear_cache()
    
    def process_large_dataset(self, data: List[Any], 
                            processing_function: callable,
                            item_memory_estimate: float = 0.1,
                            **kwargs) -> List[BatchProcessingResult]:
        """Process large dataset with automatic batching."""
        if not data:
            return []
        
        # Create memory-aware batches
        batches = self.batcher.create_batches(data, item_memory_estimate)
        results = []
        
        logger.info(f"Processing {len(data)} items in {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} (size: {len(batch)})")
            
            result = self.process_batch(batch, processing_function, **kwargs)
            results.append(result)
            
            # Adaptive batch resizing based on memory usage
            if result.gpu_utilization > 0:
                self.batcher.adaptive_batch_resize(result.gpu_utilization / 100.0)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        memory_stats = self.device_manager.get_memory_stats()
        
        return {
            'device_info': {
                'device': str(self.device_manager.device),
                'device_type': self.device_manager.device_type,
                'device_properties': self.device_manager.device_properties
            },
            'memory_stats': {
                'total_memory_gb': memory_stats.total_memory,
                'allocated_memory_gb': memory_stats.allocated_memory,
                'free_memory_gb': memory_stats.free_memory,
                'utilization_percent': memory_stats.utilization_percent
            },
            'inference_stats': self.inference_stats.copy(),
            'config': {
                'mixed_precision': self.mixed_precision.enabled,
                'batch_size': self.config.batch_size,
                'auto_scaling': self.auto_scaler.enabled,
                'current_scaling_factor': self.auto_scaler.scaling_factor
            }
        }
    
    def benchmark_performance(self, test_data: List[Any], 
                            processing_function: callable,
                            iterations: int = 5) -> Dict[str, Any]:
        """Benchmark performance with test data."""
        logger.info(f"Running performance benchmark with {iterations} iterations")
        
        benchmark_results = {
            'iterations': iterations,
            'data_size': len(test_data),
            'results': [],
            'average_throughput': 0.0,
            'average_memory_usage': 0.0,
            'average_gpu_utilization': 0.0
        }
        
        for i in range(iterations):
            logger.debug(f"Benchmark iteration {i+1}/{iterations}")
            
            # Clear cache before each iteration
            self.device_manager.clear_cache()
            
            # Process data
            results = self.process_large_dataset(test_data, processing_function)
            
            # Aggregate results
            total_throughput = sum(r.throughput for r in results)
            avg_memory = sum(r.memory_peak for r in results) / len(results) if results else 0
            avg_gpu_util = sum(r.gpu_utilization for r in results) / len(results) if results else 0
            
            iteration_result = {
                'iteration': i + 1,
                'throughput': total_throughput,
                'memory_peak_gb': avg_memory,
                'gpu_utilization_percent': avg_gpu_util,
                'batch_count': len(results)
            }
            
            benchmark_results['results'].append(iteration_result)
        
        # Calculate averages
        if benchmark_results['results']:
            benchmark_results['average_throughput'] = sum(
                r['throughput'] for r in benchmark_results['results']
            ) / len(benchmark_results['results'])
            
            benchmark_results['average_memory_usage'] = sum(
                r['memory_peak_gb'] for r in benchmark_results['results']
            ) / len(benchmark_results['results'])
            
            benchmark_results['average_gpu_utilization'] = sum(
                r['gpu_utilization_percent'] for r in benchmark_results['results']
            ) / len(benchmark_results['results'])
        
        logger.info(f"Benchmark completed. Average throughput: {benchmark_results['average_throughput']:.2f} items/sec")
        return benchmark_results


def create_gpu_inference_engine(config: Optional[GPUConfig] = None) -> GPUAcceleratedInference:
    """Create GPU-accelerated inference engine."""
    return GPUAcceleratedInference(config)


def get_optimal_gpu_config() -> GPUConfig:
    """Get optimal GPU configuration for current system."""
    config = GPUConfig()
    
    # Detect system capabilities
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Adjust config based on GPU memory
        if gpu_memory >= 12.0:  # RX6800M or similar
            config.batch_size = 64
            config.memory_fraction = 0.8
            config.rocm_optimization = True
        elif gpu_memory >= 8.0:
            config.batch_size = 32
            config.memory_fraction = 0.75
        else:
            config.batch_size = 16
            config.memory_fraction = 0.7
    else:
        # CPU fallback
        config.device = "cpu"
        config.mixed_precision = False
        config.batch_size = 8
    
    return config