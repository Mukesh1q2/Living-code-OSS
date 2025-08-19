"""
GPU performance benchmarks and utilization tests for Sanskrit Rewrite Engine.

This module provides comprehensive benchmarking capabilities for GPU acceleration,
including memory optimization tests, throughput measurements, and RX6800M specific
performance analysis.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import profile, record_function, ProfilerActivity
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

from .gpu_acceleration import GPUAcceleratedInference, GPUConfig, GPUMemoryStats
from .tokenizer import SanskritTokenizer
from .panini_engine import PaniniRuleEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tests."""
    test_data_sizes: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    sequence_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    iterations_per_test: int = 5
    warmup_iterations: int = 2
    memory_stress_test: bool = True
    throughput_test: bool = True
    latency_test: bool = True
    scaling_test: bool = True
    profiling_enabled: bool = True
    save_results: bool = True
    results_path: str = "./benchmark_results"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    test_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    memory_stats: GPUMemoryStats
    execution_time: float
    throughput: float
    errors: List[str] = field(default_factory=list)
    profiling_data: Optional[Dict] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    timestamp: str
    system_info: Dict[str, Any]
    gpu_info: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class SystemProfiler:
    """System resource profiler for benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.stats_history = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True
        self.stats_history = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.debug("System monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.stats_history:
            return {}
        
        # Calculate statistics
        cpu_usage = [s['cpu_percent'] for s in self.stats_history]
        memory_usage = [s['memory_percent'] for s in self.stats_history]
        
        stats = {
            'duration': len(self.stats_history),
            'cpu_usage': {
                'mean': statistics.mean(cpu_usage),
                'max': max(cpu_usage),
                'min': min(cpu_usage),
                'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(memory_usage),
                'max': max(memory_usage),
                'min': min(memory_usage),
                'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
            },
            'samples': len(self.stats_history)
        }
        
        logger.debug("System monitoring stopped")
        return stats
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        while self.monitoring:
            try:
                stats = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available': psutil.virtual_memory().available / 1e9
                }
                self.stats_history.append(stats)
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break


class SanskritBenchmarkData:
    """Generator for Sanskrit benchmark data."""
    
    def __init__(self):
        self.tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize Sanskrit tokenizer."""
        try:
            from .tokenizer import SanskritTokenizer
            self.tokenizer = SanskritTokenizer()
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
    
    def generate_sanskrit_texts(self, count: int, avg_length: int = 100) -> List[str]:
        """Generate Sanskrit text samples for benchmarking."""
        # Sample Sanskrit texts of varying complexity
        base_texts = [
            "dharma artha kāma mokṣa",
            "satyam eva jayate",
            "vasudhaiva kuṭumbakam",
            "ahiṃsā paramo dharmaḥ",
            "vidyā dadāti vinayam",
            "rāma kṛṣṇa govinda",
            "oṃ namaḥ śivāya",
            "gaṅgā yamunā sarasvatī",
            "sūrya candra agni vāyu",
            "brahmā viṣṇu maheśvara"
        ]
        
        # Compound words for complexity testing
        compound_texts = [
            "rājaputra mahārāja",
            "devālaya grāmavāsa",
            "nīlotpala śubhakāma",
            "cakrapāṇi gajavaktra",
            "padmākṣa śaṅkarācārya"
        ]
        
        # Complex grammatical constructions
        complex_texts = [
            "gacchati karoti bhavati paśyati śṛṇoti",
            "rāmasya lakṣmaṇasya sītāyāḥ ca",
            "dharmeṇa arthena kāmena mokṣeṇa ca",
            "vidyayā vinayena satyena ahiṃsayā ca"
        ]
        
        all_texts = base_texts + compound_texts + complex_texts
        generated_texts = []
        
        for i in range(count):
            # Select base text
            base_text = all_texts[i % len(all_texts)]
            
            # Extend to desired length
            text = base_text
            while len(text) < avg_length:
                additional = all_texts[(i + len(text)) % len(all_texts)]
                text += " " + additional
            
            # Trim to approximate length
            if len(text) > avg_length * 1.2:
                text = text[:avg_length]
            
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_rule_application_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate rule application test data."""
        rule_patterns = [
            {"input": "a + i", "expected": "e", "rule_type": "sandhi"},
            {"input": "a + u", "expected": "o", "rule_type": "sandhi"},
            {"input": "rāma + asya", "expected": "rāmasya", "rule_type": "sandhi"},
            {"input": "gam + ti", "expected": "gacchati", "rule_type": "morphology"},
            {"input": "kar + tavya", "expected": "kartavya", "rule_type": "morphology"},
            {"input": "rāja + putra", "expected": "rājaputra", "rule_type": "compound"},
            {"input": "deva + ālaya", "expected": "devālaya", "rule_type": "compound"}
        ]
        
        data = []
        for i in range(count):
            pattern = rule_patterns[i % len(rule_patterns)]
            data.append({
                "id": f"rule_test_{i}",
                "input_text": pattern["input"],
                "expected_output": pattern["expected"],
                "rule_type": pattern["rule_type"],
                "complexity": len(pattern["input"].split()) + len(pattern["expected"])
            })
        
        return data
    
    def generate_mixed_complexity_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate data with mixed complexity levels."""
        data = []
        
        for i in range(count):
            complexity_level = (i % 4) + 1  # 1-4 complexity levels
            
            if complexity_level == 1:
                # Simple
                text = self.generate_sanskrit_texts(1, 50)[0]
                processing_type = "tokenization"
            elif complexity_level == 2:
                # Medium
                text = self.generate_sanskrit_texts(1, 150)[0]
                processing_type = "morphological_analysis"
            elif complexity_level == 3:
                # Complex
                text = self.generate_sanskrit_texts(1, 300)[0]
                processing_type = "rule_application"
            else:
                # Very complex
                text = self.generate_sanskrit_texts(1, 500)[0]
                processing_type = "full_analysis"
            
            data.append({
                "id": f"mixed_{i}",
                "text": text,
                "complexity": complexity_level,
                "processing_type": processing_type,
                "estimated_memory": complexity_level * 0.1  # GB estimate
            })
        
        return data


class GPUBenchmarkRunner:
    """Main GPU benchmark runner."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.gpu_engine = None
        self.data_generator = SanskritBenchmarkData()
        self.profiler = SystemProfiler()
        self.results_path = Path(self.config.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU engine
        self._initialize_gpu_engine()
    
    def _initialize_gpu_engine(self):
        """Initialize GPU acceleration engine."""
        try:
            gpu_config = GPUConfig(
                mixed_precision=True,
                batch_size=32,
                auto_scaling=True,
                benchmark_mode=True
            )
            self.gpu_engine = GPUAcceleratedInference(gpu_config)
            logger.info("GPU benchmark engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GPU engine: {e}")
            self.gpu_engine = None
    
    def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        logger.info("Starting full GPU benchmark suite")
        
        suite = BenchmarkSuite(
            suite_name="Sanskrit GPU Acceleration Benchmark",
            timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
            system_info=self._get_system_info(),
            gpu_info=self._get_gpu_info()
        )
        
        # Run individual benchmark tests
        if self.config.throughput_test:
            suite.results.extend(self._run_throughput_benchmarks())
        
        if self.config.latency_test:
            suite.results.extend(self._run_latency_benchmarks())
        
        if self.config.memory_stress_test:
            suite.results.extend(self._run_memory_stress_tests())
        
        if self.config.scaling_test:
            suite.results.extend(self._run_scaling_tests())
        
        # Generate summary
        suite.summary = self._generate_benchmark_summary(suite.results)
        
        # Save results
        if self.config.save_results:
            self._save_benchmark_results(suite)
        
        logger.info("Benchmark suite completed")
        return suite
    
    def _run_throughput_benchmarks(self) -> List[BenchmarkResult]:
        """Run throughput benchmark tests."""
        logger.info("Running throughput benchmarks")
        results = []
        
        for data_size in self.config.test_data_sizes:
            for batch_size in self.config.batch_sizes:
                # Generate test data
                test_data = self.data_generator.generate_mixed_complexity_data(data_size)
                
                # Configure test
                config = {
                    "test_type": "throughput",
                    "data_size": data_size,
                    "batch_size": batch_size,
                    "iterations": self.config.iterations_per_test
                }
                
                # Run benchmark
                result = self._run_single_benchmark(
                    test_name=f"throughput_data{data_size}_batch{batch_size}",
                    config=config,
                    test_data=test_data,
                    processing_function=self._mock_processing_function
                )
                
                results.append(result)
        
        return results
    
    def _run_latency_benchmarks(self) -> List[BenchmarkResult]:
        """Run latency benchmark tests."""
        logger.info("Running latency benchmarks")
        results = []
        
        # Test single-item processing latency
        for seq_length in self.config.sequence_lengths:
            test_data = self.data_generator.generate_sanskrit_texts(1, seq_length)
            
            config = {
                "test_type": "latency",
                "sequence_length": seq_length,
                "batch_size": 1,
                "iterations": self.config.iterations_per_test * 2  # More iterations for latency
            }
            
            result = self._run_single_benchmark(
                test_name=f"latency_seq{seq_length}",
                config=config,
                test_data=test_data,
                processing_function=self._mock_processing_function
            )
            
            results.append(result)
        
        return results
    
    def _run_memory_stress_tests(self) -> List[BenchmarkResult]:
        """Run memory stress tests."""
        logger.info("Running memory stress tests")
        results = []
        
        # Gradually increase memory usage
        memory_test_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in memory_test_sizes:
            # Generate large dataset
            test_data = self.data_generator.generate_mixed_complexity_data(size)
            
            config = {
                "test_type": "memory_stress",
                "data_size": size,
                "estimated_memory_gb": size * 0.1,
                "iterations": 1  # Single iteration for stress test
            }
            
            result = self._run_single_benchmark(
                test_name=f"memory_stress_{size}",
                config=config,
                test_data=test_data,
                processing_function=self._memory_intensive_function
            )
            
            results.append(result)
            
            # Check if we hit memory limits
            if result.errors:
                logger.warning(f"Memory stress test failed at size {size}")
                break
        
        return results
    
    def _run_scaling_tests(self) -> List[BenchmarkResult]:
        """Run auto-scaling tests."""
        logger.info("Running auto-scaling tests")
        results = []
        
        # Test with varying complexity data
        complexity_levels = [1, 2, 3, 4]
        
        for complexity in complexity_levels:
            # Generate data with specific complexity
            test_data = []
            for i in range(100):
                data_item = {
                    "id": f"scaling_test_{i}",
                    "complexity": complexity,
                    "processing_time_estimate": complexity * 0.1
                }
                test_data.append(data_item)
            
            config = {
                "test_type": "auto_scaling",
                "complexity_level": complexity,
                "data_size": len(test_data),
                "iterations": self.config.iterations_per_test
            }
            
            result = self._run_single_benchmark(
                test_name=f"scaling_complexity{complexity}",
                config=config,
                test_data=test_data,
                processing_function=self._complexity_aware_function
            )
            
            results.append(result)
        
        return results
    
    def _run_single_benchmark(self, test_name: str, config: Dict[str, Any], 
                            test_data: List[Any], processing_function: Callable) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.debug(f"Running benchmark: {test_name}")
        
        # Initialize result
        result = BenchmarkResult(
            test_name=test_name,
            config=config,
            metrics={},
            memory_stats=GPUMemoryStats(),
            execution_time=0.0,
            throughput=0.0
        )
        
        try:
            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                if self.gpu_engine:
                    self.gpu_engine.process_batch(
                        test_data[:min(10, len(test_data))], 
                        processing_function
                    )
            
            # Clear cache after warmup
            if self.gpu_engine:
                self.gpu_engine.device_manager.clear_cache()
            
            # Start monitoring
            self.profiler.start_monitoring()
            
            # Run actual benchmark
            start_time = time.time()
            
            if self.gpu_engine:
                # Use GPU engine
                batch_results = self.gpu_engine.process_large_dataset(
                    test_data, processing_function
                )
                
                # Aggregate results
                total_throughput = sum(r.throughput for r in batch_results)
                avg_memory = sum(r.memory_peak for r in batch_results) / len(batch_results) if batch_results else 0
                avg_gpu_util = sum(r.gpu_utilization for r in batch_results) / len(batch_results) if batch_results else 0
                
                result.metrics.update({
                    "total_throughput": total_throughput,
                    "average_memory_peak_gb": avg_memory,
                    "average_gpu_utilization": avg_gpu_util,
                    "batch_count": len(batch_results)
                })
                
                # Get final memory stats
                result.memory_stats = self.gpu_engine.device_manager.get_memory_stats()
            else:
                # CPU fallback
                processing_function(test_data)
                result.metrics["cpu_processing"] = True
            
            end_time = time.time()
            result.execution_time = end_time - start_time
            result.throughput = len(test_data) / result.execution_time if result.execution_time > 0 else 0
            
            # Stop monitoring and get system stats
            system_stats = self.profiler.stop_monitoring()
            result.metrics.update(system_stats)
            
        except Exception as e:
            logger.error(f"Benchmark {test_name} failed: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _mock_processing_function(self, data: List[Any]) -> List[Any]:
        """Mock processing function for benchmarking."""
        # Simulate processing with some computation
        results = []
        for item in data:
            if isinstance(item, str):
                # Simulate text processing
                result = item.upper()
            elif isinstance(item, dict):
                # Simulate structured data processing
                result = {k: str(v) for k, v in item.items()}
            else:
                result = str(item)
            
            results.append(result)
            
            # Add small delay to simulate processing
            time.sleep(0.001)
        
        return results
    
    def _memory_intensive_function(self, data: List[Any]) -> List[Any]:
        """Memory-intensive processing function for stress testing."""
        results = []
        
        # Create memory-intensive operations
        for item in data:
            # Simulate large tensor operations
            if TORCH_AVAILABLE and torch:
                # Create temporary tensors
                temp_tensor = torch.randn(1000, 1000)
                if torch.cuda.is_available():
                    temp_tensor = temp_tensor.cuda()
                
                # Perform operations
                result_tensor = torch.matmul(temp_tensor, temp_tensor.T)
                result = result_tensor.sum().item()
                
                # Clean up
                del temp_tensor, result_tensor
            else:
                # CPU intensive operation
                result = sum(i * i for i in range(1000))
            
            results.append(result)
        
        # Force garbage collection
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def _complexity_aware_function(self, data: List[Any]) -> List[Any]:
        """Processing function that scales with data complexity."""
        results = []
        
        for item in data:
            complexity = item.get('complexity', 1) if isinstance(item, dict) else 1
            
            # Scale processing time with complexity
            processing_iterations = complexity * 100
            
            result = 0
            for i in range(processing_iterations):
                result += i * complexity
            
            results.append(result)
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"gpu_available": False}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_info.update({
                    "gpu_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
        
        return gpu_info
    
    def _generate_benchmark_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if not r.errors]
        
        if not successful_results:
            return {"error": "No successful benchmark results"}
        
        # Calculate summary statistics
        throughputs = [r.throughput for r in successful_results if r.throughput > 0]
        execution_times = [r.execution_time for r in successful_results]
        memory_peaks = [r.memory_stats.allocated_memory for r in successful_results if r.memory_stats.allocated_memory > 0]
        
        summary = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results),
            "throughput_stats": {
                "mean": statistics.mean(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0,
                "min": min(throughputs) if throughputs else 0,
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0
            },
            "execution_time_stats": {
                "mean": statistics.mean(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "min": min(execution_times) if execution_times else 0,
                "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            "memory_stats": {
                "peak_usage_gb": max(memory_peaks) if memory_peaks else 0,
                "average_usage_gb": statistics.mean(memory_peaks) if memory_peaks else 0
            }
        }
        
        return summary
    
    def _save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file."""
        results_file = self.results_path / f"benchmark_{suite.timestamp}.json"
        
        try:
            # Convert to serializable format
            suite_dict = {
                "suite_name": suite.suite_name,
                "timestamp": suite.timestamp,
                "system_info": suite.system_info,
                "gpu_info": suite.gpu_info,
                "summary": suite.summary,
                "results": []
            }
            
            for result in suite.results:
                result_dict = {
                    "test_name": result.test_name,
                    "config": result.config,
                    "metrics": result.metrics,
                    "execution_time": result.execution_time,
                    "throughput": result.throughput,
                    "errors": result.errors,
                    "memory_stats": {
                        "total_memory": result.memory_stats.total_memory,
                        "allocated_memory": result.memory_stats.allocated_memory,
                        "utilization_percent": result.memory_stats.utilization_percent
                    }
                }
                suite_dict["results"].append(result_dict)
            
            with open(results_file, 'w') as f:
                json.dump(suite_dict, f, indent=2)
            
            logger.info(f"Benchmark results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")


def run_gpu_benchmarks(config: Optional[BenchmarkConfig] = None) -> BenchmarkSuite:
    """Run GPU benchmarks with specified configuration."""
    runner = GPUBenchmarkRunner(config)
    return runner.run_full_benchmark_suite()


def main():
    """Main function for running benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanskrit GPU Acceleration Benchmarks")
    parser.add_argument("--config", type=str, help="Path to benchmark config file")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--stress", action="store_true", help="Run stress test only")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create config
    if args.quick:
        config = BenchmarkConfig(
            test_data_sizes=[10, 50],
            batch_sizes=[8, 32],
            iterations_per_test=2,
            results_path=args.output
        )
    elif args.stress:
        config = BenchmarkConfig(
            test_data_sizes=[1000, 5000],
            memory_stress_test=True,
            throughput_test=False,
            latency_test=False,
            scaling_test=False,
            results_path=args.output
        )
    else:
        config = BenchmarkConfig(results_path=args.output)
    
    # Run benchmarks
    logger.info("Starting GPU benchmarks...")
    suite = run_gpu_benchmarks(config)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"Total tests: {suite.summary.get('total_tests', 0)}")
    print(f"Successful: {suite.summary.get('successful_tests', 0)}")
    print(f"Failed: {suite.summary.get('failed_tests', 0)}")
    
    if 'throughput_stats' in suite.summary:
        throughput = suite.summary['throughput_stats']
        print(f"Average throughput: {throughput.get('mean', 0):.2f} items/sec")
        print(f"Peak throughput: {throughput.get('max', 0):.2f} items/sec")
    
    if 'memory_stats' in suite.summary:
        memory = suite.summary['memory_stats']
        print(f"Peak memory usage: {memory.get('peak_usage_gb', 0):.2f} GB")
    
    print("="*60)


if __name__ == "__main__":
    main()