"""
Main execution script for Sanskrit Rewrite Engine GPU acceleration.

This script provides command-line interface for running GPU-accelerated
Sanskrit processing, benchmarks, and demonstrations.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import GPU acceleration components
try:
    from .gpu_acceleration import (
        GPUAcceleratedInference, GPUConfig, 
        create_gpu_inference_engine, get_optimal_gpu_config
    )
    from .memory_optimization import (
        MemoryOptimizer, MemoryConfig,
        create_memory_optimizer, get_rx6800m_memory_config
    )
    from .gpu_benchmarks import (
        GPUBenchmarkRunner, BenchmarkConfig,
        run_gpu_benchmarks
    )
    GPU_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GPU acceleration not available: {e}")
    GPU_AVAILABLE = False

# Import core Sanskrit components
try:
    from .tokenizer import SanskritTokenizer
    from .panini_engine import PaniniRuleEngine
    from .essential_sutras import create_essential_sutras
    SANSKRIT_CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Sanskrit core components not available: {e}")
    SANSKRIT_CORE_AVAILABLE = False


def create_sample_sanskrit_data(count: int = 100) -> list:
    """Create sample Sanskrit data for testing."""
    sample_texts = [
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
    
    # Generate test data
    test_data = []
    for i in range(count):
        base_text = sample_texts[i % len(sample_texts)]
        
        # Add variations
        if i % 3 == 0:
            text = f"{base_text} + asya"  # Genitive form
        elif i % 3 == 1:
            text = f"a + i"  # Sandhi rule
        else:
            text = base_text
        
        test_data.append({
            "id": f"sample_{i}",
            "text": text,
            "type": "sanskrit_processing",
            "complexity": (i % 4) + 1
        })
    
    return test_data


def mock_sanskrit_processing(data: list) -> list:
    """Mock Sanskrit processing function for demonstration."""
    results = []
    
    for item in data:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item)
        
        # Mock processing based on text content
        if "+" in text:
            # Simulate rule application
            if "a + i" in text:
                result = text.replace("a + i", "e")
            elif "a + u" in text:
                result = text.replace("a + u", "o")
            elif "+ asya" in text:
                result = text.replace(" + asya", "asya")
            else:
                result = text.replace(" + ", "")
        else:
            # Simulate tokenization
            result = f"tokenized: {text}"
        
        results.append(result)
    
    return results


def run_demo(args: argparse.Namespace) -> None:
    """Run GPU acceleration demonstration."""
    logger.info("Running Sanskrit GPU acceleration demonstration")
    
    if not GPU_AVAILABLE:
        logger.error("GPU acceleration not available")
        return
    
    # Create GPU configuration
    if args.rx6800m:
        gpu_config = GPUConfig(
            device="auto",
            mixed_precision=True,
            batch_size=64,
            rocm_optimization=True,
            auto_scaling=True
        )
        memory_config = get_rx6800m_memory_config()
    else:
        gpu_config = get_optimal_gpu_config()
        memory_config = MemoryConfig()
    
    logger.info(f"GPU Config: device={gpu_config.device}, batch_size={gpu_config.batch_size}")
    logger.info(f"Memory Config: max_memory={memory_config.max_memory_gb}GB")
    
    # Create components
    gpu_engine = create_gpu_inference_engine(gpu_config)
    memory_optimizer = create_memory_optimizer(memory_config)
    
    # Start memory monitoring
    memory_optimizer.start_monitoring()
    
    try:
        # Create test data
        test_data = create_sample_sanskrit_data(args.data_size)
        logger.info(f"Created {len(test_data)} test samples")
        
        # Store in memory optimizer
        success = memory_optimizer.store_large_dataset("demo_data", test_data)
        if success:
            logger.info("Test data stored in memory optimizer")
        
        # Load and process
        loaded_data = memory_optimizer.load_large_dataset("demo_data")
        if loaded_data:
            logger.info("Test data loaded from memory optimizer")
            
            # Process with GPU acceleration
            logger.info("Starting GPU-accelerated processing...")
            results = gpu_engine.process_large_dataset(
                loaded_data, 
                mock_sanskrit_processing,
                item_memory_estimate=0.01
            )
            
            # Display results
            total_items = sum(len(result.outputs) for result in results)
            total_time = sum(result.processing_time for result in results)
            avg_throughput = sum(result.throughput for result in results) / len(results)
            avg_gpu_util = sum(result.gpu_utilization for result in results) / len(results)
            
            logger.info(f"Processing completed:")
            logger.info(f"  Total items: {total_items}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Average throughput: {avg_throughput:.2f} items/sec")
            logger.info(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
            
            # Show sample results
            if results and results[0].outputs:
                logger.info("Sample results:")
                for i, output in enumerate(results[0].outputs[:3]):
                    logger.info(f"  {i+1}: {output}")
        
        # Get performance stats
        perf_stats = gpu_engine.get_performance_stats()
        logger.info("Performance Statistics:")
        logger.info(f"  Device: {perf_stats['device_info']['device']}")
        logger.info(f"  Memory utilization: {perf_stats['memory_stats']['utilization_percent']:.1f}%")
        logger.info(f"  Total inferences: {perf_stats['inference_stats']['total_inferences']}")
        
        # Get memory report
        memory_report = memory_optimizer.get_memory_report()
        logger.info("Memory Report:")
        logger.info(f"  Cache hit rate: {memory_report['cache_stats']['hit_rate']:.2f}")
        logger.info(f"  Cache utilization: {memory_report['cache_stats']['utilization']:.2f}")
        
    finally:
        # Cleanup
        memory_optimizer.stop_monitoring()
        memory_optimizer.cleanup()
        logger.info("Demo completed")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run GPU benchmarks."""
    logger.info("Running GPU benchmarks")
    
    if not GPU_AVAILABLE:
        logger.error("GPU acceleration not available")
        return
    
    # Create benchmark configuration
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
        config = BenchmarkConfig(
            results_path=args.output,
            iterations_per_test=args.iterations
        )
    
    # Run benchmarks
    suite = run_gpu_benchmarks(config)
    
    # Display results
    logger.info("Benchmark Results:")
    logger.info(f"  Total tests: {suite.summary.get('total_tests', 0)}")
    logger.info(f"  Successful: {suite.summary.get('successful_tests', 0)}")
    logger.info(f"  Failed: {suite.summary.get('failed_tests', 0)}")
    
    if 'throughput_stats' in suite.summary:
        throughput = suite.summary['throughput_stats']
        logger.info(f"  Average throughput: {throughput.get('mean', 0):.2f} items/sec")
        logger.info(f"  Peak throughput: {throughput.get('max', 0):.2f} items/sec")
    
    if 'memory_stats' in suite.summary:
        memory = suite.summary['memory_stats']
        logger.info(f"  Peak memory usage: {memory.get('peak_usage_gb', 0):.2f} GB")
    
    logger.info(f"Results saved to: {args.output}")


def run_memory_test(args: argparse.Namespace) -> None:
    """Run memory optimization tests."""
    logger.info("Running memory optimization tests")
    
    if not GPU_AVAILABLE:
        logger.error("GPU acceleration not available")
        return
    
    # Create memory optimizer
    if args.rx6800m:
        config = get_rx6800m_memory_config()
    else:
        config = MemoryConfig()
    
    optimizer = create_memory_optimizer(config)
    optimizer.start_monitoring()
    
    try:
        # Test different data sizes
        test_sizes = [100, 500, 1000, 2000]
        
        for size in test_sizes:
            logger.info(f"Testing with {size} items...")
            
            # Create test data
            test_data = create_sample_sanskrit_data(size)
            
            # Store data
            success = optimizer.store_large_dataset(f"test_{size}", test_data)
            logger.info(f"  Storage success: {success}")
            
            # Load data
            loaded_data = optimizer.load_large_dataset(f"test_{size}")
            logger.info(f"  Load success: {loaded_data is not None}")
            
            # Get memory report
            report = optimizer.get_memory_report()
            memory_usage = report['memory_stats']['utilization_percent']
            cache_hit_rate = report['cache_stats']['hit_rate']
            
            logger.info(f"  Memory usage: {memory_usage:.1f}%")
            logger.info(f"  Cache hit rate: {cache_hit_rate:.2f}")
            
            # Optimize memory if usage is high
            if memory_usage > 80:
                logger.info("  Optimizing memory...")
                opt_results = optimizer.optimize_memory_usage()
                logger.info(f"  Memory freed: {opt_results['memory_freed_gb']:.2f} GB")
    
    finally:
        optimizer.stop_monitoring()
        optimizer.cleanup()
        logger.info("Memory test completed")


def run_serve(args: argparse.Namespace) -> None:
    """Run GPU acceleration service."""
    logger.info("Starting GPU acceleration service")
    
    if not GPU_AVAILABLE:
        logger.error("GPU acceleration not available")
        return
    
    # This would start a web service or API
    # For now, just run a continuous demo
    try:
        while True:
            logger.info("Service running... (Press Ctrl+C to stop)")
            
            # Create and run a small demo
            gpu_config = get_optimal_gpu_config()
            gpu_engine = create_gpu_inference_engine(gpu_config)
            
            test_data = create_sample_sanskrit_data(10)
            result = gpu_engine.process_batch(test_data, mock_sanskrit_processing)
            
            logger.info(f"Processed {len(result.outputs)} items, "
                       f"throughput: {result.throughput:.2f} items/sec")
            
            import time
            time.sleep(30)  # Wait 30 seconds
            
    except KeyboardInterrupt:
        logger.info("Service stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sanskrit Rewrite Engine GPU Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sanskrit_rewrite_engine --demo --rx6800m
  python -m sanskrit_rewrite_engine --benchmark --quick
  python -m sanskrit_rewrite_engine --memory-test --data-size 1000
  python -m sanskrit_rewrite_engine --serve
        """
    )
    
    # Main commands
    parser.add_argument("--demo", action="store_true", 
                       help="Run GPU acceleration demonstration")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run GPU benchmarks")
    parser.add_argument("--memory-test", action="store_true",
                       help="Run memory optimization tests")
    parser.add_argument("--serve", action="store_true",
                       help="Start GPU acceleration service")
    
    # Configuration options
    parser.add_argument("--rx6800m", action="store_true",
                       help="Use RX6800M optimized configuration")
    parser.add_argument("--data-size", type=int, default=100,
                       help="Size of test data (default: 100)")
    parser.add_argument("--output", type=str, default="./benchmark_results",
                       help="Output directory for results")
    
    # Benchmark options
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark")
    parser.add_argument("--stress", action="store_true",
                       help="Run stress test")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of benchmark iterations")
    
    # Logging options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Check if any command was specified
    if not any([args.demo, args.benchmark, args.memory_test, args.serve]):
        parser.print_help()
        return
    
    # Run specified command
    try:
        if args.demo:
            run_demo(args)
        elif args.benchmark:
            run_benchmark(args)
        elif args.memory_test:
            run_memory_test(args)
        elif args.serve:
            run_serve(args)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()