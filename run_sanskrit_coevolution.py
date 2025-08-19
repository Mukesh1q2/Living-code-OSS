#!/usr/bin/env python3
"""
Run Sanskrit Co-evolutionary Training Loop.

This script demonstrates and tests the Sanskrit Challenger-Solver co-evolutionary
training system for R-Zero framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sanskrit_rewrite_engine.sanskrit_coevolution import (
    run_sanskrit_coevolution, test_coevolution_convergence, CoevolutionConfig
)
from sanskrit_rewrite_engine.r_zero_config import SanskritRZeroConfig, setup_sanskrit_r_zero_environment


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sanskrit_coevolution.log')
        ]
    )


def verify_environment():
    """Verify that the environment is properly set up."""
    logger = logging.getLogger(__name__)
    
    # Check required directories
    required_dirs = [
        "r_zero_storage",
        "sanskrit_datasets", 
        "r_zero_checkpoints",
        "sanskrit_corpus"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check for R-Zero main directory
    r_zero_main = Path("R-Zero-main")
    if not r_zero_main.exists():
        logger.warning("R-Zero-main directory not found. Some features may not work.")
    
    # Set environment variables
    os.environ['STORAGE_PATH'] = str(Path("r_zero_storage").absolute())
    
    logger.info("Environment verification completed")


def create_sample_config() -> CoevolutionConfig:
    """Create a sample configuration for testing."""
    return CoevolutionConfig(
        max_iterations=20,
        problems_per_iteration=10,
        convergence_threshold=0.85,
        performance_window=5,
        save_frequency=5,
        evaluation_frequency=3,
        evaluation_problems=20
    )


def run_quick_test():
    """Run a quick test of the co-evolutionary system."""
    logger = logging.getLogger(__name__)
    logger.info("Running quick co-evolution test...")
    
    try:
        # Run a short test
        results = test_coevolution_convergence(num_iterations=3)
        
        logger.info("Quick test completed successfully!")
        logger.info(f"Test results: {results.get('experiment_id', 'N/A')}")
        logger.info(f"Final performance: {results.get('final_performance', {}).get('solver_performance', {}).get('overall', 0.0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


def run_full_training(config_path: str = None, iterations: int = None):
    """Run full co-evolutionary training."""
    logger = logging.getLogger(__name__)
    logger.info("Starting full co-evolutionary training...")
    
    try:
        # Create configuration
        if iterations:
            config = CoevolutionConfig(
                max_iterations=iterations,
                problems_per_iteration=15,
                convergence_threshold=0.9,
                save_frequency=max(1, iterations // 5)
            )
        else:
            config = create_sample_config()
        
        # Run training
        results = run_sanskrit_coevolution(
            config_path=config_path,
            coevolution_config=config
        )
        
        # Report results
        logger.info("Co-evolutionary training completed!")
        logger.info(f"Experiment ID: {results.get('experiment_id', 'N/A')}")
        logger.info(f"Total iterations: {results.get('total_iterations', 0)}")
        logger.info(f"Converged: {results.get('converged', False)}")
        logger.info(f"Final performance: {results.get('final_performance', {}).get('solver_performance', {}).get('overall', 0.0):.3f}")
        logger.info(f"Performance improvement: {results.get('training_statistics', {}).get('performance_improvement', 0.0):.3f}")
        
        # Print performance by problem type
        performance_by_type = results.get('performance_by_type', {})
        if performance_by_type:
            logger.info("Performance by problem type:")
            for problem_type, performance in performance_by_type.items():
                logger.info(f"  {problem_type}: {performance:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        raise


def analyze_results(results_file: str):
    """Analyze results from a completed training run."""
    logger = logging.getLogger(__name__)
    
    try:
        import json
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Analyzing results from: {results_file}")
        logger.info(f"Experiment ID: {results.get('experiment_id', 'N/A')}")
        logger.info(f"Training duration: {results.get('start_time', 'N/A')} to {results.get('end_time', 'N/A')}")
        
        # Training statistics
        stats = results.get('training_statistics', {})
        logger.info(f"Total problems generated: {stats.get('total_problems_generated', 0)}")
        logger.info(f"Total problems solved: {stats.get('total_problems_solved', 0)}")
        logger.info(f"Average solver performance: {stats.get('average_solver_performance', 0.0):.3f}")
        logger.info(f"Final solver performance: {stats.get('final_solver_performance', 0.0):.3f}")
        logger.info(f"Performance improvement: {stats.get('performance_improvement', 0.0):.3f}")
        
        # Convergence analysis
        converged = results.get('converged', False)
        total_iterations = results.get('total_iterations', 0)
        max_iterations = results.get('config', {}).get('max_iterations', 0)
        
        logger.info(f"Convergence: {'Yes' if converged else 'No'}")
        logger.info(f"Iterations used: {total_iterations}/{max_iterations}")
        
        if converged:
            efficiency = total_iterations / max_iterations if max_iterations > 0 else 0
            logger.info(f"Training efficiency: {efficiency:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Results analysis failed: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Sanskrit Co-evolutionary Training")
    parser.add_argument("--mode", choices=["test", "train", "analyze"], default="test",
                       help="Mode to run: test (quick test), train (full training), analyze (analyze results)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--results_file", type=str, help="Results file to analyze (for analyze mode)")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                       help="Logging level")
    parser.add_argument("--setup_env", action="store_true", help="Set up R-Zero environment")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Verify environment
        verify_environment()
        
        # Set up R-Zero environment if requested
        if args.setup_env:
            logger.info("Setting up R-Zero environment...")
            setup = setup_sanskrit_r_zero_environment(args.config)
            env_info = setup.get_environment_info()
            logger.info(f"Environment setup completed: {env_info}")
        
        # Run based on mode
        if args.mode == "test":
            logger.info("Running quick test...")
            success = run_quick_test()
            if success:
                logger.info("Quick test passed!")
                return 0
            else:
                logger.error("Quick test failed!")
                return 1
                
        elif args.mode == "train":
            logger.info("Running full training...")
            results = run_full_training(args.config, args.iterations)
            if results:
                logger.info("Training completed successfully!")
                return 0
            else:
                logger.error("Training failed!")
                return 1
                
        elif args.mode == "analyze":
            if not args.results_file:
                logger.error("Results file required for analyze mode")
                return 1
            
            logger.info("Analyzing results...")
            results = analyze_results(args.results_file)
            if results:
                logger.info("Analysis completed successfully!")
                return 0
            else:
                logger.error("Analysis failed!")
                return 1
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())