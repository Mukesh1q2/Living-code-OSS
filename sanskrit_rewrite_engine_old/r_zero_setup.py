#!/usr/bin/env python3
"""
R-Zero setup script for Sanskrit reasoning integration.

This script sets up the R-Zero environment and dependencies for Sanskrit
grammatical reasoning tasks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our components
from .r_zero_config import SanskritRZeroConfig, RZeroEnvironmentSetup, setup_sanskrit_r_zero_environment
from .verl_integration import SanskritVERLIntegrator, check_verl_availability
from .sanskrit_reward_function import compute_sanskrit_score


def check_r_zero_dependencies() -> Dict[str, Any]:
    """
    Check R-Zero dependencies and availability.
    
    Returns:
        Dictionary with dependency status
    """
    logger.info("Checking R-Zero dependencies...")
    
    status = {
        'r_zero_main_exists': Path('R-Zero-main').exists(),
        'external_r_zero_exists': Path('external_models/r-zero').exists(),
        'verl_availability': check_verl_availability(),
        'python_version': sys.version,
        'required_packages': {}
    }
    
    # Check required packages
    required_packages = [
        'torch', 'transformers', 'accelerate', 'datasets', 
        'yaml', 'numpy', 'pandas'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            status['required_packages'][package] = True
        except ImportError:
            status['required_packages'][package] = False
    
    return status


def setup_storage_directories(config: SanskritRZeroConfig) -> None:
    """
    Set up storage directories for R-Zero.
    
    Args:
        config: Sanskrit R-Zero configuration
    """
    logger.info("Setting up storage directories...")
    
    directories = [
        config.storage_path,
        config.model_path,
        config.dataset_path,
        config.checkpoint_path,
        config.sanskrit_corpus_path,
        config.sutra_rules_path,
        f"{config.storage_path}/logs",
        f"{config.storage_path}/traces",
        f"{config.storage_path}/rewards",
        f"{config.storage_path}/experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_sample_sanskrit_dataset(config: SanskritRZeroConfig) -> None:
    """
    Create a sample Sanskrit dataset for testing.
    
    Args:
        config: Sanskrit R-Zero configuration
    """
    logger.info("Creating sample Sanskrit dataset...")
    
    sample_data = [
        {
            "sanskrit_problem": "Apply sandhi to: a + i",
            "expected_solution": "e",
            "problem_type": "SANDHI_APPLICATION",
            "difficulty": "BEGINNER",
            "sutra_references": ["6.1.87"]
        },
        {
            "sanskrit_problem": "Apply sandhi to: a + u", 
            "expected_solution": "o",
            "problem_type": "SANDHI_APPLICATION",
            "difficulty": "BEGINNER",
            "sutra_references": ["6.1.87"]
        },
        {
            "sanskrit_problem": "Analyze morphology of: gacchati",
            "expected_solution": "gam(root) + ti(suffix)",
            "problem_type": "MORPHOLOGICAL_ANALYSIS", 
            "difficulty": "INTERMEDIATE",
            "sutra_references": ["3.4.78"]
        },
        {
            "sanskrit_problem": "Apply sandhi to: rāma + asya",
            "expected_solution": "rāmasya",
            "problem_type": "SANDHI_APPLICATION",
            "difficulty": "INTERMEDIATE",
            "sutra_references": ["6.1.101"]
        },
        {
            "sanskrit_problem": "Analyze compound: rājapuruṣa",
            "expected_solution": "rāja + puruṣa (tatpuruṣa)",
            "problem_type": "COMPOUND_ANALYSIS",
            "difficulty": "ADVANCED",
            "sutra_references": ["2.1.22"]
        }
    ]
    
    # Save training data
    import json
    train_file = Path(config.dataset_path) / "sanskrit_train.json"
    val_file = Path(config.dataset_path) / "sanskrit_val.json"
    
    # Split data (80% train, 20% val)
    split_idx = int(len(sample_data) * 0.8)
    train_data = sample_data[:split_idx]
    val_data = sample_data[split_idx:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created training dataset: {train_file} ({len(train_data)} examples)")
    logger.info(f"Created validation dataset: {val_file} ({len(val_data)} examples)")


def test_reward_function() -> None:
    """Test the Sanskrit reward function."""
    logger.info("Testing Sanskrit reward function...")
    
    # Test data
    predicts = ["e", "o", "gam + ti"]
    ground_truths = ["e", "o", "gam + ti"]
    
    try:
        scores = compute_sanskrit_score(predicts, ground_truths)
        
        logger.info("Reward function test results:")
        for i, score in enumerate(scores):
            logger.info(f"  Example {i+1}: overall={score['overall']:.3f}, "
                       f"format={score['format']:.3f}, accuracy={score['accuracy']:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"Reward function test failed: {e}")
        return False


def test_verl_integration() -> bool:
    """Test VERL integration."""
    logger.info("Testing VERL integration...")
    
    try:
        integrator = SanskritVERLIntegrator(storage_path="./test_verl_integration")
        
        # Test environment setup
        config = {
            'model_path': 'Qwen/Qwen2.5-7B-Instruct',
            'max_sanskrit_length': 256,
            'batch_size': 4,
            'learning_rate': 1e-6
        }
        
        environment = integrator.setup_sanskrit_training_environment(config)
        
        # Test training step
        training_data = [
            {'problem': 'a + i', 'answer': 'e'},
            {'problem': 'a + u', 'answer': 'o'}
        ]
        
        results = integrator.run_sanskrit_training_step(environment, training_data)
        
        logger.info(f"VERL integration test results: {results}")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_verl_integration"):
            shutil.rmtree("./test_verl_integration")
        
        return results.get('step_successful', False)
        
    except Exception as e:
        logger.error(f"VERL integration test failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting R-Zero setup for Sanskrit reasoning...")
    
    # Check dependencies
    deps_status = check_r_zero_dependencies()
    logger.info(f"Dependency check results: {deps_status}")
    
    # Create configuration
    config = SanskritRZeroConfig()
    logger.info(f"Created configuration with storage path: {config.storage_path}")
    
    # Setup environment
    setup = setup_sanskrit_r_zero_environment()
    logger.info("Environment setup completed")
    
    # Setup directories
    setup_storage_directories(config)
    
    # Create sample dataset
    create_sample_sanskrit_dataset(config)
    
    # Test reward function
    reward_test_passed = test_reward_function()
    
    # Test VERL integration
    verl_test_passed = test_verl_integration()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("R-Zero Setup Summary:")
    logger.info(f"  Storage path: {config.storage_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Dataset path: {config.dataset_path}")
    logger.info(f"  R-Zero main exists: {deps_status['r_zero_main_exists']}")
    logger.info(f"  External R-Zero exists: {deps_status['external_r_zero_exists']}")
    logger.info(f"  Reward function test: {'PASSED' if reward_test_passed else 'FAILED'}")
    logger.info(f"  VERL integration test: {'PASSED' if verl_test_passed else 'FAILED'}")
    logger.info("="*50)
    
    if reward_test_passed and verl_test_passed:
        logger.info("✅ R-Zero setup completed successfully!")
        return True
    else:
        logger.warning("⚠️  R-Zero setup completed with some issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)