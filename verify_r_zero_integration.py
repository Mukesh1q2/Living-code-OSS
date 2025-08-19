#!/usr/bin/env python3
"""
Verification script for R-Zero integration with Sanskrit reasoning.

This script verifies that all components of the R-Zero integration are working
correctly and provides a comprehensive status report.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_r_zero_directories() -> Dict[str, bool]:
    """Verify R-Zero directory structure."""
    logger.info("Verifying R-Zero directory structure...")
    
    directories = {
        'R-Zero-main': Path('R-Zero-main').exists(),
        'external_models/r-zero': Path('external_models/r-zero').exists(),
        'r_zero_storage': Path('r_zero_storage').exists(),
        'sanskrit_datasets': Path('sanskrit_datasets').exists(),
        'r_zero_models': Path('r_zero_models').exists(),
        'r_zero_checkpoints': Path('r_zero_checkpoints').exists()
    }
    
    return directories


def verify_configuration_files() -> Dict[str, bool]:
    """Verify configuration files exist."""
    logger.info("Verifying configuration files...")
    
    files = {
        'sanskrit_config.yaml': Path('r_zero_storage/config.yaml').exists(),
        'sanskrit_format.jinja': Path('r_zero_storage/sanskrit_format.jinja').exists(),
        'train_dataset': Path('sanskrit_datasets/sanskrit_train.json').exists(),
        'val_dataset': Path('sanskrit_datasets/sanskrit_val.json').exists()
    }
    
    return files


def verify_python_modules() -> Dict[str, bool]:
    """Verify Python modules can be imported."""
    logger.info("Verifying Python module imports...")
    
    modules = {}
    
    # Core modules
    try:
        from sanskrit_rewrite_engine.r_zero_config import SanskritRZeroConfig
        modules['r_zero_config'] = True
    except ImportError as e:
        logger.error(f"Failed to import r_zero_config: {e}")
        modules['r_zero_config'] = False
    
    try:
        from sanskrit_rewrite_engine.verl_integration import SanskritVERLIntegrator
        modules['verl_integration'] = True
    except ImportError as e:
        logger.error(f"Failed to import verl_integration: {e}")
        modules['verl_integration'] = False
    
    try:
        from sanskrit_rewrite_engine.sanskrit_reward_function import compute_sanskrit_score
        modules['sanskrit_reward_function'] = True
    except ImportError as e:
        logger.error(f"Failed to import sanskrit_reward_function: {e}")
        modules['sanskrit_reward_function'] = False
    
    try:
        from sanskrit_rewrite_engine.rl_environment import SanskritRLEnvironment
        modules['rl_environment'] = True
    except ImportError as e:
        logger.error(f"Failed to import rl_environment: {e}")
        modules['rl_environment'] = False
    
    return modules


def verify_reward_function() -> Dict[str, Any]:
    """Verify reward function works correctly."""
    logger.info("Verifying reward function...")
    
    try:
        from sanskrit_rewrite_engine.sanskrit_reward_function import compute_sanskrit_score
        from sanskrit_rewrite_engine.r_zero_integration import SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
        
        # Test data
        predicts = ["e", "o", "wrong"]
        ground_truths = ["e", "o", "correct"]
        problems = [
            SanskritProblem(
                id=f"verify_{i}",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text=f"test_{i}",
                expected_output=gt
            )
            for i, gt in enumerate(ground_truths)
        ]
        
        scores = compute_sanskrit_score(predicts, ground_truths, problems, None)
        
        # Verify structure
        if len(scores) != 3:
            return {'working': False, 'error': f'Expected 3 scores, got {len(scores)}'}
        
        for i, score in enumerate(scores):
            required_keys = ['overall', 'format', 'accuracy']
            for key in required_keys:
                if key not in score:
                    return {'working': False, 'error': f'Missing key {key} in score {i}'}
        
        # First two should have higher scores than the third
        if not (scores[0]['overall'] > scores[2]['overall'] and scores[1]['overall'] > scores[2]['overall']):
            return {'working': False, 'error': 'Reward function not discriminating correctly'}
        
        return {
            'working': True,
            'sample_scores': scores,
            'average_score': sum(s['overall'] for s in scores) / len(scores)
        }
        
    except Exception as e:
        return {'working': False, 'error': str(e)}


def verify_verl_integration() -> Dict[str, Any]:
    """Verify VERL integration."""
    logger.info("Verifying VERL integration...")
    
    try:
        from sanskrit_rewrite_engine.verl_integration import SanskritVERLIntegrator, check_verl_availability
        
        # Check availability
        availability = check_verl_availability()
        
        # Create integrator
        integrator = SanskritVERLIntegrator(storage_path="./verify_verl_test")
        
        # Test environment setup
        config = {
            'model_path': 'test_model',
            'max_sanskrit_length': 128,
            'batch_size': 2,
            'learning_rate': 1e-6
        }
        
        environment = integrator.setup_sanskrit_training_environment(config)
        
        # Test training step
        training_data = [
            {'problem': 'a + i', 'answer': 'e'},
            {'problem': 'a + u', 'answer': 'o'}
        ]
        
        results = integrator.run_sanskrit_training_step(environment, training_data)
        
        # Cleanup
        import shutil
        if os.path.exists("./verify_verl_test"):
            shutil.rmtree("./verify_verl_test")
        
        return {
            'working': True,
            'availability': availability,
            'environment_components': list(environment.keys()),
            'training_results': results
        }
        
    except Exception as e:
        return {'working': False, 'error': str(e)}


def verify_environment_variables() -> Dict[str, str]:
    """Verify environment variables are set."""
    logger.info("Verifying environment variables...")
    
    return {
        'STORAGE_PATH': os.environ.get('STORAGE_PATH', 'Not set'),
        'MODEL_PATH': os.environ.get('MODEL_PATH', 'Not set'),
        'SANSKRIT_CORPUS_PATH': os.environ.get('SANSKRIT_CORPUS_PATH', 'Not set')
    }


def run_comprehensive_verification() -> Dict[str, Any]:
    """Run comprehensive verification of R-Zero integration."""
    logger.info("Starting comprehensive R-Zero integration verification...")
    
    results = {
        'directories': verify_r_zero_directories(),
        'configuration_files': verify_configuration_files(),
        'python_modules': verify_python_modules(),
        'reward_function': verify_reward_function(),
        'verl_integration': verify_verl_integration(),
        'environment_variables': verify_environment_variables()
    }
    
    return results


def print_verification_report(results: Dict[str, Any]) -> None:
    """Print a formatted verification report."""
    print("\n" + "="*80)
    print("R-ZERO INTEGRATION VERIFICATION REPORT")
    print("="*80)
    
    # Directory verification
    print("\n📁 DIRECTORY STRUCTURE:")
    for directory, exists in results['directories'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {directory}")
    
    # Configuration files
    print("\n📄 CONFIGURATION FILES:")
    for file, exists in results['configuration_files'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    # Python modules
    print("\n🐍 PYTHON MODULES:")
    for module, imported in results['python_modules'].items():
        status = "✅" if imported else "❌"
        print(f"  {status} {module}")
    
    # Reward function
    print("\n🎯 REWARD FUNCTION:")
    reward_result = results['reward_function']
    if reward_result['working']:
        print(f"  ✅ Working correctly")
        print(f"  📊 Average score: {reward_result['average_score']:.3f}")
    else:
        print(f"  ❌ Error: {reward_result['error']}")
    
    # VERL integration
    print("\n🔄 VERL INTEGRATION:")
    verl_result = results['verl_integration']
    if verl_result['working']:
        print(f"  ✅ Working correctly")
        print(f"  🔧 Components: {', '.join(verl_result['environment_components'])}")
        training_success = verl_result['training_results'].get('step_successful', False)
        print(f"  🏃 Training step: {'✅' if training_success else '❌'}")
    else:
        print(f"  ❌ Error: {verl_result['error']}")
    
    # Environment variables
    print("\n🌍 ENVIRONMENT VARIABLES:")
    for var, value in results['environment_variables'].items():
        status = "✅" if value != 'Not set' else "❌"
        print(f"  {status} {var}: {value}")
    
    # Overall status
    print("\n" + "="*80)
    
    # Calculate overall success
    dir_success = all(results['directories'].values())
    config_success = all(results['configuration_files'].values())
    module_success = all(results['python_modules'].values())
    reward_success = results['reward_function']['working']
    verl_success = results['verl_integration']['working']
    env_success = all(v != 'Not set' for v in results['environment_variables'].values())
    
    overall_success = all([dir_success, config_success, module_success, reward_success, verl_success])
    
    if overall_success:
        print("🎉 OVERALL STATUS: ✅ ALL SYSTEMS OPERATIONAL")
        print("   R-Zero integration is ready for Sanskrit reasoning tasks!")
    else:
        print("⚠️  OVERALL STATUS: ❌ SOME ISSUES DETECTED")
        print("   Please review the failed components above.")
    
    print("="*80)


def main():
    """Main verification function."""
    try:
        results = run_comprehensive_verification()
        print_verification_report(results)
        
        # Return appropriate exit code
        overall_success = (
            all(results['directories'].values()) and
            all(results['configuration_files'].values()) and
            all(results['python_modules'].values()) and
            results['reward_function']['working'] and
            results['verl_integration']['working']
        )
        
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        print(f"\n❌ VERIFICATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)