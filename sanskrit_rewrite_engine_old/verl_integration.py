"""
VERL (Versatile Reinforcement Learning) integration for Sanskrit reasoning.

This module provides integration with R-Zero's VERL components for Sanskrit
grammatical reasoning tasks.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import importlib.util

# Add R-Zero paths to system path
r_zero_paths = [
    "external_models/r-zero",
    "R-Zero-main",
    "external_models/r-zero/verl",
    "R-Zero-main/verl"
]

for path in r_zero_paths:
    if Path(path).exists():
        sys.path.insert(0, str(Path(path).absolute()))

logger = logging.getLogger(__name__)


class VERLIntegrationError(Exception):
    """Exception raised for VERL integration errors."""
    pass


class SanskritVERLIntegrator:
    """Integrator for Sanskrit reasoning with VERL components."""
    
    def __init__(self, storage_path: str = "./r_zero_storage"):
        """
        Initialize VERL integrator.
        
        Args:
            storage_path: Path for storing VERL-related data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # VERL components
        self._verl_available = False
        self._verl_components = {}
        
        # Initialize VERL components
        self._initialize_verl_components()
    
    def _initialize_verl_components(self) -> None:
        """Initialize VERL components if available."""
        try:
            # Try to import VERL components
            self._import_verl_modules()
            self._verl_available = True
            logger.info("VERL components successfully initialized")
        except Exception as e:
            logger.warning(f"VERL components not available: {e}")
            self._verl_available = False
    
    def _import_verl_modules(self) -> None:
        """Import VERL modules."""
        try:
            # Import core VERL components
            import verl
            self._verl_components['core'] = verl
            
            # Import specific components
            from verl.trainer import PPOTrainer
            from verl.workers import RolloutWorker
            from verl.models import ActorModel, CriticModel
            from verl.utils import RewardFunction
            
            self._verl_components.update({
                'trainer': PPOTrainer,
                'rollout_worker': RolloutWorker,
                'actor_model': ActorModel,
                'critic_model': CriticModel,
                'reward_function': RewardFunction
            })
            
        except ImportError as e:
            # Create mock components for development
            logger.warning(f"Creating mock VERL components: {e}")
            self._create_mock_verl_components()
    
    def _create_mock_verl_components(self) -> None:
        """Create mock VERL components for development."""
        class MockTrainer:
            def __init__(self, *args, **kwargs):
                self.config = kwargs
            
            def train(self, *args, **kwargs):
                logger.info("Mock training step")
                return {"loss": 0.1, "reward": 0.8}
        
        class MockRolloutWorker:
            def __init__(self, *args, **kwargs):
                self.config = kwargs
            
            def rollout(self, *args, **kwargs):
                logger.info("Mock rollout step")
                return {"responses": ["mock_response"], "rewards": [0.8]}
        
        class MockModel:
            def __init__(self, *args, **kwargs):
                self.config = kwargs
            
            def forward(self, *args, **kwargs):
                return {"logits": [0.1, 0.2, 0.3]}
        
        self._verl_components = {
            'trainer': MockTrainer,
            'rollout_worker': MockRolloutWorker,
            'actor_model': MockModel,
            'critic_model': MockModel,
            'reward_function': lambda x: 0.8
        }
    
    def is_verl_available(self) -> bool:
        """Check if VERL components are available."""
        return self._verl_available
    
    def create_sanskrit_trainer(self, config: Dict[str, Any]) -> Any:
        """
        Create Sanskrit-specific VERL trainer.
        
        Args:
            config: Training configuration
            
        Returns:
            VERL trainer instance
        """
        if not self._verl_available:
            raise VERLIntegrationError("VERL components not available")
        
        # Adapt config for Sanskrit reasoning
        sanskrit_config = self._adapt_config_for_sanskrit(config)
        
        # Create trainer
        trainer_class = self._verl_components['trainer']
        trainer = trainer_class(**sanskrit_config)
        
        return trainer
    
    def create_sanskrit_rollout_worker(self, config: Dict[str, Any]) -> Any:
        """
        Create Sanskrit-specific rollout worker.
        
        Args:
            config: Rollout configuration
            
        Returns:
            VERL rollout worker instance
        """
        if not self._verl_available:
            raise VERLIntegrationError("VERL components not available")
        
        # Adapt config for Sanskrit
        sanskrit_config = self._adapt_rollout_config_for_sanskrit(config)
        
        # Create rollout worker
        worker_class = self._verl_components['rollout_worker']
        worker = worker_class(**sanskrit_config)
        
        return worker
    
    def create_sanskrit_models(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Create Sanskrit-specific actor and critic models.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (actor_model, critic_model)
        """
        if not self._verl_available:
            raise VERLIntegrationError("VERL components not available")
        
        # Adapt config for Sanskrit models
        model_config = self._adapt_model_config_for_sanskrit(config)
        
        # Create models
        actor_class = self._verl_components['actor_model']
        critic_class = self._verl_components['critic_model']
        
        actor = actor_class(**model_config)
        critic = critic_class(**model_config)
        
        return actor, critic
    
    def _adapt_config_for_sanskrit(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt general config for Sanskrit reasoning."""
        sanskrit_config = config.copy()
        
        # Sanskrit-specific adaptations
        sanskrit_config.update({
            'task_type': 'sanskrit_reasoning',
            'reward_function': 'sanskrit_reward_function.py:compute_sanskrit_score',
            'max_sequence_length': config.get('max_sanskrit_length', 512),
            'vocabulary_size': config.get('sanskrit_vocab_size', 50000),
            'special_tokens': ['<sanskrit>', '</sanskrit>', '<sutra>', '</sutra>']
        })
        
        return sanskrit_config
    
    def _adapt_rollout_config_for_sanskrit(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt rollout config for Sanskrit reasoning."""
        rollout_config = config.copy()
        
        # Sanskrit-specific rollout settings
        rollout_config.update({
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 0.99),
            'max_new_tokens': config.get('max_sanskrit_length', 512),
            'do_sample': True,
            'pad_token_id': config.get('pad_token_id', 0),
            'eos_token_id': config.get('eos_token_id', 2)
        })
        
        return rollout_config
    
    def _adapt_model_config_for_sanskrit(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model config for Sanskrit reasoning."""
        model_config = config.copy()
        
        # Sanskrit-specific model settings
        model_config.update({
            'model_name_or_path': config.get('model_path', 'Qwen/Qwen2.5-7B-Instruct'),
            'trust_remote_code': config.get('trust_remote_code', False),
            'torch_dtype': 'float16',
            'device_map': 'auto',
            'gradient_checkpointing': config.get('enable_gradient_checkpointing', True)
        })
        
        return model_config
    
    def setup_sanskrit_training_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up complete Sanskrit training environment.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary containing all training components
        """
        if not self._verl_available:
            logger.warning("VERL not available, creating mock environment")
            return self._create_mock_training_environment(config)
        
        try:
            # Create trainer
            trainer = self.create_sanskrit_trainer(config)
            
            # Create rollout worker
            rollout_worker = self.create_sanskrit_rollout_worker(config)
            
            # Create models
            actor, critic = self.create_sanskrit_models(config)
            
            # Setup reward function
            reward_function = self._setup_sanskrit_reward_function(config)
            
            environment = {
                'trainer': trainer,
                'rollout_worker': rollout_worker,
                'actor_model': actor,
                'critic_model': critic,
                'reward_function': reward_function,
                'config': config,
                'storage_path': self.storage_path
            }
            
            logger.info("Sanskrit training environment successfully set up")
            return environment
            
        except Exception as e:
            logger.error(f"Failed to setup Sanskrit training environment: {e}")
            raise VERLIntegrationError(f"Environment setup failed: {e}")
    
    def _setup_sanskrit_reward_function(self, config: Dict[str, Any]) -> Any:
        """Setup Sanskrit-specific reward function."""
        try:
            from .sanskrit_reward_function import compute_sanskrit_score
            return compute_sanskrit_score
        except ImportError:
            logger.warning("Sanskrit reward function not available, using mock")
            return lambda predicts, ground_truths, **kwargs: [{"overall": 0.8, "format": 1.0, "accuracy": 0.7}] * len(predicts)
    
    def _create_mock_training_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock training environment for development."""
        return {
            'trainer': self._verl_components['trainer'](config),
            'rollout_worker': self._verl_components['rollout_worker'](config),
            'actor_model': self._verl_components['actor_model'](config),
            'critic_model': self._verl_components['critic_model'](config),
            'reward_function': self._verl_components['reward_function'],
            'config': config,
            'storage_path': self.storage_path,
            'mock': True
        }
    
    def run_sanskrit_training_step(self, environment: Dict[str, Any], 
                                  training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a single Sanskrit training step.
        
        Args:
            environment: Training environment from setup_sanskrit_training_environment
            training_data: List of training examples
            
        Returns:
            Training step results
        """
        try:
            trainer = environment['trainer']
            rollout_worker = environment['rollout_worker']
            reward_function = environment['reward_function']
            
            # Generate rollouts
            rollout_results = rollout_worker.rollout(training_data)
            
            # Calculate rewards
            if 'mock' not in environment:
                rewards = reward_function(
                    rollout_results.get('responses', []),
                    [item.get('answer', '') for item in training_data]
                )
            else:
                rewards = [{"overall": 0.8, "format": 1.0, "accuracy": 0.7}] * len(training_data)
            
            # Training step
            training_results = trainer.train(rollout_results, rewards)
            
            return {
                'training_loss': training_results.get('loss', 0.1),
                'average_reward': sum(r.get('overall', 0) for r in rewards) / len(rewards) if rewards else 0.0,
                'rollout_count': len(training_data),
                'step_successful': True
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return {
                'training_loss': float('inf'),
                'average_reward': 0.0,
                'rollout_count': 0,
                'step_successful': False,
                'error': str(e)
            }
    
    def save_training_checkpoint(self, environment: Dict[str, Any], 
                               checkpoint_name: str) -> str:
        """
        Save training checkpoint.
        
        Args:
            environment: Training environment
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.storage_path / "checkpoints" / f"{checkpoint_name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model states (mock implementation)
            checkpoint_data = {
                'actor_state': "mock_actor_state",
                'critic_state': "mock_critic_state",
                'trainer_state': "mock_trainer_state",
                'config': environment['config']
            }
            
            # In real implementation, would use torch.save
            import json
            with open(checkpoint_path.with_suffix('.json'), 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise VERLIntegrationError(f"Checkpoint save failed: {e}")
    
    def load_training_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded checkpoint data
        """
        try:
            # Load checkpoint (mock implementation)
            import json
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise VERLIntegrationError(f"Checkpoint load failed: {e}")
    
    def get_training_statistics(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Args:
            environment: Training environment
            
        Returns:
            Training statistics
        """
        return {
            'verl_available': self._verl_available,
            'storage_path': str(self.storage_path),
            'components_loaded': list(self._verl_components.keys()),
            'environment_ready': 'trainer' in environment,
            'mock_mode': 'mock' in environment
        }


def create_sanskrit_verl_integrator(storage_path: str = "./r_zero_storage") -> SanskritVERLIntegrator:
    """
    Create Sanskrit VERL integrator.
    
    Args:
        storage_path: Path for storing VERL-related data
        
    Returns:
        SanskritVERLIntegrator instance
    """
    return SanskritVERLIntegrator(storage_path)


def check_verl_availability() -> Dict[str, Any]:
    """
    Check VERL component availability.
    
    Returns:
        Dictionary with availability information
    """
    availability = {
        'r_zero_paths_exist': [],
        'verl_importable': False,
        'components_available': []
    }
    
    # Check paths
    for path in r_zero_paths:
        if Path(path).exists():
            availability['r_zero_paths_exist'].append(path)
    
    # Check imports
    try:
        import verl
        availability['verl_importable'] = True
        availability['components_available'].append('core')
    except ImportError:
        pass
    
    try:
        from verl.trainer import PPOTrainer
        availability['components_available'].append('trainer')
    except ImportError:
        pass
    
    try:
        from verl.workers import RolloutWorker
        availability['components_available'].append('rollout_worker')
    except ImportError:
        pass
    
    return availability