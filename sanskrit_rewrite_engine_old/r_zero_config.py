"""
R-Zero configuration for Sanskrit reasoning tasks.

This module provides configuration management for integrating R-Zero framework
with Sanskrit grammatical reasoning tasks.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SanskritRZeroConfig:
    """Configuration for Sanskrit R-Zero integration."""
    
    # Storage paths
    storage_path: str = "./r_zero_storage"
    model_path: str = "./r_zero_models"
    dataset_path: str = "./sanskrit_datasets"
    checkpoint_path: str = "./r_zero_checkpoints"
    
    # Sanskrit-specific parameters
    sanskrit_corpus_path: str = "./sanskrit_corpus"
    sutra_rules_path: str = "./sutra_rules"
    max_sanskrit_length: int = 512
    max_derivation_steps: int = 20
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-6
    max_epochs: int = 10
    validation_split: float = 0.2
    
    # R-Zero specific
    rollout_batch_size: int = 256
    temperature: float = 1.0
    top_p: float = 0.99
    kl_coef: float = 1e-2
    
    # Sanskrit reward weights
    grammatical_weight: float = 0.4
    semantic_weight: float = 0.3
    efficiency_weight: float = 0.3
    
    def __post_init__(self):
        """Ensure all paths exist."""
        for path_attr in ['storage_path', 'model_path', 'dataset_path', 
                         'checkpoint_path', 'sanskrit_corpus_path', 'sutra_rules_path']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SanskritRZeroConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'SanskritRZeroConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class RZeroEnvironmentSetup:
    """Setup R-Zero environment for Sanskrit reasoning."""
    
    def __init__(self, config: SanskritRZeroConfig):
        self.config = config
        self.r_zero_path = Path("external_models/r-zero")
        self.main_r_zero_path = Path("R-Zero-main")
    
    def setup_environment(self) -> None:
        """Set up the R-Zero environment."""
        # Set environment variables
        os.environ['STORAGE_PATH'] = self.config.storage_path
        os.environ['MODEL_PATH'] = self.config.model_path
        os.environ['SANSKRIT_CORPUS_PATH'] = self.config.sanskrit_corpus_path
        
        # Create necessary directories
        self._create_directories()
        
        # Copy and adapt configuration files
        self._setup_config_files()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.storage_path,
            self.config.model_path,
            self.config.dataset_path,
            self.config.checkpoint_path,
            self.config.sanskrit_corpus_path,
            self.config.sutra_rules_path,
            f"{self.config.storage_path}/logs",
            f"{self.config.storage_path}/traces",
            f"{self.config.storage_path}/rewards"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_config_files(self) -> None:
        """Set up R-Zero configuration files for Sanskrit."""
        # Create Sanskrit-specific config.yaml
        sanskrit_config = self._create_sanskrit_config()
        
        config_path = Path(self.config.storage_path) / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sanskrit_config, f, default_flow_style=False, allow_unicode=True)
    
    def _create_sanskrit_config(self) -> Dict[str, Any]:
        """Create Sanskrit-specific R-Zero configuration."""
        return {
            'data': {
                'train_files': f'{self.config.dataset_path}/sanskrit_train.json',
                'val_files': f'{self.config.dataset_path}/sanskrit_val.json',
                'prompt_key': 'sanskrit_problem',
                'answer_key': 'expected_solution',
                'max_prompt_length': self.config.max_sanskrit_length,
                'max_response_length': self.config.max_sanskrit_length,
                'rollout_batch_size': self.config.rollout_batch_size,
                'val_batch_size': self.config.batch_size,
                'format_prompt': './sanskrit_format.jinja',
                'shuffle': True,
                'seed': 42,
                'filter_overlong_prompts': True
            },
            'algorithm': {
                'adv_estimator': 'grpo',
                'disable_kl': False,
                'use_kl_loss': True,
                'kl_penalty': 'low_var_kl',
                'kl_coef': self.config.kl_coef,
                'mock_data': 'test'
            },
            'worker': {
                'actor': {
                    'global_batch_size': self.config.batch_size,
                    'micro_batch_size_per_device_for_update': 2,
                    'micro_batch_size_per_device_for_experience': 8,
                    'max_grad_norm': 1.0,
                    'padding_free': True,
                    'model': {
                        'model_path': 'Qwen/Qwen2.5-7B-Instruct',
                        'enable_gradient_checkpointing': True,
                        'trust_remote_code': False
                    },
                    'optim': {
                        'lr': self.config.learning_rate,
                        'weight_decay': 1e-2,
                        'strategy': 'adamw',
                        'lr_warmup_ratio': 0.0
                    },
                    'fsdp': {
                        'enable_full_shard': True,
                        'enable_cpu_offload': False,
                        'enable_rank0_init': True
                    },
                    'offload': {
                        'offload_params': True,
                        'offload_optimizer': True
                    }
                },
                'rollout': {
                    'n': 5,
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'gpu_memory_utilization': 0.7,
                    'enforce_eager': False,
                    'enable_chunked_prefill': False,
                    'tensor_parallel_size': 1,  # Adjusted for single GPU
                    'val_override_config': {
                        'temperature': 1.0,
                        'n': 1
                    }
                },
                'ref': {
                    'fsdp': {
                        'enable_full_shard': True,
                        'enable_cpu_offload': True,
                        'enable_rank0_init': True
                    },
                    'offload': {
                        'offload_params': True
                    }
                },
                'reward': {
                    'reward_type': 'batch',
                    'reward_function': './sanskrit_reward_function.py:compute_sanskrit_score'
                }
            },
            'trainer': {
                'total_epochs': self.config.max_epochs,
                'max_steps': None,
                'project_name': 'sanskrit_r_zero',
                'experiment_name': 'sanskrit_grammatical_reasoning',
                'logger': ['console', 'wandb'],
                'nnodes': 1,
                'n_gpus_per_node': 1,  # Adjusted for single GPU
                'val_freq': 3,
                'val_before_train': True,
                'val_only': False,
                'val_generations_to_log': 3,
                'save_freq': 5,
                'save_limit': 3,
                'save_checkpoint_path': self.config.checkpoint_path,
                'load_checkpoint_path': None
            }
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'storage_path': os.environ.get('STORAGE_PATH'),
            'model_path': os.environ.get('MODEL_PATH'),
            'sanskrit_corpus_path': os.environ.get('SANSKRIT_CORPUS_PATH'),
            'r_zero_available': self.r_zero_path.exists(),
            'main_r_zero_available': self.main_r_zero_path.exists(),
            'config_file': Path(self.config.storage_path) / "config.yaml"
        }


def setup_sanskrit_r_zero_environment(config_path: Optional[str] = None) -> RZeroEnvironmentSetup:
    """
    Set up Sanskrit R-Zero environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        RZeroEnvironmentSetup instance
    """
    if config_path and Path(config_path).exists():
        config = SanskritRZeroConfig.load_from_yaml(config_path)
    else:
        config = SanskritRZeroConfig()
    
    setup = RZeroEnvironmentSetup(config)
    setup.setup_environment()
    
    return setup