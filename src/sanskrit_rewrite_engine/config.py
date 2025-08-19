"""
Configuration management for the Sanskrit Rewrite Engine.

This module provides configuration classes and utilities for managing
engine settings and behavior.
"""

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


@dataclass
class EngineConfig:
    """Configuration for the Sanskrit Rewrite Engine."""
    
    # Processing settings
    max_iterations: int = 10
    enable_tracing: bool = True
    max_trace_depth: int = 100
    
    # Rule settings
    rule_directories: List[str] = field(default_factory=lambda: ["data/rules"])
    default_rule_set: str = "basic_sandhi"
    
    # Advanced rule application settings
    enable_infinite_loop_detection: bool = True
    max_same_rule_applications: int = 50
    max_text_state_repetitions: int = 5
    enable_rule_conflict_resolution: bool = True
    enable_conditional_rule_activation: bool = True
    
    # Tokenization settings
    preserve_whitespace: bool = True
    preserve_markers: bool = True
    
    # Performance settings
    max_text_length: int = 10000
    timeout_seconds: int = 30
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Memory settings
    memory_limit_mb: int = 500
    
    # Lazy processing settings
    chunk_size: int = 1000
    enable_lazy_processing: bool = True
    
    # Debug settings
    debug_mode: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'EngineConfig':
        """Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            EngineConfig instance
            
        Raises:
            ValueError: If file format is unsupported or invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.json']:
                    data = json.load(f)
                elif config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                    
            return cls.from_dict(data)
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error parsing configuration file {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineConfig':
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            EngineConfig instance
        """
        # Filter out unknown keys to avoid TypeError
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'max_iterations': self.max_iterations,
            'enable_tracing': self.enable_tracing,
            'max_trace_depth': self.max_trace_depth,
            'rule_directories': self.rule_directories,
            'default_rule_set': self.default_rule_set,
            'enable_infinite_loop_detection': self.enable_infinite_loop_detection,
            'max_same_rule_applications': self.max_same_rule_applications,
            'max_text_state_repetitions': self.max_text_state_repetitions,
            'enable_rule_conflict_resolution': self.enable_rule_conflict_resolution,
            'enable_conditional_rule_activation': self.enable_conditional_rule_activation,
            'preserve_whitespace': self.preserve_whitespace,
            'preserve_markers': self.preserve_markers,
            'max_text_length': self.max_text_length,
            'timeout_seconds': self.timeout_seconds,
            'cache_size': self.cache_size,
            'cache_ttl': self.cache_ttl,
            'memory_limit_mb': self.memory_limit_mb,
            'chunk_size': self.chunk_size,
            'enable_lazy_processing': self.enable_lazy_processing,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        data = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.json']:
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                # Default to JSON
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        
        if self.max_trace_depth <= 0:
            errors.append("max_trace_depth must be positive")
        
        if self.max_text_length <= 0:
            errors.append("max_text_length must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if not self.rule_directories:
            errors.append("rule_directories cannot be empty")
        
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append(f"Invalid log_level: {self.log_level}")
        
        if self.max_same_rule_applications <= 0:
            errors.append("max_same_rule_applications must be positive")
        
        if self.max_text_state_repetitions <= 0:
            errors.append("max_text_state_repetitions must be positive")
        
        return errors
    
    def merge(self, other: 'EngineConfig') -> 'EngineConfig':
        """Merge this configuration with another, with other taking precedence.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            New merged configuration
        """
        merged_data = self.to_dict()
        merged_data.update(other.to_dict())
        return self.from_dict(merged_data)


def load_default_config() -> EngineConfig:
    """Load default configuration.
    
    Returns:
        Default EngineConfig instance
    """
    return EngineConfig()


def load_config_with_overrides(
    base_config: Optional[EngineConfig] = None,
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> EngineConfig:
    """Load configuration with optional file and runtime overrides.
    
    Args:
        base_config: Base configuration (defaults to default config)
        config_file: Optional configuration file to load
        overrides: Optional runtime overrides
        
    Returns:
        Final configuration with all overrides applied
    """
    # Start with base config
    if base_config is None:
        config = load_default_config()
    else:
        config = base_config
    
    # Apply file-based config
    if config_file is not None:
        file_config = EngineConfig.from_file(config_file)
        config = config.merge(file_config)
    
    # Apply runtime overrides
    if overrides is not None:
        override_config = EngineConfig.from_dict(overrides)
        config = config.merge(override_config)
    
    # Validate final configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
    
    return config