"""
Configuration management for Sanskrit Rewrite Engine MCP Server

This module provides configuration management, validation, and default settings
for the MCP server integration.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Main MCP server configuration."""
    # Server settings
    server_name: str = "sanskrit-rewrite-engine"
    server_version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    
    # Workspace settings
    workspace_path: str = "."
    project_name: str = "sanskrit-project"
    
    # Security settings
    allowed_directories: List[str] = field(default_factory=list)
    blocked_extensions: List[str] = field(default_factory=lambda: [
        '.exe', '.bat', '.sh', '.ps1', '.dll', '.so', '.dylib'
    ])
    max_file_size_mb: int = 50
    enable_git_operations: bool = True
    enable_code_execution: bool = False
    require_confirmation: bool = True
    
    # Execution limits
    max_execution_time: int = 30
    max_memory_mb: int = 256
    max_cpu_percent: int = 50
    max_open_files: int = 100
    max_processes: int = 5
    
    # Allowed Python modules for code execution
    allowed_modules: List[str] = field(default_factory=lambda: [
        'math', 'json', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'string', 're', 'unicodedata',
        'decimal', 'fractions', 'statistics', 'random'
    ])
    
    # Blocked Python modules
    blocked_modules: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'telnetlib', 'ssl', 'hashlib',
        'hmac', 'secrets', 'ctypes', 'multiprocessing', 'threading',
        'asyncio', 'concurrent', 'queue', 'pickle', 'shelve',
        'dbm', 'sqlite3', 'importlib', 'pkgutil', 'modulefinder'
    ])
    
    # Logging settings
    log_level: str = "INFO"
    audit_log_enabled: bool = True
    audit_log_path: Optional[str] = None
    execution_audit_enabled: bool = True
    
    # Sanskrit processing settings
    enable_sanskrit_processing: bool = True
    sanskrit_corpus_path: Optional[str] = None
    enable_tracing: bool = True
    max_processing_passes: int = 20
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_compression: bool = True
    
    # Backup settings
    enable_backups: bool = True
    backup_directory: str = "backups"
    max_backup_age_days: int = 30
    max_backup_count: int = 100


@dataclass
class MCPToolConfig:
    """Configuration for individual MCP tools."""
    name: str
    enabled: bool = True
    require_confirmation: bool = False
    rate_limit_per_minute: int = 60
    max_concurrent_calls: int = 5
    timeout_seconds: int = 30
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class MCPConfigManager:
    """Manages MCP server configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = MCPServerConfig()
        self.tool_configs: Dict[str, MCPToolConfig] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Try workspace-specific config first
        workspace_config = Path(".kiro/settings/mcp_server.json")
        if workspace_config.exists():
            return str(workspace_config)
        
        # Fall back to user config
        user_config = Path.home() / ".kiro/settings/mcp_server.json"
        return str(user_config)
    
    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            logger.info(f"Config file not found at {self.config_path}, using defaults")
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update main config
            if 'server' in config_data:
                self._update_config_from_dict(config_data['server'])
            
            # Load tool configurations
            if 'tools' in config_data:
                for tool_name, tool_config in config_data['tools'].items():
                    self.tool_configs[tool_name] = MCPToolConfig(
                        name=tool_name,
                        **tool_config
                    )
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _create_default_config(self):
        """Create default configuration file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Set workspace-specific defaults
            if self.config.workspace_path == ".":
                self.config.workspace_path = os.getcwd()
            
            # Set default allowed directories
            if not self.config.allowed_directories:
                self.config.allowed_directories = [
                    self.config.workspace_path,
                    os.path.join(self.config.workspace_path, "sanskrit_corpus"),
                    os.path.join(self.config.workspace_path, "output"),
                    os.path.join(self.config.workspace_path, "temp")
                ]
            
            # Set default audit log path
            if not self.config.audit_log_path:
                self.config.audit_log_path = os.path.join(
                    self.config.workspace_path, "mcp_audit.log"
                )
            
            # Set default Sanskrit corpus path
            if not self.config.sanskrit_corpus_path:
                self.config.sanskrit_corpus_path = os.path.join(
                    self.config.workspace_path, "sanskrit_corpus"
                )
            
            # Save default configuration
            self.save_config()
            logger.info(f"Created default configuration at {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_data = {
                "server": asdict(self.config),
                "tools": {name: asdict(tool_config) for name, tool_config in self.tool_configs.items()},
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate workspace path
        if not os.path.exists(self.config.workspace_path):
            issues.append(f"Workspace path does not exist: {self.config.workspace_path}")
        
        # Validate allowed directories
        for directory in self.config.allowed_directories:
            if not os.path.exists(directory):
                issues.append(f"Allowed directory does not exist: {directory}")
        
        # Validate numeric limits
        if self.config.max_file_size_mb <= 0:
            issues.append("max_file_size_mb must be positive")
        
        if self.config.max_execution_time <= 0:
            issues.append("max_execution_time must be positive")
        
        if self.config.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        
        if not (0 < self.config.max_cpu_percent <= 100):
            issues.append("max_cpu_percent must be between 1 and 100")
        
        # Validate port
        if not (1 <= self.config.port <= 65535):
            issues.append("port must be between 1 and 65535")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level not in valid_log_levels:
            issues.append(f"log_level must be one of: {valid_log_levels}")
        
        return issues
    
    def get_tool_config(self, tool_name: str) -> MCPToolConfig:
        """Get configuration for a specific tool."""
        if tool_name not in self.tool_configs:
            # Create default tool config
            self.tool_configs[tool_name] = MCPToolConfig(name=tool_name)
        
        return self.tool_configs[tool_name]
    
    def set_tool_config(self, tool_name: str, config: MCPToolConfig):
        """Set configuration for a specific tool."""
        self.tool_configs[tool_name] = config
    
    def get_security_config(self):
        """Get security configuration object."""
        from .mcp_server import SecurityConfig
        
        return SecurityConfig(
            allowed_directories=self.config.allowed_directories,
            blocked_extensions=self.config.blocked_extensions,
            max_file_size_mb=self.config.max_file_size_mb,
            enable_git_operations=self.config.enable_git_operations,
            enable_code_execution=self.config.enable_code_execution,
            execution_timeout_seconds=self.config.max_execution_time,
            audit_log_path=self.config.audit_log_path,
            require_confirmation=self.config.require_confirmation
        )
    
    def get_workspace_config(self):
        """Get workspace configuration object."""
        from .mcp_server import WorkspaceConfig
        
        return WorkspaceConfig(
            root_path=self.config.workspace_path,
            project_name=self.config.project_name,
            git_repo_path=self.config.workspace_path if self.config.enable_git_operations else None,
            sanskrit_corpus_path=self.config.sanskrit_corpus_path,
            output_directory="output",
            temp_directory="temp",
            backup_directory=self.config.backup_directory
        )
    
    def get_execution_limits(self):
        """Get execution limits configuration."""
        from .safe_execution import ExecutionLimits
        
        return ExecutionLimits(
            max_execution_time=self.config.max_execution_time,
            max_memory_mb=self.config.max_memory_mb,
            max_cpu_percent=self.config.max_cpu_percent,
            max_file_size_mb=self.config.max_file_size_mb,
            max_open_files=self.config.max_open_files,
            max_processes=self.config.max_processes,
            allowed_modules=self.config.allowed_modules,
            blocked_modules=self.config.blocked_modules
        )


def create_sample_config(output_path: str):
    """Create a sample configuration file."""
    config_manager = MCPConfigManager()
    
    # Add sample tool configurations
    config_manager.set_tool_config("read_file", MCPToolConfig(
        name="read_file",
        enabled=True,
        require_confirmation=False,
        rate_limit_per_minute=120,
        max_concurrent_calls=10
    ))
    
    config_manager.set_tool_config("write_file", MCPToolConfig(
        name="write_file",
        enabled=True,
        require_confirmation=True,
        rate_limit_per_minute=60,
        max_concurrent_calls=5
    ))
    
    config_manager.set_tool_config("delete_file", MCPToolConfig(
        name="delete_file",
        enabled=True,
        require_confirmation=True,
        rate_limit_per_minute=30,
        max_concurrent_calls=2
    ))
    
    config_manager.set_tool_config("git_commit", MCPToolConfig(
        name="git_commit",
        enabled=True,
        require_confirmation=True,
        rate_limit_per_minute=20,
        max_concurrent_calls=1
    ))
    
    # Save to specified path
    config_manager.config_path = output_path
    config_manager.save_config()


def load_config_from_file(config_path: str) -> MCPConfigManager:
    """Load configuration from specific file."""
    return MCPConfigManager(config_path)


def get_default_config() -> MCPServerConfig:
    """Get default configuration."""
    return MCPServerConfig()


if __name__ == "__main__":
    # Create sample configuration
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "mcp_server_config.json"
    
    create_sample_config(output_path)
    print(f"Sample configuration created at: {output_path}")
    
    # Validate configuration
    config_manager = load_config_from_file(output_path)
    issues = config_manager.validate_config()
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")