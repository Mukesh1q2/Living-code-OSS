"""
MCP (Model Context Protocol) Server Integration for Sanskrit Rewrite Engine

This module provides a secure MCP server that enables controlled file system access,
workspace management, and safe execution environment for the Sanskrit processing system.
Implements requirements 13.1, 13.2, 13.3 for standardized interfaces and external integration.
"""

import os
import json
import asyncio
import logging
import subprocess
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None
from contextlib import contextmanager

# MCP Protocol imports (would be from actual MCP library)
try:
    from mcp import Server, Tool, Resource
    from mcp.types import TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    # Mock MCP classes for development
    MCP_AVAILABLE = False
    
    class Server:
        def __init__(self, name: str, version: str): pass
        def list_tools(self): return []
        def call_tool(self, name: str, arguments: dict): pass
        def list_resources(self): return []
        def read_resource(self, uri: str): pass
    
    class Tool:
        def __init__(self, name: str, description: str, input_schema: dict): pass
    
    class Resource:
        def __init__(self, uri: str, name: str, description: str, mime_type: str): pass
    
    class TextContent:
        def __init__(self, type: str, text: str): pass
    
    class ImageContent:
        def __init__(self, type: str, data: str, mime_type: str): pass
    
    class EmbeddedResource:
        def __init__(self, type: str, resource: dict): pass

# Import Sanskrit engine components
from .tokenizer import SanskritTokenizer
from .panini_engine import PaniniRuleEngine, PaniniEngineResult
from .essential_sutras import create_essential_sutras
from .semantic_pipeline import process_sanskrit_text, ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for MCP server."""
    allowed_directories: List[str] = field(default_factory=list)
    blocked_extensions: List[str] = field(default_factory=lambda: ['.exe', '.bat', '.sh', '.ps1'])
    max_file_size_mb: int = 10
    enable_git_operations: bool = True
    enable_code_execution: bool = False
    execution_timeout_seconds: int = 30
    audit_log_path: Optional[str] = None
    require_confirmation: bool = True


@dataclass
class WorkspaceConfig:
    """Workspace configuration for project management."""
    root_path: str
    project_name: str
    git_repo_path: Optional[str] = None
    sanskrit_corpus_path: Optional[str] = None
    output_directory: str = "output"
    temp_directory: str = "temp"
    backup_directory: str = "backups"


@dataclass
class AuditLogEntry:
    """Audit log entry for security tracking."""
    timestamp: datetime
    operation: str
    file_path: str
    user_id: Optional[str]
    success: bool
    error_message: Optional[str] = None
    file_hash: Optional[str] = None


class SecuritySandbox:
    """Security sandbox for safe file operations and code execution."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log: List[AuditLogEntry] = []
        
    def is_path_allowed(self, file_path: str) -> bool:
        """Check if file path is within allowed directories."""
        abs_path = os.path.abspath(file_path)
        
        # Check if path is within allowed directories
        for allowed_dir in self.config.allowed_directories:
            allowed_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_abs):
                return True
        
        return False
    
    def is_extension_allowed(self, file_path: str) -> bool:
        """Check if file extension is allowed."""
        ext = Path(file_path).suffix.lower()
        return ext not in self.config.blocked_extensions
    
    def check_file_size(self, file_path: str) -> bool:
        """Check if file size is within limits."""
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return size_mb <= self.config.max_file_size_mb
        except OSError:
            return False
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for audit trail."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except OSError:
            return ""
    
    def log_operation(self, operation: str, file_path: str, success: bool, 
                     user_id: Optional[str] = None, error_message: Optional[str] = None):
        """Log operation to audit trail."""
        file_hash = self.calculate_file_hash(file_path) if success and os.path.exists(file_path) else None
        
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            operation=operation,
            file_path=file_path,
            user_id=user_id,
            success=success,
            error_message=error_message,
            file_hash=file_hash
        )
        
        self.audit_log.append(entry)
        
        # Write to audit log file if configured
        if self.config.audit_log_path:
            self._write_audit_log(entry)
    
    def _write_audit_log(self, entry: AuditLogEntry):
        """Write audit log entry to file."""
        try:
            with open(self.config.audit_log_path, 'a', encoding='utf-8') as f:
                log_data = {
                    'timestamp': entry.timestamp.isoformat(),
                    'operation': entry.operation,
                    'file_path': entry.file_path,
                    'user_id': entry.user_id,
                    'success': entry.success,
                    'error_message': entry.error_message,
                    'file_hash': entry.file_hash
                }
                f.write(json.dumps(log_data) + '\n')
        except OSError as e:
            logger.error(f"Failed to write audit log: {e}")
    
    @contextmanager
    def safe_execution_environment(self):
        """Create a safe execution environment with temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="sanskrit_mcp_")
        try:
            # Set restrictive permissions
            os.chmod(temp_dir, 0o700)
            yield temp_dir
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


class WorkspaceManager:
    """Manages project workspace and file operations."""
    
    def __init__(self, config: WorkspaceConfig, security: SecuritySandbox):
        self.config = config
        self.security = security
        self.root_path = Path(config.root_path)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.root_path,
            self.root_path / self.config.output_directory,
            self.root_path / self.config.temp_directory,
            self.root_path / self.config.backup_directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def list_files(self, directory: str = "", pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in workspace directory."""
        try:
            target_dir = self.root_path / directory if directory else self.root_path
            
            if not self.security.is_path_allowed(str(target_dir)):
                raise PermissionError(f"Access denied to directory: {directory}")
            
            files = []
            for file_path in target_dir.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        'name': file_path.name,
                        'path': str(file_path.relative_to(self.root_path)),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_path.suffix.lower()
                    })
            
            self.security.log_operation("list_files", str(target_dir), True)
            return files
            
        except Exception as e:
            self.security.log_operation("list_files", directory, False, error_message=str(e))
            raise
    
    def read_file(self, file_path: str) -> str:
        """Read file content with security checks."""
        try:
            full_path = self.root_path / file_path
            
            if not self.security.is_path_allowed(str(full_path)):
                raise PermissionError(f"Access denied to file: {file_path}")
            
            if not self.security.check_file_size(str(full_path)):
                raise ValueError(f"File too large: {file_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.security.log_operation("read_file", str(full_path), True)
            return content
            
        except Exception as e:
            self.security.log_operation("read_file", file_path, False, error_message=str(e))
            raise
    
    def write_file(self, file_path: str, content: str, backup: bool = True) -> bool:
        """Write file content with security checks and optional backup."""
        try:
            full_path = self.root_path / file_path
            
            if not self.security.is_path_allowed(str(full_path)):
                raise PermissionError(f"Access denied to file: {file_path}")
            
            if not self.security.is_extension_allowed(file_path):
                raise PermissionError(f"File extension not allowed: {file_path}")
            
            # Create backup if file exists and backup is requested
            if backup and full_path.exists():
                self._create_backup(full_path)
            
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.security.log_operation("write_file", str(full_path), True)
            return True
            
        except Exception as e:
            self.security.log_operation("write_file", file_path, False, error_message=str(e))
            raise
    
    def delete_file(self, file_path: str, backup: bool = True) -> bool:
        """Delete file with security checks and optional backup."""
        try:
            full_path = self.root_path / file_path
            
            if not self.security.is_path_allowed(str(full_path)):
                raise PermissionError(f"Access denied to file: {file_path}")
            
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Create backup if requested
            if backup:
                self._create_backup(full_path)
            
            # Delete file
            full_path.unlink()
            
            self.security.log_operation("delete_file", str(full_path), True)
            return True
            
        except Exception as e:
            self.security.log_operation("delete_file", file_path, False, error_message=str(e))
            raise
    
    def _create_backup(self, file_path: Path):
        """Create backup of file."""
        backup_dir = self.root_path / self.config.backup_directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")


class GitIntegration:
    """Git integration for version control operations."""
    
    def __init__(self, repo_path: str, security: SecuritySandbox):
        self.repo_path = repo_path
        self.security = security
        self.repo = None
        self._initialize_repo()
    
    def _initialize_repo(self):
        """Initialize or open Git repository."""
        if not GIT_AVAILABLE:
            logger.warning("Git not available, Git integration disabled")
            return
            
        try:
            if os.path.exists(os.path.join(self.repo_path, '.git')):
                self.repo = git.Repo(self.repo_path)
            else:
                self.repo = git.Repo.init(self.repo_path)
            logger.info(f"Git repository initialized at {self.repo_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        if not self.repo:
            return {"error": "Git repository not available"}
        
        try:
            status = {
                "branch": self.repo.active_branch.name,
                "is_dirty": self.repo.is_dirty(),
                "untracked_files": self.repo.untracked_files,
                "modified_files": [item.a_path for item in self.repo.index.diff(None)],
                "staged_files": [item.a_path for item in self.repo.index.diff("HEAD")]
            }
            
            self.security.log_operation("git_status", self.repo_path, True)
            return status
            
        except Exception as e:
            self.security.log_operation("git_status", self.repo_path, False, error_message=str(e))
            raise
    
    def add_files(self, file_paths: List[str]) -> bool:
        """Add files to Git staging area."""
        if not self.repo:
            raise RuntimeError("Git repository not available")
        
        try:
            self.repo.index.add(file_paths)
            self.security.log_operation("git_add", f"files: {file_paths}", True)
            return True
            
        except Exception as e:
            self.security.log_operation("git_add", f"files: {file_paths}", False, error_message=str(e))
            raise
    
    def commit(self, message: str, author_name: str = "MCP Server", 
               author_email: str = "mcp@sanskrit-engine.local") -> str:
        """Commit staged changes."""
        if not self.repo:
            raise RuntimeError("Git repository not available")
        
        try:
            # Configure author
            with self.repo.config_writer() as git_config:
                git_config.set_value("user", "name", author_name)
                git_config.set_value("user", "email", author_email)
            
            # Commit changes
            commit = self.repo.index.commit(message)
            commit_hash = commit.hexsha[:8]
            
            self.security.log_operation("git_commit", f"commit: {commit_hash}", True)
            return commit_hash
            
        except Exception as e:
            self.security.log_operation("git_commit", "commit", False, error_message=str(e))
            raise
    
    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get commit history."""
        if not self.repo:
            return []
        
        try:
            commits = []
            for commit in self.repo.iter_commits(max_count=max_count):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                    "files_changed": len(commit.stats.files)
                })
            
            self.security.log_operation("git_history", self.repo_path, True)
            return commits
            
        except Exception as e:
            self.security.log_operation("git_history", self.repo_path, False, error_message=str(e))
            raise


class SanskritMCPServer:
    """Main MCP server for Sanskrit Rewrite Engine integration."""
    
    def __init__(self, workspace_config: WorkspaceConfig, security_config: SecurityConfig):
        self.workspace_config = workspace_config
        self.security_config = security_config
        
        # Initialize components
        self.security = SecuritySandbox(security_config)
        self.workspace = WorkspaceManager(workspace_config, self.security)
        self.git = GitIntegration(workspace_config.root_path, self.security) if security_config.enable_git_operations else None
        
        # Initialize Sanskrit processing components
        self.tokenizer = SanskritTokenizer()
        self.rule_engine = None
        self._initialize_sanskrit_engine()
        
        # Initialize MCP server
        self.server = Server("sanskrit-rewrite-engine", "1.0.0")
        self._register_tools()
        self._register_resources()
    
    def _initialize_sanskrit_engine(self):
        """Initialize Sanskrit processing engine."""
        try:
            essential_sutras = create_essential_sutras()
            # Initialize rule engine with essential sutras
            # This would be implemented based on the actual PaniniRuleEngine
            logger.info("Sanskrit processing engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Sanskrit engine: {e}")
    
    def _register_tools(self):
        """Register MCP tools for file operations and Sanskrit processing."""
        
        # File system tools
        self.server.list_tools().extend([
            Tool(
                name="list_files",
                description="List files in workspace directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path (relative to workspace)"},
                        "pattern": {"type": "string", "description": "File pattern (default: '*')"}
                    }
                }
            ),
            Tool(
                name="read_file",
                description="Read file content from workspace",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path relative to workspace"}
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="write_file",
                description="Write content to file in workspace",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path relative to workspace"},
                        "content": {"type": "string", "description": "File content to write"},
                        "backup": {"type": "boolean", "description": "Create backup before writing (default: true)"}
                    },
                    "required": ["file_path", "content"]
                }
            ),
            Tool(
                name="delete_file",
                description="Delete file from workspace",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path relative to workspace"},
                        "backup": {"type": "boolean", "description": "Create backup before deleting (default: true)"}
                    },
                    "required": ["file_path"]
                }
            )
        ])
        
        # Git tools
        if self.git:
            self.server.list_tools().extend([
                Tool(
                    name="git_status",
                    description="Get Git repository status",
                    input_schema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="git_add",
                    description="Add files to Git staging area",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "file_paths": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["file_paths"]
                    }
                ),
                Tool(
                    name="git_commit",
                    description="Commit staged changes",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "author_name": {"type": "string", "description": "Author name"},
                            "author_email": {"type": "string", "description": "Author email"}
                        },
                        "required": ["message"]
                    }
                ),
                Tool(
                    name="git_history",
                    description="Get commit history",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "max_count": {"type": "integer", "description": "Maximum number of commits (default: 10)"}
                        }
                    }
                )
            ])
        
        # Sanskrit processing tools
        self.server.list_tools().extend([
            Tool(
                name="process_sanskrit_text",
                description="Process Sanskrit text using the rewrite engine",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Sanskrit text to process"},
                        "options": {
                            "type": "object",
                            "properties": {
                                "enable_tracing": {"type": "boolean", "description": "Enable detailed tracing"},
                                "max_passes": {"type": "integer", "description": "Maximum processing passes"}
                            }
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="tokenize_sanskrit",
                description="Tokenize Sanskrit text",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Sanskrit text to tokenize"}
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="get_audit_log",
                description="Get security audit log",
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum number of entries (default: 100)"}
                    }
                }
            )
        ])
    
    def _register_resources(self):
        """Register MCP resources for workspace files and Sanskrit data."""
        # This would register resources that can be read by the MCP client
        pass
    
    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Handle MCP tool calls."""
        try:
            if name == "list_files":
                return await self._handle_list_files(arguments)
            elif name == "read_file":
                return await self._handle_read_file(arguments)
            elif name == "write_file":
                return await self._handle_write_file(arguments)
            elif name == "delete_file":
                return await self._handle_delete_file(arguments)
            elif name == "git_status":
                return await self._handle_git_status(arguments)
            elif name == "git_add":
                return await self._handle_git_add(arguments)
            elif name == "git_commit":
                return await self._handle_git_commit(arguments)
            elif name == "git_history":
                return await self._handle_git_history(arguments)
            elif name == "process_sanskrit_text":
                return await self._handle_process_sanskrit_text(arguments)
            elif name == "tokenize_sanskrit":
                return await self._handle_tokenize_sanskrit(arguments)
            elif name == "get_audit_log":
                return await self._handle_get_audit_log(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Tool call failed: {name} - {e}")
            return {"error": str(e)}
    
    async def _handle_list_files(self, arguments: dict) -> dict:
        """Handle list_files tool call."""
        directory = arguments.get("directory", "")
        pattern = arguments.get("pattern", "*")
        
        files = self.workspace.list_files(directory, pattern)
        return {"files": files}
    
    async def _handle_read_file(self, arguments: dict) -> dict:
        """Handle read_file tool call."""
        file_path = arguments["file_path"]
        content = self.workspace.read_file(file_path)
        return {"content": content}
    
    async def _handle_write_file(self, arguments: dict) -> dict:
        """Handle write_file tool call."""
        file_path = arguments["file_path"]
        content = arguments["content"]
        backup = arguments.get("backup", True)
        
        success = self.workspace.write_file(file_path, content, backup)
        return {"success": success}
    
    async def _handle_delete_file(self, arguments: dict) -> dict:
        """Handle delete_file tool call."""
        file_path = arguments["file_path"]
        backup = arguments.get("backup", True)
        
        success = self.workspace.delete_file(file_path, backup)
        return {"success": success}
    
    async def _handle_git_status(self, arguments: dict) -> dict:
        """Handle git_status tool call."""
        if not self.git:
            return {"error": "Git integration not enabled"}
        
        status = self.git.get_status()
        return status
    
    async def _handle_git_add(self, arguments: dict) -> dict:
        """Handle git_add tool call."""
        if not self.git:
            return {"error": "Git integration not enabled"}
        
        file_paths = arguments["file_paths"]
        success = self.git.add_files(file_paths)
        return {"success": success}
    
    async def _handle_git_commit(self, arguments: dict) -> dict:
        """Handle git_commit tool call."""
        if not self.git:
            return {"error": "Git integration not enabled"}
        
        message = arguments["message"]
        author_name = arguments.get("author_name", "MCP Server")
        author_email = arguments.get("author_email", "mcp@sanskrit-engine.local")
        
        commit_hash = self.git.commit(message, author_name, author_email)
        return {"commit_hash": commit_hash}
    
    async def _handle_git_history(self, arguments: dict) -> dict:
        """Handle git_history tool call."""
        if not self.git:
            return {"error": "Git integration not enabled"}
        
        max_count = arguments.get("max_count", 10)
        history = self.git.get_commit_history(max_count)
        return {"commits": history}
    
    async def _handle_process_sanskrit_text(self, arguments: dict) -> dict:
        """Handle process_sanskrit_text tool call."""
        text = arguments["text"]
        options = arguments.get("options", {})
        
        try:
            # Process Sanskrit text using the semantic pipeline
            result = process_sanskrit_text(text)
            
            # Convert result to serializable format
            return {
                "input_text": text,
                "processed_result": {
                    "tokens": [{"text": token.text, "kind": token.kind.value, "tags": list(token.tags)} 
                              for token in result.tokens] if hasattr(result, 'tokens') else [],
                    "semantic_graph": result.semantic_graph.to_dict() if hasattr(result, 'semantic_graph') else {},
                    "transformations": result.transformations if hasattr(result, 'transformations') else [],
                    "success": True
                }
            }
            
        except Exception as e:
            return {
                "input_text": text,
                "processed_result": {
                    "success": False,
                    "error": str(e)
                }
            }
    
    async def _handle_tokenize_sanskrit(self, arguments: dict) -> dict:
        """Handle tokenize_sanskrit tool call."""
        text = arguments["text"]
        
        try:
            tokens = self.tokenizer.tokenize(text)
            return {
                "input_text": text,
                "tokens": [
                    {
                        "text": token.text,
                        "kind": token.kind.value,
                        "tags": list(token.tags),
                        "meta": dict(token.meta)
                    }
                    for token in tokens
                ]
            }
            
        except Exception as e:
            return {
                "input_text": text,
                "error": str(e)
            }
    
    async def _handle_get_audit_log(self, arguments: dict) -> dict:
        """Handle get_audit_log tool call."""
        limit = arguments.get("limit", 100)
        
        recent_entries = self.security.audit_log[-limit:] if len(self.security.audit_log) > limit else self.security.audit_log
        
        return {
            "audit_entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "operation": entry.operation,
                    "file_path": entry.file_path,
                    "user_id": entry.user_id,
                    "success": entry.success,
                    "error_message": entry.error_message,
                    "file_hash": entry.file_hash
                }
                for entry in recent_entries
            ]
        }


def create_mcp_server(workspace_path: str, allowed_directories: List[str] = None) -> SanskritMCPServer:
    """Create and configure MCP server for Sanskrit Rewrite Engine."""
    
    # Default allowed directories
    if allowed_directories is None:
        allowed_directories = [
            workspace_path,
            os.path.join(workspace_path, "sanskrit_corpus"),
            os.path.join(workspace_path, "output"),
            os.path.join(workspace_path, "temp")
        ]
    
    # Security configuration
    security_config = SecurityConfig(
        allowed_directories=allowed_directories,
        blocked_extensions=['.exe', '.bat', '.sh', '.ps1', '.dll', '.so'],
        max_file_size_mb=50,
        enable_git_operations=True,
        enable_code_execution=False,
        execution_timeout_seconds=30,
        audit_log_path=os.path.join(workspace_path, "mcp_audit.log"),
        require_confirmation=True
    )
    
    # Workspace configuration
    workspace_config = WorkspaceConfig(
        root_path=workspace_path,
        project_name="sanskrit-rewrite-engine",
        git_repo_path=workspace_path,
        sanskrit_corpus_path=os.path.join(workspace_path, "sanskrit_corpus"),
        output_directory="output",
        temp_directory="temp",
        backup_directory="backups"
    )
    
    return SanskritMCPServer(workspace_config, security_config)


async def run_mcp_server(server: SanskritMCPServer, host: str = "localhost", port: int = 8000):
    """Run the MCP server."""
    logger.info(f"Starting Sanskrit MCP Server on {host}:{port}")
    
    # This would start the actual MCP server
    # For now, we'll simulate server operation
    try:
        while True:
            await asyncio.sleep(1)
            # Server would handle MCP protocol messages here
            
    except KeyboardInterrupt:
        logger.info("MCP Server stopped")


if __name__ == "__main__":
    # Example usage
    import sys
    
    workspace_path = sys.argv[1] if len(sys.argv) > 1 else "."
    server = create_mcp_server(workspace_path)
    
    # Run server
    asyncio.run(run_mcp_server(server))