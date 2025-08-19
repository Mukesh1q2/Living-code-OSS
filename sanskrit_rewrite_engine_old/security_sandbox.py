"""
Enhanced Security and Sandbox Layer for Sanskrit Rewrite Engine

This module provides a comprehensive security and sandbox layer that implements:
- Safe execution environment for generated code
- Controlled file system access with allowlisted directories
- Resource limits and monitoring for code execution
- Audit logs of all file and system access
- User permission management and access controls

Implements requirements 13.2, 13.3, 13.4 for atomic operations, structured trace data,
and detailed error information with context.
"""

import os
import sys
import json
import time
import threading
import tempfile
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from enum import Enum
import uuid

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """User permission levels."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    RESTRICTED = "restricted"


class SecurityEvent(Enum):
    """Security event types for audit logging."""
    FILE_ACCESS = "file_access"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    CODE_EXECUTION = "code_execution"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SECURITY_VIOLATION = "security_violation"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"


@dataclass
class UserProfile:
    """User profile with permissions and access controls."""
    user_id: str
    username: str
    permission_level: PermissionLevel
    allowed_directories: Set[str] = field(default_factory=set)
    blocked_extensions: Set[str] = field(default_factory=set)
    max_file_size_mb: int = 10
    max_execution_time: int = 30
    max_memory_mb: int = 256
    can_execute_code: bool = False
    can_modify_files: bool = True
    can_delete_files: bool = False
    session_timeout_minutes: int = 60
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class SecurityAuditEntry:
    """Enhanced audit log entry with detailed security information."""
    timestamp: datetime
    event_id: str
    event_type: SecurityEvent
    user_id: Optional[str]
    resource_path: str
    operation: str
    success: bool
    error_message: Optional[str] = None
    resource_hash: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    execution_time_ms: Optional[int] = None
    memory_used_mb: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    """Comprehensive resource limits for code execution."""
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 256  # MB
    max_cpu_percent: int = 50  # CPU usage percentage
    max_file_size_mb: int = 10  # MB per file
    max_total_file_size_mb: int = 100  # Total file size limit
    max_open_files: int = 50
    max_processes: int = 3
    max_network_connections: int = 0  # Disable network by default
    max_disk_write_mb: int = 50  # Disk write limit
    allowed_syscalls: Set[str] = field(default_factory=set)
    blocked_syscalls: Set[str] = field(default_factory=lambda: {
        'execve', 'fork', 'clone', 'socket', 'connect', 'bind', 'listen'
    })
    allowed_modules: List[str] = field(default_factory=lambda: [
        'math', 'json', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'string', 're', 'unicodedata'
    ])
    blocked_modules: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'telnetlib', 'ssl', 'hashlib',
        'hmac', 'secrets', 'ctypes', 'multiprocessing', 'threading'
    ])


class PermissionManager:
    """Manages user permissions and access controls."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.users: Dict[str, UserProfile] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.permission_cache: Dict[str, Dict[str, bool]] = {}
        self._load_user_config()
    
    def _load_user_config(self):
        """Load user configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for user_data in config.get('users', []):
                        user = UserProfile(
                            user_id=user_data['user_id'],
                            username=user_data['username'],
                            permission_level=PermissionLevel(user_data['permission_level']),
                            allowed_directories=set(user_data.get('allowed_directories', [])),
                            blocked_extensions=set(user_data.get('blocked_extensions', [])),
                            max_file_size_mb=user_data.get('max_file_size_mb', 10),
                            max_execution_time=user_data.get('max_execution_time', 30),
                            max_memory_mb=user_data.get('max_memory_mb', 256),
                            can_execute_code=user_data.get('can_execute_code', False),
                            can_modify_files=user_data.get('can_modify_files', True),
                            can_delete_files=user_data.get('can_delete_files', False),
                            session_timeout_minutes=user_data.get('session_timeout_minutes', 60),
                            is_active=user_data.get('is_active', True)
                        )
                        self.users[user.user_id] = user
            except Exception as e:
                logger.error(f"Failed to load user config: {e}")
    
    def save_user_config(self):
        """Save user configuration to file."""
        if not self.config_path:
            return
        
        try:
            config = {
                'users': [
                    {
                        'user_id': user.user_id,
                        'username': user.username,
                        'permission_level': user.permission_level.value,
                        'allowed_directories': list(user.allowed_directories),
                        'blocked_extensions': list(user.blocked_extensions),
                        'max_file_size_mb': user.max_file_size_mb,
                        'max_execution_time': user.max_execution_time,
                        'max_memory_mb': user.max_memory_mb,
                        'can_execute_code': user.can_execute_code,
                        'can_modify_files': user.can_modify_files,
                        'can_delete_files': user.can_delete_files,
                        'session_timeout_minutes': user.session_timeout_minutes,
                        'is_active': user.is_active
                    }
                    for user in self.users.values()
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save user config: {e}")
    
    def create_user(self, username: str, permission_level: PermissionLevel) -> UserProfile:
        """Create a new user profile."""
        user_id = str(uuid.uuid4())
        user = UserProfile(
            user_id=user_id,
            username=username,
            permission_level=permission_level
        )
        
        # Set default permissions based on level
        if permission_level == PermissionLevel.ADMIN:
            user.can_execute_code = True
            user.can_modify_files = True
            user.can_delete_files = True
            user.max_file_size_mb = 100
            user.max_execution_time = 300
            user.max_memory_mb = 1024
        elif permission_level == PermissionLevel.USER:
            user.can_execute_code = True
            user.can_modify_files = True
            user.can_delete_files = False
            user.max_file_size_mb = 50
            user.max_execution_time = 60
            user.max_memory_mb = 512
        elif permission_level == PermissionLevel.GUEST:
            user.can_execute_code = False
            user.can_modify_files = False
            user.can_delete_files = False
            user.max_file_size_mb = 10
            user.max_execution_time = 30
            user.max_memory_mb = 256
        else:  # RESTRICTED
            user.can_execute_code = False
            user.can_modify_files = False
            user.can_delete_files = False
            user.max_file_size_mb = 1
            user.max_execution_time = 10
            user.max_memory_mb = 128
        
        self.users[user_id] = user
        self.save_user_config()
        return user
    
    def authenticate_user(self, user_id: str, session_id: Optional[str] = None) -> Optional[UserProfile]:
        """Authenticate user and create session."""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None
        
        # Create or update session
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': None,  # Would be set by calling code
            'user_agent': None   # Would be set by calling code
        }
        
        user.last_login = datetime.now()
        return user
    
    def check_permission(self, user_id: str, operation: str, resource_path: str) -> bool:
        """Check if user has permission for operation on resource."""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        # Check session validity
        if not self._is_session_valid(user_id):
            return False
        
        # Cache key for performance
        cache_key = f"{user_id}:{operation}:{resource_path}"
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]
        
        # Check operation permissions
        result = self._check_operation_permission(user, operation, resource_path)
        
        # Cache result
        self.permission_cache[cache_key] = result
        return result
    
    def _check_operation_permission(self, user: UserProfile, operation: str, resource_path: str) -> bool:
        """Check specific operation permission."""
        # Check operation-specific permissions first
        if operation in ['write', 'create', 'modify'] and not user.can_modify_files:
            return False
        
        if operation == 'delete' and not user.can_delete_files:
            return False
        
        if operation == 'execute' and not user.can_execute_code:
            return False
        
        # For code execution, skip path and extension checks
        if operation == 'execute' and resource_path == 'code_execution':
            return True
        
        # Check directory access for file operations
        if not self._is_path_allowed(user, resource_path):
            return False
        
        # Check file extension
        if resource_path:
            ext = Path(resource_path).suffix.lower()
            if ext in user.blocked_extensions:
                return False
        
        return True
    
    def _is_path_allowed(self, user: UserProfile, path: str) -> bool:
        """Check if path is in user's allowed directories."""
        if not user.allowed_directories:
            return True  # No restrictions
        
        abs_path = os.path.abspath(path)
        for allowed_dir in user.allowed_directories:
            allowed_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_abs):
                return True
        
        return False
    
    def _is_session_valid(self, user_id: str) -> bool:
        """Check if user session is still valid."""
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session['user_id'] == user_id:
                user = self.users[user_id]
                timeout = timedelta(minutes=user.session_timeout_minutes)
                if datetime.now() - session['last_activity'] < timeout:
                    session['last_activity'] = datetime.now()
                    return True
                else:
                    # Mark session for expiration
                    expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        return False


class EnhancedSecurityAuditor:
    """Enhanced security auditor with comprehensive logging and monitoring."""
    
    def __init__(self, audit_log_path: str, max_log_size_mb: int = 100):
        self.audit_log_path = audit_log_path
        self.max_log_size_mb = max_log_size_mb
        self.audit_entries: List[SecurityAuditEntry] = []
        self.security_metrics: Dict[str, Any] = {
            'total_events': 0,
            'security_violations': 0,
            'failed_authentications': 0,
            'resource_limit_violations': 0,
            'suspicious_activities': []
        }
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure audit log directory exists."""
        log_dir = os.path.dirname(self.audit_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log_security_event(self, event_type: SecurityEvent, user_id: Optional[str],
                          resource_path: str, operation: str, success: bool,
                          **kwargs) -> str:
        """Log a security event with detailed information."""
        event_id = str(uuid.uuid4())
        
        entry = SecurityAuditEntry(
            timestamp=datetime.now(),
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            resource_path=resource_path,
            operation=operation,
            success=success,
            error_message=kwargs.get('error_message'),
            resource_hash=kwargs.get('resource_hash'),
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            session_id=kwargs.get('session_id'),
            execution_time_ms=kwargs.get('execution_time_ms'),
            memory_used_mb=kwargs.get('memory_used_mb'),
            additional_data=kwargs.get('additional_data', {})
        )
        
        self.audit_entries.append(entry)
        self._write_audit_entry(entry)
        self._update_security_metrics(entry)
        self._check_suspicious_activity(entry)
        
        return event_id
    
    def _write_audit_entry(self, entry: SecurityAuditEntry):
        """Write audit entry to log file."""
        try:
            # Check log file size and rotate if necessary
            if os.path.exists(self.audit_log_path):
                size_mb = os.path.getsize(self.audit_log_path) / (1024 * 1024)
                if size_mb > self.max_log_size_mb:
                    self._rotate_log_file()
            
            log_data = {
                'timestamp': entry.timestamp.isoformat(),
                'event_id': entry.event_id,
                'event_type': entry.event_type.value,
                'user_id': entry.user_id,
                'resource_path': entry.resource_path,
                'operation': entry.operation,
                'success': entry.success,
                'error_message': entry.error_message,
                'resource_hash': entry.resource_hash,
                'ip_address': entry.ip_address,
                'user_agent': entry.user_agent,
                'session_id': entry.session_id,
                'execution_time_ms': entry.execution_time_ms,
                'memory_used_mb': entry.memory_used_mb,
                'additional_data': entry.additional_data
            }
            
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _rotate_log_file(self):
        """Rotate audit log file when it gets too large."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_path = f"{self.audit_log_path}.{timestamp}"
            shutil.move(self.audit_log_path, rotated_path)
            logger.info(f"Rotated audit log to {rotated_path}")
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")
    
    def _update_security_metrics(self, entry: SecurityAuditEntry):
        """Update security metrics based on audit entry."""
        self.security_metrics['total_events'] += 1
        
        if entry.event_type == SecurityEvent.SECURITY_VIOLATION:
            self.security_metrics['security_violations'] += 1
        
        if entry.event_type == SecurityEvent.PERMISSION_DENIED:
            self.security_metrics['failed_authentications'] += 1
        
        if entry.event_type == SecurityEvent.RESOURCE_LIMIT_EXCEEDED:
            self.security_metrics['resource_limit_violations'] += 1
    
    def _check_suspicious_activity(self, entry: SecurityAuditEntry):
        """Check for suspicious activity patterns."""
        # Check for repeated failures from same user
        if not entry.success and entry.user_id:
            recent_failures = [
                e for e in self.audit_entries[-50:]  # Check last 50 events
                if e.user_id == entry.user_id and not e.success
                and (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if len(recent_failures) > 5:
                self.security_metrics['suspicious_activities'].append({
                    'type': 'repeated_failures',
                    'user_id': entry.user_id,
                    'count': len(recent_failures),
                    'timestamp': datetime.now().isoformat()
                })
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_entries = [e for e in self.audit_entries if e.timestamp > cutoff_time]
        
        report = {
            'time_period_hours': hours,
            'total_events': len(recent_entries),
            'successful_operations': len([e for e in recent_entries if e.success]),
            'failed_operations': len([e for e in recent_entries if not e.success]),
            'security_violations': len([e for e in recent_entries if e.event_type == SecurityEvent.SECURITY_VIOLATION]),
            'permission_denials': len([e for e in recent_entries if e.event_type == SecurityEvent.PERMISSION_DENIED]),
            'code_executions': len([e for e in recent_entries if e.event_type == SecurityEvent.CODE_EXECUTION]),
            'file_operations': len([e for e in recent_entries if e.event_type in [SecurityEvent.FILE_ACCESS, SecurityEvent.FILE_WRITE, SecurityEvent.FILE_DELETE]]),
            'unique_users': len(set(e.user_id for e in recent_entries if e.user_id)),
            'top_operations': self._get_top_operations(recent_entries),
            'suspicious_activities': self.security_metrics['suspicious_activities'][-10:]  # Last 10 suspicious activities
        }
        
        return report
    
    def _get_top_operations(self, entries: List[SecurityAuditEntry]) -> List[Dict[str, Any]]:
        """Get top operations by frequency."""
        operation_counts = {}
        for entry in entries:
            key = f"{entry.operation}:{entry.event_type.value}"
            operation_counts[key] = operation_counts.get(key, 0) + 1
        
        return [
            {'operation': op, 'count': count}
            for op, count in sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]


class ComprehensiveSecuritySandbox:
    """Comprehensive security sandbox combining all security features."""
    
    def __init__(self, workspace_path: str, config_path: Optional[str] = None):
        self.workspace_path = workspace_path
        self.permission_manager = PermissionManager(config_path)
        self.auditor = EnhancedSecurityAuditor(
            os.path.join(workspace_path, "security_audit.log")
        )
        self.resource_monitors: Dict[str, Any] = {}
        self._setup_default_users()
    
    def _setup_default_users(self):
        """Setup default users if none exist."""
        if not self.permission_manager.users:
            # Create default admin user
            admin = self.permission_manager.create_user("admin", PermissionLevel.ADMIN)
            admin.allowed_directories.add(self.workspace_path)
            
            # Create default user
            user = self.permission_manager.create_user("user", PermissionLevel.USER)
            user.allowed_directories.add(os.path.join(self.workspace_path, "user_workspace"))
            
            self.permission_manager.save_user_config()
    
    def authenticate_and_authorize(self, user_id: str, operation: str, 
                                 resource_path: str, **kwargs) -> bool:
        """Authenticate user and authorize operation."""
        # Authenticate user
        user = self.permission_manager.authenticate_user(user_id)
        if not user:
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_DENIED,
                user_id,
                resource_path,
                operation,
                False,
                error_message="Authentication failed",
                **kwargs
            )
            return False
        
        # Check permissions
        if not self.permission_manager.check_permission(user_id, operation, resource_path):
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_DENIED,
                user_id,
                resource_path,
                operation,
                False,
                error_message="Permission denied",
                **kwargs
            )
            return False
        
        return True
    
    @contextmanager
    def secure_file_operation(self, user_id: str, operation: str, file_path: str, **kwargs):
        """Context manager for secure file operations."""
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            # Check authorization
            if not self.authenticate_and_authorize(user_id, operation, file_path, **kwargs):
                error_message = f"Access denied for {operation} on {file_path}"
                self.auditor.log_security_event(
                    SecurityEvent.PERMISSION_DENIED,
                    user_id,
                    file_path,
                    operation,
                    False,
                    error_message=error_message,
                    **kwargs
                )
                raise PermissionError(error_message)
            
            # Yield control to caller
            yield
            success = True
            
        except Exception as e:
            if not error_message:
                error_message = str(e)
            raise
        
        finally:
            # Log the operation only if not already logged as permission denied
            if success or "Access denied" not in str(error_message):
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                event_type = SecurityEvent.FILE_ACCESS
                if operation in ['write', 'create', 'modify']:
                    event_type = SecurityEvent.FILE_WRITE
                elif operation == 'delete':
                    event_type = SecurityEvent.FILE_DELETE
                
                self.auditor.log_security_event(
                    event_type,
                    user_id,
                    file_path,
                    operation,
                    success,
                    error_message=error_message,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    **kwargs
                )
    
    @contextmanager
    def secure_code_execution(self, user_id: str, code: str, **kwargs):
        """Context manager for secure code execution."""
        start_time = time.time()
        success = False
        error_message = None
        memory_used_mb = 0
        
        try:
            # Check authorization
            if not self.authenticate_and_authorize(user_id, "execute", "code_execution", **kwargs):
                error_message = "Code execution not permitted"
                self.auditor.log_security_event(
                    SecurityEvent.PERMISSION_DENIED,
                    user_id,
                    "code_execution",
                    "execute",
                    False,
                    error_message=error_message,
                    **kwargs
                )
                raise PermissionError(error_message)
            
            # Get user limits
            user = self.permission_manager.users[user_id]
            limits = ResourceLimits(
                max_execution_time=user.max_execution_time,
                max_memory_mb=user.max_memory_mb
            )
            
            # Create execution environment
            from .safe_execution import SafeExecutionEnvironment, ExecutionContext
            executor = SafeExecutionEnvironment(limits)
            
            context = ExecutionContext(code=code, **kwargs)
            result = executor.execute_code(context)
            
            success = result.success
            memory_used_mb = result.memory_used_mb
            
            if not success:
                error_message = result.error
            
            yield result
            
        except Exception as e:
            if not error_message:
                error_message = str(e)
            raise
        
        finally:
            # Log the execution
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            event_type = SecurityEvent.CODE_EXECUTION
            if not success and "not permitted" in str(error_message):
                event_type = SecurityEvent.PERMISSION_DENIED
            
            self.auditor.log_security_event(
                event_type,
                user_id,
                "code_execution",
                "execute",
                success,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                additional_data={'code_hash': hashlib.sha256(code.encode()).hexdigest()},
                **kwargs
            )
    
    def get_user_permissions(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user permissions and status."""
        user = self.permission_manager.users.get(user_id)
        if not user:
            return None
        
        return {
            'user_id': user.user_id,
            'username': user.username,
            'permission_level': user.permission_level.value,
            'allowed_directories': list(user.allowed_directories),
            'blocked_extensions': list(user.blocked_extensions),
            'can_execute_code': user.can_execute_code,
            'can_modify_files': user.can_modify_files,
            'can_delete_files': user.can_delete_files,
            'max_file_size_mb': user.max_file_size_mb,
            'max_execution_time': user.max_execution_time,
            'max_memory_mb': user.max_memory_mb,
            'is_active': user.is_active,
            'last_login': user.last_login.isoformat() if user.last_login else None
        }
    
    def update_user_permissions(self, admin_user_id: str, target_user_id: str, 
                               updates: Dict[str, Any]) -> bool:
        """Update user permissions (admin only)."""
        # Check if admin user has permission
        admin_user = self.permission_manager.users.get(admin_user_id)
        if not admin_user or admin_user.permission_level != PermissionLevel.ADMIN:
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_DENIED,
                admin_user_id,
                f"user:{target_user_id}",
                "update_permissions",
                False,
                error_message="Admin permission required"
            )
            return False
        
        # Update target user
        target_user = self.permission_manager.users.get(target_user_id)
        if not target_user:
            return False
        
        try:
            # Apply updates
            for key, value in updates.items():
                if hasattr(target_user, key):
                    if key == 'permission_level':
                        value = PermissionLevel(value)
                    elif key in ['allowed_directories', 'blocked_extensions']:
                        value = set(value) if isinstance(value, list) else value
                    setattr(target_user, key, value)
            
            self.permission_manager.save_user_config()
            
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_CHANGE,
                admin_user_id,
                f"user:{target_user_id}",
                "update_permissions",
                True,
                additional_data=updates
            )
            
            return True
            
        except Exception as e:
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_CHANGE,
                admin_user_id,
                f"user:{target_user_id}",
                "update_permissions",
                False,
                error_message=str(e)
            )
            return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'total_users': len(self.permission_manager.users),
            'active_users': len([u for u in self.permission_manager.users.values() if u.is_active]),
            'active_sessions': len(self.permission_manager.active_sessions),
            'security_report': self.auditor.get_security_report(),
            'workspace_path': self.workspace_path,
            'audit_log_path': self.auditor.audit_log_path
        }


def create_security_sandbox(workspace_path: str, config_path: Optional[str] = None) -> ComprehensiveSecuritySandbox:
    """Create and configure comprehensive security sandbox."""
    return ComprehensiveSecuritySandbox(workspace_path, config_path)


if __name__ == "__main__":
    # Example usage
    sandbox = create_security_sandbox(".")
    
    # Create a test user
    user = sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
    user.allowed_directories.add(".")
    user.can_execute_code = True
    
    # Test secure file operation
    try:
        with sandbox.secure_file_operation(user.user_id, "read", "test.txt"):
            print("File operation authorized")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    
    # Test secure code execution
    try:
        with sandbox.secure_code_execution(user.user_id, "print('Hello, World!')") as result:
            print(f"Code execution result: {result.output}")
    except PermissionError as e:
        print(f"Code execution denied: {e}")
    
    # Get security status
    status = sandbox.get_security_status()
    print(f"Security status: {json.dumps(status, indent=2, default=str)}")