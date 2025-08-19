"""
Security Integration Module for Sanskrit Rewrite Engine

This module provides a unified security interface that integrates all security
components: sandbox, configuration, auditing, and safe execution.
Implements requirements 13.2, 13.3, 13.4 for atomic operations,
structured trace data, and detailed error information.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

from .security_sandbox import (
    ComprehensiveSecuritySandbox,
    PermissionLevel,
    SecurityEvent,
    create_security_sandbox
)
from .security_config import (
    SecurityConfigurationManager,
    SecurityPolicy,
    ThreatLevel,
    create_default_security_config
)
from .safe_execution import (
    SafeExecutionEnvironment,
    CodeExecutionManager,
    ExecutionLimits,
    ExecutionContext,
    ExecutionResult,
    create_safe_execution_manager
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityOperationResult:
    """Result of a security operation with detailed information."""
    success: bool
    operation: str
    resource_path: str
    user_id: Optional[str]
    execution_time_ms: int
    error_message: Optional[str] = None
    warning_messages: List[str] = None
    audit_event_id: Optional[str] = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []
        if self.additional_data is None:
            self.additional_data = {}


class IntegratedSecurityManager:
    """Integrated security manager that coordinates all security components."""
    
    def __init__(self, workspace_path: str, config_path: Optional[str] = None):
        self.workspace_path = workspace_path
        self.config_path = config_path
        
        # Initialize security components
        self.config_manager = self._initialize_config_manager()
        self.sandbox = self._initialize_sandbox()
        self.execution_manager = self._initialize_execution_manager()
        
        # Security state
        self.is_emergency_lockdown = False
        self.security_alerts: List[Dict[str, Any]] = []
        
        logger.info("Integrated security manager initialized") 
   
    def _initialize_config_manager(self) -> SecurityConfigurationManager:
        """Initialize security configuration manager."""
        if self.config_path:
            return SecurityConfigurationManager(self.config_path)
        else:
            return create_default_security_config(self.workspace_path)
    
    def _initialize_sandbox(self) -> ComprehensiveSecuritySandbox:
        """Initialize security sandbox."""
        user_config_path = os.path.join(self.workspace_path, ".kiro", "users.json")
        return ComprehensiveSecuritySandbox(self.workspace_path, user_config_path)
    
    def _initialize_execution_manager(self) -> CodeExecutionManager:
        """Initialize code execution manager."""
        config = self.config_manager.config
        limits = ExecutionLimits(
            max_execution_time=config.code_execution.max_execution_time,
            max_memory_mb=config.code_execution.max_memory_mb,
            max_cpu_percent=config.code_execution.max_cpu_percent,
            allowed_modules=config.code_execution.allowed_imports,
            blocked_modules=config.code_execution.blocked_imports
        )
        return create_safe_execution_manager(self.workspace_path, limits)
    
    def authenticate_user(self, username: str, create_if_not_exists: bool = False) -> Optional[str]:
        """Authenticate user and return user ID."""
        # Find user by username
        for user_id, user in self.sandbox.permission_manager.users.items():
            if user.username == username:
                authenticated_user = self.sandbox.permission_manager.authenticate_user(user_id)
                if authenticated_user:
                    return user_id
                break
        
        # Create user if requested and not found
        if create_if_not_exists:
            user = self.sandbox.permission_manager.create_user(username, PermissionLevel.USER)
            user.allowed_directories.add(self.workspace_path)
            return user.user_id
        
        return None
    
    @contextmanager
    def secure_operation(self, user_id: str, operation: str, resource_path: str, **kwargs):
        """Context manager for secure operations with comprehensive error handling."""
        start_time = datetime.now()
        result = SecurityOperationResult(
            success=False,
            operation=operation,
            resource_path=resource_path,
            user_id=user_id,
            execution_time_ms=0
        )
        
        try:
            # Check emergency lockdown
            if self.is_emergency_lockdown:
                raise PermissionError("System is in emergency lockdown mode")
            
            # Pre-operation security checks
            self._perform_pre_operation_checks(user_id, operation, resource_path, **kwargs)
            
            # Yield control to caller
            yield result
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            self._handle_security_exception(e, user_id, operation, resource_path)
            raise
        
        finally:
            # Calculate execution time
            execution_time = datetime.now() - start_time
            result.execution_time_ms = int(execution_time.total_seconds() * 1000)
            
            # Log the operation
            result.audit_event_id = self._log_security_operation(result, **kwargs)
            
            # Post-operation analysis
            self._perform_post_operation_analysis(result)
    
    def _perform_pre_operation_checks(self, user_id: str, operation: str, 
                                    resource_path: str, **kwargs):
        """Perform pre-operation security checks."""
        # Check user authentication and authorization
        if not self.sandbox.authenticate_and_authorize(user_id, operation, resource_path, **kwargs):
            raise PermissionError(f"Access denied for {operation} on {resource_path}")
        
        # Check rate limiting
        if self.config_manager.config.enable_rate_limiting:
            if not self._check_rate_limit(user_id):
                raise PermissionError("Rate limit exceeded")
        
        # Check IP whitelisting if enabled
        if self.config_manager.config.enable_ip_whitelisting:
            client_ip = kwargs.get('client_ip')
            if client_ip and client_ip not in self.config_manager.config.whitelisted_ips:
                raise PermissionError(f"IP address {client_ip} not whitelisted")
        
        # Additional security checks based on operation type
        if operation == 'execute':
            self._check_code_execution_security(user_id, resource_path, **kwargs)
        elif operation in ['write', 'create', 'modify']:
            self._check_file_modification_security(user_id, resource_path, **kwargs)
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        # Simplified rate limiting - in production, use Redis or similar
        # For now, just return True
        return True
    
    def _check_code_execution_security(self, user_id: str, code: str, **kwargs):
        """Perform additional security checks for code execution."""
        config = self.config_manager.config.code_execution
        
        if not config.enabled:
            raise PermissionError("Code execution is disabled")
        
        # Check code for dangerous patterns
        from .safe_execution import SecurityValidator
        validator = SecurityValidator(ExecutionLimits(
            allowed_modules=config.allowed_imports,
            blocked_modules=config.blocked_imports
        ))
        
        validation_result = validator.validate_code(code)
        if not validation_result['is_safe']:
            raise PermissionError(f"Code validation failed: {validation_result['issues']}")
    
    def _check_file_modification_security(self, user_id: str, file_path: str, **kwargs):
        """Perform additional security checks for file modifications."""
        config = self.config_manager.config.file_access
        
        # Check file size if content is provided
        content = kwargs.get('content', '')
        if content and len(content.encode('utf-8')) > config.max_file_size_mb * 1024 * 1024:
            raise PermissionError(f"File content exceeds size limit of {config.max_file_size_mb}MB")
        
        # Check sensitive paths
        if config.require_approval_for_sensitive_paths:
            abs_path = os.path.abspath(file_path)
            for sensitive_path in config.sensitive_paths:
                if abs_path.startswith(os.path.abspath(sensitive_path)):
                    # In a real implementation, this would trigger an approval workflow
                    logger.warning(f"Sensitive path access: {file_path}")
    
    def _handle_security_exception(self, exception: Exception, user_id: str, 
                                 operation: str, resource_path: str):
        """Handle security exceptions and generate alerts if necessary."""
        # Check if this is a security violation
        if isinstance(exception, PermissionError):
            # Generate security alert
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'security_violation',
                'user_id': user_id,
                'operation': operation,
                'resource_path': resource_path,
                'error': str(exception),
                'severity': 'high'
            }
            self.security_alerts.append(alert)
            
            # Check if we should trigger emergency lockdown
            recent_violations = [
                a for a in self.security_alerts[-10:]  # Last 10 alerts
                if a['type'] == 'security_violation' and a['user_id'] == user_id
            ]
            
            if len(recent_violations) >= 5:  # 5 violations from same user
                self._trigger_emergency_lockdown(f"Multiple security violations from user {user_id}")
    
    def _trigger_emergency_lockdown(self, reason: str):
        """Trigger emergency lockdown mode."""
        self.is_emergency_lockdown = True
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': 'emergency_lockdown',
            'reason': reason,
            'severity': 'critical'
        }
        self.security_alerts.append(alert)
        
        logger.critical(f"EMERGENCY LOCKDOWN TRIGGERED: {reason}")
        
        # In a real implementation, this would:
        # - Notify administrators
        # - Disable all non-essential operations
        # - Create incident report
        # - Potentially shut down the system
    
    def _log_security_operation(self, result: SecurityOperationResult, **kwargs) -> str:
        """Log security operation to audit system."""
        event_type = SecurityEvent.FILE_ACCESS
        
        if result.operation == 'execute':
            event_type = SecurityEvent.CODE_EXECUTION
        elif result.operation in ['write', 'create', 'modify']:
            event_type = SecurityEvent.FILE_WRITE
        elif result.operation == 'delete':
            event_type = SecurityEvent.FILE_DELETE
        elif not result.success:
            event_type = SecurityEvent.SECURITY_VIOLATION
        
        return self.sandbox.auditor.log_security_event(
            event_type,
            result.user_id,
            result.resource_path,
            result.operation,
            result.success,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            additional_data=result.additional_data,
            **kwargs
        )
    
    def _perform_post_operation_analysis(self, result: SecurityOperationResult):
        """Perform post-operation security analysis."""
        # Anomaly detection
        if self.config_manager.config.enable_anomaly_detection:
            self._detect_anomalies(result)
        
        # Update security metrics
        self._update_security_metrics(result)
    
    def _detect_anomalies(self, result: SecurityOperationResult):
        """Detect anomalous behavior patterns."""
        # Simple anomaly detection - in production, use ML models
        
        # Check for unusual execution times
        if result.execution_time_ms > 10000:  # 10 seconds
            result.warning_messages.append("Unusually long execution time detected")
        
        # Check for unusual file access patterns
        if result.operation == 'read' and 'system' in result.resource_path.lower():
            result.warning_messages.append("System file access detected")
    
    def _update_security_metrics(self, result: SecurityOperationResult):
        """Update security metrics based on operation result."""
        # This would update various security metrics
        # For now, just log the operation
        logger.debug(f"Security operation completed: {result.operation} - {result.success}")
    
    # High-level security operations
    
    def secure_file_read(self, user_id: str, file_path: str, **kwargs) -> str:
        """Securely read file content."""
        with self.secure_operation(user_id, "read", file_path, **kwargs) as result:
            with self.sandbox.secure_file_operation(user_id, "read", file_path, **kwargs):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                result.additional_data['content_length'] = len(content)
                return content
    
    def secure_file_write(self, user_id: str, file_path: str, content: str, **kwargs) -> bool:
        """Securely write file content."""
        with self.secure_operation(user_id, "write", file_path, content=content, **kwargs) as result:
            with self.sandbox.secure_file_operation(user_id, "write", file_path, **kwargs):
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                result.additional_data['content_length'] = len(content)
                return True
    
    def secure_file_delete(self, user_id: str, file_path: str, **kwargs) -> bool:
        """Securely delete file."""
        with self.secure_operation(user_id, "delete", file_path, **kwargs) as result:
            with self.sandbox.secure_file_operation(user_id, "delete", file_path, **kwargs):
                if os.path.exists(file_path):
                    os.remove(file_path)
                    result.additional_data['file_deleted'] = True
                else:
                    result.additional_data['file_deleted'] = False
                return True
    
    def secure_code_execution(self, user_id: str, code: str, language: str = "python", **kwargs) -> ExecutionResult:
        """Securely execute code."""
        with self.secure_operation(user_id, "execute", "code_execution", code=code, **kwargs) as result:
            with self.sandbox.secure_code_execution(user_id, code, **kwargs) as exec_result:
                result.additional_data.update({
                    'language': language,
                    'code_length': len(code),
                    'execution_success': exec_result.success,
                    'memory_used_mb': exec_result.memory_used_mb,
                    'output_length': len(exec_result.output)
                })
                return exec_result
    
    def list_directory_secure(self, user_id: str, directory_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Securely list directory contents."""
        with self.secure_operation(user_id, "list", directory_path, **kwargs) as result:
            if not self.sandbox.permission_manager.check_permission(user_id, "read", directory_path):
                raise PermissionError(f"Access denied to directory: {directory_path}")
            
            files = []
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path):
                        stat = os.stat(item_path)
                        files.append({
                            'name': item,
                            'path': item_path,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'type': 'file'
                        })
                    elif os.path.isdir(item_path):
                        files.append({
                            'name': item,
                            'path': item_path,
                            'type': 'directory'
                        })
            
            result.additional_data['files_count'] = len(files)
            return files
    
    # Security management operations
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        config_report = self.config_manager.generate_security_report()
        sandbox_status = self.sandbox.get_security_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'emergency_lockdown': self.is_emergency_lockdown,
            'recent_alerts': self.security_alerts[-10:],  # Last 10 alerts
            'configuration': config_report,
            'sandbox': sandbox_status,
            'workspace_path': self.workspace_path
        }
    
    def update_security_policy(self, admin_user_id: str, policy: SecurityPolicy) -> bool:
        """Update security policy (admin only)."""
        # Check admin permissions
        admin_user = self.sandbox.permission_manager.users.get(admin_user_id)
        if not admin_user or admin_user.permission_level != PermissionLevel.ADMIN:
            return False
        
        # Apply new policy
        self.config_manager.apply_policy_template(policy)
        self.config_manager.save_configuration()
        
        # Reinitialize execution manager with new limits
        self.execution_manager = self._initialize_execution_manager()
        
        # Log policy change
        self.sandbox.auditor.log_security_event(
            SecurityEvent.PERMISSION_CHANGE,
            admin_user_id,
            "security_policy",
            "update_policy",
            True,
            additional_data={'new_policy': policy.value}
        )
        
        return True
    
    def disable_emergency_lockdown(self, admin_user_id: str, reason: str) -> bool:
        """Disable emergency lockdown (admin only)."""
        # Check admin permissions
        admin_user = self.sandbox.permission_manager.users.get(admin_user_id)
        if not admin_user or admin_user.permission_level != PermissionLevel.ADMIN:
            return False
        
        self.is_emergency_lockdown = False
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': 'lockdown_disabled',
            'admin_user_id': admin_user_id,
            'reason': reason,
            'severity': 'info'
        }
        self.security_alerts.append(alert)
        
        logger.info(f"Emergency lockdown disabled by {admin_user_id}: {reason}")
        return True
    
    def export_security_logs(self, admin_user_id: str, format: str = 'json') -> str:
        """Export security logs (admin only)."""
        # Check admin permissions
        admin_user = self.sandbox.permission_manager.users.get(admin_user_id)
        if not admin_user or admin_user.permission_level != PermissionLevel.ADMIN:
            raise PermissionError("Admin permission required")
        
        # Get audit entries
        audit_entries = self.sandbox.auditor.audit_entries
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'exported_by': admin_user_id,
            'total_entries': len(audit_entries),
            'entries': [
                {
                    'timestamp': entry.timestamp.isoformat(),
                    'event_id': entry.event_id,
                    'event_type': entry.event_type.value,
                    'user_id': entry.user_id,
                    'resource_path': entry.resource_path,
                    'operation': entry.operation,
                    'success': entry.success,
                    'error_message': entry.error_message,
                    'execution_time_ms': entry.execution_time_ms,
                    'additional_data': entry.additional_data
                }
                for entry in audit_entries
            ]
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


def create_integrated_security_manager(workspace_path: str, 
                                     config_path: Optional[str] = None) -> IntegratedSecurityManager:
    """Create integrated security manager for workspace."""
    return IntegratedSecurityManager(workspace_path, config_path)


if __name__ == "__main__":
    # Example usage
    security_manager = create_integrated_security_manager(".")
    
    # Authenticate user
    user_id = security_manager.authenticate_user("test_user", create_if_not_exists=True)
    
    if user_id:
        try:
            # Secure file operations
            content = security_manager.secure_file_read(user_id, "README.md")
            print(f"File content length: {len(content)}")
            
            # Secure code execution
            result = security_manager.secure_code_execution(
                user_id, 
                "print('Hello from secure execution!')"
            )
            print(f"Code execution result: {result.output}")
            
            # Get security status
            status = security_manager.get_security_status()
            print(f"Security status: {json.dumps(status, indent=2, default=str)}")
            
        except Exception as e:
            print(f"Security operation failed: {e}")
    else:
        print("Failed to authenticate user")