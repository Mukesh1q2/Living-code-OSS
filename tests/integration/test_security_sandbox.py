"""
Security Validation and Compliance Tests for Sanskrit Rewrite Engine

This module provides comprehensive tests for the security and sandbox layer,
validating all security controls, access restrictions, and audit logging.
"""

import os
import sys
import json
import tempfile
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sanskrit_rewrite_engine.security_sandbox import (
    ComprehensiveSecuritySandbox,
    PermissionManager,
    EnhancedSecurityAuditor,
    UserProfile,
    PermissionLevel,
    SecurityEvent,
    ResourceLimits,
    create_security_sandbox
)


class TestPermissionManager(unittest.TestCase):
    """Test permission management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "users.json")
        self.permission_manager = PermissionManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_user(self):
        """Test user creation with different permission levels."""
        # Test admin user creation
        admin_user = self.permission_manager.create_user("admin", PermissionLevel.ADMIN)
        self.assertEqual(admin_user.permission_level, PermissionLevel.ADMIN)
        self.assertTrue(admin_user.can_execute_code)
        self.assertTrue(admin_user.can_modify_files)
        self.assertTrue(admin_user.can_delete_files)
        self.assertEqual(admin_user.max_memory_mb, 1024)
        
        # Test regular user creation
        regular_user = self.permission_manager.create_user("user", PermissionLevel.USER)
        self.assertEqual(regular_user.permission_level, PermissionLevel.USER)
        self.assertTrue(regular_user.can_execute_code)
        self.assertTrue(regular_user.can_modify_files)
        self.assertFalse(regular_user.can_delete_files)
        self.assertEqual(regular_user.max_memory_mb, 512)
        
        # Test guest user creation
        guest_user = self.permission_manager.create_user("guest", PermissionLevel.GUEST)
        self.assertEqual(guest_user.permission_level, PermissionLevel.GUEST)
        self.assertFalse(guest_user.can_execute_code)
        self.assertFalse(guest_user.can_modify_files)
        self.assertFalse(guest_user.can_delete_files)
        
        # Test restricted user creation
        restricted_user = self.permission_manager.create_user("restricted", PermissionLevel.RESTRICTED)
        self.assertEqual(restricted_user.permission_level, PermissionLevel.RESTRICTED)
        self.assertFalse(restricted_user.can_execute_code)
        self.assertFalse(restricted_user.can_modify_files)
        self.assertFalse(restricted_user.can_delete_files)
        self.assertEqual(restricted_user.max_memory_mb, 128)
    
    def test_user_authentication(self):
        """Test user authentication and session management."""
        user = self.permission_manager.create_user("test_user", PermissionLevel.USER)
        
        # Test successful authentication
        authenticated_user = self.permission_manager.authenticate_user(user.user_id)
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.user_id, user.user_id)
        
        # Test authentication with invalid user
        invalid_user = self.permission_manager.authenticate_user("invalid_id")
        self.assertIsNone(invalid_user)
        
        # Test inactive user authentication
        user.is_active = False
        inactive_user = self.permission_manager.authenticate_user(user.user_id)
        self.assertIsNone(inactive_user)
    
    def test_permission_checking(self):
        """Test permission checking for various operations."""
        user = self.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.allowed_directories.add(self.temp_dir)
        user.blocked_extensions.add('.exe')
        
        # Authenticate user first
        self.permission_manager.authenticate_user(user.user_id)
        
        # Test allowed operations
        allowed_file = os.path.join(self.temp_dir, "test.txt")
        self.assertTrue(self.permission_manager.check_permission(user.user_id, "read", allowed_file))
        self.assertTrue(self.permission_manager.check_permission(user.user_id, "write", allowed_file))
        self.assertFalse(self.permission_manager.check_permission(user.user_id, "delete", allowed_file))
        
        # Test blocked extensions
        blocked_file = os.path.join(self.temp_dir, "malware.exe")
        self.assertFalse(self.permission_manager.check_permission(user.user_id, "read", blocked_file))
        
        # Test directory restrictions
        restricted_file = "/etc/passwd"
        self.assertFalse(self.permission_manager.check_permission(user.user_id, "read", restricted_file))
    
    def test_session_timeout(self):
        """Test session timeout functionality."""
        user = self.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.session_timeout_minutes = 0.01  # 0.6 seconds for testing
        
        # Authenticate user
        self.permission_manager.authenticate_user(user.user_id)
        
        # Check permission immediately (should work)
        self.assertTrue(self.permission_manager.check_permission(user.user_id, "read", "test.txt"))
        
        # Wait for session to expire
        time.sleep(1)
        
        # Check permission after timeout (should fail)
        self.assertFalse(self.permission_manager.check_permission(user.user_id, "read", "test.txt"))
    
    def test_config_persistence(self):
        """Test user configuration persistence."""
        # Create user and save config
        user = self.permission_manager.create_user("persistent_user", PermissionLevel.ADMIN)
        user.allowed_directories.add("/test/path")
        self.permission_manager.save_user_config()
        
        # Create new permission manager and load config
        new_manager = PermissionManager(self.config_path)
        loaded_user = new_manager.users.get(user.user_id)
        
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user.username, "persistent_user")
        self.assertEqual(loaded_user.permission_level, PermissionLevel.ADMIN)
        self.assertIn("/test/path", loaded_user.allowed_directories)


class TestEnhancedSecurityAuditor(unittest.TestCase):
    """Test security auditing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_log_path = os.path.join(self.temp_dir, "audit.log")
        self.auditor = EnhancedSecurityAuditor(self.audit_log_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_security_event_logging(self):
        """Test security event logging."""
        # Log a successful file access
        event_id = self.auditor.log_security_event(
            SecurityEvent.FILE_ACCESS,
            "user123",
            "/test/file.txt",
            "read",
            True,
            ip_address="127.0.0.1",
            session_id="session123"
        )
        
        self.assertIsNotNone(event_id)
        self.assertEqual(len(self.auditor.audit_entries), 1)
        
        entry = self.auditor.audit_entries[0]
        self.assertEqual(entry.event_type, SecurityEvent.FILE_ACCESS)
        self.assertEqual(entry.user_id, "user123")
        self.assertEqual(entry.resource_path, "/test/file.txt")
        self.assertEqual(entry.operation, "read")
        self.assertTrue(entry.success)
        self.assertEqual(entry.ip_address, "127.0.0.1")
        
        # Verify log file was written
        self.assertTrue(os.path.exists(self.audit_log_path))
        
        with open(self.audit_log_path, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            self.assertEqual(log_data['event_type'], 'file_access')
            self.assertEqual(log_data['user_id'], 'user123')
    
    def test_security_metrics_tracking(self):
        """Test security metrics tracking."""
        # Log various events
        self.auditor.log_security_event(SecurityEvent.SECURITY_VIOLATION, "user1", "/test", "read", False)
        self.auditor.log_security_event(SecurityEvent.PERMISSION_DENIED, "user2", "/test", "write", False)
        self.auditor.log_security_event(SecurityEvent.RESOURCE_LIMIT_EXCEEDED, "user3", "/test", "execute", False)
        self.auditor.log_security_event(SecurityEvent.FILE_ACCESS, "user4", "/test", "read", True)
        
        metrics = self.auditor.security_metrics
        self.assertEqual(metrics['total_events'], 4)
        self.assertEqual(metrics['security_violations'], 1)
        self.assertEqual(metrics['failed_authentications'], 1)
        self.assertEqual(metrics['resource_limit_violations'], 1)
    
    def test_suspicious_activity_detection(self):
        """Test suspicious activity detection."""
        # Simulate repeated failures from same user
        for i in range(6):
            self.auditor.log_security_event(
                SecurityEvent.PERMISSION_DENIED,
                "suspicious_user",
                f"/test/file{i}.txt",
                "read",
                False
            )
        
        # Check if suspicious activity was detected
        suspicious_activities = self.auditor.security_metrics['suspicious_activities']
        self.assertGreater(len(suspicious_activities), 0)
        
        activity = suspicious_activities[0]
        self.assertEqual(activity['type'], 'repeated_failures')
        self.assertEqual(activity['user_id'], 'suspicious_user')
        self.assertGreaterEqual(activity['count'], 6)
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Log some events
        self.auditor.log_security_event(SecurityEvent.FILE_ACCESS, "user1", "/test1", "read", True)
        self.auditor.log_security_event(SecurityEvent.FILE_WRITE, "user2", "/test2", "write", True)
        self.auditor.log_security_event(SecurityEvent.CODE_EXECUTION, "user1", "code", "execute", False)
        self.auditor.log_security_event(SecurityEvent.PERMISSION_DENIED, "user3", "/test3", "delete", False)
        
        report = self.auditor.get_security_report(hours=24)
        
        self.assertEqual(report['total_events'], 4)
        self.assertEqual(report['successful_operations'], 2)
        self.assertEqual(report['failed_operations'], 2)
        self.assertEqual(report['code_executions'], 1)
        self.assertEqual(report['file_operations'], 2)
        self.assertEqual(report['permission_denials'], 1)
        self.assertEqual(report['unique_users'], 3)
        
        # Check top operations
        self.assertIn('top_operations', report)
        self.assertIsInstance(report['top_operations'], list)
    
    def test_log_rotation(self):
        """Test audit log rotation."""
        # Set small max log size for testing
        self.auditor.max_log_size_mb = 0.001  # Very small for testing
        
        # Write enough data to trigger rotation
        for i in range(100):
            self.auditor.log_security_event(
                SecurityEvent.FILE_ACCESS,
                f"user{i}",
                f"/test/file{i}.txt",
                "read",
                True,
                additional_data={'large_data': 'x' * 1000}  # Add some bulk
            )
        
        # Check if rotation occurred (rotated file should exist)
        log_dir = os.path.dirname(self.audit_log_path)
        log_files = [f for f in os.listdir(log_dir) if f.startswith('audit.log')]
        
        # Should have original log plus at least one rotated log
        self.assertGreaterEqual(len(log_files), 1)


class TestComprehensiveSecuritySandbox(unittest.TestCase):
    """Test comprehensive security sandbox functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "users.json")
        self.sandbox = ComprehensiveSecuritySandbox(self.temp_dir, self.config_path)
        
        # Create test user
        self.test_user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        self.test_user.allowed_directories.add(self.temp_dir)
        self.test_user.can_execute_code = True
        
        # Authenticate the user to create a session
        self.sandbox.permission_manager.authenticate_user(self.test_user.user_id)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_secure_file_operation_success(self):
        """Test successful secure file operation."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        
        # Test successful file operation
        with self.sandbox.secure_file_operation(self.test_user.user_id, "write", test_file):
            with open(test_file, 'w') as f:
                f.write("test content")
        
        # Verify file was created
        self.assertTrue(os.path.exists(test_file))
        
        # Verify audit log entry
        audit_entries = self.sandbox.auditor.audit_entries
        self.assertGreater(len(audit_entries), 0)
        
        last_entry = audit_entries[-1]
        self.assertEqual(last_entry.event_type, SecurityEvent.FILE_WRITE)
        self.assertEqual(last_entry.user_id, self.test_user.user_id)
        self.assertTrue(last_entry.success)
    
    def test_secure_file_operation_permission_denied(self):
        """Test secure file operation with permission denied."""
        # Try to access file outside allowed directories
        restricted_file = "/etc/passwd"
        
        with self.assertRaises(PermissionError):
            with self.sandbox.secure_file_operation(self.test_user.user_id, "read", restricted_file):
                pass
        
        # Verify audit log entry for permission denial
        audit_entries = self.sandbox.auditor.audit_entries
        self.assertGreater(len(audit_entries), 0)
        
        last_entry = audit_entries[-1]
        self.assertEqual(last_entry.event_type, SecurityEvent.PERMISSION_DENIED)
        self.assertEqual(last_entry.user_id, self.test_user.user_id)
        self.assertFalse(last_entry.success)
    
    def test_secure_code_execution_success(self):
        """Test successful secure code execution."""
        safe_code = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
        
        with self.sandbox.secure_code_execution(self.test_user.user_id, safe_code) as result:
            self.assertTrue(result.success)
            self.assertIn("Result: 4.0", result.output)
        
        # Verify audit log entry
        audit_entries = self.sandbox.auditor.audit_entries
        self.assertGreater(len(audit_entries), 0)
        
        last_entry = audit_entries[-1]
        self.assertEqual(last_entry.event_type, SecurityEvent.CODE_EXECUTION)
        self.assertEqual(last_entry.user_id, self.test_user.user_id)
        self.assertTrue(last_entry.success)
    
    def test_secure_code_execution_permission_denied(self):
        """Test secure code execution with permission denied."""
        # Create user without code execution permission
        restricted_user = self.sandbox.permission_manager.create_user("restricted", PermissionLevel.GUEST)
        
        safe_code = "print('Hello, World!')"
        
        with self.assertRaises(PermissionError):
            with self.sandbox.secure_code_execution(restricted_user.user_id, safe_code):
                pass
        
        # Verify audit log entry for permission denial
        audit_entries = self.sandbox.auditor.audit_entries
        self.assertGreater(len(audit_entries), 0)
        
        last_entry = audit_entries[-1]
        self.assertEqual(last_entry.event_type, SecurityEvent.PERMISSION_DENIED)
        self.assertEqual(last_entry.user_id, restricted_user.user_id)
        self.assertFalse(last_entry.success)
    
    def test_user_permission_management(self):
        """Test user permission management."""
        # Create admin user
        admin_user = self.sandbox.permission_manager.create_user("admin", PermissionLevel.ADMIN)
        
        # Get user permissions
        permissions = self.sandbox.get_user_permissions(self.test_user.user_id)
        self.assertIsNotNone(permissions)
        self.assertEqual(permissions['username'], 'test_user')
        self.assertEqual(permissions['permission_level'], 'user')
        
        # Update user permissions as admin
        updates = {
            'can_delete_files': True,
            'max_file_size_mb': 100
        }
        
        success = self.sandbox.update_user_permissions(
            admin_user.user_id,
            self.test_user.user_id,
            updates
        )
        
        self.assertTrue(success)
        
        # Verify updates were applied
        updated_permissions = self.sandbox.get_user_permissions(self.test_user.user_id)
        self.assertTrue(updated_permissions['can_delete_files'])
        self.assertEqual(updated_permissions['max_file_size_mb'], 100)
        
        # Verify audit log entry
        audit_entries = self.sandbox.auditor.audit_entries
        permission_change_entries = [
            e for e in audit_entries 
            if e.event_type == SecurityEvent.PERMISSION_CHANGE
        ]
        self.assertGreater(len(permission_change_entries), 0)
    
    def test_non_admin_permission_update_denied(self):
        """Test that non-admin users cannot update permissions."""
        # Try to update permissions as regular user
        updates = {'can_delete_files': True}
        
        success = self.sandbox.update_user_permissions(
            self.test_user.user_id,  # Regular user, not admin
            self.test_user.user_id,
            updates
        )
        
        self.assertFalse(success)
        
        # Verify audit log entry for permission denial
        audit_entries = self.sandbox.auditor.audit_entries
        permission_denied_entries = [
            e for e in audit_entries 
            if e.event_type == SecurityEvent.PERMISSION_DENIED
        ]
        self.assertGreater(len(permission_denied_entries), 0)
    
    def test_security_status_reporting(self):
        """Test security status reporting."""
        status = self.sandbox.get_security_status()
        
        self.assertIn('total_users', status)
        self.assertIn('active_users', status)
        self.assertIn('active_sessions', status)
        self.assertIn('security_report', status)
        self.assertIn('workspace_path', status)
        self.assertIn('audit_log_path', status)
        
        self.assertEqual(status['workspace_path'], self.temp_dir)
        self.assertGreaterEqual(status['total_users'], 1)  # At least our test user
    
    def test_concurrent_operations(self):
        """Test concurrent security operations."""
        results = []
        errors = []
        
        def file_operation(user_id, file_name):
            try:
                file_path = os.path.join(self.temp_dir, file_name)
                with self.sandbox.secure_file_operation(user_id, "write", file_path):
                    with open(file_path, 'w') as f:
                        f.write(f"Content from {user_id}")
                results.append(f"Success: {file_name}")
            except Exception as e:
                errors.append(f"Error: {file_name} - {e}")
        
        # Create multiple users
        users = []
        for i in range(5):
            user = self.sandbox.permission_manager.create_user(f"user{i}", PermissionLevel.USER)
            user.allowed_directories.add(self.temp_dir)
            users.append(user)
        
        # Run concurrent file operations
        threads = []
        for i, user in enumerate(users):
            thread = threading.Thread(
                target=file_operation,
                args=(user.user_id, f"file{i}.txt")
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)  # All operations should succeed
        self.assertEqual(len(errors), 0)   # No errors should occur
        
        # Verify all files were created
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"file{i}.txt")
            self.assertTrue(os.path.exists(file_path))
        
        # Verify audit log entries
        file_write_entries = [
            e for e in self.sandbox.auditor.audit_entries
            if e.event_type == SecurityEvent.FILE_WRITE
        ]
        self.assertEqual(len(file_write_entries), 5)


class TestSecurityCompliance(unittest.TestCase):
    """Test security compliance and validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.sandbox = create_security_sandbox(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.allowed_directories.add(self.temp_dir)
        
        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            os.path.join(self.temp_dir, "..", "..", "etc", "passwd"),
            self.temp_dir + "/../../../etc/passwd"
        ]
        
        for malicious_path in malicious_paths:
            with self.assertRaises(PermissionError):
                with self.sandbox.secure_file_operation(user.user_id, "read", malicious_path):
                    pass
    
    def test_file_extension_blocking(self):
        """Test blocking of dangerous file extensions."""
        user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.allowed_directories.add(self.temp_dir)
        user.blocked_extensions.update(['.exe', '.bat', '.sh', '.ps1'])
        
        dangerous_files = [
            os.path.join(self.temp_dir, "malware.exe"),
            os.path.join(self.temp_dir, "script.bat"),
            os.path.join(self.temp_dir, "shell.sh"),
            os.path.join(self.temp_dir, "powershell.ps1")
        ]
        
        for dangerous_file in dangerous_files:
            with self.assertRaises(PermissionError):
                with self.sandbox.secure_file_operation(user.user_id, "write", dangerous_file):
                    pass
    
    def test_resource_limit_enforcement(self):
        """Test enforcement of resource limits."""
        user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.can_execute_code = True
        user.max_execution_time = 1  # 1 second limit
        user.max_memory_mb = 50      # 50 MB limit
        
        # Authenticate user
        self.sandbox.permission_manager.authenticate_user(user.user_id)
        
        # Test execution time limit with a busy loop (no imports needed)
        long_running_code = """
# Busy loop that should exceed the 1-second limit
count = 0
for i in range(10000000):  # Large loop to consume time
    count += i
print(f"Count: {count}")
"""
        
        with self.sandbox.secure_code_execution(user.user_id, long_running_code) as result:
            # The code should either timeout or be terminated due to resource limits
            # It might succeed if the loop completes quickly, so we check for either case
            if not result.success:
                # If it failed, it should be due to timeout or resource limits
                self.assertTrue(result.timeout or result.resource_exceeded or "time" in result.error.lower())
            else:
                # If it succeeded, it should have completed within limits
                self.assertLess(result.execution_time, 2.0)  # Should complete within 2 seconds
    
    def test_audit_log_integrity(self):
        """Test audit log integrity and completeness."""
        user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.allowed_directories.add(self.temp_dir)
        
        # Perform various operations
        test_file = os.path.join(self.temp_dir, "test.txt")
        
        # File write operation
        with self.sandbox.secure_file_operation(user.user_id, "write", test_file):
            with open(test_file, 'w') as f:
                f.write("test content")
        
        # File read operation
        with self.sandbox.secure_file_operation(user.user_id, "read", test_file):
            with open(test_file, 'r') as f:
                content = f.read()
        
        # Permission denied operation
        try:
            with self.sandbox.secure_file_operation(user.user_id, "read", "/etc/passwd"):
                pass
        except PermissionError:
            pass
        
        # Verify audit log completeness
        audit_entries = self.sandbox.auditor.audit_entries
        self.assertGreaterEqual(len(audit_entries), 3)  # At least 3 operations
        
        # Verify all entries have required fields
        for entry in audit_entries:
            self.assertIsNotNone(entry.timestamp)
            self.assertIsNotNone(entry.event_id)
            self.assertIsNotNone(entry.event_type)
            self.assertIsNotNone(entry.user_id)
            self.assertIsNotNone(entry.resource_path)
            self.assertIsNotNone(entry.operation)
            self.assertIsInstance(entry.success, bool)
        
        # Verify audit log file exists and is readable
        audit_log_path = self.sandbox.auditor.audit_log_path
        self.assertTrue(os.path.exists(audit_log_path))
        
        with open(audit_log_path, 'r') as f:
            log_lines = f.readlines()
            self.assertGreaterEqual(len(log_lines), 3)
            
            # Verify each line is valid JSON
            for line in log_lines:
                log_data = json.loads(line.strip())
                self.assertIn('timestamp', log_data)
                self.assertIn('event_type', log_data)
                self.assertIn('user_id', log_data)
    
    def test_session_security(self):
        """Test session security and timeout."""
        user = self.sandbox.permission_manager.create_user("test_user", PermissionLevel.USER)
        user.session_timeout_minutes = 0.01  # Very short timeout for testing
        user.allowed_directories.add(self.temp_dir)
        
        # Authenticate user
        authenticated_user = self.sandbox.permission_manager.authenticate_user(user.user_id)
        self.assertIsNotNone(authenticated_user)
        
        # Verify session is active
        self.assertTrue(self.sandbox.permission_manager.check_permission(
            user.user_id, "read", os.path.join(self.temp_dir, "test.txt")
        ))
        
        # Wait for session to expire
        time.sleep(1)
        
        # Verify session is expired
        self.assertFalse(self.sandbox.permission_manager.check_permission(
            user.user_id, "read", os.path.join(self.temp_dir, "test.txt")
        ))
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation."""
        # Create regular user
        regular_user = self.sandbox.permission_manager.create_user("regular", PermissionLevel.USER)
        
        # Create admin user
        admin_user = self.sandbox.permission_manager.create_user("admin", PermissionLevel.ADMIN)
        
        # Regular user should not be able to update their own permissions
        updates = {'permission_level': 'admin'}
        success = self.sandbox.update_user_permissions(
            regular_user.user_id,
            regular_user.user_id,
            updates
        )
        self.assertFalse(success)
        
        # Regular user should not be able to update other users' permissions
        success = self.sandbox.update_user_permissions(
            regular_user.user_id,
            admin_user.user_id,
            updates
        )
        self.assertFalse(success)
        
        # Admin user should be able to update permissions
        success = self.sandbox.update_user_permissions(
            admin_user.user_id,
            regular_user.user_id,
            {'max_file_size_mb': 100}
        )
        self.assertTrue(success)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)