"""
Security tests for Sanskrit Rewrite Engine MCP Server

This module contains comprehensive security tests including penetration testing
for the MCP server integration, file system security, and safe execution environment.
"""

import os
import sys
import tempfile
import shutil
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sanskrit_rewrite_engine.mcp_server import (
    SanskritMCPServer, SecuritySandbox, WorkspaceManager, GitIntegration,
    SecurityConfig, WorkspaceConfig, create_mcp_server
)
from sanskrit_rewrite_engine.safe_execution import (
    SafeExecutionEnvironment, CodeExecutionManager, SecurityValidator,
    ExecutionLimits, ExecutionContext, create_safe_execution_manager
)


class TestSecuritySandbox:
    """Test security sandbox functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.allowed_dir = os.path.join(self.temp_dir, "allowed")
        self.blocked_dir = os.path.join(self.temp_dir, "blocked")
        
        os.makedirs(self.allowed_dir)
        os.makedirs(self.blocked_dir)
        
        self.config = SecurityConfig(
            allowed_directories=[self.allowed_dir],
            blocked_extensions=['.exe', '.bat', '.sh'],
            max_file_size_mb=1,
            audit_log_path=os.path.join(self.temp_dir, "audit.log")
        )
        
        self.sandbox = SecuritySandbox(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_access_control(self):
        """Test path access control."""
        # Allowed path should be accessible
        allowed_file = os.path.join(self.allowed_dir, "test.txt")
        assert self.sandbox.is_path_allowed(allowed_file)
        
        # Blocked path should not be accessible
        blocked_file = os.path.join(self.blocked_dir, "test.txt")
        assert not self.sandbox.is_path_allowed(blocked_file)
        
        # Path traversal attempts should be blocked
        traversal_path = os.path.join(self.allowed_dir, "..", "blocked", "test.txt")
        assert not self.sandbox.is_path_allowed(traversal_path)
    
    def test_extension_filtering(self):
        """Test file extension filtering."""
        # Safe extensions should be allowed
        assert self.sandbox.is_extension_allowed("test.txt")
        assert self.sandbox.is_extension_allowed("data.json")
        assert self.sandbox.is_extension_allowed("script.py")
        
        # Dangerous extensions should be blocked
        assert not self.sandbox.is_extension_allowed("malware.exe")
        assert not self.sandbox.is_extension_allowed("script.bat")
        assert not self.sandbox.is_extension_allowed("shell.sh")
    
    def test_file_size_limits(self):
        """Test file size limits."""
        # Create small file (should pass)
        small_file = os.path.join(self.allowed_dir, "small.txt")
        with open(small_file, 'w') as f:
            f.write("small content")
        assert self.sandbox.check_file_size(small_file)
        
        # Create large file (should fail)
        large_file = os.path.join(self.allowed_dir, "large.txt")
        with open(large_file, 'w') as f:
            f.write("x" * (2 * 1024 * 1024))  # 2MB file
        assert not self.sandbox.check_file_size(large_file)
    
    def test_audit_logging(self):
        """Test audit logging functionality."""
        test_file = os.path.join(self.allowed_dir, "test.txt")
        
        # Log successful operation
        self.sandbox.log_operation("test_read", test_file, True, "test_user")
        
        # Log failed operation
        self.sandbox.log_operation("test_write", test_file, False, "test_user", "Permission denied")
        
        # Check audit log entries
        assert len(self.sandbox.audit_log) == 2
        assert self.sandbox.audit_log[0].operation == "test_read"
        assert self.sandbox.audit_log[0].success == True
        assert self.sandbox.audit_log[1].operation == "test_write"
        assert self.sandbox.audit_log[1].success == False
        assert self.sandbox.audit_log[1].error_message == "Permission denied"
        
        # Check audit log file
        if self.config.audit_log_path:
            assert os.path.exists(self.config.audit_log_path)


class TestWorkspaceManager:
    """Test workspace manager security."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir)
        
        self.security_config = SecurityConfig(
            allowed_directories=[self.workspace_dir],
            blocked_extensions=['.exe', '.bat'],
            max_file_size_mb=1
        )
        
        self.workspace_config = WorkspaceConfig(
            root_path=self.workspace_dir,
            project_name="test_project"
        )
        
        self.sandbox = SecuritySandbox(self.security_config)
        self.workspace = WorkspaceManager(self.workspace_config, self.sandbox)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_secure_file_operations(self):
        """Test secure file operations."""
        # Test writing allowed file
        content = "test content"
        assert self.workspace.write_file("test.txt", content)
        
        # Test reading file
        read_content = self.workspace.read_file("test.txt")
        assert read_content == content
        
        # Test writing blocked extension
        with pytest.raises(PermissionError):
            self.workspace.write_file("malware.exe", "malicious content")
        
        # Test path traversal attack
        with pytest.raises(PermissionError):
            self.workspace.write_file("../../../etc/passwd", "hacked")
    
    def test_backup_functionality(self):
        """Test backup functionality."""
        # Create initial file
        self.workspace.write_file("backup_test.txt", "original content")
        
        # Modify file with backup
        self.workspace.write_file("backup_test.txt", "modified content", backup=True)
        
        # Check backup was created
        backup_dir = Path(self.workspace_dir) / "backups"
        backup_files = list(backup_dir.glob("backup_test_*.txt"))
        assert len(backup_files) > 0
    
    def test_directory_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        # Various directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for attempt in traversal_attempts:
            with pytest.raises(PermissionError):
                self.workspace.read_file(attempt)
            
            with pytest.raises(PermissionError):
                self.workspace.write_file(attempt, "malicious content")


class TestSafeExecution:
    """Test safe code execution environment."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.limits = ExecutionLimits(
            max_execution_time=5,
            max_memory_mb=64,
            max_cpu_percent=50
        )
        self.executor = SafeExecutionEnvironment(self.limits)
        self.manager = create_safe_execution_manager(self.temp_dir, self.limits)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_safe_code_execution(self):
        """Test execution of safe code."""
        safe_code = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
        
        context = ExecutionContext(code=safe_code)
        result = self.executor.execute_code(context)
        
        assert result.success
        assert "Result: 4.0" in result.output
        assert not result.security_violation
        assert not result.timeout
    
    def test_dangerous_code_blocking(self):
        """Test blocking of dangerous code."""
        dangerous_codes = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.call(['ls', '-la'])",
            "eval('__import__(\"os\").system(\"ls\")')",
            "exec('import os; os.listdir(\"/\")')",
            "open('/etc/passwd', 'r').read()",
            "__import__('os').system('whoami')"
        ]
        
        for dangerous_code in dangerous_codes:
            context = ExecutionContext(code=dangerous_code)
            result = self.executor.execute_code(context)
            
            assert not result.success
            assert result.security_violation
            assert "Security validation failed" in result.error
    
    def test_resource_limits(self):
        """Test resource limit enforcement."""
        # Test memory limit
        memory_bomb = """
data = []
for i in range(1000000):
    data.append('x' * 1000)
"""
        
        result = self.manager.execute_code(memory_bomb)
        # Should either fail validation or be terminated by resource monitor
        assert not result.success
        
        # Test time limit
        infinite_loop = """
while True:
    pass
"""
        
        result = self.manager.execute_code(infinite_loop)
        assert not result.success
        assert result.timeout or result.resource_exceeded
    
    def test_import_restrictions(self):
        """Test import restrictions."""
        # Allowed imports should work
        allowed_code = """
import math
import json
import datetime
print("Allowed imports work")
"""
        
        result = self.manager.execute_code(allowed_code)
        assert result.success
        
        # Blocked imports should fail
        blocked_imports = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "from os import system",
            "import ctypes"
        ]
        
        for blocked_import in blocked_imports:
            result = self.manager.execute_code(blocked_import)
            assert not result.success
    
    def test_execution_audit_logging(self):
        """Test execution audit logging."""
        code = "print('Hello, World!')"
        
        # Execute code
        result = self.manager.execute_code(code, user_id="test_user")
        
        # Check audit log
        stats = self.manager.get_execution_stats()
        assert stats["total_executions"] == 1
        assert len(stats["recent_executions"]) >= 2  # start and complete events
        
        # Check audit log file
        audit_log_path = os.path.join(self.temp_dir, "execution_audit.log")
        assert os.path.exists(audit_log_path)


class TestMCPServerSecurity:
    """Test MCP server security integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir)
        
        # Create test files
        self.test_file = os.path.join(self.workspace_dir, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("test content")
        
        self.server = create_mcp_server(self.workspace_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_file_access_security(self):
        """Test file access security through MCP interface."""
        # Test reading allowed file
        result = await self.server.call_tool("read_file", {"file_path": "test.txt"})
        assert "content" in result
        assert result["content"] == "test content"
        
        # Test reading non-existent file
        result = await self.server.call_tool("read_file", {"file_path": "nonexistent.txt"})
        assert "error" in result
        
        # Test writing file
        result = await self.server.call_tool("write_file", {
            "file_path": "new_file.txt",
            "content": "new content"
        })
        assert result.get("success", False)
    
    @pytest.mark.asyncio
    async def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd"
        ]
        
        for attempt in traversal_attempts:
            # Test read attempt
            result = await self.server.call_tool("read_file", {"file_path": attempt})
            assert "error" in result
            assert "Access denied" in result["error"] or "Permission" in result["error"]
            
            # Test write attempt
            result = await self.server.call_tool("write_file", {
                "file_path": attempt,
                "content": "malicious content"
            })
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_sanskrit_processing_security(self):
        """Test Sanskrit processing security."""
        # Test normal Sanskrit processing
        result = await self.server.call_tool("process_sanskrit_text", {
            "text": "dharma artha kāma mokṣa"
        })
        assert "processed_result" in result
        
        # Test with potentially malicious input
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\x00\x01\x02\x03"  # null bytes and control characters
        ]
        
        for malicious_input in malicious_inputs:
            result = await self.server.call_tool("process_sanskrit_text", {
                "text": malicious_input
            })
            # Should not crash and should handle gracefully
            assert "processed_result" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_audit_log_access(self):
        """Test audit log access and security."""
        # Perform some operations to generate audit entries
        await self.server.call_tool("read_file", {"file_path": "test.txt"})
        await self.server.call_tool("write_file", {
            "file_path": "audit_test.txt",
            "content": "audit test"
        })
        
        # Get audit log
        result = await self.server.call_tool("get_audit_log", {"limit": 10})
        assert "audit_entries" in result
        assert len(result["audit_entries"]) > 0
        
        # Check audit entry structure
        entry = result["audit_entries"][0]
        required_fields = ["timestamp", "operation", "file_path", "success"]
        for field in required_fields:
            assert field in entry


class TestPenetrationTesting:
    """Penetration testing for MCP server."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir)
        self.server = create_mcp_server(self.workspace_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_injection_attacks(self):
        """Test various injection attacks."""
        # SQL injection attempts (even though we don't use SQL)
        sql_injections = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; DELETE FROM files; --",
            "' UNION SELECT * FROM passwords --"
        ]
        
        for injection in sql_injections:
            result = await self.server.call_tool("read_file", {"file_path": injection})
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_command_injection(self):
        """Test command injection attempts."""
        command_injections = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(ls -la)",
            "; cat /etc/shadow"
        ]
        
        for injection in command_injections:
            # Test in file paths
            result = await self.server.call_tool("read_file", {"file_path": injection})
            assert "error" in result
            
            # Test in file content
            result = await self.server.call_tool("write_file", {
                "file_path": "test.txt",
                "content": injection
            })
            # Should succeed but not execute commands
            if result.get("success"):
                # Verify content was written as-is, not executed
                read_result = await self.server.call_tool("read_file", {"file_path": "test.txt"})
                assert read_result.get("content") == injection
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_attempts(self):
        """Test buffer overflow attempts."""
        # Large inputs
        large_string = "A" * 10000
        very_large_string = "B" * 100000
        
        # Test with large file path
        result = await self.server.call_tool("read_file", {"file_path": large_string})
        assert "error" in result
        
        # Test with large content
        result = await self.server.call_tool("write_file", {
            "file_path": "large_test.txt",
            "content": very_large_string
        })
        # Should either succeed or fail gracefully due to size limits
        assert "success" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_unicode_and_encoding_attacks(self):
        """Test Unicode and encoding-based attacks."""
        unicode_attacks = [
            "test\x00.txt",  # Null byte injection
            "test\r\n.txt",  # CRLF injection
            "test\u202e.txt",  # Right-to-left override
            "test\ufeff.txt",  # Byte order mark
            "test\u0000.txt",  # Unicode null
            "test\u2028.txt",  # Line separator
            "test\u2029.txt"   # Paragraph separator
        ]
        
        for attack in unicode_attacks:
            result = await self.server.call_tool("read_file", {"file_path": attack})
            # Should handle gracefully without crashing
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion(self):
        """Test resource exhaustion attacks."""
        # Test many rapid requests
        tasks = []
        for i in range(100):
            task = self.server.call_tool("read_file", {"file_path": f"test_{i}.txt"})
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle all requests without crashing
        assert len(results) == 100
        for result in results:
            assert isinstance(result, (dict, Exception))
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_attempts(self):
        """Test privilege escalation attempts."""
        escalation_attempts = [
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Windows\\System32\\config\\SYSTEM",
            "/proc/self/environ",
            "/proc/version",
            "/sys/class/dmi/id/product_uuid"
        ]
        
        for attempt in escalation_attempts:
            result = await self.server.call_tool("read_file", {"file_path": attempt})
            assert "error" in result
            assert "Access denied" in result["error"] or "Permission" in result["error"]


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])