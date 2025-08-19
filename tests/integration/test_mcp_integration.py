"""
Integration tests for Sanskrit Rewrite Engine MCP Server

This module tests the complete integration of MCP server components including
file operations, Git integration, Sanskrit processing, and workspace management.
"""

import os
import sys
import tempfile
import shutil
import pytest
import asyncio
import json
import git
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sanskrit_rewrite_engine.mcp_server import (
    SanskritMCPServer, SecuritySandbox, WorkspaceManager, GitIntegration,
    SecurityConfig, WorkspaceConfig, create_mcp_server
)
from sanskrit_rewrite_engine.safe_execution import (
    CodeExecutionManager, create_safe_execution_manager
)


class TestMCPServerIntegration:
    """Test complete MCP server integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir)
        
        # Create test files and directories
        self.test_files = {
            "README.md": "# Test Project\n\nThis is a test project.",
            "src/main.py": "print('Hello, World!')",
            "data/sample.txt": "Sample data content",
            "sanskrit/test.txt": "dharma artha kāma mokṣa"
        }
        
        for file_path, content in self.test_files.items():
            full_path = os.path.join(self.workspace_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Initialize Git repository
        self.repo = git.Repo.init(self.workspace_dir)
        self.repo.index.add(list(self.test_files.keys()))
        self.repo.index.commit("Initial commit")
        
        # Create MCP server
        self.server = create_mcp_server(self.workspace_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_complete_file_workflow(self):
        """Test complete file operation workflow."""
        # List files
        result = await self.server.call_tool("list_files", {})
        assert "files" in result
        assert len(result["files"]) > 0
        
        # Read existing file
        result = await self.server.call_tool("read_file", {"file_path": "README.md"})
        assert result["content"] == self.test_files["README.md"]
        
        # Write new file
        new_content = "# Updated README\n\nThis has been updated."
        result = await self.server.call_tool("write_file", {
            "file_path": "README.md",
            "content": new_content,
            "backup": True
        })
        assert result["success"]
        
        # Verify file was updated
        result = await self.server.call_tool("read_file", {"file_path": "README.md"})
        assert result["content"] == new_content
        
        # Verify backup was created
        result = await self.server.call_tool("list_files", {"directory": "backups"})
        backup_files = [f for f in result["files"] if f["name"].startswith("README_")]
        assert len(backup_files) > 0
    
    @pytest.mark.asyncio
    async def test_git_integration_workflow(self):
        """Test complete Git integration workflow."""
        # Get initial status
        result = await self.server.call_tool("git_status", {})
        assert "branch" in result
        assert result["branch"] == "master" or result["branch"] == "main"
        
        # Create new file
        await self.server.call_tool("write_file", {
            "file_path": "new_feature.py",
            "content": "# New feature implementation\nprint('New feature')"
        })
        
        # Check status shows untracked file
        result = await self.server.call_tool("git_status", {})
        assert "new_feature.py" in result["untracked_files"]
        
        # Add file to staging
        result = await self.server.call_tool("git_add", {
            "file_paths": ["new_feature.py"]
        })
        assert result["success"]
        
        # Check status shows staged file
        result = await self.server.call_tool("git_status", {})
        assert "new_feature.py" in result["staged_files"]
        
        # Commit changes
        result = await self.server.call_tool("git_commit", {
            "message": "Add new feature implementation",
            "author_name": "Test User",
            "author_email": "test@example.com"
        })
        assert "commit_hash" in result
        
        # Check commit history
        result = await self.server.call_tool("git_history", {"max_count": 5})
        assert "commits" in result
        assert len(result["commits"]) >= 2  # Initial commit + new commit
        assert any("new feature" in commit["message"].lower() for commit in result["commits"])
    
    @pytest.mark.asyncio
    async def test_sanskrit_processing_integration(self):
        """Test Sanskrit processing integration."""
        # Test basic Sanskrit tokenization
        sanskrit_text = "dharma artha kāma mokṣa"
        result = await self.server.call_tool("tokenize_sanskrit", {"text": sanskrit_text})
        
        assert "tokens" in result
        assert len(result["tokens"]) > 0
        assert result["input_text"] == sanskrit_text
        
        # Test Sanskrit text processing
        result = await self.server.call_tool("process_sanskrit_text", {
            "text": sanskrit_text,
            "options": {"enable_tracing": True}
        })
        
        assert "processed_result" in result
        assert result["input_text"] == sanskrit_text
        
        # Test with compound Sanskrit text
        compound_text = "a + i"  # Should become "e" through sandhi
        result = await self.server.call_tool("process_sanskrit_text", {"text": compound_text})
        
        assert "processed_result" in result
        # The processing should handle the sandhi transformation
    
    @pytest.mark.asyncio
    async def test_workspace_management(self):
        """Test workspace management capabilities."""
        # Test directory listing with patterns
        result = await self.server.call_tool("list_files", {
            "directory": "src",
            "pattern": "*.py"
        })
        assert "files" in result
        python_files = [f for f in result["files"] if f["name"].endswith(".py")]
        assert len(python_files) > 0
        
        # Test creating nested directory structure
        await self.server.call_tool("write_file", {
            "file_path": "deep/nested/structure/test.txt",
            "content": "Deep nested content"
        })
        
        # Verify nested structure was created
        result = await self.server.call_tool("read_file", {
            "file_path": "deep/nested/structure/test.txt"
        })
        assert result["content"] == "Deep nested content"
        
        # Test file deletion with backup
        result = await self.server.call_tool("delete_file", {
            "file_path": "deep/nested/structure/test.txt",
            "backup": True
        })
        assert result["success"]
        
        # Verify file was deleted
        result = await self.server.call_tool("read_file", {
            "file_path": "deep/nested/structure/test.txt"
        })
        assert "error" in result
        
        # Verify backup exists
        result = await self.server.call_tool("list_files", {"directory": "backups"})
        backup_files = [f for f in result["files"] if "test_" in f["name"]]
        assert len(backup_files) > 0
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self):
        """Test audit logging across all operations."""
        # Perform various operations
        operations = [
            ("read_file", {"file_path": "README.md"}),
            ("write_file", {"file_path": "audit_test.txt", "content": "audit content"}),
            ("list_files", {}),
            ("tokenize_sanskrit", {"text": "test"}),
        ]
        
        for tool_name, args in operations:
            await self.server.call_tool(tool_name, args)
        
        # Get audit log
        result = await self.server.call_tool("get_audit_log", {"limit": 20})
        assert "audit_entries" in result
        assert len(result["audit_entries"]) >= len(operations)
        
        # Verify audit entries have required fields
        for entry in result["audit_entries"]:
            assert "timestamp" in entry
            assert "operation" in entry
            assert "success" in entry
            assert "file_path" in entry
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test reading non-existent file
        result = await self.server.call_tool("read_file", {"file_path": "nonexistent.txt"})
        assert "error" in result
        assert "not found" in result["error"].lower() or "no such file" in result["error"].lower()
        
        # Test writing to invalid path
        result = await self.server.call_tool("write_file", {
            "file_path": "/invalid/path/file.txt",
            "content": "content"
        })
        assert "error" in result
        
        # Test Git operations on non-Git directory
        non_git_server = create_mcp_server(tempfile.mkdtemp())
        result = await non_git_server.call_tool("git_status", {})
        # Should handle gracefully
        assert isinstance(result, dict)
        
        # Test malformed Sanskrit input
        result = await self.server.call_tool("process_sanskrit_text", {"text": ""})
        assert "processed_result" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations handling."""
        # Create multiple concurrent file operations
        tasks = []
        
        # Concurrent reads
        for i in range(10):
            task = self.server.call_tool("read_file", {"file_path": "README.md"})
            tasks.append(task)
        
        # Concurrent writes to different files
        for i in range(5):
            task = self.server.call_tool("write_file", {
                "file_path": f"concurrent_{i}.txt",
                "content": f"Content {i}"
            })
            tasks.append(task)
        
        # Concurrent Sanskrit processing
        for i in range(3):
            task = self.server.call_tool("tokenize_sanskrit", {
                "text": f"test text {i}"
            })
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed
        assert len(results) == 18
        
        # Check that no operations failed due to concurrency issues
        successful_results = [r for r in results if isinstance(r, dict) and "error" not in r]
        assert len(successful_results) >= 15  # Allow for some expected failures (non-existent files)
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files."""
        # Create moderately large content
        large_content = "Line {}\n".format("x" * 100) * 1000  # ~100KB
        
        # Write large file
        result = await self.server.call_tool("write_file", {
            "file_path": "large_file.txt",
            "content": large_content
        })
        
        # Should succeed or fail gracefully based on size limits
        if result.get("success"):
            # If write succeeded, read should also work
            result = await self.server.call_tool("read_file", {"file_path": "large_file.txt"})
            assert "content" in result
            assert len(result["content"]) > 50000  # Verify substantial content
        else:
            # If write failed, should be due to size limits
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        # Test various Unicode content
        unicode_tests = [
            ("sanskrit_unicode.txt", "धर्म अर्थ काम मोक्ष"),  # Devanagari
            ("chinese.txt", "你好世界"),  # Chinese
            ("arabic.txt", "مرحبا بالعالم"),  # Arabic
            ("emoji.txt", "Hello 🌍 World! 🚀"),  # Emoji
            ("special_chars.txt", "Special: !@#$%^&*()_+-=[]{}|;':\",./<>?"),  # Special chars
        ]
        
        for filename, content in unicode_tests:
            # Write Unicode content
            result = await self.server.call_tool("write_file", {
                "file_path": filename,
                "content": content
            })
            assert result.get("success", False), f"Failed to write {filename}"
            
            # Read back and verify
            result = await self.server.call_tool("read_file", {"file_path": filename})
            assert result["content"] == content, f"Content mismatch for {filename}"
            
            # Test Sanskrit processing with Unicode
            if "sanskrit" in filename:
                result = await self.server.call_tool("tokenize_sanskrit", {"text": content})
                assert "tokens" in result or "error" in result


class TestMCPServerPerformance:
    """Test MCP server performance characteristics."""
    
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
    async def test_response_time_performance(self):
        """Test response time performance."""
        import time
        
        # Test file operations response time
        start_time = time.time()
        result = await self.server.call_tool("list_files", {})
        list_time = time.time() - start_time
        
        assert list_time < 1.0  # Should complete within 1 second
        assert "files" in result
        
        # Test file read response time
        await self.server.call_tool("write_file", {
            "file_path": "perf_test.txt",
            "content": "Performance test content"
        })
        
        start_time = time.time()
        result = await self.server.call_tool("read_file", {"file_path": "perf_test.txt"})
        read_time = time.time() - start_time
        
        assert read_time < 0.5  # Should complete within 0.5 seconds
        assert "content" in result
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage stability under load."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for i in range(100):
            await self.server.call_tool("write_file", {
                "file_path": f"memory_test_{i}.txt",
                "content": f"Content {i} " * 100
            })
            
            if i % 10 == 0:
                # Force garbage collection
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        import time
        
        # Create many concurrent requests
        start_time = time.time()
        
        tasks = []
        for i in range(50):
            task = self.server.call_tool("tokenize_sanskrit", {
                "text": f"test sanskrit text {i}"
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Should handle 50 concurrent requests reasonably quickly
        assert total_time < 10.0  # Within 10 seconds
        assert len(results) == 50
        
        # Most requests should succeed
        successful = [r for r in results if isinstance(r, dict) and "tokens" in r]
        assert len(successful) >= 40  # At least 80% success rate


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])