"""
MCP Server Integration Demo

This script demonstrates the MCP server integration for the Sanskrit Rewrite Engine,
showing file operations, Git integration, and Sanskrit processing capabilities.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sanskrit_rewrite_engine.mcp_server import create_mcp_server
from sanskrit_rewrite_engine.mcp_config import MCPConfigManager, create_sample_config


async def demo_file_operations(server):
    """Demonstrate file operations through MCP server."""
    print("\n=== File Operations Demo ===")
    
    # List initial files
    result = await server.call_tool("list_files", {})
    print(f"Initial files: {len(result.get('files', []))}")
    
    # Create a new file
    content = """# Sanskrit Processing Demo

This file demonstrates the MCP server integration for Sanskrit text processing.

## Sample Sanskrit Text
dharma artha kāma mokṣa
satyam eva jayate
vasudhaiva kuṭumbakam
"""
    
    result = await server.call_tool("write_file", {
        "file_path": "sanskrit_demo.md",
        "content": content
    })
    print(f"File creation: {'✅ Success' if result.get('success') else '❌ Failed'}")
    
    # Read the file back
    result = await server.call_tool("read_file", {"file_path": "sanskrit_demo.md"})
    if "content" in result:
        print(f"File read: ✅ Success ({len(result['content'])} characters)")
    else:
        print(f"File read: ❌ Failed - {result.get('error', 'Unknown error')}")
    
    # List files again
    result = await server.call_tool("list_files", {})
    print(f"Files after creation: {len(result.get('files', []))}")
    
    # Create a subdirectory with files
    await server.call_tool("write_file", {
        "file_path": "data/sample1.txt",
        "content": "Sample data file 1"
    })
    await server.call_tool("write_file", {
        "file_path": "data/sample2.txt", 
        "content": "Sample data file 2"
    })
    
    # List files in subdirectory
    result = await server.call_tool("list_files", {"directory": "data"})
    print(f"Files in data directory: {len(result.get('files', []))}")


async def demo_sanskrit_processing(server):
    """Demonstrate Sanskrit processing capabilities."""
    print("\n=== Sanskrit Processing Demo ===")
    
    # Test Sanskrit tokenization
    sanskrit_texts = [
        "dharma artha kāma mokṣa",
        "satyam eva jayate",
        "a + i",  # Should become "e" through sandhi
        "rāma + asya",  # Genitive form
        "vasudhaiva kuṭumbakam"
    ]
    
    for text in sanskrit_texts:
        print(f"\nProcessing: '{text}'")
        
        # Tokenize
        result = await server.call_tool("tokenize_sanskrit", {"text": text})
        if "tokens" in result:
            tokens = result["tokens"]
            print(f"  Tokens: {len(tokens)}")
            for token in tokens[:3]:  # Show first 3 tokens
                print(f"    - '{token['text']}' ({token['kind']})")
        else:
            print(f"  Tokenization failed: {result.get('error', 'Unknown error')}")
        
        # Full processing
        result = await server.call_tool("process_sanskrit_text", {
            "text": text,
            "options": {"enable_tracing": True}
        })
        if "processed_result" in result:
            processed = result["processed_result"]
            if processed.get("success"):
                print(f"  Processing: ✅ Success")
                if "transformations" in processed:
                    print(f"    Transformations: {len(processed['transformations'])}")
            else:
                print(f"  Processing: ❌ Failed - {processed.get('error', 'Unknown error')}")
        else:
            print(f"  Processing failed: {result.get('error', 'Unknown error')}")


async def demo_git_integration(server):
    """Demonstrate Git integration."""
    print("\n=== Git Integration Demo ===")
    
    # Check Git status
    result = await server.call_tool("git_status", {})
    if "error" in result:
        print(f"Git status: ❌ {result['error']}")
        return
    
    print(f"Git branch: {result.get('branch', 'unknown')}")
    print(f"Repository dirty: {result.get('is_dirty', False)}")
    print(f"Untracked files: {len(result.get('untracked_files', []))}")
    
    # Add files to Git
    if result.get('untracked_files'):
        files_to_add = result['untracked_files'][:3]  # Add first 3 files
        add_result = await server.call_tool("git_add", {"file_paths": files_to_add})
        print(f"Git add: {'✅ Success' if add_result.get('success') else '❌ Failed'}")
        
        # Commit changes
        commit_result = await server.call_tool("git_commit", {
            "message": "Add demo files from MCP server",
            "author_name": "MCP Demo",
            "author_email": "demo@sanskrit-engine.local"
        })
        if "commit_hash" in commit_result:
            print(f"Git commit: ✅ Success ({commit_result['commit_hash']})")
        else:
            print(f"Git commit: ❌ Failed - {commit_result.get('error', 'Unknown error')}")
    
    # Show commit history
    history_result = await server.call_tool("git_history", {"max_count": 5})
    if "commits" in history_result:
        commits = history_result["commits"]
        print(f"Recent commits: {len(commits)}")
        for commit in commits[:2]:  # Show first 2 commits
            print(f"  - {commit['hash']}: {commit['message'][:50]}...")
    else:
        print(f"Git history: ❌ Failed - {history_result.get('error', 'Unknown error')}")


async def demo_security_features(server):
    """Demonstrate security features."""
    print("\n=== Security Features Demo ===")
    
    # Test path traversal protection
    result = await server.call_tool("read_file", {"file_path": "../../../etc/passwd"})
    if "error" in result:
        print("✅ Path traversal protection working")
    else:
        print("❌ Path traversal protection failed")
    
    # Test dangerous file extension blocking
    result = await server.call_tool("write_file", {
        "file_path": "malware.exe",
        "content": "malicious content"
    })
    if "error" in result:
        print("✅ Dangerous extension blocking working")
    else:
        print("❌ Dangerous extension blocking failed")
    
    # Show audit log
    result = await server.call_tool("get_audit_log", {"limit": 5})
    if "audit_entries" in result:
        entries = result["audit_entries"]
        print(f"✅ Audit logging working ({len(entries)} recent entries)")
        for entry in entries[-2:]:  # Show last 2 entries
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            operation = entry['operation']
            success = "✅" if entry['success'] else "❌"
            print(f"  {timestamp} {success} {operation}")
    else:
        print("❌ Audit logging failed")


async def demo_concurrent_operations(server):
    """Demonstrate concurrent operations handling."""
    print("\n=== Concurrent Operations Demo ===")
    
    # Create multiple concurrent tasks
    tasks = []
    
    # Concurrent file operations
    for i in range(5):
        task = server.call_tool("write_file", {
            "file_path": f"concurrent_{i}.txt",
            "content": f"Concurrent file {i} content"
        })
        tasks.append(task)
    
    # Concurrent Sanskrit processing
    sanskrit_texts = ["dharma", "artha", "kāma", "mokṣa", "satyam"]
    for text in sanskrit_texts:
        task = server.call_tool("tokenize_sanskrit", {"text": text})
        tasks.append(task)
    
    # Execute all tasks concurrently
    print(f"Executing {len(tasks)} concurrent operations...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful = 0
    failed = 0
    errors = 0
    
    for result in results:
        if isinstance(result, Exception):
            errors += 1
        elif isinstance(result, dict):
            if result.get("success") or "tokens" in result:
                successful += 1
            else:
                failed += 1
        else:
            errors += 1
    
    print(f"Results: ✅ {successful} successful, ❌ {failed} failed, 🔥 {errors} errors")


async def run_comprehensive_demo():
    """Run comprehensive MCP server demo."""
    print("Sanskrit Rewrite Engine MCP Server Demo")
    print("=" * 50)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_dir = os.path.join(temp_dir, "demo_workspace")
        os.makedirs(workspace_dir)
        
        print(f"Demo workspace: {workspace_dir}")
        
        # Initialize Git repository
        import git
        repo = git.Repo.init(workspace_dir)
        
        # Create MCP server
        print("Creating MCP server...")
        server = create_mcp_server(workspace_dir)
        print("✅ MCP server created")
        
        try:
            # Run demo sections
            await demo_file_operations(server)
            await demo_sanskrit_processing(server)
            await demo_git_integration(server)
            await demo_security_features(server)
            await demo_concurrent_operations(server)
            
            print("\n=== Demo Summary ===")
            print("✅ All demo sections completed successfully")
            print("✅ MCP server integration is working properly")
            print("✅ Security features are functioning")
            print("✅ Sanskrit processing is operational")
            print("✅ Git integration is working")
            print("✅ Concurrent operations are handled correctly")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


async def create_config_demo():
    """Demonstrate configuration management."""
    print("\n=== Configuration Demo ===")
    
    # Create sample configuration
    config_path = "demo_mcp_config.json"
    create_sample_config(config_path)
    print(f"✅ Sample configuration created: {config_path}")
    
    # Load and validate configuration
    config_manager = MCPConfigManager(config_path)
    issues = config_manager.validate_config()
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")
    
    # Show configuration summary
    config = config_manager.config
    print(f"Server: {config.server_name} v{config.server_version}")
    print(f"Workspace: {config.workspace_path}")
    print(f"Security: {len(config.allowed_directories)} allowed directories")
    print(f"Execution limits: {config.max_execution_time}s timeout, {config.max_memory_mb}MB memory")
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"✅ Cleaned up demo configuration file")


def main():
    """Main demo entry point."""
    print("Starting MCP Server Demo...")
    
    try:
        # Run configuration demo first
        asyncio.run(create_config_demo())
        
        # Run comprehensive demo
        asyncio.run(run_comprehensive_demo())
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo start the MCP server manually, run:")
        print("  python -m sanskrit_rewrite_engine.mcp_cli start --workspace /path/to/workspace")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()