"""
Command Line Interface for Sanskrit Rewrite Engine MCP Server

This module provides a CLI for managing, configuring, and running the MCP server.
"""

import os
import sys
import argparse
import asyncio
import logging
import json
import signal
from pathlib import Path
from typing import Optional, List
import tempfile
import shutil

from .mcp_server import create_mcp_server, run_mcp_server
from .mcp_config import MCPConfigManager, create_sample_config
from .safe_execution import create_safe_execution_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPServerCLI:
    """Command line interface for MCP server management."""
    
    def __init__(self):
        self.config_manager: Optional[MCPConfigManager] = None
        self.server = None
        self.running = False
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser."""
        parser = argparse.ArgumentParser(
            description="Sanskrit Rewrite Engine MCP Server CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start MCP server with default configuration
  python -m sanskrit_rewrite_engine.mcp_cli start
  
  # Start server with custom workspace
  python -m sanskrit_rewrite_engine.mcp_cli start --workspace /path/to/project
  
  # Create sample configuration
  python -m sanskrit_rewrite_engine.mcp_cli config create --output mcp_config.json
  
  # Validate configuration
  python -m sanskrit_rewrite_engine.mcp_cli config validate --config mcp_config.json
  
  # Test server functionality
  python -m sanskrit_rewrite_engine.mcp_cli test --workspace /path/to/project
  
  # Show server status
  python -m sanskrit_rewrite_engine.mcp_cli status
            """
        )
        
        # Global options
        parser.add_argument("--verbose", "-v", action="store_true",
                           help="Enable verbose logging")
        parser.add_argument("--debug", action="store_true",
                           help="Enable debug logging")
        parser.add_argument("--config", type=str,
                           help="Path to configuration file")
        parser.add_argument("--workspace", type=str, default=".",
                           help="Workspace directory path")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Start command
        start_parser = subparsers.add_parser("start", help="Start MCP server")
        start_parser.add_argument("--host", type=str, default="localhost",
                                 help="Server host (default: localhost)")
        start_parser.add_argument("--port", type=int, default=8000,
                                 help="Server port (default: 8000)")
        start_parser.add_argument("--daemon", action="store_true",
                                 help="Run as daemon process")
        
        # Stop command
        stop_parser = subparsers.add_parser("stop", help="Stop MCP server")
        stop_parser.add_argument("--force", action="store_true",
                                help="Force stop server")
        
        # Status command
        status_parser = subparsers.add_parser("status", help="Show server status")
        
        # Config commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_command")
        
        create_config_parser = config_subparsers.add_parser("create", help="Create sample configuration")
        create_config_parser.add_argument("--output", type=str, default="mcp_config.json",
                                         help="Output configuration file path")
        create_config_parser.add_argument("--format", choices=["json", "yaml"], default="json",
                                         help="Configuration file format")
        
        validate_config_parser = config_subparsers.add_parser("validate", help="Validate configuration")
        
        show_config_parser = config_subparsers.add_parser("show", help="Show current configuration")
        show_config_parser.add_argument("--format", choices=["json", "yaml"], default="json",
                                       help="Output format")
        
        # Test command
        test_parser = subparsers.add_parser("test", help="Test server functionality")
        test_parser.add_argument("--quick", action="store_true",
                                help="Run quick tests only")
        test_parser.add_argument("--security", action="store_true",
                                help="Run security tests")
        test_parser.add_argument("--output", type=str,
                                help="Test results output file")
        
        # Tools command
        tools_parser = subparsers.add_parser("tools", help="Manage MCP tools")
        tools_subparsers = tools_parser.add_subparsers(dest="tools_command")
        
        list_tools_parser = tools_subparsers.add_parser("list", help="List available tools")
        
        enable_tool_parser = tools_subparsers.add_parser("enable", help="Enable tool")
        enable_tool_parser.add_argument("tool_name", help="Tool name to enable")
        
        disable_tool_parser = tools_subparsers.add_parser("disable", help="Disable tool")
        disable_tool_parser.add_argument("tool_name", help="Tool name to disable")
        
        # Audit command
        audit_parser = subparsers.add_parser("audit", help="Audit log management")
        audit_parser.add_argument("--show", action="store_true",
                                 help="Show recent audit entries")
        audit_parser.add_argument("--limit", type=int, default=50,
                                 help="Number of entries to show")
        audit_parser.add_argument("--export", type=str,
                                 help="Export audit log to file")
        
        return parser
    
    def setup_logging(self, verbose: bool = False, debug: bool = False):
        """Setup logging configuration."""
        if debug:
            level = logging.DEBUG
        elif verbose:
            level = logging.INFO
        else:
            level = logging.WARNING
        
        logging.getLogger().setLevel(level)
        
        # Set specific logger levels
        logging.getLogger("sanskrit_rewrite_engine").setLevel(level)
    
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration."""
        try:
            self.config_manager = MCPConfigManager(config_path)
            
            # Validate configuration
            issues = self.config_manager.validate_config()
            if issues:
                logger.warning("Configuration issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    async def start_server(self, args):
        """Start MCP server."""
        logger.info("Starting Sanskrit MCP Server...")
        
        try:
            # Create server
            workspace_path = args.workspace or self.config_manager.config.workspace_path
            self.server = create_mcp_server(workspace_path)
            
            # Update host and port from args
            host = args.host or self.config_manager.config.host
            port = args.port or self.config_manager.config.port
            
            logger.info(f"Server starting on {host}:{port}")
            logger.info(f"Workspace: {workspace_path}")
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                logger.info("Received shutdown signal")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            self.running = True
            
            if args.daemon:
                logger.info("Running in daemon mode")
                # In a real implementation, this would fork the process
                # For now, just run normally
            
            # Run server
            await run_mcp_server(self.server, host, port)
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            sys.exit(1)
    
    def stop_server(self, args):
        """Stop MCP server."""
        logger.info("Stopping MCP server...")
        
        if self.server:
            self.running = False
            logger.info("Server stopped")
        else:
            logger.warning("No server instance found")
    
    def show_status(self, args):
        """Show server status."""
        print("Sanskrit MCP Server Status")
        print("=" * 30)
        
        if self.config_manager:
            config = self.config_manager.config
            print(f"Workspace: {config.workspace_path}")
            print(f"Host: {config.host}")
            print(f"Port: {config.port}")
            print(f"Git Operations: {'Enabled' if config.enable_git_operations else 'Disabled'}")
            print(f"Code Execution: {'Enabled' if config.enable_code_execution else 'Disabled'}")
            print(f"Sanskrit Processing: {'Enabled' if config.enable_sanskrit_processing else 'Disabled'}")
            print(f"Audit Logging: {'Enabled' if config.audit_log_enabled else 'Disabled'}")
        
        print(f"Server Running: {'Yes' if self.running else 'No'}")
    
    def create_config(self, args):
        """Create sample configuration."""
        try:
            output_path = args.output
            if args.format == "yaml":
                if not output_path.endswith(('.yaml', '.yml')):
                    output_path = output_path.replace('.json', '.yaml')
            
            create_sample_config(output_path)
            print(f"Sample configuration created: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            sys.exit(1)
    
    def validate_config(self, args):
        """Validate configuration."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        issues = self.config_manager.validate_config()
        
        if issues:
            print("Configuration Issues:")
            for issue in issues:
                print(f"  ❌ {issue}")
            sys.exit(1)
        else:
            print("✅ Configuration is valid")
    
    def show_config(self, args):
        """Show current configuration."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        config_dict = {
            "server": self.config_manager.config.__dict__,
            "tools": {name: tool.__dict__ for name, tool in self.config_manager.tool_configs.items()}
        }
        
        if args.format == "yaml":
            import yaml
            print(yaml.dump(config_dict, default_flow_style=False, indent=2))
        else:
            print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    
    async def run_tests(self, args):
        """Run server tests."""
        logger.info("Running MCP server tests...")
        
        try:
            # Create temporary workspace for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                test_workspace = os.path.join(temp_dir, "test_workspace")
                os.makedirs(test_workspace)
                
                # Create test server
                test_server = create_mcp_server(test_workspace)
                
                # Run basic functionality tests
                await self._run_basic_tests(test_server)
                
                if args.security:
                    await self._run_security_tests(test_server)
                
                if not args.quick:
                    await self._run_comprehensive_tests(test_server)
                
                logger.info("All tests completed successfully")
                
        except Exception as e:
            logger.error(f"Tests failed: {e}")
            sys.exit(1)
    
    async def _run_basic_tests(self, server):
        """Run basic functionality tests."""
        logger.info("Running basic functionality tests...")
        
        # Test file operations
        result = await server.call_tool("write_file", {
            "file_path": "test.txt",
            "content": "Test content"
        })
        assert result.get("success"), "File write test failed"
        
        result = await server.call_tool("read_file", {"file_path": "test.txt"})
        assert result.get("content") == "Test content", "File read test failed"
        
        # Test Sanskrit processing
        result = await server.call_tool("tokenize_sanskrit", {"text": "dharma"})
        assert "tokens" in result, "Sanskrit tokenization test failed"
        
        logger.info("✅ Basic tests passed")
    
    async def _run_security_tests(self, server):
        """Run security tests."""
        logger.info("Running security tests...")
        
        # Test path traversal protection
        result = await server.call_tool("read_file", {"file_path": "../../../etc/passwd"})
        assert "error" in result, "Path traversal protection test failed"
        
        # Test dangerous file extension blocking
        result = await server.call_tool("write_file", {
            "file_path": "malware.exe",
            "content": "malicious"
        })
        assert "error" in result, "Dangerous extension blocking test failed"
        
        logger.info("✅ Security tests passed")
    
    async def _run_comprehensive_tests(self, server):
        """Run comprehensive tests."""
        logger.info("Running comprehensive tests...")
        
        # Test concurrent operations
        tasks = []
        for i in range(10):
            task = server.call_tool("tokenize_sanskrit", {"text": f"test {i}"})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = [r for r in results if isinstance(r, dict) and "tokens" in r]
        assert len(successful) >= 8, "Concurrent operations test failed"
        
        logger.info("✅ Comprehensive tests passed")
    
    def list_tools(self, args):
        """List available tools."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        print("Available MCP Tools:")
        print("=" * 20)
        
        # Standard tools
        standard_tools = [
            "list_files", "read_file", "write_file", "delete_file",
            "git_status", "git_add", "git_commit", "git_history",
            "process_sanskrit_text", "tokenize_sanskrit", "get_audit_log"
        ]
        
        for tool_name in standard_tools:
            tool_config = self.config_manager.get_tool_config(tool_name)
            status = "✅ Enabled" if tool_config.enabled else "❌ Disabled"
            print(f"  {tool_name}: {status}")
    
    def enable_tool(self, args):
        """Enable a tool."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        tool_config = self.config_manager.get_tool_config(args.tool_name)
        tool_config.enabled = True
        self.config_manager.set_tool_config(args.tool_name, tool_config)
        self.config_manager.save_config()
        
        print(f"✅ Tool '{args.tool_name}' enabled")
    
    def disable_tool(self, args):
        """Disable a tool."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        tool_config = self.config_manager.get_tool_config(args.tool_name)
        tool_config.enabled = False
        self.config_manager.set_tool_config(args.tool_name, tool_config)
        self.config_manager.save_config()
        
        print(f"❌ Tool '{args.tool_name}' disabled")
    
    def show_audit_log(self, args):
        """Show audit log entries."""
        if not self.config_manager:
            logger.error("No configuration loaded")
            sys.exit(1)
        
        audit_log_path = self.config_manager.config.audit_log_path
        if not audit_log_path or not os.path.exists(audit_log_path):
            print("No audit log found")
            return
        
        try:
            with open(audit_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Show recent entries
            recent_lines = lines[-args.limit:] if len(lines) > args.limit else lines
            
            print(f"Recent Audit Log Entries ({len(recent_lines)} of {len(lines)}):")
            print("=" * 50)
            
            for line in recent_lines:
                try:
                    entry = json.loads(line.strip())
                    timestamp = entry.get('timestamp', 'Unknown')
                    operation = entry.get('operation', 'Unknown')
                    file_path = entry.get('file_path', 'Unknown')
                    success = entry.get('success', False)
                    status = "✅" if success else "❌"
                    
                    print(f"{timestamp} {status} {operation}: {file_path}")
                    
                except json.JSONDecodeError:
                    continue
            
            if args.export:
                shutil.copy2(audit_log_path, args.export)
                print(f"Audit log exported to: {args.export}")
                
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
    
    async def run(self):
        """Main CLI entry point."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Setup logging
        self.setup_logging(args.verbose, args.debug)
        
        # Load configuration
        self.load_config(args.config)
        
        # Handle commands
        try:
            if args.command == "start":
                await self.start_server(args)
            elif args.command == "stop":
                self.stop_server(args)
            elif args.command == "status":
                self.show_status(args)
            elif args.command == "config":
                if args.config_command == "create":
                    self.create_config(args)
                elif args.config_command == "validate":
                    self.validate_config(args)
                elif args.config_command == "show":
                    self.show_config(args)
                else:
                    parser.print_help()
            elif args.command == "test":
                await self.run_tests(args)
            elif args.command == "tools":
                if args.tools_command == "list":
                    self.list_tools(args)
                elif args.tools_command == "enable":
                    self.enable_tool(args)
                elif args.tools_command == "disable":
                    self.disable_tool(args)
                else:
                    parser.print_help()
            elif args.command == "audit":
                self.show_audit_log(args)
            else:
                parser.print_help()
                
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli = MCPServerCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()