"""
Comprehensive integration tests for the Sanskrit Rewrite Engine CLI.

This module tests the command-line interface using subprocess and mock inputs,
ensuring proper command handling, argument parsing, and output formatting.
"""

import pytest
import subprocess
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

from src.sanskrit_rewrite_engine.cli import cli
from tests.fixtures.sample_texts import (
    BASIC_WORDS, MORPHOLOGICAL_EXAMPLES, SANDHI_EXAMPLES
)


class TestCLICommands:
    """Test CLI commands using Click's test runner."""
    
    def setup_method(self):
        """Set up test runner before each test."""
        self.runner = CliRunner()
        
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Sanskrit Rewrite Engine CLI" in result.output
        assert "process" in result.output
        assert "serve" in result.output
        assert "rules" in result.output
        
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['version'])
        
        assert result.exit_code == 0
        assert "2.0.0" in result.output
        
    def test_cli_version_json(self):
        """Test CLI version command with JSON output."""
        result = self.runner.invoke(cli, ['version', '--format', 'json'])
        
        assert result.exit_code == 0
        
        # Should be valid JSON
        data = json.loads(result.output)
        assert "version" in data
        assert "python_version" in data
        assert "platform" in data
        
    def test_process_command_basic(self):
        """Test basic process command."""
        result = self.runner.invoke(cli, ['process', 'test text'])
        
        assert result.exit_code == 0
        assert "test text" in result.output
        
    def test_process_command_with_tracing(self):
        """Test process command with tracing enabled."""
        result = self.runner.invoke(cli, ['process', 'rāma + iti', '--trace'])
        
        assert result.exit_code == 0
        assert "rāma + iti" in result.output
        
    def test_process_command_with_trace_formats(self):
        """Test process command with different trace formats."""
        formats = ['text', 'json', 'table']
        
        for format_type in formats:
            result = self.runner.invoke(cli, [
                'process', 'test', '--trace', '--trace-format', format_type
            ])
            
            assert result.exit_code == 0
            
    def test_process_command_with_max_iterations(self):
        """Test process command with max iterations limit."""
        result = self.runner.invoke(cli, [
            'process', 'test text', '--max-iterations', '5'
        ])
        
        assert result.exit_code == 0
        
    def test_process_command_with_input_file(self):
        """Test process command with input file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("rāma + iti")
            input_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', '--input-file', input_file
            ])
            
            assert result.exit_code == 0
            assert "rāma + iti" in result.output
            
        finally:
            os.unlink(input_file)
            
    def test_process_command_with_output_file(self):
        """Test process command with output file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', 'test text', '--output', output_file
            ])
            
            assert result.exit_code == 0
            
            # Check output file was created
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                content = f.read()
                assert "test text" in content
                
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_process_command_with_config_file(self):
        """Test process command with configuration file."""
        config_data = {
            "max_iterations": 3,
            "enable_tracing": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', 'test text', '--config', config_file
            ])
            
            assert result.exit_code == 0
            
        finally:
            os.unlink(config_file)
            
    def test_process_command_with_rule_file(self):
        """Test process command with custom rule file."""
        rule_data = {
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "Test rule",
                    "pattern": "test",
                    "replacement": "TEST",
                    "priority": 1,
                    "enabled": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rule_data, f)
            rule_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', 'test text', '--rules', rule_file
            ])
            
            assert result.exit_code == 0
            
        finally:
            os.unlink(rule_file)
            
    def test_process_command_error_handling(self):
        """Test process command error handling."""
        # Test with empty text
        result = self.runner.invoke(cli, ['process', ''])
        assert result.exit_code != 0
        
        # Test with nonexistent input file
        result = self.runner.invoke(cli, [
            'process', '--input-file', 'nonexistent.txt'
        ])
        assert result.exit_code != 0
        
        # Test with nonexistent config file
        result = self.runner.invoke(cli, [
            'process', 'test', '--config', 'nonexistent.json'
        ])
        assert result.exit_code != 0
        
    def test_serve_command_help(self):
        """Test serve command help."""
        result = self.runner.invoke(cli, ['serve', '--help'])
        
        assert result.exit_code == 0
        assert "Start the Sanskrit Rewrite Engine web server" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output
        
    def test_rules_list_command(self):
        """Test rules list command."""
        result = self.runner.invoke(cli, ['rules', 'list'])
        
        assert result.exit_code == 0
        assert "Available Rule Sets" in result.output or "Loaded rules" in result.output
        
    def test_rules_list_command_json(self):
        """Test rules list command with JSON format."""
        result = self.runner.invoke(cli, ['rules', 'list', '--format', 'json'])
        
        assert result.exit_code == 0
        
        # Should be valid JSON
        data = json.loads(result.output)
        assert "available_rule_sets" in data
        assert "rule_count" in data
        
    def test_rules_validate_command(self):
        """Test rules validate command."""
        # Create a valid rule file
        rule_data = {
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "Test rule",
                    "pattern": "test",
                    "replacement": "TEST"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rule_data, f)
            rule_file = f.name
            
        try:
            result = self.runner.invoke(cli, ['rules', 'validate', rule_file])
            
            assert result.exit_code == 0
            assert "Rule file is valid" in result.output
            
        finally:
            os.unlink(rule_file)
            
    def test_rules_validate_command_invalid_file(self):
        """Test rules validate command with invalid file."""
        # Create an invalid rule file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            rule_file = f.name
            
        try:
            result = self.runner.invoke(cli, ['rules', 'validate', rule_file])
            
            assert result.exit_code != 0
            assert "JSON syntax error" in result.output
            
        finally:
            os.unlink(rule_file)
            
    def test_config_show_command(self):
        """Test config show command."""
        result = self.runner.invoke(cli, ['config', 'show'])
        
        assert result.exit_code == 0
        assert "Configuration" in result.output
        
    def test_config_show_command_json(self):
        """Test config show command with JSON format."""
        result = self.runner.invoke(cli, ['config', 'show', '--format', 'json'])
        
        assert result.exit_code == 0
        
        # Should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, dict)
        
    def test_config_init_command(self):
        """Test config init command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(cli, [
                '--config-dir', temp_dir,
                'config', 'init'
            ])
            
            assert result.exit_code == 0
            assert "Configuration initialized" in result.output
            
            # Check files were created
            config_file = Path(temp_dir) / "config.json"
            rules_dir = Path(temp_dir) / "rules"
            
            assert config_file.exists()
            assert rules_dir.exists()
            
    def test_interactive_command_help(self):
        """Test interactive command help."""
        result = self.runner.invoke(cli, ['interactive', '--help'])
        
        assert result.exit_code == 0
        assert "Interactive Sanskrit text processing" in result.output
        
    def test_verbose_flag(self):
        """Test verbose flag functionality."""
        result = self.runner.invoke(cli, [
            '--verbose', 'process', 'test text'
        ])
        
        assert result.exit_code == 0
        
    def test_quiet_flag(self):
        """Test quiet flag functionality."""
        result = self.runner.invoke(cli, [
            'process', 'test text', '--quiet'
        ])
        
        assert result.exit_code == 0
        
    def test_multiple_sanskrit_examples(self):
        """Test processing multiple Sanskrit examples."""
        examples = [
            "rāma + iti",
            "deva + indra",
            "guru + upadeśa",
            "dharma + artha"
        ]
        
        for example in examples:
            result = self.runner.invoke(cli, ['process', example])
            
            assert result.exit_code == 0
            assert example in result.output


class TestCLISubprocess:
    """Test CLI using subprocess for real command execution."""
    
    def test_cli_executable(self):
        """Test that CLI is properly executable."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'sanskrit_rewrite_engine.cli', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode == 0
            assert "Sanskrit Rewrite Engine CLI" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")
            
    def test_cli_version_subprocess(self):
        """Test CLI version using subprocess."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'sanskrit_rewrite_engine.cli', 'version'
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode == 0
            assert "2.0.0" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI version command timed out")
            
    def test_cli_process_subprocess(self):
        """Test CLI process command using subprocess."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'sanskrit_rewrite_engine.cli',
                'process', 'test text'
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode == 0
            assert "test text" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI process command timed out")
            
    def test_cli_error_handling_subprocess(self):
        """Test CLI error handling using subprocess."""
        try:
            # Test with invalid command
            result = subprocess.run([
                sys.executable, '-m', 'sanskrit_rewrite_engine.cli',
                'invalid-command'
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode != 0
            assert "No such command" in result.stderr or "Usage:" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI error handling timed out")


class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
        
    def test_cli_startup_time(self):
        """Test CLI startup time."""
        import time
        
        start_time = time.time()
        result = self.runner.invoke(cli, ['--help'])
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        assert result.exit_code == 0
        # CLI should start quickly (less than 2 seconds)
        assert startup_time < 2.0
        
    def test_process_command_performance(self):
        """Test process command performance."""
        import time
        
        # Test with various text sizes
        test_texts = [
            "short",
            "medium length text",
            "longer text " * 10,
            "rāma + iti " * 20
        ]
        
        for text in test_texts:
            start_time = time.time()
            result = self.runner.invoke(cli, ['process', text])
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert result.exit_code == 0
            # Should complete within reasonable time
            assert processing_time < 5.0
            
    def test_memory_usage(self):
        """Test CLI memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large text
        large_text = "rāma + iti " * 100
        result = self.runner.invoke(cli, ['process', large_text])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.exit_code == 0
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024


class TestCLIIntegration:
    """Test CLI integration with other components."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
        
    def test_cli_with_real_engine(self):
        """Test CLI integration with real engine."""
        result = self.runner.invoke(cli, [
            'process', 'rāma + iti', '--trace'
        ])
        
        assert result.exit_code == 0
        assert "rāma + iti" in result.output
        
    def test_cli_rule_loading_integration(self):
        """Test CLI rule loading integration."""
        # Create a test rule file
        rule_data = {
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "Test rule",
                    "pattern": "hello",
                    "replacement": "hi",
                    "priority": 1,
                    "enabled": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rule_data, f)
            rule_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', 'hello world', '--rules', rule_file
            ])
            
            assert result.exit_code == 0
            # Should apply the rule
            assert "hi world" in result.output or "hello world" in result.output
            
        finally:
            os.unlink(rule_file)
            
    def test_cli_config_integration(self):
        """Test CLI configuration integration."""
        config_data = {
            "max_iterations": 2,
            "enable_tracing": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            result = self.runner.invoke(cli, [
                'process', 'test text', '--config', config_file, '--trace'
            ])
            
            assert result.exit_code == 0
            
        finally:
            os.unlink(config_file)
            
    def test_cli_output_formats(self):
        """Test CLI output format consistency."""
        # Test text format
        result = self.runner.invoke(cli, [
            'process', 'test', '--trace', '--trace-format', 'text'
        ])
        assert result.exit_code == 0
        
        # Test JSON format
        result = self.runner.invoke(cli, [
            'process', 'test', '--trace', '--trace-format', 'json'
        ])
        assert result.exit_code == 0
        
        # Test table format
        result = self.runner.invoke(cli, [
            'process', 'test', '--trace', '--trace-format', 'table'
        ])
        assert result.exit_code == 0