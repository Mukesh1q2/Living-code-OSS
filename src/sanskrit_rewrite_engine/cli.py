"""
Command-line interface for Sanskrit Rewrite Engine.

This module provides comprehensive CLI commands for the Sanskrit processing engine,
including text transformation, rule management, and server operations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from click import echo, style

from .engine import SanskritRewriteEngine, TransformationResult


# CLI Configuration
DEFAULT_CONFIG_DIR = Path.home() / ".sanskrit-rewrite-engine"
DEFAULT_RULES_DIR = Path("data/rules")


@click.group()
@click.version_option(version="2.0.0", prog_name="Sanskrit Rewrite Engine")
@click.option('--config-dir', 
              type=click.Path(path_type=Path),
              default=DEFAULT_CONFIG_DIR,
              help='Configuration directory path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx: click.Context, config_dir: Path, verbose: bool):
    """Sanskrit Rewrite Engine CLI.
    
    A sophisticated computational linguistics system for Sanskrit text processing
    based on Pāṇinian grammatical principles.
    
    \b
    Examples:
        sanskrit-cli process "rāma + iti" --trace
        sanskrit-cli process "deva + indra" --trace-format json
        sanskrit-cli serve --port 8080 --reload
        sanskrit-cli rules list --format table
        sanskrit-cli config show --format json
        sanskrit-cli config init
    
    \b
    For more help on specific commands:
        sanskrit-cli process --help
        sanskrit-cli serve --help
        sanskrit-cli rules --help
        sanskrit-cli config --help
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store global configuration
    ctx.obj['config_dir'] = config_dir
    ctx.obj['verbose'] = verbose
    
    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)


@cli.command()
@click.argument('text', required=False)
@click.option('--rules', '-r', 
              multiple=True,
              help='Rule files to load (can be specified multiple times)')
@click.option('--rule-set', '-s',
              help='Named rule set to use (e.g., "basic_sandhi", "compounds")')
@click.option('--trace', '-t', is_flag=True, 
              help='Show detailed transformation trace')
@click.option('--trace-format', 
              type=click.Choice(['text', 'json', 'table']),
              default='text',
              help='Format for trace output')
@click.option('--output', '-o', 
              type=click.Path(path_type=Path),
              help='Output file (default: stdout)')
@click.option('--max-iterations', '-m',
              type=int, default=10,
              help='Maximum transformation iterations')
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              help='Configuration file to use')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress all output except results')
@click.option('--input-file', '-i',
              type=click.Path(exists=True, path_type=Path),
              help='Read input text from file instead of argument')
@click.pass_context
def process(ctx: click.Context, 
           text: str,
           rules: tuple,
           rule_set: Optional[str],
           trace: bool,
           trace_format: str,
           output: Optional[Path],
           max_iterations: int,
           config: Optional[Path],
           quiet: bool,
           input_file: Optional[Path]):
    """Process Sanskrit text with grammatical transformations.
    
    Apply Sanskrit grammatical rules to transform input text according to
    Pāṇinian principles. Supports rule-based sandhi, compound formation,
    and morphological analysis.
    
    \b
    Examples:
        sanskrit-cli process "rāma + iti"
        sanskrit-cli process "deva + indra" --trace --trace-format json
        sanskrit-cli process "text" --rules sandhi.json --rules compounds.json
        sanskrit-cli process "text" --rule-set basic_sandhi --max-iterations 5
        sanskrit-cli process --input-file input.txt --output result.txt
    
    \b
    Rule Selection:
        --rules: Load specific rule files
        --rule-set: Use predefined rule collections
        If neither specified, uses default rules
    
    \b
    Trace Formats:
        text: Human-readable trace (default)
        json: Machine-readable JSON format
        table: Tabular format with columns
    """
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Handle input from file if specified
        if input_file:
            if not quiet and verbose:
                echo(f"Reading input from: {input_file}")
            text = input_file.read_text(encoding='utf-8').strip()
            if not text:
                raise click.ClickException("Input file is empty")
        elif not text:
            raise click.ClickException("Either TEXT argument or --input-file must be provided")
        
        if not text.strip():
            raise click.ClickException("Input text cannot be empty")
        
        # Initialize engine with configuration
        engine_config = {}
        if config:
            if not quiet and verbose:
                echo(f"Loading configuration from: {config}")
            with open(config, 'r', encoding='utf-8') as f:
                engine_config = json.load(f)
        
        # Override max iterations if specified
        if max_iterations != 10:
            engine_config['max_iterations'] = max_iterations
            
        engine = SanskritRewriteEngine(config=engine_config)
        
        # Load rules
        if rules:
            for rule_file in rules:
                if not quiet and verbose:
                    echo(f"Loading rules from: {rule_file}")
                try:
                    engine.load_rules(rule_file)
                except Exception as e:
                    raise click.ClickException(f"Failed to load rules from {rule_file}: {e}")
        
        elif rule_set:
            # Load predefined rule set
            rule_file = _find_rule_set(rule_set, ctx.obj['config_dir'])
            if rule_file:
                if not quiet and verbose:
                    echo(f"Loading rule set '{rule_set}' from: {rule_file}")
                engine.load_rules(str(rule_file))
            else:
                available_sets = _list_available_rule_sets(ctx.obj['config_dir'])
                raise click.ClickException(
                    f"Rule set '{rule_set}' not found. Available: {', '.join(available_sets)}"
                )
        
        # Process text
        if not quiet and verbose:
            echo(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
        result = engine.process(text, enable_tracing=trace)
        
        # Format output
        output_content = _format_result(result, trace, trace_format, verbose and not quiet)
        
        # Write output
        if output:
            output.write_text(output_content, encoding='utf-8')
            if not quiet:
                echo(f"Output written to: {output}")
        else:
            echo(output_content)
            
        # Exit with error code if processing failed
        if not result.success:
            if not quiet:
                echo(style("Processing failed", fg='red'), err=True)
            sys.exit(1)
            
    except click.ClickException:
        raise
    except Exception as e:
        if verbose:
            import traceback
            echo(f"Error: {e}\n{traceback.format_exc()}", err=True)
        else:
            echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', '-h', default='localhost', 
              help='Host address to bind to')
@click.option('--port', '-p', default=8000, type=int,
              help='Port number to bind to')
@click.option('--reload', is_flag=True, 
              help='Enable auto-reload for development')
@click.option('--workers', '-w', type=int, default=1,
              help='Number of worker processes')
@click.option('--log-level', 
              type=click.Choice(['debug', 'info', 'warning', 'error']),
              default='info',
              help='Logging level')
@click.option('--access-log/--no-access-log', default=True,
              help='Enable/disable access logging')
@click.option('--config', '-c',
              type=click.Path(exists=True, path_type=Path),
              help='Server configuration file')
@click.pass_context
def serve(ctx: click.Context,
          host: str, 
          port: int, 
          reload: bool,
          workers: int,
          log_level: str,
          access_log: bool,
          config: Optional[Path]):
    """Start the Sanskrit Rewrite Engine web server.
    
    Launches a FastAPI-based web server providing REST API endpoints
    for Sanskrit text processing and analysis.
    
    \b
    Examples:
        sanskrit-cli serve
        sanskrit-cli serve --port 8080 --reload
        sanskrit-cli serve --host 0.0.0.0 --workers 4
        sanskrit-cli serve --config server-config.json
    
    \b
    Server Features:
        - REST API at /api/process and /api/analyze
        - Interactive documentation at /docs
        - Health check at /health
        - CORS support for web frontends
        - Request logging and monitoring
    
    \b
    Development vs Production:
        --reload: Use for development (single worker, auto-reload)
        --workers: Use multiple workers for production
    """
    verbose = ctx.obj.get('verbose', False)
    
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install 'sanskrit-rewrite-engine[web]'"
        )
    
    try:
        # Load server configuration if provided
        server_config = {}
        if config:
            if verbose:
                echo(f"Loading server configuration from: {config}")
            with open(config, 'r', encoding='utf-8') as f:
                server_config = json.load(f)
        
        # Prepare uvicorn configuration
        uvicorn_config = {
            "app": "sanskrit_rewrite_engine.server:app",
            "host": host,
            "port": port,
            "log_level": log_level,
            "access_log": access_log,
        }
        
        # Add reload or workers (mutually exclusive)
        if reload:
            uvicorn_config["reload"] = True
            if workers > 1:
                echo("Warning: --workers ignored when --reload is enabled", err=True)
        else:
            uvicorn_config["workers"] = workers
        
        # Override with config file settings
        uvicorn_config.update(server_config)
        
        # Display startup information
        echo(style("Sanskrit Rewrite Engine Server", fg='green', bold=True))
        echo(f"Starting server on {style(f'http://{host}:{port}', fg='blue')}")
        echo(f"API Documentation: {style(f'http://{host}:{port}/docs', fg='blue')}")
        echo(f"Health Check: {style(f'http://{host}:{port}/health', fg='blue')}")
        
        if reload:
            echo(style("Development mode: Auto-reload enabled", fg='yellow'))
        else:
            echo(f"Production mode: {workers} worker(s)")
        
        echo("Press Ctrl+C to stop the server")
        echo("-" * 50)
        
        # Start server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        echo("\nServer stopped by user")
    except Exception as e:
        if verbose:
            import traceback
            echo(f"Error starting server: {e}\n{traceback.format_exc()}", err=True)
        else:
            echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@cli.group()
def rules():
    """Manage transformation rules and rule sets.
    
    Commands for listing, validating, and managing Sanskrit transformation
    rules used by the processing engine.
    """
    pass


@rules.command('list')
@click.option('--format', 'output_format',
              type=click.Choice(['text', 'json', 'table']),
              default='text',
              help='Output format')
@click.option('--category', '-c',
              help='Filter by rule category')
@click.option('--enabled-only', is_flag=True,
              help='Show only enabled rules')
@click.pass_context
def list_rules(ctx: click.Context, output_format: str, category: Optional[str], enabled_only: bool):
    """List available transformation rules.
    
    \b
    Examples:
        sanskrit-cli rules list
        sanskrit-cli rules list --format json
        sanskrit-cli rules list --category sandhi
        sanskrit-cli rules list --enabled-only
    """
    try:
        engine = SanskritRewriteEngine()
        
        # Load default rules to get rule information
        available_rule_sets = _list_available_rule_sets(ctx.obj['config_dir'])
        
        if output_format == 'json':
            result = {
                "available_rule_sets": available_rule_sets,
                "rule_count": engine.get_rule_count(),
                "categories": ["sandhi", "morphology", "compounds", "inflection"]
            }
            echo(json.dumps(result, indent=2))
        else:
            echo(style("Available Rule Sets:", fg='green', bold=True))
            for rule_set in available_rule_sets:
                echo(f"  • {rule_set}")
            
            echo(f"\nLoaded rules: {engine.get_rule_count()}")
            echo("\nRule categories: sandhi, morphology, compounds, inflection")
            
    except Exception as e:
        echo(f"Error listing rules: {e}", err=True)
        sys.exit(1)


@rules.command('validate')
@click.argument('rule_file', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_rules(ctx: click.Context, rule_file: Path):
    """Validate a rule file for syntax and completeness.
    
    \b
    Examples:
        sanskrit-cli rules validate data/rules/sandhi.json
        sanskrit-cli rules validate my-custom-rules.json
    """
    try:
        with open(rule_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        # Basic validation
        required_fields = ['rule_set', 'version', 'rules']
        for field in required_fields:
            if field not in rules_data:
                raise ValueError(f"Missing required field: {field}")
        
        rules = rules_data['rules']
        if not isinstance(rules, list):
            raise ValueError("'rules' must be a list")
        
        # Validate each rule
        for i, rule in enumerate(rules):
            rule_required = ['id', 'name', 'pattern', 'replacement']
            for field in rule_required:
                if field not in rule:
                    raise ValueError(f"Rule {i}: Missing required field '{field}'")
        
        echo(style("✓ Rule file is valid", fg='green'))
        echo(f"Rule set: {rules_data['rule_set']}")
        echo(f"Version: {rules_data['version']}")
        echo(f"Rules count: {len(rules)}")
        
    except json.JSONDecodeError as e:
        echo(f"JSON syntax error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        echo(f"Validation error: {e}", err=True)
        sys.exit(1)


@cli.group()
def config():
    """Manage engine configuration and settings.
    
    Commands for viewing and managing Sanskrit Rewrite Engine configuration.
    """
    pass


@cli.command()
@click.option('--format', 'output_format',
              type=click.Choice(['text', 'json']),
              default='text',
              help='Output format')
def version(output_format: str):
    """Show version information.
    
    \b
    Examples:
        sanskrit-cli version
        sanskrit-cli version --format json
    """
    version_info = {
        "version": "2.0.0",
        "python_version": sys.version,
        "platform": sys.platform
    }
    
    if output_format == 'json':
        echo(json.dumps(version_info, indent=2))
    else:
        echo(style("Sanskrit Rewrite Engine", fg='green', bold=True))
        echo(f"Version: {version_info['version']}")
        echo(f"Python: {version_info['python_version']}")
        echo(f"Platform: {version_info['platform']}")


@cli.command()
@click.argument('text', required=False)
@click.option('--interactive', '-I', is_flag=True,
              help='Start interactive mode for processing multiple texts')
def interactive(text: Optional[str], interactive: bool):
    """Interactive Sanskrit text processing.
    
    Start an interactive session for processing multiple Sanskrit texts
    with immediate feedback and rule tracing.
    
    \b
    Examples:
        sanskrit-cli interactive
        sanskrit-cli interactive "initial text"
    """
    try:
        engine = SanskritRewriteEngine()
        
        echo(style("Sanskrit Rewrite Engine - Interactive Mode", fg='green', bold=True))
        echo("Type 'help' for commands, 'quit' to exit")
        echo("-" * 50)
        
        if text:
            echo(f"Processing initial text: {text}")
            result = engine.process(text, enable_tracing=True)
            echo(_format_result(result, True, 'text', False))
            echo()
        
        while True:
            try:
                user_input = click.prompt("Sanskrit", type=str, default="").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    echo("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    echo("Commands:")
                    echo("  help    - Show this help")
                    echo("  quit    - Exit interactive mode")
                    echo("  config  - Show current configuration")
                    echo("  rules   - Show loaded rules count")
                    echo("  Any other text will be processed")
                    continue
                elif user_input.lower() == 'config':
                    config_data = engine.get_config()
                    for key, value in config_data.items():
                        echo(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'rules':
                    echo(f"Loaded rules: {engine.get_rule_count()}")
                    continue
                elif not user_input:
                    continue
                
                # Process the text
                result = engine.process(user_input, enable_tracing=True)
                echo(_format_result(result, True, 'text', False))
                echo()
                
            except KeyboardInterrupt:
                echo("\nGoodbye!")
                break
            except EOFError:
                echo("\nGoodbye!")
                break
                
    except Exception as e:
        echo(f"Error in interactive mode: {e}", err=True)
        sys.exit(1)


@config.command('show')
@click.option('--format', 'output_format',
              type=click.Choice(['text', 'json']),
              default='text',
              help='Output format')
@click.pass_context
def show_config(ctx: click.Context, output_format: str):
    """Show current engine configuration.
    
    \b
    Examples:
        sanskrit-cli config show
        sanskrit-cli config show --format json
    """
    try:
        engine = SanskritRewriteEngine()
        config_data = engine.get_config()
        
        if output_format == 'json':
            echo(json.dumps(config_data, indent=2))
        else:
            echo(style("Sanskrit Rewrite Engine Configuration:", fg='green', bold=True))
            for key, value in config_data.items():
                echo(f"  {key}: {value}")
                
    except Exception as e:
        echo(f"Error showing configuration: {e}", err=True)
        sys.exit(1)


@config.command('init')
@click.option('--force', is_flag=True,
              help='Overwrite existing configuration')
@click.pass_context
def init_config(ctx: click.Context, force: bool):
    """Initialize default configuration files.
    
    Creates default configuration and rule files in the user's
    configuration directory.
    
    \b
    Examples:
        sanskrit-cli config init
        sanskrit-cli config init --force
    """
    config_dir = ctx.obj['config_dir']
    
    try:
        # Create default configuration
        config_file = config_dir / "config.json"
        
        if config_file.exists() and not force:
            raise click.ClickException(
                f"Configuration already exists at {config_file}. Use --force to overwrite."
            )
        
        default_config = {
            "max_iterations": 10,
            "enable_tracing": True,
            "timeout_seconds": 30,
            "max_text_length": 10000,
            "rule_directories": ["data/rules", str(config_dir / "rules")]
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        # Create rules directory
        rules_dir = config_dir / "rules"
        rules_dir.mkdir(exist_ok=True)
        
        echo(style("✓ Configuration initialized", fg='green'))
        echo(f"Configuration file: {config_file}")
        echo(f"Rules directory: {rules_dir}")
        
    except Exception as e:
        echo(f"Error initializing configuration: {e}", err=True)
        sys.exit(1)


# Helper functions
def _format_result(result: TransformationResult, 
                  show_trace: bool, 
                  trace_format: str, 
                  verbose: bool) -> str:
    """Format processing result for output."""
    lines = []
    
    if verbose:
        lines.append(style("Sanskrit Text Processing Result", fg='green', bold=True))
        lines.append("-" * 40)
    
    # Basic result
    lines.append(f"Input:  {result.input_text}")
    lines.append(f"Output: {result.output_text}")
    
    if verbose:
        lines.append(f"Success: {result.success}")
        if result.transformations_applied:
            lines.append(f"Transformations: {len(result.transformations_applied)}")
    
    # Error handling
    if not result.success and result.error_message:
        lines.append(style(f"Error: {result.error_message}", fg='red'))
    
    # Trace information
    if show_trace and result.trace:
        lines.append("")
        if trace_format == 'json':
            lines.append(json.dumps(result.trace, indent=2))
        elif trace_format == 'table':
            lines.append(_format_trace_table(result.trace))
        else:  # text format
            lines.append(style("Transformation Trace:", fg='blue', bold=True))
            for i, step in enumerate(result.trace, 1):
                if isinstance(step, dict):
                    step_type = step.get('step', 'unknown')
                    lines.append(f"{i:2d}. {step_type}: {step}")
                else:
                    lines.append(f"{i:2d}. {step}")
    
    return "\n".join(lines)


def _format_trace_table(trace: List[Dict[str, Any]]) -> str:
    """Format trace as a table."""
    if not trace:
        return "No trace data available"
    
    # Simple table formatting
    lines = [style("Step | Type | Details", fg='blue', bold=True)]
    lines.append("-" * 50)
    
    for i, step in enumerate(trace, 1):
        if isinstance(step, dict):
            step_type = step.get('step', 'unknown')
            details = str(step).replace('\n', ' ')[:40] + "..."
            lines.append(f"{i:4d} | {step_type:12s} | {details}")
        else:
            lines.append(f"{i:4d} | {'unknown':12s} | {str(step)[:40]}...")
    
    return "\n".join(lines)


def _find_rule_set(rule_set_name: str, config_dir: Path) -> Optional[Path]:
    """Find a rule set file by name."""
    # Search in multiple locations
    search_paths = [
        DEFAULT_RULES_DIR,
        config_dir / "rules",
        Path(".")
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            # Try exact match
            rule_file = search_path / f"{rule_set_name}.json"
            if rule_file.exists():
                return rule_file
            
            # Try pattern matching
            for file_path in search_path.glob("*.json"):
                if rule_set_name.lower() in file_path.stem.lower():
                    return file_path
    
    return None


def _list_available_rule_sets(config_dir: Path) -> List[str]:
    """List available rule sets."""
    rule_sets = set()
    
    # Search in multiple locations
    search_paths = [
        DEFAULT_RULES_DIR,
        config_dir / "rules",
        Path(".")
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            for file_path in search_path.glob("*.json"):
                rule_sets.add(file_path.stem)
    
    return sorted(list(rule_sets))


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()