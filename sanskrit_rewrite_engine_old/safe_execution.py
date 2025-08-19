"""
Safe Execution Environment for Sanskrit Rewrite Engine MCP Server

This module provides a secure sandbox for executing generated code with
resource limits, monitoring, and audit logging. Implements security
requirements for controlled code execution.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import signal
import threading
import time
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLimits:
    """Resource limits for code execution."""
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 256  # MB
    max_cpu_percent: int = 50  # CPU usage percentage
    max_file_size_mb: int = 10  # MB
    max_open_files: int = 100
    max_processes: int = 5
    allowed_modules: List[str] = field(default_factory=lambda: [
        'math', 'json', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'string', 're', 'unicodedata'
    ])
    blocked_modules: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'telnetlib', 'ssl', 'hashlib',
        'hmac', 'secrets', 'ctypes', 'multiprocessing', 'threading'
    ])


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    exit_code: Optional[int] = None
    timeout: bool = False
    resource_exceeded: bool = False
    security_violation: bool = False


@dataclass
class ExecutionContext:
    """Context for code execution."""
    code: str
    language: str = "python"
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    input_data: Optional[str] = None
    allowed_imports: List[str] = field(default_factory=list)
    timeout: int = 30


class SecurityValidator:
    """Validates code for security issues before execution."""
    
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.dangerous_patterns = [
            # File system operations
            r'open\s*\(',
            r'file\s*\(',
            r'with\s+open',
            r'os\.',
            r'sys\.',
            r'subprocess\.',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'__import__',
            r'importlib',
            # Network operations
            r'socket\.',
            r'urllib\.',
            r'http\.',
            r'requests\.',
            # System operations
            r'ctypes\.',
            r'multiprocessing\.',
            r'threading\.',
            # Dangerous builtins
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
        ]
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for security issues."""
        issues = []
        warnings = []
        
        # Check for dangerous patterns
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern found: {pattern}")
        
        # Check imports
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if hasattr(self.limits, 'blocked_modules') and alias.name in self.limits.blocked_modules:
                            issues.append(f"Blocked module import: {alias.name}")
                        elif hasattr(self.limits, 'allowed_modules') and alias.name not in self.limits.allowed_modules:
                            warnings.append(f"Unusual module import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if hasattr(self.limits, 'blocked_modules') and node.module in self.limits.blocked_modules:
                        issues.append(f"Blocked module import: {node.module}")
                    elif hasattr(self.limits, 'allowed_modules') and node.module not in self.limits.allowed_modules:
                        warnings.append(f"Unusual module import: {node.module}")
        
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


class ResourceMonitor:
    """Monitors resource usage during code execution."""
    
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.process = None
        self.monitoring = False
        self.stats = {
            "max_memory_mb": 0,
            "max_cpu_percent": 0,
            "execution_time": 0,
            "resource_violations": []
        }
    
    def start_monitoring(self, process):
        """Start monitoring process resources."""
        if not PSUTIL_AVAILABLE:
            return
        self.process = process
        self.monitoring = True
        self.stats = {
            "max_memory_mb": 0,
            "max_cpu_percent": 0,
            "execution_time": 0,
            "resource_violations": []
        }
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        if not PSUTIL_AVAILABLE:
            return
            
        start_time = time.time()
        
        while self.monitoring and self.process and self.process.is_running():
            try:
                # Get memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.stats["max_memory_mb"] = max(self.stats["max_memory_mb"], memory_mb)
                
                # Get CPU usage
                cpu_percent = self.process.cpu_percent()
                self.stats["max_cpu_percent"] = max(self.stats["max_cpu_percent"], cpu_percent)
                
                # Update execution time
                self.stats["execution_time"] = time.time() - start_time
                
                # Check limits
                if memory_mb > self.limits.max_memory_mb:
                    self.stats["resource_violations"].append(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")
                    self._terminate_process("Memory limit exceeded")
                    break
                
                if cpu_percent > self.limits.max_cpu_percent:
                    self.stats["resource_violations"].append(f"CPU limit exceeded: {cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
                
                if self.stats["execution_time"] > self.limits.max_execution_time:
                    self.stats["resource_violations"].append(f"Time limit exceeded: {self.stats['execution_time']:.1f}s > {self.limits.max_execution_time}s")
                    self._terminate_process("Time limit exceeded")
                    break
                
                # Check number of open files
                try:
                    open_files = len(self.process.open_files())
                    if open_files > self.limits.max_open_files:
                        self.stats["resource_violations"].append(f"Open files limit exceeded: {open_files} > {self.limits.max_open_files}")
                        self._terminate_process("Open files limit exceeded")
                        break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def _terminate_process(self, reason: str):
        """Terminate the monitored process."""
        logger.warning(f"Terminating process: {reason}")
        try:
            if self.process and self.process.is_running():
                self.process.terminate()
                # Wait for graceful termination
                try:
                    self.process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    # Force kill if graceful termination fails
                    self.process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class SafeExecutionEnvironment:
    """Safe execution environment with sandboxing and monitoring."""
    
    def __init__(self, limits: ExecutionLimits = None):
        self.limits = limits or ExecutionLimits()
        self.validator = SecurityValidator(self.limits)
        self.monitor = ResourceMonitor(self.limits)
    
    def execute_code(self, context: ExecutionContext) -> ExecutionResult:
        """Execute code in safe environment."""
        # Validate code first
        validation = self.validator.validate_code(context.code)
        if not validation["is_safe"]:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security validation failed: {'; '.join(validation['issues'])}",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                security_violation=True
            )
        
        # Create temporary execution environment
        with self._create_sandbox() as sandbox_dir:
            return self._execute_in_sandbox(context, sandbox_dir)
    
    @contextmanager
    def _create_sandbox(self):
        """Create temporary sandbox directory."""
        sandbox_dir = tempfile.mkdtemp(prefix="sanskrit_exec_")
        try:
            # Set restrictive permissions
            os.chmod(sandbox_dir, 0o700)
            
            # Create restricted Python environment
            self._setup_restricted_environment(sandbox_dir)
            
            yield sandbox_dir
        finally:
            # Cleanup sandbox
            try:
                shutil.rmtree(sandbox_dir)
            except OSError as e:
                logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")
    
    def _setup_restricted_environment(self, sandbox_dir: str):
        """Setup restricted Python environment in sandbox."""
        # Create restricted Python script template
        restricted_template = '''
import sys
import os

# Remove dangerous modules from sys.modules
dangerous_modules = {dangerous_modules}
for module in dangerous_modules:
    if module in sys.modules:
        del sys.modules[module]

# Override __import__ to restrict imports
import builtins
original_import = builtins.__import__
allowed_modules = {allowed_modules}

def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in allowed_modules or any(name.startswith(allowed + '.') for allowed in allowed_modules):
        return original_import(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Import of '" + name + "' is not allowed")

builtins.__import__ = restricted_import

# Remove dangerous builtins
dangerous_builtins = ['eval', 'exec', 'compile', 'open', 'file']
for builtin in dangerous_builtins:
    if hasattr(builtins, builtin):
        delattr(builtins, builtin)

# Execute user code will be inserted here
'''
        
        # Save template to sandbox
        template_path = os.path.join(sandbox_dir, "restricted_runner.py")
        with open(template_path, 'w') as f:
            f.write(restricted_template.format(
                dangerous_modules=repr(self.limits.blocked_modules),
                allowed_modules=repr(self.limits.allowed_modules)
            ))
    
    def _execute_in_sandbox(self, context: ExecutionContext, sandbox_dir: str) -> ExecutionResult:
        """Execute code in sandbox directory."""
        start_time = time.time()
        
        # Create execution script with restricted environment
        script_path = os.path.join(sandbox_dir, "user_script.py")
        
        # Create restricted execution wrapper
        wrapper_code = f'''
import sys
import os

# Remove dangerous modules from sys.modules
dangerous_modules = {repr(self.limits.blocked_modules)}
for module in dangerous_modules:
    if module in sys.modules:
        del sys.modules[module]

# Override __import__ to restrict imports
import builtins
original_import = builtins.__import__
allowed_modules = {repr(self.limits.allowed_modules)}

def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in allowed_modules or any(name.startswith(allowed + '.') for allowed in allowed_modules):
        return original_import(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Import of '{{name}}' is not allowed")

builtins.__import__ = restricted_import

# Remove dangerous builtins
dangerous_builtins = ['eval', 'exec', 'compile', 'open', 'file']
for builtin in dangerous_builtins:
    if hasattr(builtins, builtin):
        delattr(builtins, builtin)

# Execute user code
try:
{chr(10).join("    " + line for line in context.code.split(chr(10)))}
except Exception as e:
    print(f"Execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
        
        with open(script_path, 'w') as f:
            f.write(wrapper_code)
        
        # Prepare execution command
        python_executable = sys.executable
        cmd = [python_executable, script_path]
        
        # Set up environment
        env = os.environ.copy()
        env.update(context.environment_vars)
        
        # Limit resources using resource module (Unix only)
        def set_limits():
            if not RESOURCE_AVAILABLE:
                return
            if hasattr(resource, 'RLIMIT_AS'):
                # Memory limit
                memory_limit = self.limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            if hasattr(resource, 'RLIMIT_CPU'):
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (self.limits.max_execution_time, self.limits.max_execution_time))
            
            if hasattr(resource, 'RLIMIT_NOFILE'):
                # File descriptor limit
                resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits.max_open_files, self.limits.max_open_files))
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=sandbox_dir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=set_limits if (os.name != 'nt' and RESOURCE_AVAILABLE) else None
            )
            
            # Start monitoring
            if PSUTIL_AVAILABLE:
                try:
                    ps_process = psutil.Process(process.pid)
                    self.monitor.start_monitoring(ps_process)
                except psutil.NoSuchProcess:
                    pass
            
            # Execute with timeout
            try:
                stdout, stderr = process.communicate(
                    input=context.input_data,
                    timeout=self.limits.max_execution_time
                )
                timeout = False
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                timeout = True
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            execution_time = time.time() - start_time
            
            # Check for resource violations
            resource_exceeded = len(self.monitor.stats["resource_violations"]) > 0
            
            return ExecutionResult(
                success=process.returncode == 0 and not timeout and not resource_exceeded,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                memory_used_mb=self.monitor.stats["max_memory_mb"],
                cpu_percent=self.monitor.stats["max_cpu_percent"],
                exit_code=process.returncode,
                timeout=timeout,
                resource_exceeded=resource_exceeded,
                security_violation=False
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {e}",
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                cpu_percent=0,
                security_violation=True
            )


class CodeExecutionManager:
    """Manages safe code execution with audit logging."""
    
    def __init__(self, limits: ExecutionLimits = None, audit_log_path: Optional[str] = None):
        self.limits = limits or ExecutionLimits()
        self.executor = SafeExecutionEnvironment(self.limits)
        self.audit_log_path = audit_log_path
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute_code(self, code: str, language: str = "python", 
                    user_id: Optional[str] = None, **kwargs) -> ExecutionResult:
        """Execute code with full audit logging."""
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = datetime.now()
        
        # Create execution context
        context = ExecutionContext(
            code=code,
            language=language,
            **kwargs
        )
        
        # Log execution start
        self._log_execution_start(execution_id, context, user_id)
        
        try:
            # Execute code
            result = self.executor.execute_code(context)
            
            # Log execution result
            self._log_execution_result(execution_id, result, user_id)
            
            return result
            
        except Exception as e:
            # Log execution error
            error_result = ExecutionResult(
                success=False,
                output="",
                error=f"Execution manager error: {e}",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                security_violation=True
            )
            self._log_execution_result(execution_id, error_result, user_id)
            return error_result
    
    def _log_execution_start(self, execution_id: str, context: ExecutionContext, user_id: Optional[str]):
        """Log execution start."""
        log_entry = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "event": "execution_start",
            "user_id": user_id,
            "language": context.language,
            "code_hash": hashlib.sha256(context.code.encode()).hexdigest(),
            "code_length": len(context.code)
        }
        
        self.execution_history.append(log_entry)
        self._write_audit_log(log_entry)
    
    def _log_execution_result(self, execution_id: str, result: ExecutionResult, user_id: Optional[str]):
        """Log execution result."""
        log_entry = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "event": "execution_complete",
            "user_id": user_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "memory_used_mb": result.memory_used_mb,
            "cpu_percent": result.cpu_percent,
            "timeout": result.timeout,
            "resource_exceeded": result.resource_exceeded,
            "security_violation": result.security_violation,
            "exit_code": result.exit_code,
            "output_length": len(result.output),
            "error_length": len(result.error)
        }
        
        self.execution_history.append(log_entry)
        self._write_audit_log(log_entry)
    
    def _write_audit_log(self, log_entry: Dict[str, Any]):
        """Write audit log entry to file."""
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except OSError as e:
                logger.error(f"Failed to write execution audit log: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len([e for e in self.execution_history if e["event"] == "execution_complete"])
        successful_executions = len([e for e in self.execution_history if e["event"] == "execution_complete" and e["success"]])
        security_violations = len([e for e in self.execution_history if e["event"] == "execution_complete" and e["security_violation"]])
        timeouts = len([e for e in self.execution_history if e["event"] == "execution_complete" and e["timeout"]])
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "security_violations": security_violations,
            "timeouts": timeouts,
            "recent_executions": self.execution_history[-10:]
        }


# Import hashlib for code hashing
import hashlib


def create_safe_execution_manager(workspace_path: str, 
                                 limits: ExecutionLimits = None) -> CodeExecutionManager:
    """Create a safe execution manager for the workspace."""
    if limits is None:
        limits = ExecutionLimits(
            max_execution_time=30,
            max_memory_mb=256,
            max_cpu_percent=50,
            max_file_size_mb=10,
            max_open_files=50,
            max_processes=3
        )
    
    audit_log_path = os.path.join(workspace_path, "execution_audit.log")
    return CodeExecutionManager(limits, audit_log_path)


if __name__ == "__main__":
    # Example usage
    manager = create_safe_execution_manager(".")
    
    # Test safe code
    safe_code = """
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
    
    result = manager.execute_code(safe_code)
    print(f"Execution result: {result}")
    
    # Test unsafe code
    unsafe_code = """
import os
os.system("ls -la")
"""
    
    result = manager.execute_code(unsafe_code)
    print(f"Unsafe execution result: {result}")