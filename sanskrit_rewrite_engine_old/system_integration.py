"""
System Integration Module for Sanskrit Rewrite Engine

This module provides comprehensive system integration, bringing together all components
into a cohesive Sanskrit reasoning system with end-to-end workflows, health monitoring,
logging, and deployment automation.

Requirements: All requirements final validation
"""

import os
import sys
import json
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Core system imports
try:
    from .tokenizer import SanskritTokenizer
    from .panini_engine import PaniniRuleEngine, PaniniEngineResult
    from .essential_sutras import create_essential_sutras, create_essential_paribhasas
    from .morphological_analyzer import SanskritMorphologicalAnalyzer
    from .syntax_tree import SyntaxTreeBuilder
    from .semantic_graph import SemanticGraphBuilder
    from .semantic_pipeline import SemanticProcessor, process_sanskrit_text
    from .reasoning_core import ReasoningCore
    from .symbolic_computation import SymbolicComputationEngine
    from .hybrid_reasoning import HybridSanskritReasoner
    from .multi_domain_mapper import MultiDomainMapper
    from .r_zero_integration import RZeroIntegration
    from .safe_execution import CodeExecutionManager, create_safe_execution_manager
    from .mcp_server import SanskritMCPServer, SecurityConfig, WorkspaceConfig
    from .api_server import app as fastapi_app
    from .performance_optimization import PerformanceOptimizer
    from .gpu_acceleration import GPUDeviceManager, MixedPrecisionManager
    from .security_integration import IntegratedSecurityManager
    from .fallback_mechanisms import FallbackManager
except ImportError as e:
    print(f"Warning: Some components not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sanskrit_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"

class ComponentStatus(Enum):
    """Component status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    active_connections: int = 0
    requests_per_minute: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    component_statuses: Dict[str, ComponentStatus] = field(default_factory=dict)

@dataclass
class WorkflowResult:
    """Result of an end-to-end workflow execution."""
    workflow_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    execution_time: float
    components_used: List[str]
    trace_data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)

class SanskritSystemIntegrator:
    """
    Main system integrator that coordinates all Sanskrit reasoning components
    into a cohesive system with comprehensive monitoring and management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system integrator."""
        self.config_path = config_path or "system_config.json"
        self.config = self._load_config()
        
        # System state
        self.status = SystemStatus.INITIALIZING
        self.components: Dict[str, Any] = {}
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring and logging
        self.audit_logger = self._setup_audit_logging()
        self.health_monitor = None
        self.metrics_collector = None
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get("max_threads", 10))
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.get("max_processes", 4))
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        default_config = {
            "max_threads": 10,
            "max_processes": 4,
            "health_check_interval": 30,
            "metrics_retention_hours": 24,
            "auto_scaling": True,
            "gpu_acceleration": True,
            "security_level": "high",
            "logging_level": "INFO",
            "workspace_root": str(Path.cwd()),
            "temp_directory": str(Path.cwd() / "temp"),
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "allowed_extensions": [".py", ".txt", ".md", ".json", ".yaml"],
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "enable_websockets": True,
            "enable_streaming": True
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup comprehensive audit logging."""
        audit_logger = logging.getLogger("sanskrit_audit")
        audit_logger.setLevel(logging.INFO)
        
        # Create audit log handler
        audit_handler = logging.FileHandler("sanskrit_audit.log")
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        
        return audit_logger
    
    def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing Sanskrit system components...")
        
        try:
            # Core linguistic components
            self._init_linguistic_core()
            
            # Reasoning and computation components
            self._init_reasoning_core()
            
            # Integration and security components
            self._init_integration_layer()
            
            # Performance and monitoring components
            self._init_monitoring_layer()
            
            self.status = SystemStatus.HEALTHY
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = SystemStatus.CRITICAL
            raise
    
    def _init_linguistic_core(self):
        """Initialize linguistic processing components."""
        try:
            # Tokenizer
            self.components['tokenizer'] = SanskritTokenizer()
            self.component_statuses['tokenizer'] = ComponentStatus.ACTIVE
            
            # Panini rule engine
            rule_engine = PaniniRuleEngine()
            rule_engine.load_rules(create_essential_sutras())
            rule_engine.load_paribhasas(create_essential_paribhasas())
            self.components['rule_engine'] = rule_engine
            self.component_statuses['rule_engine'] = ComponentStatus.ACTIVE
            
            # Morphological analyzer
            self.components['morphological_analyzer'] = SanskritMorphologicalAnalyzer()
            self.component_statuses['morphological_analyzer'] = ComponentStatus.ACTIVE
            
            # Syntax tree builder
            self.components['syntax_builder'] = SyntaxTreeBuilder()
            self.component_statuses['syntax_builder'] = ComponentStatus.ACTIVE
            
            # Semantic graph builder
            self.components['semantic_builder'] = SemanticGraphBuilder()
            self.component_statuses['semantic_builder'] = ComponentStatus.ACTIVE
            
            # Semantic processor
            self.components['semantic_processor'] = SemanticProcessor(
                tokenizer=self.components['tokenizer'],
                rule_engine=self.components['rule_engine'],
                morphological_analyzer=self.components['morphological_analyzer'],
                syntax_builder=self.components['syntax_builder'],
                semantic_builder=self.components['semantic_builder']
            )
            self.component_statuses['semantic_processor'] = ComponentStatus.ACTIVE
            
            logger.info("Linguistic core components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize linguistic core: {e}")
            self.component_statuses.update({
                'tokenizer': ComponentStatus.ERROR,
                'rule_engine': ComponentStatus.ERROR,
                'morphological_analyzer': ComponentStatus.ERROR,
                'syntax_builder': ComponentStatus.ERROR,
                'semantic_builder': ComponentStatus.ERROR,
                'semantic_processor': ComponentStatus.ERROR
            })
            raise
    
    def _init_reasoning_core(self):
        """Initialize reasoning and computation components."""
        try:
            # Reasoning core
            self.components['reasoning_core'] = ReasoningCore()
            self.component_statuses['reasoning_core'] = ComponentStatus.ACTIVE
            
            # Symbolic computation engine
            self.components['symbolic_engine'] = SymbolicComputationEngine()
            self.component_statuses['symbolic_engine'] = ComponentStatus.ACTIVE
            
            # Hybrid reasoner
            self.components['hybrid_reasoner'] = HybridSanskritReasoner()
            self.component_statuses['hybrid_reasoner'] = ComponentStatus.ACTIVE
            
            # Multi-domain mapper
            self.components['domain_mapper'] = MultiDomainMapper()
            self.component_statuses['domain_mapper'] = ComponentStatus.ACTIVE
            
            # R-Zero integration
            try:
                self.components['r_zero'] = RZeroIntegration()
                self.component_statuses['r_zero'] = ComponentStatus.ACTIVE
            except Exception as e:
                logger.warning(f"R-Zero integration not available: {e}")
                self.component_statuses['r_zero'] = ComponentStatus.INACTIVE
            
            logger.info("Reasoning core components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoning core: {e}")
            self.component_statuses.update({
                'reasoning_core': ComponentStatus.ERROR,
                'symbolic_engine': ComponentStatus.ERROR,
                'hybrid_reasoner': ComponentStatus.ERROR,
                'domain_mapper': ComponentStatus.ERROR
            })
            raise
    
    def _init_integration_layer(self):
        """Initialize integration and security components."""
        try:
            # Safe execution manager
            self.components['execution_manager'] = create_safe_execution_manager(
                self.config['workspace_root']
            )
            self.component_statuses['execution_manager'] = ComponentStatus.ACTIVE
            
            # Security configuration
            security_config = SecurityConfig(
                allowed_directories=[Path(self.config['workspace_root'])],
                max_file_size=self.config['max_file_size'],
                allowed_extensions=set(self.config['allowed_extensions'])
            )
            
            workspace_config = WorkspaceConfig(
                workspace_root=Path(self.config['workspace_root']),
                temp_directory=Path(self.config['temp_directory'])
            )
            
            # MCP server
            self.components['mcp_server'] = SanskritMCPServer(
                security_config, workspace_config
            )
            self.component_statuses['mcp_server'] = ComponentStatus.ACTIVE
            
            # Integrated security manager
            self.components['security_manager'] = IntegratedSecurityManager()
            self.component_statuses['security_manager'] = ComponentStatus.ACTIVE
            
            # Fallback manager
            self.components['fallback_manager'] = FallbackManager()
            self.component_statuses['fallback_manager'] = ComponentStatus.ACTIVE
            
            logger.info("Integration layer components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize integration layer: {e}")
            self.component_statuses.update({
                'execution_manager': ComponentStatus.ERROR,
                'mcp_server': ComponentStatus.ERROR,
                'security_manager': ComponentStatus.ERROR,
                'fallback_manager': ComponentStatus.ERROR
            })
            raise
    
    def _init_monitoring_layer(self):
        """Initialize monitoring and performance components."""
        try:
            # GPU device manager (if available)
            if self.config.get('gpu_acceleration', True):
                try:
                    self.components['gpu_manager'] = GPUDeviceManager()
                    self.components['precision_manager'] = MixedPrecisionManager()
                    self.component_statuses['gpu_manager'] = ComponentStatus.ACTIVE
                    self.component_statuses['precision_manager'] = ComponentStatus.ACTIVE
                except Exception as e:
                    logger.warning(f"GPU acceleration not available: {e}")
                    self.component_statuses['gpu_manager'] = ComponentStatus.INACTIVE
                    self.component_statuses['precision_manager'] = ComponentStatus.INACTIVE
            
            # Performance optimizer
            self.components['performance_optimizer'] = PerformanceOptimizer()
            self.component_statuses['performance_optimizer'] = ComponentStatus.ACTIVE
            
            # Start health monitoring
            self._start_health_monitoring()
            
            logger.info("Monitoring layer components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring layer: {e}")
            self.component_statuses.update({
                'performance_optimizer': ComponentStatus.ERROR
            })
            raise
    
    def _start_health_monitoring(self):
        """Start system health monitoring."""
        def health_check_loop():
            while self.status != SystemStatus.SHUTDOWN:
                try:
                    metrics = self._collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent metrics
                    cutoff_time = datetime.now() - timedelta(
                        hours=self.config.get('metrics_retention_hours', 24)
                    )
                    self.metrics_history = [
                        m for m in self.metrics_history if m.timestamp > cutoff_time
                    ]
                    
                    # Check system health
                    self._check_system_health(metrics)
                    
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                
                time.sleep(self.config.get('health_check_interval', 30))
        
        self.health_monitor = threading.Thread(target=health_check_loop, daemon=True)
        self.health_monitor.start()
        logger.info("Health monitoring started")
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics (if available)
            gpu_usage = 0.0
            if 'gpu_manager' in self.components:
                try:
                    gpu_usage = self.components['gpu_manager'].get_gpu_utilization()
                except:
                    pass
            
            # Component statuses
            component_statuses = self.component_statuses.copy()
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                component_statuses=component_statuses
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return SystemMetrics()
    
    def _check_system_health(self, metrics: SystemMetrics):
        """Check system health and update status."""
        try:
            # Check resource usage
            if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                if self.status == SystemStatus.HEALTHY:
                    self.status = SystemStatus.DEGRADED
                    logger.warning("System performance degraded due to high resource usage")
            
            # Check component health
            error_components = [
                name for name, status in metrics.component_statuses.items()
                if status == ComponentStatus.ERROR
            ]
            
            if error_components:
                if len(error_components) > len(metrics.component_statuses) / 2:
                    self.status = SystemStatus.CRITICAL
                    logger.error(f"System critical: Multiple component failures: {error_components}")
                elif self.status == SystemStatus.HEALTHY:
                    self.status = SystemStatus.DEGRADED
                    logger.warning(f"System degraded: Component failures: {error_components}")
            elif self.status == SystemStatus.DEGRADED:
                # Check if we can recover
                if metrics.cpu_usage < 70 and metrics.memory_usage < 70:
                    self.status = SystemStatus.HEALTHY
                    logger.info("System health recovered")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def execute_workflow(self, workflow_type: str, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute an end-to-end workflow."""
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.audit_logger.info(f"Starting workflow {workflow_id}: {workflow_type}")
        
        try:
            self.active_workflows[workflow_id] = {
                'type': workflow_type,
                'start_time': start_time,
                'input_data': input_data
            }
            
            if workflow_type == "sanskrit_to_code":
                result = await self._execute_sanskrit_to_code_workflow(input_data)
            elif workflow_type == "text_analysis":
                result = await self._execute_text_analysis_workflow(input_data)
            elif workflow_type == "reasoning_query":
                result = await self._execute_reasoning_workflow(input_data)
            elif workflow_type == "mathematical_computation":
                result = await self._execute_math_workflow(input_data)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            execution_time = time.time() - start_time
            
            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                input_data=input_data,
                output_data=result,
                success=True,
                execution_time=execution_time,
                components_used=result.get('components_used', []),
                trace_data=result.get('trace_data', {})
            )
            
            self.audit_logger.info(f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Workflow {workflow_id} failed: {str(e)}"
            logger.error(error_msg)
            self.audit_logger.error(error_msg)
            
            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                input_data=input_data,
                output_data={},
                success=False,
                execution_time=execution_time,
                components_used=[],
                trace_data={'error': str(e), 'traceback': traceback.format_exc()},
                errors=[str(e)]
            )
        
        finally:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return workflow_result
    
    async def _execute_sanskrit_to_code_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Sanskrit text to code generation workflow."""
        sanskrit_text = input_data.get('text', '')
        target_language = input_data.get('target_language', 'python')
        
        components_used = []
        
        # Step 1: Process Sanskrit text
        semantic_result = await process_sanskrit_text(
            text=sanskrit_text,
            tokenizer=self.components['tokenizer'],
            rule_engine=self.components['rule_engine'],
            enable_tracing=True
        )
        components_used.extend(['tokenizer', 'rule_engine', 'semantic_processor'])
        
        # Step 2: Extract semantic concepts
        concepts = self.components['semantic_processor'].extract_semantic_concepts(
            semantic_result.semantic_graph
        )
        
        # Step 3: Map to target domain
        code_mapping = self.components['domain_mapper'].map_to_programming_domain(
            concepts, target_language
        )
        components_used.append('domain_mapper')
        
        # Step 4: Generate code
        if 'hybrid_reasoner' in self.components:
            code_result = await self.components['hybrid_reasoner'].generate_code(
                semantic_concepts=concepts,
                target_language=target_language,
                code_mapping=code_mapping
            )
            components_used.append('hybrid_reasoner')
        else:
            # Fallback code generation
            code_result = self.components['fallback_manager'].generate_fallback_code(
                concepts, target_language
            )
            components_used.append('fallback_manager')
        
        # Step 5: Validate and execute (if requested)
        if input_data.get('execute_code', False):
            execution_result = self.components['execution_manager'].execute_code(
                code=code_result['code'],
                language=target_language,
                user_id=input_data.get('user_id', 'system')
            )
            components_used.append('execution_manager')
        else:
            execution_result = None
        
        return {
            'sanskrit_analysis': {
                'tokens': [str(t) for t in semantic_result.tokens],
                'transformations': semantic_result.transformations,
                'semantic_graph': semantic_result.semantic_graph
            },
            'semantic_concepts': concepts,
            'code_mapping': code_mapping,
            'generated_code': code_result,
            'execution_result': execution_result.to_dict() if execution_result else None,
            'components_used': components_used,
            'trace_data': {
                'semantic_trace': semantic_result.trace_data,
                'mapping_trace': code_mapping.get('trace', {}),
                'generation_trace': code_result.get('trace', {})
            }
        }
    
    async def _execute_text_analysis_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive Sanskrit text analysis workflow."""
        text = input_data.get('text', '')
        analysis_depth = input_data.get('depth', 'full')  # minimal, standard, full
        
        components_used = []
        
        # Step 1: Tokenization and basic processing
        semantic_result = await process_sanskrit_text(
            text=text,
            tokenizer=self.components['tokenizer'],
            rule_engine=self.components['rule_engine'],
            enable_tracing=True
        )
        components_used.extend(['tokenizer', 'rule_engine', 'semantic_processor'])
        
        # Step 2: Morphological analysis
        morphological_analysis = self.components['morphological_analyzer'].analyze_text(text)
        components_used.append('morphological_analyzer')
        
        # Step 3: Syntax analysis (if requested)
        syntax_tree = None
        if analysis_depth in ['standard', 'full']:
            syntax_tree = self.components['syntax_builder'].build_tree(
                semantic_result.tokens, morphological_analysis
            )
            components_used.append('syntax_builder')
        
        # Step 4: Semantic graph construction (if requested)
        semantic_graph = None
        if analysis_depth == 'full':
            semantic_graph = self.components['semantic_builder'].build_graph(
                syntax_tree, morphological_analysis
            )
            components_used.append('semantic_builder')
        
        return {
            'input_text': text,
            'tokenization': {
                'tokens': [str(t) for t in semantic_result.tokens],
                'transformations': semantic_result.transformations
            },
            'morphological_analysis': morphological_analysis.to_dict() if morphological_analysis else None,
            'syntax_tree': syntax_tree.to_dict() if syntax_tree else None,
            'semantic_graph': semantic_graph.to_dict() if semantic_graph else None,
            'components_used': components_used,
            'trace_data': semantic_result.trace_data
        }
    
    async def _execute_reasoning_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning query workflow."""
        query = input_data.get('query', '')
        context = input_data.get('context', {})
        
        components_used = []
        
        # Step 1: Process query through reasoning core
        reasoning_result = await self.components['reasoning_core'].process_query(
            query=query,
            context=context
        )
        components_used.append('reasoning_core')
        
        # Step 2: Enhance with hybrid reasoning (if available)
        if 'hybrid_reasoner' in self.components:
            enhanced_result = await self.components['hybrid_reasoner'].enhance_reasoning(
                base_result=reasoning_result,
                query=query,
                context=context
            )
            components_used.append('hybrid_reasoner')
        else:
            enhanced_result = reasoning_result
        
        # Step 3: Apply symbolic computation (if mathematical)
        if self._is_mathematical_query(query):
            symbolic_result = self.components['symbolic_engine'].process_mathematical_query(
                query, enhanced_result
            )
            components_used.append('symbolic_engine')
        else:
            symbolic_result = None
        
        return {
            'query': query,
            'reasoning_result': reasoning_result,
            'enhanced_result': enhanced_result,
            'symbolic_result': symbolic_result,
            'components_used': components_used,
            'trace_data': {
                'reasoning_trace': getattr(reasoning_result, 'trace_data', {}),
                'enhancement_trace': getattr(enhanced_result, 'trace_data', {}),
                'symbolic_trace': getattr(symbolic_result, 'trace_data', {}) if symbolic_result else {}
            }
        }
    
    async def _execute_math_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mathematical computation workflow."""
        expression = input_data.get('expression', '')
        domain = input_data.get('domain', 'general')
        
        components_used = []
        
        # Step 1: Parse mathematical expression
        parsed_expr = self.components['symbolic_engine'].parse_expression(expression)
        components_used.append('symbolic_engine')
        
        # Step 2: Apply Vedic mathematics (if applicable)
        vedic_result = self.components['symbolic_engine'].apply_vedic_mathematics(
            parsed_expr, domain
        )
        
        # Step 3: Perform symbolic computation
        computation_result = self.components['symbolic_engine'].compute(
            parsed_expr, vedic_optimizations=vedic_result
        )
        
        # Step 4: Verify result (if requested)
        verification_result = None
        if input_data.get('verify', False):
            verification_result = self.components['symbolic_engine'].verify_result(
                expression, computation_result
            )
        
        return {
            'expression': expression,
            'parsed_expression': str(parsed_expr),
            'vedic_optimizations': vedic_result,
            'computation_result': computation_result,
            'verification_result': verification_result,
            'components_used': components_used,
            'trace_data': {
                'parsing_trace': getattr(parsed_expr, 'trace_data', {}),
                'computation_trace': getattr(computation_result, 'trace_data', {}),
                'verification_trace': getattr(verification_result, 'trace_data', {}) if verification_result else {}
            }
        }
    
    def _is_mathematical_query(self, query: str) -> bool:
        """Check if a query is mathematical in nature."""
        math_keywords = [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'algebra', 'geometry', 'arithmetic', 'mathematics',
            '+', '-', '*', '/', '=', 'x', 'y', 'z'
        ]
        return any(keyword in query.lower() for keyword in math_keywords)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()
        
        return {
            'status': self.status.value,
            'timestamp': datetime.now().isoformat(),
            'components': {
                name: status.value for name, status in self.component_statuses.items()
            },
            'metrics': {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'gpu_usage': latest_metrics.gpu_usage,
                'active_connections': latest_metrics.active_connections,
                'requests_per_minute': latest_metrics.requests_per_minute,
                'average_response_time': latest_metrics.average_response_time,
                'error_rate': latest_metrics.error_rate
            },
            'active_workflows': len(self.active_workflows),
            'uptime': self._get_uptime()
        }
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        # This is a simplified implementation
        # In a real system, you'd track the actual start time
        return "System uptime tracking not implemented"
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Initiating system shutdown...")
        self.status = SystemStatus.SHUTDOWN
        
        # Wait for active workflows to complete (with timeout)
        shutdown_timeout = 30  # seconds
        start_time = time.time()
        
        while self.active_workflows and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self.active_workflows)} workflows to complete...")
            await asyncio.sleep(1)
        
        # Force shutdown remaining workflows
        if self.active_workflows:
            logger.warning(f"Force shutting down {len(self.active_workflows)} remaining workflows")
            self.active_workflows.clear()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Cleanup components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'cleanup'):
                    await component.cleanup()
            except Exception as e:
                logger.error(f"Error shutting down component {name}: {e}")
        
        logger.info("System shutdown completed")

# Global system integrator instance
_system_integrator: Optional[SanskritSystemIntegrator] = None

def get_system_integrator(config_path: Optional[str] = None) -> SanskritSystemIntegrator:
    """Get or create the global system integrator instance."""
    global _system_integrator
    if _system_integrator is None:
        _system_integrator = SanskritSystemIntegrator(config_path)
    return _system_integrator

async def initialize_system(config_path: Optional[str] = None) -> SanskritSystemIntegrator:
    """Initialize the complete Sanskrit reasoning system."""
    integrator = get_system_integrator(config_path)
    logger.info("Sanskrit reasoning system initialized successfully")
    return integrator

async def shutdown_system():
    """Shutdown the complete Sanskrit reasoning system."""
    global _system_integrator
    if _system_integrator:
        await _system_integrator.shutdown()
        _system_integrator = None

# Convenience functions for common workflows
async def process_sanskrit_to_code(
    text: str,
    target_language: str = 'python',
    execute_code: bool = False,
    user_id: str = 'system'
) -> WorkflowResult:
    """Process Sanskrit text and generate code."""
    integrator = get_system_integrator()
    return await integrator.execute_workflow('sanskrit_to_code', {
        'text': text,
        'target_language': target_language,
        'execute_code': execute_code,
        'user_id': user_id
    })

async def analyze_sanskrit_text(
    text: str,
    depth: str = 'full'
) -> WorkflowResult:
    """Perform comprehensive Sanskrit text analysis."""
    integrator = get_system_integrator()
    return await integrator.execute_workflow('text_analysis', {
        'text': text,
        'depth': depth
    })

async def process_reasoning_query(
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> WorkflowResult:
    """Process a reasoning query."""
    integrator = get_system_integrator()
    return await integrator.execute_workflow('reasoning_query', {
        'query': query,
        'context': context or {}
    })

async def compute_mathematical_expression(
    expression: str,
    domain: str = 'general',
    verify: bool = False
) -> WorkflowResult:
    """Compute a mathematical expression."""
    integrator = get_system_integrator()
    return await integrator.execute_workflow('mathematical_computation', {
        'expression': expression,
        'domain': domain,
        'verify': verify
    })