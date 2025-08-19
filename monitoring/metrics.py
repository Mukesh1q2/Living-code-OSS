"""
Metrics collection and monitoring for Vidya Quantum Interface
"""
import time
import psutil
import asyncio
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from functools import wraps
import logging

from config import config

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'vidya_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'vidya_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'vidya_active_websocket_connections',
    'Number of active WebSocket connections'
)

SANSKRIT_PROCESSING_TIME = Histogram(
    'vidya_sanskrit_processing_seconds',
    'Sanskrit text processing time',
    ['operation_type']
)

QUANTUM_EFFECT_FPS = Gauge(
    'vidya_quantum_effect_fps',
    'Quantum effect rendering FPS',
    ['effect_type']
)

AI_MODEL_INFERENCE_TIME = Histogram(
    'vidya_ai_inference_seconds',
    'AI model inference time',
    ['model_name', 'model_type']
)

MEMORY_USAGE = Gauge(
    'vidya_memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

CPU_USAGE = Gauge(
    'vidya_cpu_usage_percent',
    'CPU usage percentage',
    ['component']
)

ERROR_COUNT = Counter(
    'vidya_errors_total',
    'Total errors',
    ['error_type', 'component']
)

CACHE_OPERATIONS = Counter(
    'vidya_cache_operations_total',
    'Cache operations',
    ['operation', 'result']
)

USER_INTERACTIONS = Counter(
    'vidya_user_interactions_total',
    'User interactions',
    ['interaction_type']
)

CONSCIOUSNESS_STATE_CHANGES = Counter(
    'vidya_consciousness_state_changes_total',
    'Vidya consciousness state changes',
    ['from_state', 'to_state']
)

# Application info
APP_INFO = Info(
    'vidya_app_info',
    'Application information'
)


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_monitor_task = None
        
        # Set application info
        APP_INFO.info({
            'version': '1.0.0',
            'environment': config.environment,
            'python_version': '3.11',
        })
    
    async def start_system_monitoring(self):
        """Start system resource monitoring"""
        self.system_monitor_task = asyncio.create_task(self._monitor_system_resources())
    
    async def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_system_resources(self):
        """Monitor system resources periodically"""
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.labels(component='system').set(memory.used)
                MEMORY_USAGE.labels(component='available').set(memory.available)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.labels(component='system').set(cpu_percent)
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                MEMORY_USAGE.labels(component='process').set(process_memory.rss)
                
                process_cpu = process.cpu_percent()
                CPU_USAGE.labels(component='process').set(process_cpu)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_websocket_connection(self, connected: bool):
        """Record WebSocket connection changes"""
        if connected:
            ACTIVE_CONNECTIONS.inc()
        else:
            ACTIVE_CONNECTIONS.dec()
    
    def record_sanskrit_processing(self, operation_type: str, duration: float):
        """Record Sanskrit processing metrics"""
        SANSKRIT_PROCESSING_TIME.labels(operation_type=operation_type).observe(duration)
    
    def record_quantum_effect_fps(self, effect_type: str, fps: float):
        """Record quantum effect performance"""
        QUANTUM_EFFECT_FPS.labels(effect_type=effect_type).set(fps)
    
    def record_ai_inference(self, model_name: str, model_type: str, duration: float):
        """Record AI model inference metrics"""
        AI_MODEL_INFERENCE_TIME.labels(model_name=model_name, model_type=model_type).observe(duration)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation"""
        CACHE_OPERATIONS.labels(operation=operation, result=result).inc()
    
    def record_user_interaction(self, interaction_type: str):
        """Record user interaction"""
        USER_INTERACTIONS.labels(interaction_type=interaction_type).inc()
    
    def record_consciousness_state_change(self, from_state: str, to_state: str):
        """Record Vidya consciousness state change"""
        CONSCIOUSNESS_STATE_CHANGES.labels(from_state=from_state, to_state=to_state).inc()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name == 'sanskrit_processing':
                    operation_type = labels.get('operation_type', 'unknown') if labels else 'unknown'
                    metrics_collector.record_sanskrit_processing(operation_type, duration)
                elif metric_name == 'ai_inference':
                    model_name = labels.get('model_name', 'unknown') if labels else 'unknown'
                    model_type = labels.get('model_type', 'unknown') if labels else 'unknown'
                    metrics_collector.record_ai_inference(model_name, model_type, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name == 'sanskrit_processing':
                    operation_type = labels.get('operation_type', 'unknown') if labels else 'unknown'
                    metrics_collector.record_sanskrit_processing(operation_type, duration)
                elif metric_name == 'ai_inference':
                    model_name = labels.get('model_name', 'unknown') if labels else 'unknown'
                    model_type = labels.get('model_type', 'unknown') if labels else 'unknown'
                    metrics_collector.record_ai_inference(model_name, model_type, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_errors(component: str):
    """Decorator to track errors"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                metrics_collector.record_error(error_type, component)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                metrics_collector.record_error(error_type, component)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class HealthChecker:
    """Health check utilities"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result if isinstance(result, dict) else {}
                }
                
                if not result:
                    results['status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['status'] = 'unhealthy'
        
        return results


# Global health checker
health_checker = HealthChecker()


def start_metrics_server(port: int = 9000):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


# Register default health checks
def check_memory():
    """Check memory usage"""
    memory = psutil.virtual_memory()
    return {
        'available_mb': memory.available / 1024 / 1024,
        'used_percent': memory.percent,
        'healthy': memory.percent < 90
    }

def check_disk():
    """Check disk usage"""
    disk = psutil.disk_usage('/')
    return {
        'free_gb': disk.free / 1024 / 1024 / 1024,
        'used_percent': (disk.used / disk.total) * 100,
        'healthy': (disk.used / disk.total) * 100 < 90
    }

async def check_redis():
    """Check Redis connection"""
    try:
        import redis
        r = redis.from_url(config.redis_url)
        r.ping()
        return {'healthy': True, 'connection': 'ok'}
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

# Register health checks
health_checker.register_check('memory', check_memory)
health_checker.register_check('disk', check_disk)
health_checker.register_check('redis', check_redis)