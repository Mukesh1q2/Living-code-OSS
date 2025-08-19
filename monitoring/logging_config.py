"""
Centralized logging configuration for Vidya Quantum Interface
"""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from config import config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, 'processing_time') or 'performance' in record.getMessage().lower()


class SecurityFilter(logging.Filter):
    """Filter for security-related logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        security_keywords = ['auth', 'login', 'security', 'unauthorized', 'forbidden']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in security_keywords)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration based on environment"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Base configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            },
            'json': {
                '()': JSONFormatter,
            },
        },
        'filters': {
            'performance': {
                '()': PerformanceFilter,
            },
            'security': {
                '()': SecurityFilter,
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout,
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': logs_dir / 'vidya.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': logs_dir / 'vidya_errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
            },
            'performance_file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'json',
                'filename': logs_dir / 'vidya_performance.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 3,
                'filters': ['performance'],
            },
            'security_file': {
                'level': 'WARNING',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'json',
                'filename': logs_dir / 'vidya_security.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'filters': ['security'],
            },
        },
        'loggers': {
            'vidya_quantum_interface': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file', 'performance_file', 'security_file'],
                'propagate': False,
            },
            'sanskrit_rewrite_engine': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'fastapi': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
        },
    }
    
    # Environment-specific adjustments
    if config.environment == 'development':
        logging_config['handlers']['console']['level'] = 'DEBUG'
        logging_config['loggers']['vidya_quantum_interface']['level'] = 'DEBUG'
    
    elif config.environment == 'production':
        # Add structured logging for production
        logging_config['handlers']['json_file'] = {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': logs_dir / 'vidya_structured.log',
            'maxBytes': 52428800,  # 50MB
            'backupCount': 10,
        }
        
        # Add CloudWatch handler if on AWS
        if hasattr(config, 'aws_cloudwatch_log_group'):
            try:
                import watchtower
                logging_config['handlers']['cloudwatch'] = {
                    'level': 'INFO',
                    'class': 'watchtower.CloudWatchLogsHandler',
                    'formatter': 'json',
                    'log_group': config.aws_cloudwatch_log_group,
                    'stream_name': 'vidya-backend',
                }
                logging_config['loggers']['vidya_quantum_interface']['handlers'].append('cloudwatch')
            except ImportError:
                pass
        
        # Reduce console logging in production
        logging_config['handlers']['console']['level'] = 'WARNING'
    
    return logging_config


def setup_logging():
    """Setup logging configuration"""
    logging_config = get_logging_config()
    logging.config.dictConfig(logging_config)
    
    # Set up performance logging
    performance_logger = logging.getLogger('vidya_quantum_interface.performance')
    
    # Set up security logging
    security_logger = logging.getLogger('vidya_quantum_interface.security')
    
    return logging.getLogger('vidya_quantum_interface')


class LoggingMiddleware:
    """Middleware for request/response logging"""
    
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger('vidya_quantum_interface.requests')
    
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            start_time = datetime.utcnow()
            
            # Log request
            self.logger.info(
                "Request started",
                extra={
                    'method': scope['method'],
                    'path': scope['path'],
                    'query_string': scope['query_string'].decode(),
                    'client': scope.get('client'),
                    'request_id': id(scope),
                }
            )
            
            # Process request
            await self.app(scope, receive, send)
            
            # Log response (simplified - would need response capture in real implementation)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "Request completed",
                extra={
                    'method': scope['method'],
                    'path': scope['path'],
                    'processing_time': processing_time,
                    'request_id': id(scope),
                }
            )
        else:
            await self.app(scope, receive, send)


# Performance monitoring utilities
class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger('vidya_quantum_interface.performance')
    
    def log_processing_time(self, operation: str, duration: float, **kwargs):
        """Log processing time for operations"""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'operation': operation,
                'processing_time': duration,
                **kwargs
            }
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        self.logger.info(
            f"Memory usage: {operation}",
            extra={
                'operation': operation,
                'memory_mb': memory_mb,
                **kwargs
            }
        )
    
    def log_quantum_effect_performance(self, effect: str, fps: float, **kwargs):
        """Log quantum effect performance"""
        self.logger.info(
            f"Quantum effect performance: {effect}",
            extra={
                'effect': effect,
                'fps': fps,
                **kwargs
            }
        )


# Security monitoring utilities
class SecurityMonitor:
    """Security monitoring utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger('vidya_quantum_interface.security')
    
    def log_authentication_attempt(self, user_id: str, success: bool, **kwargs):
        """Log authentication attempts"""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"Authentication {'successful' if success else 'failed'}",
            extra={
                'user_id': user_id,
                'success': success,
                **kwargs
            }
        )
    
    def log_unauthorized_access(self, path: str, client_ip: str, **kwargs):
        """Log unauthorized access attempts"""
        self.logger.warning(
            "Unauthorized access attempt",
            extra={
                'path': path,
                'client_ip': client_ip,
                **kwargs
            }
        )
    
    def log_rate_limit_exceeded(self, client_ip: str, endpoint: str, **kwargs):
        """Log rate limit violations"""
        self.logger.warning(
            "Rate limit exceeded",
            extra={
                'client_ip': client_ip,
                'endpoint': endpoint,
                **kwargs
            }
        )


# Global instances
performance_monitor = PerformanceMonitor()
security_monitor = SecurityMonitor()