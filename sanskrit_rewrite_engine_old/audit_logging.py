"""
Comprehensive Audit Logging System for Sanskrit Rewrite Engine

This module provides detailed audit trails for all system operations,
security events, performance metrics, and user activities.

Requirements: All requirements final validation - logging and audit trails
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
from queue import Queue, Empty
import gzip
import shutil

class AuditEventType(Enum):
    """Types of audit events."""
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    FILE_ACCESS = "file_access"
    CODE_EXECUTION = "code_execution"
    RULE_APPLICATION = "rule_application"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"
    PERFORMANCE_METRIC = "performance_metric"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    COMPONENT_STATUS = "component_status"
    CONFIGURATION_CHANGE = "configuration_change"

class AuditLevel(Enum):
    """Audit logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"

@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType = AuditEventType.API_REQUEST
    level: AuditLevel = AuditLevel.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    action: str = ""
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

@dataclass
class AuditConfig:
    """Audit logging configuration."""
    log_directory: str = "audit_logs"
    max_file_size_mb: int = 100
    max_files: int = 30
    compression_enabled: bool = True
    real_time_alerts: bool = True
    security_events_separate: bool = True
    performance_metrics_separate: bool = True
    log_level: AuditLevel = AuditLevel.INFO
    buffer_size: int = 1000
    flush_interval_seconds: int = 5
    include_stack_trace: bool = False
    anonymize_user_data: bool = False

class AuditLogger:
    """
    Comprehensive audit logging system with real-time processing,
    log rotation, compression, and security event monitoring.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize the audit logger."""
        self.config = config or AuditConfig()
        self.log_directory = Path(self.config.log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Event queue for async processing
        self.event_queue = Queue(maxsize=self.config.buffer_size)
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        # Log files
        self.main_log_file = None
        self.security_log_file = None
        self.performance_log_file = None
        
        # Statistics
        self.events_logged = 0
        self.events_dropped = 0
        self.last_flush = datetime.now()
        
        # Initialize logging
        self._setup_log_files()
        self._start_processing_thread()
    
    def _setup_log_files(self):
        """Setup log files with rotation."""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Main audit log
        main_log_path = self.log_directory / f"audit_{timestamp}.log"
        self.main_log_file = open(main_log_path, 'a', encoding='utf-8')
        
        # Security events log (if separate)
        if self.config.security_events_separate:
            security_log_path = self.log_directory / f"security_{timestamp}.log"
            self.security_log_file = open(security_log_path, 'a', encoding='utf-8')
        
        # Performance metrics log (if separate)
        if self.config.performance_metrics_separate:
            performance_log_path = self.log_directory / f"performance_{timestamp}.log"
            self.performance_log_file = open(performance_log_path, 'a', encoding='utf-8')
    
    def _start_processing_thread(self):
        """Start the background processing thread."""
        self.processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_events(self):
        """Background thread to process audit events."""
        while not self.shutdown_event.is_set():
            try:
                # Process events in batches
                events_to_process = []
                
                # Collect events with timeout
                try:
                    # Get first event (blocking with timeout)
                    event = self.event_queue.get(timeout=self.config.flush_interval_seconds)
                    events_to_process.append(event)
                    
                    # Get additional events (non-blocking)
                    while len(events_to_process) < 100:  # Batch size limit
                        try:
                            event = self.event_queue.get_nowait()
                            events_to_process.append(event)
                        except Empty:
                            break
                
                except Empty:
                    # Timeout - check if we need to flush anyway
                    if (datetime.now() - self.last_flush).total_seconds() > self.config.flush_interval_seconds:
                        self._flush_logs()
                    continue
                
                # Process the batch
                for event in events_to_process:
                    self._write_event(event)
                    self.events_logged += 1
                
                # Flush if needed
                if (datetime.now() - self.last_flush).total_seconds() > self.config.flush_interval_seconds:
                    self._flush_logs()
                
            except Exception as e:
                # Log processing error to system log
                logging.getLogger(__name__).error(f"Audit processing error: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write an event to the appropriate log file."""
        try:
            # Check log level
            if not self._should_log_event(event):
                return
            
            # Anonymize if configured
            if self.config.anonymize_user_data:
                event = self._anonymize_event(event)
            
            # Convert to JSON
            log_line = event.to_json() + '\n'
            
            # Write to appropriate log file
            if event.level == AuditLevel.SECURITY and self.security_log_file:
                self.security_log_file.write(log_line)
            elif event.event_type == AuditEventType.PERFORMANCE_METRIC and self.performance_log_file:
                self.performance_log_file.write(log_line)
            else:
                self.main_log_file.write(log_line)
            
            # Real-time alerts for critical events
            if self.config.real_time_alerts and event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL, AuditLevel.SECURITY]:
                self._send_real_time_alert(event)
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to write audit event: {e}")
    
    def _should_log_event(self, event: AuditEvent) -> bool:
        """Check if event should be logged based on level."""
        level_order = {
            AuditLevel.DEBUG: 0,
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2,
            AuditLevel.ERROR: 3,
            AuditLevel.CRITICAL: 4,
            AuditLevel.SECURITY: 5
        }
        
        return level_order.get(event.level, 0) >= level_order.get(self.config.log_level, 1)
    
    def _anonymize_event(self, event: AuditEvent) -> AuditEvent:
        """Anonymize sensitive data in event."""
        # Create a copy to avoid modifying original
        anonymized_event = AuditEvent(**asdict(event))
        
        # Anonymize user ID
        if anonymized_event.user_id:
            anonymized_event.user_id = f"user_{hash(anonymized_event.user_id) % 10000:04d}"
        
        # Anonymize IP address
        if anonymized_event.ip_address:
            parts = anonymized_event.ip_address.split('.')
            if len(parts) == 4:
                anonymized_event.ip_address = f"{parts[0]}.{parts[1]}.xxx.xxx"
        
        # Anonymize sensitive details
        if 'email' in anonymized_event.details:
            email = anonymized_event.details['email']
            if '@' in email:
                local, domain = email.split('@', 1)
                anonymized_event.details['email'] = f"{local[:2]}***@{domain}"
        
        return anonymized_event
    
    def _send_real_time_alert(self, event: AuditEvent):
        """Send real-time alert for critical events."""
        # This is a placeholder - implement actual alerting mechanism
        # (email, Slack, webhook, etc.)
        alert_message = f"ALERT: {event.level.value} - {event.action} - {event.error_message or 'No details'}"
        logging.getLogger("audit_alerts").critical(alert_message)
    
    def _flush_logs(self):
        """Flush all log files."""
        try:
            if self.main_log_file:
                self.main_log_file.flush()
            if self.security_log_file:
                self.security_log_file.flush()
            if self.performance_log_file:
                self.performance_log_file.flush()
            
            self.last_flush = datetime.now()
            
            # Check for log rotation
            self._check_log_rotation()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to flush logs: {e}")
    
    def _check_log_rotation(self):
        """Check if log files need rotation."""
        try:
            max_size = self.config.max_file_size_mb * 1024 * 1024
            
            # Check main log
            if self.main_log_file and os.path.getsize(self.main_log_file.name) > max_size:
                self._rotate_log_file('main')
            
            # Check security log
            if self.security_log_file and os.path.getsize(self.security_log_file.name) > max_size:
                self._rotate_log_file('security')
            
            # Check performance log
            if self.performance_log_file and os.path.getsize(self.performance_log_file.name) > max_size:
                self._rotate_log_file('performance')
            
            # Clean up old files
            self._cleanup_old_logs()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Log rotation failed: {e}")
    
    def _rotate_log_file(self, log_type: str):
        """Rotate a specific log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if log_type == 'main' and self.main_log_file:
            old_path = Path(self.main_log_file.name)
            self.main_log_file.close()
            
            # Compress old file if enabled
            if self.config.compression_enabled:
                compressed_path = old_path.with_suffix('.log.gz')
                with open(old_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                old_path.unlink()
            
            # Create new file
            new_path = self.log_directory / f"audit_{timestamp}.log"
            self.main_log_file = open(new_path, 'a', encoding='utf-8')
        
        # Similar logic for security and performance logs...
    
    def _cleanup_old_logs(self):
        """Clean up old log files beyond retention limit."""
        try:
            # Get all log files sorted by modification time
            log_files = []
            for pattern in ['audit_*.log*', 'security_*.log*', 'performance_*.log*']:
                log_files.extend(self.log_directory.glob(pattern))
            
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove files beyond limit
            if len(log_files) > self.config.max_files:
                for old_file in log_files[self.config.max_files:]:
                    try:
                        old_file.unlink()
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to delete old log {old_file}: {e}")
                        
        except Exception as e:
            logging.getLogger(__name__).error(f"Log cleanup failed: {e}")
    
    def log_event(self, event: AuditEvent):
        """Log an audit event."""
        try:
            self.event_queue.put_nowait(event)
        except:
            # Queue is full - drop event and increment counter
            self.events_dropped += 1
            if self.events_dropped % 100 == 0:  # Log every 100 dropped events
                logging.getLogger(__name__).warning(f"Dropped {self.events_dropped} audit events due to full queue")
    
    def log_system_event(self, action: str, details: Optional[Dict[str, Any]] = None, level: AuditLevel = AuditLevel.INFO):
        """Log a system event."""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_START if 'start' in action.lower() else AuditEventType.SYSTEM_SHUTDOWN,
            level=level,
            component="system",
            action=action,
            details=details or {}
        )
        self.log_event(event)
    
    def log_api_request(self, method: str, endpoint: str, user_id: Optional[str] = None, 
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       request_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log an API request."""
        event = AuditEvent(
            event_type=AuditEventType.API_REQUEST,
            level=AuditLevel.INFO,
            user_id=user_id,
            component="api",
            action=f"{method} {endpoint}",
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            details=details or {}
        )
        self.log_event(event)
    
    def log_api_response(self, method: str, endpoint: str, status_code: int, duration_ms: float,
                        user_id: Optional[str] = None, request_id: Optional[str] = None,
                        error_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log an API response."""
        event = AuditEvent(
            event_type=AuditEventType.API_RESPONSE,
            level=AuditLevel.ERROR if status_code >= 400 else AuditLevel.INFO,
            user_id=user_id,
            component="api",
            action=f"{method} {endpoint} -> {status_code}",
            request_id=request_id,
            duration_ms=duration_ms,
            success=status_code < 400,
            error_message=error_message,
            details=details or {}
        )
        self.log_event(event)
    
    def log_file_access(self, operation: str, file_path: str, user_id: Optional[str] = None,
                       success: bool = True, error_message: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None):
        """Log file access operations."""
        event = AuditEvent(
            event_type=AuditEventType.FILE_ACCESS,
            level=AuditLevel.ERROR if not success else AuditLevel.INFO,
            user_id=user_id,
            component="file_system",
            action=operation,
            resource=file_path,
            success=success,
            error_message=error_message,
            details=details or {}
        )
        self.log_event(event)
    
    def log_code_execution(self, language: str, user_id: Optional[str] = None,
                          success: bool = True, duration_ms: Optional[float] = None,
                          error_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log code execution events."""
        event = AuditEvent(
            event_type=AuditEventType.CODE_EXECUTION,
            level=AuditLevel.ERROR if not success else AuditLevel.INFO,
            user_id=user_id,
            component="execution_engine",
            action=f"execute_{language}",
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            details=details or {}
        )
        self.log_event(event)
    
    def log_security_event(self, action: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, severity: str = "medium",
                          details: Optional[Dict[str, Any]] = None):
        """Log security events."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            level=AuditLevel.SECURITY,
            user_id=user_id,
            component="security",
            action=action,
            ip_address=ip_address,
            details={**(details or {}), "severity": severity}
        )
        self.log_event(event)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "",
                              component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        event = AuditEvent(
            event_type=AuditEventType.PERFORMANCE_METRIC,
            level=AuditLevel.INFO,
            component=component or "performance",
            action=metric_name,
            details={
                **(details or {}),
                "value": value,
                "unit": unit
            }
        )
        self.log_event(event)
    
    def log_workflow_event(self, workflow_id: str, workflow_type: str, action: str,
                          user_id: Optional[str] = None, duration_ms: Optional[float] = None,
                          success: bool = True, error_message: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None):
        """Log workflow events."""
        event_type = AuditEventType.WORKFLOW_START if action == "start" else AuditEventType.WORKFLOW_END
        
        event = AuditEvent(
            event_type=event_type,
            level=AuditLevel.ERROR if not success else AuditLevel.INFO,
            user_id=user_id,
            component="workflow_engine",
            action=f"{action}_{workflow_type}",
            resource=workflow_id,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            details=details or {}
        )
        self.log_event(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        return {
            "events_logged": self.events_logged,
            "events_dropped": self.events_dropped,
            "queue_size": self.event_queue.qsize(),
            "last_flush": self.last_flush.isoformat(),
            "log_directory": str(self.log_directory),
            "config": asdict(self.config)
        }
    
    def shutdown(self):
        """Shutdown the audit logger."""
        logging.getLogger(__name__).info("Shutting down audit logger...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        # Process remaining events
        remaining_events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                remaining_events.append(event)
            except Empty:
                break
        
        for event in remaining_events:
            self._write_event(event)
        
        # Close log files
        if self.main_log_file:
            self.main_log_file.close()
        if self.security_log_file:
            self.security_log_file.close()
        if self.performance_log_file:
            self.performance_log_file.close()
        
        logging.getLogger(__name__).info("Audit logger shutdown completed")

# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None

def get_audit_logger(config: Optional[AuditConfig] = None) -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(config)
    return _audit_logger

def shutdown_audit_logger():
    """Shutdown the global audit logger."""
    global _audit_logger
    if _audit_logger:
        _audit_logger.shutdown()
        _audit_logger = None

# Convenience functions
def log_system_event(action: str, details: Optional[Dict[str, Any]] = None, level: AuditLevel = AuditLevel.INFO):
    """Log a system event using the global logger."""
    get_audit_logger().log_system_event(action, details, level)

def log_api_request(method: str, endpoint: str, user_id: Optional[str] = None, 
                   ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                   request_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """Log an API request using the global logger."""
    get_audit_logger().log_api_request(method, endpoint, user_id, ip_address, user_agent, request_id, details)

def log_security_event(action: str, user_id: Optional[str] = None,
                      ip_address: Optional[str] = None, severity: str = "medium",
                      details: Optional[Dict[str, Any]] = None):
    """Log a security event using the global logger."""
    get_audit_logger().log_security_event(action, user_id, ip_address, severity, details)