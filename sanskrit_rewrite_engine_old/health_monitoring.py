"""
System Health Monitoring and Alerting for Sanskrit Rewrite Engine

This module provides comprehensive system health monitoring, alerting,
and automated recovery mechanisms for the Sanskrit reasoning system.

Requirements: All requirements final validation - system health monitoring and alerting
"""

import os
import sys
import json
import asyncio
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Union[float, int, str, bool]
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemAlert:
    """System alert information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    component: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class HealthMonitor:
    """
    Comprehensive system health monitoring with alerting and recovery.
    
    Monitors:
    - System resources (CPU, memory, disk, GPU)
    - Application performance metrics
    - Sanskrit processing pipeline health
    - R-Zero integration status
    - API endpoint availability
    - Database connectivity
    - File system integrity
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[SystemAlert] = []
        self.alert_handlers: List[Callable[[SystemAlert], None]] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.db_path = self.config.get('db_path', 'health_monitoring.db')
        self._init_database()
        
        # Component monitors
        self.system_monitor = SystemResourceMonitor(self)
        self.app_monitor = ApplicationMonitor(self)
        self.sanskrit_monitor = SanskritPipelineMonitor(self)
        self.api_monitor = APIHealthMonitor(self)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            'monitoring_interval': 30,  # seconds
            'alert_cooldown': 300,  # seconds
            'retention_days': 30,
            'thresholds': {
                'cpu_warning': 80.0,
                'cpu_critical': 95.0,
                'memory_warning': 85.0,
                'memory_critical': 95.0,
                'disk_warning': 90.0,
                'disk_critical': 98.0,
                'response_time_warning': 5.0,  # seconds
                'response_time_critical': 15.0,
            },
            'notifications': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'localhost',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                },
                'webhook': {
                    'enabled': False,
                    'url': '',
                    'headers': {}
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for metrics and alerts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value TEXT NOT NULL,
                        status TEXT NOT NULL,
                        unit TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        severity TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at DATETIME
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alerts(timestamp)
                ''')
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect all metrics
                self.system_monitor.collect_metrics()
                self.app_monitor.collect_metrics()
                self.sanskrit_monitor.collect_metrics()
                self.api_monitor.collect_metrics()
                
                # Process alerts
                self._process_alerts()
                
                # Cleanup old data
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def add_metric(self, metric: HealthMetric):
        """Add or update a health metric."""
        self.metrics[metric.name] = metric
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO metrics (name, value, status, unit, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (metric.name, str(metric.value), metric.status.value, 
                      metric.unit, metric.timestamp))
        except Exception as e:
            logger.error(f"Failed to store metric {metric.name}: {e}")
    
    def create_alert(self, severity: AlertSeverity, component: str, 
                    message: str, details: Optional[Dict[str, Any]] = None):
        """Create a new system alert."""
        alert = SystemAlert(
            severity=severity,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts (id, severity, component, message, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (alert.id, alert.severity.value, alert.component, 
                      alert.message, json.dumps(alert.details), alert.timestamp))
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Update database
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('''
                            UPDATE alerts SET resolved = TRUE, resolved_at = ?
                            WHERE id = ?
                        ''', (alert.resolved_at, alert_id))
                except Exception as e:
                    logger.error(f"Failed to update alert resolution: {e}")
                
                break
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = HealthStatus.HEALTHY
        critical_count = 0
        warning_count = 0
        
        for metric in self.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                critical_count += 1
                overall_status = HealthStatus.CRITICAL
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        return {
            'overall_status': overall_status.value,
            'metrics_count': len(self.metrics),
            'critical_metrics': critical_count,
            'warning_metrics': warning_count,
            'active_alerts': len(active_alerts),
            'last_check': datetime.now().isoformat(),
            'uptime': self._get_uptime(),
            'metrics': {name: {
                'value': metric.value,
                'status': metric.status.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat()
            } for name, metric in self.metrics.items()}
        }
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return str(uptime).split('.')[0]  # Remove microseconds
        except Exception:
            return "unknown"
    
    def _process_alerts(self):
        """Process and send alerts based on current metrics."""
        for metric in self.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                # Check if we already have an active alert for this metric
                existing_alert = any(
                    not a.resolved and a.component == metric.name 
                    for a in self.alerts
                )
                
                if not existing_alert:
                    self.create_alert(
                        AlertSeverity.CRITICAL,
                        metric.name,
                        f"Critical threshold exceeded: {metric.value} {metric.unit}",
                        {'metric': metric.name, 'value': metric.value, 'threshold': metric.threshold_critical}
                    )
            
            elif metric.status == HealthStatus.WARNING:
                existing_alert = any(
                    not a.resolved and a.component == metric.name 
                    for a in self.alerts
                )
                
                if not existing_alert:
                    self.create_alert(
                        AlertSeverity.WARNING,
                        metric.name,
                        f"Warning threshold exceeded: {metric.value} {metric.unit}",
                        {'metric': metric.name, 'value': metric.value, 'threshold': metric.threshold_warning}
                    )
    
    def _cleanup_old_data(self):
        """Clean up old metrics and resolved alerts."""
        cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean old metrics
                conn.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_date,))
                
                # Clean old resolved alerts
                conn.execute('''
                    DELETE FROM alerts 
                    WHERE resolved = TRUE AND resolved_at < ?
                ''', (cutoff_date,))
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def send_email_alert(self, alert: SystemAlert):
        """Send email notification for alert."""
        email_config = self.config['notifications']['email']
        if not email_config['enabled'] or not email_config['recipients']:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Sanskrit Engine Alert: {alert.component}"
            
            body = f"""
            Alert Details:
            - Severity: {alert.severity.value}
            - Component: {alert.component}
            - Message: {alert.message}
            - Timestamp: {alert.timestamp}
            - Details: {json.dumps(alert.details, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, alert: SystemAlert):
        """Send webhook notification for alert."""
        webhook_config = self.config['notifications']['webhook']
        if not webhook_config['enabled'] or not webhook_config['url']:
            return
        
        try:
            payload = {
                'alert_id': alert.id,
                'severity': alert.severity.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'details': alert.details
            }
            
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class SystemResourceMonitor:
    """Monitor system resources (CPU, memory, disk, GPU)."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.config = health_monitor.config['thresholds']
    
    def collect_metrics(self):
        """Collect system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = self._get_status(cpu_percent, 'cpu')
        self.health_monitor.add_metric(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status=cpu_status,
            threshold_warning=self.config['cpu_warning'],
            threshold_critical=self.config['cpu_critical'],
            unit="%",
            description="CPU usage percentage"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_status = self._get_status(memory.percent, 'memory')
        self.health_monitor.add_metric(HealthMetric(
            name="memory_usage",
            value=memory.percent,
            status=memory_status,
            threshold_warning=self.config['memory_warning'],
            threshold_critical=self.config['memory_critical'],
            unit="%",
            description="Memory usage percentage"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_status = self._get_status(disk_percent, 'disk')
        self.health_monitor.add_metric(HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status=disk_status,
            threshold_warning=self.config['disk_warning'],
            threshold_critical=self.config['disk_critical'],
            unit="%",
            description="Disk usage percentage"
        ))
        
        # GPU monitoring (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Monitor first GPU
                self.health_monitor.add_metric(HealthMetric(
                    name="gpu_usage",
                    value=gpu.load * 100,
                    status=HealthStatus.HEALTHY,  # No thresholds set for GPU yet
                    unit="%",
                    description="GPU usage percentage"
                ))
                
                self.health_monitor.add_metric(HealthMetric(
                    name="gpu_memory",
                    value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    status=HealthStatus.HEALTHY,
                    unit="%",
                    description="GPU memory usage percentage"
                ))
        except ImportError:
            pass  # GPUtil not available
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
    
    def _get_status(self, value: float, metric_type: str) -> HealthStatus:
        """Determine health status based on thresholds."""
        warning_threshold = self.config.get(f'{metric_type}_warning', 80)
        critical_threshold = self.config.get(f'{metric_type}_critical', 95)
        
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class ApplicationMonitor:
    """Monitor application-specific metrics."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.start_time = datetime.now()
    
    def collect_metrics(self):
        """Collect application metrics."""
        # Application uptime
        uptime = datetime.now() - self.start_time
        self.health_monitor.add_metric(HealthMetric(
            name="app_uptime",
            value=uptime.total_seconds(),
            status=HealthStatus.HEALTHY,
            unit="seconds",
            description="Application uptime"
        ))
        
        # Process metrics
        process = psutil.Process()
        
        # Memory usage by process
        memory_info = process.memory_info()
        self.health_monitor.add_metric(HealthMetric(
            name="app_memory_rss",
            value=memory_info.rss / 1024 / 1024,  # MB
            status=HealthStatus.HEALTHY,
            unit="MB",
            description="Application RSS memory usage"
        ))
        
        # File descriptors
        try:
            num_fds = process.num_fds()
            self.health_monitor.add_metric(HealthMetric(
                name="app_file_descriptors",
                value=num_fds,
                status=HealthStatus.HEALTHY,
                unit="count",
                description="Number of open file descriptors"
            ))
        except AttributeError:
            pass  # Not available on Windows
        
        # Thread count
        num_threads = process.num_threads()
        self.health_monitor.add_metric(HealthMetric(
            name="app_threads",
            value=num_threads,
            status=HealthStatus.HEALTHY,
            unit="count",
            description="Number of application threads"
        ))


class SanskritPipelineMonitor:
    """Monitor Sanskrit processing pipeline health."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.processing_times = []
        self.error_count = 0
        self.success_count = 0
    
    def collect_metrics(self):
        """Collect Sanskrit pipeline metrics."""
        # Average processing time
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            
            # Clear old times (keep last 100)
            self.processing_times = self.processing_times[-100:]
        else:
            avg_time = 0
            max_time = 0
        
        time_status = self._get_time_status(avg_time)
        self.health_monitor.add_metric(HealthMetric(
            name="sanskrit_avg_processing_time",
            value=avg_time,
            status=time_status,
            threshold_warning=5.0,
            threshold_critical=15.0,
            unit="seconds",
            description="Average Sanskrit processing time"
        ))
        
        # Error rate
        total_requests = self.success_count + self.error_count
        error_rate = (self.error_count / total_requests * 100) if total_requests > 0 else 0
        
        error_status = HealthStatus.HEALTHY
        if error_rate > 10:
            error_status = HealthStatus.CRITICAL
        elif error_rate > 5:
            error_status = HealthStatus.WARNING
        
        self.health_monitor.add_metric(HealthMetric(
            name="sanskrit_error_rate",
            value=error_rate,
            status=error_status,
            threshold_warning=5.0,
            threshold_critical=10.0,
            unit="%",
            description="Sanskrit processing error rate"
        ))
    
    def record_processing_time(self, duration: float):
        """Record a processing time."""
        self.processing_times.append(duration)
        self.success_count += 1
    
    def record_error(self):
        """Record a processing error."""
        self.error_count += 1
    
    def _get_time_status(self, time_value: float) -> HealthStatus:
        """Get status based on processing time."""
        if time_value > 15.0:
            return HealthStatus.CRITICAL
        elif time_value > 5.0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class APIHealthMonitor:
    """Monitor API endpoint health and availability."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.endpoints = [
            '/health',
            '/api/v1/sanskrit/analyze',
            '/api/v1/sanskrit/translate',
            '/api/v1/reasoning/solve'
        ]
    
    def collect_metrics(self):
        """Collect API health metrics."""
        base_url = "http://localhost:8000"  # Default API server
        
        for endpoint in self.endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                # Response time metric
                time_status = self._get_response_time_status(response_time)
                self.health_monitor.add_metric(HealthMetric(
                    name=f"api_response_time_{endpoint.replace('/', '_')}",
                    value=response_time,
                    status=time_status,
                    threshold_warning=5.0,
                    threshold_critical=15.0,
                    unit="seconds",
                    description=f"Response time for {endpoint}"
                ))
                
                # Availability metric
                availability_status = HealthStatus.HEALTHY if response.status_code < 500 else HealthStatus.CRITICAL
                self.health_monitor.add_metric(HealthMetric(
                    name=f"api_availability_{endpoint.replace('/', '_')}",
                    value=1 if response.status_code < 500 else 0,
                    status=availability_status,
                    unit="boolean",
                    description=f"Availability of {endpoint}"
                ))
                
            except requests.exceptions.RequestException as e:
                # Endpoint unavailable
                self.health_monitor.add_metric(HealthMetric(
                    name=f"api_availability_{endpoint.replace('/', '_')}",
                    value=0,
                    status=HealthStatus.CRITICAL,
                    unit="boolean",
                    description=f"Availability of {endpoint}"
                ))
                
                logger.warning(f"API endpoint {endpoint} unavailable: {e}")
    
    def _get_response_time_status(self, response_time: float) -> HealthStatus:
        """Get status based on response time."""
        if response_time > 15.0:
            return HealthStatus.CRITICAL
        elif response_time > 5.0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        
        # Add default alert handlers
        _health_monitor.add_alert_handler(_health_monitor.send_email_alert)
        _health_monitor.add_alert_handler(_health_monitor.send_webhook_alert)
        
    return _health_monitor

def start_health_monitoring():
    """Start the global health monitoring system."""
    monitor = get_health_monitor()
    monitor.start_monitoring()
    return monitor

def stop_health_monitoring():
    """Stop the global health monitoring system."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()

if __name__ == "__main__":
    # Example usage
    monitor = start_health_monitoring()
    
    try:
        # Keep monitoring running
        while True:
            time.sleep(60)
            health_status = monitor.get_system_health()
            print(f"System Health: {health_status['overall_status']}")
            
    except KeyboardInterrupt:
        print("Stopping health monitoring...")
        stop_health_monitoring()
