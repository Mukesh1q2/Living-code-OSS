"""
Security Configuration and Policy Management for Sanskrit Rewrite Engine

This module provides centralized security configuration management,
policy enforcement, and security best practices implementation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityPolicy(Enum):
    """Security policy levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    CUSTOM = "custom"


class ThreatLevel(Enum):
    """Threat level classifications."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class FileAccessPolicy:
    """File access policy configuration."""
    allowed_directories: List[str] = field(default_factory=list)
    blocked_directories: List[str] = field(default_factory=list)
    allowed_extensions: List[str] = field(default_factory=list)
    blocked_extensions: List[str] = field(default_factory=lambda: [
        '.exe', '.bat', '.cmd', '.sh', '.ps1', '.dll', '.so', '.dylib'
    ])
    max_file_size_mb: int = 50
    max_total_files: int = 1000
    require_approval_for_sensitive_paths: bool = True
    sensitive_paths: List[str] = field(default_factory=lambda: [
        '/etc', '/sys', '/proc', '/dev', '/boot',
        'C:\\Windows', 'C:\\System32', 'C:\\Program Files'
    ])


@dataclass
class CodeExecutionPolicy:
    """Code execution policy configuration."""
    enabled: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ['python'])
    blocked_imports: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'telnetlib', 'ssl', 'ctypes',
        'multiprocessing', 'threading', 'asyncio'
    ])
    allowed_imports: List[str] = field(default_factory=lambda: [
        'math', 'json', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'string', 're', 'unicodedata'
    ])
    max_execution_time: int = 30
    max_memory_mb: int = 256
    max_cpu_percent: int = 50
    enable_network_access: bool = False
    enable_file_system_access: bool = False
    sandbox_mode: bool = True


@dataclass
class AuditPolicy:
    """Audit and logging policy configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    log_all_file_access: bool = True
    log_all_code_execution: bool = True
    log_permission_denials: bool = True
    log_security_violations: bool = True
    retention_days: int = 90
    max_log_size_mb: int = 100
    enable_real_time_alerts: bool = True
    alert_on_suspicious_activity: bool = True
    alert_threshold_failed_attempts: int = 5
    alert_threshold_time_window_minutes: int = 5


@dataclass
class NetworkPolicy:
    """Network access policy configuration."""
    enabled: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    blocked_ports: List[int] = field(default_factory=lambda: [22, 23, 25, 53, 80, 443, 993, 995])
    max_connections: int = 0
    connection_timeout: int = 10
    enable_ssl_verification: bool = True


@dataclass
class SessionPolicy:
    """Session management policy configuration."""
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    require_reauthentication_for_sensitive_ops: bool = True
    enable_session_encryption: bool = True
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    idle_timeout_minutes: int = 30


@dataclass
class SecurityConfiguration:
    """Comprehensive security configuration."""
    policy_level: SecurityPolicy = SecurityPolicy.MODERATE
    file_access: FileAccessPolicy = field(default_factory=FileAccessPolicy)
    code_execution: CodeExecutionPolicy = field(default_factory=CodeExecutionPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    session: SessionPolicy = field(default_factory=SessionPolicy)
    
    # Global settings
    enable_security_headers: bool = True
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    enable_ip_whitelisting: bool = False
    whitelisted_ips: List[str] = field(default_factory=list)
    enable_geo_blocking: bool = False
    blocked_countries: List[str] = field(default_factory=list)
    
    # Compliance settings
    enable_gdpr_compliance: bool = True
    enable_hipaa_compliance: bool = False
    enable_sox_compliance: bool = False
    data_retention_days: int = 365
    enable_data_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    
    # Monitoring and alerting
    enable_intrusion_detection: bool = True
    enable_anomaly_detection: bool = True
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    monitoring_interval_seconds: int = 60
    
    # Emergency settings
    emergency_lockdown_enabled: bool = False
    emergency_contact: Optional[str] = None
    incident_response_plan: Optional[str] = None


class SecurityConfigurationManager:
    """Manages security configuration and policy enforcement."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "security_config.yaml"
        self.config: SecurityConfiguration = SecurityConfiguration()
        self.policy_templates: Dict[SecurityPolicy, SecurityConfiguration] = {}
        self._initialize_policy_templates()
        self._load_configuration()
    
    def _initialize_policy_templates(self):
        """Initialize predefined security policy templates."""
        
        # Strict policy template
        strict_config = SecurityConfiguration(
            policy_level=SecurityPolicy.STRICT,
            file_access=FileAccessPolicy(
                blocked_extensions=[
                    '.exe', '.bat', '.cmd', '.sh', '.ps1', '.dll', '.so', '.dylib',
                    '.msi', '.deb', '.rpm', '.dmg', '.pkg', '.app', '.scr', '.com',
                    '.pif', '.vbs', '.js', '.jar', '.class'
                ],
                max_file_size_mb=10,
                max_total_files=100,
                require_approval_for_sensitive_paths=True
            ),
            code_execution=CodeExecutionPolicy(
                enabled=False,
                max_execution_time=10,
                max_memory_mb=128,
                max_cpu_percent=25,
                sandbox_mode=True
            ),
            audit=AuditPolicy(
                log_level="DEBUG",
                retention_days=180,
                enable_real_time_alerts=True,
                alert_threshold_failed_attempts=3
            ),
            network=NetworkPolicy(enabled=False),
            session=SessionPolicy(
                session_timeout_minutes=30,
                max_concurrent_sessions=2,
                require_reauthentication_for_sensitive_ops=True
            ),
            rate_limit_requests_per_minute=50
        )
        
        # Moderate policy template
        moderate_config = SecurityConfiguration(
            policy_level=SecurityPolicy.MODERATE,
            file_access=FileAccessPolicy(
                max_file_size_mb=50,
                max_total_files=500
            ),
            code_execution=CodeExecutionPolicy(
                enabled=True,
                max_execution_time=30,
                max_memory_mb=256,
                max_cpu_percent=50,
                sandbox_mode=True
            ),
            audit=AuditPolicy(
                log_level="INFO",
                retention_days=90,
                alert_threshold_failed_attempts=5
            ),
            network=NetworkPolicy(
                enabled=False,
                max_connections=5
            ),
            session=SessionPolicy(
                session_timeout_minutes=60,
                max_concurrent_sessions=5
            )
        )
        
        # Permissive policy template
        permissive_config = SecurityConfiguration(
            policy_level=SecurityPolicy.PERMISSIVE,
            file_access=FileAccessPolicy(
                blocked_extensions=['.exe', '.bat', '.sh'],
                max_file_size_mb=100,
                max_total_files=1000,
                require_approval_for_sensitive_paths=False
            ),
            code_execution=CodeExecutionPolicy(
                enabled=True,
                max_execution_time=60,
                max_memory_mb=512,
                max_cpu_percent=75,
                enable_file_system_access=True,
                sandbox_mode=False
            ),
            audit=AuditPolicy(
                log_level="WARN",
                retention_days=30,
                alert_threshold_failed_attempts=10
            ),
            network=NetworkPolicy(
                enabled=True,
                max_connections=10
            ),
            session=SessionPolicy(
                session_timeout_minutes=120,
                max_concurrent_sessions=10,
                require_reauthentication_for_sensitive_ops=False
            ),
            rate_limit_requests_per_minute=200
        )
        
        self.policy_templates = {
            SecurityPolicy.STRICT: strict_config,
            SecurityPolicy.MODERATE: moderate_config,
            SecurityPolicy.PERMISSIVE: permissive_config
        }
    
    def _load_configuration(self):
        """Load security configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Convert dict to SecurityConfiguration
                self.config = self._dict_to_config(config_data)
                logger.info(f"Loaded security configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load security configuration: {e}")
                logger.info("Using default security configuration")
        else:
            logger.info("No security configuration file found, using default configuration")
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> SecurityConfiguration:
        """Convert dictionary to SecurityConfiguration object."""
        # This is a simplified conversion - in practice, you'd want more robust handling
        config = SecurityConfiguration()
        
        if 'policy_level' in config_data:
            config.policy_level = SecurityPolicy(config_data['policy_level'])
        
        if 'file_access' in config_data:
            fa_data = config_data['file_access']
            config.file_access = FileAccessPolicy(**fa_data)
        
        if 'code_execution' in config_data:
            ce_data = config_data['code_execution']
            config.code_execution = CodeExecutionPolicy(**ce_data)
        
        if 'audit' in config_data:
            audit_data = config_data['audit']
            config.audit = AuditPolicy(**audit_data)
        
        if 'network' in config_data:
            net_data = config_data['network']
            config.network = NetworkPolicy(**net_data)
        
        if 'session' in config_data:
            session_data = config_data['session']
            config.session = SessionPolicy(**session_data)
        
        # Copy other fields
        for key, value in config_data.items():
            if hasattr(config, key) and key not in ['file_access', 'code_execution', 'audit', 'network', 'session']:
                setattr(config, key, value)
        
        return config
    
    def save_configuration(self):
        """Save security configuration to file."""
        try:
            config_dict = asdict(self.config)
            
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Saved security configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save security configuration: {e}")
    
    def apply_policy_template(self, policy: SecurityPolicy):
        """Apply a predefined security policy template."""
        if policy in self.policy_templates:
            self.config = self.policy_templates[policy]
            logger.info(f"Applied {policy.value} security policy template")
        else:
            logger.error(f"Unknown security policy: {policy}")
    
    def validate_configuration(self) -> List[str]:
        """Validate security configuration and return any issues."""
        issues = []
        
        # Validate file access policy
        if self.config.file_access.max_file_size_mb <= 0:
            issues.append("File access: max_file_size_mb must be positive")
        
        if self.config.file_access.max_total_files <= 0:
            issues.append("File access: max_total_files must be positive")
        
        # Validate code execution policy
        if self.config.code_execution.enabled:
            if self.config.code_execution.max_execution_time <= 0:
                issues.append("Code execution: max_execution_time must be positive")
            
            if self.config.code_execution.max_memory_mb <= 0:
                issues.append("Code execution: max_memory_mb must be positive")
            
            if not (0 < self.config.code_execution.max_cpu_percent <= 100):
                issues.append("Code execution: max_cpu_percent must be between 1 and 100")
        
        # Validate audit policy
        if self.config.audit.retention_days <= 0:
            issues.append("Audit: retention_days must be positive")
        
        if self.config.audit.max_log_size_mb <= 0:
            issues.append("Audit: max_log_size_mb must be positive")
        
        # Validate session policy
        if self.config.session.session_timeout_minutes <= 0:
            issues.append("Session: session_timeout_minutes must be positive")
        
        if self.config.session.max_concurrent_sessions <= 0:
            issues.append("Session: max_concurrent_sessions must be positive")
        
        # Validate rate limiting
        if self.config.rate_limit_requests_per_minute <= 0:
            issues.append("Rate limiting: requests_per_minute must be positive")
        
        return issues
    
    def get_security_recommendations(self) -> List[Dict[str, Any]]:
        """Get security recommendations based on current configuration."""
        recommendations = []
        
        # Check if code execution is enabled without sandboxing
        if self.config.code_execution.enabled and not self.config.code_execution.sandbox_mode:
            recommendations.append({
                'level': ThreatLevel.HIGH,
                'category': 'Code Execution',
                'issue': 'Code execution is enabled without sandboxing',
                'recommendation': 'Enable sandbox mode for code execution',
                'action': 'Set code_execution.sandbox_mode = True'
            })
        
        # Check if network access is enabled
        if self.config.network.enabled:
            recommendations.append({
                'level': ThreatLevel.MEDIUM,
                'category': 'Network Access',
                'issue': 'Network access is enabled',
                'recommendation': 'Consider disabling network access if not required',
                'action': 'Set network.enabled = False'
            })
        
        # Check audit settings
        if not self.config.audit.enabled:
            recommendations.append({
                'level': ThreatLevel.CRITICAL,
                'category': 'Auditing',
                'issue': 'Audit logging is disabled',
                'recommendation': 'Enable audit logging for security monitoring',
                'action': 'Set audit.enabled = True'
            })
        
        # Check session timeout
        if self.config.session.session_timeout_minutes > 120:
            recommendations.append({
                'level': ThreatLevel.MEDIUM,
                'category': 'Session Management',
                'issue': 'Session timeout is too long',
                'recommendation': 'Reduce session timeout to improve security',
                'action': 'Set session.session_timeout_minutes <= 120'
            })
        
        # Check file size limits
        if self.config.file_access.max_file_size_mb > 100:
            recommendations.append({
                'level': ThreatLevel.LOW,
                'category': 'File Access',
                'issue': 'File size limit is high',
                'recommendation': 'Consider reducing file size limit',
                'action': 'Set file_access.max_file_size_mb <= 100'
            })
        
        # Check blocked extensions
        dangerous_extensions = {'.exe', '.bat', '.sh', '.ps1', '.dll', '.so'}
        current_blocked = set(self.config.file_access.blocked_extensions)
        missing_blocks = dangerous_extensions - current_blocked
        
        if missing_blocks:
            recommendations.append({
                'level': ThreatLevel.HIGH,
                'category': 'File Access',
                'issue': f'Dangerous file extensions not blocked: {missing_blocks}',
                'recommendation': 'Block dangerous file extensions',
                'action': f'Add {missing_blocks} to file_access.blocked_extensions'
            })
        
        return recommendations
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security configuration report."""
        validation_issues = self.validate_configuration()
        recommendations = self.get_security_recommendations()
        
        # Calculate security score
        total_checks = 10
        passed_checks = total_checks - len(validation_issues)
        security_score = (passed_checks / total_checks) * 100
        
        # Adjust score based on recommendations
        critical_issues = len([r for r in recommendations if r['level'] == ThreatLevel.CRITICAL])
        high_issues = len([r for r in recommendations if r['level'] == ThreatLevel.HIGH])
        medium_issues = len([r for r in recommendations if r['level'] == ThreatLevel.MEDIUM])
        
        security_score -= (critical_issues * 20) + (high_issues * 10) + (medium_issues * 5)
        security_score = max(0, security_score)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'policy_level': self.config.policy_level.value,
            'security_score': round(security_score, 1),
            'validation_issues': validation_issues,
            'recommendations': recommendations,
            'configuration_summary': {
                'file_access_enabled': True,
                'code_execution_enabled': self.config.code_execution.enabled,
                'network_access_enabled': self.config.network.enabled,
                'audit_logging_enabled': self.config.audit.enabled,
                'session_timeout_minutes': self.config.session.session_timeout_minutes,
                'max_file_size_mb': self.config.file_access.max_file_size_mb,
                'blocked_extensions_count': len(self.config.file_access.blocked_extensions),
                'rate_limit_enabled': self.config.enable_rate_limiting
            },
            'compliance_status': {
                'gdpr_compliant': self.config.enable_gdpr_compliance,
                'hipaa_compliant': self.config.enable_hipaa_compliance,
                'sox_compliant': self.config.enable_sox_compliance,
                'data_encryption_enabled': self.config.enable_data_encryption
            }
        }
    
    def export_configuration(self, format: str = 'yaml') -> str:
        """Export configuration in specified format."""
        config_dict = asdict(self.config)
        
        if format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            return json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_configuration(self, config_str: str, format: str = 'yaml'):
        """Import configuration from string."""
        try:
            if format.lower() == 'yaml':
                config_data = yaml.safe_load(config_str)
            elif format.lower() == 'json':
                config_data = json.loads(config_str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.config = self._dict_to_config(config_data)
            logger.info("Successfully imported security configuration")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            raise


def create_default_security_config(workspace_path: str) -> SecurityConfigurationManager:
    """Create default security configuration for workspace."""
    config_path = os.path.join(workspace_path, ".kiro", "security_config.yaml")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    manager = SecurityConfigurationManager(config_path)
    
    # Apply moderate policy by default
    manager.apply_policy_template(SecurityPolicy.MODERATE)
    
    # Customize for Sanskrit workspace
    manager.config.file_access.allowed_directories = [
        workspace_path,
        os.path.join(workspace_path, "sanskrit_corpus"),
        os.path.join(workspace_path, "output"),
        os.path.join(workspace_path, "temp")
    ]
    
    manager.config.file_access.sensitive_paths.extend([
        os.path.join(workspace_path, ".git"),
        os.path.join(workspace_path, ".kiro", "settings")
    ])
    
    # Save configuration
    manager.save_configuration()
    
    return manager


if __name__ == "__main__":
    # Example usage
    manager = create_default_security_config(".")
    
    # Generate security report
    report = manager.generate_security_report()
    print("Security Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Export configuration
    yaml_config = manager.export_configuration('yaml')
    print("\nConfiguration (YAML):")
    print(yaml_config)