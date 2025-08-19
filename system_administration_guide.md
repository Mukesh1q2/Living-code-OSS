# Sanskrit Rewrite Engine - System Administration Guide

## Overview

This guide provides comprehensive instructions for deploying, monitoring, and maintaining the Sanskrit Rewrite Engine in production environments. The system is designed for high availability, scalability, and security.

## Architecture Overview

The Sanskrit Rewrite Engine consists of several key components:

- **Linguistic Core**: Sanskrit grammar engine, tokenization, and morphological analysis
- **Reasoning Engine**: Logic programming backend with symbolic computation
- **R-Zero Integration**: Self-learning and model adaptation framework
- **API Server**: REST and WebSocket endpoints for client interaction
- **Web Interface**: React-based frontend with real-time collaboration
- **Security Layer**: Sandboxed execution and access controls
- **Monitoring System**: Health monitoring, alerting, and audit logging

## System Requirements

### Minimum Requirements
- **CPU**: 8 cores (Intel/AMD x64 or ARM64)
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD
- **GPU**: Optional but recommended (NVIDIA RTX 3060 or better, AMD RX 6800M or better)
- **Network**: 1 Gbps connection
- **OS**: Ubuntu 20.04+, CentOS 8+, or Windows Server 2019+

### Recommended Production Requirements
- **CPU**: 16+ cores
- **Memory**: 32+ GB RAM
- **Storage**: 500+ GB NVMe SSD
- **GPU**: NVIDIA RTX 4080 or better, AMD RX 7800 XT or better
- **Network**: 10 Gbps connection
- **Load Balancer**: HAProxy or NGINX
- **Database**: PostgreSQL 13+ or MongoDB 5+

## Installation

### Docker Deployment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/sanskrit-rewrite-engine.git
   cd sanskrit-rewrite-engine
   ```

2. **Build Docker images**:
   ```bash
   # CPU-only version
   docker build -t sanskrit-engine:latest .
   
   # GPU-enabled version
   docker build -t sanskrit-engine:gpu --target gpu .
   ```

3. **Run with Docker Compose**:
   ```bash
   # Copy and customize configuration
   cp docker/docker-compose.yml docker-compose.prod.yml
   
   # Start services
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Kubernetes Deployment

1. **Prepare configuration**:
   ```bash
   # Create namespace
   kubectl create namespace sanskrit-engine
   
   # Apply configurations
   kubectl apply -f k8s/
   ```

2. **Deploy using Helm** (if available):
   ```bash
   helm install sanskrit-engine ./helm-chart \
     --namespace sanskrit-engine \
     --set image.tag=latest \
     --set gpu.enabled=true
   ```

### Manual Installation

1. **Install Python dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y build-essential git curl
   
   # CentOS/RHEL
   sudo yum groupinstall -y "Development Tools"
   sudo yum install -y git curl
   ```

3. **Configure environment**:
   ```bash
   # Copy configuration template
   cp config/production.yaml.template config/production.yaml
   
   # Edit configuration
   nano config/production.yaml
   ```

4. **Start services**:
   ```bash
   # Start API server
   python -m sanskrit_rewrite_engine.api_server --config config/production.yaml
   
   # Start frontend (in separate terminal)
   cd frontend
   npm install
   npm run build
   npm start
   ```

## Configuration

### Main Configuration File

The main configuration is stored in `config/production.yaml`:

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_connections: 1000
  
# Database Configuration
database:
  type: "postgresql"  # or "mongodb"
  host: "localhost"
  port: 5432
  name: "sanskrit_engine"
  username: "sanskrit_user"
  password: "${DB_PASSWORD}"
  
# GPU Configuration
gpu:
  enabled: true
  device_id: 0
  memory_limit: "8GB"
  
# R-Zero Configuration
r_zero:
  enabled: true
  model_path: "r_zero_models/"
  checkpoint_interval: 3600  # seconds
  
# Security Configuration
security:
  sandbox_enabled: true
  max_execution_time: 30  # seconds
  allowed_directories:
    - "/app/workspace"
    - "/tmp/sanskrit_temp"
  
# Monitoring Configuration
monitoring:
  health_check_interval: 30  # seconds
  metrics_retention_days: 30
  alert_email: "admin@yourorg.com"
  
# Logging Configuration
logging:
  level: "INFO"
  audit_enabled: true
  log_rotation: true
  max_file_size: "100MB"
```

### Environment Variables

Set these environment variables for production:

```bash
# Database
export DB_PASSWORD="your_secure_password"
export DB_HOST="your_db_host"

# Security
export SECRET_KEY="your_secret_key_here"
export JWT_SECRET="your_jwt_secret_here"

# External Services
export OPENAI_API_KEY="your_openai_key"  # Optional
export ANTHROPIC_API_KEY="your_anthropic_key"  # Optional

# Monitoring
export ALERT_WEBHOOK_URL="https://hooks.slack.com/your/webhook"
```

## Monitoring and Alerting

### Health Monitoring

The system includes comprehensive health monitoring:

```python
from sanskrit_rewrite_engine.health_monitoring import start_health_monitoring

# Start monitoring
monitor = start_health_monitoring()

# Check system health
health_status = monitor.get_system_health()
print(f"System Status: {health_status['overall_status']}")
```

### Key Metrics to Monitor

1. **System Resources**:
   - CPU usage (warning: >80%, critical: >95%)
   - Memory usage (warning: >85%, critical: >95%)
   - Disk usage (warning: >90%, critical: >98%)
   - GPU utilization and memory

2. **Application Metrics**:
   - API response times
   - Request throughput
   - Error rates
   - Sanskrit processing accuracy

3. **Business Metrics**:
   - Active users
   - Processing volume
   - Feature usage statistics

### Setting Up Alerts

Configure alerts in the monitoring configuration:

```yaml
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "alerts@yourorg.com"
    password: "${SMTP_PASSWORD}"
    recipients:
      - "admin@yourorg.com"
      - "devops@yourorg.com"
  
  webhook:
    enabled: true
    url: "${ALERT_WEBHOOK_URL}"
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
  
  thresholds:
    cpu_warning: 80.0
    cpu_critical: 95.0
    memory_warning: 85.0
    memory_critical: 95.0
    response_time_warning: 5.0
    response_time_critical: 15.0
```

## Security

### Access Control

1. **API Authentication**:
   ```python
   # JWT-based authentication
   headers = {
       "Authorization": f"Bearer {jwt_token}"
   }
   ```

2. **File System Security**:
   - Sandboxed execution environment
   - Allowlisted directories only
   - Resource limits enforced

3. **Network Security**:
   - HTTPS/TLS encryption
   - Rate limiting
   - IP allowlisting (optional)

### Security Best Practices

1. **Regular Updates**:
   ```bash
   # Update system packages
   sudo apt-get update && sudo apt-get upgrade
   
   # Update Python dependencies
   pip install -r requirements.txt --upgrade
   ```

2. **Backup Strategy**:
   ```bash
   # Database backup
   pg_dump sanskrit_engine > backup_$(date +%Y%m%d).sql
   
   # Model checkpoints backup
   tar -czf models_backup_$(date +%Y%m%d).tar.gz r_zero_models/
   ```

3. **Audit Logging**:
   - All API requests logged
   - File access tracked
   - Security events monitored
   - Regular log analysis

## Performance Optimization

### CPU Optimization

1. **Process Configuration**:
   ```yaml
   server:
     workers: 8  # 2x CPU cores
     worker_class: "uvicorn.workers.UvicornWorker"
     max_requests: 1000
     max_requests_jitter: 100
   ```

2. **Caching Strategy**:
   ```python
   # Redis caching for frequent requests
   cache_config = {
       "backend": "redis",
       "host": "localhost",
       "port": 6379,
       "ttl": 3600  # 1 hour
   }
   ```

### Memory Optimization

1. **Memory Limits**:
   ```yaml
   resources:
     limits:
       memory: "8Gi"
     requests:
       memory: "4Gi"
   ```

2. **Garbage Collection**:
   ```python
   import gc
   
   # Force garbage collection after heavy operations
   gc.collect()
   ```

### GPU Optimization

1. **Memory Management**:
   ```python
   # Clear GPU cache periodically
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

2. **Batch Processing**:
   ```python
   # Process requests in batches for better GPU utilization
   batch_size = 32
   ```

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
   ```bash
   # Check memory usage
   free -h
   
   # Check process memory
   ps aux --sort=-%mem | head -10
   
   # Restart service if needed
   systemctl restart sanskrit-engine
   ```

2. **GPU Issues**:
   ```bash
   # Check GPU status
   nvidia-smi
   
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Database Connection Issues**:
   ```bash
   # Test database connection
   psql -h localhost -U sanskrit_user -d sanskrit_engine
   
   # Check connection pool
   SELECT * FROM pg_stat_activity;
   ```

### Log Analysis

1. **Application Logs**:
   ```bash
   # View recent logs
   tail -f logs/sanskrit_engine.log
   
   # Search for errors
   grep -i error logs/sanskrit_engine.log
   ```

2. **Audit Logs**:
   ```bash
   # View audit events
   tail -f audit_logs/audit_$(date +%Y%m%d).log
   
   # Security events
   grep -i security audit_logs/security_$(date +%Y%m%d).log
   ```

### Performance Debugging

1. **Profile Performance**:
   ```python
   from sanskrit_rewrite_engine.performance_profiler import profile_function
   
   @profile_function
   def slow_function():
       # Your code here
       pass
   ```

2. **Memory Profiling**:
   ```bash
   # Install memory profiler
   pip install memory-profiler
   
   # Profile memory usage
   mprof run python your_script.py
   mprof plot
   ```

## Backup and Recovery

### Automated Backups

1. **Database Backup Script**:
   ```bash
   #!/bin/bash
   # backup_db.sh
   
   DATE=$(date +%Y%m%d_%H%M%S)
   BACKUP_DIR="/backups/database"
   
   mkdir -p $BACKUP_DIR
   
   pg_dump sanskrit_engine | gzip > $BACKUP_DIR/backup_$DATE.sql.gz
   
   # Keep only last 30 days
   find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
   ```

2. **Model Backup Script**:
   ```bash
   #!/bin/bash
   # backup_models.sh
   
   DATE=$(date +%Y%m%d_%H%M%S)
   BACKUP_DIR="/backups/models"
   
   mkdir -p $BACKUP_DIR
   
   tar -czf $BACKUP_DIR/models_$DATE.tar.gz r_zero_models/ r_zero_checkpoints/
   
   # Keep only last 7 days for models (they're large)
   find $BACKUP_DIR -name "models_*.tar.gz" -mtime +7 -delete
   ```

3. **Cron Configuration**:
   ```bash
   # Add to crontab
   0 2 * * * /path/to/backup_db.sh
   0 3 * * 0 /path/to/backup_models.sh  # Weekly model backup
   ```

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Stop application
   systemctl stop sanskrit-engine
   
   # Restore database
   gunzip -c backup_20240812.sql.gz | psql sanskrit_engine
   
   # Start application
   systemctl start sanskrit-engine
   ```

2. **Model Recovery**:
   ```bash
   # Extract models
   tar -xzf models_20240812.tar.gz
   
   # Restart service to load models
   systemctl restart sanskrit-engine
   ```

## Scaling

### Horizontal Scaling

1. **Load Balancer Configuration** (NGINX):
   ```nginx
   upstream sanskrit_backend {
       server 10.0.1.10:8000;
       server 10.0.1.11:8000;
       server 10.0.1.12:8000;
   }
   
   server {
       listen 80;
       server_name sanskrit.yourorg.com;
       
       location / {
           proxy_pass http://sanskrit_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Kubernetes Horizontal Pod Autoscaler**:
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: sanskrit-engine-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: sanskrit-engine
     minReplicas: 3
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

### Vertical Scaling

1. **Resource Limits**:
   ```yaml
   resources:
     limits:
       cpu: "4000m"
       memory: "16Gi"
       nvidia.com/gpu: 1
     requests:
       cpu: "2000m"
       memory: "8Gi"
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly Tasks**:
   - Review system logs
   - Check disk usage
   - Update security patches
   - Verify backups

2. **Monthly Tasks**:
   - Performance analysis
   - Capacity planning
   - Security audit
   - Update dependencies

3. **Quarterly Tasks**:
   - Disaster recovery testing
   - Performance benchmarking
   - Architecture review
   - Training data updates

### Maintenance Scripts

1. **Log Cleanup**:
   ```bash
   #!/bin/bash
   # cleanup_logs.sh
   
   # Compress old logs
   find /var/log/sanskrit-engine -name "*.log" -mtime +7 -exec gzip {} \;
   
   # Delete very old logs
   find /var/log/sanskrit-engine -name "*.log.gz" -mtime +30 -delete
   ```

2. **Health Check**:
   ```bash
   #!/bin/bash
   # health_check.sh
   
   # Check API health
   curl -f http://localhost:8000/health || echo "API health check failed"
   
   # Check database
   psql -h localhost -U sanskrit_user -d sanskrit_engine -c "SELECT 1;" || echo "Database check failed"
   
   # Check GPU
   nvidia-smi || echo "GPU check failed"
   ```

## Support and Documentation

### Getting Help

1. **Documentation**: Check the `/docs` directory for detailed API documentation
2. **Logs**: Always check application and audit logs first
3. **Community**: Join our Discord/Slack for community support
4. **Issues**: Report bugs on GitHub with full logs and reproduction steps

### Emergency Contacts

- **Primary Admin**: admin@yourorg.com
- **DevOps Team**: devops@yourorg.com
- **On-call**: +1-555-SANSKRIT (24/7)

### Useful Commands

```bash
# Service management
systemctl status sanskrit-engine
systemctl restart sanskrit-engine
systemctl logs -f sanskrit-engine

# Docker management
docker ps
docker logs sanskrit-engine
docker exec -it sanskrit-engine bash

# Kubernetes management
kubectl get pods -n sanskrit-engine
kubectl logs -f deployment/sanskrit-engine -n sanskrit-engine
kubectl describe pod <pod-name> -n sanskrit-engine

# Performance monitoring
htop
iotop
nvidia-smi -l 1
```

This guide should be updated regularly as the system evolves. Always test changes in a staging environment before applying to production.