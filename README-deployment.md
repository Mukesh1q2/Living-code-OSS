# Vidya Quantum Interface - Deployment Guide

This guide covers deployment options for the Vidya Quantum Interface, from local development to production cloud deployment.

## Quick Start

### Local Development
```bash
# Clone and setup
git clone <repository>
cd vidya-quantum-interface

# Copy environment configuration
cp .env.example .env

# Start with Docker Compose
docker-compose up -d

# Or start development servers separately
npm run dev:frontend &
python -m uvicorn vidya_quantum_interface.api_server:app --reload
```

### Production Deployment
```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production
```

## Deployment Options

### 1. Local Development (Docker Compose)

**Best for**: Development, testing, demos

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services**:
- Frontend: http://localhost (Nginx + React)
- Backend: http://localhost:8000 (FastAPI)
- Redis: localhost:6379

### 2. Staging Environment

**Best for**: Integration testing, QA

```bash
# Deploy to staging
./scripts/deploy.sh staging

# With monitoring
docker-compose -f docker-compose.yml -f docker-compose.staging.yml --profile monitoring up -d
```

**Additional Services**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

### 3. Production Environment

**Best for**: Live deployment

```bash
# Deploy to production
./scripts/deploy.sh production

# With full monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

**Features**:
- High availability (3 replicas)
- Resource limits and reservations
- Health checks and auto-restart
- Comprehensive monitoring
- Alerting system

### 4. Kubernetes Deployment

**Best for**: Cloud-native, scalable deployment

```bash
# Deploy to Kubernetes
export KUBERNETES_NAMESPACE=vidya-production
export VERSION=latest
./scripts/k8s-deploy.sh production
```

**Features**:
- Auto-scaling (HPA)
- Rolling updates
- Service mesh ready
- Cloud provider integration

## Environment Configuration

### Environment Variables

Create environment-specific `.env` files:

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# .env.staging  
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=info

# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=warning
```

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | - |
| `ENABLE_QUANTUM_EFFECTS` | Enable quantum visualizations | `true` |
| `CLOUD_PROVIDER` | Cloud provider (aws/gcp/azure) | `local` |

## Cloud Provider Setup

### AWS Deployment

1. **Configure AWS credentials**:
```bash
aws configure
```

2. **Set environment variables**:
```bash
export CLOUD_PROVIDER=aws
export AWS_REGION=us-west-2
export AWS_S3_BUCKET=vidya-storage
```

3. **Deploy with AWS integration**:
```bash
./scripts/deploy.sh production
```

**AWS Services Used**:
- ECS/EKS for container orchestration
- S3 for storage
- CloudWatch for logging
- Bedrock for AI processing
- Lambda for serverless functions

### Google Cloud Platform

1. **Configure GCP credentials**:
```bash
gcloud auth login
gcloud config set project your-project-id
```

2. **Deploy to GKE**:
```bash
export CLOUD_PROVIDER=gcp
./scripts/k8s-deploy.sh production
```

### Azure Deployment

1. **Configure Azure CLI**:
```bash
az login
```

2. **Deploy to AKS**:
```bash
export CLOUD_PROVIDER=azure
./scripts/k8s-deploy.sh production
```

## Monitoring and Observability

### Metrics Collection

The application exposes Prometheus metrics at `/metrics`:

- HTTP request metrics
- Sanskrit processing performance
- Quantum effect rendering FPS
- Memory and CPU usage
- WebSocket connections
- Error rates

### Logging

Structured logging with multiple outputs:

- **Console**: Development debugging
- **Files**: Rotating log files
- **CloudWatch**: Production cloud logging
- **JSON**: Structured logs for analysis

### Health Checks

Health check endpoints:

- `/health` - Basic health check
- `/health/detailed` - Comprehensive system status
- `/metrics` - Prometheus metrics

### Grafana Dashboards

Pre-configured dashboards for:

- Application performance
- System resources
- Sanskrit processing metrics
- Quantum effect performance
- User interaction analytics

## Security Considerations

### Production Security

1. **Environment Variables**: Use secrets management
2. **HTTPS**: Enable TLS termination
3. **Authentication**: Implement proper auth
4. **Rate Limiting**: Prevent abuse
5. **CORS**: Configure allowed origins
6. **Input Validation**: Sanitize all inputs

### Secrets Management

```bash
# Kubernetes secrets
kubectl create secret generic vidya-secrets \
  --from-literal=huggingface-api-key=your-key \
  --from-literal=secret-key=your-secret

# Docker secrets
echo "your-api-key" | docker secret create huggingface_api_key -
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vidya-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vidya-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Performance Optimization

1. **Caching**: Redis for API responses
2. **CDN**: Static asset delivery
3. **Database**: Connection pooling
4. **AI Models**: Local inference when possible
5. **WebGL**: Optimized quantum effects

## Troubleshooting

### Common Issues

1. **Container won't start**:
```bash
docker logs vidya-backend
docker logs vidya-frontend
```

2. **Health check failures**:
```bash
curl http://localhost:8000/health
curl http://localhost/health
```

3. **Performance issues**:
```bash
# Check metrics
curl http://localhost:9000/metrics

# Check logs
docker-compose logs -f backend
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=debug
docker-compose up -d
```

## CI/CD Pipeline

The included GitHub Actions workflow provides:

1. **Automated Testing**: Unit, integration, and security tests
2. **Image Building**: Multi-stage Docker builds
3. **Security Scanning**: Vulnerability assessment
4. **Deployment**: Automated staging and production deployment
5. **Monitoring**: Post-deployment health checks

### Pipeline Stages

1. **Test** → Run all tests and linting
2. **Build** → Create and push Docker images  
3. **Deploy Staging** → Automatic staging deployment
4. **Deploy Production** → Manual production deployment
5. **Monitor** → Health checks and notifications

## Support and Maintenance

### Backup Strategy

1. **Database**: Automated Redis snapshots
2. **Logs**: Centralized log aggregation
3. **Configuration**: Version-controlled configs
4. **Models**: AI model versioning

### Update Process

1. **Rolling Updates**: Zero-downtime deployments
2. **Rollback**: Automatic rollback on failure
3. **Testing**: Comprehensive test suite
4. **Monitoring**: Real-time health monitoring

For additional support, check the troubleshooting guide or create an issue in the repository.