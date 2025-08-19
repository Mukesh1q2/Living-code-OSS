#!/bin/bash

# Vidya Quantum Interface Deployment Script
set -e

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
REGISTRY=${DOCKER_REGISTRY:-localhost:5000}

echo "🚀 Deploying Vidya Quantum Interface"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Registry: $REGISTRY"

# Load environment-specific configuration
if [ -f ".env.$ENVIRONMENT" ]; then
    echo "Loading environment configuration from .env.$ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | xargs)
else
    echo "⚠️  No environment file found for $ENVIRONMENT, using defaults"
fi

# Build and tag images
echo "📦 Building Docker images..."

# Backend image
docker build -t $REGISTRY/vidya-backend:$VERSION -f Dockerfile.backend .
docker tag $REGISTRY/vidya-backend:$VERSION $REGISTRY/vidya-backend:latest

# Frontend image
docker build -t $REGISTRY/vidya-frontend:$VERSION -f vidya-quantum-interface/Dockerfile.frontend ./vidya-quantum-interface
docker tag $REGISTRY/vidya-frontend:$VERSION $REGISTRY/vidya-frontend:latest

# Push images to registry
if [ "$REGISTRY" != "localhost:5000" ]; then
    echo "📤 Pushing images to registry..."
    docker push $REGISTRY/vidya-backend:$VERSION
    docker push $REGISTRY/vidya-backend:latest
    docker push $REGISTRY/vidya-frontend:$VERSION
    docker push $REGISTRY/vidya-frontend:latest
fi

# Deploy based on environment
case $ENVIRONMENT in
    "local")
        echo "🏠 Deploying locally with Docker Compose..."
        docker-compose down
        docker-compose up -d
        ;;
    "staging"|"production")
        if command -v kubectl &> /dev/null; then
            echo "☸️  Deploying to Kubernetes..."
            ./scripts/k8s-deploy.sh $ENVIRONMENT $VERSION
        elif command -v docker-compose &> /dev/null; then
            echo "🐳 Deploying with Docker Compose..."
            docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml up -d
        else
            echo "❌ No deployment method available"
            exit 1
        fi
        ;;
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Health check
echo "🏥 Performing health checks..."
sleep 10

# Check backend health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    exit 1
fi

# Check frontend health
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend health check failed"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost"

if [ "$ENVIRONMENT" != "local" ]; then
    echo "Monitoring: http://localhost:3001 (Grafana)"
    echo "Metrics: http://localhost:9090 (Prometheus)"
fi