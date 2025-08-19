#!/bin/bash

# Kubernetes deployment script for Vidya Quantum Interface
set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE=${KUBERNETES_NAMESPACE:-vidya}

echo "☸️  Deploying to Kubernetes"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."

# ConfigMap for environment variables
kubectl apply -f k8s/configmap-$ENVIRONMENT.yaml -n $NAMESPACE

# Secrets (should be managed separately in production)
if [ -f "k8s/secrets-$ENVIRONMENT.yaml" ]; then
    kubectl apply -f k8s/secrets-$ENVIRONMENT.yaml -n $NAMESPACE
fi

# Redis deployment
kubectl apply -f k8s/redis.yaml -n $NAMESPACE

# Backend deployment
envsubst < k8s/backend.yaml | kubectl apply -f - -n $NAMESPACE

# Frontend deployment
envsubst < k8s/frontend.yaml | kubectl apply -f - -n $NAMESPACE

# Services
kubectl apply -f k8s/services.yaml -n $NAMESPACE

# Ingress (if available)
if [ -f "k8s/ingress-$ENVIRONMENT.yaml" ]; then
    kubectl apply -f k8s/ingress-$ENVIRONMENT.yaml -n $NAMESPACE
fi

# HPA (Horizontal Pod Autoscaler)
if [ "$ENVIRONMENT" = "production" ]; then
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE
fi

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/vidya-backend -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/vidya-frontend -n $NAMESPACE

# Show deployment status
echo "📊 Deployment status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "✅ Kubernetes deployment completed!"

# Get service URLs
if kubectl get ingress -n $NAMESPACE > /dev/null 2>&1; then
    echo "🌐 Application URLs:"
    kubectl get ingress -n $NAMESPACE
else
    echo "🔗 Port forward to access services:"
    echo "kubectl port-forward svc/vidya-frontend 8080:80 -n $NAMESPACE"
    echo "kubectl port-forward svc/vidya-backend 8000:8000 -n $NAMESPACE"
fi