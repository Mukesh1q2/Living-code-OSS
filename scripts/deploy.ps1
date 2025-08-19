# Vidya Quantum Interface Deployment Script (PowerShell)
param(
    [string]$Environment = "staging",
    [string]$Version = "latest",
    [string]$Registry = "localhost:5000"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying Vidya Quantum Interface" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Cyan
Write-Host "Registry: $Registry" -ForegroundColor Cyan

# Load environment-specific configuration
$envFile = ".env.$Environment"
if (Test-Path $envFile) {
    Write-Host "Loading environment configuration from $envFile" -ForegroundColor Yellow
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
} else {
    Write-Host "⚠️  No environment file found for $Environment, using defaults" -ForegroundColor Yellow
}

# Build and tag images
Write-Host "📦 Building Docker images..." -ForegroundColor Blue

# Backend image
Write-Host "Building backend image..." -ForegroundColor Gray
docker build -t "$Registry/vidya-backend:$Version" -f Dockerfile.backend .
if ($LASTEXITCODE -ne 0) { throw "Backend build failed" }

docker tag "$Registry/vidya-backend:$Version" "$Registry/vidya-backend:latest"

# Frontend image
Write-Host "Building frontend image..." -ForegroundColor Gray
docker build -t "$Registry/vidya-frontend:$Version" -f vidya-quantum-interface/Dockerfile.frontend ./vidya-quantum-interface
if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }

docker tag "$Registry/vidya-frontend:$Version" "$Registry/vidya-frontend:latest"

# Push images to registry
if ($Registry -ne "localhost:5000") {
    Write-Host "📤 Pushing images to registry..." -ForegroundColor Blue
    docker push "$Registry/vidya-backend:$Version"
    docker push "$Registry/vidya-backend:latest"
    docker push "$Registry/vidya-frontend:$Version"
    docker push "$Registry/vidya-frontend:latest"
}

# Deploy based on environment
switch ($Environment) {
    "local" {
        Write-Host "🏠 Deploying locally with Docker Compose..." -ForegroundColor Green
        docker-compose down
        docker-compose up -d
    }
    { $_ -in @("staging", "production") } {
        if (Get-Command kubectl -ErrorAction SilentlyContinue) {
            Write-Host "☸️  Deploying to Kubernetes..." -ForegroundColor Green
            & "./scripts/k8s-deploy.ps1" $Environment $Version
        } elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) {
            Write-Host "🐳 Deploying with Docker Compose..." -ForegroundColor Green
            docker-compose -f docker-compose.yml -f "docker-compose.$Environment.yml" up -d
        } else {
            throw "❌ No deployment method available"
        }
    }
    default {
        throw "❌ Unknown environment: $Environment"
    }
}

# Health check
Write-Host "🏥 Performing health checks..." -ForegroundColor Blue
Start-Sleep -Seconds 10

# Check backend health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is healthy" -ForegroundColor Green
    } else {
        throw "Backend returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "❌ Backend health check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Check frontend health
try {
    $response = Invoke-WebRequest -Uri "http://localhost/health" -UseBasicParsing -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Frontend is healthy" -ForegroundColor Green
    } else {
        throw "Frontend returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "❌ Frontend health check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost" -ForegroundColor Cyan

if ($Environment -ne "local") {
    Write-Host "Monitoring: http://localhost:3001 (Grafana)" -ForegroundColor Cyan
    Write-Host "Metrics: http://localhost:9090 (Prometheus)" -ForegroundColor Cyan
}