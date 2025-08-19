# Start Vidya Quantum Interface Backend
$ErrorActionPreference = "Continue"

Write-Host "🚀 Starting Vidya Quantum Interface Backend..." -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment not found. Run .\setup-local.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Check if .env exists
if (!(Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating default configuration..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
}

# Load environment variables
if (Test-Path ".env") {
    Write-Host "Loading environment variables..." -ForegroundColor Gray
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$" -and !$_.StartsWith("#")) {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

# Create basic API server if it doesn't exist
if (!(Test-Path "vidya_quantum_interface")) {
    Write-Host "Creating basic API server structure..." -ForegroundColor Gray
    New-Item -ItemType Directory -Path "vidya_quantum_interface" -Force | Out-Null
}

# Check if API server exists, create basic one if not
$apiServerPath = "vidya_quantum_interface\api_server.py"
if (!(Test-Path $apiServerPath)) {
    Write-Host "Creating basic API server..." -ForegroundColor Gray
    
    $apiServerContent = @"
"""
Vidya Quantum Interface API Server
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

app = FastAPI(
    title="Vidya Quantum Interface",
    description="Sanskrit AI Consciousness Interface",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Vidya Quantum Interface API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "vidya-quantum-interface"}

@app.post("/api/sanskrit/analyze")
async def analyze_sanskrit(text: dict):
    """Basic Sanskrit analysis endpoint"""
    input_text = text.get("text", "")
    
    # Basic response for now
    return {
        "input": input_text,
        "analysis": {
            "words": input_text.split(),
            "word_count": len(input_text.split()),
            "character_count": len(input_text),
            "status": "analyzed"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {"metrics": "basic_metrics_placeholder"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
"@
    
    $apiServerContent | Out-File -FilePath $apiServerPath -Encoding UTF8
}

# Create __init__.py if it doesn't exist
$initPath = "vidya_quantum_interface\__init__.py"
if (!(Test-Path $initPath)) {
    '"""Vidya Quantum Interface Package"""' | Out-File -FilePath $initPath -Encoding UTF8
}

Write-Host "Starting FastAPI server..." -ForegroundColor Blue
Write-Host "Backend will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Blue

# Start the server
try {
    python -m uvicorn vidya_quantum_interface.api_server:app --reload --host 0.0.0.0 --port 8000
} catch {
    Write-Host "❌ Failed to start backend server" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Make sure you've run .\setup-local.ps1 first" -ForegroundColor Yellow
}