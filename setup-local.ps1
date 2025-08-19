# Vidya Quantum Interface - Local Setup Script
param(
    [switch]$SkipRedis = $false
)

$ErrorActionPreference = "Continue"

Write-Host "🚀 Setting up Vidya Quantum Interface for Local Development" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Blue

# Step 1: Create virtual environment for Python
Write-Host "`n📦 Step 1: Setting up Python environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists, activating..." -ForegroundColor Gray
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Gray
& ".\venv\Scripts\Activate.ps1"

# Step 2: Install Python dependencies
Write-Host "`n📚 Step 2: Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "Upgrading pip..." -ForegroundColor Gray
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Host "Installing requirements.txt..." -ForegroundColor Gray
    pip install -r requirements.txt
} else {
    Write-Host "⚠️  requirements.txt not found, installing basic dependencies..." -ForegroundColor Yellow
    pip install fastapi uvicorn redis pydantic python-multipart
}

if (Test-Path "requirements-dev.txt") {
    Write-Host "Installing development dependencies..." -ForegroundColor Gray
    pip install -r requirements-dev.txt
} else {
    Write-Host "Installing basic dev dependencies..." -ForegroundColor Gray
    pip install pytest pytest-cov black flake8 mypy
}

# Install package in development mode
Write-Host "Installing package in development mode..." -ForegroundColor Gray
pip install -e .

# Step 3: Setup Node.js dependencies
Write-Host "`n🌐 Step 3: Setting up Node.js dependencies..." -ForegroundColor Yellow
if (Test-Path "vidya-quantum-interface") {
    Set-Location "vidya-quantum-interface"
    
    if (Test-Path "package.json") {
        Write-Host "Installing Node.js dependencies..." -ForegroundColor Gray
        npm install
    } else {
        Write-Host "⚠️  package.json not found in vidya-quantum-interface/" -ForegroundColor Yellow
    }
    
    Set-Location ".."
} else {
    Write-Host "⚠️  vidya-quantum-interface directory not found" -ForegroundColor Yellow
}

# Step 4: Create necessary directories
Write-Host "`n📁 Step 4: Creating necessary directories..." -ForegroundColor Yellow
$directories = @("logs", "data", "models", "sutra_rules")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Gray
    }
}

# Step 5: Redis setup (optional)
if (!$SkipRedis) {
    Write-Host "`n🔴 Step 5: Redis setup..." -ForegroundColor Yellow
    Write-Host "Checking if Redis is available..." -ForegroundColor Gray
    
    try {
        $redisTest = redis-cli ping 2>&1
        if ($redisTest -eq "PONG") {
            Write-Host "✅ Redis is running!" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️  Redis not found. You have options:" -ForegroundColor Yellow
        Write-Host "   1. Install Redis locally" -ForegroundColor Cyan
        Write-Host "   2. Use Docker: docker run -d -p 6379:6379 redis:7-alpine" -ForegroundColor Cyan
        Write-Host "   3. Skip Redis (some features may not work)" -ForegroundColor Cyan
    }
}

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Blue
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\start-backend.ps1" -ForegroundColor Cyan
Write-Host "2. In another terminal: .\start-frontend.ps1" -ForegroundColor Cyan
Write-Host "3. Open http://localhost:5173 in your browser" -ForegroundColor Cyan