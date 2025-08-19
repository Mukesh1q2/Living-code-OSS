# Upload Vidya Quantum Interface to GitHub Repository
# Repository: https://github.com/Mukesh1q2/Living-code-OSS

param(
    [string]$CommitMessage = "feat: complete Vidya Quantum Interface implementation",
    [switch]$Force = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Upload to GitHub - Vidya Quantum Interface" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\upload-to-github.ps1                    # Upload with default message" -ForegroundColor Cyan
    Write-Host "  .\upload-to-github.ps1 -CommitMessage 'Custom message'" -ForegroundColor Cyan
    Write-Host "  .\upload-to-github.ps1 -Force             # Force push (use carefully)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Repository: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor White
    exit 0
}

Write-Host "🚀 Uploading Vidya Quantum Interface to GitHub" -ForegroundColor Green
Write-Host "Repository: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Blue

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found. Please install Git from: https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# Check if we're in a git repository
if (!(Test-Path ".git")) {
    Write-Host "📁 Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git repository found" -ForegroundColor Green
}

# Check if remote origin exists
$remoteExists = git remote get-url origin 2>$null
if (!$remoteExists) {
    Write-Host "🔗 Adding remote origin..." -ForegroundColor Yellow
    git remote add origin https://github.com/Mukesh1q2/Living-code-OSS.git
    Write-Host "✅ Remote origin added" -ForegroundColor Green
} else {
    Write-Host "✅ Remote origin exists: $remoteExists" -ForegroundColor Green
    # Update remote URL to ensure it's correct
    git remote set-url origin https://github.com/Mukesh1q2/Living-code-OSS.git
}

# Check git status
Write-Host "📊 Checking repository status..." -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "📝 Found changes to commit:" -ForegroundColor Yellow
    git status --short
} else {
    Write-Host "✅ No changes to commit" -ForegroundColor Green
}

# Add all files
Write-Host "📦 Adding all files..." -ForegroundColor Yellow
git add .

# Create comprehensive commit message
$fullCommitMessage = @"
$CommitMessage

🕉️ Vidya Quantum Interface - Complete Implementation

## Features Added:
- ✅ Sanskrit morphological analysis engine with Paninian grammar rules
- ✅ Quantum consciousness visualization with WebGL rendering
- ✅ React/TypeScript frontend with accessibility features
- ✅ FastAPI Python backend with comprehensive API
- ✅ Docker containerization for all services
- ✅ Kubernetes deployment manifests
- ✅ Multi-cloud support (AWS, GCP, Azure)
- ✅ Comprehensive monitoring with Prometheus/Grafana
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Complete documentation and setup guides

## Technical Stack:
- Frontend: React 18, TypeScript, Three.js, WebGL
- Backend: Python 3.11, FastAPI, Redis, SQLAlchemy
- AI/ML: HuggingFace Transformers, PyTorch, Multi-model support
- Deployment: Docker, Kubernetes, Multi-cloud
- Monitoring: Prometheus, Grafana, Structured logging
- Testing: Pytest, Jest, Comprehensive test coverage

## Repository Structure:
- 📁 vidya-quantum-interface/ - React frontend
- 📁 vidya_quantum_interface/ - Python backend
- 📁 sanskrit_rewrite_engine/ - Sanskrit processing
- 📁 cloud/ - Cloud integrations
- 📁 config/ - Environment configurations
- 📁 monitoring/ - Logging and metrics
- 📁 k8s/ - Kubernetes manifests
- 📁 scripts/ - Deployment scripts
- 📁 .github/workflows/ - CI/CD pipelines

## Getting Started:
1. Clone: git clone https://github.com/Mukesh1q2/Living-code-OSS.git
2. Setup: .\start-simple.ps1 setup
3. Backend: .\start-simple.ps1 backend
4. Frontend: .\start-simple.ps1 frontend
5. Access: http://localhost:5173

Ready for production deployment with comprehensive monitoring,
security features, and scalable architecture.

Sanskrit + AI + Quantum Consciousness = Vidya ✨
"@

# Commit changes
Write-Host "💾 Creating commit..." -ForegroundColor Yellow
git commit -m $fullCommitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  No changes to commit or commit failed" -ForegroundColor Yellow
} else {
    Write-Host "✅ Commit created successfully" -ForegroundColor Green
}

# Set main branch
Write-Host "🌿 Setting main branch..." -ForegroundColor Yellow
git branch -M main

# Push to GitHub
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "Repository: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor Cyan

try {
    if ($Force) {
        Write-Host "⚠️  Force pushing..." -ForegroundColor Yellow
        git push -f origin main
    } else {
        git push -u origin main
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "❌ Push failed. You may need to authenticate or resolve conflicts." -ForegroundColor Red
        Write-Host "Try running: git pull origin main" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Error pushing to GitHub: $_" -ForegroundColor Red
    Write-Host "💡 Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're authenticated with GitHub" -ForegroundColor White
    Write-Host "2. Check if you have write access to the repository" -ForegroundColor White
    Write-Host "3. Try: git pull origin main (if repository exists)" -ForegroundColor White
    Write-Host "4. Use: .\upload-to-github.ps1 -Force (if you want to overwrite)" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "🎉 Upload Complete!" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Blue
Write-Host ""
Write-Host "Repository URL: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor Cyan
Write-Host "Clone URL: git clone https://github.com/Mukesh1q2/Living-code-OSS.git" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit your repository on GitHub" -ForegroundColor White
Write-Host "2. Add repository description and topics" -ForegroundColor White
Write-Host "3. Enable GitHub Pages (if desired)" -ForegroundColor White
Write-Host "4. Configure branch protection rules" -ForegroundColor White
Write-Host "5. Add collaborators if needed" -ForegroundColor White
Write-Host ""
Write-Host "🕉️ Vidya Quantum Interface is now live on GitHub! ✨" -ForegroundColor Green