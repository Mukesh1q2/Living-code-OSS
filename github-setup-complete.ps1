# Complete GitHub Setup for Vidya Quantum Interface
# Repository: https://github.com/Mukesh1q2/Living-code-OSS

Write-Host "🚀 Complete GitHub Setup - Vidya Quantum Interface" -ForegroundColor Green
Write-Host "Repository: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Blue

# Step 1: Upload all files
Write-Host "`n📤 Step 1: Uploading all files to GitHub..." -ForegroundColor Yellow
& ".\upload-to-github.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Upload failed. Please check the error messages above." -ForegroundColor Red
    exit 1
}

# Step 2: Repository configuration instructions
Write-Host "`n⚙️  Step 2: Repository Configuration" -ForegroundColor Yellow
Write-Host "Please complete these steps on GitHub:" -ForegroundColor White
Write-Host ""

Write-Host "🏷️  Repository Description:" -ForegroundColor Cyan
Write-Host "🕉️ Sanskrit AI Consciousness Interface - Where Ancient Wisdom Meets Quantum Computing. Advanced Sanskrit text analysis with immersive quantum visualizations, AI integration, and production-ready deployment." -ForegroundColor White
Write-Host ""

Write-Host "🏷️  Repository Topics (comma-separated):" -ForegroundColor Cyan
$topics = @(
    "sanskrit", "ai", "quantum-computing", "consciousness", "nlp", "linguistics",
    "ancient-wisdom", "morphology", "panini", "grammar", "vedic", "fastapi",
    "react", "typescript", "webgl", "visualization", "docker", "kubernetes",
    "aws", "gcp", "azure", "machine-learning", "transformers", "python"
)
Write-Host ($topics -join ", ") -ForegroundColor White
Write-Host ""

Write-Host "🌐 Website URL (optional):" -ForegroundColor Cyan
Write-Host "https://mukesh1q2.github.io/Living-code-OSS" -ForegroundColor White
Write-Host ""

# Step 3: GitHub CLI setup (if available)
Write-Host "`n🔧 Step 3: Automated Configuration (if GitHub CLI is available)" -ForegroundColor Yellow

try {
    $ghVersion = gh --version 2>$null
    if ($ghVersion) {
        Write-Host "✅ GitHub CLI found. Attempting automated setup..." -ForegroundColor Green
        
        # Set repository description
        try {
            gh repo edit Mukesh1q2/Living-code-OSS --description "🕉️ Sanskrit AI Consciousness Interface - Where Ancient Wisdom Meets Quantum Computing. Advanced Sanskrit text analysis with immersive quantum visualizations, AI integration, and production-ready deployment."
            Write-Host "✅ Repository description updated" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not update description automatically" -ForegroundColor Yellow
        }
        
        # Add topics
        try {
            gh repo edit Mukesh1q2/Living-code-OSS --add-topic ($topics -join ",")
            Write-Host "✅ Repository topics added" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not add topics automatically" -ForegroundColor Yellow
        }
        
        # Enable features
        try {
            gh repo edit Mukesh1q2/Living-code-OSS --enable-issues --enable-projects --enable-wiki
            Write-Host "✅ Repository features enabled" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not enable features automatically" -ForegroundColor Yellow
        }
        
    } else {
        Write-Host "⚠️  GitHub CLI not found. Manual setup required." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  GitHub CLI not available. Please configure manually." -ForegroundColor Yellow
}

# Step 4: Manual configuration checklist
Write-Host "`n📋 Step 4: Manual Configuration Checklist" -ForegroundColor Yellow
Write-Host "Visit: https://github.com/Mukesh1q2/Living-code-OSS/settings" -ForegroundColor Cyan
Write-Host ""

$checklist = @(
    "✅ Repository description added",
    "✅ Topics/tags added for discoverability", 
    "✅ Website URL added (optional)",
    "✅ Issues enabled",
    "✅ Projects enabled", 
    "✅ Wiki enabled",
    "✅ Discussions enabled (recommended)",
    "✅ Security features enabled (Dependabot, etc.)",
    "✅ Branch protection rules configured",
    "✅ GitHub Actions enabled"
)

foreach ($item in $checklist) {
    Write-Host "  $item" -ForegroundColor White
}

# Step 5: GitHub Actions secrets
Write-Host "`n🔐 Step 5: GitHub Actions Secrets (for deployment)" -ForegroundColor Yellow
Write-Host "Go to: https://github.com/Mukesh1q2/Living-code-OSS/settings/secrets/actions" -ForegroundColor Cyan
Write-Host ""
Write-Host "Add these secrets for deployment:" -ForegroundColor White

$secrets = @(
    "HUGGINGFACE_API_KEY - Your HuggingFace API key",
    "OPENAI_API_KEY - Your OpenAI API key (optional)",
    "ANTHROPIC_API_KEY - Your Anthropic API key (optional)",
    "DOCKER_REGISTRY - Docker registry URL (optional)",
    "KUBECONFIG - Kubernetes configuration (for K8s deployment)",
    "GRAFANA_PASSWORD - Grafana admin password"
)

foreach ($secret in $secrets) {
    Write-Host "  • $secret" -ForegroundColor Gray
}

# Step 6: GitHub Pages setup
Write-Host "`n📄 Step 6: GitHub Pages (optional)" -ForegroundColor Yellow
Write-Host "Go to: https://github.com/Mukesh1q2/Living-code-OSS/settings/pages" -ForegroundColor Cyan
Write-Host "• Source: Deploy from a branch" -ForegroundColor White
Write-Host "• Branch: main" -ForegroundColor White
Write-Host "• Folder: / (root) or /docs" -ForegroundColor White

# Step 7: Project board setup
Write-Host "`n📊 Step 7: Project Board (optional)" -ForegroundColor Yellow
Write-Host "Go to: https://github.com/Mukesh1q2/Living-code-OSS/projects" -ForegroundColor Cyan
Write-Host "• Create new project" -ForegroundColor White
Write-Host "• Add columns: Backlog, In Progress, Review, Done" -ForegroundColor White
Write-Host "• Link to issues and pull requests" -ForegroundColor White

# Final summary
Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=================" -ForegroundColor Blue
Write-Host ""
Write-Host "Your Vidya Quantum Interface is now on GitHub:" -ForegroundColor White
Write-Host "🔗 Repository: https://github.com/Mukesh1q2/Living-code-OSS" -ForegroundColor Cyan
Write-Host "📖 README: https://github.com/Mukesh1q2/Living-code-OSS#readme" -ForegroundColor Cyan
Write-Host "🐛 Issues: https://github.com/Mukesh1q2/Living-code-OSS/issues" -ForegroundColor Cyan
Write-Host "🔄 Actions: https://github.com/Mukesh1q2/Living-code-OSS/actions" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps for users:" -ForegroundColor Yellow
Write-Host "1. git clone https://github.com/Mukesh1q2/Living-code-OSS.git" -ForegroundColor White
Write-Host "2. cd Living-code-OSS" -ForegroundColor White
Write-Host "3. .\start-simple.ps1 setup" -ForegroundColor White
Write-Host "4. .\start-simple.ps1 backend" -ForegroundColor White
Write-Host "5. .\start-simple.ps1 frontend" -ForegroundColor White
Write-Host ""
Write-Host "🕉️ Sanskrit + AI + Quantum Consciousness = Vidya ✨" -ForegroundColor Green