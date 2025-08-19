# GitHub Repository Initialization Script for Vidya Quantum Interface
param(
    [string]$RepoName = "vidya-quantum-interface",
    [string]$GitHubUsername = "",
    [switch]$Private = $false,
    [switch]$Help = $false
)

if ($Help -or $GitHubUsername -eq "") {
    Write-Host "GitHub Repository Initialization for Vidya Quantum Interface" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\init-github-repo.ps1 -GitHubUsername 'yourusername'" -ForegroundColor Cyan
    Write-Host "  .\init-github-repo.ps1 -GitHubUsername 'yourusername' -Private" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -GitHubUsername : Your GitHub username (required)" -ForegroundColor Cyan
    Write-Host "  -RepoName       : Repository name (default: vidya-quantum-interface)" -ForegroundColor Cyan
    Write-Host "  -Private        : Create private repository (default: public)" -ForegroundColor Cyan
    Write-Host "  -Help           : Show this help message" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Prerequisites:" -ForegroundColor Yellow
    Write-Host "1. Install GitHub CLI: https://cli.github.com/" -ForegroundColor White
    Write-Host "2. Login to GitHub: gh auth login" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host "🚀 Initializing Vidya Quantum Interface GitHub Repository" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Blue

# Check if GitHub CLI is installed
try {
    $ghVersion = gh --version
    Write-Host "✅ GitHub CLI found: $($ghVersion.Split("`n")[0])" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI not found. Please install it from: https://cli.github.com/" -ForegroundColor Red
    exit 1
}

# Check if user is logged in
try {
    $ghUser = gh auth status 2>&1
    if ($ghUser -match "Logged in") {
        Write-Host "✅ GitHub CLI authenticated" -ForegroundColor Green
    } else {
        Write-Host "❌ Not logged in to GitHub. Run: gh auth login" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ GitHub authentication failed. Run: gh auth login" -ForegroundColor Red
    exit 1
}

# Initialize git repository if not already initialized
if (!(Test-Path ".git")) {
    Write-Host "📁 Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git repository already exists" -ForegroundColor Green
}

# Create .gitignore if it doesn't exist
if (!(Test-Path ".gitignore")) {
    Write-Host "⚠️  .gitignore not found. Please ensure it exists." -ForegroundColor Yellow
} else {
    Write-Host "✅ .gitignore found" -ForegroundColor Green
}

# Add all files to git
Write-Host "📝 Adding files to Git..." -ForegroundColor Yellow
git add .

# Create initial commit
Write-Host "💾 Creating initial commit..." -ForegroundColor Yellow
git commit -m "feat: initial commit - Vidya Quantum Interface

🕉️ Sanskrit AI Consciousness Interface - Where Ancient Wisdom Meets Quantum Computing

Features:
- Sanskrit morphological analysis engine
- Quantum consciousness visualization
- React/TypeScript frontend
- FastAPI Python backend
- Docker containerization
- Kubernetes deployment
- Multi-cloud support (AWS, GCP, Azure)
- Comprehensive monitoring and logging
- CI/CD pipeline with GitHub Actions

This initial release provides a complete foundation for Sanskrit text analysis
with immersive quantum visualizations and production-ready deployment options."

# Create GitHub repository
Write-Host "🌐 Creating GitHub repository..." -ForegroundColor Yellow

$repoDescription = "🕉️ Sanskrit AI Consciousness Interface - Where Ancient Wisdom Meets Quantum Computing. Advanced Sanskrit text analysis with immersive quantum visualizations, AI integration, and production-ready deployment."

$visibility = if ($Private) { "--private" } else { "--public" }

try {
    gh repo create $RepoName $visibility --description $repoDescription --clone=false
    Write-Host "✅ GitHub repository created: https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Repository might already exist or there was an error" -ForegroundColor Yellow
    Write-Host "Error: $_" -ForegroundColor Red
}

# Add remote origin
Write-Host "🔗 Adding remote origin..." -ForegroundColor Yellow
try {
    git remote add origin "https://github.com/$GitHubUsername/$RepoName.git"
    Write-Host "✅ Remote origin added" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Remote origin might already exist" -ForegroundColor Yellow
    git remote set-url origin "https://github.com/$GitHubUsername/$RepoName.git"
    Write-Host "✅ Remote origin updated" -ForegroundColor Green
}

# Push to GitHub
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Yellow
try {
    git branch -M main
    git push -u origin main
    Write-Host "✅ Code pushed to GitHub successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to push to GitHub" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "You may need to authenticate or check your permissions" -ForegroundColor Yellow
}

# Set up repository settings
Write-Host "⚙️  Configuring repository settings..." -ForegroundColor Yellow

# Add topics/tags
$topics = @(
    "sanskrit", "ai", "quantum-computing", "consciousness", "nlp", "linguistics",
    "ancient-wisdom", "morphology", "panini", "grammar", "vedic", "fastapi",
    "react", "typescript", "webgl", "visualization", "docker", "kubernetes",
    "aws", "gcp", "azure", "machine-learning", "transformers", "python"
)

try {
    gh repo edit --add-topic ($topics -join ",")
    Write-Host "✅ Repository topics added" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not add topics automatically" -ForegroundColor Yellow
}

# Enable GitHub Pages (if public repository)
if (!$Private) {
    try {
        gh api repos/$GitHubUsername/$RepoName/pages -X POST -f source[branch]=main -f source[path]=/docs
        Write-Host "✅ GitHub Pages enabled" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Could not enable GitHub Pages automatically" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🎉 Repository setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""
Write-Host "Repository URL: https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Cyan
Write-Host "Clone URL: git clone https://github.com/$GitHubUsername/$RepoName.git" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit your repository on GitHub" -ForegroundColor White
Write-Host "2. Add a detailed description and website URL" -ForegroundColor White
Write-Host "3. Configure branch protection rules" -ForegroundColor White
Write-Host "4. Set up GitHub Actions secrets for deployment" -ForegroundColor White
Write-Host "5. Invite collaborators if needed" -ForegroundColor White
Write-Host ""
Write-Host "Repository Topics Added:" -ForegroundColor Yellow
Write-Host ($topics -join ", ") -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🕉️✨" -ForegroundColor Green