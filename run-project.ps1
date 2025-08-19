# Complete Vidya Quantum Interface Runner
param(
    [switch]$Setup = $false,
    [switch]$Backend = $false,
    [switch]$Frontend = $false,
    [switch]$Both = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Vidya Quantum Interface - Local Development Runner" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Blue
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run-project.ps1 -Setup      # Initial setup (run this first)" -ForegroundColor Cyan
    Write-Host "  .\run-project.ps1 -Backend    # Start backend only" -ForegroundColor Cyan
    Write-Host "  .\run-project.ps1 -Frontend   # Start frontend only" -ForegroundColor Cyan
    Write-Host "  .\run-project.ps1 -Both       # Start both (requires 2 terminals)" -ForegroundColor Cyan
    Write-Host "  .\run-project.ps1 -Help       # Show this help" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "For first time setup:" -ForegroundColor Yellow
    Write-Host "1. .\run-project.ps1 -Setup" -ForegroundColor Green
    Write-Host "2. .\run-project.ps1 -Backend   (in one terminal)" -ForegroundColor Green
    Write-Host "3. .\run-project.ps1 -Frontend  (in another terminal)" -ForegroundColor Green
    exit 0
}

if ($Setup) {
    Write-Host "🚀 Running initial setup..." -ForegroundColor Green
    & ".\setup-local.ps1"
    exit 0
}

if ($Backend) {
    Write-Host "🔧 Starting backend server..." -ForegroundColor Blue
    & ".\start-backend.ps1"
    exit 0
}

if ($Frontend) {
    Write-Host "🌐 Starting frontend server..." -ForegroundColor Blue
    & ".\start-frontend.ps1"
    exit 0
}

if ($Both) {
    Write-Host "⚠️  Starting both servers requires two separate terminals!" -ForegroundColor Yellow
    Write-Host "Opening backend in current terminal..." -ForegroundColor Gray
    Write-Host "Please open another PowerShell window and run: .\run-project.ps1 -Frontend" -ForegroundColor Cyan
    Start-Sleep -Seconds 3
    & ".\start-backend.ps1"
    exit 0
}

# Default: Show help
Write-Host "Vidya Quantum Interface - Local Development" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Blue
Write-Host ""
Write-Host "No parameters specified. Here's what you can do:" -ForegroundColor Yellow
Write-Host ""
Write-Host "First time setup:" -ForegroundColor Cyan
Write-Host "  .\run-project.ps1 -Setup" -ForegroundColor White
Write-Host ""
Write-Host "Start development servers:" -ForegroundColor Cyan
Write-Host "  .\run-project.ps1 -Backend    # Terminal 1" -ForegroundColor White
Write-Host "  .\run-project.ps1 -Frontend   # Terminal 2" -ForegroundColor White
Write-Host ""
Write-Host "Or get help:" -ForegroundColor Cyan
Write-Host "  .\run-project.ps1 -Help" -ForegroundColor White
Write-Host ""
Write-Host "Quick start for first time users:" -ForegroundColor Green
$response = Read-Host "Would you like to run the initial setup now? (y/n)"
if ($response -eq "y" -or $response -eq "Y" -or $response -eq "yes") {
    & ".\setup-local.ps1"
}