param(
  [string]$BaseUrl = "http://127.0.0.1:1234/v1",
  [string]$TextModel = "gpt-oss-20b",
  [string]$ApiKey = "lm-studio",
  [string]$Host = "127.0.0.1",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

# Resolve project root (this script is expected to be in the scripts/ dir)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Info "Project root: $ProjectRoot"

# Activate venv
$VenvActivate = Join-Path $ProjectRoot "sanskrit-env\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
  Write-Info "Activating virtual environment..."
  . $VenvActivate
  Write-Ok "Virtual environment activated"
}
else {
  Write-Warn "Virtual environment not found at $VenvActivate"
  Write-Warn "Proceeding with system Python. Consider creating/using the provided venv."
}

# Ensure required Python packages
Write-Info "Ensuring required Python packages are installed..."
python -m pip install --disable-pip-version-check -q -e . | Out-Null
python -m pip install --disable-pip-version-check -q requests | Out-Null
Write-Ok "Dependencies installed/updated"

# Configure LM Studio (OpenAI-compatible) env vars
$env:LMSTUDIO_BASE_URL = $BaseUrl
$env:LMSTUDIO_TEXT_MODEL = $TextModel
$env:OPENAI_API_KEY = $ApiKey

Write-Info "LM Studio endpoint: $($env:LMSTUDIO_BASE_URL)"
Write-Info "Text model: $($env:LMSTUDIO_TEXT_MODEL)"
Write-Info "Backend API: http://$Host:$Port"

# Optional: quick connectivity check to LM Studio (non-fatal)
try {
  Write-Info "Checking LM Studio availability..."
  $resp = Invoke-WebRequest -Method Post -Uri ("$($env:LMSTUDIO_BASE_URL.TrimEnd('/'))/chat/completions") -Headers @{ Authorization = "Bearer $ApiKey"; "Content-Type" = "application/json" } -Body '{"model":"' + $TextModel + '","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' -TimeoutSec 3
  if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) { Write-Ok "LM Studio reachable" } else { Write-Warn "LM Studio responded with status $($resp.StatusCode)" }
}
catch { Write-Warn "LM Studio check failed: $($_.Exception.Message)" }

# Start backend API (FastAPI via uvicorn)
Write-Info "Starting Vidya API server..."
python -m uvicorn vidya_quantum_interface.api_server:app --reload --host $Host --port $Port

