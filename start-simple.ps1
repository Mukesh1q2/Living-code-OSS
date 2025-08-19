# Simple starter script for Vidya Quantum Interface
param(
    [string]$Action = "help"
)

switch ($Action.ToLower()) {
    "setup" {
        Write-Host "🚀 Setting up Vidya Quantum Interface..." -ForegroundColor Green
        
        # Check Python
        try {
            $pythonVersion = python --version
            Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
        } catch {
            Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
            exit 1
        }
        
        # Check Node.js
        try {
            $nodeVersion = node --version
            Write-Host "✅ Found Node.js: $nodeVersion" -ForegroundColor Green
        } catch {
            Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
            exit 1
        }
        
        # Create virtual environment
        if (!(Test-Path "venv")) {
            Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
            python -m venv venv
        }
        
        # Activate and install Python dependencies
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
        python -m pip install --upgrade pip
        pip install fastapi uvicorn redis pydantic python-multipart pytest
        
        # Create basic project structure
        Write-Host "Creating project structure..." -ForegroundColor Yellow
        if (!(Test-Path "vidya_quantum_interface")) {
            New-Item -ItemType Directory -Path "vidya_quantum_interface" -Force | Out-Null
        }
        
        # Create basic API server
        $apiContent = @'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="Vidya Quantum Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Vidya Quantum Interface API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/sanskrit/analyze")
async def analyze_sanskrit(data: dict):
    text = data.get("text", "")
    return {
        "input": text,
        "analysis": {
            "words": text.split(),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    }
'@
        $apiContent | Out-File -FilePath "vidya_quantum_interface\api_server.py" -Encoding UTF8
        
        # Create __init__.py
        '"""Vidya Quantum Interface"""' | Out-File -FilePath "vidya_quantum_interface\__init__.py" -Encoding UTF8
        
        # Setup frontend
        if (!(Test-Path "frontend")) {
            Write-Host "Setting up frontend..." -ForegroundColor Yellow
            New-Item -ItemType Directory -Path "frontend" -Force | Out-Null
            Set-Location "frontend"
            
            # Create package.json
            $packageJson = @'
{
  "name": "vidya-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.1.1",
    "vite": "^5.0.0"
  }
}
'@
            $packageJson | Out-File -FilePath "package.json" -Encoding UTF8
            
            # Install npm dependencies
            npm install
            
            # Create basic React app
            New-Item -ItemType Directory -Path "src" -Force | Out-Null
            
            $appContent = @'
import React, { useState } from 'react'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)

  const analyze = async () => {
    const response = await fetch('/api/sanskrit/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    const data = await response.json()
    setResult(data)
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>🕉️ Vidya Quantum Interface</h1>
      <textarea 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter Sanskrit text..."
        style={{ width: '100%', height: '100px', margin: '10px 0' }}
      />
      <button onClick={analyze} style={{ padding: '10px 20px' }}>
        Analyze
      </button>
      {result && (
        <div style={{ marginTop: '20px', padding: '10px', background: '#f0f0f0' }}>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export default App
'@
            $appContent | Out-File -FilePath "src\App.jsx" -Encoding UTF8
            
            $mainContent = @'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(<App />)
'@
            $mainContent | Out-File -FilePath "src\main.jsx" -Encoding UTF8
            
            $indexHtml = @'
<!DOCTYPE html>
<html>
<head>
  <title>Vidya Quantum Interface</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
'@
            $indexHtml | Out-File -FilePath "index.html" -Encoding UTF8
            
            $viteConfig = @'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
'@
            $viteConfig | Out-File -FilePath "vite.config.js" -Encoding UTF8
            
            Set-Location ".."
        }
        
        Write-Host "✅ Setup complete!" -ForegroundColor Green
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. .\start-simple.ps1 backend" -ForegroundColor Cyan
        Write-Host "2. .\start-simple.ps1 frontend (in new terminal)" -ForegroundColor Cyan
    }
    
    "backend" {
        Write-Host "🔧 Starting backend..." -ForegroundColor Blue
        & ".\venv\Scripts\Activate.ps1"
        python -m uvicorn vidya_quantum_interface.api_server:app --reload --host 0.0.0.0 --port 8000
    }
    
    "frontend" {
        Write-Host "🌐 Starting frontend..." -ForegroundColor Blue
        Set-Location "frontend"
        npm run dev
    }
    
    default {
        Write-Host "Vidya Quantum Interface - Simple Runner" -ForegroundColor Green
        Write-Host "=====================================" -ForegroundColor Blue
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "  .\start-simple.ps1 setup     # First time setup" -ForegroundColor Cyan
        Write-Host "  .\start-simple.ps1 backend   # Start backend server" -ForegroundColor Cyan
        Write-Host "  .\start-simple.ps1 frontend  # Start frontend server" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "First time? Run: .\start-simple.ps1 setup" -ForegroundColor Green
    }
}