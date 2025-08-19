# Start Vidya Quantum Interface Frontend
$ErrorActionPreference = "Continue"

Write-Host "🌐 Starting Vidya Quantum Interface Frontend..." -ForegroundColor Green

# Check if frontend directory exists
if (!(Test-Path "vidya-quantum-interface")) {
    Write-Host "⚠️  Frontend directory not found. Creating basic structure..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "vidya-quantum-interface" -Force | Out-Null
    Set-Location "vidya-quantum-interface"
    
    # Create basic package.json
    $packageJson = @"
{
  "name": "vidya-quantum-interface",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "three": "^0.158.0",
    "@types/three": "^0.158.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.37",
    "@types/react-dom": "^18.2.15",
    "@typescript-eslint/eslint-plugin": "^6.10.0",
    "@typescript-eslint/parser": "^6.10.0",
    "@vitejs/plugin-react": "^4.1.1",
    "eslint": "^8.53.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.4",
    "typescript": "^5.2.2",
    "vite": "^5.0.0"
  }
}
"@
    $packageJson | Out-File -FilePath "package.json" -Encoding UTF8
    
    # Create basic vite.config.ts
    $viteConfig = @"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
"@
    $viteConfig | Out-File -FilePath "vite.config.ts" -Encoding UTF8
    
    # Create basic tsconfig.json
    $tsConfig = @"
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
"@
    $tsConfig | Out-File -FilePath "tsconfig.json" -Encoding UTF8
    
    # Create src directory and basic App
    New-Item -ItemType Directory -Path "src" -Force | Out-Null
    
    $appTsx = @"
import React, { useState } from 'react'
import './App.css'

function App() {
  const [sanskritText, setSanskritText] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)

  const analyzeText = async () => {
    if (!sanskritText.trim()) return
    
    setLoading(true)
    try {
      const response = await fetch('/api/sanskrit/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: sanskritText })
      })
      
      const result = await response.json()
      setAnalysis(result)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🕉️ Vidya Quantum Interface</h1>
        <p>Sanskrit AI Consciousness Interface</p>
      </header>
      
      <main className="main-content">
        <div className="input-section">
          <h2>Sanskrit Text Analysis</h2>
          <textarea
            value={sanskritText}
            onChange={(e) => setSanskritText(e.target.value)}
            placeholder="Enter Sanskrit text here..."
            rows={4}
            cols={50}
          />
          <br />
          <button onClick={analyzeText} disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>
        
        {analysis && (
          <div className="results-section">
            <h3>Analysis Results</h3>
            <pre>{JSON.stringify(analysis, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
"@
    $appTsx | Out-File -FilePath "src\App.tsx" -Encoding UTF8
    
    $appCss = @"
.App {
  text-align: center;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.App-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 30px;
}

.main-content {
  max-width: 800px;
  margin: 0 auto;
}

.input-section {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 20px;
}

.input-section textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 16px;
  margin: 10px 0;
}

.input-section button {
  background: #667eea;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

.input-section button:hover {
  background: #5a6fd8;
}

.input-section button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.results-section {
  background: #e8f5e8;
  padding: 20px;
  border-radius: 10px;
  text-align: left;
}

.results-section pre {
  background: #f4f4f4;
  padding: 15px;
  border-radius: 5px;
  overflow-x: auto;
}
"@
    $appCss | Out-File -FilePath "src\App.css" -Encoding UTF8
    
    $mainTsx = @"
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"@
    $mainTsx | Out-File -FilePath "src\main.tsx" -Encoding UTF8
    
    $indexCss = @"
:root {
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  display: flex;
  place-items: center;
  min-width: 320px;
  min-height: 100vh;
}

#root {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}
"@
    $indexCss | Out-File -FilePath "src\index.css" -Encoding UTF8
    
    # Create index.html
    $indexHtml = @"
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vidya Quantum Interface</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"@
    $indexHtml | Out-File -FilePath "index.html" -Encoding UTF8
    
    Set-Location ".."
}

# Navigate to frontend directory
Set-Location "vidya-quantum-interface"

# Check if package.json exists
if (!(Test-Path "package.json")) {
    Write-Host "❌ package.json not found in vidya-quantum-interface/" -ForegroundColor Red
    Write-Host "Run .\setup-local.ps1 first to set up the project" -ForegroundColor Yellow
    Set-Location ".."
    exit 1
}

# Install dependencies if node_modules doesn't exist
if (!(Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Gray
    npm install
}

Write-Host "Starting Vite development server..." -ForegroundColor Blue
Write-Host "Frontend will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Blue

# Start the development server
try {
    npm run dev
} catch {
    Write-Host "❌ Failed to start frontend server" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Make sure you've run .\setup-local.ps1 first" -ForegroundColor Yellow
} finally {
    Set-Location ".."
}