@echo off
REM Development environment setup script for Sanskrit Rewrite Engine (Windows)

setlocal enabledelayedexpansion

echo Sanskrit Rewrite Engine - Development Setup
echo ==============================================

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo Error: pyproject.toml not found. Please run this script from the project root.
    exit /b 1
)

REM Python Backend Setup
echo Setting up Python backend...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    exit /b 1
)
echo ✓ Python version: 
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -e .[dev]
if errorlevel 1 (
    echo Error: Failed to install Python dependencies
    exit /b 1
)
echo ✓ Python dependencies installed

REM Frontend Setup
echo Setting up frontend...

REM Check Node.js version
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js not found. Please install Node.js 18+
    exit /b 1
)

for /f "tokens=1 delims=v" %%i in ('node --version') do set NODE_VERSION_FULL=%%i
for /f "tokens=1 delims=." %%i in ("%NODE_VERSION_FULL:v=%") do set NODE_VERSION=%%i

if %NODE_VERSION% lss 18 (
    echo Error: Node.js 18+ required. Current version: %NODE_VERSION_FULL%
    exit /b 1
)
echo ✓ Node.js version: %NODE_VERSION_FULL%

REM Navigate to frontend directory
cd frontend

REM Create environment file if it doesn't exist
if not exist ".env.local" (
    echo Creating frontend environment file...
    copy .env.example .env.local >nul
    echo ✓ Environment file created at frontend\.env.local
    echo Please review and update the environment variables as needed.
)

REM Install frontend dependencies
echo Installing frontend dependencies...
call npm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    exit /b 1
)
echo ✓ Frontend dependencies installed

REM Return to root directory
cd ..

echo.
echo Development environment setup completed!
echo.
echo Next steps:
echo 1. Review and update frontend\.env.local with your configuration
echo 2. Start the backend: venv\Scripts\activate.bat ^&^& sanskrit-cli serve
echo 3. Start the frontend: cd frontend ^&^& npm start
echo 4. Open http://localhost:3000 in your browser
echo.
echo Useful commands:
echo - Backend tests: pytest
echo - Frontend tests: cd frontend ^&^& npm test
echo - Build frontend: scripts\build-frontend.bat
echo - Type checking: cd frontend ^&^& npm run type-check

endlocal