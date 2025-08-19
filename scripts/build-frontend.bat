@echo off
REM Frontend build automation script for Sanskrit Rewrite Engine (Windows)

setlocal enabledelayedexpansion

REM Configuration
set FRONTEND_DIR=frontend
set BUILD_DIR=%FRONTEND_DIR%\build
set NODE_VERSION_REQUIRED=18

echo Sanskrit Rewrite Engine - Frontend Build Script
echo ==================================================

REM Check if we're in the right directory
if not exist "%FRONTEND_DIR%" (
    echo Error: Frontend directory not found. Please run this script from the project root.
    exit /b 1
)

REM Check Node.js version
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js not found. Please install Node.js %NODE_VERSION_REQUIRED%+
    exit /b 1
)

for /f "tokens=1 delims=v" %%i in ('node --version') do set NODE_VERSION_FULL=%%i
for /f "tokens=1 delims=." %%i in ("%NODE_VERSION_FULL:v=%") do set NODE_VERSION=%%i

if %NODE_VERSION% lss %NODE_VERSION_REQUIRED% (
    echo Error: Node.js %NODE_VERSION_REQUIRED%+ required. Current version: %NODE_VERSION_FULL%
    exit /b 1
)
echo ✓ Node.js version: %NODE_VERSION_FULL%

REM Navigate to frontend directory
cd "%FRONTEND_DIR%"

REM Check if package.json exists
if not exist "package.json" (
    echo Error: package.json not found in frontend directory
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    call npm ci
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        exit /b 1
    )
    echo ✓ Dependencies installed
) else (
    echo ✓ Dependencies up to date
)

REM Run type checking
echo Running type checking...
call npm run type-check
if errorlevel 1 (
    echo Error: Type checking failed
    exit /b 1
)
echo ✓ Type checking passed

REM Run linting
echo Running linting...
call npm run lint
if errorlevel 1 (
    echo Error: Linting failed
    exit /b 1
)
echo ✓ Linting passed

REM Run tests
echo Running tests...
call npm run test:coverage
if errorlevel 1 (
    echo Error: Tests failed
    exit /b 1
)
echo ✓ Tests passed

REM Clean previous build
if exist "build" (
    echo Cleaning previous build...
    rmdir /s /q "build"
)

REM Build the application
echo Building application...
call npm run build
if errorlevel 1 (
    echo Error: Build failed
    exit /b 1
)

REM Verify build was successful
if exist "build\index.html" (
    echo ✓ Build completed successfully
    echo.
    echo Build Information:
    echo Build directory: %cd%\build
    echo.
    echo Frontend build completed successfully!
    echo To serve the build locally: npm run preview
) else (
    echo Error: Build failed or build directory not found
    exit /b 1
)

endlocal