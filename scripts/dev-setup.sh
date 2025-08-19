#!/bin/bash
# Development environment setup script for Sanskrit Rewrite Engine

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Sanskrit Rewrite Engine - Development Setup${NC}"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Python Backend Setup
echo -e "${YELLOW}Setting up Python backend...${NC}"

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${GREEN}✓ Python version: $(python3 --version)${NC}"
else
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -e .[dev]
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Frontend Setup
echo -e "${YELLOW}Setting up frontend...${NC}"

# Check Node.js version
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt "18" ]; then
        echo -e "${RED}Error: Node.js 18+ required. Current version: $(node --version)${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js version: $(node --version)${NC}"
else
    echo -e "${RED}Error: Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

# Navigate to frontend directory and install dependencies
cd frontend

if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}Creating frontend environment file...${NC}"
    cp .env.example .env.local
    echo -e "${GREEN}✓ Environment file created at frontend/.env.local${NC}"
    echo -e "${BLUE}Please review and update the environment variables as needed.${NC}"
fi

echo -e "${YELLOW}Installing frontend dependencies...${NC}"
npm install
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

# Return to root directory
cd ..

# Setup pre-commit hooks if available
if [ -f ".pre-commit-config.yaml" ]; then
    echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

echo -e "${GREEN}Development environment setup completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review and update frontend/.env.local with your configuration"
echo "2. Start the backend: source venv/bin/activate && sanskrit-cli serve"
echo "3. Start the frontend: cd frontend && npm start"
echo "4. Open http://localhost:3000 in your browser"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "- Backend tests: pytest"
echo "- Frontend tests: cd frontend && npm test"
echo "- Build frontend: ./scripts/build-frontend.sh"
echo "- Lint code: pre-commit run --all-files"