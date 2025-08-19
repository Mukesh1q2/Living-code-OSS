#!/bin/bash
# Frontend build automation script for Sanskrit Rewrite Engine

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_DIR="frontend"
BUILD_DIR="$FRONTEND_DIR/build"
NODE_VERSION_REQUIRED="18"

echo -e "${BLUE}Sanskrit Rewrite Engine - Frontend Build Script${NC}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}Error: Frontend directory not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Check Node.js version
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt "$NODE_VERSION_REQUIRED" ]; then
        echo -e "${RED}Error: Node.js $NODE_VERSION_REQUIRED+ required. Current version: $(node --version)${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js version: $(node --version)${NC}"
else
    echo -e "${RED}Error: Node.js not found. Please install Node.js $NODE_VERSION_REQUIRED+${NC}"
    exit 1
fi

# Navigate to frontend directory
cd "$FRONTEND_DIR"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo -e "${RED}Error: package.json not found in frontend directory${NC}"
    exit 1
fi

# Install dependencies if node_modules doesn't exist or package-lock.json is newer
if [ ! -d "node_modules" ] || [ "package-lock.json" -nt "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm ci
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies up to date${NC}"
fi

# Run type checking
echo -e "${YELLOW}Running type checking...${NC}"
npm run type-check
echo -e "${GREEN}✓ Type checking passed${NC}"

# Run linting
echo -e "${YELLOW}Running linting...${NC}"
npm run lint
echo -e "${GREEN}✓ Linting passed${NC}"

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
npm run test:coverage
echo -e "${GREEN}✓ Tests passed${NC}"

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Build the application
echo -e "${YELLOW}Building application...${NC}"
npm run build

# Verify build was successful
if [ -d "build" ] && [ -f "build/index.html" ]; then
    echo -e "${GREEN}✓ Build completed successfully${NC}"
    
    # Show build size information
    echo -e "${BLUE}Build Information:${NC}"
    echo "Build directory: $(pwd)/build"
    echo "Build size: $(du -sh build | cut -f1)"
    echo "Main files:"
    find build -name "*.js" -o -name "*.css" | head -5 | while read file; do
        echo "  - $file ($(du -sh "$file" | cut -f1))"
    done
else
    echo -e "${RED}Error: Build failed or build directory not found${NC}"
    exit 1
fi

echo -e "${GREEN}Frontend build completed successfully!${NC}"
echo -e "${BLUE}To serve the build locally: npm run preview${NC}"