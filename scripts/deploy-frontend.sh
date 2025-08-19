#!/bin/bash
# Frontend deployment script for Sanskrit Rewrite Engine

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
DEPLOY_DIR="${DEPLOY_DIR:-/var/www/sanskrit-rewrite-engine}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/sanskrit-frontend}"

echo -e "${BLUE}Sanskrit Rewrite Engine - Frontend Deployment${NC}"
echo "================================================"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found. Please run build script first.${NC}"
    echo "Run: ./scripts/build-frontend.sh"
    exit 1
fi

# Create backup of current deployment if it exists
if [ -d "$DEPLOY_DIR" ]; then
    echo -e "${YELLOW}Creating backup of current deployment...${NC}"
    BACKUP_NAME="frontend-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r "$DEPLOY_DIR" "$BACKUP_DIR/$BACKUP_NAME"
    echo -e "${GREEN}✓ Backup created: $BACKUP_DIR/$BACKUP_NAME${NC}"
fi

# Create deployment directory if it doesn't exist
mkdir -p "$DEPLOY_DIR"

# Copy build files to deployment directory
echo -e "${YELLOW}Deploying frontend files...${NC}"
cp -r "$BUILD_DIR"/* "$DEPLOY_DIR/"

# Set appropriate permissions
chmod -R 644 "$DEPLOY_DIR"
find "$DEPLOY_DIR" -type d -exec chmod 755 {} \;

echo -e "${GREEN}✓ Frontend deployed successfully to: $DEPLOY_DIR${NC}"

# Optional: Restart web server (uncomment and modify as needed)
# echo -e "${YELLOW}Restarting web server...${NC}"
# sudo systemctl restart nginx
# echo -e "${GREEN}✓ Web server restarted${NC}"

echo -e "${BLUE}Deployment completed successfully!${NC}"
echo "Frontend is now available at the configured web server location."