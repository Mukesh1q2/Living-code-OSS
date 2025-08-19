#!/usr/bin/env python3
"""
Startup script for Sanskrit Rewrite Engine API Server
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the API server."""
    try:
        import uvicorn
        from api_server import app
        
        logger.info("Starting Sanskrit Rewrite Engine API Server...")
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()