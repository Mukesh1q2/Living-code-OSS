#!/usr/bin/env python3
"""
Run the consolidated FastAPI server.
"""

import uvicorn
from src.sanskrit_rewrite_engine.server import app

if __name__ == "__main__":
    print("🚀 Starting Consolidated Sanskrit Rewrite Engine Server...")
    print("=" * 60)
    print("✅ FastAPI server with all consolidated features")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )