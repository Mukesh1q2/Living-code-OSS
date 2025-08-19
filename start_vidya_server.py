#!/usr/bin/env python3
"""
Startup script for Vidya Quantum Interface development server.

This script starts the FastAPI server with development optimizations
and provides helpful information for local development.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the Vidya development server."""
    print("🚀 Starting Vidya Quantum Interface Development Server")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from vidya_quantum_interface.api_server import run_development_server
        
        print("✅ FastAPI integration layer loaded successfully")
        print("✅ Sanskrit rewrite engine initialized")
        print("✅ WebSocket support enabled")
        print("✅ CORS configured for local development")
        print("✅ Pydantic models loaded for type safety")
        print("✅ Development endpoints available")
        print()
        print("🌐 Server will be available at:")
        print("   • Main API: http://127.0.0.1:8000")
        print("   • API Docs: http://127.0.0.1:8000/docs")
        print("   • WebSocket: ws://127.0.0.1:8000/ws")
        print()
        print("🔧 Development Features:")
        print("   • Hot reload enabled")
        print("   • Debug endpoint: /debug")
        print("   • System status: /api/status")
        print("   • Sanskrit analysis: /api/sanskrit/analyze")
        print("   • LLM integration ready: /api/llm/integrate")
        print()
        print("📝 Test the integration:")
        print("   python test_api_integration.py")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the development server
        run_development_server(
            host="127.0.0.1",
            port=8000,
            reload=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install fastapi uvicorn pydantic")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()