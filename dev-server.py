#!/usr/bin/env python3
"""
Vidya Quantum Interface - Concurrent Development Server
Runs both Python backend and React frontend concurrently for local development
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from threading import Thread

class ConcurrentDevServer:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting Vidya Backend Server...")
        
        backend_dir = Path("vidya-backend")
        if not backend_dir.exists():
            print("❌ Backend directory not found!")
            return
            
        try:
            # Change to backend directory and start server
            self.backend_process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream backend output
            def stream_backend_output():
                for line in iter(self.backend_process.stdout.readline, ''):
                    if self.running:
                        print(f"🐍 Backend: {line.strip()}")
                        
            Thread(target=stream_backend_output, daemon=True).start()
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
    
    def start_frontend(self):
        """Start the React frontend development server"""
        print("⚛️  Starting Vidya Frontend Server...")
        
        frontend_dir = Path("vidya-frontend")
        if not frontend_dir.exists():
            print("❌ Frontend directory not found!")
            return
            
        try:
            # Install dependencies if node_modules doesn't exist
            if not (frontend_dir / "node_modules").exists():
                print("📦 Installing frontend dependencies...")
                subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            
            # Start development server
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream frontend output
            def stream_frontend_output():
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if self.running:
                        print(f"⚛️  Frontend: {line.strip()}")
                        
            Thread(target=stream_frontend_output, daemon=True).start()
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
    
    def install_backend_dependencies(self):
        """Install Python backend dependencies"""
        print("🐍 Installing backend dependencies...")
        
        backend_dir = Path("vidya-backend")
        requirements_file = backend_dir / "requirements.txt"
        
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Backend dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install backend dependencies: {e}")
        else:
            print("⚠️  No requirements.txt found for backend")
    
    def stop_servers(self):
        """Stop both servers gracefully"""
        print("\n🛑 Stopping development servers...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ Backend server stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("✅ Frontend server stopped")
    
    def run(self):
        """Run both servers concurrently"""
        print("🌟 Vidya Quantum Interface - Development Server")
        print("=" * 50)
        
        # Install dependencies
        self.install_backend_dependencies()
        
        # Start servers
        self.start_backend()
        time.sleep(2)  # Give backend time to start
        self.start_frontend()
        
        print("\n🎉 Development servers started!")
        print("📍 Backend API: http://localhost:8000")
        print("📍 Frontend App: http://localhost:3000")
        print("📍 API Docs: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop all servers")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_servers()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the concurrent development server
    dev_server = ConcurrentDevServer()
    dev_server.run()