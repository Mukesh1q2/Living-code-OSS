#!/usr/bin/env python3
"""
Test script to verify the Vidya development environment setup
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def test_backend_dependencies():
    """Test that backend dependencies are installed"""
    print("🧪 Testing backend dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import websockets
        import pydantic
        print("✅ All backend dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing backend dependency: {e}")
        return False

def test_frontend_dependencies():
    """Test that frontend dependencies are installed"""
    print("🧪 Testing frontend dependencies...")
    
    frontend_dir = Path("vidya-frontend")
    node_modules = frontend_dir / "node_modules"
    
    if node_modules.exists():
        print("✅ Frontend dependencies are installed")
        return True
    else:
        print("❌ Frontend node_modules not found")
        return False

def test_backend_server():
    """Test that backend server can start and respond"""
    print("🧪 Testing backend server startup...")
    
    backend_dir = Path("vidya-backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    try:
        # Start backend server in background
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                print("✅ Backend server responds to health check")
                result = True
            else:
                print(f"❌ Backend server returned status {response.status_code}")
                result = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend server not responding: {e}")
            result = False
        
        # Stop the server
        process.terminate()
        process.wait(timeout=5)
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing backend server: {e}")
        return False

def test_frontend_build():
    """Test that frontend can build successfully"""
    print("🧪 Testing frontend build...")
    
    frontend_dir = Path("vidya-frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    try:
        # Check if we already built successfully earlier
        dist_dir = frontend_dir / "dist"
        if dist_dir.exists():
            print("✅ Frontend build artifacts found (already built)")
            return True
        
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True  # Use shell on Windows
        )
        
        if result.returncode == 0:
            print("✅ Frontend builds successfully")
            return True
        else:
            print(f"❌ Frontend build failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Frontend build timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend build: {e}")
        return False

def test_shared_types():
    """Test that shared TypeScript interfaces exist"""
    print("🧪 Testing shared TypeScript interfaces...")
    
    shared_types_file = Path("vidya-frontend/src/types/shared.ts")
    
    if shared_types_file.exists():
        content = shared_types_file.read_text()
        
        # Check for key interfaces
        required_interfaces = [
            "QuantumState",
            "VidyaConsciousness",
            "NetworkNode",
            "WebSocketMessage",
            "SanskritToken"
        ]
        
        missing_interfaces = []
        for interface in required_interfaces:
            if f"interface {interface}" not in content:
                missing_interfaces.append(interface)
        
        if not missing_interfaces:
            print("✅ All required TypeScript interfaces are present")
            return True
        else:
            print(f"❌ Missing interfaces: {missing_interfaces}")
            return False
    else:
        print("❌ Shared types file not found")
        return False

def main():
    """Run all tests"""
    print("🌟 Vidya Quantum Interface - Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Backend Dependencies", test_backend_dependencies),
        ("Frontend Dependencies", test_frontend_dependencies),
        ("Shared TypeScript Interfaces", test_shared_types),
        ("Frontend Build", test_frontend_build),
        ("Backend Server", test_backend_server),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Vidya development environment is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run: python dev-server.py")
        print("   2. Open: http://localhost:3000")
        print("   3. Start developing the quantum consciousness!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()