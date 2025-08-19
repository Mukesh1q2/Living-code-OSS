#!/usr/bin/env python3
"""
Test script for the Sanskrit Rewrite Engine Frontend
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def main():
    """Main test function"""
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print("Frontend directory not found!")
        sys.exit(1)
    
    print("Testing Sanskrit Rewrite Engine Frontend...")
    print("=" * 50)
    
    # Check if Node.js is available
    node_result = run_command(["node", "--version"], check=False)
    if not node_result:
        print("Node.js is not installed or not in PATH")
        sys.exit(1)
    
    # Check if npm is available
    npm_result = run_command(["npm", "--version"], check=False)
    if not npm_result:
        print("npm is not installed or not in PATH")
        sys.exit(1)
    
    print(f"Node.js version: {node_result.stdout.strip()}")
    print(f"npm version: {npm_result.stdout.strip()}")
    print()
    
    # Install dependencies
    print("Installing dependencies...")
    install_result = run_command(["npm", "install"], cwd=frontend_dir)
    if not install_result:
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    print("\nRunning tests...")
    test_result = run_command(
        ["npm", "test", "--", "--run", "--watchAll=false"], 
        cwd=frontend_dir,
        check=False
    )
    
    if test_result and test_result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        if test_result and test_result.stderr:
            print(f"Test errors: {test_result.stderr}")
    
    # Check TypeScript compilation
    print("\nChecking TypeScript compilation...")
    tsc_result = run_command(
        ["npx", "tsc", "--noEmit"], 
        cwd=frontend_dir,
        check=False
    )
    
    if tsc_result and tsc_result.returncode == 0:
        print("✅ TypeScript compilation successful!")
    else:
        print("❌ TypeScript compilation failed")
        if tsc_result and tsc_result.stderr:
            print(f"TypeScript errors: {tsc_result.stderr}")
    
    # Try to build the project
    print("\nBuilding project...")
    build_result = run_command(
        ["npm", "run", "build"], 
        cwd=frontend_dir,
        check=False
    )
    
    if build_result and build_result.returncode == 0:
        print("✅ Build successful!")
        
        # Check if build directory exists
        build_dir = frontend_dir / "build"
        if build_dir.exists():
            print(f"Build output created at: {build_dir}")
            
            # List some key files
            key_files = ["index.html", "static/js", "static/css"]
            for file_path in key_files:
                full_path = build_dir / file_path
                if full_path.exists():
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print("❌ Build directory not created")
    else:
        print("❌ Build failed")
        if build_result and build_result.stderr:
            print(f"Build errors: {build_result.stderr}")
    
    print("\n" + "=" * 50)
    print("Frontend testing complete!")

if __name__ == "__main__":
    main()