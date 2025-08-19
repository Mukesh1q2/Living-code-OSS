#!/usr/bin/env python3
"""
Test script for the consolidated FastAPI server.
"""

import requests
import json
import sys
import time

def test_server(base_url="http://localhost:8000"):
    """Test all endpoints of the consolidated server."""
    print("Testing Consolidated Sanskrit Rewrite Engine Server")
    print("=" * 55)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            data = response.json()
            print(f"   Version: {data.get('version')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Components: {len(data.get('components', {}))}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test 3: Rules endpoint
    print("\n3. Testing rules endpoint...")
    try:
        response = requests.get(f"{base_url}/api/rules")
        if response.status_code == 200:
            print("✅ Rules endpoint working")
            data = response.json()
            print(f"   Rules count: {data.get('count')}")
            if data.get('rules'):
                print(f"   First rule: {data['rules'][0].get('name')}")
        else:
            print(f"❌ Rules endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Rules endpoint error: {e}")
    
    # Test 4: Process endpoint
    print("\n4. Testing process endpoint...")
    try:
        test_data = {
            "text": "rāma + iti",
            "enable_tracing": True
        }
        response = requests.post(f"{base_url}/api/process", json=test_data)
        if response.status_code == 200:
            print("✅ Process endpoint working")
            data = response.json()
            print(f"   Input: {data.get('input')}")
            print(f"   Output: {data.get('output')}")
            print(f"   Success: {data.get('success')}")
            print(f"   Transformations: {len(data.get('transformations', {}))}")
        else:
            print(f"❌ Process endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Process endpoint error: {e}")
    
    # Test 5: Analyze endpoint
    print("\n5. Testing analyze endpoint...")
    try:
        test_data = {
            "text": "rāma + iti"
        }
        response = requests.post(f"{base_url}/api/analyze", json=test_data)
        if response.status_code == 200:
            print("✅ Analyze endpoint working")
            data = response.json()
            print(f"   Input: {data.get('input')}")
            print(f"   Success: {data.get('success')}")
            analysis = data.get('analysis', {})
            print(f"   Word count: {analysis.get('basic_stats', {}).get('word_count')}")
        else:
            print(f"❌ Analyze endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Analyze endpoint error: {e}")
    
    # Test 6: OpenAPI docs
    print("\n6. Testing OpenAPI documentation...")
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            print("✅ OpenAPI documentation available")
            data = response.json()
            print(f"   Title: {data.get('info', {}).get('title')}")
            print(f"   Paths: {len(data.get('paths', {}))}")
        else:
            print(f"❌ OpenAPI documentation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ OpenAPI documentation error: {e}")
    
    print("\n" + "=" * 55)
    print("Test completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing server at: {base_url}")
    print("Make sure the server is running first!")
    print()
    
    test_server(base_url)