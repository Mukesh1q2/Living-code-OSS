#!/usr/bin/env python3
"""
Test script to verify the Sanskrit Rewrite Engine server is working
"""

import requests
import json
import time
import sys

def test_server_connection():
    """Test if server is running and responding"""
    print("🔍 Testing server connection...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - is it running on port 8000?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def test_health_endpoint():
    """Test health check endpoint"""
    print("\n🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Version: {health_data.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Health endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_process_endpoint():
    """Test text processing endpoint"""
    print("\n🔄 Testing text processing...")
    
    test_cases = [
        "rāma + iti",
        "deva + indra", 
        "mahā + ātman",
        "hello world"
    ]
    
    for text in test_cases:
        try:
            payload = {
                "text": text,
                "max_passes": 20,
                "enable_tracing": True
            }
            
            response = requests.post(
                "http://localhost:8000/api/process",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ '{text}' → '{result.get('output', 'no output')}'")
                
                if result.get('transformations'):
                    print(f"   Rules applied: {list(result['transformations'].keys())}")
                else:
                    print("   No transformations applied")
            else:
                print(f"❌ Processing '{text}' failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error processing '{text}': {e}")

def test_rules_endpoint():
    """Test rules listing endpoint"""
    print("\n📋 Testing rules endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/api/rules", timeout=5)
        if response.status_code == 200:
            rules_data = response.json()
            rules = rules_data.get('rules', [])
            print(f"✅ Rules endpoint working - {len(rules)} rules available")
            for rule in rules[:3]:  # Show first 3 rules
                print(f"   - {rule.get('name', 'unknown')}: {rule.get('description', 'no description')}")
            return True
        else:
            print(f"❌ Rules endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Rules endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Sanskrit Rewrite Engine Server Test Suite")
    print("=" * 50)
    
    # Test server connection
    if not test_server_connection():
        print("\n💡 To start the server, run:")
        print("   python simple_server.py")
        print("   OR")
        print("   python sanskrit_rewrite_engine/run_server.py")
        sys.exit(1)
    
    # Run other tests
    test_health_endpoint()
    test_process_endpoint()
    test_rules_endpoint()
    
    print("\n" + "=" * 50)
    print("🎉 Server testing completed!")
    print("\n💡 Frontend should now be able to connect to the backend.")
    print("   Frontend URL: http://localhost:3000")
    print("   Backend URL: http://localhost:8000")

if __name__ == "__main__":
    main()