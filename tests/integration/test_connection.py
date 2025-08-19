#!/usr/bin/env python3
"""
Test connection to Sanskrit Rewrite Engine server
"""

import json
import time
import sys

# Try to import requests, fall back to urllib if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.parse
    REQUESTS_AVAILABLE = False
    print("⚠️  requests module not available, using urllib")

def test_with_requests():
    """Test using requests library"""
    print("🔍 Testing server connection with requests...")
    
    try:
        # Test basic connection
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            data = response.json()
            print(f"   Message: {data.get('message', 'No message')}")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
        
        # Test processing
        print("\n🔄 Testing text processing...")
        test_data = {
            "text": "rāma + iti",
            "enable_tracing": True
        }
        
        response = requests.post("http://localhost:8000/api/process", 
                               json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing successful!")
            print(f"   Input: {result.get('input', 'unknown')}")
            print(f"   Output: {result.get('output', 'unknown')}")
            if result.get('transformations'):
                print(f"   Rules: {list(result['transformations'].keys())}")
        else:
            print(f"❌ Processing failed with status {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_urllib():
    """Test using urllib (fallback)"""
    print("🔍 Testing server connection with urllib...")
    
    try:
        # Test basic connection
        response = urllib.request.urlopen("http://localhost:8000/", timeout=5)
        if response.status == 200:
            print("✅ Server is responding")
            data = json.loads(response.read().decode('utf-8'))
            print(f"   Message: {data.get('message', 'No message')}")
        else:
            print(f"❌ Server returned status {response.status}")
            return False
        
        # Test processing
        print("\n🔄 Testing text processing...")
        test_data = {
            "text": "rāma + iti",
            "enable_tracing": True
        }
        
        data = json.dumps(test_data).encode('utf-8')
        req = urllib.request.Request("http://localhost:8000/api/process",
                                   data=data,
                                   headers={'Content-Type': 'application/json'})
        
        response = urllib.request.urlopen(req, timeout=10)
        
        if response.status == 200:
            result = json.loads(response.read().decode('utf-8'))
            print(f"✅ Processing successful!")
            print(f"   Input: {result.get('input', 'unknown')}")
            print(f"   Output: {result.get('output', 'unknown')}")
            if result.get('transformations'):
                print(f"   Rules: {list(result['transformations'].keys())}")
        else:
            print(f"❌ Processing failed with status {response.status}")
            return False
        
        return True
        
    except urllib.error.URLError as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Sanskrit Rewrite Engine Connection Test")
    print("=" * 45)
    
    # Test connection
    if REQUESTS_AVAILABLE:
        success = test_with_requests()
    else:
        success = test_with_urllib()
    
    if success:
        print("\n🎉 All tests passed!")
        print("💡 Your frontend should now be able to connect to the backend")
        print("   Frontend: http://localhost:3000")
        print("   Backend:  http://localhost:8000")
    else:
        print("\n❌ Tests failed!")
        print("💡 Make sure the server is running:")
        print("   python start_server.py")
    
    print("=" * 45)

if __name__ == "__main__":
    main()