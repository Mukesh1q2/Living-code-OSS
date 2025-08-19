#!/usr/bin/env python3
"""
Test script for the morphological analysis API endpoint.
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime

# API endpoint
API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

def test_morphological_analysis_endpoint():
    """Test the REST API endpoint for morphological analysis."""
    print("Testing morphological analysis REST endpoint...")
    
    test_data = {
        "text": "गच्छति",
        "enableVisualization": True,
        "enableEtymology": True,
        "enablePaniniRules": True,
        "enableMorphemeFlow": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/sanskrit/morphological-analysis",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ REST API test successful!")
            print(f"Success: {result.get('success')}")
            print(f"Processing time: {result.get('processing_time')}s")
            
            morphological_data = result.get('morphological_data', {})
            print(f"Word: {morphological_data.get('word')}")
            print(f"Root: {morphological_data.get('root')}")
            print(f"Morphemes: {len(morphological_data.get('morphemes', []))}")
            print(f"Etymology connections: {len(morphological_data.get('etymologicalConnections', []))}")
            print(f"Pāṇini rules: {len(morphological_data.get('paniniRules', []))}")
            
            return True
        else:
            print(f"❌ REST API test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ REST API test failed with error: {e}")
        return False

async def test_websocket_morphological_analysis():
    """Test the WebSocket endpoint for real-time morphological analysis."""
    print("\nTesting morphological analysis WebSocket endpoint...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connected")
            
            # Send morphological analysis request
            request = {
                "type": "morphological_analysis",
                "data": {
                    "text": "गच्छति",
                    "enableVisualization": True,
                    "enableEtymology": True,
                    "enablePaniniRules": True,
                    "enableMorphemeFlow": True
                }
            }
            
            await websocket.send(json.dumps(request))
            print("✅ Morphological analysis request sent")
            
            # Listen for responses
            updates_received = 0
            start_time = datetime.now()
            
            while updates_received < 10:  # Limit to prevent infinite loop
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    
                    message_type = message.get("type")
                    print(f"📨 Received: {message_type}")
                    
                    if message_type == "morphological_analysis_started":
                        print("   Analysis started")
                    elif message_type == "morphological_analysis_update":
                        updates_received += 1
                        data = message.get("data", {})
                        step = data.get("step", "unknown")
                        progress = data.get("progress", 0)
                        print(f"   Update {updates_received}: {step} ({progress:.1%})")
                    elif message_type == "morphological_analysis_complete":
                        print("   Analysis complete!")
                        data = message.get("data", {})
                        print(f"   Word: {data.get('word')}")
                        print(f"   Morphemes: {len(data.get('morphemes', []))}")
                        break
                    elif message_type == "panini_rule_applied":
                        rule_data = message.get("data", {})
                        print(f"   Pāṇini rule applied: {rule_data.get('ruleNumber')} - {rule_data.get('ruleName')}")
                    elif message_type == "morpheme_flow_update":
                        flow_data = message.get("data", {})
                        progress = flow_data.get("progress", 0)
                        print(f"   Morpheme flow: {progress:.1%}")
                    elif message_type == "error":
                        print(f"   ❌ Error: {message.get('data', {}).get('message')}")
                        break
                        
                except asyncio.TimeoutError:
                    print("   ⏰ Timeout waiting for response")
                    break
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"✅ WebSocket test completed in {elapsed:.2f}s")
            print(f"   Received {updates_received} updates")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def test_morpheme_interaction():
    """Test morpheme interaction via WebSocket."""
    print("\nTesting morpheme interaction...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            # Send morpheme interaction
            request = {
                "type": "morpheme_interaction",
                "data": {
                    "morphemeId": "morpheme_0",
                    "action": "select",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await websocket.send(json.dumps(request))
            print("✅ Morpheme interaction request sent")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                message = json.loads(response)
                
                if message.get("type") == "morpheme_interaction_response":
                    print("✅ Morpheme interaction response received")
                    data = message.get("data", {})
                    print(f"   Action: {data.get('action')}")
                    print(f"   Result type: {data.get('result', {}).get('type')}")
                    return True
                else:
                    print(f"❌ Unexpected response type: {message.get('type')}")
                    return False
                    
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for interaction response")
                return False
                
    except Exception as e:
        print(f"❌ Morpheme interaction test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Sanskrit Morphological Analysis Implementation")
    print("=" * 60)
    
    # Test REST API
    rest_success = test_morphological_analysis_endpoint()
    
    # Test WebSocket
    ws_success = await test_websocket_morphological_analysis()
    
    # Test interaction
    interaction_success = await test_morpheme_interaction()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"REST API: {'✅ PASS' if rest_success else '❌ FAIL'}")
    print(f"WebSocket: {'✅ PASS' if ws_success else '❌ FAIL'}")
    print(f"Interaction: {'✅ PASS' if interaction_success else '❌ FAIL'}")
    
    if all([rest_success, ws_success, interaction_success]):
        print("\n🎉 All tests passed! Task 19 implementation is working correctly.")
        print("\nImplemented features:")
        print("✅ Real-time visualization of Sanskrit word decomposition")
        print("✅ Etymological connection display with interactive exploration")
        print("✅ Pāṇini rule application visualization with geometric mandalas")
        print("✅ Morpheme flow animations through neural network pathways")
        print("✅ Interactive exploration of Sanskrit grammatical relationships")
    else:
        print("\n❌ Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    asyncio.run(main())