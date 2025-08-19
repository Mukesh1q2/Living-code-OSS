#!/usr/bin/env python3
"""
Test script for the Sanskrit Engine Adapter

This script tests the basic functionality of the Sanskrit adapter
to ensure it works correctly with the existing Sanskrit engine.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from vidya_quantum_interface import SanskritEngineAdapter
    print("✓ Successfully imported SanskritEngineAdapter")
except ImportError as e:
    print(f"✗ Failed to import SanskritEngineAdapter: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic Sanskrit adapter functionality"""
    print("\n=== Testing Basic Sanskrit Adapter Functionality ===")
    
    try:
        # Initialize the adapter
        print("1. Initializing Sanskrit adapter...")
        adapter = SanskritEngineAdapter()
        print("✓ Sanskrit adapter initialized successfully")
        
        # Test simple text processing
        print("\n2. Testing simple text processing...")
        test_text = "rama"
        result = await adapter.process_text_simple(test_text)
        
        print(f"✓ Processed text: '{test_text}'")
        print(f"  - Success: {result.get('success', False)}")
        print(f"  - Error: {result.get('error', 'None')}")
        
        if result.get('visualization_data'):
            viz_data = result['visualization_data']
            print(f"  - Tokens: {len(viz_data.get('tokens', []))}")
            print(f"  - Network nodes: {len(viz_data.get('network_nodes', []))}")
            print(f"  - Connections: {len(viz_data.get('connections', []))}")
            print(f"  - Quantum effects: {len(viz_data.get('quantum_effects', []))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False


async def test_streaming_functionality():
    """Test streaming Sanskrit processing"""
    print("\n=== Testing Streaming Functionality ===")
    
    try:
        adapter = SanskritEngineAdapter()
        test_text = "गच्छति"
        
        print(f"3. Testing streaming processing for: '{test_text}'")
        
        update_count = 0
        async for update in adapter.process_text_streaming(test_text, enable_visualization=True):
            update_count += 1
            print(f"  Update {update_count}: {update.update_type} (progress: {update.progress:.1%})")
            
            if update.update_type == 'processing_complete':
                break
            
            # Limit updates to prevent spam
            if update_count > 10:
                print("  ... (limiting output)")
                break
        
        print(f"✓ Streaming completed with {update_count} updates")
        return True
        
    except Exception as e:
        print(f"✗ Error in streaming functionality test: {e}")
        return False


async def test_error_handling():
    """Test error handling and graceful degradation"""
    print("\n=== Testing Error Handling ===")
    
    try:
        adapter = SanskritEngineAdapter()
        
        # Test with empty text
        print("4. Testing with empty text...")
        result = await adapter.process_text_simple("")
        print(f"  - Empty text handled: {result.get('success', False)}")
        
        # Test with very long text
        print("5. Testing with long text...")
        long_text = "a" * 1000
        result = await adapter.process_text_simple(long_text)
        print(f"  - Long text handled: {result.get('success', False)}")
        
        # Test stopping non-existent stream
        print("6. Testing stream stop functionality...")
        success = adapter.stop_stream("non_existent_id")
        print(f"  - Non-existent stream stop handled: {not success}")
        
        print("✓ Error handling tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False


async def test_visualization_data():
    """Test visualization data generation"""
    print("\n=== Testing Visualization Data Generation ===")
    
    try:
        adapter = SanskritEngineAdapter()
        test_text = "सत्यम्"
        
        print(f"7. Testing visualization data for: '{test_text}'")
        result = await adapter.process_text_simple(test_text)
        
        if result.get('visualization_data'):
            viz_data = result['visualization_data']
            
            # Check tokens
            tokens = viz_data.get('tokens', [])
            if tokens:
                print(f"  ✓ Generated {len(tokens)} tokens")
                sample_token = tokens[0]
                print(f"    - Sample token text: '{sample_token.get('text', 'N/A')}'")
                print(f"    - Has morphology: {'morphology' in sample_token}")
                print(f"    - Has quantum properties: {'quantum_properties' in sample_token}")
                print(f"    - Has visualization data: {'visualization_data' in sample_token}")
            
            # Check network nodes
            nodes = viz_data.get('network_nodes', [])
            if nodes:
                print(f"  ✓ Generated {len(nodes)} network nodes")
                node_types = set(node.get('type', 'unknown') for node in nodes)
                print(f"    - Node types: {', '.join(node_types)}")
            
            # Check connections
            connections = viz_data.get('connections', [])
            if connections:
                print(f"  ✓ Generated {len(connections)} connections")
                connection_types = set(conn.get('type', 'unknown') for conn in connections)
                print(f"    - Connection types: {', '.join(connection_types)}")
            
            # Check quantum effects
            effects = viz_data.get('quantum_effects', [])
            if effects:
                print(f"  ✓ Generated {len(effects)} quantum effects")
                effect_types = set(effect.get('type', 'unknown') for effect in effects)
                print(f"    - Effect types: {', '.join(effect_types)}")
        
        print("✓ Visualization data generation test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error in visualization data test: {e}")
        return False


async def main():
    """Run all tests"""
    print("Sanskrit Engine Adapter Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_streaming_functionality,
        test_error_handling,
        test_visualization_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)