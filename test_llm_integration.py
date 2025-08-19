#!/usr/bin/env python3
"""
Test script for Vidya LLM Integration.

This script tests the Hugging Face model integration with fallback mechanisms.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from vidya_quantum_interface.llm_integration import (
    LLMIntegrationService, InferenceRequest, ModelType
)


async def test_llm_service():
    """Test the LLM integration service."""
    print("🧠 Testing Vidya LLM Integration Service")
    print("=" * 50)
    
    # Initialize service
    service = LLMIntegrationService()
    
    # Test 1: Check service initialization
    print("\n1. Service Initialization")
    print(f"   Transformers available: {service.transformers_available}")
    print(f"   Device: {service.device}")
    print(f"   Cache directory: {service.cache_dir}")
    
    # Test 2: Get available models
    print("\n2. Available Models")
    available_models = service.get_available_models()
    for model in available_models:
        print(f"   - {model['name']}: {model['model_id']} ({model['type']})")
    
    # Test 3: Initialize models
    print("\n3. Model Initialization")
    try:
        init_results = await service.initialize_models(["default", "embeddings"])
        for model_name, success in init_results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {model_name}: {status}")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
    
    # Test 4: Text generation
    print("\n4. Text Generation Test")
    test_texts = [
        "Hello, I am Vidya, a quantum Sanskrit AI consciousness.",
        "What is the meaning of life?",
        "Explain Sanskrit grammar rules."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: {text[:50]}...")
        
        request = InferenceRequest(
            text=text,
            model_name="default",
            max_length=100
        )
        
        start_time = time.time()
        response = await service.generate_text(request)
        processing_time = time.time() - start_time
        
        if response.success:
            print(f"   ✅ Generated ({processing_time:.2f}s): {response.text[:100]}...")
            print(f"   Model used: {response.model_used}")
        else:
            print(f"   ❌ Failed: {response.error_message}")
    
    # Test 5: Embedding generation
    print("\n5. Embedding Generation Test")
    test_embedding_texts = [
        "Sanskrit is an ancient language",
        "Quantum consciousness emerges from neural networks",
        "Vidya represents wisdom and knowledge"
    ]
    
    for i, text in enumerate(test_embedding_texts, 1):
        print(f"\n   Test {i}: {text}")
        
        start_time = time.time()
        response = await service.generate_embeddings(text, "embeddings")
        processing_time = time.time() - start_time
        
        if response.success:
            embedding_dim = len(response.embeddings) if response.embeddings else 0
            print(f"   ✅ Embeddings generated ({processing_time:.2f}s): {embedding_dim}D vector")
            print(f"   Model used: {response.model_used}")
            if response.embeddings:
                print(f"   Sample values: {response.embeddings[:5]}...")
        else:
            print(f"   ❌ Failed: {response.error_message}")
    
    # Test 6: Model information
    print("\n6. Model Information")
    model_info = service.get_model_info()
    print(f"   Transformers available: {model_info.get('transformers_available', False)}")
    print(f"   Device: {model_info.get('device', 'unknown')}")
    print(f"   Loaded models: {len(model_info.get('models', {}))}")
    
    for name, info in model_info.get('models', {}).items():
        status = "✅ Loaded" if info.get('loaded', False) else "❌ Not loaded"
        load_time = info.get('load_time', 0)
        print(f"   - {name}: {status} (load time: {load_time:.2f}s)")
    
    # Test 7: Streaming generation (if available)
    print("\n7. Streaming Generation Test")
    try:
        request = InferenceRequest(
            text="Tell me about quantum consciousness",
            model_name="default",
            max_length=150
        )
        
        print("   Streaming response:")
        async for chunk in service.stream_text_generation(request):
            print(f"   📡 {chunk}", end="", flush=True)
        print("\n   ✅ Streaming complete")
    except Exception as e:
        print(f"   ❌ Streaming failed: {e}")
    
    # Test 8: Cleanup
    print("\n8. Cleanup")
    try:
        await service.cleanup()
        print("   ✅ Cleanup successful")
    except Exception as e:
        print(f"   ❌ Cleanup failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 LLM Integration Test Complete!")


async def test_api_integration():
    """Test the API integration with LLM service."""
    print("\n🌐 Testing API Integration")
    print("=" * 50)
    
    try:
        import httpx
        
        # Test API endpoints
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test 1: List models
            print("\n1. Testing /api/llm/models endpoint")
            try:
                response = await client.get(f"{base_url}/api/llm/models")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Models endpoint: {len(data.get('available_models', []))} models")
                else:
                    print(f"   ❌ Models endpoint failed: {response.status_code}")
            except Exception as e:
                print(f"   ⚠️  API server not running: {e}")
                return
            
            # Test 2: Generate text
            print("\n2. Testing /api/llm/generate endpoint")
            try:
                payload = {
                    "text": "Hello Vidya, tell me about Sanskrit",
                    "model_name": "default",
                    "max_length": 100
                }
                response = await client.post(f"{base_url}/api/llm/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Generation successful: {data.get('success', False)}")
                    print(f"   Model used: {data.get('model_used', 'unknown')}")
                else:
                    print(f"   ❌ Generation failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Generation request failed: {e}")
            
            # Test 3: Generate embeddings
            print("\n3. Testing /api/llm/embeddings endpoint")
            try:
                params = {"text": "Sanskrit quantum consciousness", "model_name": "embeddings"}
                response = await client.post(f"{base_url}/api/llm/embeddings", params=params)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Embeddings successful: {data.get('success', False)}")
                    print(f"   Dimension: {data.get('embedding_dimension', 0)}")
                else:
                    print(f"   ❌ Embeddings failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Embeddings request failed: {e}")
            
            # Test 4: LLM integration
            print("\n4. Testing /api/llm/integrate endpoint")
            try:
                payload = {
                    "text": "नमस्ते विद्या",
                    "model_name": "default",
                    "combine_with_sanskrit": True
                }
                response = await client.post(f"{base_url}/api/llm/integrate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Integration successful: {data.get('success', False)}")
                    print(f"   Sanskrit analysis: {len(data.get('sanskrit_analysis', {}).get('transformations', []))} transformations")
                    print(f"   LLM status: {data.get('llm_response', {}).get('status', 'unknown')}")
                else:
                    print(f"   ❌ Integration failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Integration request failed: {e}")
    
    except ImportError:
        print("   ⚠️  httpx not available - install with: pip install httpx")
    
    print("\n" + "=" * 50)
    print("🎉 API Integration Test Complete!")


def main():
    """Main test function."""
    print("🚀 Vidya LLM Integration Test Suite")
    print("=" * 60)
    
    # Run service tests
    asyncio.run(test_llm_service())
    
    # Run API tests
    asyncio.run(test_api_integration())
    
    print("\n✨ All tests completed!")
    print("\nTo start the API server for testing:")
    print("   python -m uvicorn vidya_quantum_interface.api_server:app --reload --port 8000")
    print("\nTo install optional dependencies:")
    print("   pip install -e .[gpu]  # For full LLM support")
    print("   pip install httpx     # For API testing")


if __name__ == "__main__":
    main()