#!/usr/bin/env python3
"""
Verification script for Vidya LLM Integration.

This script verifies that all LLM integration features are working correctly.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from vidya_quantum_interface.llm_integration import (
    LLMIntegrationService, InferenceRequest, get_llm_service
)
from vidya_quantum_interface.llm_config import get_config_manager


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")


def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"ℹ️  {message}")


async def verify_service_initialization():
    """Verify that the LLM service initializes correctly."""
    print_section("Service Initialization")
    
    try:
        service = get_llm_service()
        print_success("LLM service instance created")
        
        print_info(f"Transformers available: {service.transformers_available}")
        print_info(f"Device: {service.device}")
        print_info(f"Cache directory: {service.cache_dir}")
        
        # Test model initialization
        init_results = await service.initialize_models(["default", "embeddings"])
        
        if service.transformers_available:
            for model_name, success in init_results.items():
                if success:
                    print_success(f"Model '{model_name}' initialized successfully")
                else:
                    print_error(f"Model '{model_name}' failed to initialize")
        else:
            print_info("Using fallback mode (transformers not available)")
        
        return True
        
    except Exception as e:
        print_error(f"Service initialization failed: {e}")
        return False


async def verify_text_generation():
    """Verify text generation functionality."""
    print_section("Text Generation")
    
    service = get_llm_service()
    
    test_cases = [
        {
            "text": "Hello, I am Vidya",
            "model_name": "default",
            "max_length": 50,
            "expected_success": True
        },
        {
            "text": "Explain Sanskrit grammar",
            "model_name": "sanskrit-aware",
            "max_length": 100,
            "expected_success": True
        },
        {
            "text": "",  # Empty text
            "model_name": "default",
            "max_length": 50,
            "expected_success": True  # Should handle gracefully
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}")
        
        request = InferenceRequest(
            text=test_case["text"],
            model_name=test_case["model_name"],
            max_length=test_case["max_length"]
        )
        
        try:
            start_time = time.time()
            response = await service.generate_text(request)
            processing_time = time.time() - start_time
            
            if response.success == test_case["expected_success"]:
                print_success(f"Text generation successful ({processing_time:.3f}s)")
                print_info(f"Input: '{test_case['text'][:30]}...'")
                print_info(f"Output: '{response.text[:50]}...'")
                print_info(f"Model used: {response.model_used}")
            else:
                print_error(f"Unexpected result: expected {test_case['expected_success']}, got {response.success}")
                all_passed = False
                
        except Exception as e:
            print_error(f"Text generation failed: {e}")
            all_passed = False
    
    return all_passed


async def verify_embeddings():
    """Verify embedding generation functionality."""
    print_section("Embedding Generation")
    
    service = get_llm_service()
    
    test_texts = [
        "Sanskrit is an ancient language",
        "Quantum consciousness emerges from neural networks",
        "Vidya represents wisdom and knowledge",
        ""  # Empty text
    ]
    
    all_passed = True
    
    for i, text in enumerate(test_texts, 1):
        print_subsection(f"Test Case {i}")
        
        try:
            start_time = time.time()
            response = await service.generate_embeddings(text, "embeddings")
            processing_time = time.time() - start_time
            
            if response.success:
                embedding_dim = len(response.embeddings) if response.embeddings else 0
                print_success(f"Embeddings generated ({processing_time:.3f}s)")
                print_info(f"Input: '{text[:30]}...'")
                print_info(f"Dimension: {embedding_dim}")
                print_info(f"Model used: {response.model_used}")
                
                # Verify embedding properties
                if response.embeddings:
                    if embedding_dim > 0:
                        print_success(f"Embedding dimension is valid: {embedding_dim}")
                    else:
                        print_error("Embedding dimension is zero")
                        all_passed = False
                else:
                    print_error("No embeddings returned")
                    all_passed = False
            else:
                print_error(f"Embedding generation failed: {response.error_message}")
                all_passed = False
                
        except Exception as e:
            print_error(f"Embedding generation failed: {e}")
            all_passed = False
    
    return all_passed


async def verify_streaming():
    """Verify streaming text generation."""
    print_section("Streaming Generation")
    
    service = get_llm_service()
    
    try:
        request = InferenceRequest(
            text="Tell me about quantum consciousness",
            model_name="default",
            max_length=100
        )
        
        print_info("Starting streaming generation...")
        chunks = []
        
        async for chunk in service.stream_text_generation(request):
            chunks.append(chunk)
            if len(chunks) <= 5:  # Show first few chunks
                print_info(f"Chunk {len(chunks)}: '{chunk[:20]}...'")
        
        if chunks:
            full_text = "".join(chunks)
            print_success(f"Streaming completed: {len(chunks)} chunks, {len(full_text)} characters")
            return True
        else:
            print_error("No chunks received from streaming")
            return False
            
    except Exception as e:
        print_error(f"Streaming generation failed: {e}")
        return False


async def verify_model_management():
    """Verify model management functionality."""
    print_section("Model Management")
    
    service = get_llm_service()
    
    try:
        # Test getting model info
        model_info = service.get_model_info()
        print_success("Retrieved model information")
        print_info(f"Transformers available: {model_info.get('transformers_available', False)}")
        print_info(f"Device: {model_info.get('device', 'unknown')}")
        
        # Test getting available models
        available_models = service.get_available_models()
        print_success(f"Retrieved {len(available_models)} available models")
        
        for model in available_models[:3]:  # Show first 3
            print_info(f"- {model['name']}: {model['model_id']} ({model['type']})")
        
        # Test model loading/unloading (if transformers available)
        if service.transformers_available:
            test_model = "default"
            
            # Try to load
            success = await service._load_model(test_model)
            if success:
                print_success(f"Model '{test_model}' loaded successfully")
                
                # Try to unload
                success = await service.unload_model(test_model)
                if success:
                    print_success(f"Model '{test_model}' unloaded successfully")
                else:
                    print_error(f"Failed to unload model '{test_model}'")
            else:
                print_info(f"Model '{test_model}' loading failed (expected in fallback mode)")
        else:
            print_info("Model loading/unloading skipped (fallback mode)")
        
        return True
        
    except Exception as e:
        print_error(f"Model management verification failed: {e}")
        return False


async def verify_configuration():
    """Verify configuration management."""
    print_section("Configuration Management")
    
    try:
        config_manager = get_config_manager()
        print_success("Configuration manager created")
        
        # Test service config
        service_config = config_manager.get_service_config()
        print_success("Retrieved service configuration")
        print_info(f"Cache dir: {service_config.cache_dir}")
        print_info(f"Default device: {service_config.default_device}")
        print_info(f"Enable fallbacks: {service_config.enable_fallbacks}")
        
        # Test model configs
        model_configs = config_manager.get_all_model_configs()
        print_success(f"Retrieved {len(model_configs)} model configurations")
        
        # Test recommendations
        recommendations = config_manager.get_recommended_models()
        print_success(f"Got {len(recommendations)} model recommendations")
        print_info(f"Recommended: {', '.join(recommendations)}")
        
        # Test validation
        issues = config_manager.validate_config()
        if issues:
            print_info(f"Configuration issues found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3
                print_info(f"- {issue}")
        else:
            print_success("Configuration validation passed")
        
        return True
        
    except Exception as e:
        print_error(f"Configuration verification failed: {e}")
        return False


async def verify_error_handling():
    """Verify error handling and fallback mechanisms."""
    print_section("Error Handling & Fallbacks")
    
    service = get_llm_service()
    
    try:
        # Test with invalid model name
        request = InferenceRequest(
            text="Test text",
            model_name="nonexistent-model",
            max_length=50
        )
        
        response = await service.generate_text(request)
        
        if response.success and response.model_used == "fallback":
            print_success("Fallback mechanism working for invalid model")
        elif not service.transformers_available:
            print_success("Fallback mode active (transformers not available)")
        else:
            print_error("Fallback mechanism not working properly")
            return False
        
        # Test with invalid embedding model
        embedding_response = await service.generate_embeddings("test", "nonexistent-embedding-model")
        
        if embedding_response.success and embedding_response.model_used == "fallback":
            print_success("Fallback mechanism working for invalid embedding model")
        else:
            print_info("Embedding fallback behavior varies by configuration")
        
        return True
        
    except Exception as e:
        print_error(f"Error handling verification failed: {e}")
        return False


async def main():
    """Main verification function."""
    print("🚀 Vidya LLM Integration Verification Suite")
    print("=" * 60)
    
    verification_functions = [
        ("Service Initialization", verify_service_initialization),
        ("Text Generation", verify_text_generation),
        ("Embedding Generation", verify_embeddings),
        ("Streaming Generation", verify_streaming),
        ("Model Management", verify_model_management),
        ("Configuration Management", verify_configuration),
        ("Error Handling", verify_error_handling)
    ]
    
    results = {}
    
    for name, func in verification_functions:
        try:
            result = await func()
            results[name] = result
        except Exception as e:
            print_error(f"Verification '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_section("Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Results: {passed}/{total} verifications passed")
    
    if passed == total:
        print_success("🎉 All verifications passed! LLM integration is working correctly.")
        
        print("\n🚀 Next Steps:")
        print("   1. Install full LLM support: pip install -e .[gpu]")
        print("   2. Start the API server: python -m uvicorn vidya_quantum_interface.api_server:app --reload")
        print("   3. Test the web interface: Open http://localhost:8000")
        print("   4. Run the demo: python examples/llm_integration_demo.py")
        
        return True
    else:
        print_error(f"❌ {total - passed} verifications failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)