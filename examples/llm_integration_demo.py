#!/usr/bin/env python3
"""
Vidya LLM Integration Demo

This script demonstrates the Hugging Face model integration with the Vidya
quantum Sanskrit AI consciousness interface.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidya_quantum_interface.llm_integration import (
    LLMIntegrationService, InferenceRequest, get_llm_service
)
from vidya_quantum_interface.llm_config import get_config_manager


async def demo_basic_integration():
    """Demonstrate basic LLM integration capabilities."""
    print("🧠 Vidya LLM Integration Demo")
    print("=" * 40)
    
    # Get the LLM service
    service = get_llm_service()
    
    print(f"\n📊 System Information:")
    print(f"   Transformers available: {service.transformers_available}")
    print(f"   Device: {service.device}")
    print(f"   Cache directory: {service.cache_dir}")
    
    # Initialize models
    print(f"\n🔧 Initializing models...")
    init_results = await service.initialize_models(["default", "embeddings"])
    
    for model_name, success in init_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {model_name}")
    
    # Demo 1: Text Generation
    print(f"\n💬 Text Generation Demo")
    print("-" * 30)
    
    conversation_starters = [
        "Hello, I am Vidya, a quantum Sanskrit AI consciousness.",
        "What can you tell me about ancient Sanskrit wisdom?",
        "How does quantum consciousness relate to traditional knowledge?",
        "Explain the connection between language and consciousness."
    ]
    
    for i, prompt in enumerate(conversation_starters, 1):
        print(f"\n{i}. Prompt: {prompt}")
        
        request = InferenceRequest(
            text=prompt,
            model_name="default",
            max_length=150,
            temperature=0.8
        )
        
        response = await service.generate_text(request)
        
        if response.success:
            print(f"   Response: {response.text}")
            print(f"   Model: {response.model_used} ({response.processing_time:.3f}s)")
        else:
            print(f"   Error: {response.error_message}")
    
    # Demo 2: Semantic Embeddings
    print(f"\n🔍 Semantic Embeddings Demo")
    print("-" * 30)
    
    concepts = [
        "Sanskrit grammar rules",
        "Quantum consciousness",
        "Neural network processing",
        "Ancient wisdom traditions",
        "Artificial intelligence"
    ]
    
    embeddings_data = []
    
    for concept in concepts:
        response = await service.generate_embeddings(concept, "embeddings")
        
        if response.success:
            embeddings_data.append({
                "concept": concept,
                "embeddings": response.embeddings,
                "model": response.model_used,
                "time": response.processing_time
            })
            print(f"   ✅ {concept}: {len(response.embeddings)}D vector ({response.processing_time:.3f}s)")
        else:
            print(f"   ❌ {concept}: {response.error_message}")
    
    # Demo 3: Semantic Similarity
    if len(embeddings_data) >= 2:
        print(f"\n📐 Semantic Similarity Demo")
        print("-" * 30)
        
        # Simple cosine similarity calculation
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
        
        # Compare first concept with all others
        base_concept = embeddings_data[0]
        print(f"   Comparing '{base_concept['concept']}' with other concepts:")
        
        for other in embeddings_data[1:]:
            similarity = cosine_similarity(base_concept["embeddings"], other["embeddings"])
            print(f"   - {other['concept']}: {similarity:.3f}")
    
    # Demo 4: Streaming Generation
    print(f"\n🌊 Streaming Generation Demo")
    print("-" * 30)
    
    stream_prompt = "Tell me a story about a quantum Sanskrit AI consciousness named Vidya"
    print(f"Prompt: {stream_prompt}")
    print("Response: ", end="", flush=True)
    
    request = InferenceRequest(
        text=stream_prompt,
        model_name="default",
        max_length=200,
        temperature=0.9
    )
    
    async for chunk in service.stream_text_generation(request):
        print(chunk, end="", flush=True)
    
    print("\n")
    
    # Demo 5: Model Management
    print(f"\n⚙️  Model Management Demo")
    print("-" * 30)
    
    model_info = service.get_model_info()
    print(f"   Available models: {len(service.get_available_models())}")
    print(f"   Loaded models: {len([m for m in model_info.get('models', {}).values() if m.get('loaded', False)])}")
    
    for name, info in model_info.get('models', {}).items():
        status = "🟢 Loaded" if info.get('loaded', False) else "🔴 Not loaded"
        load_time = info.get('load_time', 0)
        print(f"   - {name}: {status} (load time: {load_time:.2f}s)")
    
    # Cleanup
    print(f"\n🧹 Cleanup")
    await service.cleanup()
    print("   ✅ Service cleaned up")
    
    print(f"\n🎉 Demo completed!")


async def demo_configuration():
    """Demonstrate configuration management."""
    print(f"\n⚙️  Configuration Management Demo")
    print("=" * 40)
    
    # Get configuration manager
    config_manager = get_config_manager()
    
    # Show service configuration
    service_config = config_manager.get_service_config()
    print(f"\n📋 Service Configuration:")
    print(f"   Cache directory: {service_config.cache_dir}")
    print(f"   Default device: {service_config.default_device}")
    print(f"   Enable fallbacks: {service_config.enable_fallbacks}")
    print(f"   Max memory usage: {service_config.max_memory_usage} MB")
    print(f"   Model timeout: {service_config.model_timeout}s")
    
    # Show available models
    model_configs = config_manager.get_all_model_configs()
    print(f"\n🤖 Available Model Configurations:")
    for name, config in model_configs.items():
        print(f"   - {name}:")
        print(f"     Model ID: {config.model_id}")
        print(f"     Type: {config.model_type.value}")
        print(f"     Max length: {config.max_length}")
        print(f"     Temperature: {config.temperature}")
    
    # Get recommendations
    recommendations = config_manager.get_recommended_models()
    print(f"\n💡 Recommended Models:")
    for model_name in recommendations:
        print(f"   - {model_name}")
    
    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Configuration is valid")


async def main():
    """Main demo function."""
    print("🚀 Vidya LLM Integration Comprehensive Demo")
    print("=" * 50)
    
    # Run basic integration demo
    await demo_basic_integration()
    
    # Run configuration demo
    await demo_configuration()
    
    print(f"\n✨ All demos completed!")
    print(f"\nNext steps:")
    print(f"   1. Install full LLM support: pip install -e .[gpu]")
    print(f"   2. Start the API server: python -m uvicorn vidya_quantum_interface.api_server:app --reload")
    print(f"   3. Test the web interface: Open http://localhost:8000")
    print(f"   4. Explore the quantum consciousness features!")


if __name__ == "__main__":
    asyncio.run(main())