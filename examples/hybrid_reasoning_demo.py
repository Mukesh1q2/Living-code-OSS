"""
Demonstration of hybrid Sanskrit reasoning system.

This script shows how to use the hybrid reasoning system that combines
R-Zero, external models, and Sanskrit grammatical rules.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sanskrit_rewrite_engine.hybrid_integration import (
    HybridSanskritSystem, HybridConfig, create_default_config, reason_sanskrit
)
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
)


async def demo_basic_reasoning():
    """Demonstrate basic hybrid reasoning capabilities."""
    print("=== Basic Hybrid Reasoning Demo ===\n")
    
    # Create configuration (without API keys for demo)
    config = create_default_config()
    config.enable_gpt = False  # Disable for demo
    config.enable_claude = False  # Disable for demo
    config.log_level = "INFO"
    
    # Initialize system
    system = HybridSanskritSystem(config)
    
    # Test cases
    test_cases = [
        {
            'text': 'a + i',
            'type': SanskritProblemType.SANDHI_APPLICATION,
            'difficulty': SanskritDifficultyLevel.BEGINNER,
            'description': 'Simple vowel sandhi'
        },
        {
            'text': 'rāma + iti',
            'type': SanskritProblemType.SANDHI_APPLICATION,
            'difficulty': SanskritDifficultyLevel.INTERMEDIATE,
            'description': 'Word boundary sandhi'
        },
        {
            'text': 'gacchati',
            'type': SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            'difficulty': SanskritDifficultyLevel.INTERMEDIATE,
            'description': 'Verb morphology analysis'
        },
        {
            'text': 'Create Sanskrit expression for conditional loop',
            'type': SanskritProblemType.TRANSLATION_SYNTHESIS,
            'difficulty': SanskritDifficultyLevel.EXPERT,
            'description': 'Complex translation synthesis'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Input: {test_case['text']}")
        print(f"Type: {test_case['type'].value}")
        print(f"Difficulty: {test_case['difficulty'].value}")
        
        try:
            result = await system.reason(
                input_text=test_case['text'],
                problem_type=test_case['type'],
                difficulty=test_case['difficulty']
            )
            
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Model Used: {result.model_used.value}")
            print(f"Reasoning Steps: {len(result.reasoning_steps)}")
            
            if result.reasoning_steps:
                print("Key Reasoning Steps:")
                for step in result.reasoning_steps[:3]:  # Show first 3 steps
                    print(f"  - {step}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 50)
    
    await system.shutdown()


async def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Demo ===\n")
    
    config = create_default_config()
    config.enable_gpt = False
    config.enable_claude = False
    config.log_level = "WARNING"  # Reduce log noise
    
    system = HybridSanskritSystem(config)
    
    # Batch of Sanskrit problems
    batch_inputs = [
        {'text': 'a + u', 'type': SanskritProblemType.SANDHI_APPLICATION},
        {'text': 'e + a', 'type': SanskritProblemType.SANDHI_APPLICATION},
        {'text': 'karoti', 'type': SanskritProblemType.MORPHOLOGICAL_ANALYSIS},
        {'text': 'bhavati', 'type': SanskritProblemType.MORPHOLOGICAL_ANALYSIS},
        {'text': 'rājaputra', 'type': SanskritProblemType.COMPOUND_ANALYSIS}
    ]
    
    print(f"Processing batch of {len(batch_inputs)} Sanskrit problems...")
    
    try:
        results = await system.reason_batch(batch_inputs, max_concurrent=3)
        
        print(f"Batch processing completed. Results:")
        for i, (input_data, result) in enumerate(zip(batch_inputs, results), 1):
            print(f"{i}. {input_data['text']} → {result.answer} "
                  f"(confidence: {result.confidence:.2f}, model: {result.model_used.value})")
        
        # Calculate statistics
        avg_confidence = sum(r.confidence for r in results) / len(results)
        model_usage = {}
        for result in results:
            model = result.model_used.value
            model_usage[model] = model_usage.get(model, 0) + 1
        
        print(f"\nBatch Statistics:")
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Model Usage: {model_usage}")
        
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
    
    await system.shutdown()


async def demo_fallback_mechanisms():
    """Demonstrate fallback mechanisms."""
    print("\n=== Fallback Mechanisms Demo ===\n")
    
    config = create_default_config()
    config.enable_fallback = True
    config.confidence_threshold = 0.9  # High threshold to trigger fallbacks
    config.log_level = "INFO"
    
    system = HybridSanskritSystem(config)
    
    # Test cases designed to trigger fallbacks
    fallback_cases = [
        {
            'text': 'extremely_complex_unknown_sanskrit_construct_xyz',
            'description': 'Unknown construct to trigger fallback'
        },
        {
            'text': 'ambiguous context dependent expression',
            'description': 'Ambiguous expression requiring external help'
        }
    ]
    
    for i, case in enumerate(fallback_cases, 1):
        print(f"Fallback Test {i}: {case['description']}")
        print(f"Input: {case['text']}")
        
        try:
            result = await system.reason(case['text'])
            
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Model Used: {result.model_used.value}")
            
            # Check if fallback was used
            if 'fallback_used' in result.validation_results:
                print("✓ Fallback mechanism was triggered")
            
            if 'complete_failure' in result.validation_results:
                print("⚠ Complete failure - all methods exhausted")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 50)
    
    await system.shutdown()


async def demo_system_status():
    """Demonstrate system status and monitoring."""
    print("\n=== System Status Demo ===\n")
    
    config = create_default_config()
    system = HybridSanskritSystem(config)
    
    # Get system status
    status = system.get_system_status()
    
    print("System Configuration:")
    for key, value in status['config'].items():
        print(f"  {key}: {value}")
    
    print(f"\nAvailable Models: {status['models']['available_models']}")
    print(f"Model Types: {', '.join(status['models']['model_types'])}")
    
    print(f"\nCache Status:")
    print(f"  Cached Results: {status['cache']['cached_results']}")
    print(f"  Cache Enabled: {status['cache']['cache_enabled']}")
    print(f"  Max Cache Size: {status['cache']['max_cache_size']}")
    
    if status['fallback']['available_strategies']:
        print(f"\nFallback Strategies: {', '.join(status['fallback']['available_strategies'])}")
    
    # Test caching
    print(f"\nTesting result caching...")
    test_text = "a + i"
    
    # First call
    result1 = await system.reason(test_text)
    print(f"First call result: {result1.answer}")
    
    # Second call (should use cache)
    result2 = await system.reason(test_text)
    print(f"Second call result: {result2.answer}")
    
    # Check cache status
    status_after = system.get_system_status()
    print(f"Cached results after test: {status_after['cache']['cached_results']}")
    
    await system.shutdown()


async def demo_convenience_function():
    """Demonstrate convenience function usage."""
    print("\n=== Convenience Function Demo ===\n")
    
    print("Using convenience function for quick reasoning...")
    
    try:
        result = await reason_sanskrit("rāma + eva")
        print(f"Input: rāma + eva")
        print(f"Output: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model: {result.model_used.value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Options Demo ===\n")
    
    # Default configuration
    default_config = HybridConfig()
    print("Default Configuration:")
    print(f"  GPT Enabled: {default_config.enable_gpt}")
    print(f"  Claude Enabled: {default_config.enable_claude}")
    print(f"  R-Zero Enabled: {default_config.enable_r_zero}")
    print(f"  Confidence Threshold: {default_config.confidence_threshold}")
    print(f"  Max Reasoning Time: {default_config.max_reasoning_time_ms}ms")
    print(f"  Cache Results: {default_config.cache_results}")
    print(f"  Enable Consensus: {default_config.enable_consensus}")
    print(f"  Enable Fallback: {default_config.enable_fallback}")
    
    # Custom configuration
    print(f"\nCustom Configuration Example:")
    custom_config = HybridConfig(
        enable_gpt=True,
        gpt_api_key="your-api-key-here",
        enable_claude=True,
        claude_api_key="your-claude-key-here",
        confidence_threshold=0.8,
        max_reasoning_time_ms=15000,
        cache_results=True,
        max_cache_size=500,
        log_level="DEBUG"
    )
    print(f"  Higher confidence threshold: {custom_config.confidence_threshold}")
    print(f"  Shorter reasoning time: {custom_config.max_reasoning_time_ms}ms")
    print(f"  Smaller cache: {custom_config.max_cache_size}")
    print(f"  Debug logging: {custom_config.log_level}")


async def main():
    """Run all demonstrations."""
    print("Sanskrit Hybrid Reasoning System Demonstration")
    print("=" * 60)
    
    # Configuration demo (synchronous)
    demo_configuration_options()
    
    # Async demonstrations
    await demo_basic_reasoning()
    await demo_batch_processing()
    await demo_fallback_mechanisms()
    await demo_system_status()
    await demo_convenience_function()
    
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("\nNote: This demo uses mock external models.")
    print("For real usage, provide valid API keys in the configuration.")


if __name__ == "__main__":
    asyncio.run(main())