# LLM Integration Guide for Vidya Quantum Interface

This guide explains how to use the Hugging Face model integration in the Vidya quantum Sanskrit AI consciousness interface.

## Overview

The LLM integration provides:

- **Local Model Inference**: Run Hugging Face models locally for text generation and embeddings
- **Fallback Mechanisms**: Graceful degradation when models are unavailable
- **Model Management**: Load, unload, and manage multiple models
- **Semantic Analysis**: Generate embeddings for text understanding
- **Real-time Processing**: Stream text generation for responsive interfaces

## Quick Start

### 1. Install Dependencies

For basic functionality (with fallbacks):
```bash
pip install -r requirements.txt
```

For full LLM support:
```bash
pip install -r requirements_llm.txt
# or
pip install -e .[gpu]
```

### 2. Basic Usage

```python
from vidya_quantum_interface.llm_integration import get_llm_service, InferenceRequest

# Get the service
service = get_llm_service()

# Initialize models
await service.initialize_models(["default", "embeddings"])

# Generate text
request = InferenceRequest(
    text="Hello, I am Vidya",
    model_name="default",
    max_length=100
)
response = await service.generate_text(request)
print(response.text)

# Generate embeddings
embedding_response = await service.generate_embeddings("Sanskrit consciousness")
print(f"Embedding dimension: {len(embedding_response.embeddings)}")
```

### 3. API Usage

Start the server:
```bash
python -m uvicorn vidya_quantum_interface.api_server:app --reload --port 8000
```

Test endpoints:
```bash
# List available models
curl http://localhost:8000/api/llm/models

# Generate text
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello Vidya", "model_name": "default", "max_length": 100}'

# Generate embeddings
curl -X POST "http://localhost:8000/api/llm/embeddings?text=Sanskrit%20wisdom&model_name=embeddings"

# Test integration with Sanskrit engine
curl -X POST http://localhost:8000/api/llm/integrate \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते विद्या", "model_name": "default", "combine_with_sanskrit": true}'
```

## Configuration

### Environment Variables

Configure the service using environment variables:

```bash
export VIDYA_LLM_CACHE_DIR="/path/to/cache"
export VIDYA_LLM_DEVICE="cuda"  # or "cpu", "mps"
export VIDYA_LLM_ENABLE_FALLBACKS="true"
export VIDYA_LLM_MAX_MEMORY="8192"  # MB
export VIDYA_LLM_TIMEOUT="300"  # seconds
export VIDYA_LLM_LOG_LEVEL="INFO"
```

### Configuration File

Create `llm_config.json`:

```json
{
  "service": {
    "cache_dir": "/path/to/cache",
    "default_device": "auto",
    "enable_fallbacks": true,
    "max_memory_usage": 4096,
    "model_timeout": 300,
    "log_level": "INFO"
  },
  "models": {
    "custom-model": {
      "model_id": "microsoft/DialoGPT-medium",
      "model_type": "text-generation",
      "device": "auto",
      "max_length": 512,
      "temperature": 0.8,
      "top_p": 0.9,
      "do_sample": true,
      "local_files_only": false,
      "trust_remote_code": false,
      "torch_dtype": "auto"
    }
  }
}
```

## Available Models

### Default Models

| Model Name | Model ID | Type | Description |
|------------|----------|------|-------------|
| `default` | `microsoft/DialoGPT-small` | Text Generation | Small conversational model |
| `embeddings` | `sentence-transformers/all-MiniLM-L6-v2` | Text Embedding | Lightweight embedding model |
| `sanskrit-aware` | `microsoft/DialoGPT-medium` | Text Generation | Medium-sized model |
| `large-generation` | `microsoft/DialoGPT-large` | Text Generation | Large model (requires more memory) |
| `embeddings-large` | `sentence-transformers/all-mpnet-base-v2` | Text Embedding | High-quality embeddings |

### Model Recommendations

Based on system capabilities:

- **4GB RAM**: `default`, `embeddings`
- **8GB RAM**: Add `sanskrit-aware`
- **16GB+ RAM**: Add `large-generation`, `embeddings-large`

## Features

### Text Generation

```python
request = InferenceRequest(
    text="Explain quantum consciousness",
    model_name="sanskrit-aware",
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

response = await service.generate_text(request)
if response.success:
    print(f"Generated: {response.text}")
    print(f"Model: {response.model_used}")
    print(f"Time: {response.processing_time:.2f}s")
```

### Streaming Generation

```python
request = InferenceRequest(
    text="Tell me about Sanskrit",
    model_name="default",
    max_length=300
)

async for chunk in service.stream_text_generation(request):
    print(chunk, end="", flush=True)
```

### Semantic Embeddings

```python
# Generate embeddings
response = await service.generate_embeddings(
    "Sanskrit quantum consciousness",
    model_name="embeddings"
)

if response.success:
    embeddings = response.embeddings  # List of floats
    dimension = len(embeddings)  # Usually 384 or 768
    
    # Use for semantic similarity, clustering, etc.
```

### Model Management

```python
# Load a model
success = await service._load_model("large-generation")

# Get model information
info = service.get_model_info("default")
print(f"Loaded: {info.get('loaded', False)}")
print(f"Load time: {info.get('load_time', 0):.2f}s")

# Unload to free memory
await service.unload_model("large-generation")

# List available models
models = service.get_available_models()
for model in models:
    print(f"{model['name']}: {model['model_id']}")
```

## Fallback Mechanisms

When Hugging Face transformers are not available or models fail to load, the service provides intelligent fallbacks:

### Text Generation Fallbacks

- Rule-based response generation
- Context-aware fallback messages
- Maintains API compatibility

### Embedding Fallbacks

- Hash-based deterministic embeddings
- Consistent dimensionality (384D)
- Suitable for development and testing

### Fallback Detection

```python
response = await service.generate_text(request)
if response.metadata and response.metadata.get("fallback_mode"):
    print("Using fallback mode")
    print(f"Reason: {response.metadata.get('reason')}")
```

## Integration with Sanskrit Engine

The LLM service integrates seamlessly with the existing Sanskrit rewrite engine:

```python
# Combined processing
llm_request = LLMIntegrationRequest(
    text="नमस्ते विद्या",
    model_name="sanskrit-aware",
    combine_with_sanskrit=True
)

response = await test_llm_integration(llm_request)

# Access both results
sanskrit_analysis = response.sanskrit_analysis
llm_response = response.llm_response
combined_result = response.combined_result
```

## Performance Optimization

### Memory Management

```python
# Monitor memory usage
model_info = service.get_model_info()
for name, info in model_info.get('models', {}).items():
    if info.get('loaded'):
        print(f"{name}: {info.get('memory_usage', 0)} MB")

# Unload unused models
await service.unload_model("large-generation")

# Force cleanup
await service.cleanup()
```

### Device Selection

```python
# Automatic device detection
service = LLMIntegrationService()
print(f"Using device: {service.device}")

# Manual device specification
config = ModelConfig(
    name="gpu-model",
    model_id="microsoft/DialoGPT-medium",
    model_type=ModelType.TEXT_GENERATION,
    device="cuda"  # or "cpu", "mps"
)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
responses = []

for text in texts:
    request = InferenceRequest(text=text, model_name="default")
    response = await service.generate_text(request)
    responses.append(response)
```

## Error Handling

```python
try:
    response = await service.generate_text(request)
    if not response.success:
        print(f"Generation failed: {response.error_message}")
        # Handle fallback or retry logic
except Exception as e:
    print(f"Service error: {e}")
    # Handle service-level errors
```

## Testing

Run the test suite:

```bash
python test_llm_integration.py
```

Run the demo:

```bash
python examples/llm_integration_demo.py
```

## Troubleshooting

### Common Issues

1. **Transformers not available**
   - Install: `pip install transformers torch`
   - Service will use fallback mode

2. **CUDA out of memory**
   - Reduce `max_length` in model config
   - Use smaller models (`default` instead of `large-generation`)
   - Unload unused models

3. **Model download fails**
   - Check internet connection
   - Set `local_files_only=True` for offline mode
   - Use cached models

4. **Slow inference**
   - Use GPU if available
   - Reduce `max_length`
   - Use smaller models for development

### Debug Information

```python
# Get debug information
debug_info = service.get_model_info()
print(json.dumps(debug_info, indent=2))

# Check system capabilities
from vidya_quantum_interface.llm_config import get_config_manager
config_manager = get_config_manager()
capabilities = config_manager._detect_system_capabilities()
print(f"System capabilities: {capabilities}")
```

## Next Steps

1. **Explore Advanced Features**: Try different models and parameters
2. **Integrate with Frontend**: Connect to the React quantum interface
3. **Custom Models**: Add your own Hugging Face models
4. **Production Deployment**: Configure for cloud deployment
5. **Performance Tuning**: Optimize for your specific use case

For more information, see the [API Reference](api_reference.md) and [Developer Guide](developer_guide.md).