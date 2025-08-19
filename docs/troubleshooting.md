# Sanskrit Rewrite Engine - Troubleshooting Guide

## Common Issues and Solutions

This guide covers frequently encountered issues and their solutions when using the Sanskrit Rewrite Engine.

## Installation Issues

### Python Version Compatibility

**Issue**: `ImportError` or `SyntaxError` when importing the engine

```
SyntaxError: invalid syntax (type hints not supported)
```

**Solution**: Ensure you're using Python 3.8 or higher

```bash
python --version  # Should show 3.8+
pip install --upgrade python
```

### Dependency Installation Problems

**Issue**: `ModuleNotFoundError` for required packages

```
ModuleNotFoundError: No module named 'sanskrit_rewrite_engine'
```

**Solution**: Install all dependencies

```bash
pip install -r requirements.txt
# Or for development
pip install -r requirements-dev.txt
```

**Issue**: Conflicting package versions

```
ERROR: pip's dependency resolver does not currently consider all the ways...
```

**Solution**: Use a virtual environment to isolate dependencies

```bash
python -m venv sanskrit_env
source sanskrit_env/bin/activate  # On Windows: sanskrit_env\Scripts\activate
pip install -r requirements.txt
```

## Runtime Issues

### Memory and Performance Problems

**Issue**: High memory usage during large text processing

```python
MemoryError: Unable to allocate array
```

**Solution**: Process text in smaller chunks

```python
from sanskrit_rewrite_engine import SanskritEngine

engine = SanskritEngine()

# Instead of processing entire large text at once
# large_text = "very long sanskrit text..."
# result = engine.rewrite(large_text)

# Process in chunks
def process_in_chunks(text, chunk_size=1000):
    results = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        results.append(engine.rewrite(chunk))
    return ''.join(results)
```

**Issue**: Slow processing speed for complex transformations

**Solution**: Enable caching and use optimized settings

```python
engine = SanskritEngine(
    enable_cache=True,
    cache_size=10000,
    optimization_level='high'
)
```

### Unicode and Encoding Issues

**Issue**: Incorrect character rendering or `UnicodeDecodeError`

```python
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution**: Ensure proper UTF-8 encoding

```python
# When reading files
with open('sanskrit_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# When writing output
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(rewritten_text)
```

**Issue**: Devanagari script not displaying correctly

**Solution**: Verify font support and system configuration

```python
# Check if text contains Devanagari characters
import unicodedata

def has_devanagari(text):
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, ''):
            return True
    return False
```

## Configuration Issues

### Rule Loading Problems

**Issue**: Custom rules not being applied

```python
RuleLoadError: Unable to load custom rules from file
```

**Solution**: Verify rule file format and syntax

```python
# Correct rule file format (rules.json)
{
    "transformation_rules": [
        {
            "pattern": "source_pattern",
            "replacement": "target_pattern",
            "context": "optional_context"
        }
    ]
}

# Load rules properly
engine = SanskritEngine()
engine.load_custom_rules('path/to/rules.json')
```

**Issue**: Rule conflicts or unexpected transformations

**Solution**: Check rule priority and ordering

```python
# Rules are applied in order - more specific rules should come first
engine.set_rule_priority(['specific_rule_1', 'general_rule_2'])
```

### Dictionary and Lexicon Issues

**Issue**: Words not being recognized or transformed

```python
LexiconError: Word not found in dictionary
```

**Solution**: Update or expand the lexicon

```python
# Add custom words to lexicon
engine.add_to_lexicon([
    {'word': 'custom_word', 'meaning': 'definition', 'category': 'noun'}
])

# Or load additional dictionary
engine.load_dictionary('supplementary_dict.json')
```

## API and Integration Issues

### Import and Module Issues

**Issue**: Cannot import specific components

```python
ImportError: cannot import name 'SanskritEngine' from 'sanskrit_rewrite_engine'
```

**Solution**: Check installation and import paths

```python
# Verify installation
import sanskrit_rewrite_engine
print(sanskrit_rewrite_engine.__version__)

# Correct import syntax
from sanskrit_rewrite_engine import SanskritEngine
from sanskrit_rewrite_engine.transformers import TextTransformer
```

### Configuration File Issues

**Issue**: Configuration file not found or invalid format

```python
ConfigError: Configuration file 'config.yaml' not found
```

**Solution**: Create or fix configuration file

```yaml
# config.yaml
engine:
  language: sanskrit
  script: devanagari
  transformation_mode: conservative
  
rules:
  enable_sandhi: true
  enable_morphology: true
  custom_rules_path: "rules/"
  
output:
  format: unicode
  preserve_formatting: true
```

## Testing and Validation Issues

### Test Failures

**Issue**: Unit tests failing after installation

**Solution**: Run tests with proper environment setup

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_transformations.py -v
```

**Issue**: Validation errors for Sanskrit text

**Solution**: Use built-in validation tools

```python
from sanskrit_rewrite_engine.validators import SanskritValidator

validator = SanskritValidator()

# Validate input text
is_valid, errors = validator.validate_text(input_text)
if not is_valid:
    print(f"Validation errors: {errors}")
```

## Performance Optimization

### Caching Strategies

**Issue**: Repeated processing of similar texts is slow

**Solution**: Implement intelligent caching

```python
# Enable persistent caching
engine = SanskritEngine(
    cache_type='persistent',
    cache_file='sanskrit_cache.db'
)

# Clear cache when needed
engine.clear_cache()
```

### Batch Processing

**Issue**: Processing multiple files individually is inefficient

**Solution**: Use batch processing capabilities

```python
# Process multiple files at once
files = ['text1.txt', 'text2.txt', 'text3.txt']
results = engine.batch_process(files, output_dir='processed/')
```

## Debugging and Logging

### Enable Debug Mode

**Issue**: Need detailed information about transformation process

**Solution**: Enable comprehensive logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Enable engine debug mode
engine = SanskritEngine(debug=True, log_level='DEBUG')

# View transformation steps
result = engine.rewrite(text, verbose=True)
```

### Common Error Patterns

**Issue**: Unexpected transformation results

**Solution**: Use step-by-step analysis

```python
# Analyze transformation steps
analysis = engine.analyze_transformation(input_text)
for step in analysis.steps:
    print(f"Step {step.number}: {step.description}")
    print(f"Input: {step.input}")
    print(f"Output: {step.output}")
    print(f"Rule applied: {step.rule}")
```

## Getting Help

### Community Resources

- **GitHub Issues**: Report bugs and request features at [repository URL]
- **Documentation**: Comprehensive guides at [docs URL]
- **Examples**: Sample code and use cases in the `examples/` directory

### Diagnostic Information

When reporting issues, include this diagnostic information:

```python
import sanskrit_rewrite_engine as sre
import sys
import platform

print(f"Sanskrit Rewrite Engine version: {sre.__version__}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Available transformers: {sre.list_transformers()}")
```

### Performance Profiling

For performance issues, use the built-in profiler:

```python
# Profile transformation performance
with engine.profile() as profiler:
    result = engine.rewrite(large_text)

# View profiling results
profiler.print_stats()
```

This troubleshooting guide covers the most common issues encountered when using the Sanskrit Rewrite Engine. For additional help, consult the full documentation or reach out to the community.
