# Sanskrit Rewrite Engine - Frequently Asked Questions (FAQ)

## General Questions

### What is the Sanskrit Rewrite Engine?

The Sanskrit Rewrite Engine is a sophisticated computational linguistics system designed for Sanskrit text processing, grammatical analysis, and transformation. It uses token-based processing with rule-based transformations to perform operations like sandhi resolution, morphological analysis, and compound formation.

### Who should use this engine?

The engine is designed for:
- **Sanskrit scholars and linguists** working with digital texts
- **Software developers** building Sanskrit processing applications
- **Computational linguists** researching Sanskrit grammar
- **Researchers** analyzing large Sanskrit corpora
- **Students** learning Sanskrit grammar through computational tools

### What makes this different from simple string replacement?

Unlike basic string replacement, the Sanskrit Rewrite Engine:
- Uses **token-based processing** for linguistic accuracy
- Applies **rule priorities** and **convergence detection**
- Provides **detailed transformation traces** for debugging
- Supports **context-sensitive transformations**
- Includes **guard systems** to prevent infinite loops
- Offers **extensible rule definitions** in JSON format

## Installation and Setup

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 2GB RAM
- 200MB storage
- Internet connection for installation

**Recommended:**
- Python 3.9+
- 4GB+ RAM
- 1GB+ storage
- Multi-core processor

### How do I install the engine?

```bash
# Basic installation
pip install sanskrit-rewrite-engine

# With development tools
pip install sanskrit-rewrite-engine[dev]

# With all features
pip install sanskrit-rewrite-engine[all]
```

### Why do I get import errors after installation?

Common causes and solutions:

1. **Wrong Python environment**:
   ```bash
   # Check which Python you're using
   which python
   pip show sanskrit-rewrite-engine
   ```

2. **Virtual environment not activated**:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Package not installed in current environment**:
   ```bash
   pip install sanskrit-rewrite-engine
   ```

### How do I update to the latest version?

```bash
# Update to latest version
pip install --upgrade sanskrit-rewrite-engine

# Check current version
python -c "import sanskrit_rewrite_engine; print(sanskrit_rewrite_engine.__version__)"
```

## Usage Questions

### How do I process basic Sanskrit text?

```python
from sanskrit_rewrite_engine import SanskritRewriteEngine

# Create engine
engine = SanskritRewriteEngine()

# Process text
result = engine.process("rāma + iti")
print(result.get_output_text())  # "rāmeti"
```

### What input formats are supported?

The engine supports:
- **IAST transliteration** (recommended): `rāma`, `devī`, `śāstra`
- **Simple ASCII**: `rama`, `devi`, `shastra`
- **Mixed formats**: `rāma + iti`
- **Morphological markers**: `rāma + GEN` (with appropriate rules)

### How do I enable detailed tracing?

```python
# Enable tracing during processing
result = engine.process("rāma + iti", enable_tracing=True)

# Examine traces
for pass_trace in result.traces:
    print(f"Pass {pass_trace.pass_number}:")
    for transform in pass_trace.transformations:
        print(f"  Applied {transform.rule_name} at position {transform.index}")
```

### How do I add custom rules?

```python
from sanskrit_rewrite_engine import Rule, Token, TokenKind

def match_custom(tokens, index):
    """Match custom pattern"""
    return (index < len(tokens) - 1 and 
            tokens[index].text == "custom" and 
            tokens[index + 1].text == "pattern")

def apply_custom(tokens, index):
    """Apply custom transformation"""
    new_token = Token("result", TokenKind.OTHER, set(), {})
    return tokens[:index] + [new_token] + tokens[index + 2:], index + 1

custom_rule = Rule(
    priority=1,
    id=999,
    name="custom_rule",
    description="Custom transformation",
    match_fn=match_custom,
    apply_fn=apply_custom
)

engine.add_rule(custom_rule)
```

### How do I load rules from JSON files?

```python
# Load rules from JSON file
engine.load_rules_from_file("my_rules.json")

# JSON format example:
{
  "rules": [
    {
      "id": "vowel_sandhi_a_i",
      "name": "a + i → e",
      "description": "Combine 'a' and 'i' vowels",
      "pattern": "a\\s*\\+\\s*i",
      "replacement": "e",
      "priority": 1
    }
  ]
}
```

## Performance Questions

### Why is processing slow?

Common causes and solutions:

1. **Tracing enabled in production**:
   ```python
   # Disable tracing for better performance
   result = engine.process(text, enable_tracing=False)
   ```

2. **Too many transformation passes**:
   ```python
   # Limit maximum passes
   result = engine.process(text, max_passes=10)
   ```

3. **Heavy dependencies loaded**:
   ```bash
   # Install only needed extras
   pip install sanskrit-rewrite-engine[dev]  # Not [all]
   ```

4. **Large input texts**:
   ```python
   # Process in chunks for very large texts
   chunks = split_text_into_chunks(large_text)
   results = [engine.process(chunk) for chunk in chunks]
   ```

### How can I improve performance?

```python
from sanskrit_rewrite_engine.config import EngineConfig

# Performance-optimized configuration
config = EngineConfig(
    performance_mode=True,
    enable_tracing=False,
    max_passes=10,
    enable_caching=True
)

engine = SanskritRewriteEngine(config=config)
```

### How much memory does the engine use?

Typical memory usage:
- **Basic engine**: 50-100MB
- **With all rules loaded**: 100-200MB
- **With tracing enabled**: +50-100MB per processing session
- **Large text processing**: Scales with input size

## API and Integration Questions

### How do I use the REST API?

```bash
# Start server
sanskrit-web --port 8000

# Process text via API
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "rāma + iti", "trace": true}'
```

### What's the API response format?

```json
{
  "input_text": "rāma + iti",
  "output_text": "rāmeti",
  "transformations_applied": ["vowel_sandhi_a_i"],
  "converged": true,
  "passes": 2,
  "trace": [...],
  "success": true,
  "error_message": null
}
```

### How do I integrate with Flask/Django?

```python
# Flask integration
from flask import Flask, request, jsonify
from sanskrit_rewrite_engine import SanskritRewriteEngine

app = Flask(__name__)
engine = SanskritRewriteEngine()

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    result = engine.process(data['text'])
    return jsonify({
        'output': result.get_output_text(),
        'success': result.success
    })
```

### How do I handle errors in the API?

```python
try:
    result = engine.process(text)
    if result.success:
        return result.get_output_text()
    else:
        print(f"Processing failed: {result.error_message}")
except Exception as e:
    print(f"Engine error: {str(e)}")
```

## Rule System Questions

### How do rule priorities work?

Rules are applied in priority order (lower numbers = higher priority):
- **Priority 0**: Meta-rules and control logic
- **Priority 1-5**: Phonological rules (sandhi)
- **Priority 6-10**: Morphological rules
- **Priority 11-15**: Syntactic rules
- **Priority 16+**: Cleanup and formatting rules

### What happens if rules conflict?

The engine handles conflicts through:
1. **Priority ordering**: Higher priority rules apply first
2. **Guard system**: Prevents infinite loops
3. **Application limits**: Rules can have maximum application counts
4. **Convergence detection**: Stops when no more changes occur

### How do I debug rule application?

```python
# Enable detailed tracing
result = engine.process(text, enable_tracing=True)

# Examine rule applications
for pass_trace in result.traces:
    for transform in pass_trace.transformations:
        print(f"Rule: {transform.rule_name}")
        print(f"Position: {transform.index}")
        print(f"Before: {[t.text for t in transform.tokens_before]}")
        print(f"After: {[t.text for t in transform.tokens_after]}")
```

### Can I disable specific rules?

```python
# Disable rule by ID
engine.disable_rule(rule_id=123)

# Enable rule by ID
engine.enable_rule(rule_id=123)

# Get list of active rules
active_rules = engine.get_active_rules()
```

## Linguistic Questions

### What Sanskrit features are supported?

Currently supported:
- **Vowel sandhi**: a+i→e, a+u→o, etc.
- **Basic consonant sandhi**: t+c→tc, etc.
- **Compound formation**: deva+rāja→devarāja
- **Morphological markers**: Basic case/number marking
- **Cleanup operations**: Removing processing markers

### What about Vedic Sanskrit?

The engine focuses on Classical Sanskrit but can be extended:
```python
# Add Vedic-specific rules
vedic_rules = load_vedic_rule_set()
for rule in vedic_rules:
    engine.add_rule(rule)
```

### How accurate is the grammatical analysis?

The engine provides:
- **High accuracy** for basic sandhi operations (95%+)
- **Good accuracy** for compound analysis (80-90%)
- **Basic support** for morphological analysis
- **Extensible framework** for adding more sophisticated rules

### Can it handle manuscript variations?

The engine can be configured for manuscript-specific variations:
```python
# Load manuscript-specific rules
engine.load_rules_from_file("manuscript_variants.json")

# Configure for specific manuscript tradition
config = EngineConfig(
    manuscript_tradition="kashmir_shaivism",
    allow_variants=True
)
```

## Development Questions

### How do I contribute to the project?

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Submit pull request** with description

See the [Contributing Guide](contributing_guide.md) for details.

### How do I run the tests?

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=sanskrit_rewrite_engine

# Run specific test file
python -m pytest tests/test_engine.py -v

# Run performance tests
python -m pytest tests/test_performance.py --benchmark-only
```

### How do I add new rule types?

```python
# Extend the Rule class
from sanskrit_rewrite_engine.rule import Rule

class ConditionalRule(Rule):
    def __init__(self, condition_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_fn = condition_fn
    
    def can_apply(self, tokens, index):
        if not super().can_apply():
            return False
        return self.condition_fn(tokens, index)
```

### How do I profile performance?

```python
import cProfile
import pstats

# Profile engine performance
profiler = cProfile.Profile()
profiler.enable()

# Run your code
for _ in range(1000):
    result = engine.process("test text")

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Troubleshooting

### Common Error Messages

#### "TokenizationError: Invalid character"
```python
# Cause: Unsupported characters in input
# Solution: Clean input text or extend tokenizer
text = clean_sanskrit_text(input_text)
result = engine.process(text)
```

#### "ConvergenceError: Maximum passes exceeded"
```python
# Cause: Rules creating infinite loops
# Solution: Check rule definitions or increase limit
result = engine.process(text, max_passes=50)
```

#### "RuleApplicationError: Rule function failed"
```python
# Cause: Bug in custom rule function
# Solution: Debug rule function
def debug_rule_function(tokens, index):
    try:
        return original_function(tokens, index)
    except Exception as e:
        print(f"Rule error at {index}: {e}")
        raise
```

### Getting Help

1. **Check documentation**: Start with user guides and API reference
2. **Search issues**: Look for similar problems on GitHub
3. **Enable debugging**: Use tracing and logging for more information
4. **Ask community**: Post questions in GitHub discussions
5. **Report bugs**: Create detailed issue reports with examples

### Performance Issues

#### "Processing is very slow"
- Disable tracing in production
- Reduce maximum passes
- Use performance mode
- Check for rule conflicts

#### "High memory usage"
- Disable detailed tracing
- Process texts in smaller chunks
- Enable garbage collection
- Monitor memory usage

#### "Server timeouts"
- Increase server timeout settings
- Use async processing for large texts
- Implement request queuing
- Scale horizontally with multiple instances

## Advanced Topics

### Custom Tokenizers

```python
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer

class CustomTokenizer(SanskritTokenizer):
    def __init__(self):
        super().__init__()
        # Add custom character sets
        self.custom_vowels = {'ṝ', 'ḹ'}  # Long r, l
        
    def identify_token_kind(self, text):
        if text in self.custom_vowels:
            return TokenKind.VOWEL
        return super().identify_token_kind(text)
```

### Plugin System

```python
from sanskrit_rewrite_engine.plugins import Plugin

class MorphologyPlugin(Plugin):
    def register_rules(self, registry):
        # Add morphological rules
        pass
    
    def register_tokenizer_extensions(self, tokenizer):
        # Extend tokenization
        pass

# Load plugin
engine.load_plugin(MorphologyPlugin())
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["rāma + iti", "deva + indra", "mahā + ātman"]

# Sequential processing
results = [engine.process(text) for text in texts]

# Batch processing (if available)
results = engine.process_batch(texts)
```

---

This FAQ covers the most common questions about the Sanskrit Rewrite Engine. For more detailed information, see the complete documentation in the [User Guide](user_guide_complete.md), [Developer Guide](developer_guide.md), and [API Reference](api_reference.md).

*Last updated: January 15, 2024*