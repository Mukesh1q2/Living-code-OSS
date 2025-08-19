# Sanskrit Rewrite Engine - Developer Guide

## Overview

The Sanskrit Rewrite Engine is a sophisticated token-level transformation system that serves as the core of the Symbolic Sanskrit Core (SSC). This guide provides comprehensive information for developers working with or extending the engine.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Advanced Features](#advanced-features)
5. [API Reference](#api-reference)
6. [Testing](#testing)
7. [Performance Optimization](#performance-optimization)
8. [Extension Points](#extension-points)
9. [Common Patterns](#common-patterns)
10. [Error Handling](#error-handling)
11. [Best Practices](#best-practices)
12. [Contributing](#contributing)
13. [Support](#support)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from sanskrit_rewrite_engine import SanskritRewriteEngine, SanskritTokenizer, RuleRegistry

# Initialize components
tokenizer = SanskritTokenizer()
registry = RuleRegistry()
engine = SanskritRewriteEngine(tokenizer, registry)

# Process Sanskrit text
result = engine.process("rāma + iti")
print(result.get_output_text())  # "rāmeti"
```

## Architecture Overview

The engine follows a multi-stage pipeline:

1. **Tokenization**: Convert text to typed tokens
2. **Rule Application**: Apply transformation rules with guards
3. **Iterative Processing**: Continue until convergence
4. **Trace Collection**: Record all transformations

## Core Components

### Token System

Tokens are the fundamental units of processing:

```python
from sanskrit_rewrite_engine.token import Token, TokenKind

# Create tokens
vowel_token = Token("a", TokenKind.VOWEL, {"initial"}, {})
consonant_token = Token("k", TokenKind.CONSONANT, set(), {})

# Token operations
vowel_token.add_tag("stressed")
vowel_token.set_meta("position", 0)
```

### Rule Definition

Rules define transformations with match and apply functions:

```python
from sanskrit_rewrite_engine.rule import Rule

def match_vowel_sandhi(tokens, index):
    """Match 'a' + 'i' pattern"""
    if index + 1 >= len(tokens):
        return False
    return (tokens[index].text == "a" and 
            tokens[index + 1].text == "i")

def apply_vowel_sandhi(tokens, index):
    """Transform 'a' + 'i' to 'e'"""
    new_token = Token("e", TokenKind.VOWEL, {"sandhi"}, {})
    return tokens[:index] + [new_token] + tokens[index + 2:], index + 1

# Create rule
sandhi_rule = Rule(
    priority=1,
    id=1,
    name="a_i_sandhi",
    description="Transform a + i to e",
    match_fn=match_vowel_sandhi,
    apply_fn=apply_vowel_sandhi
)
```

### Rule Registry

Manage collections of rules:

```python
from sanskrit_rewrite_engine.rule import RuleRegistry

registry = RuleRegistry()
registry.add_rule(sandhi_rule)

# Load predefined rule sets
registry.load_sandhi_rules()
registry.load_morphological_rules()
```

## Advanced Features

### Meta-Rules

Control rule application dynamically:

```python
from sanskrit_rewrite_engine.rule import MetaRule

def enable_compound_rules(tokens, registry):
    """Enable compound rules when '+' marker found"""
    has_compound = any(token.text == "+" for token in tokens)
    if has_compound:
        registry.enable_rule_group("compound")

meta_rule = MetaRule(
    priority=0,
    id=100,
    name="compound_activation",
    condition_fn=lambda tokens, registry: True,
    action_fn=enable_compound_rules
)
```

### Tracing and Debugging

Monitor rule applications:

```python
result = engine.process("text", enable_tracing=True)

# Examine traces
for pass_trace in result.traces:
    print(f"Pass {pass_trace.pass_number}:")
    for transform in pass_trace.transformations:
        print(f"  {transform.rule_name} at {transform.index}")
```

### Custom Tokenizers

Extend tokenization for specific needs:

```python
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer

class CustomTokenizer(SanskritTokenizer):
    def __init__(self):
        super().__init__()
        self.custom_patterns = {...}
    
    def _identify_token_kind(self, text):
        # Custom logic
        return super()._identify_token_kind(text)
```

## API Reference

### Core Classes

#### SanskritRewriteEngine

Main processing engine.

**Methods:**
- `process(text: str, max_passes: int = 20) -> RewriteResult`
- `add_rule(rule: Rule) -> None`
- `remove_rule(rule_id: int) -> None`

#### Token

Represents a text unit with linguistic properties.

**Properties:**
- `text: str` - Token content
- `kind: TokenKind` - Token type (VOWEL, CONSONANT, etc.)
- `tags: Set[str]` - Semantic tags
- `meta: Dict[str, Any]` - Metadata

**Methods:**
- `has_tag(tag: str) -> bool`
- `add_tag(tag: str) -> None`
- `get_meta(key: str, default=None) -> Any`

#### Rule

Defines a transformation rule.

**Properties:**
- `priority: int` - Application priority (lower = earlier)
- `match_fn: Callable` - Pattern matching function
- `apply_fn: Callable` - Transformation function
- `max_applications: Optional[int]` - Application limit

### Utility Functions

```python
from sanskrit_rewrite_engine.utils import (
    is_vowel, is_consonant, create_sandhi_rule,
    load_rule_set, validate_sanskrit_text
)

# Check token types
if is_vowel(token):
    # Handle vowel

# Create common rules
rule = create_sandhi_rule("a", "i", "e")

# Load predefined rules
rules = load_rule_set("classical_sandhi.json")
```

## Testing

### Unit Tests

```python
import unittest
from sanskrit_rewrite_engine import SanskritRewriteEngine

class TestSandhiRules(unittest.TestCase):
    def setUp(self):
        self.engine = SanskritRewriteEngine()
        
    def test_vowel_sandhi(self):
        result = self.engine.process("rāma + iti")
        self.assertEqual(result.get_output_text(), "rāmeti")
```

### Integration Tests

```python
def test_complex_transformation():
    """Test multi-rule application"""
    engine = SanskritRewriteEngine()
    engine.load_all_rules()
    
    input_text = "deva + indra + iti"
    result = engine.process(input_text)
    
    assert result.converged
    assert len(result.traces) > 1
```

## Performance Optimization

### Rule Optimization

- Use specific match conditions to avoid unnecessary checks
- Set appropriate priorities to minimize rule conflicts
- Limit rule applications when possible

```python
# Efficient rule
def efficient_match(tokens, index):
    # Quick checks first
    if index >= len(tokens) - 1:
        return False
    if tokens[index].kind != TokenKind.VOWEL:
        return False
    # More expensive checks last
    return complex_pattern_check(tokens, index)
```

### Memory Management

- Use token pooling for common patterns
- Configure trace retention based on needs
- Process large texts in chunks

```python
# Configure for performance
config = EngineConfig(
    enable_tracing=False,  # Disable for production
    performance_mode=True,
    max_passes=10
)
engine = SanskritRewriteEngine(config=config)
```

## Extension Points

### Custom Rules

Implement domain-specific transformations:

```python
class VedicSandhiRule(Rule):
    """Vedic-specific sandhi rules"""
    
    def __init__(self):
        super().__init__(
            priority=0,  # High priority
            name="vedic_sandhi",
            match_fn=self.match_vedic_pattern,
            apply_fn=self.apply_vedic_transformation
        )
    
    def match_vedic_pattern(self, tokens, index):
        # Vedic-specific logic
        pass
```

### Plugin System

Extend functionality through plugins:

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

## Common Patterns

### Sandhi Processing

```python
# Standard vowel sandhi
engine.add_sandhi_rules([
    ("a", "i", "e"),
    ("a", "u", "o"),
    ("a", "ā", "ā")
])

# Process with sandhi
result = engine.process("rāma + iti")
```

### Morphological Analysis

```python
# Add morphological markers
engine.add_morphology_rules()

# Process inflected forms
result = engine.process("rāma : GEN")  # → rāmasya
```

### Compound Formation

```python
# Enable compound processing
engine.enable_compound_rules()

# Process compounds
result = engine.process("deva + rāja")  # → devarāja
```

## Error Handling

### Common Errors

1. **TokenizationError**: Invalid input text
2. **RuleApplicationError**: Rule function failures
3. **ConvergenceError**: Non-converging transformations
4. **GuardViolationError**: Infinite loop prevention

### Error Recovery

```python
try:
    result = engine.process(text)
except ConvergenceError as e:
    # Handle non-convergence
    partial_result = e.partial_result
    print(f"Stopped after {e.passes} passes")
except RuleApplicationError as e:
    # Handle rule errors
    print(f"Rule {e.rule.name} failed at position {e.index}")
```

## Best Practices

### Rule Design

1. **Specificity**: Make match functions as specific as possible
2. **Idempotency**: Ensure rules don't conflict with their own output
3. **Documentation**: Include clear descriptions and examples
4. **Testing**: Test rules in isolation and combination

### Performance

1. **Profiling**: Use built-in profiling for optimization
2. **Caching**: Cache expensive computations
3. **Batching**: Process similar texts together
4. **Monitoring**: Track performance metrics

### Debugging

1. **Tracing**: Enable detailed tracing during development
2. **Logging**: Use structured logging for production
3. **Validation**: Validate inputs and outputs
4. **Testing**: Comprehensive test coverage

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## Support

- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the knowledge base
- **Examples**: See the examples/ directory