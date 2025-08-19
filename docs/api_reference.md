# Sanskrit Rewrite Engine - API Reference

## Overview

This document provides comprehensive API documentation for the Sanskrit Rewrite Engine. All classes, methods, and functions are documented with parameters, return types, and usage examples.

## Quick Start Examples

### Basic Text Processing
```python
from sanskrit_rewrite_engine import SanskritRewriteEngine

# Initialize engine
engine = SanskritRewriteEngine()

# Process Sanskrit text
result = engine.process("rāma + iti")
print(f"Output: {result.get_output_text()}")  # "rāmeti"
```

### Advanced Processing with Configuration
```python
from sanskrit_rewrite_engine import SanskritRewriteEngine
from sanskrit_rewrite_engine.config import EngineConfig

# Configure engine
config = EngineConfig(
    max_passes=30,
    enable_tracing=True,
    performance_mode=False
)

engine = SanskritRewriteEngine(config=config)
result = engine.process("deva + indra + iti")

# Examine detailed results
print(f"Converged: {result.converged}")
print(f"Passes: {result.passes}")
print(f"Transformations: {result.get_transformation_summary()}")
```

### REST API Usage
```bash
# Start server
sanskrit-web --port 8000

# Process text via API
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "rāma + iti",
    "max_passes": 20,
    "enable_tracing": true
  }'
```

## Table of Contents

1. [Core Classes](#core-classes)
2. [Configuration Classes](#configuration-classes)
3. [Utility Functions](#utility-functions)
4. [Exception Classes](#exception-classes)
5. [Constants](#constants)
6. [Usage Examples](#usage-examples)
7. [Integration Patterns](#integration-patterns)
8. [Performance Considerations](#performance-considerations)

## Core Classes

### SanskritRewriteEngine

Main processing engine for Sanskrit text transformations.

```python
class SanskritRewriteEngine:
    def __init__(self, tokenizer: SanskritTokenizer, registry: RuleRegistry, config: Optional[EngineConfig] = None)
```

#### Constructor Parameters

- `tokenizer` (SanskritTokenizer): Tokenizer instance for text processing
- `registry` (RuleRegistry): Rule registry containing transformation rules
- `config` (EngineConfig, optional): Configuration settings

#### Methods

##### process()

```python
def process(self, text: str, max_passes: int = 20, enable_tracing: bool = True) -> RewriteResult
```

Process Sanskrit text through the transformation pipeline.

**Parameters:**
- `text` (str): Input Sanskrit text
- `max_passes` (int): Maximum number of transformation passes
- `enable_tracing` (bool): Whether to collect detailed traces

**Returns:**
- `RewriteResult`: Complete processing result with traces

**Example:**
```python
engine = SanskritRewriteEngine(tokenizer, registry)
result = engine.process("rāma + iti")
print(result.get_output_text())  # "rāmeti"
```

##### add_rule()

```python
def add_rule(self, rule: Rule) -> None
```

Add a transformation rule to the engine.

**Parameters:**
- `rule` (Rule): Rule to add

**Example:**
```python
rule = Rule(priority=1, id=1, name="test", match_fn=match, apply_fn=apply)
engine.add_rule(rule)
```

##### remove_rule()

```python
def remove_rule(self, rule_id: int) -> bool
```

Remove a rule by ID.

**Parameters:**
- `rule_id` (int): ID of rule to remove

**Returns:**
- `bool`: True if rule was removed, False if not found

##### get_active_rules()

```python
def get_active_rules(self) -> List[Rule]
```

Get list of currently active rules.

**Returns:**
- `List[Rule]`: Active rules sorted by priority

##### reset()

```python
def reset(self) -> None
```

Reset engine state, clearing guards and application counters.

### Token

Represents a unit of text with linguistic properties.

```python
@dataclass
class Token:
    text: str
    kind: TokenKind
    tags: Set[str]
    meta: Dict[str, Any]
    position: Optional[int] = None
```

#### Properties

- `text` (str): Token content
- `kind` (TokenKind): Token type (VOWEL, CONSONANT, MARKER, OTHER)
- `tags` (Set[str]): Semantic tags
- `meta` (Dict[str, Any]): Metadata dictionary
- `position` (int, optional): Original position in text

#### Methods

##### has_tag()

```python
def has_tag(self, tag: str) -> bool
```

Check if token has a specific tag.

**Parameters:**
- `tag` (str): Tag to check

**Returns:**
- `bool`: True if tag exists

##### add_tag()

```python
def add_tag(self, tag: str) -> None
```

Add a tag to the token.

**Parameters:**
- `tag` (str): Tag to add

##### remove_tag()

```python
def remove_tag(self, tag: str) -> bool
```

Remove a tag from the token.

**Parameters:**
- `tag` (str): Tag to remove

**Returns:**
- `bool`: True if tag was removed

##### get_meta()

```python
def get_meta(self, key: str, default: Any = None) -> Any
```

Get metadata value.

**Parameters:**
- `key` (str): Metadata key
- `default` (Any): Default value if key not found

**Returns:**
- `Any`: Metadata value or default

##### set_meta()

```python
def set_meta(self, key: str, value: Any) -> None
```

Set metadata value.

**Parameters:**
- `key` (str): Metadata key
- `value` (Any): Value to set

### TokenKind

Enumeration of token types.

```python
class TokenKind(Enum):
    VOWEL = "VOWEL"
    CONSONANT = "CONSONANT"
    MARKER = "MARKER"
    OTHER = "OTHER"
```

### Rule

Defines a transformation rule.

```python
@dataclass
class Rule:
    priority: int
    id: int
    name: str
    description: str
    match_fn: Callable[[List[Token], int], bool]
    apply_fn: Callable[[List[Token], int], Tuple[List[Token], int]]
    max_applications: Optional[int] = None
    applications: int = 0
    active: bool = True
    sutra_ref: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
```

#### Properties

- `priority` (int): Application priority (lower = earlier)
- `id` (int): Unique rule identifier
- `name` (str): Rule name
- `description` (str): Human-readable description
- `match_fn` (Callable): Function to test if rule applies
- `apply_fn` (Callable): Function to apply transformation
- `max_applications` (int, optional): Maximum applications allowed
- `applications` (int): Current application count
- `active` (bool): Whether rule is active
- `sutra_ref` (str, optional): Pāṇini sūtra reference
- `meta_data` (Dict): Additional metadata

#### Methods

##### can_apply()

```python
def can_apply(self) -> bool
```

Check if rule can be applied (active and under limit).

**Returns:**
- `bool`: True if rule can apply

##### reset_applications()

```python
def reset_applications(self) -> None
```

Reset application counter to zero.

##### increment_applications()

```python
def increment_applications(self) -> None
```

Increment application counter.

### RuleRegistry

Manages collections of transformation rules.

```python
class RuleRegistry:
    def __init__(self)
```

#### Methods

##### add_rule()

```python
def add_rule(self, rule: Rule) -> None
```

Add a rule to the registry.

**Parameters:**
- `rule` (Rule): Rule to add

**Raises:**
- `ValueError`: If rule ID already exists

##### remove_rule()

```python
def remove_rule(self, rule_id: int) -> bool
```

Remove a rule by ID.

**Parameters:**
- `rule_id` (int): ID of rule to remove

**Returns:**
- `bool`: True if rule was removed

##### get_rule()

```python
def get_rule(self, rule_id: int) -> Optional[Rule]
```

Get rule by ID.

**Parameters:**
- `rule_id` (int): Rule ID

**Returns:**
- `Optional[Rule]`: Rule if found, None otherwise

##### get_active_rules()

```python
def get_active_rules(self) -> List[Rule]
```

Get all active rules sorted by priority.

**Returns:**
- `List[Rule]`: Active rules

##### enable_rule()

```python
def enable_rule(self, rule_id: int) -> bool
```

Enable a rule.

**Parameters:**
- `rule_id` (int): Rule ID

**Returns:**
- `bool`: True if rule was enabled

##### disable_rule()

```python
def disable_rule(self, rule_id: int) -> bool
```

Disable a rule.

**Parameters:**
- `rule_id` (int): Rule ID

**Returns:**
- `bool`: True if rule was disabled

##### load_rule_set()

```python
def load_rule_set(self, filename: str) -> int
```

Load rules from JSON file.

**Parameters:**
- `filename` (str): Path to rule set file

**Returns:**
- `int`: Number of rules loaded

##### save_rule_set()

```python
def save_rule_set(self, filename: str, rule_ids: Optional[List[int]] = None) -> None
```

Save rules to JSON file.

**Parameters:**
- `filename` (str): Output file path
- `rule_ids` (List[int], optional): Specific rules to save (all if None)

### SanskritTokenizer

Tokenizes Sanskrit text into typed tokens.

```python
class SanskritTokenizer:
    def __init__(self, vowels: Optional[Set[str]] = None, consonants: Optional[Set[str]] = None)
```

#### Constructor Parameters

- `vowels` (Set[str], optional): Set of vowel characters
- `consonants` (Set[str], optional): Set of consonant characters

#### Methods

##### tokenize()

```python
def tokenize(self, text: str) -> List[Token]
```

Tokenize Sanskrit text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `List[Token]`: List of tokens

**Example:**
```python
tokenizer = SanskritTokenizer()
tokens = tokenizer.tokenize("rāma")
# [Token("r", CONSONANT), Token("ā", VOWEL), Token("m", CONSONANT), Token("a", VOWEL)]
```

##### add_vowel()

```python
def add_vowel(self, vowel: str) -> None
```

Add a vowel character.

**Parameters:**
- `vowel` (str): Vowel to add

##### add_consonant()

```python
def add_consonant(self, consonant: str) -> None
```

Add a consonant character.

**Parameters:**
- `consonant` (str): Consonant to add

##### is_vowel()

```python
def is_vowel(self, char: str) -> bool
```

Check if character is a vowel.

**Parameters:**
- `char` (str): Character to check

**Returns:**
- `bool`: True if vowel

##### is_consonant()

```python
def is_consonant(self, char: str) -> bool
```

Check if character is a consonant.

**Parameters:**
- `char` (str): Character to check

**Returns:**
- `bool`: True if consonant

### RewriteResult

Contains the complete result of text processing.

```python
@dataclass
class RewriteResult:
    input_text: str
    input_tokens: List[Token]
    output_tokens: List[Token]
    converged: bool
    passes: int
    traces: List[PassTrace]
    errors: List[str]
```

#### Properties

- `input_text` (str): Original input text
- `input_tokens` (List[Token]): Initial tokens
- `output_tokens` (List[Token]): Final tokens
- `converged` (bool): Whether processing converged
- `passes` (int): Number of passes executed
- `traces` (List[PassTrace]): Detailed transformation traces
- `errors` (List[str]): Any errors encountered

#### Methods

##### get_output_text()

```python
def get_output_text(self) -> str
```

Get final output as text string.

**Returns:**
- `str`: Concatenated output tokens

##### get_transformation_summary()

```python
def get_transformation_summary(self) -> Dict[str, int]
```

Get summary of transformations applied.

**Returns:**
- `Dict[str, int]`: Rule names and application counts

##### get_rule_trace()

```python
def get_rule_trace(self) -> List[str]
```

Get ordered list of rules applied.

**Returns:**
- `List[str]`: Rule names in application order

### PassTrace

Trace information for a single processing pass.

```python
@dataclass
class PassTrace:
    pass_number: int
    tokens_before: List[Token]
    tokens_after: List[Token]
    transformations: List[TransformationTrace]
    meta_rule_applications: List[str]
```

#### Properties

- `pass_number` (int): Pass number (1-indexed)
- `tokens_before` (List[Token]): Tokens at start of pass
- `tokens_after` (List[Token]): Tokens at end of pass
- `transformations` (List[TransformationTrace]): Individual transformations
- `meta_rule_applications` (List[str]): Meta-rules applied

### TransformationTrace

Trace information for a single transformation.

```python
@dataclass
class TransformationTrace:
    rule_name: str
    rule_id: int
    index: int
    tokens_before: List[Token]
    tokens_after: List[Token]
    timestamp: datetime
```

#### Properties

- `rule_name` (str): Name of applied rule
- `rule_id` (int): ID of applied rule
- `index` (int): Position where rule was applied
- `tokens_before` (List[Token]): Tokens before transformation
- `tokens_after` (List[Token]): Tokens after transformation
- `timestamp` (datetime): When transformation occurred

## Configuration Classes

### EngineConfig

Configuration settings for the rewrite engine.

```python
@dataclass
class EngineConfig:
    max_passes: int = 20
    enable_tracing: bool = True
    trace_detail_level: str = "full"
    performance_mode: bool = False
    rule_set_path: Optional[str] = None
    custom_tokenizer_config: Optional[Dict] = None
    memory_limit: Optional[int] = None
    timeout_seconds: Optional[int] = None
```

#### Properties

- `max_passes` (int): Maximum transformation passes
- `enable_tracing` (bool): Whether to collect traces
- `trace_detail_level` (str): Trace detail ("minimal", "standard", "full")
- `performance_mode` (bool): Enable performance optimizations
- `rule_set_path` (str, optional): Path to default rule set
- `custom_tokenizer_config` (Dict, optional): Custom tokenizer settings
- `memory_limit` (int, optional): Memory limit in MB
- `timeout_seconds` (int, optional): Processing timeout

## Utility Functions

### Rule Builders

#### create_sandhi_rule()

```python
def create_sandhi_rule(from_vowel1: str, from_vowel2: str, to_vowel: str, priority: int = 1) -> Rule
```

Create a vowel sandhi rule.

**Parameters:**
- `from_vowel1` (str): First vowel
- `from_vowel2` (str): Second vowel
- `to_vowel` (str): Result vowel
- `priority` (int): Rule priority

**Returns:**
- `Rule`: Configured sandhi rule

**Example:**
```python
rule = create_sandhi_rule("a", "i", "e")
```

#### create_compound_rule()

```python
def create_compound_rule(priority: int = 1) -> Rule
```

Create a compound formation rule.

**Parameters:**
- `priority` (int): Rule priority

**Returns:**
- `Rule`: Compound formation rule

#### create_cleanup_rule()

```python
def create_cleanup_rule(marker: str, priority: int = 10) -> Rule
```

Create a marker cleanup rule.

**Parameters:**
- `marker` (str): Marker to remove
- `priority` (int): Rule priority

**Returns:**
- `Rule`: Cleanup rule

### Text Processing Utilities

#### is_vowel()

```python
def is_vowel(token: Token) -> bool
```

Check if token is a vowel.

**Parameters:**
- `token` (Token): Token to check

**Returns:**
- `bool`: True if vowel token

#### is_consonant()

```python
def is_consonant(token: Token) -> bool
```

Check if token is a consonant.

**Parameters:**
- `token` (Token): Token to check

**Returns:**
- `bool`: True if consonant token

#### tokens_to_text()

```python
def tokens_to_text(tokens: List[Token]) -> str
```

Convert token list to text string.

**Parameters:**
- `tokens` (List[Token]): Tokens to convert

**Returns:**
- `str`: Concatenated text

#### text_to_tokens()

```python
def text_to_tokens(text: str, tokenizer: SanskritTokenizer) -> List[Token]
```

Convert text to token list.

**Parameters:**
- `text` (str): Text to tokenize
- `tokenizer` (SanskritTokenizer): Tokenizer to use

**Returns:**
- `List[Token]`: Token list

### Validation Functions

#### validate_rule()

```python
def validate_rule(rule: Rule) -> List[str]
```

Validate rule definition.

**Parameters:**
- `rule` (Rule): Rule to validate

**Returns:**
- `List[str]`: List of validation errors (empty if valid)

#### validate_token_sequence()

```python
def validate_token_sequence(tokens: List[Token]) -> bool
```

Validate token sequence consistency.

**Parameters:**
- `tokens` (List[Token]): Tokens to validate

**Returns:**
- `bool`: True if sequence is valid

#### validate_sanskrit_text()

```python
def validate_sanskrit_text(text: str) -> Tuple[bool, List[str]]
```

Validate Sanskrit text format.

**Parameters:**
- `text` (str): Text to validate

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, error_messages)

## Exception Classes

### RewriteEngineError

Base exception for all engine errors.

```python
class RewriteEngineError(Exception):
    def __init__(self, message: str, context: Optional[Dict] = None)
```

### TokenizationError

Error during tokenization.

```python
class TokenizationError(RewriteEngineError):
    def __init__(self, text: str, position: int, message: str)
```

#### Properties

- `text` (str): Text being tokenized
- `position` (int): Error position
- `message` (str): Error description

### RuleApplicationError

Error during rule application.

```python
class RuleApplicationError(RewriteEngineError):
    def __init__(self, rule: Rule, tokens: List[Token], index: int, message: str)
```

#### Properties

- `rule` (Rule): Rule that failed
- `tokens` (List[Token]): Token context
- `index` (int): Application position
- `message` (str): Error description

### ConvergenceError

Error when processing doesn't converge.

```python
class ConvergenceError(RewriteEngineError):
    def __init__(self, max_passes: int, final_tokens: List[Token])
```

#### Properties

- `max_passes` (int): Maximum passes reached
- `final_tokens` (List[Token]): Final token state

### GuardViolationError

Error when guard conditions are violated.

```python
class GuardViolationError(RewriteEngineError):
    def __init__(self, rule: Rule, index: int, message: str)
```

#### Properties

- `rule` (Rule): Rule that violated guards
- `index` (int): Violation position
- `message` (str): Violation description

## Constants

### Default Character Sets

```python
DEFAULT_VOWELS = {
    'a', 'ā', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ',
    'e', 'ai', 'o', 'au', 'ṃ', 'ḥ'
}

DEFAULT_CONSONANTS = {
    'k', 'kh', 'g', 'gh', 'ṅ',
    'c', 'ch', 'j', 'jh', 'ñ',
    't', 'th', 'd', 'dh', 'n',
    'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
    'p', 'ph', 'b', 'bh', 'm',
    'y', 'r', 'l', 'v',
    'ś', 'ṣ', 's', 'h'
}

DEFAULT_MARKERS = {'+', '_', ':', '[', ']', '(', ')'}
```

### Rule Priorities

```python
class RulePriority:
    META_RULES = 0
    PHONOLOGICAL = 1
    MORPHOLOGICAL = 5
    SYNTACTIC = 10
    CLEANUP = 20
```

## Usage Examples

### Basic Processing

```python
from sanskrit_rewrite_engine import SanskritRewriteEngine, SanskritTokenizer, RuleRegistry

# Setup
tokenizer = SanskritTokenizer()
registry = RuleRegistry()
registry.load_default_rules()
engine = SanskritRewriteEngine(tokenizer, registry)

# Process text
result = engine.process("rāma + iti")
print(f"Input: {result.input_text}")
print(f"Output: {result.get_output_text()}")
print(f"Converged: {result.converged}")
print(f"Passes: {result.passes}")
```

### Custom Rule Creation

```python
from sanskrit_rewrite_engine import Rule, create_sandhi_rule

# Create custom rule
def match_custom(tokens, index):
    return (index < len(tokens) - 1 and 
            tokens[index].text == "x" and 
            tokens[index + 1].text == "y")

def apply_custom(tokens, index):
    new_token = Token("z", TokenKind.OTHER, set(), {})
    return tokens[:index] + [new_token] + tokens[index + 2:], index + 1

custom_rule = Rule(
    priority=1,
    id=999,
    name="custom_xy_rule",
    description="Transform x + y to z",
    match_fn=match_custom,
    apply_fn=apply_custom
)

engine.add_rule(custom_rule)
```

### Advanced Configuration

```python
from sanskrit_rewrite_engine import EngineConfig

config = EngineConfig(
    max_passes=50,
    enable_tracing=True,
    trace_detail_level="full",
    performance_mode=False,
    memory_limit=1024,  # 1GB
    timeout_seconds=30
)

engine = SanskritRewriteEngine(tokenizer, registry, config)
```

This API reference provides complete documentation for all public interfaces in the Sanskrit Rewrite Engine.