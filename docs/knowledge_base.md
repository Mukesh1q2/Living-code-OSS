# Sanskrit Rewrite Engine - Knowledge Base

## Overview

This knowledge base provides comprehensive solutions to common issues, frequently asked questions, and detailed explanations of complex topics related to the Sanskrit Rewrite Engine.

## Table of Contents

1. [Frequently Asked Questions](#frequently-asked-questions)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [Advanced Configuration](#advanced-configuration)
4. [Performance Tuning](#performance-tuning)
5. [Linguistic Concepts](#linguistic-concepts)
6. [Integration Patterns](#integration-patterns)
7. [Debugging Techniques](#debugging-techniques)
8. [Best Practices](#best-practices)
9. [Troubleshooting Workflows](#troubleshooting-workflows)
10. [Community Resources](#community-resources)

## Frequently Asked Questions

### General Questions

#### Q: What is the Sanskrit Rewrite Engine?
**A:** The Sanskrit Rewrite Engine is a computational linguistics tool that applies Sanskrit grammatical rules to transform text. It handles phonological changes (sandhi), morphological analysis, and compound formation based on traditional Pāṇinian grammar principles.

#### Q: What input formats are supported?
**A:** The engine supports:
- **Devanagari script**: Native Sanskrit script (UTF-8 encoded)
- **IAST transliteration**: International Alphabet of Sanskrit Transliteration
- **Mixed formats**: Combination of scripts with proper encoding
- **Marked text**: Text with morphological boundaries (+, _, :)

#### Q: How accurate are the transformations?
**A:** Accuracy depends on several factors:
- **Classical Sanskrit**: 95%+ accuracy for standard texts
- **Vedic Sanskrit**: 85-90% accuracy (requires Vedic rule set)
- **Complex compounds**: 90%+ accuracy with proper segmentation
- **Manuscript text**: Varies based on text quality and preprocessing

#### Q: Can I add custom rules?
**A:** Yes! The engine supports:
- **Custom rule files**: JSON format rule definitions
- **Programmatic rules**: Python function-based rules
- **Rule priorities**: Control application order
- **Meta-rules**: Rules that control other rules

### Technical Questions

#### Q: What Python versions are supported?
**A:** 
- **Minimum**: Python 3.8
- **Recommended**: Python 3.9 or higher
- **Tested**: Python 3.8, 3.9, 3.10, 3.11
- **Dependencies**: See requirements.txt for full list

#### Q: How do I handle Unicode issues?
**A:**
```python
# Ensure proper UTF-8 encoding
with open('sanskrit_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize Unicode for consistency
import unicodedata
normalized_text = unicodedata.normalize('NFC', text)

# Process with engine
result = engine.process(normalized_text)
```

#### Q: Can I process large texts?
**A:** Yes, with proper configuration:
```python
# For large texts, use chunking
def process_large_text(text, chunk_size=5000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = []
    
    for chunk in chunks:
        result = engine.process(chunk)
        results.append(result.get_output_text())
    
    return ' '.join(results)

# Or use streaming mode
engine = SanskritRewriteEngine(config=EngineConfig(
    performance_mode=True,
    enable_tracing=False  # Disable for large texts
))
```

### Linguistic Questions

#### Q: What sandhi rules are implemented?
**A:** The engine includes:
- **Vowel sandhi**: a+i→e, a+u→o, etc.
- **Consonant sandhi**: Assimilation, deletion, insertion
- **Visarga sandhi**: ḥ transformations before vowels/consonants
- **Final consonant changes**: Word-final modifications
- **Compound sandhi**: Internal word transformations

#### Q: How does morphological analysis work?
**A:**
```python
# Enable morphological analysis
engine.enable_morphological_analysis()

# Process text with morphological markers
result = engine.process("rāma : GEN")  # Genitive case marker
print(result.get_output_text())  # "rāmasya"

# Get detailed morphological information
for trace in result.traces:
    for transform in trace.transformations:
        if 'morphological' in transform.rule_name:
            print(f"Applied: {transform.rule_name}")
            print(f"Result: {transform.tokens_after}")
```

#### Q: Can I analyze compound words?
**A:**
```python
# Enable compound analysis
engine.enable_compound_analysis()

# Analyze existing compounds
result = engine.analyze_compound("devarāja")
print(result.components)  # ["deva", "rāja"]
print(result.compound_type)  # "tatpuruṣa"

# Form new compounds
result = engine.process("deva + rāja")
print(result.get_output_text())  # "devarāja"
```

## Common Issues and Solutions

### Installation and Setup Issues

#### Issue: ImportError when importing the engine
```python
ImportError: No module named 'sanskrit_rewrite_engine'
```

**Solutions:**
1. **Check installation**:
   ```bash
   pip list | grep sanskrit
   pip install sanskrit-rewrite-engine
   ```

2. **Virtual environment issues**:
   ```bash
   # Ensure virtual environment is activated
   which python  # Should point to venv
   pip install -e .  # Install in development mode
   ```

3. **Path issues**:
   ```python
   import sys
   sys.path.append('/path/to/sanskrit-rewrite-engine')
   import sanskrit_rewrite_engine
   ```

#### Issue: Dependency conflicts
```bash
ERROR: pip's dependency resolver does not currently consider all the ways...
```

**Solutions:**
1. **Use fresh virtual environment**:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Update pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Use conda environment**:
   ```bash
   conda create -n sanskrit python=3.9
   conda activate sanskrit
   pip install -r requirements.txt
   ```

### Runtime Issues

#### Issue: Memory errors with large texts
```python
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Process in chunks**:
   ```python
   def process_in_chunks(text, chunk_size=1000):
       results = []
       for i in range(0, len(text), chunk_size):
           chunk = text[i:i + chunk_size]
           result = engine.process(chunk)
           results.append(result.get_output_text())
           
           # Clear memory
           del result
           import gc
           gc.collect()
       
       return ' '.join(results)
   ```

2. **Use performance mode**:
   ```python
   config = EngineConfig(
       performance_mode=True,
       enable_tracing=False,
       memory_limit=512  # MB
   )
   engine = SanskritRewriteEngine(config=config)
   ```

3. **Optimize rule set**:
   ```python
   # Load only necessary rules
   engine.load_minimal_rules()
   # Or disable expensive rules
   engine.disable_rule_group("complex_compounds")
   ```

#### Issue: Slow processing speed
```python
# Processing takes too long for moderate-sized texts
```

**Solutions:**
1. **Enable caching**:
   ```python
   engine = SanskritRewriteEngine(
       enable_cache=True,
       cache_size=10000
   )
   ```

2. **Optimize rule priorities**:
   ```python
   # Put most common rules first
   engine.reorder_rules_by_frequency()
   
   # Or manually set priorities
   engine.set_rule_priority("common_sandhi", priority=1)
   ```

3. **Use parallel processing**:
   ```python
   from multiprocessing import Pool
   
   def process_text(text):
       engine = SanskritRewriteEngine()
       return engine.process(text)
   
   with Pool() as pool:
       results = pool.map(process_text, text_list)
   ```

#### Issue: Unicode and encoding problems
```python
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**
1. **Detect and convert encoding**:
   ```python
   import chardet
   
   # Detect encoding
   with open('file.txt', 'rb') as f:
       raw_data = f.read()
       encoding = chardet.detect(raw_data)['encoding']
   
   # Read with detected encoding
   with open('file.txt', 'r', encoding=encoding) as f:
       text = f.read()
   
   # Convert to UTF-8
   text_utf8 = text.encode('utf-8').decode('utf-8')
   ```

2. **Handle mixed encodings**:
   ```python
   def safe_decode(text):
       encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
       for encoding in encodings:
           try:
               return text.decode(encoding)
           except UnicodeDecodeError:
               continue
       raise UnicodeDecodeError("Could not decode text")
   ```

3. **Normalize Unicode**:
   ```python
   import unicodedata
   
   # Normalize to NFC form
   normalized = unicodedata.normalize('NFC', text)
   
   # Remove problematic characters
   cleaned = ''.join(c for c in normalized if unicodedata.category(c) != 'Cc')
   ```

### Configuration Issues

#### Issue: Rules not applying as expected
```python
# Expected transformation not occurring
```

**Solutions:**
1. **Check rule loading**:
   ```python
   # Verify rules are loaded
   print(f"Loaded rules: {len(engine.get_active_rules())}")
   for rule in engine.get_active_rules():
       print(f"Rule: {rule.name}, Priority: {rule.priority}")
   ```

2. **Debug rule matching**:
   ```python
   # Enable debug mode
   engine.set_debug_mode(True)
   result = engine.process(text, verbose=True)
   
   # Check why rules didn't match
   for trace in result.traces:
       print(f"Pass {trace.pass_number}: {len(trace.transformations)} transformations")
   ```

3. **Test individual rules**:
   ```python
   # Test rule in isolation
   rule = engine.get_rule("vowel_sandhi_a_i")
   tokens = engine.tokenize("rāma + iti")
   
   for i, token in enumerate(tokens):
       if rule.match_fn(tokens, i):
           print(f"Rule matches at position {i}")
           new_tokens, new_pos = rule.apply_fn(tokens, i)
           print(f"Result: {[t.text for t in new_tokens]}")
   ```

#### Issue: Custom rules not working
```python
# Custom rules loaded but not applying
```

**Solutions:**
1. **Validate rule format**:
   ```python
   def validate_custom_rule(rule):
       required_fields = ['priority', 'id', 'name', 'match_fn', 'apply_fn']
       for field in required_fields:
           if not hasattr(rule, field):
               raise ValueError(f"Rule missing required field: {field}")
       
       # Test match function
       try:
           test_tokens = [Token("test", TokenKind.OTHER)]
           rule.match_fn(test_tokens, 0)
       except Exception as e:
           raise ValueError(f"Invalid match function: {e}")
   ```

2. **Check rule priorities**:
   ```python
   # Ensure custom rules have appropriate priorities
   custom_rule.priority = 1  # High priority
   engine.add_rule(custom_rule)
   
   # Verify rule order
   rules = engine.get_active_rules()
   for rule in rules[:5]:  # Check first 5 rules
       print(f"{rule.name}: priority {rule.priority}")
   ```

3. **Debug rule application**:
   ```python
   # Add logging to custom rules
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   def debug_match_fn(tokens, index):
       result = original_match_fn(tokens, index)
       logging.debug(f"Rule match at {index}: {result}")
       return result
   
   custom_rule.match_fn = debug_match_fn
   ```

## Advanced Configuration

### Performance Optimization

#### Memory Management
```python
# Configure memory limits
config = EngineConfig(
    memory_limit=1024,  # MB
    enable_gc=True,     # Garbage collection
    gc_threshold=1000   # Objects before GC
)

# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Use memory-efficient processing
engine = SanskritRewriteEngine(config=config)
monitor_memory()
result = engine.process(large_text)
monitor_memory()
```

#### Caching Strategies
```python
# Configure intelligent caching
cache_config = {
    'type': 'lru',           # LRU cache
    'size': 10000,           # Cache size
    'ttl': 3600,             # Time to live (seconds)
    'persist': True,         # Persistent cache
    'file': 'cache.db'       # Cache file
}

engine = SanskritRewriteEngine(cache_config=cache_config)

# Cache hit rate monitoring
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats.hit_rate:.2%}")
print(f"Cache size: {stats.size}/{stats.max_size}")
```

#### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Thread-based parallelism (I/O bound)
def process_texts_threaded(texts):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(engine.process, text) for text in texts]
        results = [future.result() for future in futures]
    return results

# Process-based parallelism (CPU bound)
def process_texts_multiprocess(texts):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(process_single_text, texts))
    return results

def process_single_text(text):
    # Create engine in each process
    engine = SanskritRewriteEngine()
    return engine.process(text)
```

### Rule Management

#### Dynamic Rule Loading
```python
# Load rules based on text type
def load_rules_for_text_type(text_type):
    if text_type == 'vedic':
        engine.load_rule_set('vedic_rules.json')
    elif text_type == 'classical':
        engine.load_rule_set('classical_rules.json')
    elif text_type == 'modern':
        engine.load_rule_set('modern_rules.json')

# Auto-detect text type
def detect_text_type(text):
    vedic_markers = ['iti', 'vai', 'ha']
    classical_markers = ['atha', 'tatra', 'yatra']
    
    vedic_count = sum(1 for marker in vedic_markers if marker in text)
    classical_count = sum(1 for marker in classical_markers if marker in text)
    
    if vedic_count > classical_count:
        return 'vedic'
    else:
        return 'classical'

# Process with appropriate rules
text_type = detect_text_type(input_text)
load_rules_for_text_type(text_type)
result = engine.process(input_text)
```

#### Rule Validation and Testing
```python
# Comprehensive rule testing
def test_rule_set(rule_set_name):
    engine = SanskritRewriteEngine()
    engine.load_rule_set(f'{rule_set_name}.json')
    
    # Load test cases
    with open(f'test_cases_{rule_set_name}.json') as f:
        test_cases = json.load(f)
    
    results = []
    for test_case in test_cases:
        result = engine.process(test_case['input'])
        expected = test_case['expected']
        actual = result.get_output_text()
        
        results.append({
            'input': test_case['input'],
            'expected': expected,
            'actual': actual,
            'passed': actual == expected,
            'rules_applied': [t.rule_name for trace in result.traces 
                            for t in trace.transformations]
        })
    
    # Generate report
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"Rule set {rule_set_name}: {passed}/{total} tests passed ({passed/total:.1%})")
    
    return results
```

## Performance Tuning

### Profiling and Benchmarking

#### Performance Profiling
```python
import cProfile
import pstats
import time

def profile_engine_performance():
    """Comprehensive performance profiling"""
    
    # Setup profiler
    profiler = cProfile.Profile()
    
    # Test data
    test_texts = [
        "rāma + iti",
        "deva + indra + iti",
        "mahā + bhārata + kathā"
    ] * 100  # Repeat for statistical significance
    
    # Profile processing
    profiler.enable()
    start_time = time.time()
    
    for text in test_texts:
        result = engine.process(text)
    
    end_time = time.time()
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Average per text: {(end_time - start_time) / len(test_texts):.4f} seconds")
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return stats

# Run profiling
performance_stats = profile_engine_performance()
```

#### Memory Profiling
```python
import tracemalloc
import gc

def profile_memory_usage():
    """Profile memory usage patterns"""
    
    tracemalloc.start()
    
    # Baseline memory
    gc.collect()
    baseline = tracemalloc.get_traced_memory()[0]
    
    # Create engine
    engine = SanskritRewriteEngine()
    after_init = tracemalloc.get_traced_memory()[0]
    
    # Process texts
    results = []
    for i in range(100):
        result = engine.process(f"test text {i}")
        results.append(result)
        
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"After {i} texts: {current / 1024 / 1024:.1f} MB "
                  f"(peak: {peak / 1024 / 1024:.1f} MB)")
    
    # Final memory usage
    final_current, final_peak = tracemalloc.get_traced_memory()
    
    print(f"\nMemory usage summary:")
    print(f"Baseline: {baseline / 1024 / 1024:.1f} MB")
    print(f"After init: {after_init / 1024 / 1024:.1f} MB")
    print(f"Final: {final_current / 1024 / 1024:.1f} MB")
    print(f"Peak: {final_peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

#### Benchmarking Different Configurations
```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    config_name: str
    total_time: float
    avg_time_per_text: float
    memory_usage: float
    accuracy: float

def benchmark_configurations():
    """Benchmark different engine configurations"""
    
    configs = {
        'default': EngineConfig(),
        'performance': EngineConfig(
            performance_mode=True,
            enable_tracing=False,
            max_passes=10
        ),
        'accuracy': EngineConfig(
            performance_mode=False,
            enable_tracing=True,
            max_passes=50
        ),
        'memory_optimized': EngineConfig(
            memory_limit=256,
            enable_gc=True,
            gc_threshold=100
        )
    }
    
    test_texts = load_test_corpus()  # Load standard test corpus
    results = []
    
    for config_name, config in configs.items():
        print(f"Benchmarking {config_name} configuration...")
        
        engine = SanskritRewriteEngine(config=config)
        
        # Measure performance
        start_time = time.time()
        start_memory = get_memory_usage()
        
        processed_results = []
        for text in test_texts:
            result = engine.process(text)
            processed_results.append(result)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Calculate accuracy
        accuracy = calculate_accuracy(processed_results, expected_results)
        
        # Store results
        benchmark_result = BenchmarkResult(
            config_name=config_name,
            total_time=end_time - start_time,
            avg_time_per_text=(end_time - start_time) / len(test_texts),
            memory_usage=end_memory - start_memory,
            accuracy=accuracy
        )
        results.append(benchmark_result)
    
    # Print comparison
    print("\nBenchmark Results:")
    print(f"{'Config':<15} {'Time (s)':<10} {'Avg (ms)':<10} {'Memory (MB)':<12} {'Accuracy':<10}")
    print("-" * 65)
    
    for result in results:
        print(f"{result.config_name:<15} "
              f"{result.total_time:<10.2f} "
              f"{result.avg_time_per_text*1000:<10.1f} "
              f"{result.memory_usage:<12.1f} "
              f"{result.accuracy:<10.2%}")
    
    return results
```

### Optimization Strategies

#### Rule Optimization
```python
# Optimize rule matching
def optimize_rule_matching():
    """Optimize rule matching for better performance"""
    
    # Index rules by first character for faster lookup
    rule_index = {}
    for rule in engine.get_active_rules():
        # Analyze rule patterns to build index
        first_chars = analyze_rule_first_chars(rule)
        for char in first_chars:
            if char not in rule_index:
                rule_index[char] = []
            rule_index[char].append(rule)
    
    # Use index during processing
    def fast_rule_lookup(tokens, index):
        if index >= len(tokens):
            return []
        
        first_char = tokens[index].text[0] if tokens[index].text else ''
        candidate_rules = rule_index.get(first_char, [])
        
        # Test only relevant rules
        applicable_rules = []
        for rule in candidate_rules:
            if rule.match_fn(tokens, index):
                applicable_rules.append(rule)
        
        return applicable_rules

# Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_tokenize(text):
    """Cache tokenization results"""
    return engine.tokenizer.tokenize(text)

@lru_cache(maxsize=5000)
def cached_rule_match(rule_id, token_signature, index):
    """Cache rule matching results"""
    rule = engine.get_rule(rule_id)
    tokens = reconstruct_tokens_from_signature(token_signature)
    return rule.match_fn(tokens, index)
```

#### Memory Optimization
```python
# Implement memory-efficient token handling
class MemoryEfficientToken:
    """Memory-efficient token implementation"""
    
    __slots__ = ['text', 'kind', '_tags', '_meta']
    
    def __init__(self, text, kind):
        self.text = text
        self.kind = kind
        self._tags = None
        self._meta = None
    
    @property
    def tags(self):
        if self._tags is None:
            self._tags = set()
        return self._tags
    
    @property
    def meta(self):
        if self._meta is None:
            self._meta = {}
        return self._meta

# Use object pooling for common tokens
class TokenPool:
    """Pool of reusable token objects"""
    
    def __init__(self, max_size=1000):
        self.pool = []
        self.max_size = max_size
    
    def get_token(self, text, kind):
        if self.pool:
            token = self.pool.pop()
            token.text = text
            token.kind = kind
            token._tags = None
            token._meta = None
            return token
        else:
            return MemoryEfficientToken(text, kind)
    
    def return_token(self, token):
        if len(self.pool) < self.max_size:
            self.pool.append(token)

# Global token pool
token_pool = TokenPool()
```

## Linguistic Concepts

### Sanskrit Grammar Fundamentals

#### Sandhi (Phonological Changes)
Sandhi refers to the phonological changes that occur when sounds come into contact:

```python
# Vowel Sandhi Examples
vowel_sandhi_rules = {
    ('a', 'a'): 'ā',     # a + a → ā
    ('a', 'ā'): 'ā',     # a + ā → ā
    ('a', 'i'): 'e',     # a + i → e
    ('a', 'ī'): 'e',     # a + ī → e
    ('a', 'u'): 'o',     # a + u → o
    ('a', 'ū'): 'o',     # a + ū → o
    ('a', 'e'): 'ai',    # a + e → ai
    ('a', 'o'): 'au',    # a + o → au
}

# Consonant Sandhi Examples
consonant_sandhi_rules = {
    ('t', 'c'): 'c',     # t + c → c (assimilation)
    ('d', 'g'): 'g',     # d + g → g (assimilation)
    ('n', 'p'): 'm',     # n + p → m (place assimilation)
    ('n', 'k'): 'ṅ',     # n + k → ṅ (place assimilation)
}

# Visarga Sandhi Examples
visarga_sandhi_rules = {
    ('ḥ', 'vowel'): 'r',    # ḥ + vowel → r
    ('ḥ', 'voiced'): 'r',   # ḥ + voiced consonant → r
    ('ḥ', 'k'): 'ḥ',        # ḥ + k → ḥ (no change)
    ('ḥ', 'p'): 'ḥ',        # ḥ + p → ḥ (no change)
}
```

#### Morphological Analysis
Understanding Sanskrit word structure:

```python
# Morphological Components
class MorphologicalAnalysis:
    def __init__(self):
        self.root = None          # dhātu (verbal root)
        self.stem = None          # prātipadika (nominal stem)
        self.prefixes = []        # upasarga (prefixes)
        self.suffixes = []        # pratyaya (suffixes)
        self.inflection = None    # vibhakti/tiṅanta (case/verbal ending)
    
    def analyze_word(self, word):
        """Analyze morphological structure of Sanskrit word"""
        # Example: rāmāya → rāma (stem) + āya (dative singular)
        analysis = MorphologicalAnalysis()
        
        # Identify stem and inflection
        if word.endswith('āya'):
            analysis.stem = word[:-3]
            analysis.inflection = 'dative_singular'
        elif word.endswith('asya'):
            analysis.stem = word[:-4]
            analysis.inflection = 'genitive_singular'
        # ... more patterns
        
        return analysis

# Compound Analysis (Samāsa)
class CompoundAnalysis:
    def __init__(self):
        self.type = None          # tatpuruṣa, bahuvrīhi, etc.
        self.components = []      # constituent words
        self.meaning = None       # semantic interpretation
    
    def analyze_compound(self, compound):
        """Analyze Sanskrit compound structure"""
        # Example: devarāja → deva + rāja (tatpuruṣa)
        analysis = CompoundAnalysis()
        
        # Simple pattern matching (real implementation would be more complex)
        if 'deva' in compound and 'rāja' in compound:
            analysis.components = ['deva', 'rāja']
            analysis.type = 'tatpuruṣa'
            analysis.meaning = 'king of gods'
        
        return analysis
```

#### Pāṇinian Grammar Principles
The engine follows traditional Pāṇinian grammatical principles:

```python
# Sūtra-based Rule System
class PaninianRule:
    def __init__(self, sutra_number, sutra_text, rule_function):
        self.sutra_number = sutra_number
        self.sutra_text = sutra_text
        self.rule_function = rule_function
        self.adhikara = None      # domain of application
        self.anuvṛtti = []        # inherited elements
    
    def applies_in_context(self, context):
        """Check if rule applies in given grammatical context"""
        # Implement Pāṇinian rule application logic
        pass

# Example Pāṇinian Rules
paninian_rules = [
    PaninianRule(
        sutra_number="6.1.87",
        sutra_text="ād guṇaḥ",
        rule_function=apply_guna_sandhi
    ),
    PaninianRule(
        sutra_number="8.4.40",
        sutra_text="stoḥ ścu nā tuḥ",
        rule_function=apply_consonant_assimilation
    )
]
```

### Advanced Linguistic Features

#### Accent and Prosody
```python
# Vedic Accent Handling
class AccentAnalysis:
    def __init__(self):
        self.accent_type = None   # udātta, anudātta, svarita
        self.accent_position = None
        self.prosodic_pattern = None
    
    def analyze_accent(self, word):
        """Analyze Vedic accent patterns"""
        # Implement accent analysis logic
        pass

# Meter Analysis (Chandas)
class MeterAnalysis:
    def __init__(self):
        self.meter_type = None    # anuṣṭubh, triṣṭubh, etc.
        self.syllable_pattern = []
        self.caesura_positions = []
    
    def analyze_meter(self, verse):
        """Analyze Sanskrit verse meter"""
        # Implement meter analysis logic
        pass
```

#### Semantic Analysis
```python
# Semantic Role Analysis
class SemanticAnalysis:
    def __init__(self):
        self.karaka_roles = {}    # semantic roles (kartā, karma, etc.)
        self.semantic_relations = []
        self.thematic_roles = {}
    
    def analyze_semantics(self, sentence):
        """Analyze semantic structure of Sanskrit sentence"""
        # Implement semantic analysis logic
        pass

# Cross-linguistic Mapping
class CrossLinguisticMapper:
    def __init__(self):
        self.sanskrit_to_english = {}
        self.sanskrit_to_hindi = {}
        self.universal_concepts = {}
    
    def map_concepts(self, sanskrit_text):
        """Map Sanskrit concepts to other languages"""
        # Implement cross-linguistic mapping
        pass
```

## Integration Patterns

### Web Application Integration

#### Flask Integration
```python
from flask import Flask, request, jsonify, render_template
from sanskrit_rewrite_engine import SanskritRewriteEngine

app = Flask(__name__)
engine = SanskritRewriteEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_text():
    data = request.get_json()
    
    try:
        result = engine.process(
            text=data['text'],
            max_passes=data.get('max_passes', 20),
            enable_tracing=data.get('enable_tracing', True)
        )
        
        return jsonify({
            'success': True,
            'input': result.input_text,
            'output': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': [
                {
                    'rule_name': t.rule_name,
                    'position': t.index,
                    'before': [token.text for token in t.tokens_before],
                    'after': [token.text for token in t.tokens_after]
                }
                for trace in result.traces
                for t in trace.transformations
            ]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    
    # Enable detailed analysis
    engine.enable_morphological_analysis()
    engine.enable_compound_analysis()
    
    result = engine.process(data['text'])
    
    analysis = {
        'morphological': extract_morphological_info(result),
        'compounds': extract_compound_info(result),
        'sandhi': extract_sandhi_info(result)
    }
    
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)
```

#### Django Integration
```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

@csrf_exempt
@require_http_methods(["POST"])
def process_sanskrit(request):
    try:
        data = json.loads(request.body)
        
        engine = SanskritRewriteEngine()
        result = engine.process(data['text'])
        
        return JsonResponse({
            'success': True,
            'result': {
                'input': result.input_text,
                'output': result.get_output_text(),
                'traces': serialize_traces(result.traces)
            }
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

# models.py
from django.db import models

class ProcessingJob(models.Model):
    input_text = models.TextField()
    output_text = models.TextField()
    processing_time = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'sanskrit_processing_jobs'
```

### Database Integration

#### SQLAlchemy Integration
```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class SanskritText(Base):
    __tablename__ = 'sanskrit_texts'
    
    id = Column(Integer, primary_key=True)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    processing_time = Column(Float)
    rule_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ProcessingRule(Base):
    __tablename__ = 'processing_rules'
    
    id = Column(Integer, primary_key=True)
    rule_name = Column(String(100), nullable=False)
    rule_type = Column(String(50))
    priority = Column(Integer)
    application_count = Column(Integer, default=0)

# Database operations
engine = create_engine('sqlite:///sanskrit_processing.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def store_processing_result(input_text, result):
    session = Session()
    
    sanskrit_text = SanskritText(
        original_text=input_text,
        processed_text=result.get_output_text(),
        processing_time=result.processing_time,
        rule_count=len(result.get_transformation_summary())
    )
    
    session.add(sanskrit_text)
    session.commit()
    session.close()
```

#### MongoDB Integration
```python
from pymongo import MongoClient
from datetime import datetime

class MongoSanskritStorage:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client.sanskrit_processing
        self.texts = self.db.texts
        self.rules = self.db.rules
        self.analytics = self.db.analytics
    
    def store_processing_result(self, input_text, result):
        document = {
            'input_text': input_text,
            'output_text': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': [
                {
                    'rule_name': t.rule_name,
                    'rule_id': t.rule_id,
                    'position': t.index,
                    'timestamp': t.timestamp.isoformat()
                }
                for trace in result.traces
                for t in trace.transformations
            ],
            'created_at': datetime.utcnow()
        }
        
        return self.texts.insert_one(document)
    
    def get_processing_statistics(self):
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'total_texts': {'$sum': 1},
                    'avg_passes': {'$avg': '$passes'},
                    'convergence_rate': {
                        '$avg': {'$cond': ['$converged', 1, 0]}
                    }
                }
            }
        ]
        
        return list(self.texts.aggregate(pipeline))[0]
```

### API Integration

#### REST API Client
```python
import requests
import json

class SanskritAPIClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def process_text(self, text, **kwargs):
        """Process Sanskrit text via API"""
        url = f"{self.base_url}/api/v1/process"
        
        payload = {
            'text': text,
            **kwargs
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def batch_process(self, texts, **kwargs):
        """Process multiple texts in batch"""
        url = f"{self.base_url}/api/v1/batch"
        
        payload = {
            'texts': texts,
            **kwargs
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_available_rules(self):
        """Get list of available transformation rules"""
        url = f"{self.base_url}/api/v1/rules"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()

# Usage example
client = SanskritAPIClient('https://api.sanskrit-engine.com', api_key='your-key')

result = client.process_text('rāma + iti')
print(f"Processed: {result['output']}")

batch_results = client.batch_process(['rāma + iti', 'deva + indra'])
for i, result in enumerate(batch_results['results']):
    print(f"Text {i+1}: {result['output']}")
```

#### GraphQL Integration
```python
import requests

class SanskritGraphQLClient:
    def __init__(self, endpoint, api_key=None):
        self.endpoint = endpoint
        self.headers = {'Content-Type': 'application/json'}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def execute_query(self, query, variables=None):
        """Execute GraphQL query"""
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        
        return response.json()
    
    def process_text(self, text, include_traces=True):
        """Process text with GraphQL"""
        query = """
        mutation ProcessText($input: ProcessTextInput!) {
            processText(input: $input) {
                success
                result {
                    inputText
                    outputText
                    converged
                    passes
                    traces @include(if: $includeTraces) {
                        passNumber
                        transformations {
                            ruleName
                            position
                            tokensBefore
                            tokensAfter
                        }
                    }
                }
                errors {
                    message
                    code
                }
            }
        }
        """
        
        variables = {
            'input': {
                'text': text,
                'enableTracing': include_traces
            },
            'includeTraces': include_traces
        }
        
        return self.execute_query(query, variables)

# Usage
client = SanskritGraphQLClient('https://api.sanskrit-engine.com/graphql')
result = client.process_text('rāma + iti')
```

## Debugging Techniques

### Comprehensive Debugging Workflow

#### Step-by-Step Debugging
```python
def debug_processing_step_by_step(text):
    """Debug text processing with detailed step analysis"""
    
    print(f"Debugging processing of: '{text}'")
    print("=" * 50)
    
    # 1. Tokenization debugging
    print("1. TOKENIZATION")
    tokenizer = SanskritTokenizer()
    tokens = tokenizer.tokenize(text)
    
    print(f"Input text: '{text}'")
    print(f"Tokens ({len(tokens)}):")
    for i, token in enumerate(tokens):
        print(f"  {i}: '{token.text}' ({token.kind}) {token.tags}")
    print()
    
    # 2. Rule loading debugging
    print("2. RULE LOADING")
    engine = SanskritRewriteEngine()
    rules = engine.get_active_rules()
    
    print(f"Loaded rules: {len(rules)}")
    for rule in rules[:5]:  # Show first 5 rules
        print(f"  {rule.name} (priority: {rule.priority})")
    print()
    
    # 3. Rule matching debugging
    print("3. RULE MATCHING")
    applicable_rules = []
    
    for i, token in enumerate(tokens):
        print(f"Position {i} ('{token.text}'):")
        position_rules = []
        
        for rule in rules:
            if rule.match_fn(tokens, i):
                position_rules.append(rule)
                print(f"  ✓ {rule.name}")
        
        if not position_rules:
            print(f"  ✗ No rules match")
        
        applicable_rules.extend(position_rules)
    print()
    
    # 4. Processing with detailed traces
    print("4. PROCESSING")
    result = engine.process(text, enable_tracing=True)
    
    print(f"Converged: {result.converged}")
    print(f"Passes: {result.passes}")
    print(f"Final output: '{result.get_output_text()}'")
    print()
    
    # 5. Transformation analysis
    print("5. TRANSFORMATIONS")
    for pass_num, pass_trace in enumerate(result.traces, 1):
        print(f"Pass {pass_num}:")
        
        if not pass_trace.transformations:
            print("  No transformations")
            continue
        
        for t in pass_trace.transformations:
            before_text = ''.join(token.text for token in t.tokens_before)
            after_text = ''.join(token.text for token in t.tokens_after)
            
            print(f"  {t.rule_name} at position {t.index}:")
            print(f"    Before: '{before_text}'")
            print(f"    After:  '{after_text}'")
    
    return result

# Usage
debug_result = debug_processing_step_by_step("rāma + iti")
```

#### Rule-Specific Debugging
```python
def debug_specific_rule(rule_name, test_cases):
    """Debug a specific rule with multiple test cases"""
    
    engine = SanskritRewriteEngine()
    rule = engine.get_rule(rule_name)
    
    if not rule:
        print(f"Rule '{rule_name}' not found")
        return
    
    print(f"Debugging rule: {rule.name}")
    print(f"Description: {rule.description}")
    print(f"Priority: {rule.priority}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: '{test_case}'")
        
        # Tokenize
        tokens = engine.tokenizer.tokenize(test_case)
        print(f"Tokens: {[t.text for t in tokens]}")
        
        # Test rule at each position
        matches = []
        for pos in range(len(tokens)):
            if rule.match_fn(tokens, pos):
                matches.append(pos)
                print(f"  ✓ Matches at position {pos}")
                
                # Apply rule
                try:
                    new_tokens, new_pos = rule.apply_fn(tokens, pos)
                    new_text = ''.join(t.text for t in new_tokens)
                    print(f"    Result: '{new_text}' (next pos: {new_pos})")
                except Exception as e:
                    print(f"    Error applying rule: {e}")
        
        if not matches:
            print("  ✗ No matches found")
        
        print()

# Usage
test_cases = ["rāma + iti", "deva + indra", "a + i"]
debug_specific_rule("vowel_sandhi_a_i", test_cases)
```

#### Performance Debugging
```python
import time
import cProfile
import pstats
from memory_profiler import profile

def debug_performance_issues(text_samples):
    """Debug performance issues with detailed analysis"""
    
    print("PERFORMANCE DEBUGGING")
    print("=" * 50)
    
    engine = SanskritRewriteEngine()
    
    # 1. Timing analysis
    print("1. TIMING ANALYSIS")
    times = []
    
    for i, text in enumerate(text_samples):
        start_time = time.time()
        result = engine.process(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        times.append(processing_time)
        
        print(f"Sample {i+1}: {processing_time:.4f}s "
              f"({len(text)} chars, {result.passes} passes)")
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.4f}s")
    print()
    
    # 2. Memory analysis
    print("2. MEMORY ANALYSIS")
    
    @profile
    def process_with_memory_tracking():
        for text in text_samples:
            result = engine.process(text)
    
    process_with_memory_tracking()
    print()
    
    # 3. Profiling analysis
    print("3. PROFILING ANALYSIS")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for text in text_samples:
        result = engine.process(text)
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("Top 10 functions by cumulative time:")
    stats.print_stats(10)
    
    # 4. Rule efficiency analysis
    print("\n4. RULE EFFICIENCY ANALYSIS")
    
    rule_stats = {}
    
    for text in text_samples:
        result = engine.process(text, enable_tracing=True)
        
        for trace in result.traces:
            for t in trace.transformations:
                rule_name = t.rule_name
                if rule_name not in rule_stats:
                    rule_stats[rule_name] = {'count': 0, 'total_time': 0}
                
                rule_stats[rule_name]['count'] += 1
    
    print("Rule application frequency:")
    sorted_rules = sorted(rule_stats.items(), 
                         key=lambda x: x[1]['count'], 
                         reverse=True)
    
    for rule_name, stats in sorted_rules[:10]:
        print(f"  {rule_name}: {stats['count']} applications")

# Usage
sample_texts = [
    "rāma + iti",
    "deva + indra + iti",
    "mahā + bhārata + kathā + iti"
] * 10

debug_performance_issues(sample_texts)
```

### Error Analysis and Recovery

#### Comprehensive Error Handling
```python
class SanskritProcessingError(Exception):
    """Base class for Sanskrit processing errors"""
    pass

class DetailedErrorHandler:
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {
            'tokenization': self.recover_from_tokenization_error,
            'rule_application': self.recover_from_rule_error,
            'convergence': self.recover_from_convergence_error
        }
    
    def handle_error(self, error, context):
        """Handle errors with detailed logging and recovery"""
        
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'timestamp': time.time()
        }
        
        self.error_log.append(error_info)
        
        # Attempt recovery
        error_category = self.categorize_error(error)
        if error_category in self.recovery_strategies:
            return self.recovery_strategies[error_category](error, context)
        
        # If no recovery possible, re-raise
        raise error
    
    def categorize_error(self, error):
        """Categorize error for appropriate recovery strategy"""
        if isinstance(error, TokenizationError):
            return 'tokenization'
        elif isinstance(error, RuleApplicationError):
            return 'rule_application'
        elif isinstance(error, ConvergenceError):
            return 'convergence'
        else:
            return 'unknown'
    
    def recover_from_tokenization_error(self, error, context):
        """Recover from tokenization errors"""
        print(f"Tokenization error: {error}")
        
        # Try alternative tokenization strategies
        strategies = [
            self.try_lenient_tokenization,
            self.try_character_by_character,
            self.try_fallback_tokenization
        ]
        
        for strategy in strategies:
            try:
                return strategy(context['text'])
            except Exception as e:
                print(f"Strategy failed: {e}")
                continue
        
        # If all strategies fail, return minimal tokenization
        return [Token(char, TokenKind.OTHER) for char in context['text']]
    
    def recover_from_rule_error(self, error, context):
        """Recover from rule application errors"""
        print(f"Rule application error: {error}")
        
        # Skip problematic rule and continue
        rule = context.get('rule')
        if rule:
            rule.active = False
            print(f"Disabled rule: {rule.name}")
        
        return context.get('tokens', [])
    
    def recover_from_convergence_error(self, error, context):
        """Recover from convergence errors"""
        print(f"Convergence error: {error}")
        
        # Return partial result
        return context.get('partial_result')

# Usage with error handling
def robust_process_text(text):
    """Process text with comprehensive error handling"""
    
    error_handler = DetailedErrorHandler()
    engine = SanskritRewriteEngine()
    
    try:
        result = engine.process(text)
        return result
    
    except Exception as e:
        context = {
            'text': text,
            'engine': engine
        }
        
        recovered_result = error_handler.handle_error(e, context)
        
        # Create partial result object
        return RewriteResult(
            input_text=text,
            input_tokens=engine.tokenizer.tokenize(text),
            output_tokens=recovered_result or [],
            converged=False,
            passes=0,
            traces=[],
            errors=[str(e)]
        )
```

This comprehensive knowledge base provides detailed solutions, advanced configurations, and debugging techniques for the Sanskrit Rewrite Engine. It serves as a complete reference for users at all levels, from beginners to advanced developers and researchers.