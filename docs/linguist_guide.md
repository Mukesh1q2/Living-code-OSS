# Sanskrit Rewrite Engine - Linguist's Guide

## Introduction

Welcome to the Sanskrit Rewrite Engine! This guide is designed specifically for Sanskrit scholars, linguists, and researchers who want to use computational tools for Sanskrit text analysis and transformation.

## What is the Sanskrit Rewrite Engine?

The Sanskrit Rewrite Engine is a computational system that applies Sanskrit grammatical rules automatically. Unlike simple find-and-replace tools, it understands Sanskrit phonology, morphology, and syntax to perform linguistically accurate transformations.

### Key Capabilities

- **Sandhi Resolution**: Automatically apply vowel and consonant sandhi rules
- **Morphological Analysis**: Break down words into roots, suffixes, and inflections
- **Compound Analysis**: Analyze and form Sanskrit compounds (samāsa)
- **Rule Tracing**: See exactly which grammatical rules were applied
- **Pāṇinian Framework**: Based on traditional Sanskrit grammatical principles

## Getting Started

### Installation for Linguists

```bash
# Install the engine
pip install sanskrit-rewrite-engine

# Verify installation
python -c "import sanskrit_rewrite_engine; print('Ready for Sanskrit processing!')"
```

### Basic Text Processing

#### Web Interface (Recommended for Beginners)
1. Start the web server: `sanskrit-web --port 8000`
2. Open http://localhost:8000 in your browser
3. Enter Sanskrit text and view transformations
4. Examine detailed rule traces

#### Python Interface (For Advanced Users)
```python
from sanskrit_rewrite_engine import SanskritRewriteEngine

# Initialize engine with Classical Sanskrit rules
engine = SanskritRewriteEngine()

# Process text with detailed analysis
result = engine.process("rāma + iti", enable_tracing=True)

print(f"Input: {result.input_text}")
print(f"Output: {result.get_output_text()}")
print(f"Rules applied: {result.get_transformation_summary()}")
```

### Example Transformations

#### Vowel Sandhi (स्वरसन्धि)
```python
examples = [
    ("rāma + iti", "rāmeti", "a + i → e (Pāṇini 6.1.87)"),
    ("deva + indra", "devendra", "a + i → e"),
    ("mahā + ātman", "mahātman", "ā + ā → ā (Pāṇini 6.1.101)"),
    ("te + eva", "taiva", "e + e → ai (Pāṇini 6.1.88)"),
    ("go + indra", "gavindra", "o + i → avi (Pāṇini 6.1.78)")
]

for input_text, expected, rule in examples:
    result = engine.process(input_text)
    print(f"{input_text:15} → {result.get_output_text():15} ({rule})")
```

#### Consonant Sandhi (व्यञ्जनसन्धि)
```python
consonant_examples = [
    ("tat + ca", "tac ca", "t + c → c (Pāṇini 8.4.40)"),
    ("sat + jana", "saj jana", "t + j → j (Pāṇini 8.4.40)"),
    ("vāk + īśa", "vāg īśa", "k + vowel → g (Pāṇini 8.2.30)"),
    ("ṣaṭ + aṅga", "ṣaḍ aṅga", "ṭ + vowel → ḍ (Pāṇini 8.2.39)")
]

for input_text, expected, rule in consonant_examples:
    result = engine.process(input_text)
    print(f"{input_text:15} → {result.get_output_text():15} ({rule})")
```

#### Compound Formation (समास)
```python
compound_examples = [
    ("deva + rāja", "devarāja", "Tatpuruṣa compound"),
    ("rāja + putra", "rājaputra", "Tatpuruṣa compound"),
    ("mahā + rāja", "mahārāja", "Karmadhāraya compound"),
    ("pīta + ambara", "pītāmbara", "Bahuvrīhi compound")
]

for input_text, expected, compound_type in compound_examples:
    result = engine.process(input_text)
    print(f"{input_text:15} → {result.get_output_text():15} ({compound_type})")
```

## Understanding Rule Application

### Rule Priorities

The engine applies rules in a specific order based on traditional Sanskrit grammar:

1. **Meta-rules** (paribhāṣā): Control when other rules apply
2. **Phonological rules**: Sandhi transformations
3. **Morphological rules**: Inflection and derivation
4. **Cleanup rules**: Remove processing markers

### Tracing Transformations

Every transformation is recorded with:
- **Rule name**: Traditional Sanskrit grammatical term
- **Sūtra reference**: Pāṇini sūtra number when applicable
- **Before/after states**: Text before and after transformation
- **Position**: Where in the text the rule applied

Example trace:
```
Pass 1:
  Position 4: vowel_sandhi_a_i (6.1.87)
    Before: [rāma] [+] [iti]
    After:  [rāme] [ti]
  
Pass 2:
  Position 4: compound_formation
    Before: [rāme] [ti]
    After:  [rāmeti]
```

## Working with Different Text Types

### Classical Sanskrit

The engine handles classical Sanskrit texts with:
- Standard Devanagari input
- IAST transliteration support
- Classical sandhi rules
- Morphological analysis

```
Input (Devanagari): राम इति
Input (IAST):       rāma iti
Output:             rāmeti
```

### Vedic Sanskrit

For Vedic texts, enable Vedic mode:
- Vedic accent preservation
- Archaic forms recognition
- Vedic-specific sandhi rules

### Manuscript Transcription

When working with manuscripts:
- Use uncertainty markers: `rāma[?]`
- Mark damaged text: `rāma<...>iti`
- Indicate editorial additions: `rāma{+iti}`

## Advanced Features for Linguists

### Custom Rule Sets

Create specialized rule sets for:
- Specific texts or authors
- Regional variations
- Historical periods
- Grammatical schools

### Morphological Analysis

Deep analysis of word structure:
```
Input:  rāmāya
Analysis:
  Root:     rāma (proper noun)
  Suffix:   -āya (dative singular)
  Meaning:  "to/for Rāma"
```

### Compound Analysis

Detailed compound breakdown:
```
Input:  rājarṣi
Analysis:
  Type:     tatpuruṣa compound
  Members:  rāja + ṛṣi
  Meaning:  "royal sage"
  Pattern:  noun + noun → compound noun
```

### Derivational Analysis

Track word formation:
```
Input:  kārayati
Analysis:
  Root:     √kṛ (to do)
  Suffix:   -aya (causative)
  Inflection: -ti (3rd person singular)
  Meaning:  "causes to do"
```

## Research Applications

### Corpus Analysis

Use the engine for large-scale text analysis:
- Batch process multiple texts
- Extract statistical patterns
- Identify linguistic variations
- Track historical changes

### Comparative Studies

Compare different manuscript traditions:
- Identify variant readings
- Analyze scribal practices
- Study regional differences
- Track textual transmission

### Pedagogical Applications

Teaching Sanskrit grammar:
- Demonstrate rule application
- Create interactive exercises
- Visualize transformations
- Generate practice examples

## Working with Specific Grammatical Phenomena

### Sandhi Rules

#### External Sandhi (pada-sandhi)
Between separate words:
```
rāmaḥ + iti → rāma iti (visarga before vowel)
tat + eva → tad eva (final t before vowel)
```

#### Internal Sandhi (aṅga-sandhi)
Within compound words:
```
su + ukta → sukta (vowel deletion)
nis + panna → niṣpanna (consonant assimilation)
```

### Morphological Processes

#### Nominal Inflection
```
rāma (stem) + -āya (dat.sg.) → rāmāya
```

#### Verbal Conjugation
```
√bhū (root) + -ti (3sg.pres.) → bhavati
```

#### Derivational Morphology
```
√kṛ (root) + -aka (agent suffix) → kāraka
```

### Compound Formation

#### Tatpuruṣa Compounds
```
rāja + putra → rājaputra (king's son)
```

#### Bahuvrīhi Compounds
```
mahā + ātman → mahātman (great-souled)
```

#### Dvandva Compounds
```
rāma + lakṣmaṇa → rāmalakṣmaṇau (Rāma and Lakṣmaṇa)
```

## Troubleshooting Common Issues

### Input Format Problems

**Issue**: Text not processing correctly
**Solution**: Check input encoding (UTF-8 required)

**Issue**: Devanagari not recognized
**Solution**: Ensure proper Unicode encoding

### Rule Application Issues

**Issue**: Expected transformation not occurring
**Solution**: Check if input matches rule conditions exactly

**Issue**: Unexpected transformations
**Solution**: Review rule priorities and contexts

### Performance Issues

**Issue**: Slow processing of large texts
**Solution**: Process in smaller chunks or disable detailed tracing

## Best Practices for Linguists

### Text Preparation

1. **Clean Input**: Remove extraneous formatting
2. **Consistent Encoding**: Use UTF-8 throughout
3. **Mark Uncertainties**: Use standard editorial conventions
4. **Segment Appropriately**: Break long texts into manageable units

### Rule Validation

1. **Cross-Reference**: Check against traditional sources
2. **Test Cases**: Create comprehensive test examples
3. **Edge Cases**: Consider unusual or archaic forms
4. **Documentation**: Record rule sources and justifications

### Research Methodology

1. **Systematic Approach**: Process texts consistently
2. **Version Control**: Track changes to rule sets
3. **Reproducibility**: Document processing parameters
4. **Validation**: Cross-check results with manual analysis

## Integration with Other Tools

### Text Editors

- **Emacs**: Sanskrit input methods and processing
- **Vim**: Custom syntax highlighting and commands
- **VS Code**: Extensions for Sanskrit development

### Databases

- **Corpus Databases**: Bulk processing and storage
- **Lexical Databases**: Integration with dictionaries
- **Manuscript Databases**: Variant analysis

### Analysis Tools

- **Statistical Software**: R, Python for pattern analysis
- **Visualization Tools**: Network graphs, tree diagrams
- **Concordance Tools**: KWIC analysis and indexing

## Community and Support

### Getting Help

- **Documentation**: Comprehensive guides and references
- **Forums**: Community discussions and Q&A
- **Tutorials**: Step-by-step learning materials
- **Examples**: Real-world use cases and solutions

### Contributing

Linguists can contribute by:
- **Rule Validation**: Checking grammatical accuracy
- **Test Cases**: Providing challenging examples
- **Documentation**: Improving explanations and examples
- **Feedback**: Reporting issues and suggesting improvements

### Collaboration

- **Research Projects**: Joint computational linguistics research
- **Corpus Development**: Building comprehensive text collections
- **Tool Integration**: Connecting with other Sanskrit tools
- **Standards Development**: Establishing best practices

## Glossary

**Sandhi**: Phonological changes at morpheme boundaries
**Samāsa**: Compound word formation
**Prakriyā**: Derivational process or analysis
**Sūtra**: Grammatical rule (especially from Pāṇini)
**Paribhāṣā**: Meta-rule governing other rules
**Adhikāra**: Domain or scope of rule application
**Anuvṛtti**: Continuation or inheritance of rule elements

## Further Reading

- Pāṇini's Aṣṭādhyāyī (primary source)
- Whitney's Sanskrit Grammar (comprehensive reference)
- Cardona's Pāṇini studies (modern analysis)
- Computational linguistics papers on Sanskrit
- Digital humanities resources for Sanskrit
## Advanc
ed Linguistic Features

### Morphological Analysis

The engine can analyze word structure and morphological components:

```python
# Analyze morphological structure
morphological_examples = [
    "rāmasya",      # rāma + GEN.SG
    "devānām",      # deva + GEN.PL  
    "gacchati",     # √gam + PRES.3SG
    "jagāma",       # √gam + PERF.3SG
    "gamiṣyati"     # √gam + FUT.3SG
]

for word in morphological_examples:
    result = engine.analyze_morphology(word)
    print(f"{word:12} → {result.root} + {result.suffixes}")
```

### Rule Tracing for Linguistic Research

Understanding exactly which rules apply is crucial for linguistic research:

```python
# Detailed rule analysis
result = engine.process("rāma + iti", enable_tracing=True)

print("Transformation Analysis:")
print("=" * 50)

for pass_num, pass_trace in enumerate(result.traces, 1):
    print(f"\nPass {pass_num}:")
    
    for transform in pass_trace.transformations:
        print(f"  Rule: {transform.rule_name}")
        print(f"  Sūtra: {transform.sutra_reference}")
        print(f"  Position: {transform.index}")
        
        # Show token-level changes
        before_tokens = [t.text for t in transform.tokens_before]
        after_tokens = [t.text for t in transform.tokens_after]
        
        print(f"  Before: {' '.join(before_tokens)}")
        print(f"  After:  {' '.join(after_tokens)}")
        print(f"  Change: {transform.description}")
        print()
```

### Working with Manuscripts and Variants

The engine can handle manuscript variations and dialectal differences:

```python
from sanskrit_rewrite_engine.config import EngineConfig

# Configure for specific manuscript tradition
config = EngineConfig(
    manuscript_tradition="kashmir_shaivism",
    allow_variants=True,
    strict_sandhi=False  # Allow looser sandhi rules
)

engine = SanskritRewriteEngine(config=config)

# Process text with manuscript-specific rules
manuscript_text = "śiva + īśa"  # Might have variant sandhi
result = engine.process(manuscript_text)
```

### Custom Rule Development

Linguists can add their own rules based on specific texts or traditions:

```python
from sanskrit_rewrite_engine import Rule, Token, TokenKind

def create_vedic_sandhi_rule():
    """Example: Vedic sandhi rule for specific contexts"""
    
    def match_vedic_pattern(tokens, index):
        """Match Vedic-specific sandhi pattern"""
        if index + 1 >= len(tokens):
            return False
        
        # Example: Vedic ā + a → ā (different from Classical)
        return (tokens[index].text == "ā" and 
                tokens[index + 1].text == "a" and
                tokens[index].has_tag("vedic_context"))
    
    def apply_vedic_sandhi(tokens, index):
        """Apply Vedic sandhi transformation"""
        # ā + a → ā (Vedic rule)
        new_token = Token("ā", TokenKind.VOWEL, {"vedic_sandhi"}, {})
        return tokens[:index] + [new_token] + tokens[index + 2:], index + 1
    
    return Rule(
        priority=0,  # High priority for Vedic rules
        id=9001,
        name="vedic_aa_sandhi",
        description="Vedic ā + a → ā sandhi",
        match_fn=match_vedic_pattern,
        apply_fn=apply_vedic_sandhi,
        sutra_ref="Vedic Grammar §123",
        meta_data={"period": "vedic", "source": "Whitney_1889"}
    )

# Add custom rule to engine
vedic_rule = create_vedic_sandhi_rule()
engine.add_rule(vedic_rule)
```

## Corpus Processing

### Batch Processing Large Texts

For processing entire texts or corpora:

```python
import os
from pathlib import Path

def process_corpus(corpus_directory):
    """Process all Sanskrit files in a directory"""
    engine = SanskritRewriteEngine()
    results = {}
    
    for file_path in Path(corpus_directory).glob("*.txt"):
        print(f"Processing {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into sentences or verses
        sentences = text.split('\n')
        processed_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                result = engine.process(sentence.strip())
                processed_sentences.append(result.get_output_text())
        
        results[file_path.name] = processed_sentences
    
    return results

# Process entire corpus
corpus_results = process_corpus("sanskrit_texts/")
```

### Statistical Analysis

Analyze transformation patterns across texts:

```python
def analyze_transformation_patterns(results):
    """Analyze which rules are most commonly applied"""
    rule_counts = {}
    
    for filename, sentences in results.items():
        for sentence in sentences:
            result = engine.process(sentence, enable_tracing=True)
            
            for pass_trace in result.traces:
                for transform in pass_trace.transformations:
                    rule_name = transform.rule_name
                    rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
    
    # Sort by frequency
    sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Most Common Transformations:")
    print("=" * 40)
    for rule_name, count in sorted_rules[:10]:
        print(f"{rule_name:25} {count:5} times")

# Analyze patterns
analyze_transformation_patterns(corpus_results)
```

## Research Applications

### Comparative Grammar Studies

Compare different grammatical traditions:

```python
# Classical Sanskrit processing
classical_engine = SanskritRewriteEngine(tradition="classical")
classical_result = classical_engine.process("deva + ṛṣi")

# Vedic Sanskrit processing  
vedic_engine = SanskritRewriteEngine(tradition="vedic")
vedic_result = vedic_engine.process("deva + ṛṣi")

print(f"Classical: {classical_result.get_output_text()}")
print(f"Vedic:     {vedic_result.get_output_text()}")

# Compare rule applications
print("\nRule Differences:")
classical_rules = set(classical_result.get_transformation_summary().keys())
vedic_rules = set(vedic_result.get_transformation_summary().keys())

unique_classical = classical_rules - vedic_rules
unique_vedic = vedic_rules - classical_rules

print(f"Classical-only rules: {unique_classical}")
print(f"Vedic-only rules: {unique_vedic}")
```

### Manuscript Collation

Compare different manuscript readings:

```python
def compare_manuscript_readings(base_text, variants):
    """Compare how different manuscript readings are processed"""
    engine = SanskritRewriteEngine()
    
    results = {}
    results['base'] = engine.process(base_text)
    
    for variant_name, variant_text in variants.items():
        results[variant_name] = engine.process(variant_text)
    
    # Compare outputs
    print(f"Base text: {base_text}")
    print(f"Base result: {results['base'].get_output_text()}")
    print()
    
    for variant_name, result in results.items():
        if variant_name != 'base':
            print(f"{variant_name}: {result.get_output_text()}")
            
            # Show rule differences
            base_rules = set(results['base'].get_transformation_summary().keys())
            variant_rules = set(result.get_transformation_summary().keys())
            
            if base_rules != variant_rules:
                print(f"  Rule differences: {variant_rules - base_rules}")

# Example manuscript comparison
variants = {
    "kashmir_ms": "śiva + īśa",
    "bengal_ms": "śiva + īśa", 
    "south_ms": "śiva + īśa"
}

compare_manuscript_readings("śiva + īśa", variants)
```

### Pedagogical Applications

Create learning materials and exercises:

```python
def generate_sandhi_exercises(difficulty_level="beginner"):
    """Generate sandhi exercises for students"""
    
    if difficulty_level == "beginner":
        patterns = [
            ("a", "i", "e"),
            ("a", "u", "o"), 
            ("ā", "ā", "ā")
        ]
    elif difficulty_level == "intermediate":
        patterns = [
            ("e", "a", "a"),
            ("o", "a", "ava"),
            ("ai", "a", "āya")
        ]
    else:  # advanced
        patterns = [
            ("aḥ", "ka", "aḥka"),
            ("t", "ca", "cca"),
            ("n", "ma", "mma")
        ]
    
    exercises = []
    for vowel1, vowel2, result in patterns:
        # Create exercise with blanks
        exercise = f"rāma{vowel1} + {vowel2}ti = rāma___ti"
        answer = f"rāma{result}ti"
        exercises.append((exercise, answer))
    
    return exercises

# Generate exercises
beginner_exercises = generate_sandhi_exercises("beginner")
for exercise, answer in beginner_exercises:
    print(f"Exercise: {exercise}")
    print(f"Answer:   {answer}")
    print()
```

## Integration with Traditional Tools

### Pāṇini Sūtra References

The engine can reference traditional grammatical sources:

```python
# Get sūtra references for transformations
result = engine.process("rāma + iti", enable_tracing=True)

for pass_trace in result.traces:
    for transform in pass_trace.transformations:
        if transform.sutra_reference:
            print(f"Rule: {transform.rule_name}")
            print(f"Sūtra: {transform.sutra_reference}")
            print(f"Description: {transform.description}")
            print()
```

### Dictionary Integration

Connect with Sanskrit dictionaries for enhanced analysis:

```python
def enhanced_analysis(text):
    """Combine transformation with dictionary lookup"""
    result = engine.process(text)
    
    # Process each resulting word
    words = result.get_output_text().split()
    
    for word in words:
        # Look up in dictionary (pseudo-code)
        dictionary_entry = lookup_in_dictionary(word)
        
        print(f"Word: {word}")
        print(f"Meaning: {dictionary_entry.meaning}")
        print(f"Root: {dictionary_entry.root}")
        print(f"Grammar: {dictionary_entry.grammatical_info}")
        print()

# Enhanced analysis
enhanced_analysis("rāma + iti")
```

## Best Practices for Linguists

### 1. Always Enable Tracing for Research
```python
# For research, always use detailed tracing
result = engine.process(text, enable_tracing=True)
```

### 2. Validate Results Against Traditional Sources
```python
def validate_against_traditional_grammar(input_text, expected_output):
    """Validate engine output against known grammatical rules"""
    result = engine.process(input_text)
    
    if result.get_output_text() != expected_output:
        print(f"Discrepancy found:")
        print(f"Input: {input_text}")
        print(f"Engine: {result.get_output_text()}")
        print(f"Expected: {expected_output}")
        
        # Examine rule applications
        for pass_trace in result.traces:
            for transform in pass_trace.transformations:
                print(f"Applied: {transform.rule_name}")
```

### 3. Document Custom Rules Thoroughly
```python
# Always include comprehensive metadata for custom rules
custom_rule = Rule(
    priority=1,
    id=custom_id,
    name="descriptive_name",
    description="Detailed description of the transformation",
    match_fn=match_function,
    apply_fn=apply_function,
    sutra_ref="Traditional source reference",
    meta_data={
        "source": "Scholarly source",
        "date_added": "2024-01-15",
        "author": "Researcher name",
        "notes": "Additional context and limitations"
    }
)
```

### 4. Test Edge Cases
```python
# Test boundary conditions and unusual cases
edge_cases = [
    "a + a + a",           # Multiple identical vowels
    "ṛ + ṛ",               # Vocalic r combinations
    "aḥ + aḥ",             # Visarga combinations
    "m + m",               # Consonant clusters
    "śrī + īśa + ānanda"   # Complex multi-word combinations
]

for case in edge_cases:
    result = engine.process(case)
    print(f"{case:20} → {result.get_output_text()}")
```

## Troubleshooting for Linguists

### Common Issues and Solutions

#### Issue: Unexpected Sandhi Results
```python
# Debug unexpected transformations
result = engine.process("problematic_text", enable_tracing=True)

print("Debugging transformation:")
for i, pass_trace in enumerate(result.traces):
    print(f"Pass {i+1}:")
    for transform in pass_trace.transformations:
        print(f"  {transform.rule_name}: {transform.description}")
```

#### Issue: Missing Traditional Rules
```python
# Add missing traditional rules
def add_missing_rule():
    # Define the missing rule based on traditional grammar
    missing_rule = Rule(
        priority=appropriate_priority,
        id=unique_id,
        name="traditional_rule_name",
        description="Based on Pāṇini X.Y.Z",
        match_fn=traditional_match_function,
        apply_fn=traditional_apply_function,
        sutra_ref="Pāṇini X.Y.Z"
    )
    
    engine.add_rule(missing_rule)
    return missing_rule

# Test the new rule
new_rule = add_missing_rule()
test_result = engine.process("test_case_for_new_rule")
```

#### Issue: Performance with Large Texts
```python
# Optimize for large corpus processing
from sanskrit_rewrite_engine.config import EngineConfig

# Performance-optimized configuration
config = EngineConfig(
    performance_mode=True,
    enable_tracing=False,  # Disable for large texts
    max_passes=15,         # Reduce if convergence is quick
    enable_caching=True    # Cache common transformations
)

optimized_engine = SanskritRewriteEngine(config=config)
```

## Contributing to Linguistic Accuracy

### Reporting Issues
When you find linguistic inaccuracies:

1. **Document the issue** with traditional sources
2. **Provide test cases** with expected outputs  
3. **Reference sūtras** or grammatical authorities
4. **Suggest corrections** with implementation details

### Adding New Rules
To contribute new grammatical rules:

1. **Research thoroughly** in traditional sources
2. **Implement carefully** with proper testing
3. **Document extensively** with references
4. **Test edge cases** and interactions with existing rules

---

This guide provides a comprehensive foundation for using the Sanskrit Rewrite Engine in linguistic research and scholarship. The engine's design respects traditional grammatical principles while providing modern computational capabilities for Sanskrit text analysis.

*For technical implementation details, see the [Developer Guide](developer_guide.md). For complete API documentation, see the [API Reference](api_reference.md).*

*Last updated: January 15, 2024*