# Sanskrit Rewrite Engine - Complete User Guide

## Overview

This comprehensive user guide covers all aspects of using the Sanskrit Rewrite Engine, from basic text processing to advanced research applications. It's designed to serve users with different backgrounds and needs.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [User Personas and Workflows](#user-personas-and-workflows)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Resources and Support](#resources-and-support)

## Getting Started

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large texts)
- **Storage**: 500MB free space for installation and cache

### Installation

#### Quick Installation
```bash
# Install from PyPI (recommended)
pip install sanskrit-rewrite-engine

# Verify installation
python -c "import sanskrit_rewrite_engine; print('Installation successful!')"
```

#### Development Installation
```bash
# Clone repository
git clone https://github.com/your-org/sanskrit-rewrite-engine.git
cd sanskrit-rewrite-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

#### Docker Installation
```bash
# Pull Docker image
docker pull sanskrit-rewrite-engine:latest

# Run container
docker run -it sanskrit-rewrite-engine:latest python
```

### First Steps

#### Basic Text Processing
```python
from sanskrit_rewrite_engine import SanskritRewriteEngine

# Create engine instance
engine = SanskritRewriteEngine()

# Process simple text
result = engine.process("rāma + iti")
print(f"Input: {result.input_text}")
print(f"Output: {result.get_output_text()}")
print(f"Transformations: {len(result.get_transformation_summary())}")
```

#### Web Interface (if available)
1. Open your web browser
2. Navigate to `http://localhost:8000` (or your deployment URL)
3. Enter Sanskrit text in the input field
4. Click "Process" to see transformations
5. Review results and transformation traces

## Basic Usage

### Text Input Formats

#### Devanagari Script
```python
# Native Devanagari input
devanagari_text = "राम इति"
result = engine.process(devanagari_text)
print(result.get_output_text())  # "रामेति"
```

#### IAST Transliteration
```python
# IAST (International Alphabet of Sanskrit Transliteration)
iast_text = "rāma iti"
result = engine.process(iast_text)
print(result.get_output_text())  # "rāmeti"
```

#### Mixed Input
```python
# Combination of scripts
mixed_text = "rāma + इति"
result = engine.process(mixed_text)
```

#### Marked Text
```python
# Text with morphological markers
marked_text = "rāma : GEN + iti"  # Genitive case marker
result = engine.process(marked_text)
print(result.get_output_text())  # "rāmasya iti"
```

### Common Transformations

#### Sandhi (Phonological Changes)
```python
# Vowel sandhi
examples = [
    "rāma + iti",      # → rāmeti (a + i → e)
    "deva + indra",    # → devendra (a + i → e)
    "mahā + ātman",    # → mahātman (ā + ā → ā)
    "su + ukta"        # → sukta (u + u → u)
]

for text in examples:
    result = engine.process(text)
    print(f"{text} → {result.get_output_text()}")
```

#### Compound Formation
```python
# Compound words (samāsa)
compounds = [
    "deva + rāja",     # → devarāja (god-king)
    "mahā + bhārata",  # → mahābhārata (great Bharata)
    "su + putra"       # → suputra (good son)
]

for compound in compounds:
    result = engine.process(compound)
    print(f"{compound} → {result.get_output_text()}")
```

#### Morphological Inflection
```python
# Case inflections
inflections = [
    "rāma : NOM",      # → rāmaḥ (nominative)
    "rāma : GEN",      # → rāmasya (genitive)
    "rāma : DAT",      # → rāmāya (dative)
    "rāma : ACC"       # → rāmam (accusative)
]

for inflection in inflections:
    result = engine.process(inflection)
    print(f"{inflection} → {result.get_output_text()}")
```

### Understanding Results

#### Result Structure
```python
result = engine.process("rāma + iti")

# Basic information
print(f"Input: {result.input_text}")
print(f"Output: {result.get_output_text()}")
print(f"Converged: {result.converged}")
print(f"Passes: {result.passes}")

# Transformation summary
summary = result.get_transformation_summary()
for rule_name, count in summary.items():
    print(f"{rule_name}: {count} applications")
```

#### Detailed Traces
```python
# Examine transformation traces
for pass_num, pass_trace in enumerate(result.traces, 1):
    print(f"Pass {pass_num}:")
    
    for transformation in pass_trace.transformations:
        print(f"  Rule: {transformation.rule_name}")
        print(f"  Position: {transformation.index}")
        
        before = ''.join(t.text for t in transformation.tokens_before)
        after = ''.join(t.text for t in transformation.tokens_after)
        
        print(f"  Before: '{before}'")
        print(f"  After: '{after}'")
        print()
```

## Advanced Features

### Custom Rule Sets

#### Loading Predefined Rule Sets
```python
# Load specific rule sets
engine.load_rule_set("classical_sandhi.json")
engine.load_rule_set("vedic_rules.json")
engine.load_rule_set("morphological_rules.json")

# Check loaded rules
rules = engine.get_active_rules()
print(f"Loaded {len(rules)} rules")
```

#### Creating Custom Rules
```python
from sanskrit_rewrite_engine import Rule, TokenKind

# Define custom transformation
def match_custom_pattern(tokens, index):
    """Match specific pattern"""
    if index >= len(tokens) - 1:
        return False
    return (tokens[index].text == "custom" and 
            tokens[index + 1].text == "pattern")

def apply_custom_transformation(tokens, index):
    """Apply custom transformation"""
    new_token = Token("transformed", TokenKind.OTHER, {"custom"}, {})
    return tokens[:index] + [new_token] + tokens[index + 2:], index + 1

# Create rule
custom_rule = Rule(
    priority=1,
    id=9999,
    name="custom_transformation",
    description="Transform custom pattern",
    match_fn=match_custom_pattern,
    apply_fn=apply_custom_transformation
)

# Add to engine
engine.add_rule(custom_rule)
```

#### Rule Configuration Files
```json
{
    "rule_set_name": "custom_rules",
    "version": "1.0",
    "rules": [
        {
            "id": 1001,
            "name": "custom_sandhi",
            "priority": 1,
            "description": "Custom sandhi rule",
            "pattern": {
                "match": ["x", "+", "y"],
                "replace": ["z"]
            },
            "conditions": {
                "context": "word_boundary",
                "script": "any"
            }
        }
    ]
}
```

### Performance Optimization

#### Configuration Options
```python
from sanskrit_rewrite_engine import EngineConfig

# Performance-optimized configuration
config = EngineConfig(
    performance_mode=True,
    enable_tracing=False,      # Disable for speed
    max_passes=10,             # Limit iterations
    memory_limit=512,          # MB limit
    enable_cache=True,         # Enable caching
    cache_size=5000           # Cache entries
)

engine = SanskritRewriteEngine(config=config)
```

#### Batch Processing
```python
# Process multiple texts efficiently
texts = ["rāma + iti", "deva + indra", "mahā + ātman"]

# Sequential processing
results = []
for text in texts:
    result = engine.process(text)
    results.append(result)

# Parallel processing (for large batches)
from multiprocessing import Pool

def process_text(text):
    engine = SanskritRewriteEngine()
    return engine.process(text)

with Pool() as pool:
    results = pool.map(process_text, texts)
```

#### Memory Management
```python
# For large texts, use chunking
def process_large_text(text, chunk_size=1000):
    """Process large text in chunks"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = []
    
    for chunk in chunks:
        result = engine.process(chunk)
        results.append(result.get_output_text())
        
        # Clear memory
        del result
        import gc
        gc.collect()
    
    return ' '.join(results)

# Usage
large_text = "very long sanskrit text..." * 1000
processed = process_large_text(large_text)
```

### Analysis and Debugging

#### Detailed Analysis
```python
# Enable comprehensive analysis
engine.enable_morphological_analysis()
engine.enable_compound_analysis()
engine.enable_semantic_analysis()

result = engine.process("devarāja")

# Get analysis results
morphology = result.get_morphological_analysis()
compounds = result.get_compound_analysis()
semantics = result.get_semantic_analysis()

print(f"Morphology: {morphology}")
print(f"Compounds: {compounds}")
print(f"Semantics: {semantics}")
```

#### Rule Debugging
```python
# Debug specific rules
engine.set_debug_mode(True)
engine.enable_rule_tracing("vowel_sandhi")

result = engine.process("rāma + iti", verbose=True)

# Examine rule applications
for trace in result.traces:
    for transformation in trace.transformations:
        if "vowel_sandhi" in transformation.rule_name:
            print(f"Sandhi rule applied: {transformation.rule_name}")
            print(f"Context: {transformation.tokens_before}")
            print(f"Result: {transformation.tokens_after}")
```

#### Performance Profiling
```python
import time

# Profile processing time
start_time = time.time()
result = engine.process(text)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.4f} seconds")
print(f"Tokens processed: {len(result.input_tokens)}")
print(f"Rules applied: {len(result.get_transformation_summary())}")

# Memory profiling
import tracemalloc

tracemalloc.start()
result = engine.process(text)
current, peak = tracemalloc.get_traced_memory()

print(f"Memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

## User Personas and Workflows

### Sanskrit Scholars and Linguists

#### Workflow: Manuscript Analysis
```python
# Analyze manuscript variants
manuscripts = [
    "manuscript_a.txt",
    "manuscript_b.txt",
    "manuscript_c.txt"
]

# Process each manuscript
results = {}
for manuscript in manuscripts:
    with open(manuscript, 'r', encoding='utf-8') as f:
        text = f.read()
    
    result = engine.process(text)
    results[manuscript] = result

# Compare results
for ms_name, result in results.items():
    print(f"{ms_name}: {result.get_output_text()}")
    
    # Analyze transformations
    transformations = result.get_transformation_summary()
    print(f"Transformations: {transformations}")
```

#### Workflow: Grammatical Analysis
```python
# Detailed grammatical analysis
def analyze_grammar(text):
    """Comprehensive grammatical analysis"""
    
    # Enable all analysis features
    engine.enable_morphological_analysis()
    engine.enable_syntactic_analysis()
    engine.enable_semantic_analysis()
    
    result = engine.process(text, enable_tracing=True)
    
    analysis = {
        'original': text,
        'processed': result.get_output_text(),
        'morphology': extract_morphological_features(result),
        'syntax': extract_syntactic_features(result),
        'semantics': extract_semantic_features(result),
        'rules_applied': result.get_transformation_summary()
    }
    
    return analysis

# Analyze sample text
sample = "rāmasya putraḥ sītāyāḥ priyaḥ"
analysis = analyze_grammar(sample)

print(f"Original: {analysis['original']}")
print(f"Processed: {analysis['processed']}")
print(f"Morphology: {analysis['morphology']}")
```

### Computational Linguists and Researchers

#### Workflow: Corpus Processing
```python
import os
import json
from pathlib import Path

def process_corpus(corpus_directory, output_file):
    """Process entire Sanskrit corpus"""
    
    results = []
    
    # Process all text files in corpus
    for file_path in Path(corpus_directory).glob("*.txt"):
        print(f"Processing {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process text
        result = engine.process(text)
        
        # Store results
        file_result = {
            'filename': file_path.name,
            'original_text': text,
            'processed_text': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': result.get_transformation_summary(),
            'processing_time': getattr(result, 'processing_time', 0)
        }
        
        results.append(file_result)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# Process corpus
corpus_results = process_corpus("sanskrit_corpus/", "corpus_results.json")

# Generate statistics
total_texts = len(corpus_results)
converged_texts = sum(1 for r in corpus_results if r['converged'])
avg_passes = sum(r['passes'] for r in corpus_results) / total_texts

print(f"Corpus Statistics:")
print(f"Total texts: {total_texts}")
print(f"Converged: {converged_texts} ({converged_texts/total_texts:.1%})")
print(f"Average passes: {avg_passes:.1f}")
```

#### Workflow: Rule Development and Testing
```python
def develop_and_test_rule():
    """Develop and test new grammatical rules"""
    
    # Define test cases
    test_cases = [
        {"input": "test input 1", "expected": "expected output 1"},
        {"input": "test input 2", "expected": "expected output 2"},
        # ... more test cases
    ]
    
    # Create new rule
    def new_rule_match(tokens, index):
        # Rule matching logic
        pass
    
    def new_rule_apply(tokens, index):
        # Rule application logic
        pass
    
    new_rule = Rule(
        priority=1,
        id=2000,
        name="experimental_rule",
        description="Experimental grammatical rule",
        match_fn=new_rule_match,
        apply_fn=new_rule_apply
    )
    
    # Test rule
    engine.add_rule(new_rule)
    
    results = []
    for test_case in test_cases:
        result = engine.process(test_case["input"])
        actual = result.get_output_text()
        expected = test_case["expected"]
        
        test_result = {
            'input': test_case["input"],
            'expected': expected,
            'actual': actual,
            'passed': actual == expected,
            'transformations': result.get_transformation_summary()
        }
        
        results.append(test_result)
    
    # Analyze results
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"Rule Test Results: {passed}/{total} passed ({passed/total:.1%})")
    
    return results

# Develop and test new rule
test_results = develop_and_test_rule()
```

### Software Developers

#### Workflow: API Integration
```python
from flask import Flask, request, jsonify
from sanskrit_rewrite_engine import SanskritRewriteEngine

app = Flask(__name__)
engine = SanskritRewriteEngine()

@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint for text processing"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Process text
        result = engine.process(text)
        
        # Return results
        return jsonify({
            'success': True,
            'input': result.input_text,
            'output': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': result.get_transformation_summary()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for detailed analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Enable analysis features
        engine.enable_morphological_analysis()
        engine.enable_compound_analysis()
        
        result = engine.process(text)
        
        return jsonify({
            'success': True,
            'analysis': {
                'morphological': extract_morphological_info(result),
                'compounds': extract_compound_info(result),
                'transformations': result.get_transformation_summary()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

#### Workflow: Database Integration
```python
import sqlite3
import json
from datetime import datetime

class SanskritDatabase:
    def __init__(self, db_path="sanskrit_processing.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                converged BOOLEAN,
                passes INTEGER,
                transformations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_result(self, result):
        """Store processing result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_results 
            (input_text, output_text, converged, passes, transformations)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            result.input_text,
            result.get_output_text(),
            result.converged,
            result.passes,
            json.dumps(result.get_transformation_summary())
        ))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self):
        """Get processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_processed,
                AVG(passes) as avg_passes,
                SUM(CASE WHEN converged THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as convergence_rate
            FROM processing_results
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_processed': stats[0],
            'avg_passes': stats[1],
            'convergence_rate': stats[2]
        }

# Usage
db = SanskritDatabase()
engine = SanskritRewriteEngine()

# Process and store results
texts = ["rāma + iti", "deva + indra", "mahā + ātman"]

for text in texts:
    result = engine.process(text)
    db.store_result(result)

# Get statistics
stats = db.get_statistics()
print(f"Database Statistics: {stats}")
```

### Digital Humanities Researchers

#### Workflow: Textual Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def analyze_textual_patterns(texts):
    """Analyze patterns in Sanskrit texts"""
    
    results = []
    
    for text in texts:
        result = engine.process(text, enable_tracing=True)
        
        # Extract features
        features = {
            'text_length': len(text),
            'token_count': len(result.input_tokens),
            'transformation_count': len(result.get_transformation_summary()),
            'passes_required': result.passes,
            'converged': result.converged,
            'sandhi_applications': count_sandhi_applications(result),
            'compound_formations': count_compound_formations(result),
            'morphological_changes': count_morphological_changes(result)
        }
        
        results.append(features)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Text length distribution
    axes[0, 0].hist(df['text_length'], bins=20)
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].set_xlabel('Characters')
    
    # Transformation count vs text length
    axes[0, 1].scatter(df['text_length'], df['transformation_count'])
    axes[0, 1].set_title('Transformations vs Text Length')
    axes[0, 1].set_xlabel('Text Length')
    axes[0, 1].set_ylabel('Transformations')
    
    # Convergence rate
    convergence_rate = df['converged'].mean()
    axes[1, 0].bar(['Converged', 'Not Converged'], 
                   [convergence_rate, 1 - convergence_rate])
    axes[1, 0].set_title('Convergence Rate')
    axes[1, 0].set_ylabel('Proportion')
    
    # Passes distribution
    axes[1, 1].hist(df['passes_required'], bins=10)
    axes[1, 1].set_title('Passes Required Distribution')
    axes[1, 1].set_xlabel('Passes')
    
    plt.tight_layout()
    plt.savefig('textual_analysis.png')
    plt.show()
    
    return df

def count_sandhi_applications(result):
    """Count sandhi rule applications"""
    count = 0
    for trace in result.traces:
        for transformation in trace.transformations:
            if 'sandhi' in transformation.rule_name.lower():
                count += 1
    return count

def count_compound_formations(result):
    """Count compound formation applications"""
    count = 0
    for trace in result.traces:
        for transformation in trace.transformations:
            if 'compound' in transformation.rule_name.lower():
                count += 1
    return count

def count_morphological_changes(result):
    """Count morphological rule applications"""
    count = 0
    for trace in result.traces:
        for transformation in trace.transformations:
            if 'morphological' in transformation.rule_name.lower():
                count += 1
    return count

# Load and analyze corpus
corpus_texts = load_sanskrit_corpus()  # Your corpus loading function
analysis_df = analyze_textual_patterns(corpus_texts)

# Statistical summary
print("Textual Analysis Summary:")
print(analysis_df.describe())
```

## Integration Examples

### Web Application Integration

#### React Frontend Component
```javascript
// SanskritProcessor.jsx
import React, { useState } from 'react';
import axios from 'axios';

const SanskritProcessor = () => {
    const [inputText, setInputText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const processText = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await axios.post('/api/process', {
                text: inputText,
                enable_tracing: true
            });

            setResult(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'Processing failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="sanskrit-processor">
            <h2>Sanskrit Text Processor</h2>
            
            <div className="input-section">
                <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter Sanskrit text (Devanagari or IAST)"
                    rows={4}
                    cols={50}
                />
                <br />
                <button 
                    onClick={processText} 
                    disabled={loading || !inputText.trim()}
                >
                    {loading ? 'Processing...' : 'Process Text'}
                </button>
            </div>

            {error && (
                <div className="error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {result && (
                <div className="result-section">
                    <h3>Results</h3>
                    <div className="result-item">
                        <strong>Input:</strong> {result.input}
                    </div>
                    <div className="result-item">
                        <strong>Output:</strong> {result.output}
                    </div>
                    <div className="result-item">
                        <strong>Converged:</strong> {result.converged ? 'Yes' : 'No'}
                    </div>
                    <div className="result-item">
                        <strong>Passes:</strong> {result.passes}
                    </div>
                    
                    {result.transformations && (
                        <div className="transformations">
                            <h4>Transformations Applied</h4>
                            <ul>
                                {Object.entries(result.transformations).map(([rule, count]) => (
                                    <li key={rule}>
                                        {rule}: {count} applications
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default SanskritProcessor;
```

#### Vue.js Component
```vue
<!-- SanskritProcessor.vue -->
<template>
  <div class="sanskrit-processor">
    <h2>Sanskrit Text Processor</h2>
    
    <div class="input-section">
      <textarea
        v-model="inputText"
        placeholder="Enter Sanskrit text (Devanagari or IAST)"
        rows="4"
        cols="50"
      ></textarea>
      <br />
      <button 
        @click="processText" 
        :disabled="loading || !inputText.trim()"
      >
        {{ loading ? 'Processing...' : 'Process Text' }}
      </button>
    </div>

    <div v-if="error" class="error">
      <strong>Error:</strong> {{ error }}
    </div>

    <div v-if="result" class="result-section">
      <h3>Results</h3>
      <div class="result-item">
        <strong>Input:</strong> {{ result.input }}
      </div>
      <div class="result-item">
        <strong>Output:</strong> {{ result.output }}
      </div>
      <div class="result-item">
        <strong>Converged:</strong> {{ result.converged ? 'Yes' : 'No' }}
      </div>
      <div class="result-item">
        <strong>Passes:</strong> {{ result.passes }}
      </div>
      
      <div v-if="result.transformations" class="transformations">
        <h4>Transformations Applied</h4>
        <ul>
          <li v-for="(count, rule) in result.transformations" :key="rule">
            {{ rule }}: {{ count }} applications
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'SanskritProcessor',
  data() {
    return {
      inputText: '',
      result: null,
      loading: false,
      error: null
    };
  },
  methods: {
    async processText() {
      this.loading = true;
      this.error = null;

      try {
        const response = await axios.post('/api/process', {
          text: this.inputText,
          enable_tracing: true
        });

        this.result = response.data;
      } catch (err) {
        this.error = err.response?.data?.error || 'Processing failed';
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

<style scoped>
.sanskrit-processor {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.input-section {
  margin-bottom: 20px;
}

.error {
  color: red;
  margin: 10px 0;
}

.result-section {
  border: 1px solid #ccc;
  padding: 15px;
  border-radius: 5px;
}

.result-item {
  margin: 10px 0;
}

.transformations {
  margin-top: 15px;
}
</style>
```

### Command Line Interface

#### CLI Application
```python
#!/usr/bin/env python3
"""
Sanskrit Rewrite Engine CLI
Command-line interface for processing Sanskrit text
"""

import argparse
import sys
import json
from pathlib import Path
from sanskrit_rewrite_engine import SanskritRewriteEngine, EngineConfig

def main():
    parser = argparse.ArgumentParser(
        description='Process Sanskrit text using the Sanskrit Rewrite Engine'
    )
    
    # Input options
    parser.add_argument('text', nargs='?', help='Sanskrit text to process')
    parser.add_argument('-f', '--file', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    
    # Processing options
    parser.add_argument('--max-passes', type=int, default=20,
                       help='Maximum number of processing passes')
    parser.add_argument('--no-tracing', action='store_true',
                       help='Disable transformation tracing')
    parser.add_argument('--performance-mode', action='store_true',
                       help='Enable performance optimizations')
    
    # Rule options
    parser.add_argument('--rules', help='Custom rule set file')
    parser.add_argument('--list-rules', action='store_true',
                       help='List available rules')
    
    # Output options
    parser.add_argument('--format', choices=['text', 'json', 'detailed'],
                       default='text', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create engine configuration
    config = EngineConfig(
        max_passes=args.max_passes,
        enable_tracing=not args.no_tracing,
        performance_mode=args.performance_mode
    )
    
    # Initialize engine
    engine = SanskritRewriteEngine(config=config)
    
    # Load custom rules if specified
    if args.rules:
        engine.load_rule_set(args.rules)
    
    # Handle list rules command
    if args.list_rules:
        rules = engine.get_active_rules()
        print(f"Available rules ({len(rules)}):")
        for rule in rules:
            print(f"  {rule.name} (priority: {rule.priority})")
        return
    
    # Get input text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Cannot decode file '{args.file}' as UTF-8", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        input_text = args.text
    else:
        # Read from stdin
        input_text = sys.stdin.read().strip()
    
    if not input_text:
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)
    
    # Process text
    try:
        result = engine.process(input_text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Format output
    if args.format == 'text':
        output = result.get_output_text()
    elif args.format == 'json':
        output = json.dumps({
            'input': result.input_text,
            'output': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': result.get_transformation_summary()
        }, indent=2, ensure_ascii=False)
    elif args.format == 'detailed':
        output = format_detailed_output(result, args.verbose)
    
    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            if args.verbose:
                print(f"Output written to '{args.output}'")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output)

def format_detailed_output(result, verbose=False):
    """Format detailed output with transformation traces"""
    lines = []
    
    lines.append(f"Input: {result.input_text}")
    lines.append(f"Output: {result.get_output_text()}")
    lines.append(f"Converged: {result.converged}")
    lines.append(f"Passes: {result.passes}")
    lines.append("")
    
    # Transformation summary
    summary = result.get_transformation_summary()
    if summary:
        lines.append("Transformations Applied:")
        for rule_name, count in summary.items():
            lines.append(f"  {rule_name}: {count}")
        lines.append("")
    
    # Detailed traces (if verbose)
    if verbose and result.traces:
        lines.append("Detailed Transformation Traces:")
        for pass_num, pass_trace in enumerate(result.traces, 1):
            lines.append(f"Pass {pass_num}:")
            
            if not pass_trace.transformations:
                lines.append("  No transformations")
                continue
            
            for t in pass_trace.transformations:
                before = ''.join(token.text for token in t.tokens_before)
                after = ''.join(token.text for token in t.tokens_after)
                
                lines.append(f"  {t.rule_name} at position {t.index}:")
                lines.append(f"    Before: '{before}'")
                lines.append(f"    After:  '{after}'")
            lines.append("")
    
    return '\n'.join(lines)

if __name__ == '__main__':
    main()
```

#### Usage Examples
```bash
# Basic usage
sanskrit-cli "rāma + iti"

# Process file
sanskrit-cli -f input.txt -o output.txt

# JSON output with custom rules
sanskrit-cli "deva + indra" --rules custom_rules.json --format json

# Verbose detailed output
sanskrit-cli "mahā + bhārata" --format detailed --verbose

# Performance mode for large texts
sanskrit-cli -f large_text.txt --performance-mode --no-tracing

# List available rules
sanskrit-cli --list-rules
```

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems
```bash
# Issue: Permission denied during installation
sudo pip install sanskrit-rewrite-engine

# Issue: Python version compatibility
python3.9 -m pip install sanskrit-rewrite-engine

# Issue: Virtual environment problems
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install sanskrit-rewrite-engine
```

#### Runtime Errors
```python
# Issue: Unicode encoding errors
try:
    result = engine.process(text)
except UnicodeDecodeError:
    # Try different encoding
    text_utf8 = text.encode('utf-8', errors='ignore').decode('utf-8')
    result = engine.process(text_utf8)

# Issue: Memory errors with large texts
def safe_process_large_text(text, max_chunk_size=1000):
    if len(text) <= max_chunk_size:
        return engine.process(text)
    
    # Process in chunks
    chunks = [text[i:i+max_chunk_size] 
              for i in range(0, len(text), max_chunk_size)]
    
    results = []
    for chunk in chunks:
        result = engine.process(chunk)
        results.append(result.get_output_text())
    
    return ' '.join(results)

# Issue: Rule conflicts
# Check rule priorities and disable conflicting rules
conflicting_rules = engine.find_conflicting_rules()
for rule_id in conflicting_rules:
    engine.disable_rule(rule_id)
```

#### Performance Issues
```python
# Issue: Slow processing
# Enable performance optimizations
config = EngineConfig(
    performance_mode=True,
    enable_tracing=False,
    max_passes=10,
    enable_cache=True
)
engine = SanskritRewriteEngine(config=config)

# Issue: High memory usage
# Use memory-efficient settings
config = EngineConfig(
    memory_limit=256,  # MB
    enable_gc=True,
    gc_threshold=100
)

# Clear cache periodically
engine.clear_cache()
```

### Debugging Techniques

#### Enable Debug Mode
```python
# Enable comprehensive debugging
engine.set_debug_mode(True)
engine.set_log_level('DEBUG')

# Process with verbose output
result = engine.process(text, verbose=True)

# Examine debug information
debug_info = engine.get_debug_info()
print(f"Debug info: {debug_info}")
```

#### Trace Analysis
```python
def analyze_traces(result):
    """Analyze transformation traces for debugging"""
    
    print(f"Total passes: {result.passes}")
    print(f"Converged: {result.converged}")
    
    for pass_num, pass_trace in enumerate(result.traces, 1):
        print(f"\nPass {pass_num}:")
        print(f"  Transformations: {len(pass_trace.transformations)}")
        
        for t in pass_trace.transformations:
            print(f"    {t.rule_name} at {t.index}")
            
            before_text = ''.join(token.text for token in t.tokens_before)
            after_text = ''.join(token.text for token in t.tokens_after)
            
            print(f"      '{before_text}' → '{after_text}'")

# Analyze processing traces
result = engine.process("complex text", enable_tracing=True)
analyze_traces(result)
```

## Best Practices

### Input Preparation
1. **Encoding**: Always use UTF-8 encoding for Sanskrit text
2. **Normalization**: Normalize Unicode text to NFC form
3. **Cleaning**: Remove extraneous whitespace and formatting
4. **Validation**: Validate input before processing

```python
import unicodedata

def prepare_input(text):
    """Prepare Sanskrit text for processing"""
    
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc')
    
    return text

# Usage
clean_text = prepare_input(raw_input)
result = engine.process(clean_text)
```

### Performance Optimization
1. **Caching**: Enable caching for repeated processing
2. **Batch Processing**: Process multiple texts together
3. **Rule Selection**: Use only necessary rules
4. **Memory Management**: Monitor and limit memory usage

```python
# Optimized processing setup
config = EngineConfig(
    enable_cache=True,
    cache_size=5000,
    performance_mode=True,
    memory_limit=512
)

engine = SanskritRewriteEngine(config=config)

# Load only necessary rules
engine.load_rule_set("essential_rules.json")
```

### Error Handling
1. **Graceful Degradation**: Handle errors without crashing
2. **Logging**: Log errors for debugging
3. **Recovery**: Implement recovery strategies
4. **Validation**: Validate results

```python
import logging

def robust_process(text):
    """Process text with robust error handling"""
    
    try:
        # Validate input
        if not text or not text.strip():
            raise ValueError("Empty input text")
        
        # Process text
        result = engine.process(text)
        
        # Validate result
        if not result.converged:
            logging.warning(f"Processing did not converge for: {text}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing '{text}': {e}")
        
        # Return minimal result
        return create_fallback_result(text, str(e))

def create_fallback_result(text, error_message):
    """Create fallback result for failed processing"""
    return RewriteResult(
        input_text=text,
        input_tokens=[],
        output_tokens=[],
        converged=False,
        passes=0,
        traces=[],
        errors=[error_message]
    )
```

## Resources and Support

### Documentation
- **API Reference**: Complete function and class documentation
- **Developer Guide**: Comprehensive development information
- **Linguist Guide**: Sanskrit-specific usage information
- **Researcher Guide**: Research applications and methodologies

### Community
- **GitHub Repository**: Source code and issue tracking
- **Discussions**: Community Q&A and feature requests
- **Examples**: Sample code and use cases
- **Tutorials**: Step-by-step learning materials

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and references
- **Community Forums**: User discussions and help
- **Email Support**: Direct support for complex issues

### Contributing
- **Code Contributions**: Bug fixes and new features
- **Documentation**: Improvements and additions
- **Testing**: Test cases and validation
- **Community**: Helping other users

This complete user guide provides comprehensive coverage of the Sanskrit Rewrite Engine for users of all levels and backgrounds. It includes practical examples, troubleshooting information, and best practices to help users get the most out of the system.