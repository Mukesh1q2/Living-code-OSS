# Video Tutorial Scripts for Sanskrit Rewrite Engine

## Tutorial 1: Getting Started (5 minutes)

### Script Overview
Introduction to the Sanskrit Rewrite Engine for complete beginners.

### Scene 1: Introduction (30 seconds)
**Narrator:** "Welcome to the Sanskrit Rewrite Engine tutorial series. In this first video, we'll learn how to install and use the engine for basic Sanskrit text processing."

**Screen:** Show title slide with Sanskrit text examples

### Scene 2: Installation (1 minute)
**Narrator:** "Let's start by installing the engine. Open your terminal or command prompt."

**Screen:** Terminal window
```bash
pip install sanskrit-rewrite-engine
```

**Narrator:** "The installation includes all necessary dependencies. Let's verify it works."

**Screen:** Python interpreter
```python
import sanskrit_rewrite_engine
print("Installation successful!")
```

### Scene 3: Basic Usage (2 minutes)
**Narrator:** "Now let's process our first Sanskrit text. We'll transform 'rāma + iti' using sandhi rules."

**Screen:** Python code editor
```python
from sanskrit_rewrite_engine import SanskritRewriteEngine

engine = SanskritRewriteEngine()
result = engine.process("rāma + iti")
print(f"Input: {result.input_text}")
print(f"Output: {result.get_output_text()}")
```

**Narrator:** "The engine automatically applied the vowel sandhi rule, combining 'a' and 'i' to form 'e'."

### Scene 4: Understanding Results (1 minute)
**Narrator:** "Let's examine what happened during processing."

**Screen:** Code showing trace analysis
```python
print(f"Converged: {result.converged}")
print(f"Passes: {result.passes}")
summary = result.get_transformation_summary()
for rule, count in summary.items():
    print(f"{rule}: {count} applications")
```

### Scene 5: Wrap-up (30 seconds)
**Narrator:** "In this tutorial, we learned basic installation and usage. Next, we'll explore advanced features like custom rules and morphological analysis."

---

## Tutorial 2: Advanced Features (8 minutes)

### Script Overview
Deep dive into morphological analysis, compound formation, and custom rules.

### Scene 1: Morphological Analysis (2 minutes)
**Narrator:** "The engine can analyze Sanskrit word structure. Let's enable morphological analysis."

**Screen:** Code demonstration
```python
engine.enable_morphological_analysis()
result = engine.process("rāma : GEN")  # Genitive case
print(f"Result: {result.get_output_text()}")  # rāmasya
```

### Scene 2: Compound Formation (2 minutes)
**Narrator:** "Sanskrit compounds are automatically formed using traditional rules."

**Screen:** Multiple examples
```python
examples = [
    "deva + rāja",     # devarāja
    "mahā + bhārata",  # mahābhārata
    "su + putra"       # suputra
]
```

### Scene 3: Custom Rules (3 minutes)
**Narrator:** "You can create custom transformation rules for specialized texts."

**Screen:** Rule creation example
```python
from sanskrit_rewrite_engine import Rule

def custom_match(tokens, index):
    return tokens[index].text == "special"

def custom_apply(tokens, index):
    # Custom transformation logic
    pass

custom_rule = Rule(
    priority=1,
    name="custom_rule",
    match_fn=custom_match,
    apply_fn=custom_apply
)
```

### Scene 4: Performance Tips (1 minute)
**Narrator:** "For large texts, use performance optimizations."

**Screen:** Configuration example
```python
config = EngineConfig(performance_mode=True)
engine = SanskritRewriteEngine(config=config)
```

---

## Tutorial 3: Research Applications (10 minutes)

### Script Overview
Using the engine for computational linguistics research and corpus analysis.

### Scene 1: Corpus Processing (3 minutes)
**Narrator:** "Researchers can process entire Sanskrit corpora systematically."

**Screen:** Batch processing code
```python
def process_corpus(directory):
    results = []
    for file in directory.glob("*.txt"):
        with open(file) as f:
            text = f.read()
        result = engine.process(text)
        results.append(result)
    return results
```

### Scene 2: Statistical Analysis (3 minutes)
**Narrator:** "Extract linguistic patterns and statistics from processed texts."

**Screen:** Analysis code with visualizations
```python
import pandas as pd
import matplotlib.pyplot as plt

# Create analysis DataFrame
df = pd.DataFrame(analysis_results)
df['transformation_count'].hist()
plt.title('Transformation Distribution')
plt.show()
```

### Scene 3: Rule Development (4 minutes)
**Narrator:** "Develop and test new grammatical rules systematically."

**Screen:** Rule testing framework
```python
def test_rule(rule, test_cases):
    results = []
    for case in test_cases:
        result = engine.process(case['input'])
        results.append({
            'expected': case['expected'],
            'actual': result.get_output_text(),
            'passed': result.get_output_text() == case['expected']
        })
    return results
```

---

## Tutorial 4: Web Integration (7 minutes)

### Script Overview
Building web applications with the Sanskrit Rewrite Engine.

### Scene 1: Flask API (3 minutes)
**Narrator:** "Create a REST API for Sanskrit text processing."

**Screen:** Flask application code
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
engine = SanskritRewriteEngine()

@app.route('/api/process', methods=['POST'])
def process_text():
    data = request.get_json()
    result = engine.process(data['text'])
    return jsonify({
        'input': result.input_text,
        'output': result.get_output_text()
    })
```

### Scene 2: Frontend Integration (3 minutes)
**Narrator:** "Build a user-friendly web interface."

**Screen:** HTML/JavaScript code
```html
<textarea id="input" placeholder="Enter Sanskrit text"></textarea>
<button onclick="processText()">Process</button>
<div id="output"></div>

<script>
async function processText() {
    const text = document.getElementById('input').value;
    const response = await fetch('/api/process', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    });
    const result = await response.json();
    document.getElementById('output').textContent = result.output;
}
</script>
```

### Scene 3: Deployment (1 minute)
**Narrator:** "Deploy your application using Docker or cloud services."

**Screen:** Dockerfile and deployment commands
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## Tutorial 5: Troubleshooting and Debugging (6 minutes)

### Script Overview
Common issues and debugging techniques.

### Scene 1: Common Errors (2 minutes)
**Narrator:** "Let's address the most common issues users encounter."

**Screen:** Error examples and solutions
```python
# Unicode encoding issues
try:
    result = engine.process(text)
except UnicodeDecodeError:
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    result = engine.process(text)
```

### Scene 2: Debug Mode (2 minutes)
**Narrator:** "Enable debug mode to understand what's happening during processing."

**Screen:** Debug configuration
```python
engine.set_debug_mode(True)
result = engine.process(text, verbose=True)

# Examine traces
for trace in result.traces:
    print(f"Applied: {trace.rule_name}")
```

### Scene 3: Performance Issues (2 minutes)
**Narrator:** "Optimize performance for large texts and complex processing."

**Screen:** Performance monitoring
```python
import time
start_time = time.time()
result = engine.process(large_text)
processing_time = time.time() - start_time
print(f"Processing took {processing_time:.2f} seconds")
```

---

## Production Notes

### Equipment Needed
- Screen recording software (OBS Studio, Camtasia)
- High-quality microphone
- Code editor with syntax highlighting
- Terminal/command prompt
- Web browser for demonstrations

### Recording Guidelines
1. **Resolution**: 1920x1080 minimum
2. **Frame Rate**: 30 FPS
3. **Audio**: Clear narration, no background noise
4. **Code**: Large, readable fonts
5. **Pacing**: Slow enough to follow along

### Post-Production
1. **Editing**: Remove pauses, add transitions
2. **Captions**: Include subtitles for accessibility
3. **Chapters**: Add chapter markers for navigation
4. **Thumbnails**: Create engaging preview images
5. **Descriptions**: Include code examples and links

### Distribution
- YouTube channel with playlists
- Documentation website embedding
- GitHub repository links
- Community forums sharing

These tutorial scripts provide comprehensive coverage of the Sanskrit Rewrite Engine from basic usage to advanced applications, suitable for video production and user education.