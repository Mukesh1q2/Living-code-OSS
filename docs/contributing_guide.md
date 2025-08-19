# Contributing to Sanskrit Rewrite Engine

## Welcome Contributors!

Thank you for your interest in contributing to the Sanskrit Rewrite Engine! This guide will help you get started with contributing code, documentation, tests, and other improvements to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Contribution Guidelines](#code-contribution-guidelines)
4. [Documentation Contributions](#documentation-contributions)
5. [Testing Guidelines](#testing-guidelines)
6. [Issue Reporting](#issue-reporting)
7. [Pull Request Process](#pull-request-process)
8. [Code Style and Standards](#code-style-and-standards)
9. [Community Guidelines](#community-guidelines)
10. [Recognition and Credits](#recognition-and-credits)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of Sanskrit grammar (helpful but not required)
- Familiarity with computational linguistics concepts (optional)

### First-Time Contributors

If you're new to open source or this project:

1. **Start Small**: Look for issues labeled `good-first-issue` or `help-wanted`
2. **Read the Code**: Familiarize yourself with the codebase structure
3. **Join Discussions**: Participate in GitHub discussions and issues
4. **Ask Questions**: Don't hesitate to ask for clarification

## Development Environment Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/sanskrit-rewrite-engine.git
cd sanskrit-rewrite-engine

# Add the original repository as upstream
git remote add upstream https://github.com/ORIGINAL_OWNER/sanskrit-rewrite-engine.git
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Run tests to ensure everything is working
python -m pytest tests/ -v

# Run a simple example
python examples/basic_usage.py
```

## Code Contribution Guidelines

### Areas for Contribution

#### Core Engine Development
- **Token System**: Improve tokenization accuracy and performance
- **Rule Engine**: Add new grammatical rules or optimize existing ones
- **Guard System**: Enhance loop prevention and application limits
- **Trace System**: Improve debugging and analysis capabilities

#### Rule Development
- **Sandhi Rules**: Implement additional phonological transformations
- **Morphological Rules**: Add support for complex word formations
- **Compound Rules**: Enhance compound analysis and formation
- **Meta-Rules**: Develop context-sensitive rule control

#### Integration Features
- **API Enhancements**: Improve external system integration
- **Performance Optimizations**: Speed and memory improvements
- **Error Handling**: Better error messages and recovery
- **Validation**: Input/output validation and consistency checks

### Code Structure

```
sanskrit_rewrite_engine/
├── __init__.py              # Main package initialization
├── token.py                 # Token system implementation
├── rule.py                  # Rule definition and management
├── tokenizer.py             # Text tokenization
├── engine.py                # Main rewrite engine
├── guard.py                 # Guard system for loop prevention
├── trace.py                 # Transformation tracing
├── utils.py                 # Utility functions
├── exceptions.py            # Custom exceptions
└── validators.py            # Input/output validation
```

### Adding New Features

#### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

#### 2. Implement Your Feature

Follow these guidelines:
- Write clear, documented code
- Add comprehensive tests
- Update documentation
- Follow existing code patterns

#### 3. Example: Adding a New Rule Type

```python
# In rule.py
class ConditionalRule(Rule):
    """Rule that applies only when specific conditions are met"""
    
    def __init__(self, condition_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_fn = condition_fn
    
    def can_apply(self, tokens: List[Token], index: int) -> bool:
        """Check if rule can apply given current conditions"""
        if not super().can_apply():
            return False
        return self.condition_fn(tokens, index)

# In tests/test_rule.py
def test_conditional_rule():
    """Test conditional rule application"""
    def condition(tokens, index):
        return len(tokens) > 5  # Only apply to longer sequences
    
    rule = ConditionalRule(
        condition_fn=condition,
        priority=1,
        id=1,
        name="test_conditional",
        match_fn=lambda tokens, i: tokens[i].text == "test",
        apply_fn=lambda tokens, i: (tokens, i)
    )
    
    short_tokens = [Token("test", TokenKind.OTHER)]
    long_tokens = [Token("test", TokenKind.OTHER)] * 6
    
    assert not rule.can_apply(short_tokens, 0)
    assert rule.can_apply(long_tokens, 0)
```

## Documentation Contributions

### Types of Documentation

#### Code Documentation
- **Docstrings**: All public functions and classes must have docstrings
- **Type Hints**: Use comprehensive type annotations
- **Inline Comments**: Explain complex logic and algorithms

#### User Documentation
- **Guides**: Step-by-step tutorials for different user types
- **Examples**: Practical code examples and use cases
- **API Reference**: Complete function and class documentation
- **Troubleshooting**: Common issues and solutions

#### Developer Documentation
- **Architecture**: System design and component interactions
- **Contributing**: Guidelines for contributors (this document)
- **Testing**: Testing strategies and requirements
- **Performance**: Optimization techniques and benchmarks

### Documentation Standards

#### Docstring Format

```python
def process_sanskrit_text(text: str, rules: List[Rule]) -> RewriteResult:
    """Process Sanskrit text using specified transformation rules.
    
    This function tokenizes the input text, applies the given rules in
    priority order, and returns a complete result with transformation traces.
    
    Args:
        text: Input Sanskrit text in IAST or Devanagari
        rules: List of transformation rules to apply
        
    Returns:
        RewriteResult containing processed text and transformation traces
        
    Raises:
        TokenizationError: If input text cannot be tokenized
        RuleApplicationError: If rule application fails
        
    Example:
        >>> rules = [create_sandhi_rule("a", "i", "e")]
        >>> result = process_sanskrit_text("rāma + iti", rules)
        >>> print(result.get_output_text())
        'rāmeti'
    """
```

#### Markdown Documentation

- Use clear headings and structure
- Include code examples with syntax highlighting
- Add cross-references between related sections
- Keep examples practical and tested

## Testing Guidelines

### Test Categories

#### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Cover edge cases and error conditions
- Aim for >90% code coverage

#### Integration Tests
- Test component interactions
- Use real data and configurations
- Verify end-to-end workflows
- Test performance characteristics

#### Validation Tests
- Test against known Sanskrit corpora
- Verify linguistic accuracy
- Compare with traditional grammar sources
- Test cross-platform compatibility

### Writing Tests

#### Test Structure

```python
import pytest
from sanskrit_rewrite_engine import SanskritRewriteEngine, Token, TokenKind

class TestSandhiRules:
    """Test suite for sandhi transformation rules"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = SanskritRewriteEngine()
        self.engine.load_sandhi_rules()
    
    def test_vowel_sandhi_a_i(self):
        """Test a + i → e transformation"""
        # Arrange
        input_text = "rāma + iti"
        expected_output = "rāmeti"
        
        # Act
        result = self.engine.process(input_text)
        
        # Assert
        assert result.converged
        assert result.get_output_text() == expected_output
        assert any("vowel_sandhi" in trace.rule_name 
                  for pass_trace in result.traces 
                  for trace in pass_trace.transformations)
    
    @pytest.mark.parametrize("input_text,expected", [
        ("deva + indra", "devendra"),
        ("mahā + ātman", "mahātman"),
        ("su + ukta", "sukta")
    ])
    def test_multiple_sandhi_cases(self, input_text, expected):
        """Test multiple sandhi transformations"""
        result = self.engine.process(input_text)
        assert result.get_output_text() == expected
    
    def test_sandhi_with_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(TokenizationError):
            self.engine.process("invalid\x00text")
```

#### Test Data

Create comprehensive test datasets:

```python
# tests/data/sandhi_examples.json
{
    "vowel_sandhi": [
        {"input": "rāma + iti", "output": "rāmeti", "rule": "a_i_sandhi"},
        {"input": "deva + indra", "output": "devendra", "rule": "a_i_sandhi"}
    ],
    "consonant_sandhi": [
        {"input": "tat + ca", "output": "tac ca", "rule": "t_c_assimilation"}
    ]
}
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=sanskrit_rewrite_engine --cov-report=html

# Run specific test file
python -m pytest tests/test_sandhi.py -v

# Run tests matching pattern
python -m pytest -k "sandhi" -v

# Run performance tests
python -m pytest tests/test_performance.py --benchmark-only
```

## Issue Reporting

### Before Reporting an Issue

1. **Search Existing Issues**: Check if the issue already exists
2. **Update Dependencies**: Ensure you're using the latest version
3. **Reproduce the Issue**: Create a minimal reproduction case
4. **Check Documentation**: Verify expected behavior

### Issue Template

```markdown
## Issue Description
Brief description of the problem

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Package Version: [e.g., 1.2.3]

## Code Example
```python
# Minimal code to reproduce the issue
```

## Additional Context
Any other relevant information
```

### Issue Labels

- `bug`: Something isn't working correctly
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed
- `performance`: Performance-related issues
- `linguistic`: Sanskrit grammar or linguistics issues

## Pull Request Process

### Before Submitting

1. **Create an Issue**: Discuss major changes before implementing
2. **Follow Guidelines**: Ensure code follows project standards
3. **Write Tests**: Add comprehensive test coverage
4. **Update Documentation**: Keep docs in sync with code changes
5. **Test Thoroughly**: Run full test suite locally

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #[issue number]

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality manually if needed
4. **Feedback**: Address reviewer comments and suggestions
5. **Approval**: Maintainer approves and merges PR

## Code Style and Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these additions:

#### Formatting
- **Line Length**: Maximum 88 characters (Black formatter default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Group imports (standard library, third-party, local)

#### Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Members**: `_leading_underscore`

#### Type Hints
```python
from typing import List, Dict, Optional, Union, Callable

def process_tokens(
    tokens: List[Token], 
    rules: Dict[str, Rule],
    max_passes: Optional[int] = None
) -> Tuple[List[Token], bool]:
    """Process tokens with type hints"""
    pass
```

### Code Quality Tools

#### Linting and Formatting
```bash
# Install tools
pip install black isort flake8 mypy

# Format code
black sanskrit_rewrite_engine/
isort sanskrit_rewrite_engine/

# Check style
flake8 sanskrit_rewrite_engine/

# Type checking
mypy sanskrit_rewrite_engine/
```

#### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Documentation Style

#### Code Comments
```python
class RuleEngine:
    """Manages and applies transformation rules to token sequences.
    
    The RuleEngine coordinates rule application, manages priorities,
    and prevents infinite loops through the guard system.
    """
    
    def apply_rules(self, tokens: List[Token]) -> List[Token]:
        """Apply all active rules to token sequence.
        
        Rules are applied in priority order until no more transformations
        are possible or maximum passes are reached.
        """
        # Sort rules by priority (lower numbers = higher priority)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        
        # Apply rules iteratively until convergence
        for pass_num in range(self.max_passes):
            # ... implementation
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat all community members with respect
- **Be Inclusive**: Welcome people of all backgrounds and experience levels
- **Be Collaborative**: Work together constructively
- **Be Patient**: Help others learn and grow
- **Be Professional**: Maintain professional communication

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Pull Requests**: Code review and collaboration
- **Documentation**: Comprehensive guides and references

### Getting Help

- **Documentation**: Check existing guides and API reference
- **Search Issues**: Look for similar problems and solutions
- **Ask Questions**: Create a discussion or issue for help
- **Community**: Engage with other contributors and users

## Recognition and Credits

### Contributor Recognition

We value all contributions and recognize contributors through:

- **Contributors File**: Listed in CONTRIBUTORS.md
- **Release Notes**: Mentioned in version release notes
- **GitHub**: Automatic contribution tracking
- **Documentation**: Author credits in relevant sections

### Types of Contributions

All contributions are valuable:
- **Code**: New features, bug fixes, optimizations
- **Documentation**: Guides, examples, API docs
- **Testing**: Test cases, validation, quality assurance
- **Issues**: Bug reports, feature requests, discussions
- **Community**: Helping others, answering questions

### Contribution Workflow Summary

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly
5. **Document** your changes
6. **Submit** a pull request
7. **Collaborate** on review process
8. **Celebrate** your contribution!

## Advanced Contribution Topics

### Performance Contributions

#### Profiling and Benchmarking
```python
import cProfile
import pstats
from sanskrit_rewrite_engine import SanskritRewriteEngine

def profile_engine_performance():
    """Profile engine performance for optimization"""
    engine = SanskritRewriteEngine()
    
    # Profile critical operations
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run performance-critical code
    for _ in range(1000):
        result = engine.process("test input")
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### Memory Optimization
```python
import tracemalloc
import gc

def memory_profile():
    """Profile memory usage for optimization"""
    tracemalloc.start()
    
    # Your code here
    engine = SanskritRewriteEngine()
    results = []
    for i in range(1000):
        result = engine.process(f"test {i}")
        results.append(result)
    
    # Force garbage collection
    gc.collect()
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

### Linguistic Contributions

#### Adding New Sanskrit Rules
```python
def create_vedic_sandhi_rule():
    """Example of adding Vedic-specific sandhi rules"""
    
    def match_vedic_pattern(tokens: List[Token], index: int) -> bool:
        """Match Vedic-specific phonological patterns"""
        # Implement Vedic pattern matching
        pass
    
    def apply_vedic_transformation(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
        """Apply Vedic-specific transformations"""
        # Implement Vedic transformation
        pass
    
    return Rule(
        priority=0,  # High priority for Vedic rules
        id=generate_rule_id(),
        name="vedic_sandhi_special",
        description="Vedic-specific sandhi transformation",
        match_fn=match_vedic_pattern,
        apply_fn=apply_vedic_transformation,
        sutra_ref="Vedic Grammar Reference",
        meta_data={"period": "vedic", "source": "scholarly_reference"}
    )
```

### Integration Contributions

#### API Extensions
```python
from flask import Flask, request, jsonify
from sanskrit_rewrite_engine import SanskritRewriteEngine

app = Flask(__name__)
engine = SanskritRewriteEngine()

@app.route('/api/v1/process', methods=['POST'])
def api_process_text():
    """REST API endpoint for text processing"""
    data = request.get_json()
    
    try:
        result = engine.process(
            text=data['text'],
            max_passes=data.get('max_passes', 20)
        )
        
        return jsonify({
            'success': True,
            'input': result.input_text,
            'output': result.get_output_text(),
            'converged': result.converged,
            'passes': result.passes,
            'transformations': len(result.traces)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
```

Thank you for contributing to the Sanskrit Rewrite Engine! Your contributions help advance computational Sanskrit linguistics and make these tools accessible to researchers, scholars, and developers worldwide.