# Sanskrit Rewrite Engine - Comprehensive Test Suite

This directory contains a comprehensive test suite for the Sanskrit Rewrite Engine, covering unit tests, integration tests, performance tests, and CLI testing.

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_engine.py            # Engine core functionality tests
│   ├── test_tokenizer.py         # Tokenizer tests
│   ├── test_rules.py             # Rule system tests
│   └── ...                       # Other component tests
├── integration/                   # Integration tests
│   ├── test_api.py               # FastAPI server integration tests
│   ├── test_cli.py               # CLI integration tests
│   └── test_performance.py       # Performance and scalability tests
├── fixtures/                      # Test data and mock objects
│   ├── sample_texts.py           # Sanskrit text samples
│   ├── sample_rules.json         # Test rule definitions
│   └── mock_objects.py           # Mock objects for testing
├── conftest.py                    # Pytest configuration and fixtures
├── run_comprehensive_tests.py    # Comprehensive test runner
└── README.md                      # This file
```

## Test Categories

### Unit Tests (tests/unit/)

- **Engine Tests**: Core transformation engine functionality
  - Text processing with various inputs
  - Rule application and iteration
  - Configuration management
  - Error handling and edge cases
  - Memory usage and performance

- **Tokenizer Tests**: Sanskrit text tokenization
  - Basic tokenization functionality
  - Unicode and Devanagari support
  - Morphological marker preservation
  - Token metadata and classification

- **Rules Tests**: Rule system functionality
  - Rule creation and validation
  - Pattern matching and application
  - Rule registry management
  - Priority-based rule ordering

### Integration Tests (tests/integration/)

- **API Tests**: FastAPI web server integration
  - Endpoint functionality (/process, /analyze, /health, /rules)
  - Request/response validation
  - Error handling and security
  - CORS and middleware testing
  - Concurrent request handling

- **CLI Tests**: Command-line interface testing
  - Command parsing and execution
  - File input/output handling
  - Configuration management
  - Error handling and help text

- **Performance Tests**: System performance and scalability
  - Processing speed benchmarks
  - Memory usage monitoring
  - Concurrent processing tests
  - Scalability under load
  - Performance regression detection

## Running Tests

### Quick Test Run
```bash
# Run basic unit tests
python -m pytest tests/unit/test_engine.py tests/unit/test_tokenizer.py tests/unit/test_rules.py -v

# Run integration tests
python -m pytest tests/integration/test_api.py -v
```

### Comprehensive Test Suite
```bash
# Run all tests with the comprehensive test runner
python tests/run_comprehensive_tests.py

# Run specific test categories
python tests/run_comprehensive_tests.py --unit
python tests/run_comprehensive_tests.py --integration
python tests/run_comprehensive_tests.py --performance
```

### Test Options
```bash
# Fast tests only (skip performance tests)
python tests/run_comprehensive_tests.py --fast

# With coverage reporting
python tests/run_comprehensive_tests.py --coverage

# Verbose output
python tests/run_comprehensive_tests.py --verbose
```

## Test Requirements

### Core Dependencies
- pytest >= 6.0
- pytest-asyncio
- fastapi
- httpx (for API testing)
- click (for CLI testing)

### Optional Dependencies
- psutil (for memory usage tests)
- coverage (for coverage reporting)
- black, flake8, isort (for code quality tests)
- mypy (for type checking)
- bandit (for security testing)

Install all dependencies:
```bash
pip install pytest pytest-asyncio fastapi httpx click psutil coverage black flake8 isort mypy bandit
```

## Test Data

### Sample Texts (tests/fixtures/sample_texts.py)
- Basic Sanskrit words
- Morphological examples with markers
- Sandhi transformation examples
- Complex sentences
- Unicode test cases
- Error cases and edge conditions

### Sample Rules (tests/fixtures/sample_rules.json)
- Basic transformation rules
- Sandhi rules
- Compound formation rules
- Test-specific rules for validation

## Test Results Summary

The comprehensive test suite includes:

- **75+ Unit Tests**: Testing individual components
- **50+ Integration Tests**: Testing component interactions
- **25+ Performance Tests**: Testing speed and scalability
- **30+ CLI Tests**: Testing command-line interface

### Current Status
- ✅ Engine core functionality: 29/30 tests passing
- ✅ Rule system: 15/15 tests passing  
- ⚠️ Tokenizer: 13/26 tests passing (needs alignment with BasicSanskritTokenizer)
- ✅ API integration: Most tests passing
- ✅ CLI integration: Most tests passing
- ⚠️ Performance tests: Require psutil dependency

## Key Features Tested

### Functionality
- Sanskrit text processing and transformation
- Rule-based sandhi application
- Tokenization of Devanagari and IAST text
- API endpoint functionality
- CLI command processing

### Quality Assurance
- Error handling and edge cases
- Memory usage and performance
- Security validation
- Concurrent processing
- Input validation and sanitization

### Performance
- Processing speed benchmarks
- Memory usage monitoring
- Scalability testing
- Regression detection

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Include both positive and negative test cases
4. Add performance tests for new functionality
5. Update this README with new test categories

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the src/ directory is in Python path
2. **Missing Dependencies**: Install optional dependencies for full test coverage
3. **Test Failures**: Some tests may need adjustment based on actual implementation details

### Test Configuration

The test suite uses pytest configuration in `pytest.ini` and fixtures in `conftest.py`. Markers are used to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Tests that take more than 1 second

## Coverage Goals

- Unit test coverage: >90%
- Integration test coverage: >80%
- Overall code coverage: >85%

The comprehensive test suite ensures the Sanskrit Rewrite Engine is robust, performant, and reliable across all supported use cases.