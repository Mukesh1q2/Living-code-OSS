"""
Pytest configuration and fixtures for Sanskrit Rewrite Engine tests.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock

from tests.fixtures.sample_texts import (
    BASIC_WORDS, 
    MORPHOLOGICAL_EXAMPLES,
    COMPOUND_EXAMPLES,
    SANDHI_EXAMPLES,
    TRANSLITERATION_EXAMPLES,
    COMPLEX_SENTENCES,
    ERROR_CASES
)
from tests.fixtures.mock_objects import (
    MockSanskritRewriteEngine,
    MockBasicSanskritTokenizer,
    MockRuleRegistry,
    MockToken,
    MockRule,
    MockTransformationResult,
    TEST_CONFIG,
    SAMPLE_RULES,
    create_mock_fastapi_app,
    create_mock_requests_response
)


@pytest.fixture
def sample_texts():
    """Provide sample Sanskrit texts for testing."""
    return {
        "basic_words": BASIC_WORDS,
        "morphological": MORPHOLOGICAL_EXAMPLES,
        "compounds": COMPOUND_EXAMPLES,
        "sandhi": SANDHI_EXAMPLES,
        "transliteration": TRANSLITERATION_EXAMPLES,
        "complex": COMPLEX_SENTENCES,
        "error_cases": ERROR_CASES
    }


@pytest.fixture
def mock_engine():
    """Provide a mock Sanskrit Rewrite Engine."""
    return MockSanskritRewriteEngine(TEST_CONFIG)


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    return MockBasicSanskritTokenizer()


@pytest.fixture
def mock_rule_registry():
    """Provide a mock rule registry."""
    return MockRuleRegistry()


@pytest.fixture
def sample_rule():
    """Provide a sample rule for testing."""
    return MockRule(
        id="test_rule",
        name="Test Rule",
        description="A test rule",
        pattern="test",
        replacement="TEST",
        priority=1,
        enabled=True,
        metadata={"test": True}
    )


@pytest.fixture
def sample_token():
    """Provide a sample token for testing."""
    return MockToken(
        text="test",
        start_pos=0,
        end_pos=4,
        token_type="WORD",
        metadata={"test": True}
    )


@pytest.fixture
def sample_transformation_result():
    """Provide a sample transformation result."""
    return MockTransformationResult(
        input_text="test input",
        output_text="test output",
        transformations_applied=["test_rule"],
        trace=[{"rule": "test_rule", "applied": True}],
        success=True
    )


@pytest.fixture
def temp_rule_file():
    """Create a temporary rule file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": SAMPLE_RULES
        }, f, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(TEST_CONFIG, f, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_fastapi_app():
    """Provide a mock FastAPI app."""
    return create_mock_fastapi_app()


@pytest.fixture
def mock_http_response():
    """Provide a mock HTTP response."""
    return create_mock_requests_response()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sanskrit_text_samples():
    """Provide various Sanskrit text samples for comprehensive testing."""
    return {
        "simple": "rama",
        "with_markers": "rāma + iti",
        "compound": "rāja + putra",
        "complex": "rāmo rājā daśarathasya putraḥ",
        "devanagari": "राम",
        "mixed": "rāma and राम",
        "empty": "",
        "whitespace": "   ",
        "special_chars": "!@#$%"
    }


@pytest.fixture
def performance_test_data():
    """Provide test data for performance testing."""
    from tests.fixtures.sample_texts import PERFORMANCE_TEST_TEXTS
    return PERFORMANCE_TEST_TEXTS


@pytest.fixture
def unicode_test_cases():
    """Provide Unicode test cases."""
    from tests.fixtures.sample_texts import UNICODE_TEST_CASES
    return UNICODE_TEST_CASES


@pytest.fixture
def edge_cases():
    """Provide edge case test data."""
    from tests.fixtures.sample_texts import EDGE_CASES
    return EDGE_CASES


@pytest.fixture
def stress_test_data():
    """Provide stress test data."""
    from tests.fixtures.sample_texts import STRESS_TEST_TEXTS
    return STRESS_TEST_TEXTS


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for tests."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_server_response():
    """Provide mock server response for API testing."""
    def _create_response(status_code=200, json_data=None, headers=None):
        from unittest.mock import Mock
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data or {"success": True}
        response.headers = headers or {}
        response.text = str(json_data) if json_data else "OK"
        return response
    return _create_response


@pytest.fixture(scope="session")
def test_engine():
    """Provide a test engine instance for session-wide use."""
    from sanskrit_rewrite_engine.engine import SanskritRewriteEngine
    return SanskritRewriteEngine()


@pytest.fixture
def benchmark_timer():
    """Provide a timer for benchmarking tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
            
    return Timer()


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    yield
    # Any cleanup code can go here


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and scalability tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "fast: Tests that complete quickly"
    )


# Custom pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add unit marker for tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker for tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests with "slow" in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.fast)