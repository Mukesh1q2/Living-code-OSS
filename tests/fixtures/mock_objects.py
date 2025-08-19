"""
Mock objects and fixtures for testing.
"""

from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MockToken:
    """Mock token for testing."""
    text: str
    start_pos: int
    end_pos: int
    token_type: str = "WORD"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockRule:
    """Mock rule for testing."""
    id: str
    name: str
    description: str
    pattern: str
    replacement: str
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockTransformationResult:
    """Mock transformation result for testing."""
    input_text: str
    output_text: str
    transformations_applied: List[str]
    trace: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None


class MockSanskritRewriteEngine:
    """Mock Sanskrit Rewrite Engine for testing."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.rules_loaded = []
        
    def process(self, text: str) -> MockTransformationResult:
        """Mock process method."""
        return MockTransformationResult(
            input_text=text,
            output_text=text.replace("+", ""),  # Simple mock transformation
            transformations_applied=["mock_rule"],
            trace=[{"rule": "mock_rule", "applied": True}],
            success=True
        )
        
    def load_rules(self, rule_file: str) -> None:
        """Mock rule loading."""
        self.rules_loaded.append(rule_file)
        
    def add_rule(self, rule: MockRule) -> None:
        """Mock rule addition."""
        self.rules_loaded.append(rule.id)


class MockBasicSanskritTokenizer:
    """Mock tokenizer for testing."""
    
    def tokenize(self, text: str) -> List[MockToken]:
        """Mock tokenization."""
        tokens = []
        for i, char in enumerate(text):
            if char.isalpha():
                tokens.append(MockToken(
                    text=char,
                    start_pos=i,
                    end_pos=i+1,
                    token_type="WORD"
                ))
        return tokens
        
    def is_sanskrit_char(self, char: str) -> bool:
        """Mock Sanskrit character detection."""
        sanskrit_chars = "अआइईउऊऋॠऌॡएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        return char in sanskrit_chars


class MockRuleRegistry:
    """Mock rule registry for testing."""
    
    def __init__(self):
        self._rules = []
        
    def load_from_json(self, file_path: str) -> None:
        """Mock JSON loading."""
        self._rules.append(MockRule(
            id="mock_rule",
            name="Mock Rule",
            description="A mock rule for testing",
            pattern="\\+",
            replacement=""
        ))
        
    def add_rule(self, rule: MockRule) -> None:
        """Mock rule addition."""
        self._rules.append(rule)
        
    def get_applicable_rules(self, text: str, position: int) -> List[MockRule]:
        """Mock rule matching."""
        return [rule for rule in self._rules if rule.pattern in text]
        
    def get_rules_by_priority(self) -> List[MockRule]:
        """Mock priority sorting."""
        return sorted(self._rules, key=lambda r: r.priority)


def create_mock_fastapi_app():
    """Create a mock FastAPI app for testing."""
    app = Mock()
    app.post = Mock()
    app.get = Mock()
    app.add_middleware = Mock()
    return app


def create_mock_requests_response(status_code: int = 200, json_data: Dict = None):
    """Create a mock requests response."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data or {"success": True}
    response.text = str(json_data) if json_data else "OK"
    return response


# Common test configurations
TEST_CONFIG = {
    "max_iterations": 5,
    "enable_tracing": True,
    "rule_directories": ["tests/fixtures/rules"],
    "default_rule_set": "test_rules"
}

# Sample rule definitions for testing
SAMPLE_RULES = [
    {
        "id": "test_rule_1",
        "name": "Test Rule 1",
        "description": "Remove plus signs",
        "pattern": "\\+",
        "replacement": "",
        "priority": 1,
        "enabled": True,
        "metadata": {"category": "test"}
    },
    {
        "id": "test_rule_2", 
        "name": "Test Rule 2",
        "description": "Replace spaces with underscores",
        "pattern": " ",
        "replacement": "_",
        "priority": 2,
        "enabled": True,
        "metadata": {"category": "test"}
    }
]