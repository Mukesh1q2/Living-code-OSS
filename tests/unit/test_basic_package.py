"""
Basic tests to verify the package structure works correctly.
"""

import pytest


def test_package_imports():
    """Test that the main package components can be imported."""
    # Test core engine import
    from src.sanskrit_rewrite_engine.engine import SanskritRewriteEngine, TransformationResult
    
    # Test tokenizer import
    from src.sanskrit_rewrite_engine.tokenizer import BasicSanskritTokenizer, Token
    
    # Test rules import
    from src.sanskrit_rewrite_engine.rules import Rule, RuleRegistry
    
    # Test CLI import
    from src.sanskrit_rewrite_engine.cli import main
    
    # Test server import
    from src.sanskrit_rewrite_engine.server import create_app
    
    assert SanskritRewriteEngine is not None
    assert BasicSanskritTokenizer is not None
    assert Rule is not None
    assert RuleRegistry is not None
    assert main is not None
    assert create_app is not None


def test_engine_basic_functionality():
    """Test basic engine functionality."""
    from src.sanskrit_rewrite_engine.engine import SanskritRewriteEngine
    
    engine = SanskritRewriteEngine()
    result = engine.process("test text")
    
    assert result.success is True
    assert result.input_text == "test text"
    assert result.output_text == "test text"  # No transformation yet
    assert isinstance(result.transformations_applied, list)
    assert isinstance(result.trace, list)


def test_tokenizer_basic_functionality():
    """Test basic tokenizer functionality."""
    from src.sanskrit_rewrite_engine.tokenizer import BasicSanskritTokenizer
    
    tokenizer = BasicSanskritTokenizer()
    tokens = tokenizer.tokenize("test text")
    
    assert len(tokens) == 2
    assert tokens[0].text == "test"
    assert tokens[1].text == "text"


def test_rule_registry_basic_functionality():
    """Test basic rule registry functionality."""
    from src.sanskrit_rewrite_engine.rules import RuleRegistry, Rule
    
    registry = RuleRegistry()
    rule = Rule(
        id="test_rule",
        name="Test Rule",
        description="A test rule",
        pattern="test",
        replacement="tested"
    )
    
    registry.add_rule(rule)
    assert registry.get_rule_count() == 1
    
    rules = registry.get_rules_by_priority()
    assert len(rules) == 1
    assert rules[0].id == "test_rule"