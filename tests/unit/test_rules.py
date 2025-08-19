"""
Tests for the rule system.
"""

import json
import pytest
import tempfile
import os
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.sanskrit_rewrite_engine.rules import Rule, RuleRegistry


class TestRule:
    """Test the Rule class."""
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            pattern="a",
            replacement="b"
        )
        
        assert rule.id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.description == "A test rule"
        assert rule.pattern == "a"
        assert rule.replacement == "b"
        assert rule.priority == 1
        assert rule.enabled is True
        assert rule.metadata == {}
    
    def test_rule_matches_simple(self):
        """Test simple pattern matching."""
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern="hello",
            replacement="hi"
        )
        
        # Should match at position 0
        assert rule.matches("hello world", 0) is True
        
        # Should not match at position 1
        assert rule.matches("hello world", 1) is False
        
        # Should match at position 6
        assert rule.matches("say hello", 4) is True
    
    def test_rule_matches_regex(self):
        """Test regex pattern matching."""
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern="a\\s*\\+\\s*i",
            replacement="e"
        )
        
        assert rule.matches("a + i", 0) is True
        assert rule.matches("a+i", 0) is True
        assert rule.matches("a  +  i", 0) is True
        assert rule.matches("hello a + i", 6) is True
        assert rule.matches("ai", 0) is False
    
    def test_rule_apply_simple(self):
        """Test simple rule application."""
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern="hello",
            replacement="hi"
        )
        
        text, pos = rule.apply("hello world", 0)
        assert text == "hi world"
        assert pos == 2  # Position after "hi"
        
        text, pos = rule.apply("say hello there", 4)
        assert text == "say hi there"
        assert pos == 6  # Position after "hi"
    
    def test_rule_apply_regex(self):
        """Test regex rule application."""
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern="a\\s*\\+\\s*i",
            replacement="e"
        )
        
        text, pos = rule.apply("a + i", 0)
        assert text == "e"
        assert pos == 1  # Position after "e"
        
        # Test matching at position where pattern starts
        # "rāma + iti" - the pattern "a + i" should match at position 3 (the 'a' in rāma)
        text, pos = rule.apply("rāma + iti", 3)
        assert text == "rāmeti"
        assert pos == 4  # Position after "e"
    
    def test_rule_apply_with_groups(self):
        """Test regex rule application with capture groups."""
        rule = Rule(
            id="test",
            name="Test",
            description="Test",
            pattern="([a-z0-9]+)\\s*\\+\\s*([a-z0-9]+)",
            replacement="$1$2"
        )
        
        text, pos = rule.apply("word1 + word2", 0)
        assert text == "word1word2"
        assert pos == 10  # Position after "word1word2"


class TestRuleRegistry:
    """Test the RuleRegistry class."""
    
    def test_registry_creation(self):
        """Test basic registry creation."""
        registry = RuleRegistry()
        assert registry.get_rule_count() == 0
    
    def test_add_rule(self):
        """Test adding rules to registry."""
        registry = RuleRegistry()
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            pattern="a",
            replacement="b"
        )
        
        registry.add_rule(rule)
        assert registry.get_rule_count() == 1
        assert registry.get_rule_by_id("test_rule") == rule
    
    def test_add_duplicate_rule_id(self):
        """Test that adding duplicate rule IDs raises error."""
        registry = RuleRegistry()
        rule1 = Rule(id="test", name="Test 1", description="Test", pattern="a", replacement="b")
        rule2 = Rule(id="test", name="Test 2", description="Test", pattern="c", replacement="d")
        
        registry.add_rule(rule1)
        
        with pytest.raises(ValueError, match="Rule with ID 'test' already exists"):
            registry.add_rule(rule2)
    
    def test_get_applicable_rules(self):
        """Test getting applicable rules."""
        registry = RuleRegistry()
        
        rule1 = Rule(id="rule1", name="Rule 1", description="Test", pattern="hello", replacement="hi")
        rule2 = Rule(id="rule2", name="Rule 2", description="Test", pattern="world", replacement="earth")
        rule3 = Rule(id="rule3", name="Rule 3", description="Test", pattern="foo", replacement="bar")
        
        registry.add_rule(rule1)
        registry.add_rule(rule2)
        registry.add_rule(rule3)
        
        # Test at position 0 - should match "hello"
        applicable = registry.get_applicable_rules("hello world", 0)
        assert len(applicable) == 1
        assert applicable[0].id == "rule1"
        
        # Test at position 6 - should match "world"
        applicable = registry.get_applicable_rules("hello world", 6)
        assert len(applicable) == 1
        assert applicable[0].id == "rule2"
        
        # Test at position with no matches
        applicable = registry.get_applicable_rules("hello world", 3)
        assert len(applicable) == 0
    
    def test_get_rules_by_priority(self):
        """Test getting rules sorted by priority."""
        registry = RuleRegistry()
        
        rule1 = Rule(id="rule1", name="Rule 1", description="Test", pattern="a", replacement="b", priority=3)
        rule2 = Rule(id="rule2", name="Rule 2", description="Test", pattern="c", replacement="d", priority=1)
        rule3 = Rule(id="rule3", name="Rule 3", description="Test", pattern="e", replacement="f", priority=2)
        
        registry.add_rule(rule1)
        registry.add_rule(rule2)
        registry.add_rule(rule3)
        
        rules = registry.get_rules_by_priority()
        assert len(rules) == 3
        assert rules[0].id == "rule2"  # Priority 1
        assert rules[1].id == "rule3"  # Priority 2
        assert rules[2].id == "rule1"  # Priority 3
    
    def test_get_best_matching_rule(self):
        """Test conflict resolution with best matching rule."""
        registry = RuleRegistry()
        
        # Add rules with different priorities that could match the same text
        rule1 = Rule(id="rule1", name="Rule 1", description="Test", pattern="a", replacement="x", priority=2)
        rule2 = Rule(id="rule2", name="Rule 2", description="Test", pattern="a", replacement="y", priority=1)
        rule3 = Rule(id="rule3", name="Rule 3", description="Test", pattern="ab", replacement="z", priority=3)
        
        registry.add_rule(rule1)
        registry.add_rule(rule2)
        registry.add_rule(rule3)
        
        # Should return rule2 (priority 1) when matching "a"
        best_rule = registry.get_best_matching_rule("abc", 0)
        assert best_rule is not None
        assert best_rule.id == "rule2"
    
    def test_enable_disable_rules(self):
        """Test enabling and disabling rules."""
        registry = RuleRegistry()
        rule = Rule(id="test", name="Test", description="Test", pattern="a", replacement="b")
        registry.add_rule(rule)
        
        # Rule should be enabled by default
        assert rule.enabled is True
        
        # Disable rule
        assert registry.disable_rule("test") is True
        assert rule.enabled is False
        
        # Enable rule
        assert registry.enable_rule("test") is True
        assert rule.enabled is True
        
        # Test with non-existent rule
        assert registry.enable_rule("nonexistent") is False
        assert registry.disable_rule("nonexistent") is False
    
    def test_remove_rule(self):
        """Test removing rules."""
        registry = RuleRegistry()
        rule = Rule(id="test", name="Test", description="Test", pattern="a", replacement="b")
        registry.add_rule(rule)
        
        assert registry.get_rule_count() == 1
        assert registry.remove_rule("test") is True
        assert registry.get_rule_count() == 0
        assert registry.get_rule_by_id("test") is None
        
        # Test removing non-existent rule
        assert registry.remove_rule("nonexistent") is False
    
    def test_get_rules_by_category(self):
        """Test getting rules by category."""
        registry = RuleRegistry()
        
        rule1 = Rule(
            id="rule1", name="Rule 1", description="Test", pattern="a", replacement="b",
            metadata={"category": "vowel_sandhi"}
        )
        rule2 = Rule(
            id="rule2", name="Rule 2", description="Test", pattern="c", replacement="d",
            metadata={"category": "consonant_sandhi"}
        )
        rule3 = Rule(
            id="rule3", name="Rule 3", description="Test", pattern="e", replacement="f",
            metadata={"category": "vowel_sandhi"}
        )
        
        registry.add_rule(rule1)
        registry.add_rule(rule2)
        registry.add_rule(rule3)
        
        vowel_rules = registry.get_rules_by_category("vowel_sandhi")
        assert len(vowel_rules) == 2
        assert {rule.id for rule in vowel_rules} == {"rule1", "rule3"}
        
        consonant_rules = registry.get_rules_by_category("consonant_sandhi")
        assert len(consonant_rules) == 1
        assert consonant_rules[0].id == "rule2"
    
    def test_load_from_json(self):
        """Test loading rules from JSON file."""
        registry = RuleRegistry()
        
        # Create temporary JSON file
        rule_data = {
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": [
                {
                    "id": "test_rule_1",
                    "name": "Test Rule 1",
                    "description": "First test rule",
                    "pattern": "a",
                    "replacement": "b",
                    "priority": 1,
                    "enabled": True,
                    "metadata": {"category": "test"}
                },
                {
                    "id": "test_rule_2",
                    "name": "Test Rule 2",
                    "description": "Second test rule",
                    "pattern": "c",
                    "replacement": "d",
                    "priority": 2,
                    "enabled": False,
                    "metadata": {"category": "test"}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rule_data, f)
            temp_file = f.name
        
        try:
            registry.load_from_json(temp_file)
            
            assert registry.get_rule_count() == 2
            
            rule1 = registry.get_rule_by_id("test_rule_1")
            assert rule1 is not None
            assert rule1.name == "Test Rule 1"
            assert rule1.pattern == "a"
            assert rule1.replacement == "b"
            assert rule1.priority == 1
            assert rule1.enabled is True
            assert rule1.metadata["category"] == "test"
            
            rule2 = registry.get_rule_by_id("test_rule_2")
            assert rule2 is not None
            assert rule2.enabled is False
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_json_invalid_file(self):
        """Test loading from invalid JSON file."""
        registry = RuleRegistry()
        
        with pytest.raises(ValueError, match="Error loading rules"):
            registry.load_from_json("nonexistent_file.json")
    
    def test_load_from_json_duplicate_ids(self):
        """Test loading JSON with duplicate rule IDs."""
        registry = RuleRegistry()
        
        rule_data = {
            "rule_set": "test_rules",
            "version": "1.0",
            "rules": [
                {
                    "id": "duplicate_id",
                    "name": "Rule 1",
                    "description": "First rule",
                    "pattern": "a",
                    "replacement": "b"
                },
                {
                    "id": "duplicate_id",
                    "name": "Rule 2",
                    "description": "Second rule",
                    "pattern": "c",
                    "replacement": "d"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rule_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Duplicate rule ID 'duplicate_id'"):
                registry.load_from_json(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_clear(self):
        """Test clearing all rules."""
        registry = RuleRegistry()
        
        rule1 = Rule(id="rule1", name="Rule 1", description="Test", pattern="a", replacement="b")
        rule2 = Rule(id="rule2", name="Rule 2", description="Test", pattern="c", replacement="d")
        
        registry.add_rule(rule1)
        registry.add_rule(rule2)
        
        assert registry.get_rule_count() == 2
        
        registry.clear()
        
        assert registry.get_rule_count() == 0
        assert registry.get_rule_by_id("rule1") is None
        assert registry.get_rule_by_id("rule2") is None