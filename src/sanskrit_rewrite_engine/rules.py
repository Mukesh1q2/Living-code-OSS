"""
Rule definition and management system.

This module provides the rule system for Sanskrit transformations.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Match


@dataclass
class Rule:
    """Represents a transformation rule."""
    id: str
    name: str
    description: str
    pattern: str          # Regex or simple pattern to match
    replacement: str      # Replacement text or template
    priority: int = 1     # Lower numbers = higher priority
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, text: str, position: int) -> bool:
        """Check if rule matches at given position.
        
        Args:
            text: Text to check
            position: Position in text
            
        Returns:
            True if rule matches at position
        """
        try:
            # Check if pattern matches at the given position
            match = re.search(self.pattern, text[position:])
            return match is not None and match.start() == 0
        except re.error:
            # If regex is invalid, fall back to simple string matching
            return text[position:].startswith(self.pattern)
        
    def _convert_replacement_syntax(self, replacement: str) -> str:
        """Convert $1, $2 style replacements to Python's \1, \2 style.
        
        Args:
            replacement: Replacement string potentially with $1, $2 syntax
            
        Returns:
            Replacement string with Python-compatible syntax
        """
        # Convert $1, $2, etc. to \1, \2, etc.
        import re as regex_module
        return regex_module.sub(r'\$(\d+)', r'\\\1', replacement)
    
    def apply(self, text: str, position: int) -> Tuple[str, int]:
        """Apply rule and return new text and next position.
        
        Args:
            text: Text to transform
            position: Current position
            
        Returns:
            Tuple of (new_text, next_position)
        """
        try:
            # Apply regex substitution from the given position
            remaining_text = text[position:]
            match = re.search(self.pattern, remaining_text)
            
            if match and match.start() == 0:
                # Convert replacement syntax if needed
                python_replacement = self._convert_replacement_syntax(self.replacement)
                
                # Replace the matched portion using re.sub to handle capture groups
                new_remaining = re.sub(self.pattern, python_replacement, remaining_text, count=1)
                new_text = text[:position] + new_remaining
                
                # Calculate new position after replacement
                # We need to find the actual length of the replacement text
                replacement_text = re.sub(self.pattern, python_replacement, match.group(0))
                replacement_length = len(replacement_text)
                new_position = position + replacement_length
                
                return new_text, new_position
            else:
                # No match, move to next position
                return text, position + 1
                
        except re.error:
            # If regex is invalid, fall back to simple string replacement
            if text[position:].startswith(self.pattern):
                new_text = text[:position] + self.replacement + text[position + len(self.pattern):]
                new_position = position + len(self.replacement)
                return new_text, new_position
            else:
                return text, position + 1


class RuleRegistry:
    """Registry for managing transformation rules."""
    
    def __init__(self):
        """Initialize empty rule registry."""
        self._rules: List[Rule] = []
        self._rules_by_id: Dict[str, Rule] = {}
        
    def load_from_json(self, file_path: str) -> None:
        """Load rules from JSON configuration.
        
        Args:
            file_path: Path to JSON rule file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'rules' in data:
                for rule_data in data['rules']:
                    rule = Rule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        description=rule_data['description'],
                        pattern=rule_data['pattern'],
                        replacement=rule_data['replacement'],
                        priority=rule_data.get('priority', 1),
                        enabled=rule_data.get('enabled', True),
                        metadata=rule_data.get('metadata', {})
                    )
                    if rule.id in self._rules_by_id:
                        raise ValueError(f"Duplicate rule ID '{rule.id}' in {file_path}")
                    self._rules.append(rule)
                    self._rules_by_id[rule.id] = rule
                    
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error loading rules from {file_path}: {e}")
        
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the registry.
        
        Args:
            rule: Rule to add
            
        Raises:
            ValueError: If rule with same ID already exists
        """
        if rule.id in self._rules_by_id:
            raise ValueError(f"Rule with ID '{rule.id}' already exists")
        
        self._rules.append(rule)
        self._rules_by_id[rule.id] = rule
        
    def get_applicable_rules(self, text: str, position: int) -> List[Rule]:
        """Get rules that can apply at the given position.
        
        Args:
            text: Text to check
            position: Position in text
            
        Returns:
            List of applicable rules
        """
        applicable = []
        for rule in self._rules:
            if rule.enabled and rule.matches(text, position):
                applicable.append(rule)
        return applicable
        
    def get_rules_by_priority(self) -> List[Rule]:
        """Get all enabled rules sorted by priority.
        
        Returns:
            List of rules sorted by priority (lower number = higher priority)
        """
        enabled_rules = [rule for rule in self._rules if rule.enabled]
        return sorted(enabled_rules, key=lambda r: r.priority)
        
    def clear(self) -> None:
        """Clear all rules from registry."""
        self._rules.clear()
        self._rules_by_id.clear()
        
    def get_rule_count(self) -> int:
        """Get total number of rules.
        
        Returns:
            Number of rules in registry
        """
        return len(self._rules)
    
    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by its ID.
        
        Args:
            rule_id: ID of the rule to retrieve
            
        Returns:
            Rule if found, None otherwise
        """
        return self._rules_by_id.get(rule_id)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by its ID.
        
        Args:
            rule_id: ID of the rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        if rule_id in self._rules_by_id:
            rule = self._rules_by_id[rule_id]
            self._rules.remove(rule)
            del self._rules_by_id[rule_id]
            return True
        return False
    
    def get_best_matching_rule(self, text: str, position: int) -> Optional[Rule]:
        """Get the highest priority rule that matches at the given position.
        
        Uses conflict resolution: returns the rule with highest priority (lowest number)
        that matches at the position. If multiple rules have the same priority,
        returns the first one found.
        
        Args:
            text: Text to check
            position: Position in text
            
        Returns:
            Best matching rule or None if no rules match
        """
        applicable_rules = self.get_applicable_rules(text, position)
        if not applicable_rules:
            return None
            
        # Sort by priority (lower number = higher priority)
        applicable_rules.sort(key=lambda r: r.priority)
        return applicable_rules[0]
    
    def get_rules_by_category(self, category: str) -> List[Rule]:
        """Get all rules in a specific category.
        
        Args:
            category: Category name to filter by
            
        Returns:
            List of rules in the specified category
        """
        return [
            rule for rule in self._rules 
            if rule.enabled and rule.metadata.get('category') == category
        ]
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule by its ID.
        
        Args:
            rule_id: ID of the rule to enable
            
        Returns:
            True if rule was found and enabled, False otherwise
        """
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by its ID.
        
        Args:
            rule_id: ID of the rule to disable
            
        Returns:
            True if rule was found and disabled, False otherwise
        """
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False