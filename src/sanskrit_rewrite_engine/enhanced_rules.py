"""
Enhanced rule format system for complex linguistic rules.

This module provides an extensible rule format that supports complex linguistic
transformations, conditional logic, rule dependencies, and Pāṇini sūtra encoding.
"""

import logging
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path

from .interfaces import RuleType, ProcessingContext, AdvancedRule
from .future_architecture import (
    ExtensibleSutraRule, SutraReference, SutraCategory,
    AdvancedSanskritToken
)


logger = logging.getLogger(__name__)


class RuleConditionType(Enum):
    """Types of rule conditions."""
    PHONOLOGICAL = "phonological"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    POSITIONAL = "positional"
    FREQUENCY = "frequency"


class RuleOperator(Enum):
    """Operators for rule conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_SET = "in_set"
    NOT_IN_SET = "not_in_set"


@dataclass
class RuleCondition:
    """A condition that must be met for rule application."""
    condition_type: RuleConditionType
    operator: RuleOperator
    target_field: str
    expected_value: Any
    description: Optional[str] = None
    weight: float = 1.0  # Weight for condition evaluation
    
    def evaluate(self, token: Any, context: ProcessingContext) -> bool:
        """Evaluate the condition against a token and context."""
        try:
            # Get the actual value from token or context
            actual_value = self._get_target_value(token, context)
            
            # Apply the operator
            return self._apply_operator(actual_value, self.expected_value)
            
        except Exception as e:
            logger.error(f"Error evaluating condition {self.description}: {e}")
            return False
    
    def _get_target_value(self, token: Any, context: ProcessingContext) -> Any:
        """Get the target value from token or context."""
        if self.condition_type == RuleConditionType.PHONOLOGICAL:
            if hasattr(token, 'token_metadata'):
                return token.token_metadata.phonological_features.get(self.target_field)
            elif hasattr(token, 'phonetic_features'):
                return token.phonetic_features.get(self.target_field)
        
        elif self.condition_type == RuleConditionType.MORPHOLOGICAL:
            if hasattr(token, 'token_metadata'):
                return token.token_metadata.morphological_features.get(self.target_field)
            elif hasattr(token, 'morphological_features'):
                return token.morphological_features.get(self.target_field)
        
        elif self.condition_type == RuleConditionType.CONTEXTUAL:
            return context.metadata.get(self.target_field)
        
        elif self.condition_type == RuleConditionType.POSITIONAL:
            if self.target_field == 'position':
                return getattr(token, 'position', 0)
            elif self.target_field == 'start_pos':
                return getattr(token, 'start_pos', 0)
            elif self.target_field == 'end_pos':
                return getattr(token, 'end_pos', 0)
        
        # Default: try to get attribute directly
        return getattr(token, self.target_field, None)
    
    def _apply_operator(self, actual: Any, expected: Any) -> bool:
        """Apply the operator to compare actual and expected values."""
        if actual is None:
            return self.operator in [RuleOperator.NOT_EQUALS, RuleOperator.NOT_CONTAINS, RuleOperator.NOT_IN_SET]
        
        if self.operator == RuleOperator.EQUALS:
            return actual == expected
        elif self.operator == RuleOperator.NOT_EQUALS:
            return actual != expected
        elif self.operator == RuleOperator.CONTAINS:
            return str(expected) in str(actual)
        elif self.operator == RuleOperator.NOT_CONTAINS:
            return str(expected) not in str(actual)
        elif self.operator == RuleOperator.STARTS_WITH:
            return str(actual).startswith(str(expected))
        elif self.operator == RuleOperator.ENDS_WITH:
            return str(actual).endswith(str(expected))
        elif self.operator == RuleOperator.MATCHES_REGEX:
            return bool(re.search(str(expected), str(actual)))
        elif self.operator == RuleOperator.GREATER_THAN:
            return float(actual) > float(expected)
        elif self.operator == RuleOperator.LESS_THAN:
            return float(actual) < float(expected)
        elif self.operator == RuleOperator.IN_SET:
            return actual in expected
        elif self.operator == RuleOperator.NOT_IN_SET:
            return actual not in expected
        
        return False


@dataclass
class RuleAction:
    """An action to perform when a rule is applied."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def execute(self, token: Any, context: ProcessingContext) -> Any:
        """Execute the action on a token."""
        if self.action_type == "replace_text":
            return self._replace_text(token)
        elif self.action_type == "add_feature":
            return self._add_feature(token)
        elif self.action_type == "set_metadata":
            return self._set_metadata(token)
        elif self.action_type == "transform_phonetic":
            return self._transform_phonetic(token)
        elif self.action_type == "split_token":
            return self._split_token(token)
        elif self.action_type == "merge_tokens":
            return self._merge_tokens(token, context)
        else:
            logger.warning(f"Unknown action type: {self.action_type}")
            return token
    
    def _replace_text(self, token: Any) -> Any:
        """Replace text in token."""
        pattern = self.parameters.get('pattern', '')
        replacement = self.parameters.get('replacement', '')
        
        if hasattr(token, '_surface_form'):
            token._surface_form = re.sub(pattern, replacement, token._surface_form)
        elif hasattr(token, 'text'):
            token.text = re.sub(pattern, replacement, token.text)
        
        return token
    
    def _add_feature(self, token: Any) -> Any:
        """Add a linguistic feature to token."""
        feature_name = self.parameters.get('feature_name')
        feature_value = self.parameters.get('feature_value')
        feature_type = self.parameters.get('feature_type', 'morphological')
        
        if hasattr(token, 'token_metadata'):
            if feature_type == 'phonological':
                token.token_metadata.phonological_features[feature_name] = feature_value
            elif feature_type == 'morphological':
                token.token_metadata.morphological_features[feature_name] = feature_value
            elif feature_type == 'syntactic':
                token.token_metadata.syntactic_features[feature_name] = feature_value
            elif feature_type == 'semantic':
                token.token_metadata.semantic_features[feature_name] = feature_value
        
        return token
    
    def _set_metadata(self, token: Any) -> Any:
        """Set metadata on token."""
        metadata_key = self.parameters.get('key')
        metadata_value = self.parameters.get('value')
        
        if hasattr(token, 'metadata'):
            token.metadata[metadata_key] = metadata_value
        elif hasattr(token, '_metadata'):
            token._metadata[metadata_key] = metadata_value
        
        return token
    
    def _transform_phonetic(self, token: Any) -> Any:
        """Transform phonetic representation."""
        if hasattr(token, 'phonetic_representation'):
            transformation = self.parameters.get('transformation', {})
            current = token.phonetic_representation or token.surface_form
            
            for pattern, replacement in transformation.items():
                current = re.sub(pattern, replacement, current)
            
            token.phonetic_representation = current
        
        return token
    
    def _split_token(self, token: Any) -> List[Any]:
        """Split token into multiple tokens."""
        split_pattern = self.parameters.get('pattern', r'\s+')
        
        if hasattr(token, 'surface_form'):
            parts = re.split(split_pattern, token.surface_form)
            # Create new tokens for each part
            new_tokens = []
            for i, part in enumerate(parts):
                if part.strip():
                    new_token = AdvancedSanskritToken(part.strip())
                    new_token._position = token._position + i
                    new_tokens.append(new_token)
            return new_tokens
        
        return [token]
    
    def _merge_tokens(self, token: Any, context: ProcessingContext) -> Any:
        """Merge with adjacent tokens (placeholder)."""
        # This would require access to token sequence
        logger.info("Token merging not implemented in this context")
        return token


class ComplexLinguisticRule(ExtensibleSutraRule):
    """Complex linguistic rule with conditions, actions, and dependencies."""
    
    def __init__(self, rule_id: str, rule_type: RuleType, pattern: str, replacement: str):
        super().__init__(rule_id, rule_type, pattern, replacement)
        self.conditions: List[RuleCondition] = []
        self.actions: List[RuleAction] = []
        self.dependencies: List[str] = []  # Rule IDs that must be applied first
        self.conflicts: List[str] = []  # Rule IDs that conflict with this rule
        self.scope_restrictions: Dict[str, Any] = {}
        self.application_count: int = 0
        self.max_applications: Optional[int] = None
        self.success_rate: float = 0.0
        self.last_applied: Optional[str] = None  # Timestamp or context ID
    
    def add_condition(self, condition: RuleCondition) -> None:
        """Add a condition to the rule."""
        self.conditions.append(condition)
    
    def add_action(self, action: RuleAction) -> None:
        """Add an action to the rule."""
        self.actions.append(action)
    
    def set_dependencies(self, dependencies: List[str]) -> None:
        """Set rule dependencies."""
        self.dependencies = dependencies.copy()
    
    def set_conflicts(self, conflicts: List[str]) -> None:
        """Set conflicting rules."""
        self.conflicts = conflicts.copy()
    
    def matches(self, target: Any, context: ProcessingContext) -> bool:
        """Enhanced matching with complex conditions."""
        # Check basic pattern match first
        if not super().matches(target, context):
            return False
        
        # Check if max applications reached
        if self.max_applications and self.application_count >= self.max_applications:
            return False
        
        # Check dependencies
        applied_rules = context.metadata.get('applied_rules', [])
        for dep in self.dependencies:
            if dep not in applied_rules:
                return False
        
        # Check conflicts
        for conflict in self.conflicts:
            if conflict in applied_rules:
                return False
        
        # Check scope restrictions
        if self.scope_restrictions:
            for key, expected_value in self.scope_restrictions.items():
                actual_value = context.metadata.get(key)
                if actual_value != expected_value:
                    return False
        
        # Evaluate all conditions
        condition_results = []
        total_weight = 0.0
        
        for condition in self.conditions:
            result = condition.evaluate(target, context)
            condition_results.append(result)
            total_weight += condition.weight
        
        if not condition_results:
            return True  # No conditions means always match
        
        # Calculate weighted success rate
        success_weight = sum(
            condition.weight for condition, result in zip(self.conditions, condition_results)
            if result
        )
        
        success_rate = success_weight / total_weight if total_weight > 0 else 0.0
        
        # Require at least 70% of weighted conditions to pass
        return success_rate >= 0.7
    
    def apply(self, target: Any, context: ProcessingContext) -> Any:
        """Apply rule with complex actions."""
        # Apply basic transformation first
        result = super().apply(target, context)
        
        # Execute additional actions
        for action in self.actions:
            try:
                result = action.execute(result, context)
            except Exception as e:
                logger.error(f"Error executing action in rule {self.rule_id}: {e}")
        
        # Update application statistics
        self.application_count += 1
        self.last_applied = context.processing_id
        
        # Record rule application in context
        applied_rules = context.metadata.get('applied_rules', [])
        applied_rules.append(self.rule_id)
        context.metadata['applied_rules'] = applied_rules
        
        return result
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize complex rule to JSON."""
        base_data = super().to_json()
        
        # Add complex rule data
        base_data.update({
            'conditions': [
                {
                    'condition_type': cond.condition_type.value,
                    'operator': cond.operator.value,
                    'target_field': cond.target_field,
                    'expected_value': cond.expected_value,
                    'description': cond.description,
                    'weight': cond.weight
                }
                for cond in self.conditions
            ],
            'actions': [
                {
                    'action_type': action.action_type,
                    'parameters': action.parameters,
                    'description': action.description
                }
                for action in self.actions
            ],
            'dependencies': self.dependencies,
            'conflicts': self.conflicts,
            'scope_restrictions': self.scope_restrictions,
            'max_applications': self.max_applications,
            'application_statistics': {
                'application_count': self.application_count,
                'success_rate': self.success_rate,
                'last_applied': self.last_applied
            }
        })
        
        return base_data
    
    @classmethod
    def from_json(cls, rule_data: Dict[str, Any]) -> 'ComplexLinguisticRule':
        """Create complex rule from JSON data."""
        # Create base rule
        rule = cls(
            rule_id=rule_data['rule_id'],
            rule_type=RuleType(rule_data['rule_type']),
            pattern=rule_data['pattern'],
            replacement=rule_data['replacement']
        )
        
        # Load base properties
        rule._priority = rule_data.get('priority', 1)
        rule._enabled = rule_data.get('enabled', True)
        rule._metadata.update(rule_data.get('metadata', {}))
        
        # Load sūtra reference
        if 'sutra_reference' in rule_data:
            sutra_data = rule_data['sutra_reference']
            sutra_ref = SutraReference(
                sutra_number=sutra_data['sutra_number'],
                sutra_text=sutra_data['sutra_text'],
                category=SutraCategory(sutra_data['category']),
                translation=sutra_data.get('translation'),
                examples=sutra_data.get('examples', [])
            )
            rule.set_sutra_reference(sutra_ref)
        
        # Load conditions
        for cond_data in rule_data.get('conditions', []):
            condition = RuleCondition(
                condition_type=RuleConditionType(cond_data['condition_type']),
                operator=RuleOperator(cond_data['operator']),
                target_field=cond_data['target_field'],
                expected_value=cond_data['expected_value'],
                description=cond_data.get('description'),
                weight=cond_data.get('weight', 1.0)
            )
            rule.add_condition(condition)
        
        # Load actions
        for action_data in rule_data.get('actions', []):
            action = RuleAction(
                action_type=action_data['action_type'],
                parameters=action_data.get('parameters', {}),
                description=action_data.get('description')
            )
            rule.add_action(action)
        
        # Load dependencies and conflicts
        rule.dependencies = rule_data.get('dependencies', [])
        rule.conflicts = rule_data.get('conflicts', [])
        rule.scope_restrictions = rule_data.get('scope_restrictions', {})
        rule.max_applications = rule_data.get('max_applications')
        
        # Load statistics
        stats = rule_data.get('application_statistics', {})
        rule.application_count = stats.get('application_count', 0)
        rule.success_rate = stats.get('success_rate', 0.0)
        rule.last_applied = stats.get('last_applied')
        
        return rule


class RuleSetManager:
    """Manager for complex rule sets with dependencies and validation."""
    
    def __init__(self):
        self.rules: Dict[str, ComplexLinguisticRule] = {}
        self.rule_sets: Dict[str, List[str]] = {}  # Named collections of rule IDs
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.conflict_graph: Dict[str, Set[str]] = {}
    
    def add_rule(self, rule: ComplexLinguisticRule) -> None:
        """Add a rule to the manager."""
        self.rules[rule.rule_id] = rule
        
        # Update dependency graph
        self.dependency_graph[rule.rule_id] = set(rule.dependencies)
        
        # Update conflict graph
        self.conflict_graph[rule.rule_id] = set(rule.conflicts)
        
        logger.info(f"Added rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the manager."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            
            # Clean up graphs
            if rule_id in self.dependency_graph:
                del self.dependency_graph[rule_id]
            if rule_id in self.conflict_graph:
                del self.conflict_graph[rule_id]
            
            # Remove from other rules' dependencies and conflicts
            for other_rule in self.rules.values():
                if rule_id in other_rule.dependencies:
                    other_rule.dependencies.remove(rule_id)
                if rule_id in other_rule.conflicts:
                    other_rule.conflicts.remove(rule_id)
            
            logger.info(f"Removed rule: {rule_id}")
            return True
        
        return False
    
    def create_rule_set(self, name: str, rule_ids: List[str]) -> bool:
        """Create a named rule set."""
        # Validate that all rules exist
        for rule_id in rule_ids:
            if rule_id not in self.rules:
                logger.error(f"Rule not found: {rule_id}")
                return False
        
        self.rule_sets[name] = rule_ids.copy()
        logger.info(f"Created rule set: {name} with {len(rule_ids)} rules")
        return True
    
    def get_rule_set(self, name: str) -> List[ComplexLinguisticRule]:
        """Get rules in a named rule set."""
        rule_ids = self.rule_sets.get(name, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    def validate_dependencies(self) -> List[str]:
        """Validate rule dependencies and detect cycles."""
        issues = []
        
        # Check for missing dependencies
        for rule_id, rule in self.rules.items():
            for dep in rule.dependencies:
                if dep not in self.rules:
                    issues.append(f"Rule {rule_id} depends on missing rule: {dep}")
        
        # Check for circular dependencies
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for rule_id in self.rules:
            if rule_id not in visited:
                if has_cycle(rule_id, visited, set()):
                    issues.append(f"Circular dependency detected involving rule: {rule_id}")
        
        return issues
    
    def get_application_order(self, rule_ids: List[str]) -> List[str]:
        """Get optimal application order for a set of rules."""
        # Topological sort based on dependencies
        in_degree = {rule_id: 0 for rule_id in rule_ids}
        
        # Calculate in-degrees
        for rule_id in rule_ids:
            for dep in self.dependency_graph.get(rule_id, []):
                if dep in in_degree:
                    in_degree[rule_id] += 1
        
        # Kahn's algorithm
        queue = [rule_id for rule_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent rules
            for rule_id in rule_ids:
                if current in self.dependency_graph.get(rule_id, []):
                    in_degree[rule_id] -= 1
                    if in_degree[rule_id] == 0:
                        queue.append(rule_id)
        
        return result
    
    def load_from_json(self, file_path: str) -> None:
        """Load rules from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load individual rules
            for rule_data in data.get('rules', []):
                rule = ComplexLinguisticRule.from_json(rule_data)
                self.add_rule(rule)
            
            # Load rule sets
            for set_name, rule_ids in data.get('rule_sets', {}).items():
                self.create_rule_set(set_name, rule_ids)
            
            logger.info(f"Loaded {len(self.rules)} rules from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load rules from {file_path}: {e}")
    
    def save_to_json(self, file_path: str) -> None:
        """Save rules to JSON file."""
        try:
            data = {
                'rules': [rule.to_json() for rule in self.rules.values()],
                'rule_sets': self.rule_sets
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.rules)} rules to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save rules to {file_path}: {e}")


# Factory functions for creating complex rules

def create_sandhi_rule(rule_id: str, pattern: str, replacement: str,
                      phonological_conditions: Optional[List[Dict[str, Any]]] = None) -> ComplexLinguisticRule:
    """Create a sandhi rule with phonological conditions."""
    rule = ComplexLinguisticRule(rule_id, RuleType.SANDHI, pattern, replacement)
    
    # Add default phonological conditions
    if phonological_conditions:
        for cond_data in phonological_conditions:
            condition = RuleCondition(
                condition_type=RuleConditionType.PHONOLOGICAL,
                operator=RuleOperator(cond_data.get('operator', 'equals')),
                target_field=cond_data['field'],
                expected_value=cond_data['value'],
                description=cond_data.get('description')
            )
            rule.add_condition(condition)
    
    return rule


def create_morphological_rule(rule_id: str, pattern: str, replacement: str,
                            morphological_features: Optional[Dict[str, Any]] = None) -> ComplexLinguisticRule:
    """Create a morphological rule with feature additions."""
    rule = ComplexLinguisticRule(rule_id, RuleType.MORPHOLOGICAL, pattern, replacement)
    
    # Add feature-setting actions
    if morphological_features:
        for feature_name, feature_value in morphological_features.items():
            action = RuleAction(
                action_type="add_feature",
                parameters={
                    'feature_name': feature_name,
                    'feature_value': feature_value,
                    'feature_type': 'morphological'
                },
                description=f"Set {feature_name} to {feature_value}"
            )
            rule.add_action(action)
    
    return rule


def create_compound_rule(rule_id: str, pattern: str, replacement: str,
                        compound_type: str) -> ComplexLinguisticRule:
    """Create a compound formation rule."""
    rule = ComplexLinguisticRule(rule_id, RuleType.COMPOUND, pattern, replacement)
    
    # Add compound-specific action
    action = RuleAction(
        action_type="add_feature",
        parameters={
            'feature_name': 'compound_type',
            'feature_value': compound_type,
            'feature_type': 'morphological'
        },
        description=f"Mark as {compound_type} compound"
    )
    rule.add_action(action)
    
    return rule