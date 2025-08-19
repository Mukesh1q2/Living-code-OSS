"""
Extensible rule format for future Pāṇini sūtra encoding and complex linguistic rules.

This module provides advanced rule representations that can evolve to support
sophisticated Sanskrit grammatical rules including sūtra references, complex
conditions, and hierarchical rule relationships.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging

from .interfaces import (
    AdvancedRule, RuleType, ProcessingContext, LinguisticFeature,
    AdvancedToken, ProcessingStage
)


logger = logging.getLogger(__name__)


class RuleConditionType(Enum):
    """Types of rule conditions."""
    PATTERN_MATCH = "pattern_match"
    CONTEXT_DEPENDENT = "context_dependent"
    FEATURE_BASED = "feature_based"
    PHONOLOGICAL = "phonological"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    POSITIONAL = "positional"
    FREQUENCY_BASED = "frequency_based"


class RuleScope(Enum):
    """Scope of rule application."""
    LOCAL = "local"  # Single token/position
    CONTEXTUAL = "contextual"  # Surrounding context
    GLOBAL = "global"  # Entire text
    HIERARCHICAL = "hierarchical"  # Rule hierarchy


class SutraCategory(Enum):
    """Categories of Pāṇini sūtras for future encoding."""
    SAMJNA = "samjna"  # Definition rules
    PARIBHASHA = "paribhasha"  # Meta-rules
    VIDHI = "vidhi"  # Operational rules
    NIYAMA = "niyama"  # Restrictive rules
    ATIDESA = "atidesa"  # Extension rules
    APAVADA = "apavada"  # Exception rules


@dataclass
class RuleCondition:
    """Represents a condition for rule application."""
    condition_type: RuleConditionType
    expression: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False
    weight: float = 1.0
    description: str = ""


@dataclass
class SutraReference:
    """Reference to a Pāṇini sūtra."""
    sutra_number: str  # e.g., "1.1.1"
    sutra_text: str  # Original sūtra text
    category: SutraCategory
    adhyaya: int  # Chapter
    pada: int  # Section
    sutra_index: int  # Sūtra number within pada
    translation: Optional[str] = None
    commentary: Optional[str] = None
    related_sutras: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleHierarchy:
    """Represents hierarchical relationships between rules."""
    parent_rules: List[str] = field(default_factory=list)
    child_rules: List[str] = field(default_factory=list)
    sibling_rules: List[str] = field(default_factory=list)
    conflicting_rules: List[str] = field(default_factory=list)
    prerequisite_rules: List[str] = field(default_factory=list)
    blocking_rules: List[str] = field(default_factory=list)


class ExtensibleRule(AdvancedRule):
    """Advanced rule implementation with extensible features."""
    
    def __init__(self, rule_id: str, rule_type: RuleType, pattern: str, replacement: str):
        self._rule_id = rule_id
        self._rule_type = rule_type
        self._pattern = pattern
        self._replacement = replacement
        self._sutra_reference: Optional[SutraReference] = None
        self._conditions: List[RuleCondition] = []
        self._hierarchy: RuleHierarchy = RuleHierarchy()
        self._scope = RuleScope.LOCAL
        self._priority = 1
        self._enabled = True
        self._metadata: Dict[str, Any] = {}
        self._application_count = 0
        self._success_rate = 1.0
        self._context_requirements: Dict[str, Any] = {}
        self._feature_requirements: List[str] = []
        self._compiled_pattern: Optional[re.Pattern] = None
        self._custom_matcher: Optional[Callable] = None
        self._custom_applicator: Optional[Callable] = None
    
    @property
    def rule_id(self) -> str:
        return self._rule_id
    
    @property
    def rule_type(self) -> RuleType:
        return self._rule_type
    
    @property
    def sutra_reference(self) -> Optional[str]:
        if self._sutra_reference:
            return self._sutra_reference.sutra_number
        return None
    
    def get_sutra_reference(self) -> Optional[SutraReference]:
        """Get full sūtra reference object."""
        return self._sutra_reference
    
    def set_sutra_reference(self, sutra_ref: SutraReference) -> None:
        """Set sūtra reference."""
        self._sutra_reference = sutra_ref
    
    def add_condition(self, condition: RuleCondition) -> None:
        """Add a condition for rule application."""
        self._conditions.append(condition)
    
    def set_hierarchy(self, hierarchy: RuleHierarchy) -> None:
        """Set rule hierarchy relationships."""
        self._hierarchy = hierarchy
    
    def set_scope(self, scope: RuleScope) -> None:
        """Set rule application scope."""
        self._scope = scope
    
    def set_custom_matcher(self, matcher: Callable[[Any, ProcessingContext], bool]) -> None:
        """Set custom matching function."""
        self._custom_matcher = matcher
    
    def set_custom_applicator(self, applicator: Callable[[Any, ProcessingContext], Any]) -> None:
        """Set custom application function."""
        self._custom_applicator = applicator
    
    def matches(self, target: Any, context: ProcessingContext) -> bool:
        """Check if rule matches target in context."""
        if not self._enabled:
            return False
        
        # Use custom matcher if available
        if self._custom_matcher:
            try:
                return self._custom_matcher(target, context)
            except Exception as e:
                logger.error(f"Custom matcher failed for rule {self._rule_id}: {e}")
                return False
        
        # Check basic pattern match
        if not self._basic_pattern_match(target):
            return False
        
        # Check conditions
        if not self._check_conditions(target, context):
            return False
        
        # Check context requirements
        if not self._check_context_requirements(context):
            return False
        
        # Check feature requirements
        if not self._check_feature_requirements(target):
            return False
        
        return True
    
    def apply(self, target: Any, context: ProcessingContext) -> Any:
        """Apply rule to target."""
        if not self.matches(target, context):
            return target
        
        try:
            # Use custom applicator if available
            if self._custom_applicator:
                result = self._custom_applicator(target, context)
            else:
                result = self._basic_apply(target, context)
            
            # Update statistics
            self._application_count += 1
            
            # Add application metadata
            if hasattr(result, 'metadata'):
                if 'rule_applications' not in result.metadata:
                    result.metadata['rule_applications'] = []
                result.metadata['rule_applications'].append({
                    'rule_id': self._rule_id,
                    'rule_type': self._rule_type.value,
                    'sutra_reference': self.sutra_reference,
                    'context_stage': context.stage.value
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Rule application failed for {self._rule_id}: {e}")
            return target
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get rule metadata including sūtra references."""
        metadata = self._metadata.copy()
        metadata.update({
            'rule_id': self._rule_id,
            'rule_type': self._rule_type.value,
            'pattern': self._pattern,
            'replacement': self._replacement,
            'priority': self._priority,
            'enabled': self._enabled,
            'scope': self._scope.value,
            'application_count': self._application_count,
            'success_rate': self._success_rate,
            'conditions_count': len(self._conditions),
            'feature_requirements': self._feature_requirements.copy()
        })
        
        if self._sutra_reference:
            metadata['sutra_reference'] = {
                'sutra_number': self._sutra_reference.sutra_number,
                'sutra_text': self._sutra_reference.sutra_text,
                'category': self._sutra_reference.category.value,
                'translation': self._sutra_reference.translation
            }
        
        if self._hierarchy.parent_rules or self._hierarchy.child_rules:
            metadata['hierarchy'] = {
                'parent_rules': self._hierarchy.parent_rules,
                'child_rules': self._hierarchy.child_rules,
                'sibling_rules': self._hierarchy.sibling_rules,
                'conflicting_rules': self._hierarchy.conflicting_rules
            }
        
        return metadata
    
    def _basic_pattern_match(self, target: Any) -> bool:
        """Basic pattern matching."""
        if hasattr(target, 'surface_form'):
            text = target.surface_form
        elif hasattr(target, 'text'):
            text = target.text
        else:
            text = str(target)
        
        try:
            if not self._compiled_pattern:
                self._compiled_pattern = re.compile(self._pattern)
            return bool(self._compiled_pattern.search(text))
        except re.error:
            # Fallback to simple string matching
            return self._pattern in text
    
    def _check_conditions(self, target: Any, context: ProcessingContext) -> bool:
        """Check all rule conditions."""
        for condition in self._conditions:
            if not self._evaluate_condition(condition, target, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: RuleCondition, target: Any, context: ProcessingContext) -> bool:
        """Evaluate a single condition."""
        try:
            result = False
            
            if condition.condition_type == RuleConditionType.PATTERN_MATCH:
                result = self._evaluate_pattern_condition(condition, target)
            elif condition.condition_type == RuleConditionType.CONTEXT_DEPENDENT:
                result = self._evaluate_context_condition(condition, context)
            elif condition.condition_type == RuleConditionType.FEATURE_BASED:
                result = self._evaluate_feature_condition(condition, target)
            elif condition.condition_type == RuleConditionType.PHONOLOGICAL:
                result = self._evaluate_phonological_condition(condition, target)
            elif condition.condition_type == RuleConditionType.MORPHOLOGICAL:
                result = self._evaluate_morphological_condition(condition, target)
            elif condition.condition_type == RuleConditionType.POSITIONAL:
                result = self._evaluate_positional_condition(condition, target, context)
            elif condition.condition_type == RuleConditionType.FREQUENCY_BASED:
                result = self._evaluate_frequency_condition(condition, context)
            
            return result if not condition.negated else not result
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _evaluate_pattern_condition(self, condition: RuleCondition, target: Any) -> bool:
        """Evaluate pattern-based condition."""
        text = getattr(target, 'surface_form', getattr(target, 'text', str(target)))
        try:
            return bool(re.search(condition.expression, text))
        except re.error:
            return condition.expression in text
    
    def _evaluate_context_condition(self, condition: RuleCondition, context: ProcessingContext) -> bool:
        """Evaluate context-dependent condition."""
        # Check processing stage
        if 'stage' in condition.parameters:
            required_stage = ProcessingStage(condition.parameters['stage'])
            if context.stage != required_stage:
                return False
        
        # Check context features
        if 'required_features' in condition.parameters:
            for feature_name in condition.parameters['required_features']:
                if feature_name not in context.features:
                    return False
        
        return True
    
    def _evaluate_feature_condition(self, condition: RuleCondition, target: Any) -> bool:
        """Evaluate feature-based condition."""
        if not hasattr(target, 'features'):
            return False
        
        features = target.features
        feature_name = condition.parameters.get('feature_name')
        expected_value = condition.parameters.get('expected_value')
        
        if feature_name not in features:
            return False
        
        if expected_value is not None:
            feature = features[feature_name]
            return feature.value == expected_value
        
        return True
    
    def _evaluate_phonological_condition(self, condition: RuleCondition, target: Any) -> bool:
        """Evaluate phonological condition."""
        # Implement phonological analysis
        # This would check vowel/consonant patterns, syllable structure, etc.
        return True  # Placeholder
    
    def _evaluate_morphological_condition(self, condition: RuleCondition, target: Any) -> bool:
        """Evaluate morphological condition."""
        # Implement morphological analysis
        # This would check case, gender, number, etc.
        return True  # Placeholder
    
    def _evaluate_positional_condition(self, condition: RuleCondition, target: Any, context: ProcessingContext) -> bool:
        """Evaluate positional condition."""
        # Check position in text, sentence, etc.
        return True  # Placeholder
    
    def _evaluate_frequency_condition(self, condition: RuleCondition, context: ProcessingContext) -> bool:
        """Evaluate frequency-based condition."""
        # Check application frequency, success rate, etc.
        max_applications = condition.parameters.get('max_applications', float('inf'))
        return self._application_count < max_applications
    
    def _check_context_requirements(self, context: ProcessingContext) -> bool:
        """Check context requirements."""
        for req_key, req_value in self._context_requirements.items():
            if req_key == 'stage':
                if context.stage != ProcessingStage(req_value):
                    return False
            elif req_key == 'analysis_level':
                if context.analysis_level.value != req_value:
                    return False
        
        return True
    
    def _check_feature_requirements(self, target: Any) -> bool:
        """Check feature requirements."""
        if not self._feature_requirements:
            return True
        
        if not hasattr(target, 'features'):
            return False
        
        features = target.features
        for required_feature in self._feature_requirements:
            if required_feature not in features:
                return False
        
        return True
    
    def _basic_apply(self, target: Any, context: ProcessingContext) -> Any:
        """Basic rule application."""
        if hasattr(target, 'surface_form'):
            text = target.surface_form
            try:
                if not self._compiled_pattern:
                    self._compiled_pattern = re.compile(self._pattern)
                new_text = self._compiled_pattern.sub(self._replacement, text)
                target._surface_form = new_text
            except (re.error, AttributeError):
                # Fallback to simple replacement
                new_text = text.replace(self._pattern, self._replacement)
                if hasattr(target, '_surface_form'):
                    target._surface_form = new_text
        
        return target


class ExtensibleRuleRegistry:
    """Registry for extensible rules with advanced features."""
    
    def __init__(self):
        self._rules: Dict[str, ExtensibleRule] = {}
        self._rules_by_type: Dict[RuleType, List[str]] = {}
        self._rules_by_sutra: Dict[str, List[str]] = {}
        self._rule_hierarchy: Dict[str, RuleHierarchy] = {}
        self._sutra_database: Dict[str, SutraReference] = {}
    
    def register_rule(self, rule: ExtensibleRule) -> None:
        """Register an extensible rule."""
        self._rules[rule.rule_id] = rule
        
        # Index by type
        if rule.rule_type not in self._rules_by_type:
            self._rules_by_type[rule.rule_type] = []
        self._rules_by_type[rule.rule_type].append(rule.rule_id)
        
        # Index by sūtra reference
        if rule.sutra_reference:
            if rule.sutra_reference not in self._rules_by_sutra:
                self._rules_by_sutra[rule.sutra_reference] = []
            self._rules_by_sutra[rule.sutra_reference].append(rule.rule_id)
        
        logger.info(f"Registered extensible rule: {rule.rule_id}")
    
    def get_rule(self, rule_id: str) -> Optional[ExtensibleRule]:
        """Get rule by ID."""
        return self._rules.get(rule_id)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[ExtensibleRule]:
        """Get rules by type."""
        rule_ids = self._rules_by_type.get(rule_type, [])
        return [self._rules[rule_id] for rule_id in rule_ids if rule_id in self._rules]
    
    def get_rules_by_sutra(self, sutra_number: str) -> List[ExtensibleRule]:
        """Get rules by sūtra reference."""
        rule_ids = self._rules_by_sutra.get(sutra_number, [])
        return [self._rules[rule_id] for rule_id in rule_ids if rule_id in self._rules]
    
    def add_sutra_reference(self, sutra_ref: SutraReference) -> None:
        """Add sūtra reference to database."""
        self._sutra_database[sutra_ref.sutra_number] = sutra_ref
    
    def get_sutra_reference(self, sutra_number: str) -> Optional[SutraReference]:
        """Get sūtra reference by number."""
        return self._sutra_database.get(sutra_number)
    
    def load_from_extended_json(self, file_path: str) -> None:
        """Load rules from extended JSON format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load sūtra references if present
            if 'sutra_references' in data:
                for sutra_data in data['sutra_references']:
                    sutra_ref = SutraReference(
                        sutra_number=sutra_data['sutra_number'],
                        sutra_text=sutra_data['sutra_text'],
                        category=SutraCategory(sutra_data.get('category', 'vidhi')),
                        adhyaya=sutra_data.get('adhyaya', 1),
                        pada=sutra_data.get('pada', 1),
                        sutra_index=sutra_data.get('sutra_index', 1),
                        translation=sutra_data.get('translation'),
                        commentary=sutra_data.get('commentary'),
                        related_sutras=sutra_data.get('related_sutras', []),
                        metadata=sutra_data.get('metadata', {})
                    )
                    self.add_sutra_reference(sutra_ref)
            
            # Load rules
            if 'rules' in data:
                for rule_data in data['rules']:
                    rule = self._create_rule_from_data(rule_data)
                    if rule:
                        self.register_rule(rule)
            
            logger.info(f"Loaded extensible rules from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load extensible rules from {file_path}: {e}")
            raise
    
    def _create_rule_from_data(self, rule_data: Dict[str, Any]) -> Optional[ExtensibleRule]:
        """Create rule from JSON data."""
        try:
            rule = ExtensibleRule(
                rule_id=rule_data['id'],
                rule_type=RuleType(rule_data.get('type', 'phonological')),
                pattern=rule_data['pattern'],
                replacement=rule_data['replacement']
            )
            
            # Set basic properties
            rule._priority = rule_data.get('priority', 1)
            rule._enabled = rule_data.get('enabled', True)
            rule._scope = RuleScope(rule_data.get('scope', 'local'))
            
            # Set sūtra reference
            if 'sutra_reference' in rule_data:
                sutra_number = rule_data['sutra_reference']
                sutra_ref = self.get_sutra_reference(sutra_number)
                if sutra_ref:
                    rule.set_sutra_reference(sutra_ref)
            
            # Add conditions
            if 'conditions' in rule_data:
                for cond_data in rule_data['conditions']:
                    condition = RuleCondition(
                        condition_type=RuleConditionType(cond_data['type']),
                        expression=cond_data['expression'],
                        parameters=cond_data.get('parameters', {}),
                        negated=cond_data.get('negated', False),
                        weight=cond_data.get('weight', 1.0),
                        description=cond_data.get('description', '')
                    )
                    rule.add_condition(condition)
            
            # Set hierarchy
            if 'hierarchy' in rule_data:
                hierarchy = RuleHierarchy(
                    parent_rules=rule_data['hierarchy'].get('parent_rules', []),
                    child_rules=rule_data['hierarchy'].get('child_rules', []),
                    sibling_rules=rule_data['hierarchy'].get('sibling_rules', []),
                    conflicting_rules=rule_data['hierarchy'].get('conflicting_rules', []),
                    prerequisite_rules=rule_data['hierarchy'].get('prerequisite_rules', []),
                    blocking_rules=rule_data['hierarchy'].get('blocking_rules', [])
                )
                rule.set_hierarchy(hierarchy)
            
            # Set context requirements
            if 'context_requirements' in rule_data:
                rule._context_requirements = rule_data['context_requirements']
            
            # Set feature requirements
            if 'feature_requirements' in rule_data:
                rule._feature_requirements = rule_data['feature_requirements']
            
            # Set metadata
            if 'metadata' in rule_data:
                rule._metadata = rule_data['metadata']
            
            return rule
            
        except Exception as e:
            logger.error(f"Failed to create rule from data: {e}")
            return None


# Utility functions for creating extensible rules

def create_sandhi_rule(rule_id: str, pattern: str, replacement: str, 
                      sutra_reference: Optional[str] = None) -> ExtensibleRule:
    """Create a sandhi rule with appropriate settings."""
    rule = ExtensibleRule(rule_id, RuleType.SANDHI, pattern, replacement)
    rule.set_scope(RuleScope.CONTEXTUAL)
    
    if sutra_reference:
        # This would be populated from a sūtra database
        sutra_ref = SutraReference(
            sutra_number=sutra_reference,
            sutra_text="",  # Would be filled from database
            category=SutraCategory.VIDHI,
            adhyaya=1, pada=1, sutra_index=1
        )
        rule.set_sutra_reference(sutra_ref)
    
    return rule


def create_morphological_rule(rule_id: str, pattern: str, replacement: str,
                            feature_requirements: List[str] = None) -> ExtensibleRule:
    """Create a morphological rule with feature requirements."""
    rule = ExtensibleRule(rule_id, RuleType.MORPHOLOGICAL, pattern, replacement)
    rule.set_scope(RuleScope.LOCAL)
    
    if feature_requirements:
        rule._feature_requirements = feature_requirements
    
    return rule


def create_compound_rule(rule_id: str, pattern: str, replacement: str) -> ExtensibleRule:
    """Create a compound formation rule."""
    rule = ExtensibleRule(rule_id, RuleType.COMPOUND, pattern, replacement)
    rule.set_scope(RuleScope.CONTEXTUAL)
    
    # Add condition for compound markers
    condition = RuleCondition(
        condition_type=RuleConditionType.PATTERN_MATCH,
        expression=r'[+\-_]',
        description="Requires compound markers"
    )
    rule.add_condition(condition)
    
    return rule