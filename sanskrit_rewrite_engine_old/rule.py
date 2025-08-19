"""
Pāṇini rule system for Sanskrit grammatical transformations.

This module implements the core rule engine with support for:
- SutraRule dataclass with adhikāra (domain) and anuvṛtti (inheritance)
- Guarded rule application with loop prevention
- Sūtra numbering system and cross-references
- Paribhāṣā rules for cross-domain effects
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable, Tuple, Any
from enum import Enum
from .sanskrit_token import Token


class RuleType(Enum):
    """Types of Pāṇini rules."""
    SUTRA = "SUTRA"           # Regular sūtra rules
    PARIBHASA = "PARIBHASA"   # Meta-rules that control other rules
    ADHIKARA = "ADHIKARA"     # Domain rules that establish scope
    ANUVRTI = "ANUVRTI"       # Rules with inherited scope


@dataclass(frozen=True)
class SutraReference:
    """Reference to a Pāṇini sūtra with traditional numbering."""
    adhyaya: int      # Chapter (1-8)
    pada: int         # Quarter (1-4)
    sutra: int        # Sūtra number within pada
    
    def __str__(self) -> str:
        return f"{self.adhyaya}.{self.pada}.{self.sutra}"
    
    def __lt__(self, other: 'SutraReference') -> bool:
        """Compare sūtra references for ordering."""
        return (self.adhyaya, self.pada, self.sutra) < (other.adhyaya, other.pada, other.sutra)
    
    def __hash__(self) -> int:
        """Make SutraReference hashable for use in sets and dicts."""
        return hash((self.adhyaya, self.pada, self.sutra))


@dataclass
class SutraRule:
    """
    A Pāṇini sūtra rule with traditional grammatical metadata.
    
    Attributes:
        sutra_ref: Traditional sūtra reference (e.g., 1.1.1)
        name: Sanskrit name of the sūtra
        description: English description
        rule_type: Type of rule (SUTRA, PARIBHASA, etc.)
        priority: Application priority (lower = higher priority)
        match_fn: Function to check if rule applies
        apply_fn: Function to apply the transformation
        adhikara: Domain/scope of the rule
        anuvrti: Inherited scope from previous rules
        cross_refs: References to related sūtras
        max_applications: Maximum times this rule can be applied
        applications: Current application count
        active: Whether the rule is currently active
        meta_data: Additional rule metadata
    """
    sutra_ref: SutraReference
    name: str
    description: str
    rule_type: RuleType
    priority: int
    match_fn: Callable[[List[Token], int], bool]
    apply_fn: Callable[[List[Token], int], Tuple[List[Token], int]]
    adhikara: Set[str] = field(default_factory=set)
    anuvrti: Set[str] = field(default_factory=set)
    cross_refs: List[SutraReference] = field(default_factory=list)
    max_applications: Optional[int] = None
    applications: int = 0
    active: bool = True
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Unique identifier based on sūtra reference."""
        return str(self.sutra_ref)
    
    def can_apply(self) -> bool:
        """Check if the rule can be applied."""
        if not self.active:
            return False
        if self.max_applications is not None and self.applications >= self.max_applications:
            return False
        return True
    
    def reset_applications(self) -> None:
        """Reset the application counter."""
        self.applications = 0
    
    def add_adhikara(self, domain: str) -> None:
        """Add a domain to the rule's scope."""
        self.adhikara.add(domain)
    
    def add_anuvrti(self, inherited_scope: str) -> None:
        """Add inherited scope from previous rules."""
        self.anuvrti.add(inherited_scope)
    
    def has_domain(self, domain: str) -> bool:
        """Check if rule applies to a specific domain."""
        return domain in self.adhikara or domain in self.anuvrti
    
    def add_cross_reference(self, ref: SutraReference) -> None:
        """Add a cross-reference to another sūtra."""
        self.cross_refs.append(ref)
    
    def __str__(self) -> str:
        return f"SutraRule({self.sutra_ref}: {self.name})"
    
    def __repr__(self) -> str:
        return (f"SutraRule(sutra_ref={self.sutra_ref}, name='{self.name}', "
                f"type={self.rule_type}, priority={self.priority})")


@dataclass
class ParibhasaRule:
    """
    A paribhāṣā (meta-rule) that controls the application of other rules.
    
    These rules establish principles for interpreting and applying sūtras,
    handle exceptions, and manage cross-domain effects.
    """
    sutra_ref: SutraReference
    name: str
    description: str
    priority: int
    condition_fn: Callable[[List[Token], 'RuleRegistry'], bool]
    action_fn: Callable[['RuleRegistry'], None]
    scope: Set[str] = field(default_factory=set)  # Which domains this affects
    active: bool = True
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return f"paribhasa_{self.sutra_ref}"
    
    def evaluate(self, tokens: List[Token], registry: 'RuleRegistry') -> None:
        """Evaluate the paribhāṣā rule and apply its effects."""
        if self.active and self.condition_fn(tokens, registry):
            self.action_fn(registry)
    
    def affects_domain(self, domain: str) -> bool:
        """Check if this paribhāṣā affects a specific domain."""
        return not self.scope or domain in self.scope


class GuardSystem:
    """
    System to prevent infinite loops and manage rule application limits.
    
    Tracks rule applications at specific positions to prevent immediate
    reapplication of the same rule at the same location.
    """
    
    def __init__(self):
        # Maps (token_position, token_hash) -> set of applied rule IDs
        self._application_history: Dict[Tuple[int, int], Set[str]] = {}
        # Global application counts per rule
        self._global_applications: Dict[str, int] = {}
        # Position-specific locks to prevent oscillation
        self._position_locks: Dict[int, Set[str]] = {}
    
    def can_apply_rule(self, rule: SutraRule, tokens: List[Token], index: int) -> bool:
        """
        Check if a rule can be applied at a specific position.
        
        Args:
            rule: The rule to check
            tokens: Current token sequence
            index: Position in the token sequence
            
        Returns:
            True if the rule can be applied, False otherwise
        """
        if not rule.can_apply():
            return False
        
        # Check position-specific application history
        position_key = self._generate_position_key(tokens, index)
        if position_key in self._application_history:
            if rule.id in self._application_history[position_key]:
                return False
        
        # Check position locks
        if index in self._position_locks:
            if rule.id in self._position_locks[index]:
                return False
        
        return True
    
    def record_application(self, rule: SutraRule, tokens: List[Token], index: int) -> None:
        """
        Record that a rule was applied at a specific position.
        
        Args:
            rule: The applied rule
            tokens: Token sequence after application
            index: Position where rule was applied
        """
        # Record in application history
        position_key = self._generate_position_key(tokens, index)
        if position_key not in self._application_history:
            self._application_history[position_key] = set()
        self._application_history[position_key].add(rule.id)
        
        # Update global application count
        self._global_applications[rule.id] = self._global_applications.get(rule.id, 0) + 1
        rule.applications += 1
        
        # Set position lock for this rule
        if index not in self._position_locks:
            self._position_locks[index] = set()
        self._position_locks[index].add(rule.id)
    
    def reset_guards(self) -> None:
        """Reset all guard state."""
        self._application_history.clear()
        self._global_applications.clear()
        self._position_locks.clear()
    
    def clear_position_locks(self, start_index: int, end_index: int) -> None:
        """Clear position locks in a range (after successful transformations)."""
        for i in range(start_index, end_index + 1):
            if i in self._position_locks:
                del self._position_locks[i]
    
    def get_application_count(self, rule_id: str) -> int:
        """Get the global application count for a rule."""
        return self._global_applications.get(rule_id, 0)
    
    def _generate_position_key(self, tokens: List[Token], index: int) -> Tuple[int, int]:
        """
        Generate a key for position-based tracking.
        
        Uses token content hash to detect when the same transformation
        might be attempted repeatedly.
        """
        if index >= len(tokens):
            return (index, 0)
        
        # Create a hash based on the token and its immediate context
        context_tokens = tokens[max(0, index-1):min(len(tokens), index+2)]
        context_hash = hash(tuple(t.text for t in context_tokens))
        
        return (index, context_hash)


class RuleRegistry:
    """
    Registry for managing Pāṇini rules with traditional organization.
    
    Supports hierarchical rule organization, adhikāra domains,
    and paribhāṣā meta-rules.
    """
    
    def __init__(self):
        self._sutra_rules: List[SutraRule] = []
        self._paribhasa_rules: List[ParibhasaRule] = []
        self._rule_index: Dict[str, SutraRule] = {}
        self._paribhasa_index: Dict[str, ParibhasaRule] = {}
        self._domain_rules: Dict[str, List[SutraRule]] = {}
        self._adhikara_stack: List[str] = []  # Current adhikāra context
    
    def add_sutra_rule(self, rule: SutraRule) -> None:
        """Add a sūtra rule to the registry."""
        self._sutra_rules.append(rule)
        self._rule_index[rule.id] = rule
        
        # Index by domains
        for domain in rule.adhikara:
            if domain not in self._domain_rules:
                self._domain_rules[domain] = []
            self._domain_rules[domain].append(rule)
        
        # Sort rules by priority and sūtra reference
        self._sutra_rules.sort(key=lambda r: (r.priority, r.sutra_ref))
    
    def add_paribhasa_rule(self, rule: ParibhasaRule) -> None:
        """Add a paribhāṣā rule to the registry."""
        self._paribhasa_rules.append(rule)
        self._paribhasa_index[rule.id] = rule
        
        # Sort by priority
        self._paribhasa_rules.sort(key=lambda r: r.priority)
    
    def get_active_sutra_rules(self) -> List[SutraRule]:
        """Get all active sūtra rules in priority order."""
        return [rule for rule in self._sutra_rules if rule.active]
    
    def get_active_paribhasa_rules(self) -> List[ParibhasaRule]:
        """Get all active paribhāṣā rules in priority order."""
        return [rule for rule in self._paribhasa_rules if rule.active]
    
    def get_rules_for_domain(self, domain: str) -> List[SutraRule]:
        """Get all rules that apply to a specific domain."""
        return self._domain_rules.get(domain, [])
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule by ID."""
        if rule_id in self._rule_index:
            self._rule_index[rule_id].active = True
        elif rule_id in self._paribhasa_index:
            self._paribhasa_index[rule_id].active = True
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule by ID."""
        if rule_id in self._rule_index:
            self._rule_index[rule_id].active = False
        elif rule_id in self._paribhasa_index:
            self._paribhasa_index[rule_id].active = False
    
    def push_adhikara(self, domain: str) -> None:
        """Push an adhikāra domain onto the context stack."""
        self._adhikara_stack.append(domain)
    
    def pop_adhikara(self) -> Optional[str]:
        """Pop the current adhikāra domain from the context stack."""
        return self._adhikara_stack.pop() if self._adhikara_stack else None
    
    def get_current_adhikara(self) -> List[str]:
        """Get the current adhikāra context stack."""
        return self._adhikara_stack.copy()
    
    def apply_anuvrti(self, rule: SutraRule) -> None:
        """Apply anuvṛtti (inheritance) from the current adhikāra context."""
        for domain in self._adhikara_stack:
            rule.add_anuvrti(domain)
    
    def reset_all_applications(self) -> None:
        """Reset application counters for all rules."""
        for rule in self._sutra_rules:
            rule.reset_applications()
    
    def get_rule_by_reference(self, ref: SutraReference) -> Optional[SutraRule]:
        """Get a rule by its sūtra reference."""
        rule_id = str(ref)
        return self._rule_index.get(rule_id)
    
    def get_cross_referenced_rules(self, rule: SutraRule) -> List[SutraRule]:
        """Get all rules cross-referenced by the given rule."""
        cross_refs = []
        for ref in rule.cross_refs:
            cross_rule = self.get_rule_by_reference(ref)
            if cross_rule:
                cross_refs.append(cross_rule)
        return cross_refs
    
    def validate_rule_set(self) -> List[str]:
        """Validate the rule set and return any issues found."""
        issues = []
        
        # Check for duplicate sūtra references
        refs_seen = set()
        for rule in self._sutra_rules:
            if rule.sutra_ref in refs_seen:
                issues.append(f"Duplicate sūtra reference: {rule.sutra_ref}")
            refs_seen.add(rule.sutra_ref)
        
        # Check cross-references
        for rule in self._sutra_rules:
            for ref in rule.cross_refs:
                if not self.get_rule_by_reference(ref):
                    issues.append(f"Invalid cross-reference in {rule.sutra_ref}: {ref}")
        
        # Check adhikāra consistency
        for domain, rules in self._domain_rules.items():
            if not rules:
                issues.append(f"Empty domain: {domain}")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule registry."""
        return {
            'total_sutra_rules': len(self._sutra_rules),
            'active_sutra_rules': len(self.get_active_sutra_rules()),
            'total_paribhasa_rules': len(self._paribhasa_rules),
            'active_paribhasa_rules': len(self.get_active_paribhasa_rules()),
            'domains': list(self._domain_rules.keys()),
            'current_adhikara_stack': self._adhikara_stack.copy(),
            'rule_types': {
                rule_type.value: len([r for r in self._sutra_rules if r.rule_type == rule_type])
                for rule_type in RuleType
            }
        }