"""
Pāṇini rule engine for applying Sanskrit grammatical transformations.

This module implements the main engine that orchestrates the application
of Pāṇini sūtras with proper guarding, tracing, and convergence detection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
import logging

from .sanskrit_token import Token
from .rule import SutraRule, ParibhasaRule, RuleRegistry, GuardSystem
from .essential_sutras import create_essential_sutras, create_essential_paribhasas


@dataclass
class TransformationTrace:
    """Trace of a single rule application."""
    rule_name: str
    rule_id: str
    sutra_ref: str
    index: int
    tokens_before: List[Token]
    tokens_after: List[Token]
    timestamp: datetime
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Applied {self.rule_name} ({self.sutra_ref}) at position {self.index}"


@dataclass
class PassTrace:
    """Trace of a complete processing pass."""
    pass_number: int
    tokens_before: List[Token]
    tokens_after: List[Token]
    transformations: List[TransformationTrace]
    paribhasa_applications: List[str]
    convergence_achieved: bool
    
    def get_transformation_count(self) -> int:
        return len(self.transformations)
    
    def get_rules_applied(self) -> Set[str]:
        return {t.rule_name for t in self.transformations}


@dataclass
class PaniniEngineResult:
    """Result of Pāṇini rule engine processing."""
    input_text: str
    input_tokens: List[Token]
    output_tokens: List[Token]
    converged: bool
    passes: int
    traces: List[PassTrace]
    errors: List[str]
    statistics: Dict[str, Any]
    
    def get_output_text(self) -> str:
        """Reconstruct text from output tokens."""
        return ''.join(token.text for token in self.output_tokens)
    
    def get_transformation_summary(self) -> Dict[str, int]:
        """Get summary of transformations applied."""
        summary = {}
        for trace in self.traces:
            for transformation in trace.transformations:
                rule_name = transformation.rule_name
                summary[rule_name] = summary.get(rule_name, 0) + 1
        return summary
    
    def get_rules_applied(self) -> Set[str]:
        """Get set of all rules that were applied."""
        rules = set()
        for trace in self.traces:
            rules.update(trace.get_rules_applied())
        return rules


class PaniniRuleEngine:
    """
    Main engine for applying Pāṇini sūtras to Sanskrit text.
    
    Features:
    - Guarded rule application with loop prevention
    - Iterative processing until convergence
    - Comprehensive tracing and debugging
    - Support for paribhāṣā meta-rules
    - Traditional sūtra organization with adhikāra domains
    """
    
    def __init__(self, tokenizer=None, custom_rules: Optional[List[SutraRule]] = None):
        """
        Initialize the Pāṇini rule engine.
        
        Args:
            tokenizer: Sanskrit tokenizer instance
            custom_rules: Additional custom rules to include
        """
        self.tokenizer = tokenizer
        self.registry = RuleRegistry()
        self.guard_system = GuardSystem()
        self.logger = logging.getLogger(__name__)
        
        # Load essential sūtras
        self._load_essential_rules()
        
        # Add custom rules if provided
        if custom_rules:
            for rule in custom_rules:
                self.registry.add_sutra_rule(rule)
        
        # Validate rule set
        self._validate_rules()
    
    def _load_essential_rules(self) -> None:
        """Load the essential Pāṇini sūtras and paribhāṣās."""
        # Load sūtra rules
        essential_sutras = create_essential_sutras()
        for rule in essential_sutras:
            self.registry.add_sutra_rule(rule)
        
        # Load paribhāṣā rules
        essential_paribhasas = create_essential_paribhasas()
        for paribhasa in essential_paribhasas:
            self.registry.add_paribhasa_rule(paribhasa)
        
        self.logger.info(f"Loaded {len(essential_sutras)} sūtra rules and "
                        f"{len(essential_paribhasas)} paribhāṣā rules")
    
    def _validate_rules(self) -> None:
        """Validate the loaded rule set."""
        issues = self.registry.validate_rule_set()
        if issues:
            self.logger.warning(f"Rule validation issues found: {issues}")
        else:
            self.logger.info("Rule set validation passed")
    
    def process(self, tokens: List[Token], max_passes: int = 20) -> PaniniEngineResult:
        """
        Process tokens through the Pāṇini rule engine.
        
        Args:
            tokens: Input token sequence
            max_passes: Maximum number of processing passes
            
        Returns:
            PaniniEngineResult with transformation details
        """
        if not tokens:
            return PaniniEngineResult(
                input_text="",
                input_tokens=[],
                output_tokens=[],
                converged=True,
                passes=0,
                traces=[],
                errors=[],
                statistics={}
            )
        
        # Initialize processing state
        current_tokens = [token for token in tokens]  # Deep copy
        input_tokens = [token for token in tokens]
        traces = []
        errors = []
        converged = False
        
        # Reset guard system and rule applications
        self.guard_system.reset_guards()
        self.registry.reset_all_applications()
        
        self.logger.info(f"Starting Pāṇini processing with {len(tokens)} tokens, max {max_passes} passes")
        
        # Main processing loop
        for pass_num in range(1, max_passes + 1):
            self.logger.debug(f"Starting pass {pass_num}")
            
            tokens_before_pass = [token for token in current_tokens]
            
            # Apply paribhāṣā rules first
            paribhasa_applications = self._apply_paribhasa_rules(current_tokens)
            
            # Apply sūtra rules
            current_tokens, transformations = self._apply_sutra_rules(current_tokens)
            
            # Create pass trace
            pass_trace = PassTrace(
                pass_number=pass_num,
                tokens_before=tokens_before_pass,
                tokens_after=[token for token in current_tokens],
                transformations=transformations,
                paribhasa_applications=paribhasa_applications,
                convergence_achieved=len(transformations) == 0
            )
            traces.append(pass_trace)
            
            # Check for convergence
            if len(transformations) == 0:
                converged = True
                self.logger.info(f"Convergence achieved after {pass_num} passes")
                break
            
            self.logger.debug(f"Pass {pass_num}: {len(transformations)} transformations applied")
        
        if not converged:
            self.logger.warning(f"Failed to converge after {max_passes} passes")
            errors.append(f"Failed to converge after {max_passes} passes")
        
        # Generate statistics
        statistics = self._generate_statistics(traces, current_tokens)
        
        return PaniniEngineResult(
            input_text=''.join(t.text for t in input_tokens),
            input_tokens=input_tokens,
            output_tokens=current_tokens,
            converged=converged,
            passes=len(traces),
            traces=traces,
            errors=errors,
            statistics=statistics
        )
    
    def _apply_paribhasa_rules(self, tokens: List[Token]) -> List[str]:
        """
        Apply paribhāṣā (meta-rules) that control other rules.
        
        Args:
            tokens: Current token sequence
            
        Returns:
            List of paribhāṣā rule names that were applied
        """
        applied_paribhasas = []
        
        for paribhasa in self.registry.get_active_paribhasa_rules():
            try:
                paribhasa.evaluate(tokens, self.registry)
                applied_paribhasas.append(paribhasa.name)
                self.logger.debug(f"Applied paribhāṣā: {paribhasa.name}")
            except Exception as e:
                self.logger.error(f"Error applying paribhāṣā {paribhasa.name}: {e}")
        
        return applied_paribhasas
    
    def _apply_sutra_rules(self, tokens: List[Token]) -> Tuple[List[Token], List[TransformationTrace]]:
        """
        Apply sūtra rules to the token sequence.
        
        Args:
            tokens: Current token sequence
            
        Returns:
            Tuple of (transformed_tokens, transformation_traces)
        """
        current_tokens = tokens
        transformations = []
        
        # Process tokens left to right
        index = 0
        while index < len(current_tokens):
            # Find the highest priority applicable rule at this position
            applicable_rule = self._find_applicable_rule(current_tokens, index)
            
            if applicable_rule is None:
                index += 1
                continue
            
            # Record tokens before transformation
            tokens_before = [token for token in current_tokens]
            
            try:
                # Apply the rule
                new_tokens, new_index = applicable_rule.apply_fn(current_tokens, index)
                
                # Record the application in guard system
                self.guard_system.record_application(applicable_rule, current_tokens, index)
                
                # Create transformation trace
                transformation = TransformationTrace(
                    rule_name=applicable_rule.name,
                    rule_id=applicable_rule.id,
                    sutra_ref=str(applicable_rule.sutra_ref),
                    index=index,
                    tokens_before=tokens_before,
                    tokens_after=[token for token in new_tokens],
                    timestamp=datetime.now(),
                    meta_data={
                        'rule_type': applicable_rule.rule_type.value,
                        'adhikara': list(applicable_rule.adhikara),
                        'anuvrti': list(applicable_rule.anuvrti)
                    }
                )
                transformations.append(transformation)
                
                # Update tokens and position
                current_tokens = new_tokens
                index = new_index
                
                self.logger.debug(f"Applied {applicable_rule.name} at position {index}")
                
            except Exception as e:
                self.logger.error(f"Error applying rule {applicable_rule.name} at position {index}: {e}")
                index += 1
        
        return current_tokens, transformations
    
    def _find_applicable_rule(self, tokens: List[Token], index: int) -> Optional[SutraRule]:
        """
        Find the highest priority applicable rule at the given position.
        
        Args:
            tokens: Current token sequence
            index: Position to check
            
        Returns:
            The applicable rule with highest priority, or None
        """
        applicable_rules = []
        
        for rule in self.registry.get_active_sutra_rules():
            # Check if rule can be applied (guard system)
            if not self.guard_system.can_apply_rule(rule, tokens, index):
                continue
            
            # Check if rule matches at this position
            try:
                if rule.match_fn(tokens, index):
                    applicable_rules.append(rule)
            except Exception as e:
                self.logger.error(f"Error checking rule {rule.name} at position {index}: {e}")
        
        if not applicable_rules:
            return None
        
        # Return rule with highest priority (lowest priority number)
        # If priorities are equal, use sūtra reference order
        return min(applicable_rules, key=lambda r: (r.priority, r.sutra_ref))
    
    def _generate_statistics(self, traces: List[PassTrace], final_tokens: List[Token]) -> Dict[str, Any]:
        """Generate processing statistics."""
        total_transformations = sum(len(trace.transformations) for trace in traces)
        
        rule_usage = {}
        for trace in traces:
            for transformation in trace.transformations:
                rule_name = transformation.rule_name
                rule_usage[rule_name] = rule_usage.get(rule_name, 0) + 1
        
        # Token statistics
        token_stats = {
            'vowel_count': sum(1 for t in final_tokens if t.kind.value == 'VOWEL'),
            'consonant_count': sum(1 for t in final_tokens if t.kind.value == 'CONSONANT'),
            'marker_count': sum(1 for t in final_tokens if t.kind.value == 'MARKER'),
            'other_count': sum(1 for t in final_tokens if t.kind.value == 'OTHER'),
        }
        
        # Sandhi statistics
        sandhi_results = sum(1 for t in final_tokens if t.has_tag('sandhi_result'))
        
        return {
            'total_passes': len(traces),
            'total_transformations': total_transformations,
            'rule_usage': rule_usage,
            'most_used_rule': max(rule_usage.items(), key=lambda x: x[1])[0] if rule_usage else None,
            'token_statistics': token_stats,
            'sandhi_transformations': sandhi_results,
            'final_token_count': len(final_tokens),
            'registry_statistics': self.registry.get_statistics()
        }
    
    def add_custom_rule(self, rule: SutraRule) -> None:
        """Add a custom rule to the registry."""
        self.registry.add_sutra_rule(rule)
        self.logger.info(f"Added custom rule: {rule.name}")
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule by ID."""
        self.registry.enable_rule(rule_id)
        self.logger.info(f"Enabled rule: {rule_id}")
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule by ID."""
        self.registry.disable_rule(rule_id)
        self.logger.info(f"Disabled rule: {rule_id}")
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule registry."""
        return self.registry.get_statistics()
    
    def reset_engine(self) -> None:
        """Reset the engine state."""
        self.guard_system.reset_guards()
        self.registry.reset_all_applications()
        self.logger.info("Engine state reset")


class PaniniEngineBuilder:
    """Builder class for creating configured Pāṇini engines."""
    
    def __init__(self):
        self.tokenizer = None
        self.custom_rules = []
        self.disabled_rules = set()
        self.max_applications = {}
    
    def with_tokenizer(self, tokenizer):
        """Set the tokenizer to use."""
        self.tokenizer = tokenizer
        return self
    
    def with_custom_rule(self, rule: SutraRule):
        """Add a custom rule."""
        self.custom_rules.append(rule)
        return self
    
    def disable_rule(self, rule_id: str):
        """Disable a rule by ID."""
        self.disabled_rules.add(rule_id)
        return self
    
    def set_max_applications(self, rule_id: str, max_apps: int):
        """Set maximum applications for a rule."""
        self.max_applications[rule_id] = max_apps
        return self
    
    def build(self) -> PaniniRuleEngine:
        """Build the configured engine."""
        engine = PaniniRuleEngine(self.tokenizer, self.custom_rules)
        
        # Apply configuration
        for rule_id in self.disabled_rules:
            engine.disable_rule(rule_id)
        
        for rule_id, max_apps in self.max_applications.items():
            # This would need to be implemented in the registry
            pass
        
        return engine