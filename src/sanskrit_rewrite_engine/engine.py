"""
Core Sanskrit transformation engine.

This module contains the main SanskritRewriteEngine class that orchestrates
the text processing pipeline.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from .performance import (
    PerformanceOptimizer, ProcessingLimiter, MemoryMonitor,
    LazyTextProcessor, performance_timer, memory_limit_check
)


@dataclass
class TransformationResult:
    """Result of a Sanskrit text transformation."""
    input_text: str
    output_text: str
    transformations_applied: List[str]
    trace: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    convergence_reached: bool = False
    iterations_used: int = 0
    infinite_loop_detected: bool = False


@dataclass
class RuleApplicationContext:
    """Context information for rule application."""
    current_text: str
    position: int
    iteration: int
    previous_applications: List[str]
    text_history: List[str]
    rule_application_count: Dict[str, int]
    iteration_rule_count: Dict[str, int] = field(default_factory=dict)
    
    def get_context_hash(self) -> str:
        """Get a hash representing the current context state."""
        context_str = f"{self.current_text}:{self.position}:{self.iteration}"
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def has_rule_been_applied_recently(self, rule_id: str, lookback: int = 3) -> bool:
        """Check if a rule has been applied recently."""
        recent_applications = self.previous_applications[-lookback:]
        return rule_id in recent_applications
    
    def get_rule_application_frequency(self, rule_id: str) -> int:
        """Get the number of times a rule has been applied."""
        return self.rule_application_count.get(rule_id, 0)
    
    def get_iteration_rule_count(self, rule_id: str) -> int:
        """Get the number of times a rule has been applied in current iteration."""
        return self.iteration_rule_count.get(rule_id, 0)
    
    def increment_iteration_rule_count(self, rule_id: str):
        """Increment the rule count for current iteration."""
        self.iteration_rule_count[rule_id] = self.iteration_rule_count.get(rule_id, 0) + 1
    
    def reset_iteration_counts(self):
        """Reset iteration-specific rule counts."""
        self.iteration_rule_count.clear()


@dataclass
class InfiniteLoopGuard:
    """Guard against infinite loops in rule application."""
    max_same_rule_applications: int = 50
    max_text_state_repetitions: int = 5
    text_state_history: List[str] = field(default_factory=list)
    rule_application_counts: Dict[str, int] = field(default_factory=dict)
    
    def check_infinite_loop(self, text: str, rule_id: str) -> bool:
        """Check if we're in an infinite loop."""
        # Check rule application count
        self.rule_application_counts[rule_id] = self.rule_application_counts.get(rule_id, 0) + 1
        if self.rule_application_counts[rule_id] > self.max_same_rule_applications:
            return True
        
        # Check text state repetition
        self.text_state_history.append(text)
        
        # Count occurrences of current text state
        text_occurrences = self.text_state_history.count(text)
        return text_occurrences > self.max_text_state_repetitions
    
    def reset(self):
        """Reset the guard state."""
        self.text_state_history.clear()
        self.rule_application_counts.clear()


class SanskritRewriteEngine:
    """Main engine for Sanskrit text transformations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Sanskrit rewrite engine.
        
        Args:
            config: Optional configuration dictionary
        """
        # Import here to avoid circular imports
        from .tokenizer import BasicSanskritTokenizer
        from .rules import RuleRegistry
        from .config import EngineConfig, load_default_config
        import os
        
        self.tokenizer = BasicSanskritTokenizer()
        self.rule_registry = RuleRegistry()
        
        # Initialize configuration
        if config is None:
            self.config = load_default_config()
        elif isinstance(config, dict):
            self.config = EngineConfig.from_dict(config)
        else:
            self.config = config
        
        # Initialize performance optimizer
        self._performance_optimizer = PerformanceOptimizer(
            cache_size=getattr(self.config, 'cache_size', 1000),
            cache_ttl=getattr(self.config, 'cache_ttl', 3600),
            memory_limit_mb=getattr(self.config, 'memory_limit_mb', 500),
            chunk_size=getattr(self.config, 'chunk_size', 1000)
        )
        
        # Load default rules if available
        self._load_default_rules()
        
        # Index rules for performance
        self._index_rules()
        
    @performance_timer
    @memory_limit_check(500)
    def process(self, text: str, enable_tracing: Optional[bool] = None, 
                rule_context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Process Sanskrit text with advanced rule application logic and performance optimizations.
        
        Args:
            text: Input Sanskrit text to process
            enable_tracing: Override config setting for tracing
            rule_context: Optional context for conditional rule activation
            
        Returns:
            TransformationResult with processing details
        """
        start_time = time.time()
        
        # Check cache first
        rules_hash = self._get_rules_hash()
        config_hash = self._get_config_hash()
        cached_result = self._performance_optimizer.cache.get(text, rules_hash, config_hash)
        
        if cached_result is not None:
            # Update metrics for cache hit
            processing_time = time.time() - start_time
            self._performance_optimizer.update_metrics(
                processing_time, len(cached_result.transformations_applied), cache_hit=True
            )
            return cached_result
        
        # Initialize processing limiter
        limiter = self._performance_optimizer.create_limiter(
            max_time=self.config.timeout_seconds,
            max_iterations=self.config.max_iterations,
            max_text_length=self.config.max_text_length
        )
        limiter.start_processing()
        
        # Check text length limit
        if limiter.check_text_length(text):
            return TransformationResult(
                input_text=text,
                output_text=text,
                transformations_applied=[],
                trace=[],
                success=False,
                error_message=f"Text length {len(text)} exceeds maximum {self.config.max_text_length}"
            )
        
        # Use lazy processing for large texts
        use_lazy_processing = len(text) > self.config.max_text_length // 2
        if use_lazy_processing:
            return self._process_with_lazy_evaluation(text, enable_tracing, rule_context, limiter)
        
        # Continue with regular processing
        original_start_time = start_time
        start_time = time.time()
        
        # Determine if tracing is enabled
        tracing_enabled = enable_tracing if enable_tracing is not None else self.config.enable_tracing
        
        # Initialize processing state
        current_text = text
        transformations_applied = []
        trace = []
        iteration = 0
        convergence_reached = False
        infinite_loop_detected = False
        
        # Initialize advanced processing components
        loop_guard = InfiniteLoopGuard()
        context = RuleApplicationContext(
            current_text=current_text,
            position=0,
            iteration=0,
            previous_applications=[],
            text_history=[current_text],
            rule_application_count={}
        )
        
        try:
            # Validate input
            if len(text) > self.config.max_text_length:
                raise ValueError(f"Text length {len(text)} exceeds maximum {self.config.max_text_length}")
            
            # Tokenize input text
            tokens = self.tokenizer.tokenize(text)
            
            if tracing_enabled:
                trace.append({
                    'step': 'tokenization',
                    'iteration': 0,
                    'tokens_count': len(tokens),
                    'tokens': [{'text': t.text, 'type': t.token_type, 'pos': (t.start_pos, t.end_pos)} for t in tokens],
                    'timestamp': time.time() - start_time
                })
            
            # Apply transformation rules iteratively with advanced logic
            while iteration < self.config.max_iterations:
                iteration += 1
                iteration_start_time = time.time()
                context.iteration = iteration
                context.current_text = current_text
                context.reset_iteration_counts()  # Reset per-iteration counts
                
                # Check for timeout
                if time.time() - start_time > self.config.timeout_seconds:
                    raise TimeoutError(f"Processing timeout after {self.config.timeout_seconds} seconds")
                
                # Get applicable rules with advanced selection
                applicable_rules = self._get_contextual_rules(current_text, context, rule_context)
                
                if not applicable_rules:
                    if tracing_enabled:
                        trace.append({
                            'step': 'no_applicable_rules',
                            'iteration': iteration,
                            'message': 'No applicable rules for current context',
                            'timestamp': time.time() - start_time
                        })
                    convergence_reached = True
                    break
                
                # Apply rules with conflict resolution and guards
                text_changed = False
                rules_applied_this_iteration = []
                rule_conflicts = []
                
                # Process text position by position for precise rule application
                position = 0
                while position < len(current_text):
                    context.position = position
                    
                    # Get best rule for this position with conflict resolution
                    best_rule, conflicts = self._resolve_rule_conflicts(
                        current_text, position, applicable_rules, context
                    )
                    
                    if conflicts and tracing_enabled:
                        rule_conflicts.extend(conflicts)
                    
                    if best_rule is None:
                        position += 1
                        continue
                    
                    # Check infinite loop guard
                    if loop_guard.check_infinite_loop(current_text, best_rule.id):
                        infinite_loop_detected = True
                        if tracing_enabled:
                            trace.append({
                                'step': 'infinite_loop_detected',
                                'iteration': iteration,
                                'rule_id': best_rule.id,
                                'rule_name': best_rule.name,
                                'position': position,
                                'message': 'Infinite loop detected, stopping rule application',
                                'timestamp': time.time() - start_time
                            })
                        break
                    
                    # Apply the rule
                    if best_rule.matches(current_text, position):
                        old_text = current_text
                        current_text, new_position = best_rule.apply(current_text, position)
                        
                        if current_text != old_text:
                            text_changed = True
                            transformation_info = f"{best_rule.name} (id: {best_rule.id})"
                            transformations_applied.append(transformation_info)
                            rules_applied_this_iteration.append(best_rule.id)
                            
                            # Update context
                            context.previous_applications.append(best_rule.id)
                            context.rule_application_count[best_rule.id] = \
                                context.rule_application_count.get(best_rule.id, 0) + 1
                            context.increment_iteration_rule_count(best_rule.id)
                            context.text_history.append(current_text)
                            context.current_text = current_text
                            
                            if tracing_enabled:
                                trace.append({
                                    'step': 'rule_application',
                                    'iteration': iteration,
                                    'rule_id': best_rule.id,
                                    'rule_name': best_rule.name,
                                    'position': position,
                                    'before': old_text,
                                    'after': current_text,
                                    'pattern': best_rule.pattern,
                                    'replacement': best_rule.replacement,
                                    'context_hash': context.get_context_hash(),
                                    'rule_frequency': context.get_rule_application_frequency(best_rule.id),
                                    'timestamp': time.time() - start_time
                                })
                            
                            # Update position after transformation
                            position = new_position
                        else:
                            position += 1
                    else:
                        position += 1
                
                # Break if infinite loop detected
                if infinite_loop_detected:
                    break
                
                # Log iteration summary with advanced metrics
                if tracing_enabled:
                    trace.append({
                        'step': 'iteration_summary',
                        'iteration': iteration,
                        'text_changed': text_changed,
                        'rules_applied': rules_applied_this_iteration,
                        'rule_conflicts': rule_conflicts,
                        'current_text': current_text,
                        'text_length_change': len(current_text) - len(context.text_history[-2] if len(context.text_history) > 1 else text),
                        'iteration_time': time.time() - iteration_start_time,
                        'timestamp': time.time() - start_time
                    })
                
                # Check for convergence
                if not text_changed:
                    convergence_reached = True
                    if tracing_enabled:
                        trace.append({
                            'step': 'convergence',
                            'iteration': iteration,
                            'message': 'Convergence reached - no more transformations possible',
                            'timestamp': time.time() - start_time
                        })
                    break
            
            # Check if we hit max iterations
            if iteration >= self.config.max_iterations:
                if tracing_enabled:
                    trace.append({
                        'step': 'max_iterations',
                        'iteration': iteration,
                        'message': f'Reached maximum iterations ({self.config.max_iterations})',
                        'convergence_reached': convergence_reached,
                        'timestamp': time.time() - start_time
                    })
            
            # Final processing summary
            processing_time = time.time() - start_time
            
            if tracing_enabled:
                trace.append({
                    'step': 'completion',
                    'total_iterations': iteration,
                    'total_transformations': len(transformations_applied),
                    'convergence_reached': convergence_reached,
                    'infinite_loop_detected': infinite_loop_detected,
                    'processing_time': processing_time,
                    'rule_application_stats': dict(context.rule_application_count),
                    'timestamp': processing_time
                })
            
            result = TransformationResult(
                input_text=text,
                output_text=current_text,
                transformations_applied=transformations_applied,
                trace=trace if tracing_enabled else [],
                success=True,
                convergence_reached=convergence_reached,
                iterations_used=iteration,
                infinite_loop_detected=infinite_loop_detected
            )
            
            # Cache successful result
            self._performance_optimizer.cache.put(text, rules_hash, config_hash, result)
            
            # Update performance metrics
            processing_time = time.time() - original_start_time
            self._performance_optimizer.update_metrics(
                processing_time, len(transformations_applied), cache_hit=False
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            error_message = str(e)
            
            if tracing_enabled:
                trace.append({
                    'step': 'error',
                    'iteration': iteration,
                    'error': error_message,
                    'convergence_reached': convergence_reached,
                    'infinite_loop_detected': infinite_loop_detected,
                    'timestamp': time.time() - start_time
                })
            
            return TransformationResult(
                input_text=text,
                output_text=current_text,  # Return partial result
                transformations_applied=transformations_applied,
                trace=trace if tracing_enabled else [],
                success=False,
                error_message=error_message,
                convergence_reached=convergence_reached,
                iterations_used=iteration,
                infinite_loop_detected=infinite_loop_detected
            )
        
    def load_rules(self, rule_file: str) -> None:
        """Load rules from JSON file.
        
        Args:
            rule_file: Path to JSON rule file
        """
        self.rule_registry.load_from_json(rule_file)
        
    def add_rule(self, rule) -> None:
        """Add a single rule programmatically.
        
        Args:
            rule: Rule object to add
        """
        self.rule_registry.add_rule(rule)
    
    def get_rule_count(self) -> int:
        """Get the number of loaded rules.
        
        Returns:
            Number of rules in the registry
        """
        return self.rule_registry.get_rule_count()
    
    def clear_rules(self) -> None:
        """Clear all loaded rules."""
        self.rule_registry.clear()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.to_dict()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        from .config import EngineConfig
        
        current_config = self.config.to_dict()
        current_config.update(config_updates)
        self.config = EngineConfig.from_dict(current_config)
    
    def _load_default_rules(self) -> None:
        """Load default rules from the configured rule directories."""
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Try to load the default rule set
        for rule_dir in self.config.rule_directories:
            rule_file = os.path.join(rule_dir, f"{self.config.default_rule_set}.json")
            
            if os.path.exists(rule_file):
                try:
                    self.rule_registry.load_from_json(rule_file)
                    logger.info(f"Loaded default rules from {rule_file}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load rules from {rule_file}: {e}")
                    continue
        
        # If no default rules found, create some basic rules programmatically
        logger.info("No default rule files found, creating basic rules programmatically")
        self._create_basic_rules()
    
    def _get_contextual_rules(self, text: str, context: RuleApplicationContext, 
                             rule_context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Get rules applicable in the current context with conditional activation.
        
        Args:
            text: Current text being processed
            context: Current processing context
            rule_context: Optional external context for rule activation
            
        Returns:
            List of applicable rules
        """
        from .rules import Rule
        
        all_rules = self.rule_registry.get_rules_by_priority()
        applicable_rules = []
        
        for rule in all_rules:
            if not rule.enabled:
                continue
            
            # Check if rule should be activated based on context
            if not self._should_activate_rule(rule, text, context, rule_context):
                continue
            
            # Check if rule has been applied too frequently
            if self._is_rule_overused(rule, context):
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _should_activate_rule(self, rule: Any, text: str, context: RuleApplicationContext,
                             rule_context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a rule should be activated based on context.
        
        Args:
            rule: Rule to check
            text: Current text
            context: Processing context
            rule_context: Optional external context
            
        Returns:
            True if rule should be activated
        """
        # Check basic conditions
        if not rule.enabled:
            return False
        
        # Check if rule has conditional activation metadata
        conditions = rule.metadata.get('conditions', {})
        
        # Check iteration-based conditions
        if 'min_iteration' in conditions:
            if context.iteration < conditions['min_iteration']:
                return False
        
        if 'max_iteration' in conditions:
            if context.iteration > conditions['max_iteration']:
                return False
        
        # Check text-based conditions
        if 'requires_text_pattern' in conditions:
            import re
            pattern = conditions['requires_text_pattern']
            if not re.search(pattern, text):
                return False
        
        if 'excludes_text_pattern' in conditions:
            import re
            pattern = conditions['excludes_text_pattern']
            if re.search(pattern, text):
                return False
        
        # Check category-based conditions
        if rule_context and 'required_categories' in rule_context:
            rule_category = rule.metadata.get('category')
            if rule_category not in rule_context['required_categories']:
                return False
        
        # Check frequency-based conditions
        if 'max_applications_per_iteration' in conditions:
            max_apps = conditions['max_applications_per_iteration']
            current_apps = context.previous_applications.count(rule.id)
            if current_apps >= max_apps:
                return False
        
        # Check dependency conditions
        if 'requires_previous_rule' in conditions:
            required_rule = conditions['requires_previous_rule']
            if required_rule not in context.previous_applications:
                return False
        
        if 'conflicts_with_rule' in conditions:
            conflicting_rule = conditions['conflicts_with_rule']
            if conflicting_rule in context.previous_applications:
                return False
        
        return True
    
    def _is_rule_overused(self, rule: Any, context: RuleApplicationContext) -> bool:
        """Check if a rule has been applied too frequently.
        
        Args:
            rule: Rule to check
            context: Processing context
            
        Returns:
            True if rule is overused
        """
        # Check global application count
        max_global_applications = rule.metadata.get('max_global_applications', 1000)
        if context.get_rule_application_frequency(rule.id) >= max_global_applications:
            return True
        
        # Check per-iteration application limits
        conditions = rule.metadata.get('conditions', {})
        max_per_iteration = conditions.get('max_applications_per_iteration', 1000)
        
        if context.get_iteration_rule_count(rule.id) >= max_per_iteration:
            return True
        
        # Check recent application frequency
        recent_lookback = rule.metadata.get('recent_application_lookback', 5)
        if context.has_rule_been_applied_recently(rule.id, recent_lookback):
            # Allow if it's a high-priority rule and hasn't exceeded per-iteration limit
            current_iteration_count = context.get_iteration_rule_count(rule.id)
            if rule.priority <= 1 and current_iteration_count < max_per_iteration:
                return False
            return True
        
        return False
    
    def _resolve_rule_conflicts(self, text: str, position: int, rules: List[Any], 
                               context: RuleApplicationContext) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """Resolve conflicts between multiple applicable rules.
        
        Args:
            text: Current text
            position: Current position
            rules: List of potentially applicable rules
            context: Processing context
            
        Returns:
            Tuple of (best_rule, conflict_info)
        """
        # Find all rules that match at this position
        matching_rules = []
        for rule in rules:
            if rule.matches(text, position):
                matching_rules.append(rule)
        
        if not matching_rules:
            return None, []
        
        if len(matching_rules) == 1:
            return matching_rules[0], []
        
        # Multiple rules match - resolve conflicts
        conflicts = []
        
        # Group rules by priority
        priority_groups = {}
        for rule in matching_rules:
            priority = rule.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(rule)
        
        # Get highest priority group (lowest number)
        highest_priority = min(priority_groups.keys())
        highest_priority_rules = priority_groups[highest_priority]
        
        # Record conflicts for tracing
        if len(matching_rules) > 1:
            conflicts.append({
                'position': position,
                'conflicting_rules': [{'id': r.id, 'name': r.name, 'priority': r.priority} 
                                    for r in matching_rules],
                'resolution_method': 'priority',
                'selected_rule': None  # Will be filled below
            })
        
        # If still multiple rules at same priority, use additional criteria
        if len(highest_priority_rules) > 1:
            # Prefer rules that haven't been used recently
            unused_recently = [r for r in highest_priority_rules 
                             if not context.has_rule_been_applied_recently(r.id)]
            if unused_recently:
                highest_priority_rules = unused_recently
        
        if len(highest_priority_rules) > 1:
            # Prefer rules with lower application frequency
            min_frequency = min(context.get_rule_application_frequency(r.id) 
                              for r in highest_priority_rules)
            least_used = [r for r in highest_priority_rules 
                         if context.get_rule_application_frequency(r.id) == min_frequency]
            highest_priority_rules = least_used
        
        if len(highest_priority_rules) > 1:
            # Prefer rules with more specific patterns (longer patterns)
            max_pattern_length = max(len(r.pattern) for r in highest_priority_rules)
            most_specific = [r for r in highest_priority_rules 
                           if len(r.pattern) == max_pattern_length]
            highest_priority_rules = most_specific
        
        # Select the first rule (deterministic selection)
        selected_rule = highest_priority_rules[0]
        
        # Update conflict info
        if conflicts:
            conflicts[-1]['selected_rule'] = {
                'id': selected_rule.id, 
                'name': selected_rule.name, 
                'priority': selected_rule.priority
            }
        
        return selected_rule, conflicts
    
    def _create_basic_rules(self) -> None:
        """Create basic transformation rules programmatically."""
        from .rules import Rule
        
        basic_rules = [
            Rule(
                id="vowel_sandhi_a_i",
                name="a + i → e",
                description="Combine 'a' and 'i' vowels into 'e'",
                pattern=r"a\s*\+\s*i",
                replacement="e",
                priority=1,
                enabled=True,
                metadata={
                    "category": "vowel_sandhi",
                    "sutra_ref": "6.1.87",
                    "example": "rāma + iti → rāmeti",
                    "conditions": {
                        "max_applications_per_iteration": 10
                    },
                    "max_global_applications": 50
                }
            ),
            Rule(
                id="vowel_sandhi_a_u",
                name="a + u → o",
                description="Combine 'a' and 'u' vowels into 'o'",
                pattern=r"a\s*\+\s*u",
                replacement="o",
                priority=1,
                enabled=True,
                metadata={
                    "category": "vowel_sandhi",
                    "sutra_ref": "6.1.87",
                    "example": "gaccha + uktam → gacchokta",
                    "conditions": {
                        "max_applications_per_iteration": 10
                    },
                    "max_global_applications": 50
                }
            ),
            Rule(
                id="compound_formation",
                name="Compound Formation",
                description="Join words marked with + into compounds",
                pattern=r"([a-zA-Zअ-ह]+)\s*\+\s*([a-zA-Zअ-ह]+)",
                replacement=r"\1\2",
                priority=2,
                enabled=True,
                metadata={
                    "category": "compound_formation",
                    "example": "deva + rāja → devarāja",
                    "conditions": {
                        "requires_text_pattern": r"\+",
                        "max_applications_per_iteration": 20
                    },
                    "max_global_applications": 100
                }
            )
        ]
        
        for rule in basic_rules:
            try:
                self.rule_registry.add_rule(rule)
            except ValueError:
                # Rule already exists, skip
                pass
    
    def _index_rules(self) -> None:
        """Index rules for performance optimization."""
        # Clear existing index
        self._performance_optimizer.rule_index.clear()
        
        # Add all rules to index
        for rule in self.rule_registry.get_rules_by_priority():
            self._performance_optimizer.rule_index.add_rule(rule)
    
    def _get_rules_hash(self) -> str:
        """Generate hash of current rules for caching."""
        rule_data = []
        for rule in self.rule_registry.get_rules_by_priority():
            rule_data.append(f"{rule.id}:{rule.pattern}:{rule.replacement}:{rule.enabled}")
        
        combined = "|".join(rule_data)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for caching."""
        config_data = self.config.to_dict()
        # Remove non-deterministic fields
        config_data.pop('debug_mode', None)
        config_data.pop('log_level', None)
        
        config_str = str(sorted(config_data.items()))
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def _process_with_lazy_evaluation(self, text: str, enable_tracing: Optional[bool], 
                                    rule_context: Optional[Dict[str, Any]], 
                                    limiter: ProcessingLimiter) -> TransformationResult:
        """Process large text using lazy evaluation.
        
        Args:
            text: Input text
            enable_tracing: Tracing flag
            rule_context: Rule context
            limiter: Processing limiter
            
        Returns:
            TransformationResult
        """
        lazy_processor = self._performance_optimizer.create_lazy_processor(text)
        transformations_applied = []
        trace = []
        
        tracing_enabled = enable_tracing if enable_tracing is not None else self.config.enable_tracing
        
        def process_chunk(chunk_text: str) -> str:
            """Process a single chunk."""
            # Use regular processing for chunk
            chunk_result = self._process_chunk(chunk_text, enable_tracing=False, 
                                             rule_context=rule_context, limiter=limiter)
            if chunk_result.success:
                transformations_applied.extend(chunk_result.transformations_applied)
                if tracing_enabled:
                    trace.extend(chunk_result.trace)
                return chunk_result.output_text
            return chunk_text
        
        # Process chunks lazily
        try:
            for i in range(lazy_processor.chunk_count()):
                if limiter.check_timeout():
                    raise TimeoutError(f"Processing timeout after {limiter.max_processing_time} seconds")
                
                if self._performance_optimizer.memory_monitor.check_memory_limit():
                    self._performance_optimizer.metrics.memory_limits_triggered += 1
                    raise MemoryError("Memory limit exceeded during processing")
                
                lazy_processor.process_chunk(i, process_chunk)
                limiter.increment_iteration()
            
            output_text = lazy_processor.get_processed_text()
            
            return TransformationResult(
                input_text=text,
                output_text=output_text,
                transformations_applied=transformations_applied,
                trace=trace if tracing_enabled else [],
                success=True,
                convergence_reached=True,
                iterations_used=limiter.iteration_count,
                infinite_loop_detected=False
            )
            
        except (TimeoutError, MemoryError) as e:
            if isinstance(e, TimeoutError):
                self._performance_optimizer.metrics.timeouts_triggered += 1
            
            return TransformationResult(
                input_text=text,
                output_text=lazy_processor.get_processed_text(),
                transformations_applied=transformations_applied,
                trace=trace if tracing_enabled else [],
                success=False,
                error_message=str(e),
                convergence_reached=False,
                iterations_used=limiter.iteration_count,
                infinite_loop_detected=False
            )
    
    def _process_chunk(self, text: str, enable_tracing: Optional[bool], 
                      rule_context: Optional[Dict[str, Any]], 
                      limiter: ProcessingLimiter) -> TransformationResult:
        """Process a single text chunk (simplified version of main process method).
        
        Args:
            text: Chunk text to process
            enable_tracing: Tracing flag
            rule_context: Rule context
            limiter: Processing limiter
            
        Returns:
            TransformationResult for the chunk
        """
        # Simplified processing for chunks - just apply rules once
        current_text = text
        transformations_applied = []
        trace = []
        
        tracing_enabled = enable_tracing if enable_tracing is not None else self.config.enable_tracing
        
        try:
            # Get candidate rules using index
            candidate_rules = self._performance_optimizer.rule_index.get_candidate_rules(current_text, 0)
            
            # Apply rules
            for rule in candidate_rules:
                if not rule.enabled:
                    continue
                
                if limiter.check_timeout():
                    break
                
                position = 0
                while position < len(current_text):
                    if rule.matches(current_text, position):
                        old_text = current_text
                        current_text, new_position = rule.apply(current_text, position)
                        
                        if current_text != old_text:
                            transformations_applied.append(f"{rule.name} (id: {rule.id})")
                            
                            if tracing_enabled:
                                trace.append({
                                    'step': 'rule_application',
                                    'rule_id': rule.id,
                                    'rule_name': rule.name,
                                    'position': position,
                                    'before': old_text,
                                    'after': current_text
                                })
                            
                            position = new_position
                        else:
                            position += 1
                    else:
                        position += 1
            
            return TransformationResult(
                input_text=text,
                output_text=current_text,
                transformations_applied=transformations_applied,
                trace=trace,
                success=True,
                convergence_reached=True,
                iterations_used=1,
                infinite_loop_detected=False
            )
            
        except Exception as e:
            return TransformationResult(
                input_text=text,
                output_text=current_text,
                transformations_applied=transformations_applied,
                trace=trace,
                success=False,
                error_message=str(e),
                convergence_reached=False,
                iterations_used=1,
                infinite_loop_detected=False
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics.
        
        Returns:
            Performance metrics dictionary
        """
        return self._performance_optimizer.get_performance_report()
    
    def clear_cache(self) -> None:
        """Clear transformation cache."""
        self._performance_optimizer.clear_cache()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self._performance_optimizer.reset_metrics()