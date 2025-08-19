"""
Reasoning core with Prolog/Datalog backend for Sanskrit grammatical inference.

This module implements a logic programming interface that maps Sanskrit sūtras
to inference rules, supports constraint satisfaction, and provides theorem
proving capabilities using sūtra-based inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
import re
from abc import ABC, abstractmethod

from .rule import SutraRule, SutraReference, RuleRegistry
from .token import Token


class LogicalOperator(Enum):
    """Logical operators for inference rules."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"
    FORALL = "FORALL"
    EXISTS = "EXISTS"


@dataclass
class LogicalTerm:
    """A term in logical expressions."""
    name: str
    args: List['LogicalTerm'] = field(default_factory=list)
    is_variable: bool = False
    
    def __str__(self) -> str:
        if not self.args:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"
    
    def substitute(self, bindings: Dict[str, 'LogicalTerm']) -> 'LogicalTerm':
        """Apply variable substitutions."""
        if self.is_variable and self.name in bindings:
            return bindings[self.name]
        
        new_args = [arg.substitute(bindings) for arg in self.args]
        return LogicalTerm(self.name, new_args, self.is_variable)


@dataclass
class LogicalClause:
    """A logical clause (fact or rule)."""
    head: LogicalTerm
    body: List[LogicalTerm] = field(default_factory=list)
    operator: LogicalOperator = LogicalOperator.IMPLIES
    
    def is_fact(self) -> bool:
        """Check if this is a fact (no body)."""
        return len(self.body) == 0
    
    def is_rule(self) -> bool:
        """Check if this is a rule (has body)."""
        return len(self.body) > 0
    
    def __str__(self) -> str:
        if self.is_fact():
            return f"{self.head}."
        
        body_str = " ∧ ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}."


@dataclass
class InferenceRule:
    """An inference rule derived from Sanskrit sūtras."""
    id: str
    sutra_ref: Optional[SutraReference]
    name: str
    clauses: List[LogicalClause]
    priority: int = 0
    conditions: List[Callable] = field(default_factory=list)
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    def add_clause(self, clause: LogicalClause) -> None:
        """Add a logical clause to this rule."""
        self.clauses.append(clause)
    
    def get_facts(self) -> List[LogicalClause]:
        """Get all facts from this rule."""
        return [clause for clause in self.clauses if clause.is_fact()]
    
    def get_rules(self) -> List[LogicalClause]:
        """Get all rules from this rule."""
        return [clause for clause in self.clauses if clause.is_rule()]


class KnowledgeBase:
    """Knowledge base for storing logical facts and rules."""
    
    def __init__(self):
        self.facts: Set[str] = set()  # String representation of facts
        self.rules: List[LogicalClause] = []
        self.inference_rules: Dict[str, InferenceRule] = {}
        self.fact_index: Dict[str, List[LogicalClause]] = {}  # predicate -> facts
    
    def add_fact(self, fact: LogicalClause) -> None:
        """Add a fact to the knowledge base."""
        if not fact.is_fact():
            raise ValueError("Only facts can be added with add_fact()")
        
        fact_str = str(fact)
        if fact_str not in self.facts:
            self.facts.add(fact_str)
            
            # Index by predicate name
            predicate = fact.head.name
            if predicate not in self.fact_index:
                self.fact_index[predicate] = []
            self.fact_index[predicate].append(fact)
    
    def add_rule(self, rule: LogicalClause) -> None:
        """Add a rule to the knowledge base."""
        if not rule.is_rule():
            raise ValueError("Only rules can be added with add_rule()")
        
        self.rules.append(rule)
    
    def add_inference_rule(self, inference_rule: InferenceRule) -> None:
        """Add an inference rule to the knowledge base."""
        self.inference_rules[inference_rule.id] = inference_rule
        
        # Add all clauses from the inference rule
        for clause in inference_rule.clauses:
            if clause.is_fact():
                self.add_fact(clause)
            else:
                self.add_rule(clause)
    
    def get_facts_by_predicate(self, predicate: str) -> List[LogicalClause]:
        """Get all facts with a specific predicate."""
        return self.fact_index.get(predicate, [])
    
    def query(self, goal: LogicalTerm) -> List[Dict[str, LogicalTerm]]:
        """Query the knowledge base for solutions."""
        # Simple implementation - would need full resolution for complex queries
        solutions = []
        
        # Check direct facts
        for fact in self.get_facts_by_predicate(goal.name):
            bindings = self._unify(goal, fact.head)
            if bindings is not None:
                solutions.append(bindings)
        
        # Check rules
        for rule in self.rules:
            if rule.head.name == goal.name:
                bindings = self._unify(goal, rule.head)
                if bindings is not None:
                    # Would need to recursively solve body goals
                    solutions.append(bindings)
        
        return solutions
    
    def _unify(self, term1: LogicalTerm, term2: LogicalTerm) -> Optional[Dict[str, LogicalTerm]]:
        """Unify two logical terms."""
        bindings = {}
        
        if not self._unify_helper(term1, term2, bindings):
            return None
        
        return bindings
    
    def _unify_helper(self, term1: LogicalTerm, term2: LogicalTerm, bindings: Dict[str, LogicalTerm]) -> bool:
        """Helper for unification."""
        # Variable unification
        if term1.is_variable:
            if term1.name in bindings:
                return self._unify_helper(bindings[term1.name], term2, bindings)
            else:
                bindings[term1.name] = term2
                return True
        
        if term2.is_variable:
            if term2.name in bindings:
                return self._unify_helper(term1, bindings[term2.name], bindings)
            else:
                bindings[term2.name] = term1
                return True
        
        # Constant unification
        if term1.name != term2.name or len(term1.args) != len(term2.args):
            return False
        
        # Recursive unification of arguments
        for arg1, arg2 in zip(term1.args, term2.args):
            if not self._unify_helper(arg1, arg2, bindings):
                return False
        
        return True


class SutraToLogicMapper:
    """Maps Sanskrit sūtras to logical inference rules."""
    
    def __init__(self):
        self.mapping_rules = self._initialize_mapping_rules()
    
    def _initialize_mapping_rules(self) -> Dict[str, Callable]:
        """Initialize mapping rules for different types of sūtras."""
        return {
            'sandhi': self._map_sandhi_rule,
            'morphology': self._map_morphological_rule,
            'phonology': self._map_phonological_rule,
            'syntax': self._map_syntactic_rule,
            'definition': self._map_definition_rule
        }
    
    def map_sutra_to_logic(self, sutra: SutraRule) -> InferenceRule:
        """Map a Sanskrit sūtra to logical inference rules."""
        rule_type = self._determine_rule_type(sutra)
        mapper = self.mapping_rules.get(rule_type, self._map_generic_rule)
        
        return mapper(sutra)
    
    def _determine_rule_type(self, sutra: SutraRule) -> str:
        """Determine the type of sūtra for mapping."""
        if 'sandhi' in sutra.adhikara:
            return 'sandhi'
        elif 'morphology' in sutra.adhikara:
            return 'morphology'
        elif 'phonology' in sutra.adhikara:
            return 'phonology'
        elif 'syntax' in sutra.adhikara:
            return 'syntax'
        elif sutra.rule_type.value == 'ADHIKARA':
            return 'definition'
        else:
            return 'generic'
    
    def _map_sandhi_rule(self, sutra: SutraRule) -> InferenceRule:
        """Map sandhi rules to logical form."""
        inference_rule = InferenceRule(
            id=f"sandhi_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )
        
        # Example: iko yaṇ aci (i,u,ṛ,ḷ become y,v,r,l before vowels)
        if sutra.name == "iko yaṇ aci":
            # sandhi_applies(X, Y, Z) :- vowel_ik(X), vowel(Y), semivowel_yan(X, Z)
            head = LogicalTerm("sandhi_applies", [
                LogicalTerm("X", is_variable=True),
                LogicalTerm("Y", is_variable=True),
                LogicalTerm("Z", is_variable=True)
            ])
            body = [
                LogicalTerm("vowel_ik", [LogicalTerm("X", is_variable=True)]),
                LogicalTerm("vowel", [LogicalTerm("Y", is_variable=True)]),
                LogicalTerm("semivowel_yan", [LogicalTerm("X", is_variable=True), LogicalTerm("Z", is_variable=True)])
            ]
            clause = LogicalClause(head, body)
            inference_rule.add_clause(clause)
            
            # Add supporting facts
            for vowel, semivowel in [("i", "y"), ("u", "v"), ("ṛ", "r"), ("ḷ", "l")]:
                fact = LogicalClause(LogicalTerm("semivowel_yan", [
                    LogicalTerm(vowel), LogicalTerm(semivowel)
                ]))
                inference_rule.add_clause(fact)
        
        return inference_rule
    
    def _map_morphological_rule(self, sutra: SutraRule) -> InferenceRule:
        """Map morphological rules to logical form."""
        inference_rule = InferenceRule(
            id=f"morph_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )
        
        # Example mapping for morphological rules
        # morpheme_attachment(Root, Suffix, Result) :- conditions...
        
        return inference_rule
    
    def _map_phonological_rule(self, sutra: SutraRule) -> InferenceRule:
        """Map phonological rules to logical form."""
        inference_rule = InferenceRule(
            id=f"phon_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )
        
        return inference_rule
    
    def _map_syntactic_rule(self, sutra: SutraRule) -> InferenceRule:
        """Map syntactic rules to logical form."""
        inference_rule = InferenceRule(
            id=f"syn_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )
        
        return inference_rule
    
    def _map_definition_rule(self, sutra: SutraRule) -> InferenceRule:
        """Map definition rules to logical form."""
        inference_rule = InferenceRule(
            id=f"def_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )
        
        # Example: vṛddhir ādaic (vṛddhi vowels are ā, ai, au)
        if sutra.name == "vṛddhir ādaic":
            for vowel in ["ā", "ai", "au"]:
                fact = LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm(vowel)]))
                inference_rule.add_clause(fact)
        
        return inference_rule
    
    def _map_generic_rule(self, sutra: SutraRule) -> InferenceRule:
        """Generic mapping for unspecified rule types."""
        return InferenceRule(
            id=f"generic_{sutra.id}",
            sutra_ref=sutra.sutra_ref,
            name=sutra.name,
            clauses=[],
            priority=sutra.priority
        )


class ConstraintSolver:
    """Constraint satisfaction solver for complex grammatical problems."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.constraints: List[Callable] = []
    
    def add_constraint(self, constraint: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a constraint function."""
        self.constraints.append(constraint)
    
    def solve_constraints(self, variables: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Solve constraints using backtracking."""
        solutions = []
        self._backtrack_solve({}, variables, solutions)
        return solutions
    
    def _backtrack_solve(self, assignment: Dict[str, Any], 
                        remaining_vars: Dict[str, List[Any]], 
                        solutions: List[Dict[str, Any]]) -> None:
        """Backtracking constraint solver."""
        if not remaining_vars:
            # Check all constraints
            if all(constraint(assignment) for constraint in self.constraints):
                solutions.append(assignment.copy())
            return
        
        # Choose next variable
        var = next(iter(remaining_vars))
        values = remaining_vars[var]
        new_remaining = {k: v for k, v in remaining_vars.items() if k != var}
        
        for value in values:
            assignment[var] = value
            
            # Check constraints early (forward checking)
            if self._is_consistent(assignment):
                self._backtrack_solve(assignment, new_remaining, solutions)
            
            del assignment[var]
    
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if current assignment is consistent with constraints."""
        return all(constraint(assignment) for constraint in self.constraints)


class TheoremProver:
    """Theorem prover using sūtra-based inference."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.proof_cache: Dict[str, bool] = {}
    
    def prove(self, goal: LogicalTerm, max_depth: int = 10) -> Tuple[bool, List[str]]:
        """Prove a goal using resolution and sūtra-based inference."""
        proof_steps = []
        goal_str = str(goal)
        
        if goal_str in self.proof_cache:
            return self.proof_cache[goal_str], proof_steps
        
        result = self._prove_recursive(goal, max_depth, proof_steps, set())
        self.proof_cache[goal_str] = result
        
        return result, proof_steps
    
    def _prove_recursive(self, goal: LogicalTerm, depth: int, 
                        proof_steps: List[str], visited: Set[str]) -> bool:
        """Recursive proof search."""
        if depth <= 0:
            return False
        
        goal_str = str(goal)
        if goal_str in visited:
            return False  # Avoid cycles
        
        visited.add(goal_str)
        
        # Try to unify with facts
        for fact in self.kb.get_facts_by_predicate(goal.name):
            bindings = self.kb._unify(goal, fact.head)
            if bindings is not None:
                proof_steps.append(f"Unified {goal} with fact {fact.head}")
                visited.remove(goal_str)
                return True
        
        # Try to resolve with rules
        for rule in self.kb.rules:
            if rule.head.name == goal.name:
                bindings = self.kb._unify(goal, rule.head)
                if bindings is not None:
                    proof_steps.append(f"Trying rule: {rule}")
                    
                    # Prove all body goals
                    all_proved = True
                    for body_goal in rule.body:
                        substituted_goal = body_goal.substitute(bindings)
                        if not self._prove_recursive(substituted_goal, depth - 1, proof_steps, visited):
                            all_proved = False
                            break
                    
                    if all_proved:
                        proof_steps.append(f"Successfully proved {goal} using rule")
                        visited.remove(goal_str)
                        return True
        
        visited.remove(goal_str)
        return False
    
    def explain_proof(self, goal: LogicalTerm) -> str:
        """Generate human-readable proof explanation."""
        proved, steps = self.prove(goal)
        
        if proved:
            explanation = f"Proof of {goal}:\n"
            for i, step in enumerate(steps, 1):
                explanation += f"{i}. {step}\n"
            return explanation
        else:
            return f"Could not prove {goal}"


class QueryPlanner:
    """Context-aware query planner for Sanskrit rules vs LLM calls."""
    
    def __init__(self, knowledge_base: KnowledgeBase, llm_interface=None):
        self.kb = knowledge_base
        self.llm_interface = llm_interface
        self.rule_coverage = self._analyze_rule_coverage()
    
    def _analyze_rule_coverage(self) -> Dict[str, float]:
        """Analyze coverage of different grammatical domains by rules."""
        coverage = {
            'sandhi': 0.8,      # High coverage for sandhi rules
            'morphology': 0.6,  # Medium coverage for morphology
            'syntax': 0.4,      # Lower coverage for syntax
            'semantics': 0.2    # Very low coverage for semantics
        }
        return coverage
    
    def plan_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan how to handle a query - use rules or LLM."""
        query_type = self._classify_query(query, context)
        
        plan = {
            'query': query,
            'query_type': query_type,
            'use_rules': False,
            'use_llm': False,
            'confidence': 0.0,
            'reasoning': ""
        }
        
        coverage = self.rule_coverage.get(query_type, 0.0)
        
        if coverage > 0.7:
            plan['use_rules'] = True
            plan['confidence'] = coverage
            plan['reasoning'] = f"High rule coverage ({coverage:.1%}) for {query_type}"
        elif coverage > 0.3:
            plan['use_rules'] = True
            plan['use_llm'] = True
            plan['confidence'] = 0.6
            plan['reasoning'] = f"Medium rule coverage ({coverage:.1%}), using hybrid approach"
        else:
            plan['use_llm'] = True
            plan['confidence'] = 0.4
            plan['reasoning'] = f"Low rule coverage ({coverage:.1%}), using LLM"
        
        return plan
    
    def _classify_query(self, query: str, context: Dict[str, Any]) -> str:
        """Classify the type of grammatical query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sandhi', 'combination', 'euphonic']):
            return 'sandhi'
        elif any(word in query_lower for word in ['morphology', 'inflection', 'declension', 'decline']):
            return 'morphology'
        elif any(word in query_lower for word in ['syntax', 'parsing', 'structure']):
            return 'syntax'
        elif any(word in query_lower for word in ['meaning', 'semantic', 'interpretation', 'mean']):
            return 'semantics'
        else:
            return 'general'
    
    def execute_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query using the planned approach."""
        plan = self.plan_query(query, context)
        result = {
            'query': query,
            'plan': plan,
            'rule_results': None,
            'llm_results': None,
            'final_answer': None
        }
        
        if plan['use_rules']:
            result['rule_results'] = self._execute_rule_query(query, context)
        
        if plan['use_llm']:
            result['llm_results'] = self._execute_llm_query(query, context)
        
        # Combine results
        result['final_answer'] = self._combine_results(result)
        
        return result
    
    def _execute_rule_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query using logical rules."""
        # Convert query to logical form and execute
        # This is a simplified implementation
        return {
            'method': 'logical_rules',
            'results': [],
            'confidence': 0.8
        }
    
    def _execute_llm_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query using LLM interface."""
        if self.llm_interface:
            # Call LLM interface
            return {
                'method': 'llm',
                'results': "LLM response would go here",
                'confidence': 0.6
            }
        else:
            return {
                'method': 'llm',
                'results': "LLM interface not available",
                'confidence': 0.0
            }
    
    def _combine_results(self, result: Dict[str, Any]) -> str:
        """Combine rule and LLM results into final answer."""
        if result['rule_results'] and result['llm_results']:
            return "Combined rule and LLM results"
        elif result['rule_results']:
            return "Rule-based result"
        elif result['llm_results']:
            return result['llm_results']['results']
        else:
            return "No results available"


class ReasoningCore:
    """Main reasoning core integrating all components."""
    
    def __init__(self, rule_registry: Optional[RuleRegistry] = None):
        self.rule_registry = rule_registry or RuleRegistry()
        self.knowledge_base = KnowledgeBase()
        self.sutra_mapper = SutraToLogicMapper()
        self.constraint_solver = ConstraintSolver(self.knowledge_base)
        self.theorem_prover = TheoremProver(self.knowledge_base)
        self.query_planner = QueryPlanner(self.knowledge_base)
        
        # Initialize with existing rules
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> None:
        """Initialize knowledge base with rules from registry."""
        for rule in self.rule_registry.get_active_sutra_rules():
            inference_rule = self.sutra_mapper.map_sutra_to_logic(rule)
            self.knowledge_base.add_inference_rule(inference_rule)
    
    def add_sutra_rule(self, sutra: SutraRule) -> None:
        """Add a new sūtra rule to the reasoning system."""
        self.rule_registry.add_sutra_rule(sutra)
        inference_rule = self.sutra_mapper.map_sutra_to_logic(sutra)
        self.knowledge_base.add_inference_rule(inference_rule)
    
    def query(self, goal: Union[str, LogicalTerm], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the reasoning system."""
        if isinstance(goal, str):
            # Parse string to logical term (simplified)
            goal_term = self._parse_goal_string(goal)
        else:
            goal_term = goal
        
        context = context or {}
        
        # Use query planner to determine approach
        plan = self.query_planner.plan_query(str(goal_term), context)
        
        if plan['use_rules']:
            # Try logical inference first
            solutions = self.knowledge_base.query(goal_term)
            if solutions:
                return {
                    'goal': str(goal_term),
                    'solutions': solutions,
                    'method': 'logical_inference',
                    'success': True
                }
        
        # Fallback or hybrid approach
        return self.query_planner.execute_query(str(goal_term), context)
    
    def prove(self, goal: Union[str, LogicalTerm]) -> Tuple[bool, str]:
        """Prove a goal and return explanation."""
        if isinstance(goal, str):
            goal_term = self._parse_goal_string(goal)
        else:
            goal_term = goal
        
        proved, steps = self.theorem_prover.prove(goal_term)
        explanation = self.theorem_prover.explain_proof(goal_term)
        
        return proved, explanation
    
    def solve_constraints(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Solve constraint satisfaction problems."""
        variables = problem.get('variables', {})
        constraints = problem.get('constraints', [])
        
        # Add constraints to solver
        for constraint in constraints:
            self.constraint_solver.add_constraint(constraint)
        
        return self.constraint_solver.solve_constraints(variables)
    
    def _parse_goal_string(self, goal_str: str) -> LogicalTerm:
        """Parse a goal string into a LogicalTerm (simplified parser)."""
        # This is a very basic parser - would need more sophisticated parsing
        if '(' in goal_str:
            name = goal_str[:goal_str.index('(')]
            args_str = goal_str[goal_str.index('(') + 1:goal_str.rindex(')')]
            args = [LogicalTerm(arg.strip()) for arg in args_str.split(',') if arg.strip()]
            return LogicalTerm(name, args)
        else:
            return LogicalTerm(goal_str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning system."""
        return {
            'knowledge_base': {
                'facts_count': len(self.knowledge_base.facts),
                'rules_count': len(self.knowledge_base.rules),
                'inference_rules_count': len(self.knowledge_base.inference_rules)
            },
            'rule_registry': self.rule_registry.get_statistics(),
            'proof_cache_size': len(self.theorem_prover.proof_cache)
        }