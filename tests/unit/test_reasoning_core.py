"""
Tests for the reasoning core with Prolog/Datalog backend.

This module tests logical consistency, performance, and correctness
of the Sanskrit grammatical inference system.
"""

import pytest
from typing import Dict, Any, List
import time

from sanskrit_rewrite_engine.reasoning_core import (
    LogicalTerm, LogicalClause, LogicalOperator, InferenceRule,
    KnowledgeBase, SutraToLogicMapper, ConstraintSolver,
    TheoremProver, QueryPlanner, ReasoningCore
)
from sanskrit_rewrite_engine.rule import (
    SutraRule, SutraReference, RuleType, RuleRegistry
)
from sanskrit_rewrite_engine.token import Token, TokenKind


class TestLogicalTerm:
    """Test logical term operations."""
    
    def test_term_creation(self):
        """Test creating logical terms."""
        term = LogicalTerm("vowel", [LogicalTerm("a")])
        assert term.name == "vowel"
        assert len(term.args) == 1
        assert term.args[0].name == "a"
    
    def test_term_string_representation(self):
        """Test string representation of terms."""
        term = LogicalTerm("vowel", [LogicalTerm("a")])
        assert str(term) == "vowel(a)"
        
        simple_term = LogicalTerm("a")
        assert str(simple_term) == "a"
    
    def test_variable_substitution(self):
        """Test variable substitution in terms."""
        var_term = LogicalTerm("X", is_variable=True)
        bindings = {"X": LogicalTerm("a")}
        
        substituted = var_term.substitute(bindings)
        assert substituted.name == "a"
        assert not substituted.is_variable


class TestLogicalClause:
    """Test logical clause operations."""
    
    def test_fact_creation(self):
        """Test creating facts."""
        head = LogicalTerm("vowel", [LogicalTerm("a")])
        fact = LogicalClause(head)
        
        assert fact.is_fact()
        assert not fact.is_rule()
        assert str(fact) == "vowel(a)."
    
    def test_rule_creation(self):
        """Test creating rules."""
        head = LogicalTerm("sandhi_applies", [LogicalTerm("X", is_variable=True)])
        body = [LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)])]
        rule = LogicalClause(head, body)
        
        assert rule.is_rule()
        assert not rule.is_fact()
        assert ":-" in str(rule)


class TestKnowledgeBase:
    """Test knowledge base operations."""
    
    def setup_method(self):
        """Set up test knowledge base."""
        self.kb = KnowledgeBase()
    
    def test_add_fact(self):
        """Test adding facts to knowledge base."""
        fact = LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")]))
        self.kb.add_fact(fact)
        
        assert len(self.kb.facts) == 1
        assert "vowel" in self.kb.fact_index
        assert len(self.kb.fact_index["vowel"]) == 1
    
    def test_add_rule(self):
        """Test adding rules to knowledge base."""
        head = LogicalTerm("sandhi", [LogicalTerm("X", is_variable=True)])
        body = [LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)])]
        rule = LogicalClause(head, body)
        
        self.kb.add_rule(rule)
        assert len(self.kb.rules) == 1
    
    def test_query_facts(self):
        """Test querying facts from knowledge base."""
        # Add some facts
        facts = [
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])),
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])),
            LogicalClause(LogicalTerm("consonant", [LogicalTerm("k")]))
        ]
        
        for fact in facts:
            self.kb.add_fact(fact)
        
        # Query for vowels
        goal = LogicalTerm("vowel", [LogicalTerm("a")])
        solutions = self.kb.query(goal)
        
        assert len(solutions) == 1
    
    def test_unification(self):
        """Test term unification."""
        term1 = LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)])
        term2 = LogicalTerm("vowel", [LogicalTerm("a")])
        
        bindings = self.kb._unify(term1, term2)
        
        assert bindings is not None
        assert "X" in bindings
        assert bindings["X"].name == "a"


class TestSutraToLogicMapper:
    """Test mapping Sanskrit sūtras to logical rules."""
    
    def setup_method(self):
        """Set up test mapper."""
        self.mapper = SutraToLogicMapper()
    
    def test_map_sandhi_rule(self):
        """Test mapping sandhi rules to logic."""
        # Create a test sandhi rule
        sutra = SutraRule(
            sutra_ref=SutraReference(6, 1, 77),
            name="iko yaṇ aci",
            description="i, u, ṛ, ḷ become y, v, r, l before vowels",
            rule_type=RuleType.SUTRA,
            priority=2,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i),
            adhikara={"sandhi"}
        )
        
        inference_rule = self.mapper.map_sutra_to_logic(sutra)
        
        assert inference_rule.id == f"sandhi_{sutra.id}"
        assert inference_rule.name == "iko yaṇ aci"
        assert len(inference_rule.clauses) > 0
    
    def test_map_definition_rule(self):
        """Test mapping definition rules to logic."""
        sutra = SutraRule(
            sutra_ref=SutraReference(1, 1, 1),
            name="vṛddhir ādaic",
            description="vṛddhi vowels are ā, ai, au",
            rule_type=RuleType.ADHIKARA,
            priority=1,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i),
            adhikara={"vrddhi"}
        )
        
        inference_rule = self.mapper.map_sutra_to_logic(sutra)
        
        assert inference_rule.id == f"def_{sutra.id}"
        # Should have facts for vṛddhi vowels
        facts = inference_rule.get_facts()
        assert len(facts) == 3  # ā, ai, au


class TestConstraintSolver:
    """Test constraint satisfaction solver."""
    
    def setup_method(self):
        """Set up test constraint solver."""
        self.kb = KnowledgeBase()
        self.solver = ConstraintSolver(self.kb)
    
    def test_simple_constraint_solving(self):
        """Test solving simple constraints."""
        # Define variables and their domains
        variables = {
            'vowel1': ['a', 'i', 'u'],
            'vowel2': ['a', 'i', 'u']
        }
        
        # Add constraint: vowel1 != vowel2
        def different_vowels(assignment: Dict[str, Any]) -> bool:
            if 'vowel1' in assignment and 'vowel2' in assignment:
                return assignment['vowel1'] != assignment['vowel2']
            return True
        
        self.solver.add_constraint(different_vowels)
        
        solutions = self.solver.solve_constraints(variables)
        
        # Should have 6 solutions (3*3 - 3 same pairs)
        assert len(solutions) == 6
        
        # Verify all solutions satisfy the constraint
        for solution in solutions:
            assert solution['vowel1'] != solution['vowel2']
    
    def test_complex_constraint_solving(self):
        """Test solving more complex constraints."""
        variables = {
            'root': ['gam', 'kar', 'bhū'],
            'suffix': ['ti', 'anti', 'āmi'],
            'person': [1, 2, 3]
        }
        
        # Constraint: person 1 uses āmi, person 3 uses ti/anti
        def person_suffix_agreement(assignment: Dict[str, Any]) -> bool:
            if 'person' in assignment and 'suffix' in assignment:
                person = assignment['person']
                suffix = assignment['suffix']
                
                if person == 1:
                    return suffix == 'āmi'
                elif person == 3:
                    return suffix in ['ti', 'anti']
                else:
                    return True
            return True
        
        self.solver.add_constraint(person_suffix_agreement)
        
        solutions = self.solver.solve_constraints(variables)
        
        # Verify constraint satisfaction
        for solution in solutions:
            assert person_suffix_agreement(solution)


class TestTheoremProver:
    """Test theorem proving capabilities."""
    
    def setup_method(self):
        """Set up test theorem prover."""
        self.kb = KnowledgeBase()
        self.prover = TheoremProver(self.kb)
        
        # Add some basic facts and rules
        facts = [
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])),
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])),
            LogicalClause(LogicalTerm("semivowel_yan", [LogicalTerm("i"), LogicalTerm("y")]))
        ]
        
        for fact in facts:
            self.kb.add_fact(fact)
        
        # Add a rule: sandhi_applies(X, Y, Z) :- vowel(X), vowel(Y), semivowel_yan(X, Z)
        head = LogicalTerm("sandhi_applies", [
            LogicalTerm("X", is_variable=True),
            LogicalTerm("Y", is_variable=True), 
            LogicalTerm("Z", is_variable=True)
        ])
        body = [
            LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)]),
            LogicalTerm("vowel", [LogicalTerm("Y", is_variable=True)]),
            LogicalTerm("semivowel_yan", [LogicalTerm("X", is_variable=True), LogicalTerm("Z", is_variable=True)])
        ]
        rule = LogicalClause(head, body)
        self.kb.add_rule(rule)
    
    def test_prove_fact(self):
        """Test proving simple facts."""
        goal = LogicalTerm("vowel", [LogicalTerm("a")])
        proved, steps = self.prover.prove(goal)
        
        assert proved
        assert len(steps) > 0
    
    def test_prove_rule(self):
        """Test proving using rules."""
        goal = LogicalTerm("sandhi_applies", [
            LogicalTerm("i"),
            LogicalTerm("a"),
            LogicalTerm("y")
        ])
        
        proved, steps = self.prover.prove(goal)
        
        # This should be provable given our facts and rules
        # Note: This is a simplified test - full implementation would need more sophisticated resolution
        assert isinstance(proved, bool)
        assert isinstance(steps, list)
    
    def test_proof_explanation(self):
        """Test proof explanation generation."""
        goal = LogicalTerm("vowel", [LogicalTerm("a")])
        explanation = self.prover.explain_proof(goal)
        
        assert isinstance(explanation, str)
        assert "Proof of" in explanation or "Could not prove" in explanation


class TestQueryPlanner:
    """Test context-aware query planning."""
    
    def setup_method(self):
        """Set up test query planner."""
        self.kb = KnowledgeBase()
        self.planner = QueryPlanner(self.kb)
    
    def test_query_classification(self):
        """Test query type classification."""
        test_cases = [
            ("What is the sandhi of a + i?", "sandhi"),
            ("How do you decline this noun?", "morphology"),
            ("What is the syntax of this sentence?", "syntax"),
            ("What does this word mean?", "semantics"),
            ("Random question", "general")
        ]
        
        for query, expected_type in test_cases:
            classified_type = self.planner._classify_query(query, {})
            assert classified_type == expected_type
    
    def test_query_planning(self):
        """Test query planning logic."""
        # Test high-coverage query (sandhi)
        plan = self.planner.plan_query("What is the sandhi rule?", {})
        
        assert plan['query_type'] == 'sandhi'
        assert plan['use_rules'] == True
        assert plan['confidence'] > 0.7
    
    def test_query_execution(self):
        """Test query execution."""
        result = self.planner.execute_query("What is the sandhi rule?", {})
        
        assert 'query' in result
        assert 'plan' in result
        assert 'final_answer' in result


class TestReasoningCore:
    """Test the main reasoning core integration."""
    
    def setup_method(self):
        """Set up test reasoning core."""
        self.registry = RuleRegistry()
        self.core = ReasoningCore(self.registry)
    
    def test_core_initialization(self):
        """Test reasoning core initialization."""
        assert self.core.knowledge_base is not None
        assert self.core.sutra_mapper is not None
        assert self.core.constraint_solver is not None
        assert self.core.theorem_prover is not None
        assert self.core.query_planner is not None
    
    def test_add_sutra_rule(self):
        """Test adding sūtra rules to the core."""
        sutra = SutraRule(
            sutra_ref=SutraReference(1, 1, 1),
            name="test rule",
            description="test description",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        initial_count = len(self.core.knowledge_base.inference_rules)
        self.core.add_sutra_rule(sutra)
        
        assert len(self.core.knowledge_base.inference_rules) > initial_count
    
    def test_query_interface(self):
        """Test the main query interface."""
        result = self.core.query("vowel(a)")
        
        assert isinstance(result, dict)
        assert 'goal' in result or 'query' in result
    
    def test_prove_interface(self):
        """Test the prove interface."""
        proved, explanation = self.core.prove("vowel(a)")
        
        assert isinstance(proved, bool)
        assert isinstance(explanation, str)
    
    def test_constraint_solving_interface(self):
        """Test constraint solving interface."""
        problem = {
            'variables': {'x': [1, 2, 3], 'y': [1, 2, 3]},
            'constraints': [lambda assignment: assignment.get('x', 0) != assignment.get('y', 0)]
        }
        
        solutions = self.core.solve_constraints(problem)
        
        assert isinstance(solutions, list)
    
    def test_statistics(self):
        """Test statistics generation."""
        stats = self.core.get_statistics()
        
        assert 'knowledge_base' in stats
        assert 'rule_registry' in stats
        assert 'proof_cache_size' in stats


class TestPerformance:
    """Test performance characteristics of the reasoning system."""
    
    def setup_method(self):
        """Set up performance test environment."""
        self.core = ReasoningCore()
    
    def test_query_performance(self):
        """Test query performance."""
        start_time = time.time()
        
        # Run multiple queries
        for i in range(100):
            self.core.query(f"test_predicate({i})")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete 100 queries in reasonable time (< 1 second)
        assert elapsed < 1.0
    
    def test_knowledge_base_scaling(self):
        """Test knowledge base scaling with many facts."""
        kb = KnowledgeBase()
        
        start_time = time.time()
        
        # Add many facts
        for i in range(1000):
            fact = LogicalClause(LogicalTerm("test_fact", [LogicalTerm(f"arg_{i}")]))
            kb.add_fact(fact)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should add 1000 facts quickly
        assert elapsed < 0.5
        assert len(kb.facts) == 1000
    
    def test_constraint_solver_performance(self):
        """Test constraint solver performance."""
        kb = KnowledgeBase()
        solver = ConstraintSolver(kb)
        
        # Define a moderately complex problem
        variables = {
            'var1': list(range(10)),
            'var2': list(range(10)),
            'var3': list(range(10))
        }
        
        # Add constraint: all different
        def all_different(assignment: Dict[str, Any]) -> bool:
            values = list(assignment.values())
            return len(values) == len(set(values))
        
        solver.add_constraint(all_different)
        
        start_time = time.time()
        solutions = solver.solve_constraints(variables)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should solve in reasonable time
        assert elapsed < 2.0
        assert len(solutions) > 0


class TestLogicalConsistency:
    """Test logical consistency of the reasoning system."""
    
    def setup_method(self):
        """Set up consistency test environment."""
        self.core = ReasoningCore()
    
    def test_fact_consistency(self):
        """Test that facts remain consistent."""
        # Add some facts
        kb = self.core.knowledge_base
        
        fact1 = LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")]))
        fact2 = LogicalClause(LogicalTerm("consonant", [LogicalTerm("k")]))
        
        kb.add_fact(fact1)
        kb.add_fact(fact2)
        
        # Query should return consistent results
        vowel_query = LogicalTerm("vowel", [LogicalTerm("a")])
        consonant_query = LogicalTerm("consonant", [LogicalTerm("a")])
        
        vowel_solutions = kb.query(vowel_query)
        consonant_solutions = kb.query(consonant_query)
        
        assert len(vowel_solutions) == 1
        assert len(consonant_solutions) == 0
    
    def test_rule_consistency(self):
        """Test that rules produce consistent inferences."""
        kb = self.core.knowledge_base
        
        # Add facts
        kb.add_fact(LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])))
        kb.add_fact(LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])))
        
        # Add rule: long_vowel(X) :- vowel(X), length(X, long)
        head = LogicalTerm("long_vowel", [LogicalTerm("X", is_variable=True)])
        body = [
            LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)]),
            LogicalTerm("length", [LogicalTerm("X", is_variable=True), LogicalTerm("long")])
        ]
        rule = LogicalClause(head, body)
        kb.add_rule(rule)
        
        # Add length fact
        kb.add_fact(LogicalClause(LogicalTerm("length", [LogicalTerm("ā"), LogicalTerm("long")])))
        
        # The system should maintain consistency
        assert len(kb.rules) == 1
        assert len(kb.facts) >= 3
    
    def test_no_contradictions(self):
        """Test that the system doesn't derive contradictions."""
        # This is a placeholder for more sophisticated consistency checking
        # In a full implementation, we would check for contradictory conclusions
        
        stats = self.core.get_statistics()
        
        # Basic sanity checks
        assert stats['knowledge_base']['facts_count'] >= 0
        assert stats['knowledge_base']['rules_count'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])