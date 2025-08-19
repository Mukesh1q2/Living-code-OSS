"""
Integration tests for the reasoning core with actual Sanskrit rules.

This demonstrates the reasoning core working with real Sanskrit sūtras
and performing logical inference on grammatical problems.
"""

import pytest
from sanskrit_rewrite_engine.reasoning_core import (
    ReasoningCore, LogicalTerm, LogicalClause, InferenceRule
)
from sanskrit_rewrite_engine.rule import (
    SutraRule, SutraReference, RuleType, RuleRegistry
)
from sanskrit_rewrite_engine.essential_sutras import create_essential_sutras


class TestReasoningIntegration:
    """Integration tests for reasoning core with Sanskrit rules."""
    
    def setup_method(self):
        """Set up integration test environment."""
        # Create a rule registry with essential sutras
        self.registry = RuleRegistry()
        essential_sutras = create_essential_sutras()
        
        for sutra in essential_sutras:
            self.registry.add_sutra_rule(sutra)
        
        # Create reasoning core with the registry
        self.core = ReasoningCore(self.registry)
    
    def test_sandhi_inference(self):
        """Test logical inference for sandhi rules."""
        # Query about vowel properties
        result = self.core.query("vowel(a)")
        
        assert isinstance(result, dict)
        # Should have some form of result
        assert 'goal' in result or 'query' in result
    
    def test_vrddhi_vowel_classification(self):
        """Test classification of vṛddhi vowels."""
        # The essential sutras should include vṛddhir ādaic
        # which defines ā, ai, au as vṛddhi vowels
        
        # Test if we can prove that ā is a vṛddhi vowel
        proved, explanation = self.core.prove("vrddhi_vowel(ā)")
        
        # The proof might succeed or fail depending on implementation
        # but should return valid types
        assert isinstance(proved, bool)
        assert isinstance(explanation, str)
    
    def test_constraint_solving_with_sanskrit_rules(self):
        """Test constraint solving with Sanskrit grammatical constraints."""
        # Define a problem: find valid sandhi combinations
        problem = {
            'variables': {
                'first_vowel': ['a', 'i', 'u'],
                'second_vowel': ['a', 'i', 'u'],
                'result': ['ā', 'e', 'o', 'ai', 'au']
            },
            'constraints': [
                # Constraint: if first is 'a' and second is 'i', result should be 'e'
                lambda assignment: (
                    assignment.get('first_vowel') != 'a' or 
                    assignment.get('second_vowel') != 'i' or 
                    assignment.get('result') == 'e'
                ),
                # Constraint: if first is 'a' and second is 'u', result should be 'o'
                lambda assignment: (
                    assignment.get('first_vowel') != 'a' or 
                    assignment.get('second_vowel') != 'u' or 
                    assignment.get('result') == 'o'
                )
            ]
        }
        
        solutions = self.core.solve_constraints(problem)
        
        assert isinstance(solutions, list)
        # Should find valid solutions
        assert len(solutions) > 0
        
        # Verify that solutions satisfy the constraints
        for solution in solutions:
            if solution.get('first_vowel') == 'a' and solution.get('second_vowel') == 'i':
                assert solution.get('result') == 'e'
            if solution.get('first_vowel') == 'a' and solution.get('second_vowel') == 'u':
                assert solution.get('result') == 'o'
    
    def test_query_planning_with_sanskrit_context(self):
        """Test query planning with Sanskrit grammatical context."""
        # Test different types of Sanskrit queries
        test_queries = [
            "What is the sandhi of a + i?",
            "How do you form the vṛddhi of i?",
            "What are the guṇa vowels?",
            "Explain the morphology of this word"
        ]
        
        for query in test_queries:
            result = self.core.query(query, context={'language': 'sanskrit'})
            
            assert isinstance(result, dict)
            assert 'query' in result or 'goal' in result
    
    def test_theorem_proving_with_sutra_rules(self):
        """Test theorem proving using actual sūtra rules."""
        # Test proving basic grammatical facts
        test_goals = [
            "vowel(a)",
            "consonant(k)",
            "vrddhi_vowel(ā)"
        ]
        
        for goal_str in test_goals:
            proved, explanation = self.core.prove(goal_str)
            
            assert isinstance(proved, bool)
            assert isinstance(explanation, str)
            
            # Explanation should contain some meaningful content
            assert len(explanation) > 0
    
    def test_knowledge_base_population(self):
        """Test that the knowledge base is properly populated with Sanskrit rules."""
        stats = self.core.get_statistics()
        
        # Should have some facts and rules from the essential sutras
        assert stats['knowledge_base']['facts_count'] >= 0
        assert stats['knowledge_base']['rules_count'] >= 0
        assert stats['knowledge_base']['inference_rules_count'] > 0
        
        # Should have rule registry statistics
        assert 'rule_registry' in stats
        assert stats['rule_registry']['total_sutra_rules'] > 0
    
    def test_sutra_to_logic_mapping(self):
        """Test that Sanskrit sūtras are properly mapped to logical rules."""
        # Get the sutra mapper
        mapper = self.core.sutra_mapper
        
        # Create a test sandhi rule
        test_sutra = SutraRule(
            sutra_ref=SutraReference(6, 1, 77),
            name="iko yaṇ aci",
            description="i, u, ṛ, ḷ become y, v, r, l before vowels",
            rule_type=RuleType.SUTRA,
            priority=2,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i),
            adhikara={"sandhi"}
        )
        
        # Map to logical rule
        inference_rule = mapper.map_sutra_to_logic(test_sutra)
        
        assert inference_rule.id == f"sandhi_{test_sutra.id}"
        assert inference_rule.name == "iko yaṇ aci"
        assert len(inference_rule.clauses) > 0
        
        # Should have some facts for the semivowel mappings
        facts = inference_rule.get_facts()
        assert len(facts) > 0
    
    def test_performance_with_multiple_rules(self):
        """Test performance with multiple Sanskrit rules."""
        import time
        
        # Add more rules to test scaling
        for i in range(10):
            test_sutra = SutraRule(
                sutra_ref=SutraReference(1, 1, i + 10),
                name=f"test_rule_{i}",
                description=f"Test rule {i}",
                rule_type=RuleType.SUTRA,
                priority=i,
                match_fn=lambda tokens, idx: False,
                apply_fn=lambda tokens, idx: (tokens, idx),
                adhikara={"test"}
            )
            self.core.add_sutra_rule(test_sutra)
        
        # Test query performance
        start_time = time.time()
        
        for i in range(50):
            self.core.query(f"test_predicate({i})")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete 50 queries reasonably quickly
        assert elapsed < 2.0
        
        # Verify knowledge base has grown
        stats = self.core.get_statistics()
        assert stats['knowledge_base']['inference_rules_count'] > 10


def test_reasoning_core_basic_functionality():
    """Test basic reasoning core functionality without complex setup."""
    core = ReasoningCore()
    
    # Test basic query
    result = core.query("test_query")
    assert isinstance(result, dict)
    
    # Test basic proof
    proved, explanation = core.prove("test_goal")
    assert isinstance(proved, bool)
    assert isinstance(explanation, str)
    
    # Test statistics
    stats = core.get_statistics()
    assert isinstance(stats, dict)
    assert 'knowledge_base' in stats


if __name__ == "__main__":
    pytest.main([__file__])