"""
Demonstration of the Sanskrit Reasoning Core with Prolog/Datalog backend.

This script demonstrates the key capabilities of the reasoning core:
1. Logic programming interface for Sanskrit rules
2. Mapping between Sanskrit rules and inference rules
3. Constraint satisfaction for grammatical problems
4. Theorem proving using sūtra-based inference
5. Context-aware query planning
"""

from sanskrit_rewrite_engine.reasoning_core import (
    ReasoningCore, LogicalTerm, LogicalClause, InferenceRule
)
from sanskrit_rewrite_engine.rule import (
    SutraRule, SutraReference, RuleType, RuleRegistry
)
from sanskrit_rewrite_engine.essential_sutras import create_essential_sutras


def demonstrate_logical_terms():
    """Demonstrate logical term creation and manipulation."""
    print("=== LOGICAL TERMS DEMONSTRATION ===")
    
    # Create simple terms
    vowel_a = LogicalTerm("vowel", [LogicalTerm("a")])
    print(f"Created term: {vowel_a}")
    
    # Create variable terms
    vowel_x = LogicalTerm("vowel", [LogicalTerm("X", is_variable=True)])
    print(f"Created variable term: {vowel_x}")
    
    # Demonstrate substitution
    bindings = {"X": LogicalTerm("i")}
    substituted = vowel_x.substitute(bindings)
    print(f"After substitution X->i: {substituted}")
    
    print()


def demonstrate_knowledge_base():
    """Demonstrate knowledge base operations."""
    print("=== KNOWLEDGE BASE DEMONSTRATION ===")
    
    from sanskrit_rewrite_engine.reasoning_core import KnowledgeBase
    
    kb = KnowledgeBase()
    
    # Add facts
    facts = [
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])),
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])),
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("u")])),
        LogicalClause(LogicalTerm("consonant", [LogicalTerm("k")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("ā")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("ai")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("au")]))
    ]
    
    for fact in facts:
        kb.add_fact(fact)
        print(f"Added fact: {fact}")
    
    # Add a rule
    head = LogicalTerm("long_vowel", [LogicalTerm("X", is_variable=True)])
    body = [LogicalTerm("vrddhi_vowel", [LogicalTerm("X", is_variable=True)])]
    rule = LogicalClause(head, body)
    kb.add_rule(rule)
    print(f"Added rule: {rule}")
    
    # Query the knowledge base
    print("\nQuerying knowledge base:")
    goal = LogicalTerm("vowel", [LogicalTerm("a")])
    solutions = kb.query(goal)
    print(f"Query {goal} -> {len(solutions)} solutions")
    
    goal = LogicalTerm("vrddhi_vowel", [LogicalTerm("ā")])
    solutions = kb.query(goal)
    print(f"Query {goal} -> {len(solutions)} solutions")
    
    print()


def demonstrate_sutra_mapping():
    """Demonstrate mapping Sanskrit sūtras to logical rules."""
    print("=== SŪTRA TO LOGIC MAPPING DEMONSTRATION ===")
    
    from sanskrit_rewrite_engine.reasoning_core import SutraToLogicMapper
    
    mapper = SutraToLogicMapper()
    
    # Create a test sandhi rule
    sandhi_sutra = SutraRule(
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
    inference_rule = mapper.map_sutra_to_logic(sandhi_sutra)
    
    print(f"Mapped sūtra: {sandhi_sutra.name}")
    print(f"Inference rule ID: {inference_rule.id}")
    print(f"Number of clauses: {len(inference_rule.clauses)}")
    
    for i, clause in enumerate(inference_rule.clauses):
        print(f"  Clause {i+1}: {clause}")
    
    # Create a definition rule
    definition_sutra = SutraRule(
        sutra_ref=SutraReference(1, 1, 1),
        name="vṛddhir ādaic",
        description="vṛddhi vowels are ā, ai, au",
        rule_type=RuleType.ADHIKARA,
        priority=1,
        match_fn=lambda tokens, i: False,
        apply_fn=lambda tokens, i: (tokens, i),
        adhikara={"vrddhi"}
    )
    
    definition_rule = mapper.map_sutra_to_logic(definition_sutra)
    print(f"\nMapped definition: {definition_sutra.name}")
    print(f"Number of facts: {len(definition_rule.get_facts())}")
    
    for fact in definition_rule.get_facts():
        print(f"  Fact: {fact}")
    
    print()


def demonstrate_constraint_solving():
    """Demonstrate constraint satisfaction solving."""
    print("=== CONSTRAINT SOLVING DEMONSTRATION ===")
    
    from sanskrit_rewrite_engine.reasoning_core import ConstraintSolver, KnowledgeBase
    
    kb = KnowledgeBase()
    solver = ConstraintSolver(kb)
    
    # Define a Sanskrit sandhi problem
    variables = {
        'first_vowel': ['a', 'i', 'u'],
        'second_vowel': ['a', 'i', 'u'],
        'result': ['ā', 'e', 'o', 'ai', 'au']
    }
    
    # Add Sanskrit sandhi constraints
    def guṇa_constraint(assignment):
        """Constraint for guṇa sandhi: a + i = e, a + u = o"""
        first = assignment.get('first_vowel')
        second = assignment.get('second_vowel')
        result = assignment.get('result')
        
        if first == 'a' and second == 'i':
            return result == 'e'
        elif first == 'a' and second == 'u':
            return result == 'o'
        return True
    
    def similar_vowel_constraint(assignment):
        """Constraint for similar vowel lengthening: a + a = ā"""
        first = assignment.get('first_vowel')
        second = assignment.get('second_vowel')
        result = assignment.get('result')
        
        if first == 'a' and second == 'a':
            return result == 'ā'
        return True
    
    solver.add_constraint(guṇa_constraint)
    solver.add_constraint(similar_vowel_constraint)
    
    print("Solving Sanskrit sandhi constraints...")
    print("Variables:", variables)
    print("Constraints: guṇa sandhi (a+i=e, a+u=o) and similar vowel lengthening (a+a=ā)")
    
    solutions = solver.solve_constraints(variables)
    
    print(f"\nFound {len(solutions)} valid solutions:")
    for i, solution in enumerate(solutions[:10]):  # Show first 10
        print(f"  {i+1}: {solution['first_vowel']} + {solution['second_vowel']} = {solution['result']}")
    
    if len(solutions) > 10:
        print(f"  ... and {len(solutions) - 10} more solutions")
    
    print()


def demonstrate_theorem_proving():
    """Demonstrate theorem proving capabilities."""
    print("=== THEOREM PROVING DEMONSTRATION ===")
    
    from sanskrit_rewrite_engine.reasoning_core import TheoremProver, KnowledgeBase
    
    kb = KnowledgeBase()
    prover = TheoremProver(kb)
    
    # Add Sanskrit grammatical facts
    facts = [
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])),
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])),
        LogicalClause(LogicalTerm("vowel", [LogicalTerm("u")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("ā")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("ai")])),
        LogicalClause(LogicalTerm("vrddhi_vowel", [LogicalTerm("au")])),
        LogicalClause(LogicalTerm("guna_vowel", [LogicalTerm("a")])),
        LogicalClause(LogicalTerm("guna_vowel", [LogicalTerm("e")])),
        LogicalClause(LogicalTerm("guna_vowel", [LogicalTerm("o")]))
    ]
    
    for fact in facts:
        kb.add_fact(fact)
    
    # Add rules
    # Rule: strong_vowel(X) :- vrddhi_vowel(X)
    head = LogicalTerm("strong_vowel", [LogicalTerm("X", is_variable=True)])
    body = [LogicalTerm("vrddhi_vowel", [LogicalTerm("X", is_variable=True)])]
    rule1 = LogicalClause(head, body)
    kb.add_rule(rule1)
    
    # Rule: strong_vowel(X) :- guna_vowel(X)
    head = LogicalTerm("strong_vowel", [LogicalTerm("X", is_variable=True)])
    body = [LogicalTerm("guna_vowel", [LogicalTerm("X", is_variable=True)])]
    rule2 = LogicalClause(head, body)
    kb.add_rule(rule2)
    
    print("Knowledge base contains:")
    print(f"  {len(kb.facts)} facts")
    print(f"  {len(kb.rules)} rules")
    
    # Test theorem proving
    test_goals = [
        LogicalTerm("vowel", [LogicalTerm("a")]),
        LogicalTerm("vrddhi_vowel", [LogicalTerm("ā")]),
        LogicalTerm("strong_vowel", [LogicalTerm("ā")]),
        LogicalTerm("strong_vowel", [LogicalTerm("e")])
    ]
    
    print("\nTesting theorem proving:")
    for goal in test_goals:
        proved, steps = prover.prove(goal)
        print(f"  Goal: {goal}")
        print(f"  Proved: {proved}")
        if proved and steps:
            print(f"  Steps: {len(steps)}")
        print()


def demonstrate_query_planning():
    """Demonstrate context-aware query planning."""
    print("=== QUERY PLANNING DEMONSTRATION ===")
    
    from sanskrit_rewrite_engine.reasoning_core import QueryPlanner, KnowledgeBase
    
    kb = KnowledgeBase()
    planner = QueryPlanner(kb)
    
    # Test different types of queries
    test_queries = [
        "What is the sandhi of a + i?",
        "How do you decline this noun?",
        "What is the syntax of this sentence?",
        "What does this word mean?",
        "Explain the morphological analysis",
        "Random grammatical question"
    ]
    
    print("Testing query classification and planning:")
    for query in test_queries:
        plan = planner.plan_query(query, {})
        
        print(f"\nQuery: '{query}'")
        print(f"  Type: {plan['query_type']}")
        print(f"  Use rules: {plan['use_rules']}")
        print(f"  Use LLM: {plan['use_llm']}")
        print(f"  Confidence: {plan['confidence']:.2f}")
        print(f"  Reasoning: {plan['reasoning']}")


def demonstrate_full_reasoning_core():
    """Demonstrate the complete reasoning core integration."""
    print("=== FULL REASONING CORE DEMONSTRATION ===")
    
    # Create reasoning core with essential sutras
    registry = RuleRegistry()
    essential_sutras = create_essential_sutras()
    
    print(f"Loading {len(essential_sutras)} essential sūtras...")
    for sutra in essential_sutras:
        registry.add_sutra_rule(sutra)
    
    core = ReasoningCore(registry)
    
    # Show statistics
    stats = core.get_statistics()
    print(f"\nReasoning core statistics:")
    print(f"  Knowledge base facts: {stats['knowledge_base']['facts_count']}")
    print(f"  Knowledge base rules: {stats['knowledge_base']['rules_count']}")
    print(f"  Inference rules: {stats['knowledge_base']['inference_rules_count']}")
    print(f"  Total sūtra rules: {stats['rule_registry']['total_sutra_rules']}")
    
    # Test queries
    print(f"\nTesting queries:")
    test_queries = [
        "vowel(a)",
        "vrddhi_vowel(ā)",
        "What is sandhi?",
        "How does morphology work?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = core.query(query)
        
        if 'solutions' in result:
            print(f"  Found {len(result['solutions'])} solutions")
        elif 'final_answer' in result:
            print(f"  Answer: {result['final_answer']}")
        else:
            print(f"  Result type: {result.get('method', 'unknown')}")
    
    # Test theorem proving
    print(f"\nTesting theorem proving:")
    test_goals = ["vowel(a)", "vrddhi_vowel(ā)"]
    
    for goal in test_goals:
        proved, explanation = core.prove(goal)
        print(f"\nGoal: {goal}")
        print(f"Proved: {proved}")
        if explanation:
            print(f"Explanation: {explanation[:100]}...")


def main():
    """Run all demonstrations."""
    print("SANSKRIT REASONING CORE DEMONSTRATION")
    print("=" * 50)
    print()
    
    demonstrate_logical_terms()
    demonstrate_knowledge_base()
    demonstrate_sutra_mapping()
    demonstrate_constraint_solving()
    demonstrate_theorem_proving()
    demonstrate_query_planning()
    demonstrate_full_reasoning_core()
    
    print("=" * 50)
    print("DEMONSTRATION COMPLETE")
    print()
    print("The reasoning core successfully demonstrates:")
    print("✓ Logic programming interface for Sanskrit rules")
    print("✓ Mapping between Sanskrit sūtras and inference rules")
    print("✓ Constraint satisfaction for grammatical problems")
    print("✓ Theorem proving using sūtra-based inference")
    print("✓ Context-aware query planning (Sanskrit rules vs LLM calls)")
    print("✓ Integration with existing Sanskrit rule system")


if __name__ == "__main__":
    main()