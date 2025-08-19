#!/usr/bin/env python3
"""
Multi-Domain Mapping System Demonstration

This script demonstrates the cross-domain semantic mapping capabilities
of the Sanskrit Rewrite Engine, showing translations between Sanskrit,
programming languages, mathematical formulas, and knowledge graphs.
"""

from sanskrit_rewrite_engine.multi_domain_mapper import (
    MultiDomainMapper, DomainType, ProgrammingLanguage,
    AlgorithmicSanskritExpression
)
import json


def demonstrate_sanskrit_to_programming():
    """Demonstrate Sanskrit to programming language translation."""
    print("=" * 60)
    print("SANSKRIT → PROGRAMMING TRANSLATION")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test cases for Sanskrit to programming translation
    test_cases = [
        "यदि x > 0 तदा print('positive')",
        "कार्यम् add x y = x + y",
        "योगः a b",
    ]
    
    for sanskrit_code in test_cases:
        print(f"\nSanskrit: {sanskrit_code}")
        
        # Translate to Python
        python_mapping = mapper.translate(
            sanskrit_code,
            DomainType.SANSKRIT,
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        print(f"Python:   {repr(python_mapping.target_content)}")
        print(f"Confidence: {python_mapping.confidence}")
        print(f"Semantic Preservation: {mapper.validate_semantic_preservation(python_mapping)}")


def demonstrate_sanskrit_to_mathematics():
    """Demonstrate Sanskrit to mathematical formula translation."""
    print("\n" + "=" * 60)
    print("SANSKRIT → MATHEMATICS TRANSLATION")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test cases for Sanskrit to mathematics translation
    test_cases = [
        "योगः x y",
        "गुणनम् a b",
        "ज्या theta",
        "कोज्या alpha",
    ]
    
    for sanskrit_math in test_cases:
        print(f"\nSanskrit: {sanskrit_math}")
        
        math_mapping = mapper.translate(
            sanskrit_math,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        print(f"Mathematics: {math_mapping.target_content}")
        print(f"Formula Type: {math_mapping.metadata.get('formula_type', 'unknown')}")
        print(f"Confidence: {math_mapping.confidence}")


def demonstrate_sanskrit_to_knowledge_graph():
    """Demonstrate Sanskrit to knowledge graph translation."""
    print("\n" + "=" * 60)
    print("SANSKRIT → KNOWLEDGE GRAPH TRANSLATION")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test cases for Sanskrit to knowledge graph translation
    test_cases = [
        "गुरुः शिष्यः धर्मः",
        "राजा प्रजा न्याय",
    ]
    
    for sanskrit_text in test_cases:
        print(f"\nSanskrit: {sanskrit_text}")
        
        kg_mapping = mapper.translate(
            sanskrit_text,
            DomainType.SANSKRIT,
            DomainType.KNOWLEDGE_GRAPH
        )
        
        # Parse and display the knowledge graph
        try:
            graph_data = json.loads(kg_mapping.target_content)
            print(f"Knowledge Graph:")
            print(f"  Nodes: {len(graph_data.get('nodes', {}))}")
            print(f"  Edges: {len(graph_data.get('edges', {}))}")
            print(f"  Graph ID: {graph_data.get('id', 'unknown')}")
        except:
            print(f"Knowledge Graph: {kg_mapping.target_content[:100]}...")
        
        print(f"Confidence: {kg_mapping.confidence}")


def demonstrate_algorithmic_sanskrit_dsl():
    """Demonstrate the Algorithmic Sanskrit DSL."""
    print("\n" + "=" * 60)
    print("ALGORITHMIC SANSKRIT DSL")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test cases for Algorithmic Sanskrit DSL
    test_expressions = [
        "यदि x > 0 तदा print('positive')",
        "कार्यम् factorial n = n * factorial(n-1)",
        "योगः sum_result x y",
    ]
    
    for expression_text in test_expressions:
        print(f"\nAlgorithmic Sanskrit: {expression_text}")
        
        # Parse the expression
        expression = mapper.create_algorithmic_sanskrit(expression_text)
        
        print(f"Semantic Type: {expression.semantic_type}")
        print(f"Parameters: {expression.parameters}")
        print(f"Documentation: {expression.documentation}")
        
        # Compile to Python
        try:
            python_code = mapper.compile_algorithmic_sanskrit(
                expression,
                DomainType.PROGRAMMING,
                language='python'
            )
            print(f"Compiled Python: {repr(python_code)}")
        except Exception as e:
            print(f"Compilation Error: {e}")


def demonstrate_bidirectional_mapping():
    """Demonstrate bidirectional mapping and consistency validation."""
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL MAPPING & CONSISTENCY")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test bidirectional mapping
    sanskrit_text = "योगः x y"
    
    print(f"Original Sanskrit: {sanskrit_text}")
    
    # Create bidirectional mapping
    bidirectional_result = mapper.create_bidirectional_mapping(
        sanskrit_text,
        DomainType.SANSKRIT,
        DomainType.MATHEMATICS
    )
    
    forward_mapping = bidirectional_result['forward_mapping']
    reverse_mapping = bidirectional_result['reverse_mapping']
    
    print(f"Forward Translation: {forward_mapping.target_content}")
    print(f"Reverse Translation: {reverse_mapping.target_content}")
    print(f"Consistency Score: {bidirectional_result['consistency_score']:.2f}")
    print(f"Is Consistent: {bidirectional_result['is_consistent']}")


def demonstrate_semantic_preservation():
    """Demonstrate cross-domain semantic preservation testing."""
    print("\n" + "=" * 60)
    print("SEMANTIC PRESERVATION TESTING")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Test semantic preservation across different domains
    test_cases = [
        ("यदि x > 0 तदा action", DomainType.PROGRAMMING),
        ("योगः a b", DomainType.MATHEMATICS),
        ("गुरुः शिष्यः", DomainType.KNOWLEDGE_GRAPH),
    ]
    
    for sanskrit_text, target_domain in test_cases:
        print(f"\nSanskrit: {sanskrit_text}")
        print(f"Target Domain: {target_domain.value}")
        
        # Translate
        if target_domain == DomainType.PROGRAMMING:
            mapping = mapper.translate(
                sanskrit_text, DomainType.SANSKRIT, target_domain,
                language=ProgrammingLanguage.PYTHON
            )
        else:
            mapping = mapper.translate(
                sanskrit_text, DomainType.SANSKRIT, target_domain
            )
        
        print(f"Translation: {mapping.target_content}")
        
        # Validate semantic preservation
        preservation_score = mapper.validate_semantic_preservation(mapping)
        print(f"Semantic Preservation Score: {preservation_score:.2f}")
        
        if preservation_score > 0.7:
            print("✓ High semantic preservation")
        elif preservation_score > 0.5:
            print("~ Moderate semantic preservation")
        else:
            print("✗ Low semantic preservation")


def demonstrate_mapping_statistics():
    """Demonstrate mapping statistics and analytics."""
    print("\n" + "=" * 60)
    print("MAPPING STATISTICS & ANALYTICS")
    print("=" * 60)
    
    mapper = MultiDomainMapper()
    
    # Perform several translations to generate statistics
    test_translations = [
        ("यदि x > 0", DomainType.PROGRAMMING, {'language': ProgrammingLanguage.PYTHON}),
        ("योगः a b", DomainType.MATHEMATICS, {}),
        ("गुरुः शिष्यः", DomainType.KNOWLEDGE_GRAPH, {}),
        ("कार्यम् test", DomainType.PROGRAMMING, {'language': ProgrammingLanguage.PYTHON}),
        ("ज्या theta", DomainType.MATHEMATICS, {}),
    ]
    
    for sanskrit_text, target_domain, kwargs in test_translations:
        mapper.translate(sanskrit_text, DomainType.SANSKRIT, target_domain, **kwargs)
    
    # Get and display statistics
    stats = mapper.get_mapping_statistics()
    
    print(f"Total Mappings: {stats['total_mappings']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    
    print("\nBy Source Domain:")
    for domain, count in stats['by_source_domain'].items():
        print(f"  {domain}: {count}")
    
    print("\nBy Target Domain:")
    for domain, count in stats['by_target_domain'].items():
        print(f"  {domain}: {count}")
    
    print("\nBy Mapping Type:")
    for mapping_type, count in stats['by_mapping_type'].items():
        print(f"  {mapping_type}: {count}")
    
    # Demonstrate export/import
    print("\nExporting mappings...")
    exported_data = mapper.export_mappings('json')
    print(f"Exported {len(json.loads(exported_data))} mappings")
    
    # Create new mapper and import
    new_mapper = MultiDomainMapper()
    imported_count = new_mapper.import_mappings(exported_data, 'json')
    print(f"Imported {imported_count} mappings into new mapper")


def main():
    """Run all demonstrations."""
    print("MULTI-DOMAIN MAPPING SYSTEM DEMONSTRATION")
    print("Sanskrit Rewrite Engine - Cross-Domain Semantic Translation")
    print("=" * 80)
    
    try:
        demonstrate_sanskrit_to_programming()
        demonstrate_sanskrit_to_mathematics()
        demonstrate_sanskrit_to_knowledge_graph()
        demonstrate_algorithmic_sanskrit_dsl()
        demonstrate_bidirectional_mapping()
        demonstrate_semantic_preservation()
        demonstrate_mapping_statistics()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()