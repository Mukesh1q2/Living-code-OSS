"""
Tests for multi-domain mapping system.

This module tests cross-domain semantic preservation and translation
accuracy between Sanskrit, programming languages, mathematical formulas,
and knowledge graphs.
"""

import pytest
import json
from unittest.mock import Mock, patch

from sanskrit_rewrite_engine.multi_domain_mapper import (
    DomainType, DomainMapping, AlgorithmicSanskritExpression,
    SanskritToProgrammingTranslator, SanskritToMathTranslator,
    SanskritToKnowledgeGraphTranslator, AlgorithmicSanskritDSL,
    ProgrammingLanguage
)
from sanskrit_rewrite_engine.bidirectional_mapper import (
    BidirectionalMapper, MultiDomainMapper
)


class TestDomainMapping:
    """Test domain mapping data structure."""
    
    def test_domain_mapping_creation(self):
        """Test creating a domain mapping."""
        mapping = DomainMapping(
            id="test_mapping",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="यदि सत्यम् तदा कार्यम्",
            target_content="if True:\n    action()",
            mapping_type="sanskrit_to_programming"
        )
        
        assert mapping.id == "test_mapping"
        assert mapping.source_domain == DomainType.SANSKRIT
        assert mapping.target_domain == DomainType.PROGRAMMING
        assert mapping.confidence == 1.0  # default value
    
    def test_domain_mapping_to_dict(self):
        """Test converting domain mapping to dictionary."""
        mapping = DomainMapping(
            id="test_mapping",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="test",
            target_content="test_output",
            mapping_type="test_type"
        )
        
        mapping_dict = mapping.to_dict()
        
        assert mapping_dict['id'] == "test_mapping"
        assert mapping_dict['source_domain'] == "sanskrit"
        assert mapping_dict['target_domain'] == "programming"
        assert 'created_at' in mapping_dict


class TestSanskritToProgrammingTranslator:
    """Test Sanskrit to programming language translation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = SanskritToProgrammingTranslator()
    
    def test_conditional_translation_python(self):
        """Test translating Sanskrit conditional to Python."""
        sanskrit_code = "यदि x > 0 तदा print('positive')"
        
        mapping = self.translator.translate(
            sanskrit_code, 
            DomainType.SANSKRIT, 
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        assert mapping.source_domain == DomainType.SANSKRIT
        assert mapping.target_domain == DomainType.PROGRAMMING
        assert "if" in mapping.target_content
        assert mapping.metadata['target_language'] == 'python'    

    def test_function_translation_python(self):
        """Test translating Sanskrit function to Python."""
        sanskrit_code = "कार्यम् add x y = x + y"
        
        mapping = self.translator.translate(
            sanskrit_code,
            DomainType.SANSKRIT,
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        assert "def" in mapping.target_content
        assert "add" in mapping.target_content
    
    def test_arithmetic_translation(self):
        """Test translating Sanskrit arithmetic to code."""
        sanskrit_code = "योगः a b"
        
        mapping = self.translator.translate(
            sanskrit_code,
            DomainType.SANSKRIT,
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        assert "+" in mapping.target_content
    
    def test_validation_python_syntax(self):
        """Test validation of Python syntax."""
        valid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="test",
            target_content="x = 1\nprint(x)",
            mapping_type="test",
            metadata={'target_language': 'python'}
        )
        
        assert self.translator.validate_translation(valid_mapping) == True
        
        invalid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="test",
            target_content="x = 1\nprint(x",  # Missing closing parenthesis
            mapping_type="test",
            metadata={'target_language': 'python'}
        )
        
        assert self.translator.validate_translation(invalid_mapping) == False


class TestSanskritToMathTranslator:
    """Test Sanskrit to mathematical formula translation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = SanskritToMathTranslator()
    
    def test_arithmetic_translation(self):
        """Test translating Sanskrit arithmetic to math."""
        sanskrit_math = "योगः x y"
        
        mapping = self.translator.translate(
            sanskrit_math,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        assert "+" in mapping.target_content
        assert mapping.metadata['formula_type'] == 'arithmetic'
    
    def test_trigonometric_translation(self):
        """Test translating Sanskrit trigonometric functions."""
        sanskrit_math = "ज्या x"
        
        mapping = self.translator.translate(
            sanskrit_math,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        assert "sin" in mapping.target_content
        assert mapping.metadata['formula_type'] == 'trigonometry'
    
    def test_formula_validation(self):
        """Test mathematical formula validation."""
        valid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.MATHEMATICS,
            source_content="test",
            target_content="x + y = z",
            mapping_type="test"
        )
        
        assert self.translator.validate_translation(valid_mapping) == True
        
        invalid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.MATHEMATICS,
            source_content="test",
            target_content="x + y = z)",  # Unbalanced parentheses
            mapping_type="test"
        )
        
        assert self.translator.validate_translation(invalid_mapping) == False


class TestSanskritToKnowledgeGraphTranslator:
    """Test Sanskrit to knowledge graph translation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = SanskritToKnowledgeGraphTranslator()
    
    def test_entity_recognition(self):
        """Test recognizing entities in Sanskrit text."""
        sanskrit_text = "गुरुः शिष्यः पठति"
        
        mapping = self.translator.translate(
            sanskrit_text,
            DomainType.SANSKRIT,
            DomainType.KNOWLEDGE_GRAPH
        )
        
        # Parse the JSON output
        graph_data = json.loads(mapping.target_content)
        
        assert 'nodes' in graph_data
        assert 'edges' in graph_data
        assert mapping.metadata['node_count'] >= 0
        assert mapping.metadata['edge_count'] >= 0
    
    def test_knowledge_graph_validation(self):
        """Test knowledge graph validation."""
        valid_graph = {
            'id': 'test',
            'nodes': {},
            'edges': {},
            'source_text': 'test'
        }
        
        valid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.KNOWLEDGE_GRAPH,
            source_content="test",
            target_content=json.dumps(valid_graph),
            mapping_type="test"
        )
        
        assert self.translator.validate_translation(valid_mapping) == True
        
        invalid_mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.KNOWLEDGE_GRAPH,
            source_content="test",
            target_content="invalid json",
            mapping_type="test"
        )
        
        assert self.translator.validate_translation(invalid_mapping) == False


class TestAlgorithmicSanskritDSL:
    """Test algorithmic Sanskrit domain-specific language."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dsl = AlgorithmicSanskritDSL()
    
    def test_conditional_parsing(self):
        """Test parsing conditional expressions."""
        expression = "यदि x > 0 तदा print(x) अन्यथा print('zero')"
        
        parsed = self.dsl.parse_expression(expression)
        
        assert parsed.semantic_type == 'conditional'
        assert parsed.expression == expression
        assert len(parsed.parameters) > 0
    
    def test_function_parsing(self):
        """Test parsing function definitions."""
        expression = "कार्यम् factorial n = n * factorial(n-1)"
        
        parsed = self.dsl.parse_expression(expression)
        
        assert parsed.semantic_type == 'function'
        assert 'factorial' in parsed.expression
    
    def test_expression_validation(self):
        """Test validating algorithmic Sanskrit expressions."""
        conditional_expr = AlgorithmicSanskritExpression(
            expression="यदि x > 0 तदा action",
            semantic_type="conditional"
        )
        
        assert self.dsl.validate_expression(conditional_expr) == True
        
        invalid_conditional = AlgorithmicSanskritExpression(
            expression="invalid conditional",
            semantic_type="conditional"
        )
        
        assert self.dsl.validate_expression(invalid_conditional) == False
    
    def test_compile_to_python(self):
        """Test compiling to Python."""
        conditional_expr = AlgorithmicSanskritExpression(
            expression="यदि x > 0 तदा print('positive')",
            semantic_type="conditional"
        )
        
        python_code = self.dsl.compile_to_target(
            conditional_expr, 
            DomainType.PROGRAMMING,
            language='python'
        )
        
        assert "if" in python_code
        assert "print" in python_code


class TestBidirectionalMapper:
    """Test bidirectional mapping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = BidirectionalMapper()
    
    def test_reverse_translation_patterns(self):
        """Test reverse translation using patterns."""
        code = "if x > 0:\n    print('positive')"
        
        mapping = self.mapper.translate_reverse(
            code,
            DomainType.PROGRAMMING,
            DomainType.SANSKRIT
        )
        
        assert "यदि" in mapping.target_content  # 'if' should be translated to 'यदि'
        assert mapping.mapping_type == "reverse_translation"
    
    def test_bidirectional_consistency(self):
        """Test consistency of bidirectional translation."""
        original_content = "if x > 0"
        
        forward_mapping = DomainMapping(
            id="forward",
            source_domain=DomainType.PROGRAMMING,
            target_domain=DomainType.SANSKRIT,
            source_content=original_content,
            target_content="यदि x > 0",
            mapping_type="forward"
        )
        
        reverse_mapping = DomainMapping(
            id="reverse",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="यदि x > 0",
            target_content="if x > 0",
            mapping_type="reverse"
        )
        
        consistency = self.mapper.validate_bidirectional_consistency(
            forward_mapping, reverse_mapping
        )
        
        assert consistency > 0.0  # Should have some similarity
    
    def test_similarity_calculation(self):
        """Test text similarity calculation."""
        text1 = "if x > 0"
        text2 = "if x > 0"
        
        similarity = self.mapper._calculate_similarity(text1, text2)
        assert similarity == 1.0  # Identical texts
        
        text3 = "while x > 0"
        similarity2 = self.mapper._calculate_similarity(text1, text3)
        assert 0.0 < similarity2 < 1.0  # Partial similarity


class TestMultiDomainMapper:
    """Test the main multi-domain mapping system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = MultiDomainMapper()
    
    def test_sanskrit_to_programming_translation(self):
        """Test Sanskrit to programming translation."""
        sanskrit_code = "यदि x > 0"
        
        mapping = self.mapper.translate(
            sanskrit_code,
            DomainType.SANSKRIT,
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        assert mapping.source_domain == DomainType.SANSKRIT
        assert mapping.target_domain == DomainType.PROGRAMMING
        assert len(self.mapper.mapping_history) == 1
    
    def test_sanskrit_to_math_translation(self):
        """Test Sanskrit to mathematics translation."""
        sanskrit_math = "योगः x y"
        
        mapping = self.mapper.translate(
            sanskrit_math,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        assert "+" in mapping.target_content
        assert mapping.source_domain == DomainType.SANSKRIT
        assert mapping.target_domain == DomainType.MATHEMATICS
    
    def test_algorithmic_sanskrit_creation(self):
        """Test creating algorithmic Sanskrit expressions."""
        expression = "कार्यम् add x y = x + y"
        
        parsed = self.mapper.create_algorithmic_sanskrit(expression)
        
        assert parsed.semantic_type == 'function'
        assert 'add' in parsed.expression
    
    def test_algorithmic_sanskrit_compilation(self):
        """Test compiling algorithmic Sanskrit."""
        expression = self.mapper.create_algorithmic_sanskrit(
            "यदि x > 0 तदा print('positive')"
        )
        
        python_code = self.mapper.compile_algorithmic_sanskrit(
            expression,
            DomainType.PROGRAMMING,
            language='python'
        )
        
        assert "if" in python_code
    
    def test_semantic_preservation_validation(self):
        """Test semantic preservation validation."""
        mapping = DomainMapping(
            id="test",
            source_domain=DomainType.SANSKRIT,
            target_domain=DomainType.PROGRAMMING,
            source_content="यदि सत्यम्",
            target_content="if True:",
            mapping_type="test"
        )
        
        score = self.mapper.validate_semantic_preservation(mapping)
        
        assert 0.0 <= score <= 1.0
    
    def test_mapping_statistics(self):
        """Test getting mapping statistics."""
        # Perform some translations
        self.mapper.translate("योगः x y", DomainType.SANSKRIT, DomainType.MATHEMATICS)
        self.mapper.translate("यदि x", DomainType.SANSKRIT, DomainType.PROGRAMMING)
        
        stats = self.mapper.get_mapping_statistics()
        
        assert stats['total_mappings'] == 2
        assert 'by_source_domain' in stats
        assert 'by_target_domain' in stats
        assert 'average_confidence' in stats
    
    def test_export_import_mappings(self):
        """Test exporting and importing mappings."""
        # Create some mappings
        self.mapper.translate("योगः x y", DomainType.SANSKRIT, DomainType.MATHEMATICS)
        
        # Export mappings
        exported_data = self.mapper.export_mappings('json')
        
        # Create new mapper and import
        new_mapper = MultiDomainMapper()
        imported_count = new_mapper.import_mappings(exported_data, 'json')
        
        assert imported_count > 0
        assert len(new_mapper.mapping_history) == imported_count


class TestCrossDomainSemanticPreservation:
    """Test semantic preservation across domain translations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = MultiDomainMapper()
    
    def test_conditional_semantic_preservation(self):
        """Test that conditional logic is preserved across domains."""
        sanskrit_conditional = "यदि x > 0 तदा print('positive')"
        
        # Translate to programming
        prog_mapping = self.mapper.translate(
            sanskrit_conditional,
            DomainType.SANSKRIT,
            DomainType.PROGRAMMING,
            language=ProgrammingLanguage.PYTHON
        )
        
        # Check that conditional structure is preserved
        assert "if" in prog_mapping.target_content.lower()
        
        # Validate semantic preservation
        preservation_score = self.mapper.validate_semantic_preservation(prog_mapping)
        assert preservation_score > 0.5  # Should have reasonable preservation
    
    def test_mathematical_semantic_preservation(self):
        """Test that mathematical operations are preserved."""
        sanskrit_math = "योगः x y"
        
        # Translate to mathematics
        math_mapping = self.mapper.translate(
            sanskrit_math,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        # Check that addition operation is preserved
        assert "+" in math_mapping.target_content
        
        # Validate semantic preservation
        preservation_score = self.mapper.validate_semantic_preservation(math_mapping)
        assert preservation_score > 0.8  # Mathematical operations should preserve well
    
    def test_round_trip_consistency(self):
        """Test consistency of round-trip translations."""
        original_sanskrit = "योगः x y"
        
        # Sanskrit -> Math -> Sanskrit (via reverse patterns)
        math_mapping = self.mapper.translate(
            original_sanskrit,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        bidirectional_result = self.mapper.create_bidirectional_mapping(
            original_sanskrit,
            DomainType.SANSKRIT,
            DomainType.MATHEMATICS
        )
        
        assert bidirectional_result['consistency_score'] >= 0.0
        assert 'forward_mapping' in bidirectional_result
        assert 'reverse_mapping' in bidirectional_result


if __name__ == '__main__':
    pytest.main([__file__])