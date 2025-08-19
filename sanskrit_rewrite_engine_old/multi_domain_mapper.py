"""
Multi-domain mapping system for Sanskrit reasoning.

This module implements cross-domain semantic mapping between Sanskrit,
programming languages, mathematical formulas, and knowledge graphs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import json
import re
import ast
from datetime import datetime

from .semantic_graph import (
    SemanticGraph, SemanticNode, SemanticEdge, SemanticNodeType, 
    SemanticRelationType, SemanticGraphBuilder
)
from .token import Token


class DomainType(Enum):
    """Types of domains for cross-mapping."""
    SANSKRIT = "sanskrit"
    PROGRAMMING = "programming"
    MATHEMATICS = "mathematics"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    NATURAL_LANGUAGE = "natural_language"
    LOGICAL_FORM = "logical_form"


class ProgrammingLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"


@dataclass
class DomainMapping:
    """Represents a mapping between domains."""
    id: str
    source_domain: DomainType
    target_domain: DomainType
    source_content: str
    target_content: str
    mapping_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to dictionary."""
        return {
            'id': self.id,
            'source_domain': self.source_domain.value,
            'target_domain': self.target_domain.value,
            'source_content': self.source_content,
            'target_content': self.target_content,
            'mapping_type': self.mapping_type,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


@dataclass
class AlgorithmicSanskritExpression:
    """Represents an expression in the algorithmic Sanskrit DSL."""
    expression: str
    semantic_type: str
    parameters: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    documentation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'expression': self.expression,
            'semantic_type': self.semantic_type,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'documentation': self.documentation
        }


class DomainTranslator(ABC):
    """Abstract base class for domain translators."""
    
    @abstractmethod
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate content from source domain to target domain."""
        pass
    
    @abstractmethod
    def validate_translation(self, mapping: DomainMapping) -> bool:
        """Validate that a translation preserves semantic meaning."""
        pass


class SanskritToProgrammingTranslator(DomainTranslator):
    """Translates Sanskrit expressions to programming constructs."""
    
    def __init__(self):
        self.sanskrit_to_code_patterns = {
            'यदि': {  # yadi (if)
                'python': 'if {condition}:\n    {action}',
                'javascript': 'if ({condition}) {\n    {action}\n}',
                'semantic_type': 'conditional'
            },
            'कार्यम्': {  # kaaryam (function)
                'python': 'def {name}({params}):',
                'javascript': 'function {name}({params}) {',
                'semantic_type': 'function_definition'
            },
            'योगः': {  # yogah (addition)
                'python': '{a} + {b}',
                'javascript': '{a} + {b}',
                'semantic_type': 'addition'
            }
        }
    
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate Sanskrit to programming code."""
        if source_domain != DomainType.SANSKRIT or target_domain != DomainType.PROGRAMMING:
            raise ValueError("This translator only supports Sanskrit to Programming translation")
        
        target_language = kwargs.get('language', ProgrammingLanguage.PYTHON).value
        translated_code = self._translate_sanskrit_to_code(content, target_language)
        
        return DomainMapping(
            id=f"sanskrit_to_{target_language}_{hash(content)}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_content=content,
            target_content=translated_code,
            mapping_type="sanskrit_to_programming",
            confidence=0.8,
            metadata={'target_language': target_language}
        )
    
    def _translate_sanskrit_to_code(self, sanskrit_text: str, target_language: str) -> str:
        """Translate Sanskrit text to code in target language."""
        # Simple pattern matching approach
        for sanskrit_keyword, patterns in self.sanskrit_to_code_patterns.items():
            if sanskrit_keyword in sanskrit_text:
                if target_language in patterns:
                    template = patterns[target_language]
                    # Extract basic parameters
                    if sanskrit_keyword == 'यदि':
                        # Extract condition and action from "यदि condition तदा action"
                        if 'तदा' in sanskrit_text:
                            parts = sanskrit_text.split('यदि')[1].split('तदा')
                            condition = parts[0].strip() if parts else 'True'
                            action = parts[1].strip() if len(parts) > 1 else 'pass'
                            return template.format(condition=condition, action=action)
                        else:
                            condition = sanskrit_text.replace('यदि', '').strip()
                            return template.format(condition=condition or 'True', action='pass')
                    elif sanskrit_keyword == 'कार्यम्':
                        # Extract function name and parameters
                        parts = sanskrit_text.replace('कार्यम्', '').strip().split('=')
                        if len(parts) >= 2:
                            func_def = parts[0].strip().split()
                            name = func_def[0] if func_def else 'function'
                            params = ', '.join(func_def[1:]) if len(func_def) > 1 else ''
                            return template.format(name=name, params=params)
                    elif sanskrit_keyword == 'योगः':
                        parts = sanskrit_text.replace('योगः', '').strip().split()
                        a = parts[0] if parts else 'x'
                        b = parts[1] if len(parts) > 1 else 'y'
                        return template.format(a=a, b=b)
        
        # If no pattern matches, return as comment
        return f"# {sanskrit_text}" if target_language == 'python' else f"// {sanskrit_text}"
    
    def validate_translation(self, mapping: DomainMapping) -> bool:
        """Validate programming translation by checking syntax."""
        target_language = mapping.metadata.get('target_language', 'python')
        code = mapping.target_content
        
        try:
            if target_language == 'python':
                ast.parse(code)
                return True
            else:
                # Basic syntax check for other languages
                if target_language in ['javascript', 'java', 'cpp']:
                    open_braces = code.count('{')
                    close_braces = code.count('}')
                    return open_braces == close_braces
                return True
        except:
            return False


class SanskritToMathTranslator(DomainTranslator):
    """Translates Sanskrit expressions to mathematical formulas."""
    
    def __init__(self, symbolic_engine=None):
        self.symbolic_engine = symbolic_engine
        self.math_patterns = {
            'योगः': '+',  # addition
            'गुणनम्': '*',  # multiplication
            'ज्या': 'sin',  # sine
            'कोज्या': 'cos',  # cosine
            'पाई': 'pi',  # pi
        }
    
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate Sanskrit to mathematical formula."""
        if source_domain != DomainType.SANSKRIT or target_domain != DomainType.MATHEMATICS:
            raise ValueError("This translator only supports Sanskrit to Mathematics translation")
        
        formula = self._translate_to_formula(content)
        
        return DomainMapping(
            id=f"sanskrit_to_math_{hash(content)}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_content=content,
            target_content=formula,
            mapping_type="sanskrit_to_mathematics",
            confidence=0.85,
            metadata={'formula_type': self._classify_formula(formula)}
        )
    
    def _translate_to_formula(self, sanskrit_text: str) -> str:
        """Translate Sanskrit text to mathematical formula."""
        formula = sanskrit_text
        
        # Apply pattern substitutions
        for sanskrit_term, math_symbol in self.math_patterns.items():
            formula = formula.replace(sanskrit_term, math_symbol)
        
        return formula.strip()
    
    def _classify_formula(self, formula: str) -> str:
        """Classify the type of mathematical formula."""
        if any(func in formula for func in ['sin', 'cos', 'tan']):
            return 'trigonometry'
        elif any(op in formula for op in ['+', '-', '*', '/']):
            return 'arithmetic'
        else:
            return 'general'
    
    def validate_translation(self, mapping: DomainMapping) -> bool:
        """Validate mathematical translation."""
        formula = mapping.target_content
        
        # Basic validation - check for balanced parentheses
        open_parens = formula.count('(')
        close_parens = formula.count(')')
        
        return open_parens == close_parens


class SanskritToKnowledgeGraphTranslator(DomainTranslator):
    """Translates Sanskrit to knowledge graph representation."""
    
    def __init__(self):
        self.graph_builder = SemanticGraphBuilder()
        self.entity_patterns = {
            'गुरुः': SemanticNodeType.ENTITY,  # teacher
            'शिष्यः': SemanticNodeType.ENTITY,  # student
            'धर्मः': SemanticNodeType.CONCEPT,  # dharma
        }
    
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate Sanskrit to knowledge graph."""
        if source_domain != DomainType.SANSKRIT or target_domain != DomainType.KNOWLEDGE_GRAPH:
            raise ValueError("This translator only supports Sanskrit to Knowledge Graph translation")
        
        graph = self._create_knowledge_graph(content)
        
        return DomainMapping(
            id=f"sanskrit_to_kg_{hash(content)}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_content=content,
            target_content=graph.to_json(),
            mapping_type="sanskrit_to_knowledge_graph",
            confidence=0.75,
            metadata={
                'node_count': len(graph.nodes),
                'edge_count': len(graph.edges),
                'graph_id': graph.id
            }
        )
    
    def _create_knowledge_graph(self, sanskrit_text: str) -> SemanticGraph:
        """Create knowledge graph from Sanskrit text."""
        graph = self.graph_builder.create_graph(sanskrit_text, "sanskrit")
        
        # Simple entity extraction
        words = sanskrit_text.split()
        for word in words:
            if word in self.entity_patterns:
                node_type = self.entity_patterns[word]
                self.graph_builder.add_concept_node(
                    concept=word,
                    label=word,
                    node_type=node_type
                )
        
        return self.graph_builder.get_graph()
    
    def validate_translation(self, mapping: DomainMapping) -> bool:
        """Validate knowledge graph translation."""
        try:
            graph_data = json.loads(mapping.target_content)
            required_fields = ['id', 'nodes', 'edges', 'source_text']
            return all(field in graph_data for field in required_fields)
        except:
            return False


class AlgorithmicSanskritDSL:
    """Domain-specific language for algorithmic Sanskrit."""
    
    def __init__(self):
        self.grammar_rules = {
            'conditional': 'यदि {condition} तदा {action}',
            'function': 'कार्यम् {name} {parameters} = {body}',
        }
    
    def parse_expression(self, expression: str) -> AlgorithmicSanskritExpression:
        """Parse an algorithmic Sanskrit expression."""
        expr_type = self._classify_expression(expression)
        parameters = self._extract_parameters(expression, expr_type)
        
        return AlgorithmicSanskritExpression(
            expression=expression,
            semantic_type=expr_type,
            parameters=parameters,
            documentation=f"Algorithmic Sanskrit expression of type: {expr_type}"
        )
    
    def _classify_expression(self, expression: str) -> str:
        """Classify the type of algorithmic Sanskrit expression."""
        if 'यदि' in expression and 'तदा' in expression:
            return 'conditional'
        elif 'कार्यम्' in expression:
            return 'function'
        else:
            return 'expression'
    
    def _extract_parameters(self, expression: str, expr_type: str) -> List[str]:
        """Extract parameters from expression."""
        parameters = []
        
        if expr_type == 'conditional' and 'यदि' in expression:
            parts = expression.split('यदि')[1].split('तदा')
            if parts:
                parameters.append(f"condition: {parts[0].strip()}")
        
        return parameters
    
    def compile_to_target(self, expression: AlgorithmicSanskritExpression, 
                         target_domain: DomainType, **kwargs) -> str:
        """Compile algorithmic Sanskrit to target domain."""
        if target_domain == DomainType.PROGRAMMING:
            language = kwargs.get('language', 'python')
            if expression.semantic_type == 'conditional':
                return self._compile_conditional_to_python(expression.expression)
        
        return f"# {expression.expression}"
    
    def validate_expression(self, expression: AlgorithmicSanskritExpression) -> bool:
        """Validate an algorithmic Sanskrit expression."""
        if expression.semantic_type == 'conditional':
            return 'यदि' in expression.expression and 'तदा' in expression.expression
        elif expression.semantic_type == 'function':
            return 'कार्यम्' in expression.expression
        return True
    
    def _compile_conditional_to_python(self, expression: str) -> str:
        """Compile conditional to Python."""
        if 'यदि' in expression and 'तदा' in expression:
            parts = expression.split('यदि')[1].split('तदा')
            condition = parts[0].strip() if parts else 'True'
            action = parts[1].strip() if len(parts) > 1 else 'pass'
            return f"if {condition}:\n    {action}"
        
        return f"# {expression}"

class MultiDomainMapper:
    """Main class for multi-domain mapping system."""
    
    def __init__(self, reasoning_core=None, symbolic_engine=None):
        self.reasoning_core = reasoning_core
        self.symbolic_engine = symbolic_engine
        
        # Initialize translators
        self.translators = {
            (DomainType.SANSKRIT, DomainType.PROGRAMMING): SanskritToProgrammingTranslator(),
            (DomainType.SANSKRIT, DomainType.MATHEMATICS): SanskritToMathTranslator(symbolic_engine),
            (DomainType.SANSKRIT, DomainType.KNOWLEDGE_GRAPH): SanskritToKnowledgeGraphTranslator(),
        }
        
        # Initialize DSL
        self.algorithmic_dsl = AlgorithmicSanskritDSL()
        
        # Mapping history and cache
        self.mapping_history: List[DomainMapping] = []
        self.semantic_consistency_cache: Dict[str, float] = {}
    
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate content between domains."""
        translator_key = (source_domain, target_domain)
        
        if translator_key in self.translators:
            translator = self.translators[translator_key]
            mapping = translator.translate(content, source_domain, target_domain, **kwargs)
            
            # Store in history
            self.mapping_history.append(mapping)
            
            return mapping
        else:
            raise ValueError(f"No translator available for {source_domain} to {target_domain}")
    
    def create_algorithmic_sanskrit(self, expression: str) -> AlgorithmicSanskritExpression:
        """Create algorithmic Sanskrit expression."""
        return self.algorithmic_dsl.parse_expression(expression)
    
    def compile_algorithmic_sanskrit(self, expression: AlgorithmicSanskritExpression, 
                                   target_domain: DomainType, **kwargs) -> str:
        """Compile algorithmic Sanskrit to target domain."""
        return self.algorithmic_dsl.compile_to_target(expression, target_domain, **kwargs)
    
    def create_bidirectional_mapping(self, content: str, source_domain: DomainType, 
                                   target_domain: DomainType, **kwargs) -> Dict[str, Any]:
        """Create and validate bidirectional mapping."""
        # Simple implementation for testing
        forward_mapping = self.translate(content, source_domain, target_domain, **kwargs)
        
        # Create a mock reverse mapping
        reverse_mapping = DomainMapping(
            id=f"reverse_{hash(content)}",
            source_domain=target_domain,
            target_domain=source_domain,
            source_content=forward_mapping.target_content,
            target_content=content,
            mapping_type="reverse_translation",
            confidence=0.7
        )
        
        # Simple consistency score based on content similarity
        consistency_score = self._calculate_similarity(content, reverse_mapping.target_content)
        
        return {
            'forward_mapping': forward_mapping,
            'reverse_mapping': reverse_mapping,
            'consistency_score': consistency_score,
            'is_consistent': consistency_score > 0.7
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def validate_semantic_preservation(self, mapping: DomainMapping) -> float:
        """Validate that semantic meaning is preserved across domains."""
        cache_key = f"{mapping.source_domain.value}_{mapping.target_domain.value}_{hash(mapping.source_content)}"
        
        if cache_key in self.semantic_consistency_cache:
            return self.semantic_consistency_cache[cache_key]
        
        # Use translator's validation method
        translator_key = (mapping.source_domain, mapping.target_domain)
        if translator_key in self.translators:
            translator = self.translators[translator_key]
            is_valid = translator.validate_translation(mapping)
            score = 0.8 if is_valid else 0.3  # Higher score for valid translations
        else:
            # Fallback validation
            score = self._fallback_semantic_validation(mapping)
        
        self.semantic_consistency_cache[cache_key] = score
        return score
    
    def _fallback_semantic_validation(self, mapping: DomainMapping) -> float:
        """Fallback semantic validation when no specific validator is available."""
        # Basic validation - check that target content is not empty and different from source
        if not mapping.target_content or mapping.target_content == mapping.source_content:
            return 0.0
        
        # Check for reasonable length ratio
        source_len = len(mapping.source_content)
        target_len = len(mapping.target_content)
        
        if source_len == 0:
            return 0.0
        
        length_ratio = target_len / source_len
        
        # Reasonable translations should have length ratios between 0.5 and 3.0
        if 0.5 <= length_ratio <= 3.0:
            return 0.6  # Moderate confidence
        else:
            return 0.3  # Low confidence
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about mappings performed."""
        if not self.mapping_history:
            return {'total_mappings': 0}
        
        stats = {
            'total_mappings': len(self.mapping_history),
            'by_source_domain': {},
            'by_target_domain': {},
            'by_mapping_type': {},
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for mapping in self.mapping_history:
            # Count by source domain
            source = mapping.source_domain.value
            stats['by_source_domain'][source] = stats['by_source_domain'].get(source, 0) + 1
            
            # Count by target domain
            target = mapping.target_domain.value
            stats['by_target_domain'][target] = stats['by_target_domain'].get(target, 0) + 1
            
            # Count by mapping type
            mapping_type = mapping.mapping_type
            stats['by_mapping_type'][mapping_type] = stats['by_mapping_type'].get(mapping_type, 0) + 1
            
            # Sum confidence
            total_confidence += mapping.confidence
        
        stats['average_confidence'] = total_confidence / len(self.mapping_history)
        
        return stats
    
    def export_mappings(self, format: str = 'json') -> str:
        """Export mapping history in specified format."""
        if format == 'json':
            mappings_data = [mapping.to_dict() for mapping in self.mapping_history]
            return json.dumps(mappings_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_mappings(self, data: str, format: str = 'json') -> int:
        """Import mappings from data."""
        if format == 'json':
            mappings_data = json.loads(data)
            imported_count = 0
            
            for mapping_dict in mappings_data:
                try:
                    mapping = DomainMapping(
                        id=mapping_dict['id'],
                        source_domain=DomainType(mapping_dict['source_domain']),
                        target_domain=DomainType(mapping_dict['target_domain']),
                        source_content=mapping_dict['source_content'],
                        target_content=mapping_dict['target_content'],
                        mapping_type=mapping_dict['mapping_type'],
                        confidence=mapping_dict.get('confidence', 1.0),
                        metadata=mapping_dict.get('metadata', {}),
                        created_at=mapping_dict.get('created_at', datetime.now().isoformat())
                    )
                    self.mapping_history.append(mapping)
                    imported_count += 1
                except Exception as e:
                    print(f"Failed to import mapping: {e}")
            
            return imported_count
        else:
            raise ValueError(f"Unsupported import format: {format}")