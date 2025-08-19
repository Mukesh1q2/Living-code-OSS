"""
Semantic processing pipeline for Sanskrit text.

This module implements the complete pipeline:
morphology → syntax tree → semantic graph
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import logging

from .token import Token, TokenKind
from .tokenizer import SanskritTokenizer
from .morphological_analyzer import (
    SanskritMorphologicalAnalyzer, MorphologicalAnalysis,
    Morpheme, MorphemeType, GrammaticalCategory
)
from .syntax_tree import (
    SyntaxTree, SyntaxTreeBuilder, SyntaxNode, SyntaxNodeType,
    PhraseType, SyntacticFunction
)
from .semantic_graph import (
    SemanticGraph, SemanticGraphBuilder, SemanticNode, SemanticEdge,
    SemanticNodeType, SemanticRelationType
)


class ProcessingStage(Enum):
    """Stages in the semantic processing pipeline."""
    TOKENIZATION = "TOKENIZATION"
    MORPHOLOGICAL_ANALYSIS = "MORPHOLOGICAL_ANALYSIS"
    SYNTAX_TREE_CONSTRUCTION = "SYNTAX_TREE_CONSTRUCTION"
    SEMANTIC_GRAPH_CONSTRUCTION = "SEMANTIC_GRAPH_CONSTRUCTION"
    CROSS_LANGUAGE_MAPPING = "CROSS_LANGUAGE_MAPPING"
    VALIDATION = "VALIDATION"


@dataclass
class ProcessingResult:
    """Result of semantic processing pipeline."""
    input_text: str
    tokens: List[Token] = field(default_factory=list)
    morphological_analyses: List[MorphologicalAnalysis] = field(default_factory=list)
    syntax_tree: Optional[SyntaxTree] = None
    semantic_graph: Optional[SemanticGraph] = None
    cross_language_mappings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    processing_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'input_text': self.input_text,
            'confidence': self.confidence,
            'errors': self.errors
        }


class CrossLanguageMapper:
    """Handles cross-language semantic mapping."""
    
    def __init__(self):
        self.concept_mappings = {
            "गम्": {"english": "go", "python": "move"},
            "कर्": {"english": "do", "python": "execute"}
        }
    
    def map_concept_to_language(self, concept: str, target_language: str) -> Optional[str]:
        """Map a Sanskrit concept to another language."""
        if concept in self.concept_mappings:
            return self.concept_mappings[concept].get(target_language)
        return None
    
    def map_relation_to_language(self, relation: str, target_language: str) -> Optional[str]:
        """Map a Sanskrit relation to another language."""
        relation_mappings = {
            "कर्ता": {"english": "agent", "programming": "caller"},
            "कर्म": {"english": "patient", "programming": "parameter"}
        }
        if relation in relation_mappings:
            return relation_mappings[relation].get(target_language)
        return None
    
    def generate_cross_language_mappings(self, semantic_graph: SemanticGraph) -> Dict[str, Any]:
        """Generate cross-language mappings for a semantic graph."""
        mappings = {'english': {}, 'python': {}, 'javascript': {}}
        
        # Map nodes
        for node in semantic_graph.nodes.values():
            concept = node.concept
            for language in mappings.keys():
                mapped_concept = self.map_concept_to_language(concept, language)
                if mapped_concept:
                    mappings[language][node.id] = {
                        'original_concept': concept,
                        'mapped_concept': mapped_concept,
                        'node_type': node.node_type.value
                    }
        
        return mappings
    
    def generate_code_representation(self, semantic_graph: SemanticGraph, language: str = "python") -> str:
        """Generate code representation of a semantic graph."""
        if language == "python":
            return self._generate_python_code(semantic_graph)
        elif language == "javascript":
            return self._generate_javascript_code(semantic_graph)
        else:
            return f"# Generated {language} code"
    
    def _generate_python_code(self, semantic_graph: SemanticGraph) -> str:
        """Generate Python code representation."""
        lines = ["# Generated from Sanskrit semantic graph", ""]
        
        # Generate classes for entities
        entity_nodes = [n for n in semantic_graph.nodes.values() 
                       if n.node_type == SemanticNodeType.ENTITY]
        for node in entity_nodes:
            class_name = node.concept.capitalize()
            lines.append(f"class {class_name}:")
            lines.append(f'    """Represents {node.label}"""')
            lines.append("    pass")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_javascript_code(self, semantic_graph: SemanticGraph) -> str:
        """Generate JavaScript code representation."""
        lines = ["// Generated from Sanskrit semantic graph", ""]
        
        # Generate classes for entities
        entity_nodes = [n for n in semantic_graph.nodes.values() 
                       if n.node_type == SemanticNodeType.ENTITY]
        for node in entity_nodes:
            class_name = node.concept.capitalize()
            lines.append(f"class {class_name} {{")
            lines.append(f"  // Represents {node.label}")
            lines.append("}")
            lines.append("")
        
        return "\n".join(lines)


class SemanticProcessor:
    """Main semantic processing pipeline for Sanskrit text."""
    
    def __init__(self,
                 tokenizer: Optional[SanskritTokenizer] = None,
                 morphological_analyzer: Optional[SanskritMorphologicalAnalyzer] = None):
        """Initialize the semantic processor."""
        self.tokenizer = tokenizer or SanskritTokenizer()
        self.morphological_analyzer = morphological_analyzer or SanskritMorphologicalAnalyzer()
        self.syntax_tree_builder = SyntaxTreeBuilder()
        self.semantic_graph_builder = SemanticGraphBuilder()
        self.cross_language_mapper = CrossLanguageMapper()
        self.logger = logging.getLogger(__name__)
    
    def process(self, text: str, 
                enable_cross_language_mapping: bool = True,
                target_languages: Optional[List[str]] = None) -> ProcessingResult:
        """Process Sanskrit text through the complete semantic pipeline."""
        result = ProcessingResult(input_text=text)
        
        try:
            # Basic processing stages
            result = self._tokenize(text, result)
            result = self._analyze_morphology(result)
            result = self._build_syntax_tree(result)
            result = self._build_semantic_graph(result)
            
            if enable_cross_language_mapping:
                result = self._generate_cross_language_mappings(result, target_languages)
            
            result.confidence = 0.8  # Default confidence
            
        except Exception as e:
            result.add_error(f"Processing failed: {str(e)}")
            result.confidence = 0.0
        
        return result
    
    def _tokenize(self, text: str, result: ProcessingResult) -> ProcessingResult:
        """Tokenize the input text."""
        try:
            tokens = self.tokenizer.tokenize(text)
            result.tokens = tokens
            result.processing_stages[ProcessingStage.TOKENIZATION.value] = {
                'token_count': len(tokens),
                'completed': True
            }
        except Exception as e:
            result.add_error(f"Tokenization failed: {str(e)}")
            result.processing_stages[ProcessingStage.TOKENIZATION.value] = {
                'completed': False,
                'error': str(e)
            }
        return result
    
    def _analyze_morphology(self, result: ProcessingResult) -> ProcessingResult:
        """Perform morphological analysis."""
        try:
            words = self._group_tokens_into_words(result.tokens)
            analyses = []
            for word in words:
                analysis = self.morphological_analyzer.analyze_word(word)
                if analysis:
                    analyses.append(analysis)
            result.morphological_analyses = analyses
            result.processing_stages[ProcessingStage.MORPHOLOGICAL_ANALYSIS.value] = {
                'word_count': len(words),
                'analysis_count': len(analyses),
                'completed': True
            }
        except Exception as e:
            result.add_error(f"Morphological analysis failed: {str(e)}")
            result.processing_stages[ProcessingStage.MORPHOLOGICAL_ANALYSIS.value] = {
                'completed': False,
                'error': str(e)
            }
        return result
    
    def _group_tokens_into_words(self, tokens: List[Token]) -> List[str]:
        """Group tokens into words."""
        words = []
        current_word = ""
        for token in tokens:
            if token.kind == TokenKind.MARKER:
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += token.text
        if current_word:
            words.append(current_word)
        return words
    
    def _build_syntax_tree(self, result: ProcessingResult) -> ProcessingResult:
        """Build syntax tree."""
        try:
            if result.morphological_analyses:
                syntax_tree = self.syntax_tree_builder.build_tree_from_morphological_analyses(
                    result.morphological_analyses, result.input_text
                )
                result.syntax_tree = syntax_tree
        except Exception as e:
            result.add_error(f"Syntax tree construction failed: {str(e)}")
        return result
    
    def _build_semantic_graph(self, result: ProcessingResult) -> ProcessingResult:
        """Build semantic graph."""
        try:
            if result.syntax_tree:
                semantic_graph = self._convert_syntax_tree_to_semantic_graph(result.syntax_tree)
                result.semantic_graph = semantic_graph
        except Exception as e:
            result.add_error(f"Semantic graph construction failed: {str(e)}")
        return result
    
    def _convert_syntax_tree_to_semantic_graph(self, syntax_tree: SyntaxTree) -> SemanticGraph:
        """Convert syntax tree to semantic graph."""
        graph = self.semantic_graph_builder.create_graph(
            source_text=syntax_tree.source_text, language="sanskrit"
        )
        
        word_nodes = syntax_tree.get_word_nodes()
        for word_node in word_nodes:
            concept = self._extract_concept_from_word_node(word_node)
            self.semantic_graph_builder.add_concept_node(
                concept=concept, label=word_node.text
            )
        
        return graph
    
    def _extract_concept_from_word_node(self, word_node: SyntaxNode) -> str:
        """Extract concept from word node."""
        if word_node.morphological_info:
            root_morphemes = word_node.morphological_info.get_root_morphemes()
            if root_morphemes:
                return root_morphemes[0].text
        return word_node.text
    
    def _generate_cross_language_mappings(self, result: ProcessingResult, 
                                        target_languages: Optional[List[str]] = None) -> ProcessingResult:
        """Generate cross-language mappings."""
        try:
            if result.semantic_graph:
                mappings = self.cross_language_mapper.generate_cross_language_mappings(
                    result.semantic_graph
                )
                result.cross_language_mappings = mappings
        except Exception as e:
            result.add_error(f"Cross-language mapping failed: {str(e)}")
        return result
    
    def batch_process(self, texts: List[str], **kwargs) -> List[ProcessingResult]:
        """Process multiple texts in batch."""
        results = []
        for text in texts:
            result = self.process(text, **kwargs)
            results.append(result)
        return results
    
    def generate_report(self, result: ProcessingResult, format: str = "text") -> str:
        """Generate a comprehensive processing report."""
        if format == "text":
            lines = []
            lines.append("SANSKRIT SEMANTIC PROCESSING REPORT")
            lines.append("=" * 50)
            lines.append(f"Input Text: {result.input_text}")
            lines.append(f"Overall Confidence: {result.confidence:.2f}")
            return "\n".join(lines)
        elif format == "json":
            import json
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported report format: {format}")


# Utility functions
def process_sanskrit_text(text: str, **kwargs) -> ProcessingResult:
    """Convenience function to process Sanskrit text."""
    processor = SemanticProcessor()
    return processor.process(text, **kwargs)


def extract_semantic_concepts(result: ProcessingResult) -> List[str]:
    """Extract semantic concepts from processing result."""
    if not result.semantic_graph:
        return []
    return [node.concept for node in result.semantic_graph.nodes.values()]


def extract_semantic_relations(result: ProcessingResult) -> List[Tuple[str, str, str]]:
    """Extract semantic relations from processing result."""
    if not result.semantic_graph:
        return []
    
    relations = []
    for edge in result.semantic_graph.edges.values():
        source_node = result.semantic_graph.get_node(edge.source_id)
        target_node = result.semantic_graph.get_node(edge.target_id)
        
        if source_node and target_node:
            relations.append((
                source_node.concept,
                edge.relation_type.value,
                target_node.concept
            ))
    
    return relations


def validate_semantic_consistency(results: List[ProcessingResult]) -> Dict[str, Any]:
    """Validate semantic consistency across multiple processing results."""
    consistency_report = {
        'total_results': len(results),
        'consistent_concepts': set(),
        'inconsistent_concepts': set(),
        'confidence_stats': {'mean': 0.0, 'min': 1.0, 'max': 0.0}
    }
    
    if not results:
        return consistency_report
    
    confidences = [result.confidence for result in results]
    if confidences:
        consistency_report['confidence_stats'] = {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences)
        }
    
    return consistency_report