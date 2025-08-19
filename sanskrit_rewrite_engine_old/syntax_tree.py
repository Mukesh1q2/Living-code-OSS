"""
Syntax tree representation for Sanskrit text processing.

This module implements syntax tree structures that serve as an intermediate
representation between morphological analysis and semantic graphs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import json

from .token import Token, TokenKind
from .morphological_analyzer import (
    MorphologicalAnalysis, Morpheme, MorphemeType, 
    GrammaticalCategory, SamasaType
)


class SyntaxNodeType(Enum):
    """Types of syntax tree nodes."""
    ROOT = "ROOT"                       # Root of the syntax tree
    SENTENCE = "SENTENCE"               # Complete sentence
    CLAUSE = "CLAUSE"                   # Subordinate clause
    PHRASE = "PHRASE"                   # Phrase (noun phrase, verb phrase, etc.)
    WORD = "WORD"                       # Individual word
    MORPHEME = "MORPHEME"               # Morpheme within a word
    COMPOUND = "COMPOUND"               # Compound word (samāsa)
    SANDHI = "SANDHI"                   # Sandhi junction


class PhraseType(Enum):
    """Types of phrases in Sanskrit syntax."""
    NOUN_PHRASE = "NOUN_PHRASE"         # Noun phrase (substantive)
    VERB_PHRASE = "VERB_PHRASE"         # Verb phrase (predicate)
    ADJECTIVE_PHRASE = "ADJECTIVE_PHRASE"  # Adjectival phrase
    ADVERB_PHRASE = "ADVERB_PHRASE"     # Adverbial phrase
    PREPOSITIONAL_PHRASE = "PREPOSITIONAL_PHRASE"  # Prepositional phrase
    PARTICIPIAL_PHRASE = "PARTICIPIAL_PHRASE"  # Participial phrase


class SyntacticFunction(Enum):
    """Syntactic functions in Sanskrit grammar."""
    SUBJECT = "SUBJECT"                 # Kartā (agent/subject)
    OBJECT = "OBJECT"                   # Karma (patient/object)
    INDIRECT_OBJECT = "INDIRECT_OBJECT" # Sampradāna (recipient)
    INSTRUMENT = "INSTRUMENT"           # Karaṇa (instrument)
    SOURCE = "SOURCE"                   # Apādāna (source/ablation)
    LOCATION = "LOCATION"               # Adhikaraṇa (location)
    POSSESSOR = "POSSESSOR"             # Sambandha (genitive relation)
    PREDICATE = "PREDICATE"             # Main predicate
    MODIFIER = "MODIFIER"               # Adjectival/adverbial modifier
    COMPLEMENT = "COMPLEMENT"           # Complement
    VOCATIVE = "VOCATIVE"               # Vocative (address)


@dataclass
class SyntaxNode:
    """
    A node in the syntax tree.
    
    Attributes:
        id: Unique identifier for the node
        node_type: Type of syntax node
        phrase_type: Type of phrase (if applicable)
        syntactic_function: Syntactic function in the sentence
        label: Human-readable label
        text: Text content represented by this node
        morphological_info: Morphological analysis information
        grammatical_features: Grammatical features (case, number, etc.)
        children: Child nodes
        parent_id: ID of parent node
        confidence: Confidence in this syntactic analysis
        metadata: Additional node-specific information
    """
    id: str
    node_type: SyntaxNodeType
    label: str = ""
    text: str = ""
    phrase_type: Optional[PhraseType] = None
    syntactic_function: Optional[SyntacticFunction] = None
    morphological_info: Optional[MorphologicalAnalysis] = None
    grammatical_features: Dict[str, Any] = field(default_factory=dict)
    children: List['SyntaxNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'SyntaxNode') -> None:
        """Add a child node."""
        child.parent_id = self.id
        self.children.append(child)
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child node by ID."""
        self.children = [child for child in self.children if child.id != child_id]
    
    def get_child_by_id(self, child_id: str) -> Optional['SyntaxNode']:
        """Get a child node by ID."""
        for child in self.children:
            if child.id == child_id:
                return child
        return None
    
    def get_children_by_type(self, node_type: SyntaxNodeType) -> List['SyntaxNode']:
        """Get all children of a specific type."""
        return [child for child in self.children if child.node_type == node_type]
    
    def get_children_by_function(self, function: SyntacticFunction) -> List['SyntaxNode']:
        """Get all children with a specific syntactic function."""
        return [child for child in self.children if child.syntactic_function == function]
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def get_text(self) -> str:
        """Get the text content of this node and all its children."""
        if self.text:
            return self.text
        
        # Concatenate children's text
        child_texts = []
        for child in self.children:
            child_text = child.get_text()
            if child_text:
                child_texts.append(child_text)
        
        return ' '.join(child_texts)
    
    def add_grammatical_feature(self, feature: str, value: Any) -> None:
        """Add a grammatical feature."""
        self.grammatical_features[feature] = value
    
    def get_grammatical_feature(self, feature: str, default=None) -> Any:
        """Get a grammatical feature value."""
        return self.grammatical_features.get(feature, default)
    
    def has_grammatical_feature(self, feature: str) -> bool:
        """Check if node has a specific grammatical feature."""
        return feature in self.grammatical_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'node_type': self.node_type.value,
            'label': self.label,
            'text': self.text,
            'phrase_type': self.phrase_type.value if self.phrase_type else None,
            'syntactic_function': self.syntactic_function.value if self.syntactic_function else None,
            'grammatical_features': self.grammatical_features,
            'children': [child.to_dict() for child in self.children],
            'parent_id': self.parent_id,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class SyntaxTree:
    """
    A complete syntax tree for a Sanskrit sentence or text.
    
    Attributes:
        id: Unique identifier for the tree
        root: Root node of the tree
        source_text: Original text that generated this tree
        morphological_analyses: Morphological analyses for words
        confidence: Overall confidence in the syntactic analysis
        metadata: Additional tree-specific information
    """
    id: str
    root: SyntaxNode
    source_text: str = ""
    morphological_analyses: List[MorphologicalAnalysis] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_nodes(self) -> List[SyntaxNode]:
        """Get all nodes in the tree."""
        nodes = []
        self._collect_nodes(self.root, nodes)
        return nodes
    
    def _collect_nodes(self, node: SyntaxNode, nodes: List[SyntaxNode]) -> None:
        """Recursively collect all nodes."""
        nodes.append(node)
        for child in node.children:
            self._collect_nodes(child, nodes)
    
    def get_node_by_id(self, node_id: str) -> Optional[SyntaxNode]:
        """Get a node by its ID."""
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_nodes_by_type(self, node_type: SyntaxNodeType) -> List[SyntaxNode]:
        """Get all nodes of a specific type."""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.node_type == node_type]
    
    def get_nodes_by_function(self, function: SyntacticFunction) -> List[SyntaxNode]:
        """Get all nodes with a specific syntactic function."""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.syntactic_function == function]
    
    def get_leaf_nodes(self) -> List[SyntaxNode]:
        """Get all leaf nodes (terminal nodes)."""
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.is_leaf()]
    
    def get_word_nodes(self) -> List[SyntaxNode]:
        """Get all word-level nodes."""
        return self.get_nodes_by_type(SyntaxNodeType.WORD)
    
    def get_phrase_nodes(self) -> List[SyntaxNode]:
        """Get all phrase-level nodes."""
        return self.get_nodes_by_type(SyntaxNodeType.PHRASE)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary for serialization."""
        return {
            'id': self.id,
            'root': self.root.to_dict(),
            'source_text': self.source_text,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert tree to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def visualize_tree(self, format: str = "text") -> str:
        """Generate a visualization of the syntax tree."""
        if format == "text":
            return self._generate_text_tree()
        elif format == "mermaid":
            return self._generate_mermaid_tree()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_tree(self) -> str:
        """Generate a text-based tree visualization."""
        lines = []
        lines.append(f"Syntax Tree: {self.source_text}")
        lines.append("=" * 50)
        self._generate_text_node(self.root, lines, 0)
        return "\n".join(lines)
    
    def _generate_text_node(self, node: SyntaxNode, lines: List[str], depth: int) -> None:
        """Generate text representation of a node."""
        indent = "  " * depth
        
        # Node information
        node_info = f"{node.node_type.value}"
        if node.phrase_type:
            node_info += f" ({node.phrase_type.value})"
        if node.syntactic_function:
            node_info += f" [{node.syntactic_function.value}]"
        
        # Add text content if it's a leaf node
        if node.is_leaf() and node.text:
            node_info += f": '{node.text}'"
        elif node.label:
            node_info += f": {node.label}"
        
        lines.append(f"{indent}{node_info}")
        
        # Add grammatical features
        if node.grammatical_features:
            features = ", ".join(f"{k}={v}" for k, v in node.grammatical_features.items())
            lines.append(f"{indent}  Features: {features}")
        
        # Recursively add children
        for child in node.children:
            self._generate_text_node(child, lines, depth + 1)
    
    def _generate_mermaid_tree(self) -> str:
        """Generate a Mermaid diagram for the syntax tree."""
        lines = ["graph TD"]
        self._generate_mermaid_node(self.root, lines)
        return "\n".join(lines)
    
    def _generate_mermaid_node(self, node: SyntaxNode, lines: List[str]) -> None:
        """Generate Mermaid representation of a node."""
        # Node label
        label = node.label or node.node_type.value
        if node.text and node.is_leaf():
            label = f"{label}: {node.text}"
        
        lines.append(f'    {node.id}["{label}"]')
        
        # Add edges to children
        for child in node.children:
            lines.append(f'    {node.id} --> {child.id}')
            self._generate_mermaid_node(child, lines)


class SyntaxTreeBuilder:
    """
    Builder class for constructing syntax trees from morphological analyses.
    """
    
    def __init__(self):
        self.node_counter = 0
    
    def build_tree_from_morphological_analyses(self, 
                                             analyses: List[MorphologicalAnalysis],
                                             source_text: str = "") -> SyntaxTree:
        """
        Build a syntax tree from morphological analyses.
        
        Args:
            analyses: List of morphological analyses for words
            source_text: Original source text
            
        Returns:
            Complete syntax tree
        """
        # Create root node
        root = self._create_node(
            node_type=SyntaxNodeType.ROOT,
            label="ROOT"
        )
        
        # Create sentence node
        sentence_node = self._create_node(
            node_type=SyntaxNodeType.SENTENCE,
            label="SENTENCE",
            text=source_text
        )
        root.add_child(sentence_node)
        
        # Process each morphological analysis
        for analysis in analyses:
            word_node = self._build_word_node(analysis)
            sentence_node.add_child(word_node)
        
        # Determine syntactic functions
        self._assign_syntactic_functions(sentence_node)
        
        # Build phrases
        self._build_phrases(sentence_node)
        
        # Create syntax tree
        tree = SyntaxTree(
            id=f"tree_{self.node_counter}",
            root=root,
            source_text=source_text,
            morphological_analyses=analyses
        )
        
        return tree
    
    def _create_node(self, 
                    node_type: SyntaxNodeType,
                    label: str = "",
                    text: str = "",
                    **kwargs) -> SyntaxNode:
        """Create a new syntax node with a unique ID."""
        self.node_counter += 1
        return SyntaxNode(
            id=f"node_{self.node_counter}",
            node_type=node_type,
            label=label,
            text=text,
            **kwargs
        )
    
    def _build_word_node(self, analysis: MorphologicalAnalysis) -> SyntaxNode:
        """Build a word node from morphological analysis."""
        word_node = self._create_node(
            node_type=SyntaxNodeType.WORD,
            label=analysis.word,
            text=analysis.word,
            morphological_info=analysis
        )
        
        # Add grammatical features from morphological analysis
        for category in analysis.grammatical_categories:
            word_node.add_grammatical_feature(category.value, True)
        
        # Add morpheme children
        for morpheme in analysis.morphemes:
            morpheme_node = self._create_node(
                node_type=SyntaxNodeType.MORPHEME,
                label=f"{morpheme.text} ({morpheme.type.value})",
                text=morpheme.text
            )
            
            # Add morpheme grammatical features
            for feature, value in morpheme.grammatical_info.items():
                morpheme_node.add_grammatical_feature(feature, value)
            
            word_node.add_child(morpheme_node)
        
        # Handle compound analysis
        if analysis.compound_analysis:
            compound_node = self._create_node(
                node_type=SyntaxNodeType.COMPOUND,
                label=f"Compound ({analysis.compound_analysis.type.value})",
                text=analysis.compound_analysis.compound_text
            )
            
            # Add constituent morphemes
            for constituent in analysis.compound_analysis.constituents:
                constituent_node = self._create_node(
                    node_type=SyntaxNodeType.MORPHEME,
                    label=f"{constituent.text} ({constituent.type.value})",
                    text=constituent.text
                )
                compound_node.add_child(constituent_node)
            
            word_node.add_child(compound_node)
        
        return word_node
    
    def _assign_syntactic_functions(self, sentence_node: SyntaxNode) -> None:
        """Assign syntactic functions to word nodes based on grammatical features."""
        word_nodes = sentence_node.get_children_by_type(SyntaxNodeType.WORD)
        
        for word_node in word_nodes:
            function = self._determine_syntactic_function(word_node)
            if function:
                word_node.syntactic_function = function
    
    def _determine_syntactic_function(self, word_node: SyntaxNode) -> Optional[SyntacticFunction]:
        """Determine syntactic function based on grammatical features."""
        # Check for case markers (vibhakti)
        if word_node.has_grammatical_feature('PRATHAMA'):
            return SyntacticFunction.SUBJECT
        elif word_node.has_grammatical_feature('DVITIYA'):
            return SyntacticFunction.OBJECT
        elif word_node.has_grammatical_feature('TRITIYA'):
            return SyntacticFunction.INSTRUMENT
        elif word_node.has_grammatical_feature('CHATURTHI'):
            return SyntacticFunction.INDIRECT_OBJECT
        elif word_node.has_grammatical_feature('PANCHAMI'):
            return SyntacticFunction.SOURCE
        elif word_node.has_grammatical_feature('SHASHTI'):
            return SyntacticFunction.POSSESSOR
        elif word_node.has_grammatical_feature('SAPTAMI'):
            return SyntacticFunction.LOCATION
        elif word_node.has_grammatical_feature('SAMBODHAN'):
            return SyntacticFunction.VOCATIVE
        
        # Check for verbal forms
        if word_node.morphological_info:
            root_morphemes = word_node.morphological_info.get_root_morphemes()
            if any(m.type == MorphemeType.DHATU for m in root_morphemes):
                return SyntacticFunction.PREDICATE
        
        return None
    
    def _build_phrases(self, sentence_node: SyntaxNode) -> None:
        """Build phrase-level structures from word nodes."""
        word_nodes = sentence_node.get_children_by_type(SyntaxNodeType.WORD)
        
        # Group words into phrases based on syntactic functions and proximity
        phrases = self._group_words_into_phrases(word_nodes)
        
        # Replace word nodes with phrase nodes
        for phrase_words in phrases:
            if len(phrase_words) > 1:
                phrase_node = self._create_phrase_node(phrase_words)
                
                # Remove individual word nodes from sentence
                for word_node in phrase_words:
                    sentence_node.remove_child(word_node.id)
                
                # Add phrase node to sentence
                sentence_node.add_child(phrase_node)
    
    def _group_words_into_phrases(self, word_nodes: List[SyntaxNode]) -> List[List[SyntaxNode]]:
        """Group words into phrases based on syntactic and semantic criteria."""
        phrases = []
        current_phrase = []
        
        for word_node in word_nodes:
            if self._should_start_new_phrase(word_node, current_phrase):
                if current_phrase:
                    phrases.append(current_phrase)
                current_phrase = [word_node]
            else:
                current_phrase.append(word_node)
        
        if current_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def _should_start_new_phrase(self, word_node: SyntaxNode, current_phrase: List[SyntaxNode]) -> bool:
        """Determine if a word should start a new phrase."""
        if not current_phrase:
            return True
        
        # Start new phrase for different syntactic functions
        last_word = current_phrase[-1]
        if (word_node.syntactic_function and last_word.syntactic_function and
            word_node.syntactic_function != last_word.syntactic_function):
            return True
        
        # Start new phrase for predicates
        if word_node.syntactic_function == SyntacticFunction.PREDICATE:
            return True
        
        return False
    
    def _create_phrase_node(self, word_nodes: List[SyntaxNode]) -> SyntaxNode:
        """Create a phrase node from a group of word nodes."""
        # Determine phrase type based on the head word
        phrase_type = self._determine_phrase_type(word_nodes)
        
        # Determine syntactic function (use the function of the head word)
        syntactic_function = None
        for word_node in word_nodes:
            if word_node.syntactic_function:
                syntactic_function = word_node.syntactic_function
                break
        
        phrase_node = self._create_node(
            node_type=SyntaxNodeType.PHRASE,
            phrase_type=phrase_type,
            syntactic_function=syntactic_function,
            label=f"{phrase_type.value if phrase_type else 'PHRASE'}"
        )
        
        # Add word nodes as children
        for word_node in word_nodes:
            phrase_node.add_child(word_node)
        
        return phrase_node
    
    def _determine_phrase_type(self, word_nodes: List[SyntaxNode]) -> Optional[PhraseType]:
        """Determine the type of phrase based on its constituent words."""
        # Look for the head word to determine phrase type
        for word_node in word_nodes:
            if word_node.syntactic_function == SyntacticFunction.PREDICATE:
                return PhraseType.VERB_PHRASE
            elif word_node.syntactic_function in [
                SyntacticFunction.SUBJECT, 
                SyntacticFunction.OBJECT,
                SyntacticFunction.INDIRECT_OBJECT
            ]:
                return PhraseType.NOUN_PHRASE
        
        # Default to noun phrase
        return PhraseType.NOUN_PHRASE


# Utility functions
def create_syntax_node(node_type: SyntaxNodeType, 
                      node_id: str,
                      **kwargs) -> SyntaxNode:
    """Create a syntax node with the given type and ID."""
    return SyntaxNode(
        id=node_id,
        node_type=node_type,
        **kwargs
    )


def merge_syntax_trees(trees: List[SyntaxTree]) -> SyntaxTree:
    """Merge multiple syntax trees into one (for compound sentences)."""
    if not trees:
        raise ValueError("No trees to merge")
    
    if len(trees) == 1:
        return trees[0]
    
    # Create a new root
    builder = SyntaxTreeBuilder()
    root = builder._create_node(
        node_type=SyntaxNodeType.ROOT,
        label="MERGED_ROOT"
    )
    
    # Add all sentence nodes as children
    source_texts = []
    all_analyses = []
    
    for tree in trees:
        sentence_nodes = tree.root.get_children_by_type(SyntaxNodeType.SENTENCE)
        for sentence_node in sentence_nodes:
            root.add_child(sentence_node)
        
        source_texts.append(tree.source_text)
        all_analyses.extend(tree.morphological_analyses)
    
    merged_tree = SyntaxTree(
        id=f"merged_tree_{builder.node_counter}",
        root=root,
        source_text=" ".join(source_texts),
        morphological_analyses=all_analyses
    )
    
    return merged_tree