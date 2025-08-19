"""
Semantic graph representation for Sanskrit text processing.

This module implements a universal semantic graph system that can:
- Map Sanskrit sentences to semantic graphs
- Link to mathematical expressions, programming structures, and logical forms
- Support cross-language semantic mapping (Sanskrit ↔ English ↔ Code)
- Provide semantic consistency validation across transformations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import json
import uuid
from datetime import datetime

from .token import Token, TokenKind
from .morphological_analyzer import (
    MorphologicalAnalysis, Morpheme, MorphemeType, 
    GrammaticalCategory, SamasaType
)


class SemanticNodeType(Enum):
    """Types of nodes in the semantic graph."""
    CONCEPT = "CONCEPT"                 # Abstract concept
    ENTITY = "ENTITY"                   # Concrete entity
    ACTION = "ACTION"                   # Action or process
    PROPERTY = "PROPERTY"               # Property or attribute
    RELATION = "RELATION"               # Relationship between concepts
    QUANTIFIER = "QUANTIFIER"           # Quantification (all, some, etc.)
    LOGICAL_OPERATOR = "LOGICAL_OPERATOR"  # Logical operators (and, or, not)
    MATHEMATICAL = "MATHEMATICAL"       # Mathematical expression
    PROGRAMMING = "PROGRAMMING"         # Programming construct
    LINGUISTIC = "LINGUISTIC"           # Linguistic construct


class SemanticRelationType(Enum):
    """Types of semantic relations."""
    # Core semantic relations
    AGENT = "AGENT"                     # Who/what performs action
    PATIENT = "PATIENT"                 # Who/what receives action
    INSTRUMENT = "INSTRUMENT"           # How action is performed
    LOCATION = "LOCATION"               # Where action occurs
    TIME = "TIME"                       # When action occurs
    MANNER = "MANNER"                   # How action is performed
    PURPOSE = "PURPOSE"                 # Why action is performed
    
    # Logical relations
    CONJUNCTION = "CONJUNCTION"         # AND relation
    DISJUNCTION = "DISJUNCTION"         # OR relation
    NEGATION = "NEGATION"               # NOT relation
    IMPLICATION = "IMPLICATION"         # IF-THEN relation
    EQUIVALENCE = "EQUIVALENCE"         # IFF relation
    
    # Mathematical relations
    EQUALS = "EQUALS"                   # Mathematical equality
    GREATER_THAN = "GREATER_THAN"       # Mathematical comparison
    LESS_THAN = "LESS_THAN"             # Mathematical comparison
    FUNCTION_APPLICATION = "FUNCTION_APPLICATION"  # Function applied to argument
    
    # Programming relations
    ASSIGNMENT = "ASSIGNMENT"           # Variable assignment
    FUNCTION_CALL = "FUNCTION_CALL"     # Function invocation
    INHERITANCE = "INHERITANCE"         # Class inheritance
    COMPOSITION = "COMPOSITION"         # Object composition
    
    # Sanskrit-specific relations
    KARAKA = "KARAKA"                   # Sanskrit grammatical relation
    SAMASA = "SAMASA"                   # Compound relation
    SANDHI = "SANDHI"                   # Phonological relation


@dataclass
class SemanticNode:
    """
    A node in the semantic graph representing a concept or entity.
    
    Attributes:
        id: Unique identifier for the node
        node_type: Type of semantic node
        label: Human-readable label
        concept: Core concept represented by this node
        properties: Properties and attributes of the node
        linguistic_info: Linguistic information (morphology, syntax)
        cross_language_mappings: Mappings to other languages
        confidence: Confidence in this semantic interpretation
        metadata: Additional node-specific information
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: SemanticNodeType = SemanticNodeType.CONCEPT
    label: str = ""
    concept: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    linguistic_info: Dict[str, Any] = field(default_factory=dict)
    cross_language_mappings: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the node."""
        self.properties[key] = value
    
    def get_property(self, key: str, default=None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)
    
    def add_cross_language_mapping(self, language: str, mapping: str) -> None:
        """Add a cross-language mapping."""
        self.cross_language_mappings[language] = mapping
    
    def get_cross_language_mapping(self, language: str) -> Optional[str]:
        """Get cross-language mapping for a specific language."""
        return self.cross_language_mappings.get(language)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'node_type': self.node_type.value,
            'label': self.label,
            'concept': self.concept,
            'properties': self.properties,
            'linguistic_info': self.linguistic_info,
            'cross_language_mappings': self.cross_language_mappings,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class SemanticEdge:
    """
    An edge in the semantic graph representing a relationship.
    
    Attributes:
        id: Unique identifier for the edge
        source_id: ID of the source node
        target_id: ID of the target node
        relation_type: Type of semantic relation
        label: Human-readable label for the relation
        properties: Properties of the relation
        confidence: Confidence in this relation
        metadata: Additional edge-specific information
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: SemanticRelationType = SemanticRelationType.KARAKA
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the edge."""
        self.properties[key] = value
    
    def get_property(self, key: str, default=None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'label': self.label,
            'properties': self.properties,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class SemanticGraph:
    """
    A semantic graph representing the meaning of text.
    
    Attributes:
        id: Unique identifier for the graph
        nodes: Dictionary of nodes indexed by ID
        edges: Dictionary of edges indexed by ID
        root_nodes: IDs of root nodes (entry points)
        source_text: Original text that generated this graph
        language: Primary language of the source text
        confidence: Overall confidence in the semantic interpretation
        metadata: Additional graph-specific information
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, SemanticNode] = field(default_factory=dict)
    edges: Dict[str, SemanticEdge] = field(default_factory=dict)
    root_nodes: Set[str] = field(default_factory=set)
    source_text: str = ""
    language: str = "sanskrit"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: SemanticNode) -> str:
        """Add a node to the graph and return its ID."""
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, edge: SemanticEdge) -> str:
        """Add an edge to the graph and return its ID."""
        # Validate that source and target nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found in graph")
        
        self.edges[edge.id] = edge
        return edge.id
    
    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[SemanticEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_outgoing_edges(self, node_id: str) -> List[SemanticEdge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges.values() if edge.source_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[SemanticEdge]:
        """Get all edges targeting a node."""
        return [edge for edge in self.edges.values() if edge.target_id == node_id]
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighboring node IDs."""
        neighbors = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)
        return list(neighbors)
    
    def add_root_node(self, node_id: str) -> None:
        """Mark a node as a root node."""
        if node_id in self.nodes:
            self.root_nodes.add(node_id)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id in self.nodes:
            # Remove all edges connected to this node
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            
            # Remove the node
            del self.nodes[node_id]
            self.root_nodes.discard(node_id)
    
    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the graph."""
        if edge_id in self.edges:
            del self.edges[edge_id]
    
    def merge_graph(self, other_graph: 'SemanticGraph') -> None:
        """Merge another semantic graph into this one."""
        # Add all nodes
        for node in other_graph.nodes.values():
            self.nodes[node.id] = node
        
        # Add all edges
        for edge in other_graph.edges.values():
            self.edges[edge.id] = edge
        
        # Merge root nodes
        self.root_nodes.update(other_graph.root_nodes)
        
        # Update confidence (weighted average)
        total_nodes = len(self.nodes) + len(other_graph.nodes)
        if total_nodes > 0:
            self.confidence = (
                (self.confidence * len(self.nodes) + 
                 other_graph.confidence * len(other_graph.nodes)) / total_nodes
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            'id': self.id,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'root_nodes': list(self.root_nodes),
            'source_text': self.source_text,
            'language': self.language,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def visualize_mermaid(self) -> str:
        """Generate a Mermaid diagram representation of the graph."""
        lines = ["graph TD"]
        
        # Add nodes
        for node in self.nodes.values():
            node_shape = self._get_mermaid_node_shape(node.node_type)
            label = node.label or node.concept or node.id[:8]
            lines.append(f'    {node.id}["{label}"]')
        
        # Add edges
        for edge in self.edges.values():
            arrow = self._get_mermaid_arrow(edge.relation_type)
            label = edge.label or edge.relation_type.value
            lines.append(f'    {edge.source_id} {arrow} {edge.target_id}')
            lines.append(f'    {edge.source_id} -.->|{label}| {edge.target_id}')
        
        return "\n".join(lines)
    
    def _get_mermaid_node_shape(self, node_type: SemanticNodeType) -> str:
        """Get Mermaid node shape based on semantic node type."""
        shapes = {
            SemanticNodeType.CONCEPT: "[]",
            SemanticNodeType.ENTITY: "()",
            SemanticNodeType.ACTION: "{}",
            SemanticNodeType.PROPERTY: "<>",
            SemanticNodeType.RELATION: "[]",
            SemanticNodeType.QUANTIFIER: "(())",
            SemanticNodeType.LOGICAL_OPERATOR: "[]",
            SemanticNodeType.MATHEMATICAL: "[]",
            SemanticNodeType.PROGRAMMING: "[]",
            SemanticNodeType.LINGUISTIC: "[]"
        }
        return shapes.get(node_type, "[]")
    
    def _get_mermaid_arrow(self, relation_type: SemanticRelationType) -> str:
        """Get Mermaid arrow style based on relation type."""
        arrows = {
            SemanticRelationType.AGENT: "-->",
            SemanticRelationType.PATIENT: "-->",
            SemanticRelationType.INSTRUMENT: "-.->",
            SemanticRelationType.LOCATION: "-.->",
            SemanticRelationType.TIME: "-.->",
            SemanticRelationType.CONJUNCTION: "===",
            SemanticRelationType.DISJUNCTION: "-..-",
            SemanticRelationType.IMPLICATION: "==>",
            SemanticRelationType.EQUALS: "===",
            SemanticRelationType.FUNCTION_APPLICATION: "-->",
            SemanticRelationType.ASSIGNMENT: "-->",
            SemanticRelationType.KARAKA: "-->",
            SemanticRelationType.SAMASA: "---",
            SemanticRelationType.SANDHI: "-..-"
        }
        return arrows.get(relation_type, "-->")


class SemanticGraphBuilder:
    """
    Builder class for constructing semantic graphs from various inputs.
    """
    
    def __init__(self):
        self.current_graph: Optional[SemanticGraph] = None
        self.node_cache: Dict[str, str] = {}  # concept -> node_id mapping
    
    def create_graph(self, source_text: str = "", language: str = "sanskrit") -> SemanticGraph:
        """Create a new semantic graph."""
        self.current_graph = SemanticGraph(
            source_text=source_text,
            language=language
        )
        self.node_cache.clear()
        return self.current_graph
    
    def add_concept_node(self, 
                        concept: str, 
                        label: str = "",
                        node_type: SemanticNodeType = SemanticNodeType.CONCEPT,
                        properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a concept node to the current graph."""
        if not self.current_graph:
            raise ValueError("No current graph. Call create_graph() first.")
        
        # Check if concept already exists
        if concept in self.node_cache:
            return self.node_cache[concept]
        
        node = SemanticNode(
            node_type=node_type,
            label=label or concept,
            concept=concept,
            properties=properties or {}
        )
        
        node_id = self.current_graph.add_node(node)
        self.node_cache[concept] = node_id
        return node_id
    
    def add_relation(self,
                    source_concept: str,
                    target_concept: str,
                    relation_type: SemanticRelationType,
                    label: str = "",
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a relation between two concepts."""
        if not self.current_graph:
            raise ValueError("No current graph. Call create_graph() first.")
        
        # Ensure both concepts exist as nodes
        source_id = self.node_cache.get(source_concept)
        target_id = self.node_cache.get(target_concept)
        
        if not source_id:
            source_id = self.add_concept_node(source_concept)
        if not target_id:
            target_id = self.add_concept_node(target_concept)
        
        edge = SemanticEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            label=label or relation_type.value,
            properties=properties or {}
        )
        
        return self.current_graph.add_edge(edge)
    
    def set_root_concept(self, concept: str) -> None:
        """Mark a concept as a root node."""
        if not self.current_graph:
            raise ValueError("No current graph. Call create_graph() first.")
        
        node_id = self.node_cache.get(concept)
        if node_id:
            self.current_graph.add_root_node(node_id)
    
    def get_graph(self) -> Optional[SemanticGraph]:
        """Get the current graph."""
        return self.current_graph


# Utility functions for semantic graph operations
def create_semantic_node(concept: str, 
                        node_type: SemanticNodeType = SemanticNodeType.CONCEPT,
                        label: str = "",
                        **kwargs) -> SemanticNode:
    """Create a semantic node with the given concept and type."""
    return SemanticNode(
        concept=concept,
        label=label or concept,
        node_type=node_type,
        properties=kwargs.get('properties', {}),
        linguistic_info=kwargs.get('linguistic_info', {}),
        cross_language_mappings=kwargs.get('cross_language_mappings', {}),
        confidence=kwargs.get('confidence', 1.0),
        metadata=kwargs.get('metadata', {})
    )


def create_semantic_edge(source_id: str,
                        target_id: str,
                        relation_type: SemanticRelationType,
                        **kwargs) -> SemanticEdge:
    """Create a semantic edge with the given relation type."""
    return SemanticEdge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        label=kwargs.get('label', relation_type.value),
        properties=kwargs.get('properties', {}),
        confidence=kwargs.get('confidence', 1.0),
        metadata=kwargs.get('metadata', {})
    )


def merge_semantic_graphs(graphs: List[SemanticGraph]) -> SemanticGraph:
    """Merge multiple semantic graphs into one."""
    if not graphs:
        return SemanticGraph()
    
    merged = SemanticGraph(
        source_text=" ".join(g.source_text for g in graphs if g.source_text),
        language=graphs[0].language
    )
    
    for graph in graphs:
        merged.merge_graph(graph)
    
    return merged