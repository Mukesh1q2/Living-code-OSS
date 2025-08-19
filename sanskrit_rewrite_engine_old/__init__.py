"""
Sanskrit Rewrite Engine - A sophisticated token-level transformation system
for Sanskrit text processing based on Pāṇinian grammar principles.
"""

from .sanskrit_token import Token, TokenKind
from .tokenizer import SanskritTokenizer
from .transliterator import DevanagariIASTTransliterator
from .rule import SutraRule, SutraReference, RuleType, ParibhasaRule, GuardSystem, RuleRegistry
from .panini_engine import PaniniRuleEngine, PaniniEngineBuilder, PaniniEngineResult
from .essential_sutras import create_essential_sutras, create_essential_paribhasas
from .morphological_analyzer import (
    SanskritMorphologicalAnalyzer, MorphologicalDatabase, MorphologicalAnalysis,
    Morpheme, MorphemeType, GrammaticalCategory, SamasaType, CompoundAnalysis,
    CompoundAnalyzer, ContextDisambiguator
)
from .syntax_tree import (
    SyntaxTree, SyntaxTreeBuilder, SyntaxNode, SyntaxNodeType,
    PhraseType, SyntacticFunction, create_syntax_node, merge_syntax_trees
)
from .semantic_graph import (
    SemanticGraph, SemanticNode, SemanticEdge, SemanticGraphBuilder,
    SemanticNodeType, SemanticRelationType, create_semantic_node, 
    create_semantic_edge, merge_semantic_graphs
)
from .semantic_pipeline import (
    SemanticProcessor, CrossLanguageMapper, ProcessingResult,
    process_sanskrit_text, extract_semantic_concepts, extract_semantic_relations,
    validate_semantic_consistency
)
from .symbolic_computation import (
    MathematicalDomain, VedicMathSutra, MathematicalTransformation,
    SutraToMathMapper, SymbolicAlgebraEngine, MathematicalProofVerifier,
    SymbolicComputationEngine, create_symbolic_computation_engine,
    apply_vedic_math, verify_mathematical_identity
)
from .mcp_server import (
    SanskritMCPServer, SecuritySandbox, WorkspaceManager, GitIntegration,
    SecurityConfig, WorkspaceConfig, create_mcp_server, run_mcp_server
)
from .safe_execution import (
    SafeExecutionEnvironment, CodeExecutionManager, SecurityValidator,
    ExecutionLimits, ExecutionContext, ExecutionResult, create_safe_execution_manager
)
from .mcp_config import (
    MCPServerConfig, MCPToolConfig, MCPConfigManager,
    create_sample_config, load_config_from_file, get_default_config
)

__version__ = "0.1.0"
__all__ = [
    "SanskritTokenizer",
    "Token", 
    "TokenKind",
    "DevanagariIASTTransliterator",
    "SutraRule",
    "SutraReference", 
    "RuleType",
    "ParibhasaRule",
    "GuardSystem",
    "RuleRegistry",
    "PaniniRuleEngine",
    "PaniniEngineBuilder",
    "PaniniEngineResult",
    "create_essential_sutras",
    "create_essential_paribhasas",
    "SanskritMorphologicalAnalyzer",
    "MorphologicalDatabase",
    "MorphologicalAnalysis",
    "Morpheme",
    "MorphemeType",
    "GrammaticalCategory",
    "SamasaType",
    "CompoundAnalysis",
    "CompoundAnalyzer",
    "ContextDisambiguator",
    "SyntaxTree",
    "SyntaxTreeBuilder",
    "SyntaxNode",
    "SyntaxNodeType",
    "PhraseType",
    "SyntacticFunction",
    "create_syntax_node",
    "merge_syntax_trees",
    "SemanticGraph",
    "SemanticNode",
    "SemanticEdge",
    "SemanticGraphBuilder",
    "SemanticNodeType",
    "SemanticRelationType",
    "create_semantic_node",
    "create_semantic_edge",
    "merge_semantic_graphs",
    "SemanticProcessor",
    "CrossLanguageMapper",
    "ProcessingResult",
    "process_sanskrit_text",
    "extract_semantic_concepts",
    "extract_semantic_relations",
    "validate_semantic_consistency",
    "MathematicalDomain",
    "VedicMathSutra",
    "MathematicalTransformation",
    "SutraToMathMapper",
    "SymbolicAlgebraEngine",
    "MathematicalProofVerifier",
    "SymbolicComputationEngine",
    "create_symbolic_computation_engine",
    "apply_vedic_math",
    "verify_mathematical_identity",
    "SanskritMCPServer",
    "SecuritySandbox",
    "WorkspaceManager",
    "GitIntegration",
    "SecurityConfig",
    "WorkspaceConfig",
    "create_mcp_server",
    "run_mcp_server",
    "SafeExecutionEnvironment",
    "CodeExecutionManager",
    "SecurityValidator",
    "ExecutionLimits",
    "ExecutionContext",
    "ExecutionResult",
    "create_safe_execution_manager",
    "MCPServerConfig",
    "MCPToolConfig",
    "MCPConfigManager",
    "create_sample_config",
    "load_config_from_file",
    "get_default_config"
]