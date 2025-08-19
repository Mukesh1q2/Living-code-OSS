"""
Sanskrit Rewrite Engine

A sophisticated computational linguistics system for Sanskrit text processing.
"""

__version__ = "2.0.0"
__author__ = "Sanskrit Rewrite Engine Team"
__email__ = "sanskrit-engine@example.com"

# Core imports for the public API
from .engine import SanskritRewriteEngine
from .tokenizer import BasicSanskritTokenizer
from .rules import Rule, RuleRegistry
from .performance import PerformanceOptimizer, TransformationCache

# Future architecture imports
from .interfaces import (
    ProcessingStage, RuleType, TokenType, AnalysisLevel,
    AdvancedToken, AdvancedRule, ProcessingContext, LinguisticFeature
)
from .future_architecture import (
    AdvancedSanskritToken, ExtensibleSutraRule, SutraReference, SutraCategory,
    ArchitectureMigrationManager, setup_future_architecture
)
from .plugin_system import PluginLoader, LinguisticPlugin
from .enhanced_rules import ComplexLinguisticRule, RuleSetManager
from .ml_integration import SanskritMLIntegration
from .architecture_integration import (
    FutureArchitectureManager, setup_future_architecture as setup_arch,
    create_enhanced_engine, validate_future_architecture
)

__all__ = [
    # Core components
    "SanskritRewriteEngine",
    "BasicSanskritTokenizer", 
    "Rule",
    "RuleRegistry",
    "PerformanceOptimizer",
    "TransformationCache",
    
    # Future architecture interfaces
    "ProcessingStage",
    "RuleType", 
    "TokenType",
    "AnalysisLevel",
    "AdvancedToken",
    "AdvancedRule",
    "ProcessingContext",
    "LinguisticFeature",
    
    # Advanced implementations
    "AdvancedSanskritToken",
    "ExtensibleSutraRule",
    "SutraReference",
    "SutraCategory",
    "ComplexLinguisticRule",
    
    # Management and integration
    "ArchitectureMigrationManager",
    "FutureArchitectureManager",
    "PluginLoader",
    "LinguisticPlugin",
    "RuleSetManager",
    "SanskritMLIntegration",
    
    # Utility functions
    "setup_future_architecture",
    "setup_arch",
    "create_enhanced_engine",
    "validate_future_architecture",
]