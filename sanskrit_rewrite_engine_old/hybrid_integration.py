"""
Integration module for hybrid Sanskrit reasoning system.

This module provides a unified interface for the hybrid reasoning system,
integrating R-Zero, external models, fallback mechanisms, and Sanskrit rules.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .hybrid_reasoning import (
    HybridSanskritReasoner, ModelType, ExternalModelInterface,
    GPTInterface, ClaudeInterface, RZeroInterface, ReasoningResult
)
from .fallback_mechanisms import FallbackManager, FallbackContext, FallbackReason
from .reasoning_core import ReasoningCore
from .r_zero_integration import SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
from .rule import RuleRegistry


@dataclass
class HybridConfig:
    """Configuration for hybrid Sanskrit reasoning system."""
    
    # Model configurations
    enable_gpt: bool = False
    gpt_api_key: Optional[str] = None
    gpt_model: str = "gpt-4"
    
    enable_claude: bool = False
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-opus"
    
    enable_r_zero: bool = True
    r_zero_path: str = "./external_models/r-zero"
    
    # Sanskrit rule engine
    rule_registry_path: Optional[str] = None
    sanskrit_corpus_path: Optional[str] = None
    
    # Reasoning parameters
    confidence_threshold: float = 0.7
    max_reasoning_time_ms: int = 30000
    enable_consensus: bool = True
    enable_fallback: bool = True
    
    # Performance settings
    cache_results: bool = True
    max_cache_size: int = 1000
    enable_parallel_queries: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


class HybridSanskritSystem:
    """
    Main interface for hybrid Sanskrit reasoning system.
    
    Provides a unified API for Sanskrit reasoning that automatically
    selects the best approach (rules, R-Zero, external models) based
    on the task complexity and available resources.
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.reasoning_core = self._initialize_reasoning_core()
        self.external_models = self._initialize_external_models()
        self.hybrid_reasoner = self._initialize_hybrid_reasoner()
        self.fallback_manager = self._initialize_fallback_manager()
        
        # Result cache
        self.result_cache: Dict[str, ReasoningResult] = {}
        
        self.logger.info("Hybrid Sanskrit reasoning system initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            if self.config.log_file:
                handler = logging.FileHandler(self.config.log_file)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_reasoning_core(self) -> ReasoningCore:
        """Initialize the Sanskrit reasoning core."""
        rule_registry = RuleRegistry()
        
        if self.config.rule_registry_path:
            # Load custom rules if specified
            rule_registry.load_from_file(self.config.rule_registry_path)
        
        return ReasoningCore(rule_registry)
    
    def _initialize_external_models(self) -> List[ExternalModelInterface]:
        """Initialize external model interfaces."""
        models = []
        
        # Initialize GPT interface
        if self.config.enable_gpt and self.config.gpt_api_key:
            try:
                gpt_interface = GPTInterface(
                    api_key=self.config.gpt_api_key,
                    model_name=self.config.gpt_model
                )
                models.append(gpt_interface)
                self.logger.info(f"GPT interface initialized: {self.config.gpt_model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize GPT interface: {e}")
        
        # Initialize Claude interface
        if self.config.enable_claude and self.config.claude_api_key:
            try:
                claude_interface = ClaudeInterface(
                    api_key=self.config.claude_api_key,
                    model_name=self.config.claude_model
                )
                models.append(claude_interface)
                self.logger.info(f"Claude interface initialized: {self.config.claude_model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Claude interface: {e}")
        
        # Initialize R-Zero interface
        if self.config.enable_r_zero:
            try:
                r_zero_interface = RZeroInterface(self.config.r_zero_path)
                models.append(r_zero_interface)
                self.logger.info("R-Zero interface initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize R-Zero interface: {e}")
        
        return models
    
    def _initialize_hybrid_reasoner(self) -> HybridSanskritReasoner:
        """Initialize the hybrid reasoning system."""
        return HybridSanskritReasoner(
            reasoning_core=self.reasoning_core,
            external_models=self.external_models
        )
    
    def _initialize_fallback_manager(self) -> Optional[FallbackManager]:
        """Initialize fallback manager if enabled."""
        if not self.config.enable_fallback:
            return None
        
        external_models_dict = {
            model.get_model_info().model_type: model 
            for model in self.external_models
        }
        
        return FallbackManager(
            reasoning_core=self.reasoning_core,
            external_models=external_models_dict,
            corpus_path=self.config.sanskrit_corpus_path
        )
    
    async def reason(self, 
                    input_text: str,
                    problem_type: Optional[SanskritProblemType] = None,
                    difficulty: Optional[SanskritDifficultyLevel] = None,
                    context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Perform Sanskrit reasoning on input text.
        
        Args:
            input_text: Sanskrit text to process
            problem_type: Type of Sanskrit problem (auto-detected if None)
            difficulty: Difficulty level (auto-detected if None)
            context: Additional context for reasoning
            
        Returns:
            ReasoningResult with answer and metadata
        """
        # Check cache first
        cache_key = self._generate_cache_key(input_text, problem_type, difficulty, context)
        if self.config.cache_results and cache_key in self.result_cache:
            self.logger.debug(f"Returning cached result for: {input_text[:50]}...")
            return self.result_cache[cache_key]
        
        # Auto-detect problem type and difficulty if not provided
        if problem_type is None:
            problem_type = self._detect_problem_type(input_text, context)
        
        if difficulty is None:
            difficulty = self._detect_difficulty(input_text, problem_type, context)
        
        # Create Sanskrit problem
        problem = SanskritProblem(
            id=f"hybrid_{hash(input_text)}",
            type=problem_type,
            difficulty=difficulty,
            input_text=input_text,
            expected_output="",  # Will be filled by reasoning
            context=context or {}
        )
        
        try:
            # Perform hybrid reasoning
            result = await self.hybrid_reasoner.reason(problem, context)
            
            # Cache result if enabled
            if self.config.cache_results:
                self._cache_result(cache_key, result)
            
            self.logger.info(f"Reasoning completed for: {input_text[:50]}... "
                           f"(confidence: {result.confidence:.2f}, model: {result.model_used.value})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reasoning failed for: {input_text[:50]}... Error: {e}")
            
            # Try fallback if enabled
            if self.fallback_manager:
                return await self._execute_fallback(problem, context, str(e))
            else:
                raise
    
    async def reason_batch(self, 
                          inputs: List[Dict[str, Any]],
                          max_concurrent: int = 5) -> List[ReasoningResult]:
        """
        Perform reasoning on multiple inputs concurrently.
        
        Args:
            inputs: List of input dictionaries with keys: text, type, difficulty, context
            max_concurrent: Maximum number of concurrent reasoning operations
            
        Returns:
            List of ReasoningResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def reason_single(input_dict: Dict[str, Any]) -> ReasoningResult:
            async with semaphore:
                return await self.reason(
                    input_text=input_dict['text'],
                    problem_type=input_dict.get('type'),
                    difficulty=input_dict.get('difficulty'),
                    context=input_dict.get('context')
                )
        
        tasks = [reason_single(input_dict) for input_dict in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch reasoning failed for input {i}: {result}")
                # Create error result
                error_result = ReasoningResult(
                    query=inputs[i]['text'],
                    answer=f"Error: {str(result)}",
                    confidence=0.0,
                    model_used=ModelType.SANSKRIT_RULES,
                    reasoning_steps=[f"Error occurred: {str(result)}"],
                    validation_results={'error': True}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_fallback(self, 
                              problem: SanskritProblem,
                              context: Optional[Dict[str, Any]],
                              error_message: str) -> ReasoningResult:
        """Execute fallback reasoning when primary methods fail."""
        fallback_context = FallbackContext(
            original_problem=problem,
            attempted_methods=['hybrid_reasoning'],
            partial_results=[],
            failure_reasons=[FallbackReason.RULE_COVERAGE_INSUFFICIENT],
            confidence_threshold=self.config.confidence_threshold,
            time_budget_ms=self.config.max_reasoning_time_ms
        )
        
        try:
            fallback_result = await self.fallback_manager.execute_fallback(fallback_context)
            self.logger.warning(f"Fallback reasoning used: {fallback_result.strategy_used.value}")
            return fallback_result.result
        except Exception as fallback_error:
            self.logger.error(f"Fallback reasoning also failed: {fallback_error}")
            
            # Return minimal error result
            return ReasoningResult(
                query=problem.input_text,
                answer=f"Reasoning failed: {error_message}",
                confidence=0.0,
                model_used=ModelType.SANSKRIT_RULES,
                reasoning_steps=[f"Primary reasoning failed: {error_message}",
                               f"Fallback reasoning failed: {str(fallback_error)}"],
                validation_results={'complete_failure': True}
            )
    
    def _detect_problem_type(self, 
                           input_text: str, 
                           context: Optional[Dict[str, Any]]) -> SanskritProblemType:
        """Auto-detect the type of Sanskrit problem."""
        text_lower = input_text.lower()
        
        # Simple heuristics for problem type detection
        if '+' in input_text or 'sandhi' in text_lower:
            return SanskritProblemType.SANDHI_APPLICATION
        elif any(word in text_lower for word in ['analyze', 'morphology', 'inflection']):
            return SanskritProblemType.MORPHOLOGICAL_ANALYSIS
        elif any(word in text_lower for word in ['derive', 'derivation', 'etymology']):
            return SanskritProblemType.WORD_DERIVATION
        elif any(word in text_lower for word in ['compound', 'samasa']):
            return SanskritProblemType.COMPOUND_ANALYSIS
        elif any(word in text_lower for word in ['validate', 'correct', 'grammar']):
            return SanskritProblemType.GRAMMATICAL_VALIDATION
        elif any(word in text_lower for word in ['translate', 'synthesize', 'create']):
            return SanskritProblemType.TRANSLATION_SYNTHESIS
        else:
            return SanskritProblemType.RULE_APPLICATION  # Default
    
    def _detect_difficulty(self, 
                         input_text: str,
                         problem_type: SanskritProblemType,
                         context: Optional[Dict[str, Any]]) -> SanskritDifficultyLevel:
        """Auto-detect the difficulty level of the problem."""
        # Simple heuristics based on text length and complexity
        text_length = len(input_text.split())
        
        if text_length <= 5:
            return SanskritDifficultyLevel.BEGINNER
        elif text_length <= 15:
            return SanskritDifficultyLevel.INTERMEDIATE
        elif text_length <= 30:
            return SanskritDifficultyLevel.ADVANCED
        else:
            return SanskritDifficultyLevel.EXPERT
    
    def _generate_cache_key(self, 
                          input_text: str,
                          problem_type: Optional[SanskritProblemType],
                          difficulty: Optional[SanskritDifficultyLevel],
                          context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for result caching."""
        key_parts = [
            input_text,
            problem_type.value if problem_type else "auto",
            difficulty.value if difficulty else "auto",
            str(sorted(context.items())) if context else "no_context"
        ]
        return "|".join(key_parts)
    
    def _cache_result(self, cache_key: str, result: ReasoningResult) -> None:
        """Cache reasoning result."""
        if len(self.result_cache) >= self.config.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status information about the hybrid system."""
        return {
            'config': {
                'gpt_enabled': self.config.enable_gpt,
                'claude_enabled': self.config.enable_claude,
                'r_zero_enabled': self.config.enable_r_zero,
                'fallback_enabled': self.config.enable_fallback,
                'consensus_enabled': self.config.enable_consensus
            },
            'models': {
                'available_models': len(self.external_models),
                'model_types': [model.get_model_info().model_type.value 
                              for model in self.external_models]
            },
            'cache': {
                'cached_results': len(self.result_cache),
                'cache_enabled': self.config.cache_results,
                'max_cache_size': self.config.max_cache_size
            },
            'reasoning_core': self.reasoning_core.get_statistics() if self.reasoning_core else None,
            'fallback': {
                'available_strategies': (self.fallback_manager.get_available_strategies() 
                                       if self.fallback_manager else [])
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.result_cache.clear()
        self.logger.info("Result cache cleared")
    
    async def shutdown(self) -> None:
        """Shutdown the hybrid system gracefully."""
        self.logger.info("Shutting down hybrid Sanskrit reasoning system")
        
        # Clear cache
        self.clear_cache()
        
        # Close any open connections or resources
        # (In a real implementation, this would close API connections, etc.)
        
        self.logger.info("Hybrid system shutdown complete")


# Convenience functions for easy usage

async def reason_sanskrit(text: str, 
                         config: Optional[HybridConfig] = None,
                         **kwargs) -> ReasoningResult:
    """
    Convenience function for single Sanskrit reasoning operation.
    
    Args:
        text: Sanskrit text to process
        config: System configuration (uses default if None)
        **kwargs: Additional arguments passed to reason()
        
    Returns:
        ReasoningResult
    """
    if config is None:
        config = HybridConfig()
    
    system = HybridSanskritSystem(config)
    try:
        return await system.reason(text, **kwargs)
    finally:
        await system.shutdown()


def create_default_config(gpt_key: Optional[str] = None,
                         claude_key: Optional[str] = None) -> HybridConfig:
    """
    Create a default configuration for the hybrid system.
    
    Args:
        gpt_key: OpenAI API key
        claude_key: Anthropic API key
        
    Returns:
        HybridConfig with sensible defaults
    """
    return HybridConfig(
        enable_gpt=gpt_key is not None,
        gpt_api_key=gpt_key,
        enable_claude=claude_key is not None,
        claude_api_key=claude_key,
        enable_r_zero=True,
        confidence_threshold=0.7,
        enable_consensus=True,
        enable_fallback=True,
        cache_results=True
    )