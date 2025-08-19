"""
Fallback mechanisms for Sanskrit reasoning when rules are insufficient.

This module provides sophisticated fallback strategies for handling Sanskrit
tasks that cannot be resolved through traditional grammatical rules alone.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
import logging
from abc import ABC, abstractmethod

from .hybrid_reasoning import (
    ReasoningResult, ModelType, TaskComplexity, SanskritProblem, 
    SanskritProblemType, ExternalModelInterface
)
from .reasoning_core import ReasoningCore


class FallbackReason(Enum):
    """Reasons why fallback is needed."""
    RULE_COVERAGE_INSUFFICIENT = "RULE_COVERAGE_INSUFFICIENT"
    AMBIGUOUS_CONTEXT = "AMBIGUOUS_CONTEXT"
    INCOMPLETE_KNOWLEDGE = "INCOMPLETE_KNOWLEDGE"
    PARSING_FAILURE = "PARSING_FAILURE"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    CONFIDENCE_TOO_LOW = "CONFIDENCE_TOO_LOW"


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    EXTERNAL_MODEL_QUERY = "EXTERNAL_MODEL_QUERY"
    CORPUS_SIMILARITY_SEARCH = "CORPUS_SIMILARITY_SEARCH"
    PARTIAL_RULE_APPLICATION = "PARTIAL_RULE_APPLICATION"
    HUMAN_EXPERT_CONSULTATION = "HUMAN_EXPERT_CONSULTATION"
    APPROXIMATION_WITH_WARNING = "APPROXIMATION_WITH_WARNING"
    GRACEFUL_DEGRADATION = "GRACEFUL_DEGRADATION"


@dataclass
class FallbackContext:
    """Context information for fallback decisions."""
    original_problem: SanskritProblem
    attempted_methods: List[str]
    partial_results: List[ReasoningResult]
    failure_reasons: List[FallbackReason]
    confidence_threshold: float
    time_budget_ms: int
    allow_approximation: bool = True
    require_explanation: bool = True


@dataclass
class FallbackResult:
    """Result from a fallback mechanism."""
    strategy_used: FallbackStrategy
    result: ReasoningResult
    fallback_confidence: float
    warnings: List[str] = field(default_factory=list)
    approximations_made: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallbackStrategy_ABC(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    async def can_handle(self, context: FallbackContext) -> bool:
        """Check if this strategy can handle the fallback situation."""
        pass
    
    @abstractmethod
    async def execute(self, context: FallbackContext) -> FallbackResult:
        """Execute the fallback strategy."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get strategy priority (lower = higher priority)."""
        pass


class ExternalModelFallback(FallbackStrategy_ABC):
    """Fallback to external models when rules fail."""
    
    def __init__(self, external_models: Dict[ModelType, ExternalModelInterface]):
        self.external_models = external_models
        self.logger = logging.getLogger(__name__)
    
    async def can_handle(self, context: FallbackContext) -> bool:
        """Check if external models can handle the problem."""
        # Can handle most problems if we have available models
        return len(self.external_models) > 0
    
    async def execute(self, context: FallbackContext) -> FallbackResult:
        """Execute external model fallback."""
        problem = context.original_problem
        
        # Select best available model for the task
        best_model = self._select_best_model(problem)
        
        if not best_model:
            raise RuntimeError("No suitable external model available")
        
        # Create enhanced prompt with context about the failure
        prompt = self._create_fallback_prompt(problem, context)
        
        # Query the model
        model_result = await best_model.query(prompt, {
            'fallback_context': True,
            'attempted_methods': context.attempted_methods,
            'failure_reasons': [r.value for r in context.failure_reasons]
        })
        
        # Create reasoning result
        reasoning_result = ReasoningResult(
            query=problem.input_text,
            answer=model_result['response'],
            confidence=model_result['confidence'] * 0.9,  # Slight penalty for fallback
            model_used=best_model.get_model_info().model_type,
            reasoning_steps=model_result.get('reasoning', []),
            validation_results={'fallback_used': True}
        )
        
        warnings = []
        if model_result['confidence'] < context.confidence_threshold:
            warnings.append(f"Model confidence ({model_result['confidence']:.2f}) below threshold")
        
        return FallbackResult(
            strategy_used=FallbackStrategy.EXTERNAL_MODEL_QUERY,
            result=reasoning_result,
            fallback_confidence=model_result['confidence'],
            warnings=warnings,
            metadata={'model_used': best_model.get_model_info().model_type.value}
        )
    
    def get_priority(self) -> int:
        """High priority fallback strategy."""
        return 1
    
    def _select_best_model(self, problem: SanskritProblem) -> Optional[ExternalModelInterface]:
        """Select the best model for the problem type."""
        suitable_models = []
        
        for model_type, model in self.external_models.items():
            capability = model.get_model_info()
            if problem.type in capability.supported_tasks:
                suitable_models.append((model, capability.accuracy_score))
        
        if not suitable_models:
            # Return any available model as last resort
            return next(iter(self.external_models.values())) if self.external_models else None
        
        # Return model with highest accuracy
        return max(suitable_models, key=lambda x: x[1])[0]
    
    def _create_fallback_prompt(self, problem: SanskritProblem, context: FallbackContext) -> str:
        """Create an enhanced prompt for fallback scenarios."""
        prompt = f"""Sanskrit Problem (Fallback Mode):

Original Task: {problem.type.value}
Input: {problem.input_text}
Difficulty: {problem.difficulty.value}

IMPORTANT: Traditional Sanskrit grammatical rules were insufficient for this problem.
Attempted methods: {', '.join(context.attempted_methods)}
Failure reasons: {', '.join([r.value for r in context.failure_reasons])}

Please provide your best analysis considering:
1. The limitations of rule-based approaches for this specific case
2. Contextual interpretation that may be required
3. Any ambiguities or special cases involved
4. Your confidence level and reasoning

Your response should include:
- The most likely correct answer
- Explanation of why rule-based methods failed
- Alternative interpretations if applicable
- Confidence assessment (0.0 to 1.0)

Answer:"""
        
        return prompt


class CorpusSimilarityFallback(FallbackStrategy_ABC):
    """Fallback using corpus similarity search."""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.logger = logging.getLogger(__name__)
        # In a real implementation, this would load and index a Sanskrit corpus
        self.corpus_examples = self._load_corpus_examples()
    
    async def can_handle(self, context: FallbackContext) -> bool:
        """Check if corpus search can help."""
        # Can help with most problems if we have relevant examples
        return len(self.corpus_examples) > 0
    
    async def execute(self, context: FallbackContext) -> FallbackResult:
        """Execute corpus similarity fallback."""
        problem = context.original_problem
        
        # Find similar examples in corpus
        similar_examples = self._find_similar_examples(problem.input_text)
        
        if not similar_examples:
            raise RuntimeError("No similar examples found in corpus")
        
        # Analyze patterns from similar examples
        pattern_analysis = self._analyze_patterns(similar_examples)
        
        # Generate answer based on patterns
        answer = self._generate_answer_from_patterns(problem, pattern_analysis)
        
        reasoning_steps = [
            f"Found {len(similar_examples)} similar examples in corpus",
            f"Identified pattern: {pattern_analysis['primary_pattern']}",
            f"Applied pattern to generate answer: {answer}"
        ]
        
        reasoning_result = ReasoningResult(
            query=problem.input_text,
            answer=answer,
            confidence=pattern_analysis['confidence'],
            model_used=ModelType.SANSKRIT_RULES,  # Using corpus as rule extension
            reasoning_steps=reasoning_steps,
            validation_results={'corpus_based': True}
        )
        
        return FallbackResult(
            strategy_used=FallbackStrategy.CORPUS_SIMILARITY_SEARCH,
            result=reasoning_result,
            fallback_confidence=pattern_analysis['confidence'],
            metadata={
                'similar_examples_count': len(similar_examples),
                'pattern_confidence': pattern_analysis['confidence']
            }
        )
    
    def get_priority(self) -> int:
        """Medium priority fallback strategy."""
        return 2
    
    def _load_corpus_examples(self) -> List[Dict[str, str]]:
        """Load corpus examples (mock implementation)."""
        # In a real implementation, this would load from actual Sanskrit corpus
        return [
            {'input': 'rāma + iti', 'output': 'rāmeti', 'pattern': 'a_vowel_sandhi'},
            {'input': 'kṛṣṇa + eva', 'output': 'kṛṣṇaiva', 'pattern': 'a_vowel_sandhi'},
            {'input': 'deva + ālaya', 'output': 'devālaya', 'pattern': 'a_vowel_sandhi'},
        ]
    
    def _find_similar_examples(self, input_text: str) -> List[Dict[str, str]]:
        """Find similar examples in corpus."""
        # Simple similarity based on common patterns
        similar = []
        for example in self.corpus_examples:
            if self._calculate_similarity(input_text, example['input']) > 0.5:
                similar.append(example)
        return similar
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)."""
        # Simple character-based similarity
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)
        return len(common_chars) / len(total_chars) if total_chars else 0.0
    
    def _analyze_patterns(self, examples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze patterns from similar examples."""
        patterns = {}
        for example in examples:
            pattern = example.get('pattern', 'unknown')
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        if patterns:
            primary_pattern = max(patterns.items(), key=lambda x: x[1])[0]
            confidence = patterns[primary_pattern] / len(examples)
        else:
            primary_pattern = 'unknown'
            confidence = 0.3
        
        return {
            'primary_pattern': primary_pattern,
            'confidence': confidence,
            'pattern_distribution': patterns
        }
    
    def _generate_answer_from_patterns(self, problem: SanskritProblem, 
                                     pattern_analysis: Dict[str, Any]) -> str:
        """Generate answer based on identified patterns."""
        # Simplified pattern application
        if pattern_analysis['primary_pattern'] == 'a_vowel_sandhi':
            # Apply basic vowel sandhi
            if ' + ' in problem.input_text:
                parts = problem.input_text.split(' + ')
                if len(parts) == 2 and parts[0].endswith('a'):
                    return parts[0][:-1] + parts[1]
        
        return f"Pattern-based result for: {problem.input_text}"


class PartialRuleApplicationFallback(FallbackStrategy_ABC):
    """Fallback using partial rule application with approximation."""
    
    def __init__(self, reasoning_core: ReasoningCore):
        self.reasoning_core = reasoning_core
        self.logger = logging.getLogger(__name__)
    
    async def can_handle(self, context: FallbackContext) -> bool:
        """Check if partial rule application is possible."""
        return context.allow_approximation and len(context.partial_results) > 0
    
    async def execute(self, context: FallbackContext) -> FallbackResult:
        """Execute partial rule application fallback."""
        problem = context.original_problem
        
        # Combine partial results
        combined_result = self._combine_partial_results(context.partial_results)
        
        # Apply additional heuristics
        enhanced_result = self._apply_heuristics(combined_result, problem)
        
        warnings = [
            "Result based on partial rule application",
            "May not be fully accurate due to incomplete rule coverage"
        ]
        
        approximations = [
            "Used heuristic approximation for uncovered cases",
            "Combined multiple partial results"
        ]
        
        return FallbackResult(
            strategy_used=FallbackStrategy.PARTIAL_RULE_APPLICATION,
            result=enhanced_result,
            fallback_confidence=enhanced_result.confidence * 0.8,  # Penalty for approximation
            warnings=warnings,
            approximations_made=approximations,
            metadata={'partial_results_count': len(context.partial_results)}
        )
    
    def get_priority(self) -> int:
        """Lower priority fallback strategy."""
        return 3
    
    def _combine_partial_results(self, partial_results: List[ReasoningResult]) -> ReasoningResult:
        """Combine multiple partial results."""
        if not partial_results:
            raise ValueError("No partial results to combine")
        
        if len(partial_results) == 1:
            return partial_results[0]
        
        # Simple combination: take highest confidence result as base
        best_result = max(partial_results, key=lambda r: r.confidence)
        
        # Combine reasoning steps
        all_steps = []
        for result in partial_results:
            all_steps.extend([f"Partial: {step}" for step in result.reasoning_steps])
        
        return ReasoningResult(
            query=best_result.query,
            answer=best_result.answer,
            confidence=best_result.confidence,
            model_used=best_result.model_used,
            reasoning_steps=all_steps,
            validation_results={'combined_partial': True}
        )
    
    def _apply_heuristics(self, result: ReasoningResult, problem: SanskritProblem) -> ReasoningResult:
        """Apply heuristics to enhance partial results."""
        # Simple heuristic enhancements
        enhanced_answer = result.answer
        
        # Add common Sanskrit transformations if missing
        if problem.type == SanskritProblemType.SANDHI_APPLICATION:
            if ' + ' in problem.input_text and enhanced_answer == problem.input_text:
                # Apply basic joining if no transformation occurred
                enhanced_answer = problem.input_text.replace(' + ', '')
        
        return ReasoningResult(
            query=result.query,
            answer=enhanced_answer,
            confidence=result.confidence * 0.9,  # Slight penalty for heuristics
            model_used=result.model_used,
            reasoning_steps=result.reasoning_steps + ["Applied heuristic enhancements"],
            validation_results=result.validation_results
        )


class GracefulDegradationFallback(FallbackStrategy_ABC):
    """Graceful degradation when all else fails."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def can_handle(self, context: FallbackContext) -> bool:
        """Always can handle as last resort."""
        return True
    
    async def execute(self, context: FallbackContext) -> FallbackResult:
        """Execute graceful degradation."""
        problem = context.original_problem
        
        # Provide basic analysis with clear limitations
        answer = self._generate_basic_response(problem)
        
        reasoning_steps = [
            "All advanced reasoning methods failed",
            "Providing basic structural analysis",
            "Result should be verified by Sanskrit expert"
        ]
        
        reasoning_result = ReasoningResult(
            query=problem.input_text,
            answer=answer,
            confidence=0.2,  # Very low confidence
            model_used=ModelType.SANSKRIT_RULES,
            reasoning_steps=reasoning_steps,
            validation_results={'graceful_degradation': True}
        )
        
        warnings = [
            "All reasoning methods failed",
            "Result is basic structural analysis only",
            "Expert verification strongly recommended",
            "Confidence is very low"
        ]
        
        return FallbackResult(
            strategy_used=FallbackStrategy.GRACEFUL_DEGRADATION,
            result=reasoning_result,
            fallback_confidence=0.1,
            warnings=warnings,
            metadata={'last_resort': True}
        )
    
    def get_priority(self) -> int:
        """Lowest priority - last resort."""
        return 999
    
    def _generate_basic_response(self, problem: SanskritProblem) -> str:
        """Generate basic response when all else fails."""
        responses = {
            SanskritProblemType.SANDHI_APPLICATION: f"Basic joining: {problem.input_text.replace(' + ', '')}",
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: f"Word structure analysis needed for: {problem.input_text}",
            SanskritProblemType.WORD_DERIVATION: f"Derivation analysis required for: {problem.input_text}",
            SanskritProblemType.COMPOUND_ANALYSIS: f"Compound breakdown needed for: {problem.input_text}",
            SanskritProblemType.RULE_APPLICATION: f"Rule application required for: {problem.input_text}",
            SanskritProblemType.GRAMMATICAL_VALIDATION: f"Grammatical validation needed for: {problem.input_text}",
            SanskritProblemType.TRANSLATION_SYNTHESIS: f"Translation synthesis required for: {problem.input_text}"
        }
        
        return responses.get(problem.type, f"Analysis required for: {problem.input_text}")


class FallbackManager:
    """Manages fallback strategies and orchestrates their execution."""
    
    def __init__(self, reasoning_core: ReasoningCore, 
                 external_models: Dict[ModelType, ExternalModelInterface],
                 corpus_path: Optional[str] = None):
        self.reasoning_core = reasoning_core
        self.strategies: List[FallbackStrategy_ABC] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback strategies
        self._initialize_strategies(external_models, corpus_path)
    
    def _initialize_strategies(self, external_models: Dict[ModelType, ExternalModelInterface],
                             corpus_path: Optional[str]):
        """Initialize available fallback strategies."""
        # External model fallback
        if external_models:
            self.strategies.append(ExternalModelFallback(external_models))
        
        # Corpus similarity fallback
        if corpus_path:
            self.strategies.append(CorpusSimilarityFallback(corpus_path))
        
        # Partial rule application fallback
        self.strategies.append(PartialRuleApplicationFallback(self.reasoning_core))
        
        # Graceful degradation (always available)
        self.strategies.append(GracefulDegradationFallback())
        
        # Sort by priority
        self.strategies.sort(key=lambda s: s.get_priority())
    
    async def execute_fallback(self, context: FallbackContext) -> FallbackResult:
        """Execute the best available fallback strategy."""
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(context):
                    self.logger.info(f"Executing fallback strategy: {strategy.__class__.__name__}")
                    return await strategy.execute(context)
            except Exception as e:
                self.logger.error(f"Fallback strategy {strategy.__class__.__name__} failed: {e}")
                continue
        
        raise RuntimeError("All fallback strategies failed")
    
    def add_strategy(self, strategy: FallbackStrategy_ABC):
        """Add a custom fallback strategy."""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available fallback strategies."""
        return [strategy.__class__.__name__ for strategy in self.strategies]