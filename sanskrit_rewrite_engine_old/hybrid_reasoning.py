"""
Hybrid Sanskrit reasoning system integrating R-Zero with external models.

This module provides hybrid reasoning capabilities that combine:
- R-Zero's multi-model self-evolving reasoning
- External LLM models (GPT/Claude) for complex Sanskrit tasks
- Sanskrit rule validation using model consensus
- Fallback mechanisms for out-of-scope tasks
- Intelligent model selection based on task complexity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
import json
import asyncio
import logging
from abc import ABC, abstractmethod

from .reasoning_core import ReasoningCore, LogicalTerm
from .r_zero_integration import SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
from .token import Token


class ModelType(Enum):
    """Types of models available for reasoning."""
    R_ZERO = "R_ZERO"
    GPT_4 = "GPT_4"
    GPT_4_TURBO = "GPT_4_TURBO"
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"
    CLAUDE_3_HAIKU = "CLAUDE_3_HAIKU"
    GEMINI_2_0_FLASH = "GEMINI_2_0_FLASH"
    GEMINI_1_5_PRO = "GEMINI_1_5_PRO"
    SANSKRIT_RULES = "SANSKRIT_RULES"
    HYBRID = "HYBRID"
    ENSEMBLE = "ENSEMBLE"


class TaskComplexity(Enum):
    """Complexity levels for Sanskrit tasks."""
    SIMPLE = "SIMPLE"          # Basic rule application
    MODERATE = "MODERATE"      # Multi-step transformations
    COMPLEX = "COMPLEX"        # Context-dependent reasoning
    EXPERT = "EXPERT"          # Advanced interpretation


@dataclass
class ModelCapability:
    """Describes a model's capabilities for Sanskrit tasks."""
    model_type: ModelType
    supported_tasks: Set[SanskritProblemType]
    complexity_range: Tuple[TaskComplexity, TaskComplexity]
    confidence_threshold: float
    response_time_ms: int
    cost_per_query: float
    accuracy_score: float
    
    def can_handle(self, task_type: SanskritProblemType, complexity: TaskComplexity) -> bool:
        """Check if model can handle a specific task and complexity."""
        return (task_type in self.supported_tasks and 
                self.complexity_range[0].value <= complexity.value <= self.complexity_range[1].value)


@dataclass
class ReasoningResult:
    """Result from a reasoning operation."""
    query: str
    answer: str
    confidence: float
    model_used: ModelType
    reasoning_steps: List[str]
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if result meets reliability threshold."""
        return self.confidence >= threshold


class ExternalModelInterface(ABC):
    """Abstract interface for external models."""
    
    @abstractmethod
    async def query(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query the external model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelCapability:
        """Get model capability information."""
        pass


class GPTInterface(ExternalModelInterface):
    """Interface for GPT models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    async def query(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query GPT model."""
        # This would integrate with OpenAI API
        # For now, return a mock response
        self.logger.info(f"Querying GPT with prompt: {prompt[:100]}...")
        
        return {
            "response": f"GPT response for: {prompt[:50]}...",
            "confidence": 0.8,
            "model": self.model_name,
            "reasoning": ["GPT reasoning step 1", "GPT reasoning step 2"]
        }
    
    def get_model_info(self) -> ModelCapability:
        """Get GPT model capabilities."""
        return ModelCapability(
            model_type=ModelType.GPT_4,
            supported_tasks={
                SanskritProblemType.TRANSLATION_SYNTHESIS,
                SanskritProblemType.GRAMMATICAL_VALIDATION,
                SanskritProblemType.MORPHOLOGICAL_ANALYSIS
            },
            complexity_range=(TaskComplexity.MODERATE, TaskComplexity.EXPERT),
            confidence_threshold=0.7,
            response_time_ms=2000,
            cost_per_query=0.02,
            accuracy_score=0.85
        )


class ClaudeInterface(ExternalModelInterface):
    """Interface for Claude models."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus"):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    async def query(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query Claude model."""
        # This would integrate with Anthropic API
        # For now, return a mock response
        self.logger.info(f"Querying Claude with prompt: {prompt[:100]}...")
        
        return {
            "response": f"Claude response for: {prompt[:50]}...",
            "confidence": 0.85,
            "model": self.model_name,
            "reasoning": ["Claude reasoning step 1", "Claude reasoning step 2"]
        }
    
    def get_model_info(self) -> ModelCapability:
        """Get Claude model capabilities."""
        return ModelCapability(
            model_type=ModelType.CLAUDE,
            supported_tasks={
                SanskritProblemType.TRANSLATION_SYNTHESIS,
                SanskritProblemType.GRAMMATICAL_VALIDATION,
                SanskritProblemType.COMPOUND_ANALYSIS,
                SanskritProblemType.WORD_DERIVATION
            },
            complexity_range=(TaskComplexity.MODERATE, TaskComplexity.EXPERT),
            confidence_threshold=0.75,
            response_time_ms=1800,
            cost_per_query=0.015,
            accuracy_score=0.88
        )


class RZeroInterface(ExternalModelInterface):
    """Interface for R-Zero reasoning system."""
    
    def __init__(self, r_zero_path: str):
        self.r_zero_path = r_zero_path
        self.logger = logging.getLogger(__name__)
    
    async def query(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query R-Zero system."""
        self.logger.info(f"Querying R-Zero with prompt: {prompt[:100]}...")
        
        # This would integrate with R-Zero system
        return {
            "response": f"R-Zero response for: {prompt[:50]}...",
            "confidence": 0.9,
            "model": "R-Zero",
            "reasoning": ["R-Zero reasoning step 1", "R-Zero reasoning step 2"]
        }
    
    def get_model_info(self) -> ModelCapability:
        """Get R-Zero model capabilities."""
        return ModelCapability(
            model_type=ModelType.R_ZERO,
            supported_tasks=set(SanskritProblemType),  # All tasks
            complexity_range=(TaskComplexity.SIMPLE, TaskComplexity.EXPERT),
            confidence_threshold=0.8,
            response_time_ms=3000,
            cost_per_query=0.0,  # Local model
            accuracy_score=0.92
        )


class TaskComplexityAnalyzer:
    """Analyzes Sanskrit task complexity for model selection."""
    
    def __init__(self):
        self.complexity_indicators = {
            TaskComplexity.SIMPLE: {
                'max_tokens': 10,
                'rule_count': 1,
                'context_dependency': False,
                'ambiguity_level': 0.1
            },
            TaskComplexity.MODERATE: {
                'max_tokens': 25,
                'rule_count': 3,
                'context_dependency': True,
                'ambiguity_level': 0.3
            },
            TaskComplexity.COMPLEX: {
                'max_tokens': 50,
                'rule_count': 5,
                'context_dependency': True,
                'ambiguity_level': 0.6
            },
            TaskComplexity.EXPERT: {
                'max_tokens': 100,
                'rule_count': 10,
                'context_dependency': True,
                'ambiguity_level': 0.8
            }
        }
    
    def analyze_complexity(self, problem: SanskritProblem, context: Dict[str, Any]) -> TaskComplexity:
        """Analyze the complexity of a Sanskrit problem."""
        score = 0
        
        # Token count analysis
        token_count = len(problem.input_text.split())
        if token_count > 50:
            score += 3
        elif token_count > 25:
            score += 2
        elif token_count > 10:
            score += 1
        
        # Problem type complexity
        type_complexity = {
            SanskritProblemType.SANDHI_APPLICATION: 1,
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: 2,
            SanskritProblemType.RULE_APPLICATION: 1,
            SanskritProblemType.WORD_DERIVATION: 3,
            SanskritProblemType.COMPOUND_ANALYSIS: 2,
            SanskritProblemType.GRAMMATICAL_VALIDATION: 2,
            SanskritProblemType.TRANSLATION_SYNTHESIS: 4
        }
        score += type_complexity.get(problem.type, 2)
        
        # Context dependency
        if context.get('requires_context', False):
            score += 2
        
        # Sutra references (more references = more complex)
        score += min(len(problem.sutra_references), 3)
        
        # Map score to complexity
        if score <= 3:
            return TaskComplexity.SIMPLE
        elif score <= 6:
            return TaskComplexity.MODERATE
        elif score <= 9:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT


class ModelSelector:
    """Selects the best model(s) for a given Sanskrit task."""
    
    def __init__(self, models: List[ExternalModelInterface]):
        self.models = {model.get_model_info().model_type: model for model in models}
        self.capabilities = {model_type: model.get_model_info() 
                           for model_type, model in self.models.items()}
        self.complexity_analyzer = TaskComplexityAnalyzer()
    
    def select_models(self, problem: SanskritProblem, context: Dict[str, Any]) -> List[ModelType]:
        """Select the best model(s) for a problem."""
        complexity = self.complexity_analyzer.analyze_complexity(problem, context)
        
        # Find capable models
        capable_models = []
        for model_type, capability in self.capabilities.items():
            if capability.can_handle(problem.type, complexity):
                capable_models.append((model_type, capability))
        
        if not capable_models:
            # Fallback to most general model
            return [ModelType.R_ZERO]
        
        # Sort by accuracy and cost efficiency
        capable_models.sort(key=lambda x: (-x[1].accuracy_score, x[1].cost_per_query))
        
        # Return top models (up to 3 for consensus)
        return [model[0] for model in capable_models[:3]]
    
    def should_use_consensus(self, problem: SanskritProblem, context: Dict[str, Any]) -> bool:
        """Determine if consensus validation is needed."""
        complexity = self.complexity_analyzer.analyze_complexity(problem, context)
        
        # Use consensus for complex tasks or when accuracy is critical
        return (complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT] or
                context.get('require_high_confidence', False))


class ConsensusValidator:
    """Validates results using consensus from multiple models."""
    
    def __init__(self, reasoning_core: ReasoningCore):
        self.reasoning_core = reasoning_core
        self.logger = logging.getLogger(__name__)
    
    async def validate_consensus(self, results: List[ReasoningResult], 
                               problem: SanskritProblem) -> ReasoningResult:
        """Validate results using consensus mechanism."""
        if not results:
            raise ValueError("No results to validate")
        
        if len(results) == 1:
            return results[0]
        
        # Analyze agreement
        answers = [result.answer for result in results]
        confidence_scores = [result.confidence for result in results]
        
        # Simple consensus: majority vote with confidence weighting
        answer_scores = {}
        for result in results:
            if result.answer not in answer_scores:
                answer_scores[result.answer] = 0
            answer_scores[result.answer] += result.confidence
        
        # Select answer with highest weighted score
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        consensus_confidence = best_answer[1] / len(results)
        
        # Combine reasoning steps
        all_reasoning_steps = []
        for result in results:
            all_reasoning_steps.extend([f"{result.model_used.value}: {step}" 
                                      for step in result.reasoning_steps])
        
        # Rule-based validation
        rule_validation = self._validate_with_rules(best_answer[0], problem)
        
        return ReasoningResult(
            query=problem.input_text,
            answer=best_answer[0],
            confidence=min(consensus_confidence, rule_validation['confidence']),
            model_used=ModelType.HYBRID,
            reasoning_steps=all_reasoning_steps,
            validation_results={
                'consensus_score': consensus_confidence,
                'rule_validation': rule_validation,
                'model_agreement': len(set(answers)) == 1,
                'participating_models': [r.model_used.value for r in results]
            }
        )
    
    def _validate_with_rules(self, answer: str, problem: SanskritProblem) -> Dict[str, Any]:
        """Validate answer using Sanskrit grammatical rules."""
        try:
            # Use reasoning core to validate
            validation_query = f"validate_sanskrit_form({answer})"
            result = self.reasoning_core.query(validation_query)
            
            return {
                'is_valid': result.get('success', False),
                'confidence': 0.9 if result.get('success', False) else 0.3,
                'rule_violations': result.get('violations', [])
            }
        except Exception as e:
            self.logger.error(f"Rule validation failed: {e}")
            return {
                'is_valid': False,
                'confidence': 0.1,
                'rule_violations': [f"Validation error: {str(e)}"]
            }


class HybridSanskritReasoner:
    """Main hybrid reasoning system combining all components."""
    
    def __init__(self, reasoning_core: ReasoningCore, 
                 external_models: List[ExternalModelInterface]):
        self.reasoning_core = reasoning_core
        self.model_selector = ModelSelector(external_models)
        self.consensus_validator = ConsensusValidator(reasoning_core)
        self.external_models = {model.get_model_info().model_type: model 
                              for model in external_models}
        self.logger = logging.getLogger(__name__)
    
    async def reason(self, problem: SanskritProblem, 
                    context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """Perform hybrid reasoning on a Sanskrit problem."""
        context = context or {}
        
        # First, try Sanskrit rules
        rule_result = await self._try_rule_based_reasoning(problem, context)
        
        # If rules are sufficient and confident, return
        if rule_result and rule_result.confidence >= 0.8:
            self.logger.info(f"Rule-based reasoning succeeded with confidence {rule_result.confidence}")
            return rule_result
        
        # Otherwise, use external models
        selected_models = self.model_selector.select_models(problem, context)
        use_consensus = self.model_selector.should_use_consensus(problem, context)
        
        if use_consensus and len(selected_models) > 1:
            return await self._consensus_reasoning(problem, context, selected_models)
        else:
            return await self._single_model_reasoning(problem, context, selected_models[0])
    
    async def _try_rule_based_reasoning(self, problem: SanskritProblem, 
                                      context: Dict[str, Any]) -> Optional[ReasoningResult]:
        """Try to solve using Sanskrit grammatical rules."""
        try:
            # Convert problem to logical query
            query = self._problem_to_query(problem)
            result = self.reasoning_core.query(query, context)
            
            if result.get('success', False):
                return ReasoningResult(
                    query=problem.input_text,
                    answer=str(result.get('solutions', [''])[0]),
                    confidence=0.9,
                    model_used=ModelType.SANSKRIT_RULES,
                    reasoning_steps=result.get('reasoning_steps', []),
                    validation_results={'rule_based': True}
                )
        except Exception as e:
            self.logger.warning(f"Rule-based reasoning failed: {e}")
        
        return None
    
    async def _single_model_reasoning(self, problem: SanskritProblem, 
                                    context: Dict[str, Any], 
                                    model_type: ModelType) -> ReasoningResult:
        """Reason using a single external model."""
        model = self.external_models[model_type]
        prompt = self._create_prompt(problem, context)
        
        result = await model.query(prompt, context)
        
        return ReasoningResult(
            query=problem.input_text,
            answer=result['response'],
            confidence=result['confidence'],
            model_used=model_type,
            reasoning_steps=result.get('reasoning', []),
            validation_results={'single_model': True}
        )
    
    async def _consensus_reasoning(self, problem: SanskritProblem, 
                                 context: Dict[str, Any], 
                                 model_types: List[ModelType]) -> ReasoningResult:
        """Reason using consensus from multiple models."""
        tasks = []
        for model_type in model_types:
            if model_type in self.external_models:
                model = self.external_models[model_type]
                prompt = self._create_prompt(problem, context)
                tasks.append(self._query_model_with_type(model, model_type, prompt, context))
        
        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ReasoningResult)]
        
        if not valid_results:
            raise RuntimeError("All model queries failed")
        
        return await self.consensus_validator.validate_consensus(valid_results, problem)
    
    async def _query_model_with_type(self, model: ExternalModelInterface, 
                                   model_type: ModelType, prompt: str, 
                                   context: Dict[str, Any]) -> ReasoningResult:
        """Query a model and wrap result with type information."""
        result = await model.query(prompt, context)
        
        return ReasoningResult(
            query=prompt,
            answer=result['response'],
            confidence=result['confidence'],
            model_used=model_type,
            reasoning_steps=result.get('reasoning', []),
            validation_results={'model_type': model_type.value}
        )
    
    def _problem_to_query(self, problem: SanskritProblem) -> str:
        """Convert Sanskrit problem to logical query."""
        query_templates = {
            SanskritProblemType.SANDHI_APPLICATION: f"apply_sandhi({problem.input_text})",
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: f"analyze_morphology({problem.input_text})",
            SanskritProblemType.WORD_DERIVATION: f"derive_word({problem.input_text})",
            SanskritProblemType.COMPOUND_ANALYSIS: f"analyze_compound({problem.input_text})",
            SanskritProblemType.RULE_APPLICATION: f"apply_rule({problem.input_text})",
            SanskritProblemType.GRAMMATICAL_VALIDATION: f"validate_grammar({problem.input_text})",
            SanskritProblemType.TRANSLATION_SYNTHESIS: f"synthesize_sanskrit({problem.input_text})"
        }
        
        return query_templates.get(problem.type, f"process_sanskrit({problem.input_text})")
    
    def _create_prompt(self, problem: SanskritProblem, context: Dict[str, Any]) -> str:
        """Create a prompt for external models."""
        prompt = f"""Sanskrit Task: {problem.type.value}
Input: {problem.input_text}
Difficulty: {problem.difficulty.value}

Instructions: {problem.explanation or 'Process the Sanskrit input according to grammatical rules.'}

Context: {json.dumps(context, indent=2) if context else 'None'}

Please provide:
1. The correct answer
2. Step-by-step reasoning
3. Relevant Sanskrit grammatical rules applied
4. Confidence level (0.0 to 1.0)

Answer:"""
        
        return prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hybrid reasoning system."""
        return {
            'reasoning_core': self.reasoning_core.get_statistics(),
            'available_models': list(self.external_models.keys()),
            'model_capabilities': {
                model_type.value: capability.__dict__ 
                for model_type, capability in self.model_selector.capabilities.items()
            }
        }