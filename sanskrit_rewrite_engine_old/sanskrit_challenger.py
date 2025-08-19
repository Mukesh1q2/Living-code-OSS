"""
Sanskrit Challenger Model for R-Zero Co-evolutionary Loop.

This module adapts R-Zero's Challenger model to generate Sanskrit grammatical problems
for training the Solver model in a co-evolutionary manner.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import vllm
from transformers import AutoTokenizer

from .r_zero_integration import (
    SanskritProblem, SanskritProblemType, SanskritDifficultyLevel,
    SanskritProblemGenerator
)
from .r_zero_config import SanskritRZeroConfig


logger = logging.getLogger(__name__)


@dataclass
class ChallengerConfig:
    """Configuration for Sanskrit Challenger model."""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 2048
    num_samples: int = 10
    gpu_memory_utilization: float = 0.7
    seed: int = 42


class SanskritChallenger:
    """
    Sanskrit Challenger model that generates increasingly difficult Sanskrit problems.
    
    This model learns to create challenging Sanskrit grammatical problems that push
    the Solver model to improve its Sanskrit reasoning capabilities.
    """
    
    def __init__(self, 
                 config: ChallengerConfig,
                 problem_generator: SanskritProblemGenerator,
                 storage_path: str = "./r_zero_storage"):
        """
        Initialize the Sanskrit Challenger.
        
        Args:
            config: Challenger configuration
            problem_generator: Sanskrit problem generator
            storage_path: Path for storing generated problems
        """
        self.config = config
        self.problem_generator = problem_generator
        self.storage_path = Path(storage_path)
        self.generated_questions_path = self.storage_path / "generated_question"
        self.generated_questions_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Problem generation templates
        self._initialize_templates()
    
    def _initialize_model(self):
        """Initialize the VLLM model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model = vllm.LLM(
                model=self.config.model_path,
                tokenizer=self.config.model_path,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                seed=self.config.seed,
            )
            logger.info(f"Initialized Challenger model: {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Challenger model: {e}")
            raise
    
    def _initialize_templates(self):
        """Initialize problem generation templates."""
        self.system_prompt = """You are an expert Sanskrit grammarian and computational linguist specializing in Pāṇinian grammar.

Your task is to generate challenging Sanskrit grammatical problems that test deep understanding of:
- Sandhi (phonological transformations)
- Morphological analysis (dhātu, pratyaya identification)
- Word derivation (śabda-prakriyā)
- Compound analysis (samāsa)
- Rule application (sūtra-based transformations)
- Grammatical validation

Generate problems that are:
1. Linguistically accurate according to Pāṇinian principles
2. Appropriately challenging for the specified difficulty level
3. Clear and unambiguous in their requirements
4. Solvable using systematic grammatical analysis

Format your output exactly as:
<problem_type>TYPE</problem_type>
<difficulty>LEVEL</difficulty>
<input>Sanskrit text or expression to analyze</input>
<expected_output>Correct answer with explanation</expected_output>
<sutra_references>Relevant Pāṇini sūtra numbers</sutra_references>
<explanation>Brief explanation of the grammatical principle</explanation>"""

        self.problem_type_descriptions = {
            SanskritProblemType.SANDHI_APPLICATION: "Apply sandhi rules to combine words or morphemes",
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: "Analyze word structure into morphemes and grammatical categories",
            SanskritProblemType.WORD_DERIVATION: "Provide step-by-step derivation from root to final form",
            SanskritProblemType.COMPOUND_ANALYSIS: "Analyze compound structure and identify samāsa type",
            SanskritProblemType.RULE_APPLICATION: "Apply specific Pāṇini rules to given input",
            SanskritProblemType.GRAMMATICAL_VALIDATION: "Validate grammatical correctness and suggest corrections",
        }
    
    def generate_problems(self, 
                         problem_type: SanskritProblemType,
                         difficulty: SanskritDifficultyLevel,
                         num_problems: int = 10,
                         context: Optional[Dict[str, Any]] = None) -> List[SanskritProblem]:
        """
        Generate Sanskrit problems using the Challenger model.
        
        Args:
            problem_type: Type of problems to generate
            difficulty: Difficulty level
            num_problems: Number of problems to generate
            context: Additional context for problem generation
            
        Returns:
            List of generated Sanskrit problems
        """
        logger.info(f"Generating {num_problems} {problem_type.value} problems at {difficulty.value} level")
        
        # Create generation prompt
        user_prompt = self._create_generation_prompt(problem_type, difficulty, context)
        
        # Generate problems using the model
        generated_texts = self._generate_with_model(user_prompt, num_problems)
        
        # Parse generated problems
        problems = []
        for i, text in enumerate(generated_texts):
            try:
                problem = self._parse_generated_problem(text, problem_type, difficulty, i)
                if problem:
                    problems.append(problem)
            except Exception as e:
                logger.warning(f"Failed to parse generated problem {i}: {e}")
        
        # Fallback to template-based generation if model generation fails
        if len(problems) < num_problems // 2:
            logger.warning("Model generation insufficient, using template fallback")
            fallback_problems = self.problem_generator.generate_problems(
                count=num_problems - len(problems),
                problem_types=[problem_type],
                difficulty_distribution={difficulty: 1.0}
            )
            problems.extend(fallback_problems)
        
        logger.info(f"Successfully generated {len(problems)} problems")
        return problems[:num_problems]
    
    def _create_generation_prompt(self, 
                                 problem_type: SanskritProblemType,
                                 difficulty: SanskritDifficultyLevel,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for problem generation."""
        type_description = self.problem_type_descriptions.get(problem_type, "Sanskrit grammatical analysis")
        
        difficulty_guidance = {
            SanskritDifficultyLevel.BEGINNER: "Use simple, common words and basic rules",
            SanskritDifficultyLevel.INTERMEDIATE: "Use moderately complex constructions and multiple rules",
            SanskritDifficultyLevel.ADVANCED: "Use complex constructions requiring deep grammatical knowledge",
            SanskritDifficultyLevel.EXPERT: "Use highly complex, rare constructions and subtle rule interactions"
        }
        
        context_info = ""
        if context:
            if "focus_areas" in context:
                context_info += f"\nFocus on these areas: {', '.join(context['focus_areas'])}"
            if "avoid_patterns" in context:
                context_info += f"\nAvoid these patterns: {', '.join(context['avoid_patterns'])}"
        
        return f"""Generate a {difficulty.value.lower()} level Sanskrit problem for: {type_description}

Difficulty guidance: {difficulty_guidance[difficulty]}
{context_info}

Create a problem that tests understanding of Pāṇinian grammar principles.
Ensure the problem is solvable and linguistically accurate.

Generate exactly one problem following the specified format."""
    
    def _generate_with_model(self, user_prompt: str, num_samples: int) -> List[str]:
        """Generate text using the VLLM model."""
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False,
                add_generation_prompt=True, 
                add_special_tokens=True
            )
        else:
            prompt = f"system: {chat[0]['content']}\nuser: {chat[1]['content']}"
        
        sample_params = vllm.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            n=1,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        # Generate multiple samples
        completions = self.model.generate([prompt] * num_samples, sampling_params=sample_params)
        
        return [completion.outputs[0].text for completion in completions]
    
    def _parse_generated_problem(self, 
                                text: str, 
                                problem_type: SanskritProblemType,
                                difficulty: SanskritDifficultyLevel,
                                index: int) -> Optional[SanskritProblem]:
        """Parse generated text into a SanskritProblem."""
        try:
            # Extract components using regex
            import re
            
            type_match = re.search(r'<problem_type>(.*?)</problem_type>', text, re.DOTALL)
            difficulty_match = re.search(r'<difficulty>(.*?)</difficulty>', text, re.DOTALL)
            input_match = re.search(r'<input>(.*?)</input>', text, re.DOTALL)
            output_match = re.search(r'<expected_output>(.*?)</expected_output>', text, re.DOTALL)
            sutra_match = re.search(r'<sutra_references>(.*?)</sutra_references>', text, re.DOTALL)
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', text, re.DOTALL)
            
            if not (input_match and output_match):
                return None
            
            input_text = input_match.group(1).strip()
            expected_output = output_match.group(1).strip()
            
            # Extract sutra references
            sutra_references = []
            if sutra_match:
                sutra_text = sutra_match.group(1).strip()
                # Extract sūtra numbers (e.g., "1.1.1", "6.1.87")
                sutra_numbers = re.findall(r'\d+\.\d+\.\d+', sutra_text)
                sutra_references = sutra_numbers
            
            explanation = explanation_match.group(1).strip() if explanation_match else None
            
            return SanskritProblem(
                id=f"challenger_{problem_type.value.lower()}_{difficulty.value.lower()}_{index:04d}",
                type=problem_type,
                difficulty=difficulty,
                input_text=input_text,
                expected_output=expected_output,
                sutra_references=sutra_references,
                explanation=explanation,
                metadata={
                    "generated_by": "challenger_model",
                    "model_path": self.config.model_path,
                    "generation_params": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Error parsing generated problem: {e}")
            return None
    
    def save_generated_problems(self, 
                               problems: List[SanskritProblem],
                               filename: str) -> str:
        """Save generated problems to file."""
        filepath = self.generated_questions_path / f"{filename}.json"
        
        # Convert problems to R-Zero format
        r_zero_problems = [problem.to_r_zero_format() for problem in problems]
        
        # Add metadata
        data = {
            "problems": r_zero_problems,
            "metadata": {
                "generator": "sanskrit_challenger",
                "model_path": self.config.model_path,
                "generation_time": str(Path().cwd()),
                "total_problems": len(problems)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(problems)} problems to {filepath}")
        return str(filepath)
    
    def generate_adaptive_problems(self, 
                                  solver_performance: Dict[str, float],
                                  num_problems: int = 20) -> List[SanskritProblem]:
        """
        Generate problems adaptively based on Solver performance.
        
        Args:
            solver_performance: Performance metrics by problem type
            num_problems: Number of problems to generate
            
        Returns:
            List of adaptively generated problems
        """
        logger.info("Generating adaptive problems based on Solver performance")
        
        # Identify weak areas
        weak_areas = []
        for problem_type, performance in solver_performance.items():
            if performance < 0.7:  # Below 70% accuracy
                try:
                    weak_areas.append(SanskritProblemType(problem_type))
                except ValueError:
                    continue
        
        if not weak_areas:
            # If no weak areas, focus on advanced problems
            weak_areas = list(SanskritProblemType)
        
        # Generate more problems for weak areas
        problems = []
        problems_per_type = max(1, num_problems // len(weak_areas))
        
        for problem_type in weak_areas:
            # Increase difficulty for areas where Solver is performing well
            current_performance = solver_performance.get(problem_type.value, 0.5)
            
            if current_performance > 0.8:
                difficulty = SanskritDifficultyLevel.EXPERT
            elif current_performance > 0.6:
                difficulty = SanskritDifficultyLevel.ADVANCED
            elif current_performance > 0.4:
                difficulty = SanskritDifficultyLevel.INTERMEDIATE
            else:
                difficulty = SanskritDifficultyLevel.BEGINNER
            
            type_problems = self.generate_problems(
                problem_type=problem_type,
                difficulty=difficulty,
                num_problems=problems_per_type,
                context={"focus_areas": ["challenging_cases", "edge_cases"]}
            )
            problems.extend(type_problems)
        
        return problems[:num_problems]


def create_sanskrit_challenger(config_path: Optional[str] = None) -> SanskritChallenger:
    """
    Create a Sanskrit Challenger instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured SanskritChallenger
    """
    # Load R-Zero config
    if config_path and Path(config_path).exists():
        r_zero_config = SanskritRZeroConfig.load_from_yaml(config_path)
    else:
        r_zero_config = SanskritRZeroConfig()
    
    # Create challenger config
    challenger_config = ChallengerConfig(
        temperature=r_zero_config.temperature,
        top_p=r_zero_config.top_p,
        seed=42
    )
    
    # Initialize components (simplified for now)
    from .tokenizer import SanskritTokenizer
    from .panini_engine import PaniniRuleEngine
    from .morphological_analyzer import SanskritMorphologicalAnalyzer
    from .derivation_simulator import ShabdaPrakriyaSimulator
    
    # Create basic instances
    tokenizer = SanskritTokenizer()
    rule_engine = PaniniRuleEngine(tokenizer)
    morphological_analyzer = SanskritMorphologicalAnalyzer(tokenizer, rule_engine)
    derivation_simulator = ShabdaPrakriyaSimulator(rule_engine, morphological_analyzer)
    
    problem_generator = SanskritProblemGenerator(
        tokenizer=tokenizer,
        rule_engine=rule_engine,
        morphological_analyzer=morphological_analyzer,
        derivation_simulator=derivation_simulator
    )
    
    return SanskritChallenger(
        config=challenger_config,
        problem_generator=problem_generator,
        storage_path=r_zero_config.storage_path
    )