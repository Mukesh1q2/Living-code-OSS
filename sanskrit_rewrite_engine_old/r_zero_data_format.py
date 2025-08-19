"""
R-Zero Data Format Converter for Sanskrit Reasoning.

This module provides utilities to convert Sanskrit reasoning problems
and solutions into the data format expected by the R-Zero framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
import json
import random
from datetime import datetime
import logging

from .r_zero_integration import (
    SanskritProblem, SanskritSolution, SanskritProblemType, 
    SanskritDifficultyLevel, SanskritProblemGenerator
)


@dataclass
class RZeroDataset:
    """
    Dataset in R-Zero compatible format.
    
    Attributes:
        problems: List of problems in R-Zero format
        metadata: Dataset metadata
        statistics: Dataset statistics
    """
    problems: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def save_to_json(self, filepath: str) -> None:
        """Save dataset to JSON file."""
        data = {
            'problems': self.problems,
            'metadata': self.metadata,
            'statistics': self.statistics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'RZeroDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            problems=data.get('problems', []),
            metadata=data.get('metadata', {}),
            statistics=data.get('statistics', {})
        )


class SanskritRZeroDataConverter:
    """
    Converter for Sanskrit data to R-Zero format.
    
    Handles conversion of Sanskrit reasoning problems and solutions
    into the format expected by R-Zero's training pipeline.
    """
    
    def __init__(self, problem_generator: SanskritProblemGenerator):
        """
        Initialize the data converter.
        
        Args:
            problem_generator: Sanskrit problem generator
        """
        self.problem_generator = problem_generator
        self.logger = logging.getLogger(__name__)
    
    def create_training_dataset(self, 
                               num_problems: int = 1000,
                               problem_distribution: Optional[Dict[SanskritProblemType, float]] = None,
                               difficulty_distribution: Optional[Dict[SanskritDifficultyLevel, float]] = None) -> RZeroDataset:
        """
        Create a training dataset in R-Zero format.
        
        Args:
            num_problems: Number of problems to generate
            problem_distribution: Distribution of problem types
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            R-Zero compatible dataset
        """
        # Set default distributions
        if problem_distribution is None:
            problem_distribution = {
                SanskritProblemType.SANDHI_APPLICATION: 0.25,
                SanskritProblemType.MORPHOLOGICAL_ANALYSIS: 0.20,
                SanskritProblemType.WORD_DERIVATION: 0.20,
                SanskritProblemType.COMPOUND_ANALYSIS: 0.15,
                SanskritProblemType.RULE_APPLICATION: 0.10,
                SanskritProblemType.GRAMMATICAL_VALIDATION: 0.05,
                SanskritProblemType.TRANSLATION_SYNTHESIS: 0.05
            }
        
        if difficulty_distribution is None:
            difficulty_distribution = {
                SanskritDifficultyLevel.BEGINNER: 0.3,
                SanskritDifficultyLevel.INTERMEDIATE: 0.4,
                SanskritDifficultyLevel.ADVANCED: 0.2,
                SanskritDifficultyLevel.EXPERT: 0.1
            }
        
        # Generate problems with specified distributions
        problems = []
        for _ in range(num_problems):
            # Select problem type based on distribution
            problem_type = self._select_from_distribution(problem_distribution)
            difficulty = self._select_from_distribution(difficulty_distribution)
            
            # Generate problem
            problem = self.problem_generator._generate_problem_by_type(
                problem_id=f"sanskrit_train_{len(problems):06d}",
                problem_type=problem_type,
                difficulty=difficulty
            )
            
            if problem:
                problems.append(problem)
        
        # Convert to R-Zero format
        r_zero_problems = [self.convert_problem_to_r_zero(p) for p in problems]
        
        # Calculate statistics
        statistics = self._calculate_dataset_statistics(problems)
        
        # Create metadata
        metadata = {
            'dataset_type': 'sanskrit_reasoning_training',
            'creation_date': datetime.now().isoformat(),
            'num_problems': len(r_zero_problems),
            'problem_distribution': {k.value: v for k, v in problem_distribution.items()},
            'difficulty_distribution': {k.value: v for k, v in difficulty_distribution.items()},
            'generator_version': '1.0'
        }
        
        dataset = RZeroDataset(
            problems=r_zero_problems,
            metadata=metadata,
            statistics=statistics
        )
        
        self.logger.info(f"Created R-Zero dataset with {len(r_zero_problems)} problems")
        return dataset
    
    def convert_problem_to_r_zero(self, problem: SanskritProblem) -> Dict[str, Any]:
        """
        Convert a Sanskrit problem to R-Zero format.
        
        Args:
            problem: Sanskrit problem to convert
            
        Returns:
            Problem in R-Zero format
        """
        r_zero_format = problem.to_r_zero_format()
        
        # Add R-Zero specific fields
        r_zero_format.update({
            'id': problem.id,
            'difficulty': problem.difficulty.value,
            'type': problem.type.value,
            'sutra_references': problem.sutra_references,
            'explanation': problem.explanation
        })
        
        return r_zero_format
    
    def create_evaluation_dataset(self, 
                                 num_problems: int = 200,
                                 focus_on_difficult: bool = True) -> RZeroDataset:
        """
        Create an evaluation dataset with challenging problems.
        
        Args:
            num_problems: Number of evaluation problems
            focus_on_difficult: Whether to focus on difficult problems
            
        Returns:
            R-Zero compatible evaluation dataset
        """
        if focus_on_difficult:
            difficulty_distribution = {
                SanskritDifficultyLevel.BEGINNER: 0.1,
                SanskritDifficultyLevel.INTERMEDIATE: 0.2,
                SanskritDifficultyLevel.ADVANCED: 0.4,
                SanskritDifficultyLevel.EXPERT: 0.3
            }
        else:
            difficulty_distribution = {
                SanskritDifficultyLevel.BEGINNER: 0.25,
                SanskritDifficultyLevel.INTERMEDIATE: 0.35,
                SanskritDifficultyLevel.ADVANCED: 0.25,
                SanskritDifficultyLevel.EXPERT: 0.15
            }
        
        # Generate evaluation problems
        problems = self.problem_generator.generate_problems(
            count=num_problems,
            difficulty_distribution=difficulty_distribution
        )
        
        # Convert to R-Zero format
        r_zero_problems = [self.convert_problem_to_r_zero(p) for p in problems]
        
        # Calculate statistics
        statistics = self._calculate_dataset_statistics(problems)
        
        # Create metadata
        metadata = {
            'dataset_type': 'sanskrit_reasoning_evaluation',
            'creation_date': datetime.now().isoformat(),
            'num_problems': len(r_zero_problems),
            'focus_on_difficult': focus_on_difficult,
            'difficulty_distribution': {k.value: v for k, v in difficulty_distribution.items()},
            'generator_version': '1.0'
        }
        
        dataset = RZeroDataset(
            problems=r_zero_problems,
            metadata=metadata,
            statistics=statistics
        )
        
        self.logger.info(f"Created R-Zero evaluation dataset with {len(r_zero_problems)} problems")
        return dataset
    
    def create_challenger_dataset(self, 
                                 base_problems: List[SanskritProblem],
                                 variations_per_problem: int = 3) -> RZeroDataset:
        """
        Create a dataset for the Challenger model in R-Zero.
        
        Args:
            base_problems: Base problems to create variations from
            variations_per_problem: Number of variations per base problem
            
        Returns:
            R-Zero compatible challenger dataset
        """
        challenger_problems = []
        
        for base_problem in base_problems:
            # Add the original problem
            challenger_problems.append(base_problem)
            
            # Create variations
            for i in range(variations_per_problem):
                variation = self._create_problem_variation(base_problem, i)
                if variation:
                    challenger_problems.append(variation)
        
        # Convert to R-Zero format
        r_zero_problems = [self.convert_problem_to_r_zero(p) for p in challenger_problems]
        
        # Calculate statistics
        statistics = self._calculate_dataset_statistics(challenger_problems)
        
        # Create metadata
        metadata = {
            'dataset_type': 'sanskrit_reasoning_challenger',
            'creation_date': datetime.now().isoformat(),
            'num_problems': len(r_zero_problems),
            'base_problems': len(base_problems),
            'variations_per_problem': variations_per_problem,
            'generator_version': '1.0'
        }
        
        dataset = RZeroDataset(
            problems=r_zero_problems,
            metadata=metadata,
            statistics=statistics
        )
        
        self.logger.info(f"Created R-Zero challenger dataset with {len(r_zero_problems)} problems")
        return dataset
    
    def _select_from_distribution(self, distribution: Dict[Any, float]) -> Any:
        """Select an item based on probability distribution."""
        rand_val = random.random()
        cumulative = 0.0
        
        for item, prob in distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                return item
        
        # Fallback to first item
        return next(iter(distribution.keys()))
    
    def _calculate_dataset_statistics(self, problems: List[SanskritProblem]) -> Dict[str, Any]:
        """Calculate statistics for a dataset."""
        if not problems:
            return {}
        
        # Count by type
        type_counts = {}
        for problem in problems:
            type_name = problem.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by difficulty
        difficulty_counts = {}
        for problem in problems:
            difficulty_name = problem.difficulty.value
            difficulty_counts[difficulty_name] = difficulty_counts.get(difficulty_name, 0) + 1
        
        # Average input/output lengths
        input_lengths = [len(p.input_text) for p in problems]
        output_lengths = [len(p.expected_output) for p in problems]
        
        # Sutra reference statistics
        sutra_refs = []
        for problem in problems:
            sutra_refs.extend(problem.sutra_references)
        
        unique_sutras = len(set(sutra_refs))
        
        return {
            'total_problems': len(problems),
            'type_distribution': type_counts,
            'difficulty_distribution': difficulty_counts,
            'average_input_length': sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            'average_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'unique_sutra_references': unique_sutras,
            'total_sutra_references': len(sutra_refs)
        }
    
    def _create_problem_variation(self, base_problem: SanskritProblem, variation_index: int) -> Optional[SanskritProblem]:
        """Create a variation of a base problem."""
        try:
            # Create variation based on problem type
            if base_problem.type == SanskritProblemType.SANDHI_APPLICATION:
                return self._create_sandhi_variation(base_problem, variation_index)
            elif base_problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
                return self._create_morphology_variation(base_problem, variation_index)
            elif base_problem.type == SanskritProblemType.WORD_DERIVATION:
                return self._create_derivation_variation(base_problem, variation_index)
            else:
                # For other types, create a simple variation by modifying difficulty
                new_difficulty = self._get_next_difficulty(base_problem.difficulty)
                return SanskritProblem(
                    id=f"{base_problem.id}_var_{variation_index}",
                    type=base_problem.type,
                    difficulty=new_difficulty,
                    input_text=base_problem.input_text,
                    expected_output=base_problem.expected_output,
                    context=base_problem.context.copy(),
                    sutra_references=base_problem.sutra_references.copy(),
                    explanation=f"Variation of {base_problem.explanation}",
                    metadata={**base_problem.metadata, 'variation_of': base_problem.id}
                )
        except Exception as e:
            self.logger.error(f"Error creating variation for problem {base_problem.id}: {str(e)}")
            return None
    
    def _create_sandhi_variation(self, base_problem: SanskritProblem, variation_index: int) -> SanskritProblem:
        """Create a variation of a sandhi problem."""
        # Simple variation: change the input vowels
        vowel_pairs = [("a", "i"), ("a", "u"), ("i", "a"), ("u", "a"), ("e", "a")]
        vowel1, vowel2 = vowel_pairs[variation_index % len(vowel_pairs)]
        
        new_input = f"{vowel1} + {vowel2}"
        
        # Determine expected output based on sandhi rules
        if vowel1 == "a" and vowel2 == "i":
            new_output = "e"
        elif vowel1 == "a" and vowel2 == "u":
            new_output = "o"
        elif vowel1 == "i" and vowel2 == "a":
            new_output = "ya"
        elif vowel1 == "u" and vowel2 == "a":
            new_output = "va"
        else:
            new_output = vowel1 + vowel2  # No sandhi
        
        return SanskritProblem(
            id=f"{base_problem.id}_var_{variation_index}",
            type=base_problem.type,
            difficulty=base_problem.difficulty,
            input_text=new_input,
            expected_output=new_output,
            context=base_problem.context.copy(),
            sutra_references=base_problem.sutra_references.copy(),
            explanation=f"Sandhi variation: {vowel1} + {vowel2} → {new_output}",
            metadata={**base_problem.metadata, 'variation_of': base_problem.id}
        )
    
    def _create_morphology_variation(self, base_problem: SanskritProblem, variation_index: int) -> SanskritProblem:
        """Create a variation of a morphology problem."""
        # Simple variation: use different words with similar structure
        morphology_words = ["gacchati", "karoti", "bhavati", "paśyati", "śṛṇoti"]
        new_word = morphology_words[variation_index % len(morphology_words)]
        
        # Create expected output (simplified)
        if new_word.endswith("ati"):
            root = new_word[:-3]
            new_output = f"{root}(ROOT) + ati(SUFFIX) | Categories: PRESENT, THIRD_PERSON, SINGULAR"
        else:
            new_output = f"{new_word}(ROOT) | Categories: VERBAL_FORM"
        
        return SanskritProblem(
            id=f"{base_problem.id}_var_{variation_index}",
            type=base_problem.type,
            difficulty=base_problem.difficulty,
            input_text=new_word,
            expected_output=new_output,
            context=base_problem.context.copy(),
            sutra_references=base_problem.sutra_references.copy(),
            explanation=f"Morphological analysis variation of {new_word}",
            metadata={**base_problem.metadata, 'variation_of': base_problem.id}
        )
    
    def _create_derivation_variation(self, base_problem: SanskritProblem, variation_index: int) -> SanskritProblem:
        """Create a variation of a derivation problem."""
        # Simple variation: use different root-suffix combinations
        derivations = [
            ("gam", "ti", "gacchati"),
            ("kar", "ti", "karoti"),
            ("bhū", "ti", "bhavati"),
            ("dṛś", "ti", "paśyati")
        ]
        
        root, suffix, result = derivations[variation_index % len(derivations)]
        new_input = f"{root} + {suffix}"
        new_output = f"{root} → {result}"
        
        return SanskritProblem(
            id=f"{base_problem.id}_var_{variation_index}",
            type=base_problem.type,
            difficulty=base_problem.difficulty,
            input_text=new_input,
            expected_output=new_output,
            context=base_problem.context.copy(),
            sutra_references=base_problem.sutra_references.copy(),
            explanation=f"Derivation variation: {root} + {suffix} → {result}",
            metadata={**base_problem.metadata, 'variation_of': base_problem.id}
        )
    
    def _get_next_difficulty(self, current_difficulty: SanskritDifficultyLevel) -> SanskritDifficultyLevel:
        """Get the next difficulty level."""
        difficulties = [
            SanskritDifficultyLevel.BEGINNER,
            SanskritDifficultyLevel.INTERMEDIATE,
            SanskritDifficultyLevel.ADVANCED,
            SanskritDifficultyLevel.EXPERT
        ]
        
        try:
            current_index = difficulties.index(current_difficulty)
            next_index = (current_index + 1) % len(difficulties)
            return difficulties[next_index]
        except ValueError:
            return SanskritDifficultyLevel.INTERMEDIATE


# Utility functions for R-Zero integration
def create_r_zero_training_data(problem_generator: SanskritProblemGenerator,
                               num_problems: int = 1000) -> RZeroDataset:
    """Create R-Zero training dataset."""
    converter = SanskritRZeroDataConverter(problem_generator)
    return converter.create_training_dataset(num_problems)


def create_r_zero_evaluation_data(problem_generator: SanskritProblemGenerator,
                                 num_problems: int = 200) -> RZeroDataset:
    """Create R-Zero evaluation dataset."""
    converter = SanskritRZeroDataConverter(problem_generator)
    return converter.create_evaluation_dataset(num_problems)


def save_dataset_for_r_zero(dataset: RZeroDataset, 
                           base_filename: str,
                           split_ratio: float = 0.8) -> Tuple[str, str]:
    """
    Save dataset in R-Zero format with train/validation split.
    
    Args:
        dataset: Dataset to save
        base_filename: Base filename (without extension)
        split_ratio: Ratio for train/validation split
        
    Returns:
        Tuple of (train_filename, val_filename)
    """
    # Split dataset
    total_problems = len(dataset.problems)
    train_size = int(total_problems * split_ratio)
    
    train_problems = dataset.problems[:train_size]
    val_problems = dataset.problems[train_size:]
    
    # Create train dataset
    train_dataset = RZeroDataset(
        problems=train_problems,
        metadata={**dataset.metadata, 'split': 'train', 'split_ratio': split_ratio},
        statistics=dataset.statistics
    )
    
    # Create validation dataset
    val_dataset = RZeroDataset(
        problems=val_problems,
        metadata={**dataset.metadata, 'split': 'validation', 'split_ratio': 1.0 - split_ratio},
        statistics=dataset.statistics
    )
    
    # Save files
    train_filename = f"{base_filename}_train.json"
    val_filename = f"{base_filename}_val.json"
    
    train_dataset.save_to_json(train_filename)
    val_dataset.save_to_json(val_filename)
    
    return train_filename, val_filename