"""
Sanskrit-specific reward functions for R-Zero training.

This module adapts R-Zero's reward_function/math.py for Sanskrit grammatical scoring,
providing comprehensive evaluation of Sanskrit reasoning tasks including:
- Sandhi correctness and morphological accuracy
- Semantic consistency rewards for cross-language translations  
- Rule application efficiency scoring
- Multi-objective rewards (accuracy + efficiency + linguistic validity)

Adapted from R-Zero's math.py reward structure for Sanskrit domain.
"""

import re
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import Sanskrit components
from .tokenizer import SanskritTokenizer
from .panini_engine import PaniniRuleEngine
from .morphological_analyzer import SanskritMorphologicalAnalyzer
from .r_zero_integration import SanskritGrammaticalValidator, SanskritProblem, SanskritProblemType


logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Types of reward components for Sanskrit evaluation."""
    GRAMMATICAL_CORRECTNESS = "grammatical_correctness"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    RULE_APPLICATION_EFFICIENCY = "rule_application_efficiency"
    FORMAT_COMPLIANCE = "format_compliance"
    SANDHI_ACCURACY = "sandhi_accuracy"
    MORPHOLOGICAL_ACCURACY = "morphological_accuracy"
    LINGUISTIC_VALIDITY = "linguistic_validity"
    CROSS_LANGUAGE_CONSISTENCY = "cross_language_consistency"


@dataclass
class SanskritReward:
    """Sanskrit-specific reward structure compatible with R-Zero."""
    overall_score: float
    components: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to R-Zero compatible format matching math.py structure."""
        return {
            "overall": self.overall_score,
            "format": self.components.get(RewardComponent.FORMAT_COMPLIANCE.value, 0.0),
            "accuracy": self.components.get(RewardComponent.GRAMMATICAL_CORRECTNESS.value, 0.0),
            "sandhi": self.components.get(RewardComponent.SANDHI_ACCURACY.value, 0.0),
            "morphology": self.components.get(RewardComponent.MORPHOLOGICAL_ACCURACY.value, 0.0),
            "efficiency": self.components.get(RewardComponent.RULE_APPLICATION_EFFICIENCY.value, 0.0),
            "semantic": self.components.get(RewardComponent.SEMANTIC_CONSISTENCY.value, 0.0),
            "linguistic": self.components.get(RewardComponent.LINGUISTIC_VALIDITY.value, 0.0),
            "cross_lang": self.components.get(RewardComponent.CROSS_LANGUAGE_CONSISTENCY.value, 0.0)
        }


# Core reward functions adapted from R-Zero's math.py structure

def format_reward(predict: str, problem_type: Optional[SanskritProblemType] = None) -> float:
    """
    Evaluate format compliance for Sanskrit predictions.
    Adapted from R-Zero's format_reward function.
    """
    if not predict or not predict.strip():
        return 0.0
    
    predict = predict.strip()
    
    # Basic format patterns for different Sanskrit problem types
    if problem_type == SanskritProblemType.SANDHI_APPLICATION:
        # Expect format: input → output or just the result
        sandhi_pattern = re.compile(r".*[→=].*|^[a-zA-Zāīūṛṝḷḹēōṃḥ]+$", re.DOTALL)
        return 1.0 if sandhi_pattern.search(predict) else 0.5
    
    elif problem_type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
        # Expect morphological breakdown with + or parentheses
        morph_pattern = re.compile(r".*[\+\(\)].*|.*root.*|.*suffix.*", re.DOTALL | re.IGNORECASE)
        return 1.0 if morph_pattern.search(predict) else 0.5
    
    elif problem_type == SanskritProblemType.WORD_DERIVATION:
        # Expect step-by-step derivation with arrows or steps
        deriv_pattern = re.compile(r".*[→].*|.*step.*|.*rule.*|.*sūtra.*", re.DOTALL | re.IGNORECASE)
        return 1.0 if deriv_pattern.search(predict) else 0.5
    
    # General Sanskrit text validation
    # Check for Devanagari script
    devanagari_chars = re.compile(r'[अ-ह्ा-ृे-ौं-ःऽ]')
    has_devanagari = bool(devanagari_chars.search(predict))
    
    # Check for IAST with diacritics (distinctive Sanskrit)
    iast_diacritics = re.compile(r'[āīūṛṝḷḹēōṃḥ]')
    has_iast_diacritics = bool(iast_diacritics.search(predict))
    
    # Check for common Sanskrit words in IAST without diacritics
    sanskrit_words = ['gacchati', 'karoti', 'bhavati', 'dharma', 'karma', 'yoga', 'moksa', 'atman', 'brahman']
    has_sanskrit_words = any(word in predict.lower() for word in sanskrit_words)
    
    # Check for reasonable length and structure
    reasonable_length = 1 <= len(predict) <= 500
    
    # More discriminating scoring
    if (has_devanagari or has_iast_diacritics or has_sanskrit_words) and reasonable_length:
        return 1.0
    elif reasonable_length:
        # Non-Sanskrit text gets lower score
        return 0.3
    else:
        return 0.0


def sandhi_accuracy_reward(predict: str, ground_truth: str, input_text: str = "") -> float:
    """
    Evaluate sandhi transformation accuracy.
    Specialized reward for Sanskrit phonological rules.
    """
    if not predict or not ground_truth:
        return 0.0
    
    predict = predict.strip()
    ground_truth = ground_truth.strip()
    
    # Exact match gets full score
    if predict == ground_truth:
        return 1.0
    
    # String similarity as base score
    similarity = calculate_string_similarity(predict, ground_truth)
    score = similarity
    
    # Bonus for phonologically valid transformations
    if input_text and '+' in input_text:
        parts = input_text.split('+')
        if len(parts) == 2:
            part1, part2 = parts[0].strip(), parts[1].strip()
            
            # Vowel sandhi patterns - give partial credit for correct patterns
            if part1.endswith('a') and part2.startswith('i'):
                if 'e' in predict:
                    score = max(score, 0.3)  # a + i → e
            elif part1.endswith('a') and part2.startswith('u'):
                if 'o' in predict:
                    score = max(score, 0.3)  # a + u → o
            elif part1.endswith('a') and part2.startswith('a'):
                if 'ā' in predict:
                    score = max(score, 0.3)  # a + a → ā
    
    return min(1.0, score)


def morphological_accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Evaluate morphological analysis accuracy.
    Specialized reward for Sanskrit morphological decomposition.
    """
    if not predict or not ground_truth:
        return 0.0
    
    predict = predict.strip().lower()
    ground_truth = ground_truth.strip().lower()
    
    # Exact match
    if predict == ground_truth:
        return 1.0
    
    # Check for morphological components
    morph_markers = ['root', 'suffix', 'prefix', 'dhātu', 'pratyaya', '+', '(', ')']
    
    pred_components = set()
    gt_components = set()
    
    for marker in morph_markers:
        if marker in predict:
            pred_components.add(marker)
        if marker in ground_truth:
            gt_components.add(marker)
    
    # Component overlap score
    if gt_components:
        component_score = len(pred_components.intersection(gt_components)) / len(gt_components)
    else:
        component_score = 0.5
    
    # String similarity
    similarity = calculate_string_similarity(predict, ground_truth)
    
    # Combined score
    return (component_score * 0.6 + similarity * 0.4)


def rule_efficiency_reward(predict: str, problem: Optional[SanskritProblem] = None) -> float:
    """
    Evaluate rule application efficiency.
    Rewards concise, clear, and well-structured responses.
    """
    if not predict:
        return 0.0
    
    predict = predict.strip()
    
    # Length efficiency (penalize overly verbose responses)
    length_penalty = max(0.0, 1.0 - len(predict) / 200.0)
    
    # Clarity bonus for structured responses
    clarity_bonus = 0.0
    structure_markers = ['→', '=', ':', 'step', 'rule', 'because', 'therefore']
    for marker in structure_markers:
        if marker.lower() in predict.lower():
            clarity_bonus += 0.1
    
    clarity_bonus = min(0.3, clarity_bonus)
    
    # Complexity penalty for overly nested structures
    complexity_penalty = 0.0
    if predict.count('(') > 5 or predict.count('[') > 3:
        complexity_penalty = 0.2
    
    # Efficiency score
    efficiency = length_penalty + clarity_bonus - complexity_penalty
    return max(0.0, min(1.0, efficiency))


def linguistic_validity_reward(predict: str, validator: Optional[SanskritGrammaticalValidator] = None) -> float:
    """
    Evaluate linguistic validity of Sanskrit text.
    Uses grammatical validator if available, otherwise heuristics.
    """
    if not predict:
        return 0.0
    
    if validator:
        try:
            validation_result = validator.validate_sanskrit_text(predict)
            return validation_result.get('confidence', 0.0)
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
    
    # Fallback heuristic validation
    score = 0.0
    
    # Check for Sanskrit characters
    sanskrit_pattern = r'[अ-ह्ा-ृे-ौं-ःऽ]|[aāiīuūṛṝḷḹeaioauṃḥ]'
    has_sanskrit = bool(re.search(sanskrit_pattern, predict))
    
    # Check for invalid character sequences (no invalid combinations)
    invalid_sequences = ['qq', 'xx', 'zz', 'ḥa', 'ṃa']  # Common invalid patterns
    has_invalid = any(seq in predict.lower() for seq in invalid_sequences)
    
    # Check for reasonable word structure
    words = predict.split()
    reasonable_structure = words and all(len(word) >= 1 for word in words)
    
    # More discriminating scoring
    if has_sanskrit and not has_invalid and reasonable_structure:
        score = 0.9  # High but not perfect without validator
    elif has_sanskrit and reasonable_structure:
        score = 0.7  # Good Sanskrit with minor issues
    elif not has_sanskrit and not has_invalid and reasonable_structure:
        score = 0.4  # Non-Sanskrit but valid structure
    elif has_invalid:
        score = 0.2  # Invalid sequences
    else:
        score = 0.1  # Poor structure
    
    return min(1.0, score)


def cross_language_consistency_reward(predict: str, ground_truth: str, 
                                    source_lang: str = "sanskrit", 
                                    target_lang: str = "english") -> float:
    """
    Evaluate consistency in cross-language translations.
    Specialized for Sanskrit ↔ other language mappings.
    """
    if not predict or not ground_truth:
        return 0.0
    
    predict = predict.strip().lower()
    ground_truth = ground_truth.strip().lower()
    
    # Exact match gets full score
    if predict == ground_truth:
        return 1.0
    
    # Basic similarity
    similarity = calculate_string_similarity(predict, ground_truth)
    
    # Language-specific consistency checks
    if source_lang == "sanskrit" and target_lang == "english":
        # Check if Sanskrit concepts are properly translated
        sanskrit_concepts = {
            'dharma': ['duty', 'righteousness', 'law'],
            'karma': ['action', 'deed', 'work'],
            'mokṣa': ['liberation', 'release', 'freedom'],
            'yoga': ['union', 'discipline', 'practice']
        }
        
        concept_score = 0.0
        concept_count = 0
        
        for sanskrit_term, english_equivalents in sanskrit_concepts.items():
            if sanskrit_term in ground_truth:
                concept_count += 1
                if any(equiv in predict for equiv in english_equivalents):
                    concept_score += 1.0
        
        if concept_count > 0:
            concept_consistency = concept_score / concept_count
            similarity = (similarity + concept_consistency) / 2
    
    return similarity


def calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using Jaccard similarity."""
    if not str1 or not str2:
        return 0.0
    
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    
    if str1 == str2:
        return 1.0
    
    # Jaccard similarity on character level
    set1, set2 = set(str1), set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


# Multi-objective reward configuration
@dataclass
class SanskritRewardConfig:
    """Configuration for Sanskrit reward calculation."""
    format_weight: float = 0.1
    accuracy_weight: float = 0.3
    efficiency_weight: float = 0.2
    linguistic_weight: float = 0.2
    semantic_weight: float = 0.2
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = self.format_weight + self.accuracy_weight + self.efficiency_weight + self.linguistic_weight + self.semantic_weight
        if abs(total - 1.0) > 1e-6:
            self.format_weight /= total
            self.accuracy_weight /= total
            self.efficiency_weight /= total
            self.linguistic_weight /= total
            self.semantic_weight /= total


# Main R-Zero compatible function (adapted from math.py structure)
def compute_score(predicts: List[str], 
                 ground_truths: List[str],
                 problems: Optional[List[SanskritProblem]] = None,
                 validator: Optional[SanskritGrammaticalValidator] = None,
                 format_weight: float = 0.1,
                 accuracy_weight: float = 0.3,
                 efficiency_weight: float = 0.2,
                 linguistic_weight: float = 0.2,
                 semantic_weight: float = 0.2,
                 **kwargs) -> List[Dict[str, float]]:
    """
    Compute Sanskrit scores for R-Zero framework.
    
    Multi-objective reward function combining:
    - Format compliance (structure and presentation)
    - Grammatical accuracy (Sanskrit correctness)
    - Rule application efficiency (conciseness and clarity)
    - Linguistic validity (phonological and morphological correctness)
    - Semantic consistency (meaning preservation)
    
    Args:
        predicts: List of model predictions
        ground_truths: List of expected answers
        problems: Optional list of Sanskrit problems for context
        validator: Optional Sanskrit grammatical validator
        format_weight: Weight for format compliance (default: 0.1)
        accuracy_weight: Weight for grammatical accuracy (default: 0.3)
        efficiency_weight: Weight for rule efficiency (default: 0.2)
        linguistic_weight: Weight for linguistic validity (default: 0.2)
        semantic_weight: Weight for semantic consistency (default: 0.2)
        **kwargs: Additional arguments
        
    Returns:
        List of score dictionaries compatible with R-Zero
    """
    # Normalize weights to sum to 1.0
    total_weight = format_weight + accuracy_weight + efficiency_weight + linguistic_weight + semantic_weight
    if abs(total_weight - 1.0) > 1e-6:
        format_weight /= total_weight
        accuracy_weight /= total_weight
        efficiency_weight /= total_weight
        linguistic_weight /= total_weight
        semantic_weight /= total_weight
    
    scores = []
    
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        problem = problems[i] if problems and i < len(problems) else None
        
        try:
            # Clean prediction (similar to math.py preprocessing)
            predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
            
            # Calculate individual reward components
            format_score = format_reward(predict, problem.type if problem else None)
            
            # Accuracy varies by problem type
            if problem and problem.type == SanskritProblemType.SANDHI_APPLICATION:
                accuracy_score = sandhi_accuracy_reward(predict, ground_truth, 
                                                      problem.input_text if problem else "")
            elif problem and problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
                accuracy_score = morphological_accuracy_reward(predict, ground_truth)
            else:
                # General grammatical accuracy
                accuracy_score = linguistic_validity_reward(predict, validator)
            
            # Rule application efficiency
            efficiency_score = rule_efficiency_reward(predict, problem)
            
            # Linguistic validity
            linguistic_score = linguistic_validity_reward(predict, validator)
            
            # Semantic consistency
            semantic_score = cross_language_consistency_reward(predict, ground_truth)
            
            # Multi-objective overall score
            overall_score = (
                format_weight * format_score +
                accuracy_weight * accuracy_score +
                efficiency_weight * efficiency_score +
                linguistic_weight * linguistic_score +
                semantic_weight * semantic_score
            )
            
            # Problem-specific bonuses
            bonus_score = 0.0
            if problem:
                if problem.type == SanskritProblemType.SANDHI_APPLICATION:
                    sandhi_score = sandhi_accuracy_reward(predict, ground_truth, problem.input_text)
                    bonus_score = sandhi_score * 0.1
                elif problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
                    morph_score = morphological_accuracy_reward(predict, ground_truth)
                    bonus_score = morph_score * 0.1
            
            overall_score = min(1.0, overall_score + bonus_score)
            
            # Build score dictionary (matching R-Zero math.py format)
            score_dict = {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "efficiency": efficiency_score,
                "linguistic": linguistic_score,
                "semantic": semantic_score
            }
            
            # Add problem-specific scores
            if problem:
                if problem.type == SanskritProblemType.SANDHI_APPLICATION:
                    score_dict["sandhi"] = sandhi_accuracy_reward(predict, ground_truth, problem.input_text)
                elif problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
                    score_dict["morphology"] = morphological_accuracy_reward(predict, ground_truth)
                elif problem.type == SanskritProblemType.TRANSLATION_SYNTHESIS:
                    score_dict["cross_lang"] = cross_language_consistency_reward(predict, ground_truth)
            
            scores.append(score_dict)
            
        except Exception as e:
            logger.error(f"Error calculating reward for prediction {i}: {e}")
            # Fallback score (matching math.py error handling pattern)
            scores.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
                "efficiency": 0.0,
                "linguistic": 0.0,
                "semantic": 0.0,
                "error": str(e)
            })
    
    return scores


# Backward compatibility function
def compute_sanskrit_score(predicts: List[str], 
                          ground_truths: List[str],
                          problems: Optional[List[SanskritProblem]] = None,
                          validator: Optional[SanskritGrammaticalValidator] = None,
                          **kwargs) -> List[Dict[str, float]]:
    """
    Backward compatibility wrapper for compute_score.
    Maintains existing API while using new multi-objective implementation.
    """
    return compute_score(predicts, ground_truths, problems, validator, **kwargs)


# Utility functions for specific reward components (updated for new structure)
def evaluate_sandhi_correctness(prediction: str, input_text: str, ground_truth: str = "") -> float:
    """Evaluate sandhi transformation correctness."""
    return sandhi_accuracy_reward(prediction, ground_truth, input_text)


def evaluate_morphological_analysis(prediction: str, ground_truth: str) -> float:
    """Evaluate morphological analysis accuracy."""
    return morphological_accuracy_reward(prediction, ground_truth)


def evaluate_grammatical_validity(text: str, validator: Optional[SanskritGrammaticalValidator] = None) -> float:
    """Evaluate grammatical validity of Sanskrit text."""
    return linguistic_validity_reward(text, validator)


def evaluate_rule_application_efficiency(prediction: str, problem: Optional[SanskritProblem] = None) -> float:
    """Evaluate efficiency of rule application."""
    return rule_efficiency_reward(prediction, problem)


def evaluate_semantic_consistency(prediction: str, ground_truth: str, 
                                source_lang: str = "sanskrit", target_lang: str = "english") -> float:
    """Evaluate semantic consistency in translations."""
    return cross_language_consistency_reward(prediction, ground_truth, source_lang, target_lang)


# Advanced multi-objective evaluation
def evaluate_comprehensive_sanskrit_quality(prediction: str, 
                                          ground_truth: str,
                                          problem: Optional[SanskritProblem] = None,
                                          validator: Optional[SanskritGrammaticalValidator] = None,
                                          config: Optional[SanskritRewardConfig] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of Sanskrit prediction quality.
    
    Returns detailed breakdown of all reward components.
    """
    if config is None:
        config = SanskritRewardConfig()
    
    # Calculate all components
    format_score = format_reward(prediction, problem.type if problem else None)
    
    if problem and problem.type == SanskritProblemType.SANDHI_APPLICATION:
        accuracy_score = sandhi_accuracy_reward(prediction, ground_truth, problem.input_text)
    elif problem and problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
        accuracy_score = morphological_accuracy_reward(prediction, ground_truth)
    else:
        accuracy_score = linguistic_validity_reward(prediction, validator)
    
    efficiency_score = rule_efficiency_reward(prediction, problem)
    linguistic_score = linguistic_validity_reward(prediction, validator)
    semantic_score = cross_language_consistency_reward(prediction, ground_truth)
    
    # Multi-objective overall score
    overall_score = (
        config.format_weight * format_score +
        config.accuracy_weight * accuracy_score +
        config.efficiency_weight * efficiency_score +
        config.linguistic_weight * linguistic_score +
        config.semantic_weight * semantic_score
    )
    
    return {
        "overall": overall_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "efficiency": efficiency_score,
        "linguistic": linguistic_score,
        "semantic": semantic_score,
        "weights": {
            "format": config.format_weight,
            "accuracy": config.accuracy_weight,
            "efficiency": config.efficiency_weight,
            "linguistic": config.linguistic_weight,
            "semantic": config.semantic_weight
        }
    }