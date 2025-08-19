#!/usr/bin/env python3
"""
Demonstration of Sanskrit-aware reward functions for R-Zero training.

This script showcases the effectiveness and stability of the Sanskrit reward
functions adapted from R-Zero's math.py, demonstrating multi-objective scoring
for various Sanskrit reasoning tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sanskrit_rewrite_engine.sanskrit_reward_function import (
    compute_score,
    evaluate_comprehensive_sanskrit_quality,
    SanskritRewardConfig
)
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem,
    SanskritProblemType,
    SanskritDifficultyLevel
)


def demonstrate_sandhi_rewards():
    """Demonstrate sandhi accuracy rewards."""
    print("=== Sandhi Application Rewards ===")
    
    sandhi_problem = SanskritProblem(
        id="demo_sandhi",
        type=SanskritProblemType.SANDHI_APPLICATION,
        difficulty=SanskritDifficultyLevel.BEGINNER,
        input_text="a + i",
        expected_output="e",
        explanation="Basic vowel sandhi: a + i → e"
    )
    
    test_cases = [
        ("e", "Perfect sandhi application"),
        ("ai", "Incorrect but phonologically reasonable"),
        ("x", "Completely wrong"),
        ("a + i → e", "Verbose but correct explanation"),
        ("", "Empty response")
    ]
    
    for prediction, description in test_cases:
        scores = compute_score([prediction], ["e"], [sandhi_problem])
        score = scores[0]
        print(f"{description:35} | Overall: {score['overall']:.3f} | Sandhi: {score.get('sandhi', 0):.3f} | Format: {score['format']:.3f}")
    
    print()


def demonstrate_morphology_rewards():
    """Demonstrate morphological analysis rewards."""
    print("=== Morphological Analysis Rewards ===")
    
    morph_problem = SanskritProblem(
        id="demo_morph",
        type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
        difficulty=SanskritDifficultyLevel.INTERMEDIATE,
        input_text="gacchati",
        expected_output="gam(root) + a(thematic) + ti(3sg.pres)",
        explanation="Morphological analysis of 'gacchati'"
    )
    
    test_cases = [
        ("gam(root) + a(thematic) + ti(3sg.pres)", "Perfect analysis"),
        ("gam(root) + ti(suffix)", "Partial analysis"),
        ("gacchati is a verb", "Descriptive but incomplete"),
        ("wrong analysis", "Incorrect analysis"),
        ("gam + a + ti", "Simplified correct structure")
    ]
    
    ground_truth = "gam(root) + a(thematic) + ti(3sg.pres)"
    
    for prediction, description in test_cases:
        scores = compute_score([prediction], [ground_truth], [morph_problem])
        score = scores[0]
        print(f"{description:35} | Overall: {score['overall']:.3f} | Morphology: {score.get('morphology', 0):.3f} | Format: {score['format']:.3f}")
    
    print()


def demonstrate_multi_objective_scoring():
    """Demonstrate multi-objective reward combination."""
    print("=== Multi-Objective Scoring ===")
    
    # Test with different weight configurations
    configs = [
        ("Accuracy-focused", SanskritRewardConfig(accuracy_weight=0.6, efficiency_weight=0.1, format_weight=0.1, linguistic_weight=0.1, semantic_weight=0.1)),
        ("Efficiency-focused", SanskritRewardConfig(accuracy_weight=0.2, efficiency_weight=0.5, format_weight=0.1, linguistic_weight=0.1, semantic_weight=0.1)),
        ("Balanced", SanskritRewardConfig(accuracy_weight=0.3, efficiency_weight=0.2, format_weight=0.1, linguistic_weight=0.2, semantic_weight=0.2)),
        ("Linguistic-focused", SanskritRewardConfig(accuracy_weight=0.2, efficiency_weight=0.1, format_weight=0.1, linguistic_weight=0.5, semantic_weight=0.1))
    ]
    
    prediction = "gacchati"
    ground_truth = "gacchati"
    
    for config_name, config in configs:
        result = evaluate_comprehensive_sanskrit_quality(
            prediction=prediction,
            ground_truth=ground_truth,
            config=config
        )
        print(f"{config_name:20} | Overall: {result['overall']:.3f} | Accuracy: {result['accuracy']:.3f} | Efficiency: {result['efficiency']:.3f} | Linguistic: {result['linguistic']:.3f}")
    
    print()


def demonstrate_cross_language_consistency():
    """Demonstrate cross-language consistency rewards."""
    print("=== Cross-Language Consistency ===")
    
    translation_problem = SanskritProblem(
        id="demo_translation",
        type=SanskritProblemType.TRANSLATION_SYNTHESIS,
        difficulty=SanskritDifficultyLevel.ADVANCED,
        input_text="dharma",
        expected_output="duty, righteousness, law",
        explanation="Translation of Sanskrit concept 'dharma'"
    )
    
    test_cases = [
        ("duty, righteousness, law", "Perfect translation"),
        ("duty", "Partial but correct translation"),
        ("righteousness", "Single correct concept"),
        ("moral law", "Related concept"),
        ("completely unrelated", "Wrong translation")
    ]
    
    ground_truth = "duty, righteousness, law"
    
    for prediction, description in test_cases:
        scores = compute_score([prediction], [ground_truth], [translation_problem])
        score = scores[0]
        print(f"{description:35} | Overall: {score['overall']:.3f} | Cross-lang: {score.get('cross_lang', 0):.3f} | Semantic: {score['semantic']:.3f}")
    
    print()


def demonstrate_reward_stability():
    """Demonstrate reward function stability."""
    print("=== Reward Function Stability ===")
    
    prediction = "gacchati"
    ground_truth = "gacchati"
    
    # Run multiple times to check consistency
    scores = []
    for i in range(10):
        result = compute_score([prediction], [ground_truth])
        scores.append(result[0]['overall'])
    
    print(f"Prediction: '{prediction}' vs Ground Truth: '{ground_truth}'")
    print(f"Scores across 10 runs: {scores}")
    print(f"All scores identical: {all(s == scores[0] for s in scores)}")
    print(f"Mean: {sum(scores)/len(scores):.6f}, Std Dev: {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.6f}")
    
    print()


def demonstrate_reward_monotonicity():
    """Demonstrate that better predictions get higher rewards."""
    print("=== Reward Monotonicity ===")
    
    ground_truth = "gam(root) + ti(suffix)"
    
    predictions = [
        ("gam(root) + ti(suffix)", "Perfect match"),
        ("gam + ti", "Good structure"),
        ("gam root ti suffix", "Reasonable attempt"),
        ("verb form", "Vague description"),
        ("wrong", "Incorrect")
    ]
    
    results = []
    for prediction, description in predictions:
        scores = compute_score([prediction], [ground_truth])
        score = scores[0]['overall']
        results.append((score, description, prediction))
    
    # Sort by score (descending)
    results.sort(reverse=True)
    
    print("Predictions ranked by reward score:")
    for i, (score, description, prediction) in enumerate(results, 1):
        print(f"{i}. {description:20} | Score: {score:.3f} | Prediction: '{prediction}'")
    
    print()


def demonstrate_r_zero_compatibility():
    """Demonstrate R-Zero framework compatibility."""
    print("=== R-Zero Framework Compatibility ===")
    
    # Simulate R-Zero batch processing
    predicts = [
        "e",  # Sandhi result
        "gam(root) + ti(suffix)",  # Morphological analysis
        "gam → gacch → gacchati",  # Derivation
        "dharma"  # Sanskrit term
    ]
    
    ground_truths = [
        "e",
        "gam(root) + a(thematic) + ti(3sg.pres)",
        "gam → gacch → gacchati",
        "dharma"
    ]
    
    # Process batch
    scores = compute_score(predicts, ground_truths)
    
    print("Batch processing results (R-Zero compatible format):")
    for i, score in enumerate(scores):
        print(f"Sample {i+1}:")
        print(f"  Prediction: '{predicts[i]}'")
        print(f"  Ground Truth: '{ground_truths[i]}'")
        print(f"  Overall Score: {score['overall']:.3f}")
        print(f"  Component Scores: Format={score['format']:.3f}, Accuracy={score['accuracy']:.3f}, Efficiency={score['efficiency']:.3f}")
        print()


def main():
    """Run all demonstrations."""
    print("Sanskrit Reward Function Demonstration")
    print("=" * 50)
    print()
    
    demonstrate_sandhi_rewards()
    demonstrate_morphology_rewards()
    demonstrate_multi_objective_scoring()
    demonstrate_cross_language_consistency()
    demonstrate_reward_stability()
    demonstrate_reward_monotonicity()
    demonstrate_r_zero_compatibility()
    
    print("Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ Sandhi correctness evaluation")
    print("✓ Morphological accuracy assessment")
    print("✓ Multi-objective reward combination")
    print("✓ Cross-language consistency checking")
    print("✓ Reward function stability")
    print("✓ Monotonic reward behavior")
    print("✓ R-Zero framework compatibility")


if __name__ == "__main__":
    main()