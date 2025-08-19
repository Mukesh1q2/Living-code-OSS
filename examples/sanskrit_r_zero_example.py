#!/usr/bin/env python3
"""
Sanskrit R-Zero Integration Example.

This script demonstrates how to use the Sanskrit Rewrite Engine
with the R-Zero self-evolving reasoning framework.
"""

import os
import json
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Sanskrit engine components
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine
from sanskrit_rewrite_engine.morphological_analyzer import SanskritMorphologicalAnalyzer
from sanskrit_rewrite_engine.derivation_simulator import ShabdaPrakriyaSimulator
from sanskrit_rewrite_engine.rule import RuleRegistry

# Import R-Zero integration components
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblemGenerator, SanskritGrammaticalValidator, 
    SanskritReasoningMetrics, compute_sanskrit_score,
    SanskritProblemType, SanskritDifficultyLevel
)
from sanskrit_rewrite_engine.r_zero_data_format import (
    SanskritRZeroDataConverter, create_r_zero_training_data,
    save_dataset_for_r_zero
)
from sanskrit_rewrite_engine.rl_environment import (
    SanskritRLEnvironment, SanskritAction, ActionType,
    SanskritRLTrainer, create_sanskrit_rl_environment
)


def setup_sanskrit_engine():
    """Set up the Sanskrit engine components."""
    logger.info("Setting up Sanskrit engine components...")
    
    # Initialize tokenizer
    tokenizer = SanskritTokenizer()
    
    # Initialize rule engine
    rule_engine = PaniniRuleEngine(tokenizer)
    
    # Initialize morphological analyzer
    morphological_analyzer = SanskritMorphologicalAnalyzer()
    
    # Initialize derivation simulator
    rule_registry = RuleRegistry()
    derivation_simulator = ShabdaPrakriyaSimulator(
        rule_engine, morphological_analyzer, rule_registry
    )
    
    logger.info("Sanskrit engine components initialized successfully")
    return tokenizer, rule_engine, morphological_analyzer, derivation_simulator


def demonstrate_problem_generation():
    """Demonstrate Sanskrit problem generation for R-Zero."""
    logger.info("=== Demonstrating Sanskrit Problem Generation ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    
    # Create problem generator
    problem_generator = SanskritProblemGenerator(
        tokenizer, rule_engine, morphological_analyzer, derivation_simulator
    )
    
    # Generate problems of different types
    logger.info("Generating Sanskrit problems...")
    problems = problem_generator.generate_problems(
        count=20,
        problem_types=[
            SanskritProblemType.SANDHI_APPLICATION,
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            SanskritProblemType.WORD_DERIVATION
        ],
        difficulty_distribution={
            SanskritDifficultyLevel.BEGINNER: 0.4,
            SanskritDifficultyLevel.INTERMEDIATE: 0.4,
            SanskritDifficultyLevel.ADVANCED: 0.2
        }
    )
    
    # Display sample problems
    logger.info(f"Generated {len(problems)} problems")
    for i, problem in enumerate(problems[:5]):  # Show first 5
        logger.info(f"Problem {i+1}:")
        logger.info(f"  Type: {problem.type.value}")
        logger.info(f"  Difficulty: {problem.difficulty.value}")
        logger.info(f"  Input: {problem.input_text}")
        logger.info(f"  Expected Output: {problem.expected_output}")
        logger.info(f"  Explanation: {problem.explanation}")
        logger.info("")
    
    return problems


def demonstrate_data_format_conversion():
    """Demonstrate conversion to R-Zero data format."""
    logger.info("=== Demonstrating R-Zero Data Format Conversion ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    problem_generator = SanskritProblemGenerator(
        tokenizer, rule_engine, morphological_analyzer, derivation_simulator
    )
    
    # Create data converter
    data_converter = SanskritRZeroDataConverter(problem_generator)
    
    # Create training dataset
    logger.info("Creating R-Zero training dataset...")
    training_dataset = data_converter.create_training_dataset(num_problems=100)
    
    logger.info(f"Training dataset created with {len(training_dataset.problems)} problems")
    logger.info(f"Dataset metadata: {training_dataset.metadata}")
    logger.info(f"Dataset statistics: {training_dataset.statistics}")
    
    # Create evaluation dataset
    logger.info("Creating R-Zero evaluation dataset...")
    eval_dataset = data_converter.create_evaluation_dataset(num_problems=20)
    
    logger.info(f"Evaluation dataset created with {len(eval_dataset.problems)} problems")
    
    # Save datasets
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data with train/val split
    train_file, val_file = save_dataset_for_r_zero(
        training_dataset, 
        os.path.join(output_dir, "sanskrit_training"),
        split_ratio=0.8
    )
    logger.info(f"Training data saved to: {train_file}")
    logger.info(f"Validation data saved to: {val_file}")
    
    # Save evaluation data
    eval_file = os.path.join(output_dir, "sanskrit_evaluation.json")
    eval_dataset.save_to_json(eval_file)
    logger.info(f"Evaluation data saved to: {eval_file}")
    
    # Show sample R-Zero format
    sample_problem = training_dataset.problems[0]
    logger.info("Sample problem in R-Zero format:")
    logger.info(json.dumps(sample_problem, indent=2, ensure_ascii=False))
    
    return training_dataset, eval_dataset


def demonstrate_grammatical_validation():
    """Demonstrate Sanskrit grammatical validation."""
    logger.info("=== Demonstrating Sanskrit Grammatical Validation ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    
    # Create validator
    validator = SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    # Test validation on various Sanskrit texts
    test_texts = [
        "gacchati",      # Valid: he/she goes
        "karoti",        # Valid: he/she does
        "a + i",         # Valid: sandhi input
        "e",             # Valid: sandhi result
        "invalid_xyz",   # Invalid: not Sanskrit
        "gacchanti",     # Context-dependent: they go (plural)
    ]
    
    logger.info("Validating Sanskrit texts...")
    for text in test_texts:
        result = validator.validate_sanskrit_text(text)
        logger.info(f"Text: '{text}'")
        logger.info(f"  Valid: {result['is_valid']}")
        logger.info(f"  Confidence: {result['confidence']:.3f}")
        logger.info(f"  Metrics: {result['metrics']}")
        if result['errors']:
            logger.info(f"  Errors: {result['errors']}")
        logger.info("")


def demonstrate_reasoning_metrics():
    """Demonstrate Sanskrit reasoning evaluation metrics."""
    logger.info("=== Demonstrating Sanskrit Reasoning Metrics ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    validator = SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    # Create metrics calculator
    metrics_calculator = SanskritReasoningMetrics(validator)
    
    # Create test problems and solutions
    from sanskrit_rewrite_engine.r_zero_integration import SanskritProblem
    
    test_cases = [
        {
            'problem': SanskritProblem(
                id="test_sandhi_001",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text="a + i",
                expected_output="e",
                explanation="Vowel sandhi: a + i → e"
            ),
            'solutions': ["e", "ai", "wrong_answer"]
        },
        {
            'problem': SanskritProblem(
                id="test_morph_001",
                type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="gacchati",
                expected_output="gam(ROOT) + ti(SUFFIX)",
                explanation="Morphological analysis of 'gacchati'"
            ),
            'solutions': ["gam(ROOT) + ti(SUFFIX)", "gacch + ati", "unknown"]
        }
    ]
    
    logger.info("Evaluating solutions...")
    for i, test_case in enumerate(test_cases):
        problem = test_case['problem']
        solutions = test_case['solutions']
        
        logger.info(f"Test Case {i+1}: {problem.type.value}")
        logger.info(f"Problem: {problem.input_text}")
        logger.info(f"Expected: {problem.expected_output}")
        logger.info("")
        
        for j, solution in enumerate(solutions):
            metrics = metrics_calculator.evaluate_solution(problem, solution)
            logger.info(f"  Solution {j+1}: '{solution}'")
            logger.info(f"    Overall Score: {metrics['overall_score']:.3f}")
            logger.info(f"    Exact Match: {metrics['exact_match']:.3f}")
            logger.info(f"    Grammatical Correctness: {metrics['grammatical_correctness']:.3f}")
            logger.info(f"    Semantic Similarity: {metrics['semantic_similarity']:.3f}")
            logger.info("")


def demonstrate_rl_environment():
    """Demonstrate Sanskrit reinforcement learning environment."""
    logger.info("=== Demonstrating Sanskrit RL Environment ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    validator = SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    # Create RL environment
    rl_env = create_sanskrit_rl_environment(rule_engine, validator, max_steps=10)
    
    # Create a simple test problem
    from sanskrit_rewrite_engine.r_zero_integration import SanskritProblem
    
    test_problem = SanskritProblem(
        id="rl_test_001",
        type=SanskritProblemType.SANDHI_APPLICATION,
        difficulty=SanskritDifficultyLevel.BEGINNER,
        input_text="a + i",
        expected_output="e",
        explanation="RL test problem"
    )
    
    # Reset environment
    logger.info("Resetting RL environment...")
    initial_state = rl_env.reset(test_problem)
    logger.info(f"Initial state: {len(initial_state.tokens)} tokens")
    logger.info(f"Available rules: {len(initial_state.available_rules)}")
    logger.info(f"Target output: {initial_state.target_output}")
    
    # Demonstrate some actions
    logger.info("Demonstrating RL actions...")
    
    # Action 1: Skip
    action1 = SanskritAction(type=ActionType.SKIP)
    state1, reward1, done1, info1 = rl_env.step(action1)
    logger.info(f"Action 1 (SKIP): Reward = {reward1.total_reward:.3f}, Done = {done1}")
    
    # Action 2: Terminate
    action2 = SanskritAction(type=ActionType.TERMINATE)
    state2, reward2, done2, info2 = rl_env.step(action2)
    logger.info(f"Action 2 (TERMINATE): Reward = {reward2.total_reward:.3f}, Done = {done2}")
    
    # Get episode summary
    episode_summary = rl_env.get_episode_summary()
    logger.info(f"Episode summary: {episode_summary['total_steps']} steps, "
               f"total reward = {episode_summary['total_reward']:.3f}")


def demonstrate_r_zero_score_computation():
    """Demonstrate R-Zero compatible score computation."""
    logger.info("=== Demonstrating R-Zero Score Computation ===")
    
    # Setup components
    tokenizer, rule_engine, morphological_analyzer, derivation_simulator = setup_sanskrit_engine()
    validator = SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    # Create test data
    from sanskrit_rewrite_engine.r_zero_integration import SanskritProblem
    
    predicts = [
        "e",                           # Correct sandhi
        "gam(ROOT) + ti(SUFFIX)",      # Correct morphology
        "wrong_answer",                # Incorrect
        "a + i becomes e",             # Correct but different format
    ]
    
    ground_truths = [
        "e",
        "gam(ROOT) + ti(SUFFIX)",
        "correct_answer",
        "e",
    ]
    
    problems = [
        SanskritProblem(
            id=f"score_test_{i}",
            type=SanskritProblemType.SANDHI_APPLICATION if i % 2 == 0 else SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text=f"input_{i}",
            expected_output=gt
        )
        for i, gt in enumerate(ground_truths)
    ]
    
    # Compute scores
    logger.info("Computing R-Zero compatible scores...")
    scores = compute_sanskrit_score(predicts, ground_truths, problems, validator)
    
    # Display results
    for i, (predict, ground_truth, score) in enumerate(zip(predicts, ground_truths, scores)):
        logger.info(f"Example {i+1}:")
        logger.info(f"  Prediction: '{predict}'")
        logger.info(f"  Ground Truth: '{ground_truth}'")
        logger.info(f"  Overall Score: {score['overall']:.3f}")
        logger.info(f"  Format Score: {score['format']:.3f}")
        logger.info(f"  Accuracy Score: {score['accuracy']:.3f}")
        logger.info(f"  Exact Match: {score['exact_match']:.3f}")
        logger.info("")


def main():
    """Main demonstration function."""
    logger.info("Starting Sanskrit R-Zero Integration Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demonstrate each component
        problems = demonstrate_problem_generation()
        training_dataset, eval_dataset = demonstrate_data_format_conversion()
        demonstrate_grammatical_validation()
        demonstrate_reasoning_metrics()
        demonstrate_rl_environment()
        demonstrate_r_zero_score_computation()
        
        logger.info("=" * 60)
        logger.info("Sanskrit R-Zero Integration Demonstration Completed Successfully!")
        
        # Summary
        logger.info("\nSummary:")
        logger.info(f"- Generated {len(problems)} Sanskrit problems")
        logger.info(f"- Created training dataset with {len(training_dataset.problems)} problems")
        logger.info(f"- Created evaluation dataset with {len(eval_dataset.problems)} problems")
        logger.info("- Demonstrated grammatical validation")
        logger.info("- Demonstrated reasoning metrics")
        logger.info("- Demonstrated RL environment")
        logger.info("- Demonstrated R-Zero score computation")
        
        logger.info("\nOutput files created in 'output/' directory:")
        logger.info("- sanskrit_training_train.json")
        logger.info("- sanskrit_training_val.json")
        logger.info("- sanskrit_evaluation.json")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()