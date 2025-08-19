"""
Sanskrit Question Evaluation for R-Zero Framework.

This module adapts R-Zero's question evaluation components for Sanskrit
grammatical reasoning problems.
"""

import os
import json
import argparse
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import vllm
from transformers import AutoTokenizer

from .sanskrit_solver import SanskritSolver, SolverConfig
from .r_zero_integration import SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
from .sanskrit_reward_function import SanskritRewardCalculator, compute_sanskrit_score
from .r_zero_config import SanskritRZeroConfig


logger = logging.getLogger(__name__)
STORAGE_PATH = os.getenv("STORAGE_PATH", "./r_zero_storage")


def extract_boxed_answer(response: str) -> str:
    """Extract answer from \\boxed{} tags."""
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', response)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Fallback patterns
    answer_patterns = [
        r'Answer:\s*(.+?)(?:\n|$)',
        r'Final answer:\s*(.+?)(?:\n|$)',
        r'Solution:\s*(.+?)(?:\n|$)',
        r'Result:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    return response.strip()


def grade_sanskrit_answer(predicted: str, expected: str, problem_type: str = None) -> bool:
    """
    Grade Sanskrit answer for correctness.
    
    Args:
        predicted: Predicted answer
        expected: Expected answer
        problem_type: Type of Sanskrit problem
        
    Returns:
        True if answer is correct, False otherwise
    """
    if not predicted or not expected:
        return False
    
    # Normalize answers
    pred_norm = predicted.lower().strip()
    exp_norm = expected.lower().strip()
    
    # Exact match
    if pred_norm == exp_norm:
        return True
    
    # Problem-specific grading
    if problem_type == "SANDHI_APPLICATION":
        # For sandhi, check if the transformation result is present
        if '→' in expected:
            exp_result = expected.split('→')[-1].strip().lower()
            return exp_result in pred_norm
    
    elif problem_type == "MORPHOLOGICAL_ANALYSIS":
        # For morphology, check if key components are identified
        key_terms = ['root', 'suffix', 'dhātu', 'pratyaya']
        exp_terms = [term for term in key_terms if term in exp_norm]
        pred_terms = [term for term in key_terms if term in pred_norm]
        
        if exp_terms and len(set(exp_terms) & set(pred_terms)) >= len(exp_terms) // 2:
            return True
    
    elif problem_type == "COMPOUND_ANALYSIS":
        # For compounds, check if samāsa type is correctly identified
        samasa_types = ['tatpuruṣa', 'karmadhāraya', 'dvandva', 'bahuvrīhi']
        exp_type = next((t for t in samasa_types if t in exp_norm), None)
        pred_type = next((t for t in samasa_types if t in pred_norm), None)
        
        if exp_type and pred_type and exp_type == pred_type:
            return True
    
    # Fuzzy matching for similar answers
    # Calculate Jaccard similarity
    exp_words = set(exp_norm.split())
    pred_words = set(pred_norm.split())
    
    if exp_words and pred_words:
        intersection = len(exp_words & pred_words)
        union = len(exp_words | pred_words)
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= 0.7  # 70% similarity threshold
    
    return False


def evaluate_sanskrit_questions(questions_file: str,
                               model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                               num_samples: int = 10,
                               suffix: str = "001",
                               save_name: str = "evaluation") -> List[Dict[str, Any]]:
    """
    Evaluate Sanskrit questions using the Solver model.
    
    Args:
        questions_file: Path to file containing questions
        model_path: Path to the solver model
        num_samples: Number of solution attempts per question
        suffix: Suffix for output files
        save_name: Base name for saving results
        
    Returns:
        List of evaluation results
    """
    logger.info(f"Evaluating Sanskrit questions from {questions_file}")
    
    # Load questions
    questions_path = Path(STORAGE_PATH) / "generated_question" / questions_file
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    with open(questions_path, "r", encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Filter valid questions (score >= 0)
    valid_questions = [q for q in questions_data if q.get('score', -1) >= 0]
    logger.info(f"Loaded {len(valid_questions)} valid questions")
    
    if not valid_questions:
        logger.warning("No valid questions found")
        return []
    
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=0.85,
        seed=int(suffix) if suffix.isdigit() else 42,
    )
    
    # Prepare prompts
    system_prompt = """You are an expert Sanskrit grammarian with deep knowledge of Pāṇinian grammar.

Solve the given Sanskrit grammatical problem step by step.
Show your reasoning clearly and provide your final answer within \\boxed{} tags.

Key principles:
- Apply sandhi rules correctly for phonological transformations
- Identify morphological components (dhātu, pratyaya) accurately  
- Use systematic derivation processes (śabda-prakriyā)
- Analyze compounds (samāsa) by type and structure
- Reference relevant sūtra numbers when applicable"""
    
    chats = []
    for question_data in valid_questions:
        question = question_data['question']
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {question}\n\nPlease solve this step-by-step and provide your final answer within \\boxed{{}} tags."}
        ]
        chats.append(chat)
    
    # Convert to prompts
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                chat, 
                tokenize=False,
                add_generation_prompt=True, 
                add_special_tokens=True
            ) 
            for chat in chats
        ]
    else:
        prompts = [
            f"system: {chat[0]['content']}\nuser: {chat[1]['content']}" 
            for chat in chats
        ]
    
    # Generation parameters
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        n=num_samples,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # Generate responses
    logger.info("Generating solutions...")
    responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    
    # Process results
    results = []
    reward_calculator = SanskritRewardCalculator()
    
    for i, (question_data, response) in enumerate(zip(valid_questions, responses)):
        try:
            question = question_data['question']
            expected_answer = question_data.get('answer', '')
            problem_type = question_data.get('metadata', {}).get('type', '')
            
            # Extract answers from all response samples
            sample_answers = []
            for output in response.outputs:
                answer = extract_boxed_answer(output.text)
                sample_answers.append(answer)
            
            # Find consensus answer (most frequent)
            answer_counts = {}
            for answer in sample_answers:
                # Group similar answers
                found_group = False
                for existing_answer in answer_counts:
                    if grade_sanskrit_answer(answer, existing_answer, problem_type):
                        answer_counts[existing_answer] += 1
                        found_group = True
                        break
                
                if not found_group:
                    answer_counts[answer] = 1
            
            # Select most frequent answer
            if answer_counts:
                consensus_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                consensus_count = answer_counts[consensus_answer]
                consensus_score = consensus_count / len(sample_answers)
            else:
                consensus_answer = sample_answers[0] if sample_answers else ""
                consensus_score = 0.0
            
            # Grade the consensus answer
            is_correct = grade_sanskrit_answer(consensus_answer, expected_answer, problem_type)
            
            # Calculate detailed reward
            if expected_answer:
                # Create a mock problem for reward calculation
                mock_problem = type('MockProblem', (), {
                    'type': type('MockType', (), {'value': problem_type})(),
                    'input_text': question,
                    'expected_output': expected_answer,
                    'difficulty': type('MockDifficulty', (), {'value': 'INTERMEDIATE'})()
                })()
                
                reward = reward_calculator.calculate_reward(
                    consensus_answer, expected_answer, mock_problem
                )
                detailed_score = reward.overall_score
            else:
                detailed_score = consensus_score
            
            result = {
                "question_id": question_data.get('id', f"q_{i}"),
                "question": question,
                "expected_answer": expected_answer,
                "consensus_answer": consensus_answer,
                "consensus_score": consensus_score,
                "is_correct": is_correct,
                "detailed_score": detailed_score,
                "sample_answers": sample_answers,
                "problem_type": problem_type,
                "metadata": question_data.get('metadata', {})
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(valid_questions)} questions")
        
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            results.append({
                "question_id": question_data.get('id', f"q_{i}"),
                "question": question_data.get('question', ''),
                "error": str(e),
                "is_correct": False,
                "detailed_score": 0.0
            })
    
    # Calculate overall statistics
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    avg_detailed_score = sum(r.get('detailed_score', 0.0) for r in results) / total_count if total_count > 0 else 0.0
    
    logger.info(f"Evaluation completed: {correct_count}/{total_count} correct ({accuracy:.2%})")
    logger.info(f"Average detailed score: {avg_detailed_score:.3f}")
    
    # Save results
    output_dir = Path(STORAGE_PATH) / "generated_question"
    output_file = output_dir / f"{save_name}_{suffix}_results.json"
    
    evaluation_data = {
        "results": results,
        "statistics": {
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_detailed_score": avg_detailed_score
        },
        "metadata": {
            "model_path": model_path,
            "num_samples": num_samples,
            "questions_file": questions_file,
            "evaluation_timestamp": str(Path().cwd())
        }
    }
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved evaluation results to {output_file}")
    return results


def evaluate_with_solver_model(questions_file: str,
                              solver_config_path: Optional[str] = None,
                              save_name: str = "solver_evaluation",
                              suffix: str = "001") -> List[Dict[str, Any]]:
    """
    Evaluate questions using the Sanskrit Solver model.
    
    Args:
        questions_file: Path to file containing questions
        solver_config_path: Path to solver configuration
        save_name: Base name for saving results
        suffix: Suffix for output files
        
    Returns:
        List of evaluation results
    """
    logger.info(f"Evaluating questions using Solver model")
    
    # Load questions
    questions_path = Path(STORAGE_PATH) / "generated_question" / questions_file
    with open(questions_path, "r", encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Convert to SanskritProblem objects
    problems = []
    for q_data in questions_data:
        if q_data.get('score', -1) >= 0:  # Valid questions only
            try:
                problem_type = SanskritProblemType(q_data.get('metadata', {}).get('type', 'SANDHI_APPLICATION'))
                difficulty = SanskritDifficultyLevel(q_data.get('metadata', {}).get('difficulty', 'INTERMEDIATE'))
                
                problem = SanskritProblem(
                    id=q_data.get('id', f"eval_{len(problems)}"),
                    type=problem_type,
                    difficulty=difficulty,
                    input_text=q_data['question'],
                    expected_output=q_data.get('answer', ''),
                    metadata=q_data.get('metadata', {})
                )
                problems.append(problem)
            except Exception as e:
                logger.warning(f"Skipping invalid problem: {e}")
    
    logger.info(f"Converted {len(problems)} questions to problems")
    
    # Create solver
    if solver_config_path and Path(solver_config_path).exists():
        r_zero_config = SanskritRZeroConfig.load_from_yaml(solver_config_path)
    else:
        r_zero_config = SanskritRZeroConfig()
    
    solver_config = SolverConfig(
        num_samples=5,  # Fewer samples for evaluation
        seed=int(suffix) if suffix.isdigit() else 42
    )
    
    # Initialize solver (simplified)
    from .tokenizer import SanskritTokenizer
    from .panini_engine import PaniniRuleEngine
    from .morphological_analyzer import SanskritMorphologicalAnalyzer
    from .r_zero_integration import SanskritGrammaticalValidator
    
    tokenizer = SanskritTokenizer()
    rule_engine = PaniniRuleEngine(tokenizer)
    morphological_analyzer = SanskritMorphologicalAnalyzer(tokenizer, rule_engine)
    validator = SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    solver = SanskritSolver(
        config=solver_config,
        validator=validator,
        storage_path=r_zero_config.storage_path
    )
    
    # Solve problems
    solutions = solver.solve_problems(problems)
    
    # Evaluate performance
    performance = solver.evaluate_performance(problems, solutions)
    
    # Save results
    results_file = solver.save_results(problems, solutions, f"{save_name}_{suffix}")
    
    logger.info(f"Solver evaluation completed with performance: {performance}")
    return solutions


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate Sanskrit grammatical questions")
    parser.add_argument("--questions_file", type=str, required=True,
                       help="File containing questions to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model path for evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of solution attempts per question")
    parser.add_argument("--suffix", type=str, default="001",
                       help="Suffix for output files")
    parser.add_argument("--save_name", type=str, default="evaluation",
                       help="Base name for saving results")
    parser.add_argument("--use_solver", action="store_true",
                       help="Use Solver model instead of direct evaluation")
    parser.add_argument("--config_path", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.use_solver:
            results = evaluate_with_solver_model(
                questions_file=args.questions_file,
                solver_config_path=args.config_path,
                save_name=args.save_name,
                suffix=args.suffix
            )
            print(f"Evaluated {len(results)} questions using Solver model")
        else:
            results = evaluate_sanskrit_questions(
                questions_file=args.questions_file,
                model_path=args.model,
                num_samples=args.num_samples,
                suffix=args.suffix,
                save_name=args.save_name
            )
            
            # Print summary statistics
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            print(f"Evaluation completed:")
            print(f"  Total questions: {total_count}")
            print(f"  Correct answers: {correct_count}")
            print(f"  Accuracy: {accuracy:.2%}")
            
            # Show sample results
            print("\nSample results:")
            for i, result in enumerate(results[:3]):
                print(f"\n{i+1}. Question: {result['question'][:100]}...")
                print(f"   Expected: {result.get('expected_answer', 'N/A')[:50]}...")
                print(f"   Got: {result.get('consensus_answer', 'N/A')[:50]}...")
                print(f"   Correct: {result.get('is_correct', False)}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()