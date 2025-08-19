"""
Sanskrit Question Generation for R-Zero Framework.

This module adapts R-Zero's question generation components for Sanskrit
grammatical reasoning problems.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import vllm
from transformers import AutoTokenizer

from .sanskrit_challenger import SanskritChallenger, ChallengerConfig
from .r_zero_integration import SanskritProblemType, SanskritDifficultyLevel
from .r_zero_config import SanskritRZeroConfig


logger = logging.getLogger(__name__)
STORAGE_PATH = os.getenv("STORAGE_PATH", "./r_zero_storage")


def extract_sanskrit_problem(text: str) -> Dict[str, str]:
    """Extract Sanskrit problem components from generated text."""
    import re
    
    # Look for structured problem format
    problem_match = re.search(r'<problem>(.*?)</problem>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if problem_match and answer_match:
        return {
            "question": problem_match.group(1).strip(),
            "answer": answer_match.group(1).strip(),
            "score": 0
        }
    
    # Fallback: look for boxed answers
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed_matches:
        # Try to extract question from text before the boxed answer
        parts = text.split('\\boxed{')
        if len(parts) > 1:
            question = parts[0].strip()
            answer = boxed_matches[-1].strip()
            return {
                "question": question,
                "answer": answer,
                "score": 0
            }
    
    # If no structured format found, return the whole text as question
    return {
        "question": text.strip(),
        "answer": "",
        "score": -1
    }


def generate_sanskrit_questions(model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                               num_samples: int = 100,
                               problem_type: Optional[str] = None,
                               difficulty: Optional[str] = None,
                               save_name: str = "sanskrit_questions",
                               suffix: str = "001") -> List[Dict[str, Any]]:
    """
    Generate Sanskrit grammatical questions using the Challenger model.
    
    Args:
        model_path: Path to the language model
        num_samples: Number of questions to generate
        problem_type: Specific problem type to focus on
        difficulty: Specific difficulty level
        save_name: Base name for saving results
        suffix: Suffix for the output file
        
    Returns:
        List of generated questions
    """
    logger.info(f"Generating {num_samples} Sanskrit questions")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        seed=int(suffix) if suffix.isdigit() else 42,
    )
    
    # Create system prompt for Sanskrit question generation
    system_prompt = """You are an expert Sanskrit grammarian and teacher specializing in Pāṇinian grammar.

Your task is to generate challenging Sanskrit grammatical problems that test deep understanding of:
- Sandhi (phonological transformations)
- Morphological analysis (dhātu, pratyaya identification)  
- Word derivation (śabda-prakriyā)
- Compound analysis (samāsa)
- Rule application (sūtra-based transformations)
- Grammatical validation

Generate problems that are:
1. Linguistically accurate according to Pāṇinian principles
2. Appropriately challenging for advanced students
3. Clear and unambiguous in their requirements
4. Solvable using systematic grammatical analysis

Format your output exactly as:
<problem>
[Clear problem statement with Sanskrit text to analyze]
</problem>

<answer>
[Correct solution with brief explanation]
</answer>

Do NOT output anything else—no extra explanations, no additional markup."""
    
    # Create user prompt based on parameters
    user_prompt = "Generate one new, challenging Sanskrit grammatical reasoning question now."
    
    if problem_type:
        type_descriptions = {
            "SANDHI_APPLICATION": "sandhi (phonological transformation)",
            "MORPHOLOGICAL_ANALYSIS": "morphological analysis",
            "WORD_DERIVATION": "word derivation (śabda-prakriyā)",
            "COMPOUND_ANALYSIS": "compound analysis (samāsa)",
            "RULE_APPLICATION": "Pāṇini rule application",
            "GRAMMATICAL_VALIDATION": "grammatical validation"
        }
        if problem_type in type_descriptions:
            user_prompt = f"Generate a challenging {type_descriptions[problem_type]} problem in Sanskrit."
    
    if difficulty:
        difficulty_guidance = {
            "BEGINNER": "suitable for beginners (simple, common words)",
            "INTERMEDIATE": "at intermediate level (moderate complexity)",
            "ADVANCED": "at advanced level (complex constructions)",
            "EXPERT": "at expert level (highly complex, rare constructions)"
        }
        if difficulty in difficulty_guidance:
            user_prompt += f" Make it {difficulty_guidance[difficulty]}."
    
    user_prompt += " Remember to format the output exactly as instructed."
    
    # Create chat format
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, 
            tokenize=False,
            add_generation_prompt=True, 
            add_special_tokens=True
        )
    else:
        prompt = f"system: {chat[0]['content']}\nuser: {chat[1]['content']}"
    
    # Generation parameters
    sample_params = vllm.SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # Generate questions
    logger.info("Generating questions with model...")
    completions = model.generate([prompt] * num_samples, sampling_params=sample_params)
    
    # Process results
    results = []
    for i, completion in enumerate(completions):
        response = completion.outputs[0].text
        try:
            question_data = extract_sanskrit_problem(response)
            question_data["id"] = f"sanskrit_{save_name}_{suffix}_{i:04d}"
            results.append(question_data)
        except Exception as e:
            logger.warning(f"Failed to parse question {i}: {e}")
            results.append({
                "id": f"sanskrit_{save_name}_{suffix}_{i:04d}",
                "question": response,
                "answer": "",
                "score": -1
            })
    
    # Save results
    output_dir = Path(STORAGE_PATH) / "generated_question"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{save_name}_{suffix}.json"
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated {len(results)} questions, saved to {output_file}")
    return results


def generate_with_challenger_model(challenger_config_path: Optional[str] = None,
                                  num_samples: int = 100,
                                  save_name: str = "challenger_questions",
                                  suffix: str = "001") -> List[Dict[str, Any]]:
    """
    Generate questions using the Sanskrit Challenger model.
    
    Args:
        challenger_config_path: Path to challenger configuration
        num_samples: Number of questions to generate
        save_name: Base name for saving results
        suffix: Suffix for the output file
        
    Returns:
        List of generated questions
    """
    logger.info(f"Generating {num_samples} questions using Challenger model")
    
    # Load configuration
    if challenger_config_path and Path(challenger_config_path).exists():
        r_zero_config = SanskritRZeroConfig.load_from_yaml(challenger_config_path)
    else:
        r_zero_config = SanskritRZeroConfig()
    
    # Create challenger config
    challenger_config = ChallengerConfig(
        num_samples=min(10, num_samples // 10),  # Generate in batches
        seed=int(suffix) if suffix.isdigit() else 42
    )
    
    # Initialize challenger (simplified)
    from .tokenizer import SanskritTokenizer
    from .panini_engine import PaniniRuleEngine
    from .morphological_analyzer import SanskritMorphologicalAnalyzer
    from .derivation_simulator import ShabdaPrakriyaSimulator
    from .r_zero_integration import SanskritProblemGenerator
    
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
    
    challenger = SanskritChallenger(
        config=challenger_config,
        problem_generator=problem_generator,
        storage_path=r_zero_config.storage_path
    )
    
    # Generate problems across different types and difficulties
    all_problems = []
    problem_types = list(SanskritProblemType)
    difficulties = list(SanskritDifficultyLevel)
    
    problems_per_batch = max(1, num_samples // (len(problem_types) * len(difficulties)))
    
    for problem_type in problem_types:
        for difficulty in difficulties:
            batch_problems = challenger.generate_problems(
                problem_type=problem_type,
                difficulty=difficulty,
                num_problems=problems_per_batch
            )
            all_problems.extend(batch_problems)
    
    # Convert to R-Zero format
    results = []
    for i, problem in enumerate(all_problems[:num_samples]):
        r_zero_format = problem.to_r_zero_format()
        results.append({
            "id": f"challenger_{save_name}_{suffix}_{i:04d}",
            "question": r_zero_format["problem"],
            "answer": r_zero_format["answer"],
            "score": 0,
            "metadata": r_zero_format.get("metadata", {})
        })
    
    # Save results
    output_dir = Path(STORAGE_PATH) / "generated_question"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{save_name}_{suffix}.json"
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated {len(results)} questions using Challenger, saved to {output_file}")
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate Sanskrit grammatical questions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model path for question generation")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of questions to generate")
    parser.add_argument("--problem_type", type=str, choices=[t.value for t in SanskritProblemType],
                       help="Specific problem type to focus on")
    parser.add_argument("--difficulty", type=str, choices=[d.value for d in SanskritDifficultyLevel],
                       help="Specific difficulty level")
    parser.add_argument("--save_name", type=str, default="sanskrit_questions",
                       help="Base name for saving results")
    parser.add_argument("--suffix", type=str, default="001",
                       help="Suffix for the output file")
    parser.add_argument("--use_challenger", action="store_true",
                       help="Use Challenger model instead of direct generation")
    parser.add_argument("--config_path", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.use_challenger:
            results = generate_with_challenger_model(
                challenger_config_path=args.config_path,
                num_samples=args.num_samples,
                save_name=args.save_name,
                suffix=args.suffix
            )
        else:
            results = generate_sanskrit_questions(
                model_path=args.model,
                num_samples=args.num_samples,
                problem_type=args.problem_type,
                difficulty=args.difficulty,
                save_name=args.save_name,
                suffix=args.suffix
            )
        
        print(f"Successfully generated {len(results)} Sanskrit questions")
        
        # Print sample questions
        print("\nSample questions:")
        for i, result in enumerate(results[:3]):
            print(f"\n{i+1}. {result['question'][:100]}...")
            if result['answer']:
                print(f"   Answer: {result['answer'][:50]}...")
    
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise


if __name__ == "__main__":
    main()