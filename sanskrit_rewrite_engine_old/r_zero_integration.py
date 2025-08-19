"""
R-Zero integration for Sanskrit reasoning system.

This module provides the integration layer between the Sanskrit Rewrite Engine
and the R-Zero self-evolving reasoning framework. It includes:
- Sanskrit-specific training data format compatible with R-Zero
- Sanskrit problem-solution pairs for Challenger-Solver training
- Sanskrit grammatical correctness validation functions
- Sanskrit reasoning evaluation metrics
- Sanskrit rule application as reinforcement learning environment
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
import json
import re
import random
from datetime import datetime
import logging

from .token import Token, TokenKind
from .panini_engine import PaniniRuleEngine, PaniniEngineResult
from .morphological_analyzer import SanskritMorphologicalAnalyzer, MorphologicalAnalysis
from .derivation_simulator import ShabdaPrakriyaSimulator, DerivationTree, DerivationContext
from .tokenizer import SanskritTokenizer
from .essential_sutras import create_essential_sutras


class SanskritProblemType(Enum):
    """Types of Sanskrit reasoning problems."""
    SANDHI_APPLICATION = "SANDHI_APPLICATION"
    MORPHOLOGICAL_ANALYSIS = "MORPHOLOGICAL_ANALYSIS"
    WORD_DERIVATION = "WORD_DERIVATION"
    COMPOUND_ANALYSIS = "COMPOUND_ANALYSIS"
    RULE_APPLICATION = "RULE_APPLICATION"
    GRAMMATICAL_VALIDATION = "GRAMMATICAL_VALIDATION"
    TRANSLATION_SYNTHESIS = "TRANSLATION_SYNTHESIS"


class SanskritDifficultyLevel(Enum):
    """Difficulty levels for Sanskrit problems."""
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    EXPERT = "EXPERT"


@dataclass
class SanskritProblem:
    """
    A Sanskrit reasoning problem for R-Zero training.
    
    Attributes:
        id: Unique problem identifier
        type: Type of Sanskrit problem
        difficulty: Difficulty level
        input_text: Input Sanskrit text or expression
        expected_output: Expected solution/output
        context: Additional context information
        sutra_references: Relevant Pāṇini sūtra references
        explanation: Human-readable explanation
        metadata: Additional problem-specific data
    """
    id: str
    type: SanskritProblemType
    difficulty: SanskritDifficultyLevel
    input_text: str
    expected_output: str
    context: Dict[str, Any] = field(default_factory=dict)
    sutra_references: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_r_zero_format(self) -> Dict[str, Any]:
        """Convert to R-Zero compatible format."""
        return {
            "problem": self._format_problem_text(),
            "answer": self.expected_output,
            "metadata": {
                "id": self.id,
                "type": self.type.value,
                "difficulty": self.difficulty.value,
                "sutra_references": self.sutra_references,
                "explanation": self.explanation,
                **self.metadata
            }
        }
    
    def _format_problem_text(self) -> str:
        """Format problem text for R-Zero training."""
        problem_templates = {
            SanskritProblemType.SANDHI_APPLICATION: 
                f"TYPE: SANDHI\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Apply sandhi rules and return the final form with explanation.\"",
            
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
                f"TYPE: MORPHOLOGY\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Analyze morphological structure and identify dhātu, pratyaya, and grammatical categories.\"",
            
            SanskritProblemType.WORD_DERIVATION:
                f"TYPE: DERIVATION\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Provide step-by-step śabda-prakriyā derivation with sūtra citations.\"",
            
            SanskritProblemType.COMPOUND_ANALYSIS:
                f"TYPE: COMPOUND\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Analyze compound structure, identify samāsa type and constituents.\"",
            
            SanskritProblemType.RULE_APPLICATION:
                f"TYPE: RULE\\nINPUT: \"{self.input_text}\"\\nCONTEXT: \"{self.context.get('rule_context', '')}\"\\nINSTRUCTIONS: \"Apply the specified Pāṇini rule and show the transformation.\"",
            
            SanskritProblemType.GRAMMATICAL_VALIDATION:
                f"TYPE: VALIDATION\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Validate grammatical correctness and identify any errors with corrections.\"",
            
            SanskritProblemType.TRANSLATION_SYNTHESIS:
                f"TYPE: SYNTHESIS\\nINPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Synthesize Sanskrit expression from the given algorithmic or mathematical description.\""
        }
        
        return problem_templates.get(self.type, f"INPUT: \"{self.input_text}\"\\nINSTRUCTIONS: \"Process the Sanskrit input and provide appropriate analysis.\"")


@dataclass
class SanskritSolution:
    """
    A solution to a Sanskrit reasoning problem.
    
    Attributes:
        problem_id: ID of the associated problem
        solution_text: The solution text
        confidence: Confidence score (0.0 to 1.0)
        reasoning_steps: Step-by-step reasoning
        rule_applications: Rules that were applied
        validation_results: Validation against Sanskrit grammar
        metadata: Additional solution data
    """
    problem_id: str
    solution_text: str
    confidence: float = 1.0
    reasoning_steps: List[str] = field(default_factory=list)
    rule_applications: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SanskritProblemGenerator:
    """
    Generator for Sanskrit reasoning problems compatible with R-Zero.
    
    Creates diverse Sanskrit problems across different types and difficulty levels
    for training the Challenger-Solver co-evolutionary loop.
    """
    
    def __init__(self, 
                 tokenizer: SanskritTokenizer,
                 rule_engine: PaniniRuleEngine,
                 morphological_analyzer: SanskritMorphologicalAnalyzer,
                 derivation_simulator: ShabdaPrakriyaSimulator):
        """
        Initialize the problem generator.
        
        Args:
            tokenizer: Sanskrit tokenizer
            rule_engine: Pāṇini rule engine
            morphological_analyzer: Morphological analyzer
            derivation_simulator: Word derivation simulator
        """
        self.tokenizer = tokenizer
        self.rule_engine = rule_engine
        self.morphological_analyzer = morphological_analyzer
        self.derivation_simulator = derivation_simulator
        self.logger = logging.getLogger(__name__)
        
        # Problem templates and data
        self._initialize_problem_templates()
        self._initialize_sanskrit_corpus()
    
    def _initialize_problem_templates(self):
        """Initialize problem generation templates."""
        self.sandhi_templates = [
            ("a", "i", "e"),  # a + i → e
            ("a", "u", "o"),  # a + u → o
            ("a", "a", "ā"),  # a + a → ā
            ("i", "a", "ya"), # i + a → ya
            ("u", "a", "va"), # u + a → va
        ]
        
        self.morphology_examples = [
            "gacchati",  # goes
            "karoti",    # does
            "bhavati",   # becomes
            "paśyati",   # sees
            "śṛṇoti",    # hears
        ]
        
        self.compound_examples = [
            ("rāja", "putra", "rājaputra"),      # king's son
            ("grāma", "vāsa", "grāmavāsa"),     # village dwelling
            ("nīla", "utpala", "nīlotpala"),    # blue lotus
            ("mahā", "rāja", "mahārāja"),       # great king
        ]
        
        self.derivation_examples = [
            ("gam", "ti", "gacchati"),          # go + present → goes
            ("kar", "tavya", "kartavya"),       # do + gerundive → to be done
            ("bhū", "ta", "bhūta"),             # be + past participle → been
        ]
    
    def _initialize_sanskrit_corpus(self):
        """Initialize Sanskrit text corpus for problem generation."""
        self.corpus_words = [
            "dharma", "artha", "kāma", "mokṣa",  # Four goals
            "satya", "ahiṃsā", "dayā", "kṣamā",  # Virtues
            "vidyā", "jñāna", "buddhi", "mati",  # Knowledge terms
            "rāma", "kṛṣṇa", "śiva", "viṣṇu",    # Deities
            "gaṅgā", "yamunā", "sarasvatī",      # Rivers
            "sūrya", "candra", "agni", "vāyu",   # Elements
        ]
        
        self.compound_patterns = [
            ("tatpuruṣa", ["rājaputra", "devālaya", "grāmavāsa"]),
            ("karmadhāraya", ["nīlotpala", "mahārāja", "śubhakāma"]),
            ("dvandva", ["rāmalakṣmaṇa", "pitāmātā", "sītārāma"]),
            ("bahuvrīhi", ["cakrapāṇi", "gajavaktra", "padmākṣa"]),
        ]
    
    def generate_problems(self, 
                         count: int = 100,
                         problem_types: Optional[List[SanskritProblemType]] = None,
                         difficulty_distribution: Optional[Dict[SanskritDifficultyLevel, float]] = None) -> List[SanskritProblem]:
        """
        Generate a set of Sanskrit reasoning problems.
        
        Args:
            count: Number of problems to generate
            problem_types: Types of problems to generate (all types if None)
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of generated Sanskrit problems
        """
        if problem_types is None:
            problem_types = list(SanskritProblemType)
        
        if difficulty_distribution is None:
            difficulty_distribution = {
                SanskritDifficultyLevel.BEGINNER: 0.3,
                SanskritDifficultyLevel.INTERMEDIATE: 0.4,
                SanskritDifficultyLevel.ADVANCED: 0.2,
                SanskritDifficultyLevel.EXPERT: 0.1
            }
        
        problems = []
        
        for i in range(count):
            # Select problem type and difficulty
            problem_type = random.choice(problem_types)
            difficulty = self._select_difficulty(difficulty_distribution)
            
            # Generate problem based on type
            problem = self._generate_problem_by_type(
                problem_id=f"sanskrit_{problem_type.value.lower()}_{i:04d}",
                problem_type=problem_type,
                difficulty=difficulty
            )
            
            if problem:
                problems.append(problem)
        
        self.logger.info(f"Generated {len(problems)} Sanskrit problems")
        return problems
    
    def _select_difficulty(self, distribution: Dict[SanskritDifficultyLevel, float]) -> SanskritDifficultyLevel:
        """Select difficulty level based on distribution."""
        rand_val = random.random()
        cumulative = 0.0
        
        for difficulty, prob in distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                return difficulty
        
        return SanskritDifficultyLevel.INTERMEDIATE  # Fallback
    
    def _generate_problem_by_type(self, 
                                 problem_id: str,
                                 problem_type: SanskritProblemType,
                                 difficulty: SanskritDifficultyLevel) -> Optional[SanskritProblem]:
        """Generate a problem of a specific type."""
        generators = {
            SanskritProblemType.SANDHI_APPLICATION: self._generate_sandhi_problem,
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: self._generate_morphology_problem,
            SanskritProblemType.WORD_DERIVATION: self._generate_derivation_problem,
            SanskritProblemType.COMPOUND_ANALYSIS: self._generate_compound_problem,
            SanskritProblemType.RULE_APPLICATION: self._generate_rule_problem,
            SanskritProblemType.GRAMMATICAL_VALIDATION: self._generate_validation_problem,
            SanskritProblemType.TRANSLATION_SYNTHESIS: self._generate_synthesis_problem,
        }
        
        generator = generators.get(problem_type)
        if generator:
            return generator(problem_id, difficulty)
        
        return None
    
    def _generate_sandhi_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a sandhi application problem."""
        # Select sandhi rule based on difficulty
        if difficulty == SanskritDifficultyLevel.BEGINNER:
            templates = self.sandhi_templates[:3]  # Simple vowel sandhi
        else:
            templates = self.sandhi_templates  # All sandhi rules
        
        vowel1, vowel2, result = random.choice(templates)
        
        # Create compound input
        word1 = random.choice(["rāma", "kṛṣṇa", "dharma"])
        word2 = random.choice(["iti", "eva", "api"])
        
        # Modify words to end/start with the selected vowels
        if word1.endswith('a') and vowel1 == 'a':
            input_text = f"{word1} + {vowel2}{word2[1:]}"
            expected_output = f"{word1[:-1]}{result}{word2[1:]}"
        else:
            input_text = f"{vowel1} + {vowel2}"
            expected_output = result
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=difficulty,
            input_text=input_text,
            expected_output=expected_output,
            sutra_references=["6.1.87", "6.1.101"],  # Common sandhi sūtras
            explanation=f"Vowel sandhi: {vowel1} + {vowel2} → {result}",
            metadata={"sandhi_type": "vowel", "rule_applied": f"{vowel1}+{vowel2}→{result}"}
        )
    
    def _generate_morphology_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a morphological analysis problem."""
        word = random.choice(self.morphology_examples)
        
        # Analyze the word using our morphological analyzer
        analysis = self.morphological_analyzer.analyze_word(word)
        
        # Format expected output
        morphemes = []
        for morpheme in analysis.morphemes:
            morphemes.append(f"{morpheme.text}({morpheme.type.value})")
        
        expected_output = " + ".join(morphemes)
        
        # Add grammatical categories
        if analysis.grammatical_categories:
            categories = [cat.value for cat in analysis.grammatical_categories]
            expected_output += f" | Categories: {', '.join(categories)}"
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            difficulty=difficulty,
            input_text=word,
            expected_output=expected_output,
            explanation=f"Morphological analysis of '{word}'",
            metadata={"word_class": "verb", "analysis_confidence": analysis.confidence}
        )
    
    def _generate_derivation_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a word derivation problem."""
        root, suffix, derived_word = random.choice(self.derivation_examples)
        
        # Create derivation context
        context = DerivationContext(source_text=derived_word)
        
        # Simulate derivation
        derivation_tree = self.derivation_simulator.derive_word(context)
        
        # Format expected output with derivation steps
        steps = derivation_tree.get_derivation_summary()
        expected_output = " → ".join(steps) if steps else f"{root} + {suffix} → {derived_word}"
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.WORD_DERIVATION,
            difficulty=difficulty,
            input_text=f"{root} + {suffix}",
            expected_output=expected_output,
            sutra_references=[step.sutra_citation for step in derivation_tree.derivation_steps if step.sutra_citation],
            explanation=f"Derivation of '{derived_word}' from root '{root}' with suffix '{suffix}'",
            metadata={"derivation_confidence": derivation_tree.confidence, "steps_count": len(derivation_tree.derivation_steps)}
        )
    
    def _generate_compound_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a compound analysis problem."""
        compound_type, examples = random.choice(self.compound_patterns)
        compound = random.choice(examples)
        
        # Analyze compound structure (simplified)
        if compound_type == "tatpuruṣa":
            # Determinative compound - second element is head
            parts = self._split_compound(compound)
            expected_output = f"Type: {compound_type} | Parts: {' + '.join(parts)} | Head: {parts[-1]}"
        elif compound_type == "karmadhāraya":
            # Descriptive compound - both elements describe the same entity
            parts = self._split_compound(compound)
            expected_output = f"Type: {compound_type} | Parts: {' + '.join(parts)} | Relation: descriptive"
        else:
            parts = self._split_compound(compound)
            expected_output = f"Type: {compound_type} | Parts: {' + '.join(parts)}"
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.COMPOUND_ANALYSIS,
            difficulty=difficulty,
            input_text=compound,
            expected_output=expected_output,
            explanation=f"Analysis of {compound_type} compound '{compound}'",
            metadata={"compound_type": compound_type, "constituent_count": len(parts)}
        )
    
    def _generate_rule_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a rule application problem."""
        # Get a random rule from the essential sūtras
        essential_rules = create_essential_sutras()
        rule = random.choice(essential_rules)
        
        # Create a simple input that the rule can apply to
        input_text = "a + i"  # Simple vowel combination
        expected_output = "e"   # Result of vowel sandhi
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.RULE_APPLICATION,
            difficulty=difficulty,
            input_text=input_text,
            expected_output=expected_output,
            context={"rule_context": f"Apply rule: {rule.name}"},
            sutra_references=[str(rule.sutra_ref)],
            explanation=f"Application of {rule.name} ({rule.sutra_ref})",
            metadata={"rule_id": rule.id, "rule_priority": rule.priority}
        )
    
    def _generate_validation_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a grammatical validation problem."""
        # Create both correct and incorrect forms
        correct_forms = ["gacchati", "karoti", "bhavati"]
        incorrect_forms = ["gacchanti", "karanti", "bhavanti"]  # Wrong number
        
        if random.choice([True, False]):
            # Correct form
            word = random.choice(correct_forms)
            expected_output = f"VALID: '{word}' is grammatically correct."
        else:
            # Incorrect form
            word = random.choice(incorrect_forms)
            correct_form = word.replace("anti", "ati")  # Simple correction
            expected_output = f"INVALID: '{word}' should be '{correct_form}' (singular form required)."
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.GRAMMATICAL_VALIDATION,
            difficulty=difficulty,
            input_text=word,
            expected_output=expected_output,
            explanation=f"Grammatical validation of '{word}'",
            metadata={"validation_type": "verbal_form"}
        )
    
    def _generate_synthesis_problem(self, problem_id: str, difficulty: SanskritDifficultyLevel) -> SanskritProblem:
        """Generate a translation synthesis problem."""
        algorithmic_descriptions = [
            "Repeat an action until a condition is met",
            "Choose between two alternatives based on a condition",
            "Combine multiple elements into one",
            "Transform input to output through a process"
        ]
        
        sanskrit_equivalents = [
            "yāvat + condition + tāvat + action",
            "yadi + condition + tarhi + action1 + anyathā + action2",
            "samyoga + elements → ekatva",
            "pariṇāma + input → output"
        ]
        
        description = random.choice(algorithmic_descriptions)
        sanskrit_form = random.choice(sanskrit_equivalents)
        
        return SanskritProblem(
            id=problem_id,
            type=SanskritProblemType.TRANSLATION_SYNTHESIS,
            difficulty=difficulty,
            input_text=description,
            expected_output=sanskrit_form,
            explanation=f"Sanskrit synthesis of algorithmic concept: {description}",
            metadata={"synthesis_type": "algorithmic_to_sanskrit"}
        )
    
    def _split_compound(self, compound: str) -> List[str]:
        """Split compound into constituent parts (simplified)."""
        # This is a simplified implementation
        # In practice, this would use sophisticated compound analysis
        compound_splits = {
            "rājaputra": ["rāja", "putra"],
            "devālaya": ["deva", "ālaya"],
            "grāmavāsa": ["grāma", "vāsa"],
            "nīlotpala": ["nīla", "utpala"],
            "mahārāja": ["mahā", "rāja"],
            "śubhakāma": ["śubha", "kāma"],
            "rāmalakṣmaṇa": ["rāma", "lakṣmaṇa"],
            "pitāmātā": ["pitā", "mātā"],
            "sītārāma": ["sītā", "rāma"],
            "cakrapāṇi": ["cakra", "pāṇi"],
            "gajavaktra": ["gaja", "vaktra"],
            "padmākṣa": ["padma", "akṣa"],
        }
        
        return compound_splits.get(compound, [compound[:len(compound)//2], compound[len(compound)//2:]])


class SanskritGrammaticalValidator:
    """
    Validator for Sanskrit grammatical correctness.
    
    Provides comprehensive validation functions for evaluating Sanskrit
    text against Pāṇinian grammatical rules.
    """
    
    def __init__(self, 
                 rule_engine: PaniniRuleEngine,
                 morphological_analyzer: SanskritMorphologicalAnalyzer):
        """
        Initialize the grammatical validator.
        
        Args:
            rule_engine: Pāṇini rule engine for validation
            morphological_analyzer: Morphological analyzer
        """
        self.rule_engine = rule_engine
        self.morphological_analyzer = morphological_analyzer
        self.logger = logging.getLogger(__name__)
    
    def validate_sanskrit_text(self, text: str) -> Dict[str, Any]:
        """
        Validate Sanskrit text for grammatical correctness.
        
        Args:
            text: Sanskrit text to validate
            
        Returns:
            Validation results with scores and error details
        """
        validation_result = {
            "text": text,
            "is_valid": True,
            "confidence": 1.0,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "metrics": {}
        }
        
        try:
            # Tokenize the text
            tokens = self.rule_engine.tokenizer.tokenize(text)
            
            # Check phonological validity
            phonological_score = self._validate_phonology(tokens)
            validation_result["metrics"]["phonological_score"] = phonological_score
            
            # Check morphological validity
            morphological_score = self._validate_morphology(text)
            validation_result["metrics"]["morphological_score"] = morphological_score
            
            # Check syntactic validity
            syntactic_score = self._validate_syntax(tokens)
            validation_result["metrics"]["syntactic_score"] = syntactic_score
            
            # Check sandhi correctness
            sandhi_score = self._validate_sandhi(tokens)
            validation_result["metrics"]["sandhi_score"] = sandhi_score
            
            # Calculate overall confidence
            overall_confidence = (phonological_score + morphological_score + 
                                syntactic_score + sandhi_score) / 4.0
            validation_result["confidence"] = overall_confidence
            validation_result["is_valid"] = overall_confidence >= 0.7
            
        except Exception as e:
            self.logger.error(f"Validation error for text '{text}': {str(e)}")
            validation_result["is_valid"] = False
            validation_result["confidence"] = 0.0
            validation_result["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    def _validate_phonology(self, tokens: List[Token]) -> float:
        """Validate phonological correctness."""
        score = 1.0
        
        # Check for invalid character sequences
        for token in tokens:
            if not self._is_valid_sanskrit_sequence(token.text):
                score -= 0.1
        
        return max(0.0, score)
    
    def _validate_morphology(self, text: str) -> float:
        """Validate morphological correctness."""
        try:
            analysis = self.morphological_analyzer.analyze_word(text)
            return analysis.confidence
        except:
            return 0.5  # Partial score if analysis fails
    
    def _validate_syntax(self, tokens: List[Token]) -> float:
        """Validate syntactic correctness."""
        # Simplified syntax validation
        # In a full implementation, this would check word order, agreement, etc.
        return 0.8  # Placeholder score
    
    def _validate_sandhi(self, tokens: List[Token]) -> float:
        """Validate sandhi correctness."""
        try:
            # Process through rule engine to check if sandhi is correctly applied
            result = self.rule_engine.process(tokens)
            
            # If processing converged without errors, sandhi is likely correct
            if result.converged and not result.errors:
                return 1.0
            else:
                return 0.6  # Partial score
        except:
            return 0.4  # Low score if processing fails
    
    def _is_valid_sanskrit_sequence(self, text: str) -> bool:
        """Check if text contains valid Sanskrit character sequences."""
        # Define valid Sanskrit characters
        valid_chars = set('aāiīuūṛṝḷḹeēoōṃḥkgṅcjñṭḍṇtdnpbmyrlvśṣsh ')
        
        # Check if all characters are valid
        return all(c.lower() in valid_chars for c in text)


class SanskritReasoningMetrics:
    """
    Evaluation metrics for Sanskrit reasoning tasks.
    
    Provides comprehensive metrics for evaluating the performance of
    Sanskrit reasoning systems in the R-Zero framework.
    """
    
    def __init__(self, validator: SanskritGrammaticalValidator):
        """
        Initialize the metrics calculator.
        
        Args:
            validator: Sanskrit grammatical validator
        """
        self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    def evaluate_solution(self, 
                         problem: SanskritProblem, 
                         solution: str,
                         reference_solution: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a solution to a Sanskrit reasoning problem.
        
        Args:
            problem: The Sanskrit problem
            solution: The proposed solution
            reference_solution: Optional reference solution for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Grammatical correctness
        validation_result = self.validator.validate_sanskrit_text(solution)
        metrics["grammatical_correctness"] = validation_result["confidence"]
        
        # Exact match with expected output
        if reference_solution is None:
            reference_solution = problem.expected_output
        
        metrics["exact_match"] = 1.0 if solution.strip() == reference_solution.strip() else 0.0
        
        # Semantic similarity (simplified)
        metrics["semantic_similarity"] = self._calculate_semantic_similarity(solution, reference_solution)
        
        # Problem-specific metrics
        problem_specific_metrics = self._calculate_problem_specific_metrics(problem, solution)
        metrics.update(problem_specific_metrics)
        
        # Overall score
        weights = {
            "grammatical_correctness": 0.3,
            "exact_match": 0.4,
            "semantic_similarity": 0.3
        }
        
        overall_score = sum(metrics.get(key, 0.0) * weight for key, weight in weights.items())
        metrics["overall_score"] = overall_score
        
        return metrics
    
    def _calculate_semantic_similarity(self, solution: str, reference: str) -> float:
        """Calculate semantic similarity between solution and reference."""
        # Simplified semantic similarity based on word overlap
        solution_words = set(solution.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 1.0 if not solution_words else 0.0
        
        intersection = solution_words.intersection(reference_words)
        union = solution_words.union(reference_words)
        
        return len(intersection) / len(union) if union else 1.0
    
    def _calculate_problem_specific_metrics(self, problem: SanskritProblem, solution: str) -> Dict[str, float]:
        """Calculate metrics specific to the problem type."""
        metrics = {}
        
        if problem.type == SanskritProblemType.SANDHI_APPLICATION:
            metrics["sandhi_accuracy"] = self._evaluate_sandhi_solution(problem, solution)
        
        elif problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
            metrics["morphology_accuracy"] = self._evaluate_morphology_solution(problem, solution)
        
        elif problem.type == SanskritProblemType.WORD_DERIVATION:
            metrics["derivation_accuracy"] = self._evaluate_derivation_solution(problem, solution)
        
        elif problem.type == SanskritProblemType.COMPOUND_ANALYSIS:
            metrics["compound_accuracy"] = self._evaluate_compound_solution(problem, solution)
        
        return metrics
    
    def _evaluate_sandhi_solution(self, problem: SanskritProblem, solution: str) -> float:
        """Evaluate sandhi application solution."""
        # Check if the solution contains the expected sandhi result
        expected = problem.expected_output.lower()
        solution_lower = solution.lower()
        
        if expected in solution_lower:
            return 1.0
        else:
            # Partial credit for containing relevant Sanskrit terms
            return 0.3 if any(char in 'aāiīuūeēoō' for char in solution_lower) else 0.0
    
    def _evaluate_morphology_solution(self, problem: SanskritProblem, solution: str) -> float:
        """Evaluate morphological analysis solution."""
        # Check for presence of morphological terms
        morphology_terms = ['dhātu', 'pratyaya', 'root', 'suffix', 'stem']
        solution_lower = solution.lower()
        
        term_count = sum(1 for term in morphology_terms if term in solution_lower)
        return min(1.0, term_count / 3.0)  # Normalize to max 1.0
    
    def _evaluate_derivation_solution(self, problem: SanskritProblem, solution: str) -> float:
        """Evaluate word derivation solution."""
        # Check for derivation indicators
        derivation_indicators = ['→', '+', 'step', 'rule', 'sūtra']
        solution_lower = solution.lower()
        
        indicator_count = sum(1 for indicator in derivation_indicators if indicator in solution_lower)
        return min(1.0, indicator_count / 2.0)  # Normalize to max 1.0
    
    def _evaluate_compound_solution(self, problem: SanskritProblem, solution: str) -> float:
        """Evaluate compound analysis solution."""
        # Check for compound analysis terms
        compound_terms = ['tatpuruṣa', 'karmadhāraya', 'dvandva', 'bahuvrīhi', 'compound', 'samāsa']
        solution_lower = solution.lower()
        
        term_count = sum(1 for term in compound_terms if term in solution_lower)
        return min(1.0, term_count / 2.0)  # Normalize to max 1.0


def compute_sanskrit_score(predicts: List[str], 
                          ground_truths: List[str], 
                          problems: List[SanskritProblem],
                          validator: SanskritGrammaticalValidator,
                          format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute Sanskrit reasoning scores compatible with R-Zero reward function format.
    
    Args:
        predicts: List of predicted solutions
        ground_truths: List of ground truth solutions
        problems: List of Sanskrit problems
        validator: Sanskrit grammatical validator
        format_weight: Weight for format scoring
        
    Returns:
        List of score dictionaries compatible with R-Zero
    """
    metrics_calculator = SanskritReasoningMetrics(validator)
    scores = []
    
    for predict, ground_truth, problem in zip(predicts, ground_truths, problems):
        # Calculate comprehensive metrics
        evaluation_metrics = metrics_calculator.evaluate_solution(problem, predict, ground_truth)
        
        # Format score (check if solution follows expected format)
        format_score = 1.0 if _check_sanskrit_format(predict, problem) else 0.0
        
        # Accuracy score (grammatical correctness + semantic accuracy)
        accuracy_score = (evaluation_metrics["grammatical_correctness"] + 
                         evaluation_metrics["semantic_similarity"]) / 2.0
        
        # Overall score
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "grammatical_correctness": evaluation_metrics["grammatical_correctness"],
            "semantic_similarity": evaluation_metrics["semantic_similarity"],
            "exact_match": evaluation_metrics["exact_match"]
        })
    
    return scores


def _check_sanskrit_format(solution: str, problem: SanskritProblem) -> bool:
    """Check if solution follows expected format for the problem type."""
    solution_lower = solution.lower()
    
    # Basic format checks based on problem type
    if problem.type == SanskritProblemType.SANDHI_APPLICATION:
        return '→' in solution or 'becomes' in solution_lower or 'result' in solution_lower
    
    elif problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS:
        return '+' in solution or 'root' in solution_lower or 'suffix' in solution_lower
    
    elif problem.type == SanskritProblemType.WORD_DERIVATION:
        return '→' in solution or 'step' in solution_lower or 'derivation' in solution_lower
    
    elif problem.type == SanskritProblemType.COMPOUND_ANALYSIS:
        return 'type:' in solution_lower or 'compound' in solution_lower or 'parts:' in solution_lower
    
    # Default: check if solution is not empty and contains Sanskrit-like content
    return len(solution.strip()) > 0 and any(c in 'aāiīuūeēoōṃḥ' for c in solution)