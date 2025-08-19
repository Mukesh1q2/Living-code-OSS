"""
Test suite for Sanskrit R-Zero integration.

This module provides comprehensive tests for Sanskrit reasoning data quality,
format compliance, and integration with the R-Zero framework.
"""

import pytest
import json
import tempfile
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Import R-Zero integration components
from sanskrit_rewrite_engine.r_zero_config import SanskritRZeroConfig, RZeroEnvironmentSetup
from sanskrit_rewrite_engine.verl_integration import SanskritVERLIntegrator, check_verl_availability
from sanskrit_rewrite_engine.sanskrit_reward_function import (
    SanskritRewardCalculator, compute_sanskrit_score, RewardComponent
)

# Import existing components
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem, SanskritSolution, SanskritProblemType, SanskritDifficultyLevel,
    SanskritProblemGenerator, SanskritGrammaticalValidator, SanskritReasoningMetrics,
    compute_sanskrit_score
)
from sanskrit_rewrite_engine.r_zero_data_format import (
    SanskritRZeroDataConverter, RZeroDataset, create_r_zero_training_data,
    save_dataset_for_r_zero
)
from sanskrit_rewrite_engine.rl_environment import (
    SanskritRLEnvironment, SanskritAction, ActionType, SanskritState,
    SanskritRLTrainer
)
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine
from sanskrit_rewrite_engine.morphological_analyzer import SanskritMorphologicalAnalyzer
from sanskrit_rewrite_engine.derivation_simulator import ShabdaPrakriyaSimulator


class TestRZeroConfiguration:
    """Test R-Zero configuration setup."""
    
    def test_sanskrit_r_zero_config_creation(self):
        """Test creation of Sanskrit R-Zero configuration."""
        config = SanskritRZeroConfig()
        
        assert config.storage_path == "./r_zero_storage"
        assert config.model_path == "./r_zero_models"
        assert config.max_sanskrit_length == 512
        assert config.grammatical_weight == 0.4
        assert config.semantic_weight == 0.3
        assert config.efficiency_weight == 0.3
    
    def test_config_serialization(self):
        """Test configuration serialization to/from YAML."""
        config = SanskritRZeroConfig(
            storage_path="./test_storage",
            learning_rate=2e-6,
            batch_size=64
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save and load
            config.save_to_yaml(temp_file)
            loaded_config = SanskritRZeroConfig.load_from_yaml(temp_file)
            
            assert loaded_config.storage_path == "./test_storage"
            assert loaded_config.learning_rate == 2e-6
            assert loaded_config.batch_size == 64
            
        finally:
            os.unlink(temp_file)
    
    def test_environment_setup(self):
        """Test R-Zero environment setup."""
        config = SanskritRZeroConfig(storage_path="./test_r_zero_storage")
        setup = RZeroEnvironmentSetup(config)
        
        # Test environment setup
        setup.setup_environment()
        
        # Check environment variables
        assert os.environ.get('STORAGE_PATH') == "./test_r_zero_storage"
        assert os.environ.get('MODEL_PATH') == "./r_zero_models"
        
        # Check directory creation
        assert os.path.exists("./test_r_zero_storage")
        assert os.path.exists("./test_r_zero_storage/logs")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_r_zero_storage"):
            shutil.rmtree("./test_r_zero_storage")


class TestVERLIntegration:
    """Test VERL integration components."""
    
    @pytest.fixture
    def verl_integrator(self):
        """Create VERL integrator for testing."""
        return SanskritVERLIntegrator(storage_path="./test_verl_storage")
    
    def test_verl_availability_check(self):
        """Test VERL availability checking."""
        availability = check_verl_availability()
        
        assert 'r_zero_paths_exist' in availability
        assert 'verl_importable' in availability
        assert 'components_available' in availability
        assert isinstance(availability['r_zero_paths_exist'], list)
    
    def test_verl_integrator_initialization(self, verl_integrator):
        """Test VERL integrator initialization."""
        assert verl_integrator.storage_path.exists()
        assert isinstance(verl_integrator.is_verl_available(), bool)
        
        # Cleanup
        import shutil
        if os.path.exists("./test_verl_storage"):
            shutil.rmtree("./test_verl_storage")
    
    def test_sanskrit_training_environment_setup(self, verl_integrator):
        """Test Sanskrit training environment setup."""
        config = {
            'model_path': 'Qwen/Qwen2.5-7B-Instruct',
            'max_sanskrit_length': 256,
            'batch_size': 16,
            'learning_rate': 1e-6
        }
        
        environment = verl_integrator.setup_sanskrit_training_environment(config)
        
        assert 'trainer' in environment
        assert 'rollout_worker' in environment
        assert 'actor_model' in environment
        assert 'critic_model' in environment
        assert 'reward_function' in environment
        assert 'config' in environment
        
        # Cleanup
        import shutil
        if os.path.exists("./test_verl_storage"):
            shutil.rmtree("./test_verl_storage")
    
    def test_training_step_execution(self, verl_integrator):
        """Test training step execution."""
        config = {'batch_size': 4, 'learning_rate': 1e-6}
        environment = verl_integrator.setup_sanskrit_training_environment(config)
        
        training_data = [
            {'problem': 'a + i', 'answer': 'e'},
            {'problem': 'a + u', 'answer': 'o'},
            {'problem': 'gacchati', 'answer': 'gam + ti'},
            {'problem': 'rāma + asya', 'answer': 'rāmasya'}
        ]
        
        results = verl_integrator.run_sanskrit_training_step(environment, training_data)
        
        assert 'training_loss' in results
        assert 'average_reward' in results
        assert 'rollout_count' in results
        assert 'step_successful' in results
        assert results['rollout_count'] == 4
        
        # Cleanup
        import shutil
        if os.path.exists("./test_verl_storage"):
            shutil.rmtree("./test_verl_storage")


class TestSanskritRewardFunction:
    """Test Sanskrit-specific reward functions."""
    
    @pytest.fixture
    def reward_calculator(self):
        """Create reward calculator for testing."""
        return SanskritRewardCalculator()
    
    def test_reward_calculation_basic(self, reward_calculator):
        """Test basic reward calculation."""
        prediction = "e"
        ground_truth = "e"
        
        reward = reward_calculator.calculate_reward(prediction, ground_truth)
        
        assert reward.overall_score > 0.8  # Should be high for exact match
        assert RewardComponent.FORMAT_COMPLIANCE.value in reward.components
        assert RewardComponent.GRAMMATICAL_CORRECTNESS.value in reward.components
        assert RewardComponent.SEMANTIC_CONSISTENCY.value in reward.components
        assert 'exact_match' in reward.details
    
    def test_sandhi_specific_reward(self, reward_calculator):
        """Test sandhi-specific reward calculation."""
        problem = SanskritProblem(
            id="sandhi_test",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        # Correct answer
        reward_correct = reward_calculator.calculate_reward("e", "e", problem)
        assert RewardComponent.SANDHI_ACCURACY.value in reward_correct.components
        assert reward_correct.components[RewardComponent.SANDHI_ACCURACY.value] > 0.9
        
        # Incorrect answer
        reward_incorrect = reward_calculator.calculate_reward("o", "e", problem)
        assert reward_incorrect.components[RewardComponent.SANDHI_ACCURACY.value] < 0.5
    
    def test_morphological_specific_reward(self, reward_calculator):
        """Test morphological analysis specific reward."""
        problem = SanskritProblem(
            id="morph_test",
            type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            difficulty=SanskritDifficultyLevel.INTERMEDIATE,
            input_text="gacchati",
            expected_output="gam(root) + ti(suffix)"
        )
        
        # Good morphological analysis
        prediction = "gam(root) + ti(suffix)"
        reward = reward_calculator.calculate_reward(prediction, "gam(root) + ti(suffix)", problem)
        
        assert RewardComponent.MORPHOLOGICAL_ACCURACY.value in reward.components
        assert reward.components[RewardComponent.MORPHOLOGICAL_ACCURACY.value] > 0.8
    
    def test_format_compliance_checking(self, reward_calculator):
        """Test format compliance checking."""
        # Test with Sanskrit characters
        sanskrit_text = "गच्छति"
        score = reward_calculator._check_format_compliance(sanskrit_text, None)
        assert score > 0.3  # Should get points for Sanskrit characters
        
        # Test with IAST
        iast_text = "gacchati"
        score = reward_calculator._check_format_compliance(iast_text, None)
        assert score > 0.0
        
        # Test empty text
        empty_score = reward_calculator._check_format_compliance("", None)
        assert empty_score == 0.0
    
    def test_compute_sanskrit_score_function(self):
        """Test the main compute_sanskrit_score function."""
        predicts = ["e", "o", "gam + ti"]
        ground_truths = ["e", "o", "gam + ti"]
        problems = [
            SanskritProblem(
                id=f"test_{i}",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text=f"input_{i}",
                expected_output=gt
            )
            for i, gt in enumerate(ground_truths)
        ]
        
        # Create mock validator
        mock_validator = Mock()
        mock_validator.validate_sanskrit_text.return_value = {
            'confidence': 0.9,
            'is_valid': True
        }
        
        scores = compute_sanskrit_score(predicts, ground_truths, problems, mock_validator)
        
        assert len(scores) == 3
        for score in scores:
            assert 'overall' in score
            assert 'format' in score
            assert 'accuracy' in score
            assert score['overall'] > 0.5  # All are correct matches


class TestSanskritProblemGeneration:
    """Test Sanskrit problem generation for R-Zero."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        tokenizer = Mock(spec=SanskritTokenizer)
        rule_engine = Mock(spec=PaniniRuleEngine)
        morphological_analyzer = Mock(spec=SanskritMorphologicalAnalyzer)
        derivation_simulator = Mock(spec=ShabdaPrakriyaSimulator)
        
        return {
            'tokenizer': tokenizer,
            'rule_engine': rule_engine,
            'morphological_analyzer': morphological_analyzer,
            'derivation_simulator': derivation_simulator
        }
    
    @pytest.fixture
    def problem_generator(self, mock_components):
        """Create a Sanskrit problem generator."""
        return SanskritProblemGenerator(
            tokenizer=mock_components['tokenizer'],
            rule_engine=mock_components['rule_engine'],
            morphological_analyzer=mock_components['morphological_analyzer'],
            derivation_simulator=mock_components['derivation_simulator']
        )
    
    def test_problem_generation_basic(self, problem_generator):
        """Test basic problem generation."""
        problems = problem_generator.generate_problems(count=10)
        
        assert len(problems) == 10
        assert all(isinstance(p, SanskritProblem) for p in problems)
        assert all(p.id.startswith('sanskrit_') for p in problems)
        assert all(p.type in SanskritProblemType for p in problems)
        assert all(p.difficulty in SanskritDifficultyLevel for p in problems)
    
    def test_problem_type_distribution(self, problem_generator):
        """Test that problem types are distributed correctly."""
        problem_types = [SanskritProblemType.SANDHI_APPLICATION, SanskritProblemType.MORPHOLOGICAL_ANALYSIS]
        problems = problem_generator.generate_problems(
            count=20,
            problem_types=problem_types
        )
        
        assert len(problems) == 20
        assert all(p.type in problem_types for p in problems)
    
    def test_difficulty_distribution(self, problem_generator):
        """Test difficulty distribution."""
        difficulty_dist = {
            SanskritDifficultyLevel.BEGINNER: 0.5,
            SanskritDifficultyLevel.ADVANCED: 0.5
        }
        
        problems = problem_generator.generate_problems(
            count=100,
            difficulty_distribution=difficulty_dist
        )
        
        beginner_count = sum(1 for p in problems if p.difficulty == SanskritDifficultyLevel.BEGINNER)
        advanced_count = sum(1 for p in problems if p.difficulty == SanskritDifficultyLevel.ADVANCED)
        
        # Allow some variance in distribution
        assert 40 <= beginner_count <= 60
        assert 40 <= advanced_count <= 60
    
    def test_sandhi_problem_generation(self, problem_generator):
        """Test sandhi problem generation."""
        problem = problem_generator._generate_sandhi_problem(
            "test_sandhi_001",
            SanskritDifficultyLevel.BEGINNER
        )
        
        assert problem.type == SanskritProblemType.SANDHI_APPLICATION
        assert problem.difficulty == SanskritDifficultyLevel.BEGINNER
        assert '+' in problem.input_text
        assert len(problem.expected_output) > 0
        assert len(problem.sutra_references) > 0
    
    def test_morphology_problem_generation(self, problem_generator):
        """Test morphological analysis problem generation."""
        # Mock the morphological analyzer
        mock_analysis = Mock()
        mock_analysis.morphemes = []
        mock_analysis.grammatical_categories = set()
        mock_analysis.confidence = 0.9
        
        problem_generator.morphological_analyzer.analyze_word.return_value = mock_analysis
        
        problem = problem_generator._generate_morphology_problem(
            "test_morph_001",
            SanskritDifficultyLevel.INTERMEDIATE
        )
        
        assert problem.type == SanskritProblemType.MORPHOLOGICAL_ANALYSIS
        assert problem.difficulty == SanskritDifficultyLevel.INTERMEDIATE
        assert len(problem.input_text) > 0
        assert len(problem.expected_output) > 0


class TestSanskritGrammaticalValidator:
    """Test Sanskrit grammatical validation."""
    
    @pytest.fixture
    def mock_validator_components(self):
        """Create mock components for validator."""
        rule_engine = Mock(spec=PaniniRuleEngine)
        morphological_analyzer = Mock(spec=SanskritMorphologicalAnalyzer)
        
        # Mock tokenizer
        tokenizer = Mock()
        rule_engine.tokenizer = tokenizer
        
        return rule_engine, morphological_analyzer
    
    @pytest.fixture
    def validator(self, mock_validator_components):
        """Create a Sanskrit grammatical validator."""
        rule_engine, morphological_analyzer = mock_validator_components
        return SanskritGrammaticalValidator(rule_engine, morphological_analyzer)
    
    def test_validate_sanskrit_text_valid(self, validator):
        """Test validation of valid Sanskrit text."""
        # Mock successful validation
        validator.rule_engine.tokenizer.tokenize.return_value = []
        
        mock_analysis = Mock()
        mock_analysis.confidence = 0.9
        validator.morphological_analyzer.analyze_word.return_value = mock_analysis
        
        mock_result = Mock()
        mock_result.converged = True
        mock_result.errors = []
        validator.rule_engine.process.return_value = mock_result
        
        result = validator.validate_sanskrit_text("gacchati")
        
        assert result['is_valid'] is True
        assert result['confidence'] > 0.7
        assert len(result['errors']) == 0
    
    def test_validate_sanskrit_text_invalid(self, validator):
        """Test validation of invalid Sanskrit text."""
        # Mock failed validation
        validator.rule_engine.tokenizer.tokenize.return_value = []
        
        mock_analysis = Mock()
        mock_analysis.confidence = 0.3
        validator.morphological_analyzer.analyze_word.return_value = mock_analysis
        
        mock_result = Mock()
        mock_result.converged = False
        mock_result.errors = ['Invalid sandhi']
        validator.rule_engine.process.return_value = mock_result
        
        result = validator.validate_sanskrit_text("invalid_text")
        
        assert result['is_valid'] is False
        assert result['confidence'] < 0.7
    
    def test_phonological_validation(self, validator):
        """Test phonological validation."""
        from sanskrit_rewrite_engine.token import Token, TokenKind
        
        # Valid Sanskrit tokens
        valid_tokens = [Token("ga", TokenKind.CONSONANT), Token("ccha", TokenKind.CONSONANT)]
        score = validator._validate_phonology(valid_tokens)
        assert score > 0.8
        
        # Invalid tokens
        invalid_tokens = [Token("xyz", TokenKind.OTHER)]
        score = validator._validate_phonology(invalid_tokens)
        assert score < 1.0


class TestSanskritReasoningMetrics:
    """Test Sanskrit reasoning evaluation metrics."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator."""
        validator = Mock(spec=SanskritGrammaticalValidator)
        validator.validate_sanskrit_text.return_value = {
            'confidence': 0.9,
            'is_valid': True
        }
        return validator
    
    @pytest.fixture
    def metrics_calculator(self, mock_validator):
        """Create a metrics calculator."""
        return SanskritReasoningMetrics(mock_validator)
    
    def test_evaluate_solution_exact_match(self, metrics_calculator):
        """Test solution evaluation with exact match."""
        problem = SanskritProblem(
            id="test_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        solution = "e"
        metrics = metrics_calculator.evaluate_solution(problem, solution)
        
        assert metrics['exact_match'] == 1.0
        assert metrics['grammatical_correctness'] == 0.9
        assert metrics['overall_score'] > 0.8
    
    def test_evaluate_solution_no_match(self, metrics_calculator):
        """Test solution evaluation with no match."""
        problem = SanskritProblem(
            id="test_002",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        solution = "wrong_answer"
        metrics = metrics_calculator.evaluate_solution(problem, solution)
        
        assert metrics['exact_match'] == 0.0
        assert 'overall_score' in metrics
    
    def test_problem_specific_metrics(self, metrics_calculator):
        """Test problem-specific metrics calculation."""
        # Test sandhi problem
        sandhi_problem = SanskritProblem(
            id="sandhi_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        metrics = metrics_calculator._calculate_problem_specific_metrics(sandhi_problem, "e")
        assert 'sandhi_accuracy' in metrics
        
        # Test morphology problem
        morph_problem = SanskritProblem(
            id="morph_001",
            type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="gacchati",
            expected_output="gam(ROOT) + ti(SUFFIX)"
        )
        
        metrics = metrics_calculator._calculate_problem_specific_metrics(
            morph_problem, "gam(ROOT) + ti(SUFFIX)"
        )
        assert 'morphology_accuracy' in metrics


class TestRZeroDataFormat:
    """Test R-Zero data format conversion."""
    
    @pytest.fixture
    def mock_problem_generator(self):
        """Create a mock problem generator."""
        generator = Mock(spec=SanskritProblemGenerator)
        
        # Mock problem generation
        sample_problem = SanskritProblem(
            id="test_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e",
            sutra_references=["6.1.87"],
            explanation="Vowel sandhi"
        )
        
        generator._generate_problem_by_type.return_value = sample_problem
        return generator
    
    @pytest.fixture
    def data_converter(self, mock_problem_generator):
        """Create a data converter."""
        return SanskritRZeroDataConverter(mock_problem_generator)
    
    def test_convert_problem_to_r_zero(self, data_converter):
        """Test conversion of problem to R-Zero format."""
        problem = SanskritProblem(
            id="test_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e",
            sutra_references=["6.1.87"],
            explanation="Vowel sandhi"
        )
        
        r_zero_format = data_converter.convert_problem_to_r_zero(problem)
        
        assert 'problem' in r_zero_format
        assert 'answer' in r_zero_format
        assert 'metadata' in r_zero_format
        assert r_zero_format['answer'] == "e"
        assert r_zero_format['metadata']['id'] == "test_001"
        assert r_zero_format['metadata']['type'] == "SANDHI_APPLICATION"
    
    def test_create_training_dataset(self, data_converter):
        """Test creation of training dataset."""
        dataset = data_converter.create_training_dataset(num_problems=10)
        
        assert isinstance(dataset, RZeroDataset)
        assert len(dataset.problems) == 10
        assert 'dataset_type' in dataset.metadata
        assert dataset.metadata['dataset_type'] == 'sanskrit_reasoning_training'
        assert 'total_problems' in dataset.statistics
    
    def test_create_evaluation_dataset(self, data_converter):
        """Test creation of evaluation dataset."""
        dataset = data_converter.create_evaluation_dataset(num_problems=5)
        
        assert isinstance(dataset, RZeroDataset)
        assert len(dataset.problems) == 5
        assert dataset.metadata['dataset_type'] == 'sanskrit_reasoning_evaluation'
    
    def test_dataset_save_load(self, data_converter):
        """Test dataset save and load functionality."""
        dataset = data_converter.create_training_dataset(num_problems=3)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Save dataset
            dataset.save_to_json(temp_filename)
            
            # Load dataset
            loaded_dataset = RZeroDataset.load_from_json(temp_filename)
            
            assert len(loaded_dataset.problems) == len(dataset.problems)
            assert loaded_dataset.metadata == dataset.metadata
            
        finally:
            os.unlink(temp_filename)
    
    def test_save_dataset_for_r_zero(self, data_converter):
        """Test saving dataset with train/validation split."""
        dataset = data_converter.create_training_dataset(num_problems=10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_filename = os.path.join(temp_dir, "test_dataset")
            
            train_file, val_file = save_dataset_for_r_zero(dataset, base_filename, split_ratio=0.8)
            
            # Check files exist
            assert os.path.exists(train_file)
            assert os.path.exists(val_file)
            
            # Load and check splits
            train_dataset = RZeroDataset.load_from_json(train_file)
            val_dataset = RZeroDataset.load_from_json(val_file)
            
            assert len(train_dataset.problems) == 8
            assert len(val_dataset.problems) == 2
            assert train_dataset.metadata['split'] == 'train'
            assert val_dataset.metadata['split'] == 'validation'


class TestSanskritRLEnvironment:
    """Test Sanskrit reinforcement learning environment."""
    
    @pytest.fixture
    def mock_rl_components(self):
        """Create mock components for RL environment."""
        rule_engine = Mock(spec=PaniniRuleEngine)
        validator = Mock(spec=SanskritGrammaticalValidator)
        
        # Mock tokenizer
        tokenizer = Mock()
        rule_engine.tokenizer = tokenizer
        
        # Mock registry
        registry = Mock()
        rule_engine.registry = registry
        
        return rule_engine, validator
    
    @pytest.fixture
    def rl_environment(self, mock_rl_components):
        """Create an RL environment."""
        rule_engine, validator = mock_rl_components
        return SanskritRLEnvironment(rule_engine, validator, max_steps=10)
    
    def test_environment_reset(self, rl_environment):
        """Test environment reset."""
        from sanskrit_rewrite_engine.token import Token, TokenKind
        
        # Mock tokenization
        tokens = [Token("a", TokenKind.VOWEL), Token("i", TokenKind.VOWEL)]
        rl_environment.rule_engine.tokenizer.tokenize.return_value = tokens
        rl_environment.rule_engine.registry.get_active_sutra_rules.return_value = []
        
        problem = SanskritProblem(
            id="test_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        state = rl_environment.reset(problem)
        
        assert isinstance(state, SanskritState)
        assert len(state.tokens) == 2
        assert state.step_count == 0
        assert state.target_output == "e"
    
    def test_action_execution(self, rl_environment):
        """Test action execution in environment."""
        from sanskrit_rewrite_engine.token import Token, TokenKind
        
        # Setup environment
        tokens = [Token("a", TokenKind.VOWEL)]
        rl_environment.rule_engine.tokenizer.tokenize.return_value = tokens
        rl_environment.rule_engine.registry.get_active_sutra_rules.return_value = []
        
        state = rl_environment.reset()
        
        # Test SKIP action
        action = SanskritAction(type=ActionType.SKIP)
        next_state, reward, done, info = rl_environment.step(action)
        
        assert isinstance(next_state, SanskritState)
        assert next_state.step_count == 1
        assert 'skip_action' in reward.components
    
    def test_termination_conditions(self, rl_environment):
        """Test environment termination conditions."""
        from sanskrit_rewrite_engine.token import Token, TokenKind
        
        # Setup environment
        tokens = [Token("a", TokenKind.VOWEL)]
        rl_environment.rule_engine.tokenizer.tokenize.return_value = tokens
        rl_environment.rule_engine.registry.get_active_sutra_rules.return_value = []
        
        state = rl_environment.reset()
        
        # Test TERMINATE action
        action = SanskritAction(type=ActionType.TERMINATE)
        next_state, reward, done, info = rl_environment.step(action)
        
        assert done is True
        assert 'final_grammatical_score' in reward.components
    
    def test_state_vector_conversion(self, rl_environment):
        """Test state to vector conversion."""
        from sanskrit_rewrite_engine.token import Token, TokenKind
        
        tokens = [Token("a", TokenKind.VOWEL), Token("i", TokenKind.VOWEL)]
        state = SanskritState(tokens=tokens, available_rules=[], step_count=5)
        
        state_vector = state.get_state_vector()
        
        assert isinstance(state_vector, type(state_vector))  # numpy array
        assert len(state_vector) > 0


class TestSanskritScoreComputation:
    """Test Sanskrit score computation for R-Zero."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator."""
        validator = Mock(spec=SanskritGrammaticalValidator)
        validator.validate_sanskrit_text.return_value = {
            'confidence': 0.9,
            'is_valid': True
        }
        return validator
    
    def test_compute_sanskrit_score_basic(self, mock_validator):
        """Test basic Sanskrit score computation."""
        predicts = ["e", "o", "ya"]
        ground_truths = ["e", "o", "va"]
        problems = [
            SanskritProblem(
                id=f"test_{i}",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text=f"input_{i}",
                expected_output=gt
            )
            for i, gt in enumerate(ground_truths)
        ]
        
        scores = compute_sanskrit_score(predicts, ground_truths, problems, mock_validator)
        
        assert len(scores) == 3
        assert all('overall' in score for score in scores)
        assert all('format' in score for score in scores)
        assert all('accuracy' in score for score in scores)
        
        # First two should have higher scores (exact matches)
        assert scores[0]['overall'] > scores[2]['overall']
        assert scores[1]['overall'] > scores[2]['overall']
    
    def test_format_checking(self, mock_validator):
        """Test format checking for different problem types."""
        from sanskrit_rewrite_engine.r_zero_integration import _check_sanskrit_format
        
        # Test sandhi format
        sandhi_problem = SanskritProblem(
            id="sandhi_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        assert _check_sanskrit_format("a + i → e", sandhi_problem) is True
        assert _check_sanskrit_format("result is e", sandhi_problem) is True
        assert _check_sanskrit_format("xyz", sandhi_problem) is False
        
        # Test morphology format
        morph_problem = SanskritProblem(
            id="morph_001",
            type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="gacchati",
            expected_output="gam + ti"
        )
        
        assert _check_sanskrit_format("gam(root) + ti(suffix)", morph_problem) is True
        assert _check_sanskrit_format("root: gam", morph_problem) is True
        assert _check_sanskrit_format("", morph_problem) is False


class TestDataQualityAndCompliance:
    """Test data quality and format compliance."""
    
    def test_problem_id_uniqueness(self):
        """Test that generated problems have unique IDs."""
        from sanskrit_rewrite_engine.r_zero_integration import SanskritProblemGenerator
        
        # Mock components
        tokenizer = Mock()
        rule_engine = Mock()
        morphological_analyzer = Mock()
        derivation_simulator = Mock()
        
        generator = SanskritProblemGenerator(
            tokenizer, rule_engine, morphological_analyzer, derivation_simulator
        )
        
        problems = generator.generate_problems(count=50)
        problem_ids = [p.id for p in problems]
        
        assert len(problem_ids) == len(set(problem_ids)), "Problem IDs are not unique"
    
    def test_problem_completeness(self):
        """Test that all problems have required fields."""
        from sanskrit_rewrite_engine.r_zero_integration import SanskritProblemGenerator
        
        # Mock components
        tokenizer = Mock()
        rule_engine = Mock()
        morphological_analyzer = Mock()
        derivation_simulator = Mock()
        
        generator = SanskritProblemGenerator(
            tokenizer, rule_engine, morphological_analyzer, derivation_simulator
        )
        
        problems = generator.generate_problems(count=10)
        
        for problem in problems:
            assert problem.id is not None and len(problem.id) > 0
            assert problem.type in SanskritProblemType
            assert problem.difficulty in SanskritDifficultyLevel
            assert problem.input_text is not None and len(problem.input_text) > 0
            assert problem.expected_output is not None and len(problem.expected_output) > 0
    
    def test_r_zero_format_compliance(self):
        """Test that converted problems comply with R-Zero format."""
        problem = SanskritProblem(
            id="test_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e",
            sutra_references=["6.1.87"],
            explanation="Vowel sandhi"
        )
        
        r_zero_format = problem.to_r_zero_format()
        
        # Check required R-Zero fields
        assert 'problem' in r_zero_format
        assert 'answer' in r_zero_format
        assert 'metadata' in r_zero_format
        
        # Check field types
        assert isinstance(r_zero_format['problem'], str)
        assert isinstance(r_zero_format['answer'], str)
        assert isinstance(r_zero_format['metadata'], dict)
        
        # Check metadata completeness
        metadata = r_zero_format['metadata']
        assert 'id' in metadata
        assert 'type' in metadata
        assert 'difficulty' in metadata
    
    def test_dataset_statistics_accuracy(self):
        """Test that dataset statistics are calculated correctly."""
        from sanskrit_rewrite_engine.r_zero_data_format import SanskritRZeroDataConverter
        
        # Mock problem generator
        generator = Mock()
        converter = SanskritRZeroDataConverter(generator)
        
        # Create test problems
        problems = [
            SanskritProblem(
                id=f"test_{i}",
                type=SanskritProblemType.SANDHI_APPLICATION if i < 3 else SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
                difficulty=SanskritDifficultyLevel.BEGINNER if i < 2 else SanskritDifficultyLevel.ADVANCED,
                input_text=f"input_{i}",
                expected_output=f"output_{i}",
                sutra_references=["6.1.87"] if i % 2 == 0 else []
            )
            for i in range(5)
        ]
        
        stats = converter._calculate_dataset_statistics(problems)
        
        assert stats['total_problems'] == 5
        assert stats['type_distribution']['SANDHI_APPLICATION'] == 3
        assert stats['type_distribution']['MORPHOLOGICAL_ANALYSIS'] == 2
        assert stats['difficulty_distribution']['BEGINNER'] == 2
        assert stats['difficulty_distribution']['ADVANCED'] == 3
        assert stats['unique_sutra_references'] == 1  # Only "6.1.87"
        assert stats['total_sutra_references'] == 3  # 3 problems have sutra references


# Integration tests
class TestEndToEndIntegration:
    """Test end-to-end integration with R-Zero."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from problem generation to R-Zero format."""
        # Mock all components
        tokenizer = Mock()
        rule_engine = Mock()
        morphological_analyzer = Mock()
        derivation_simulator = Mock()
        validator = Mock()
        
        # Setup mocks
        rule_engine.tokenizer = tokenizer
        validator.validate_sanskrit_text.return_value = {'confidence': 0.9, 'is_valid': True}
        
        # Create pipeline components
        from sanskrit_rewrite_engine.r_zero_integration import SanskritProblemGenerator
        from sanskrit_rewrite_engine.r_zero_data_format import SanskritRZeroDataConverter
        
        generator = SanskritProblemGenerator(
            tokenizer, rule_engine, morphological_analyzer, derivation_simulator
        )
        converter = SanskritRZeroDataConverter(generator)
        
        # Generate and convert data
        dataset = converter.create_training_dataset(num_problems=5)
        
        # Verify pipeline output
        assert len(dataset.problems) == 5
        assert all('problem' in p for p in dataset.problems)
        assert all('answer' in p for p in dataset.problems)
        assert 'dataset_type' in dataset.metadata
        assert 'total_problems' in dataset.statistics
    
    def test_reward_function_integration(self):
        """Test integration with R-Zero reward function."""
        from sanskrit_rewrite_engine.r_zero_integration import compute_sanskrit_score
        
        # Mock validator
        validator = Mock()
        validator.validate_sanskrit_text.return_value = {'confidence': 0.8, 'is_valid': True}
        
        # Test data
        predicts = ["e", "wrong"]
        ground_truths = ["e", "correct"]
        problems = [
            SanskritProblem(
                id="test_1",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text="a + i",
                expected_output="e"
            ),
            SanskritProblem(
                id="test_2",
                type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="gacchati",
                expected_output="correct"
            )
        ]
        
        scores = compute_sanskrit_score(predicts, ground_truths, problems, validator)
        
        # Verify reward function format compatibility
        assert len(scores) == 2
        for score in scores:
            assert 'overall' in score
            assert 'format' in score
            assert 'accuracy' in score
            assert isinstance(score['overall'], (int, float))
            assert 0.0 <= score['overall'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])