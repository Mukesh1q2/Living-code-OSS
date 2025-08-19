"""
Tests for Sanskrit Co-evolutionary Loop.

This module provides comprehensive tests for the Sanskrit Challenger-Solver
co-evolutionary training loop.
"""

import pytest
import json
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Mock vllm module before importing our modules
sys.modules['vllm'] = Mock()
sys.modules['vllm.SamplingParams'] = Mock()
sys.modules['vllm.LLM'] = Mock()

from sanskrit_rewrite_engine.sanskrit_challenger import SanskritChallenger, ChallengerConfig
from sanskrit_rewrite_engine.sanskrit_solver import SanskritSolver, SolverConfig
from sanskrit_rewrite_engine.sanskrit_coevolution import (
    SanskritCoevolutionLoop, CoevolutionConfig, IterationResult,
    run_sanskrit_coevolution, test_coevolution_convergence
)
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem, SanskritSolution, SanskritProblemType, SanskritDifficultyLevel
)


class TestSanskritChallenger:
    """Test cases for Sanskrit Challenger model."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.chat_template = True
        tokenizer.apply_chat_template = Mock(return_value="mocked_prompt")
        return tokenizer
    
    @pytest.fixture
    def mock_model(self):
        """Mock VLLM model for testing."""
        model = Mock()
        
        # Mock completion output
        mock_output = Mock()
        mock_output.text = """<problem_type>SANDHI_APPLICATION</problem_type>
<difficulty>INTERMEDIATE</difficulty>
<input>a + i</input>
<expected_output>e</expected_output>
<sutra_references>6.1.87</sutra_references>
<explanation>Vowel sandhi: a + i → e</explanation>"""
        
        mock_completion = Mock()
        mock_completion.outputs = [mock_output]
        
        model.generate = Mock(return_value=[mock_completion])
        return model
    
    @pytest.fixture
    def mock_problem_generator(self):
        """Mock problem generator for testing."""
        generator = Mock()
        
        # Create sample problem
        sample_problem = SanskritProblem(
            id="test_problem_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.INTERMEDIATE,
            input_text="a + i",
            expected_output="e",
            sutra_references=["6.1.87"],
            explanation="Vowel sandhi transformation"
        )
        
        generator.generate_problems = Mock(return_value=[sample_problem])
        return generator
    
    @pytest.fixture
    def temp_storage(self):
        """Temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def challenger_config(self):
        """Challenger configuration for testing."""
        return ChallengerConfig(
            model_path="test_model",
            num_samples=2,
            max_tokens=512,
            seed=42
        )
    
    def test_challenger_initialization(self, challenger_config, mock_problem_generator, temp_storage):
        """Test Challenger initialization."""
        with patch('sanskrit_rewrite_engine.sanskrit_challenger.AutoTokenizer') as mock_tokenizer_class, \
             patch('sanskrit_rewrite_engine.sanskrit_challenger.vllm.LLM') as mock_model_class:
            
            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_model_class.return_value = Mock()
            
            challenger = SanskritChallenger(
                config=challenger_config,
                problem_generator=mock_problem_generator,
                storage_path=temp_storage
            )
            
            assert challenger.config == challenger_config
            assert challenger.problem_generator == mock_problem_generator
            assert challenger.storage_path == Path(temp_storage)
    
    @patch('sanskrit_rewrite_engine.sanskrit_challenger.vllm.LLM')
    @patch('sanskrit_rewrite_engine.sanskrit_challenger.AutoTokenizer')
    def test_problem_generation(self, mock_tokenizer_class, mock_model_class, 
                               challenger_config, mock_problem_generator, temp_storage):
        """Test problem generation functionality."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer()
        mock_model_class.return_value = self.mock_model()
        
        challenger = SanskritChallenger(
            config=challenger_config,
            problem_generator=mock_problem_generator,
            storage_path=temp_storage
        )
        
        # Generate problems
        problems = challenger.generate_problems(
            problem_type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.INTERMEDIATE,
            num_problems=2
        )
        
        assert len(problems) >= 1
        assert all(isinstance(p, SanskritProblem) for p in problems)
        assert all(p.type == SanskritProblemType.SANDHI_APPLICATION for p in problems)
    
    @patch('sanskrit_rewrite_engine.sanskrit_challenger.vllm.LLM')
    @patch('sanskrit_rewrite_engine.sanskrit_challenger.AutoTokenizer')
    def test_adaptive_problem_generation(self, mock_tokenizer_class, mock_model_class,
                                       challenger_config, mock_problem_generator, temp_storage):
        """Test adaptive problem generation based on solver performance."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer()
        mock_model_class.return_value = self.mock_model()
        
        challenger = SanskritChallenger(
            config=challenger_config,
            problem_generator=mock_problem_generator,
            storage_path=temp_storage
        )
        
        # Mock solver performance (low performance in sandhi)
        solver_performance = {
            "SANDHI_APPLICATION": 0.3,  # Low performance
            "MORPHOLOGICAL_ANALYSIS": 0.8,  # High performance
        }
        
        problems = challenger.generate_adaptive_problems(
            solver_performance=solver_performance,
            num_problems=5
        )
        
        assert len(problems) >= 1
        # Should generate more problems for weak areas
        sandhi_problems = [p for p in problems if p.type == SanskritProblemType.SANDHI_APPLICATION]
        assert len(sandhi_problems) > 0


class TestSanskritSolver:
    """Test cases for Sanskrit Solver model."""
    
    @pytest.fixture
    def solver_config(self):
        """Solver configuration for testing."""
        return SolverConfig(
            model_path="test_model",
            num_samples=2,
            max_tokens=512,
            seed=42
        )
    
    @pytest.fixture
    def mock_validator(self):
        """Mock validator for testing."""
        validator = Mock()
        validator.validate_sanskrit_text = Mock(return_value={
            "confidence": 0.8,
            "is_valid": True,
            "errors": []
        })
        return validator
    
    @pytest.fixture
    def sample_problems(self):
        """Sample problems for testing."""
        return [
            SanskritProblem(
                id="test_problem_001",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="a + i",
                expected_output="e",
                sutra_references=["6.1.87"]
            ),
            SanskritProblem(
                id="test_problem_002",
                type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text="gacchati",
                expected_output="gam(ROOT) + ti(SUFFIX)",
                sutra_references=["3.4.78"]
            )
        ]
    
    @patch('sanskrit_rewrite_engine.sanskrit_solver.vllm.LLM')
    @patch('sanskrit_rewrite_engine.sanskrit_solver.AutoTokenizer')
    def test_solver_initialization(self, mock_tokenizer_class, mock_model_class,
                                  solver_config, mock_validator, temp_storage):
        """Test Solver initialization."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.return_value = Mock()
        
        solver = SanskritSolver(
            config=solver_config,
            validator=mock_validator,
            storage_path=temp_storage
        )
        
        assert solver.config == solver_config
        assert solver.validator == mock_validator
    
    @patch('sanskrit_rewrite_engine.sanskrit_solver.vllm.LLM')
    @patch('sanskrit_rewrite_engine.sanskrit_solver.AutoTokenizer')
    def test_problem_solving(self, mock_tokenizer_class, mock_model_class,
                           solver_config, mock_validator, sample_problems, temp_storage):
        """Test problem solving functionality."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = True
        mock_tokenizer.apply_chat_template = Mock(return_value="mocked_prompt")
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model response
        mock_output = Mock()
        mock_output.text = "Step 1: Apply sandhi rule\nStep 2: a + i → e\n\\boxed{e}"
        
        mock_completion = Mock()
        mock_completion.outputs = [mock_output, mock_output]  # Multiple samples
        
        mock_model = Mock()
        mock_model.generate = Mock(return_value=[mock_completion])
        mock_model_class.return_value = mock_model
        
        solver = SanskritSolver(
            config=solver_config,
            validator=mock_validator,
            storage_path=temp_storage
        )
        
        # Solve problems
        solutions = solver.solve_problems(sample_problems[:1])  # Test with one problem
        
        assert len(solutions) == 1
        assert isinstance(solutions[0], SanskritSolution)
        assert solutions[0].problem_id == sample_problems[0].id
        assert solutions[0].solution_text == "e"
    
    @patch('sanskrit_rewrite_engine.sanskrit_solver.vllm.LLM')
    @patch('sanskrit_rewrite_engine.sanskrit_solver.AutoTokenizer')
    def test_performance_evaluation(self, mock_tokenizer_class, mock_model_class,
                                   solver_config, mock_validator, sample_problems, temp_storage):
        """Test performance evaluation."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.return_value = Mock()
        
        solver = SanskritSolver(
            config=solver_config,
            validator=mock_validator,
            storage_path=temp_storage
        )
        
        # Create mock solutions
        solutions = [
            SanskritSolution(
                problem_id="test_problem_001",
                solution_text="e",
                confidence=0.9
            ),
            SanskritSolution(
                problem_id="test_problem_002",
                solution_text="gam(ROOT) + ti(SUFFIX)",
                confidence=0.8
            )
        ]
        
        performance = solver.evaluate_performance(sample_problems, solutions)
        
        assert isinstance(performance, dict)
        assert "overall" in performance
        assert performance["overall"] > 0.0


class TestCoevolutionLoop:
    """Test cases for the co-evolutionary loop."""
    
    @pytest.fixture
    def coevolution_config(self):
        """Co-evolution configuration for testing."""
        return CoevolutionConfig(
            max_iterations=5,
            problems_per_iteration=3,
            convergence_threshold=0.9,
            save_frequency=2
        )
    
    @pytest.fixture
    def mock_challenger(self):
        """Mock challenger for testing."""
        challenger = Mock()
        
        # Mock problem generation
        sample_problem = SanskritProblem(
            id="challenger_problem_001",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.INTERMEDIATE,
            input_text="a + i",
            expected_output="e"
        )
        
        challenger.generate_problems = Mock(return_value=[sample_problem])
        challenger.generate_adaptive_problems = Mock(return_value=[sample_problem])
        
        return challenger
    
    @pytest.fixture
    def mock_solver(self):
        """Mock solver for testing."""
        solver = Mock()
        
        # Mock problem solving
        sample_solution = SanskritSolution(
            problem_id="challenger_problem_001",
            solution_text="e",
            confidence=0.8
        )
        
        solver.solve_problems = Mock(return_value=[sample_solution])
        solver.evaluate_performance = Mock(return_value={
            "SANDHI_APPLICATION": 0.8,
            "overall": 0.8
        })
        
        return solver
    
    def test_coevolution_initialization(self, coevolution_config, mock_challenger, 
                                      mock_solver, temp_storage):
        """Test co-evolution loop initialization."""
        loop = SanskritCoevolutionLoop(
            config=coevolution_config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        assert loop.config == coevolution_config
        assert loop.challenger == mock_challenger
        assert loop.solver == mock_solver
        assert loop.storage_path == Path(temp_storage)
    
    def test_single_iteration(self, coevolution_config, mock_challenger, 
                             mock_solver, temp_storage):
        """Test a single iteration of the co-evolutionary loop."""
        loop = SanskritCoevolutionLoop(
            config=coevolution_config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        result = loop._run_single_iteration(1)
        
        assert isinstance(result, IterationResult)
        assert result.iteration == 1
        assert result.problems_generated > 0
        assert result.problems_solved > 0
        assert isinstance(result.solver_performance, dict)
        assert result.challenger_effectiveness >= 0.0
    
    def test_convergence_detection(self, coevolution_config, mock_challenger, 
                                  mock_solver, temp_storage):
        """Test convergence detection."""
        loop = SanskritCoevolutionLoop(
            config=coevolution_config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        # Add high-performance results to simulate convergence
        for i in range(coevolution_config.performance_window):
            result = IterationResult(
                iteration=i + 1,
                problems_generated=3,
                problems_solved=3,
                solver_performance={"overall": 0.95},  # High performance
                challenger_effectiveness=0.8,
                convergence_metrics={"current_performance": 0.95}
            )
            loop.iteration_results.append(result)
        
        assert loop._check_convergence() == True
    
    def test_performance_tracking(self, coevolution_config, mock_challenger, 
                                 mock_solver, temp_storage):
        """Test performance history tracking."""
        loop = SanskritCoevolutionLoop(
            config=coevolution_config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        result = IterationResult(
            iteration=1,
            problems_generated=3,
            problems_solved=3,
            solver_performance={"SANDHI_APPLICATION": 0.7, "overall": 0.75},
            challenger_effectiveness=0.6,
            convergence_metrics={}
        )
        
        loop._update_performance_history(result)
        
        assert "SANDHI_APPLICATION" in loop.performance_history
        assert "overall" in loop.performance_history
        assert loop.performance_history["SANDHI_APPLICATION"] == [0.7]
        assert loop.performance_history["overall"] == [0.75]
    
    @patch('sanskrit_rewrite_engine.sanskrit_coevolution.create_sanskrit_challenger')
    @patch('sanskrit_rewrite_engine.sanskrit_coevolution.create_sanskrit_solver')
    def test_full_coevolution_run(self, mock_create_solver, mock_create_challenger, temp_storage):
        """Test full co-evolutionary training run."""
        # Setup mocks
        mock_create_challenger.return_value = self.mock_challenger()
        mock_create_solver.return_value = self.mock_solver()
        
        config = CoevolutionConfig(
            max_iterations=3,
            problems_per_iteration=2,
            convergence_threshold=0.95
        )
        
        # This should complete without errors
        results = run_sanskrit_coevolution(coevolution_config=config)
        
        assert isinstance(results, dict)
        assert "experiment_id" in results
        assert "total_iterations" in results
        assert "final_performance" in results


class TestCoevolutionIntegration:
    """Integration tests for the complete co-evolutionary system."""
    
    def test_convergence_simulation(self, temp_storage):
        """Test convergence behavior with simulated improvement."""
        # This test simulates the co-evolutionary process
        # In practice, this would require actual models
        
        config = CoevolutionConfig(
            max_iterations=10,
            problems_per_iteration=5,
            convergence_threshold=0.9
        )
        
        # Mock gradual improvement
        performance_trajectory = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]
        
        mock_challenger = Mock()
        mock_solver = Mock()
        
        # Simulate improving performance
        def mock_evaluate_performance(problems, solutions):
            iteration = len(mock_solver.evaluate_performance.call_args_list)
            if iteration < len(performance_trajectory):
                perf = performance_trajectory[iteration]
                return {"overall": perf, "SANDHI_APPLICATION": perf}
            return {"overall": 0.95, "SANDHI_APPLICATION": 0.95}
        
        mock_solver.evaluate_performance = Mock(side_effect=mock_evaluate_performance)
        mock_solver.solve_problems = Mock(return_value=[
            SanskritSolution(problem_id="test", solution_text="test", confidence=0.8)
        ])
        
        mock_challenger.generate_problems = Mock(return_value=[
            SanskritProblem(
                id="test", type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="test", expected_output="test"
            )
        ])
        mock_challenger.generate_adaptive_problems = Mock(return_value=[
            SanskritProblem(
                id="test", type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="test", expected_output="test"
            )
        ])
        
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        results = loop.run_coevolution()
        
        assert results["converged"] == True
        assert results["total_iterations"] <= config.max_iterations
        assert results["final_performance"]["solver_performance"]["overall"] >= config.convergence_threshold
    
    def test_data_persistence(self, temp_storage):
        """Test that training data is properly saved and can be loaded."""
        config = CoevolutionConfig(
            max_iterations=2,
            problems_per_iteration=2,
            save_frequency=1
        )
        
        # Create minimal mocks
        mock_challenger = Mock()
        mock_solver = Mock()
        
        sample_problem = SanskritProblem(
            id="test_problem",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.INTERMEDIATE,
            input_text="a + i",
            expected_output="e"
        )
        
        sample_solution = SanskritSolution(
            problem_id="test_problem",
            solution_text="e",
            confidence=0.8
        )
        
        mock_challenger.generate_problems = Mock(return_value=[sample_problem])
        mock_challenger.generate_adaptive_problems = Mock(return_value=[sample_problem])
        mock_solver.solve_problems = Mock(return_value=[sample_solution])
        mock_solver.evaluate_performance = Mock(return_value={"overall": 0.7})
        
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_storage
        )
        
        results = loop.run_coevolution()
        
        # Check that files were created
        experiments_dir = Path(temp_storage) / "experiments"
        assert experiments_dir.exists()
        
        # Check for iteration directories
        iteration_dirs = list(experiments_dir.glob("iteration_*"))
        assert len(iteration_dirs) >= 1
        
        # Check for final results file
        final_results_files = list(experiments_dir.glob("final_results_*.json"))
        assert len(final_results_files) == 1
        
        # Verify final results can be loaded
        with open(final_results_files[0], 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results["experiment_id"] == results["experiment_id"]
        assert loaded_results["total_iterations"] == results["total_iterations"]


# Utility functions for testing
def create_test_problems(count: int = 5) -> List[SanskritProblem]:
    """Create test problems for testing."""
    problems = []
    problem_types = list(SanskritProblemType)
    difficulties = list(SanskritDifficultyLevel)
    
    for i in range(count):
        problem_type = problem_types[i % len(problem_types)]
        difficulty = difficulties[i % len(difficulties)]
        
        problem = SanskritProblem(
            id=f"test_problem_{i:03d}",
            type=problem_type,
            difficulty=difficulty,
            input_text=f"test_input_{i}",
            expected_output=f"test_output_{i}",
            sutra_references=[f"{i}.1.{i}"],
            explanation=f"Test problem {i}"
        )
        problems.append(problem)
    
    return problems


def create_test_solutions(problems: List[SanskritProblem]) -> List[SanskritSolution]:
    """Create test solutions for given problems."""
    solutions = []
    
    for problem in problems:
        solution = SanskritSolution(
            problem_id=problem.id,
            solution_text=problem.expected_output,
            confidence=0.8,
            reasoning_steps=[f"Step 1: Analyze {problem.input_text}"],
            metadata={"test_solution": True}
        )
        solutions.append(solution)
    
    return solutions


# Fixtures for temporary storage
@pytest.fixture
def temp_storage():
    """Temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])