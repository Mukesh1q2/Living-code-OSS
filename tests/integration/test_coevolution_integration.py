#!/usr/bin/env python3
"""
Simple integration test for Sanskrit Co-evolutionary Loop.

This test verifies the basic functionality without requiring external dependencies
like vllm or transformers.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock external dependencies
sys.modules['vllm'] = Mock()
sys.modules['transformers'] = Mock()

from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem, SanskritSolution, SanskritProblemType, SanskritDifficultyLevel
)
from sanskrit_rewrite_engine.sanskrit_coevolution import (
    CoevolutionConfig, IterationResult, SanskritCoevolutionLoop
)


def test_basic_data_structures():
    """Test basic data structure creation and manipulation."""
    print("Testing basic data structures...")
    
    # Test SanskritProblem creation
    problem = SanskritProblem(
        id="test_001",
        type=SanskritProblemType.SANDHI_APPLICATION,
        difficulty=SanskritDifficultyLevel.INTERMEDIATE,
        input_text="a + i",
        expected_output="e",
        sutra_references=["6.1.87"],
        explanation="Vowel sandhi transformation"
    )
    
    assert problem.id == "test_001"
    assert problem.type == SanskritProblemType.SANDHI_APPLICATION
    assert problem.input_text == "a + i"
    assert problem.expected_output == "e"
    
    # Test R-Zero format conversion
    r_zero_format = problem.to_r_zero_format()
    assert "problem" in r_zero_format
    assert "answer" in r_zero_format
    assert "metadata" in r_zero_format
    
    print("✓ Basic data structures work correctly")


def test_solution_creation():
    """Test solution creation and validation."""
    print("Testing solution creation...")
    
    solution = SanskritSolution(
        problem_id="test_001",
        solution_text="e",
        confidence=0.85,
        reasoning_steps=["Apply sandhi rule", "a + i → e"],
        rule_applications=["6.1.87"],
        metadata={"test": True}
    )
    
    assert solution.problem_id == "test_001"
    assert solution.solution_text == "e"
    assert solution.confidence == 0.85
    assert len(solution.reasoning_steps) == 2
    
    print("✓ Solution creation works correctly")


def test_coevolution_config():
    """Test co-evolution configuration."""
    print("Testing co-evolution configuration...")
    
    config = CoevolutionConfig(
        max_iterations=10,
        problems_per_iteration=5,
        convergence_threshold=0.9,
        save_frequency=2
    )
    
    assert config.max_iterations == 10
    assert config.problems_per_iteration == 5
    assert config.convergence_threshold == 0.9
    
    print("✓ Co-evolution configuration works correctly")


def test_iteration_result():
    """Test iteration result tracking."""
    print("Testing iteration result tracking...")
    
    result = IterationResult(
        iteration=1,
        problems_generated=5,
        problems_solved=5,
        solver_performance={"overall": 0.7, "SANDHI_APPLICATION": 0.8},
        challenger_effectiveness=0.6,
        convergence_metrics={"improvement": 0.1}
    )
    
    assert result.iteration == 1
    assert result.problems_generated == 5
    assert result.solver_performance["overall"] == 0.7
    assert result.challenger_effectiveness == 0.6
    
    print("✓ Iteration result tracking works correctly")


def test_coevolution_loop_initialization():
    """Test co-evolution loop initialization with mocks."""
    print("Testing co-evolution loop initialization...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock challenger and solver
        mock_challenger = Mock()
        mock_solver = Mock()
        
        config = CoevolutionConfig(max_iterations=5, problems_per_iteration=3)
        
        # Initialize loop
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_dir
        )
        
        assert loop.config == config
        assert loop.challenger == mock_challenger
        assert loop.solver == mock_solver
        assert loop.storage_path == Path(temp_dir)
        
        # Check that directories were created
        assert (Path(temp_dir) / "experiments").exists()
        assert (Path(temp_dir) / "traces").exists()
        
        print("✓ Co-evolution loop initialization works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def test_performance_tracking():
    """Test performance tracking functionality."""
    print("Testing performance tracking...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        mock_challenger = Mock()
        mock_solver = Mock()
        config = CoevolutionConfig()
        
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_dir
        )
        
        # Test performance history update
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
        
        print("✓ Performance tracking works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def test_convergence_detection():
    """Test convergence detection logic."""
    print("Testing convergence detection...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        mock_challenger = Mock()
        mock_solver = Mock()
        config = CoevolutionConfig(
            convergence_threshold=0.9,
            performance_window=3
        )
        
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_dir
        )
        
        # Add high-performance results
        for i in range(3):
            result = IterationResult(
                iteration=i + 1,
                problems_generated=3,
                problems_solved=3,
                solver_performance={"overall": 0.95},  # Above threshold
                challenger_effectiveness=0.8,
                convergence_metrics={}
            )
            loop.iteration_results.append(result)
        
        # Should detect convergence
        assert loop._check_convergence() == True
        
        # Test with low performance
        loop.iteration_results.clear()
        for i in range(3):
            result = IterationResult(
                iteration=i + 1,
                problems_generated=3,
                problems_solved=3,
                solver_performance={"overall": 0.5},  # Below threshold
                challenger_effectiveness=0.8,
                convergence_metrics={}
            )
            loop.iteration_results.append(result)
        
        # Should not detect convergence
        assert loop._check_convergence() == False
        
        print("✓ Convergence detection works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def test_data_persistence():
    """Test data saving and loading."""
    print("Testing data persistence...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        mock_challenger = Mock()
        mock_solver = Mock()
        config = CoevolutionConfig()
        
        loop = SanskritCoevolutionLoop(
            config=config,
            challenger=mock_challenger,
            solver=mock_solver,
            storage_path=temp_dir
        )
        
        # Create sample data
        problems = [
            SanskritProblem(
                id="test_001",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="a + i",
                expected_output="e"
            )
        ]
        
        solutions = [
            SanskritSolution(
                problem_id="test_001",
                solution_text="e",
                confidence=0.8
            )
        ]
        
        # Save iteration data
        loop._save_iteration_data(1, problems, solutions)
        
        # Check that files were created
        iteration_dir = Path(temp_dir) / "experiments" / "iteration_001"
        assert iteration_dir.exists()
        assert (iteration_dir / "problems.json").exists()
        assert (iteration_dir / "solutions.json").exists()
        
        # Verify data can be loaded
        import json
        with open(iteration_dir / "problems.json", 'r') as f:
            loaded_problems = json.load(f)
        
        assert len(loaded_problems) == 1
        assert loaded_problems[0]["metadata"]["id"] == "test_001"
        
        print("✓ Data persistence works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all integration tests."""
    print("Running Sanskrit Co-evolutionary Loop Integration Tests")
    print("=" * 60)
    
    try:
        test_basic_data_structures()
        test_solution_creation()
        test_coevolution_config()
        test_iteration_result()
        test_coevolution_loop_initialization()
        test_performance_tracking()
        test_convergence_detection()
        test_data_persistence()
        
        print("=" * 60)
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)