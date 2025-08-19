"""
Sanskrit Co-evolutionary Loop for R-Zero Framework.

This module implements the co-evolutionary training loop between the Sanskrit
Challenger and Solver models, enabling continuous improvement through adversarial
problem generation and solving.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .sanskrit_challenger import SanskritChallenger, create_sanskrit_challenger
from .sanskrit_solver import SanskritSolver, create_sanskrit_solver
from .r_zero_integration import SanskritProblem, SanskritSolution, SanskritProblemType, SanskritDifficultyLevel
from .r_zero_config import SanskritRZeroConfig
from .sanskrit_reward_function import SanskritRewardCalculator


logger = logging.getLogger(__name__)


@dataclass
class CoevolutionConfig:
    """Configuration for co-evolutionary training."""
    max_iterations: int = 100
    problems_per_iteration: int = 20
    convergence_threshold: float = 0.95
    performance_window: int = 5
    adaptation_rate: float = 0.1
    save_frequency: int = 10
    
    # Problem generation parameters
    difficulty_adaptation: bool = True
    type_balancing: bool = True
    
    # Evaluation parameters
    evaluation_frequency: int = 5
    evaluation_problems: int = 50


@dataclass
class IterationResult:
    """Results from a single co-evolution iteration."""
    iteration: int
    problems_generated: int
    problems_solved: int
    solver_performance: Dict[str, float]
    challenger_effectiveness: float
    convergence_metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SanskritCoevolutionLoop:
    """
    Co-evolutionary training loop for Sanskrit reasoning.
    
    Manages the interaction between Challenger (problem generator) and Solver
    (problem solver) models to continuously improve Sanskrit reasoning capabilities.
    """
    
    def __init__(self, 
                 config: CoevolutionConfig,
                 challenger: SanskritChallenger,
                 solver: SanskritSolver,
                 storage_path: str = "./r_zero_storage"):
        """
        Initialize the co-evolutionary loop.
        
        Args:
            config: Co-evolution configuration
            challenger: Sanskrit Challenger model
            solver: Sanskrit Solver model
            storage_path: Path for storing results
        """
        self.config = config
        self.challenger = challenger
        self.solver = solver
        self.storage_path = Path(storage_path)
        
        # Create directories
        self.experiments_path = self.storage_path / "experiments"
        self.traces_path = self.storage_path / "traces"
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        self.traces_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.iteration_results: List[IterationResult] = []
        self.performance_history: Dict[str, List[float]] = {}
        self.reward_calculator = SanskritRewardCalculator()
        
        # Experiment metadata
        self.experiment_id = f"sanskrit_coevolution_{int(time.time())}"
        self.start_time = datetime.now()
        
        logger.info(f"Initialized co-evolutionary loop: {self.experiment_id}")
    
    def run_coevolution(self) -> Dict[str, Any]:
        """
        Run the complete co-evolutionary training loop.
        
        Returns:
            Final results and statistics
        """
        logger.info(f"Starting co-evolutionary training for {self.config.max_iterations} iterations")
        
        try:
            for iteration in range(1, self.config.max_iterations + 1):
                logger.info(f"Starting iteration {iteration}/{self.config.max_iterations}")
                
                # Run single iteration
                result = self._run_single_iteration(iteration)
                self.iteration_results.append(result)
                
                # Update performance history
                self._update_performance_history(result)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Convergence achieved at iteration {iteration}")
                    break
                
                # Save intermediate results
                if iteration % self.config.save_frequency == 0:
                    self._save_intermediate_results(iteration)
                
                # Log progress
                self._log_iteration_progress(result)
            
            # Save final results
            final_results = self._compile_final_results()
            self._save_final_results(final_results)
            
            logger.info("Co-evolutionary training completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Co-evolutionary training failed: {e}")
            raise
    
    def _run_single_iteration(self, iteration: int) -> IterationResult:
        """Run a single iteration of the co-evolutionary loop."""
        
        # 1. Generate problems using Challenger
        logger.info(f"Iteration {iteration}: Generating problems")
        
        if iteration == 1:
            # First iteration: generate balanced problems
            problems = self._generate_initial_problems()
        else:
            # Adaptive problem generation based on Solver performance
            solver_performance = self._get_recent_performance()
            problems = self.challenger.generate_adaptive_problems(
                solver_performance=solver_performance,
                num_problems=self.config.problems_per_iteration
            )
        
        logger.info(f"Generated {len(problems)} problems")
        
        # 2. Solve problems using Solver
        logger.info(f"Iteration {iteration}: Solving problems")
        solutions = self.solver.solve_problems(problems)
        logger.info(f"Generated {len(solutions)} solutions")
        
        # 3. Evaluate Solver performance
        solver_performance = self.solver.evaluate_performance(problems, solutions)
        
        # 4. Evaluate Challenger effectiveness
        challenger_effectiveness = self._evaluate_challenger_effectiveness(problems, solutions)
        
        # 5. Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(solver_performance)
        
        # 6. Save iteration data
        self._save_iteration_data(iteration, problems, solutions)
        
        return IterationResult(
            iteration=iteration,
            problems_generated=len(problems),
            problems_solved=len(solutions),
            solver_performance=solver_performance,
            challenger_effectiveness=challenger_effectiveness,
            convergence_metrics=convergence_metrics
        )
    
    def _generate_initial_problems(self) -> List[SanskritProblem]:
        """Generate initial balanced set of problems."""
        problems = []
        problem_types = list(SanskritProblemType)
        difficulties = list(SanskritDifficultyLevel)
        
        problems_per_type = max(1, self.config.problems_per_iteration // len(problem_types))
        
        for problem_type in problem_types:
            for difficulty in difficulties:
                type_problems = self.challenger.generate_problems(
                    problem_type=problem_type,
                    difficulty=difficulty,
                    num_problems=max(1, problems_per_type // len(difficulties))
                )
                problems.extend(type_problems)
        
        return problems[:self.config.problems_per_iteration]
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent Solver performance for adaptive problem generation."""
        if not self.iteration_results:
            return {}
        
        # Use performance from last few iterations
        recent_results = self.iteration_results[-self.config.performance_window:]
        
        # Average performance across recent iterations
        performance_sums = {}
        performance_counts = {}
        
        for result in recent_results:
            for problem_type, performance in result.solver_performance.items():
                if problem_type not in performance_sums:
                    performance_sums[problem_type] = 0.0
                    performance_counts[problem_type] = 0
                
                performance_sums[problem_type] += performance
                performance_counts[problem_type] += 1
        
        # Calculate averages
        avg_performance = {}
        for problem_type in performance_sums:
            avg_performance[problem_type] = performance_sums[problem_type] / performance_counts[problem_type]
        
        return avg_performance
    
    def _evaluate_challenger_effectiveness(self, 
                                         problems: List[SanskritProblem],
                                         solutions: List[SanskritSolution]) -> float:
        """Evaluate how effectively the Challenger is creating challenging problems."""
        if not problems or not solutions:
            return 0.0
        
        # Challenger is effective if it creates problems that are:
        # 1. Solvable but challenging (not too easy, not impossible)
        # 2. Diverse in difficulty and type
        # 3. Push the Solver to improve
        
        effectiveness_scores = []
        
        for problem, solution in zip(problems, solutions):
            # Calculate problem difficulty vs solution quality
            if problem.expected_output:
                reward = self.reward_calculator.calculate_reward(
                    solution.solution_text, problem.expected_output, problem
                )
                solution_quality = reward.overall_score
            else:
                solution_quality = solution.confidence
            
            # Ideal range: challenging but solvable (0.3 - 0.8)
            if 0.3 <= solution_quality <= 0.8:
                effectiveness_scores.append(1.0)
            elif solution_quality < 0.3:
                # Too difficult
                effectiveness_scores.append(0.5)
            else:
                # Too easy
                effectiveness_scores.append(0.7)
        
        return sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
    
    def _calculate_convergence_metrics(self, solver_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate metrics to assess convergence."""
        metrics = {}
        
        # Overall performance trend
        if 'overall' in solver_performance:
            metrics['current_performance'] = solver_performance['overall']
        
        # Performance stability (variance in recent iterations)
        if len(self.iteration_results) >= 3:
            recent_performances = [r.solver_performance.get('overall', 0.0) 
                                 for r in self.iteration_results[-3:]]
            variance = sum((p - sum(recent_performances)/len(recent_performances))**2 
                          for p in recent_performances) / len(recent_performances)
            metrics['performance_stability'] = 1.0 / (1.0 + variance)  # Higher is more stable
        
        # Improvement rate
        if len(self.iteration_results) >= 2:
            current_perf = solver_performance.get('overall', 0.0)
            previous_perf = self.iteration_results[-1].solver_performance.get('overall', 0.0)
            metrics['improvement_rate'] = max(0.0, current_perf - previous_perf)
        
        return metrics
    
    def _update_performance_history(self, result: IterationResult):
        """Update performance history tracking."""
        for problem_type, performance in result.solver_performance.items():
            if problem_type not in self.performance_history:
                self.performance_history[problem_type] = []
            self.performance_history[problem_type].append(performance)
    
    def _check_convergence(self) -> bool:
        """Check if the co-evolutionary loop has converged."""
        if len(self.iteration_results) < self.config.performance_window:
            return False
        
        # Check if performance has stabilized at high level
        recent_results = self.iteration_results[-self.config.performance_window:]
        recent_performances = [r.solver_performance.get('overall', 0.0) for r in recent_results]
        
        # All recent performances above threshold
        if all(p >= self.config.convergence_threshold for p in recent_performances):
            return True
        
        # Performance variance is low (stable)
        avg_performance = sum(recent_performances) / len(recent_performances)
        variance = sum((p - avg_performance)**2 for p in recent_performances) / len(recent_performances)
        
        if avg_performance >= 0.85 and variance < 0.01:  # High performance, low variance
            return True
        
        return False
    
    def _save_iteration_data(self, 
                           iteration: int,
                           problems: List[SanskritProblem],
                           solutions: List[SanskritSolution]):
        """Save data from a single iteration."""
        iteration_dir = self.experiments_path / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Save problems
        problems_data = [p.to_r_zero_format() for p in problems]
        with open(iteration_dir / "problems.json", 'w', encoding='utf-8') as f:
            json.dump(problems_data, f, indent=2, ensure_ascii=False)
        
        # Save solutions
        solutions_data = []
        for solution in solutions:
            solutions_data.append({
                "problem_id": solution.problem_id,
                "solution_text": solution.solution_text,
                "confidence": solution.confidence,
                "reasoning_steps": solution.reasoning_steps,
                "metadata": solution.metadata
            })
        
        with open(iteration_dir / "solutions.json", 'w', encoding='utf-8') as f:
            json.dump(solutions_data, f, indent=2, ensure_ascii=False)
    
    def _save_intermediate_results(self, iteration: int):
        """Save intermediate results during training."""
        results_file = self.traces_path / f"coevolution_trace_{self.experiment_id}.json"
        
        trace_data = {
            "experiment_id": self.experiment_id,
            "current_iteration": iteration,
            "config": {
                "max_iterations": self.config.max_iterations,
                "problems_per_iteration": self.config.problems_per_iteration,
                "convergence_threshold": self.config.convergence_threshold
            },
            "iteration_results": [
                {
                    "iteration": r.iteration,
                    "problems_generated": r.problems_generated,
                    "problems_solved": r.problems_solved,
                    "solver_performance": r.solver_performance,
                    "challenger_effectiveness": r.challenger_effectiveness,
                    "convergence_metrics": r.convergence_metrics,
                    "timestamp": r.timestamp
                }
                for r in self.iteration_results
            ],
            "performance_history": self.performance_history
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved intermediate results to {results_file}")
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile final results from the co-evolutionary training."""
        if not self.iteration_results:
            return {}
        
        final_result = self.iteration_results[-1]
        
        # Calculate overall statistics
        all_performances = [r.solver_performance.get('overall', 0.0) for r in self.iteration_results]
        all_effectiveness = [r.challenger_effectiveness for r in self.iteration_results]
        
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_iterations": len(self.iteration_results),
            "converged": self._check_convergence(),
            
            "final_performance": {
                "solver_performance": final_result.solver_performance,
                "challenger_effectiveness": final_result.challenger_effectiveness,
                "convergence_metrics": final_result.convergence_metrics
            },
            
            "training_statistics": {
                "total_problems_generated": sum(r.problems_generated for r in self.iteration_results),
                "total_problems_solved": sum(r.problems_solved for r in self.iteration_results),
                "average_solver_performance": sum(all_performances) / len(all_performances),
                "final_solver_performance": all_performances[-1],
                "performance_improvement": all_performances[-1] - all_performances[0] if len(all_performances) > 1 else 0.0,
                "average_challenger_effectiveness": sum(all_effectiveness) / len(all_effectiveness),
            },
            
            "performance_by_type": {
                problem_type: self.performance_history[problem_type][-1] if self.performance_history.get(problem_type) else 0.0
                for problem_type in SanskritProblemType
            },
            
            "config": {
                "max_iterations": self.config.max_iterations,
                "problems_per_iteration": self.config.problems_per_iteration,
                "convergence_threshold": self.config.convergence_threshold
            }
        }
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results to file."""
        results_file = self.experiments_path / f"final_results_{self.experiment_id}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final results to {results_file}")
    
    def _log_iteration_progress(self, result: IterationResult):
        """Log progress for a single iteration."""
        logger.info(f"Iteration {result.iteration} completed:")
        logger.info(f"  Problems generated: {result.problems_generated}")
        logger.info(f"  Problems solved: {result.problems_solved}")
        logger.info(f"  Solver performance: {result.solver_performance.get('overall', 0.0):.3f}")
        logger.info(f"  Challenger effectiveness: {result.challenger_effectiveness:.3f}")
        
        if result.convergence_metrics:
            logger.info(f"  Convergence metrics: {result.convergence_metrics}")


def run_sanskrit_coevolution(config_path: Optional[str] = None,
                           coevolution_config: Optional[CoevolutionConfig] = None) -> Dict[str, Any]:
    """
    Run Sanskrit co-evolutionary training.
    
    Args:
        config_path: Path to R-Zero configuration file
        coevolution_config: Co-evolution specific configuration
        
    Returns:
        Final training results
    """
    logger.info("Starting Sanskrit co-evolutionary training")
    
    # Load configurations
    if coevolution_config is None:
        coevolution_config = CoevolutionConfig()
    
    # Create Challenger and Solver
    challenger = create_sanskrit_challenger(config_path)
    solver = create_sanskrit_solver(config_path)
    
    # Create co-evolutionary loop
    coevolution_loop = SanskritCoevolutionLoop(
        config=coevolution_config,
        challenger=challenger,
        solver=solver
    )
    
    # Run training
    results = coevolution_loop.run_coevolution()
    
    logger.info("Sanskrit co-evolutionary training completed")
    return results


# Utility functions for testing and evaluation
def test_coevolution_convergence(num_iterations: int = 10) -> Dict[str, Any]:
    """Test co-evolution convergence with a small number of iterations."""
    config = CoevolutionConfig(
        max_iterations=num_iterations,
        problems_per_iteration=5,
        convergence_threshold=0.8,
        save_frequency=5
    )
    
    return run_sanskrit_coevolution(coevolution_config=config)


def evaluate_coevolution_quality(results_path: str) -> Dict[str, float]:
    """Evaluate the quality of co-evolutionary training results."""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        quality_metrics = {
            "convergence_achieved": 1.0 if results.get("converged", False) else 0.0,
            "final_performance": results.get("final_performance", {}).get("solver_performance", {}).get("overall", 0.0),
            "performance_improvement": results.get("training_statistics", {}).get("performance_improvement", 0.0),
            "training_efficiency": results.get("total_iterations", 0) / results.get("config", {}).get("max_iterations", 1),
            "challenger_effectiveness": results.get("training_statistics", {}).get("average_challenger_effectiveness", 0.0)
        }
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error evaluating co-evolution quality: {e}")
        return {}