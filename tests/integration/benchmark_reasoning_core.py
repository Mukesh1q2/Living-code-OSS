"""
Performance benchmarks for the reasoning core.

This module provides comprehensive performance testing for the Sanskrit
grammatical inference system, measuring throughput, latency, and scalability.
"""

import time
import statistics
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

from sanskrit_rewrite_engine.reasoning_core import (
    ReasoningCore, KnowledgeBase, LogicalTerm, LogicalClause,
    ConstraintSolver, TheoremProver
)
from sanskrit_rewrite_engine.rule import SutraRule, SutraReference, RuleType


class ReasoningCoreBenchmark:
    """Benchmark suite for reasoning core performance."""
    
    def __init__(self):
        self.results = {}
        self.core = ReasoningCore()
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("Running Reasoning Core Performance Benchmarks...")
        
        self.results['query_performance'] = self.benchmark_query_performance()
        self.results['knowledge_base_scaling'] = self.benchmark_knowledge_base_scaling()
        self.results['constraint_solving'] = self.benchmark_constraint_solving()
        self.results['theorem_proving'] = self.benchmark_theorem_proving()
        self.results['memory_usage'] = self.benchmark_memory_usage()
        
        return self.results
    
    def benchmark_query_performance(self) -> Dict[str, Any]:
        """Benchmark query performance with varying complexity."""
        print("Benchmarking query performance...")
        
        results = {
            'simple_queries': [],
            'complex_queries': [],
            'batch_queries': []
        }
        
        # Simple queries
        simple_times = []
        for i in range(1000):
            start = time.perf_counter()
            self.core.query(f"vowel(a_{i})")
            end = time.perf_counter()
            simple_times.append(end - start)
        
        results['simple_queries'] = {
            'count': len(simple_times),
            'mean_time': statistics.mean(simple_times),
            'median_time': statistics.median(simple_times),
            'std_dev': statistics.stdev(simple_times) if len(simple_times) > 1 else 0,
            'min_time': min(simple_times),
            'max_time': max(simple_times),
            'throughput_qps': 1000 / sum(simple_times)
        }
        
        # Complex queries with multiple terms
        complex_times = []
        for i in range(100):
            start = time.perf_counter()
            self.core.query(f"sandhi_applies(vowel_{i}, vowel_{i+1}, result_{i})")
            end = time.perf_counter()
            complex_times.append(end - start)
        
        results['complex_queries'] = {
            'count': len(complex_times),
            'mean_time': statistics.mean(complex_times),
            'median_time': statistics.median(complex_times),
            'std_dev': statistics.stdev(complex_times) if len(complex_times) > 1 else 0,
            'throughput_qps': 100 / sum(complex_times)
        }
        
        # Batch queries
        batch_queries = [f"test_predicate({i})" for i in range(100)]
        start = time.perf_counter()
        for query in batch_queries:
            self.core.query(query)
        end = time.perf_counter()
        
        results['batch_queries'] = {
            'count': len(batch_queries),
            'total_time': end - start,
            'throughput_qps': len(batch_queries) / (end - start)
        }
        
        return results
    
    def benchmark_knowledge_base_scaling(self) -> Dict[str, Any]:
        """Benchmark knowledge base performance with increasing size."""
        print("Benchmarking knowledge base scaling...")
        
        results = {
            'fact_insertion': [],
            'rule_insertion': [],
            'query_performance_vs_size': []
        }
        
        kb = KnowledgeBase()
        
        # Test fact insertion scaling
        fact_counts = [100, 500, 1000, 5000, 10000]
        for count in fact_counts:
            kb_test = KnowledgeBase()
            
            start = time.perf_counter()
            for i in range(count):
                fact = LogicalClause(LogicalTerm("test_fact", [LogicalTerm(f"arg_{i}")]))
                kb_test.add_fact(fact)
            end = time.perf_counter()
            
            results['fact_insertion'].append({
                'fact_count': count,
                'insertion_time': end - start,
                'facts_per_second': count / (end - start)
            })
        
        # Test rule insertion scaling
        rule_counts = [10, 50, 100, 500, 1000]
        for count in rule_counts:
            kb_test = KnowledgeBase()
            
            start = time.perf_counter()
            for i in range(count):
                head = LogicalTerm("test_rule", [LogicalTerm(f"X_{i}", is_variable=True)])
                body = [LogicalTerm("test_fact", [LogicalTerm(f"X_{i}", is_variable=True)])]
                rule = LogicalClause(head, body)
                kb_test.add_rule(rule)
            end = time.perf_counter()
            
            results['rule_insertion'].append({
                'rule_count': count,
                'insertion_time': end - start,
                'rules_per_second': count / (end - start)
            })
        
        # Test query performance vs knowledge base size
        for fact_count in [100, 1000, 5000]:
            kb_test = KnowledgeBase()
            
            # Add facts
            for i in range(fact_count):
                fact = LogicalClause(LogicalTerm("test_fact", [LogicalTerm(f"arg_{i}")]))
                kb_test.add_fact(fact)
            
            # Measure query time
            query_times = []
            for i in range(100):
                goal = LogicalTerm("test_fact", [LogicalTerm(f"arg_{i % fact_count}")])
                start = time.perf_counter()
                kb_test.query(goal)
                end = time.perf_counter()
                query_times.append(end - start)
            
            results['query_performance_vs_size'].append({
                'kb_size': fact_count,
                'mean_query_time': statistics.mean(query_times),
                'query_throughput': 100 / sum(query_times)
            })
        
        return results
    
    def benchmark_constraint_solving(self) -> Dict[str, Any]:
        """Benchmark constraint solving performance."""
        print("Benchmarking constraint solving...")
        
        results = {
            'simple_csp': [],
            'complex_csp': [],
            'scaling_analysis': []
        }
        
        kb = KnowledgeBase()
        solver = ConstraintSolver(kb)
        
        # Simple CSP problems
        problem_sizes = [3, 5, 7, 10]
        for size in problem_sizes:
            variables = {f'var_{i}': list(range(size)) for i in range(size)}
            
            # All different constraint
            def all_different(assignment: Dict[str, Any]) -> bool:
                values = list(assignment.values())
                return len(values) == len(set(values))
            
            solver_test = ConstraintSolver(kb)
            solver_test.add_constraint(all_different)
            
            start = time.perf_counter()
            solutions = solver_test.solve_constraints(variables)
            end = time.perf_counter()
            
            results['simple_csp'].append({
                'problem_size': size,
                'solve_time': end - start,
                'solution_count': len(solutions),
                'solutions_per_second': len(solutions) / (end - start) if (end - start) > 0 else 0
            })
        
        # Complex CSP with multiple constraints
        for size in [3, 4, 5]:
            variables = {f'var_{i}': list(range(size * 2)) for i in range(size)}
            
            solver_test = ConstraintSolver(kb)
            
            # Multiple constraints
            def all_different(assignment: Dict[str, Any]) -> bool:
                values = list(assignment.values())
                return len(values) == len(set(values))
            
            def sum_constraint(assignment: Dict[str, Any]) -> bool:
                if len(assignment) == size:
                    return sum(assignment.values()) == size * (size - 1) // 2
                return True
            
            solver_test.add_constraint(all_different)
            solver_test.add_constraint(sum_constraint)
            
            start = time.perf_counter()
            solutions = solver_test.solve_constraints(variables)
            end = time.perf_counter()
            
            results['complex_csp'].append({
                'problem_size': size,
                'solve_time': end - start,
                'solution_count': len(solutions)
            })
        
        return results
    
    def benchmark_theorem_proving(self) -> Dict[str, Any]:
        """Benchmark theorem proving performance."""
        print("Benchmarking theorem proving...")
        
        results = {
            'simple_proofs': [],
            'complex_proofs': [],
            'proof_depth_analysis': []
        }
        
        kb = KnowledgeBase()
        prover = TheoremProver(kb)
        
        # Add some facts and rules for testing
        facts = [
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("a")])),
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("i")])),
            LogicalClause(LogicalTerm("vowel", [LogicalTerm("u")])),
            LogicalClause(LogicalTerm("consonant", [LogicalTerm("k")])),
            LogicalClause(LogicalTerm("consonant", [LogicalTerm("t")])),
        ]
        
        for fact in facts:
            kb.add_fact(fact)
        
        # Simple proof benchmarks
        simple_goals = [
            LogicalTerm("vowel", [LogicalTerm("a")]),
            LogicalTerm("vowel", [LogicalTerm("i")]),
            LogicalTerm("consonant", [LogicalTerm("k")])
        ]
        
        for goal in simple_goals:
            times = []
            for _ in range(100):
                start = time.perf_counter()
                proved, steps = prover.prove(goal)
                end = time.perf_counter()
                times.append(end - start)
            
            results['simple_proofs'].append({
                'goal': str(goal),
                'mean_time': statistics.mean(times),
                'median_time': statistics.median(times),
                'throughput': 100 / sum(times)
            })
        
        # Test proof depth impact
        for max_depth in [1, 3, 5, 10]:
            goal = LogicalTerm("vowel", [LogicalTerm("a")])
            
            times = []
            for _ in range(50):
                start = time.perf_counter()
                proved, steps = prover.prove(goal, max_depth=max_depth)
                end = time.perf_counter()
                times.append(end - start)
            
            results['proof_depth_analysis'].append({
                'max_depth': max_depth,
                'mean_time': statistics.mean(times),
                'throughput': 50 / sum(times)
            })
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        results = {
            'baseline_memory': process.memory_info().rss / 1024 / 1024,  # MB
            'memory_growth': []
        }
        
        # Test memory growth with increasing knowledge base size
        kb = KnowledgeBase()
        
        for fact_count in [1000, 5000, 10000, 20000]:
            # Add facts
            for i in range(fact_count):
                fact = LogicalClause(LogicalTerm("test_fact", [LogicalTerm(f"arg_{i}")]))
                kb.add_fact(fact)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            results['memory_growth'].append({
                'fact_count': fact_count,
                'memory_mb': current_memory,
                'memory_per_fact': (current_memory - results['baseline_memory']) / fact_count
            })
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report = []
        report.append("SANSKRIT REASONING CORE PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Query Performance
        if 'query_performance' in self.results:
            qp = self.results['query_performance']
            report.append("QUERY PERFORMANCE:")
            report.append(f"  Simple Queries: {qp['simple_queries']['throughput_qps']:.2f} QPS")
            report.append(f"  Complex Queries: {qp['complex_queries']['throughput_qps']:.2f} QPS")
            report.append(f"  Batch Queries: {qp['batch_queries']['throughput_qps']:.2f} QPS")
            report.append("")
        
        # Knowledge Base Scaling
        if 'knowledge_base_scaling' in self.results:
            kbs = self.results['knowledge_base_scaling']
            report.append("KNOWLEDGE BASE SCALING:")
            
            if kbs['fact_insertion']:
                last_fact_result = kbs['fact_insertion'][-1]
                report.append(f"  Fact Insertion: {last_fact_result['facts_per_second']:.2f} facts/sec")
            
            if kbs['rule_insertion']:
                last_rule_result = kbs['rule_insertion'][-1]
                report.append(f"  Rule Insertion: {last_rule_result['rules_per_second']:.2f} rules/sec")
            
            report.append("")
        
        # Constraint Solving
        if 'constraint_solving' in self.results:
            cs = self.results['constraint_solving']
            report.append("CONSTRAINT SOLVING:")
            
            if cs['simple_csp']:
                avg_time = statistics.mean([r['solve_time'] for r in cs['simple_csp']])
                report.append(f"  Average CSP Solve Time: {avg_time:.4f} seconds")
            
            report.append("")
        
        # Memory Usage
        if 'memory_usage' in self.results:
            mu = self.results['memory_usage']
            report.append("MEMORY USAGE:")
            report.append(f"  Baseline Memory: {mu['baseline_memory']:.2f} MB")
            
            if mu['memory_growth']:
                last_growth = mu['memory_growth'][-1]
                report.append(f"  Memory per Fact: {last_growth['memory_per_fact']:.6f} MB")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "reasoning_core_benchmark_results.json"):
        """Save benchmark results to file."""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
    
    def plot_results(self):
        """Generate performance plots."""
        if not self.results:
            print("No results to plot. Run benchmarks first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sanskrit Reasoning Core Performance Analysis')
        
        # Query Performance Plot
        if 'query_performance' in self.results:
            qp = self.results['query_performance']
            query_types = ['Simple', 'Complex', 'Batch']
            throughputs = [
                qp['simple_queries']['throughput_qps'],
                qp['complex_queries']['throughput_qps'],
                qp['batch_queries']['throughput_qps']
            ]
            
            axes[0, 0].bar(query_types, throughputs)
            axes[0, 0].set_title('Query Throughput (QPS)')
            axes[0, 0].set_ylabel('Queries per Second')
        
        # Knowledge Base Scaling Plot
        if 'knowledge_base_scaling' in self.results:
            kbs = self.results['knowledge_base_scaling']
            
            if kbs['fact_insertion']:
                fact_counts = [r['fact_count'] for r in kbs['fact_insertion']]
                insertion_rates = [r['facts_per_second'] for r in kbs['fact_insertion']]
                
                axes[0, 1].plot(fact_counts, insertion_rates, 'o-')
                axes[0, 1].set_title('Fact Insertion Rate vs KB Size')
                axes[0, 1].set_xlabel('Number of Facts')
                axes[0, 1].set_ylabel('Facts per Second')
        
        # Constraint Solving Performance
        if 'constraint_solving' in self.results:
            cs = self.results['constraint_solving']
            
            if cs['simple_csp']:
                problem_sizes = [r['problem_size'] for r in cs['simple_csp']]
                solve_times = [r['solve_time'] for r in cs['simple_csp']]
                
                axes[1, 0].plot(problem_sizes, solve_times, 'o-')
                axes[1, 0].set_title('CSP Solve Time vs Problem Size')
                axes[1, 0].set_xlabel('Problem Size')
                axes[1, 0].set_ylabel('Solve Time (seconds)')
        
        # Memory Usage Plot
        if 'memory_usage' in self.results:
            mu = self.results['memory_usage']
            
            if mu['memory_growth']:
                fact_counts = [r['fact_count'] for r in mu['memory_growth']]
                memory_usage = [r['memory_mb'] for r in mu['memory_growth']]
                
                axes[1, 1].plot(fact_counts, memory_usage, 'o-')
                axes[1, 1].set_title('Memory Usage vs KB Size')
                axes[1, 1].set_xlabel('Number of Facts')
                axes[1, 1].set_ylabel('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.savefig('reasoning_core_performance.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run the complete benchmark suite."""
    benchmark = ReasoningCoreBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Generate and print report
    report = benchmark.generate_report()
    print(report)
    
    # Save results
    benchmark.save_results()
    
    # Generate plots (if matplotlib is available)
    try:
        benchmark.plot_results()
    except ImportError:
        print("Matplotlib not available. Skipping plots.")


if __name__ == "__main__":
    main()