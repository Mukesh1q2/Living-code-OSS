"""
Mathematical and programming benchmarks for the Sanskrit Rewrite Engine.

This module provides specialized benchmarking for:
- Mathematical formula translation accuracy
- Programming construct generation
- Cross-domain semantic mapping validation
- Vedic mathematics sutra application
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .symbolic_computation import (
    SymbolicComputationEngine, create_symbolic_computation_engine,
    apply_vedic_math, verify_mathematical_identity
)
from .multi_domain_mapper import MultiDomainMapper
from .semantic_pipeline import process_sanskrit_text
from .corpus_benchmarking import AccuracyMetrics, PerformanceMetrics


@dataclass
class MathBenchmarkItem:
    """Mathematical benchmark test item."""
    id: str
    sanskrit_expression: str
    expected_formula: str
    mathematical_domain: str
    vedic_sutra_ref: Optional[str] = None
    difficulty_level: str = "INTERMEDIATE"
    expected_result: Optional[Any] = None


@dataclass
class ProgrammingBenchmarkItem:
    """Programming benchmark test item."""
    id: str
    sanskrit_description: str
    expected_code: str
    programming_language: str
    algorithm_type: str
    difficulty_level: str = "INTERMEDIATE"
    expected_output: Optional[str] = None


@dataclass
class CrossDomainBenchmarkItem:
    """Cross-domain mapping benchmark item."""
    id: str
    source_domain: str
    target_domain: str
    source_expression: str
    expected_mapping: str
    semantic_preservation_required: bool = True


@dataclass
class MathProgrammingBenchmarkResult:
    """Results from mathematical and programming benchmarks."""
    timestamp: datetime
    math_accuracy: AccuracyMetrics
    programming_accuracy: AccuracyMetrics
    cross_domain_accuracy: AccuracyMetrics
    performance_metrics: PerformanceMetrics
    vedic_math_coverage: Dict[str, int]
    programming_language_coverage: Dict[str, int]
    semantic_preservation_rate: float
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MathProgrammingBenchmarker:
    """
    Specialized benchmarker for mathematical and programming capabilities.
    
    Features:
    - Mathematical formula translation validation
    - Programming construct generation testing
    - Cross-domain semantic mapping evaluation
    - Vedic mathematics sutra application analysis
    - Performance profiling for computational tasks
    """
    
    def __init__(self):
        """Initialize the math/programming benchmarker."""
        self.symbolic_engine = create_symbolic_computation_engine()
        self.domain_mapper = MultiDomainMapper()
        self.logger = logging.getLogger(__name__)
        
        # Load benchmark datasets
        self.math_benchmarks = self._load_math_benchmarks()
        self.programming_benchmarks = self._load_programming_benchmarks()
        self.cross_domain_benchmarks = self._load_cross_domain_benchmarks()
    
    def _load_math_benchmarks(self) -> List[MathBenchmarkItem]:
        """Load mathematical benchmark dataset."""
        # Create sample mathematical benchmarks
        return [
            MathBenchmarkItem(
                id="math_001",
                sanskrit_expression="द्विगुणं त्रिगुणं च योगः",
                expected_formula="2x + 3y",
                mathematical_domain="algebra",
                vedic_sutra_ref="ekādhikena pūrveṇa",
                difficulty_level="BEGINNER"
            ),
            MathBenchmarkItem(
                id="math_002",
                sanskrit_expression="वर्गयोगः चतुर्गुणः",
                expected_formula="x² + y² = 4z",
                mathematical_domain="algebra",
                difficulty_level="INTERMEDIATE"
            ),
            MathBenchmarkItem(
                id="math_003",
                sanskrit_expression="त्रिकोणमितिः सिनुस् कोसिनुस्",
                expected_formula="sin²(θ) + cos²(θ) = 1",
                mathematical_domain="trigonometry",
                difficulty_level="INTERMEDIATE"
            ),
            MathBenchmarkItem(
                id="math_004",
                sanskrit_expression="अवकलनं समाकलनं च",
                expected_formula="∫(d/dx f(x))dx = f(x) + C",
                mathematical_domain="calculus",
                difficulty_level="ADVANCED"
            ),
            MathBenchmarkItem(
                id="math_005",
                sanskrit_expression="निखिलं नवतश्चरमं दशतः",
                expected_formula="(100-a)(100-b) = 10000 - 100(a+b) + ab",
                mathematical_domain="vedic_math",
                vedic_sutra_ref="nikhilam navatashcaramam dashatah",
                difficulty_level="ADVANCED"
            )
        ]
    
    def _load_programming_benchmarks(self) -> List[ProgrammingBenchmarkItem]:
        """Load programming benchmark dataset."""
        return [
            ProgrammingBenchmarkItem(
                id="prog_001",
                sanskrit_description="गणना एकादशि द्वादशि",
                expected_code="for i in range(1, 13): print(i)",
                programming_language="python",
                algorithm_type="iteration",
                difficulty_level="BEGINNER"
            ),
            ProgrammingBenchmarkItem(
                id="prog_002",
                sanskrit_description="यदि तर्हि अन्यथा",
                expected_code="if condition: action1() else: action2()",
                programming_language="python",
                algorithm_type="conditional",
                difficulty_level="BEGINNER"
            ),
            ProgrammingBenchmarkItem(
                id="prog_003",
                sanskrit_description="पुनरावृत्तिः स्वयं आह्वानम्",
                expected_code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                programming_language="python",
                algorithm_type="recursion",
                difficulty_level="INTERMEDIATE"
            ),
            ProgrammingBenchmarkItem(
                id="prog_004",
                sanskrit_description="क्रमबद्धता आरोहण अवरोहण",
                expected_code="def quicksort(arr): return sorted(arr)",
                programming_language="python",
                algorithm_type="sorting",
                difficulty_level="INTERMEDIATE"
            ),
            ProgrammingBenchmarkItem(
                id="prog_005",
                sanskrit_description="वृक्षस्य अन्वेषणम् गहनता",
                expected_code="def dfs(graph, start, visited=None): ...",
                programming_language="python",
                algorithm_type="graph_traversal",
                difficulty_level="ADVANCED"
            )
        ]
    
    def _load_cross_domain_benchmarks(self) -> List[CrossDomainBenchmarkItem]:
        """Load cross-domain mapping benchmark dataset."""
        return [
            CrossDomainBenchmarkItem(
                id="cross_001",
                source_domain="sanskrit_grammar",
                target_domain="programming",
                source_expression="सन्धिः योगः",
                expected_mapping="string concatenation",
                semantic_preservation_required=True
            ),
            CrossDomainBenchmarkItem(
                id="cross_002",
                source_domain="vedic_math",
                target_domain="algebra",
                source_expression="एकाधिकेन पूर्वेण",
                expected_mapping="(x+1)² = x² + 2x + 1",
                semantic_preservation_required=True
            ),
            CrossDomainBenchmarkItem(
                id="cross_003",
                source_domain="sanskrit_logic",
                target_domain="boolean_algebra",
                source_expression="अन्वयव्यतिरेकाभ्याम्",
                expected_mapping="A ∧ B ≡ ¬(¬A ∨ ¬B)",
                semantic_preservation_required=True
            ),
            CrossDomainBenchmarkItem(
                id="cross_004",
                source_domain="panini_grammar",
                target_domain="formal_grammar",
                source_expression="प्रत्ययः धातुना सह",
                expected_mapping="suffix → root + affix",
                semantic_preservation_required=True
            )
        ]
    
    def benchmark_mathematical_translation(self) -> Tuple[AccuracyMetrics, Dict[str, int]]:
        """
        Benchmark mathematical formula translation accuracy.
        
        Returns:
            Tuple of (accuracy metrics, vedic sutra coverage)
        """
        self.logger.info("Benchmarking mathematical translation...")
        
        accuracy = AccuracyMetrics()
        vedic_coverage = {}
        
        for item in self.math_benchmarks:
            try:
                accuracy.total_tests += 1
                
                # Process Sanskrit mathematical expression
                result = process_sanskrit_text(item.sanskrit_expression)
                
                # Apply symbolic computation
                if item.vedic_sutra_ref:
                    vedic_result = apply_vedic_math(item.sanskrit_expression, item.vedic_sutra_ref)
                    vedic_coverage[item.vedic_sutra_ref] = vedic_coverage.get(item.vedic_sutra_ref, 0) + 1
                
                # Translate to mathematical formula
                translated_formula = self.symbolic_engine.translate_to_formula(item.sanskrit_expression)
                
                # Compare with expected result
                if self._compare_mathematical_expressions(translated_formula, item.expected_formula):
                    accuracy.correct_predictions += 1
                else:
                    # Determine error type based on complexity
                    if len(translated_formula) > len(item.expected_formula):
                        accuracy.false_positives += 1
                    else:
                        accuracy.false_negatives += 1
                
                self.logger.debug(f"Math item {item.id}: Expected '{item.expected_formula}', Got '{translated_formula}'")
                
            except Exception as e:
                self.logger.error(f"Error processing math item {item.id}: {e}")
                accuracy.false_negatives += 1
        
        return accuracy, vedic_coverage
    
    def benchmark_programming_generation(self) -> Tuple[AccuracyMetrics, Dict[str, int]]:
        """
        Benchmark programming construct generation.
        
        Returns:
            Tuple of (accuracy metrics, language coverage)
        """
        self.logger.info("Benchmarking programming generation...")
        
        accuracy = AccuracyMetrics()
        language_coverage = {}
        
        for item in self.programming_benchmarks:
            try:
                accuracy.total_tests += 1
                language_coverage[item.programming_language] = \
                    language_coverage.get(item.programming_language, 0) + 1
                
                # Process Sanskrit description
                semantic_result = process_sanskrit_text(item.sanskrit_description)
                
                # Generate code using domain mapper
                generated_code = self.domain_mapper.map_to_programming(
                    item.sanskrit_description, 
                    item.programming_language
                )
                
                # Compare with expected code
                if self._compare_code_semantics(generated_code, item.expected_code):
                    accuracy.correct_predictions += 1
                else:
                    # Determine error type
                    if self._is_syntactically_valid(generated_code, item.programming_language):
                        accuracy.false_positives += 1  # Valid but incorrect
                    else:
                        accuracy.false_negatives += 1  # Invalid syntax
                
                self.logger.debug(f"Prog item {item.id}: Expected '{item.expected_code}', Got '{generated_code}'")
                
            except Exception as e:
                self.logger.error(f"Error processing programming item {item.id}: {e}")
                accuracy.false_negatives += 1
        
        return accuracy, language_coverage
    
    def benchmark_cross_domain_mapping(self) -> Tuple[AccuracyMetrics, float]:
        """
        Benchmark cross-domain semantic mapping.
        
        Returns:
            Tuple of (accuracy metrics, semantic preservation rate)
        """
        self.logger.info("Benchmarking cross-domain mapping...")
        
        accuracy = AccuracyMetrics()
        semantic_preservations = 0
        total_semantic_tests = 0
        
        for item in self.cross_domain_benchmarks:
            try:
                accuracy.total_tests += 1
                
                # Perform cross-domain mapping
                mapped_result = self.domain_mapper.map_between_domains(
                    item.source_expression,
                    item.source_domain,
                    item.target_domain
                )
                
                # Compare with expected mapping
                if self._compare_cross_domain_mapping(mapped_result, item.expected_mapping):
                    accuracy.correct_predictions += 1
                else:
                    accuracy.false_negatives += 1
                
                # Check semantic preservation if required
                if item.semantic_preservation_required:
                    total_semantic_tests += 1
                    if self._validate_semantic_preservation(
                        item.source_expression, mapped_result, 
                        item.source_domain, item.target_domain
                    ):
                        semantic_preservations += 1
                
                self.logger.debug(f"Cross-domain item {item.id}: Expected '{item.expected_mapping}', Got '{mapped_result}'")
                
            except Exception as e:
                self.logger.error(f"Error processing cross-domain item {item.id}: {e}")
                accuracy.false_negatives += 1
        
        semantic_preservation_rate = semantic_preservations / total_semantic_tests if total_semantic_tests > 0 else 0.0
        
        return accuracy, semantic_preservation_rate
    
    def run_comprehensive_benchmark(self) -> MathProgrammingBenchmarkResult:
        """
        Run comprehensive mathematical and programming benchmarks.
        
        Returns:
            Complete benchmark results
        """
        self.logger.info("Starting comprehensive math/programming benchmark...")
        
        start_time = time.perf_counter()
        
        # Run individual benchmarks
        math_accuracy, vedic_coverage = self.benchmark_mathematical_translation()
        prog_accuracy, lang_coverage = self.benchmark_programming_generation()
        cross_accuracy, semantic_rate = self.benchmark_cross_domain_mapping()
        
        end_time = time.perf_counter()
        
        # Create performance metrics
        performance = PerformanceMetrics()
        performance.processing_times = [end_time - start_time]
        
        # Generate recommendations
        recommendations = self._generate_math_prog_recommendations(
            math_accuracy, prog_accuracy, cross_accuracy, semantic_rate
        )
        
        return MathProgrammingBenchmarkResult(
            timestamp=datetime.now(),
            math_accuracy=math_accuracy,
            programming_accuracy=prog_accuracy,
            cross_domain_accuracy=cross_accuracy,
            performance_metrics=performance,
            vedic_math_coverage=vedic_coverage,
            programming_language_coverage=lang_coverage,
            semantic_preservation_rate=semantic_rate,
            errors=[],
            recommendations=recommendations
        )
    
    def _compare_mathematical_expressions(self, actual: str, expected: str) -> bool:
        """Compare mathematical expressions for semantic equivalence."""
        # Simplified comparison - could be enhanced with symbolic math
        # Remove whitespace and normalize
        actual_norm = actual.replace(" ", "").lower()
        expected_norm = expected.replace(" ", "").lower()
        
        # Direct string comparison for now
        # TODO: Implement symbolic equivalence checking
        return actual_norm == expected_norm
    
    def _compare_code_semantics(self, actual: str, expected: str) -> bool:
        """Compare code for semantic equivalence."""
        # Simplified semantic comparison
        # Remove whitespace and normalize
        actual_norm = actual.replace(" ", "").replace("\n", "").lower()
        expected_norm = expected.replace(" ", "").replace("\n", "").lower()
        
        # Check for key semantic elements
        return actual_norm == expected_norm or self._check_code_equivalence(actual, expected)
    
    def _check_code_equivalence(self, code1: str, code2: str) -> bool:
        """Check if two code snippets are semantically equivalent."""
        # Basic heuristic checks
        # TODO: Implement more sophisticated AST-based comparison
        
        # Check for similar keywords
        keywords1 = set(word for word in code1.split() if word in ['if', 'else', 'for', 'while', 'def', 'return'])
        keywords2 = set(word for word in code2.split() if word in ['if', 'else', 'for', 'while', 'def', 'return'])
        
        return len(keywords1.intersection(keywords2)) > 0
    
    def _is_syntactically_valid(self, code: str, language: str) -> bool:
        """Check if code is syntactically valid for the given language."""
        # Basic syntax validation
        if language.lower() == "python":
            try:
                compile(code, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        
        # For other languages, do basic checks
        return len(code.strip()) > 0 and not code.strip().startswith("ERROR")
    
    def _compare_cross_domain_mapping(self, actual: str, expected: str) -> bool:
        """Compare cross-domain mappings for correctness."""
        # Normalize and compare
        actual_norm = actual.strip().lower()
        expected_norm = expected.strip().lower()
        
        # Allow for reasonable variations in expression
        return actual_norm == expected_norm or self._check_semantic_similarity(actual, expected)
    
    def _check_semantic_similarity(self, text1: str, text2: str) -> bool:
        """Check semantic similarity between two texts."""
        # Simple word overlap heuristic
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Jaccard similarity threshold
        return overlap / union > 0.5
    
    def _validate_semantic_preservation(self, 
                                      source: str, 
                                      target: str, 
                                      source_domain: str, 
                                      target_domain: str) -> bool:
        """Validate that semantic meaning is preserved across domains."""
        # Simplified semantic preservation check
        # TODO: Implement more sophisticated semantic analysis
        
        # Check that key concepts are preserved
        source_concepts = self._extract_key_concepts(source, source_domain)
        target_concepts = self._extract_key_concepts(target, target_domain)
        
        # Ensure at least 70% concept overlap
        if not source_concepts or not target_concepts:
            return False
        
        overlap = len(source_concepts.intersection(target_concepts))
        return overlap / len(source_concepts) >= 0.7
    
    def _extract_key_concepts(self, text: str, domain: str) -> set:
        """Extract key concepts from text based on domain."""
        # Domain-specific concept extraction
        concepts = set()
        
        if domain == "sanskrit_grammar":
            grammar_terms = ["sandhi", "sutra", "pratyaya", "dhatu", "vibhakti"]
            concepts.update(term for term in grammar_terms if term in text.lower())
        
        elif domain == "mathematics" or domain == "vedic_math":
            math_terms = ["sum", "product", "square", "formula", "equation"]
            concepts.update(term for term in math_terms if term in text.lower())
        
        elif domain == "programming":
            prog_terms = ["function", "loop", "condition", "variable", "algorithm"]
            concepts.update(term for term in prog_terms if term in text.lower())
        
        # Add general concepts (words)
        concepts.update(word.lower() for word in text.split() if len(word) > 3)
        
        return concepts
    
    def _generate_math_prog_recommendations(self, 
                                         math_acc: AccuracyMetrics,
                                         prog_acc: AccuracyMetrics,
                                         cross_acc: AccuracyMetrics,
                                         semantic_rate: float) -> List[str]:
        """Generate recommendations for math/programming improvements."""
        recommendations = []
        
        # Mathematical translation recommendations
        if math_acc.accuracy < 0.7:
            recommendations.append(
                f"Mathematical translation accuracy is low ({math_acc.accuracy:.2%}). "
                f"Consider expanding the symbolic computation engine and adding more "
                f"Vedic mathematics sutra implementations."
            )
        
        # Programming generation recommendations
        if prog_acc.accuracy < 0.6:
            recommendations.append(
                f"Programming generation accuracy is low ({prog_acc.accuracy:.2%}). "
                f"Enhance the domain mapper with more programming language patterns "
                f"and algorithmic templates."
            )
        
        # Cross-domain mapping recommendations
        if cross_acc.accuracy < 0.8:
            recommendations.append(
                f"Cross-domain mapping accuracy is low ({cross_acc.accuracy:.2%}). "
                f"Improve semantic mapping algorithms and add more domain-specific "
                f"transformation rules."
            )
        
        # Semantic preservation recommendations
        if semantic_rate < 0.8:
            recommendations.append(
                f"Semantic preservation rate is low ({semantic_rate:.2%}). "
                f"Implement more sophisticated semantic analysis and concept "
                f"preservation validation."
            )
        
        return recommendations
    
    def generate_report(self, result: MathProgrammingBenchmarkResult) -> str:
        """Generate a comprehensive report for math/programming benchmarks."""
        report_lines = []
        
        report_lines.append("MATHEMATICAL & PROGRAMMING BENCHMARK REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Mathematical Translation Results
        report_lines.append("MATHEMATICAL TRANSLATION")
        report_lines.append("-" * 25)
        math = result.math_accuracy
        report_lines.append(f"Accuracy: {math.accuracy:.2%} ({math.correct_predictions}/{math.total_tests})")
        report_lines.append(f"Precision: {math.precision:.2%}")
        report_lines.append(f"Recall: {math.recall:.2%}")
        report_lines.append(f"F1 Score: {math.f1_score:.2%}")
        
        if result.vedic_math_coverage:
            report_lines.append("Vedic Sutra Coverage:")
            for sutra, count in result.vedic_math_coverage.items():
                report_lines.append(f"  {sutra}: {count} applications")
        report_lines.append("")
        
        # Programming Generation Results
        report_lines.append("PROGRAMMING GENERATION")
        report_lines.append("-" * 22)
        prog = result.programming_accuracy
        report_lines.append(f"Accuracy: {prog.accuracy:.2%} ({prog.correct_predictions}/{prog.total_tests})")
        report_lines.append(f"Precision: {prog.precision:.2%}")
        report_lines.append(f"Recall: {prog.recall:.2%}")
        report_lines.append(f"F1 Score: {prog.f1_score:.2%}")
        
        if result.programming_language_coverage:
            report_lines.append("Language Coverage:")
            for lang, count in result.programming_language_coverage.items():
                report_lines.append(f"  {lang}: {count} tests")
        report_lines.append("")
        
        # Cross-Domain Mapping Results
        report_lines.append("CROSS-DOMAIN MAPPING")
        report_lines.append("-" * 20)
        cross = result.cross_domain_accuracy
        report_lines.append(f"Accuracy: {cross.accuracy:.2%} ({cross.correct_predictions}/{cross.total_tests})")
        report_lines.append(f"Semantic Preservation Rate: {result.semantic_preservation_rate:.2%}")
        report_lines.append("")
        
        # Performance
        report_lines.append("PERFORMANCE")
        report_lines.append("-" * 11)
        perf = result.performance_metrics
        report_lines.append(f"Total Processing Time: {perf.mean_processing_time:.3f}s")
        report_lines.append("")
        
        # Recommendations
        if result.recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 15)
            for i, rec in enumerate(result.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)


def run_math_programming_benchmarks() -> MathProgrammingBenchmarkResult:
    """Run comprehensive mathematical and programming benchmarks."""
    benchmarker = MathProgrammingBenchmarker()
    return benchmarker.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    result = run_math_programming_benchmarks()
    
    # Generate and print report
    benchmarker = MathProgrammingBenchmarker()
    report = benchmarker.generate_report(result)
    print(report)