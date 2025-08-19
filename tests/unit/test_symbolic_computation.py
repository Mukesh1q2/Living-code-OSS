"""
Tests for symbolic computation integration module.

This module tests the integration of SymPy for algebraic computations,
Sanskrit sūtra to mathematical formula translation, and mathematical
proof verification using sūtra logic.
"""

import pytest
import sys
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Test both with and without SymPy
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from sanskrit_rewrite_engine.symbolic_computation import (
    MathematicalDomain,
    VedicMathSutra,
    MathematicalTransformation,
    SutraToMathMapper,
    SymbolicAlgebraEngine,
    MathematicalProofVerifier,
    SymbolicComputationEngine,
    create_symbolic_computation_engine,
    apply_vedic_math,
    verify_mathematical_identity
)


class TestVedicMathSutra:
    """Test VedicMathSutra dataclass."""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_vedic_math_sutra_creation(self):
        """Test creating a VedicMathSutra instance."""
        sutra = VedicMathSutra(
            name="Test Sutra",
            sanskrit_text="परीक्षा सूत्र",
            english_translation="Test formula",
            domain=MathematicalDomain.ARITHMETIC,
            formula="a + b = c",
            examples=[{"input": "1+1", "output": "2"}]
        )
        
        assert sutra.name == "Test Sutra"
        assert sutra.domain == MathematicalDomain.ARITHMETIC
        assert sutra.formula == "a + b = c"
        assert len(sutra.examples) == 1
    
    @pytest.mark.skipif(SYMPY_AVAILABLE, reason="SymPy is available")
    def test_vedic_math_sutra_without_sympy(self):
        """Test that VedicMathSutra raises ImportError without SymPy."""
        with pytest.raises(ImportError, match="SymPy is required"):
            VedicMathSutra(
                name="Test",
                sanskrit_text="Test",
                english_translation="Test",
                domain=MathematicalDomain.ARITHMETIC
            )


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestSutraToMathMapper:
    """Test SutraToMathMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = SutraToMathMapper()
    
    def test_initialization(self):
        """Test mapper initialization."""
        assert len(self.mapper.vedic_sutras) > 0
        assert "ekadhikena_purvena" in self.mapper.vedic_sutras
        assert "nikhilam" in self.mapper.vedic_sutras
        assert "urdhva_tiryagbyham" in self.mapper.vedic_sutras
        assert "paravartya_yojayet" in self.mapper.vedic_sutras
    
    def test_ekadhikena_purvena_algorithm(self):
        """Test Ekadhikena Purvena algorithm (squaring numbers ending in 5)."""
        result = self.mapper.apply_vedic_sutra("ekadhikena_purvena", 15)
        
        assert result["result"] == 225
        assert "sutra" in result
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # Test with 25
        result = self.mapper.apply_vedic_sutra("ekadhikena_purvena", 25)
        assert result["result"] == 625
        
        # Test with 35
        result = self.mapper.apply_vedic_sutra("ekadhikena_purvena", 35)
        assert result["result"] == 1225
    
    def test_ekadhikena_purvena_invalid_input(self):
        """Test Ekadhikena Purvena with invalid input."""
        result = self.mapper.apply_vedic_sutra("ekadhikena_purvena", 12)
        assert "error" in result
        assert "ending in 5" in result["error"]
    
    def test_nikhilam_algorithm(self):
        """Test Nikhilam algorithm (multiplication near base)."""
        result = self.mapper.apply_vedic_sutra("nikhilam", 97, 96, base=100)
        
        assert result["result"] == 9312
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # Test with numbers near 1000
        result = self.mapper.apply_vedic_sutra("nikhilam", 998, 997, base=1000)
        assert result["result"] == 995006
    
    def test_urdhva_tiryagbyham_algorithm(self):
        """Test Urdhva-Tiryagbyham algorithm (general multiplication)."""
        result = self.mapper.apply_vedic_sutra("urdhva_tiryagbyham", 23, 47)
        
        assert result["result"] == 1081
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # Test with single digits
        result = self.mapper.apply_vedic_sutra("urdhva_tiryagbyham", 7, 8)
        assert result["result"] == 56
    
    def test_paravartya_yojayet_algorithm(self):
        """Test Paravartya Yojayet algorithm (solving linear equations)."""
        # System: 2x + 3y = 7, 4x + 5y = 13
        equations = [([2, 3], 7), ([4, 5], 13)]
        result = self.mapper.apply_vedic_sutra("paravartya_yojayet", equations)
        
        assert "result" in result
        solutions = result["result"]
        assert len(solutions) == 2
        
        # Verify solution: x = 2, y = 1
        x, y = solutions
        assert abs(x - 2.0) < 1e-10
        assert abs(y - 1.0) < 1e-10
    
    def test_translate_sutra_to_formula(self):
        """Test translating sūtra names to formulas."""
        formula = self.mapper.translate_sutra_to_formula("ekadhikena_purvena")
        assert formula is not None
        assert "a*(a+1)" in formula or "100*a*(a+1)" in formula
        
        # Test unknown sūtra
        formula = self.mapper.translate_sutra_to_formula("unknown_sutra")
        assert formula is None
    
    def test_get_applicable_sutras(self):
        """Test getting applicable sūtras for problem types."""
        arithmetic_sutras = self.mapper.get_applicable_sutras("arithmetic")
        assert len(arithmetic_sutras) > 0
        assert "ekadhikena_purvena" in arithmetic_sutras
        
        algebra_sutras = self.mapper.get_applicable_sutras("algebra")
        assert len(algebra_sutras) > 0
        assert "urdhva_tiryagbyham" in algebra_sutras


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestSymbolicAlgebraEngine:
    """Test SymbolicAlgebraEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SymbolicAlgebraEngine()
    
    def test_parse_expression(self):
        """Test expression parsing."""
        expr = self.engine.parse_expression("x**2 + 2*x + 1")
        assert str(expr) == "x**2 + 2*x + 1"
        
        # Test caching
        expr2 = self.engine.parse_expression("x**2 + 2*x + 1")
        assert expr == expr2
    
    def test_parse_expression_error(self):
        """Test expression parsing with invalid input."""
        with pytest.raises(ValueError, match="Failed to parse expression"):
            self.engine.parse_expression("invalid expression +++")
    
    def test_sandhi_algebra_transformation(self):
        """Test sandhi-like algebraic transformation."""
        transformation = self.engine.apply_transformation("x + x + x", "sandhi_algebra")
        
        assert transformation.transformation_type == "sandhi_algebra"
        assert transformation.input_expression == "x + x + x"
        # Should simplify to 3*x
        assert "3*x" in transformation.output_expression
        assert len(transformation.verification_steps) > 0
    
    def test_compound_expansion_transformation(self):
        """Test compound expansion transformation."""
        transformation = self.engine.apply_transformation("(x + 1)*(x + 2)", "compound_expansion")
        
        assert transformation.transformation_type == "compound_expansion"
        assert transformation.input_expression == "(x + 1)*(x + 2)"
        # Should expand to x**2 + 3*x + 2
        assert "x**2" in transformation.output_expression
        assert "3*x" in transformation.output_expression
    
    def test_morphological_factoring_transformation(self):
        """Test morphological factoring transformation."""
        transformation = self.engine.apply_transformation("x**2 + 3*x + 2", "morphological_factoring")
        
        assert transformation.transformation_type == "morphological_factoring"
        assert transformation.input_expression == "x**2 + 3*x + 2"
        # Should factor to (x + 1)*(x + 2)
        assert "x + 1" in transformation.output_expression
        assert "x + 2" in transformation.output_expression
    
    def test_grammatical_simplification_transformation(self):
        """Test grammatical simplification transformation."""
        transformation = self.engine.apply_transformation("sin(x)**2 + cos(x)**2", "grammatical_simplification")
        
        assert transformation.transformation_type == "grammatical_simplification"
        assert transformation.input_expression == "sin(x)**2 + cos(x)**2"
        # Should simplify to 1
        assert transformation.output_expression == "1"
    
    def test_solve_equation_simple(self):
        """Test solving simple equations."""
        result = self.engine.solve_equation("x + 2 = 5")
        
        assert result["variable"] == "x"
        assert "3" in result["solutions"]
        assert len(result["solutions"]) == 1
    
    def test_solve_equation_quadratic(self):
        """Test solving quadratic equations."""
        result = self.engine.solve_equation("x**2 - 5*x + 6 = 0")
        
        assert result["variable"] == "x"
        assert len(result["solutions"]) == 2
        # Solutions should be 2 and 3
        solutions = [int(float(sol)) for sol in result["solutions"]]
        assert 2 in solutions
        assert 3 in solutions
    
    def test_solve_equation_multiple_variables(self):
        """Test solving equation with multiple variables."""
        with pytest.raises(ValueError, match="Multiple variables found"):
            self.engine.solve_equation("x + y = 5")
    
    def test_solve_equation_specified_variable(self):
        """Test solving equation with specified variable."""
        result = self.engine.solve_equation("x + y = 5", "x")
        
        assert result["variable"] == "x"
        assert "5 - y" in result["solutions"][0]


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestMathematicalProofVerifier:
    """Test MathematicalProofVerifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = MathematicalProofVerifier()
    
    def test_verify_algebraic_identity_valid(self):
        """Test verifying a valid algebraic identity."""
        result = self.verifier.verify_proof("(x + 1)**2 = x**2 + 2*x + 1", [])
        
        assert result["valid"] is True
        assert "verification_steps" in result
        assert len(result["verification_steps"]) > 0
    
    def test_verify_algebraic_identity_invalid(self):
        """Test verifying an invalid algebraic identity."""
        result = self.verifier.verify_proof("x + 1 = x + 2", [])
        
        assert result["valid"] is False
        assert "verification_steps" in result
    
    def test_verify_algebraic_identity_no_equals(self):
        """Test verifying statement without equals sign."""
        result = self.verifier.verify_proof("x + 1", [])
        
        assert result["valid"] is False
        assert "error" in result
        assert "must contain '='" in result["error"]
    
    def test_verify_algebraic_identity_parse_error(self):
        """Test verifying statement with parse error."""
        result = self.verifier.verify_proof("invalid +++ = also invalid +++", [])
        
        assert result["valid"] is False
        assert "error" in result
    
    def test_proof_caching(self):
        """Test that proof results are cached."""
        statement = "(x + 1)**2 = x**2 + 2*x + 1"
        
        # First call
        result1 = self.verifier.verify_proof(statement, [])
        
        # Second call should be cached
        result2 = self.verifier.verify_proof(statement, [])
        
        assert result1["valid"] == result2["valid"]
        # Check that it was cached (implementation detail)
        assert f"algebraic_identity:{statement}" in self.verifier.proof_cache


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestSymbolicComputationEngine:
    """Test SymbolicComputationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SymbolicComputationEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.sutra_mapper is not None
        assert self.engine.algebra_engine is not None
        assert self.engine.proof_verifier is not None
        assert len(self.engine.computation_history) == 0
    
    def test_vedic_math_query_square(self):
        """Test processing Vedic math query for squaring."""
        result = self.engine.process_mathematical_query("square 25 using vedic method")
        
        assert result["approach"] == "vedic_math"
        assert result["result"] == 625
        assert "steps" in result
        assert len(result["steps"]) > 0
    
    def test_vedic_math_query_multiply(self):
        """Test processing Vedic math query for multiplication."""
        result = self.engine.process_mathematical_query("multiply 97 and 96 using vedic method")
        
        assert result["approach"] == "vedic_math"
        assert result["result"] == 9312
        assert "steps" in result
    
    def test_vedic_math_query_general_multiply(self):
        """Test processing Vedic math query for general multiplication."""
        result = self.engine.process_mathematical_query("multiply 23 and 47 using vedic method")
        
        assert result["approach"] == "vedic_math"
        assert result["result"] == 1081
        assert "steps" in result
    
    def test_algebraic_query_solve(self):
        """Test processing algebraic solve query."""
        result = self.engine.process_mathematical_query("solve x + 2 = 5")
        
        assert result["approach"] == "algebraic_solving"
        assert "result" in result
        assert "3" in str(result["result"]["solutions"])
    
    def test_algebraic_query_solve_with_variable(self):
        """Test processing algebraic solve query with specified variable."""
        result = self.engine.process_mathematical_query("solve x + y = 5 for x")
        
        assert result["approach"] == "algebraic_solving"
        assert "result" in result
        assert result["result"]["variable"] == "x"
    
    def test_algebraic_query_simplify(self):
        """Test processing algebraic simplify query."""
        result = self.engine.process_mathematical_query("simplify x + x + x")
        
        assert result["approach"] == "algebraic_simplify"
        assert "3*x" in result["result"]
        assert "steps" in result
    
    def test_algebraic_query_expand(self):
        """Test processing algebraic expand query."""
        result = self.engine.process_mathematical_query("expand (x + 1)*(x + 2)")
        
        assert result["approach"] == "algebraic_expand"
        assert "x**2" in result["result"]
        assert "3*x" in result["result"]
    
    def test_algebraic_query_factor(self):
        """Test processing algebraic factor query."""
        result = self.engine.process_mathematical_query("factor x**2 + 3*x + 2")
        
        assert result["approach"] == "algebraic_factor"
        assert "x + 1" in result["result"]
        assert "x + 2" in result["result"]
    
    def test_proof_query(self):
        """Test processing proof verification query."""
        result = self.engine.process_mathematical_query("prove (x + 1)**2 = x**2 + 2*x + 1")
        
        assert result["approach"] == "proof_verification"
        assert "result" in result
        assert result["result"]["valid"] is True
    
    def test_general_query(self):
        """Test processing general mathematical query."""
        result = self.engine.process_mathematical_query("what is mathematics?")
        
        assert result["approach"] == "general"
        assert "not implemented" in result["result"]
    
    def test_computation_statistics(self):
        """Test getting computation statistics."""
        # Process some queries first
        self.engine.process_mathematical_query("square 25 using vedic method")
        self.engine.process_mathematical_query("solve x + 2 = 5")
        self.engine.process_mathematical_query("prove x = x")
        
        stats = self.engine.get_computation_statistics()
        
        assert stats["total_queries"] == 3
        assert "approach_distribution" in stats
        assert "average_confidence" in stats
        assert "available_sutras" in stats
        assert len(stats["available_sutras"]) > 0


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_symbolic_computation_engine(self):
        """Test creating symbolic computation engine."""
        engine = create_symbolic_computation_engine()
        
        assert isinstance(engine, SymbolicComputationEngine)
        assert engine.reasoning_core is None
        
        # Test with mock reasoning core
        mock_reasoning_core = Mock()
        engine_with_core = create_symbolic_computation_engine(mock_reasoning_core)
        assert engine_with_core.reasoning_core == mock_reasoning_core
    
    def test_apply_vedic_math(self):
        """Test applying Vedic math directly."""
        result = apply_vedic_math("ekadhikena_purvena", 15)
        
        assert result["result"] == 225
        assert "steps" in result
        assert result["sutra"] == "ekadhikena_purvena"
    
    def test_verify_mathematical_identity_valid(self):
        """Test verifying valid mathematical identity."""
        result = verify_mathematical_identity("(x + 1)**2", "x**2 + 2*x + 1")
        assert result is True
    
    def test_verify_mathematical_identity_invalid(self):
        """Test verifying invalid mathematical identity."""
        result = verify_mathematical_identity("x + 1", "x + 2")
        assert result is False


@pytest.mark.skipif(SYMPY_AVAILABLE, reason="SymPy is available")
class TestWithoutSymPy:
    """Test behavior when SymPy is not available."""
    
    def test_sutra_to_math_mapper_without_sympy(self):
        """Test that SutraToMathMapper raises ImportError without SymPy."""
        with pytest.raises(ImportError, match="SymPy is required"):
            SutraToMathMapper()
    
    def test_symbolic_algebra_engine_without_sympy(self):
        """Test that SymbolicAlgebraEngine raises ImportError without SymPy."""
        with pytest.raises(ImportError, match="SymPy is required"):
            SymbolicAlgebraEngine()
    
    def test_mathematical_proof_verifier_without_sympy(self):
        """Test that MathematicalProofVerifier raises ImportError without SymPy."""
        with pytest.raises(ImportError, match="SymPy is required"):
            MathematicalProofVerifier()
    
    def test_symbolic_computation_engine_without_sympy(self):
        """Test that SymbolicComputationEngine raises ImportError without SymPy."""
        with pytest.raises(ImportError, match="SymPy is required"):
            SymbolicComputationEngine()


class TestMathematicalTransformation:
    """Test MathematicalTransformation dataclass."""
    
    def test_mathematical_transformation_creation(self):
        """Test creating a MathematicalTransformation instance."""
        transformation = MathematicalTransformation(
            source_sutra="test_sutra",
            input_expression="x + x",
            output_expression="2*x",
            transformation_type="simplification",
            verification_steps=["combine like terms"],
            confidence=0.95
        )
        
        assert transformation.source_sutra == "test_sutra"
        assert transformation.input_expression == "x + x"
        assert transformation.output_expression == "2*x"
        assert transformation.transformation_type == "simplification"
        assert len(transformation.verification_steps) == 1
        assert transformation.confidence == 0.95


class TestMathematicalDomain:
    """Test MathematicalDomain enum."""
    
    def test_mathematical_domain_values(self):
        """Test MathematicalDomain enum values."""
        assert MathematicalDomain.ARITHMETIC.value == "arithmetic"
        assert MathematicalDomain.ALGEBRA.value == "algebra"
        assert MathematicalDomain.GEOMETRY.value == "geometry"
        assert MathematicalDomain.TRIGONOMETRY.value == "trigonometry"
        assert MathematicalDomain.CALCULUS.value == "calculus"
        assert MathematicalDomain.NUMBER_THEORY.value == "number_theory"
        assert MathematicalDomain.COMBINATORICS.value == "combinatorics"
        assert MathematicalDomain.VEDIC_MATH.value == "vedic_math"


if __name__ == "__main__":
    pytest.main([__file__])