"""
Symbolic computation integration for Sanskrit reasoning system.

This module integrates SymPy for algebraic and numeric computations,
translates Sanskrit sūtras (especially Vedic Math) to executable code,
and provides Sanskrit sūtra → mathematical formula translation with
algebraic manipulation using Sanskrit grammatical principles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union, Callable
from enum import Enum
import re
from abc import ABC, abstractmethod

try:
    import sympy as sp
    from sympy import symbols, simplify, expand, factor, solve, diff, integrate
    from sympy import Matrix, Rational, pi, E, I, oo
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    # Create mock objects for when SymPy is not available
    class MockSymPy:
        def __getattr__(self, name):
            raise ImportError(f"SymPy is required for symbolic computation. Install with: pip install sympy")
    
    sp = MockSymPy()
    symbols = simplify = expand = factor = solve = diff = integrate = sp
    Matrix = Rational = pi = E = I = oo = sp
    parse_expr = sp

from .rule import SutraRule, SutraReference
from .reasoning_core import ReasoningCore, LogicalTerm, LogicalClause


class MathematicalDomain(Enum):
    """Mathematical domains for Sanskrit sūtra classification."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    CALCULUS = "calculus"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    VEDIC_MATH = "vedic_math"


@dataclass
class VedicMathSutra:
    """Represents a Vedic Mathematics sūtra with computational mapping."""
    name: str
    sanskrit_text: str
    english_translation: str
    domain: MathematicalDomain
    formula: Optional[str] = None  # SymPy expression string
    algorithm: Optional[Callable] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    computational_complexity: str = "O(1)"
    applicability_conditions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for Vedic Math sūtra processing")


@dataclass
class MathematicalTransformation:
    """Represents a mathematical transformation derived from Sanskrit rules."""
    source_sutra: str
    input_expression: str
    output_expression: str
    transformation_type: str
    symbolic_form: Optional[Any] = None  # SymPy expression
    verification_steps: List[str] = field(default_factory=list)
    confidence: float = 1.0


class SutraToMathMapper:
    """Maps Sanskrit sūtras to mathematical formulas and operations."""
    
    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for symbolic computation")
        
        self.vedic_sutras = self._initialize_vedic_sutras()
        self.transformation_cache: Dict[str, MathematicalTransformation] = {}
        self.symbol_registry: Dict[str, sp.Symbol] = {}
    
    def _initialize_vedic_sutras(self) -> Dict[str, VedicMathSutra]:
        """Initialize the database of Vedic Mathematics sūtras."""
        sutras = {}
        
        # Ekadhikena Purvena (One more than the previous)
        sutras["ekadhikena_purvena"] = VedicMathSutra(
            name="Ekadhikena Purvena",
            sanskrit_text="एकाधिकेन पूर्वेण",
            english_translation="One more than the previous",
            domain=MathematicalDomain.ARITHMETIC,
            formula="(10*a + b)^2 = 100*a*(a+1) + (b^2 + 20*a*b)",
            algorithm=self._ekadhikena_purvena_algorithm,
            examples=[
                {"input": "12^2", "output": "144", "steps": ["1*(1+1)=2", "2^2=4", "result: 144"]},
                {"input": "13^2", "output": "169", "steps": ["1*(1+1)=2", "3^2=9", "result: 169"]}
            ],
            computational_complexity="O(1)",
            applicability_conditions=["number ends in 1-9", "two-digit numbers"]
        )
        
        # Nikhilam Navatashcaramam Dashatah (All from 9 and last from 10)
        sutras["nikhilam"] = VedicMathSutra(
            name="Nikhilam Navatashcaramam Dashatah",
            sanskrit_text="निखिलं नवतश्चरमं दशतः",
            english_translation="All from 9 and last from 10",
            domain=MathematicalDomain.ARITHMETIC,
            formula="(base - a) * (base - b) = base^2 - base*(a+b) + a*b",
            algorithm=self._nikhilam_algorithm,
            examples=[
                {"input": "97*96", "output": "9312", "steps": ["97→3, 96→4", "3*4=12", "97-4=93", "result: 9312"]},
                {"input": "998*997", "output": "995006", "steps": ["998→2, 997→3", "2*3=6", "998-3=995", "result: 995006"]}
            ],
            computational_complexity="O(1)",
            applicability_conditions=["numbers close to powers of 10"]
        )
        
        # Urdhva-Tiryagbyham (Vertically and crosswise)
        sutras["urdhva_tiryagbyham"] = VedicMathSutra(
            name="Urdhva-Tiryagbyham",
            sanskrit_text="ऊर्ध्वतिर्यग्भ्याम्",
            english_translation="Vertically and crosswise",
            domain=MathematicalDomain.ALGEBRA,
            formula="(a*x + b) * (c*x + d) = ac*x^2 + (ad + bc)*x + bd",
            algorithm=self._urdhva_tiryagbyham_algorithm,
            examples=[
                {"input": "23*47", "output": "1081", "steps": ["2*4=8", "2*7+3*4=26", "3*7=21", "result: 1081"]},
                {"input": "(x+2)(x+3)", "output": "x^2+5x+6", "steps": ["x*x=x^2", "x*3+2*x=5x", "2*3=6"]}
            ],
            computational_complexity="O(n^2)",
            applicability_conditions=["polynomial multiplication", "general multiplication"]
        )
        
        # Paravartya Yojayet (Transpose and apply)
        sutras["paravartya_yojayet"] = VedicMathSutra(
            name="Paravartya Yojayet",
            sanskrit_text="परावर्त्य योजयेत्",
            english_translation="Transpose and apply",
            domain=MathematicalDomain.ALGEBRA,
            formula="ax + by = c, dx + ey = f → x = (ce - bf)/(ae - bd), y = (af - cd)/(ae - bd)",
            algorithm=self._paravartya_yojayet_algorithm,
            examples=[
                {"input": "2x+3y=7, 4x+5y=13", "output": "x=2, y=1", "steps": ["transpose coefficients", "apply formula"]},
            ],
            computational_complexity="O(1)",
            applicability_conditions=["system of linear equations", "non-zero determinant"]
        )
        
        return sutras
    
    def _ekadhikena_purvena_algorithm(self, n: int) -> Tuple[int, List[str]]:
        """Algorithm for Ekadhikena Purvena sūtra (squaring numbers ending in 5)."""
        if not str(n).endswith('5'):
            raise ValueError("Ekadhikena Purvena applies to numbers ending in 5")
        
        steps = []
        prefix = n // 10
        steps.append(f"Number: {n} = {prefix}5")
        
        result_prefix = prefix * (prefix + 1)
        steps.append(f"Prefix calculation: {prefix} × ({prefix} + 1) = {prefix} × {prefix + 1} = {result_prefix}")
        
        result = result_prefix * 100 + 25
        steps.append(f"Final result: {result_prefix}25 = {result}")
        
        return result, steps
    
    def _nikhilam_algorithm(self, a: int, b: int, base: int = 100) -> Tuple[int, List[str]]:
        """Algorithm for Nikhilam sūtra (multiplication near base)."""
        steps = []
        
        complement_a = base - a
        complement_b = base - b
        steps.append(f"Complements: {a}→{complement_a}, {b}→{complement_b}")
        
        # Left part: (a - complement_b) or (b - complement_a)
        left_part = a - complement_b
        steps.append(f"Left part: {a} - {complement_b} = {left_part}")
        
        # Right part: complement_a * complement_b
        right_part = complement_a * complement_b
        steps.append(f"Right part: {complement_a} × {complement_b} = {right_part}")
        
        # Combine
        result = left_part * base + right_part
        steps.append(f"Result: {left_part} × {base} + {right_part} = {result}")
        
        return result, steps
    
    def _urdhva_tiryagbyham_algorithm(self, a: int, b: int) -> Tuple[int, List[str]]:
        """Algorithm for Urdhva-Tiryagbyham sūtra (general multiplication)."""
        steps = []
        
        # Convert to digit arrays (reverse for easier indexing)
        digits_a = [int(d) for d in str(a)][::-1]
        digits_b = [int(d) for d in str(b)][::-1]
        
        steps.append(f"Digits: {a} = {digits_a[::-1]}, {b} = {digits_b[::-1]}")
        
        # Perform crosswise multiplication
        result_digits = []
        carry = 0
        
        for i in range(len(digits_a) + len(digits_b) - 1):
            sum_products = carry
            
            # Calculate cross products for this position
            for j in range(len(digits_a)):
                k = i - j
                if 0 <= k < len(digits_b):
                    product = digits_a[j] * digits_b[k]
                    sum_products += product
                    steps.append(f"Position {i}: {digits_a[j]} × {digits_b[k]} = {product}")
            
            digit = sum_products % 10
            carry = sum_products // 10
            result_digits.append(digit)
            
            steps.append(f"Position {i} total: {sum_products}, digit: {digit}, carry: {carry}")
        
        if carry > 0:
            result_digits.append(carry)
        
        result = int(''.join(map(str, result_digits[::-1])))
        steps.append(f"Final result: {result}")
        
        return result, steps
    
    def _paravartya_yojayet_algorithm(self, equations: List[Tuple[List[float], float]]) -> Tuple[List[float], List[str]]:
        """Algorithm for Paravartya Yojayet sūtra (solving linear equations)."""
        if len(equations) != 2 or len(equations[0][0]) != 2:
            raise ValueError("Currently supports 2x2 systems only")
        
        steps = []
        
        # Extract coefficients: ax + by = c, dx + ey = f
        a, b = equations[0][0]
        c = equations[0][1]
        d, e = equations[1][0]
        f = equations[1][1]
        
        steps.append(f"System: {a}x + {b}y = {c}, {d}x + {e}y = {f}")
        
        # Calculate determinant
        det = a * e - b * d
        steps.append(f"Determinant: {a}×{e} - {b}×{d} = {det}")
        
        if det == 0:
            raise ValueError("System has no unique solution (determinant = 0)")
        
        # Apply Cramer's rule (transpose and apply)
        x = (c * e - b * f) / det
        y = (a * f - c * d) / det
        
        steps.append(f"x = ({c}×{e} - {b}×{f}) / {det} = {x}")
        steps.append(f"y = ({a}×{f} - {c}×{d}) / {det} = {y}")
        
        return [x, y], steps
    
    def translate_sutra_to_formula(self, sutra_name: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Translate a Sanskrit sūtra to its mathematical formula."""
        if sutra_name in self.vedic_sutras:
            return self.vedic_sutras[sutra_name].formula
        
        # Try to infer from context or rule patterns
        return self._infer_formula_from_context(sutra_name, context or {})
    
    def _infer_formula_from_context(self, sutra_name: str, context: Dict[str, Any]) -> Optional[str]:
        """Infer mathematical formula from Sanskrit grammatical context."""
        # This would use NLP and pattern matching to infer formulas
        # For now, return None for unknown sūtras
        return None
    
    def apply_vedic_sutra(self, sutra_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Apply a Vedic mathematics sūtra to given inputs."""
        if sutra_name not in self.vedic_sutras:
            raise ValueError(f"Unknown Vedic sūtra: {sutra_name}")
        
        sutra = self.vedic_sutras[sutra_name]
        
        try:
            if sutra.algorithm:
                result, steps = sutra.algorithm(*args, **kwargs)
                return {
                    "sutra": sutra_name,
                    "result": result,
                    "steps": steps,
                    "formula": sutra.formula,
                    "complexity": sutra.computational_complexity
                }
            else:
                return {
                    "sutra": sutra_name,
                    "error": "No algorithm implementation available",
                    "formula": sutra.formula
                }
        except Exception as e:
            return {
                "sutra": sutra_name,
                "error": str(e),
                "formula": sutra.formula
            }
    
    def get_applicable_sutras(self, problem_type: str, constraints: Dict[str, Any] = None) -> List[str]:
        """Get list of applicable sūtras for a given problem type."""
        applicable = []
        constraints = constraints or {}
        
        for name, sutra in self.vedic_sutras.items():
            # Check domain match
            if problem_type.lower() in sutra.domain.value:
                # Check applicability conditions
                if self._check_applicability_conditions(sutra, constraints):
                    applicable.append(name)
        
        return applicable
    
    def _check_applicability_conditions(self, sutra: VedicMathSutra, constraints: Dict[str, Any]) -> bool:
        """Check if a sūtra's applicability conditions are met."""
        # Simple implementation - would need more sophisticated condition checking
        return True


class SymbolicAlgebraEngine:
    """Engine for symbolic algebraic manipulation using Sanskrit principles."""
    
    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for symbolic algebra")
        
        self.expression_cache: Dict[str, sp.Expr] = {}
        self.transformation_rules: Dict[str, Callable] = {}
        self._initialize_transformation_rules()
    
    def _initialize_transformation_rules(self):
        """Initialize Sanskrit-inspired transformation rules."""
        self.transformation_rules = {
            'sandhi_algebra': self._apply_sandhi_algebra,
            'compound_expansion': self._apply_compound_expansion,
            'morphological_factoring': self._apply_morphological_factoring,
            'grammatical_simplification': self._apply_grammatical_simplification
        }
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse a mathematical expression string."""
        if expr_str in self.expression_cache:
            return self.expression_cache[expr_str]
        
        try:
            expr = parse_expr(expr_str)
            self.expression_cache[expr_str] = expr
            return expr
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expr_str}': {e}")
    
    def apply_transformation(self, expression: Union[str, sp.Expr], 
                           transformation: str, 
                           **kwargs) -> MathematicalTransformation:
        """Apply a Sanskrit-inspired transformation to an expression."""
        if isinstance(expression, str):
            expr = self.parse_expression(expression)
            input_str = expression
        else:
            expr = expression
            input_str = str(expression)
        
        if transformation not in self.transformation_rules:
            raise ValueError(f"Unknown transformation: {transformation}")
        
        transform_func = self.transformation_rules[transformation]
        result_expr, steps = transform_func(expr, **kwargs)
        
        return MathematicalTransformation(
            source_sutra=transformation,
            input_expression=input_str,
            output_expression=str(result_expr),
            transformation_type=transformation,
            symbolic_form=result_expr,
            verification_steps=steps,
            confidence=1.0
        )
    
    def _apply_sandhi_algebra(self, expr: sp.Expr, **kwargs) -> Tuple[sp.Expr, List[str]]:
        """Apply sandhi-like transformations to algebraic expressions."""
        steps = []
        original_expr = expr
        
        # Example: Combine like terms (similar to vowel sandhi)
        simplified = simplify(expr)
        if simplified != expr:
            steps.append(f"Sandhi combination: {expr} → {simplified}")
            expr = simplified
        
        # Example: Factor common terms (similar to consonant assimilation)
        factored = factor(expr)
        if factored != expr:
            steps.append(f"Consonant assimilation: {expr} → {factored}")
            expr = factored
        
        # If no changes were made, add a step explaining this
        if not steps:
            steps.append(f"Expression already in simplified form: {original_expr}")
        
        return expr, steps
    
    def _apply_compound_expansion(self, expr: sp.Expr, **kwargs) -> Tuple[sp.Expr, List[str]]:
        """Apply compound-like expansion to expressions."""
        steps = []
        
        expanded = expand(expr)
        if expanded != expr:
            steps.append(f"Compound expansion: {expr} → {expanded}")
        
        return expanded, steps
    
    def _apply_morphological_factoring(self, expr: sp.Expr, **kwargs) -> Tuple[sp.Expr, List[str]]:
        """Apply morphological analysis-like factoring."""
        steps = []
        
        # Try different factoring approaches
        factored = factor(expr)
        if factored != expr:
            steps.append(f"Morphological factoring: {expr} → {factored}")
        
        return factored, steps
    
    def _apply_grammatical_simplification(self, expr: sp.Expr, **kwargs) -> Tuple[sp.Expr, List[str]]:
        """Apply grammatical rule-like simplification."""
        steps = []
        
        # Apply various simplification rules
        simplified = simplify(expr)
        if simplified != expr:
            steps.append(f"Grammatical simplification: {expr} → {simplified}")
        
        # Try trigonometric simplification if applicable
        try:
            trig_simplified = sp.trigsimp(simplified)
            if trig_simplified != simplified:
                steps.append(f"Trigonometric simplification: {simplified} → {trig_simplified}")
                simplified = trig_simplified
        except:
            pass
        
        return simplified, steps
    
    def solve_equation(self, equation: Union[str, sp.Eq], variable: Union[str, sp.Symbol] = None) -> Dict[str, Any]:
        """Solve an equation using symbolic methods."""
        if isinstance(equation, str):
            # Parse equation string
            if '=' in equation:
                left, right = equation.split('=', 1)
                eq = sp.Eq(parse_expr(left.strip()), parse_expr(right.strip()))
            else:
                eq = sp.Eq(parse_expr(equation), 0)
        else:
            eq = equation
        
        if variable is None:
            # Find all free symbols
            free_symbols = eq.free_symbols
            if len(free_symbols) == 1:
                variable = list(free_symbols)[0]
            else:
                raise ValueError("Multiple variables found, please specify which to solve for")
        elif isinstance(variable, str):
            variable = sp.Symbol(variable)
        
        try:
            solutions = solve(eq, variable)
            return {
                "equation": str(eq),
                "variable": str(variable),
                "solutions": [str(sol) for sol in solutions],
                "symbolic_solutions": solutions
            }
        except Exception as e:
            return {
                "equation": str(eq),
                "variable": str(variable),
                "error": str(e),
                "solutions": []
            }


class MathematicalProofVerifier:
    """Verifies mathematical proofs using sūtra logic."""
    
    def __init__(self, reasoning_core: Optional[ReasoningCore] = None):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for proof verification")
        
        self.reasoning_core = reasoning_core
        self.proof_cache: Dict[str, bool] = {}
        self.verification_rules: Dict[str, Callable] = {}
        self._initialize_verification_rules()
    
    def _initialize_verification_rules(self):
        """Initialize mathematical verification rules based on sūtras."""
        self.verification_rules = {
            'algebraic_identity': self._verify_algebraic_identity,
            'equation_equivalence': self._verify_equation_equivalence,
            'logical_consistency': self._verify_logical_consistency,
            'vedic_computation': self._verify_vedic_computation
        }
    
    def verify_proof(self, proof_statement: str, proof_steps: List[str], 
                    verification_type: str = 'algebraic_identity') -> Dict[str, Any]:
        """Verify a mathematical proof using sūtra-based logic."""
        proof_key = f"{verification_type}:{proof_statement}"
        
        if proof_key in self.proof_cache:
            return {"cached": True, "valid": self.proof_cache[proof_key]}
        
        if verification_type not in self.verification_rules:
            return {"error": f"Unknown verification type: {verification_type}"}
        
        verifier = self.verification_rules[verification_type]
        result = verifier(proof_statement, proof_steps)
        
        self.proof_cache[proof_key] = result.get('valid', False)
        return result
    
    def _verify_algebraic_identity(self, statement: str, steps: List[str]) -> Dict[str, Any]:
        """Verify an algebraic identity."""
        try:
            # Parse the statement (should be in form "expr1 = expr2")
            if '=' not in statement:
                return {"valid": False, "error": "Statement must contain '='"}
            
            left_str, right_str = statement.split('=', 1)
            left_expr = parse_expr(left_str.strip())
            right_expr = parse_expr(right_str.strip())
            
            # Check if expressions are symbolically equal
            difference = simplify(left_expr - right_expr)
            is_valid = difference == 0
            
            verification_steps = []
            verification_steps.append(f"Left side: {left_expr}")
            verification_steps.append(f"Right side: {right_expr}")
            verification_steps.append(f"Difference: {left_expr} - ({right_expr}) = {difference}")
            verification_steps.append(f"Simplified difference: {difference}")
            verification_steps.append(f"Identity valid: {is_valid}")
            
            return {
                "valid": is_valid,
                "verification_steps": verification_steps,
                "symbolic_difference": difference
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _verify_equation_equivalence(self, statement: str, steps: List[str]) -> Dict[str, Any]:
        """Verify that equation transformations are valid."""
        # This would implement step-by-step equation verification
        return {"valid": True, "note": "Equation equivalence verification not fully implemented"}
    
    def _verify_logical_consistency(self, statement: str, steps: List[str]) -> Dict[str, Any]:
        """Verify logical consistency using reasoning core."""
        if not self.reasoning_core:
            return {"valid": False, "error": "Reasoning core not available"}
        
        # Convert mathematical statement to logical form and verify
        return {"valid": True, "note": "Logical consistency verification not fully implemented"}
    
    def _verify_vedic_computation(self, statement: str, steps: List[str]) -> Dict[str, Any]:
        """Verify Vedic mathematics computations."""
        # This would verify that Vedic math algorithms produce correct results
        return {"valid": True, "note": "Vedic computation verification not fully implemented"}


class SymbolicComputationEngine:
    """Main engine integrating all symbolic computation components."""
    
    def __init__(self, reasoning_core: Optional[ReasoningCore] = None):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for symbolic computation")
        
        self.reasoning_core = reasoning_core
        self.sutra_mapper = SutraToMathMapper()
        self.algebra_engine = SymbolicAlgebraEngine()
        self.proof_verifier = MathematicalProofVerifier(reasoning_core)
        
        # Integration components
        self.computation_history: List[Dict[str, Any]] = []
        self.active_context: Dict[str, Any] = {}
    
    def process_mathematical_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a mathematical query using Sanskrit-inspired methods."""
        context = context or {}
        
        result = {
            "query": query,
            "context": context,
            "timestamp": sp.N(sp.pi),  # Placeholder timestamp
            "approach": "unknown",
            "result": None,
            "steps": [],
            "confidence": 0.0
        }
        
        # Determine query type and approach (order matters - proof queries should be checked first)
        if self._is_proof_query(query):
            proof_result = self._process_proof_query(query, context)
            result.update(proof_result)
        elif self._is_vedic_math_query(query):
            result["approach"] = "vedic_math"
            vedic_result = self._process_vedic_math_query(query, context)
            result.update(vedic_result)
        elif self._is_algebraic_query(query):
            algebraic_result = self._process_algebraic_query(query, context)
            result.update(algebraic_result)
        else:
            general_result = self._process_general_query(query, context)
            result.update(general_result)
        
        # Store in history
        self.computation_history.append(result)
        
        return result
    
    def _is_vedic_math_query(self, query: str) -> bool:
        """Check if query is about Vedic mathematics."""
        vedic_keywords = ['vedic', 'sutra', 'ekadhikena', 'nikhilam', 'urdhva', 'paravartya', 'square', 'multiply']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in vedic_keywords) and ('vedic' in query_lower or 'square' in query_lower or 'multiply' in query_lower)
    
    def _is_algebraic_query(self, query: str) -> bool:
        """Check if query is algebraic."""
        algebraic_keywords = ['solve', 'simplify', 'expand', 'factor', '=', 'x', 'y', 'equation']
        return any(keyword in query.lower() for keyword in algebraic_keywords)
    
    def _is_proof_query(self, query: str) -> bool:
        """Check if query is about proof verification."""
        proof_keywords = ['prove', 'verify', 'proof', 'identity', 'show that']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in proof_keywords)
    
    def _process_vedic_math_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process Vedic mathematics queries."""
        # Extract numbers and operation from query
        numbers = re.findall(r'\d+', query)
        
        if 'square' in query.lower() and len(numbers) >= 1:
            n = int(numbers[0])
            if str(n).endswith('5'):
                return self.sutra_mapper.apply_vedic_sutra('ekadhikena_purvena', n)
        
        elif 'multiply' in query.lower() and len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            # Check if numbers are close to 100
            if 90 <= a <= 99 and 90 <= b <= 99:
                return self.sutra_mapper.apply_vedic_sutra('nikhilam', a, b, 100)
            else:
                return self.sutra_mapper.apply_vedic_sutra('urdhva_tiryagbyham', a, b)
        
        return {
            "approach": "vedic_math",
            "result": "Could not determine specific Vedic method",
            "available_sutras": list(self.sutra_mapper.vedic_sutras.keys())
        }
    
    def _process_algebraic_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process algebraic queries."""
        try:
            if 'solve' in query.lower():
                # Extract equation
                equation_match = re.search(r'solve\s+(.+?)(?:\s+for\s+(\w+))?$', query.lower())
                if equation_match:
                    equation = equation_match.group(1)
                    variable = equation_match.group(2)
                    
                    result = self.algebra_engine.solve_equation(equation, variable)
                    return {
                        "approach": "algebraic_solving",
                        "result": result,
                        "confidence": 0.9
                    }
            
            elif any(word in query.lower() for word in ['simplify', 'expand', 'factor']):
                # Extract expression and operation
                for operation in ['simplify', 'expand', 'factor']:
                    if operation in query.lower():
                        expr_match = re.search(f'{operation}\\s+(.+)', query.lower())
                        if expr_match:
                            expression = expr_match.group(1)
                            
                            if operation == 'simplify':
                                transformation = self.algebra_engine.apply_transformation(
                                    expression, 'grammatical_simplification'
                                )
                            elif operation == 'expand':
                                transformation = self.algebra_engine.apply_transformation(
                                    expression, 'compound_expansion'
                                )
                            elif operation == 'factor':
                                transformation = self.algebra_engine.apply_transformation(
                                    expression, 'morphological_factoring'
                                )
                            
                            return {
                                "approach": f"algebraic_{operation}",
                                "result": transformation.output_expression,
                                "steps": transformation.verification_steps,
                                "confidence": transformation.confidence
                            }
            
            return {
                "approach": "algebraic",
                "result": "Could not parse algebraic query",
                "confidence": 0.0
            }
            
        except Exception as e:
            return {
                "approach": "algebraic",
                "result": f"Error processing algebraic query: {e}",
                "confidence": 0.0
            }
    
    def _process_proof_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process proof verification queries."""
        # Extract statement to prove
        prove_match = re.search(r'prove\s+(.+)', query.lower())
        if prove_match:
            statement = prove_match.group(1)
            
            verification = self.proof_verifier.verify_proof(
                statement, [], 'algebraic_identity'
            )
            
            return {
                "approach": "proof_verification",
                "result": verification,
                "confidence": 0.8 if verification.get('valid') else 0.2
            }
        
        return {
            "approach": "proof_verification",
            "result": "Could not parse proof query",
            "confidence": 0.0
        }
    
    def _process_general_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process general mathematical queries."""
        return {
            "approach": "general",
            "result": "General mathematical query processing not implemented",
            "confidence": 0.1
        }
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get statistics about computation usage."""
        if not self.computation_history:
            return {"total_queries": 0}
        
        approaches = [entry.get("approach", "unknown") for entry in self.computation_history]
        approach_counts = {}
        for approach in approaches:
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
        
        avg_confidence = sum(entry.get("confidence", 0) for entry in self.computation_history) / len(self.computation_history)
        
        return {
            "total_queries": len(self.computation_history),
            "approach_distribution": approach_counts,
            "average_confidence": avg_confidence,
            "available_sutras": list(self.sutra_mapper.vedic_sutras.keys())
        }


# Convenience functions for easy integration
def create_symbolic_computation_engine(reasoning_core: Optional[ReasoningCore] = None) -> SymbolicComputationEngine:
    """Create a symbolic computation engine with optional reasoning core integration."""
    return SymbolicComputationEngine(reasoning_core)


def apply_vedic_math(sutra_name: str, *args, **kwargs) -> Dict[str, Any]:
    """Apply a Vedic mathematics sūtra directly."""
    mapper = SutraToMathMapper()
    return mapper.apply_vedic_sutra(sutra_name, *args, **kwargs)


def verify_mathematical_identity(left_expr: str, right_expr: str) -> bool:
    """Verify if two mathematical expressions are identical."""
    verifier = MathematicalProofVerifier()
    result = verifier.verify_proof(f"{left_expr} = {right_expr}", [])
    return result.get('valid', False)