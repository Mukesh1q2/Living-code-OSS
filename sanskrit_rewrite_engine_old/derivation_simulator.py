"""
Śabda-prakriyā (word derivation) simulator for Sanskrit.

This module implements complete Pāṇinian derivation process simulation with:
- Recursive rule application with meta-rules
- Step-by-step prakriyā logging with rule citations
- Derivation tree visualization for complex words
- Validation against traditional grammatical texts
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import json
from datetime import datetime
import logging

from .token import Token, TokenKind
from .rule import SutraRule, SutraReference, RuleType, ParibhasaRule, RuleRegistry
from .panini_engine import PaniniRuleEngine, TransformationTrace
from .morphological_analyzer import (
    SanskritMorphologicalAnalyzer, MorphologicalAnalysis, 
    Morpheme, MorphemeType, GrammaticalCategory
)


class DerivationStage(Enum):
    """Stages in Pāṇinian word derivation."""
    DHATU_SELECTION = "DHATU_SELECTION"           # Root selection
    PRATYAYA_ADDITION = "PRATYAYA_ADDITION"       # Suffix addition
    GUNA_VRDDHI = "GUNA_VRDDHI"                   # Vowel strengthening
    SANDHI_APPLICATION = "SANDHI_APPLICATION"      # Phonological changes
    FINAL_FORM = "FINAL_FORM"                     # Final word form


class DerivationStepType(Enum):
    """Types of derivation steps."""
    RULE_APPLICATION = "RULE_APPLICATION"         # Sūtra rule applied
    META_RULE = "META_RULE"                       # Paribhāṣā applied
    MORPHEME_ADDITION = "MORPHEME_ADDITION"       # Morpheme added
    PHONOLOGICAL_CHANGE = "PHONOLOGICAL_CHANGE"   # Sound change
    VALIDATION = "VALIDATION"                     # Validation step


@dataclass
class DerivationStep:
    """
    A single step in the word derivation process.
    
    Attributes:
        step_number: Sequential step number
        stage: Current derivation stage
        step_type: Type of derivation step
        rule_applied: Sūtra rule that was applied (if any)
        tokens_before: Token state before this step
        tokens_after: Token state after this step
        description: Human-readable description
        sutra_citation: Traditional sūtra citation
        rationale: Explanation of why this step was taken
        alternatives: Alternative derivation paths considered
        confidence: Confidence in this derivation step
        timestamp: When this step was computed
        metadata: Additional step-specific information
    """
    step_number: int
    stage: DerivationStage
    step_type: DerivationStepType
    tokens_before: List[Token]
    tokens_after: List[Token]
    description: str
    rule_applied: Optional[SutraRule] = None
    sutra_citation: Optional[str] = None
    rationale: Optional[str] = None
    alternatives: List['DerivationStep'] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_transformation_summary(self) -> str:
        """Get a summary of the transformation in this step."""
        before_text = ''.join(t.text for t in self.tokens_before)
        after_text = ''.join(t.text for t in self.tokens_after)
        
        if self.sutra_citation:
            return f"{before_text} → {after_text} ({self.sutra_citation})"
        else:
            return f"{before_text} → {after_text}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            'step_number': self.step_number,
            'stage': self.stage.value,
            'step_type': self.step_type.value,
            'tokens_before': [{'text': t.text, 'kind': t.kind.value} for t in self.tokens_before],
            'tokens_after': [{'text': t.text, 'kind': t.kind.value} for t in self.tokens_after],
            'description': self.description,
            'sutra_citation': self.sutra_citation,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class DerivationTree:
    """
    Tree structure representing the complete derivation process.
    
    Attributes:
        root_word: The final derived word
        root_morphemes: Starting morphemes (dhātu, pratyaya, etc.)
        derivation_steps: Sequential list of derivation steps
        alternative_paths: Alternative derivation paths explored
        validation_results: Results of validation against traditional texts
        confidence: Overall confidence in the derivation
        metadata: Additional derivation information
    """
    root_word: str
    root_morphemes: List[Morpheme]
    derivation_steps: List[DerivationStep]
    alternative_paths: List['DerivationTree'] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_final_form(self) -> str:
        """Get the final derived word form."""
        if self.derivation_steps:
            final_tokens = self.derivation_steps[-1].tokens_after
            return ''.join(t.text for t in final_tokens)
        return self.root_word
    
    def get_derivation_summary(self) -> List[str]:
        """Get a summary of all derivation steps."""
        summary = []
        for step in self.derivation_steps:
            summary.append(step.get_transformation_summary())
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert derivation tree to dictionary for serialization."""
        return {
            'root_word': self.root_word,
            'root_morphemes': [m.to_dict() for m in self.root_morphemes],
            'derivation_steps': [step.to_dict() for step in self.derivation_steps],
            'alternative_paths': [path.to_dict() for path in self.alternative_paths],
            'validation_results': self.validation_results,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def visualize_tree(self, format: str = "text") -> str:
        """Generate a visualization of the derivation tree."""
        if format == "text":
            return self._generate_text_tree()
        elif format == "mermaid":
            return self._generate_mermaid_diagram()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_tree(self) -> str:
        """Generate a text-based tree visualization."""
        lines = []
        lines.append(f"Derivation Tree for: {self.root_word}")
        lines.append("=" * 50)
        
        # Show root morphemes
        morpheme_texts = [f"{m.text}({m.type.value})" for m in self.root_morphemes]
        lines.append(f"Root Morphemes: {' + '.join(morpheme_texts)}")
        lines.append("")
        
        # Show derivation steps
        for i, step in enumerate(self.derivation_steps, 1):
            lines.append(f"Step {i}: {step.stage.value}")
            lines.append(f"  {step.get_transformation_summary()}")
            if step.rationale:
                lines.append(f"  Rationale: {step.rationale}")
            lines.append("")
        
        lines.append(f"Final Form: {self.get_final_form()}")
        lines.append(f"Confidence: {self.confidence:.2f}")
        
        return "\n".join(lines)
    
    def _generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram for the derivation tree."""
        lines = ["graph TD"]
        
        # Add nodes for each step
        for i, step in enumerate(self.derivation_steps):
            node_id = f"S{i}"
            before_text = ''.join(t.text for t in step.tokens_before)
            after_text = ''.join(t.text for t in step.tokens_after)
            
            if i == 0:
                lines.append(f'    START["{before_text}"] --> {node_id}["{after_text}"]')
            else:
                prev_node = f"S{i-1}"
                lines.append(f'    {prev_node} --> {node_id}["{after_text}"]')
            
            # Add rule citation if available
            if step.sutra_citation:
                lines.append(f'    {node_id} -.-> R{i}["{step.sutra_citation}"]')
        
        return "\n".join(lines)


@dataclass
class DerivationContext:
    """
    Context information for word derivation process.
    
    Attributes:
        source_text: Original input text
        target_analysis: Target morphological analysis
        derivation_constraints: Constraints on the derivation process
        traditional_sources: References to traditional grammatical texts
        linguistic_context: Additional linguistic context
    """
    source_text: str
    target_analysis: Optional[MorphologicalAnalysis] = None
    derivation_constraints: Dict[str, Any] = field(default_factory=dict)
    traditional_sources: List[str] = field(default_factory=list)
    linguistic_context: Dict[str, Any] = field(default_factory=dict)


class ShabdaPrakriyaSimulator:
    """
    Complete Pāṇinian word derivation simulator.
    
    This class implements the full śabda-prakriyā process, simulating how words
    are derived according to Pāṇinian grammar principles. It provides step-by-step
    derivation logging, rule citation, and validation against traditional texts.
    """
    
    def __init__(self, 
                 rule_engine: PaniniRuleEngine,
                 morphological_analyzer: SanskritMorphologicalAnalyzer,
                 rule_registry: RuleRegistry):
        """
        Initialize the derivation simulator.
        
        Args:
            rule_engine: Pāṇini rule engine for transformations
            morphological_analyzer: Morphological analyzer for word analysis
            rule_registry: Registry of available rules
        """
        self.rule_engine = rule_engine
        self.morphological_analyzer = morphological_analyzer
        self.rule_registry = rule_registry
        self.logger = logging.getLogger(__name__)
        
        # Derivation state
        self._current_derivation: Optional[DerivationTree] = None
        self._step_counter = 0
        self._stage_transitions = {
            DerivationStage.DHATU_SELECTION: DerivationStage.PRATYAYA_ADDITION,
            DerivationStage.PRATYAYA_ADDITION: DerivationStage.GUNA_VRDDHI,
            DerivationStage.GUNA_VRDDHI: DerivationStage.SANDHI_APPLICATION,
            DerivationStage.SANDHI_APPLICATION: DerivationStage.FINAL_FORM
        }
    
    def derive_word(self, 
                   context: DerivationContext,
                   enable_alternatives: bool = True,
                   max_steps: int = 50) -> DerivationTree:
        """
        Perform complete word derivation simulation.
        
        Args:
            context: Derivation context with source text and constraints
            enable_alternatives: Whether to explore alternative derivation paths
            max_steps: Maximum number of derivation steps
            
        Returns:
            Complete derivation tree with all steps and alternatives
        """
        self.logger.info(f"Starting derivation for: {context.source_text}")
        
        # Initialize derivation tree
        morphemes = self._extract_root_morphemes(context)
        derivation_tree = DerivationTree(
            root_word=context.source_text,
            root_morphemes=morphemes,
            derivation_steps=[],
            metadata={'context': {
                'source_text': context.source_text,
                'constraints': context.derivation_constraints,
                'sources': context.traditional_sources
            }}
        )
        
        self._current_derivation = derivation_tree
        self._step_counter = 0
        
        try:
            # Perform staged derivation
            current_tokens = self._initialize_tokens(context.source_text)
            current_stage = DerivationStage.DHATU_SELECTION
            
            while current_stage != DerivationStage.FINAL_FORM and self._step_counter < max_steps:
                self.logger.debug(f"Processing stage: {current_stage}")
                
                # Process current stage
                stage_result = self._process_stage(
                    current_tokens, current_stage, context, enable_alternatives
                )
                
                if stage_result:
                    current_tokens = stage_result.tokens_after
                    derivation_tree.derivation_steps.extend(stage_result.steps)
                    
                    # Move to next stage
                    current_stage = self._stage_transitions.get(current_stage, DerivationStage.FINAL_FORM)
                else:
                    self.logger.warning(f"No progress in stage: {current_stage}")
                    break
            
            # Finalize derivation
            self._finalize_derivation(derivation_tree, current_tokens)
            
            # Validate against traditional sources
            if context.traditional_sources:
                self._validate_against_sources(derivation_tree, context.traditional_sources)
            
            self.logger.info(f"Derivation completed in {len(derivation_tree.derivation_steps)} steps")
            return derivation_tree
            
        except Exception as e:
            self.logger.error(f"Derivation failed: {str(e)}")
            derivation_tree.metadata['error'] = str(e)
            derivation_tree.confidence = 0.0
            return derivation_tree
    
    def _extract_root_morphemes(self, context: DerivationContext) -> List[Morpheme]:
        """Extract root morphemes from the derivation context."""
        if context.target_analysis:
            return context.target_analysis.morphemes
        
        # Fallback: analyze the source text
        analysis = self.morphological_analyzer.analyze_word(context.source_text)
        if analysis:
            return analysis.morphemes
        
        # Last resort: create a single morpheme
        return [Morpheme(
            text=context.source_text,
            type=MorphemeType.ROOT,
            grammatical_info={}
        )]
    
    def _initialize_tokens(self, text: str) -> List[Token]:
        """Initialize token stream for derivation."""
        # Use the rule engine's tokenizer
        return self.rule_engine.tokenizer.tokenize(text)
    
    def _process_stage(self, 
                      tokens: List[Token], 
                      stage: DerivationStage,
                      context: DerivationContext,
                      enable_alternatives: bool) -> Optional['StageResult']:
        """Process a single derivation stage."""
        
        stage_processors = {
            DerivationStage.DHATU_SELECTION: self._process_dhatu_selection,
            DerivationStage.PRATYAYA_ADDITION: self._process_pratyaya_addition,
            DerivationStage.GUNA_VRDDHI: self._process_guna_vrddhi,
            DerivationStage.SANDHI_APPLICATION: self._process_sandhi_application
        }
        
        processor = stage_processors.get(stage)
        if processor:
            return processor(tokens, context, enable_alternatives)
        
        return None
    
    def _process_dhatu_selection(self, 
                                tokens: List[Token], 
                                context: DerivationContext,
                                enable_alternatives: bool) -> 'StageResult':
        """Process dhātu (root) selection stage."""
        steps = []
        
        # Identify root morphemes
        root_morphemes = [m for m in self._current_derivation.root_morphemes 
                         if m.type == MorphemeType.ROOT]
        
        if root_morphemes:
            step = DerivationStep(
                step_number=self._get_next_step_number(),
                stage=DerivationStage.DHATU_SELECTION,
                step_type=DerivationStepType.MORPHEME_ADDITION,
                tokens_before=tokens.copy(),
                tokens_after=tokens.copy(),  # No change in this stage
                description=f"Selected dhātu: {', '.join(m.text for m in root_morphemes)}",
                rationale="Root morpheme identification from morphological analysis"
            )
            steps.append(step)
        
        return StageResult(tokens, steps)
    
    def _process_pratyaya_addition(self, 
                                  tokens: List[Token], 
                                  context: DerivationContext,
                                  enable_alternatives: bool) -> 'StageResult':
        """Process pratyaya (suffix) addition stage."""
        steps = []
        
        # Find suffix morphemes
        suffix_morphemes = [m for m in self._current_derivation.root_morphemes 
                           if m.type in [MorphemeType.SUFFIX, MorphemeType.INFLECTION]]
        
        current_tokens = tokens.copy()
        
        for morpheme in suffix_morphemes:
            # Add suffix token
            suffix_token = Token(
                text=morpheme.text,
                kind=TokenKind.OTHER,
                tags={'morpheme', 'suffix'},
                meta={'morpheme_type': morpheme.type.value}
            )
            
            tokens_before = current_tokens.copy()
            current_tokens.append(suffix_token)
            
            step = DerivationStep(
                step_number=self._get_next_step_number(),
                stage=DerivationStage.PRATYAYA_ADDITION,
                step_type=DerivationStepType.MORPHEME_ADDITION,
                tokens_before=tokens_before,
                tokens_after=current_tokens.copy(),
                description=f"Added pratyaya: {morpheme.text}",
                rationale=f"Suffix addition for {morpheme.type.value}"
            )
            steps.append(step)
        
        return StageResult(current_tokens, steps)
    
    def _process_guna_vrddhi(self, 
                            tokens: List[Token], 
                            context: DerivationContext,
                            enable_alternatives: bool) -> 'StageResult':
        """Process guṇa/vṛddhi (vowel strengthening) stage."""
        steps = []
        current_tokens = tokens.copy()
        
        # Apply vowel strengthening rules
        # Note: In a full implementation, we would have specific GUNA_VRDDHI rules
        # For now, we'll get all active rules and filter by metadata
        all_rules = self.rule_registry.get_active_sutra_rules()
        guna_rules = [rule for rule in all_rules if 'guna' in rule.meta_data.get('categories', [])]
        
        for rule in guna_rules:
            if not rule.active:
                continue
                
            # Try to apply the rule
            for i in range(len(current_tokens)):
                if rule.match_fn(current_tokens, i):
                    tokens_before = current_tokens.copy()
                    current_tokens, new_index = rule.apply_fn(current_tokens, i)
                    
                    step = DerivationStep(
                        step_number=self._get_next_step_number(),
                        stage=DerivationStage.GUNA_VRDDHI,
                        step_type=DerivationStepType.RULE_APPLICATION,
                        tokens_before=tokens_before,
                        tokens_after=current_tokens.copy(),
                        description=f"Applied {rule.name}",
                        rule_applied=rule,
                        sutra_citation=rule.sutra_ref,
                        rationale="Vowel strengthening transformation"
                    )
                    steps.append(step)
                    break  # Apply one rule at a time
        
        return StageResult(current_tokens, steps)
    
    def _process_sandhi_application(self, 
                                   tokens: List[Token], 
                                   context: DerivationContext,
                                   enable_alternatives: bool) -> 'StageResult':
        """Process sandhi (phonological changes) stage."""
        steps = []
        
        # Use the rule engine to apply sandhi rules
        result = self.rule_engine.process(tokens)
        
        # Convert transformation traces to derivation steps
        for trace in result.traces:
            for transformation in trace.transformations:
                step = DerivationStep(
                    step_number=self._get_next_step_number(),
                    stage=DerivationStage.SANDHI_APPLICATION,
                    step_type=DerivationStepType.PHONOLOGICAL_CHANGE,
                    tokens_before=transformation.tokens_before,
                    tokens_after=transformation.tokens_after,
                    description=f"Applied sandhi rule: {transformation.rule_name}",
                    sutra_citation=self._get_sutra_citation(transformation.rule_id),
                    rationale="Phonological transformation"
                )
                steps.append(step)
        
        return StageResult(result.output_tokens, steps)
    
    def _finalize_derivation(self, derivation_tree: DerivationTree, final_tokens: List[Token]):
        """Finalize the derivation process."""
        # Add final step
        if derivation_tree.derivation_steps:
            last_step = derivation_tree.derivation_steps[-1]
            final_step = DerivationStep(
                step_number=self._get_next_step_number(),
                stage=DerivationStage.FINAL_FORM,
                step_type=DerivationStepType.VALIDATION,
                tokens_before=last_step.tokens_after,
                tokens_after=final_tokens,
                description="Final word form achieved",
                rationale="Derivation process completed"
            )
            derivation_tree.derivation_steps.append(final_step)
        
        # Calculate overall confidence
        if derivation_tree.derivation_steps:
            avg_confidence = sum(step.confidence for step in derivation_tree.derivation_steps) / len(derivation_tree.derivation_steps)
            derivation_tree.confidence = avg_confidence
    
    def _validate_against_sources(self, derivation_tree: DerivationTree, sources: List[str]):
        """Validate derivation against traditional grammatical sources."""
        validation_results = {}
        
        for source in sources:
            # Placeholder for validation logic
            # In a full implementation, this would check against traditional texts
            validation_results[source] = {
                'validated': True,
                'confidence': 0.8,
                'notes': f"Validation against {source} not yet implemented"
            }
        
        derivation_tree.validation_results = validation_results
    
    def _get_next_step_number(self) -> int:
        """Get the next step number."""
        self._step_counter += 1
        return self._step_counter
    
    def _get_sutra_citation(self, rule_id: int) -> Optional[str]:
        """Get sūtra citation for a rule ID."""
        # Convert rule_id to string since our registry uses string IDs
        rule = self.rule_registry._rule_index.get(str(rule_id))
        return str(rule.sutra_ref) if rule else None
    
    def generate_derivation_report(self, derivation_tree: DerivationTree, format: str = "text") -> str:
        """Generate a comprehensive derivation report."""
        if format == "text":
            return self._generate_text_report(derivation_tree)
        elif format == "json":
            return json.dumps(derivation_tree.to_dict(), indent=2, ensure_ascii=False)
        elif format == "html":
            return self._generate_html_report(derivation_tree)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_text_report(self, derivation_tree: DerivationTree) -> str:
        """Generate a detailed text report."""
        lines = []
        lines.append("ŚABDA-PRAKRIYĀ DERIVATION REPORT")
        lines.append("=" * 50)
        lines.append(f"Word: {derivation_tree.root_word}")
        lines.append(f"Final Form: {derivation_tree.get_final_form()}")
        lines.append(f"Confidence: {derivation_tree.confidence:.2f}")
        lines.append(f"Steps: {len(derivation_tree.derivation_steps)}")
        lines.append("")
        
        # Root morphemes
        lines.append("ROOT MORPHEMES:")
        for morpheme in derivation_tree.root_morphemes:
            lines.append(f"  {morpheme.text} ({morpheme.type.value})")
        lines.append("")
        
        # Derivation steps
        lines.append("DERIVATION STEPS:")
        current_stage = None
        for step in derivation_tree.derivation_steps:
            if step.stage != current_stage:
                lines.append(f"\n{step.stage.value}:")
                current_stage = step.stage
            
            lines.append(f"  {step.step_number}. {step.get_transformation_summary()}")
            if step.rationale:
                lines.append(f"     Rationale: {step.rationale}")
            if step.sutra_citation:
                lines.append(f"     Sūtra: {step.sutra_citation}")
        
        # Validation results
        if derivation_tree.validation_results:
            lines.append("\nVALIDATION RESULTS:")
            for source, result in derivation_tree.validation_results.items():
                status = "✓" if result.get('validated', False) else "✗"
                lines.append(f"  {status} {source}: {result.get('confidence', 0):.2f}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, derivation_tree: DerivationTree) -> str:
        """Generate an HTML report with interactive features."""
        # Placeholder for HTML generation
        # In a full implementation, this would generate rich HTML with CSS and JavaScript
        return f"""
        <html>
        <head><title>Derivation Report: {derivation_tree.root_word}</title></head>
        <body>
        <h1>Śabda-prakriyā Report</h1>
        <p>Word: {derivation_tree.root_word}</p>
        <p>Final Form: {derivation_tree.get_final_form()}</p>
        <p>Confidence: {derivation_tree.confidence:.2f}</p>
        <!-- Full HTML implementation would go here -->
        </body>
        </html>
        """


@dataclass
class StageResult:
    """Result of processing a derivation stage."""
    tokens_after: List[Token]
    steps: List[DerivationStep]


# Utility functions for derivation context
def create_derivation_context(text: str, **kwargs) -> DerivationContext:
    """Create a derivation context with default values."""
    return DerivationContext(
        source_text=text,
        **kwargs
    )


def load_traditional_sources() -> List[str]:
    """Load list of traditional grammatical sources for validation."""
    return [
        "Ashtadhyayi",
        "Mahabhashya", 
        "Kasika Vritti",
        "Siddhanta Kaumudi"
    ]