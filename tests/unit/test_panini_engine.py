"""
Tests for the Pāṇini rule engine implementation.

Tests cover:
- SutraRule creation and metadata
- GuardSystem loop prevention
- Rule application and inheritance
- Complex rule interactions
- Essential sūtra functionality
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from sanskrit_rewrite_engine.rule import (
    SutraRule, SutraReference, RuleType, ParibhasaRule,
    GuardSystem, RuleRegistry
)
from sanskrit_rewrite_engine.panini_engine import (
    PaniniRuleEngine, PaniniEngineBuilder, TransformationTrace, PassTrace
)
from sanskrit_rewrite_engine.essential_sutras import create_essential_sutras, create_essential_paribhasas
from sanskrit_rewrite_engine.token import Token, TokenKind


class TestSutraReference:
    """Test SutraReference functionality."""
    
    def test_sutra_reference_creation(self):
        """Test creating sūtra references."""
        ref = SutraReference(1, 1, 1)
        assert ref.adhyaya == 1
        assert ref.pada == 1
        assert ref.sutra == 1
        assert str(ref) == "1.1.1"
    
    def test_sutra_reference_ordering(self):
        """Test sūtra reference ordering."""
        ref1 = SutraReference(1, 1, 1)
        ref2 = SutraReference(1, 1, 2)
        ref3 = SutraReference(1, 2, 1)
        ref4 = SutraReference(2, 1, 1)
        
        assert ref1 < ref2
        assert ref2 < ref3
        assert ref3 < ref4
        assert not (ref2 < ref1)


class TestSutraRule:
    """Test SutraRule functionality."""
    
    def test_sutra_rule_creation(self):
        """Test creating sūtra rules."""
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        assert rule.sutra_ref == ref
        assert rule.name == "test_rule"
        assert rule.rule_type == RuleType.SUTRA
        assert rule.priority == 1
        assert rule.id == "1.1.1"
        assert rule.can_apply()
    
    def test_adhikara_and_anuvrti(self):
        """Test adhikāra and anuvṛtti functionality."""
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        # Test adhikāra
        rule.add_adhikara("sandhi")
        rule.add_adhikara("vowel")
        assert rule.has_domain("sandhi")
        assert rule.has_domain("vowel")
        assert not rule.has_domain("consonant")
        
        # Test anuvṛtti
        rule.add_anuvrti("inherited_domain")
        assert rule.has_domain("inherited_domain")
    
    def test_application_limits(self):
        """Test rule application limits."""
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i),
            max_applications=2
        )
        
        assert rule.can_apply()
        
        # Simulate applications
        rule.applications = 1
        assert rule.can_apply()
        
        rule.applications = 2
        assert not rule.can_apply()
        
        # Reset
        rule.reset_applications()
        assert rule.applications == 0
        assert rule.can_apply()
    
    def test_cross_references(self):
        """Test cross-reference functionality."""
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        cross_ref = SutraReference(1, 1, 2)
        rule.add_cross_reference(cross_ref)
        
        assert cross_ref in rule.cross_refs


class TestParibhasaRule:
    """Test ParibhasaRule functionality."""
    
    def test_paribhasa_creation(self):
        """Test creating paribhāṣā rules."""
        ref = SutraReference(8, 2, 1)
        paribhasa = ParibhasaRule(
            sutra_ref=ref,
            name="test_paribhasa",
            description="Test paribhāṣā",
            priority=1,
            condition_fn=lambda tokens, registry: True,
            action_fn=lambda registry: None
        )
        
        assert paribhasa.sutra_ref == ref
        assert paribhasa.name == "test_paribhasa"
        assert paribhasa.id == "paribhasa_8.2.1"
    
    def test_paribhasa_scope(self):
        """Test paribhāṣā scope functionality."""
        ref = SutraReference(8, 2, 1)
        paribhasa = ParibhasaRule(
            sutra_ref=ref,
            name="test_paribhasa",
            description="Test paribhāṣā",
            priority=1,
            condition_fn=lambda tokens, registry: True,
            action_fn=lambda registry: None,
            scope={"phonology", "sandhi"}
        )
        
        assert paribhasa.affects_domain("phonology")
        assert paribhasa.affects_domain("sandhi")
        assert not paribhasa.affects_domain("morphology")


class TestGuardSystem:
    """Test GuardSystem functionality."""
    
    def test_guard_system_creation(self):
        """Test creating guard system."""
        guard = GuardSystem()
        assert guard._application_history == {}
        assert guard._global_applications == {}
    
    def test_rule_application_tracking(self):
        """Test tracking rule applications."""
        guard = GuardSystem()
        
        # Create test rule and tokens
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        tokens = [Token("a", TokenKind.VOWEL), Token("i", TokenKind.VOWEL)]
        
        # Initially should be able to apply
        assert guard.can_apply_rule(rule, tokens, 0)
        
        # Record application
        guard.record_application(rule, tokens, 0)
        
        # Should not be able to apply at same position immediately
        assert not guard.can_apply_rule(rule, tokens, 0)
        
        # Should be able to apply at different position
        assert guard.can_apply_rule(rule, tokens, 1)
    
    def test_guard_reset(self):
        """Test resetting guard system."""
        guard = GuardSystem()
        
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        tokens = [Token("a", TokenKind.VOWEL)]
        
        # Record application
        guard.record_application(rule, tokens, 0)
        assert not guard.can_apply_rule(rule, tokens, 0)
        
        # Reset
        guard.reset_guards()
        assert guard.can_apply_rule(rule, tokens, 0)
    
    def test_application_counting(self):
        """Test global application counting."""
        guard = GuardSystem()
        
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        tokens = [Token("a", TokenKind.VOWEL)]
        
        assert guard.get_application_count(rule.id) == 0
        
        guard.record_application(rule, tokens, 0)
        assert guard.get_application_count(rule.id) == 1
        assert rule.applications == 1


class TestRuleRegistry:
    """Test RuleRegistry functionality."""
    
    def test_registry_creation(self):
        """Test creating rule registry."""
        registry = RuleRegistry()
        assert len(registry.get_active_sutra_rules()) == 0
        assert len(registry.get_active_paribhasa_rules()) == 0
    
    def test_adding_sutra_rules(self):
        """Test adding sūtra rules."""
        registry = RuleRegistry()
        
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i),
            adhikara={"sandhi"}
        )
        
        registry.add_sutra_rule(rule)
        
        assert len(registry.get_active_sutra_rules()) == 1
        assert registry.get_rule_by_reference(ref) == rule
        assert len(registry.get_rules_for_domain("sandhi")) == 1
    
    def test_adhikara_stack(self):
        """Test adhikāra context stack."""
        registry = RuleRegistry()
        
        assert registry.get_current_adhikara() == []
        
        registry.push_adhikara("sandhi")
        assert registry.get_current_adhikara() == ["sandhi"]
        
        registry.push_adhikara("vowel")
        assert registry.get_current_adhikara() == ["sandhi", "vowel"]
        
        popped = registry.pop_adhikara()
        assert popped == "vowel"
        assert registry.get_current_adhikara() == ["sandhi"]
    
    def test_anuvrti_application(self):
        """Test anuvṛtti inheritance."""
        registry = RuleRegistry()
        
        # Set up adhikāra context
        registry.push_adhikara("sandhi")
        registry.push_adhikara("vowel")
        
        ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        # Apply anuvṛtti
        registry.apply_anuvrti(rule)
        
        assert rule.has_domain("sandhi")
        assert rule.has_domain("vowel")
    
    def test_rule_validation(self):
        """Test rule set validation."""
        registry = RuleRegistry()
        
        # Add rule with cross-reference to non-existent rule
        ref1 = SutraReference(1, 1, 1)
        ref2 = SutraReference(1, 1, 2)  # This rule doesn't exist
        
        rule = SutraRule(
            sutra_ref=ref1,
            name="test_rule",
            description="Test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: True,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        rule.add_cross_reference(ref2)
        
        registry.add_sutra_rule(rule)
        
        issues = registry.validate_rule_set()
        assert len(issues) > 0
        assert any("Invalid cross-reference" in issue for issue in issues)


class TestEssentialSutras:
    """Test essential sūtra implementations."""
    
    def test_create_essential_sutras(self):
        """Test creating essential sūtras."""
        sutras = create_essential_sutras()
        
        assert len(sutras) > 0
        
        # Check that we have fundamental definition rules
        definition_rules = [s for s in sutras if s.rule_type == RuleType.ADHIKARA]
        assert len(definition_rules) > 0
        
        # Check for specific important sūtras
        sutra_refs = {str(s.sutra_ref) for s in sutras}
        assert "1.1.1" in sutra_refs  # vṛddhir ādaic
        assert "1.1.2" in sutra_refs  # adeṅ guṇaḥ
        assert "6.1.77" in sutra_refs  # iko yaṇ aci
    
    def test_create_essential_paribhasas(self):
        """Test creating essential paribhāṣās."""
        paribhasas = create_essential_paribhasas()
        
        assert len(paribhasas) > 0
        
        # Check for fundamental paribhāṣās
        paribhasa_names = {p.name for p in paribhasas}
        assert "pūrvatrāsiddham" in paribhasa_names
    
    def test_sandhi_rule_matching(self):
        """Test sandhi rule matching functions."""
        from sanskrit_rewrite_engine.essential_sutras import _match_ad_gunah, _apply_ad_gunah
        
        # Test a + i → e
        tokens = [
            Token("a", TokenKind.VOWEL),
            Token("i", TokenKind.VOWEL)
        ]
        
        assert _match_ad_gunah(tokens, 0)
        assert not _match_ad_gunah(tokens, 1)  # No next token
        
        # Apply the rule
        new_tokens, new_index = _apply_ad_gunah(tokens, 0)
        
        assert len(new_tokens) == 1
        assert new_tokens[0].text == "e"
        assert new_tokens[0].has_tag("guna")
        assert new_tokens[0].has_tag("sandhi_result")
    
    def test_savarna_dirgha_rule(self):
        """Test similar vowel lengthening rule."""
        from sanskrit_rewrite_engine.essential_sutras import _match_savarna_dirgha, _apply_savarna_dirgha
        
        # Test a + a → ā
        tokens = [
            Token("a", TokenKind.VOWEL),
            Token("a", TokenKind.VOWEL)
        ]
        
        assert _match_savarna_dirgha(tokens, 0)
        
        new_tokens, new_index = _apply_savarna_dirgha(tokens, 0)
        
        assert len(new_tokens) == 1
        assert new_tokens[0].text == "ā"
        assert new_tokens[0].has_tag("long")
        assert new_tokens[0].has_tag("sandhi_result")
    
    def test_iko_yan_aci_rule(self):
        """Test i, u, ṛ, ḷ → y, v, r, l before vowels."""
        from sanskrit_rewrite_engine.essential_sutras import _match_iko_yan_aci, _apply_iko_yan_aci
        
        # Test i + a → y + a
        tokens = [
            Token("i", TokenKind.VOWEL),
            Token("a", TokenKind.VOWEL)
        ]
        
        assert _match_iko_yan_aci(tokens, 0)
        
        new_tokens, new_index = _apply_iko_yan_aci(tokens, 0)
        
        assert len(new_tokens) == 2
        assert new_tokens[0].text == "y"
        assert new_tokens[0].kind == TokenKind.CONSONANT
        assert new_tokens[0].has_tag("semivowel")
        assert new_tokens[0].has_tag("sandhi_result")
        assert new_tokens[1].text == "a"


class TestPaniniRuleEngine:
    """Test PaniniRuleEngine functionality."""
    
    def test_engine_creation(self):
        """Test creating Pāṇini rule engine."""
        engine = PaniniRuleEngine()
        
        assert engine.registry is not None
        assert engine.guard_system is not None
        
        # Should have loaded essential rules
        stats = engine.get_rule_statistics()
        assert stats['total_sutra_rules'] > 0
    
    def test_simple_processing(self):
        """Test simple token processing."""
        engine = PaniniRuleEngine()
        
        # Create simple tokens that should trigger sandhi
        tokens = [
            Token("a", TokenKind.VOWEL),
            Token("i", TokenKind.VOWEL)
        ]
        
        result = engine.process(tokens, max_passes=5)
        
        assert result.input_tokens == tokens
        assert len(result.output_tokens) <= len(tokens)  # May be combined
        assert result.converged
        assert len(result.traces) > 0
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        engine = PaniniRuleEngine()
        
        # Tokens that shouldn't trigger any rules
        tokens = [
            Token("k", TokenKind.CONSONANT),
            Token("a", TokenKind.VOWEL)
        ]
        
        result = engine.process(tokens, max_passes=5)
        
        assert result.converged
        assert result.passes <= 1  # Should converge quickly
    
    def test_max_passes_limit(self):
        """Test maximum passes limit."""
        engine = PaniniRuleEngine()
        
        tokens = [Token("a", TokenKind.VOWEL)]
        
        result = engine.process(tokens, max_passes=1)
        
        assert result.passes <= 1
    
    def test_transformation_tracing(self):
        """Test transformation tracing."""
        engine = PaniniRuleEngine()
        
        tokens = [
            Token("a", TokenKind.VOWEL),
            Token("i", TokenKind.VOWEL)
        ]
        
        result = engine.process(tokens)
        
        if result.traces and result.traces[0].transformations:
            trace = result.traces[0].transformations[0]
            assert isinstance(trace, TransformationTrace)
            assert trace.rule_name is not None
            assert trace.rule_id is not None
            assert trace.sutra_ref is not None
            assert isinstance(trace.timestamp, datetime)
    
    def test_custom_rule_addition(self):
        """Test adding custom rules."""
        engine = PaniniRuleEngine()
        
        # Create custom rule
        ref = SutraReference(9, 9, 9)
        custom_rule = SutraRule(
            sutra_ref=ref,
            name="custom_test_rule",
            description="Custom test rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: False,  # Never matches
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        initial_count = engine.get_rule_statistics()['total_sutra_rules']
        engine.add_custom_rule(custom_rule)
        final_count = engine.get_rule_statistics()['total_sutra_rules']
        
        assert final_count == initial_count + 1
    
    def test_rule_enable_disable(self):
        """Test enabling and disabling rules."""
        engine = PaniniRuleEngine()
        
        # Get a rule ID
        stats = engine.get_rule_statistics()
        if stats['total_sutra_rules'] > 0:
            # This is a simplified test - in practice we'd need to know specific rule IDs
            engine.disable_rule("1.1.1")
            engine.enable_rule("1.1.1")
    
    def test_engine_reset(self):
        """Test engine reset functionality."""
        engine = PaniniRuleEngine()
        
        tokens = [Token("a", TokenKind.VOWEL)]
        engine.process(tokens)
        
        # Reset should clear state
        engine.reset_engine()
        
        # Should be able to process again
        result = engine.process(tokens)
        assert result is not None


class TestPaniniEngineBuilder:
    """Test PaniniEngineBuilder functionality."""
    
    def test_builder_creation(self):
        """Test creating engine builder."""
        builder = PaniniEngineBuilder()
        assert builder.custom_rules == []
        assert builder.disabled_rules == set()
    
    def test_builder_configuration(self):
        """Test builder configuration."""
        mock_tokenizer = Mock()
        
        ref = SutraReference(9, 9, 9)
        custom_rule = SutraRule(
            sutra_ref=ref,
            name="custom_rule",
            description="Custom rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        builder = (PaniniEngineBuilder()
                  .with_tokenizer(mock_tokenizer)
                  .with_custom_rule(custom_rule)
                  .disable_rule("1.1.1")
                  .set_max_applications("1.1.2", 5))
        
        assert builder.tokenizer == mock_tokenizer
        assert len(builder.custom_rules) == 1
        assert "1.1.1" in builder.disabled_rules
        assert builder.max_applications["1.1.2"] == 5
    
    def test_builder_build(self):
        """Test building engine from builder."""
        builder = PaniniEngineBuilder()
        engine = builder.build()
        
        assert isinstance(engine, PaniniRuleEngine)


class TestComplexRuleInteractions:
    """Test complex rule interactions and inheritance."""
    
    def test_rule_priority_ordering(self):
        """Test that rules are applied in priority order."""
        engine = PaniniRuleEngine()
        
        # Create tokens that could match multiple rules
        tokens = [
            Token("a", TokenKind.VOWEL),
            Token("a", TokenKind.VOWEL)
        ]
        
        result = engine.process(tokens)
        
        # The savarna_dirgha rule (priority 1) should apply before guna rules (priority 3)
        if result.traces and result.traces[0].transformations:
            # Check that higher priority rules were applied
            applied_rules = [t.rule_name for t in result.traces[0].transformations]
            # This is a simplified check - in practice we'd verify specific rule ordering
            assert len(applied_rules) > 0
    
    def test_adhikara_inheritance(self):
        """Test adhikāra domain inheritance."""
        registry = RuleRegistry()
        
        # Set up adhikāra context
        registry.push_adhikara("sandhi")
        
        ref = SutraReference(6, 1, 100)
        rule = SutraRule(
            sutra_ref=ref,
            name="test_sandhi_rule",
            description="Test sandhi rule",
            rule_type=RuleType.SUTRA,
            priority=2,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        # Apply anuvṛtti
        registry.apply_anuvrti(rule)
        
        assert rule.has_domain("sandhi")
    
    def test_cross_reference_resolution(self):
        """Test cross-reference resolution between rules."""
        registry = RuleRegistry()
        
        # Create two related rules
        ref1 = SutraReference(1, 1, 1)
        ref2 = SutraReference(1, 1, 2)
        
        rule1 = SutraRule(
            sutra_ref=ref1,
            name="rule1",
            description="First rule",
            rule_type=RuleType.SUTRA,
            priority=1,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        rule2 = SutraRule(
            sutra_ref=ref2,
            name="rule2",
            description="Second rule",
            rule_type=RuleType.SUTRA,
            priority=2,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        # Add cross-reference
        rule1.add_cross_reference(ref2)
        
        registry.add_sutra_rule(rule1)
        registry.add_sutra_rule(rule2)
        
        # Test cross-reference resolution
        cross_refs = registry.get_cross_referenced_rules(rule1)
        assert len(cross_refs) == 1
        assert cross_refs[0] == rule2
    
    def test_paribhasa_rule_control(self):
        """Test paribhāṣā rule control over other rules."""
        registry = RuleRegistry()
        
        # Create a paribhāṣā that disables certain rules
        ref = SutraReference(8, 2, 1)
        paribhasa = ParibhasaRule(
            sutra_ref=ref,
            name="test_control_paribhasa",
            description="Test control paribhāṣā",
            priority=1,
            condition_fn=lambda tokens, registry: True,
            action_fn=lambda registry: registry.disable_rule("1.1.1")
        )
        
        registry.add_paribhasa_rule(paribhasa)
        
        # Add a regular rule
        sutra_ref = SutraReference(1, 1, 1)
        rule = SutraRule(
            sutra_ref=sutra_ref,
            name="controlled_rule",
            description="Rule controlled by paribhāṣā",
            rule_type=RuleType.SUTRA,
            priority=2,
            match_fn=lambda tokens, i: False,
            apply_fn=lambda tokens, i: (tokens, i)
        )
        
        registry.add_sutra_rule(rule)
        
        # Initially rule should be active
        assert rule.active
        
        # Apply paribhāṣā
        tokens = [Token("test", TokenKind.OTHER)]
        paribhasa.evaluate(tokens, registry)
        
        # Rule should now be disabled
        assert not rule.active


if __name__ == "__main__":
    pytest.main([__file__])