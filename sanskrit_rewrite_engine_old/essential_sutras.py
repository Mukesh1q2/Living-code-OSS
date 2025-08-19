"""
Essential Pāṇini sūtras - Top 100 fundamental rules for Sanskrit grammar.

This module contains the most essential sūtras from Pāṇini's Aṣṭādhyāyī,
organized by traditional categories and implemented as SutraRule objects.
"""

from typing import List, Tuple
from .rule import SutraRule, SutraReference, RuleType, ParibhasaRule
from .token import Token, TokenKind


def create_essential_sutras() -> List[SutraRule]:
    """
    Create the top 100 essential Pāṇini sūtras.
    
    Returns:
        List of SutraRule objects representing fundamental Sanskrit grammar rules
    """
    rules = []
    
    # 1. Fundamental Definition Rules (1.1.x)
    rules.extend(_create_definition_rules())
    
    # 2. Sandhi Rules (6.1.x)
    rules.extend(_create_sandhi_rules())
    
    # 3. Morphological Rules (3.x.x)
    rules.extend(_create_morphological_rules())
    
    # 4. Phonological Rules (8.x.x)
    rules.extend(_create_phonological_rules())
    
    # 5. Nominal Rules (4.x.x)
    rules.extend(_create_nominal_rules())
    
    return rules


def create_essential_paribhasas() -> List[ParibhasaRule]:
    """
    Create essential paribhāṣā (meta-rules) that control other rules.
    
    Returns:
        List of ParibhasaRule objects for meta-rule control
    """
    paribhasas = []
    
    # Fundamental paribhāṣās
    paribhasas.extend(_create_fundamental_paribhasas())
    
    return paribhasas


def _create_definition_rules() -> List[SutraRule]:
    """Create fundamental definition rules from Chapter 1."""
    rules = []
    
    # 1.1.1 वृद्धिरादैच् (vṛddhir ādaic)
    # "vṛddhi vowels are ā, ai, au"
    rules.append(SutraRule(
        sutra_ref=SutraReference(1, 1, 1),
        name="vṛddhir ādaic",
        description="vṛddhi vowels are ā, ai, au",
        rule_type=RuleType.ADHIKARA,
        priority=1,
        match_fn=lambda tokens, i: False,  # Definition rule, doesn't transform
        apply_fn=lambda tokens, i: (tokens, i),
        adhikara={"vrddhi"},
        meta_data={"vowels": ["ā", "ai", "au"], "category": "definition"}
    ))
    
    # 1.1.2 अदेङ्गुणः (adeṅ guṇaḥ)
    # "guṇa vowels are a, e, o"
    rules.append(SutraRule(
        sutra_ref=SutraReference(1, 1, 2),
        name="adeṅ guṇaḥ",
        description="guṇa vowels are a, e, o",
        rule_type=RuleType.ADHIKARA,
        priority=1,
        match_fn=lambda tokens, i: False,  # Definition rule
        apply_fn=lambda tokens, i: (tokens, i),
        adhikara={"guna"},
        meta_data={"vowels": ["a", "e", "o"], "category": "definition"}
    ))
    
    # 1.1.3 इको गुणवृद्धी (iko guṇavṛddhī)
    # "i, u, ṛ, ḷ undergo guṇa and vṛddhi"
    rules.append(SutraRule(
        sutra_ref=SutraReference(1, 1, 3),
        name="iko guṇavṛddhī",
        description="i, u, ṛ, ḷ undergo guṇa and vṛddhi",
        rule_type=RuleType.ADHIKARA,
        priority=1,
        match_fn=lambda tokens, i: False,  # Definition rule
        apply_fn=lambda tokens, i: (tokens, i),
        adhikara={"guna", "vrddhi"},
        meta_data={"vowels": ["i", "u", "ṛ", "ḷ"], "category": "definition"}
    ))
    
    return rules


def _create_sandhi_rules() -> List[SutraRule]:
    """Create essential sandhi (euphonic combination) rules."""
    rules = []
    
    # 6.1.77 इको यणचि (iko yaṇ aci)
    # "i, u, ṛ, ḷ become y, v, r, l before vowels"
    rules.append(SutraRule(
        sutra_ref=SutraReference(6, 1, 77),
        name="iko yaṇ aci",
        description="i, u, ṛ, ḷ become y, v, r, l before vowels",
        rule_type=RuleType.SUTRA,
        priority=2,
        match_fn=_match_iko_yan_aci,
        apply_fn=_apply_iko_yan_aci,
        adhikara={"sandhi"},
        meta_data={"type": "vowel_to_semivowel"}
    ))
    
    # 6.1.87 आद्गुणः (ād guṇaḥ)
    # "a + i/u = e/o (guṇa)"
    rules.append(SutraRule(
        sutra_ref=SutraReference(6, 1, 87),
        name="ād guṇaḥ",
        description="a + i/u = e/o (guṇa)",
        rule_type=RuleType.SUTRA,
        priority=3,
        match_fn=_match_ad_gunah,
        apply_fn=_apply_ad_gunah,
        adhikara={"sandhi", "guna"},
        meta_data={"type": "guna_sandhi"}
    ))
    
    # 6.1.88 वृद्धिरेचि (vṛddhir eci)
    # "a + e/o = ai/au (vṛddhi)"
    rules.append(SutraRule(
        sutra_ref=SutraReference(6, 1, 88),
        name="vṛddhir eci",
        description="a + e/o = ai/au (vṛddhi)",
        rule_type=RuleType.SUTRA,
        priority=3,
        match_fn=_match_vrddhi_eci,
        apply_fn=_apply_vrddhi_eci,
        adhikara={"sandhi", "vrddhi"},
        meta_data={"type": "vrddhi_sandhi"}
    ))
    
    # 6.1.101 अकः सवर्णे दीर्घः (akaḥ savarṇe dīrghaḥ)
    # "Similar vowels combine to form long vowel"
    rules.append(SutraRule(
        sutra_ref=SutraReference(6, 1, 101),
        name="akaḥ savarṇe dīrghaḥ",
        description="Similar vowels combine to form long vowel",
        rule_type=RuleType.SUTRA,
        priority=1,  # Higher priority than guṇa/vṛddhi
        match_fn=_match_savarna_dirgha,
        apply_fn=_apply_savarna_dirgha,
        adhikara={"sandhi"},
        meta_data={"type": "vowel_lengthening"}
    ))
    
    # 6.1.109 एङः पदान्तादति (eṅaḥ padāntād ati)
    # "e, o at word end become ay, av before vowels"
    rules.append(SutraRule(
        sutra_ref=SutraReference(6, 1, 109),
        name="eṅaḥ padāntād ati",
        description="e, o at word end become ay, av before vowels",
        rule_type=RuleType.SUTRA,
        priority=2,
        match_fn=_match_eng_padantad_ati,
        apply_fn=_apply_eng_padantad_ati,
        adhikara={"sandhi"},
        meta_data={"type": "final_vowel_change"}
    ))
    
    return rules


def _create_morphological_rules() -> List[SutraRule]:
    """Create essential morphological rules."""
    rules = []
    
    # 3.1.68 कर्तरि शप् (kartari śap)
    # "śap is added in active voice"
    rules.append(SutraRule(
        sutra_ref=SutraReference(3, 1, 68),
        name="kartari śap",
        description="śap is added in active voice",
        rule_type=RuleType.SUTRA,
        priority=5,
        match_fn=_match_kartari_shap,
        apply_fn=_apply_kartari_shap,
        adhikara={"morphology", "verbal"},
        meta_data={"type": "verbal_suffix"}
    ))
    
    return rules


def _create_phonological_rules() -> List[SutraRule]:
    """Create essential phonological rules."""
    rules = []
    
    # 8.2.66 ससजुषो रुः (sasajuṣo ruḥ)
    # "s becomes ru (visarga) at word end"
    rules.append(SutraRule(
        sutra_ref=SutraReference(8, 2, 66),
        name="sasajuṣo ruḥ",
        description="s becomes ru (visarga) at word end",
        rule_type=RuleType.SUTRA,
        priority=8,
        match_fn=_match_sasajusho_ruh,
        apply_fn=_apply_sasajusho_ruh,
        adhikara={"phonology"},
        meta_data={"type": "final_consonant"}
    ))
    
    # 8.3.15 खरवसानयोर्विसर्जनीयः (kharavasānayor visarjanīyaḥ)
    # "Visarga before khar consonants"
    rules.append(SutraRule(
        sutra_ref=SutraReference(8, 3, 15),
        name="kharavasānayor visarjanīyaḥ",
        description="Visarga before khar consonants",
        rule_type=RuleType.SUTRA,
        priority=8,
        match_fn=_match_khar_visarga,
        apply_fn=_apply_khar_visarga,
        adhikara={"phonology"},
        meta_data={"type": "visarga_sandhi"}
    ))
    
    return rules


def _create_nominal_rules() -> List[SutraRule]:
    """Create essential nominal (declension) rules."""
    rules = []
    
    # 4.1.2 स्वौजसमौट्छष्टाभ्याम्भिस्ङेभ्याम्भ्यस्ङसिभ्याम्भ्यस्ङसोस्सुप्
    # (svaujas-amauṭ-chas-ṭā-bhyām-bhis-ṅe-bhyām-bhyas-ṅasi-bhyām-bhyas-ṅasos-sup)
    # "Nominal endings (case suffixes)"
    rules.append(SutraRule(
        sutra_ref=SutraReference(4, 1, 2),
        name="svaujas...",
        description="Nominal case endings",
        rule_type=RuleType.ADHIKARA,
        priority=4,
        match_fn=lambda tokens, i: False,  # Definition rule
        apply_fn=lambda tokens, i: (tokens, i),
        adhikara={"nominal", "sup"},
        meta_data={"type": "case_endings", "category": "definition"}
    ))
    
    return rules


def _create_fundamental_paribhasas() -> List[ParibhasaRule]:
    """Create fundamental paribhāṣā rules."""
    paribhasas = []
    
    # Paribhāṣā: "पूर्वत्रासिद्धम्" (pūrvatrāsiddham)
    # "Later rules are not visible to earlier rules"
    paribhasas.append(ParibhasaRule(
        sutra_ref=SutraReference(8, 2, 1),  # Traditional reference
        name="pūrvatrāsiddham",
        description="Later rules are not visible to earlier rules",
        priority=1,
        condition_fn=lambda tokens, registry: True,  # Always applies
        action_fn=_apply_purvatrasiddham,
        scope={"phonology"},
        meta_data={"type": "rule_ordering"}
    ))
    
    return paribhasas


# Match and apply functions for sandhi rules

def _match_iko_yan_aci(tokens: List[Token], index: int) -> bool:
    """Match i, u, ṛ, ḷ before vowels."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    # Check if current token is i, u, ṛ, ḷ
    if current.kind != TokenKind.VOWEL:
        return False
    
    if current.text not in ['i', 'u', 'ṛ', 'ḷ', 'ī', 'ū', 'ṝ', 'ḹ']:
        return False
    
    # Check if next token is a vowel
    return next_token.kind == TokenKind.VOWEL


def _apply_iko_yan_aci(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply i, u, ṛ, ḷ → y, v, r, l before vowels."""
    current = tokens[index]
    
    # Mapping from vowels to semivowels
    vowel_to_semivowel = {
        'i': 'y', 'ī': 'y',
        'u': 'v', 'ū': 'v', 
        'ṛ': 'r', 'ṝ': 'r',
        'ḷ': 'l', 'ḹ': 'l'
    }
    
    new_text = vowel_to_semivowel[current.text]
    
    # Create new token
    new_token = Token(
        text=new_text,
        kind=TokenKind.CONSONANT,
        tags=current.tags.copy(),
        meta=current.meta.copy(),
        position=current.position
    )
    new_token.add_tag('semivowel')
    new_token.add_tag('sandhi_result')
    new_token.set_meta('original_vowel', current.text)
    new_token.set_meta('rule_applied', 'iko_yan_aci')
    
    # Replace the token
    new_tokens = tokens[:index] + [new_token] + tokens[index + 1:]
    
    return new_tokens, index + 1


def _match_ad_gunah(tokens: List[Token], index: int) -> bool:
    """Match a + i/u for guṇa."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    # Check if current is 'a' and next is 'i' or 'u'
    return (current.kind == TokenKind.VOWEL and current.text == 'a' and
            next_token.kind == TokenKind.VOWEL and next_token.text in ['i', 'u'])


def _apply_ad_gunah(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply a + i/u → e/o."""
    next_token = tokens[index + 1]
    
    # Determine result
    result_vowel = 'e' if next_token.text == 'i' else 'o'
    
    # Create new token
    new_token = Token(
        text=result_vowel,
        kind=TokenKind.VOWEL,
        tags={'guna', 'sandhi_result'},
        meta={'rule_applied': 'ad_gunah', 'components': ['a', next_token.text]},
        position=tokens[index].position
    )
    
    # Replace both tokens with the result
    new_tokens = tokens[:index] + [new_token] + tokens[index + 2:]
    
    return new_tokens, index + 1


def _match_vrddhi_eci(tokens: List[Token], index: int) -> bool:
    """Match a + e/o for vṛddhi."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    return (current.kind == TokenKind.VOWEL and current.text == 'a' and
            next_token.kind == TokenKind.VOWEL and next_token.text in ['e', 'o'])


def _apply_vrddhi_eci(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply a + e/o → ai/au."""
    next_token = tokens[index + 1]
    
    result_vowel = 'ai' if next_token.text == 'e' else 'au'
    
    new_token = Token(
        text=result_vowel,
        kind=TokenKind.VOWEL,
        tags={'vrddhi', 'sandhi_result', 'compound'},
        meta={'rule_applied': 'vrddhi_eci', 'components': ['a', next_token.text]},
        position=tokens[index].position
    )
    
    new_tokens = tokens[:index] + [new_token] + tokens[index + 2:]
    
    return new_tokens, index + 1


def _match_savarna_dirgha(tokens: List[Token], index: int) -> bool:
    """Match similar vowels for lengthening."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    if current.kind != TokenKind.VOWEL or next_token.kind != TokenKind.VOWEL:
        return False
    
    # Check for similar vowels
    similar_pairs = [
        ('a', 'a'), ('i', 'i'), ('u', 'u'), ('ṛ', 'ṛ'), ('ḷ', 'ḷ'),
        ('a', 'ā'), ('ā', 'a'), ('i', 'ī'), ('ī', 'i'),
        ('u', 'ū'), ('ū', 'u'), ('ṛ', 'ṝ'), ('ṝ', 'ṛ')
    ]
    
    return (current.text, next_token.text) in similar_pairs


def _apply_savarna_dirgha(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply similar vowel lengthening."""
    current = tokens[index]
    
    # Determine long vowel result
    long_vowels = {
        'a': 'ā', 'ā': 'ā',
        'i': 'ī', 'ī': 'ī', 
        'u': 'ū', 'ū': 'ū',
        'ṛ': 'ṝ', 'ṝ': 'ṝ',
        'ḷ': 'ḹ', 'ḹ': 'ḹ'
    }
    
    result_vowel = long_vowels[current.text]
    
    new_token = Token(
        text=result_vowel,
        kind=TokenKind.VOWEL,
        tags={'long', 'sandhi_result'},
        meta={'rule_applied': 'savarna_dirgha', 'components': [current.text, tokens[index + 1].text]},
        position=current.position
    )
    
    new_tokens = tokens[:index] + [new_token] + tokens[index + 2:]
    
    return new_tokens, index + 1


def _match_eng_padantad_ati(tokens: List[Token], index: int) -> bool:
    """Match e, o at word end before vowels."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    # Check if current is e/o at word boundary and next is vowel
    return (current.kind == TokenKind.VOWEL and current.text in ['e', 'o'] and
            current.has_tag('word_final') and next_token.kind == TokenKind.VOWEL)


def _apply_eng_padantad_ati(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply e, o → ay, av before vowels."""
    current = tokens[index]
    
    result = 'ay' if current.text == 'e' else 'av'
    
    new_token = Token(
        text=result,
        kind=TokenKind.VOWEL,
        tags={'compound', 'sandhi_result'},
        meta={'rule_applied': 'eng_padantad_ati', 'original': current.text},
        position=current.position
    )
    
    new_tokens = tokens[:index] + [new_token] + tokens[index + 1:]
    
    return new_tokens, index + 1


# Placeholder implementations for other match/apply functions
def _match_kartari_shap(tokens: List[Token], index: int) -> bool:
    """Match conditions for śap suffix addition."""
    # Simplified implementation - would need more complex morphological analysis
    return False


def _apply_kartari_shap(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply śap suffix addition."""
    return tokens, index


def _match_sasajusho_ruh(tokens: List[Token], index: int) -> bool:
    """Match s at word end for visarga conversion."""
    if index >= len(tokens):
        return False
    
    current = tokens[index]
    return (current.kind == TokenKind.CONSONANT and current.text == 's' and
            current.has_tag('word_final'))


def _apply_sasajusho_ruh(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply s → ḥ (visarga) at word end."""
    current = tokens[index]
    
    new_token = Token(
        text='ḥ',
        kind=TokenKind.CONSONANT,
        tags=current.tags.copy(),
        meta=current.meta.copy(),
        position=current.position
    )
    new_token.add_tag('visarga')
    new_token.add_tag('sandhi_result')
    new_token.set_meta('rule_applied', 'sasajusho_ruh')
    new_token.set_meta('original', 's')
    
    new_tokens = tokens[:index] + [new_token] + tokens[index + 1:]
    
    return new_tokens, index + 1


def _match_khar_visarga(tokens: List[Token], index: int) -> bool:
    """Match visarga before khar consonants."""
    if index >= len(tokens) - 1:
        return False
    
    current = tokens[index]
    next_token = tokens[index + 1]
    
    # khar consonants (voiceless stops and sibilants)
    khar_consonants = {'k', 'kh', 'c', 'ch', 'ṭ', 'ṭh', 't', 'th', 'p', 'ph', 'ś', 'ṣ', 's'}
    
    return (current.text == 'ḥ' and next_token.kind == TokenKind.CONSONANT and
            next_token.text in khar_consonants)


def _apply_khar_visarga(tokens: List[Token], index: int) -> Tuple[List[Token], int]:
    """Apply visarga changes before khar consonants."""
    # Simplified - would need more complex rules for different contexts
    return tokens, index + 1


def _apply_purvatrasiddham(registry) -> None:
    """Apply the pūrvatrāsiddham principle."""
    # This would implement rule ordering logic
    # For now, just a placeholder
    pass