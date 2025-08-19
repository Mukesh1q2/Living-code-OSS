"""
Bidirectional mapping components for multi-domain translation.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

from .multi_domain_mapper import DomainType, DomainMapping, AlgorithmicSanskritDSL


class BidirectionalMapper:
    """Handles bidirectional mapping between domains."""
    
    def __init__(self):
        self.reverse_patterns = self._initialize_reverse_patterns()
        self.mapping_cache: Dict[str, DomainMapping] = {}
    
    def _initialize_reverse_patterns(self) -> Dict[Tuple[DomainType, DomainType], Dict[str, str]]:
        """Initialize patterns for reverse translation."""
        return {
            (DomainType.PROGRAMMING, DomainType.SANSKRIT): {
                'if': 'यदि',
                'else': 'अन्यथा',
                'while': 'यावत्',
                'for': 'प्रत्येकम्',
                'def': 'कार्यम्',
                'return': 'प्रत्यावर्तनम्',
                'class': 'वर्गः',
                '+': 'योगः',
                '*': 'गुणनम्',
            },
            (DomainType.MATHEMATICS, DomainType.SANSKRIT): {
                '+': 'योगः',
                '-': 'व्यवकलनम्',
                '*': 'गुणनम्',
                '/': 'भागः',
                'sin': 'ज्या',
                'cos': 'कोज्या',
                'tan': 'स्पर्शज्या',
                'pi': 'पाई',
                'e': 'ई',
            }
        }
    
    def translate_reverse(self, content: str, source_domain: DomainType, 
                         target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate from target back to source domain."""
        reverse_key = (source_domain, target_domain)
        
        if reverse_key in self.reverse_patterns:
            patterns = self.reverse_patterns[reverse_key]
            translated_content = self._apply_reverse_patterns(content, patterns)
            
            return DomainMapping(
                id=f"reverse_{source_domain.value}_to_{target_domain.value}_{hash(content)}",
                source_domain=source_domain,
                target_domain=target_domain,
                source_content=content,
                target_content=translated_content,
                mapping_type="reverse_translation",
                confidence=0.7,
                metadata={'translation_method': 'reverse_pattern_matching'}
            )
        else:
            raise ValueError(f"No reverse translator available for {source_domain} to {target_domain}")
    
    def _apply_reverse_patterns(self, content: str, patterns: Dict[str, str]) -> str:
        """Apply reverse translation patterns."""
        translated = content
        
        for source_pattern, target_pattern in patterns.items():
            translated = translated.replace(source_pattern, target_pattern)
        
        return translated
    
    def create_bidirectional_mapping(self, content: str, source_domain: DomainType, 
                                   target_domain: DomainType, **kwargs) -> Tuple[DomainMapping, DomainMapping]:
        """Create bidirectional mapping between domains."""
        # For this implementation, we'll create a simple forward mapping
        # and then reverse it using patterns
        forward_mapping = DomainMapping(
            id=f"forward_{source_domain.value}_to_{target_domain.value}_{hash(content)}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_content=content,
            target_content=content,  # Placeholder
            mapping_type="forward_translation",
            confidence=0.8
        )
        
        # Reverse translation
        reverse_mapping = self.translate_reverse(
            forward_mapping.target_content, target_domain, source_domain, **kwargs
        )
        
        return forward_mapping, reverse_mapping
    
    def validate_bidirectional_consistency(self, forward_mapping: DomainMapping, 
                                         reverse_mapping: DomainMapping) -> float:
        """Validate consistency of bidirectional translation."""
        original = forward_mapping.source_content
        round_trip = reverse_mapping.target_content
        
        # Simple similarity measure
        similarity = self._calculate_similarity(original, round_trip)
        
        return similarity
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class MultiDomainMapper:
    """Main class for multi-domain mapping system."""
    
    def __init__(self, reasoning_core=None, symbolic_engine=None):
        self.reasoning_core = reasoning_core
        self.symbolic_engine = symbolic_engine
        
        # Initialize components
        self.bidirectional_mapper = BidirectionalMapper()
        self.algorithmic_dsl = AlgorithmicSanskritDSL()
        
        # Import translators
        from .multi_domain_mapper import (
            SanskritToProgrammingTranslator, 
            SanskritToMathTranslator,
            SanskritToKnowledgeGraphTranslator
        )
        
        # Initialize translators with dependencies
        self.translators = {
            (DomainType.SANSKRIT, DomainType.PROGRAMMING): SanskritToProgrammingTranslator(),
            (DomainType.SANSKRIT, DomainType.MATHEMATICS): SanskritToMathTranslator(symbolic_engine),
            (DomainType.SANSKRIT, DomainType.KNOWLEDGE_GRAPH): SanskritToKnowledgeGraphTranslator(),
        }
        
        # Mapping history and cache
        self.mapping_history: List[DomainMapping] = []
        self.semantic_consistency_cache: Dict[str, float] = {}
    
    def translate(self, content: str, source_domain: DomainType, 
                 target_domain: DomainType, **kwargs) -> DomainMapping:
        """Translate content between domains."""
        translator_key = (source_domain, target_domain)
        
        if translator_key in self.translators:
            translator = self.translators[translator_key]
            mapping = translator.translate(content, source_domain, target_domain, **kwargs)
            
            # Store in history
            self.mapping_history.append(mapping)
            
            return mapping
        else:
            # Try bidirectional mapper
            return self.bidirectional_mapper.translate_reverse(
                content, source_domain, target_domain, **kwargs
            )
    
    def create_algorithmic_sanskrit(self, expression: str):
        """Create algorithmic Sanskrit expression."""
        return self.algorithmic_dsl.parse_expression(expression)
    
    def compile_algorithmic_sanskrit(self, expression, target_domain: DomainType, **kwargs) -> str:
        """Compile algorithmic Sanskrit to target domain."""
        return self.algorithmic_dsl.compile_to_target(expression, target_domain, **kwargs)
    
    def create_bidirectional_mapping(self, content: str, source_domain: DomainType, 
                                   target_domain: DomainType, **kwargs) -> Dict[str, Any]:
        """Create and validate bidirectional mapping."""
        forward_mapping, reverse_mapping = self.bidirectional_mapper.create_bidirectional_mapping(
            content, source_domain, target_domain, **kwargs
        )
        
        consistency_score = self.bidirectional_mapper.validate_bidirectional_consistency(
            forward_mapping, reverse_mapping
        )
        
        return {
            'forward_mapping': forward_mapping,
            'reverse_mapping': reverse_mapping,
            'consistency_score': consistency_score,
            'is_consistent': consistency_score > 0.7
        }
    
    def validate_semantic_preservation(self, mapping: DomainMapping) -> float:
        """Validate that semantic meaning is preserved across domains."""
        cache_key = f"{mapping.source_domain.value}_{mapping.target_domain.value}_{hash(mapping.source_content)}"
        
        if cache_key in self.semantic_consistency_cache:
            return self.semantic_consistency_cache[cache_key]
        
        # Use translator's validation method
        translator_key = (mapping.source_domain, mapping.target_domain)
        if translator_key in self.translators:
            translator = self.translators[translator_key]
            is_valid = translator.validate_translation(mapping)
            score = 1.0 if is_valid else 0.0
        else:
            # Fallback validation
            score = self._fallback_semantic_validation(mapping)
        
        self.semantic_consistency_cache[cache_key] = score
        return score
    
    def _fallback_semantic_validation(self, mapping: DomainMapping) -> float:
        """Fallback semantic validation when no specific validator is available."""
        # Basic validation - check that target content is not empty and different from source
        if not mapping.target_content or mapping.target_content == mapping.source_content:
            return 0.0
        
        # Check for reasonable length ratio
        source_len = len(mapping.source_content)
        target_len = len(mapping.target_content)
        
        if source_len == 0:
            return 0.0
        
        length_ratio = target_len / source_len
        
        # Reasonable translations should have length ratios between 0.5 and 3.0
        if 0.5 <= length_ratio <= 3.0:
            return 0.6  # Moderate confidence
        else:
            return 0.3  # Low confidence
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about mappings performed."""
        if not self.mapping_history:
            return {'total_mappings': 0}
        
        stats = {
            'total_mappings': len(self.mapping_history),
            'by_source_domain': {},
            'by_target_domain': {},
            'by_mapping_type': {},
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for mapping in self.mapping_history:
            # Count by source domain
            source = mapping.source_domain.value
            stats['by_source_domain'][source] = stats['by_source_domain'].get(source, 0) + 1
            
            # Count by target domain
            target = mapping.target_domain.value
            stats['by_target_domain'][target] = stats['by_target_domain'].get(target, 0) + 1
            
            # Count by mapping type
            mapping_type = mapping.mapping_type
            stats['by_mapping_type'][mapping_type] = stats['by_mapping_type'].get(mapping_type, 0) + 1
            
            # Sum confidence
            total_confidence += mapping.confidence
        
        stats['average_confidence'] = total_confidence / len(self.mapping_history)
        
        return stats
    
    def export_mappings(self, format: str = 'json') -> str:
        """Export mapping history in specified format."""
        if format == 'json':
            mappings_data = [mapping.to_dict() for mapping in self.mapping_history]
            return json.dumps(mappings_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_mappings(self, data: str, format: str = 'json') -> int:
        """Import mappings from data."""
        if format == 'json':
            mappings_data = json.loads(data)
            imported_count = 0
            
            for mapping_dict in mappings_data:
                try:
                    mapping = DomainMapping(
                        id=mapping_dict['id'],
                        source_domain=DomainType(mapping_dict['source_domain']),
                        target_domain=DomainType(mapping_dict['target_domain']),
                        source_content=mapping_dict['source_content'],
                        target_content=mapping_dict['target_content'],
                        mapping_type=mapping_dict['mapping_type'],
                        confidence=mapping_dict.get('confidence', 1.0),
                        metadata=mapping_dict.get('metadata', {}),
                        created_at=mapping_dict.get('created_at', datetime.now().isoformat())
                    )
                    self.mapping_history.append(mapping)
                    imported_count += 1
                except Exception as e:
                    print(f"Failed to import mapping: {e}")
            
            return imported_count
        else:
            raise ValueError(f"Unsupported import format: {format}")