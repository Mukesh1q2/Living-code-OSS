"""
Sanskrit Engine Adapter for Vidya Quantum Interface

This module provides an adapter that wraps the existing Sanskrit rewrite engine
with enhanced interfaces for real-time processing, visualization data generation,
and quantum consciousness integration.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator, Tuple, Set
from pathlib import Path
import uuid
from datetime import datetime

# Import existing Sanskrit engine components
try:
    from src.sanskrit_rewrite_engine.engine import SanskritRewriteEngine, TransformationResult
    from src.sanskrit_rewrite_engine.tokenizer import BasicSanskritTokenizer, Token
    from src.sanskrit_rewrite_engine.rules import Rule, RuleRegistry
    from src.sanskrit_rewrite_engine.config import EngineConfig
except ImportError as e:
    logging.warning(f"Could not import Sanskrit engine components: {e}")
    # Fallback for development
    SanskritRewriteEngine = None
    TransformationResult = None
    BasicSanskritTokenizer = None
    Token = None
    Rule = None
    RuleRegistry = None
    EngineConfig = None

logger = logging.getLogger(__name__)


@dataclass
class QuantumToken:
    """Enhanced token with quantum properties for visualization"""
    text: str
    position: Dict[str, int]  # {start, end, line, column}
    morphology: Dict[str, Any]
    quantum_properties: Dict[str, Any]
    visualization_data: Dict[str, Any]
    original_token: Optional[Any] = None  # Reference to original Token object


@dataclass
class NetworkNode:
    """Neural network node representing Sanskrit rules or morphological data"""
    id: str
    position: Dict[str, float]  # {x, y, z} coordinates
    type: str  # 'sanskrit-rule', 'morpheme', 'phoneme', etc.
    rule_data: Optional[Dict[str, Any]] = None
    activation_level: float = 0.0
    connections: List[str] = field(default_factory=list)
    quantum_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationData:
    """Complete visualization data for the quantum interface"""
    tokens: List[QuantumToken]
    network_nodes: List[NetworkNode]
    connections: List[Dict[str, Any]]
    quantum_effects: List[Dict[str, Any]]
    processing_trace: List[Dict[str, Any]]


@dataclass
class ProcessingUpdate:
    """Real-time processing update for streaming interface"""
    update_type: str  # 'token_processed', 'rule_applied', 'analysis_complete', etc.
    timestamp: float
    data: Dict[str, Any]
    progress: float  # 0.0 to 1.0
    visualization_update: Optional[Dict[str, Any]] = None


class SanskritEngineAdapter:
    """
    Adapter class that wraps existing Sanskrit processing functionality
    with enhanced interfaces for the Vidya quantum consciousness system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Sanskrit engine adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processing_id_counter = 0
        self.active_streams: Dict[str, bool] = {}
        
        # Initialize the underlying Sanskrit engine
        self._initialize_engine()
        
        # Initialize visualization components
        self._initialize_visualization_system()
        
        # Initialize quantum state management
        self._initialize_quantum_system()
        
        logger.info("Sanskrit Engine Adapter initialized successfully")
    
    def _initialize_engine(self) -> None:
        """Initialize the underlying Sanskrit rewrite engine"""
        try:
            if SanskritRewriteEngine is not None:
                self.engine = SanskritRewriteEngine(self.config.get('engine_config'))
                self.tokenizer = BasicSanskritTokenizer()
                logger.info("Sanskrit engine initialized successfully")
            else:
                # Fallback for development/testing
                self.engine = None
                self.tokenizer = None
                logger.warning("Sanskrit engine not available, using fallback mode")
        except Exception as e:
            logger.error(f"Failed to initialize Sanskrit engine: {e}")
            self.engine = None
            self.tokenizer = None
    
    def _initialize_visualization_system(self) -> None:
        """Initialize the visualization data generation system"""
        self.node_positions = {}  # Cache for node positions
        self.connection_cache = {}  # Cache for rule connections
        self.quantum_effects_cache = {}  # Cache for quantum effects
        
        # Predefined positions for common Sanskrit rule types
        self.rule_type_positions = {
            'vowel_sandhi': {'base_x': 0, 'base_y': 0, 'base_z': 0},
            'consonant_sandhi': {'base_x': 2, 'base_y': 0, 'base_z': 0},
            'morphological': {'base_x': 0, 'base_y': 2, 'base_z': 0},
            'compound': {'base_x': 2, 'base_y': 2, 'base_z': 0},
            'phonological': {'base_x': 1, 'base_y': 1, 'base_z': 1}
        }
    
    def _initialize_quantum_system(self) -> None:
        """Initialize quantum state management for consciousness integration"""
        self.quantum_states = {}
        self.entanglement_pairs = {}
        self.superposition_cache = {}
        
        # Quantum effect templates
        self.quantum_effect_templates = {
            'superposition': {
                'type': 'particle_system',
                'particles': 'probability_cloud',
                'color': '#4a90e2',
                'opacity': 0.6,
                'animation': 'quantum_fluctuation'
            },
            'entanglement': {
                'type': 'connection_line',
                'style': 'quantum_thread',
                'color': '#e24a90',
                'animation': 'instant_correlation'
            },
            'collapse': {
                'type': 'wave_collapse',
                'effect': 'decoherence',
                'duration': 1.5,
                'color': '#90e24a'
            }
        }
    
    async def process_text_streaming(self, text: str, 
                                   enable_visualization: bool = True) -> AsyncIterator[ProcessingUpdate]:
        """
        Process Sanskrit text with real-time streaming updates.
        
        Args:
            text: Input Sanskrit text to process
            enable_visualization: Whether to generate visualization data
            
        Yields:
            ProcessingUpdate objects with real-time processing information
        """
        processing_id = str(uuid.uuid4())
        self.active_streams[processing_id] = True
        
        try:
            # Initial update
            yield ProcessingUpdate(
                update_type='processing_started',
                timestamp=time.time(),
                data={'text': text, 'processing_id': processing_id},
                progress=0.0
            )
            
            # Tokenization phase
            if not self.active_streams.get(processing_id, False):
                return
            
            tokens = await self._tokenize_with_updates(text, processing_id, enable_visualization)
            
            yield ProcessingUpdate(
                update_type='tokenization_complete',
                timestamp=time.time(),
                data={'token_count': len(tokens)},
                progress=0.3,
                visualization_update={'tokens': [self._token_to_dict(t) for t in tokens]} if enable_visualization else None
            )
            
            # Rule application phase
            if not self.active_streams.get(processing_id, False):
                return
            
            async for rule_update in self._apply_rules_streaming(text, tokens, processing_id, enable_visualization):
                if not self.active_streams.get(processing_id, False):
                    return
                yield rule_update
            
            # Generate final visualization data
            if enable_visualization and self.active_streams.get(processing_id, False):
                visualization_data = await self._generate_complete_visualization(text, tokens)
                
                yield ProcessingUpdate(
                    update_type='visualization_complete',
                    timestamp=time.time(),
                    data={'visualization_ready': True},
                    progress=0.9,
                    visualization_update=visualization_data
                )
            
            # Final completion update
            if self.active_streams.get(processing_id, False):
                yield ProcessingUpdate(
                    update_type='processing_complete',
                    timestamp=time.time(),
                    data={'success': True, 'processing_id': processing_id},
                    progress=1.0
                )
        
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            yield ProcessingUpdate(
                update_type='processing_error',
                timestamp=time.time(),
                data={'error': str(e), 'processing_id': processing_id},
                progress=0.0
            )
        
        finally:
            # Clean up
            if processing_id in self.active_streams:
                del self.active_streams[processing_id]
    
    async def _tokenize_with_updates(self, text: str, processing_id: str, 
                                   enable_visualization: bool) -> List[QuantumToken]:
        """Tokenize text with streaming updates"""
        if self.tokenizer is None:
            # Fallback tokenization for development
            return await self._fallback_tokenize(text, enable_visualization)
        
        try:
            # Use existing tokenizer
            basic_tokens = self.tokenizer.tokenize(text)
            quantum_tokens = []
            
            for i, token in enumerate(basic_tokens):
                if not self.active_streams.get(processing_id, False):
                    break
                
                # Convert to quantum token with visualization data
                quantum_token = await self._create_quantum_token(token, i, enable_visualization)
                quantum_tokens.append(quantum_token)
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.01)
            
            return quantum_tokens
        
        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            return await self._fallback_tokenize(text, enable_visualization)
    
    async def _fallback_tokenize(self, text: str, enable_visualization: bool) -> List[QuantumToken]:
        """Fallback tokenization when main engine is not available"""
        tokens = []
        words = text.split()
        
        for i, word in enumerate(words):
            quantum_token = QuantumToken(
                text=word,
                position={'start': 0, 'end': len(word), 'line': 1, 'column': i + 1},
                morphology={
                    'root': 'unknown',
                    'suffixes': [],
                    'grammatical_category': 'unknown',
                    'semantic_role': 'unknown'
                },
                quantum_properties={
                    'superposition': False,
                    'entanglements': [],
                    'probability': 1.0,
                    'quantum_state': 'classical'
                },
                visualization_data={
                    'color': '#4a90e2',
                    'size': 1.0,
                    'animation': 'pulse',
                    'effects': ['glow'],
                    'position': {'x': i * 2.0, 'y': 0.0, 'z': 0.0}
                } if enable_visualization else {}
            )
            tokens.append(quantum_token)
        
        return tokens
    
    async def _create_quantum_token(self, token: Any, index: int, 
                                  enable_visualization: bool) -> QuantumToken:
        """Create a quantum token from a basic token"""
        # Extract basic token information
        text = getattr(token, 'text', str(token))
        start_pos = getattr(token, 'start_pos', 0)
        end_pos = getattr(token, 'end_pos', len(text))
        
        # Generate morphological analysis
        morphology = await self._analyze_morphology(token)
        
        # Generate quantum properties
        quantum_properties = await self._generate_quantum_properties(token, index)
        
        # Generate visualization data
        visualization_data = {}
        if enable_visualization:
            visualization_data = await self._generate_token_visualization(token, index)
        
        return QuantumToken(
            text=text,
            position={
                'start': start_pos,
                'end': end_pos,
                'line': 1,  # Simplified for now
                'column': index + 1
            },
            morphology=morphology,
            quantum_properties=quantum_properties,
            visualization_data=visualization_data,
            original_token=token
        )
    
    async def _analyze_morphology(self, token: Any) -> Dict[str, Any]:
        """Analyze morphological properties of a token"""
        # Extract existing morphological features if available
        if hasattr(token, 'morphological_features'):
            return dict(token.morphological_features)
        
        # Basic morphological analysis for fallback
        text = getattr(token, 'text', str(token))
        
        return {
            'root': self._extract_root(text),
            'suffixes': self._extract_suffixes(text),
            'grammatical_category': self._determine_category(text),
            'semantic_role': self._determine_semantic_role(text),
            'phonetic_features': self._extract_phonetic_features(text)
        }
    
    def _extract_root(self, text: str) -> str:
        """Extract the root form of a word (simplified)"""
        # This is a simplified implementation
        # In a full implementation, this would use sophisticated morphological analysis
        common_suffixes = ['ति', 'सि', 'मि', 'तु', 'सु', 'मु', 'ाम्', 'ान्', 'ात्']
        
        for suffix in common_suffixes:
            if text.endswith(suffix):
                return text[:-len(suffix)]
        
        return text
    
    def _extract_suffixes(self, text: str) -> List[str]:
        """Extract suffixes from a word (simplified)"""
        common_suffixes = ['ति', 'सि', 'मि', 'तु', 'सु', 'मु', 'ाम्', 'ान्', 'ात्']
        
        for suffix in common_suffixes:
            if text.endswith(suffix):
                return [suffix]
        
        return []
    
    def _determine_category(self, text: str) -> str:
        """Determine grammatical category (simplified)"""
        # Simplified categorization based on endings
        if text.endswith(('ति', 'सि', 'मि', 'तु', 'सु', 'मु')):
            return 'verb'
        elif text.endswith(('ः', 'म्', 'न्')):
            return 'noun'
        else:
            return 'unknown'
    
    def _determine_semantic_role(self, text: str) -> str:
        """Determine semantic role (simplified)"""
        # This would be much more sophisticated in a full implementation
        return 'content_word'
    
    def _extract_phonetic_features(self, text: str) -> Dict[str, Any]:
        """Extract phonetic features from text"""
        return {
            'syllable_count': self._count_syllables(text),
            'vowel_pattern': self._extract_vowel_pattern(text),
            'consonant_clusters': self._find_consonant_clusters(text)
        }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in Sanskrit text (simplified)"""
        vowels = 'अआइईउऊऋॠऌॡएऐओऔaāiīuūṛṝḷḹeaioau'
        return sum(1 for char in text if char in vowels)
    
    def _extract_vowel_pattern(self, text: str) -> List[str]:
        """Extract vowel pattern from text"""
        vowels = 'अआइईउऊऋॠऌॡएऐओऔaāiīuūṛṝḷḹeaioau'
        return [char for char in text if char in vowels]
    
    def _find_consonant_clusters(self, text: str) -> List[str]:
        """Find consonant clusters in text (simplified)"""
        # This is a very simplified implementation
        clusters = []
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहkkhgghṅcchj'
        
        current_cluster = ""
        for char in text:
            if char in consonants:
                current_cluster += char
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = ""
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters
    
    async def _generate_quantum_properties(self, token: Any, index: int) -> Dict[str, Any]:
        """Generate quantum properties for a token"""
        text = getattr(token, 'text', str(token))
        
        # Determine if token should be in superposition
        superposition = len(text) > 3 and 'अ' in text  # Simplified condition
        
        # Generate entanglements based on morphological similarity
        entanglements = []
        if hasattr(token, 'morphological_features'):
            # In a full implementation, this would analyze morphological relationships
            pass
        
        return {
            'superposition': superposition,
            'entanglements': entanglements,
            'probability': 1.0 if not superposition else 0.7,
            'quantum_state': 'superposition' if superposition else 'classical',
            'coherence_time': 5.0 if superposition else float('inf'),
            'measurement_count': 0
        }
    
    async def _generate_token_visualization(self, token: Any, index: int) -> Dict[str, Any]:
        """Generate visualization data for a token"""
        text = getattr(token, 'text', str(token))
        
        # Color based on token type/properties
        color = self._determine_token_color(token)
        
        # Size based on importance/length
        size = min(2.0, max(0.5, len(text) / 5.0))
        
        # Animation based on quantum properties
        animation = 'quantum_pulse' if len(text) > 3 else 'gentle_glow'
        
        # Effects based on morphological properties
        effects = ['glow']
        if 'अ' in text:  # Simplified condition for special effects
            effects.append('sanskrit_resonance')
        
        return {
            'color': color,
            'size': size,
            'animation': animation,
            'effects': effects,
            'position': {
                'x': index * 2.0,
                'y': 0.0,
                'z': 0.0
            },
            'opacity': 0.8,
            'glow_intensity': 0.5
        }
    
    def _determine_token_color(self, token: Any) -> str:
        """Determine color for token visualization"""
        text = getattr(token, 'text', str(token))
        
        # Color scheme based on Sanskrit phonetics
        if any(vowel in text for vowel in 'अआइईउऊ'):
            return '#4a90e2'  # Blue for vowels
        elif any(consonant in text for consonant in 'कखगघङ'):
            return '#e24a90'  # Pink for velars
        elif any(consonant in text for consonant in 'चछजझञ'):
            return '#90e24a'  # Green for palatals
        else:
            return '#e2904a'  # Orange for others
    
    async def _apply_rules_streaming(self, text: str, tokens: List[QuantumToken], 
                                   processing_id: str, enable_visualization: bool) -> AsyncIterator[ProcessingUpdate]:
        """Apply Sanskrit rules with streaming updates"""
        if self.engine is None:
            # Fallback rule application
            yield ProcessingUpdate(
                update_type='rule_application_complete',
                timestamp=time.time(),
                data={'rules_applied': 0, 'fallback_mode': True},
                progress=0.8
            )
            return
        
        try:
            # Process text through the engine
            result = self.engine.process(text, enable_tracing=True)
            
            if result.trace:
                total_steps = len(result.trace)
                for i, trace_step in enumerate(result.trace):
                    if not self.active_streams.get(processing_id, False):
                        return
                    
                    # Generate update for this rule application
                    update_data = {
                        'step': trace_step.get('step', 'unknown'),
                        'rule_name': trace_step.get('rule_name', 'unknown'),
                        'iteration': trace_step.get('iteration', 0),
                        'text_before': trace_step.get('before', ''),
                        'text_after': trace_step.get('after', ''),
                        'progress_detail': f"Step {i+1} of {total_steps}"
                    }
                    
                    visualization_update = None
                    if enable_visualization and trace_step.get('step') == 'rule_application':
                        visualization_update = await self._generate_rule_visualization(trace_step)
                    
                    yield ProcessingUpdate(
                        update_type='rule_applied',
                        timestamp=time.time(),
                        data=update_data,
                        progress=0.3 + (0.5 * (i + 1) / total_steps),
                        visualization_update=visualization_update
                    )
                    
                    # Small delay for real-time effect
                    await asyncio.sleep(0.05)
            
            # Final rule application summary
            yield ProcessingUpdate(
                update_type='rule_application_complete',
                timestamp=time.time(),
                data={
                    'rules_applied': len(result.transformations_applied),
                    'final_text': result.output_text,
                    'convergence_reached': result.convergence_reached,
                    'iterations_used': result.iterations_used
                },
                progress=0.8
            )
        
        except Exception as e:
            logger.error(f"Error in rule application: {e}")
            yield ProcessingUpdate(
                update_type='rule_application_error',
                timestamp=time.time(),
                data={'error': str(e)},
                progress=0.8
            )
    
    async def _generate_rule_visualization(self, trace_step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data for a rule application"""
        rule_id = trace_step.get('rule_id', 'unknown')
        rule_name = trace_step.get('rule_name', 'Unknown Rule')
        
        # Create network node for this rule
        node = NetworkNode(
            id=f"rule_{rule_id}_{int(time.time())}",
            position=self._get_rule_position(rule_id, rule_name),
            type='sanskrit-rule',
            rule_data={
                'id': rule_id,
                'name': rule_name,
                'pattern': trace_step.get('pattern', ''),
                'replacement': trace_step.get('replacement', ''),
                'application_count': 1
            },
            activation_level=1.0,
            quantum_properties={
                'rule_confidence': 0.9,
                'quantum_effect': 'rule_activation',
                'entanglement_strength': 0.8
            }
        )
        
        return {
            'type': 'rule_visualization',
            'node': self._network_node_to_dict(node),
            'quantum_effects': [
                {
                    'type': 'rule_activation',
                    'position': node.position,
                    'duration': 2.0,
                    'color': '#4a90e2'
                }
            ]
        }
    
    def _get_rule_position(self, rule_id: str, rule_name: str) -> Dict[str, float]:
        """Get 3D position for a rule node"""
        # Use cached position if available
        if rule_id in self.node_positions:
            return self.node_positions[rule_id]
        
        # Determine position based on rule type
        rule_type = self._classify_rule_type(rule_name)
        base_pos = self.rule_type_positions.get(rule_type, {'base_x': 0, 'base_y': 0, 'base_z': 0})
        
        # Add some randomization to avoid overlap
        import random
        position = {
            'x': base_pos['base_x'] + random.uniform(-0.5, 0.5),
            'y': base_pos['base_y'] + random.uniform(-0.5, 0.5),
            'z': base_pos['base_z'] + random.uniform(-0.5, 0.5)
        }
        
        # Cache the position
        self.node_positions[rule_id] = position
        return position
    
    def _classify_rule_type(self, rule_name: str) -> str:
        """Classify rule type based on rule name"""
        rule_name_lower = rule_name.lower()
        
        if 'vowel' in rule_name_lower or 'sandhi' in rule_name_lower:
            return 'vowel_sandhi'
        elif 'consonant' in rule_name_lower:
            return 'consonant_sandhi'
        elif 'compound' in rule_name_lower:
            return 'compound'
        elif 'morphological' in rule_name_lower:
            return 'morphological'
        else:
            return 'phonological'
    
    async def _generate_complete_visualization(self, text: str, 
                                             tokens: List[QuantumToken]) -> Dict[str, Any]:
        """Generate complete visualization data for the quantum interface"""
        # Generate network nodes from Pāṇini rules and morphological data
        network_nodes = await self._generate_network_nodes(tokens)
        
        # Generate connections between nodes
        connections = await self._generate_node_connections(network_nodes)
        
        # Generate quantum effects
        quantum_effects = await self._generate_quantum_effects(tokens, network_nodes)
        
        return {
            'tokens': [self._quantum_token_to_dict(token) for token in tokens],
            'network_nodes': [self._network_node_to_dict(node) for node in network_nodes],
            'connections': connections,
            'quantum_effects': quantum_effects,
            'metadata': {
                'text_length': len(text),
                'token_count': len(tokens),
                'node_count': len(network_nodes),
                'connection_count': len(connections),
                'generation_timestamp': time.time()
            }
        }
    
    async def _generate_network_nodes(self, tokens: List[QuantumToken]) -> List[NetworkNode]:
        """Generate network nodes from tokens and rules"""
        nodes = []
        
        # Create nodes for each token
        for i, token in enumerate(tokens):
            node = NetworkNode(
                id=f"token_{i}",
                position={
                    'x': token.visualization_data.get('position', {}).get('x', i * 2.0),
                    'y': token.visualization_data.get('position', {}).get('y', 0.0),
                    'z': token.visualization_data.get('position', {}).get('z', 0.0)
                },
                type='token',
                rule_data={
                    'text': token.text,
                    'morphology': token.morphology,
                    'quantum_properties': token.quantum_properties
                },
                activation_level=0.5,
                quantum_properties=token.quantum_properties
            )
            nodes.append(node)
        
        # Add morphological analysis nodes
        morphology_nodes = await self._generate_morphology_nodes(tokens)
        nodes.extend(morphology_nodes)
        
        # Add Pāṇini rule nodes (if engine is available)
        if self.engine and hasattr(self.engine, 'rule_registry'):
            rule_nodes = await self._generate_panini_rule_nodes()
            nodes.extend(rule_nodes)
        
        return nodes
    
    async def _generate_morphology_nodes(self, tokens: List[QuantumToken]) -> List[NetworkNode]:
        """Generate nodes representing morphological analysis"""
        nodes = []
        
        # Extract unique morphological features
        roots = set()
        categories = set()
        
        for token in tokens:
            if token.morphology.get('root'):
                roots.add(token.morphology['root'])
            if token.morphology.get('grammatical_category'):
                categories.add(token.morphology['grammatical_category'])
        
        # Create nodes for roots
        for i, root in enumerate(roots):
            node = NetworkNode(
                id=f"root_{root}",
                position={'x': -2.0, 'y': i * 1.5, 'z': 1.0},
                type='morphological_root',
                rule_data={'root': root, 'type': 'root'},
                activation_level=0.3,
                quantum_properties={'morphological_strength': 0.8}
            )
            nodes.append(node)
        
        # Create nodes for grammatical categories
        for i, category in enumerate(categories):
            node = NetworkNode(
                id=f"category_{category}",
                position={'x': 4.0, 'y': i * 1.5, 'z': 1.0},
                type='grammatical_category',
                rule_data={'category': category, 'type': 'category'},
                activation_level=0.4,
                quantum_properties={'categorical_strength': 0.7}
            )
            nodes.append(node)
        
        return nodes
    
    async def _generate_panini_rule_nodes(self) -> List[NetworkNode]:
        """Generate nodes representing Pāṇini rules"""
        nodes = []
        
        try:
            # Get rules from the engine's rule registry
            rules = self.engine.rule_registry.get_rules_by_priority()
            
            for i, rule in enumerate(rules[:10]):  # Limit to first 10 rules for visualization
                position = self._get_rule_position(rule.id, rule.name)
                
                node = NetworkNode(
                    id=f"panini_rule_{rule.id}",
                    position=position,
                    type='panini_rule',
                    rule_data={
                        'id': rule.id,
                        'name': rule.name,
                        'description': rule.description,
                        'pattern': rule.pattern,
                        'replacement': rule.replacement,
                        'priority': rule.priority,
                        'metadata': rule.metadata
                    },
                    activation_level=0.2,
                    quantum_properties={
                        'rule_authority': 1.0,  # Pāṇini rules have highest authority
                        'sutra_reference': rule.metadata.get('sutra_ref', 'unknown'),
                        'quantum_coherence': 0.95
                    }
                )
                nodes.append(node)
        
        except Exception as e:
            logger.error(f"Error generating Pāṇini rule nodes: {e}")
        
        return nodes
    
    async def _generate_node_connections(self, nodes: List[NetworkNode]) -> List[Dict[str, Any]]:
        """Generate connections between network nodes"""
        connections = []
        
        # Connect tokens to their morphological roots
        for node in nodes:
            if node.type == 'token' and node.rule_data:
                root = node.rule_data.get('morphology', {}).get('root')
                if root:
                    root_node_id = f"root_{root}"
                    if any(n.id == root_node_id for n in nodes):
                        connections.append({
                            'from': node.id,
                            'to': root_node_id,
                            'type': 'morphological_derivation',
                            'strength': 0.8,
                            'quantum_properties': {
                                'entanglement_strength': 0.6,
                                'connection_type': 'morphological'
                            }
                        })
        
        # Connect tokens to their grammatical categories
        for node in nodes:
            if node.type == 'token' and node.rule_data:
                category = node.rule_data.get('morphology', {}).get('grammatical_category')
                if category:
                    category_node_id = f"category_{category}"
                    if any(n.id == category_node_id for n in nodes):
                        connections.append({
                            'from': node.id,
                            'to': category_node_id,
                            'type': 'categorical_membership',
                            'strength': 0.7,
                            'quantum_properties': {
                                'entanglement_strength': 0.5,
                                'connection_type': 'categorical'
                            }
                        })
        
        # Connect Pāṇini rules to relevant tokens (simplified)
        panini_nodes = [n for n in nodes if n.type == 'panini_rule']
        token_nodes = [n for n in nodes if n.type == 'token']
        
        for panini_node in panini_nodes:
            for token_node in token_nodes:
                # Simplified connection logic - in reality this would be much more sophisticated
                if self._rule_applies_to_token(panini_node, token_node):
                    connections.append({
                        'from': panini_node.id,
                        'to': token_node.id,
                        'type': 'rule_application',
                        'strength': 0.9,
                        'quantum_properties': {
                            'entanglement_strength': 0.8,
                            'connection_type': 'transformational',
                            'sutra_authority': 1.0
                        }
                    })
        
        return connections
    
    def _rule_applies_to_token(self, rule_node: NetworkNode, token_node: NetworkNode) -> bool:
        """Check if a Pāṇini rule applies to a token (simplified)"""
        # This is a very simplified check - in reality this would involve
        # sophisticated pattern matching and contextual analysis
        
        rule_data = rule_node.rule_data or {}
        token_data = token_node.rule_data or {}
        
        pattern = rule_data.get('pattern', '')
        token_text = token_data.get('text', '')
        
        # Simple pattern matching
        if pattern and token_text:
            try:
                import re
                return bool(re.search(pattern, token_text))
            except:
                return False
        
        return False
    
    async def _generate_quantum_effects(self, tokens: List[QuantumToken], 
                                      nodes: List[NetworkNode]) -> List[Dict[str, Any]]:
        """Generate quantum effects for visualization"""
        effects = []
        
        # Generate superposition effects for tokens in quantum superposition
        for token in tokens:
            if token.quantum_properties.get('superposition', False):
                effects.append({
                    'type': 'superposition',
                    'target_id': f"token_{tokens.index(token)}",
                    'position': token.visualization_data.get('position', {'x': 0, 'y': 0, 'z': 0}),
                    'properties': self.quantum_effect_templates['superposition'],
                    'duration': token.quantum_properties.get('coherence_time', 5.0),
                    'intensity': token.quantum_properties.get('probability', 0.7)
                })
        
        # Generate entanglement effects between connected nodes
        entangled_pairs = []
        for i, token in enumerate(tokens):
            entanglements = token.quantum_properties.get('entanglements', [])
            for entangled_id in entanglements:
                if (i, entangled_id) not in entangled_pairs and (entangled_id, i) not in entangled_pairs:
                    entangled_pairs.append((i, entangled_id))
                    
                    effects.append({
                        'type': 'entanglement',
                        'source_id': f"token_{i}",
                        'target_id': f"token_{entangled_id}",
                        'properties': self.quantum_effect_templates['entanglement'],
                        'strength': 0.8,
                        'bidirectional': True
                    })
        
        # Generate rule activation effects for Pāṇini rules
        panini_nodes = [n for n in nodes if n.type == 'panini_rule']
        for node in panini_nodes:
            if node.activation_level > 0.5:
                effects.append({
                    'type': 'rule_activation',
                    'target_id': node.id,
                    'position': node.position,
                    'properties': {
                        'type': 'mandala_rotation',
                        'color': '#ffd700',  # Gold for Pāṇini rules
                        'animation': 'sacred_geometry',
                        'intensity': node.activation_level
                    },
                    'duration': 3.0
                })
        
        return effects
    
    def _quantum_token_to_dict(self, token: QuantumToken) -> Dict[str, Any]:
        """Convert QuantumToken to dictionary for JSON serialization"""
        return {
            'text': token.text,
            'position': token.position,
            'morphology': token.morphology,
            'quantum_properties': token.quantum_properties,
            'visualization_data': token.visualization_data
        }
    
    def _network_node_to_dict(self, node: NetworkNode) -> Dict[str, Any]:
        """Convert NetworkNode to dictionary for JSON serialization"""
        return {
            'id': node.id,
            'position': node.position,
            'type': node.type,
            'rule_data': node.rule_data,
            'activation_level': node.activation_level,
            'connections': node.connections,
            'quantum_properties': node.quantum_properties
        }
    
    def _token_to_dict(self, token: QuantumToken) -> Dict[str, Any]:
        """Convert token to dictionary (alias for compatibility)"""
        return self._quantum_token_to_dict(token)
    
    def stop_stream(self, processing_id: str) -> bool:
        """Stop a streaming processing operation"""
        if processing_id in self.active_streams:
            self.active_streams[processing_id] = False
            return True
        return False
    
    def get_active_streams(self) -> List[str]:
        """Get list of active streaming processing IDs"""
        return [pid for pid, active in self.active_streams.items() if active]
    
    async def process_text_simple(self, text: str) -> Dict[str, Any]:
        """Simple synchronous text processing for basic use cases"""
        try:
            # Collect all streaming updates
            updates = []
            async for update in self.process_text_streaming(text, enable_visualization=True):
                updates.append(update)
                if update.update_type == 'processing_complete':
                    break
            
            # Extract final result
            final_update = updates[-1] if updates else None
            visualization_data = None
            
            # Find the visualization data from updates
            for update in reversed(updates):
                if update.visualization_update:
                    visualization_data = update.visualization_update
                    break
            
            return {
                'success': final_update.data.get('success', False) if final_update else False,
                'processing_updates': len(updates),
                'visualization_data': visualization_data,
                'error': final_update.data.get('error') if final_update and 'error' in final_update.data else None
            }
        
        except Exception as e:
            logger.error(f"Error in simple text processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'visualization_data': None
            }


# Error handling and graceful degradation
class SanskritAdapterError(Exception):
    """Base exception for Sanskrit adapter errors"""
    pass


class ProcessingError(SanskritAdapterError):
    """Error during text processing"""
    pass


class VisualizationError(SanskritAdapterError):
    """Error during visualization generation"""
    pass


# Utility functions for error handling
def handle_processing_error(func):
    """Decorator for graceful error handling in processing functions"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            # Return fallback result
            return {
                'success': False,
                'error': str(e),
                'fallback_mode': True
            }
    return wrapper


# Export main classes and functions
__all__ = [
    'SanskritEngineAdapter',
    'QuantumToken',
    'NetworkNode',
    'VisualizationData',
    'ProcessingUpdate',
    'SanskritAdapterError',
    'ProcessingError',
    'VisualizationError'
]