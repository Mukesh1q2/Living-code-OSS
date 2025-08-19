"""
Sanskrit-LLM Response Synthesis System for Vidya Quantum Interface

This module implements the core response synthesis system that combines Sanskrit
grammatical analysis with LLM-generated content to create enhanced, contextually
aware responses that integrate ancient wisdom with modern AI capabilities.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator, Tuple, Union
from datetime import datetime
from enum import Enum
import uuid

# Import existing components
from .sanskrit_adapter import SanskritEngineAdapter, ProcessingUpdate, QuantumToken
from .llm_integration import LLMIntegrationService, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


class SynthesisQuality(Enum):
    """Quality levels for response synthesis"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PREMIUM = "premium"
    QUANTUM = "quantum"


class ResponseType(Enum):
    """Types of synthesized responses"""
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    GUIDANCE = "guidance"
    WISDOM = "wisdom"
    CONVERSATION = "conversation"
    ERROR_CORRECTION = "error_correction"


@dataclass
class SynthesisContext:
    """Context information for response synthesis"""
    user_query: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    consciousness_level: int = 1
    quantum_coherence: float = 0.5
    sanskrit_expertise: float = 0.5
    preferred_language: str = "english"
    include_sanskrit: bool = True
    include_transliteration: bool = True
    response_style: str = "balanced"  # formal, casual, balanced, scholarly


@dataclass
class SanskritAnalysisResult:
    """Results from Sanskrit grammatical analysis"""
    tokens: List[QuantumToken]
    morphological_analysis: Dict[str, Any]
    grammatical_rules: List[Dict[str, Any]]
    etymological_connections: List[Dict[str, Any]]
    semantic_patterns: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    visualization_data: Dict[str, Any]


@dataclass
class LLMGenerationResult:
    """Results from LLM text generation"""
    generated_text: str
    model_used: str
    confidence_score: float
    semantic_embeddings: Optional[List[float]]
    processing_time: float
    metadata: Dict[str, Any]
    fallback_used: bool = False


@dataclass
class SynthesizedResponse:
    """Complete synthesized response combining Sanskrit and LLM analysis"""
    response_id: str
    response_text: str
    response_type: ResponseType
    quality_level: SynthesisQuality
    
    # Component results
    sanskrit_analysis: Optional[SanskritAnalysisResult]
    llm_generation: Optional[LLMGenerationResult]
    
    # Synthesis metadata
    synthesis_confidence: float
    processing_time: float
    timestamp: datetime
    
    # Enhancement data
    sanskrit_wisdom: Optional[str] = None
    transliteration: Optional[str] = None
    etymological_insights: List[str] = field(default_factory=list)
    grammatical_explanations: List[str] = field(default_factory=list)
    
    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    engagement_score: float = 0.0
    
    # Visualization and interaction data
    quantum_effects: List[Dict[str, Any]] = field(default_factory=list)
    neural_network_updates: List[Dict[str, Any]] = field(default_factory=list)
    consciousness_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback for response improvement"""
    response_id: str
    feedback_type: str  # rating, correction, preference, report
    rating: Optional[int] = None  # 1-5 scale
    correction: Optional[str] = None
    preference_updates: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""


class ResponseSynthesizer:
    """
    Core response synthesis system that combines Sanskrit analysis with LLM outputs
    to create enhanced, contextually aware responses.
    """
    
    def __init__(self, 
                 sanskrit_adapter: SanskritEngineAdapter,
                 llm_service: LLMIntegrationService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the response synthesizer.
        
        Args:
            sanskrit_adapter: Sanskrit engine adapter for grammatical analysis
            llm_service: LLM integration service for text generation
            config: Optional configuration dictionary
        """
        self.sanskrit_adapter = sanskrit_adapter
        self.llm_service = llm_service
        self.config = config or {}
        
        # Synthesis configuration
        self.default_quality = SynthesisQuality(self.config.get('default_quality', 'enhanced'))
        self.enable_streaming = self.config.get('enable_streaming', True)
        self.max_processing_time = self.config.get('max_processing_time', 30.0)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Response caching
        self.response_cache: Dict[str, SynthesizedResponse] = {}
        self.feedback_history: List[UserFeedback] = []
        
        # Quality assessment weights
        self.quality_weights = {
            'coherence': 0.3,
            'relevance': 0.3,
            'accuracy': 0.2,
            'engagement': 0.2
        }
        
        # Sanskrit wisdom templates
        self.wisdom_templates = self._load_wisdom_templates()
        
        logger.info("Response Synthesizer initialized successfully")
    
    def _load_wisdom_templates(self) -> Dict[str, List[str]]:
        """Load Sanskrit wisdom templates for different contexts"""
        return {
            'greeting': [
                'नमस्ते - I bow to the divine in you',
                'स्वागतम् - Welcome to this space of learning',
                'सत्यमेव जयते - Truth alone triumphs'
            ],
            'learning': [
                'विद्या ददाति विनयम् - Knowledge gives humility',
                'अहिंसा परमो धर्मः - Non-violence is the highest virtue',
                'यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः - Where women are honored, divinity blossoms'
            ],
            'wisdom': [
                'तत्त्वमसि - Thou art That',
                'अहं ब्रह्मास्मि - I am Brahman',
                'सर्वं खल्विदं ब्रह्म - All this is indeed Brahman'
            ],
            'guidance': [
                'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन - You have the right to action, never to the fruits',
                'योगः कर्मसु कौशलम् - Yoga is skill in action',
                'मन एव मनुष्याणां कारणं बन्धमोक्षयोः - Mind alone is the cause of bondage and liberation'
            ]
        }
    
    async def synthesize_response(self, 
                                context: SynthesisContext,
                                quality_level: Optional[SynthesisQuality] = None) -> SynthesizedResponse:
        """
        Synthesize a complete response combining Sanskrit analysis and LLM generation.
        
        Args:
            context: Synthesis context with user query and preferences
            quality_level: Desired quality level for synthesis
            
        Returns:
            Complete synthesized response
        """
        start_time = time.time()
        response_id = f"response_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        quality_level = quality_level or self.default_quality
        
        try:
            logger.info(f"Starting response synthesis for query: {context.user_query[:50]}...")
            
            # Check cache first
            if self.enable_caching:
                cached_response = self._check_cache(context.user_query, quality_level)
                if cached_response:
                    logger.info(f"Returning cached response for query")
                    return cached_response
            
            # Determine response type
            response_type = self._classify_response_type(context.user_query)
            
            # Parallel processing of Sanskrit analysis and LLM generation
            sanskrit_task = asyncio.create_task(
                self._perform_sanskrit_analysis(context.user_query, context)
            )
            llm_task = asyncio.create_task(
                self._perform_llm_generation(context, response_type)
            )
            
            # Wait for both analyses to complete
            sanskrit_result, llm_result = await asyncio.gather(
                sanskrit_task, llm_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(sanskrit_result, Exception):
                logger.error(f"Sanskrit analysis failed: {sanskrit_result}")
                sanskrit_result = None
            
            if isinstance(llm_result, Exception):
                logger.error(f"LLM generation failed: {llm_result}")
                llm_result = None
            
            # Synthesize the final response
            synthesized_response = await self._synthesize_components(
                response_id=response_id,
                context=context,
                sanskrit_result=sanskrit_result,
                llm_result=llm_result,
                response_type=response_type,
                quality_level=quality_level,
                start_time=start_time
            )
            
            # Cache the response
            if self.enable_caching:
                self._cache_response(context.user_query, quality_level, synthesized_response)
            
            logger.info(f"Response synthesis completed in {synthesized_response.processing_time:.2f}s")
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Return fallback response
            return await self._create_fallback_response(
                response_id, context, response_type, start_time
            )
    
    async def synthesize_response_streaming(self, 
                                          context: SynthesisContext,
                                          quality_level: Optional[SynthesisQuality] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response synthesis updates in real-time.
        
        Args:
            context: Synthesis context with user query and preferences
            quality_level: Desired quality level for synthesis
            
        Yields:
            Real-time synthesis updates
        """
        if not self.enable_streaming:
            # Fall back to batch processing
            response = await self.synthesize_response(context, quality_level)
            yield {
                'type': 'synthesis_complete',
                'response': response,
                'progress': 1.0
            }
            return
        
        response_id = f"stream_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        quality_level = quality_level or self.default_quality
        
        try:
            # Initial update
            yield {
                'type': 'synthesis_started',
                'response_id': response_id,
                'query': context.user_query,
                'progress': 0.0
            }
            
            # Determine response type
            response_type = self._classify_response_type(context.user_query)
            
            yield {
                'type': 'response_type_determined',
                'response_type': response_type.value,
                'progress': 0.1
            }
            
            # Start Sanskrit analysis streaming
            sanskrit_updates = []
            async for update in self.sanskrit_adapter.process_text_streaming(
                context.user_query, enable_visualization=True
            ):
                sanskrit_updates.append(update)
                yield {
                    'type': 'sanskrit_analysis_update',
                    'update': update,
                    'progress': 0.1 + (update.progress * 0.4)
                }
            
            # Perform LLM generation
            yield {
                'type': 'llm_generation_started',
                'progress': 0.5
            }
            
            llm_result = await self._perform_llm_generation(context, response_type)
            
            yield {
                'type': 'llm_generation_complete',
                'model_used': llm_result.model_used if llm_result else 'fallback',
                'progress': 0.7
            }
            
            # Synthesize components
            yield {
                'type': 'synthesis_processing',
                'progress': 0.8
            }
            
            # Create Sanskrit analysis result from updates
            sanskrit_result = self._compile_sanskrit_analysis(sanskrit_updates) if sanskrit_updates else None
            
            # Final synthesis
            synthesized_response = await self._synthesize_components(
                response_id=response_id,
                context=context,
                sanskrit_result=sanskrit_result,
                llm_result=llm_result,
                response_type=response_type,
                quality_level=quality_level,
                start_time=time.time()
            )
            
            # Final update
            yield {
                'type': 'synthesis_complete',
                'response': synthesized_response,
                'progress': 1.0
            }
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            yield {
                'type': 'synthesis_error',
                'error': str(e),
                'progress': 0.0
            }
    
    async def _perform_sanskrit_analysis(self, 
                                       query: str, 
                                       context: SynthesisContext) -> Optional[SanskritAnalysisResult]:
        """Perform Sanskrit grammatical analysis"""
        try:
            start_time = time.time()
            
            # Process through Sanskrit adapter
            updates = []
            async for update in self.sanskrit_adapter.process_text_streaming(
                query, enable_visualization=True
            ):
                updates.append(update)
            
            # Compile results
            result = self._compile_sanskrit_analysis(updates)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Sanskrit analysis failed: {e}")
            return None
    
    def _compile_sanskrit_analysis(self, updates: List[ProcessingUpdate]) -> SanskritAnalysisResult:
        """Compile Sanskrit analysis from processing updates"""
        tokens = []
        rules = []
        visualization_data = {}
        confidence_scores = []
        
        for update in updates:
            if update.update_type == 'tokenization_complete':
                if update.visualization_update and 'tokens' in update.visualization_update:
                    # Convert token dictionaries back to QuantumToken objects
                    token_dicts = update.visualization_update['tokens']
                    for token_dict in token_dicts:
                        # Create QuantumToken from dictionary
                        token = QuantumToken(
                            text=token_dict.get('text', ''),
                            position=token_dict.get('position', {}),
                            morphology=token_dict.get('morphology', {}),
                            quantum_properties=token_dict.get('quantum_properties', {}),
                            visualization_data=token_dict.get('visualization_data', {})
                        )
                        tokens.append(token)
            
            elif update.update_type == 'rule_applied':
                rules.append({
                    'rule_name': update.data.get('rule_name', 'unknown'),
                    'step': update.data.get('step', 'unknown'),
                    'before': update.data.get('text_before', ''),
                    'after': update.data.get('text_after', ''),
                    'iteration': update.data.get('iteration', 0)
                })
            
            elif update.update_type == 'visualization_complete':
                if update.visualization_update:
                    visualization_data.update(update.visualization_update)
            
            # Track confidence based on progress
            if update.progress > 0:
                confidence_scores.append(update.progress)
        
        # Calculate overall confidence
        confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Extract morphological analysis
        morphological_analysis = {}
        for token in tokens:
            if token.morphology:
                morphological_analysis[token.text] = token.morphology
        
        # Extract etymological connections (simplified)
        etymological_connections = []
        for token in tokens:
            if token.morphology.get('root'):
                etymological_connections.append({
                    'word': token.text,
                    'root': token.morphology['root'],
                    'meaning': f"Derived from root '{token.morphology['root']}'"
                })
        
        # Extract semantic patterns
        semantic_patterns = []
        grammatical_categories = [token.morphology.get('grammatical_category', 'unknown') for token in tokens]
        category_counts = {}
        for category in grammatical_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            if count > 1:
                semantic_patterns.append({
                    'pattern': f"Multiple {category}s",
                    'count': count,
                    'significance': 'Indicates complex grammatical structure'
                })
        
        return SanskritAnalysisResult(
            tokens=tokens,
            morphological_analysis=morphological_analysis,
            grammatical_rules=rules,
            etymological_connections=etymological_connections,
            semantic_patterns=semantic_patterns,
            confidence_score=confidence_score,
            processing_time=0.0,  # Will be set by caller
            visualization_data=visualization_data
        )
    
    async def _perform_llm_generation(self, 
                                    context: SynthesisContext, 
                                    response_type: ResponseType) -> Optional[LLMGenerationResult]:
        """Perform LLM text generation"""
        try:
            start_time = time.time()
            
            # Prepare LLM request based on context and response type
            prompt = self._create_llm_prompt(context, response_type)
            
            # Select appropriate model based on context
            model_name = self._select_llm_model(context, response_type)
            
            # Create inference request
            inference_request = InferenceRequest(
                text=prompt,
                model_name=model_name,
                max_length=self._get_max_length(response_type),
                temperature=self._get_temperature(context, response_type),
                return_full_text=False
            )
            
            # Generate text
            inference_response = await self.llm_service.generate_text(inference_request)
            
            # Generate embeddings for semantic analysis
            embedding_response = await self.llm_service.generate_embeddings(
                context.user_query, "embeddings"
            )
            
            processing_time = time.time() - start_time
            
            return LLMGenerationResult(
                generated_text=inference_response.text or "",
                model_used=inference_response.model_used,
                confidence_score=0.8 if inference_response.success else 0.3,
                semantic_embeddings=embedding_response.embeddings if embedding_response.success else None,
                processing_time=processing_time,
                metadata=inference_response.metadata or {},
                fallback_used=not inference_response.success
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
    
    def _create_llm_prompt(self, context: SynthesisContext, response_type: ResponseType) -> str:
        """Create appropriate LLM prompt based on context and response type"""
        base_prompt = f"User query: {context.user_query}\n\n"
        
        # Add context from conversation history
        if context.conversation_history:
            base_prompt += "Previous conversation:\n"
            for entry in context.conversation_history[-3:]:  # Last 3 entries
                base_prompt += f"User: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}\n"
            base_prompt += "\n"
        
        # Add response type specific instructions
        if response_type == ResponseType.EXPLANATION:
            base_prompt += "Provide a clear, educational explanation that helps the user understand the concept. "
        elif response_type == ResponseType.ANALYSIS:
            base_prompt += "Analyze the query in detail, breaking down the components and their relationships. "
        elif response_type == ResponseType.GUIDANCE:
            base_prompt += "Offer helpful guidance and practical advice related to the query. "
        elif response_type == ResponseType.WISDOM:
            base_prompt += "Share relevant wisdom and deeper insights that illuminate the topic. "
        elif response_type == ResponseType.CONVERSATION:
            base_prompt += "Engage in natural, helpful conversation while addressing the query. "
        elif response_type == ResponseType.ERROR_CORRECTION:
            base_prompt += "Gently correct any misconceptions while providing accurate information. "
        
        # Add style preferences
        if context.response_style == "formal":
            base_prompt += "Use a formal, academic tone. "
        elif context.response_style == "casual":
            base_prompt += "Use a friendly, conversational tone. "
        elif context.response_style == "scholarly":
            base_prompt += "Use a scholarly tone with appropriate references and depth. "
        else:
            base_prompt += "Use a balanced, approachable tone. "
        
        # Add Sanskrit integration note if requested
        if context.include_sanskrit:
            base_prompt += "When relevant, incorporate Sanskrit concepts and terminology with explanations. "
        
        base_prompt += "\nResponse:"
        
        return base_prompt
    
    def _select_llm_model(self, context: SynthesisContext, response_type: ResponseType) -> str:
        """Select appropriate LLM model based on context and response type"""
        # Use more sophisticated model for complex queries
        if (context.consciousness_level >= 3 or 
            response_type in [ResponseType.ANALYSIS, ResponseType.WISDOM] or
            len(context.user_query) > 100):
            return "sanskrit-aware"
        
        return "default"
    
    def _get_max_length(self, response_type: ResponseType) -> int:
        """Get appropriate max length for response type"""
        length_map = {
            ResponseType.EXPLANATION: 512,
            ResponseType.ANALYSIS: 768,
            ResponseType.GUIDANCE: 384,
            ResponseType.WISDOM: 256,
            ResponseType.CONVERSATION: 256,
            ResponseType.ERROR_CORRECTION: 384
        }
        return length_map.get(response_type, 384)
    
    def _get_temperature(self, context: SynthesisContext, response_type: ResponseType) -> float:
        """Get appropriate temperature for response generation"""
        # More creative for wisdom and conversation
        if response_type in [ResponseType.WISDOM, ResponseType.CONVERSATION]:
            return 0.8
        # More precise for analysis and corrections
        elif response_type in [ResponseType.ANALYSIS, ResponseType.ERROR_CORRECTION]:
            return 0.5
        # Balanced for others
        else:
            return 0.7
    
    def _classify_response_type(self, query: str) -> ResponseType:
        """Classify the type of response needed based on the query"""
        query_lower = query.lower()
        
        # Check for question patterns
        if query.endswith('?') or any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            if any(word in query_lower for word in ['explain', 'understand', 'mean', 'definition']):
                return ResponseType.EXPLANATION
            elif any(word in query_lower for word in ['analyze', 'analysis', 'break down', 'examine']):
                return ResponseType.ANALYSIS
            else:
                return ResponseType.EXPLANATION
        
        # Check for guidance requests
        if any(word in query_lower for word in ['help', 'guide', 'advice', 'suggest', 'recommend', 'should']):
            return ResponseType.GUIDANCE
        
        # Check for wisdom/philosophical queries
        if any(word in query_lower for word in ['wisdom', 'philosophy', 'meaning of life', 'purpose', 'truth']):
            return ResponseType.WISDOM
        
        # Check for corrections
        if any(word in query_lower for word in ['wrong', 'incorrect', 'mistake', 'error', 'fix']):
            return ResponseType.ERROR_CORRECTION
        
        # Default to conversation
        return ResponseType.CONVERSATION 
   
    async def _synthesize_components(self,
                                   response_id: str,
                                   context: SynthesisContext,
                                   sanskrit_result: Optional[SanskritAnalysisResult],
                                   llm_result: Optional[LLMGenerationResult],
                                   response_type: ResponseType,
                                   quality_level: SynthesisQuality,
                                   start_time: float) -> SynthesizedResponse:
        """Synthesize Sanskrit analysis and LLM generation into final response"""
        
        # Base response text
        response_text = ""
        
        # Start with LLM generated text if available
        if llm_result and llm_result.generated_text:
            response_text = llm_result.generated_text.strip()
        else:
            # Fallback to template-based response
            response_text = self._generate_fallback_response(context.user_query, response_type)
        
        # Enhance with Sanskrit analysis if available
        if sanskrit_result and context.include_sanskrit:
            response_text = await self._enhance_with_sanskrit(
                response_text, sanskrit_result, context, quality_level
            )
        
        # Add Sanskrit wisdom if appropriate
        sanskrit_wisdom = None
        if context.include_sanskrit and quality_level in [SynthesisQuality.ENHANCED, SynthesisQuality.PREMIUM, SynthesisQuality.QUANTUM]:
            sanskrit_wisdom = self._select_sanskrit_wisdom(context.user_query, response_type)
        
        # Generate transliteration if requested
        transliteration = None
        if context.include_transliteration and sanskrit_wisdom:
            transliteration = self._generate_transliteration(sanskrit_wisdom)
        
        # Calculate quality metrics
        quality_metrics = await self._assess_response_quality(
            response_text, context, sanskrit_result, llm_result
        )
        
        # Generate quantum effects and neural network updates
        quantum_effects = self._generate_quantum_effects(sanskrit_result, llm_result, quality_level)
        neural_network_updates = self._generate_neural_network_updates(sanskrit_result, context)
        consciousness_updates = self._generate_consciousness_updates(context, quality_metrics)
        
        # Extract etymological insights
        etymological_insights = []
        if sanskrit_result:
            etymological_insights = [
                conn['meaning'] for conn in sanskrit_result.etymological_connections
            ]
        
        # Extract grammatical explanations
        grammatical_explanations = []
        if sanskrit_result:
            grammatical_explanations = [
                f"Applied {rule['rule_name']}: {rule['before']} → {rule['after']}"
                for rule in sanskrit_result.grammatical_rules
            ]
        
        # Calculate synthesis confidence
        synthesis_confidence = self._calculate_synthesis_confidence(
            sanskrit_result, llm_result, quality_metrics
        )
        
        processing_time = time.time() - start_time
        
        return SynthesizedResponse(
            response_id=response_id,
            response_text=response_text,
            response_type=response_type,
            quality_level=quality_level,
            sanskrit_analysis=sanskrit_result,
            llm_generation=llm_result,
            synthesis_confidence=synthesis_confidence,
            processing_time=processing_time,
            timestamp=datetime.now(),
            sanskrit_wisdom=sanskrit_wisdom,
            transliteration=transliteration,
            etymological_insights=etymological_insights,
            grammatical_explanations=grammatical_explanations,
            coherence_score=quality_metrics['coherence'],
            relevance_score=quality_metrics['relevance'],
            accuracy_score=quality_metrics['accuracy'],
            engagement_score=quality_metrics['engagement'],
            quantum_effects=quantum_effects,
            neural_network_updates=neural_network_updates,
            consciousness_updates=consciousness_updates
        )
    
    async def _enhance_with_sanskrit(self,
                                   base_text: str,
                                   sanskrit_result: SanskritAnalysisResult,
                                   context: SynthesisContext,
                                   quality_level: SynthesisQuality) -> str:
        """Enhance response text with Sanskrit analysis insights"""
        
        enhanced_text = base_text
        
        # Add morphological insights for premium quality
        if quality_level in [SynthesisQuality.PREMIUM, SynthesisQuality.QUANTUM]:
            if sanskrit_result.morphological_analysis:
                morphology_insights = []
                for word, analysis in sanskrit_result.morphological_analysis.items():
                    if analysis.get('root') and analysis['root'] != word:
                        morphology_insights.append(
                            f"'{word}' derives from root '{analysis['root']}'"
                        )
                
                if morphology_insights:
                    enhanced_text += f"\n\n**Morphological Insights:** {'; '.join(morphology_insights[:3])}"
        
        # Add etymological connections
        if sanskrit_result.etymological_connections and quality_level != SynthesisQuality.BASIC:
            etymology_text = []
            for conn in sanskrit_result.etymological_connections[:2]:  # Limit to 2 for readability
                etymology_text.append(f"{conn['word']} ({conn['meaning']})")
            
            if etymology_text:
                enhanced_text += f"\n\n**Etymology:** {'; '.join(etymology_text)}"
        
        # Add semantic patterns for quantum quality
        if quality_level == SynthesisQuality.QUANTUM and sanskrit_result.semantic_patterns:
            pattern_insights = []
            for pattern in sanskrit_result.semantic_patterns[:2]:
                pattern_insights.append(f"{pattern['pattern']}: {pattern['significance']}")
            
            if pattern_insights:
                enhanced_text += f"\n\n**Semantic Patterns:** {'; '.join(pattern_insights)}"
        
        return enhanced_text
    
    def _generate_fallback_response(self, query: str, response_type: ResponseType) -> str:
        """Generate fallback response when LLM is unavailable"""
        
        fallback_templates = {
            ResponseType.EXPLANATION: f"I understand you're asking about '{query[:50]}...'. Let me provide some insights based on the available information.",
            ResponseType.ANALYSIS: f"Analyzing your query '{query[:50]}...', I can identify several key components to examine.",
            ResponseType.GUIDANCE: f"Regarding your question about '{query[:50]}...', here are some helpful considerations.",
            ResponseType.WISDOM: f"Your inquiry about '{query[:50]}...' touches on profound concepts worth exploring.",
            ResponseType.CONVERSATION: f"Thank you for sharing '{query[:50]}...'. I'd be happy to discuss this with you.",
            ResponseType.ERROR_CORRECTION: f"I notice some aspects of '{query[:50]}...' that could benefit from clarification."
        }
        
        return fallback_templates.get(response_type, f"I understand your query about '{query[:50]}...' and will do my best to help.")
    
    def _select_sanskrit_wisdom(self, query: str, response_type: ResponseType) -> Optional[str]:
        """Select appropriate Sanskrit wisdom based on query and response type"""
        
        query_lower = query.lower()
        
        # Map response types to wisdom categories
        wisdom_mapping = {
            ResponseType.EXPLANATION: 'learning',
            ResponseType.ANALYSIS: 'learning',
            ResponseType.GUIDANCE: 'guidance',
            ResponseType.WISDOM: 'wisdom',
            ResponseType.CONVERSATION: 'greeting',
            ResponseType.ERROR_CORRECTION: 'guidance'
        }
        
        # Check for specific keywords
        if any(word in query_lower for word in ['hello', 'hi', 'namaste', 'greet']):
            category = 'greeting'
        elif any(word in query_lower for word in ['learn', 'teach', 'study', 'knowledge']):
            category = 'learning'
        elif any(word in query_lower for word in ['wisdom', 'truth', 'reality', 'consciousness']):
            category = 'wisdom'
        elif any(word in query_lower for word in ['help', 'guide', 'advice', 'should']):
            category = 'guidance'
        else:
            category = wisdom_mapping.get(response_type, 'learning')
        
        # Select wisdom from category
        wisdom_options = self.wisdom_templates.get(category, self.wisdom_templates['learning'])
        
        # Simple selection based on query length (could be more sophisticated)
        index = len(query) % len(wisdom_options)
        return wisdom_options[index]
    
    def _generate_transliteration(self, sanskrit_text: str) -> str:
        """Generate transliteration for Sanskrit text"""
        # Simple transliteration mapping (in production, use proper transliteration library)
        transliteration_map = {
            'नमस्ते': 'namaste',
            'स्वागतम्': 'svāgatam',
            'सत्यमेव जयते': 'satyameva jayate',
            'विद्या ददाति विनयम्': 'vidyā dadāti vinayam',
            'अहिंसा परमो धर्मः': 'ahiṃsā paramo dharmaḥ',
            'तत्त्वमसि': 'tattvamasi',
            'अहं ब्रह्मास्मि': 'ahaṃ brahmāsmi',
            'सर्वं खल्विदं ब्रह्म': 'sarvaṃ khalvidaṃ brahma',
            'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन': 'karmaṇyevādhikāraste mā phaleṣu kadācana',
            'योगः कर्मसु कौशलम्': 'yogaḥ karmasu kauśalam',
            'मन एव मनुष्याणां कारणं बन्धमोक्षयोः': 'mana eva manuṣyāṇāṃ kāraṇaṃ bandhamokṣayoḥ'
        }
        
        return transliteration_map.get(sanskrit_text, sanskrit_text)
    
    async def _assess_response_quality(self,
                                     response_text: str,
                                     context: SynthesisContext,
                                     sanskrit_result: Optional[SanskritAnalysisResult],
                                     llm_result: Optional[LLMGenerationResult]) -> Dict[str, float]:
        """Assess the quality of the synthesized response"""
        
        # Coherence assessment
        coherence_score = self._assess_coherence(response_text, context)
        
        # Relevance assessment
        relevance_score = self._assess_relevance(response_text, context.user_query)
        
        # Accuracy assessment
        accuracy_score = self._assess_accuracy(response_text, sanskrit_result, llm_result)
        
        # Engagement assessment
        engagement_score = self._assess_engagement(response_text, context)
        
        return {
            'coherence': coherence_score,
            'relevance': relevance_score,
            'accuracy': accuracy_score,
            'engagement': engagement_score
        }
    
    def _assess_coherence(self, response_text: str, context: SynthesisContext) -> float:
        """Assess response coherence"""
        # Simple coherence metrics
        sentences = response_text.split('.')
        
        # Check for reasonable sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        length_score = min(1.0, avg_sentence_length / 20.0)  # Optimal around 20 words
        
        # Check for logical flow (simplified)
        flow_score = 0.8  # Default assumption of good flow
        
        # Check for completeness
        completeness_score = 1.0 if len(response_text) > 50 else len(response_text) / 50.0
        
        return (length_score + flow_score + completeness_score) / 3.0
    
    def _assess_relevance(self, response_text: str, query: str) -> float:
        """Assess response relevance to query"""
        # Simple keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response_text.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(response_words))
        relevance_score = overlap / len(query_words)
        
        return min(1.0, relevance_score)
    
    def _assess_accuracy(self,
                        response_text: str,
                        sanskrit_result: Optional[SanskritAnalysisResult],
                        llm_result: Optional[LLMGenerationResult]) -> float:
        """Assess response accuracy"""
        # Base accuracy on component confidence
        accuracy_scores = []
        
        if sanskrit_result:
            accuracy_scores.append(sanskrit_result.confidence_score)
        
        if llm_result:
            accuracy_scores.append(llm_result.confidence_score)
        
        if not accuracy_scores:
            return 0.5  # Default for fallback responses
        
        return sum(accuracy_scores) / len(accuracy_scores)
    
    def _assess_engagement(self, response_text: str, context: SynthesisContext) -> float:
        """Assess response engagement level"""
        # Check for engaging elements
        engagement_factors = []
        
        # Questions engage users
        if '?' in response_text:
            engagement_factors.append(0.2)
        
        # Sanskrit elements add engagement
        if any(char in response_text for char in 'अआइईउऊकखगघङचछजझञ'):
            engagement_factors.append(0.3)
        
        # Personal pronouns create connection
        personal_pronouns = ['you', 'your', 'we', 'our', 'I']
        if any(pronoun in response_text.lower() for pronoun in personal_pronouns):
            engagement_factors.append(0.2)
        
        # Appropriate length
        length_factor = min(0.3, len(response_text) / 500.0)
        engagement_factors.append(length_factor)
        
        return min(1.0, sum(engagement_factors))
    
    def _generate_quantum_effects(self,
                                sanskrit_result: Optional[SanskritAnalysisResult],
                                llm_result: Optional[LLMGenerationResult],
                                quality_level: SynthesisQuality) -> List[Dict[str, Any]]:
        """Generate quantum effects for visualization"""
        
        effects = []
        
        # Sanskrit-based effects
        if sanskrit_result:
            for token in sanskrit_result.tokens:
                if token.quantum_properties.get('superposition'):
                    effects.append({
                        'type': 'superposition',
                        'target': token.text,
                        'position': token.visualization_data.get('position', {}),
                        'duration': 2.0,
                        'intensity': 0.7
                    })
                
                if token.quantum_properties.get('entanglements'):
                    effects.append({
                        'type': 'entanglement',
                        'source': token.text,
                        'targets': token.quantum_properties['entanglements'],
                        'strength': 0.8,
                        'duration': 3.0
                    })
        
        # LLM-based effects
        if llm_result and not llm_result.fallback_used:
            effects.append({
                'type': 'consciousness_activation',
                'intensity': llm_result.confidence_score,
                'duration': 2.5,
                'color': '#4a90e2'
            })
        
        # Quality-based effects
        if quality_level == SynthesisQuality.QUANTUM:
            effects.append({
                'type': 'quantum_synthesis',
                'complexity': len(effects),
                'duration': 4.0,
                'color': '#b383ff'
            })
        
        return effects
    
    def _generate_neural_network_updates(self,
                                       sanskrit_result: Optional[SanskritAnalysisResult],
                                       context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate neural network visualization updates"""
        
        updates = []
        
        if sanskrit_result:
            # Add nodes for Sanskrit tokens
            for i, token in enumerate(sanskrit_result.tokens):
                updates.append({
                    'type': 'node_activation',
                    'node_id': f'token_{i}_{token.text}',
                    'position': token.visualization_data.get('position', {}),
                    'activation_level': 0.8,
                    'node_type': 'sanskrit_token',
                    'metadata': {
                        'text': token.text,
                        'morphology': token.morphology
                    }
                })
            
            # Add nodes for grammatical rules
            for i, rule in enumerate(sanskrit_result.grammatical_rules):
                updates.append({
                    'type': 'rule_node_activation',
                    'node_id': f'rule_{i}_{rule["rule_name"]}',
                    'rule_name': rule['rule_name'],
                    'activation_level': 0.9,
                    'node_type': 'panini_rule'
                })
        
        # Add consciousness level indicator
        updates.append({
            'type': 'consciousness_update',
            'level': context.consciousness_level,
            'coherence': context.quantum_coherence
        })
        
        return updates
    
    def _generate_consciousness_updates(self,
                                      context: SynthesisContext,
                                      quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate consciousness state updates"""
        
        # Calculate consciousness evolution based on interaction quality
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        evolution_amount = quality_score * 0.1  # Small evolution per interaction
        
        return {
            'evolution_trigger': 'response_synthesis',
            'evolution_amount': evolution_amount,
            'quality_score': quality_score,
            'interaction_complexity': len(context.user_query) / 100.0,
            'sanskrit_integration': context.include_sanskrit,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_synthesis_confidence(self,
                                      sanskrit_result: Optional[SanskritAnalysisResult],
                                      llm_result: Optional[LLMGenerationResult],
                                      quality_metrics: Dict[str, float]) -> float:
        """Calculate overall synthesis confidence"""
        
        confidence_factors = []
        
        # Sanskrit analysis confidence
        if sanskrit_result:
            confidence_factors.append(sanskrit_result.confidence_score * 0.4)
        
        # LLM generation confidence
        if llm_result:
            confidence_factors.append(llm_result.confidence_score * 0.4)
        
        # Quality metrics confidence
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        confidence_factors.append(quality_score * 0.2)
        
        if not confidence_factors:
            return 0.3  # Low confidence for fallback responses
        
        return sum(confidence_factors)
    
    async def _create_fallback_response(self,
                                      response_id: str,
                                      context: SynthesisContext,
                                      response_type: ResponseType,
                                      start_time: float) -> SynthesizedResponse:
        """Create fallback response when synthesis fails"""
        
        fallback_text = self._generate_fallback_response(context.user_query, response_type)
        processing_time = time.time() - start_time
        
        return SynthesizedResponse(
            response_id=response_id,
            response_text=fallback_text,
            response_type=response_type,
            quality_level=SynthesisQuality.BASIC,
            sanskrit_analysis=None,
            llm_generation=None,
            synthesis_confidence=0.3,
            processing_time=processing_time,
            timestamp=datetime.now(),
            coherence_score=0.5,
            relevance_score=0.4,
            accuracy_score=0.3,
            engagement_score=0.4
        )
    
    # Caching methods
    def _check_cache(self, query: str, quality_level: SynthesisQuality) -> Optional[SynthesizedResponse]:
        """Check if response is cached"""
        cache_key = f"{hash(query)}_{quality_level.value}"
        return self.response_cache.get(cache_key)
    
    def _cache_response(self, query: str, quality_level: SynthesisQuality, response: SynthesizedResponse):
        """Cache response for future use"""
        cache_key = f"{hash(query)}_{quality_level.value}"
        self.response_cache[cache_key] = response
        
        # Limit cache size
        if len(self.response_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k].timestamp)
            del self.response_cache[oldest_key]
    
    # User feedback integration
    async def process_user_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Process user feedback to improve future responses"""
        
        self.feedback_history.append(feedback)
        
        # Find the original response
        original_response = None
        for response in self.response_cache.values():
            if response.response_id == feedback.response_id:
                original_response = response
                break
        
        if not original_response:
            return {'success': False, 'message': 'Original response not found'}
        
        # Process different types of feedback
        improvements = {}
        
        if feedback.feedback_type == 'rating' and feedback.rating is not None:
            # Adjust quality weights based on rating
            if feedback.rating < 3:
                improvements['quality_adjustment'] = 'negative'
            elif feedback.rating > 4:
                improvements['quality_adjustment'] = 'positive'
        
        elif feedback.feedback_type == 'correction' and feedback.correction:
            # Store correction for future reference
            improvements['correction_noted'] = feedback.correction
        
        elif feedback.feedback_type == 'preference' and feedback.preference_updates:
            # Update user preferences
            improvements['preferences_updated'] = feedback.preference_updates
        
        # Log feedback for analysis
        logger.info(f"Processed feedback for response {feedback.response_id}: {feedback.feedback_type}")
        
        return {
            'success': True,
            'feedback_processed': feedback.feedback_type,
            'improvements': improvements,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about synthesis performance"""
        
        if not self.response_cache:
            return {'message': 'No synthesis data available'}
        
        responses = list(self.response_cache.values())
        
        # Calculate averages
        avg_processing_time = sum(r.processing_time for r in responses) / len(responses)
        avg_confidence = sum(r.synthesis_confidence for r in responses) / len(responses)
        avg_coherence = sum(r.coherence_score for r in responses) / len(responses)
        avg_relevance = sum(r.relevance_score for r in responses) / len(responses)
        avg_accuracy = sum(r.accuracy_score for r in responses) / len(responses)
        avg_engagement = sum(r.engagement_score for r in responses) / len(responses)
        
        # Count by quality level
        quality_counts = {}
        for response in responses:
            quality = response.quality_level.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        # Count by response type
        type_counts = {}
        for response in responses:
            resp_type = response.response_type.value
            type_counts[resp_type] = type_counts.get(resp_type, 0) + 1
        
        return {
            'total_responses': len(responses),
            'average_processing_time': avg_processing_time,
            'average_confidence': avg_confidence,
            'quality_metrics': {
                'coherence': avg_coherence,
                'relevance': avg_relevance,
                'accuracy': avg_accuracy,
                'engagement': avg_engagement
            },
            'quality_distribution': quality_counts,
            'response_type_distribution': type_counts,
            'feedback_count': len(self.feedback_history),
            'cache_size': len(self.response_cache)
        }