"""
Quality Assessment System for Sanskrit-LLM Response Synthesis

This module provides comprehensive quality assessment capabilities for synthesized
responses, including coherence analysis, relevance scoring, accuracy validation,
and engagement measurement.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for response assessment"""
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    ENGAGEMENT = "engagement"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    SANSKRIT_INTEGRATION = "sanskrit_integration"


@dataclass
class QualityMetric:
    """Individual quality metric with score and explanation"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    contributing_factors: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    """Complete quality assessment for a response"""
    overall_score: float
    overall_confidence: float
    metrics: Dict[QualityDimension, QualityMetric]
    assessment_time: float
    timestamp: datetime
    
    # Detailed analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Comparative analysis
    benchmark_comparison: Optional[Dict[str, float]] = None
    historical_trend: Optional[str] = None


class QualityAssessor:
    """
    Advanced quality assessment system for Sanskrit-LLM synthesized responses.
    
    Provides multi-dimensional quality analysis including linguistic coherence,
    semantic relevance, factual accuracy, user engagement, and Sanskrit integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality assessor."""
        self.config = config or {}
        
        # Assessment weights for different dimensions
        self.dimension_weights = {
            QualityDimension.COHERENCE: 0.20,
            QualityDimension.RELEVANCE: 0.25,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.ENGAGEMENT: 0.15,
            QualityDimension.COMPLETENESS: 0.10,
            QualityDimension.CLARITY: 0.05,
            QualityDimension.CULTURAL_SENSITIVITY: 0.03,
            QualityDimension.SANSKRIT_INTEGRATION: 0.02
        }
        
        # Historical assessments for trend analysis
        self.assessment_history: List[QualityAssessment] = []
        
        # Benchmark scores for comparison
        self.benchmarks = {
            'basic_response': 0.6,
            'enhanced_response': 0.75,
            'premium_response': 0.85,
            'quantum_response': 0.95
        }
        
        # Sanskrit-specific assessment patterns
        self.sanskrit_patterns = self._initialize_sanskrit_patterns()
        
        # Linguistic analysis tools
        self.linguistic_analyzers = self._initialize_linguistic_analyzers()
        
        logger.info("Quality Assessor initialized successfully")
    
    def _initialize_sanskrit_patterns(self) -> Dict[str, Any]:
        """Initialize Sanskrit-specific quality assessment patterns"""
        return {
            'devanagari_script': re.compile(r'[\u0900-\u097F]+'),
            'transliteration_patterns': [
                r'[aāiīuūṛṝḷḹeaioau]',  # Vowels
                r'[kkhgghṅcchj]',        # Consonants
                r'[ṃḥ]'                  # Anusvara, visarga
            ],
            'common_terms': [
                'dharma', 'karma', 'yoga', 'moksha', 'samsara',
                'atman', 'brahman', 'vedanta', 'upanishad', 'sutra'
            ],
            'grammatical_terms': [
                'sandhi', 'vibhakti', 'pratyaya', 'dhatu', 'guna',
                'vriddhi', 'samprasarana', 'upadha', 'it', 'agama'
            ]
        }
    
    def _initialize_linguistic_analyzers(self) -> Dict[str, Any]:
        """Initialize linguistic analysis tools"""
        return {
            'sentence_patterns': {
                'simple': re.compile(r'^[^.!?]*[.!?]$'),
                'compound': re.compile(r'[,;:]'),
                'complex': re.compile(r'\b(because|since|although|while|if|when|where)\b')
            },
            'readability_factors': {
                'avg_sentence_length': 15,  # Optimal words per sentence
                'avg_word_length': 5,       # Optimal characters per word
                'syllable_complexity': 2.5  # Average syllables per word
            },
            'engagement_indicators': [
                r'\?',                      # Questions
                r'\b(you|your)\b',         # Direct address
                r'\b(we|our|us)\b',        # Inclusive language
                r'\b(imagine|consider|think)\b',  # Thought prompts
                r'[!]',                    # Exclamations
            ]
        }
    
    async def assess_response_quality(self,
                                    response_text: str,
                                    user_query: str,
                                    context: Optional[Dict[str, Any]] = None,
                                    sanskrit_analysis: Optional[Any] = None,
                                    llm_result: Optional[Any] = None) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of a synthesized response.
        """
        start_time = time.time()
        context = context or {}
        
        try:
            logger.info(f"Starting quality assessment for response: {response_text[:50]}...")
            
            # Assess each quality dimension
            metrics = {}
            
            # Coherence assessment
            metrics[QualityDimension.COHERENCE] = await self._assess_coherence(
                response_text, context
            )
            
            # Relevance assessment
            metrics[QualityDimension.RELEVANCE] = await self._assess_relevance(
                response_text, user_query, context
            )
            
            # Accuracy assessment
            metrics[QualityDimension.ACCURACY] = await self._assess_accuracy(
                response_text, sanskrit_analysis, llm_result, context
            )
            
            # Engagement assessment
            metrics[QualityDimension.ENGAGEMENT] = await self._assess_engagement(
                response_text, user_query, context
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            overall_confidence = self._calculate_overall_confidence(metrics)
            
            # Generate analysis insights
            strengths, weaknesses, recommendations = self._generate_insights(metrics, response_text)
            
            assessment_time = time.time() - start_time
            
            assessment = QualityAssessment(
                overall_score=overall_score,
                overall_confidence=overall_confidence,
                metrics=metrics,
                assessment_time=assessment_time,
                timestamp=datetime.now(),
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations
            )
            
            # Store for historical analysis
            self.assessment_history.append(assessment)
            
            # Limit history size
            if len(self.assessment_history) > 100:
                self.assessment_history.pop(0)
            
            logger.info(f"Quality assessment completed in {assessment_time:.2f}s, score: {overall_score:.2f}")
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return fallback assessment
            return self._create_fallback_assessment(start_time)
    
    async def _assess_coherence(self, response_text: str, context: Dict[str, Any]) -> QualityMetric:
        """Assess response coherence and logical flow"""
        
        factors = []
        score_components = []
        
        # Sentence structure analysis
        sentences = self._split_sentences(response_text)
        
        if not sentences:
            return QualityMetric(
                dimension=QualityDimension.COHERENCE,
                score=0.0,
                confidence=1.0,
                explanation="No coherent sentences found",
                contributing_factors=["Empty or malformed response"]
            )
        
        # Average sentence length (optimal around 15-20 words)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        length_score = max(0.0, min(1.0, length_score))
        score_components.append(length_score * 0.5)
        factors.append(f"Average sentence length: {avg_sentence_length:.1f} words")
        
        # Logical flow between sentences
        flow_score = self._assess_logical_flow(sentences)
        score_components.append(flow_score * 0.5)
        factors.append(f"Logical flow score: {flow_score:.2f}")
        
        overall_score = sum(score_components)
        confidence = 0.8  # High confidence in structural analysis
        
        # Generate improvement suggestions
        suggestions = []
        if avg_sentence_length > 25:
            suggestions.append("Consider breaking down longer sentences for better readability")
        elif avg_sentence_length < 10:
            suggestions.append("Consider combining short sentences for better flow")
        
        if flow_score < 0.7:
            suggestions.append("Improve logical connections between sentences")
        
        return QualityMetric(
            dimension=QualityDimension.COHERENCE,
            score=overall_score,
            confidence=confidence,
            explanation=f"Coherence assessment based on sentence structure and logical flow",
            contributing_factors=factors,
            improvement_suggestions=suggestions
        )
    
    async def _assess_relevance(self, response_text: str, user_query: str, context: Dict[str, Any]) -> QualityMetric:
        """Assess response relevance to user query"""
        
        factors = []
        score_components = []
        
        # Keyword overlap analysis
        query_keywords = self._extract_keywords(user_query.lower())
        response_keywords = self._extract_keywords(response_text.lower())
        
        if not query_keywords:
            keyword_overlap = 0.5  # Neutral score for empty query
        else:
            overlap_count = len(set(query_keywords) & set(response_keywords))
            keyword_overlap = overlap_count / len(query_keywords)
        
        score_components.append(keyword_overlap * 0.6)
        factors.append(f"Keyword overlap: {keyword_overlap:.2f} ({overlap_count}/{len(query_keywords)})")
        
        # Query type matching
        query_type_match = self._assess_query_type_match(user_query, response_text)
        score_components.append(query_type_match * 0.4)
        factors.append(f"Query type match: {query_type_match:.2f}")
        
        overall_score = sum(score_components)
        confidence = 0.7  # Moderate confidence in relevance assessment
        
        # Generate improvement suggestions
        suggestions = []
        if keyword_overlap < 0.3:
            suggestions.append("Include more keywords from the user's query")
        
        if query_type_match < 0.6:
            suggestions.append("Match response format to query type (question, explanation, etc.)")
        
        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=overall_score,
            confidence=confidence,
            explanation="Relevance assessment based on keyword overlap and query matching",
            contributing_factors=factors,
            improvement_suggestions=suggestions
        )
    
    async def _assess_accuracy(self, 
                             response_text: str, 
                             sanskrit_analysis: Optional[Any], 
                             llm_result: Optional[Any], 
                             context: Dict[str, Any]) -> QualityMetric:
        """Assess response accuracy and factual correctness"""
        
        factors = []
        score_components = []
        
        # Component accuracy (from Sanskrit and LLM analysis)
        if sanskrit_analysis and hasattr(sanskrit_analysis, 'confidence_score'):
            sanskrit_accuracy = sanskrit_analysis.confidence_score
            score_components.append(sanskrit_accuracy * 0.5)
            factors.append(f"Sanskrit analysis accuracy: {sanskrit_accuracy:.2f}")
        else:
            score_components.append(0.6 * 0.5)  # Default moderate accuracy
            factors.append("Sanskrit analysis not available")
        
        if llm_result and hasattr(llm_result, 'confidence_score'):
            llm_accuracy = llm_result.confidence_score
            score_components.append(llm_accuracy * 0.5)
            factors.append(f"LLM generation accuracy: {llm_accuracy:.2f}")
        else:
            score_components.append(0.5 * 0.5)  # Lower default for missing LLM
            factors.append("LLM result not available")
        
        overall_score = sum(score_components)
        confidence = 0.6  # Lower confidence due to limited fact-checking capabilities
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=overall_score,
            confidence=confidence,
            explanation="Accuracy assessment based on component confidence",
            contributing_factors=factors,
            improvement_suggestions=[]
        )
    
    async def _assess_engagement(self, response_text: str, user_query: str, context: Dict[str, Any]) -> QualityMetric:
        """Assess response engagement and user interaction potential"""
        
        factors = []
        score_components = []
        
        # Interactive elements
        question_count = len(re.findall(r'\?', response_text))
        question_score = min(1.0, question_count * 0.3)  # Up to 3-4 questions is good
        score_components.append(question_score * 0.3)
        factors.append(f"Questions asked: {question_count}")
        
        # Personal pronouns (creates connection)
        personal_pronouns = ['you', 'your', 'we', 'our', 'us']
        pronoun_count = sum(len(re.findall(rf'\b{pronoun}\b', response_text.lower())) 
                           for pronoun in personal_pronouns)
        pronoun_score = min(1.0, pronoun_count * 0.1)
        score_components.append(pronoun_score * 0.3)
        factors.append(f"Personal pronouns: {pronoun_count}")
        
        # Appropriate length (not too short, not too long)
        word_count = len(response_text.split())
        if 50 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 50:
            length_score = word_count / 50.0
        else:
            length_score = max(0.3, 1.0 - (word_count - 200) / 300.0)
        
        score_components.append(length_score * 0.4)
        factors.append(f"Response length: {word_count} words")
        
        overall_score = sum(score_components)
        confidence = 0.8  # High confidence in engagement metrics
        
        # Generate improvement suggestions
        suggestions = []
        if question_count == 0:
            suggestions.append("Consider adding questions to encourage user interaction")
        
        if pronoun_count < 2:
            suggestions.append("Use more personal pronouns to create connection")
        
        if word_count < 30:
            suggestions.append("Provide more detailed response for better engagement")
        elif word_count > 300:
            suggestions.append("Consider shortening response for better readability")
        
        return QualityMetric(
            dimension=QualityDimension.ENGAGEMENT,
            score=overall_score,
            confidence=confidence,
            explanation="Engagement assessment based on interactive elements and appropriate length",
            contributing_factors=factors,
            improvement_suggestions=suggestions
        )
    
    # Helper methods
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _assess_logical_flow(self, sentences: List[str]) -> float:
        """Assess logical flow between sentences"""
        if len(sentences) < 2:
            return 0.8  # Single sentence gets decent score
        
        # Simple flow assessment based on transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 
                           'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example']
        
        transition_count = 0
        for sentence in sentences[1:]:  # Skip first sentence
            if any(word in sentence.lower() for word in transition_words):
                transition_count += 1
        
        # Score based on appropriate use of transitions
        expected_transitions = max(1, len(sentences) // 3)
        transition_score = min(1.0, transition_count / expected_transitions)
        
        return transition_score * 0.7 + 0.3  # Base score of 0.3
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _assess_query_type_match(self, query: str, response: str) -> float:
        """Assess if response matches query type (question, explanation, etc.)"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Question queries should get explanatory responses
        if query.endswith('?') or any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            # Look for explanatory elements in response
            explanatory_words = ['because', 'since', 'therefore', 'thus', 'means', 'refers to', 'is defined as']
            if any(word in response_lower for word in explanatory_words):
                return 0.9
            else:
                return 0.6
        
        # Help/guidance queries should get actionable responses
        if any(word in query_lower for word in ['help', 'guide', 'how to', 'should i']):
            actionable_words = ['you can', 'try', 'consider', 'recommend', 'suggest', 'steps']
            if any(word in response_lower for word in actionable_words):
                return 0.9
            else:
                return 0.6
        
        # Default good match for other query types
        return 0.8
    
    def _calculate_overall_score(self, metrics: Dict[QualityDimension, QualityMetric]) -> float:
        """Calculate weighted overall quality score"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, metric in metrics.items():
            weight = self.dimension_weights.get(dimension, 0.1)
            total_score += metric.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_overall_confidence(self, metrics: Dict[QualityDimension, QualityMetric]) -> float:
        """Calculate overall confidence in the assessment"""
        confidences = [metric.confidence for metric in metrics.values()]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _generate_insights(self, 
                          metrics: Dict[QualityDimension, QualityMetric], 
                          response_text: str) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, weaknesses, and recommendations"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        for dimension, metric in metrics.items():
            if metric.score >= 0.8:
                strengths.append(f"Strong {dimension.value}: {metric.explanation}")
            elif metric.score <= 0.5:
                weaknesses.append(f"Weak {dimension.value}: {metric.explanation}")
            
            # Add improvement suggestions
            recommendations.extend(metric.improvement_suggestions)
        
        # Remove duplicates and limit length
        strengths = list(set(strengths))[:5]
        weaknesses = list(set(weaknesses))[:5]
        recommendations = list(set(recommendations))[:8]
        
        return strengths, weaknesses, recommendations
    
    def _create_fallback_assessment(self, start_time: float) -> QualityAssessment:
        """Create minimal assessment when full assessment fails"""
        fallback_metric = QualityMetric(
            dimension=QualityDimension.COHERENCE,
            score=0.5,
            confidence=0.3,
            explanation="Assessment failed, using fallback values"
        )
        
        return QualityAssessment(
            overall_score=0.5,
            overall_confidence=0.3,
            metrics={QualityDimension.COHERENCE: fallback_metric},
            assessment_time=time.time() - start_time,
            timestamp=datetime.now(),
            strengths=["Assessment completed"],
            weaknesses=["Full assessment unavailable"],
            recommendations=["Retry assessment with valid inputs"]
        )