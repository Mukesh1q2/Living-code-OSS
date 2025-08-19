"""
User Feedback Integration System for Sanskrit-LLM Response Synthesis

This module provides comprehensive user feedback collection, analysis, and integration
capabilities to continuously improve response synthesis quality through user input.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import statistics

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    RATING = "rating"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    REPORT = "report"
    SUGGESTION = "suggestion"
    APPRECIATION = "appreciation"


class FeedbackCategory(Enum):
    """Categories for feedback classification"""
    CONTENT_QUALITY = "content_quality"
    SANSKRIT_ACCURACY = "sanskrit_accuracy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    ENGAGEMENT = "engagement"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    TECHNICAL_ISSUE = "technical_issue"
    FEATURE_REQUEST = "feature_request"


@dataclass
class UserFeedback:
    """Individual user feedback entry"""
    feedback_id: str
    response_id: str
    session_id: str
    user_id: Optional[str]
    
    # Feedback content
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: Optional[int] = None  # 1-5 scale
    text_feedback: Optional[str] = None
    correction: Optional[str] = None
    
    # Context information
    original_query: str = ""
    original_response: str = ""
    context_tags: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    user_agent: Optional[str] = None
    
    # Processing status
    processed: bool = False
    applied: bool = False
    impact_score: float = 0.0


@dataclass
class FeedbackAnalysis:
    """Analysis results for feedback data"""
    total_feedback_count: int
    average_rating: float
    rating_distribution: Dict[int, int]
    category_breakdown: Dict[str, int]
    common_issues: List[str]
    improvement_suggestions: List[str]
    trend_analysis: Dict[str, Any]
    user_satisfaction_score: float


@dataclass
class ImprovementAction:
    """Action to be taken based on feedback"""
    action_id: str
    feedback_ids: List[str]
    action_type: str  # 'parameter_adjustment', 'template_update', 'model_retrain', etc.
    description: str
    priority: int  # 1-5, 5 being highest
    estimated_impact: float  # 0-1
    implementation_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)


class FeedbackIntegrator:
    """
    Comprehensive feedback integration system that collects, analyzes, and applies
    user feedback to improve Sanskrit-LLM response synthesis quality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feedback integrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Feedback storage
        self.feedback_history: List[UserFeedback] = []
        self.improvement_actions: List[ImprovementAction] = []
        
        # Analysis configuration
        self.min_feedback_for_analysis = self.config.get('min_feedback_for_analysis', 5)
        self.feedback_retention_days = self.config.get('feedback_retention_days', 90)
        self.auto_apply_threshold = self.config.get('auto_apply_threshold', 0.8)
        
        # Feedback categories and their weights
        self.category_weights = {
            FeedbackCategory.CONTENT_QUALITY: 0.25,
            FeedbackCategory.SANSKRIT_ACCURACY: 0.20,
            FeedbackCategory.RELEVANCE: 0.20,
            FeedbackCategory.CLARITY: 0.15,
            FeedbackCategory.ENGAGEMENT: 0.10,
            FeedbackCategory.CULTURAL_SENSITIVITY: 0.05,
            FeedbackCategory.TECHNICAL_ISSUE: 0.03,
            FeedbackCategory.FEATURE_REQUEST: 0.02
        }
        
        # Improvement patterns
        self.improvement_patterns = self._initialize_improvement_patterns()
        
        # User preference tracking
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Feedback Integrator initialized successfully")
    
    def _initialize_improvement_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for identifying improvement opportunities"""
        return {
            'low_rating_patterns': {
                'threshold': 2.5,
                'min_occurrences': 3,
                'actions': ['review_content', 'adjust_parameters', 'improve_templates']
            },
            'correction_patterns': {
                'sanskrit_errors': {
                    'keywords': ['wrong', 'incorrect', 'mistake', 'error', 'sanskrit'],
                    'actions': ['verify_sanskrit_analysis', 'update_terminology', 'improve_transliteration']
                },
                'factual_errors': {
                    'keywords': ['wrong', 'incorrect', 'false', 'inaccurate'],
                    'actions': ['fact_check', 'update_knowledge_base', 'improve_accuracy_validation']
                }
            },
            'preference_patterns': {
                'style_preferences': {
                    'formal': ['formal', 'academic', 'scholarly', 'professional'],
                    'casual': ['casual', 'friendly', 'conversational', 'relaxed'],
                    'detailed': ['detailed', 'comprehensive', 'thorough', 'in-depth'],
                    'concise': ['concise', 'brief', 'short', 'summary']
                },
                'content_preferences': {
                    'more_sanskrit': ['more sanskrit', 'devanagari', 'original text'],
                    'less_sanskrit': ['less sanskrit', 'simpler', 'english only'],
                    'more_examples': ['examples', 'illustrations', 'demonstrations'],
                    'more_context': ['context', 'background', 'history', 'cultural']
                }
            }
        }
    
    async def collect_feedback(self, 
                             response_id: str,
                             feedback_data: Dict[str, Any],
                             session_id: str,
                             user_id: Optional[str] = None) -> UserFeedback:
        """
        Collect and process user feedback.
        
        Args:
            response_id: ID of the response being rated
            feedback_data: Feedback data from user
            session_id: Current session ID
            user_id: Optional user ID
            
        Returns:
            Processed UserFeedback object
        """
        try:
            # Generate feedback ID
            feedback_id = f"feedback_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Parse feedback data
            feedback_type = FeedbackType(feedback_data.get('type', 'rating'))
            category = self._classify_feedback_category(feedback_data)
            
            # Create feedback object
            feedback = UserFeedback(
                feedback_id=feedback_id,
                response_id=response_id,
                session_id=session_id,
                user_id=user_id,
                feedback_type=feedback_type,
                category=category,
                rating=feedback_data.get('rating'),
                text_feedback=feedback_data.get('text'),
                correction=feedback_data.get('correction'),
                original_query=feedback_data.get('original_query', ''),
                original_response=feedback_data.get('original_response', ''),
                context_tags=feedback_data.get('context_tags', []),
                user_agent=feedback_data.get('user_agent')
            )
            
            # Process feedback
            await self._process_feedback(feedback)
            
            # Store feedback
            self.feedback_history.append(feedback)
            
            # Clean up old feedback
            await self._cleanup_old_feedback()
            
            # Check for immediate improvement opportunities
            await self._check_immediate_improvements(feedback)
            
            logger.info(f"Collected feedback {feedback_id} for response {response_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            raise
    
    def _classify_feedback_category(self, feedback_data: Dict[str, Any]) -> FeedbackCategory:
        """Classify feedback into appropriate category"""
        
        text_content = (feedback_data.get('text', '') + ' ' + 
                       feedback_data.get('correction', '')).lower()
        
        # Sanskrit-related feedback
        if any(word in text_content for word in ['sanskrit', 'devanagari', 'transliteration', 'grammar']):
            return FeedbackCategory.SANSKRIT_ACCURACY
        
        # Content quality feedback
        if any(word in text_content for word in ['quality', 'accuracy', 'correct', 'wrong', 'error']):
            return FeedbackCategory.CONTENT_QUALITY
        
        # Relevance feedback
        if any(word in text_content for word in ['relevant', 'topic', 'question', 'answer', 'related']):
            return FeedbackCategory.RELEVANCE
        
        # Clarity feedback
        if any(word in text_content for word in ['clear', 'understand', 'confusing', 'unclear', 'explain']):
            return FeedbackCategory.CLARITY
        
        # Engagement feedback
        if any(word in text_content for word in ['engaging', 'interesting', 'boring', 'interactive']):
            return FeedbackCategory.ENGAGEMENT
        
        # Cultural sensitivity feedback
        if any(word in text_content for word in ['cultural', 'respectful', 'offensive', 'appropriate']):
            return FeedbackCategory.CULTURAL_SENSITIVITY
        
        # Technical issues
        if any(word in text_content for word in ['bug', 'error', 'broken', 'not working', 'technical']):
            return FeedbackCategory.TECHNICAL_ISSUE
        
        # Feature requests
        if any(word in text_content for word in ['feature', 'add', 'would like', 'suggestion', 'improve']):
            return FeedbackCategory.FEATURE_REQUEST
        
        # Default to content quality
        return FeedbackCategory.CONTENT_QUALITY
    
    async def _process_feedback(self, feedback: UserFeedback):
        """Process individual feedback entry"""
        
        # Calculate impact score
        feedback.impact_score = self._calculate_impact_score(feedback)
        
        # Extract user preferences
        if feedback.user_id:
            await self._extract_user_preferences(feedback)
        
        # Analyze feedback content
        await self._analyze_feedback_content(feedback)
        
        # Mark as processed
        feedback.processed = True
        
        logger.debug(f"Processed feedback {feedback.feedback_id} with impact score {feedback.impact_score:.2f}")
    
    def _calculate_impact_score(self, feedback: UserFeedback) -> float:
        """Calculate the potential impact score of feedback"""
        
        impact_factors = []
        
        # Rating-based impact
        if feedback.rating is not None:
            if feedback.rating <= 2:
                impact_factors.append(0.8)  # Low ratings have high impact
            elif feedback.rating >= 4:
                impact_factors.append(0.3)  # High ratings have lower impact for changes
            else:
                impact_factors.append(0.5)  # Neutral ratings
        
        # Feedback type impact
        type_impacts = {
            FeedbackType.CORRECTION: 0.9,
            FeedbackType.REPORT: 0.8,
            FeedbackType.SUGGESTION: 0.6,
            FeedbackType.PREFERENCE: 0.4,
            FeedbackType.RATING: 0.3,
            FeedbackType.APPRECIATION: 0.1
        }
        impact_factors.append(type_impacts.get(feedback.feedback_type, 0.5))
        
        # Category impact
        category_impact = self.category_weights.get(feedback.category, 0.1)
        impact_factors.append(category_impact)
        
        # Text content impact (longer, more detailed feedback has higher impact)
        if feedback.text_feedback:
            text_length_impact = min(0.5, len(feedback.text_feedback.split()) / 50.0)
            impact_factors.append(text_length_impact)
        
        # Calculate weighted average
        return sum(impact_factors) / len(impact_factors)
    
    async def _extract_user_preferences(self, feedback: UserFeedback):
        """Extract and update user preferences from feedback"""
        
        if not feedback.user_id:
            return
        
        user_prefs = self.user_preferences.get(feedback.user_id, {})
        
        # Extract style preferences
        if feedback.text_feedback:
            text_lower = feedback.text_feedback.lower()
            
            for style, keywords in self.improvement_patterns['preference_patterns']['style_preferences'].items():
                if any(keyword in text_lower for keyword in keywords):
                    user_prefs[f'style_{style}'] = user_prefs.get(f'style_{style}', 0) + 1
            
            for content_type, keywords in self.improvement_patterns['preference_patterns']['content_preferences'].items():
                if any(keyword in text_lower for keyword in keywords):
                    user_prefs[f'content_{content_type}'] = user_prefs.get(f'content_{content_type}', 0) + 1
        
        # Extract rating preferences
        if feedback.rating is not None:
            user_prefs['average_rating'] = user_prefs.get('ratings', [])
            user_prefs['ratings'] = user_prefs.get('ratings', [])
            user_prefs['ratings'].append(feedback.rating)
            
            # Keep only last 20 ratings
            if len(user_prefs['ratings']) > 20:
                user_prefs['ratings'] = user_prefs['ratings'][-20:]
            
            user_prefs['average_rating'] = sum(user_prefs['ratings']) / len(user_prefs['ratings'])
        
        # Update preferences
        self.user_preferences[feedback.user_id] = user_prefs
        
        logger.debug(f"Updated preferences for user {feedback.user_id}")
    
    async def _analyze_feedback_content(self, feedback: UserFeedback):
        """Analyze feedback content for patterns and insights"""
        
        if not feedback.text_feedback:
            return
        
        text_lower = feedback.text_feedback.lower()
        
        # Check for correction patterns
        for error_type, pattern_info in self.improvement_patterns['correction_patterns'].items():
            if any(keyword in text_lower for keyword in pattern_info['keywords']):
                # Add context tag for this error type
                if error_type not in feedback.context_tags:
                    feedback.context_tags.append(error_type)
        
        # Extract specific issues mentioned
        issue_keywords = {
            'too_long': ['too long', 'verbose', 'wordy'],
            'too_short': ['too short', 'brief', 'more detail'],
            'too_complex': ['complex', 'difficult', 'hard to understand'],
            'too_simple': ['too simple', 'basic', 'more advanced'],
            'missing_examples': ['examples', 'illustrations', 'demonstrate'],
            'missing_context': ['context', 'background', 'why', 'history']
        }
        
        for issue, keywords in issue_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if issue not in feedback.context_tags:
                    feedback.context_tags.append(issue)
    
    async def _check_immediate_improvements(self, feedback: UserFeedback):
        """Check if feedback triggers immediate improvement actions"""
        
        # High-impact feedback triggers immediate review
        if feedback.impact_score > 0.8:
            await self._create_improvement_action(
                feedback_ids=[feedback.feedback_id],
                action_type="immediate_review",
                description=f"High-impact feedback requires immediate attention: {feedback.feedback_type.value}",
                priority=5
            )
        
        # Critical errors trigger immediate fixes
        if feedback.feedback_type == FeedbackType.REPORT and 'critical' in feedback.context_tags:
            await self._create_improvement_action(
                feedback_ids=[feedback.feedback_id],
                action_type="critical_fix",
                description="Critical issue reported by user",
                priority=5
            )
        
        # Sanskrit accuracy issues trigger verification
        if feedback.category == FeedbackCategory.SANSKRIT_ACCURACY and feedback.rating and feedback.rating <= 2:
            await self._create_improvement_action(
                feedback_ids=[feedback.feedback_id],
                action_type="sanskrit_verification",
                description="Sanskrit accuracy issue requires verification",
                priority=4
            )
    
    async def _create_improvement_action(self,
                                       feedback_ids: List[str],
                                       action_type: str,
                                       description: str,
                                       priority: int,
                                       estimated_impact: float = 0.5) -> ImprovementAction:
        """Create an improvement action based on feedback"""
        
        action = ImprovementAction(
            action_id=f"action_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            feedback_ids=feedback_ids,
            action_type=action_type,
            description=description,
            priority=priority,
            estimated_impact=estimated_impact
        )
        
        self.improvement_actions.append(action)
        
        logger.info(f"Created improvement action {action.action_id}: {description}")
        return action
    
    async def analyze_feedback_trends(self, 
                                    time_window_days: int = 30,
                                    min_feedback_count: int = 5) -> FeedbackAnalysis:
        """
        Analyze feedback trends and patterns.
        
        Args:
            time_window_days: Number of days to analyze
            min_feedback_count: Minimum feedback count for analysis
            
        Returns:
            Comprehensive feedback analysis
        """
        try:
            # Filter feedback within time window
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            recent_feedback = [
                f for f in self.feedback_history 
                if f.timestamp >= cutoff_date
            ]
            
            if len(recent_feedback) < min_feedback_count:
                logger.warning(f"Insufficient feedback for analysis: {len(recent_feedback)} < {min_feedback_count}")
                return self._create_minimal_analysis(recent_feedback)
            
            # Calculate basic statistics
            ratings = [f.rating for f in recent_feedback if f.rating is not None]
            average_rating = statistics.mean(ratings) if ratings else 0.0
            
            # Rating distribution
            rating_distribution = {}
            for rating in range(1, 6):
                rating_distribution[rating] = sum(1 for r in ratings if r == rating)
            
            # Category breakdown
            category_breakdown = {}
            for feedback in recent_feedback:
                category = feedback.category.value
                category_breakdown[category] = category_breakdown.get(category, 0) + 1
            
            # Common issues analysis
            common_issues = self._identify_common_issues(recent_feedback)
            
            # Improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(recent_feedback)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(recent_feedback)
            
            # User satisfaction score
            satisfaction_score = self._calculate_satisfaction_score(recent_feedback)
            
            analysis = FeedbackAnalysis(
                total_feedback_count=len(recent_feedback),
                average_rating=average_rating,
                rating_distribution=rating_distribution,
                category_breakdown=category_breakdown,
                common_issues=common_issues,
                improvement_suggestions=improvement_suggestions,
                trend_analysis=trend_analysis,
                user_satisfaction_score=satisfaction_score
            )
            
            logger.info(f"Completed feedback analysis: {len(recent_feedback)} feedback entries, avg rating: {average_rating:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Feedback analysis failed: {e}")
            return self._create_minimal_analysis([])
    
    def _identify_common_issues(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Identify common issues from feedback"""
        
        issue_counts = {}
        
        for feedback in feedback_list:
            # Count context tags (issues)
            for tag in feedback.context_tags:
                issue_counts[tag] = issue_counts.get(tag, 0) + 1
            
            # Count low ratings as general quality issues
            if feedback.rating and feedback.rating <= 2:
                issue_counts['low_quality'] = issue_counts.get('low_quality', 0) + 1
            
            # Count corrections as accuracy issues
            if feedback.feedback_type == FeedbackType.CORRECTION:
                issue_counts['accuracy_issues'] = issue_counts.get('accuracy_issues', 0) + 1
        
        # Sort by frequency and return top issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:10] if count >= 2]
    
    def _generate_improvement_suggestions(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Generate improvement suggestions based on feedback patterns"""
        
        suggestions = []
        
        # Analyze rating patterns
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        if ratings:
            avg_rating = statistics.mean(ratings)
            if avg_rating < 3.0:
                suggestions.append("Overall response quality needs improvement")
            elif avg_rating < 3.5:
                suggestions.append("Consider enhancing response relevance and accuracy")
        
        # Analyze category patterns
        category_counts = {}
        for feedback in feedback_list:
            category_counts[feedback.category] = category_counts.get(feedback.category, 0) + 1
        
        # Suggest improvements for most problematic categories
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count >= 3:  # Significant number of issues
                if category == FeedbackCategory.SANSKRIT_ACCURACY:
                    suggestions.append("Improve Sanskrit terminology and transliteration accuracy")
                elif category == FeedbackCategory.CLARITY:
                    suggestions.append("Enhance response clarity and readability")
                elif category == FeedbackCategory.RELEVANCE:
                    suggestions.append("Better align responses with user queries")
                elif category == FeedbackCategory.ENGAGEMENT:
                    suggestions.append("Make responses more engaging and interactive")
        
        # Analyze text feedback for specific suggestions
        text_feedback = [f.text_feedback for f in feedback_list if f.text_feedback]
        combined_text = ' '.join(text_feedback).lower()
        
        if 'too long' in combined_text or 'verbose' in combined_text:
            suggestions.append("Make responses more concise")
        
        if 'too short' in combined_text or 'more detail' in combined_text:
            suggestions.append("Provide more detailed explanations")
        
        if 'examples' in combined_text:
            suggestions.append("Include more examples and illustrations")
        
        if 'context' in combined_text:
            suggestions.append("Provide more cultural and historical context")
        
        return suggestions[:8]  # Limit to top 8 suggestions
    
    def _analyze_trends(self, feedback_list: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze trends in feedback over time"""
        
        if len(feedback_list) < 10:
            return {'status': 'insufficient_data'}
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_list, key=lambda x: x.timestamp)
        
        # Split into early and recent periods
        mid_point = len(sorted_feedback) // 2
        early_feedback = sorted_feedback[:mid_point]
        recent_feedback = sorted_feedback[mid_point:]
        
        # Compare ratings
        early_ratings = [f.rating for f in early_feedback if f.rating is not None]
        recent_ratings = [f.rating for f in recent_feedback if f.rating is not None]
        
        trend_data = {}
        
        if early_ratings and recent_ratings:
            early_avg = statistics.mean(early_ratings)
            recent_avg = statistics.mean(recent_ratings)
            
            trend_data['rating_trend'] = {
                'early_average': early_avg,
                'recent_average': recent_avg,
                'change': recent_avg - early_avg,
                'direction': 'improving' if recent_avg > early_avg else 'declining' if recent_avg < early_avg else 'stable'
            }
        
        # Compare issue frequencies
        early_issues = {}
        recent_issues = {}
        
        for feedback in early_feedback:
            for tag in feedback.context_tags:
                early_issues[tag] = early_issues.get(tag, 0) + 1
        
        for feedback in recent_feedback:
            for tag in feedback.context_tags:
                recent_issues[tag] = recent_issues.get(tag, 0) + 1
        
        # Identify improving and worsening issues
        improving_issues = []
        worsening_issues = []
        
        all_issues = set(early_issues.keys()) | set(recent_issues.keys())
        for issue in all_issues:
            early_count = early_issues.get(issue, 0)
            recent_count = recent_issues.get(issue, 0)
            
            if early_count > 0 and recent_count < early_count:
                improving_issues.append(issue)
            elif recent_count > early_count:
                worsening_issues.append(issue)
        
        trend_data['issue_trends'] = {
            'improving': improving_issues,
            'worsening': worsening_issues
        }
        
        return trend_data
    
    def _calculate_satisfaction_score(self, feedback_list: List[UserFeedback]) -> float:
        """Calculate overall user satisfaction score"""
        
        if not feedback_list:
            return 0.5  # Neutral score
        
        satisfaction_factors = []
        
        # Rating-based satisfaction
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        if ratings:
            avg_rating = statistics.mean(ratings)
            rating_satisfaction = (avg_rating - 1) / 4  # Normalize to 0-1
            satisfaction_factors.append(rating_satisfaction * 0.5)
        
        # Feedback type satisfaction
        positive_feedback = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.APPRECIATION)
        negative_feedback = sum(1 for f in feedback_list if f.feedback_type in [FeedbackType.CORRECTION, FeedbackType.REPORT])
        
        if positive_feedback + negative_feedback > 0:
            feedback_ratio = positive_feedback / (positive_feedback + negative_feedback)
            satisfaction_factors.append(feedback_ratio * 0.3)
        
        # Impact score satisfaction (lower average impact = higher satisfaction)
        impact_scores = [f.impact_score for f in feedback_list]
        if impact_scores:
            avg_impact = statistics.mean(impact_scores)
            impact_satisfaction = 1.0 - avg_impact  # Invert impact score
            satisfaction_factors.append(impact_satisfaction * 0.2)
        
        if not satisfaction_factors:
            return 0.5
        
        return sum(satisfaction_factors) / len(satisfaction_factors)
    
    def _create_minimal_analysis(self, feedback_list: List[UserFeedback]) -> FeedbackAnalysis:
        """Create minimal analysis when insufficient data"""
        
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        avg_rating = statistics.mean(ratings) if ratings else 0.0
        
        return FeedbackAnalysis(
            total_feedback_count=len(feedback_list),
            average_rating=avg_rating,
            rating_distribution={i: 0 for i in range(1, 6)},
            category_breakdown={},
            common_issues=[],
            improvement_suggestions=["Collect more feedback for detailed analysis"],
            trend_analysis={'status': 'insufficient_data'},
            user_satisfaction_score=0.5
        )
    
    async def _cleanup_old_feedback(self):
        """Clean up old feedback entries"""
        
        cutoff_date = datetime.now() - timedelta(days=self.feedback_retention_days)
        
        # Remove old feedback
        original_count = len(self.feedback_history)
        self.feedback_history = [
            f for f in self.feedback_history 
            if f.timestamp >= cutoff_date
        ]
        
        removed_count = original_count - len(self.feedback_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old feedback entries")
    
    async def apply_improvements(self, max_actions: int = 5) -> List[Dict[str, Any]]:
        """
        Apply pending improvement actions.
        
        Args:
            max_actions: Maximum number of actions to apply
            
        Returns:
            List of applied actions with results
        """
        try:
            # Get pending high-priority actions
            pending_actions = [
                action for action in self.improvement_actions
                if action.implementation_status == "pending"
            ]
            
            # Sort by priority and estimated impact
            pending_actions.sort(
                key=lambda x: (x.priority, x.estimated_impact),
                reverse=True
            )
            
            applied_actions = []
            
            for action in pending_actions[:max_actions]:
                try:
                    result = await self._apply_improvement_action(action)
                    applied_actions.append({
                        'action_id': action.action_id,
                        'action_type': action.action_type,
                        'description': action.description,
                        'result': result,
                        'success': True
                    })
                    
                    action.implementation_status = "completed"
                    
                except Exception as e:
                    logger.error(f"Failed to apply improvement action {action.action_id}: {e}")
                    applied_actions.append({
                        'action_id': action.action_id,
                        'action_type': action.action_type,
                        'description': action.description,
                        'result': str(e),
                        'success': False
                    })
                    
                    action.implementation_status = "failed"
            
            logger.info(f"Applied {len(applied_actions)} improvement actions")
            return applied_actions
            
        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            return []
    
    async def _apply_improvement_action(self, action: ImprovementAction) -> str:
        """Apply a specific improvement action"""
        
        # This is a simplified implementation
        # In a full system, this would integrate with the actual synthesis components
        
        if action.action_type == "immediate_review":
            return "Flagged for manual review"
        
        elif action.action_type == "critical_fix":
            return "Escalated to development team"
        
        elif action.action_type == "sanskrit_verification":
            return "Added to Sanskrit accuracy review queue"
        
        elif action.action_type == "parameter_adjustment":
            return "Adjusted synthesis parameters based on feedback"
        
        elif action.action_type == "template_update":
            return "Updated response templates"
        
        else:
            return f"Applied {action.action_type} improvement"
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get preferences for a specific user"""
        return self.user_preferences.get(user_id, {})
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics"""
        
        if not self.feedback_history:
            return {'message': 'No feedback data available'}
        
        # Basic statistics
        total_feedback = len(self.feedback_history)
        ratings = [f.rating for f in self.feedback_history if f.rating is not None]
        
        stats = {
            'total_feedback': total_feedback,
            'feedback_with_ratings': len(ratings),
            'average_rating': statistics.mean(ratings) if ratings else 0.0,
            'rating_distribution': {},
            'feedback_types': {},
            'categories': {},
            'recent_trends': {},
            'improvement_actions': {
                'total': len(self.improvement_actions),
                'pending': len([a for a in self.improvement_actions if a.implementation_status == "pending"]),
                'completed': len([a for a in self.improvement_actions if a.implementation_status == "completed"]),
                'failed': len([a for a in self.improvement_actions if a.implementation_status == "failed"])
            }
        }
        
        # Rating distribution
        for rating in range(1, 6):
            stats['rating_distribution'][rating] = sum(1 for r in ratings if r == rating)
        
        # Feedback type distribution
        for feedback in self.feedback_history:
            feedback_type = feedback.feedback_type.value
            stats['feedback_types'][feedback_type] = stats['feedback_types'].get(feedback_type, 0) + 1
        
        # Category distribution
        for feedback in self.feedback_history:
            category = feedback.category.value
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        return stats