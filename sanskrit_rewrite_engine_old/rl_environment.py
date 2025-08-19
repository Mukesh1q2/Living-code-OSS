"""
Reinforcement Learning Environment for Sanskrit Rule Application.

This module provides a reinforcement learning environment that allows
agents to learn Sanskrit rule application through interaction with
the Pāṇini rule engine. It's designed to integrate with R-Zero's
self-evolving reasoning framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
import random
from datetime import datetime
import logging

from .token import Token, TokenKind
from .panini_engine import PaniniRuleEngine, PaniniEngineResult
from .rule import SutraRule, RuleRegistry
from .r_zero_integration import SanskritProblem, SanskritProblemType, SanskritGrammaticalValidator


class ActionType(Enum):
    """Types of actions available in the Sanskrit RL environment."""
    APPLY_RULE = "APPLY_RULE"
    SELECT_POSITION = "SELECT_POSITION"
    SKIP = "SKIP"
    TERMINATE = "TERMINATE"


@dataclass
class SanskritAction:
    """An action in the Sanskrit rule application environment."""
    type: ActionType
    rule_id: Optional[str] = None
    position: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'type': self.type.value,
            'rule_id': self.rule_id,
            'position': self.position,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class SanskritState:
    """State representation in the Sanskrit RL environment."""
    tokens: List[Token]
    available_rules: List[SutraRule]
    applied_rules: List[str] = field(default_factory=list)
    step_count: int = 0
    target_output: Optional[str] = None
    problem_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'tokens': [{'text': t.text, 'kind': t.kind.value} for t in self.tokens],
            'available_rules': [r.id for r in self.available_rules],
            'applied_rules': self.applied_rules,
            'step_count': self.step_count,
            'target_output': self.target_output,
            'problem_context': self.problem_context,
            'metadata': self.metadata
        }
    
    def get_state_vector(self) -> np.ndarray:
        """Convert state to numerical vector for RL algorithms."""
        # Create a fixed-size state representation
        state_features = []
        
        # Token features (first 20 tokens, padded/truncated)
        max_tokens = 20
        token_features = []
        for i in range(max_tokens):
            if i < len(self.tokens):
                token = self.tokens[i]
                # Encode token kind as one-hot
                kind_encoding = [0, 0, 0, 0]  # VOWEL, CONSONANT, MARKER, OTHER
                if token.kind == TokenKind.VOWEL:
                    kind_encoding[0] = 1
                elif token.kind == TokenKind.CONSONANT:
                    kind_encoding[1] = 1
                elif token.kind == TokenKind.MARKER:
                    kind_encoding[2] = 1
                else:
                    kind_encoding[3] = 1
                
                # Add token length (normalized)
                token_length = min(len(token.text), 10) / 10.0
                token_features.extend(kind_encoding + [token_length])
            else:
                # Padding
                token_features.extend([0, 0, 0, 0, 0])
        
        state_features.extend(token_features)
        
        # Rule availability features (first 50 rules)
        max_rules = 50
        rule_features = []
        for i in range(max_rules):
            if i < len(self.available_rules):
                rule_features.append(1.0)  # Rule available
            else:
                rule_features.append(0.0)  # No rule
        
        state_features.extend(rule_features)
        
        # Step count (normalized)
        normalized_step_count = min(self.step_count, 100) / 100.0
        state_features.append(normalized_step_count)
        
        # Applied rules count (normalized)
        normalized_applied_count = min(len(self.applied_rules), 20) / 20.0
        state_features.append(normalized_applied_count)
        
        return np.array(state_features, dtype=np.float32)
    
    def _calculate_state_space_size(self) -> int:
        """Calculate the size of the state space."""
        # Based on the state vector in get_state_vector
        max_tokens = 20
        token_feature_size = 5  # 4 for kind encoding + 1 for length
        max_rules = 50
        additional_features = 2  # step count + applied rules count
        return max_tokens * token_feature_size + max_rules + additional_features


@dataclass
class SanskritReward:
    """Reward information for Sanskrit RL training."""
    total_reward: float
    components: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reward to dictionary."""
        return {
            'total_reward': self.total_reward,
            'components': self.components,
            'explanation': self.explanation,
            'metadata': self.metadata
        }


class SanskritRLEnvironment:
    """Reinforcement Learning Environment for Sanskrit rule application."""
    
    def __init__(self, 
                 rule_engine: PaniniRuleEngine,
                 validator: SanskritGrammaticalValidator,
                 max_steps: int = 50,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Sanskrit RL environment.
        
        Args:
            rule_engine: Pāṇini rule engine for rule application
            validator: Sanskrit grammatical validator
            max_steps: Maximum steps per episode
            reward_weights: Weights for different reward components
        """
        self.rule_engine = rule_engine
        self.validator = validator
        self.max_steps = max_steps
        self.logger = logging.getLogger(__name__)
        
        # Reward configuration
        self.reward_weights = reward_weights or {
            'convergence_bonus': 2.0,
            'rule_correctness': 1.0,
            'grammatical_improvement': 0.8,
            'target_match_bonus': 3.0,
            'step_penalty': -0.1,
            'invalid_action_penalty': -0.5
        }
        
        # Environment state
        self.current_state: Optional[SanskritState] = None
        self.episode_history: List[Dict[str, Any]] = []
        self.episode_count = 0
        
        # State and action space sizes
        self.state_space_size = self._calculate_state_space_size()
        self.action_space_size = self._calculate_action_space_size()
    
    def _calculate_action_space_size(self) -> int:
        """Calculate the size of the action space."""
        # Action types
        num_action_types = len(ActionType)  # APPLY_RULE, SELECT_POSITION, SKIP, TERMINATE
        
        # Maximum number of rules (for APPLY_RULE actions)
        max_rules = len(self.rule_engine.registry.get_active_sutra_rules())
        
        # Maximum positions (for SELECT_POSITION actions)
        max_positions = 50  # Reasonable upper bound
        
        # Total action space
        return num_action_types + max_rules + max_positions
    
    def _calculate_state_space_size(self) -> int:
        """Calculate the size of the state space."""
        # Based on the state vector components
        max_tokens = 20
        token_feature_size = 5  # 4 for kind encoding + 1 for length
        max_rules = 50
        additional_features = 2  # step count + applied rules count
        return max_tokens * token_feature_size + max_rules + additional_features
    
    def reset(self, 
              problem: Optional[SanskritProblem] = None,
              initial_tokens: Optional[List[Token]] = None) -> SanskritState:
        """
        Reset the environment for a new episode.
        
        Args:
            problem: Sanskrit problem to solve (optional)
            initial_tokens: Initial token sequence (optional)
            
        Returns:
            Initial state of the environment
        """
        self.logger.info(f"Environment reset for episode {self.episode_count}")
        
        # Initialize tokens
        if initial_tokens:
            tokens = initial_tokens
        elif problem:
            tokens = self.rule_engine.tokenizer.tokenize(problem.input_text)
        else:
            # Default: simple Sanskrit phrase
            tokens = self.rule_engine.tokenizer.tokenize("a + i")
        
        # Get available rules
        available_rules = self.rule_engine.registry.get_active_sutra_rules()
        
        # Create initial state
        self.current_state = SanskritState(
            tokens=tokens,
            available_rules=available_rules,
            applied_rules=[],
            step_count=0,
            target_output=problem.expected_output if problem else None,
            problem_context=problem.to_r_zero_format() if problem else None,
            metadata={}
        )
        
        # Reset episode tracking
        self.episode_history = []
        self.episode_count += 1
        
        self.logger.info(f"Environment initialized with {len(tokens)} tokens")
        return self.current_state
    
    def step(self, action: SanskritAction) -> Tuple[SanskritState, SanskritReward, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        self.logger.info(f"Executing action {action.type} for episode {self.episode_count}")
        
        # Record action in history
        step_info = {
            'step': self.current_state.step_count,
            'action': action.to_dict(),
            'state_before': self.current_state.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute action
        next_state, reward, done = self._execute_action(action)
        
        # Update step info
        step_info.update({
            'state_after': next_state.to_dict(),
            'reward': reward.to_dict(),
            'done': done
        })
        
        self.episode_history.append(step_info)
        
        # Additional info for debugging
        info = {
            'episode_id': self.episode_count,
            'episode_step': next_state.step_count,
            'action_valid': reward.components.get('invalid_action', 0) == 0,
            'reward_explanation': reward.explanation,
            'total_reward': reward.total_reward
        }
        
        self.current_state = next_state
        return next_state, reward, done, info
    
    def _execute_action(self, action: SanskritAction) -> Tuple[SanskritState, SanskritReward, bool]:
        """Execute a specific action and return the result."""
        current_tokens = self.current_state.tokens
        reward_components = {}
        done = False
        
        # Step penalty
        reward_components['step_penalty'] = self.reward_weights['step_penalty']
        
        if action.type == ActionType.APPLY_RULE:
            next_tokens, reward_components = self._apply_rule_action(action, current_tokens, reward_components)
        elif action.type == ActionType.SELECT_POSITION:
            # No change for position selection
            next_tokens = current_tokens
            reward_components['position_selection'] = 0.1  # Small positive reward
        elif action.type == ActionType.SKIP:
            # No change for skip
            next_tokens = current_tokens
            reward_components['skip_action'] = 0.0  # Neutral reward
        elif action.type == ActionType.TERMINATE:
            next_tokens = current_tokens
            reward_components.update(self._calculate_termination_reward(current_tokens))
            done = True
        else:
            # Invalid action
            next_tokens = current_tokens
            reward_components['invalid_action_penalty'] = self.reward_weights['invalid_action_penalty']
        
        # Check for episode termination conditions
        if self.current_state.step_count >= self.max_steps:
            done = True
            reward_components['max_steps_reached'] = -1.0
        
        # Create next state
        next_state = SanskritState(
            tokens=next_tokens,
            available_rules=self._get_applicable_rules(next_tokens),
            applied_rules=self.current_state.applied_rules + ([action.rule_id] if action.rule_id else []),
            step_count=self.current_state.step_count + 1,
            target_output=self.current_state.target_output,
            problem_context=self.current_state.problem_context,
            metadata=self.current_state.metadata
        )
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Generate explanation
        explanation = self._generate_reward_explanation(reward_components)
        
        reward = SanskritReward(
            total_reward=total_reward,
            components=reward_components,
            explanation=explanation,
            metadata={}
        )
        
        return next_state, reward, done
    
    def _apply_rule_action(self, action: SanskritAction, tokens: List[Token], 
                          reward_components: Dict[str, float]) -> Tuple[List[Token], Dict[str, float]]:
        """Apply a rule action."""
        # Find the rule
        rule = None
        for r in self.current_state.available_rules:
            if r.id == action.rule_id:
                rule = r
                break
        
        if rule is None:
            reward_components['invalid_action_penalty'] = self.reward_weights['invalid_action_penalty']
            return tokens, reward_components
        
        try:
            # Apply only this rule through a temporary engine with only this rule
            temp_registry = RuleRegistry()
            temp_registry.add_sutra_rule(rule)
            temp_engine = PaniniRuleEngine(self.rule_engine.tokenizer, temp_registry)
            
            # Process tokens through engine
            original_text = ''.join(t.text for t in tokens)
            result = temp_engine.process(tokens, max_passes=1)
            
            if result.converged and result.output_tokens != tokens:
                # Rule was successfully applied
                reward_components['rule_correctness'] = self.reward_weights['rule_correctness']
                
                # Check for target match bonus
                if self.current_state.target_output:
                    new_text = ''.join(t.text for t in result.output_tokens).strip()
                    if new_text == self.current_state.target_output.strip():
                        reward_components['target_match_bonus'] = self.reward_weights['target_match_bonus']
                    else:
                        # Partial credit for similarity
                        similarity = self._calculate_text_similarity(new_text, self.current_state.target_output)
                        reward_components['target_similarity'] = similarity * 0.5
                
                # Calculate grammatical improvement
                old_validation = self.validator.validate_sanskrit_text(original_text)
                new_validation = self.validator.validate_sanskrit_text(''.join(t.text for t in result.output_tokens))
                improvement = new_validation['confidence'] - old_validation['confidence']
                reward_components['grammatical_improvement'] = improvement * self.reward_weights['grammatical_improvement']
                
                return result.output_tokens, reward_components
            else:
                # Rule didn't apply or no change
                reward_components['rule_no_effect'] = -0.2
                return tokens, reward_components
                
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.id}: {e}")
            reward_components['rule_application_error'] = -0.5
            return tokens, reward_components
    
    def _calculate_termination_reward(self, tokens: List[Token]) -> Dict[str, float]:
        """Calculate reward for termination action."""
        reward_components = {}
        
        # Validate final result
        current_text = ''.join(t.text for t in tokens)
        validation = self.validator.validate_sanskrit_text(current_text)
        reward_components['final_grammatical_score'] = validation['confidence']
        
        # Check convergence bonus
        if self.current_state.target_output:
            if current_text.strip() == self.current_state.target_output.strip():
                reward_components['convergence_bonus'] = self.reward_weights['convergence_bonus']
            else:
                # Partial credit
                similarity = self._calculate_text_similarity(current_text, self.current_state.target_output)
                reward_components['target_similarity'] = similarity * 0.5
        
        return reward_components
    
    def _get_applicable_rules(self, tokens: List[Token]) -> List[SutraRule]:
        """Get rules that can be applied to the current token sequence."""
        applicable_rules = []
        
        for rule in self.rule_engine.registry.get_active_sutra_rules():
            # Check if rule can be applied at any position
            for i in range(len(tokens)):
                try:
                    if rule.match_fn(tokens, i):
                        applicable_rules.append(rule)
                        break  # Rule is applicable, no need to check other positions
                except:
                    continue  # Rule check failed, continue to next rule
        
        return applicable_rules
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple character-based similarity
        if not text1 or not text2:
            return 0.0
        
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 and not text2:
            return 1.0
        
        if text1 == text2:
            return 1.0
        
        # Calculate character overlap
        chars1 = set(text1)
        chars2 = set(text2)
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_reward_explanation(self, reward_components: Dict[str, float]) -> str:
        """Generate human-readable explanation of reward."""
        explanations = []
        
        for component, value in reward_components.items():
            if value > 0:
                explanations.append(f"+{value:.2f} for {component}")
            elif value < 0:
                explanations.append(f"{value:.2f} for {component}")
        
        if not explanations:
            return "No reward components"
        
        return "; ".join(explanations)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the current episode."""
        if not self.episode_history:
            return {}
        
        total_steps = len(self.episode_history)
        total_reward = sum(step['reward']['total_reward'] for step in self.episode_history)
        
        # Count action types
        action_counts = {}
        for step in self.episode_history:
            action_type = step['action']['type']
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Final state
        final_state = self.episode_history[-1]['state_after'] if self.episode_history else None
        
        return {
            'episode_id': self.episode_count,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'average_reward': total_reward / total_steps if total_steps > 0 else 0,
            'action_counts': action_counts,
            'final_state': final_state,
            'episode_history': self.episode_history
        }
    
    def render(self, mode: str = 'text') -> str:
        """Render the current state of the environment."""
        if self.current_state is None:
            return "Environment not initialized"
        
        if mode == 'text':
            lines = []
            lines.append(f"Episode: {self.episode_count}")
            lines.append(f"Step: {self.current_state.step_count}")
            lines.append(f"Current text: {''.join(t.text for t in self.current_state.tokens)}")
            lines.append(f"Available rules: {len(self.current_state.available_rules)}")
            lines.append(f"Applied rules: {len(self.current_state.applied_rules)}")
            
            if self.current_state.target_output:
                lines.append(f"Target: {self.current_state.target_output}")
            
            return "\n".join(lines)
        elif mode == 'json':
            import json
            return json.dumps(self.current_state.to_dict(), indent=2, ensure_ascii=False)
        else:
            return f"Unsupported render mode: {mode}"


class SanskritRLTrainer:
    """Trainer for Sanskrit RL agents."""
    
    def __init__(self, environment: SanskritRLEnvironment):
        """
        Initialize the RL trainer.
        
        Args:
            environment: Sanskrit RL environment
        """
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'success_rate': 0.0,
            'episode_rewards': [],
            'episode_lengths': []
        }
    
    def train_episode(self, problem: SanskritProblem, agent_policy) -> Dict[str, Any]:
        """
        Train for one episode using the given agent policy.
        
        Args:
            problem: Sanskrit problem to solve
            agent_policy: Function that takes state and returns action
            
        Returns:
            Episode results and statistics
        """
        # Reset environment
        state = self.environment.reset(problem)
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Get action from agent policy
            action = agent_policy(state)
            
            # Execute action
            next_state, reward, done, info = self.environment.step(action)
            
            episode_reward += reward.total_reward
            episode_steps += 1
            state = next_state
        
        # Update training statistics
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_steps'] += episode_steps
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_steps)
        
        # Calculate running averages
        self.training_stats['average_reward'] = np.mean(self.training_stats['episode_rewards'])
        
        # Calculate success rate (episodes with positive reward)
        successful_episodes = sum(1 for r in self.training_stats['episode_rewards'] if r > 0)
        self.training_stats['success_rate'] = successful_episodes / self.training_stats['episodes_completed']
        
        self.logger.info(f"Episode {self.training_stats['episodes_completed']} completed: "
                        f"reward={episode_reward:.2f}, steps={episode_steps}")
        
        episode_summary = self.environment.get_episode_summary()
        
        return {
            'episode_id': self.training_stats['episodes_completed'],
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_summary': episode_summary,
            'training_stats': self.training_stats.copy()
        }
    
    def generate_training_data(self, problems: List[SanskritProblem], 
                             episodes_per_problem: int = 5) -> List[Dict[str, Any]]:
        """
        Generate training data from Sanskrit problems.
        
        Args:
            problems: List of Sanskrit problems
            episodes_per_problem: Number of episodes per problem
            
        Returns:
            List of training episodes
        """
        training_data = []
        
        for problem in problems:
            for episode_num in range(episodes_per_problem):
                # Simple random policy for data generation
                def random_policy(state: SanskritState) -> SanskritAction:
                    action_type = random.choice(list(ActionType))
                    
                    if action_type == ActionType.APPLY_RULE:
                        if state.available_rules:
                            rule = random.choice(state.available_rules)
                            return SanskritAction(type=action_type, rule_id=rule.id)
                        else:
                            return SanskritAction(type=ActionType.SKIP)
                    elif action_type == ActionType.SELECT_POSITION:
                        position = random.randint(0, len(state.tokens) - 1) if state.tokens else 0
                        return SanskritAction(type=action_type, position=position)
                    else:
                        return SanskritAction(type=action_type)
                
                episode_data = self.train_episode(problem, random_policy)
                training_data.append(episode_data)
        
        return training_data
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'training_stats': self.training_stats,
            'environment_ready': self.environment.current_state is not None,
            'state_space_size': self.environment.state_space_size,
            'action_space_size': self.environment.action_space_size
        }


# Utility functions for R-Zero integration
def create_sanskrit_rl_environment(rule_engine: PaniniRuleEngine, 
                                 validator: SanskritGrammaticalValidator,
                                 **kwargs) -> SanskritRLEnvironment:
    """
    Create a configured Sanskrit RL environment.
    
    Args:
        rule_engine: Pāṇini rule engine
        validator: Sanskrit grammatical validator
        **kwargs: Additional configuration
        
    Returns:
        Configured Sanskrit RL environment
    """
    return SanskritRLEnvironment(rule_engine, validator, **kwargs)


def create_sanskrit_rl_trainer(environment: SanskritRLEnvironment) -> SanskritRLTrainer:
    """
    Create a Sanskrit RL trainer.
    
    Args:
        environment: Sanskrit RL environment
        
    Returns:
        Sanskrit RL trainer
    """
    return SanskritRLTrainer(environment)