"""
Persistent Sanskrit Knowledge Evolution System.

This module implements persistent knowledge evolution for the Sanskrit reasoning system,
integrating with R-Zero's model checkpointing and providing incremental learning
capabilities for Sanskrit grammatical rules and constructions.

Key Features:
- Integration with R-Zero model checkpointing and versioning
- Sanskrit-specific dataset generation and curation pipeline
- Incremental learning for new Sanskrit constructions
- Rule confidence adaptation based on R-Zero performance metrics
- Sanskrit corpus expansion through self-generated examples
- Knowledge persistence and retrieval with version control
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import threading
from collections import defaultdict, deque

# R-Zero integration imports
from .r_zero_integration import (
    SanskritProblem, 
    SanskritProblemType, 
    SanskritDifficultyLevel,
    SanskritProblemGenerator
)
from .sanskrit_reward_function import compute_score
from .panini_engine import PaniniRuleEngine, SutraRule
from .tokenizer import SanskritTokenizer
from .morphological_analyzer import SanskritMorphologicalAnalyzer
from .derivation_simulator import ShabdaPrakriyaSimulator


logger = logging.getLogger(__name__)


class KnowledgeVersionType(Enum):
    """Types of knowledge versions."""
    MAJOR = "major"  # Significant rule changes or new capabilities
    MINOR = "minor"  # Rule refinements or parameter updates
    PATCH = "patch"  # Bug fixes or small corrections
    SNAPSHOT = "snapshot"  # Regular checkpoints


class EvolutionStrategy(Enum):
    """Strategies for knowledge evolution."""
    CONSERVATIVE = "conservative"  # Only high-confidence changes
    BALANCED = "balanced"  # Moderate risk tolerance
    AGGRESSIVE = "aggressive"  # Accept lower-confidence improvements
    EXPERIMENTAL = "experimental"  # Test new approaches


@dataclass
class KnowledgeVersion:
    """Represents a version of Sanskrit knowledge."""
    version_id: str
    version_type: KnowledgeVersionType
    timestamp: datetime
    description: str
    rule_count: int
    performance_metrics: Dict[str, float]
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['version_type'] = self.version_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeVersion':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['version_type'] = KnowledgeVersionType(data['version_type'])
        return cls(**data)


@dataclass
class RuleConfidence:
    """Tracks confidence metrics for Sanskrit rules."""
    rule_id: str
    confidence_score: float
    application_count: int
    success_count: int
    failure_count: int
    last_updated: datetime
    performance_history: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def update_performance(self, success: bool, score: float = None):
        """Update performance metrics."""
        self.application_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        if score is not None:
            self.performance_history.append(score)
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Update confidence based on recent performance
            recent_scores = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            if recent_scores:
                self.confidence_score = sum(recent_scores) / len(recent_scores)
        
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class DatasetEntry:
    """Represents an entry in the Sanskrit dataset."""
    id: str
    problem: SanskritProblem
    generated_by: str  # 'human', 'self-generated', 'augmented'
    quality_score: float
    validation_status: str  # 'pending', 'validated', 'rejected'
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SanskritKnowledgeEvolution:
    """
    Main class for persistent Sanskrit knowledge evolution.
    
    Manages the evolution of Sanskrit grammatical knowledge through:
    - Integration with R-Zero checkpointing
    - Incremental learning and rule adaptation
    - Dataset generation and curation
    - Performance tracking and optimization
    """
    
    def __init__(self, 
                 storage_path: str,
                 tokenizer: SanskritTokenizer,
                 rule_engine: PaniniRuleEngine,
                 morphological_analyzer: SanskritMorphologicalAnalyzer,
                 derivation_simulator: ShabdaPrakriyaSimulator,
                 evolution_strategy: EvolutionStrategy = EvolutionStrategy.BALANCED):
        """
        Initialize the knowledge evolution system.
        
        Args:
            storage_path: Path for persistent storage
            tokenizer: Sanskrit tokenizer
            rule_engine: Panini rule engine
            morphological_analyzer: Morphological analyzer
            derivation_simulator: Derivation simulator
            evolution_strategy: Strategy for knowledge evolution
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.tokenizer = tokenizer
        self.rule_engine = rule_engine
        self.morphological_analyzer = morphological_analyzer
        self.derivation_simulator = derivation_simulator
        self.evolution_strategy = evolution_strategy
        
        # Knowledge management
        self.current_version: Optional[KnowledgeVersion] = None
        self.rule_confidences: Dict[str, RuleConfidence] = {}
        self.dataset_entries: Dict[str, DatasetEntry] = {}
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.learning_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize storage
        self._init_storage()
        self._load_knowledge()
        
        # Problem generator for dataset expansion
        self.problem_generator = SanskritProblemGenerator(
            tokenizer, rule_engine, morphological_analyzer, derivation_simulator
        )
        
        logger.info(f"Sanskrit Knowledge Evolution initialized with strategy: {evolution_strategy.value}")
    
    def _init_storage(self):
        """Initialize persistent storage."""
        # Create directory structure
        (self.storage_path / "versions").mkdir(exist_ok=True)
        (self.storage_path / "datasets").mkdir(exist_ok=True)
        (self.storage_path / "checkpoints").mkdir(exist_ok=True)
        (self.storage_path / "metrics").mkdir(exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.storage_path / "knowledge.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for knowledge tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    version_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    rule_count INTEGER,
                    performance_metrics TEXT,
                    parent_version TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rule_confidences (
                    rule_id TEXT PRIMARY KEY,
                    confidence_score REAL,
                    application_count INTEGER,
                    success_count INTEGER,
                    failure_count INTEGER,
                    last_updated TEXT,
                    performance_history TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_entries (
                    id TEXT PRIMARY KEY,
                    problem_data TEXT,
                    generated_by TEXT,
                    quality_score REAL,
                    validation_status TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def _load_knowledge(self):
        """Load existing knowledge from storage."""
        try:
            # Load current version
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM versions 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    version_data = {
                        'version_id': row[0],
                        'version_type': row[1],
                        'timestamp': row[2],
                        'description': row[3],
                        'rule_count': row[4],
                        'performance_metrics': json.loads(row[5]) if row[5] else {},
                        'parent_version': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    self.current_version = KnowledgeVersion.from_dict(version_data)
                    logger.info(f"Loaded knowledge version: {self.current_version.version_id}")
            
            # Load rule confidences
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM rule_confidences")
                for row in cursor.fetchall():
                    rule_confidence = RuleConfidence(
                        rule_id=row[0],
                        confidence_score=row[1],
                        application_count=row[2],
                        success_count=row[3],
                        failure_count=row[4],
                        last_updated=datetime.fromisoformat(row[5]),
                        performance_history=json.loads(row[6]) if row[6] else []
                    )
                    self.rule_confidences[row[0]] = rule_confidence
            
            logger.info(f"Loaded {len(self.rule_confidences)} rule confidence records")
            
        except Exception as e:
            logger.warning(f"Failed to load existing knowledge: {e}")
            self._create_initial_version()
    
    def _create_initial_version(self):
        """Create initial knowledge version."""
        version_id = f"v1.0.0_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        self.current_version = KnowledgeVersion(
            version_id=version_id,
            version_type=KnowledgeVersionType.MAJOR,
            timestamp=datetime.now(timezone.utc),
            description="Initial Sanskrit knowledge base",
            rule_count=len(self.rule_engine.rules),
            performance_metrics={}
        )
        
        # Initialize rule confidences
        for rule in self.rule_engine.rules:
            self.rule_confidences[rule.id] = RuleConfidence(
                rule_id=rule.id,
                confidence_score=0.5,  # Neutral initial confidence
                application_count=0,
                success_count=0,
                failure_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        self._save_version(self.current_version)
        logger.info(f"Created initial knowledge version: {version_id}")
    
    def create_checkpoint(self, 
                         description: str,
                         version_type: KnowledgeVersionType = KnowledgeVersionType.SNAPSHOT,
                         r_zero_checkpoint_path: Optional[str] = None) -> str:
        """
        Create a knowledge checkpoint.
        
        Args:
            description: Description of the checkpoint
            version_type: Type of version
            r_zero_checkpoint_path: Path to R-Zero model checkpoint
            
        Returns:
            Version ID of the created checkpoint
        """
        with self._lock:
            # Generate version ID
            timestamp = datetime.now(timezone.utc)
            version_id = f"{version_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Create version
            version = KnowledgeVersion(
                version_id=version_id,
                version_type=version_type,
                timestamp=timestamp,
                description=description,
                rule_count=len(self.rule_engine.rules),
                performance_metrics=performance_metrics,
                parent_version=self.current_version.version_id if self.current_version else None,
                metadata={
                    'r_zero_checkpoint': r_zero_checkpoint_path,
                    'evolution_strategy': self.evolution_strategy.value,
                    'rule_confidence_count': len(self.rule_confidences)
                }
            )
            
            # Save checkpoint
            self._save_version(version)
            self._save_checkpoint_data(version_id)
            
            self.current_version = version
            
            logger.info(f"Created checkpoint: {version_id}")
            return version_id
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        metrics = {}
        
        if self.performance_history:
            recent_scores = list(self.performance_history)[-100:]  # Last 100 evaluations
            metrics['avg_score'] = sum(recent_scores) / len(recent_scores)
            metrics['min_score'] = min(recent_scores)
            metrics['max_score'] = max(recent_scores)
        
        # Rule confidence metrics
        if self.rule_confidences:
            confidences = [rc.confidence_score for rc in self.rule_confidences.values()]
            metrics['avg_rule_confidence'] = sum(confidences) / len(confidences)
            metrics['min_rule_confidence'] = min(confidences)
            metrics['max_rule_confidence'] = max(confidences)
            
            # Success rate metrics
            success_rates = [rc.success_rate for rc in self.rule_confidences.values() if rc.application_count > 0]
            if success_rates:
                metrics['avg_success_rate'] = sum(success_rates) / len(success_rates)
        
        # Learning metrics
        for metric_name, values in self.learning_metrics.items():
            if values:
                metrics[f'{metric_name}_trend'] = values[-1] - values[0] if len(values) > 1 else 0.0
        
        return metrics
    
    def _save_version(self, version: KnowledgeVersion):
        """Save version to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO versions 
                (version_id, version_type, timestamp, description, rule_count, 
                 performance_metrics, parent_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id,
                version.version_type.value,
                version.timestamp.isoformat(),
                version.description,
                version.rule_count,
                json.dumps(version.performance_metrics),
                version.parent_version,
                json.dumps(version.metadata)
            ))
            conn.commit()
    
    def _save_checkpoint_data(self, version_id: str):
        """Save checkpoint data to files."""
        checkpoint_dir = self.storage_path / "checkpoints" / version_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rule confidences
        with open(checkpoint_dir / "rule_confidences.json", 'w') as f:
            confidences_data = {}
            for rule_id, confidence in self.rule_confidences.items():
                confidences_data[rule_id] = {
                    'confidence_score': confidence.confidence_score,
                    'application_count': confidence.application_count,
                    'success_count': confidence.success_count,
                    'failure_count': confidence.failure_count,
                    'last_updated': confidence.last_updated.isoformat(),
                    'performance_history': confidence.performance_history
                }
            json.dump(confidences_data, f, indent=2)
        
        # Save rule engine state
        with open(checkpoint_dir / "rule_engine.pkl", 'wb') as f:
            pickle.dump(self.rule_engine, f)
        
        # Save performance history
        with open(checkpoint_dir / "performance_history.json", 'w') as f:
            json.dump(list(self.performance_history), f)
        
        # Save learning metrics
        with open(checkpoint_dir / "learning_metrics.json", 'w') as f:
            json.dump(dict(self.learning_metrics), f)
    
    def update_rule_confidence(self, rule_id: str, success: bool, score: float = None):
        """
        Update confidence for a specific rule.
        
        Args:
            rule_id: ID of the rule
            success: Whether the rule application was successful
            score: Optional performance score
        """
        with self._lock:
            if rule_id not in self.rule_confidences:
                self.rule_confidences[rule_id] = RuleConfidence(
                    rule_id=rule_id,
                    confidence_score=0.5,
                    application_count=0,
                    success_count=0,
                    failure_count=0,
                    last_updated=datetime.now(timezone.utc)
                )
            
            self.rule_confidences[rule_id].update_performance(success, score)
            
            # Save to database
            confidence = self.rule_confidences[rule_id]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO rule_confidences
                    (rule_id, confidence_score, application_count, success_count,
                     failure_count, last_updated, performance_history)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule_id,
                    confidence.confidence_score,
                    confidence.application_count,
                    confidence.success_count,
                    confidence.failure_count,
                    confidence.last_updated.isoformat(),
                    json.dumps(confidence.performance_history)
                ))
                conn.commit()
    
    def generate_dataset_batch(self, 
                              batch_size: int = 100,
                              problem_types: Optional[List[SanskritProblemType]] = None,
                              difficulty_distribution: Optional[Dict[SanskritDifficultyLevel, float]] = None) -> List[DatasetEntry]:
        """
        Generate a batch of Sanskrit problems for training.
        
        Args:
            batch_size: Number of problems to generate
            problem_types: Types of problems to generate
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of generated dataset entries
        """
        problems = self.problem_generator.generate_problems(
            count=batch_size,
            problem_types=problem_types,
            difficulty_distribution=difficulty_distribution
        )
        
        entries = []
        for problem in problems:
            entry = DatasetEntry(
                id=f"gen_{problem.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                problem=problem,
                generated_by='self-generated',
                quality_score=0.8,  # Default quality for generated problems
                validation_status='pending',
                created_at=datetime.now(timezone.utc),
                metadata={'generation_strategy': self.evolution_strategy.value}
            )
            entries.append(entry)
            self.dataset_entries[entry.id] = entry
        
        # Save to database
        self._save_dataset_entries(entries)
        
        logger.info(f"Generated {len(entries)} dataset entries")
        return entries
    
    def _save_dataset_entries(self, entries: List[DatasetEntry]):
        """Save dataset entries to database."""
        with sqlite3.connect(self.db_path) as conn:
            for entry in entries:
                conn.execute("""
                    INSERT OR REPLACE INTO dataset_entries
                    (id, problem_data, generated_by, quality_score, 
                     validation_status, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    json.dumps(entry.problem.to_r_zero_format()),
                    entry.generated_by,
                    entry.quality_score,
                    entry.validation_status,
                    entry.created_at.isoformat(),
                    json.dumps(entry.metadata)
                ))
            conn.commit()
    
    def evaluate_and_evolve(self, 
                           predictions: List[str],
                           ground_truths: List[str],
                           problems: Optional[List[SanskritProblem]] = None) -> Dict[str, Any]:
        """
        Evaluate predictions and evolve knowledge based on results.
        
        Args:
            predictions: Model predictions
            ground_truths: Expected answers
            problems: Optional problem contexts
            
        Returns:
            Evaluation results and evolution metrics
        """
        # Evaluate using reward function
        scores = compute_score(predictions, ground_truths, problems)
        
        # Update performance history
        overall_scores = [score['overall'] for score in scores]
        self.performance_history.extend(overall_scores)
        
        # Update rule confidences based on performance
        if problems:
            for i, (score, problem) in enumerate(zip(scores, problems)):
                # Update confidences for rules that might have been involved
                self._update_rule_confidences_from_evaluation(problem, score)
        
        # Calculate evolution metrics
        avg_score = sum(overall_scores) / len(overall_scores)
        evolution_metrics = {
            'avg_score': avg_score,
            'score_distribution': {
                'min': min(overall_scores),
                'max': max(overall_scores),
                'std': (sum((s - avg_score) ** 2 for s in overall_scores) / len(overall_scores)) ** 0.5
            },
            'evaluation_count': len(predictions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Update learning metrics
        self.learning_metrics['overall_performance'].append(avg_score)
        
        # Trigger evolution if needed
        if self._should_trigger_evolution(evolution_metrics):
            self._trigger_knowledge_evolution(evolution_metrics)
        
        logger.info(f"Evaluated {len(predictions)} predictions, avg score: {avg_score:.3f}")
        return evolution_metrics
    
    def _update_rule_confidences_from_evaluation(self, problem: SanskritProblem, score: Dict[str, float]):
        """Update rule confidences based on evaluation results."""
        # This is a simplified approach - in practice, you'd need to track
        # which specific rules were involved in generating the prediction
        
        overall_score = score['overall']
        success = overall_score > 0.7  # Threshold for success
        
        # Update confidences for rules relevant to this problem type
        relevant_rules = self._get_relevant_rules_for_problem(problem)
        
        for rule_id in relevant_rules:
            self.update_rule_confidence(rule_id, success, overall_score)
    
    def _get_relevant_rules_for_problem(self, problem: SanskritProblem) -> List[str]:
        """Get rules relevant to a specific problem type."""
        # This is a simplified mapping - in practice, you'd have more sophisticated
        # rule-to-problem-type mappings
        
        rule_mappings = {
            SanskritProblemType.SANDHI_APPLICATION: ['sandhi_', 'vowel_'],
            SanskritProblemType.MORPHOLOGICAL_ANALYSIS: ['morph_', 'dhatu_'],
            SanskritProblemType.WORD_DERIVATION: ['deriv_', 'pratyaya_'],
        }
        
        relevant_prefixes = rule_mappings.get(problem.type, [])
        relevant_rules = []
        
        for rule in self.rule_engine.rules:
            if any(rule.id.startswith(prefix) for prefix in relevant_prefixes):
                relevant_rules.append(rule.id)
        
        return relevant_rules
    
    def _should_trigger_evolution(self, metrics: Dict[str, Any]) -> bool:
        """Determine if knowledge evolution should be triggered."""
        # Evolution triggers based on strategy
        if self.evolution_strategy == EvolutionStrategy.CONSERVATIVE:
            # Only evolve if performance is consistently high
            recent_scores = list(self.performance_history)[-50:] if len(self.performance_history) >= 50 else list(self.performance_history)
            return len(recent_scores) >= 20 and sum(recent_scores) / len(recent_scores) > 0.85
        
        elif self.evolution_strategy == EvolutionStrategy.BALANCED:
            # Evolve based on moderate performance improvements
            return len(self.performance_history) >= 100 and len(self.performance_history) % 100 == 0
        
        elif self.evolution_strategy == EvolutionStrategy.AGGRESSIVE:
            # Evolve frequently
            return len(self.performance_history) >= 50 and len(self.performance_history) % 50 == 0
        
        elif self.evolution_strategy == EvolutionStrategy.EXPERIMENTAL:
            # Evolve very frequently for experimentation
            return len(self.performance_history) >= 25 and len(self.performance_history) % 25 == 0
        
        return False
    
    def _trigger_knowledge_evolution(self, metrics: Dict[str, Any]):
        """Trigger knowledge evolution based on current metrics."""
        logger.info("Triggering knowledge evolution...")
        
        # Analyze rule performance and adapt
        self._adapt_rule_priorities()
        
        # Generate new training data
        new_entries = self.generate_dataset_batch(batch_size=50)
        
        # Create checkpoint
        checkpoint_id = self.create_checkpoint(
            description=f"Evolution triggered by performance metrics: avg_score={metrics['avg_score']:.3f}",
            version_type=KnowledgeVersionType.MINOR
        )
        
        logger.info(f"Knowledge evolution completed, checkpoint: {checkpoint_id}")
    
    def _adapt_rule_priorities(self):
        """Adapt rule priorities based on confidence scores."""
        # Sort rules by confidence
        rule_confidences_sorted = sorted(
            self.rule_confidences.items(),
            key=lambda x: x[1].confidence_score,
            reverse=True
        )
        
        # Adjust priorities (lower number = higher priority)
        for i, (rule_id, confidence) in enumerate(rule_confidences_sorted):
            # Find the rule in the engine
            for rule in self.rule_engine.rules:
                if rule.id == rule_id:
                    # Adjust priority based on confidence ranking
                    new_priority = i + 1
                    if rule.priority != new_priority:
                        logger.debug(f"Adjusting rule {rule_id} priority: {rule.priority} -> {new_priority}")
                        rule.priority = new_priority
                    break
        
        # Re-sort rules in the engine
        self.rule_engine.rules.sort(key=lambda r: (r.priority, r.id))
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of current knowledge state."""
        summary = {
            'current_version': self.current_version.to_dict() if self.current_version else None,
            'rule_count': len(self.rule_engine.rules),
            'rule_confidences': {
                'total': len(self.rule_confidences),
                'high_confidence': len([rc for rc in self.rule_confidences.values() if rc.confidence_score > 0.8]),
                'low_confidence': len([rc for rc in self.rule_confidences.values() if rc.confidence_score < 0.3]),
                'avg_confidence': sum(rc.confidence_score for rc in self.rule_confidences.values()) / len(self.rule_confidences) if self.rule_confidences else 0.0
            },
            'dataset_entries': len(self.dataset_entries),
            'performance_history_length': len(self.performance_history),
            'recent_avg_performance': sum(list(self.performance_history)[-10:]) / min(10, len(self.performance_history)) if self.performance_history else 0.0,
            'evolution_strategy': self.evolution_strategy.value,
            'storage_path': str(self.storage_path)
        }
        
        return summary
    
    def export_dataset(self, 
                      output_path: str,
                      format: str = 'r_zero',
                      filter_validated: bool = True) -> int:
        """
        Export dataset in specified format.
        
        Args:
            output_path: Path to save the dataset
            format: Export format ('r_zero', 'json', 'csv')
            filter_validated: Only export validated entries
            
        Returns:
            Number of entries exported
        """
        entries_to_export = []
        
        for entry in self.dataset_entries.values():
            if filter_validated and entry.validation_status != 'validated':
                continue
            entries_to_export.append(entry)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'r_zero':
            # Export in R-Zero compatible format
            r_zero_data = []
            for entry in entries_to_export:
                r_zero_data.append(entry.problem.to_r_zero_format())
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(r_zero_data, f, indent=2, ensure_ascii=False)
        
        elif format == 'json':
            # Export as JSON
            export_data = []
            for entry in entries_to_export:
                export_data.append({
                    'id': entry.id,
                    'problem': entry.problem.to_r_zero_format(),
                    'generated_by': entry.generated_by,
                    'quality_score': entry.quality_score,
                    'validation_status': entry.validation_status,
                    'created_at': entry.created_at.isoformat(),
                    'metadata': entry.metadata
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(entries_to_export)} dataset entries to {output_path}")
        return len(entries_to_export)
    
    def load_checkpoint(self, version_id: str) -> bool:
        """
        Load a specific knowledge checkpoint.
        
        Args:
            version_id: ID of the version to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / version_id
            
            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint {version_id} not found")
                return False
            
            # Load rule confidences
            with open(checkpoint_dir / "rule_confidences.json", 'r') as f:
                confidences_data = json.load(f)
                
                self.rule_confidences.clear()
                for rule_id, data in confidences_data.items():
                    self.rule_confidences[rule_id] = RuleConfidence(
                        rule_id=rule_id,
                        confidence_score=data['confidence_score'],
                        application_count=data['application_count'],
                        success_count=data['success_count'],
                        failure_count=data['failure_count'],
                        last_updated=datetime.fromisoformat(data['last_updated']),
                        performance_history=data['performance_history']
                    )
            
            # Load rule engine state
            with open(checkpoint_dir / "rule_engine.pkl", 'rb') as f:
                self.rule_engine = pickle.load(f)
            
            # Load performance history
            with open(checkpoint_dir / "performance_history.json", 'r') as f:
                history_data = json.load(f)
                self.performance_history.clear()
                self.performance_history.extend(history_data)
            
            # Load learning metrics
            with open(checkpoint_dir / "learning_metrics.json", 'r') as f:
                metrics_data = json.load(f)
                self.learning_metrics.clear()
                for key, values in metrics_data.items():
                    self.learning_metrics[key].extend(values)
            
            # Update current version
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM versions WHERE version_id = ?
                """, (version_id,))
                row = cursor.fetchone()
                
                if row:
                    version_data = {
                        'version_id': row[0],
                        'version_type': row[1],
                        'timestamp': row[2],
                        'description': row[3],
                        'rule_count': row[4],
                        'performance_metrics': json.loads(row[5]) if row[5] else {},
                        'parent_version': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    self.current_version = KnowledgeVersion.from_dict(version_data)
            
            logger.info(f"Successfully loaded checkpoint: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {version_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version_id FROM versions 
                    ORDER BY timestamp DESC
                """)
                all_versions = [row[0] for row in cursor.fetchall()]
            
            if len(all_versions) <= keep_count:
                return
            
            versions_to_delete = all_versions[keep_count:]
            
            for version_id in versions_to_delete:
                # Delete checkpoint directory
                checkpoint_dir = self.storage_path / "checkpoints" / version_id
                if checkpoint_dir.exists():
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                
                # Delete from database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))
                    conn.commit()
            
            logger.info(f"Cleaned up {len(versions_to_delete)} old checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")