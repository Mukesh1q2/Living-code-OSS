"""
Tests for Sanskrit Knowledge Evolution System.

This module tests the persistent knowledge evolution capabilities including:
- R-Zero model checkpointing integration
- Sanskrit-specific dataset generation and curation
- Incremental learning for new Sanskrit constructions
- Rule confidence adaptation based on performance metrics
- Sanskrit corpus expansion through self-generated examples
- Knowledge persistence and retrieval accuracy
"""

import pytest
import unittest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sanskrit_rewrite_engine.knowledge_evolution import (
    SanskritKnowledgeEvolution,
    KnowledgeVersion,
    KnowledgeVersionType,
    EvolutionStrategy,
    RuleConfidence,
    DatasetEntry
)
from sanskrit_rewrite_engine.r_zero_integration import (
    SanskritProblem,
    SanskritProblemType,
    SanskritDifficultyLevel
)
from sanskrit_rewrite_engine.tokenizer import SanskritTokenizer
from sanskrit_rewrite_engine.panini_engine import PaniniRuleEngine, SutraRule
from sanskrit_rewrite_engine.morphological_analyzer import SanskritMorphologicalAnalyzer
from sanskrit_rewrite_engine.derivation_simulator import ShabdaPrakriyaSimulator


class TestKnowledgeEvolution(unittest.TestCase):
    """Test suite for Sanskrit Knowledge Evolution system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "knowledge_test"
        
        # Create mock components
        self.tokenizer = Mock(spec=SanskritTokenizer)
        self.rule_engine = Mock(spec=PaniniRuleEngine)
        self.morphological_analyzer = Mock(spec=SanskritMorphologicalAnalyzer)
        self.derivation_simulator = Mock(spec=ShabdaPrakriyaSimulator)
        
        # Mock rules for rule engine
        mock_rules = [
            Mock(id="sandhi_rule_1", priority=1),
            Mock(id="sandhi_rule_2", priority=2),
            Mock(id="morph_rule_1", priority=3),
            Mock(id="deriv_rule_1", priority=4)
        ]
        self.rule_engine.rules = mock_rules
        
        # Create knowledge evolution system
        self.knowledge_system = SanskritKnowledgeEvolution(
            storage_path=str(self.storage_path),
            tokenizer=self.tokenizer,
            rule_engine=self.rule_engine,
            morphological_analyzer=self.morphological_analyzer,
            derivation_simulator=self.derivation_simulator,
            evolution_strategy=EvolutionStrategy.BALANCED
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test knowledge evolution system initialization."""
        # Check storage structure
        self.assertTrue(self.storage_path.exists())
        self.assertTrue((self.storage_path / "versions").exists())
        self.assertTrue((self.storage_path / "datasets").exists())
        self.assertTrue((self.storage_path / "checkpoints").exists())
        self.assertTrue((self.storage_path / "metrics").exists())
        self.assertTrue((self.storage_path / "knowledge.db").exists())
        
        # Check initial version creation
        self.assertIsNotNone(self.knowledge_system.current_version)
        self.assertEqual(self.knowledge_system.current_version.version_type, KnowledgeVersionType.MAJOR)
        
        # Check rule confidences initialization
        self.assertEqual(len(self.knowledge_system.rule_confidences), 4)
        for rule_id in ["sandhi_rule_1", "sandhi_rule_2", "morph_rule_1", "deriv_rule_1"]:
            self.assertIn(rule_id, self.knowledge_system.rule_confidences)
            self.assertEqual(self.knowledge_system.rule_confidences[rule_id].confidence_score, 0.5)
    
    def test_database_initialization(self):
        """Test SQLite database initialization."""
        db_path = self.storage_path / "knowledge.db"
        
        with sqlite3.connect(db_path) as conn:
            # Check tables exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('versions', 'rule_confidences', 'dataset_entries')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('versions', tables)
            self.assertIn('rule_confidences', tables)
            self.assertIn('dataset_entries', tables)
    
    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        # Create checkpoint
        checkpoint_id = self.knowledge_system.create_checkpoint(
            description="Test checkpoint",
            version_type=KnowledgeVersionType.MINOR
        )
        
        # Verify checkpoint was created
        self.assertIsNotNone(checkpoint_id)
        self.assertTrue(checkpoint_id.startswith("minor_"))
        
        # Check checkpoint directory exists
        checkpoint_dir = self.storage_path / "checkpoints" / checkpoint_id
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue((checkpoint_dir / "rule_confidences.json").exists())
        self.assertTrue((checkpoint_dir / "rule_engine.pkl").exists())
        self.assertTrue((checkpoint_dir / "performance_history.json").exists())
        self.assertTrue((checkpoint_dir / "learning_metrics.json").exists())
        
        # Check database entry
        with sqlite3.connect(self.storage_path / "knowledge.db") as conn:
            cursor = conn.execute("SELECT * FROM versions WHERE version_id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], checkpoint_id)  # version_id
            self.assertEqual(row[1], "minor")  # version_type
    
    def test_rule_confidence_update(self):
        """Test rule confidence updating."""
        rule_id = "sandhi_rule_1"
        
        # Initial confidence
        initial_confidence = self.knowledge_system.rule_confidences[rule_id].confidence_score
        self.assertEqual(initial_confidence, 0.5)
        
        # Update with success
        self.knowledge_system.update_rule_confidence(rule_id, success=True, score=0.9)
        
        # Check updated confidence
        updated_confidence = self.knowledge_system.rule_confidences[rule_id]
        self.assertEqual(updated_confidence.application_count, 1)
        self.assertEqual(updated_confidence.success_count, 1)
        self.assertEqual(updated_confidence.failure_count, 0)
        self.assertEqual(updated_confidence.success_rate, 1.0)
        self.assertEqual(updated_confidence.confidence_score, 0.9)
        
        # Update with failure
        self.knowledge_system.update_rule_confidence(rule_id, success=False, score=0.3)
        
        # Check updated confidence
        updated_confidence = self.knowledge_system.rule_confidences[rule_id]
        self.assertEqual(updated_confidence.application_count, 2)
        self.assertEqual(updated_confidence.success_count, 1)
        self.assertEqual(updated_confidence.failure_count, 1)
        self.assertEqual(updated_confidence.success_rate, 0.5)
        # Confidence should be average of recent scores
        self.assertEqual(updated_confidence.confidence_score, 0.6)  # (0.9 + 0.3) / 2
    
    def test_dataset_generation(self):
        """Test dataset generation."""
        # Mock problem generator
        mock_problem = SanskritProblem(
            id="test_problem_1",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e",
            explanation="Test sandhi"
        )
        
        with patch.object(self.knowledge_system.problem_generator, 'generate_problems') as mock_generate:
            mock_generate.return_value = [mock_problem]
            
            # Generate dataset batch
            entries = self.knowledge_system.generate_dataset_batch(batch_size=1)
            
            # Verify generation
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].problem.id, "test_problem_1")
            self.assertEqual(entries[0].generated_by, "self-generated")
            self.assertEqual(entries[0].validation_status, "pending")
            
            # Check database storage
            with sqlite3.connect(self.storage_path / "knowledge.db") as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM dataset_entries")
                count = cursor.fetchone()[0]
                self.assertEqual(count, 1)
    
    def test_evaluate_and_evolve(self):
        """Test evaluation and evolution process."""
        # Create test data
        predictions = ["e", "gacchati"]
        ground_truths = ["e", "gacchati"]
        problems = [
            SanskritProblem(
                id="test_1",
                type=SanskritProblemType.SANDHI_APPLICATION,
                difficulty=SanskritDifficultyLevel.BEGINNER,
                input_text="a + i",
                expected_output="e"
            ),
            SanskritProblem(
                id="test_2",
                type=SanskritProblemType.MORPHOLOGICAL_ANALYSIS,
                difficulty=SanskritDifficultyLevel.INTERMEDIATE,
                input_text="gacchati",
                expected_output="gacchati"
            )
        ]
        
        # Mock compute_score
        with patch('sanskrit_rewrite_engine.knowledge_evolution.compute_score') as mock_compute:
            mock_compute.return_value = [
                {'overall': 0.9, 'format': 1.0, 'accuracy': 0.8},
                {'overall': 0.8, 'format': 0.9, 'accuracy': 0.7}
            ]
            
            # Evaluate and evolve
            metrics = self.knowledge_system.evaluate_and_evolve(predictions, ground_truths, problems)
            
            # Check metrics
            self.assertIn('avg_score', metrics)
            self.assertIn('score_distribution', metrics)
            self.assertIn('evaluation_count', metrics)
            self.assertEqual(metrics['evaluation_count'], 2)
            self.assertEqual(metrics['avg_score'], 0.85)  # (0.9 + 0.8) / 2
            
            # Check performance history updated
            self.assertEqual(len(self.knowledge_system.performance_history), 2)
            self.assertEqual(list(self.knowledge_system.performance_history), [0.9, 0.8])
    
    def test_evolution_strategies(self):
        """Test different evolution strategies."""
        strategies = [
            EvolutionStrategy.CONSERVATIVE,
            EvolutionStrategy.BALANCED,
            EvolutionStrategy.AGGRESSIVE,
            EvolutionStrategy.EXPERIMENTAL
        ]
        
        for strategy in strategies:
            # Create system with specific strategy
            system = SanskritKnowledgeEvolution(
                storage_path=str(self.storage_path / f"test_{strategy.value}"),
                tokenizer=self.tokenizer,
                rule_engine=self.rule_engine,
                morphological_analyzer=self.morphological_analyzer,
                derivation_simulator=self.derivation_simulator,
                evolution_strategy=strategy
            )
            
            # Test evolution trigger conditions
            # Add some performance history
            system.performance_history.extend([0.8] * 100)
            
            # Check if evolution should trigger based on strategy
            metrics = {'avg_score': 0.8}
            should_trigger = system._should_trigger_evolution(metrics)
            
            # Verify strategy-specific behavior
            if strategy == EvolutionStrategy.CONSERVATIVE:
                # Should not trigger with moderate performance
                self.assertFalse(should_trigger)
            elif strategy == EvolutionStrategy.EXPERIMENTAL:
                # Should trigger frequently
                system.performance_history.clear()
                system.performance_history.extend([0.8] * 25)
                self.assertTrue(system._should_trigger_evolution(metrics))
    
    def test_rule_priority_adaptation(self):
        """Test rule priority adaptation based on confidence."""
        # Set different confidence scores
        self.knowledge_system.rule_confidences["sandhi_rule_1"].confidence_score = 0.9
        self.knowledge_system.rule_confidences["sandhi_rule_2"].confidence_score = 0.7
        self.knowledge_system.rule_confidences["morph_rule_1"].confidence_score = 0.8
        self.knowledge_system.rule_confidences["deriv_rule_1"].confidence_score = 0.6
        
        # Adapt priorities
        self.knowledge_system._adapt_rule_priorities()
        
        # Check that rules are reordered by confidence
        # Higher confidence should get lower priority numbers (higher priority)
        rule_priorities = {rule.id: rule.priority for rule in self.rule_engine.rules}
        
        # sandhi_rule_1 (0.9 confidence) should have priority 1
        # morph_rule_1 (0.8 confidence) should have priority 2
        # sandhi_rule_2 (0.7 confidence) should have priority 3
        # deriv_rule_1 (0.6 confidence) should have priority 4
        self.assertEqual(rule_priorities["sandhi_rule_1"], 1)
        self.assertEqual(rule_priorities["morph_rule_1"], 2)
        self.assertEqual(rule_priorities["sandhi_rule_2"], 3)
        self.assertEqual(rule_priorities["deriv_rule_1"], 4)
    
    def test_knowledge_summary(self):
        """Test knowledge summary generation."""
        summary = self.knowledge_system.get_knowledge_summary()
        
        # Check summary structure
        self.assertIn('current_version', summary)
        self.assertIn('rule_count', summary)
        self.assertIn('rule_confidences', summary)
        self.assertIn('dataset_entries', summary)
        self.assertIn('performance_history_length', summary)
        self.assertIn('evolution_strategy', summary)
        
        # Check values
        self.assertEqual(summary['rule_count'], 4)
        self.assertEqual(summary['rule_confidences']['total'], 4)
        self.assertEqual(summary['evolution_strategy'], 'balanced')
        self.assertEqual(summary['dataset_entries'], 0)  # No entries generated yet
    
    def test_dataset_export(self):
        """Test dataset export functionality."""
        # Create test dataset entry
        test_problem = SanskritProblem(
            id="export_test",
            type=SanskritProblemType.SANDHI_APPLICATION,
            difficulty=SanskritDifficultyLevel.BEGINNER,
            input_text="a + i",
            expected_output="e"
        )
        
        entry = DatasetEntry(
            id="export_entry_1",
            problem=test_problem,
            generated_by="test",
            quality_score=0.9,
            validation_status="validated",
            created_at=datetime.now(timezone.utc)
        )
        
        self.knowledge_system.dataset_entries[entry.id] = entry
        self.knowledge_system._save_dataset_entries([entry])
        
        # Export dataset
        export_path = self.storage_path / "exported_dataset.json"
        count = self.knowledge_system.export_dataset(
            output_path=str(export_path),
            format='r_zero',
            filter_validated=True
        )
        
        # Verify export
        self.assertEqual(count, 1)
        self.assertTrue(export_path.exists())
        
        # Check exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(len(exported_data), 1)
        self.assertEqual(exported_data[0]['answer'], 'e')
        self.assertIn('problem', exported_data[0])
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading functionality."""
        # Create initial checkpoint
        checkpoint_id = self.knowledge_system.create_checkpoint(
            description="Test checkpoint for loading",
            version_type=KnowledgeVersionType.SNAPSHOT
        )
        
        # Modify some state
        original_confidence = self.knowledge_system.rule_confidences["sandhi_rule_1"].confidence_score
        self.knowledge_system.update_rule_confidence("sandhi_rule_1", success=True, score=0.95)
        modified_confidence = self.knowledge_system.rule_confidences["sandhi_rule_1"].confidence_score
        
        # Verify state changed
        self.assertNotEqual(original_confidence, modified_confidence)
        
        # Load checkpoint
        success = self.knowledge_system.load_checkpoint(checkpoint_id)
        self.assertTrue(success)
        
        # Verify state restored
        restored_confidence = self.knowledge_system.rule_confidences["sandhi_rule_1"].confidence_score
        self.assertEqual(restored_confidence, original_confidence)
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup functionality."""
        # Create multiple checkpoints
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = self.knowledge_system.create_checkpoint(
                description=f"Test checkpoint {i}",
                version_type=KnowledgeVersionType.SNAPSHOT
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Verify all checkpoints exist
        for checkpoint_id in checkpoint_ids:
            checkpoint_dir = self.storage_path / "checkpoints" / checkpoint_id
            self.assertTrue(checkpoint_dir.exists())
        
        # Cleanup keeping only 3
        self.knowledge_system.cleanup_old_checkpoints(keep_count=3)
        
        # Verify only 3 remain
        with sqlite3.connect(self.storage_path / "knowledge.db") as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM versions")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 4)  # 3 kept + 1 initial version
        
        # Verify oldest checkpoints were deleted
        remaining_dirs = list((self.storage_path / "checkpoints").iterdir())
        self.assertEqual(len(remaining_dirs), 3)  # Only 3 checkpoint dirs remain
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Add performance history
        scores = [0.8, 0.9, 0.7, 0.85, 0.92]
        self.knowledge_system.performance_history.extend(scores)
        
        # Update rule confidences
        self.knowledge_system.update_rule_confidence("sandhi_rule_1", success=True, score=0.9)
        self.knowledge_system.update_rule_confidence("sandhi_rule_2", success=False, score=0.4)
        
        # Calculate metrics
        metrics = self.knowledge_system._calculate_performance_metrics()
        
        # Verify metrics
        self.assertIn('avg_score', metrics)
        self.assertIn('min_score', metrics)
        self.assertIn('max_score', metrics)
        self.assertIn('avg_rule_confidence', metrics)
        
        self.assertEqual(metrics['avg_score'], sum(scores) / len(scores))
        self.assertEqual(metrics['min_score'], min(scores))
        self.assertEqual(metrics['max_score'], max(scores))
    
    def test_incremental_learning_simulation(self):
        """Test incremental learning simulation."""
        # Simulate learning over time
        initial_performance = []
        
        # Generate initial performance data
        for i in range(50):
            predictions = ["e"] * 10
            ground_truths = ["e"] * 10
            
            # Mock gradually improving performance
            base_score = 0.6 + (i * 0.008)  # Gradual improvement
            mock_scores = [{'overall': min(1.0, base_score + (j * 0.01))} for j in range(10)]
            
            with patch('sanskrit_rewrite_engine.knowledge_evolution.compute_score') as mock_compute:
                mock_compute.return_value = mock_scores
                
                metrics = self.knowledge_system.evaluate_and_evolve(predictions, ground_truths)
                initial_performance.append(metrics['avg_score'])
        
        # Check that performance improved over time
        early_avg = sum(initial_performance[:10]) / 10
        late_avg = sum(initial_performance[-10:]) / 10
        
        self.assertGreater(late_avg, early_avg, "Performance should improve over time")
        
        # Check that learning metrics were recorded
        self.assertIn('overall_performance', self.knowledge_system.learning_metrics)
        self.assertEqual(len(self.knowledge_system.learning_metrics['overall_performance']), 50)


class TestRuleConfidence(unittest.TestCase):
    """Test suite for RuleConfidence class."""
    
    def test_rule_confidence_initialization(self):
        """Test RuleConfidence initialization."""
        confidence = RuleConfidence(
            rule_id="test_rule",
            confidence_score=0.7,
            application_count=10,
            success_count=7,
            failure_count=3,
            last_updated=datetime.now(timezone.utc)
        )
        
        self.assertEqual(confidence.rule_id, "test_rule")
        self.assertEqual(confidence.confidence_score, 0.7)
        self.assertEqual(confidence.success_rate, 0.7)  # 7/10
    
    def test_performance_update(self):
        """Test performance update functionality."""
        confidence = RuleConfidence(
            rule_id="test_rule",
            confidence_score=0.5,
            application_count=0,
            success_count=0,
            failure_count=0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Update with success
        confidence.update_performance(success=True, score=0.8)
        
        self.assertEqual(confidence.application_count, 1)
        self.assertEqual(confidence.success_count, 1)
        self.assertEqual(confidence.failure_count, 0)
        self.assertEqual(confidence.success_rate, 1.0)
        self.assertEqual(confidence.confidence_score, 0.8)
        
        # Update with failure
        confidence.update_performance(success=False, score=0.3)
        
        self.assertEqual(confidence.application_count, 2)
        self.assertEqual(confidence.success_count, 1)
        self.assertEqual(confidence.failure_count, 1)
        self.assertEqual(confidence.success_rate, 0.5)
        self.assertEqual(confidence.confidence_score, 0.55)  # (0.8 + 0.3) / 2


class TestKnowledgeVersion(unittest.TestCase):
    """Test suite for KnowledgeVersion class."""
    
    def test_version_serialization(self):
        """Test version serialization and deserialization."""
        version = KnowledgeVersion(
            version_id="test_v1.0.0",
            version_type=KnowledgeVersionType.MAJOR,
            timestamp=datetime.now(timezone.utc),
            description="Test version",
            rule_count=100,
            performance_metrics={'avg_score': 0.85},
            parent_version="parent_v0.9.0",
            metadata={'test_key': 'test_value'}
        )
        
        # Serialize to dict
        version_dict = version.to_dict()
        
        # Deserialize from dict
        restored_version = KnowledgeVersion.from_dict(version_dict)
        
        # Verify restoration
        self.assertEqual(restored_version.version_id, version.version_id)
        self.assertEqual(restored_version.version_type, version.version_type)
        self.assertEqual(restored_version.description, version.description)
        self.assertEqual(restored_version.rule_count, version.rule_count)
        self.assertEqual(restored_version.performance_metrics, version.performance_metrics)
        self.assertEqual(restored_version.parent_version, version.parent_version)
        self.assertEqual(restored_version.metadata, version.metadata)


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)