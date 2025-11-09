"""
Comprehensive integration tests for continual learning and knowledge preservation.
Task T033: Integration tests for knowledge preservation across training episodes.

This test file validates the integration between EWC, progressive networks,
domain adaptation, and 17Lands data integration for the MTG AI system.

Tests follow Red-Green-Refactor approach and will initially FAIL.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import sqlite3
from typing import Dict, List, Any, Tuple
import time

# Import RL components (these will be implemented)
from src.rl.continual_learning_manager import ContinualLearningManager
from src.rl.elastic_weight_consolidation import ElasticWeightConsolidation
from src.rl.progressive_networks import ProgressiveNetworks
from src.rl.domain_adaptation import DomainAdaptation
from src.rl.evaluation_metrics import EvaluationMetrics
from src.rl.model_versioning import ModelVersioning
from src.rl.training_monitor import TrainingMonitor
from src.rl.data_pipeline import SeventeenLandsDataPipeline
from src.core.mtga import GameStateManager
from src.mtg_ai.mtg_transformer_encoder import MTGTransformerEncoder
from src.mtg_ai.mtg_decision_head import MTGDecisionHead


class TestContinualLearningIntegration:
    """Integration tests for continual learning system."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_workspace):
        """Mock configuration for testing."""
        return {
            "workspace": temp_workspace,
            "model_config": {
                "state_dim": 282,
                "action_dim": 16,
                "hidden_dim": 128,
                "num_layers": 6,
                "num_heads": 8
            },
            "continual_learning": {
                "ewc_lambda": 1000.0,
                "progressive_columns": 3,
                "domain_adaptation_lr": 1e-4,
                "forgetting_threshold": 0.15,
                "performance_threshold": 0.05
            },
            "data_config": {
                "batch_size": 32,
                "sequence_length": 10,
                "validation_split": 0.2
            }
        }

    @pytest.fixture
    def mock_datasets(self, temp_workspace):
        """Create mock datasets for different MTG sets."""
        datasets = {}

        # Mock dataset structure for different sets
        for set_name in ["DMU", "NEO", "SNC", "ONE"]:
            dataset_path = Path(temp_workspace) / f"dataset_{set_name.lower()}.json"

            # Create mock training data
            mock_data = {
                "set_name": set_name,
                "games": [
                    {
                        "game_id": f"game_{set_name}_{i}",
                        "tensor_data": np.random.randn(282).tolist(),
                        "action_label": np.random.randint(0, 16).tolist(),
                        "outcome_weight": np.random.random(),
                        "game_outcome": bool(np.random.randint(0, 2)),
                        "decision_type": "combat",
                        "strategic_context": {"turn": i, "phase": "main"}
                    }
                    for i in range(20)  # Small dataset for testing
                ]
            }

            with open(dataset_path, 'w') as f:
                json.dump(mock_data, f)

            datasets[set_name] = dataset_path

        return datasets

    @pytest.fixture
    def continual_manager(self, mock_config):
        """Initialize continual learning manager."""
        # This will fail initially - class doesn't exist
        return ContinualLearningManager(mock_config)

    def test_initialization_continual_learning_system(self, continual_manager, mock_config):
        """
        T033-001: Test initialization of complete continual learning system.

        Should validate that all components integrate properly during startup.
        """
        # Test that manager initializes with all components
        assert continual_manager is not None

        # Check EWC component
        assert hasattr(continual_manager, 'ewc')
        assert continual_manager.ewc is not None
        assert continual_manager.ewc.lambda_param == mock_config["continual_learning"]["ewc_lambda"]

        # Check progressive networks
        assert hasattr(continual_manager, 'progressive_networks')
        assert continual_manager.progressive_networks is not None
        assert continual_manager.progressive_networks.num_columns == mock_config["continual_learning"]["progressive_columns"]

        # Check domain adaptation
        assert hasattr(continual_manager, 'domain_adapter')
        assert continual_manager.domain_adapter is not None

        # Check evaluation metrics
        assert hasattr(continual_manager, 'evaluator')
        assert continual_manager.evaluator is not None

        # Check model versioning
        assert hasattr(continual_manager, 'version_manager')
        assert continual_manager.version_manager is not None

    def test_knowledge_preservation_across_sets(self, continual_manager, mock_datasets):
        """
        T033-002: Test knowledge preservation when training on multiple MTG sets.

        Validates that performance on previous sets doesn't degrade catastrophically
        when training on new sets.
        """
        # Initial performance baseline
        baseline_performances = {}

        # Train on each set sequentially and measure performance
        for set_name, dataset_path in mock_datasets.items():
            # Load dataset
            dataset = continual_manager.load_dataset(dataset_path)

            # Evaluate performance before training on this set
            previous_perf = continual_manager.evaluate_all_previous_sets()
            if set_name == "DMU":  # First set
                baseline_performances[set_name] = previous_perf.get("overall", 0.5)
            else:
                # Check catastrophic forgetting
                for prev_set, prev_perf in previous_perf.items():
                    baseline = baseline_performances[prev_set]
                    forgetting = baseline - prev_perf
                    assert forgetting < continual_manager.forgetting_threshold, \
                        f"Catastrophic forgetting detected on {prev_set}: {forgetting:.3f}"

            # Train on current set
            training_result = continual_manager.train_on_set(
                set_name=set_name,
                dataset=dataset,
                preserve_knowledge=True
            )

            # Verify training succeeded
            assert training_result["success"]
            assert training_result["final_loss"] < training_result["initial_loss"]

            # Update baseline for this set
            current_perf = continual_manager.evaluate_set(set_name)
            baseline_performances[set_name] = current_perf

        # Final validation: ensure all sets maintain acceptable performance
        final_performances = continual_manager.evaluate_all_previous_sets()
        for set_name in mock_datasets.keys():
            baseline = baseline_performances[set_name]
            final_perf = final_performances[set_name]
            performance_drop = baseline - final_perf
            assert performance_drop < continual_manager.performance_threshold, \
                f"Performance degradation on {set_name}: {performance_drop:.3f}"

    def test_ewc_progressive_networks_integration(self, continual_manager, mock_datasets):
        """
        T033-003: Test integration between EWC and Progressive Networks.

        Validates that both mechanisms work together to prevent forgetting
        while enabling learning of new concepts.
        """
        # Get first dataset
        first_dataset = continual_manager.load_dataset(mock_datasets["DMU"])

        # Train initial model
        initial_result = continual_manager.train_on_set(
            set_name="DMU",
            dataset=first_dataset,
            use_ewc=False,
            use_progressive=False
        )

        # Extract Fisher information for EWC
        fisher_info = continual_manager.ewc.compute_fisher_information(
            model=continual_manager.current_model,
            dataset=first_dataset
        )

        # Train second set with both EWC and progressive networks
        second_dataset = continual_manager.load_dataset(mock_datasets["NEO"])

        progressive_result = continual_manager.train_on_set(
            set_name="NEO",
            dataset=second_dataset,
            use_ewc=True,
            use_progressive=True,
            fisher_info=fisher_info
        )

        # Validate progressive network created new column
        assert continual_manager.progressive_networks.current_columns > 1

        # Validate EWC constraints were applied
        assert continual_manager.ewc.constraint_applied

        # Check that both old and new knowledge is preserved
        dmu_performance = continual_manager.evaluate_set("DMU")
        neo_performance = continual_manager.evaluate_set("NEO")

        assert dmu_performance > 0.7  # Old knowledge preserved
        assert neo_performance > 0.6  # New knowledge learned

    def test_domain_adaptation_with_17lands_data(self, continual_manager):
        """
        T033-004: Test domain adaptation with real 17Lands data integration.

        Validates that the model can adapt to different metagame environments
        using 17Lands statistics and player data.
        """
        # Mock 17Lands data pipeline
        data_pipeline = SeventeenLandsDataPipeline()

        # Load data from different formats/metagames
        formats = ["PremierDraft", "TraditionalDraft", "Sealed"]
        adaptation_results = {}

        for format_name in formats:
            # Get format-specific data
            format_data = data_pipeline.load_format_data(format_name)

            # Adapt model to this format
            adaptation_result = continual_manager.adapt_to_domain(
                domain_data=format_data,
                domain_name=format_name,
                adaptation_method="feature_alignment"
            )

            adaptation_results[format_name] = adaptation_result

            # Validate adaptation succeeded
            assert adaptation_result["success"]
            assert adaptation_result["domain_accuracy"] > 0.5

            # Check that domain-specific features were learned
            assert adaptation_result["domain_features_learned"] > 0

        # Test cross-domain performance
        cross_domain_scores = continual_manager.evaluate_cross_domain(
            source_domain="PremierDraft",
            target_domain="TraditionalDraft"
        )

        # Should maintain reasonable performance across domains
        assert cross_domain_scores["target_performance"] > 0.4
        assert cross_domain_scores["transfer_ratio"] > 0.7

    def test_model_versioning_and_rollback(self, continual_manager, mock_datasets):
        """
        T033-005: Test model versioning and rollback capabilities.

        Validates that models can be versioned, compared, and rolled back
        when performance degradation is detected.
        """
        # Create initial version
        initial_version = continual_manager.version_manager.create_version(
            model=continual_manager.current_model,
            metadata={"set": "DMU", "epoch": 0, "performance": 0.5}
        )

        # Train and create multiple versions
        versions = [initial_version]
        for i, (set_name, dataset_path) in enumerate(list(mock_datasets.items())[:3]):
            dataset = continual_manager.load_dataset(dataset_path)
            training_result = continual_manager.train_on_set(set_name, dataset)

            version = continual_manager.version_manager.create_version(
                model=continual_manager.current_model,
                metadata={
                    "set": set_name,
                    "epoch": i + 1,
                    "performance": training_result["final_performance"]
                }
            )
            versions.append(version)

        # Test version comparison
        comparison = continual_manager.version_manager.compare_versions(
            version_a=versions[0],
            version_b=versions[-1]
        )

        assert "performance_diff" in comparison
        assert "parameter_diff" in comparison

        # Test rollback when performance degrades
        # Simulate performance degradation
        degraded_performance = 0.3  # Poor performance

        # Trigger rollback logic
        rollback_result = continual_manager.version_manager.rollback_if_degraded(
            current_performance=degraded_performance,
            threshold=0.4,
            target_version=versions[1]  # Rollback to second version
        )

        assert rollback_result["rollback_triggered"]
        assert rollback_result["target_version"] == versions[1].version_id

        # Verify model was restored
        restored_performance = continual_manager.evaluate_set("DMU")
        assert restored_performance > degraded_performance

    def test_catastrophic_forgetting_detection(self, continual_manager, mock_datasets):
        """
        T033-006: Test catastrophic forgetting detection and mitigation.

        Validates that the system can detect when catastrophic forgetting occurs
        and take appropriate mitigation actions.
        """
        # Establish baseline performance
        baseline_dataset = continual_manager.load_dataset(mock_datasets["DMU"])
        continual_manager.train_on_set("DMU", baseline_dataset)
        baseline_performance = continual_manager.evaluate_set("DMU")

        # Simulate training that causes forgetting
        forgetting_dataset = continual_manager.load_dataset(mock_datasets["NEO"])

        # Monitor for forgetting during training
        forgetting_monitor = continual_manager.monitor_forgetting_during_training(
            dataset=forgetting_dataset,
            baseline_set="DMU",
            baseline_performance=baseline_performance,
            check_interval=5  # Check every 5 batches
        )

        # Should detect forgetting
        assert forgetting_monitor["forgetting_detected"]
        assert forgetting_monitor["max_forgetting"] > continual_manager.forgetting_threshold

        # Test mitigation strategies
        mitigation_result = continual_manager.mitigate_forgetting(
            detected_forgetting=forgetting_monitor,
            strategy="ewc_replay"  # Use EWC with replay
        )

        assert mitigation_result["mitigation_applied"]

        # Verify performance recovery
        recovered_performance = continual_manager.evaluate_set("DMU")
        performance_recovery = recovered_performance - forgetting_monitor["min_performance"]
        assert performance_recovery > 0.1  # Recovered at least 10% performance

    def test_graceful_degradation_under_resource_constraints(self, continual_manager, mock_datasets):
        """
        T033-007: Test graceful degradation when system resources are constrained.

        Validates that the system degrades gracefully when memory or computation
        resources are limited.
        """
        # Simulate memory constraints
        memory_constraints = {
            "max_model_size_mb": 50,
            "max_memory_usage_mb": 200,
            "max_batch_size": 8
        }

        # Test training under memory constraints
        constrained_training = continual_manager.train_under_constraints(
            dataset=continual_manager.load_dataset(mock_datasets["DMU"]),
            constraints=memory_constraints
        )

        assert constrained_training["success"]
        assert constrained_training["memory_usage_mb"] <= memory_constraints["max_memory_usage_mb"]
        assert constrained_training["model_size_mb"] <= memory_constraints["max_model_size_mb"]

        # Test that performance is reasonable despite constraints
        constrained_performance = continual_manager.evaluate_set("DMU")
        assert constrained_performance > 0.4  # Still reasonable performance

        # Test adaptive component selection
        adaptation = continual_manager.adapt_components_to_constraints(
            available_memory_mb=100,
            target_performance=0.5
        )

        # Should disable some components to meet constraints
        assert adaptation["components_disabled"] > 0
        assert adaptation["estimated_memory_usage"] <= 100

    def test_end_to_end_continual_learning_workflow(self, continual_manager, mock_datasets):
        """
        T033-008: Test complete end-to-end continual learning workflow.

        Validates the entire pipeline from data loading through training,
        evaluation, versioning, and deployment.
        """
        workflow_results = {}

        # Step 1: Initialize with first set
        initial_dataset = continual_manager.load_dataset(mock_datasets["DMU"])
        initialization = continual_manager.initialize_with_dataset(
            dataset=initial_dataset,
            set_name="DMU"
        )
        workflow_results["initialization"] = initialization
        assert initialization["success"]

        # Step 2: Sequential training on multiple sets
        for set_name, dataset_path in list(mock_datasets.items())[1:]:
            dataset = continual_manager.load_dataset(dataset_path)

            # Pre-flight check
            preflight = continual_manager.preflight_training_check(
                dataset=dataset,
                set_name=set_name
            )
            assert preflight["can_train"]

            # Train with continual learning
            training_result = continual_manager.train_on_set(
                set_name=set_name,
                dataset=dataset,
                validate_continuously=True
            )
            workflow_results[f"training_{set_name}"] = training_result
            assert training_result["success"]

            # Post-training validation
            validation = continual_manager.post_training_validation(
                set_name=set_name,
                check_forgetting=True,
                check_performance=True
            )
            workflow_results[f"validation_{set_name}"] = validation
            assert validation["passed"]

        # Step 3: Final comprehensive evaluation
        final_evaluation = continual_manager.comprehensive_evaluation(
            test_sets=list(mock_datasets.keys())
        )
        workflow_results["final_evaluation"] = final_evaluation

        # Should maintain good performance across all sets
        assert final_evaluation["average_performance"] > 0.6
        assert final_evaluation["worst_set_performance"] > 0.4
        assert final_evaluation["forgetting_metrics"]["max_forgetting"] < 0.2

        # Step 4: Prepare for deployment
        deployment_prep = continual_manager.prepare_for_deployment(
            include_checkpoints=True,
            include_metadata=True,
            compression_level=1
        )
        workflow_results["deployment_prep"] = deployment_prep
        assert deployment_prep["ready_for_deployment"]
        assert deployment_prep["package_size_mb"] < 500  # Reasonable size

        # Verify workflow completeness
        assert len(workflow_results) == len(mock_datasets) * 2 + 2  # Each set: training + validation + init + final + deploy


class TestContinualLearningRealData:
    """Tests with real 17Lands data when available."""

    @pytest.fixture
    def real_data_pipeline(self):
        """Initialize real 17Lands data pipeline."""
        return SeventeenLandsDataPipeline()

    def test_real_17lands_integration(self, real_data_pipeline):
        """
        T033-009: Test integration with real 17Lands data.

        This test will be skipped if real data is not available.
        """
        pytest.skip("Real 17Lands data test - requires actual data files")

        # Check if real data is available
        if not real_data_pipeline.has_real_data():
            pytest.skip("No real 17Lands data available")

        # Load real dataset
        real_dataset = real_data_pipeline.load_recent_dataset(
            format="PremierDraft",
            days_back=30
        )

        # Test continual learning with real data
        config = {
            "workspace": tempfile.mkdtemp(),
            "use_real_data": True,
            "data_source": "17lands"
        }

        manager = ContinualLearningManager(config)

        # Train on real data
        result = manager.train_on_set(
            set_name="RealData",
            dataset=real_dataset,
            validate_with_real_data=True
        )

        assert result["success"]
        assert result["real_data_performance"] > 0.5


class TestContinualLearningPerformance:
    """Performance and scalability tests for continual learning."""

    def test_training_performance_scalability(self):
        """
        T033-010: Test that continual learning scales with dataset size.
        """
        pytest.skip("Performance test - requires substantial computational resources")

        # Test with different dataset sizes
        dataset_sizes = [100, 1000, 10000]
        training_times = []

        for size in dataset_sizes:
            config = {"workspace": tempfile.mkdtemp()}
            manager = ContinualLearningManager(config)

            # Generate synthetic dataset
            dataset = manager.generate_synthetic_dataset(size)

            # Measure training time
            start_time = time.time()
            result = manager.train_on_set("test", dataset)
            training_time = time.time() - start_time

            training_times.append(training_time)

            # Verify reasonable scaling (should be sub-quadratic)
            if len(training_times) > 1:
                scaling_factor = training_times[-1] / training_times[-2]
                size_factor = dataset_sizes[-1] / dataset_sizes[-2]
                assert scaling_factor < size_factor ** 1.5  # Sub-quadratic scaling

        # Clean up
        shutil.rmtree(config["workspace"])

    def test_memory_efficiency(self):
        """
        T033-011: Test memory efficiency of continual learning components.
        """
        pytest.skip("Memory efficiency test - requires memory profiling tools")

        # This would test that memory usage grows sub-linearly
        # with the number of tasks/domains learned
        pass


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "-k", "test_initialization"])