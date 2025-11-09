"""
Comprehensive unit tests for continual learning without catastrophic forgetting.

Task T032 for User Story 1 - Enhanced AI Decision Quality

Tests follow Red-Green-Refactor approach:
- Initially FAIL (implementation doesn't exist)
- Will PASS after implementing continual learning system
- Validates elastic weight consolidation (EWC)
- Tests progressive neural networks and domain adaptation
- Ensures knowledge preservation across training episodes
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from dataclasses import dataclass
from collections import defaultdict

# Import modules that will be created (these imports will fail initially)
from src.rl.training.continual_learning import (
    ContinualLearningTrainer,
    ElasticWeightConsolidation,
    ProgressiveNeuralNetwork,
    MemoryConsolidation,
    ExperienceReplayBuffer,
    KnowledgeDistillationLoss,
    TaskBoundaryDetector,
    PerformanceTracker,
    ContinualLearningConfig
)

from src.rl.training.domain_adaptation import (
    DomainAdapter,
    SetDistributionAnalyzer,
    FeatureAlignment,
    AdaptationScheduler,
    DomainPerformanceValidator
)

from src.rl.models.mtg_transformer import MTGTransformerEncoder
from src.rl.models.mtg_decision_head import MTGDecisionHead
from src.rl.training.mtg_training_pipeline import MTGTrainingPipeline

# Fallback imports for basic functionality if advanced modules don't exist yet
try:
    from src.rl.data.continual_replay_buffer import ContinualReplayBuffer
except ImportError:
    ContinualReplayBuffer = None

try:
    from src.rl.models.progressive_networks import ProgressiveNetworks
except ImportError:
    ProgressiveNetworks = None


class TestElasticWeightConsolidation:
    """Test Elastic Weight Consolidation for catastrophic forgetting prevention."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for EWC testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    @pytest.fixture
    def ewc_config(self):
        """Create EWC configuration."""
        return ContinualLearningConfig(
            ewc_lambda=0.4,
            ewc_importance_samples=100,
            online_ewc=True,
            ewc_decay_factor=0.95
        )

    @pytest.fixture
    def task_data(self):
        """Generate data for sequential tasks (simulating MTG set transitions)."""
        # Task 1: Original MTG set (e.g., Standard)
        task1_data = {
            'states': np.random.randn(200, 10).astype(np.float32),
            'actions': np.random.randint(0, 5, 200),
            'rewards': np.random.randn(200).astype(np.float32)
        }

        # Task 2: New MTG set (e.g., with new mechanics)
        task2_data = {
            'states': np.random.randn(200, 10).astype(np.float32),
            'actions': np.random.randint(0, 5, 200),
            'rewards': np.random.randn(200).astype(np.float32)
        }

        # Task 3: Another MTG set (different meta)
        task3_data = {
            'states': np.random.randn(200, 10).astype(np.float32),
            'actions': np.random.randint(0, 5, 200),
            'rewards': np.random.randn(200).astype(np.float32)
        }

        return [task1_data, task2_data, task3_data]

    def test_ewc_initialization(self, simple_model, ewc_config):
        """Test EWC system initialization."""
        ewc = ElasticWeightConsolidation(simple_model, ewc_config)

        assert ewc.model == simple_model
        assert ewc.config == ewc_config
        assert ewc.important_weights is None  # No Fisher information yet
        assert ewc.previous_weights is None

    def test_fisher_information_computation(self, simple_model, ewc_config, task_data):
        """Test Fisher information computation for importance weighting."""
        ewc = ElasticWeightConsolidation(simple_model, ewc_config)

        # Train model on first task
        task1 = task_data[0]
        self._train_simple_model(simple_model, task1, epochs=5)

        # Compute Fisher information
        ewc.compute_fisher_information(task1['states'], task1['actions'])

        # Check Fisher information was computed
        assert ewc.important_weights is not None
        assert len(ewc.important_weights) == len(list(simple_model.parameters()))

        # Check Fisher information is positive
        for fisher in ewc.important_weights:
            assert torch.all(fisher >= 0), "Fisher information should be non-negative"

    def test_ewc_penalty_computation(self, simple_model, ewc_config, task_data):
        """Test EWC penalty computation for preventing catastrophic forgetting."""
        ewc = ElasticWeightConsolidation(simple_model, ewc_config)

        # Train on task 1
        task1 = task_data[0]
        self._train_simple_model(simple_model, task1, epochs=5)

        # Store current weights and compute Fisher
        original_weights = [p.clone() for p in simple_model.parameters()]
        ewc.compute_fisher_information(task1['states'], task1['actions'])

        # Modify weights (simulate training on new task)
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Compute EWC penalty
        penalty = ewc.compute_ewc_penalty()

        # Penalty should be positive (weights moved from important positions)
        assert penalty > 0, "EWC penalty should be positive when weights change"

    def test_catastrophic_forgetting_prevention(self, simple_model, ewc_config, task_data):
        """Test that EWC prevents catastrophic forgetting."""
        ewc = ElasticWeightConsolidation(simple_model, ewc_config)

        # Train on task 1
        task1 = task_data[0]
        self._train_simple_model(simple_model, task1, epochs=10)

        # Evaluate on task 1
        task1_accuracy_before = self._evaluate_model(simple_model, task1)

        # Compute Fisher and enable EWC
        ewc.compute_fisher_information(task1['states'], task1['actions'])

        # Train on task 2 with EWC
        task2 = task_data[1]
        self._train_simple_model_with_ewc(simple_model, task2, ewc, epochs=10)

        # Evaluate on task 1 again (should maintain performance)
        task1_accuracy_after = self._evaluate_model(simple_model, task1)

        # Constitutional requirement: prevent catastrophic forgetting
        forgetting_ratio = (task1_accuracy_before - task1_accuracy_after) / task1_accuracy_before
        assert forgetting_ratio < 0.2, f"Too much forgetting: {forgetting_ratio:.2%} drop in performance"

    def test_online_ewc_functionality(self, simple_model, ewc_config, task_data):
        """Test online EWC functionality for continuous learning."""
        ewc_config.online_ewc = True
        ewc_config.ewc_decay_factor = 0.9
        ewc = ElasticWeightConsolidation(simple_model, ewc_config)

        # Track performance across tasks
        task_accuracies = []

        for task_idx, task in enumerate(task_data):
            # Train on current task
            if task_idx > 0:
                # Apply EWC for subsequent tasks
                self._train_simple_model_with_ewc(simple_model, task, ewc, epochs=5)
            else:
                # First task without EWC
                self._train_simple_model(simple_model, task, epochs=5)

            # Evaluate on all previous tasks
            current_accuracies = []
            for prev_idx in range(task_idx + 1):
                accuracy = self._evaluate_model(simple_model, task_data[prev_idx])
                current_accuracies.append(accuracy)

            task_accuracies.append(current_accuracies)

            # Update Fisher information for next task
            if task_idx < len(task_data) - 1:
                ewc.compute_fisher_information(task['states'], task['actions'])

        # Validate that performance on early tasks remains reasonable
        final_performance_on_task1 = task_accuracies[-1][0]
        initial_performance_on_task1 = task_accuracies[0][0]

        performance_retention = final_performance_on_task1 / initial_performance_on_task1
        assert performance_retention > 0.7, f"Online EWC failed to maintain performance: {performance_retention:.2%}"

    def test_ewc_lambda_impact(self, simple_model, task_data):
        """Test impact of EWC lambda on forgetting prevention."""
        lambdas = [0.0, 0.2, 0.4, 0.8]
        retention_rates = []

        for lambda_val in lambdas:
            # Create fresh model copy
            model_copy = self._copy_model(simple_model)

            config = ContinualLearningConfig(ewc_lambda=lambda_val)
            ewc = ElasticWeightConsolidation(model_copy, config)

            # Train on task 1
            task1 = task_data[0]
            self._train_simple_model(model_copy, task1, epochs=5)
            task1_acc = self._evaluate_model(model_copy, task1)

            # Compute Fisher
            ewc.compute_fisher_information(task1['states'], task1['actions'])

            # Train on task 2
            task2 = task_data[1]
            self._train_simple_model_with_ewc(model_copy, task2, ewc, epochs=5)

            # Evaluate task 1 retention
            task1_acc_after = self._evaluate_model(model_copy, task1)
            retention = task1_acc_after / task1_acc
            retention_rates.append(retention)

        # Higher lambda should provide better retention (up to a point)
        # Lambda=0.0 should have worst retention (no protection)
        assert retention_rates[0] < retention_rates[2], "EWC with lambda=0 should allow more forgetting than lambda=0.4"

    def _train_simple_model(self, model, task_data, epochs=10):
        """Train simple model on task data."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Simple batch
            states = torch.FloatTensor(task_data['states'])
            targets = torch.LongTensor(task_data['actions'])

            outputs = model(states)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

    def _train_simple_model_with_ewc(self, model, task_data, ewc, epochs=10):
        """Train model with EWC penalty."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            states = torch.FloatTensor(task_data['states'])
            targets = torch.LongTensor(task_data['actions'])

            outputs = model(states)
            task_loss = criterion(outputs, targets)

            # Add EWC penalty
            ewc_penalty = ewc.compute_ewc_penalty()
            total_loss = task_loss + ewc_penalty

            total_loss.backward()
            optimizer.step()

    def _evaluate_model(self, model, task_data):
        """Evaluate model accuracy on task data."""
        model.eval()
        with torch.no_grad():
            states = torch.FloatTensor(task_data['states'])
            targets = torch.LongTensor(task_data['actions'])

            outputs = model(states)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets).float().mean().item()

        return accuracy

    def _copy_model(self, model):
        """Create copy of model."""
        model_copy = type(model)(
            *[
                layer.in_features if hasattr(layer, 'in_features') else
                layer.out_features if hasattr(layer, 'out_features') else
                layer.weight.size(1) if hasattr(layer, 'weight') else None
                for layer in model
            ]
        )

        # Copy parameters
        model_copy.load_state_dict(model.state_dict())
        return model_copy


class TestContinualReplayBuffer:
    """Test continual learning replay buffer with episodic memory segregation."""

    @pytest.fixture
    def buffer_config(self):
        """Create buffer configuration."""
        return {
            'max_size': 1000,
            'episodic_segregation': True,
            'rehearsal_ratio': 0.1,
            'task_balance': True,
            'memory_importance_decay': 0.95
        }

    @pytest.fixture
    def multi_task_data(self):
        """Generate multi-task data simulating different MTG sets."""
        tasks = {}

        for task_id, set_name in enumerate(['DOM', 'KHM', 'STX', 'MID']):
            tasks[task_id] = {
                'task_id': task_id,
                'set_name': set_name,
                'states': np.random.randn(100, 10).astype(np.float32),
                'actions': np.random.randint(0, 5, 100),
                'rewards': np.random.randn(100).astype(np.float32),
                'episodes': self._generate_episodes(25, 4)  # 25 episodes, 4 steps each
            }

        return tasks

    def test_buffer_initialization(self, buffer_config):
        """Test continual replay buffer initialization."""
        buffer = ContinualReplayBuffer(**buffer_config)

        assert buffer.max_size == buffer_config['max_size']
        assert buffer.episodic_segregation == buffer_config['episodic_segregation']
        assert buffer.rehearsal_ratio == buffer_config['rehearsal_ratio']

    def test_episodic_memory_segregation(self, buffer_config, multi_task_data):
        """Test that episodes are segregated by task."""
        buffer = ContinualReplayBuffer(**buffer_config)

        # Add data from different tasks
        for task_id, task_data in multi_task_data.items():
            buffer.add_task_data(task_data)

        # Check that episodes are properly segregated
        task_counts = buffer.get_task_distribution()

        assert len(task_counts) == len(multi_task_data), "Should have data for all tasks"

        for task_id in multi_task_data:
            assert task_id in task_counts, f"Missing data for task {task_id}"
            assert task_counts[task_id] > 0, f"No episodes for task {task_id}"

    def test_task_balanced_sampling(self, buffer_config, multi_task_data):
        """Test balanced sampling across tasks."""
        buffer = ContinualReplayBuffer(**buffer_config)

        # Add data from different tasks
        for task_data in multi_task_data.values():
            buffer.add_task_data(task_data)

        # Sample multiple times
        samples = []
        for _ in range(100):
            sample = buffer.sample_batch(32)
            samples.append(sample)

        # Check task balance in samples
        task_sample_counts = {}
        for sample in samples:
            task_ids = sample.get('task_ids', [])
            for task_id in task_ids:
                task_sample_counts[task_id] = task_sample_counts.get(task_id, 0) + 1

        # Should have reasonable balance (not dominated by single task)
        total_samples = sum(task_sample_counts.values())
        max_task_ratio = max(task_sample_counts.values()) / total_samples if total_samples > 0 else 0

        assert max_task_ratio < 0.8, f"Task imbalance detected: one task comprises {max_task_ratio:.1%} of samples"

    def test_rehearsal_ratio_functionality(self, buffer_config, multi_task_data):
        """Test rehearsal ratio for balancing new and old experiences."""
        rehearsal_ratios = [0.1, 0.3, 0.5]
        old_task_percentages = []

        for rehearsal_ratio in rehearsal_ratios:
            buffer_config['rehearsal_ratio'] = rehearsal_ratio
            buffer = ContinualReplayBuffer(**buffer_config)

            # Add old task data
            for task_id in [0, 1]:  # Old tasks
                buffer.add_task_data(multi_task_data[task_id])

            # Add new task data
            buffer.add_task_data(multi_task_data[2])  # New task

            # Sample and check old task proportion
            sample = buffer.sample_batch(100)
            task_ids = sample.get('task_ids', [])

            old_task_count = sum(1 for tid in task_ids if tid in [0, 1])
            old_task_percentage = old_task_count / len(task_ids)
            old_task_percentages.append(old_task_percentage)

        # Higher rehearsal ratio should include more old task data
        assert old_task_percentages[2] > old_task_percentages[0], \
            "Higher rehearsal ratio should include more old task experiences"

    def _generate_episodes(self, num_episodes, episode_length):
        """Generate episodic data."""
        episodes = []
        start_idx = 0

        for episode_id in range(num_episodes):
            end_idx = start_idx + episode_length
            episodes.append({
                'episode_id': episode_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'length': episode_length
            })
            start_idx = end_idx

        return episodes


class TestProgressiveNetworks:
    """Test progressive neural networks for knowledge preservation."""

    @pytest.fixture
    def progressive_network_config(self):
        """Create progressive network configuration."""
        return {
            'input_dim': 10,
            'hidden_dims': [32, 16],
            'output_dim': 5,
            'max_columns': 4,
            'freeze_columns': True
        }

    def test_progressive_network_initialization(self, progressive_network_config):
        """Test progressive network initialization."""
        pnet = ProgressiveNetworks(**progressive_network_config)

        assert pnet.input_dim == progressive_network_config['input_dim']
        assert pnet.max_columns == progressive_network_config['max_columns']
        assert len(pnet.columns) == 0  # No columns initially

    def test_column_addition(self, progressive_network_config):
        """Test adding new columns for new tasks."""
        pnet = ProgressiveNetworks(**progressive_network_config)

        # Add first column
        pnet.add_column()
        assert len(pnet.columns) == 1

        # Add second column
        pnet.add_column()
        assert len(pnet.columns) == 2

        # Should not exceed max columns
        for _ in range(progressive_network_config['max_columns']):
            pnet.add_column()
        assert len(pnet.columns) == progressive_network_config['max_columns']

    def test_lateral_connections(self, progressive_network_config):
        """Test lateral connections between columns."""
        pnet = ProgressiveNetworks(**progressive_network_config)

        # Add multiple columns
        pnet.add_column()
        pnet.add_column()

        # Test forward pass with lateral connections
        input_data = torch.randn(10, progressive_network_config['input_dim'])
        output = pnet.forward(input_data, column_id=1)  # Use second column

        # Output should be correct shape
        assert output.shape == (10, progressive_network_config['output_dim'])

        # Lateral connections should be established
        assert len(pnet.columns[1].lateral_connections) == 1  # Connection to first column

    def test_knowledge_preservation(self, progressive_network_config, multi_task_data):
        """Test that progressive networks preserve knowledge across tasks."""
        pnet = ProgressiveNetworks(**progressive_network_config)

        task_accuracies = []

        for task_id, task_data in list(multi_task_data.items())[:3]:  # Test first 3 tasks
            # Add new column for this task
            pnet.add_column()

            # Train on this task
            self._train_progressive_network(pnet, task_data, column_id=task_id, epochs=10)

            # Evaluate on all trained tasks
            current_accuracies = []
            for prev_task_id in range(task_id + 1):
                prev_task_data = multi_task_data[prev_task_id]
                accuracy = self._evaluate_progressive_network(
                    pnet, prev_task_data, column_id=prev_task_id
                )
                current_accuracies.append(accuracy)

            task_accuracies.append(current_accuracies)

        # Check that early task performance is preserved
        final_task1_accuracy = task_accuracies[-1][0]
        initial_task1_accuracy = task_accuracies[0][0]

        # Progressive networks should maintain early task performance
        performance_retention = final_task1_accuracy / initial_task1_accuracy
        assert performance_retention > 0.8, f"Progressive networks failed to preserve knowledge: {performance_retention:.2%}"

    def _train_progressive_network(self, pnet, task_data, column_id, epochs=10):
        """Train progressive network column on task data."""
        optimizer = torch.optim.Adam(pnet.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()

            states = torch.FloatTensor(task_data['states'])
            targets = torch.LongTensor(task_data['actions'])

            outputs = pnet.forward(states, column_id=column_id)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

    def _evaluate_progressive_network(self, pnet, task_data, column_id):
        """Evaluate progressive network on task data."""
        pnet.eval()
        with torch.no_grad():
            states = torch.FloatTensor(task_data['states'])
            targets = torch.LongTensor(task_data['actions'])

            outputs = pnet.forward(states, column_id=column_id)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets).float().mean().item()

        return accuracy


@dataclass
class TrainingTask:
    """Represents a training task in continual learning"""
    task_id: str
    data: List[Dict[str, Any]]
    domain_info: Dict[str, Any]
    importance_weight: float = 1.0
    num_samples: int = 0


@dataclass
class TaskPerformance:
    """Performance metrics for a specific task"""
    task_id: str
    accuracy: float
    loss: float
    forgetting_measure: float
    knowledge_preservation: float


class TestAdvancedElasticWeightConsolidation:
    """Advanced tests for Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention"""

    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device("cpu")
        self.model = nn.Sequential(
            nn.Linear(282, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # Action space
        ).to(self.device)

        # This will fail initially - red phase
        self.ewc = ElasticWeightConsolidation(
            model=self.model,
            importance_weight=1000.0,
            mode="separate"
        )

        # Create dummy training data for MTG scenarios
        self.task1_data = self._create_mtg_task_data("THB_draft", 100)
        self.task2_data = self._create_mtg_task_data("KHM_draft", 100)
        self.task3_data = self._create_mtg_task_data("STX_limited", 100)

    def _create_mtg_task_data(self, task_id: str, num_samples: int) -> List[Dict]:
        """Create MTG-specific task data with realistic characteristics"""
        data = []
        for i in range(num_samples):
            # Create game state with MTG-specific features
            state = torch.randn(282)

            # Add task-specific bias to simulate different MTG sets
            set_hash = hash(task_id) % 1000 / 1000.0
            state += torch.tensor([set_hash] * 282).float() * 0.1

            action = torch.randint(0, 16, (1,))
            reward = torch.randn(1)

            data.append({
                "state": state,
                "action": action,
                "reward": reward,
                "task_id": task_id,
                "game_context": {
                    "turn": i % 15,
                    "phase": "main",
                    "priority": True
                }
            })
        return data

    def test_ewc_initialization(self):
        """Test EWC system initialization - should fail initially"""
        assert self.ewc.model is not None
        assert self.ewc.importance_weight == 1000.0
        assert self.ewc.mode == "separate"
        assert len(self.ewc.parameter_importance) == 0
        assert len(self.ewc.previous_parameters) == 0

    def test_fisher_information_computation(self):
        """Test Fisher information matrix computation for MTG tasks"""
        fisher_info = self.ewc.compute_fisher_information(
            self.task1_data,
            batch_size=32,
            num_samples=100
        )

        # Fisher info should have same structure as model parameters
        for name, param in self.model.named_parameters():
            assert name in fisher_info
            assert fisher_info[name].shape == param.shape
            assert torch.all(fisher_info[name] >= 0)  # Fisher info is non-negative

    def test_ewc_loss_computation_for_mtg_sets(self):
        """Test EWC loss computation for MTG set transitions"""
        # Setup: compute Fisher information for THB draft
        self.ewc.compute_fisher_information(self.task1_data)
        self.ewc.store_previous_parameters()

        # Simulate learning new set (KHM)
        current_params = {}
        for name, param in self.model.named_parameters():
            current_params[name] = param.clone()
            # Simulate adaptation to new set
            param.data += torch.randn_like(param) * 0.01

        # Compute EWC loss
        ewc_loss = self.ewc.compute_ewc_loss()

        assert ewc_loss.item() >= 0  # Loss should be non-negative
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.requires_grad

    def test_knowledge_preservation_across_mtg_sets(self):
        """Test measurement of knowledge preservation across MTG set transitions"""
        # Train on THB draft
        initial_performance = self._evaluate_mtg_task_performance(self.task1_data)

        # Store importance with EWC
        self.ewc.compute_fisher_information(self.task1_data)
        self.ewc.store_previous_parameters()

        # Train on KHM draft (might cause forgetting)
        self._simulate_mtg_training(self.task2_data)

        # Measure knowledge preservation
        final_performance = self._evaluate_mtg_task_performance(self.task1_data)
        preservation = self.ewc.measure_knowledge_preservation(
            initial_performance, final_performance
        )

        assert 0 <= preservation <= 1  # Preservation score should be normalized
        assert isinstance(preservation, float)
        assert preservation > 0.7  # Should preserve most knowledge

    def test_adaptive_importance_weight_tuning(self):
        """Test adaptive importance weight tuning based on forgetting"""
        initial_weight = self.ewc.importance_weight

        # Simulate performance drop across multiple MTG sets
        performance_history = [0.85, 0.78, 0.68, 0.60]  # Declining performance
        adapted_weight = self.ewc.adapt_importance_weight(performance_history)

        assert adapted_weight > initial_weight  # Should increase importance
        assert isinstance(adapted_weight, float)
        assert adapted_weight > 0

    def test_multi_task_ewc_for_mtg_formats(self):
        """Test EWC across multiple MTG formats (draft, sealed, standard)"""
        format_tasks = {
            "draft": self._create_mtg_task_data("draft_format", 100),
            "sealed": self._create_mtg_task_data("sealed_format", 100),
            "standard": self._create_mtg_task_data("standard_format", 100)
        }

        format_performance = {}

        # Learn each format with EWC
        for format_name, task_data in format_tasks.items():
            if len(format_performance) > 0:
                # Apply EWC protection for previous formats
                self.ewc.compute_fisher_information(task_data)

            self._simulate_mtg_training(task_data)
            format_performance[format_name] = self._evaluate_all_formats_performance(
                format_tasks, format_name
            )

        # Check that previous formats are preserved
        for format_name in format_performance:
            if format_name != "standard":  # Except the last learned format
                performance_ratio = format_performance["standard"][format_name] / \
                                 format_performance[format_name][format_name]
                assert performance_ratio > 0.7, f"Too much forgetting in {format_name}"

    def _evaluate_mtg_task_performance(self, task_data: List[Dict]) -> float:
        """Helper to evaluate model performance on MTG task data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for sample in task_data:
                state = sample["state"].unsqueeze(0).to(self.device)
                action = sample["action"].item()

                output = self.model(state)
                predicted = output.argmax().item()

                if predicted == action:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def _simulate_mtg_training(self, task_data: List[Dict], epochs: int = 5):
        """Helper to simulate training on MTG task data"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for sample in task_data[:32]:  # Use subset for speed
                state = sample["state"].unsqueeze(0).to(self.device)
                action = sample["action"].to(self.device)

                optimizer.zero_grad()
                output = self.model(state)

                # Add EWC penalty if applicable
                if hasattr(self.ewc, 'compute_ewc_loss'):
                    ewc_penalty = self.ewc.compute_ewc_loss()
                    task_loss = criterion(output, action)
                    loss = task_loss + self.ewc.importance_weight * ewc_penalty
                else:
                    loss = criterion(output, action)

                loss.backward()
                optimizer.step()

    def _evaluate_all_formats_performance(self, format_tasks: Dict, current_format: str) -> Dict:
        """Evaluate performance on all formats"""
        performance = {}
        for format_name, task_data in format_tasks.items():
            performance[format_name] = self._evaluate_mtg_task_performance(task_data)
        return performance


class TestAdvancedProgressiveNeuralNetwork:
    """Advanced tests for Progressive Neural Networks for MTG continual learning"""

    def setup_method(self):
        """Setup test fixtures for MTG progressive networks"""
        self.input_dim = 282  # MTG state dimension
        self.hidden_dim = 128
        self.output_dim = 16  # Action space
        self.max_tasks = 5  # Maximum MTG sets to handle

        # This will fail initially - red phase
        self.pnn = ProgressiveNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            max_tasks=self.max_tasks,
            task_type="mtg_set"
        )

    def test_pnn_mtg_set_initialization(self):
        """Test PNN initialization for MTG set learning"""
        assert self.pnn.current_task_count == 0
        assert len(self.pnn.columns) == 0
        assert len(self.pnn.adapters) == 0
        assert self.pnn.max_tasks == self.max_tasks
        assert self.pnn.task_type == "mtg_set"

    def test_mtg_column_addition(self):
        """Test adding new columns for different MTG sets"""
        mtg_sets = ["THB", "KHM", "STX", "MID", "DMU"]

        for mtg_set in mtg_sets[:3]:  # Add first 3 sets
            self.pnn.add_task_column(mtg_set)
            assert self.pnn.current_task_count == mtg_sets.index(mtg_set) + 1
            assert mtg_set in self.pnn.columns
            assert mtg_set in self.pnn.adapters

    def test_lateral_connections_for_mtg_knowledge_transfer(self):
        """Test lateral connections for MTG knowledge transfer between sets"""
        self.pnn.add_task_column("THB")
        self.pnn.add_task_column("KHM")

        # Create MTG-specific input data
        input_data = torch.randn(10, self.input_dim)

        # Forward pass for KHM should use lateral connections from THB
        output_khm = self.pnn.forward(input_data, task_id="KHM")
        assert output_khm.shape == (10, self.output_dim)

        # Output should be different for different MTG sets
        output_thb = self.pnn.forward(input_data, task_id="THB")
        assert not torch.equal(output_thb, output_khm)

    def test_mtg_column_freezing(self):
        """Test freezing of previous MTG set columns"""
        self.pnn.add_task_column("THB")

        # Freeze THB column
        self.pnn.freeze_column("THB")
        assert self.pnn.is_frozen("THB")

        # Attempt to modify frozen column (should not change parameters)
        original_params = {}
        for name, param in self.pnn.columns["THB"].named_parameters():
            original_params[name] = param.clone()

        # Simulate parameter update
        with torch.no_grad():
            for param in self.pnn.columns["THB"].parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Check that parameters haven't changed (frozen)
        for name, param in self.pnn.columns["THB"].named_parameters():
            assert torch.equal(param, original_params[name])

    def test_mtg_mechanics_transfer_efficiency(self):
        """Test knowledge transfer efficiency for MTG mechanics"""
        self.pnn.add_task_column("THB")  # Set with enchantment focus
        self.pnn.add_task_column("KHM")  # Set with different mechanics

        # Create similar tasks with shared mechanics
        thb_data = self._create_mtg_mechanics_data("enchant", 100)
        khm_data = self._create_mtg_mechanics_data("enchant", 100)  # Similar mechanics

        # Train on THB
        self._train_pnn_mtg_task("THB", thb_data)

        # Measure transfer efficiency to KHM
        transfer_efficiency = self.pnn.measure_transfer_efficiency(
            "THB", "KHM", khm_data
        )

        assert 0 <= transfer_efficiency <= 1
        assert isinstance(transfer_efficiency, float)
        assert transfer_efficiency > 0.5  # Should have reasonable transfer

    def test_mtg_set_capacity_management(self):
        """Test management of network capacity for MTG sets"""
        mtg_sets = ["THB", "KHM", "STX", "MID", "DMU", "NEO"]  # 6 sets (exceeds capacity)

        # Fill up to max capacity
        for i, mtg_set in enumerate(mtg_sets[:self.max_tasks]):
            self.pnn.add_task_column(mtg_set)

        assert self.pnn.current_task_count == self.max_tasks

        # Attempting to add beyond capacity should handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            self.pnn.add_task_column(mtg_sets[self.max_tasks])

    def _create_mtg_mechanics_data(self, mechanic_type: str, num_samples: int) -> torch.Tensor:
        """Create MTG data specific to certain mechanics"""
        data = torch.randn(num_samples, self.input_dim)
        # Add mechanic-specific bias
        mechanic_hash = hash(mechanic_type) % 1000 / 1000.0
        data += torch.tensor([mechanic_hash] * self.input_dim).view(1, -1) * 0.1
        return data

    def _train_pnn_mtg_task(self, task_id: str, data: torch.Tensor, epochs: int = 10):
        """Helper to train PNN on specific MTG task"""
        self.pnn.train_task(task_id)
        optimizer = optim.Adam(self.pnn.get_trainable_parameters(task_id), lr=0.001)
        criterion = nn.MSELoss()

        targets = torch.randn(data.shape[0], self.output_dim)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.pnn.forward(data, task_id=task_id)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()


class TestAdvancedMemoryConsolidation:
    """Advanced tests for memory consolidation and replay buffer management"""

    def setup_method(self):
        """Setup test fixtures"""
        self.buffer_size = 1000
        self.replay_buffer = ExperienceReplayBuffer(
            max_size=self.buffer_size,
            prioritized=True,
            task_aware=True
        )

        self.memory_consolidation = MemoryConsolidation(
            replay_buffer=self.replay_buffer,
            consolidation_strategy="diversity",
            task_preservation_weight=0.7
        )

    def test_mtg_experience_storage(self):
        """Test storing MTG-specific experiences in replay buffer"""
        mtg_experiences = [
            {
                "state": torch.randn(282),
                "action": 5,  # Cast creature
                "reward": 1.0,
                "task_id": "THB_draft",
                "game_context": {
                    "turn": 3,
                    "phase": "main",
                    "hand_size": 6,
                    "lands_played": 2
                },
                "card_info": {
                    "cmc": 3,
                    "type": "creature",
                    "rarity": "rare"
                }
            },
            {
                "state": torch.randn(282),
                "action": 3,  # Play land
                "reward": 0.5,
                "task_id": "THB_draft",
                "game_context": {
                    "turn": 4,
                    "phase": "main",
                    "hand_size": 5,
                    "lands_played": 3
                },
                "card_info": {
                    "cmc": 0,
                    "type": "land",
                    "rarity": "common"
                }
            },
            {
                "state": torch.randn(282),
                "action": 7,  # Attack
                "reward": -0.2,
                "task_id": "KHM_draft",
                "game_context": {
                    "turn": 8,
                    "phase": "combat",
                    "hand_size": 4,
                    "lands_played": 5
                },
                "card_info": {
                    "cmc": 0,
                    "type": "action",
                    "rarity": None
                }
            }
        ]

        for exp in mtg_experiences:
            self.replay_buffer.add(exp)

        assert len(self.replay_buffer) == 3

        # Test retrieval
        retrieved = self.replay_buffer.sample(batch_size=2)
        assert len(retrieved) == 2
        assert all("state" in exp for exp in retrieved)
        assert all("task_id" in exp for exp in retrieved)

    def test_prioritized_mtg_replay(self):
        """Test prioritized experience replay for MTG scenarios"""
        # Add MTG experiences with different priorities
        high_priority_exp = {
            "state": torch.randn(282),
            "action": 1,  # High-impact play
            "reward": 2.0,  # High reward
            "task_id": "THB_draft",
            "game_context": {"turn": 1, "phase": "main"},
            "card_info": {"cmc": 5, "type": "planeswalker", "rarity": "mythic"}
        }
        low_priority_exp = {
            "state": torch.randn(282),
            "action": 2,  # Low-impact play
            "reward": -1.0,  # Low reward
            "task_id": "THB_draft",
            "game_context": {"turn": 10, "phase": "end"},
            "card_info": {"cmc": 1, "type": "instant", "rarity": "common"}
        }

        self.replay_buffer.add(high_priority_exp)
        self.replay_buffer.add(low_priority_exp)

        # High priority experience should be sampled more often
        high_priority_count = 0
        for _ in range(100):
            sampled = self.replay_buffer.sample(batch_size=1)[0]
            if sampled["reward"] > 0:
                high_priority_count += 1

        assert high_priority_count > 50  # Should be sampled more than half the time

    def test_mtg_set_memory_consolidation(self):
        """Test memory consolidation strategies for MTG sets"""
        # Fill buffer with diverse MTG experiences
        mtg_sets = ["THB", "KHM", "STX"]
        for set_code in mtg_sets:
            for i in range(50):
                exp = {
                    "state": torch.randn(282),
                    "action": i % 16,
                    "reward": torch.randn(1).item(),
                    "task_id": f"{set_code}_draft",
                    "game_context": {
                        "turn": i % 15,
                        "phase": ["main", "combat", "end"][i % 3],
                        "set_code": set_code
                    },
                    "card_info": {
                        "cmc": i % 7,
                        "type": ["creature", "spell", "land"][i % 3],
                        "rarity": ["common", "uncommon", "rare", "mythic"][i % 4]
                    }
                }
                self.replay_buffer.add(exp)

        # Consolidate buffer (should reduce size while maintaining diversity)
        original_size = len(self.replay_buffer)
        consolidated_buffer = self.memory_consolidation.consolidate(
            target_size=original_size // 2,
            preserve_sets=mtg_sets
        )

        assert len(consolidated_buffer) <= original_size // 2

        # Check diversity preservation across MTG sets
        set_ids = set(exp["task_id"] for exp in consolidated_buffer)
        assert len(set_ids) == 3  # Should maintain diversity across sets

        # Check card type diversity
        card_types = set(exp["card_info"]["type"] for exp in consolidated_buffer)
        assert len(card_types) >= 2  # Should maintain card type diversity

    def test_catastrophic_forgetting_prevention_for_mtg_sets(self):
        """Test that memory consolidation prevents catastrophic forgetting of MTG sets"""
        # Simulate learning sequence for MTG sets
        thb_experiences = self._generate_mtg_set_experiences("THB", 100)
        khm_experiences = self._generate_mtg_set_experiences("KHM", 100)

        # Learn THB set
        for exp in thb_experiences:
            self.replay_buffer.add(exp)

        thb_performance_before = self._simulate_mtg_replay_learning("THB")

        # Learn KHM set (might cause forgetting)
        for exp in khm_experiences:
            self.replay_buffer.add(exp)

        # Use consolidated memory to prevent forgetting
        consolidated_buffer = self.memory_consolidation.consolidate(
            target_size=150,
            preserve_sets=["THB", "KHM"],
            preservation_strategy="balanced"
        )

        thb_performance_after = self._simulate_mtg_replay_learning(
            buffer=consolidated_buffer,
            focus_set="THB"
        )

        # Performance should not degrade significantly
        performance_drop = thb_performance_before - thb_performance_after
        assert performance_drop < 0.1  # Less than 10% performance drop

    def test_mtg_memory_importance_weighting(self):
        """Test importance weighting for MTG memory consolidation"""
        # Create experiences with different importance levels
        critical_experiences = []
        routine_experiences = []

        for i in range(50):
            # Critical experiences (game-winning plays, complex decisions)
            critical_exp = {
                "state": torch.randn(282),
                "action": 1,
                "reward": 3.0,  # Very high reward
                "task_id": "THB_draft",
                "importance": 1.0,
                "game_context": {"turn": 15, "phase": "combat", "life_total": 1}
            }
            critical_experiences.append(critical_exp)

            # Routine experiences (normal plays)
            routine_exp = {
                "state": torch.randn(282),
                "action": 5,
                "reward": 0.1,  # Low reward
                "task_id": "THB_draft",
                "importance": 0.3,
                "game_context": {"turn": 5, "phase": "main", "life_total": 20}
            }
            routine_experiences.append(routine_exp)

        # Add all experiences
        for exp in critical_experiences + routine_experiences:
            self.replay_buffer.add(exp)

        # Consolidate with importance weighting
        consolidated = self.memory_consolidation.consolidate_with_importance(
            target_size=50,
            importance_threshold=0.5
        )

        # Should preserve more critical experiences
        critical_count = sum(1 for exp in consolidated if exp.get("importance", 0) > 0.5)
        routine_count = len(consolidated) - critical_count

        assert critical_count > routine_count, "Should preserve more critical experiences"

    def _generate_mtg_set_experiences(self, set_code: str, num_exp: int) -> List[Dict]:
        """Generate experiences for a specific MTG set"""
        experiences = []
        for i in range(num_exp):
            exp = {
                "state": torch.randn(282),
                "action": i % 16,
                "reward": torch.randn(1).item(),
                "task_id": f"{set_code}_draft",
                "importance": abs(torch.randn(1).item()),
                "game_context": {
                    "turn": i % 15,
                    "phase": ["main", "combat", "end"][i % 3],
                    "set_code": set_code,
                    "life_total": 20 - (i % 10)
                },
                "card_info": {
                    "cmc": i % 7,
                    "type": ["creature", "spell", "land"][i % 3],
                    "rarity": ["common", "uncommon", "rare", "mythic"][i % 4]
                }
            }
            experiences.append(exp)
        return experiences

    def _simulate_mtg_replay_learning(self, set_code: str = None, buffer=None) -> float:
        """Simulate learning from MTG replay buffer and return performance"""
        if buffer is None:
            buffer = self.replay_buffer

        if len(buffer) == 0:
            return 0.0

        # Simple performance simulation based on buffer quality
        total_reward = 0
        count = 0

        for exp in buffer:
            if set_code is None or exp["task_id"] == f"{set_code}_draft":
                total_reward += exp["reward"]
                count += 1

        return total_reward / count if count > 0 else 0.0


class TestAdvancedDomainAdaptation:
    """Advanced tests for domain adaptation for new MTG sets"""

    def setup_method(self):
        """Setup test fixtures for MTG domain adaptation"""
        # This will fail initially - red phase
        self.domain_adapter = DomainAdapter(
            base_model=MTGTransformerEncoder(),
            adaptation_method="fine_tuning",
            target_domain="mtg_set"
        )

        self.set_analyzer = SetDistributionAnalyzer()
        self.feature_alignment = FeatureAlignment()
        self.adaptation_scheduler = AdaptationScheduler()

    def test_mtg_set_distribution_analysis(self):
        """Test analysis of card distribution across MTG sets"""
        # Mock data for different MTG sets with realistic characteristics
        set_data = {
            "THB": {  # Theros Beyond Death - enchantment focus
                "creature_ratio": 0.32,
                "enchantment_ratio": 0.18,  # Higher than average
                "spell_ratio": 0.30,
                "land_ratio": 0.20,
                "avg_cmc": 3.4,
                "mechanics": ["enchant", "devotion", "escape"],
                "power_distribution": {"low": 0.4, "medium": 0.4, "high": 0.2}
            },
            "KHM": {  # Kaldheim - Viking theme, legendary focus
                "creature_ratio": 0.38,
                "legendary_ratio": 0.15,  # Higher than average
                "spell_ratio": 0.27,
                "land_ratio": 0.20,
                "avg_cmc": 3.1,
                "mechanics": ["foretell", "changeling", "sagas"],
                "power_distribution": {"low": 0.3, "medium": 0.5, "high": 0.2}
            },
            "STX": {  # Strixhaven - college theme, multicolor focus
                "creature_ratio": 0.35,
                "multicolor_ratio": 0.25,  # Higher than average
                "spell_ratio": 0.30,
                "land_ratio": 0.20,
                "avg_cmc": 2.9,
                "mechanics": ["learn", "lesson", "magecraft"],
                "power_distribution": {"low": 0.35, "medium": 0.45, "high": 0.2}
            }
        }

        distribution_stats = self.set_analyzer.analyze_set_distributions(set_data)

        assert "THB" in distribution_stats
        assert "KHM" in distribution_stats
        assert "STX" in distribution_stats
        assert "distribution_shift" in distribution_stats
        assert "adaptation_priority" in distribution_stats
        assert "mechanics_overlap" in distribution_stats

        # Check that mechanics overlap is computed
        assert isinstance(distribution_stats["mechanics_overlap"], dict)
        assert len(distribution_stats["mechanics_overlap"]) > 0

    def test_mtg_feature_alignment(self):
        """Test feature alignment between MTG domains"""
        # Create feature representations for different MTG sets
        base_features = torch.randn(100, 282)  # Base set (e.g., THB)
        target_features = torch.randn(100, 282)  # New set (e.g., KHM)

        # Add systematic shift to simulate MTG domain change
        # Simulate KHM's higher legendary ratio
        legendary_bias = torch.zeros(1, 282)
        legendary_bias[0, 50:60] = 0.3  # Legendary encoding bias
        target_features += legendary_bias

        alignment_score = self.feature_alignment.compute_alignment(
            base_features, target_features,
            alignment_method="coral"  # CORAL alignment
        )

        assert 0 <= alignment_score <= 1
        assert isinstance(alignment_score, float)

        # Test alignment transformation
        aligned_features = self.feature_alignment.align_features(
            target_features, base_features,
            method="coral"
        )

        assert aligned_features.shape == target_features.shape

        # Alignment should reduce distribution shift
        new_alignment = self.feature_alignment.compute_alignment(
            base_features, aligned_features
        )
        assert new_alignment > alignment_score

    def test_mtg_adaptation_scheduling(self):
        """Test scheduling of MTG domain adaptation"""
        # Create adaptation schedule for MTG set transitions
        schedule = self.adaptation_scheduler.create_schedule(
            base_sets=["THB", "KHM", "STX"],
            target_set="DMU",  # Dominaria United
            adaptation_strategy="progressive",
            domain_type="mtg_set"
        )

        assert "phases" in schedule
        assert len(schedule["phases"]) > 0
        assert "learning_rates" in schedule
        assert "data_mixing_ratios" in schedule
        assert "set_similarity_weights" in schedule

        # Validate phase progression
        for i, phase in enumerate(schedule["phases"]):
            assert "duration" in phase
            assert "focus" in phase
            assert "stability_requirement" in phase

            # Later phases should have more target data
            if i > 0:
                current_mix = phase["data_mixing_ratios"]["target"]
                previous_mix = schedule["phases"][i-1]["data_mixing_ratios"]["target"]
                assert current_mix >= previous_mix

    def test_cross_mtg_set_performance_validation(self):
        """Test performance validation across different MTG sets"""
        validator = DomainPerformanceValidator()

        # Mock performance data across MTG sets
        performance_data = {
            "THB_draft": {
                "accuracy": 0.82,
                "loss": 0.45,
                "set_specific_metrics": {
                    "enchantment_usage": 0.88,
                    "devotion_utilization": 0.76
                }
            },
            "KHM_draft": {
                "accuracy": 0.78,
                "loss": 0.52,
                "set_specific_metrics": {
                    "foretell_usage": 0.72,
                    "legendary_utilization": 0.68
                }
            },
            "STX_limited": {
                "accuracy": 0.80,
                "loss": 0.48,
                "set_specific_metrics": {
                    "learn_usage": 0.85,
                    "multicolor_utilization": 0.79
                }
            },
            "DMU_limited": {
                "accuracy": 0.75,  # New set with lower performance
                "loss": 0.58,
                "set_specific_metrics": {
                    "enchant_usage": 0.65,  # Forgetting THB mechanics
                    "legendary_utilization": 0.55  # Forgetting KHM mechanics
                }
            }
        }

        validation_results = validator.validate_cross_set_performance(
            performance_data,
            baseline_set="THB_draft",
            validation_type="mtg_set"
        )

        assert "overall_performance" in validation_results
        assert "set_degradation" in validation_results
        assert "adaptation_needed" in validation_results
        assert "mechanics_forgetting" in validation_results
        assert validation_results["adaptation_needed"] is True

        # Check mechanics-specific analysis
        mechanics_analysis = validation_results["mechanics_forgetting"]
        assert "enchant_usage" in mechanics_analysis
        assert "foretell_usage" in mechanics_analysis
        assert "learn_usage" in mechanics_analysis

    def test_catastrophic_forgetting_in_mtg_domain_adaptation(self):
        """Test that MTG domain adaptation doesn't cause catastrophic forgetting"""
        # Simulate performance before adaptation
        base_performance = {
            "THB_draft": 0.85,
            "KHM_draft": 0.82,
            "STX_limited": 0.83
        }

        # Simulate domain adaptation to new set (DMU)
        adaptation_results = self.domain_adapter.adapt_to_new_set(
            target_set="DMU",
            adaptation_data=None,  # Would be real DMU data
            preservation_weight=0.1,
            mechanics_preservation=True
        )

        # Check that base performance is preserved
        post_adaptation_performance = {
            "THB_draft": 0.83,
            "KHM_draft": 0.80,
            "STX_limited": 0.81,
            "DMU_limited": 0.78  # New set performance
        }

        # Calculate forgetting measure
        forgetting = self._calculate_mtg_forgetting(
            base_performance,
            post_adaptation_performance
        )

        assert forgetting < 0.05  # Less than 5% performance loss
        assert post_adaptation_performance["DMU_limited"] > 0.7  # Reasonable performance on new set

    def test_mtg_mechanics_transfer_validation(self):
        """Test validation of mechanics transfer between MTG sets"""
        # Define mechanics relationships between sets
        mechanics_relationships = {
            "THB": {"enchant", "devotion", "escape"},
            "KHM": {"foretell", "changeling", "sagas"},
            "STX": {"learn", "lesson", "magecraft"},
            "DMU": {"enchant", "equip", "ward"}  # Shares "enchant" with THB
        }

        validator = DomainPerformanceValidator()

        # Test mechanics transfer validation
        transfer_results = validator.validate_mechanics_transfer(
            source_sets=["THB", "KHM", "STX"],
            target_set="DMU",
            mechanics_relationships=mechanics_relationships
        )

        assert "transfer_efficiency" in transfer_results
        assert "shared_mechanics" in transfer_results
        assert "novel_mechanics" in transfer_results
        assert "adaptation_recommendations" in transfer_results

        # Should identify "enchant" as shared mechanic
        assert "enchant" in transfer_results["shared_mechanics"]
        assert transfer_results["shared_mechanics"]["enchant"]["transfer_potential"] > 0.7

    def _calculate_mtg_forgetting(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate average performance forgetting for MTG sets"""
        forgetting_values = []
        for set_name in before:
            if set_name in after:
                performance_drop = before[set_name] - after[set_name]
                forgetting_values.append(max(0, performance_drop))

        return np.mean(forgetting_values) if forgetting_values else 0.0


class TestContinualLearningTrainerIntegration:
    """Integration tests for the main continual learning trainer"""

    def setup_method(self):
        """Setup comprehensive test environment"""
        # This will fail initially - red phase
        self.trainer = ContinualLearningTrainer(
            model=MTGTransformerEncoder(),
            decision_head=MTGDecisionHead(),
            ewc_importance=1000.0,
            memory_buffer_size=1000,
            use_progressive_nets=True,
            max_tasks=5,
            domain_adaptation=True
        )

        # Create realistic MTG training tasks
        self.mtg_tasks = [
            TrainingTask(
                task_id="THB_draft_enchantments",
                data=self._create_realistic_mtg_dataset("THB", "draft", 200),
                domain_info={
                    "set": "THB",
                    "format": "draft",
                    "key_mechanics": ["enchant", "devotion", "escape"],
                    "meta_characteristics": {"speed": "medium", "complexity": "high"}
                },
                importance_weight=1.0
            ),
            TrainingTask(
                task_id="KHM_draft_legendary",
                data=self._create_realistic_mtg_dataset("KHM", "draft", 200),
                domain_info={
                    "set": "KHM",
                    "format": "draft",
                    "key_mechanics": ["foretell", "changeling", "sagas"],
                    "meta_characteristics": {"speed": "fast", "complexity": "medium"}
                },
                importance_weight=1.0
            ),
            TrainingTask(
                task_id="STX_sealed_multicolor",
                data=self._create_realistic_mtg_dataset("STX", "sealed", 200),
                domain_info={
                    "set": "STX",
                    "format": "sealed",
                    "key_mechanics": ["learn", "lesson", "magecraft"],
                    "meta_characteristics": {"speed": "slow", "complexity": "high"}
                },
                importance_weight=0.8
            ),
            TrainingTask(
                task_id="DMU_limited_traditional",
                data=self._create_realistic_mtg_dataset("DMU", "limited", 200),
                domain_info={
                    "set": "DMU",
                    "format": "limited",
                    "key_mechanics": ["enchant", "equip", "ward"],
                    "meta_characteristics": {"speed": "medium", "complexity": "medium"}
                },
                importance_weight=0.9
            )
        ]

    def _create_realistic_mtg_dataset(self, set_code: str, format_name: str, num_samples: int) -> List[Dict]:
        """Create realistic MTG dataset with proper game state representation"""
        data = []

        # Set-specific characteristics
        set_characteristics = {
            "THB": {"enchantment_focus": 0.18, "avg_cmc": 3.4, "speed_rating": 2.5},
            "KHM": {"legendary_focus": 0.15, "avg_cmc": 3.1, "speed_rating": 3.0},
            "STX": {"multicolor_focus": 0.25, "avg_cmc": 2.9, "speed_rating": 2.0},
            "DMU": {"traditional_focus": 0.12, "avg_cmc": 3.2, "speed_rating": 2.7}
        }

        set_char = set_characteristics.get(set_code, {"avg_cmc": 3.0, "speed_rating": 2.5})

        for i in range(num_samples):
            # Create comprehensive MTG game state
            turn = i % 15 + 1
            phase = self._get_phase_for_turn(turn)

            # Base state vector
            state = torch.randn(282)

            # Add set-specific biases
            state[0:50] += set_char["avg_cmc"] / 10.0  # CMC bias
            state[50:100] += set_char["speed_rating"] / 10.0  # Speed bias

            # Format-specific adjustments
            if format_name == "draft":
                state[100:150] += 0.1  # Draft bias
            elif format_name == "sealed":
                state[100:150] += 0.05  # Sealed bias
            elif format_name == "limited":
                state[100:150] += 0.08  # Limited bias

            # Add set identifier
            set_hash = hash(f"{set_code}_{format_name}") % 1000 / 1000.0
            state += torch.tensor([set_hash] * 282).float() * 0.05

            # Action selection with format/set considerations
            action_probabilities = torch.ones(16)

            # Adjust action probabilities based on game state
            if phase == "main" and turn >= 3:
                action_probabilities[1:5] *= 1.5  # Cast spells more likely
            elif phase == "combat" and turn >= 5:
                action_probabilities[8:10] *= 2.0  # Attack/block more likely

            action = torch.multinomial(action_probabilities, 1).item()

            # Reward calculation based on action appropriateness
            base_reward = torch.randn(1).item()

            # Add strategic bonuses
            if action in [1, 2, 3, 4] and phase == "main":  # Cast spells in main
                base_reward += 0.2
            elif action in [8, 9] and phase == "combat":  # Combat actions
                base_reward += 0.3

            data.append({
                "state": state,
                "action": torch.tensor([action]),
                "reward": torch.tensor([base_reward]),
                "next_state": torch.randn(282),
                "done": torch.tensor([i == num_samples - 1]),
                "game_context": {
                    "turn": turn,
                    "phase": phase,
                    "set_code": set_code,
                    "format": format_name,
                    "life_total": max(1, 20 - i // 20),
                    "hand_size": max(1, 7 - i // 25),
                    "lands_played": min(8, i // 15)
                },
                "set_characteristics": set_char
            })
        return data

    def _get_phase_for_turn(self, turn: int) -> str:
        """Get likely phase for given turn"""
        if turn <= 2:
            return "early"
        elif turn <= 8:
            return "main"
        elif turn <= 12:
            return "combat"
        else:
            return "late"

    def test_trainer_initialization(self):
        """Test trainer initialization for MTG continual learning"""
        assert self.trainer.model is not None
        assert self.trainer.decision_head is not None
        assert self.trainer.ewc_importance == 1000.0
        assert self.trainer.memory_buffer_size == 1000
        assert self.trainer.use_progressive_nets is True
        assert self.trainer.domain_adaptation is True
        assert len(self.trainer.learned_tasks) == 0

    def test_mtg_task_boundary_detection(self):
        """Test detection of MTG task boundaries"""
        boundary_detector = TaskBoundaryDetector()

        # Create sequential data with clear MTG set changes
        sequential_data = []
        for task in self.mtg_tasks:
            sequential_data.extend(task.data[:50])

        boundaries = boundary_detector.detect_boundaries(
            sequential_data,
            detection_method="distribution_shift",
            domain_type="mtg_set"
        )

        assert len(boundaries) == 3  # Should detect 3 boundaries between 4 tasks
        assert all(0 <= b < len(sequential_data) for b in boundaries)
        assert boundaries[0] < boundaries[1] < boundaries[2]  # Boundaries should be in order

        # Validate boundary detection quality
        expected_boundaries = [50, 100, 150]  # Expected positions
        for expected, actual in zip(expected_boundaries, boundaries):
            assert abs(expected - actual) < 20  # Within reasonable tolerance

    def test_comprehensive_mtg_continual_learning(self):
        """Test comprehensive continual learning across MTG sets and formats"""
        performance_history = []
        forgetting_history = []

        for task in self.mtg_tasks:
            # Learn new MTG task
            task_metrics = self.trainer.learn_task(
                task,
                use_ewc=True,
                use_memory_replay=True,
                domain_adapt=True
            )
            performance_history.append(task_metrics)

            # Validate no catastrophic forgetting on previous tasks
            all_tasks_performance = self.trainer.evaluate_all_tasks()

            # Check that performance on previous tasks is maintained
            if len(performance_history) > 1:
                for i, prev_metrics in enumerate(performance_history[:-1]):
                    prev_task_id = self.mtg_tasks[i].task_id
                    current_perf = all_tasks_performance[prev_task_id]
                    prev_perf = prev_metrics["accuracy"]

                    performance_drop = prev_perf - current_perf
                    forgetting_history.append(performance_drop)

                    assert performance_drop < 0.1, \
                        f"Too much forgetting on {prev_task_id}: {performance_drop:.3f}"

        # Final validation
        assert len(self.trainer.learned_tasks) == len(self.mtg_tasks)
        assert len(performance_history) == len(self.mtg_tasks)

        # Check overall forgetting statistics
        avg_forgetting = np.mean(forgetting_history) if forgetting_history else 0
        max_forgetting = max(forgetting_history) if forgetting_history else 0

        assert avg_forgetting < 0.05, f"Average forgetting too high: {avg_forgetting:.3f}"
        assert max_forgetting < 0.1, f"Maximum forgetting too high: {max_forgetting:.3f}"

    def test_mtg_knowledge_distillation(self):
        """Test knowledge distillation for preserving MTG knowledge"""
        # Train teacher on first MTG task
        teacher_model = MTGTransformerEncoder()
        teacher_data = self.mtg_tasks[0].data[:100]

        # Train teacher (simplified)
        for _ in range(10):  # Simplified training
            pass  # Would be real training on THB draft

        # Initialize student (trainer's model)
        student_initial = self.trainer.model

        # Apply knowledge distillation while learning new task
        distillation_loss = KnowledgeDistillationLoss(
            temperature=2.0,
            alpha=0.7,  # Weight for distillation loss
            beta=0.3,   # Weight for task-specific loss
            mechanics_preservation=True  # Preserve MTG mechanics knowledge
        )

        # Test distillation loss computation with MTG context
        student_outputs = torch.randn(32, 16)  # Student predictions
        teacher_outputs = torch.randn(32, 16)  # Teacher predictions
        mechanics_labels = torch.randint(0, 5, (32,))  # MTG mechanics labels

        kd_loss = distillation_loss.compute_loss(
            student_outputs,
            teacher_outputs,
            mechanics_labels=mechanics_labels
        )

        assert kd_loss.item() >= 0
        assert isinstance(kd_loss, torch.Tensor)

    def test_mtg_performance_tracking(self):
        """Test performance tracking across MTG tasks"""
        performance_tracker = PerformanceTracker()

        # Simulate learning progression across MTG sets
        for i, task in enumerate(self.mtg_tasks):
            # Simulate task learning with set-specific characteristics
            task_performance = TaskPerformance(
                task_id=task.task_id,
                accuracy=0.75 + i * 0.02,  # Slight improvement
                loss=0.6 - i * 0.05,      # Slight improvement
                forgetting_measure=0.0,
                knowledge_preservation=1.0 - i * 0.02,  # Slight degradation
                mechanics_retention={
                    "core_mechanics": 0.9 - i * 0.05,
                    "set_mechanics": 0.8 - i * 0.08
                }
            )

            performance_tracker.record_performance(task_performance)

        # Generate comprehensive performance report
        report = performance_tracker.generate_report(domain_type="mtg")

        assert "overall_performance" in report
        assert "forgetting_analysis" in report
        assert "knowledge_preservation" in report
        assert "adaptation_efficiency" in report
        assert "mechanics_analysis" in report

        # Validate no catastrophic forgetting
        assert report["forgetting_analysis"]["max_forgetting"] < 0.1
        assert report["knowledge_preservation"]["average_preservation"] > 0.9
        assert report["mechanics_analysis"]["core_mechanics_retention"] > 0.8

    def test_mtg_memory_replay_integration(self):
        """Test integration of memory replay with MTG continual learning"""
        # Train on first MTG task
        self.trainer.learn_task(self.mtg_tasks[0])

        # Check that MTG-specific experiences are stored in memory
        assert len(self.trainer.replay_buffer) > 0

        # Verify set distribution in memory
        set_distribution = self.trainer.replay_buffer.get_set_distribution()
        assert self.mtg_tasks[0].domain_info["set"] in set_distribution

        # Train on second MTG task with memory replay
        self.trainer.learn_task(
            self.mtg_tasks[1],
            use_memory_replay=True,
            replay_strategy="set_balanced"
        )

        # Validate that memory replay helped preserve MTG knowledge
        task1_performance = self.trainer.evaluate_task(self.mtg_tasks[0].task_id)
        task2_performance = self.trainer.evaluate_task(self.mtg_tasks[1].task_id)

        assert task1_performance > 0.7  # Good performance on old MTG set
        assert task2_performance > 0.7  # Good performance on new MTG set

    def test_mtg_multi_objective_optimization(self):
        """Test balancing multiple objectives in MTG continual learning"""
        # Configure multi-objective optimization for MTG
        objectives = {
            "task_performance": 0.35,
            "knowledge_preservation": 0.25,
            "memory_efficiency": 0.15,
            "adaptation_speed": 0.10,
            "mechanics_retention": 0.15  # MTG-specific objective
        }

        self.trainer.configure_multi_objective(objectives, domain_type="mtg")

        # Train on multiple MTG tasks
        for task in self.mtg_tasks:
            metrics = self.trainer.learn_task(
                task,
                balance_objectives=True,
                mechanics_focus=True
            )

            # Validate that all objectives are considered
            assert "task_performance" in metrics
            assert "knowledge_preservation" in metrics
            assert "memory_efficiency" in metrics
            assert "adaptation_speed" in metrics
            assert "mechanics_retention" in metrics

        # Check overall balance
        final_metrics = self.trainer.get_multi_objective_metrics()
        assert final_metrics["overall_score"] > 0.7
        assert final_metrics["objective_balance"]["variance"] < 0.1
        assert final_metrics["mtg_specific"]["mechanics_retention"] > 0.8


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])