"""
Comprehensive Unit Tests for Conservative Q-Learning Algorithm

This test suite validates the Conservative Q-Learning (CQL) algorithm implementation
for User Story 1 - Enhanced AI Decision Quality (Task T026).

These tests are designed to FAIL initially (Red-Green-Refactor approach) and test:

1. Conservative Q-Learning algorithm mathematical properties
2. Algorithm convergence and stability
3. Performance validation for sub-100ms inference
4. 17Lands data integration
5. Graceful degradation to supervised baseline
6. Statistical validation requirements

Constitutional Requirements:
- Data-Driven AI Development: Optimized for 17Lands offline data
- Real-Time Responsiveness: Sub-100ms inference latency
- Explainable AI: Attention mechanisms and decision rationale
- Verifiable Testing: Comprehensive validation and monitoring

Author: Claude Code
Task: T026 - Conservative Q-Learning Implementation Tests
Date: 2025-11-09
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import time
import json
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass

# Import the actual CQL implementation (these will work once implementation exists)
try:
    from src.rl.algorithms.cql import (
        ConservativeQLearning,
        ConservativeQLearningConfig,
        CQLNetwork,
        create_conservative_q_learning
    )
    from src.rl.algorithms.base import (
        RLState,
        RLAction,
        RLTransition,
        BaseQAlgorithm
    )
    CQL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CQL modules not available: {e}")
    CQL_AVAILABLE = False
    # Create mock classes for failing tests
    ConservativeQLearning = Mock
    ConservativeQLearningConfig = Mock
    CQLNetwork = Mock
    RLState = Mock
    RLAction = Mock
    RLTransition = Mock


class TestConservativeQLearningConfig:
    """Test Conservative Q-Learning configuration validation."""

    def test_cql_config_initialization_with_defaults(self):
        """Test CQL config initialization with default values."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test basic config initialization (this should work)
        config = ConservativeQLearningConfig()

        # Test constitutional requirements
        assert config.max_inference_time_ms == 100.0  # NON-NEGOTIABLE requirement
        assert config.enable_explainability == True
        assert config.enable_performance_monitoring == True
        assert config.enable_validation_checkpoints == True

        # Test CQL-specific defaults
        assert config.cql_alpha == 5.0
        assert config.cql_temp == 1.0
        assert config.cql_lagrange == True

        # Test network defaults
        assert config.hidden_dims == [512, 256, 128]
        assert config.dueling_architecture == True
        assert config.attention_heads == 4

        # Test training defaults
        assert config.batch_size == 32
        assert config.buffer_size == 100000
        assert config.learning_rate == 1e-4

    def test_cql_config_with_custom_parameters(self):
        """Test CQL config initialization with custom parameters (RED test)."""
        """This test will FAIL initially because the current config class doesn't accept parameters."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # This should work but currently fails because __init__ doesn't accept parameters
        # This test drives the implementation improvement
        config = ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            cql_alpha=10.0,
            learning_rate=5e-4
        )

        # These assertions should pass once the implementation is improved
        assert config.state_dim == 282
        assert config.action_dim == 16
        assert config.cql_alpha == 10.0
        assert config.learning_rate == 5e-4

    def test_cql_config_validation_constitutional_requirements(self):
        """Test that config validation enforces constitutional requirements (RED test)."""
        """This test will FAIL initially because the current config doesn't validate requirements."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test inference time violation - should raise ValueError
        with pytest.raises(ValueError, match="Max inference time .* exceeds 100ms requirement"):
            ConservativeQLearningConfig(max_inference_time_ms=150.0)

        # Test explainability requirement - should raise ValueError
        with pytest.raises(ValueError, match="Explainability disabled"):
            ConservativeQLearningConfig(enable_explainability=False)

        # Test performance monitoring requirement - should raise ValueError
        with pytest.raises(ValueError, match="Performance monitoring disabled"):
            ConservativeQLearningConfig(enable_performance_monitoring=False)

    def test_cql_config_parameter_validation(self):
        """Test CQL-specific parameter validation."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test invalid CQL alpha
        with pytest.raises(ValueError, match="CQL alpha must be positive"):
            ConservativeQLearningConfig(cql_alpha=-1.0)

        # Test invalid temperature
        with pytest.raises(ValueError, match="CQL temperature must be positive"):
            ConservativeQLearningConfig(cql_temp=0.0)

        # Test invalid action dimensions
        with pytest.raises(ValueError, match="Action dimension must be positive"):
            ConservativeQLearningConfig(action_dim=0)

        # Test invalid state dimensions
        with pytest.raises(ValueError, match="State dimension must be positive"):
            ConservativeQLearningConfig(state_dim=0)


class TestCQLNetwork:
    """Test CQL neural network architecture."""

    @pytest.fixture
    def network_config(self):
        """Create network configuration for testing."""
        return {
            'state_dim': 282,  # MTG state dimension
            'action_dim': 16,  # MTG action space
            'hidden_dims': [128, 64],
            'dueling_architecture': True,
            'attention_heads': 4
        }

    def test_cql_network_initialization(self, network_config):
        """Test CQL network initialization."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        network = CQLNetwork(**network_config)

        # Check architecture components
        assert network.state_dim == network_config['state_dim']
        assert network.action_dim == network_config['action_dim']
        assert network.dueling_architecture == network_config['dueling_architecture']
        assert network.attention_heads == network_config['attention_heads']

        # Check network components exist
        assert hasattr(network, 'feature_layers')
        assert hasattr(network, 'value_stream')  # Dueling architecture
        assert hasattr(network, 'advantage_stream')  # Dueling architecture
        assert hasattr(network, 'attention')  # Attention mechanism

        # Test forward pass
        batch_size = 8
        states = torch.randn(batch_size, network_config['state_dim'])
        q_values, attention_weights = network(states)

        assert q_values.shape == (batch_size, network_config['action_dim'])
        assert attention_weights is not None  # Should return attention weights

    def test_cql_network_attention_mechanism(self, network_config):
        """Test attention mechanism for explainability."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        network = CQLNetwork(**network_config)

        # Test attention computation
        state = torch.randn(1, network_config['state_dim'])
        q_values, attention_weights = network(state)

        assert attention_weights is not None
        assert attention_weights.dim() >= 2  # Should have batch and sequence dimensions

        # Test attention importance extraction
        importance = network.get_attention_importance(state)
        assert importance.shape[-1] == network_config['state_dim']
        assert torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-6)

    def test_cql_network_dueling_architecture(self, network_config):
        """Test dueling DQN architecture implementation."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test with dueling architecture
        network_dueling = CQLNetwork(**network_config)
        states = torch.randn(4, network_config['state_dim'])
        q_values_dueling, _ = network_dueling(states)

        # Test without dueling architecture
        network_standard = CQLNetwork(**{**network_config, 'dueling_architecture': False})
        q_values_standard, _ = network_standard(states)

        # Both should produce valid Q-values
        assert q_values_dueling.shape == q_values_standard.shape
        assert not torch.isnan(q_values_dueling).any()
        assert not torch.isnan(q_values_standard).any()


class TestConservativeQLearningCore:
    """Test Conservative Q-Lening core algorithm functionality."""

    @pytest.fixture
    def cql_config(self):
        """Create CQL configuration for testing."""
        return ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            hidden_dims=[128, 64],
            learning_rate=1e-3,
            batch_size=32,
            cql_alpha=5.0,
            cql_temp=1.0,
            cql_lagrange=True,
            target_update_freq=10,
            max_inference_time_ms=100.0  # Constitutional requirement
        )

    @pytest.fixture
    def cql_agent(self, cql_config):
        """Create CQL agent for testing."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")
        # Use the correct constructor signature
        return ConservativeQLearning(state_dim=cql_config.state_dim, action_dim=cql_config.action_dim, config=cql_config)

    @pytest.fixture
    def sample_mtg_transitions(self):
        """Create realistic MTG transitions for testing."""
        batch_size = 32
        state_dim = 282
        action_dim = 16

        transitions = []
        for i in range(batch_size):
            # Create realistic MTG state
            state_vector = self._create_mtg_state_vector(state_dim)
            state = RLState(
                state_vector=state_vector,
                state_metadata={'turn': i % 20 + 1, 'phase': 'main'},
                timestamp=time.time(),
                turn_number=i % 20 + 1,
                phase='main',
                valid_actions=list(range(action_dim))
            )

            # Create action
            action_id = np.random.randint(0, action_dim)
            action = RLAction(
                action_id=action_id,
                action_type=f"mtg_action_{action_id}",
                action_parameters={},
                confidence=0.8,
                explanation="Test action"
            )

            # Create next state
            next_state_vector = self._create_mtg_state_vector(state_dim)
            next_state = RLState(
                state_vector=next_state_vector,
                state_metadata={'turn': (i % 20) + 2, 'phase': 'combat'},
                timestamp=time.time() + 1.0,
                turn_number=(i % 20) + 2,
                phase='combat'
            )

            # Create transition
            transition = RLTransition(
                state=state,
                action=action,
                reward=np.random.randn(),
                next_state=next_state,
                done=np.random.choice([True, False], p=[0.1, 0.9]),
                episode_id=f"test_episode_{i // 10}",
                step_number=i
            )
            transitions.append(transition)

        return transitions

    def _create_mtg_state_vector(self, dim: int) -> torch.Tensor:
        """Create realistic MTG state vector."""
        state = torch.zeros(dim)

        # Board state representation (first 64 dims)
        state[:64] = torch.randn(64) * 0.1

        # Hand and mana representation (next 128 dims)
        state[64:192] = torch.abs(torch.randn(128) * 0.1)

        # Phase and priority (next 48 dims)
        state[192:240] = torch.randn(48) * 0.1

        # Additional features (remaining dims)
        state[240:] = torch.randn(dim - 240) * 0.1

        return state

    def test_cql_initialization(self, cql_config):
        """Test CQL agent initialization (RED test)."""
        """This test will FAIL initially because the current constructor expects different parameters."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Current implementation expects: ConservativeQLearning(state_dim, action_dim, config)
        # But our test passes only config
        agent = ConservativeQLearning(cql_config)

        # Check core components
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.optimizer is not None

        # Check configuration
        assert agent.state_dim == cql_config.state_dim
        assert agent.action_dim == cql_config.action_dim
        assert agent.cql_alpha == cql_config.cql_alpha

        # Check training state
        assert agent.training_step == 0
        assert hasattr(agent, 'loss_history')
        assert isinstance(agent.loss_history, list)

    def test_cql_initialization_with_correct_signature(self):
        """Test CQL agent initialization with correct constructor signature."""
        """This test should work with the current implementation."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Use the correct constructor signature that current implementation expects
        config = ConservativeQLearningConfig()
        agent = ConservativeQLearning(state_dim=282, action_dim=16, config=config)

        # Check basic initialization
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.optimizer is not None
        assert agent.state_dim == 282
        assert agent.action_dim == 16
        assert agent.cql_alpha == config.cql_alpha

    def test_cql_action_selection_performance(self, cql_agent):
        """Test action selection meets sub-100ms performance requirement."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Create test state
        state = self._create_mtg_state_vector(cql_agent.state_dim)
        rl_state = RLState(
            state_vector=state,
            state_metadata={},
            timestamp=time.time(),
            turn_number=5,
            phase='main'
        )

        # Measure inference time
        start_time = time.time()
        action = cql_agent.select_action(rl_state)
        inference_time_ms = (time.time() - start_time) * 1000

        # Constitutional requirement: sub-100ms inference
        assert inference_time_ms < 100.0, f"Inference time {inference_time_ms:.2f}ms exceeds 100ms requirement"

        # Validate action structure
        assert isinstance(action, RLAction)
        assert 0 <= action.action_id < cql_agent.action_dim
        assert 0.0 <= action.confidence <= 1.0
        assert action.explanation is not None

    def test_cql_conservative_loss_computation(self, cql_agent, sample_mtg_transitions):
        """Test Conservative Q-Learning loss computation and mathematical properties (RED test)."""
        """This test will FAIL initially because _prepare_batch method might not exist or work differently."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Prepare batch - this method might not exist in current implementation
        batch = cql_agent._prepare_batch(sample_mtg_transitions[:cql_agent.cql_config.batch_size])

        # Get Q-values
        with torch.no_grad():
            q_values, _ = cql_agent.q_network(batch['states'])

        # Compute CQL loss - this method might not exist in current implementation
        cql_loss = cql_agent._calculate_cql_loss(batch['states'], q_values)

        # Validate mathematical properties
        assert isinstance(cql_loss, torch.Tensor)
        assert cql_loss.dim() == 0  # Scalar loss
        assert not torch.isnan(cql_loss)
        assert cql_loss >= 0  # Loss should be non-negative

        # Test that conservative loss penalizes overestimation
        # Higher Q-values should result in higher conservative loss
        high_q_values = q_values + 10.0  # Artificially inflate Q-values
        high_cql_loss = cql_agent._calculate_cql_loss(batch['states'], high_q_values)
        assert high_cql_loss > cql_loss

    def test_cql_mathematical_properties_direct(self):
        """Test CQL mathematical properties directly with simpler implementation."""
        """This test demonstrates the expected mathematical behavior of CQL."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test the core mathematical property: conservative loss should penalize overestimation
        batch_size = 16
        state_dim = 282
        action_dim = 16

        # Create test data
        states = torch.randn(batch_size, state_dim)
        q_values = torch.randn(batch_size, action_dim)

        # Simulate conservative loss computation
        # The CQL loss should be: E[log Σ_a exp(Q(s,a)/τ) - Q(s,π(s))/τ]
        temp = 1.0
        log_sum_exp = torch.logsumexp(q_values / temp, dim=1)
        current_q = q_values.gather(1, q_values.argmax(dim=1, keepdim=True)).squeeze()
        conservative_loss = (log_sum_exp - current_q / temp).mean()

        # Validate mathematical properties
        assert conservative_loss >= 0, "Conservative loss should be non-negative"
        assert not torch.isnan(conservative_loss), "Conservative loss should not be NaN"

        # Test with higher Q-values - should increase conservative loss
        high_q_values = q_values + 5.0
        high_log_sum_exp = torch.logsumexp(high_q_values / temp, dim=1)
        high_current_q = high_q_values.gather(1, high_q_values.argmax(dim=1, keepdim=True)).squeeze()
        high_conservative_loss = (high_log_sum_exp - high_current_q / temp).mean()

        assert high_conservative_loss > conservative_loss, "Higher Q-values should increase conservative loss"

        print(f"✓ CQL mathematical properties validated: loss={conservative_loss:.4f}, high_loss={high_conservative_loss:.4f}")

    def test_cql_training_step(self, cql_agent, sample_mtg_transitions):
        """Test single CQL training step with all loss components."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        initial_step = cql_agent.training_step
        initial_alpha = cql_agent.cql_alpha

        # Perform training step
        metrics = cql_agent.update(sample_mtg_transitions[:cql_agent.cql_config.batch_size])

        # Validate training metrics
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert 'q_loss' in metrics
        assert 'cql_loss' in metrics
        assert 'alpha' in metrics

        # Check loss values are valid
        assert not np.isnan(metrics['total_loss'])
        assert metrics['total_loss'] >= 0
        assert not np.isnan(metrics['q_loss'])
        assert metrics['q_loss'] >= 0
        assert not np.isnan(metrics['cql_loss'])
        assert metrics['cql_loss'] >= 0

        # Check training step increment
        assert cql_agent.training_step == initial_step + 1

        # Check alpha parameter (if using Lagrange multiplier)
        if cql_agent.cql_config.cql_lagrange:
            assert metrics['alpha'] != initial_alpha  # Alpha should be updated

    def test_cql_target_network_update(self, cql_agent, sample_mtg_transitions):
        """Test target network update mechanism."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Get initial target network parameters
        initial_target_params = [p.clone() for p in cql_agent.target_network.parameters()]

        # Perform training steps to trigger target update
        update_freq = cql_agent.cql_config.target_update_freq
        for i in range(update_freq + 1):
            metrics = cql_agent.update(sample_mtg_transitions[:cql_agent.cql_config.batch_size])

            # Check if target network should update
            should_update = (i + 1) % update_freq == 0

        # Verify target network was updated at correct frequency
        params_changed = False
        for init_param, current_param in zip(initial_target_params, cql_agent.target_network.parameters()):
            if not torch.allclose(init_param, current_param, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Target network should be updated after specified frequency"

    def test_cql_alpha_lagrange_update(self, cql_agent, sample_mtg_transitions):
        """Test automatic alpha tuning using Lagrange multiplier."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        if not cql_agent.cql_config.cql_lagrange:
            pytest.skip("Lagrange multiplier not enabled")

        initial_alpha = cql_agent.cql_alpha

        # Perform multiple training steps to observe alpha updates
        alpha_values = []
        for _ in range(10):
            metrics = cql_agent.update(sample_mtg_transitions[:cql_agent.cql_config.batch_size])
            alpha_values.append(metrics['alpha'])

        # Alpha should be changing (automatic tuning)
        alpha_changes = [abs(alpha_values[i] - alpha_values[i-1]) for i in range(1, len(alpha_values))]
        assert any(change > 1e-6 for change in alpha_changes), "Alpha parameter should be updated automatically"

        # Alpha should remain positive
        assert all(alpha > 0 for alpha in alpha_values), "Alpha must remain positive"

    def test_cql_explainability_features(self, cql_agent):
        """Test explainability features for constitutional compliance."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Create test state
        state = self._create_mtg_state_vector(cql_agent.state_dim)
        rl_state = RLState(
            state_vector=state,
            state_metadata={},
            timestamp=time.time()
        )

        # Select action to generate explainability data
        action = cql_agent.select_action(rl_state)

        # Get explainability data
        explainability_data = cql_agent.get_explainability_data()

        # Validate explainability requirements
        assert isinstance(explainability_data, dict)
        assert 'algorithm_type' in explainability_data
        assert explainability_data['algorithm_type'] == 'Conservative Q-Learning'
        assert 'cql_alpha' in explainability_data
        assert 'attention_heads' in explainability_data
        assert 'training_step' in explainability_data

        # Test attention weights
        attention_weights = cql_agent.get_attention_weights(rl_state)
        if cql_agent.cql_config.attention_heads > 0:
            assert attention_weights is not None
            assert isinstance(attention_weights, torch.Tensor)

    def test_cql_model_checkpointing(self, cql_agent, sample_mtg_transitions):
        """Test model saving and loading functionality."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Perform some training to create state
        for _ in range(5):
            cql_agent.update(sample_mtg_transitions[:cql_agent.cql_config.batch_size])

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name

        try:
            cql_agent.save_model(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            # Create new agent and load checkpoint
            new_agent = ConservativeQLearning(cql_agent.cql_config)
            new_agent.load_model(checkpoint_path)

            # Verify loaded state
            assert new_agent.training_step == cql_agent.training_step
            assert new_agent.cql_alpha == cql_agent.cql_alpha

            # Verify network parameters match
            for orig_param, loaded_param in zip(cql_agent.q_network.parameters(), new_agent.q_network.parameters()):
                assert torch.allclose(orig_param, loaded_param, atol=1e-6)

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


class TestCQL17LandsIntegration:
    """Test CQL integration with 17Lands replay data."""

    @pytest.fixture
    def mock_17lands_data(self):
        """Create mock 17Lands-style replay data."""
        return {
            'games': [
                {
                    'game_id': '17lands_game_001',
                    'players': [
                        {
                            'player_id': 'test_player',
                            'deck_colors': ['W', 'U'],
                            'turns': [
                                {
                                    'turn_number': 1,
                                    'actions': [
                                        {
                                            'action_type': 'play_land',
                                            'card_name': 'Plains',
                                            'timestamp': '2024-01-01T10:00:01Z',
                                            'game_state': {
                                                'life_total': 20,
                                                'hand_size': 6,
                                                'board_state': [],
                                                'mana_available': {'W': 1}
                                            }
                                        }
                                    ]
                                }
                            ],
                            'outcome': 'win',
                            'game_duration_seconds': 300
                        }
                    ]
                }
            ]
        }

    def test_cql_17lands_data_processing(self, mock_17lands_data):
        """Test processing of 17Lands replay data for CQL training."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # This test validates that CQL can process real 17Lands data
        # Implementation would need to convert 17Lands format to RLTransition objects

        config = ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            batch_size=16
        )
        agent = ConservativeQLearning(config)

        # Mock the data processing pipeline
        transitions = self._convert_17lands_to_transitions(mock_17lands_data)

        # Validate data conversion
        assert len(transitions) > 0
        for transition in transitions:
            assert isinstance(transition, RLTransition)
            assert transition.state.state_vector.shape[0] == config.state_dim
            assert 0 <= transition.action.action_id < config.action_dim

        # Test training on 17Lands data
        if len(transitions) >= config.batch_size:
            metrics = agent.update(transitions[:config.batch_size])
            assert 'total_loss' in metrics
            assert not np.isnan(metrics['total_loss'])

    def _convert_17lands_to_transitions(self, lands_data: Dict) -> List[RLTransition]:
        """Convert 17Lands data format to RLTransition objects."""
        transitions = []

        for game in lands_data['games']:
            for player in game['players']:
                for turn in player['turns']:
                    for i, action in enumerate(turn['actions']):
                        # Convert game state to state vector
                        state_vector = self._game_state_to_vector(action['game_state'])
                        state = RLState(
                            state_vector=state_vector,
                            state_metadata=action['game_state'],
                            timestamp=time.time(),
                            turn_number=turn['turn_number'],
                            phase='main'
                        )

                        # Convert action to RLAction
                        action_id = self._action_type_to_id(action['action_type'])
                        rl_action = RLAction(
                            action_id=action_id,
                            action_type=action['action_type'],
                            action_parameters={'card_name': action.get('card_name', '')},
                            confidence=1.0,  # Historical data is certain
                            explanation="Historical action from 17Lands data"
                        )

                        # Create next state (simplified)
                        next_state = RLState(
                            state_vector=state_vector + torch.randn_like(state_vector) * 0.01,
                            state_metadata=action['game_state'],
                            timestamp=time.time() + 1.0,
                            turn_number=turn['turn_number'] + 1
                        )

                        # Create transition
                        transition = RLTransition(
                            state=state,
                            action=rl_action,
                            reward=0.0,  # Would be calculated from game outcome
                            next_state=next_state,
                            done=False,
                            episode_id=game['game_id'],
                            step_number=i
                        )
                        transitions.append(transition)

        return transitions

    def _game_state_to_vector(self, game_state: Dict) -> torch.Tensor:
        """Convert game state dictionary to state vector."""
        # Simplified conversion - real implementation would be more sophisticated
        vector = torch.zeros(282)

        # Life total (dimension 0)
        vector[0] = game_state.get('life_total', 20) / 20.0

        # Hand size (dimension 1)
        vector[1] = game_state.get('hand_size', 0) / 7.0

        # Mana availability (dimensions 2-6 for WUBRG)
        mana = game_state.get('mana_available', {})
        colors = ['W', 'U', 'B', 'R', 'G']
        for i, color in enumerate(colors):
            vector[2 + i] = mana.get(color, 0) / 10.0

        return vector

    def _action_type_to_id(self, action_type: str) -> int:
        """Convert action type string to action ID."""
        action_map = {
            'play_land': 0,
            'cast_creature': 1,
            'cast_sorcery': 2,
            'cast_instant': 3,
            'activate_ability': 4,
            'attack': 5,
            'block': 6,
            'pass_priority': 7
        }
        return action_map.get(action_type, 15)  # Default to 15 (other)


class TestCQLPerformanceValidation:
    """Test CQL performance validation and constitutional compliance."""

    @pytest.fixture
    def cql_config(self):
        """Create CQL configuration with performance monitoring."""
        return ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            max_inference_time_ms=100.0,  # Constitutional requirement
            enable_performance_monitoring=True,
            enable_validation_checkpoints=True
        )

    def test_cql_inference_performance_compliance(self, cql_config):
        """Test that inference meets constitutional performance requirements."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Test multiple inferences to check average performance
        inference_times = []
        num_tests = 100

        for _ in range(num_tests):
            state = torch.randn(cql_config.state_dim)
            rl_state = RLState(
                state_vector=state,
                state_metadata={},
                timestamp=time.time()
            )

            start_time = time.time()
            action = agent.select_action(rl_state)
            inference_time_ms = (time.time() - start_time) * 1000
            inference_times.append(inference_time_ms)

            # Each inference must meet the requirement
            assert inference_time_ms < cql_config.max_inference_time_ms, \
                f"Single inference took {inference_time_ms:.2f}ms, exceeding {cql_config.max_inference_time_ms}ms limit"

        # Check average performance
        avg_time = np.mean(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)

        # Constitutional performance requirements
        assert avg_time < cql_config.max_inference_time_ms, \
            f"Average inference time {avg_time:.2f}ms exceeds requirement"
        assert p95_time < cql_config.max_inference_time_ms * 1.5, \
            f"P95 inference time {p95_time:.2f}ms exceeds relaxed requirement"

    def test_cql_memory_usage_validation(self, cql_config):
        """Test that CQL stays within memory limits."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create agent and perform training
        agent = ConservativeQLearning(cql_config)

        # Generate training data
        transitions = []
        for _ in range(1000):  # Large dataset
            state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time()
            )
            action = RLAction(
                action_id=np.random.randint(0, cql_config.action_dim),
                action_type="test_action",
                action_parameters={},
                confidence=0.8
            )
            next_state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time() + 1.0
            )
            transition = RLTransition(
                state=state,
                action=action,
                reward=np.random.randn(),
                next_state=next_state,
                done=False
            )
            transitions.append(transition)

        # Perform training
        for i in range(0, len(transitions), cql_config.batch_size):
            batch = transitions[i:i + cql_config.batch_size]
            if len(batch) >= cql_config.batch_size:
                agent.update(batch)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Check memory limits
        assert memory_increase < cql_config.max_memory_usage_gb * 1024, \
            f"Memory increase {memory_increase:.2f}MB exceeds limit of {cql_config.max_memory_usage_gb * 1024}MB"

    def test_cql_constitutional_compliance_validation(self, cql_config):
        """Test comprehensive constitutional compliance validation."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Perform some training to populate metrics
        transitions = []
        for _ in range(100):
            state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time()
            )
            action = RLAction(
                action_id=np.random.randint(0, cql_config.action_dim),
                action_type="test_action",
                action_parameters={},
                confidence=0.8
            )
            next_state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time() + 1.0
            )
            transition = RLTransition(
                state=state,
                action=action,
                reward=np.random.randn(),
                next_state=next_state,
                done=False
            )
            transitions.append(transition)

        for i in range(0, len(transitions), cql_config.batch_size):
            batch = transitions[i:i + cql_config.batch_size]
            if len(batch) >= cql_config.batch_size:
                agent.update(batch)

        # Validate constitutional compliance
        compliance = agent.validate_constitutional_compliance()

        # Check compliance structure
        assert isinstance(compliance, dict)
        assert 'compliant' in compliance
        assert 'violations' in compliance
        assert 'algorithm' in compliance
        assert 'validation_time' in compliance

        # Should be compliant with default configuration
        assert compliance['compliant'], f"Constitutional violations: {compliance['violations']}"
        assert len(compliance['violations']) == 0

        # Check specific compliance areas
        assert 'cql_parameters' in compliance
        assert 'training_stats' in compliance
        assert 'performance_stats' in compliance

        # Validate CQL-specific parameters
        cql_params = compliance['cql_parameters']
        assert 'alpha' in cql_params
        assert 'temp' in cql_params
        assert 'lagrange' in cql_params
        assert cql_params['alpha'] > 0
        assert cql_params['temp'] > 0


class TestCQLGracefulDegradation:
    """Test CQL graceful degradation to supervised baseline."""

    @pytest.fixture
    def cql_config(self):
        """Create CQL configuration for degradation testing."""
        return ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            enable_explainability=True,
            enable_performance_monitoring=True
        )

    def test_cql_degradation_on_insufficient_data(self, cql_config):
        """Test graceful degradation when replay buffer is insufficient."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Test with insufficient data
        insufficient_transitions = []
        for _ in range(5):  # Much less than batch_size
            state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time()
            )
            action = RLAction(
                action_id=0,
                action_type="test_action",
                action_parameters={},
                confidence=0.5
            )
            transition = RLTransition(
                state=state,
                action=action,
                reward=0.0,
                next_state=None,
                done=True
            )
            insufficient_transitions.append(transition)

        # Should handle insufficient data gracefully
        metrics = agent.update(insufficient_transitions)

        # Should return error information rather than crashing
        assert isinstance(metrics, dict)
        if 'error' in metrics:
            assert isinstance(metrics['error'], str)
        else:
            # If it doesn't return an error, it should handle the case appropriately
            assert 'total_loss' in metrics or 'warning' in metrics

    def test_cql_degradation_on_network_failure(self, cql_config):
        """Test graceful degradation when Q-network fails."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Simulate network failure by corrupting weights
        with torch.no_grad():
            for param in agent.q_network.parameters():
                param.fill_(float('nan'))

        # Create test transition
        state = RLState(
            state_vector=torch.randn(cql_config.state_dim),
            state_metadata={},
            timestamp=time.time()
        )

        # Should handle network failure gracefully
        try:
            action = agent.select_action(state)

            # If action selection succeeds, it should be a fallback action
            assert isinstance(action, RLAction)
            assert action.confidence <= 0.1  # Low confidence for fallback
            assert "fallback" in action.explanation.lower()

        except Exception as e:
            # If it raises an exception, it should be a controlled failure
            assert "fallback" in str(e).lower() or "graceful" in str(e).lower()

    def test_cql_supervised_baseline_fallback(self, cql_config):
        """Test fallback to supervised baseline when offline RL fails."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Simulate conditions that would trigger supervised fallback
        # (e.g., consistently high CQL loss indicating overestimation)

        # Create transitions with high reward variance (problematic for offline RL)
        problematic_transitions = []
        for i in range(100):
            state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={'high_variance': True},
                timestamp=time.time()
            )
            action = RLAction(
                action_id=i % cql_config.action_dim,
                action_type="noisy_action",
                action_parameters={},
                confidence=0.8
            )
            # High variance rewards
            reward = np.random.randn() * 10.0  # High variance
            next_state = RLState(
                state_vector=torch.randn(cql_config.state_dim),
                state_metadata={},
                timestamp=time.time() + 1.0
            )
            transition = RLTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False
            )
            problematic_transitions.append(transition)

        # Train on problematic data
        loss_values = []
        for i in range(0, len(problematic_transitions), cql_config.batch_size):
            batch = problematic_transitions[i:i + cql_config.batch_size]
            if len(batch) >= cql_config.batch_size:
                metrics = agent.update(batch)
                if 'total_loss' in metrics:
                    loss_values.append(metrics['total_loss'])

        # Check if system detects problematic training
        if len(loss_values) > 10:
            recent_avg = np.mean(loss_values[-5:])
            early_avg = np.mean(loss_values[:5])

            # If loss is increasing significantly, should trigger fallback
            if recent_avg > early_avg * 2.0:
                # This should trigger supervised baseline fallback
                # Implementation would switch to supervised learning mode
                assert hasattr(agent, '_supervised_mode') or hasattr(agent, 'fallback_mode')

    def test_cql_explainability_preserved_in_degradation(self, cql_config):
        """Test that explainability is preserved during degradation scenarios."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Test explainability during normal operation
        state = RLState(
            state_vector=torch.randn(cql_config.state_dim),
            state_metadata={},
            timestamp=time.time()
        )

        action = agent.select_action(state)
        explainability_data = agent.get_explainability_data()

        # Should have explainability data in normal operation
        assert isinstance(explainability_data, dict)
        assert 'algorithm_type' in explainability_data
        assert explainability_data['algorithm_type'] == 'Conservative Q-Learning'

        # Simulate degradation scenario
        with patch.object(agent.q_network, '__call__', side_effect=Exception("Network error")):
            # Should still provide explainability during fallback
            try:
                fallback_action = agent.select_action(state)
                fallback_explainability = agent.get_explainability_data()

                # Explainability should indicate fallback mode
                assert isinstance(fallback_explainability, dict)
                assert 'fallback_mode' in fallback_explainability or 'error' in fallback_explainability

            except Exception:
                # Even if action selection fails, explainability should be available
                error_explainability = agent.get_explainability_data()
                assert isinstance(error_explainability, dict)
                assert 'error' in error_explainability or 'fallback' in str(error_explainability).lower()


class TestCQLStatisticalValidation:
    """Test statistical validation requirements for CQL algorithm."""

    @pytest.fixture
    def cql_config(self):
        """Create CQL configuration for statistical testing."""
        return ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            batch_size=64,
            learning_rate=1e-3
        )

    def test_cql_convergence_statistical_validation(self, cql_config):
        """Test statistical validation of CQL convergence."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Generate consistent training data for convergence testing
        transitions = self._generate_convergence_test_data(1000, cql_config)

        # Track training metrics
        loss_history = []
        q_value_history = []

        # Training loop for convergence testing
        for epoch in range(50):  # 50 epochs
            epoch_losses = []
            epoch_q_values = []

            # Shuffle and batch data
            np.random.shuffle(transitions)
            for i in range(0, len(transitions), cql_config.batch_size):
                batch = transitions[i:i + cql_config.batch_size]
                if len(batch) >= cql_config.batch_size:
                    metrics = agent.update(batch)
                    if 'total_loss' in metrics:
                        epoch_losses.append(metrics['total_loss'])
                        if 'current_q_mean' in metrics:
                            epoch_q_values.append(metrics['current_q_mean'])

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                loss_history.append(avg_loss)

                if epoch_q_values:
                    avg_q_value = np.mean(epoch_q_values)
                    q_value_history.append(avg_q_value)

        # Statistical validation of convergence
        if len(loss_history) >= 10:
            # Loss should generally decrease (convergence)
            early_losses = loss_history[:10]
            late_losses = loss_history[-10:]

            # Perform statistical test for convergence
            from scipy import stats

            # T-test to check if late losses are significantly lower
            t_stat, p_value = stats.ttest_ind(late_losses, early_losses, alternative='less')

            # Should show statistically significant improvement (p < 0.05)
            assert p_value < 0.05, f"No statistically significant convergence: p={p_value:.4f}"

            # Trend analysis: loss should show decreasing trend
            loss_slope = np.polyfit(range(len(loss_history)), loss_history, 1)[0]
            assert loss_slope < 0, f"Loss trend should be negative: slope={loss_slope:.6f}"

    def test_cql_stability_statistical_validation(self, cql_config):
        """Test statistical validation of CQL training stability."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test multiple training runs with different random seeds
        num_runs = 5
        final_losses = []
        convergence_epochs = []

        for run in range(num_runs):
            # Set random seed for reproducibility
            torch.manual_seed(run)
            np.random.seed(run)

            agent = ConservativeQLearning(cql_config)
            transitions = self._generate_convergence_test_data(500, cql_config)

            run_losses = []

            # Training loop
            for epoch in range(30):
                epoch_losses = []
                np.random.shuffle(transitions)

                for i in range(0, len(transitions), cql_config.batch_size):
                    batch = transitions[i:i + cql_config.batch_size]
                    if len(batch) >= cql_config.batch_size:
                        metrics = agent.update(batch)
                        if 'total_loss' in metrics:
                            epoch_losses.append(metrics['total_loss'])

                if epoch_losses:
                    run_losses.append(np.mean(epoch_losses))

            final_losses.append(run_losses[-1])

            # Find convergence epoch (when loss stops improving significantly)
            for i in range(5, len(run_losses)):
                recent_avg = np.mean(run_losses[i-5:i])
                early_avg = np.mean(run_losses[max(0, i-10):i-5])
                if abs(recent_avg - early_avg) / early_avg < 0.01:  # 1% improvement threshold
                    convergence_epochs.append(i)
                    break
            else:
                convergence_epochs.append(len(run_losses))

        # Statistical validation of stability
        final_losses = np.array(final_losses)
        convergence_epochs = np.array(convergence_epochs)

        # Final losses should be consistent across runs (low variance)
        loss_cv = np.std(final_losses) / np.mean(final_losses)  # Coefficient of variation
        assert loss_cv < 0.2, f"Training results too variable: CV={loss_cv:.3f}"

        # Convergence should be reasonably consistent
        epoch_cv = np.std(convergence_epochs) / np.mean(convergence_epochs)
        assert epoch_cv < 0.3, f"Convergence too variable: CV={epoch_cv:.3f}"

    def test_cql_performance_benchmarks(self, cql_config):
        """Test performance against established benchmarks."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        agent = ConservativeQLearning(cql_config)

        # Generate benchmark data
        benchmark_transitions = self._generate_benchmark_data(1000, cql_config)

        # Performance metrics to track
        inference_times = []
        training_times = []
        memory_usage = []

        import psutil
        import os
        process = psutil.Process(os.getpid())

        # Inference performance benchmark
        for transition in benchmark_transitions[:100]:  # Test 100 inferences
            start_time = time.time()
            action = agent.select_action(transition.state)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

        # Training performance benchmark
        for i in range(0, min(len(benchmark_transitions), 500), cql_config.batch_size):
            batch = benchmark_transitions[i:i + cql_config.batch_size]
            if len(batch) >= cql_config.batch_size:
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                start_time = time.time()

                metrics = agent.update(batch)

                training_time = (time.time() - start_time) * 1000  # ms
                end_memory = process.memory_info().rss / 1024 / 1024  # MB

                training_times.append(training_time)
                memory_usage.append(end_memory - start_memory)

        # Validate against performance benchmarks
        inference_times = np.array(inference_times)
        training_times = np.array(training_times)
        memory_usage = np.array(memory_usage)

        # Inference benchmarks
        assert np.mean(inference_times) < 50.0, f"Inference too slow: {np.mean(inference_times):.2f}ms"
        assert np.percentile(inference_times, 95) < 100.0, f"P95 inference too slow: {np.percentile(inference_times, 95):.2f}ms"

        # Training benchmarks
        assert np.mean(training_times) < 200.0, f"Training too slow: {np.mean(training_times):.2f}ms per batch"

        # Memory benchmarks
        assert np.mean(memory_usage) < 10.0, f"Memory usage too high: {np.mean(memory_usage):.2f}MB per batch"
        assert np.max(memory_usage) < 50.0, f"Peak memory usage too high: {np.max(memory_usage):.2f}MB"

    def _generate_convergence_test_data(self, num_samples: int, config) -> List[RLTransition]:
        """Generate test data for convergence testing."""
        transitions = []

        for i in range(num_samples):
            # Create states with some structure
            state_vector = torch.randn(config.state_dim)
            state_vector[0] = i / num_samples  # Progress indicator

            state = RLState(
                state_vector=state_vector,
                state_metadata={'sample_id': i, 'progress': i / num_samples},
                timestamp=time.time(),
                turn_number=i % 20 + 1
            )

            # Create action with some logic
            action_id = int(state_vector[0].item() * config.action_dim) % config.action_dim
            action = RLAction(
                action_id=action_id,
                action_type=f"convergence_action_{action_id}",
                action_parameters={},
                confidence=0.9,
                explanation=f"Action based on state progress {i/num_samples:.2f}"
            )

            # Create next state with some correlation
            next_state_vector = state_vector + torch.randn_like(state_vector) * 0.1
            next_state = RLState(
                state_vector=next_state_vector,
                state_metadata={'sample_id': i + 1, 'progress': (i + 1) / num_samples},
                timestamp=time.time() + 1.0,
                turn_number=(i % 20) + 2
            )

            # Reward with some structure (correlated with action quality)
            reward = float(state_vector[action_id]) + np.random.randn() * 0.1

            transition = RLTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(i == num_samples - 1),
                episode_id=f"convergence_test",
                step_number=i
            )
            transitions.append(transition)

        return transitions

    def _generate_benchmark_data(self, num_samples: int, config) -> List[RLTransition]:
        """Generate benchmark data for performance testing."""
        transitions = []

        for i in range(num_samples):
            # Realistic MTG-like state
            state_vector = self._create_mtg_state_vector(config.state_dim)

            state = RLState(
                state_vector=state_vector,
                state_metadata={'benchmark_id': i},
                timestamp=time.time(),
                turn_number=i % 20 + 1,
                phase='main'
            )

            # Random action
            action_id = np.random.randint(0, config.action_dim)
            action = RLAction(
                action_id=action_id,
                action_type=f"benchmark_action_{action_id}",
                action_parameters={},
                confidence=0.8,
                explanation="Benchmark test action"
            )

            # Next state
            next_state_vector = self._create_mtg_state_vector(config.state_dim)
            next_state = RLState(
                state_vector=next_state_vector,
                state_metadata={'benchmark_id': i + 1},
                timestamp=time.time() + 1.0,
                turn_number=(i % 20) + 2
            )

            # Random reward
            reward = np.random.randn()

            transition = RLTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=np.random.random() < 0.1,
                episode_id=f"benchmark_episode_{i // 100}",
                step_number=i % 100
            )
            transitions.append(transition)

        return transitions

    def _create_mtg_state_vector(self, dim: int) -> torch.Tensor:
        """Create realistic MTG state vector."""
        state = torch.zeros(dim)

        # Board state (first 64 dims)
        state[:64] = torch.abs(torch.randn(64) * 0.2)

        # Hand and mana (next 128 dims)
        state[64:192] = torch.abs(torch.randn(128) * 0.15)

        # Phase and turn information (next 48 dims)
        state[192:240] = torch.randn(48) * 0.1

        # Additional features (remaining dims)
        if dim > 240:
            state[240:] = torch.randn(dim - 240) * 0.05

        return state


class TestCQLFactoryAndIntegration:
    """Test CQL factory functions and integration points."""

    def test_cql_factory_function(self):
        """Test Conservative Q-Learning factory function."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        # Test factory with default configuration
        agent = create_conservative_q_learning(state_dim=282, action_dim=16)

        assert isinstance(agent, ConservativeQLearning)
        assert agent.state_dim == 282
        assert agent.action_dim == 16
        assert agent.cql_config is not None

        # Test factory with custom configuration
        custom_config = ConservativeQLearningConfig(
            state_dim=128,
            action_dim=8,
            cql_alpha=10.0,
            learning_rate=5e-4
        )

        custom_agent = create_conservative_q_learning(
            state_dim=128,
            action_dim=8,
            config=custom_config
        )

        assert isinstance(custom_agent, ConservativeQLearning)
        assert custom_agent.state_dim == 128
        assert custom_agent.action_dim == 8
        assert custom_agent.cql_alpha == 10.0
        assert custom_agent.cql_config.learning_rate == 5e-4

    def test_cql_algorithm_info(self):
        """Test CQL algorithm information and metadata."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        config = ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            cql_alpha=5.0,
            attention_heads=4
        )
        agent = ConservativeQLearning(config)

        # Get algorithm info
        info = agent.get_algorithm_info()

        # Validate info structure
        assert isinstance(info, dict)
        assert 'algorithm_name' in info
        assert 'config' in info
        assert 'training_mode' in info
        assert 'compliance_status' in info

        assert info['algorithm_name'] == 'ConservativeQLearning'
        assert isinstance(info['config'], dict)
        assert info['training_mode'] is False  # Default should be False

        # Get model info
        model_info = agent.get_model_info()

        assert isinstance(model_info, dict)
        assert 'state_dim' in model_info
        assert 'action_dim' in model_info
        assert 'training_step' in model_info
        assert 'q_network_initialized' in model_info
        assert 'target_network_initialized' in model_info

        assert model_info['state_dim'] == 282
        assert model_info['action_dim'] == 16
        assert model_info['training_step'] == 0  # Initially
        assert model_info['q_network_initialized'] is True
        assert model_info['target_network_initialized'] is True

    def test_cql_training_statistics(self):
        """Test CQL training statistics tracking."""
        if not CQL_AVAILABLE:
            pytest.skip("CQL implementation not available")

        config = ConservativeQLearningConfig(
            state_dim=282,
            action_dim=16,
            batch_size=32
        )
        agent = ConservativeQLearning(config)

        # Get initial statistics
        initial_stats = agent.get_training_statistics()

        assert isinstance(initial_stats, dict)
        assert 'training_step' in initial_stats
        assert 'cql_alpha' in initial_stats
        assert 'total_training_losses' in initial_stats
        assert 'config' in initial_stats

        assert initial_stats['training_step'] == 0
        assert initial_stats['cql_alpha'] == config.cql_alpha
        assert initial_stats['total_training_losses'] == 0

        # Perform some training
        transitions = []
        for _ in range(100):
            state = RLState(
                state_vector=torch.randn(config.state_dim),
                state_metadata={},
                timestamp=time.time()
            )
            action = RLAction(
                action_id=np.random.randint(0, config.action_dim),
                action_type="test_action",
                action_parameters={},
                confidence=0.8
            )
            next_state = RLState(
                state_vector=torch.randn(config.state_dim),
                state_metadata={},
                timestamp=time.time() + 1.0
            )
            transition = RLTransition(
                state=state,
                action=action,
                reward=np.random.randn(),
                next_state=next_state,
                done=False
            )
            transitions.append(transition)

        # Train for a few steps
        for i in range(0, len(transitions), config.batch_size):
            batch = transitions[i:i + config.batch_size]
            if len(batch) >= config.batch_size:
                agent.update(batch)

        # Get updated statistics
        updated_stats = agent.get_training_statistics()

        # Should reflect training progress
        assert updated_stats['training_step'] > initial_stats['training_step']
        assert updated_stats['total_training_losses'] > 0

        # Should have loss statistics
        if 'loss_statistics' in updated_stats:
            loss_stats = updated_stats['loss_statistics']
            assert 'current_loss' in loss_stats
            assert 'avg_loss_recent' in loss_stats
            assert 'loss_std_recent' in loss_stats


# Test execution marker for pytest
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])