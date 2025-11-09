"""
Unit tests for RL models and enhanced state representation

Tests enhanced state representation system to ensure 380+ dimensional
state encoding as required for constitutional compliance.

Task T027: Enhanced State Representation Tests
User Story 1: Enhanced AI Decision Quality
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# Import the classes we need to test
try:
    from src.rl.models.dueling_dqn import DuelingDQNNetwork, DuelingDQNConfig
    DuelingDQN = DuelingDQNNetwork  # Alias for test compatibility
    print("✅ DuelingDQN import successful")
except ImportError as e:
    print(f"❌ DuelingDQN import failed: {e}")
    DuelingDQN = None
    DuelingDQNConfig = None

try:
    from src.rl.models.components.attention import MultiHeadAttention
    print("✅ MultiHeadAttention import successful")
except ImportError as e:
    print(f"❌ MultiHeadAttention import failed: {e}")
    MultiHeadAttention = None

try:
    from src.rl.models.components.layers import ResidualBlock
    print("✅ ResidualBlock import successful")
except ImportError as e:
    print(f"❌ ResidualBlock import failed: {e}")
    ResidualBlock = None

try:
    from src.rl.data.state_extractor import StateExtractor, StateFeatureConfig, MTGGameState
    print("✅ State extractor import successful")
except ImportError as e:
    print(f"❌ State extractor import failed: {e}")
    StateExtractor = None
    StateFeatureConfig = None
    MTGGameState = None


class TestStateExtractor:
    """Test enhanced MTG state representation system (380+ dimensions)."""

    @pytest.fixture
    def state_config(self):
        """Create state feature configuration."""
        if StateFeatureConfig is None:
            pytest.skip("StateFeatureConfig not implemented yet")
        return StateFeatureConfig(
            include_permanent_features=True,
            max_permanents_per_player=15,
            permanent_embedding_dim=8,
            include_hand_features=True,
            max_hand_size=10,
            card_embedding_dim=8,
            include_mana_features=True,
            mana_pool_dim=7,
            include_phase_features=True,
            include_temporal_features=True,
            include_statistical_features=True,
            include_metagame_features=True,
            include_opponent_modeling=True,
            enable_performance_monitoring=True,
            validate_feature_ranges=True
        )

    @pytest.fixture
    def state_extractor(self, state_config):
        """Create MTG state extractor for testing."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")
        return StateExtractor(config=state_config)

    @pytest.fixture
    def sample_game_state(self):
        """Create comprehensive sample MTG game state."""
        if MTGGameState is None:
            pytest.skip("MTGGameState not implemented yet")
        return MTGGameState(
            # Player state
            life=18,
            mana_pool={'red': 2, 'colorless': 1},
            hand=[
                {'name': 'Lightning Bolt', 'types': ['instant'], 'colors': ['red'],
                 'mana_cost': {'cmc': 1, 'red': 1}},
                {'name': 'Goblin Guide', 'types': ['creature'], 'colors': ['red'],
                 'mana_cost': {'cmc': 1, 'red': 1}, 'power': 2, 'toughness': 1},
                {'name': 'Mountain', 'types': ['land'], 'colors': ['red'], 'mana_cost': {'cmc': 0}},
                {'name': 'Counterspell', 'types': ['instant'], 'colors': ['blue'],
                 'mana_cost': {'cmc': 2, 'blue': 2}},
                {'name': 'Serra Angel', 'types': ['creature'], 'colors': ['white'],
                 'mana_cost': {'cmc': 5, 'white': 2}, 'power': 4, 'toughness': 4}
            ],
            library_count=45,
            graveyard_count=5,
            exile_count=0,

            # Board state
            battlefield=[
                {'name': 'Goblin Guide', 'types': ['creature'], 'colors': ['red'],
                 'power': 2, 'toughness': 1, 'tapped': False, 'mana_cost': {'cmc': 1}},
                {'name': 'Mountain', 'types': ['land'], 'colors': ['red'], 'tapped': False},
                {'name': 'Mountain', 'types': ['land'], 'colors': ['red'], 'tapped': True}
            ],
            lands=[],
            creatures=[
                {'name': 'Goblin Guide', 'types': ['creature'], 'colors': ['red'],
                 'power': 2, 'toughness': 1, 'tapped': False, 'mana_cost': {'cmc': 1}}
            ],
            artifacts_enchantments=[],

            # Game information
            turn_number=7,
            phase='main',
            step='main1',
            priority_player='player',
            active_player='player',
            storm_count=1,

            # Advanced features
            known_info={'recent_actions': [
                {'type': 'cast_spell', 'turn': 6},
                {'type': 'attack', 'turn': 6},
                {'type': 'pass_priority', 'turn': 7}
            ]},
            statistics={
                'avg_cmc_cast': 2.1,
                'land_drops': 6,
                'spells_cast': 8,
                'creatures_cast': 3,
                'damage_dealt': 12,
                'hand_size_trend': 0.1,
                'board_control_trend': -0.2,
                'win_rate_17lands': 0.65
            },
            opponent_info={
                'life': 20,
                'hand_size': 4,
                'library_count': 47,
                'graveyard_count': 3,
                'battlefield_count': 2,
                'statistics': {
                    'is_aggro': 0.7,
                    'is_control': 0.1,
                    'avg_cmc_cast': 1.8
                }
            },

            # Metadata
            timestamp=time.time(),
            format='standard',
            game_id='test_game_001'
        )

    def test_state_extractor_initialization(self, state_extractor, state_config):
        """Test state extractor initialization with 380+ dimension requirement."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test basic initialization
        assert state_extractor is not None
        assert hasattr(state_extractor, 'extract_state')
        assert hasattr(state_extractor, 'config')

        # Test configuration
        assert state_extractor.config == state_config

        # Test constitutional requirement: 380+ dimensions
        assert state_extractor.config.total_dimensions >= 380, \
            f"State dimensions {state_extractor.config.total_dimensions} below constitutional minimum of 380"

        # Test feature validation
        assert state_extractor.config.include_permanent_features
        assert state_extractor.config.include_hand_features
        assert state_extractor.config.include_mana_features
        assert state_extractor.config.include_phase_features

    def test_state_dimension_requirement(self, state_extractor):
        """Test that state meets 380+ dimension constitutional requirement."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Check that dimensions are properly calculated
        expected_dimensions = (
            120 +  # Board features
            80 +   # Hand features
            40 +   # Mana features
            48 +   # Phase features
            24 +   # Temporal features
            28 +   # Statistical features
            20 +   # Metagame features
            20     # Opponent modeling
        )

        assert state_extractor.config.total_dimensions == expected_dimensions
        assert state_extractor.config.total_dimensions >= 380

    def test_380_dimension_requirement(self, state_extractor):
        """Test explicit 380+ dimensional state representation as per requirements."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Constitutional requirement: 380+ dimensions
        actual_dimensions = state_extractor.config.total_dimensions

        # Must meet minimum 380 dimensions
        assert actual_dimensions >= 380, f"State dimensions {actual_dimensions} below constitutional minimum of 380"

        # Test expansion from baseline (23 dimensions in original system)
        baseline_dimensions = 23
        expansion_ratio = actual_dimensions / baseline_dimensions
        assert expansion_ratio >= 16.5, f"Expansion ratio {expansion_ratio:.1f}x should be >= 16.5x"

        # Verify all feature categories contribute to dimensions
        feature_checklist = {
            'board_features': 120,
            'hand_features': 80,
            'mana_features': 40,
            'phase_features': 48,
            'temporal_features': 24,
            'statistical_features': 28,
            'metagame_features': 20,
            'opponent_features': 20
        }

        total_feature_dims = sum(feature_checklist.values())
        assert total_feature_dims == actual_dimensions, f"Feature sum {total_feature_dims} != actual {actual_dimensions}"

        print(f"✅ 380+ dimensional state verified: {actual_dimensions} dimensions ({expansion_ratio:.1f}x expansion)")

    def test_basic_state_extraction(self, state_extractor, sample_game_state):
        """Test basic state extraction from game state."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        state_vector = state_extractor.extract_state(sample_game_state)

        # Check return type and dimensions
        assert isinstance(state_vector, torch.Tensor)
        assert len(state_vector) == state_extractor.config.total_dimensions
        assert state_vector.dtype == torch.float32

        # Check that features are normalized and finite
        assert torch.all(torch.isfinite(state_vector)), "State vector contains infinite or NaN values"
        assert not torch.any(torch.isnan(state_vector)), "State vector contains NaN values"

        # Check reasonable value ranges (normalized)
        assert torch.all(torch.abs(state_vector) <= 100.0), "State features should be normalized"

    def test_board_state_encoding(self, state_extractor, sample_game_state):
        """Test board state feature extraction (120 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with empty board
        empty_board_state = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=1, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0, known_info={}, statistics={},
            opponent_info={}, timestamp=time.time(), format='standard', game_id='empty_board'
        )

        empty_vector = state_extractor.extract_state(empty_board_state)
        board_vector = state_extractor.extract_state(sample_game_state)

        # Board presence should change state
        board_diff = torch.abs(board_vector - empty_vector)
        assert torch.any(board_diff > 0.01), "Board state should affect state vector"

        # Check board-specific features are encoded
        assert hasattr(state_extractor, '_extract_board_features')
        board_features = state_extractor._extract_board_features(sample_game_state)
        assert len(board_features) == 120, "Board features should be 120 dimensions"

    def test_hand_encoding(self, state_extractor, sample_game_state):
        """Test hand feature extraction (80 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with different hand sizes
        empty_hand_state = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=5, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0, known_info={}, statistics={},
            opponent_info={}, timestamp=time.time(), format='standard', game_id='empty_hand'
        )

        empty_vector = state_extractor.extract_state(empty_hand_state)
        hand_vector = state_extractor.extract_state(sample_game_state)

        # Hand size should be reflected in state
        hand_diff = torch.abs(hand_vector - empty_vector)
        assert torch.any(hand_diff > 0.01), "Hand state should affect state vector"

        # Check hand-specific features
        assert hasattr(state_extractor, '_extract_hand_features')
        hand_features = state_extractor._extract_hand_features(sample_game_state)
        assert len(hand_features) == 80, "Hand features should be 80 dimensions"

    def test_mana_encoding(self, state_extractor, sample_game_state):
        """Test mana pool and production features (40 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with no mana
        no_mana_state = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=5, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0, known_info={}, statistics={},
            opponent_info={}, timestamp=time.time(), format='standard', game_id='no_mana'
        )

        no_mana_vector = state_extractor.extract_state(no_mana_state)
        mana_vector = state_extractor.extract_state(sample_game_state)

        # Mana should be reflected in state
        mana_diff = torch.abs(mana_vector - no_mana_vector)
        assert torch.any(mana_diff > 0.01), "Mana state should affect state vector"

        # Check mana-specific features
        assert hasattr(state_extractor, '_extract_mana_features')
        mana_features = state_extractor._extract_mana_features(sample_game_state)
        assert len(mana_features) == 40, "Mana features should be 40 dimensions"

    def test_phase_encoding(self, state_extractor, sample_game_state):
        """Test game phase and timing features (48 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        phases = ['beginning', 'precombat_main', 'combat', 'postcombat_main', 'ending']
        vectors = []

        for phase in phases:
            test_state = MTGGameState(
                life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
                exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=5, phase=phase, step='main1', priority_player='player',
                active_player='player', storm_count=0, known_info={}, statistics={},
                opponent_info={}, timestamp=time.time(), format='standard', game_id=f'phase_{phase}'
            )
            vector = state_extractor.extract_state(test_state)
            vectors.append(vector)

        # Different phases should have different encodings
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                phase_diff = torch.abs(vectors[i] - vectors[j])
                assert torch.any(phase_diff > 0.01), f"Phases {phases[i]} and {phases[j]} should have different encodings"

        # Check phase-specific features
        assert hasattr(state_extractor, '_extract_phase_features')
        phase_features = state_extractor._extract_phase_features(sample_game_state)
        assert len(phase_features) == 48, "Phase features should be 48 dimensions"

    def test_17lands_data_integration(self, state_extractor, sample_game_state):
        """Test 17Lands statistics data integration."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test state with 17Lands statistics
        stats_with_17lands = MTGGameState(
            life=18, mana_pool={'red': 2}, hand=[], library_count=45, graveyard_count=5,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=7, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=1, known_info={},
            statistics={
                'win_rate_17lands': 0.65,
                'game_win_rate_17lands': 0.58,
                'play_rate_17lands': 0.12,
                'metagame_share_17lands': 0.08,
                'avg_cmc_cast': 2.1,
                'land_drops': 6,
                'spells_cast': 8,
                'damage_dealt': 12
            },
            opponent_info={'life': 20}, timestamp=time.time(), format='standard', game_id='17lands_test'
        )

        state_vector = state_extractor.extract_state(stats_with_17lands)

        # Should include statistical features
        assert hasattr(state_extractor, '_extract_statistical_features')
        stat_features = state_extractor._extract_statistical_features(stats_with_17lands)
        assert len(stat_features) == 28, "Statistical features should be 28 dimensions"

        # Verify 17Lands data is incorporated
        assert torch.all(torch.isfinite(state_vector))

    def test_temporal_features(self, state_extractor, sample_game_state):
        """Test temporal and historical features (24 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        assert hasattr(state_extractor, '_extract_temporal_features')
        temporal_features = state_extractor._extract_temporal_features(sample_game_state)
        assert len(temporal_features) == 24, "Temporal features should be 24 dimensions"

        # Test with different game histories
        early_game_state = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=55, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=2, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0,
            known_info={'recent_actions': [{'type': 'land', 'turn': 1}]},
            statistics={'hand_size_trend': 0.0, 'board_control_trend': 0.0},
            opponent_info={}, timestamp=time.time(), format='standard', game_id='early'
        )

        early_vector = state_extractor.extract_state(early_game_state)
        late_vector = state_extractor.extract_state(sample_game_state)

        temporal_diff = torch.abs(late_vector - early_vector)
        assert torch.any(temporal_diff > 0.01), "Temporal features should differ between early and late game"

    def test_opponent_modeling(self, state_extractor, sample_game_state):
        """Test opponent modeling features (20 dimensions)."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        assert hasattr(state_extractor, '_extract_opponent_features')
        opponent_features = state_extractor._extract_opponent_features(sample_game_state)
        assert len(opponent_features) == 20, "Opponent features should be 20 dimensions"

        # Test with different opponent profiles
        aggrophile_opponent = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=5, phase='main', step='main1', priority_player='opponent',
            active_player='opponent', storm_count=0, known_info={}, statistics={},
            opponent_info={
                'life': 18, 'hand_size': 5, 'statistics': {
                    'is_aggro': 0.9, 'is_control': 0.05, 'avg_cmc_cast': 1.5
                }
            },
            timestamp=time.time(), format='standard', game_id='aggro_opp'
        )

        control_opponent = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=5, phase='main', step='main1', priority_player='opponent',
            active_player='opponent', storm_count=0, known_info={}, statistics={},
            opponent_info={
                'life': 20, 'hand_size': 7, 'statistics': {
                    'is_aggro': 0.1, 'is_control': 0.8, 'avg_cmc_cast': 3.2
                }
            },
            timestamp=time.time(), format='standard', game_id='control_opp'
        )

        aggro_vector = state_extractor.extract_state(aggrophile_opponent)
        control_vector = state_extractor.extract_state(control_opponent)

        opponent_diff = torch.abs(aggro_vector - control_vector)
        assert torch.any(opponent_diff > 0.01), "Opponent modeling should differentiate opponent types"

    def test_state_consistency(self, state_extractor, sample_game_state):
        """Test state extraction consistency across multiple extractions."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Extract state multiple times
        vector1 = state_extractor.extract_state(sample_game_state)
        vector2 = state_extractor.extract_state(sample_game_state)
        vector3 = state_extractor.extract_state(sample_game_state)

        # Should be identical
        torch.testing.assert_close(vector1, vector2, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(vector2, vector3, rtol=1e-6, atol=1e-6)

    def test_state_normalization(self, state_extractor):
        """Test state feature normalization and range validation."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with extreme values
        extreme_state = MTGGameState(
            life=100,  # Very high life
            mana_pool={'red': 50, 'blue': 50},  # Extreme mana
            hand=[],  # Empty hand despite extreme resources
            library_count=0,  # Empty library
            graveyard_count=100,  # Large graveyard
            exile_count=50,  # Large exile
            battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=100,  # Very late game
            phase='ending', step='cleanup', priority_player='player',
            active_player='player', storm_count=50,  # Extreme storm
            known_info={}, statistics={}, opponent_info={},
            timestamp=time.time(), format='standard', game_id='extreme'
        )

        state_vector = state_extractor.extract_state(extreme_state)

        # All values should be normalized to reasonable ranges
        assert torch.all(torch.abs(state_vector) <= 100.0), "State features should be normalized"
        assert torch.all(torch.isfinite(state_vector)), "State features should be finite"
        assert not torch.any(torch.isnan(state_vector)), "State features should not be NaN"

    def test_performance_requirement(self, state_extractor, sample_game_state):
        """Test performance requirement: sub-100ms processing for real-time inference."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Time multiple extractions
        num_extractions = 100
        start_time = time.time()

        for _ in range(num_extractions):
            state_vector = state_extractor.extract_state(sample_game_state)
            assert len(state_vector) >= 380  # Verify dimension requirement

        end_time = time.time()
        avg_time = (end_time - start_time) / num_extractions * 1000  # Convert to ms

        # Should be fast for real-time usage (target < 100ms, ideally < 10ms)
        assert avg_time < 100.0, f"State extraction too slow: {avg_time:.3f}ms (target < 100ms)"

        # Check performance statistics
        perf_stats = state_extractor.get_performance_stats()
        assert 'avg_extraction_time_ms' in perf_stats
        assert perf_stats['avg_extraction_time_ms'] < 100.0
        assert perf_stats['constitutional_compliance'] == True

    def test_feature_importance_analysis(self, state_extractor):
        """Test feature importance analysis for explainability."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        importance = state_extractor.get_feature_importance()

        # Should return importance for all feature groups
        expected_features = [
            'board_state', 'hand', 'mana', 'timing',
            'temporal', 'statistics', 'metagame', 'opponent'
        ]

        for feature in expected_features:
            assert feature in importance, f"Missing importance for {feature}"
            assert 0.0 <= importance[feature] <= 1.0, f"Invalid importance score for {feature}"

        # Importance should sum to approximately 1.0
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.1, f"Importance scores should sum to ~1.0, got {total_importance}"

    def test_state_explanation_generation(self, state_extractor, sample_game_state):
        """Test state explanation generation for interpretability."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        state_vector = state_extractor.extract_state(sample_game_state)
        explanation = state_extractor.get_state_explanation(sample_game_state, state_vector)

        # Should include comprehensive explanation
        assert 'state_dimensions' in explanation
        assert 'feature_importance' in explanation
        assert 'key_features' in explanation
        assert 'extraction_performance' in explanation

        assert explanation['state_dimensions'] >= 380
        assert len(explanation['feature_importance']) > 0
        assert 'life_points' in explanation['key_features']
        assert 'hand_size' in explanation['key_features']

    def test_error_handling_and_graceful_degradation(self, state_extractor):
        """Test error handling and graceful degradation."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with malformed game state
        malformed_state = MTGGameState(
            life=-5,  # Invalid life
            mana_pool={'invalid_color': -10},  # Invalid mana
            hand=[{'invalid': 'card'}],  # Invalid card
            library_count=-1,  # Invalid count
            graveyard_count=None,  # None value
            exile_count=float('inf'),  # Infinite value
            battlefield=None, lands=None, creatures=None, artifacts_enchantments=None,
            turn_number=None, phase=None, step=None, priority_player=None,
            active_player=None, storm_count=-5, known_info=None, statistics=None,
            opponent_info=None, timestamp=None, format=None, game_id=None
        )

        # Should handle gracefully without crashing
        state_vector = state_extractor.extract_state(malformed_state)

        # Should return valid state vector (possibly with fallback values)
        assert isinstance(state_vector, torch.Tensor)
        assert len(state_vector) == state_extractor.config.total_dimensions
        assert torch.all(torch.isfinite(state_vector)), "State should be finite even with malformed input"

    def test_comprehensive_feature_coverage(self, state_extractor, sample_game_state):
        """Test that all required feature categories are properly covered."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        state_vector = state_extractor.extract_state(sample_game_state)

        # Test that all feature extraction methods exist and return correct dimensions
        feature_methods = [
            ('_extract_board_features', 120),
            ('_extract_hand_features', 80),
            ('_extract_mana_features', 40),
            ('_extract_phase_features', 48),
            ('_extract_temporal_features', 24),
            ('_extract_statistical_features', 28),
            ('_extract_metagame_features', 20),
            ('_extract_opponent_features', 20)
        ]

        for method_name, expected_dim in feature_methods:
            assert hasattr(state_extractor, method_name), f"Missing method: {method_name}"
            method = getattr(state_extractor, method_name)
            features = method(sample_game_state)
            assert len(features) == expected_dim, f"{method_name} should return {expected_dim} features, got {len(features)}"

    def test_dimension_expansion_from_baseline(self, state_extractor):
        """Test expansion from 23 to 380+ dimensions as specified."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # The baseline system had ~23 dimensions
        baseline_dimensions = 23
        current_dimensions = state_extractor.config.total_dimensions

        # Should be expanded significantly
        assert current_dimensions >= 380, f"Need 380+ dimensions, got {current_dimensions}"
        expansion_ratio = current_dimensions / baseline_dimensions
        assert expansion_ratio >= 16.5, f"Expansion ratio {expansion_ratio:.1f}x should be >= 16.5x"

        print(f"✅ Dimension expansion: {baseline_dimensions} → {current_dimensions} ({expansion_ratio:.1f}x)")

    def test_integration_with_mtga_voice_advisor(self, state_extractor):
        """Test integration with existing MTGA Voice Advisor infrastructure."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with data structure similar to MTGA Voice Advisor
        mtga_style_state = MTGGameState(
            life=17, mana_pool={'red': 1, 'white': 1},
            hand=[
                {'name': 'Lightning Bolt', 'types': ['instant'], 'colors': ['red'],
                 'mana_cost': {'cmc': 1, 'red': 1}},
                {'name': 'Plains', 'types': ['land'], 'colors': ['white'], 'mana_cost': {'cmc': 0}}
            ],
            library_count=48, graveyard_count=4, exile_count=0,
            battlefield=[
                {'name': 'Goblin Guide', 'types': ['creature'], 'colors': ['red'],
                 'power': 2, 'toughness': 1, 'tapped': False, 'mana_cost': {'cmc': 1}}
            ],
            lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=8, phase='combat', step='declare_attackers',
            priority_player='player', active_player='player', storm_count=0,
            known_info={'mtga_log_line': 'Player attacks with Goblin Guide'},
            statistics={'damage_dealt_this_turn': 2},
            opponent_info={'life': 15, 'battlefield_count': 1},
            timestamp=time.time(), format='standard', game_id='mtga_integration'
        )

        state_vector = state_extractor.extract_state(mtga_style_state)

        # Should handle MTGA-style data gracefully
        assert isinstance(state_vector, torch.Tensor)
        assert len(state_vector) >= 380
        assert torch.all(torch.isfinite(state_vector))

        # Should be able to generate explanations compatible with advisor system
        explanation = state_extractor.get_state_explanation(mtga_style_state, state_vector)
        assert 'key_features' in explanation
        assert 'life_points' in explanation['key_features']
        assert 'board_presence' in explanation['key_features']

    def test_batch_state_extraction(self, state_extractor):
        """Test batch state extraction for training efficiency."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Create multiple game states
        game_states = []
        for i in range(5):
            state = MTGGameState(
                life=20 - i, mana_pool={'red': i}, hand=[], library_count=50 - i,
                graveyard_count=i, exile_count=0, battlefield=[], lands=[],
                creatures=[], artifacts_enchantments=[], turn_number=5 + i,
                phase='main', step='main1', priority_player='player',
                active_player='player', storm_count=0, known_info={}, statistics={},
                opponent_info={}, timestamp=time.time(), format='standard',
                game_id=f'batch_test_{i}'
            )
            game_states.append(state)

        # Extract states in batch
        start_time = time.time()
        state_vectors = []
        for state in game_states:
            vector = state_extractor.extract_state(state)
            state_vectors.append(vector)
        end_time = time.time()

        batch_time = (end_time - start_time) / len(game_states) * 1000

        # Batch processing should be efficient
        assert batch_time < 50.0, f"Batch processing too slow: {batch_time:.3f}ms per state"

        # All vectors should have correct dimensions
        for i, vector in enumerate(state_vectors):
            assert len(vector) == state_extractor.config.total_dimensions
            assert torch.all(torch.isfinite(vector))

        # Stack into batch tensor
        batch_tensor = torch.stack(state_vectors)
        assert batch_tensor.shape == (5, state_extractor.config.total_dimensions)

    def test_state_tensor_construction(self, state_extractor, sample_game_state):
        """Test state tensor construction and validation."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test tensor construction
        state_vector = state_extractor.extract_state(sample_game_state)

        # Check tensor properties
        assert isinstance(state_vector, torch.Tensor)
        assert state_vector.dtype == torch.float32
        assert len(state_vector.shape) == 1  # Should be 1D vector
        assert state_vector.shape[0] == state_extractor.config.total_dimensions

        # Test tensor validation method
        assert hasattr(state_extractor, 'validate_state_dimensions')
        is_valid = state_extractor.validate_state_dimensions(state_vector)
        assert is_valid == True

        # Test with invalid dimensions
        invalid_vector = torch.randn(100)  # Wrong size
        is_invalid = state_extractor.validate_state_dimensions(invalid_vector)
        assert is_invalid == False

    def test_state_representation_completeness(self, state_extractor, sample_game_state):
        """Test state representation completeness and accuracy."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Extract state and verify all expected information is captured
        state_vector = state_extractor.extract_state(sample_game_state)

        # Test that all feature groups contribute to the state
        feature_contributions = {}
        total_dim = state_extractor.config.total_dimensions

        # Board state should be the largest contributor
        if state_extractor.config.include_permanent_features:
            feature_contributions['board'] = 120 / total_dim

        # Hand features
        if state_extractor.config.include_hand_features:
            feature_contributions['hand'] = 80 / total_dim

        # Mana features
        if state_extractor.config.include_mana_features:
            feature_contributions['mana'] = 40 / total_dim

        # Phase features
        if state_extractor.config.include_phase_features:
            feature_contributions['phase'] = 48 / total_dim

        # Advanced features
        if state_extractor.config.include_temporal_features:
            feature_contributions['temporal'] = 24 / total_dim
        if state_extractor.config.include_statistical_features:
            feature_contributions['statistical'] = 28 / total_dim
        if state_extractor.config.include_metagame_features:
            feature_contributions['metagame'] = 20 / total_dim
        if state_extractor.config.include_opponent_modeling:
            feature_contributions['opponent'] = 20 / total_dim

        # Verify feature distribution
        total_contribution = sum(feature_contributions.values())
        assert abs(total_contribution - 1.0) < 0.01, f"Feature contributions should sum to 1.0, got {total_contribution}"

        # Verify most important features have highest contribution
        assert feature_contributions['board'] >= 0.25, "Board state should be primary feature"
        assert feature_contributions['hand'] >= 0.15, "Hand should be major feature"

    def test_comparative_state_analysis(self, state_extractor):
        """Test comparative analysis of different game states."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Create states with different strategic situations
        winning_state = MTGGameState(
            life=20, mana_pool={'white': 5, 'blue': 3},
            hand=[], library_count=40, graveyard_count=5, exile_count=0,
            battlefield=[
                {'name': 'Serra Angel', 'types': ['creature'], 'colors': ['white'],
                 'power': 4, 'toughness': 4, 'tapped': False, 'mana_cost': {'cmc': 5}},
                {'name': 'Serra Angel', 'types': ['creature'], 'colors': ['white'],
                 'power': 4, 'toughness': 4, 'tapped': False, 'mana_cost': {'cmc': 5}}
            ],
            lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=8, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0, known_info={}, statistics={},
            opponent_info={'life': 5}, timestamp=time.time(), format='standard', game_id='winning'
        )

        losing_state = MTGGameState(
            life=3, mana_pool={}, hand=[], library_count=35, graveyard_count=8,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=8, phase='main', step='main1', priority_player='opponent',
            active_player='opponent', storm_count=0, known_info={}, statistics={},
            opponent_info={'life': 20}, timestamp=time.time(), format='standard', game_id='losing'
        )

        winning_vector = state_extractor.extract_state(winning_state)
        losing_vector = state_extractor.extract_state(losing_state)

        # Vectors should be significantly different
        state_diff = torch.abs(winning_vector - losing_vector)
        avg_difference = torch.mean(state_diff)

        assert avg_difference > 0.1, f"Winning and losing states should differ significantly (avg diff: {avg_difference:.3f})"

        # Check specific feature differences
        # Life differences should be prominent
        life_idx_start = 0  # Assuming life is encoded early
        life_diff = torch.abs(winning_vector[life_idx_start:life_idx_start+2] - losing_vector[life_idx_start:life_idx_start+2])
        assert torch.mean(life_diff) > 0.2, "Life differences should be clearly encoded"

    def test_real_time_performance_validation(self, state_extractor, sample_game_state):
        """Test real-time performance validation for sub-100ms requirement."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with different complexity states
        simple_state = MTGGameState(
            life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
            exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=1, phase='main', step='main1', priority_player='player',
            active_player='player', storm_count=0, known_info={}, statistics={},
            opponent_info={}, timestamp=time.time(), format='standard', game_id='simple'
        )

        complex_state = MTGGameState(
            life=15, mana_pool={'white': 3, 'blue': 2, 'black': 1, 'red': 1, 'green': 1},
            hand=[
                {'name': f'Card_{i}', 'types': ['creature'] if i % 2 == 0 else ['instant'],
                 'colors': ['red'], 'mana_cost': {'cmc': i, 'red': 1}}
                for i in range(7)
            ],
            library_count=40, graveyard_count=8, exile_count=2,
            battlefield=[
                {'name': f'Creature_{i}', 'types': ['creature'], 'colors': ['white'],
                 'power': i+1, 'toughness': i+2, 'tapped': i % 2 == 0, 'mana_cost': {'cmc': i+1}}
                for i in range(8)
            ],
            lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=12, phase='combat', step='declare_attackers',
            priority_player='player', active_player='player', storm_count=5,
            known_info={'recent_actions': [{'type': 'cast_spell', 'turn': t} for t in range(1, 13)]},
            statistics={
                'avg_cmc_cast': 2.5, 'land_drops': 11, 'spells_cast': 15,
                'creatures_cast': 8, 'damage_dealt': 25
            },
            opponent_info={'life': 18, 'hand_size': 5, 'battlefield_count': 4},
            timestamp=time.time(), format='standard', game_id='complex'
        )

        # Time simple state extraction
        num_tests = 100
        start_time = time.time()
        for _ in range(num_tests):
            state_extractor.extract_state(simple_state)
        simple_time = (time.time() - start_time) / num_tests * 1000

        # Time complex state extraction
        start_time = time.time()
        for _ in range(num_tests):
            state_extractor.extract_state(complex_state)
        complex_time = (time.time() - start_time) / num_tests * 1000

        # Both should meet performance requirements
        assert simple_time < 50.0, f"Simple state extraction too slow: {simple_time:.3f}ms"
        assert complex_time < 100.0, f"Complex state extraction too slow: {complex_time:.3f}ms"

        # Complex state should not be dramatically slower than simple state
        time_ratio = complex_time / simple_time
        assert time_ratio < 5.0, f"Complex state extraction {time_ratio:.1f}x slower than simple"

        print(f"Performance: Simple={simple_time:.3f}ms, Complex={complex_time:.3f}ms, Ratio={time_ratio:.1f}x")

    def test_state_vector_interpretability(self, state_extractor, sample_game_state):
        """Test that state vector features are interpretable and explainable."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        state_vector = state_extractor.extract_state(sample_game_state)
        explanation = state_extractor.get_state_explanation(sample_game_state, state_vector)

        # Test interpretability requirements
        assert 'state_dimensions' in explanation
        assert 'feature_importance' in explanation
        assert 'key_features' in explanation
        assert 'extraction_performance' in explanation

        # Key features should be easily interpretable
        key_features = explanation['key_features']
        assert 'life_points' in key_features
        assert 'hand_size' in key_features
        assert 'board_presence' in key_features
        assert 'turn_number' in key_features
        assert 'current_phase' in key_features
        assert 'mana_available' in key_features

        # Feature importance should be explainable
        importance = explanation['feature_importance']
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.1, "Feature importance should sum to ~1.0"

        # Performance should meet constitutional requirements
        perf = explanation['extraction_performance']
        assert perf['avg_extraction_time_ms'] < 100.0
        assert perf['constitutional_compliance'] == True

    def test_advanced_error_scenarios(self, state_extractor):
        """Test advanced error scenarios and edge cases."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with completely corrupted data
        corrupted_scenarios = [
            # None values everywhere
            MTGGameState(
                life=None, mana_pool=None, hand=None, library_count=None,
                graveyard_count=None, exile_count=None, battlefield=None,
                lands=None, creatures=None, artifacts_enchantments=None,
                turn_number=None, phase=None, step=None, priority_player=None,
                active_player=None, storm_count=None, known_info=None,
                statistics=None, opponent_info=None, timestamp=None,
                format=None, game_id=None
            ),
            # Extreme numeric values
            MTGGameState(
                life=float('inf'), mana_pool={'red': float('nan')},
                hand=[{'invalid': 'data'}], library_count=-1000,
                graveyard_count=10**6, exile_count=-500, battlefield=[],
                lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=-10, phase='', step='', priority_player='',
                active_player='', storm_count=-100, known_info={},
                statistics={}, opponent_info={}, timestamp=-1.0,
                format='', game_id=''
            ),
            # Very large data structures
            MTGGameState(
                life=20, mana_pool={},
                hand=[{'name': f'Card_{i}'} for i in range(1000)],  # Huge hand
                library_count=100000, graveyard_count=10000, exile_count=5000,
                battlefield=[{'name': f'Creature_{i}'} for i in range(100)],  # Huge board
                lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=10000, phase='main', step='main1',
                priority_player='player', active_player='player', storm_count=1000,
                known_info={'actions': [{'type': 'action'} for _ in range(1000)]},
                statistics={}, opponent_info={}, timestamp=time.time(),
                format='standard', game_id='huge'
            )
        ]

        for i, corrupted_state in enumerate(corrupted_scenarios):
            # Should not crash
            try:
                state_vector = state_extractor.extract_state(corrupted_state)

                # Should return valid tensor
                assert isinstance(state_vector, torch.Tensor)
                assert len(state_vector) == state_extractor.config.total_dimensions
                assert torch.all(torch.isfinite(state_vector)), f"Corrupted scenario {i} should produce finite state"

            except Exception as e:
                # If it does raise an exception, it should be a graceful one
                assert isinstance(e, (ValueError, TypeError)), f"Scenario {i} should raise graceful exception"
                assert "state" in str(e).lower() or "invalid" in str(e).lower(), f"Scenario {i} exception should be descriptive"

    def test_feature_configuration_validation(self):
        """Test state feature configuration validation."""
        if StateFeatureConfig is None:
            pytest.skip("StateFeatureConfig not implemented yet")

        # Test default configuration
        default_config = StateFeatureConfig()
        assert default_config.total_dimensions >= 380
        assert default_config.include_permanent_features
        assert default_config.include_hand_features
        assert default_config.include_mana_features
        assert default_config.include_phase_features

        # Test invalid configurations
        with pytest.raises(ValueError):
            # Configuration below 380 dimensions should fail
            invalid_config = StateFeatureConfig(
                include_permanent_features=False,
                include_hand_features=False,
                include_mana_features=False,
                include_phase_features=False
            )
            if invalid_config.total_dimensions < 380:
                raise ValueError("Configuration violates 380+ dimension requirement")

        # Test custom valid configuration
        custom_config = StateFeatureConfig(
            max_permanents_per_player=20,
            permanent_embedding_dim=10,
            max_hand_size=12,
            card_embedding_dim=10,
            enable_performance_monitoring=True,
            validate_feature_ranges=True
        )
        assert custom_config.total_dimensions >= 380
        assert custom_config.max_permanents_per_player == 20
        assert custom_config.permanent_embedding_dim == 10

    def test_memory_efficiency_validation(self, state_extractor, sample_game_state):
        """Test memory efficiency of state extraction."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Extract many states
        num_extractions = 1000
        states = []
        for i in range(num_extractions):
            # Create slightly different states
            test_state = MTGGameState(
                life=20 - i % 10, mana_pool={'red': i % 5},
                hand=[], library_count=50 - i % 20, graveyard_count=i % 10,
                exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=5 + i % 10, phase='main', step='main1',
                priority_player='player', active_player='player', storm_count=0,
                known_info={}, statistics={}, opponent_info={},
                timestamp=time.time(), format='standard', game_id=f'memory_test_{i}'
            )
            state_vector = state_extractor.extract_state(test_state)
            states.append(state_vector)

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        memory_per_state = memory_increase / num_extractions
        assert memory_per_state < 1.0, f"Memory usage too high: {memory_per_state:.3f}MB per state"

        # Clean up
        del states
        final_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = final_memory - final_cleanup_memory

        print(f"Memory efficiency: {memory_per_state:.3f}MB per state, {memory_recovered:.1f}MB recovered")

    def test_concurrent_state_extraction(self, state_extractor, sample_game_state):
        """Test thread safety of state extraction."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        import threading
        import concurrent.futures

        # Test concurrent extraction
        num_threads = 10
        extractions_per_thread = 50

        def extract_states(thread_id):
            results = []
            for i in range(extractions_per_thread):
                # Create unique state for this thread/extraction
                test_state = MTGGameState(
                    life=20, mana_pool={'red': thread_id % 5},
                    hand=[], library_count=50, graveyard_count=0, exile_count=0,
                    battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
                    turn_number=5 + i % 5, phase='main', step='main1',
                    priority_player='player', active_player='player', storm_count=0,
                    known_info={}, statistics={}, opponent_info={},
                    timestamp=time.time(), format='standard', game_id=f'thread_{thread_id}_ex_{i}'
                )
                state_vector = state_extractor.extract_state(test_state)
                results.append(len(state_vector))
            return results

        # Run concurrent extractions
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(extract_states, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all extractions were successful
        all_dimensions = []
        for thread_results in results:
            all_dimensions.extend(thread_results)

        # All should have correct dimensions
        expected_dim = state_extractor.config.total_dimensions
        assert all(dim == expected_dim for dim in all_dimensions), \
            f"All concurrent extractions should have {expected_dim} dimensions"

        # Should have extracted the correct total number
        total_extractions = len(all_dimensions)
        expected_total = num_threads * extractions_per_thread
        assert total_extractions == expected_total, \
            f"Expected {expected_total} extractions, got {total_extractions}"

        print(f"Concurrent extraction: {num_threads} threads, {total_extractions} total extractions successful")

    def test_integration_with_existing_infrastructure(self, state_extractor):
        """Test integration with existing MTGA Voice Advisor infrastructure."""
        if StateExtractor is None:
            pytest.skip("StateExtractor not implemented yet")

        # Test with data structure from existing advisor
        advisor_state_data = {
            'life': 17,
            'opponent_life': 15,
            'hand': [
                {'name': 'Lightning Bolt', 'type': 'instant', 'cost': 'R'},
                {'name': 'Mountain', 'type': 'land', 'cost': '0'}
            ],
            'board': [
                {'name': 'Goblin Guide', 'power': 2, 'toughness': 1, 'tapped': False}
            ],
            'opponent_board': [
                {'name': 'Elvish Mystic', 'power': 1, 'toughness': 1, 'tapped': True}
            ],
            'mana_pool': {'R': 1},
            'graveyard_size': 4,
            'library_size': 48,
            'turn_number': 6,
            'phase': 'combat',
            'can_play_land': False,
            'can_attack': True,
            'storm_count': 1
        }

        # Convert to MTGGameState
        mtga_state = MTGGameState(
            life=advisor_state_data['life'],
            mana_pool={'red': advisor_state_data['mana_pool'].get('R', 0)},
            hand=[
                {'name': card['name'], 'types': [card['type']], 'colors': ['red'] if 'R' in card.get('cost', '') else [],
                 'mana_cost': {'cmc': 1 if card['cost'] != '0' else 0}}
                for card in advisor_state_data['hand']
            ],
            library_count=advisor_state_data['library_size'],
            graveyard_count=advisor_state_data['graveyard_size'],
            exile_count=0,
            battlefield=[
                {'name': perm['name'], 'types': ['creature'], 'colors': ['red'],
                 'power': perm['power'], 'toughness': perm['toughness'],
                 'tapped': perm['tapped'], 'mana_cost': {'cmc': 1}}
                for perm in advisor_state_data['board']
            ],
            lands=[], creatures=[], artifacts_enchantments=[],
            turn_number=advisor_state_data['turn_number'],
            phase=advisor_state_data['phase'],
            step='declare_attackers',
            priority_player='player',
            active_player='player',
            storm_count=advisor_state_data['storm_count'],
            known_info={'can_play_land': advisor_state_data['can_play_land']},
            statistics={'can_attack': advisor_state_data['can_attack']},
            opponent_info={'life': advisor_state_data['opponent_life']},
            timestamp=time.time(), format='standard', game_id='mtga_integration'
        )

        # Extract state
        state_vector = state_extractor.extract_state(mtga_state)

        # Should work seamlessly
        assert isinstance(state_vector, torch.Tensor)
        assert len(state_vector) >= 380
        assert torch.all(torch.isfinite(state_vector))

        # Generate explanation
        explanation = state_extractor.get_state_explanation(mtga_state, state_vector)
        assert 'key_features' in explanation
        assert explanation['key_features']['life_points'] == advisor_state_data['life']
        assert explanation['key_features']['turn_number'] == advisor_state_data['turn_number']

        print("✅ Successfully integrated with MTGA Voice Advisor infrastructure")


class TestDuelingDQN:
    """Test Dueling DQN model architecture."""

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return {
            'state_dim': 380,
            'action_dim': 64,
            'hidden_dims': [512, 256, 128]
        }

    @pytest.fixture
    def dqn_model(self, model_config):
        """Create dueling DQN model."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")
        # Create config object
        config = DuelingDQNConfig(
            state_dim=model_config['state_dim'],
            action_dim=model_config['action_dim'],
            hidden_dims=model_config['hidden_dims']
        )
        return DuelingDQN(config)

    def test_model_initialization(self, dqn_model, model_config):
        """Test model initialization."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        assert dqn_model.state_dim == model_config['state_dim']
        assert dqn_model.action_dim == model_config['action_dim']
        assert dqn_model.config.hidden_dims == model_config['hidden_dims']

        # Check that model has the expected components
        assert hasattr(dqn_model, 'feature_extractor')
        assert hasattr(dqn_model, 'value_stream')
        assert hasattr(dqn_model, 'advantage_stream')

    def test_model_forward_pass(self, dqn_model):
        """Test forward pass through model."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        batch_size = 8
        state_dim = dqn_model.state_dim
        action_dim = dqn_model.action_dim

        # Create input tensor
        states = torch.randn(batch_size, state_dim)

        # Forward pass
        q_values, attention_weights = dqn_model(states)

        # Check output shape
        assert q_values.shape == (batch_size, action_dim)

        # Check output is finite
        assert torch.all(torch.isfinite(q_values))

        # Check attention weights if enabled
        if attention_weights is not None:
            assert attention_weights.shape[0] == batch_size

    def test_model_parameter_count(self, dqn_model):
        """Test model parameter count is reasonable."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        total_params = sum(p.numel() for p in dqn_model.parameters())
        trainable_params = sum(p.numel() for p in dqn_model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

        # Model should not be excessively large (for performance requirements)
        assert total_params < 10_000_000, "Model too large for real-time inference"

    def test_model_gradient_flow(self, dqn_model):
        """Test gradient flow through model."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        # Create input and target
        states = torch.randn(4, dqn_model.state_dim, requires_grad=True)
        targets = torch.randn(4, dqn_model.action_dim)

        # Forward pass (returns tuple)
        outputs, _ = dqn_model(states)
        loss = torch.nn.MSELoss()(outputs, targets)

        # Backward pass
        loss.backward()

        # Check gradients
        assert states.grad is not None
        assert torch.all(torch.isfinite(states.grad))

        # Check model gradients
        for param in dqn_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))

    def test_model_device_compatibility(self, dqn_model):
        """Test model on different devices."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        # Test on CPU
        cpu_states = torch.randn(2, dqn_model.state_dim)
        cpu_outputs, _ = dqn_model(cpu_states)
        assert cpu_outputs.device.type == 'cpu'

        # Test on CUDA if available
        if torch.cuda.is_available():
            dqn_model.cuda()
            cuda_states = torch.randn(2, dqn_model.state_dim).cuda()
            cuda_outputs, _ = dqn_model(cuda_states)
            assert cuda_outputs.device.type == 'cuda'

    def test_dueling_architecture(self, dqn_model):
        """Test dueling architecture produces reasonable value/advantage decomposition."""
        if DuelingDQN is None:
            pytest.skip("DuelingDQN not implemented yet")

        states = torch.randn(8, dqn_model.state_dim)
        q_values, _ = dqn_model(states)

        # Q-values should be reasonable
        assert torch.all(q_values >= -100)  # Not extremely negative
        assert torch.all(q_values <= 100)   # Not extremely positive

        # Different states should produce different Q-values
        states2 = torch.randn(8, dqn_model.state_dim)
        q_values2, _ = dqn_model(states2)

        # Should not be identical
        assert not torch.allclose(q_values, q_values2, atol=1e-3)


class TestModelComponents:
    """Test individual model components."""

    def test_multi_head_attention(self):
        """Test multi-head attention component."""
        if MultiHeadAttention is None:
            pytest.skip("MultiHeadAttention not implemented yet")

        batch_size = 4
        seq_len = 10
        embed_dim = 64
        num_heads = 8

        attention = MultiHeadAttention(embed_dim, num_heads)
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)

        output = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.all(torch.isfinite(output))

    def test_residual_block(self):
        """Test residual block component."""
        if ResidualBlock is None:
            pytest.skip("ResidualBlock not implemented yet")

        batch_size = 4
        input_dim = 128

        residual_block = ResidualBlock(input_dim)
        inputs = torch.randn(batch_size, input_dim)

        outputs = residual_block(inputs)

        assert outputs.shape == (batch_size, input_dim)
        assert torch.all(torch.isfinite(outputs))

        # Should have residual connection (outputs different from pure transformation)
        pure_transform = torch.nn.Linear(input_dim, input_dim)(inputs)
        assert not torch.allclose(outputs, pure_transform, atol=1e-3)
        assert hasattr(state_extractor, 'extract_state')
        assert hasattr(state_extractor, 'state_dim')
        assert state_extractor.state_dim >= 380  # Constitutional requirement

    def test_basic_state_extraction(self, state_extractor, sample_game_state):
        """Test basic state extraction from game state."""
        state_vector = state_extractor.extract_state(sample_game_state)

        # Check dimensions
        assert isinstance(state_vector, np.ndarray)
        assert len(state_vector) == state_extractor.state_dim
        assert state_vector.dtype == np.float32

        # Check that features are normalized
        assert np.all(np.isfinite(state_vector))
        assert not np.any(np.isnan(state_vector))

    def test_turn_number_encoding(self, state_extractor):
        """Test turn number feature encoding."""
        # Test early game
        early_state = {'turn_number': 3}
        early_vector = state_extractor.extract_state(early_state)

        # Test late game
        late_state = {'turn_number': 15}
        late_vector = state_extractor.extract_state(late_state)

        # Late game should have different encoding
        turn_diff = np.abs(late_vector - early_vector)
        assert np.any(turn_diff > 0.01)  # Some features should differ

    def test_life_encoding(self, state_extractor):
        """Test life total feature encoding."""
        # Test healthy life
        healthy_state = {'player_life': 20, 'opponent_life': 15}
        healthy_vector = state_extractor.extract_state(healthy_state)

        # Test low life
        low_life_state = {'player_life': 5, 'opponent_life': 20}
        low_life_vector = state_extractor.extract_state(low_life_state)

        # Low life should trigger different encoding
        life_diff = np.abs(low_life_vector - healthy_vector)
        assert np.any(life_diff > 0.01)

    def test_hand_encoding(self, state_extractor):
        """Test hand feature encoding."""
        # Test empty hand
        empty_hand_state = {'hand': []}
        empty_vector = state_extractor.extract_state(empty_hand_state)

        # Test full hand
        full_hand_state = {
            'hand': [
                {'type': 'creature', 'cost': i}
                for i in range(7)
            ]
        }
        full_vector = state_extractor.extract_state(full_hand_state)

        # Hand size should be reflected in state
        hand_diff = np.abs(full_vector - empty_vector)
        assert np.any(hand_diff > 0.01)

    def test_board_encoding(self, state_extractor):
        """Test board state encoding."""
        # Test empty board
        empty_board_state = {'board': []}
        empty_vector = state_extractor.extract_state(empty_board_state)

        # Test board with creatures
        board_state = {
            'board': [
                {'name': 'Creature1', 'power': 2, 'toughness': 1, 'tapped': False},
                {'name': 'Creature2', 'power': 3, 'toughness': 3, 'tapped': True}
            ]
        }
        board_vector = state_extractor.extract_state(board_state)

        # Board presence should change state
        board_diff = np.abs(board_vector - empty_vector)
        assert np.any(board_diff > 0.01)

    def test_mana_encoding(self, state_extractor):
        """Test mana pool encoding."""
        # Test no mana
        no_mana_state = {'mana_pool': {}}
        no_mana_vector = state_extractor.extract_state(no_mana_state)

        # Test with mana
        mana_state = {'mana_pool': {'R': 2, 'G': 1}}
        mana_vector = state_extractor.extract_state(mana_state)

        # Mana should be reflected in state
        mana_diff = np.abs(mana_vector - no_mana_vector)
        assert np.any(mana_diff > 0.01)

    def test_phase_encoding(self, state_extractor):
        """Test phase encoding."""
        phases = ['main', 'combat', 'end']
        vectors = []

        for phase in phases:
            state = {'phase': phase}
            vector = state_extractor.extract_state(state)
            vectors.append(vector)

        # Different phases should have different encodings
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                phase_diff = np.abs(vectors[i] - vectors[j])
                assert np.any(phase_diff > 0.01)

    def test_state_dimension_requirement(self, state_extractor):
        """Test that state meets 380+ dimension requirement."""
        # Test with various game states
        test_states = [
            {'turn_number': 1},
            {'turn_number': 10, 'hand': [{'type': 'creature'} for _ in range(7)]},
            {'turn_number': 15, 'board': [{'power': i, 'toughness': i} for i in range(5)]},
            {'turn_number': 20, 'hand': [], 'board': [], 'mana_pool': {}}
        ]

        for test_state in test_states:
            state_vector = state_extractor.extract_state(test_state)
            assert len(state_vector) >= 380, f"State dimension {len(state_vector)} < 380 requirement"

    def test_state_consistency(self, state_extractor, sample_game_state):
        """Test state extraction consistency."""
        # Extract state multiple times
        vector1 = state_extractor.extract_state(sample_game_state)
        vector2 = state_extractor.extract_state(sample_game_state)
        vector3 = state_extractor.extract_state(sample_game_state)

        # Should be identical
        np.testing.assert_array_equal(vector1, vector2)
        np.testing.assert_array_equal(vector2, vector3)

    def test_state_normalization(self, state_extractor):
        """Test state feature normalization."""
        # Test extreme values
        extreme_state = {
            'turn_number': 100,  # Very high turn number
            'player_life': 100,  # Very high life
            'hand': [{'type': 'creature'} for _ in range(20)],  # Very large hand
            'board': [{'power': 100, 'toughness': 100} for _ in range(10)]  # Very powerful board
        }

        state_vector = state_extractor.extract_state(extreme_state)

        # All values should be normalized to reasonable range
        assert np.all(np.abs(state_vector) <= 10.0), "State features should be normalized"
        assert np.all(np.isfinite(state_vector)), "State features should be finite"

    def test_missing_features_handling(self, state_extractor):
        """Test handling of missing or None features."""
        # State with missing features
        partial_state = {
            'turn_number': 5
            # Missing many features
        }

        # Should not crash
        state_vector = state_extractor.extract_state(partial_state)
        assert len(state_vector) == state_extractor.state_dim
        assert np.all(np.isfinite(state_vector))

        # State with None values
        none_state = {
            'turn_number': 5,
            'hand': None,
            'board': None
        }

        # Should not crash
        state_vector = state_extractor.extract_state(none_state)
        assert len(state_vector) == state_extractor.state_dim
        assert np.all(np.isfinite(state_vector))

    def test_complex_mtg_scenario(self, state_extractor):
        """Test state extraction for complex MTG scenario."""
        complex_state = {
            'turn_number': 12,
            'phase': 'combat',
            'player_life': 8,
            'opponent_life': 15,
            'hand': [
                {'name': 'Lightning Bolt', 'type': 'instant', 'cost': 'R'},
                {'name': 'Counterspell', 'type': 'instant', 'cost': 'UU'},
                {'name': 'Serra Angel', 'type': 'creature', 'cost': '3WW'},
                {'name': 'Dark Ritual', 'type': 'instant', 'cost': 'B'},
                {'name': 'Forest', 'type': 'land', 'cost': 0}
            ],
            'board': [
                {'name': 'Grizzly Bears', 'power': 2, 'toughness': 2, 'tapped': False},
                {'name': 'Serra Angel', 'power': 4, 'toughness': 4, 'tapped': True, 'flying': True},
                {'name': 'Island', 'tapped': False},
                {'name': 'Mountain', 'tapped': True},
                {'name': 'Plains', 'tapped': False}
            ],
            'opponent_board': [
                {'name': 'Dragons', 'power': 5, 'toughness': 5, 'tapped': False, 'flying': True},
                {'name': 'Swamp', 'tapped': False}
            ],
            'mana_pool': {'W': 2, 'U': 1},
            'graveyard_size': 12,
            'library_size': 35,
            'can_play_land': True,
            'can_attack': True,
            'can_block': True,
            'storm_count': 3,
            'poison_counters': 1,
            'energy_counters': 2
        }

        state_vector = state_extractor.extract_state(complex_state)

        # Verify state is properly encoded
        assert len(state_vector) >= 380
        assert np.all(np.isfinite(state_vector))
        assert state_vector.dtype == np.float32

    def test_state_vector_interpretability(self, state_extractor, sample_game_state):
        """Test that state vector features are interpretable."""
        state_vector = state_extractor.extract_state(sample_game_state)

        # Test that we can identify which features correspond to what
        # This is important for explainability (constitutional requirement)

        # Turn number should be encoded early in the vector
        turn_feature = state_vector[0]  # Assuming turn is first feature
        assert 0 <= turn_feature <= 1.0  # Should be normalized

        # Life totals should be present and normalized
        # Find life-related features (would be based on implementation)
        life_features = state_vector[10:20]  # Example range
        assert np.all(np.isfinite(life_features))

    def test_performance_requirement(self, state_extractor):
        """Test performance requirement for state extraction."""
        # State extraction should be fast (< 1ms for real-time usage)
        import time

        test_state = {
            'turn_number': 10,
            'hand': [{'type': 'creature', 'cost': i} for i in range(5)],
            'board': [{'power': i, 'toughness': i+1} for i in range(3)]
        }

        # Time multiple extractions
        num_extractions = 1000
        start_time = time.time()

        for _ in range(num_extractions):
            state_vector = state_extractor.extract_state(test_state)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_extractions * 1000  # Convert to ms

        # Should be fast for real-time usage
        assert avg_time < 1.0, f"State extraction too slow: {avg_time:.3f}ms"


# Duplicate TestDuelingDQN class removed to avoid conflicts


class TestModelComponents:
    """Test individual model components."""

    def test_multi_head_attention(self):
        """Test multi-head attention component."""
        batch_size = 4
        seq_len = 10
        embed_dim = 64
        num_heads = 8

        attention = MultiHeadAttention(embed_dim, num_heads)
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)

        output = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.all(torch.isfinite(output))

    def test_residual_block(self):
        """Test residual block component."""
        batch_size = 4
        input_dim = 128

        residual_block = ResidualBlock(input_dim)
        inputs = torch.randn(batch_size, input_dim)

        outputs = residual_block(inputs)

        assert outputs.shape == (batch_size, input_dim)
        assert torch.all(torch.isfinite(outputs))

        # Should have residual connection (outputs different from pure transformation)
        pure_transform = torch.nn.Linear(input_dim, input_dim)(inputs)
        assert not torch.allclose(outputs, pure_transform, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])