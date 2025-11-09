"""
Unit tests for multi-dimensional reward function for MTG RL system.

Tests multi-component reward function in src/rl/data/reward_function.py
validating game outcome, life/card/board advantage, tempo, strategic progress components.

This is Task T028 for User Story 1 - Enhanced AI Decision Quality.
Following Red-Green-Refactor approach - tests should FAIL initially.
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import json

# Import the reward function components (these will fail initially)
from src.rl.data.reward_function import (
    MTGRewardFunction,           # Expected class name (will fail)
    RewardConfig,                # Expected config class (will fail)
    RewardComponent,             # Expected component enum
    GameState,                   # Expected game state class
    RewardBreakdown,             # Expected breakdown class
    RewardWeights,               # Expected weights class
    get_reward_function,         # Expected factory function
    setup_reward_function        # Expected setup function
)


class TestMTGRewardFunction:
    """
    Test multi-dimensional reward function implementation.

    Tests the MTGRewardFunction class with comprehensive validation of:
    - Game outcome rewards (win/loss)
    - Life advantage calculation
    - Card advantage calculation
    - Board advantage calculation
    - Tempo and timing rewards
    - Strategic progress rewards
    - Reward shaping and normalization
    - 17Lands statistical data integration
    - Performance for real-time usage
    """

    @pytest.fixture
    def reward_config(self):
        """Create reward configuration for testing."""
        return RewardConfig(
            # Primary objectives (70% total)
            game_outcome_weight=0.4,
            strategic_progress_weight=0.3,

            # Resource advantages (20% total)
            life_advantage_weight=0.05,
            card_advantage_weight=0.08,
            board_advantage_weight=0.07,

            # Tempo and efficiency (10% total)
            tempo_weight=0.06,
            resource_efficiency_weight=0.04,

            # Disruption and interaction
            opponent_disruption_weight=0.0,

            # Reward shaping parameters
            reward_normalization=True,
            reward_clipping_range=(-2.0, 2.0),
            temporal_decay_factor=0.98,

            # 17Lands integration
            use_17lands_calibration=True,
            calibration_data_path="data/17lands_calibration.json"
        )

    @pytest.fixture
    def reward_function(self, reward_config):
        """Create MTG reward function for testing."""
        return MTGRewardFunction(config=reward_config)

    @pytest.fixture
    def sample_game_states(self):
        """Create sample game states for comprehensive reward testing."""
        return {
            'win_state': {
                'current': GameState(
                    # Player state
                    life=20,
                    hand_size=3,
                    battlefield_size=5,
                    lands_in_play=7,
                    creatures_in_play=4,
                    library_size=35,
                    graveyard_size=8,

                    # Mana and resources
                    available_mana=7,
                    total_mana_production=7,

                    # Game information
                    turn_number=15,
                    phase="end",
                    storm_count=0,

                    # Advanced state
                    board_power=12,
                    board_toughness=15,
                    opponent_life=0,
                    opponent_hand_size=2,
                    opponent_battlefield_size=1,

                    # Action information
                    action_type="attack",
                    mana_spent=3,
                    cards_drawn=1,
                    damage_dealt=20
                ),
                'previous': GameState(
                    life=18,
                    hand_size=4,
                    battlefield_size=4,
                    lands_in_play=7,
                    creatures_in_play=3,
                    library_size=36,
                    graveyard_size=7,
                    available_mana=7,
                    total_mana_production=7,
                    turn_number=14,
                    phase="combat",
                    storm_count=0,
                    board_power=9,
                    board_toughness=12,
                    opponent_life=5,
                    opponent_hand_size=3,
                    opponent_battlefield_size=2,
                    action_type="cast_creature",
                    mana_spent=4,
                    cards_drawn=0,
                    damage_dealt=0
                ),
                'game_won': True
            },

            'loss_state': {
                'current': GameState(
                    life=0,
                    hand_size=0,
                    battlefield_size=2,
                    lands_in_play=6,
                    creatures_in_play=1,
                    library_size=25,
                    graveyard_size=15,
                    available_mana=6,
                    total_mana_production=6,
                    turn_number=12,
                    phase="end",
                    storm_count=0,
                    board_power=2,
                    board_toughness=3,
                    opponent_life=20,
                    opponent_hand_size=5,
                    opponent_battlefield_size=6,
                    action_type="pass_turn",
                    mana_spent=0,
                    cards_drawn=1,
                    damage_dealt=0
                ),
                'previous': GameState(
                    life=5,
                    hand_size=1,
                    battlefield_size=2,
                    lands_in_play=6,
                    creatures_in_play=1,
                    library_size=26,
                    graveyard_size=14,
                    available_mana=6,
                    total_mana_production=6,
                    turn_number=11,
                    phase="main",
                    storm_count=0,
                    board_power=2,
                    board_toughness=3,
                    opponent_life=15,
                    opponent_hand_size=4,
                    opponent_battlefield_size=5,
                    action_type="block",
                    mana_spent=1,
                    cards_drawn=0,
                    damage_dealt=3
                ),
                'game_won': False
            },

            'tempo_gain_state': {
                'current': GameState(
                    life=18,
                    hand_size=5,
                    battlefield_size=4,
                    lands_in_play=4,
                    creatures_in_play=3,
                    library_size=40,
                    graveyard_size=5,
                    available_mana=4,
                    total_mana_production=4,
                    turn_number=8,
                    phase="post_combat",
                    storm_count=0,
                    board_power=6,
                    board_toughness=7,
                    opponent_life=17,
                    opponent_hand_size=4,
                    opponent_battlefield_size=2,
                    action_type="cast_creature",
                    mana_spent=3,
                    cards_drawn=1,
                    damage_dealt=3
                ),
                'previous': GameState(
                    life=18,
                    hand_size=6,
                    battlefield_size=2,
                    lands_in_play=3,
                    creatures_in_play=1,
                    library_size=41,
                    graveyard_size=4,
                    available_mana=3,
                    total_mana_production=3,
                    turn_number=7,
                    phase="main",
                    storm_count=0,
                    board_power=3,
                    board_toughness=3,
                    opponent_life=20,
                    opponent_hand_size=4,
                    opponent_battlefield_size=2,
                    action_type="play_land",
                    mana_spent=0,
                    cards_drawn=0,
                    damage_dealt=0
                ),
                'game_won': None
            },

            'card_advantage_state': {
                'current': GameState(
                    life=15,
                    hand_size=6,
                    battlefield_size=3,
                    lands_in_play=5,
                    creatures_in_play=2,
                    library_size=38,
                    graveyard_size=6,
                    available_mana=5,
                    total_mana_production=5,
                    turn_number=10,
                    phase="main",
                    storm_count=0,
                    board_power=4,
                    board_toughness=5,
                    opponent_life=16,
                    opponent_hand_size=3,
                    opponent_battlefield_size=3,
                    action_type="draw_spell",
                    mana_spent=2,
                    cards_drawn=2,
                    damage_dealt=0
                ),
                'previous': GameState(
                    life=15,
                    hand_size=4,
                    battlefield_size=3,
                    lands_in_play=5,
                    creatures_in_play=2,
                    library_size=40,
                    graveyard_size=6,
                    available_mana=5,
                    total_mana_production=5,
                    turn_number=9,
                    phase="end",
                    storm_count=0,
                    board_power=4,
                    board_toughness=5,
                    opponent_life=16,
                    opponent_hand_size=4,
                    opponent_battlefield_size=3,
                    action_type="pass_turn",
                    mana_spent=0,
                    cards_drawn=1,
                    damage_dealt=0
                ),
                'game_won': None
            },

            'board_advantage_state': {
                'current': GameState(
                    life=12,
                    hand_size=3,
                    battlefield_size=6,
                    lands_in_play=6,
                    creatures_in_play=4,
                    library_size=32,
                    graveyard_size=8,
                    available_mana=6,
                    total_mana_production=6,
                    turn_number=12,
                    phase="combat",
                    storm_count=0,
                    board_power=11,
                    board_toughness=13,
                    opponent_life=14,
                    opponent_hand_size=4,
                    opponent_battlefield_size=3,
                    action_type="attack",
                    mana_spent=0,
                    cards_drawn=0,
                    damage_dealt=6
                ),
                'previous': GameState(
                    life=12,
                    hand_size=4,
                    battlefield_size=4,
                    lands_in_play=6,
                    creatures_in_play=2,
                    library_size=33,
                    graveyard_size=7,
                    available_mana=6,
                    total_mana_production=6,
                    turn_number=11,
                    phase="main",
                    storm_count=0,
                    board_power=5,
                    board_toughness=6,
                    opponent_life=14,
                    opponent_hand_size=4,
                    opponent_battlefield_size=3,
                    action_type="cast_creature",
                    mana_spent=4,
                    cards_drawn=0,
                    damage_dealt=0
                ),
                'game_won': None
            }
        }

    def test_mtg_reward_function_initialization(self, reward_function, reward_config):
        """Test MTGRewardFunction initialization with configuration."""
        assert reward_function.config == reward_config
        assert hasattr(reward_function, 'calculate_reward')
        assert hasattr(reward_function, 'get_reward_breakdown')
        assert hasattr(reward_function, 'validate_reward_function')
        assert hasattr(reward_function, 'calibrate_from_17lands_data')

        # Check that 17Lands calibration is loaded
        assert hasattr(reward_function, 'lands_17lands_calibration')
        assert isinstance(reward_function.lands_17lands_calibration, dict)

    def test_game_outcome_reward_calculation(self, reward_function, sample_game_states):
        """Test game outcome reward calculation for win/loss scenarios."""
        win_data = sample_game_states['win_state']
        loss_data = sample_game_states['loss_state']

        # Test win reward
        win_breakdown = reward_function.calculate_reward(
            win_data['current'], win_data['previous'], game_won=True
        )
        assert win_breakdown.total_reward > 0.5  # Should be significantly positive
        assert RewardComponent.GAME_OUTCOME in win_breakdown.components
        assert win_breakdown.components[RewardComponent.GAME_OUTCOME] > 0

        # Test loss penalty
        loss_breakdown = reward_function.calculate_reward(
            loss_data['current'], loss_data['previous'], game_won=False
        )
        assert loss_breakdown.total_reward < -0.5  # Should be significantly negative
        assert RewardComponent.GAME_OUTCOME in loss_breakdown.components
        assert loss_breakdown.components[RewardComponent.GAME_OUTCOME] < 0

    def test_life_advantage_reward_calculation(self, reward_function, sample_game_states):
        """Test life advantage reward calculation."""
        # Test life gain scenario
        life_gain_state = {
            'current': GameState(
                life=25, opponent_life=15, hand_size=4, battlefield_size=3,
                lands_in_play=5, creatures_in_play=2, library_size=35, graveyard_size=5,
                available_mana=5, total_mana_production=5, turn_number=8, phase="main",
                storm_count=0, board_power=4, board_toughness=5, opponent_hand_size=4,
                opponent_battlefield_size=2, action_type="heal", mana_spent=2,
                cards_drawn=0, damage_dealt=0
            ),
            'previous': GameState(
                life=20, opponent_life=15, hand_size=4, battlefield_size=3,
                lands_in_play=5, creatures_in_play=2, library_size=35, graveyard_size=5,
                available_mana=5, total_mana_production=5, turn_number=7, phase="main",
                storm_count=0, board_power=4, board_toughness=5, opponent_hand_size=4,
                opponent_battlefield_size=2, action_type="pass_turn", mana_spent=0,
                cards_drawn=1, damage_dealt=0
            )
        }

        breakdown = reward_function.calculate_reward(
            life_gain_state['current'], life_gain_state['previous']
        )

        assert RewardComponent.LIFE_ADVANTAGE in breakdown.components
        life_advantage = breakdown.components[RewardComponent.LIFE_ADVANTAGE]
        assert life_advantage > 0  # Gaining life should be positive

        # Test life loss scenario
        life_loss_state = {
            'current': GameState(life=15, opponent_life=20, hand_size=4, battlefield_size=2,
                               lands_in_play=4, creatures_in_play=1, library_size=38, graveyard_size=6,
                               available_mana=4, total_mana_production=4, turn_number=9, phase="end",
                               storm_count=0, board_power=2, board_toughness=2, opponent_hand_size=5,
                               opponent_battlefield_size=3, action_type="receive_damage", mana_spent=0,
                               cards_drawn=0, damage_dealt=0),
            'previous': GameState(life=20, opponent_life=20, hand_size=4, battlefield_size=2,
                                lands_in_play=4, creatures_in_play=1, library_size=38, graveyard_size=6,
                                available_mana=4, total_mana_production=4, turn_number=8, phase="main",
                                storm_count=0, board_power=2, board_toughness=2, opponent_hand_size=5,
                                opponent_battlefield_size=3, action_type="pass_turn", mana_spent=0,
                                cards_drawn=1, damage_dealt=0)
        }

        breakdown = reward_function.calculate_reward(
            life_loss_state['current'], life_loss_state['previous']
        )

        life_advantage = breakdown.components[RewardComponent.LIFE_ADVANTAGE]
        assert life_advantage < 0  # Losing life should be negative

    def test_card_advantage_reward_calculation(self, reward_function, sample_game_states):
        """Test card advantage reward calculation."""
        card_adv_data = sample_game_states['card_advantage_state']

        breakdown = reward_function.calculate_reward(
            card_adv_data['current'], card_adv_data['previous']
        )

        assert RewardComponent.CARD_ADVANTAGE in breakdown.components
        card_advantage = breakdown.components[RewardComponent.CARD_ADVANTAGE]
        assert card_advantage > 0  # Gaining card advantage should be positive

        # Verify it considers hand size change
        hand_change = card_adv_data['current'].hand_size - card_adv_data['previous'].hand_size
        assert hand_change > 0  # Should have gained cards

    def test_board_advantage_reward_calculation(self, reward_function, sample_game_states):
        """Test board advantage reward calculation."""
        board_adv_data = sample_game_states['board_advantage_state']

        breakdown = reward_function.calculate_reward(
            board_adv_data['current'], board_adv_data['previous']
        )

        assert RewardComponent.BOARD_ADVANTAGE in breakdown.components
        board_advantage = breakdown.components[RewardComponent.BOARD_ADVANTAGE]
        assert board_advantage > 0  # Gaining board advantage should be positive

        # Should consider creature count increase
        creature_change = board_adv_data['current'].creatures_in_play - board_adv_data['previous'].creatures_in_play
        assert creature_change > 0  # Should have gained creatures

        # Should consider power/toughness increase
        power_change = board_adv_data['current'].board_power - board_adv_data['previous'].board_power
        toughness_change = board_adv_data['current'].board_toughness - board_adv_data['previous'].board_toughness
        assert power_change > 0 and toughness_change > 0

    def test_tempo_reward_calculation(self, reward_function, sample_game_states):
        """Test tempo reward calculation for timing and efficiency."""
        tempo_data = sample_game_states['tempo_gain_state']

        breakdown = reward_function.calculate_reward(
            tempo_data['current'], tempo_data['previous']
        )

        assert RewardComponent.TEMPO in breakdown.components
        tempo_reward = breakdown.components[RewardComponent.TEMPO]
        assert tempo_reward > 0  # Good tempo should be positive

        # Should reward efficient mana usage
        mana_efficiency = tempo_data['current'].mana_spent / max(tempo_data['current'].available_mana, 1)
        assert mana_efficiency > 0.5  # Should be reasonably efficient

    def test_strategic_progress_reward_calculation(self, reward_function, sample_game_states):
        """Test strategic progress reward calculation."""
        breakdown = reward_function.calculate_reward(
            sample_game_states['tempo_gain_state']['current'],
            sample_game_states['tempo_gain_state']['previous']
        )

        assert RewardComponent.STRATEGIC_PROGRESS in breakdown.components
        strategic_progress = breakdown.components[RewardComponent.STRATEGIC_PROGRESS]

        # Should calculate based on overall position improvement
        # The exact value depends on the implementation but should be reasonable
        assert -0.5 <= strategic_progress <= 0.5

    def test_reward_function_multidimensionality(self, reward_function, sample_game_states):
        """Test that reward function produces multi-dimensional outputs."""
        breakdown = reward_function.calculate_reward(
            sample_game_states['win_state']['current'],
            sample_game_states['win_state']['previous'],
            game_won=True
        )

        # Should have all expected components
        expected_components = [
            RewardComponent.GAME_OUTCOME,
            RewardComponent.LIFE_ADVANTAGE,
            RewardComponent.CARD_ADVANTAGE,
            RewardComponent.BOARD_ADVANTAGE,
            RewardComponent.TEMPO,
            RewardComponent.STRATEGIC_PROGRESS,
            RewardComponent.RESOURCE_EFFICIENCY,
            RewardComponent.OPPONENT_DISRUPTION
        ]

        for component in expected_components:
            assert component in breakdown.components, f"Missing component: {component}"
            assert isinstance(breakdown.components[component], float)

        # Total reward should be weighted sum of components
        calculated_total = 0.0
        for component, value in breakdown.components.items():
            weight = getattr(reward_function.config, f"{component.value}_weight")
            calculated_total += value * weight

        assert abs(calculated_total - breakdown.total_reward) < 1e-6

    def test_reward_normalization_and_shaping(self, reward_function):
        """Test reward normalization and shaping mechanisms."""
        # Test extreme cases to verify normalization
        extreme_states = [
            # Extreme positive
            (GameState(life=100, opponent_life=0, hand_size=10, battlefield_size=10,
                      lands_in_play=10, creatures_in_play=10, library_size=50, graveyard_size=10,
                      available_mana=10, total_mana_production=10, turn_number=1, phase="main",
                      storm_count=0, board_power=50, board_toughness=50, opponent_hand_size=0,
                      opponent_battlefield_size=0, action_type="win", mana_spent=0,
                      cards_drawn=0, damage_dealt=100),
             GameState(life=20, opponent_life=20, hand_size=5, battlefield_size=3,
                      lands_in_play=3, creatures_in_play=2, library_size=47, graveyard_size=5,
                      available_mana=3, total_mana_production=3, turn_number=1, phase="start",
                      storm_count=0, board_power=4, board_toughness=5, opponent_hand_size=5,
                      opponent_battlefield_size=2, action_type="start", mana_spent=0,
                      cards_drawn=0, damage_dealt=0)),

            # Extreme negative
            (GameState(life=0, opponent_life=100, hand_size=0, battlefield_size=0,
                      lands_in_play=0, creatures_in_play=0, library_size=0, graveyard_size=60,
                      available_mana=0, total_mana_production=0, turn_number=20, phase="end",
                      storm_count=0, board_power=0, board_toughness=0, opponent_hand_size=10,
                      opponent_battlefield_size=10, action_type="lose", mana_spent=0,
                      cards_drawn=0, damage_dealt=0),
             GameState(life=20, opponent_life=20, hand_size=5, battlefield_size=3,
                      lands_in_play=5, creatures_in_play=2, library_size=40, graveyard_size=10,
                      available_mana=5, total_mana_production=5, turn_number=10, phase="main",
                      storm_count=0, board_power=6, board_toughness=7, opponent_hand_size=4,
                      opponent_battlefield_size=3, action_type="play", mana_spent=2,
                      cards_drawn=1, damage_dealt=0))
        ]

        for current, previous in extreme_states:
            breakdown = reward_function.calculate_reward(current, previous)

            # Should be within configured clipping range
            assert reward_function.config.reward_clipping_range[0] <= breakdown.total_reward <= reward_function.config.reward_clipping_range[1]

    def test_17lands_data_integration(self, reward_function):
        """Test 17Lands statistical data integration for calibration."""
        # Should have 17Lands calibration data loaded
        assert hasattr(reward_function, 'lands_17lands_calibration')

        # Calibration should include key metrics
        expected_calibration_keys = [
            'life_advantage_win_rate',
            'card_advantage_win_rate',
            'board_advantage_win_rate',
            'tempo_advantage_win_rate',
            'resource_efficiency_win_rate'
        ]

        for key in expected_calibration_keys:
            assert key in reward_function.lands_17lands_calibration
            assert isinstance(reward_function.lands_17lands_calibration[key], float)
            assert 0.0 <= reward_function.lands_17lands_calibration[key] <= 1.0

    def test_reward_function_calibration_with_17lands(self, reward_function):
        """Test reward function calibration using 17Lands data."""
        # Mock 17Lands win rate data
        mock_17lands_data = {
            'life_advantage_win_rate': 0.65,
            'card_advantage_win_rate': 0.72,
            'board_advantage_win_rate': 0.78,
            'tempo_advantage_win_rate': 0.58,
            'resource_efficiency_win_rate': 0.61
        }

        # Calibrate reward function
        reward_function.calibrate_from_17lands_data(mock_17lands_data)

        # Verify calibration was applied
        for metric, win_rate in mock_17lands_data.items():
            calibrated_value = reward_function.lands_17lands_calibration.get(metric)
            assert calibrated_value is not None
            assert abs(calibrated_value - win_rate) < 0.01

    def test_reward_function_different_game_phases(self, reward_function):
        """Test reward function behavior across different game phases."""
        phases = ["start", "upkeep", "main", "combat", "post_combat", "end"]

        base_state = GameState(
            life=20, hand_size=5, battlefield_size=3, lands_in_play=4,
            creatures_in_play=2, library_size=40, graveyard_size=5,
            available_mana=4, total_mana_production=4, turn_number=8,
            phase="main", storm_count=0, board_power=5, board_toughness=6,
            opponent_life=18, opponent_hand_size=4, opponent_battlefield_size=2,
            action_type="play", mana_spent=2, cards_drawn=1, damage_dealt=0
        )

        phase_rewards = {}

        for phase in phases:
            current_state = GameState(
                **{k: v for k, v in base_state.__dict__.items() if k != 'phase'},
                phase=phase
            )

            previous_state = GameState(
                **{k: v for k, v in base_state.__dict__.items() if k != 'phase'},
                phase=phases[(phases.index(phase) - 1) % len(phases)]
            )

            breakdown = reward_function.calculate_reward(current_state, previous_state)
            phase_rewards[phase] = breakdown.total_reward

        # All phase rewards should be reasonable
        for phase, reward in phase_rewards.items():
            assert -2.0 <= reward <= 2.0, f"Unreasonable reward for phase {phase}: {reward}"

    def test_reward_function_win_rate_optimization_alignment(self, reward_function):
        """Test that reward function aligns with win rate optimization."""
        # Create scenarios with known win probability differences
        scenarios = [
            # High win probability scenario (strong board advantage)
            {
                'name': 'strong_board',
                'current': GameState(life=20, hand_size=4, battlefield_size=8,
                                   lands_in_play=6, creatures_in_play=5, library_size=35,
                                   graveyard_size=8, available_mana=6, total_mana_production=6,
                                   turn_number=10, phase="combat", storm_count=0,
                                   board_power=15, board_toughness=18, opponent_life=10,
                                   opponent_hand_size=3, opponent_battlefield_size=1,
                                   action_type="attack", mana_spent=0, cards_drawn=0, damage_dealt=8),
                'expected_win_rate': 0.85
            },
            # Low win probability scenario (weak position)
            {
                'name': 'weak_position',
                'current': GameState(life=5, hand_size=1, battlefield_size=1,
                                   lands_in_play=2, creatures_in_play=1, library_size=25,
                                   graveyard_size=15, available_mana=2, total_mana_production=2,
                                   turn_number=12, phase="end", storm_count=0,
                                   board_power=1, board_toughness=1, opponent_life=20,
                                   opponent_hand_size=7, opponent_battlefield_size=6,
                                   action_type="pass_turn", mana_spent=0, cards_drawn=1, damage_dealt=0),
                'expected_win_rate': 0.15
            }
        ]

        base_state = GameState(
            life=15, hand_size=3, battlefield_size=3, lands_in_play=4,
            creatures_in_play=2, library_size=38, graveyard_size=10,
            available_mana=4, total_mana_production=4, turn_number=8,
            phase="main", storm_count=0, board_power=4, board_toughness=5,
            opponent_life=17, opponent_hand_size=5, opponent_battlefield_size=3,
            action_type="play", mana_spent=2, cards_drawn=0, damage_dealt=0
        )

        for scenario in scenarios:
            breakdown = reward_function.calculate_reward(
                scenario['current'], base_state
            )

            # Higher win rate scenarios should have higher rewards
            assert breakdown.total_reward > 0, f"Positive scenario should have positive reward: {scenario['name']}"

            # Check strategic progress component specifically
            strategic_progress = breakdown.components[RewardComponent.STRATEGIC_PROGRESS]
            if scenario['expected_win_rate'] > 0.7:
                assert strategic_progress > 0, f"High win rate scenario should have positive strategic progress: {scenario['name']}"

    def test_reward_function_performance_validation(self, reward_function, sample_game_states):
        """Test reward function performance for real-time usage."""
        # Performance requirement: < 1ms per reward calculation
        max_allowed_time_ms = 1.0

        test_cases = [
            (sample_game_states['win_state']['current'], sample_game_states['win_state']['previous']),
            (sample_game_states['loss_state']['current'], sample_game_states['loss_state']['previous']),
            (sample_game_states['tempo_gain_state']['current'], sample_game_states['tempo_gain_state']['previous']),
            (sample_game_states['card_advantage_state']['current'], sample_game_states['card_advantage_state']['previous']),
            (sample_game_states['board_advantage_state']['current'], sample_game_states['board_advantage_state']['previous'])
        ]

        times = []
        for current, previous in test_cases:
            start_time = time.time()
            breakdown = reward_function.calculate_reward(current, previous)
            end_time = time.time()

            calculation_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(calculation_time)

            # Each calculation should be fast
            assert calculation_time < max_allowed_time_ms, \
                f"Reward calculation too slow: {calculation_time:.3f}ms (max: {max_allowed_time_ms}ms)"

        # Average should also be fast
        avg_time = np.mean(times)
        assert avg_time < max_allowed_time_ms * 0.5, \
            f"Average reward calculation too slow: {avg_time:.3f}ms"

    def test_reward_function_consistency_and_determinism(self, reward_function, sample_game_states):
        """Test that reward function is deterministic and consistent."""
        current = sample_game_states['tempo_gain_state']['current']
        previous = sample_game_states['tempo_gain_state']['previous']

        # Calculate reward multiple times
        rewards = []
        breakdowns = []

        for _ in range(10):
            breakdown = reward_function.calculate_reward(current, previous)
            rewards.append(breakdown.total_reward)
            breakdowns.append(breakdown)

        # All rewards should be identical
        assert len(set(rewards)) == 1, "Reward function should be deterministic"

        # All breakdowns should be identical
        for i in range(1, len(breakdowns)):
            assert breakdowns[0].total_reward == breakdowns[i].total_reward
            assert breakdowns[0].components == breakdowns[i].components

    def test_reward_function_explainability(self, reward_function, sample_game_states):
        """Test reward function explainability features."""
        breakdown = reward_function.calculate_reward(
            sample_game_states['win_state']['current'],
            sample_game_states['win_state']['previous'],
            game_won=True
        )

        # Should have detailed breakdown
        assert isinstance(breakdown, RewardBreakdown)
        assert breakdown.explanation is not None
        assert isinstance(breakdown.explanation, dict)

        # Explanation should include component details
        assert 'component_breakdown' in breakdown.explanation
        assert 'state_changes' in breakdown.explanation
        assert 'key_factors' in breakdown.explanation

        # Should be able to generate human-readable explanation
        explanation_text = reward_function.get_reward_explanation(breakdown)
        assert isinstance(explanation_text, str)
        assert len(explanation_text) > 0
        assert "Total reward:" in explanation_text

    def test_reward_function_validation(self, reward_function):
        """Test reward function validation against constitutional requirements."""
        # Generate some reward data first
        for i in range(100):
            mock_current = GameState(
                life=20 - i % 10, hand_size=5 - i % 3, battlefield_size=3 + i % 2,
                lands_in_play=4, creatures_in_play=2, library_size=40 - i,
                graveyard_size=5 + i, available_mana=4, total_mana_production=4,
                turn_number=8 + i % 10, phase="main", storm_count=0,
                board_power=4 + i % 3, board_toughness=5 + i % 2,
                opponent_life=18 - i % 8, opponent_hand_size=4 - i % 2,
                opponent_battlefield_size=3, action_type="test", mana_spent=i % 3,
                cards_drawn=i % 2, damage_dealt=i % 4
            )
            mock_previous = GameState(
                life=20, hand_size=5, battlefield_size=3, lands_in_play=4,
                creatures_in_play=2, library_size=40, graveyard_size=5,
                available_mana=4, total_mana_production=4, turn_number=8,
                phase="main", storm_count=0, board_power=4, board_toughness=5,
                opponent_life=18, opponent_hand_size=4, opponent_battlefield_size=3,
                action_type="test", mana_spent=0, cards_drawn=0, damage_dealt=0
            )

            reward_function.calculate_reward(mock_current, mock_previous)

        # Validate reward function
        validation_results = reward_function.validate_reward_function()

        assert isinstance(validation_results, dict)
        assert 'compliant' in validation_results
        assert 'violations' in validation_results
        assert 'validation_time' in validation_results
        assert 'statistics' in validation_results

        # Should have sufficient data for validation
        assert len(reward_function.reward_history) >= 100

    def test_reward_function_factory_functions(self):
        """Test reward function factory functions."""
        # Test get_reward_function
        reward_func = get_reward_function()
        assert isinstance(reward_func, MTGRewardFunction)

        # Test setup_reward_function with custom config
        custom_config = RewardConfig(
            game_outcome_weight=0.5,
            strategic_progress_weight=0.2,
            life_advantage_weight=0.1,
            card_advantage_weight=0.1,
            board_advantage_weight=0.1
        )

        custom_reward_func = setup_reward_function(weights=custom_config)
        assert isinstance(custom_reward_func, MTGRewardFunction)
        assert custom_reward_func.config == custom_config

    def test_reward_edge_cases_and_error_handling(self, reward_function):
        """Test reward function edge cases and error handling."""
        # Test with identical states (no change)
        identical_state = GameState(
            life=20, hand_size=5, battlefield_size=3, lands_in_play=4,
            creatures_in_play=2, library_size=40, graveyard_size=5,
            available_mana=4, total_mana_production=4, turn_number=8,
            phase="main", storm_count=0, board_power=4, board_toughness=5,
            opponent_life=18, opponent_hand_size=4, opponent_battlefield_size=3,
            action_type="pass", mana_spent=0, cards_drawn=1, damage_dealt=0
        )

        breakdown = reward_function.calculate_reward(identical_state, identical_state)
        assert isinstance(breakdown.total_reward, float)
        assert not np.isnan(breakdown.total_reward)
        assert not np.isinf(breakdown.total_reward)

    def test_reward_function_memory_efficiency(self, reward_function):
        """Test reward function memory efficiency."""
        import sys

        # Calculate many rewards to test memory usage
        base_state = GameState(
            life=20, hand_size=5, battlefield_size=3, lands_in_play=4,
            creatures_in_play=2, library_size=40, graveyard_size=5,
            available_mana=4, total_mana_production=4, turn_number=8,
            phase="main", storm_count=0, board_power=4, board_toughness=5,
            opponent_life=18, opponent_hand_size=4, opponent_battlefield_size=3,
            action_type="test", mana_spent=2, cards_drawn=1, damage_dealt=0
        )

        # Generate 1000 reward calculations
        for i in range(1000):
            current_state = GameState(
                **{k: v for k, v in base_state.__dict__.items()},
                turn_number=base_state.turn_number + i,
                life=base_state.life - i % 5
            )

            reward_function.calculate_reward(current_state, base_state)

        # Check that reward history is managed properly
        assert len(reward_function.reward_history) <= 10000  # Should have some limit

        # Memory usage should be reasonable (this is a rough check)
        reward_func_size = sys.getsizeof(reward_function)
        assert reward_func_size < 10 * 1024 * 1024  # Less than 10MB


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])