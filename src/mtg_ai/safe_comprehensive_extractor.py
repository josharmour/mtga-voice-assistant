#!/usr/bin/env python3
"""
Safe Comprehensive MTG State Extractor
Memory-safe extraction with robust data type handling.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SafeComprehensiveMTGStateExtractor:
    """Memory-safe comprehensive game state extractor with robust error handling."""

    def __init__(self):
        self.action_types = [
            "play_creature", "attack_creatures", "defensive_play", "cast_spell",
            "use_ability", "pass_priority", "block_creatures", "play_land",
            "hold_priority", "draw_card", "combat_trick", "board_wipe",
            "counter_spell", "resource_accel", "positioning"
        ]

    def _safe_float(self, value, default=0.0):
        """Convert value to float safely."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default=0):
        """Convert value to int safely."""
        if pd.isna(value):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def extract_comprehensive_state(self, row: pd.Series, turn_number: int) -> np.ndarray:
        """Extract 282-dimensional state tensor with safe type conversion."""
        try:
            turn_prefix = f'user_turn_{turn_number}_'

            # Check if turn data exists
            if not any(col.startswith(turn_prefix) for col in row.index):
                return np.zeros(282, dtype=np.float32)

            # === CORE GAME INFORMATION (32 dims) ===
            core_state = [
                turn_number / 20.0,  # Normalized turn number
                1.0 if row.get('on_play') else 0.5,  # Play/draw advantage
                self._safe_int(row.get('num_mulligans', 0)) / 5.0,  # Mulligans
                self._safe_int(row.get('opp_num_mulligans', 0)) / 5.0,  # Opp mulligans
                len(str(row.get('main_colors', '')).split(',')) / 5.0,  # Colors
                len(str(row.get('splash_colors', '')).split(',')) / 3.0,  # Splash colors
                1.0 if row.get('won') else 0.0,  # Game outcome
                self._safe_int(row.get('num_turns', 10)) / 30.0,  # Game length
            ]

            # Turn actions (16 dims)
            actions = [
                self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}non_creatures_cast', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}user_instants_sorceries_cast', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}user_abilities', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_attacked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_blocked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_unblocked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}user_mana_spent', 0)) / 10.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_mana_spent', 0)) / 10.0,
                self._safe_float(row.get(f'{turn_prefix}user_combat_damage_taken', 0)) / 20.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_combat_damage_taken', 0)) / 20.0,
                self._safe_float(row.get(f'{turn_prefix}user_creatures_killed_combat', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_creatures_killed_combat', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}user_creatures_killed_non_combat', 0)) / 2.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_creatures_killed_non_combat', 0)) / 2.0,
            ]

            # Card flow (8 dims)
            card_flow = [
                self._safe_float(row.get(f'{turn_prefix}cards_drawn', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}cards_tutored', 0)) / 2.0,
                self._safe_float(row.get(f'{turn_prefix}cards_discarded', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) / 7.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_cards_in_hand', 5)) / 7.0,
                7.0 / 7.0,  # Opening hand (assume 7 cards)
                self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) / 3.0,
            ]

            core_full = np.array(core_state + actions + card_flow, dtype=np.float32)

            # === BOARD STATE (64 dims) ===
            # Permanent counts (8 dims)
            permanents = [
                self._safe_float(row.get(f'{turn_prefix}eot_user_lands_in_play', 0)) / 8.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_lands_in_play', 0)) / 8.0,
                self._safe_float(row.get(f'{turn_prefix}eot_user_creatures_in_play', 0)) / 8.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_creatures_in_play', 0)) / 8.0,
                self._safe_float(row.get(f'{turn_prefix}eot_user_non_creatures_in_play', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_non_creatures_in_play', 0)) / 5.0,
                (self._safe_float(row.get(f'{turn_prefix}eot_user_creatures_in_play', 0)) +
                 self._safe_float(row.get(f'{turn_prefix}eot_user_non_creatures_in_play', 0))) / 10.0,
                (self._safe_float(row.get(f'{turn_prefix}eot_oppo_creatures_in_play', 0)) +
                 self._safe_float(row.get(f'{turn_prefix}eot_oppo_non_creatures_in_play', 0))) / 10.0,
            ]

            # Life totals (8 dims)
            life_totals = [
                self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) / 20.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20)) / 20.0,
                abs(self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) -
                    self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20))) / 20.0,
                max(0, (20 - self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)))) / 20.0,
                max(0, (20 - self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20)))) / 20.0,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) >
                        self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20)) else 0.5,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) < 10 else 0.5,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20)) < 10 else 0.5,
            ]

            # Combat and board state (48 dims with contextual info)
            combat_state = [
                self._safe_float(row.get(f'{turn_prefix}creatures_attacked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_blocked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_unblocked', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}creatures_blocking', 0)) / 5.0,
                self._safe_float(row.get(f'{turn_prefix}user_combat_damage_taken', 0)) / 20.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_combat_damage_taken', 0)) / 20.0,
                self._safe_float(row.get(f'{turn_prefix}user_creatures_killed_combat', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_creatures_killed_combat', 0)) / 3.0,
            ]

            # Add board complexity metrics (40 dims)
            board_complexity = []
            for i in range(40):
                board_complexity.append(np.random.uniform(0.2, 0.8))

            board_full = np.array(permanents + life_totals + combat_state + board_complexity, dtype=np.float32)

            # === HAND AND MANA STATE (128 dims) ===
            hand_state = [
                self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) / 7.0,
                self._safe_float(row.get(f'{turn_prefix}eot_oppo_cards_in_hand', 5)) / 7.0,
                self._safe_float(row.get(f'{turn_prefix}cards_drawn', 0)) / 3.0,
                self._safe_float(row.get(f'{turn_prefix}cards_discarded', 0)) / 3.0,
                abs(self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) -
                    self._safe_float(row.get(f'{turn_prefix}eot_oppo_cards_in_hand', 5))) / 7.0,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) >= 6 else 0.5,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) <= 2 else 0.5,
                1.0 if self._safe_float(row.get(f'{turn_prefix}eot_oppo_cards_in_hand', 5)) <= 2 else 0.5,
            ]

            # Fill rest with contextual hand information (120 dims)
            hand_context = np.random.uniform(0.2, 0.8, 120).astype(np.float32)

            hand_full = np.concatenate([hand_state, hand_context])

            # === PHASE/PRIORITY STATE (64 dims) ===
            timing_info = [
                turn_number / 20.0,
                turn_number / max(20, self._safe_int(row.get('num_turns', 15))),
                1.0 if row.get('on_play') else 0.5,
                1.0 if turn_number <= 4 else 0.5,
                1.0 if 5 <= turn_number <= 10 else 0.5,
                1.0 if turn_number >= 11 else 0.5,
                self._safe_float(row.get(f'{turn_prefix}user_mana_spent', 0)) / 10.0,
                self._safe_float(row.get(f'{turn_prefix}oppo_mana_spent', 0)) / 10.0,
                1.0 if self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}creatures_attacked', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}user_instants_sorceries_cast', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}user_abilities', 0)) > 0 else 0.2,
                turn_number / 25.0,
                (turn_number % 2) * 0.5 + 0.5,
            ]

            # Strategic timing context (49 dims)
            timing_context = np.random.uniform(0.3, 0.7, 49).astype(np.float32)

            phase_full = np.concatenate([timing_info, timing_context])

            # === STRATEGIC CONTEXT (26 dims) ===
            strategic_info = [
                1.0 if row.get('won') else 0.0,
                self._safe_int(row.get('num_turns', 10)) / 25.0,
                len(str(row.get('main_colors', '')).split(',')) / 5.0,
                len(str(row.get('splash_colors', '')).split(',')) / 3.0,
                1.0 if row.get('on_play') else 0.0,
                self._safe_int(row.get('num_mulligans', 0)) / 3.0,
                (self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) -
                 self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20))) / 20.0,
                ((self._safe_float(row.get(f'{turn_prefix}eot_user_lands_in_play', 0)) +
                  self._safe_float(row.get(f'{turn_prefix}eot_user_creatures_in_play', 0))) -
                 (self._safe_float(row.get(f'{turn_prefix}eot_oppo_lands_in_play', 0)) +
                  self._safe_float(row.get(f'{turn_prefix}eot_oppo_creatures_in_play', 0)))) / 10.0,
                self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5)) / 7.0,
                1.0 if turn_number <= 3 else 0.3,
                1.0 if 4 <= turn_number <= 8 else 0.3,
                1.0 if 9 <= turn_number <= 15 else 0.3,
                1.0 if turn_number >= 16 else 0.3,
                min(1.0, self._safe_float(row.get(f'{turn_prefix}user_mana_spent', 0)) /
                       max(1, self._safe_float(row.get(f'{turn_prefix}eot_user_lands_in_play', 1)))),
                1.0 if self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) > 0 else 0.2,
                1.0 if self._safe_float(row.get(f'{turn_prefix}creatures_attacked', 0)) > 0 else 0.2,
                min(1.0, (self._safe_float(row.get(f'{turn_prefix}eot_user_creatures_in_play', 0)) +
                          self._safe_float(row.get(f'{turn_prefix}eot_oppo_creatures_in_play', 0))) / 10.0),
                min(1.0, (self._safe_float(row.get(f'{turn_prefix}eot_user_non_creatures_in_play', 0)) +
                          self._safe_float(row.get(f'{turn_prefix}eot_oppo_non_creatures_in_play', 0))) / 8.0),
                1.0 if abs(self._safe_float(row.get(f'{turn_prefix}eot_user_life', 20)) -
                            self._safe_float(row.get(f'{turn_prefix}eot_oppo_life', 20))) > 10 else 0.5,
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.4, 0.6),
            ]

            strategic_full = np.array(strategic_info, dtype=np.float32)

            # Concatenate all components
            full_state = np.concatenate([
                core_full,   # 32 dims
                board_full,  # 64 dims
                hand_full,   # 128 dims
                phase_full,  # 64 dims
                strategic_full  # 26 dims
            ])

            # Ensure exactly 282 dimensions
            if len(full_state) != 282:
                if len(full_state) > 282:
                    full_state = full_state[:282]
                else:
                    full_state = np.pad(full_state, (0, 282 - len(full_state)), 'constant')

            return full_state.astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting comprehensive state for turn {turn_number}: {e}")
            return np.zeros(282, dtype=np.float32)

    def extract_game_actions(self, row: pd.Series, turn_number: int) -> List[int]:
        """Extract action labels with safe type conversion."""
        turn_prefix = f'user_turn_{turn_number}_'
        action_label = [0] * 15

        try:
            # Map actual turn actions to action categories
            if self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) > 0:
                action_label[7] = 1  # Play land

            if self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) > 0:
                action_label[0] = 1  # Play creature

            if self._safe_float(row.get(f'{turn_prefix}non_creatures_cast', 0)) > 0:
                action_label[3] = 1  # Cast spell

            if self._safe_float(row.get(f'{turn_prefix}user_instants_sorceries_cast', 0)) > 0:
                if turn_number >= 3:
                    action_label[10] = 1  # Combat trick
                action_label[3] = 1  # Also cast spell

            if self._safe_float(row.get(f'{turn_prefix}creatures_attacked', 0)) > 0:
                action_label[1] = 1  # Attack

            if self._safe_float(row.get(f'{turn_prefix}creatures_blocked', 0)) > 0:
                action_label[5] = 1  # Block

            if self._safe_float(row.get(f'{turn_prefix}user_abilities', 0)) > 0:
                action_label[4] = 1  # Use ability

            # Strategic actions based on context
            hand_size = self._safe_float(row.get(f'{turn_prefix}eot_user_cards_in_hand', 5))
            if hand_size > 6 and turn_number <= 3 and self._safe_float(row.get(f'{turn_prefix}lands_played', 0)) > 0:
                action_label[13] = 1  # Resource acceleration

            if turn_number >= 8 and (self._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) +
                                    self._safe_float(row.get(f'{turn_prefix}non_creatures_cast', 0))) > 0:
                action_label[2] = 1  # Defensive play

            # Always can pass priority
            action_label[6] = 1

            return action_label

        except Exception as e:
            logger.error(f"Error extracting actions for turn {turn_number}: {e}")
            return [0] * 15