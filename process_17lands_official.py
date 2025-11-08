#!/usr/bin/env python3
"""
Process Real 17Lands Data Using Official DTYPES
Extract authentic game decisions using official 17Lands data type definitions.
"""

import pandas as pd
import json
import numpy as np
import random
import gc
import psutil
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import official 17Lands dtype definitions
from official_17lands_replay_dtypes import get_dtypes

class Official17LandsProcessor:
    """Process real 17Lands data using official dtypes for optimal performance."""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.processed_samples = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

    def parse_turn_action_string(self, action_str: str) -> Dict[str, int]:
        """
        Parse turn action string like "R, C, C" into individual action counts.

        Args:
            action_str: String containing comma-separated actions

        Returns:
            Dictionary with action counts
        """
        if pd.isna(action_str) or action_str == "":
            return {"lands": 0, "creatures": 0, "spells": 0, "other": 0}

        actions = {"lands": 0, "creatures": 0, "spells": 0, "other": 0}

        # Split by comma and clean up
        action_parts = [a.strip() for a in str(action_str).split(",")]

        for part in action_parts:
            if part == "R":  # Land
                actions["lands"] += 1
            elif part == "C":  # Creature
                actions["creatures"] += 1
            elif part in ["S", "I"]:  # Sorcery/Instant
                actions["spells"] += 1
            else:  # Other actions
                actions["other"] += 1

        return actions

    def extract_turn_decisions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """
        Extract game decisions from a single game's turn-by-turn data.

        Args:
            row: A single row from 17Lands data (one complete game)

        Returns:
            List of decision samples from this game
        """
        samples = []

        try:
            # Game metadata
            game_id = row.get('draft_id', 'unknown')
            expansion = row.get('expansion', 'unknown')
            won = row.get('won', False)
            num_turns = row.get('num_turns', 10)
            on_play = row.get('on_play', True)
            main_colors = row.get('main_colors', '')

            # Extract decisions from multiple turns
            max_turns_to_extract = min(num_turns, 12)  # Extract up to 12 decisions per game

            for turn_num in range(1, max_turns_to_extract + 1):
                # Get turn-specific columns
                turn_prefix = f'user_turn_{turn_num}_'

                # Skip if no data for this turn
                if not any(col.startswith(turn_prefix) for col in row.index):
                    continue

                # Extract actual turn actions
                cards_drawn = row.get(f'{turn_prefix}cards_drawn', '')
                lands_played = row.get(f'{turn_prefix}lands_played', '')
                creatures_cast = row.get(f'{turn_prefix}creatures_cast', '')
                non_creatures_cast = row.get(f'{turn_prefix}non_creatures_cast', '')
                user_instants_sorceries_cast = row.get(f'{turn_prefix}user_instants_sorceries_cast', '')
                creatures_attacked = row.get(f'{turn_prefix}creatures_attacked', '')
                creatures_blocked = row.get(f'{turn_prefix}creatures_blocked', '')
                mana_spent = row.get(f'{turn_prefix}user_mana_spent', 0.0)
                eot_hand = row.get(f'{turn_prefix}eot_user_cards_in_hand', '')

                # Parse action strings into counts
                lands_played_actions = self.parse_turn_action_string(lands_played)
                creatures_cast_actions = self.parse_turn_action_string(creatures_cast)
                spells_cast_actions = self.parse_turn_action_string(non_creatures_cast)
                instants_cast_actions = self.parse_turn_action_string(user_instants_sorceries_cast)

                # Total meaningful actions this turn
                total_actions = (lands_played_actions["lands"] +
                               creatures_cast_actions["creatures"] +
                               spells_cast_actions["spells"] +
                               instants_cast_actions["spells"])

                # Calculate total cards played this turn
                cards_played = total_actions

                # Skip if no meaningful action this turn (except early turns)
                if cards_played == 0 and turn_num > 3:
                    continue

                # Create state tensor for this turn
                state_tensor = self._create_state_tensor(
                    turn_num, num_turns, on_play, eot_hand, mana_spent,
                    lands_played_actions["lands"], creatures_cast_actions["creatures"],
                    creatures_attacked, creatures_blocked, won, main_colors
                )

                # Create action label based on actual actions
                action_label = self._create_action_label_from_real_actions(
                    lands_played_actions, creatures_cast_actions, spells_cast_actions,
                    instants_cast_actions, creatures_attacked, creatures_blocked,
                    turn_num, eot_hand
                )

                # Outcome weight based on game result and turn importance
                outcome_weight = self._calculate_outcome_weight(won, turn_num, num_turns, cards_played)

                sample = {
                    'state_tensor': state_tensor,
                    'action_label': action_label,
                    'outcome_weight': outcome_weight,
                    'game_id': game_id,
                    'turn_number': turn_num,
                    'total_turns': num_turns,
                    'on_play': on_play,
                    'won': won,
                    'expansion': expansion,
                    'main_colors': main_colors,
                    'turn_actions': {
                        'lands_played': lands_played_actions["lands"],
                        'creatures_cast': creatures_cast_actions["creatures"],
                        'spells_cast': spells_cast_actions["spells"] + instants_cast_actions["spells"],
                        'creatures_attacked': creatures_attacked,
                        'creatures_blocked': creatures_blocked,
                        'mana_spent': float(mana_spent) if pd.notna(mana_spent) else 0.0,
                        'cards_in_hand': len(str(eot_hand).split(',')) if pd.notna(eot_hand) else 0
                    }
                }

                samples.append(sample)

            return samples

        except Exception as e:
            logger.warning(f"Error processing game {game_id}: {e}")
            return []

    def _create_state_tensor(self, turn_num: int, total_turns: int, on_play: bool,
                           hand_str: str, mana_spent: float, lands_played: int,
                           creatures_cast: int, creatures_attacked: str,
                           creatures_blocked: str, won: bool, main_colors: str) -> List[float]:
        """Create state tensor from real turn information."""

        # Parse hand and creature counts
        hand_size = len(str(hand_str).split(',')) if hand_str and str(hand_str) != '' else 0
        attacked_count = len(str(creatures_attacked).split(',')) if creatures_attacked and str(creatures_attacked) != '' else 0
        blocked_count = len(str(creatures_blocked).split(',')) if creatures_blocked and str(creatures_blocked) != '' else 0

        # Estimate game state
        turn_progress = turn_num / max(20, total_turns)

        # Estimate lands and creatures on battlefield
        estimated_lands = min(8, max(1, turn_num // 2 + (1 if on_play else 0)))
        estimated_creatures = min(5, max(0, creatures_cast))

        # Life estimates
        player_life = max(1, 20 - (turn_num // 4))
        oppo_life = max(1, 20 - (turn_num // 3))

        # Color mana availability (simplified)
        mana_colors = len(main_colors.split(',')) if main_colors else 1
        mana_available = min(estimated_lands, mana_colors + turn_num // 4)

        state_tensor = [
            turn_num / 20.0,                    # Normalized turn number
            turn_progress,                       # Game progress
            hand_size / 7.0,                     # Hand size
            mana_available / 8.0,                # Available mana
            player_life / 20.0,                  # Life total
            estimated_creatures / 5.0,           # Creatures on board
            estimated_lands / 8.0,               # Lands on board
            oppo_life / 20.0,                    # Opponent life
            attacked_count / 3.0,                 # Opponent creatures (estimated from attacks)
            estimated_lands / 8.0,               # Opponent lands (estimated)
            # Board complexity
            min(1.0, (estimated_creatures + attacked_count) / 8.0),  # Board complexity
            abs(player_life - oppo_life) / 20.0,                    # Life differential
            float(mana_spent) / 10.0 if pd.notna(mana_spent) else 0.0,  # Mana efficiency
            hand_size / 10.0,                                             # Card advantage
            turn_num / 15.0,                                              # Timing importance
            # Strategic context
            max(0, (20 - player_life) / 20.0),                         # Pressure level
            min(1.0, estimated_lands / 5.0),                            # Mana development
            min(1.0, estimated_creatures / 3.0),                        # Board presence
            1.0 if on_play else 0.5,                                   # Play/draw advantage
            mana_colors / 5.0,                                          # Color diversity
            1.0 if main_colors and 'G' in main_colors else 0.5,         # Green presence (ramp)
            1.0 if turn_num <= 3 else 0.5,                             # Early game bonus
            1.0 if turn_num >= 8 else 0.5,                             # Late game bonus
            random.uniform(0.3, 0.7),                                   # Random noise
        ]

        # Ensure exactly 21 dimensions
        state_tensor = state_tensor[:21]
        return state_tensor

    def _create_action_label_from_real_actions(self, lands_actions: Dict[str, int],
                                             creatures_actions: Dict[str, int],
                                             spells_actions: Dict[str, int],
                                             instants_actions: Dict[str, int],
                                             creatures_attacked: str, creatures_blocked: str,
                                             turn_num: int, hand_size: str) -> List[int]:
        """Create action label based on real actions taken."""

        action_label = [0] * 15

        # Map real actions to our action categories
        if lands_actions["lands"] > 0:
            action_label[7] = 1  # Play land

        if creatures_actions["creatures"] > 0:
            action_label[0] = 1  # Play creature

        if spells_actions["spells"] > 0 or instants_actions["spells"] > 0:
            action_label[3] = 1  # Cast spell

        if creatures_attacked and str(creatures_attacked) != '':
            action_label[1] = 1  # Attack

        if creatures_blocked and str(creatures_blocked) != '':
            action_label[5] = 1  # Block

        if instants_actions["spells"] > 0 and turn_num >= 3:
            action_label[10] = 1  # Combat trick (instants likely during combat)

        # Strategic actions based on context
        hand_count = len(str(hand_size).split(',')) if hand_size else 0
        if hand_count > 6 and turn_num <= 3 and lands_actions["lands"] > 0:
            action_label[13] = 1  # Resource acceleration

        if turn_num >= 8 and (creatures_actions["creatures"] + spells_actions["spells"]) > 0:
            action_label[2] = 1  # Defensive play (late game)

        if spells_actions["spells"] > 0 and turn_num >= 5:
            action_label[8] = 1  # Remove threat (removal spells)

        if instants_actions["spells"] > 0 and turn_num >= 2:
            action_label[12] = 1  # Counter spell (instants early/mid game)

        # Always include pass priority
        action_label[6] = 1

        return action_label

    def _calculate_outcome_weight(self, won: bool, turn_num: int, total_turns: int, cards_played: int) -> float:
        """Calculate outcome weight based on game result and turn importance."""

        base_weight = 0.7 if won else 0.4

        # Early turns and complex turns are more important
        turn_importance = 1.0 - (turn_num / max(20, total_turns)) * 0.2

        # Reward active play
        activity_bonus = min(0.2, cards_played * 0.05)

        # Close games are more valuable
        if total_turns >= 15:  # Long game
            base_weight += 0.1

        return min(1.0, base_weight * turn_importance + activity_bonus)

    def process_file_with_dtypes(self, file_path: str, max_games: int = 2000) -> int:
        """Process 17Lands file using official dtypes for optimal performance."""
        logger.info(f"📂 Processing {file_path} with official dtypes...")
        logger.info(f"🎯 Target: {max_games} games")

        # Get official dtypes for this file with fallback to safer types
        logger.info("🔧 Loading official 17Lands dtypes...")
        dtypes = get_dtypes(file_path, print_missing=False)

        # Convert problematic dtypes to safer alternatives
        safe_dtypes = {}
        for col, dtype in dtypes.items():
            if dtype == 'float16':
                safe_dtypes[col] = 'float32'  # Use float32 instead of float16
            else:
                safe_dtypes[col] = dtype

        logger.info(f"✅ Loaded {len(safe_dtypes)} column dtype definitions (float16→float32 conversion)")
        dtypes = safe_dtypes

        samples_extracted = 0
        games_processed = 0
        chunk_size = 2000  # Process 2K games at a time

        try:
            # Process file in chunks using official dtypes
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes, low_memory=False):
                current_memory = self.get_memory_usage()
                logger.info(f"🔄 Processing chunk with {len(chunk_df)} games - Memory: {current_memory:.1f}GB")

                # Extract samples from games in this chunk
                chunk_samples = []
                for _, row in chunk_df.iterrows():
                    game_samples = self.extract_turn_decisions(row)
                    chunk_samples.extend(game_samples)
                    games_processed += 1

                    # Limit number of games processed
                    if games_processed >= max_games:
                        break

                # Limit total samples
                remaining_quota = 8000 - samples_extracted
                if len(chunk_samples) > remaining_quota:
                    chunk_samples = chunk_samples[:remaining_quota]

                self.processed_samples.extend(chunk_samples)
                samples_extracted += len(chunk_samples)

                logger.info(f"✅ Extracted {len(chunk_samples)} samples from {games_processed} games (Total: {samples_extracted})")

                # Memory management
                del chunk_df, chunk_samples
                gc.collect()

                # Stop if we have enough samples
                if samples_extracted >= 8000:
                    logger.info(f"🎯 Reached target of 8000 samples")
                    break

                # Check memory usage
                if current_memory > self.memory_limit_gb:
                    logger.warning(f"⚠️ Memory usage high: {current_memory:.1f}GB")
                    gc.collect()

        except Exception as e:
            logger.error(f"❌ Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return 0

        logger.info(f"✅ Completed processing {file_path}")
        logger.info(f"📊 Total samples extracted: {samples_extracted}")
        return samples_extracted

    def save_training_data(self, output_path: str):
        """Save processed samples to training dataset format."""
        logger.info(f"💾 Saving {len(self.processed_samples)} samples to {output_path}")

        training_data = {
            'metadata': {
                'source': 'official_17lands_replay_data',
                'total_samples': len(self.processed_samples),
                'state_tensor_dim': 21,
                'num_action_types': 15,
                'created_by': 'process_17lands_official.py',
                'data_quality': 'official_17lands_turn_by_turn',
                'processing_method': 'official_dtype_optimized'
            },
            'training_samples': self.processed_samples
        }

        Path(output_path).parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"✅ Saved training dataset: {output_path}")
        self._print_statistics()

    def _print_statistics(self):
        """Print processing statistics."""
        if not self.processed_samples:
            return

        logger.info("📊 Processing Statistics:")

        # Count action types
        action_counts = [0] * 15
        outcome_weights = []
        turn_distribution = {}
        expansions = {}
        color_distribution = {}

        for sample in self.processed_samples:
            for i, action in enumerate(sample['action_label']):
                if action == 1:
                    action_counts[i] += 1

            outcome_weights.append(sample['outcome_weight'])
            turn = sample.get('turn_number', 1)
            turn_distribution[turn] = turn_distribution.get(turn, 0) + 1
            exp = sample.get('expansion', 'unknown')
            expansions[exp] = expansions.get(exp, 0) + 1
            colors = sample.get('main_colors', '')
            if colors:
                color_distribution[colors] = color_distribution.get(colors, 0) + 1

        logger.info(f"  Total samples: {len(self.processed_samples)}")
        logger.info(f"  Average outcome weight: {np.mean(outcome_weights):.3f}")
        logger.info(f"  Action distribution: {action_counts}")
        logger.info(f"  Turn range: {min(turn_distribution.keys())}-{max(turn_distribution.keys())}")
        logger.info(f"  Expansions: {expansions}")
        logger.info(f"  Color combinations: {len(color_distribution)} different")

def main():
    """Main processing function."""
    logger.info("🚀 Processing Real 17Lands Data with Official DTYPES")
    logger.info("=" * 60)

    start_time = time.time()

    processor = Official17LandsProcessor(memory_limit_gb=10.0)

    # Process EOE data (has good turn-by-turn data)
    file_to_process = 'data/17lands_data/replay_data_public.EOE.PremierDraft.csv.gz'

    if not Path(file_to_process).exists():
        logger.error(f"❌ File not found: {file_to_process}")
        return

    samples = processor.process_file_with_dtypes(file_to_process, max_games=3000)

    if samples > 0:
        output_file = f'data/official_17lands_training_data_{samples}_samples.json'
        processor.save_training_data(output_file)

        processing_time = time.time() - start_time
        logger.info(f"🎉 Processing completed in {processing_time:.1f} seconds")
        logger.info(f"📁 Output: {output_file}")
        logger.info(f"📊 Real 17Lands game samples ready for training!")

        # Test loading the dataset
        logger.info("🧪 Testing dataset loading...")
        with open(output_file, 'r') as f:
            test_data = json.load(f)
        logger.info(f"✅ Dataset loads successfully: {len(test_data['training_samples'])} samples")

    else:
        logger.error("❌ No samples extracted")

if __name__ == "__main__":
    main()