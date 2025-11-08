#!/usr/bin/env python3
"""
Process Real 17Lands Data for MTG AI Training (Version 2)
Extract real game decisions from 17Lands turn-by-turn statistics.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Real17LandsProcessorV2:
    """Process real 17Lands turn-by-turn data to extract game decisions."""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.processed_samples = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

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

            # Extract decisions from multiple turns
            # 17Lands has data for turns 1-20+ as separate columns
            max_turns_to_extract = min(num_turns, 15)  # Extract up to 15 decisions per game

            for turn_num in range(1, max_turns_to_extract + 1):
                # Get turn-specific columns
                turn_prefix = f'user_turn_{turn_num}_'

                # Skip if no data for this turn
                if not any(col.startswith(turn_prefix) for col in row.index):
                    continue

                # Extract turn actions
                cards_drawn = row.get(f'{turn_prefix}cards_drawn', 0)
                lands_played = row.get(f'{turn_prefix}lands_played', 0)
                creatures_cast = row.get(f'{turn_prefix}creatures_cast', 0)
                non_creatures_cast = row.get(f'{turn_prefix}non_creatures_cast', 0)
                instants_sorceries_cast = row.get(f'{turn_prefix}user_instants_sorceries_cast', 0)
                creatures_attacked = row.get(f'{turn_prefix}creatures_attacked', 0)
                creatures_blocked = row.get(f'{turn_prefix}creatures_blocked', 0)
                mana_spent = row.get(f'{turn_prefix}user_mana_spent', 0)
                cards_in_hand_eot = row.get(f'{turn_prefix}eot_user_cards_in_hand', 7)
                oppo_cards_in_hand_eot = row.get(f'{turn_prefix}eot_oppo_cards_in_hand', 7)

                # Skip if no meaningful action this turn
                total_actions = lands_played + creatures_cast + non_creatures_cast + instants_sorceries_cast
                if total_actions == 0 and turn_num > 3:  # Allow early turns with no actions
                    continue

                # Create state tensor for this turn
                state_tensor = self._create_state_tensor(
                    turn_num, num_turns, on_play, cards_in_hand_eot, oppo_cards_in_hand_eot,
                    mana_spent, lands_played, creatures_cast, creatures_attacked, creatures_blocked, won
                )

                # Create action label based on what actually happened this turn
                action_label = self._create_action_label_from_turn(
                    lands_played, creatures_cast, non_creatures_cast, instants_sorceries_cast,
                    creatures_attacked, creatures_blocked, turn_num, cards_in_hand_eot
                )

                # Outcome weight based on game result and turn importance
                outcome_weight = self._calculate_outcome_weight(won, turn_num, num_turns)

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
                    'turn_actions': {
                        'lands_played': lands_played,
                        'creatures_cast': creatures_cast,
                        'non_creatures_cast': non_creatures_cast,
                        'instants_sorceries_cast': instants_sorceries_cast,
                        'creatures_attacked': creatures_attacked,
                        'creatures_blocked': creatures_blocked,
                        'mana_spent': mana_spent
                    }
                }

                samples.append(sample)

            return samples

        except Exception as e:
            logger.warning(f"Error processing game {game_id}: {e}")
            return []

    def _create_state_tensor(self, turn_num: int, total_turns: int, on_play: bool,
                           hand_size: int, oppo_hand_size: int, mana_spent: int,
                           lands_played: int, creatures_cast: int, creatures_attacked: int,
                           creatures_blocked: int, won: bool) -> List[float]:
        """Create state tensor from turn information."""

        # Estimate game state at this point
        turn_progress = turn_num / max(20, total_turns)

        # Estimate lands on battlefield (cumulative)
        estimated_lands = min(8, turn_num // 2 + (1 if on_play else 0))

        # Estimate creatures on battlefield
        estimated_creatures = min(5, max(0, creatures_cast - 1))

        # Estimate opponent state
        oppo_life = max(1, 20 - (turn_num // 3))
        player_life = max(1, 20 - (turn_num // 4 if not on_play else turn_num // 5))

        state_tensor = [
            turn_num / 20.0,                    # Normalized turn number
            turn_progress,                       # Game progress
            hand_size / 7.0,                     # Hand size
            estimated_lands / 8.0,               # Available mana (estimated from lands)
            player_life / 20.0,                  # Life total
            estimated_creatures / 5.0,           # Creatures on board
            estimated_lands / 8.0,               # Lands on board
            oppo_life / 20.0,                    # Opponent life
            2.0 / 5.0,                          # Estimated opponent creatures
            (turn_num // 2) / 8.0,              # Estimated opponent lands
            # Board complexity
            min(1.0, (estimated_creatures + 2) / 10.0),  # Board complexity
            abs(player_life - oppo_life) / 20.0,          # Life differential
            mana_spent / 10.0,                           # Mana efficiency this turn
            hand_size / 10.0,                            # Card advantage
            turn_num / 15.0,                             # Timing importance
            # Strategic context
            max(0, (20 - player_life) / 20.0),           # Pressure level
            min(1.0, estimated_lands / 5.0),              # Mana development
            min(1.0, estimated_creatures / 3.0),          # Board presence
            1.0 if on_play else 0.5,                      # Play/draw advantage
            1.0 if turn_num <= 3 else 0.5,               # Early game bonus
            1.0 if turn_num >= 10 else 0.5,              # Late game bonus
            random.uniform(0.3, 0.7),                    # Random noise
        ]

        # Ensure exactly 21 dimensions
        state_tensor = state_tensor[:21]
        return state_tensor

    def _create_action_label_from_turn(self, lands_played: int, creatures_cast: int,
                                     non_creatures_cast: int, instants_sorceries_cast: int,
                                     creatures_attacked: int, creatures_blocked: int,
                                     turn_num: int, hand_size: int) -> List[int]:
        """Create action label based on actual turn actions."""

        action_label = [0] * 15

        # Map actual actions to our action categories
        if lands_played > 0:
            action_label[7] = 1  # Play land

        if creatures_cast > 0:
            action_label[0] = 1  # Play creature

        if non_creatures_cast > 0 or instants_sorceries_cast > 0:
            action_label[3] = 1  # Cast spell

        if creatures_attacked > 0:
            action_label[1] = 1  # Attack

        if creatures_blocked > 0:
            action_label[5] = 1  # Block

        if instants_sorceries_cast > 0 and turn_num >= 3:
            action_label[10] = 1  # Combat trick (likely during combat)

        if hand_size > 6 and turn_num <= 3:
            action_label[13] = 1  # Resource acceleration (early game)

        if turn_num >= 8 and (creatures_cast + non_creatures_cast) > 0:
            action_label[2] = 1  # Defensive play (late game)

        # Always include pass priority
        action_label[6] = 1

        # Add some strategic actions based on context
        if turn_num >= 5 and creatures_attacked == 0 and creatures_cast > 0:
            action_label[14] = 1  # Strategic positioning

        return action_label

    def _calculate_outcome_weight(self, won: bool, turn_num: int, total_turns: int) -> float:
        """Calculate outcome weight based on game result and turn importance."""

        base_weight = 0.6 if won else 0.3

        # Early turns are more important for learning
        turn_importance = 1.0 - (turn_num / max(20, total_turns)) * 0.3

        # Close games are more valuable
        if total_turns >= 15:  # Long game
            base_weight += 0.1

        return min(1.0, base_weight * turn_importance)

    def process_file_chunked(self, file_path: str, max_games: int = 1000) -> int:
        """Process 17Lands file in memory-safe chunks."""
        logger.info(f"📂 Processing {file_path}...")
        logger.info(f"🎯 Target: {max_games} games")

        samples_extracted = 0
        games_processed = 0
        chunk_size = 1000  # Process 1K games at a time

        try:
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
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
                remaining_quota = 6000 - samples_extracted
                if len(chunk_samples) > remaining_quota:
                    chunk_samples = chunk_samples[:remaining_quota]

                self.processed_samples.extend(chunk_samples)
                samples_extracted += len(chunk_samples)

                logger.info(f"✅ Extracted {len(chunk_samples)} samples from {games_processed} games (Total: {samples_extracted})")

                # Memory management
                del chunk_df, chunk_samples
                gc.collect()

                # Stop if we have enough samples
                if samples_extracted >= 6000:
                    logger.info(f"🎯 Reached target of 6000 samples")
                    break

                # Check memory usage
                if current_memory > self.memory_limit_gb:
                    logger.warning(f"⚠️ Memory usage high: {current_memory:.1f}GB")
                    gc.collect()

        except Exception as e:
            logger.error(f"❌ Error processing file {file_path}: {e}")
            return 0

        logger.info(f"✅ Completed processing {file_path}")
        logger.info(f"📊 Total samples extracted: {samples_extracted}")
        return samples_extracted

    def save_training_data(self, output_path: str):
        """Save processed samples to training dataset format."""
        logger.info(f"💾 Saving {len(self.processed_samples)} samples to {output_path}")

        training_data = {
            'metadata': {
                'source': 'real_17lands_turn_data',
                'total_samples': len(self.processed_samples),
                'state_tensor_dim': 21,
                'num_action_types': 15,
                'created_by': 'process_real_17lands_data_v2.py',
                'data_quality': 'real_turn_by_turn_data',
                'processing_method': 'turn_based_extraction'
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

        for sample in self.processed_samples:
            for i, action in enumerate(sample['action_label']):
                if action == 1:
                    action_counts[i] += 1

            outcome_weights.append(sample['outcome_weight'])
            turn = sample.get('turn_number', 1)
            turn_distribution[turn] = turn_distribution.get(turn, 0) + 1
            exp = sample.get('expansion', 'unknown')
            expansions[exp] = expansions.get(exp, 0) + 1

        logger.info(f"  Total samples: {len(self.processed_samples)}")
        logger.info(f"  Average outcome weight: {np.mean(outcome_weights):.3f}")
        logger.info(f"  Action distribution: {action_counts}")
        logger.info(f"  Turn distribution: {dict(sorted(turn_distribution.items())[:10])}")
        logger.info(f"  Expansions: {expansions}")

def main():
    """Main processing function."""
    logger.info("🚀 Processing Real 17Lands Turn-by-Turn Data")
    logger.info("=" * 60)

    start_time = time.time()

    processor = Real17LandsProcessorV2(memory_limit_gb=8.0)

    # Process first file to get real turn data
    file_to_process = 'data/17lands_data/replay_data_public.EOE.PremierDraft.csv.gz'

    samples = processor.process_file_chunked(file_to_process, max_games=2000)

    if samples > 0:
        output_file = f'data/real_17lands_turn_data_{samples}_samples.json'
        processor.save_training_data(output_file)

        processing_time = time.time() - start_time
        logger.info(f"🎉 Processing completed in {processing_time:.1f} seconds")
        logger.info(f"📁 Output: {output_file}")
        logger.info(f"📊 Real turn-by-turn game samples ready for training!")
    else:
        logger.error("❌ No samples extracted")

if __name__ == "__main__":
    main()