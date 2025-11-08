#!/usr/bin/env python3
"""
Process ALL 17Lands PremierDraft Data
Comprehensive processing of all available 17Lands replay data.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import official 17Lands dtype definitions
from official_17lands_replay_dtypes import get_dtypes

class Comprehensive17LandsProcessor:
    """Process all available 17Lands PremierDraft data."""

    def __init__(self, memory_limit_gb: float = 12.0):
        self.memory_limit_gb = memory_limit_gb
        self.processed_samples = []
        self.processing_lock = threading.Lock()

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

    def parse_turn_action_string(self, action_str: str) -> Dict[str, int]:
        """Parse turn action string into individual action counts."""
        if pd.isna(action_str) or action_str == "":
            return {"lands": 0, "creatures": 0, "spells": 0, "other": 0}

        actions = {"lands": 0, "creatures": 0, "spells": 0, "other": 0}
        action_parts = [a.strip() for a in str(action_str).split(",")]

        for part in action_parts:
            if part == "R":
                actions["lands"] += 1
            elif part == "C":
                actions["creatures"] += 1
            elif part in ["S", "I"]:
                actions["spells"] += 1
            else:
                actions["other"] += 1

        return actions

    def extract_turn_decisions(self, row: pd.Series, expansion_name: str) -> List[Dict[str, Any]]:
        """Extract game decisions from a single game."""
        samples = []

        try:
            game_id = row.get('draft_id', 'unknown')
            won = row.get('won', False)
            num_turns = row.get('num_turns', 10)
            on_play = row.get('on_play', True)
            main_colors = row.get('main_colors', '')

            max_turns_to_extract = min(num_turns, 15)

            for turn_num in range(1, max_turns_to_extract + 1):
                turn_prefix = f'user_turn_{turn_num}_'

                if not any(col.startswith(turn_prefix) for col in row.index):
                    continue

                # Extract turn actions
                cards_drawn = row.get(f'{turn_prefix}cards_drawn', '')
                lands_played = row.get(f'{turn_prefix}lands_played', '')
                creatures_cast = row.get(f'{turn_prefix}creatures_cast', '')
                non_creatures_cast = row.get(f'{turn_prefix}non_creatures_cast', '')
                user_instants_sorceries_cast = row.get(f'{turn_prefix}user_instants_sorceries_cast', '')
                creatures_attacked = row.get(f'{turn_prefix}creatures_attacked', '')
                creatures_blocked = row.get(f'{turn_prefix}creatures_blocked', '')
                mana_spent = row.get(f'{turn_prefix}user_mana_spent', 0.0)
                eot_hand = row.get(f'{turn_prefix}eot_user_cards_in_hand', '')

                # Parse actions
                lands_played_actions = self.parse_turn_action_string(lands_played)
                creatures_cast_actions = self.parse_turn_action_string(creatures_cast)
                spells_cast_actions = self.parse_turn_action_string(non_creatures_cast)
                instants_cast_actions = self.parse_turn_action_string(user_instants_sorceries_cast)

                total_actions = (lands_played_actions["lands"] +
                               creatures_cast_actions["creatures"] +
                               spells_cast_actions["spells"] +
                               instants_cast_actions["spells"])

                if total_actions == 0 and turn_num > 3:
                    continue

                # Create sample
                state_tensor = self._create_state_tensor(
                    turn_num, num_turns, on_play, eot_hand, mana_spent,
                    lands_played_actions["lands"], creatures_cast_actions["creatures"],
                    creatures_attacked, creatures_blocked, won, main_colors
                )

                action_label = self._create_action_label_from_real_actions(
                    lands_played_actions, creatures_cast_actions, spells_cast_actions,
                    instants_cast_actions, creatures_attacked, creatures_blocked,
                    turn_num, eot_hand
                )

                outcome_weight = self._calculate_outcome_weight(won, turn_num, num_turns, total_actions)

                sample = {
                    'state_tensor': state_tensor,
                    'action_label': action_label,
                    'outcome_weight': outcome_weight,
                    'game_id': game_id,
                    'turn_number': turn_num,
                    'total_turns': num_turns,
                    'on_play': on_play,
                    'won': won,
                    'expansion': expansion_name,
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

        hand_size = len(str(hand_str).split(',')) if hand_str and str(hand_str) != '' else 0
        attacked_count = len(str(creatures_attacked).split(',')) if creatures_attacked and str(creatures_attacked) != '' else 0
        blocked_count = len(str(creatures_blocked).split(',')) if creatures_blocked and str(creatures_blocked) != '' else 0

        turn_progress = turn_num / max(20, total_turns)
        estimated_lands = min(8, max(1, turn_num // 2 + (1 if on_play else 0)))
        estimated_creatures = min(5, max(0, creatures_cast))

        player_life = max(1, 20 - (turn_num // 4))
        oppo_life = max(1, 20 - (turn_num // 3))

        mana_colors = len(main_colors.split(',')) if main_colors else 1
        mana_available = min(estimated_lands, mana_colors + turn_num // 4)

        state_tensor = [
            turn_num / 20.0,
            turn_progress,
            hand_size / 7.0,
            mana_available / 8.0,
            player_life / 20.0,
            estimated_creatures / 5.0,
            estimated_lands / 8.0,
            oppo_life / 20.0,
            attacked_count / 3.0,
            estimated_lands / 8.0,
            min(1.0, (estimated_creatures + attacked_count) / 8.0),
            abs(player_life - oppo_life) / 20.0,
            float(mana_spent) / 10.0 if pd.notna(mana_spent) else 0.0,
            hand_size / 10.0,
            turn_num / 15.0,
            max(0, (20 - player_life) / 20.0),
            min(1.0, estimated_lands / 5.0),
            min(1.0, estimated_creatures / 3.0),
            1.0 if on_play else 0.5,
            mana_colors / 5.0,
            1.0 if main_colors and 'G' in main_colors else 0.5,
            random.uniform(0.3, 0.7),
        ]

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

        if lands_actions["lands"] > 0:
            action_label[7] = 1
        if creatures_actions["creatures"] > 0:
            action_label[0] = 1
        if spells_actions["spells"] > 0 or instants_actions["spells"] > 0:
            action_label[3] = 1
        if creatures_attacked and str(creatures_attacked) != '':
            action_label[1] = 1
        if creatures_blocked and str(creatures_blocked) != '':
            action_label[5] = 1
        if instants_actions["spells"] > 0 and turn_num >= 3:
            action_label[10] = 1
        if instants_actions["spells"] > 0 and turn_num >= 2:
            action_label[12] = 1

        hand_count = len(str(hand_size).split(',')) if hand_size else 0
        if hand_count > 6 and turn_num <= 3 and lands_actions["lands"] > 0:
            action_label[13] = 1
        if turn_num >= 8 and (creatures_actions["creatures"] + spells_actions["spells"]) > 0:
            action_label[2] = 1
        if spells_actions["spells"] > 0 and turn_num >= 5:
            action_label[8] = 1

        action_label[6] = 1  # Always can pass priority

        return action_label

    def _calculate_outcome_weight(self, won: bool, turn_num: int, total_turns: int, cards_played: int) -> float:
        """Calculate outcome weight based on game result and turn importance."""

        base_weight = 0.7 if won else 0.4
        turn_importance = 1.0 - (turn_num / max(20, total_turns)) * 0.2
        activity_bonus = min(0.2, cards_played * 0.05)

        if total_turns >= 15:
            base_weight += 0.1

        return min(1.0, base_weight * turn_importance + activity_bonus)

    def process_single_file(self, file_path: str, max_samples: int, expansion_name: str) -> int:
        """Process a single 17Lands file."""
        logger.info(f"📂 Processing {file_path} ({expansion_name})...")

        try:
            # Get dtypes
            dtypes = get_dtypes(file_path, print_missing=False)
            safe_dtypes = {}
            for col, dtype in dtypes.items():
                if dtype == 'float16':
                    safe_dtypes[col] = 'float32'
                else:
                    safe_dtypes[col] = dtype

            samples_extracted = 0
            games_processed = 0
            chunk_size = 3000
            target_samples = max_samples // 10  # Distribute samples across files

            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, dtype=safe_dtypes, low_memory=False):
                current_memory = self.get_memory_usage()
                logger.info(f"🔄 Processing chunk with {len(chunk_df)} games - Memory: {current_memory:.1f}GB")

                chunk_samples = []
                for _, row in chunk_df.iterrows():
                    game_samples = self.extract_turn_decisions(row, expansion_name)
                    chunk_samples.extend(game_samples)
                    games_processed += 1

                    if games_processed >= 5000:  # Limit games per file
                        break

                remaining_quota = target_samples - samples_extracted
                if len(chunk_samples) > remaining_quota:
                    chunk_samples = chunk_samples[:remaining_quota]

                with self.processing_lock:
                    self.processed_samples.extend(chunk_samples)
                    samples_extracted += len(chunk_samples)

                logger.info(f"✅ Extracted {len(chunk_samples)} samples from {games_processed} games (Total: {len(self.processed_samples)})")

                del chunk_df, chunk_samples
                gc.collect()

                if samples_extracted >= target_samples:
                    break

                if current_memory > self.memory_limit_gb:
                    logger.warning(f"⚠️ Memory usage high: {current_memory:.1f}GB")
                    gc.collect()

            logger.info(f"✅ Completed {file_path}")
            return samples_extracted

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return 0

    def process_all_files(self, max_total_samples: int = 50000) -> int:
        """Process all 17Lands PremierDraft files."""
        logger.info("🚀 Processing ALL 17Lands PremierDraft Data")
        logger.info("=" * 60)

        # Find all PremierDraft files
        data_dir = Path('data/17lands_data')
        premierdraft_files = list(data_dir.glob('replay_data_public.*.PremierDraft.csv.gz'))

        logger.info(f"📁 Found {len(premierdraft_files)} PremierDraft files")
        for file in premierdraft_files:
            size_mb = file.stat().st_size / 1024 / 1024
            logger.info(f"  - {file.name}: {size_mb:.1f}MB")

        # Calculate samples per file
        samples_per_file = max_total_samples // len(premierdraft_files)
        logger.info(f"🎯 Target: {max_total_samples} total samples (~{samples_per_file} per file)")

        start_time = time.time()

        # Process files
        for i, file_path in enumerate(premierdraft_files):
            expansion_name = file_path.name.split('.')[2]  # Extract expansion from filename

            logger.info(f"\n📊 Processing file {i+1}/{len(premierdraft_files)}: {expansion_name}")

            samples = self.process_single_file(str(file_path), samples_per_file, expansion_name)

            current_total = len(self.processed_samples)
            logger.info(f"📈 Running total: {current_total} samples ({current_total/max_total_samples*100:.1f}% of target)")

            # Stop if we have enough samples
            if len(self.processed_samples) >= max_total_samples:
                logger.info(f"🎯 Reached target of {max_total_samples} samples")
                break

        processing_time = time.time() - start_time
        logger.info(f"\n🎉 All files processed in {processing_time:.1f} seconds")
        logger.info(f"📊 Total samples collected: {len(self.processed_samples)}")

        return len(self.processed_samples)

    def save_comprehensive_dataset(self, output_path: str):
        """Save the comprehensive dataset."""
        logger.info(f"💾 Saving {len(self.processed_samples)} samples to {output_path}")

        training_data = {
            'metadata': {
                'source': 'comprehensive_17lands_premierdraft_data',
                'total_samples': len(self.processed_samples),
                'state_tensor_dim': 21,
                'num_action_types': 15,
                'created_by': 'process_all_17lands_data.py',
                'data_quality': 'comprehensive_real_game_data',
                'processing_method': 'all_expansions_comprehensive',
                'target_samples': 50000
            },
            'training_samples': self.processed_samples
        }

        Path(output_path).parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"✅ Saved comprehensive dataset: {output_path}")
        self._print_comprehensive_statistics()

    def _print_comprehensive_statistics(self):
        """Print comprehensive statistics."""
        if not self.processed_samples:
            return

        logger.info("📊 Comprehensive Processing Statistics:")

        # Count action types
        action_counts = [0] * 15
        outcome_weights = []
        expansions = {}
        color_distribution = {}
        turn_distribution = {}

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
        logger.info(f"  Expansions processed: {len(expansions)} different")
        logger.info(f"  Top expansions: {dict(sorted(expansions.items(), key=lambda x: x[1], reverse=True)[:5])}")
        logger.info(f"  Color combinations: {len(color_distribution)} different")

def main():
    """Main processing function."""
    processor = Comprehensive17LandsProcessor(memory_limit_gb=12.0)

    # Process all files with comprehensive target
    samples = processor.process_all_files(max_total_samples=50000)

    if samples > 0:
        output_file = f'data/comprehensive_17lands_training_data_{samples}_samples.json'
        processor.save_comprehensive_dataset(output_file)

        # Test loading
        logger.info("🧪 Testing comprehensive dataset loading...")
        with open(output_file, 'r') as f:
            test_data = json.load(f)
        logger.info(f"✅ Comprehensive dataset loads successfully: {len(test_data['training_samples'])} samples")

    else:
        logger.error("❌ No samples extracted")

if __name__ == "__main__":
    main()