#!/usr/bin/env python3
"""
Scale Training Data for MTG AI
Process 17lands dataset to generate more training samples for improved model performance.
"""

import pandas as pd
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features_from_replay_data(df: pd.DataFrame, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract training samples from 17lands replay data.

    Args:
        df: DataFrame containing replay data
        max_samples: Maximum number of samples to extract

    Returns:
        List of training samples with state tensors and action labels
    """
    samples = []

    # Game state features (subset of the 2579 columns)
    key_features = [
        'game_number', 'turn_number', 'player_hand_size', 'player_mana_available',
        'oppo_hand_size', 'oppo_mana_available', 'player_life', 'oppo_life',
        'player_creatures_on_battlefield', 'oppo_creatures_on_battlefield',
        'player_lands_on_battlefield', 'oppo_lands_on_battlefield'
    ]

    # Filter to columns that exist in the data
    available_features = [col for col in key_features if col in df.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")

    # Extract samples from different games
    unique_games = df['draft_id'].unique()[:max_samples // 10]  # Limit games for diversity

    for game_id in unique_games:
        game_data = df[df['draft_id'] == game_id]
        if len(game_data) < 5:  # Skip very short games
            continue

        # Extract 10 decision points per game
        turn_indices = np.linspace(0, len(game_data) - 1, min(10, len(game_data)), dtype=int)

        for turn_idx in turn_indices:
            row = game_data.iloc[turn_idx]

            # Create state tensor from available features
            state_vector = []

            # Add basic game info
            state_vector.extend([
                row.get('game_number', 0) / 20.0,  # Normalize
                row.get('turn_number', 1) / 30.0,   # Normalize
            ])

            # Add hand and mana info if available
            for feature in ['player_hand_size', 'player_mana_available', 'player_life']:
                if feature in row:
                    state_vector.append(row[feature] / 20.0)  # Normalize
                else:
                    state_vector.append(0.0)

            # Add opponent info if available
            for feature in ['oppo_hand_size', 'oppo_mana_available', 'oppo_life']:
                if feature in row:
                    state_vector.append(row[feature] / 20.0)  # Normalize
                else:
                    state_vector.append(0.0)

            # Add board state info if available
            for feature in ['player_creatures_on_battlefield', 'oppo_creatures_on_battlefield',
                          'player_lands_on_battlefield', 'oppo_lands_on_battlefield']:
                if feature in row:
                    state_vector.append(row[feature] / 10.0)  # Normalize
                else:
                    state_vector.append(0.0)

            # Fill or pad to 23 dimensions (to match our current model)
            while len(state_vector) < 23:
                state_vector.append(0.0)

            state_vector = state_vector[:23]  # Truncate if too long

            # Create action label (multi-hot encoding for 15 action types)
            # Simplified: randomly assign actions based on game state
            action_label = [0] * 15

            # Determine likely actions based on game state
            hand_size = row.get('player_hand_size', 0)
            mana = row.get('player_mana_available', 0)
            creatures = row.get('player_creatures_on_battlefield', 0)
            life = row.get('player_life', 20)

            # Simple rule-based action assignment
            if hand_size > 0 and mana >= 1:
                action_label[0] = 1  # Play creature/spell
            if creatures > 0:
                action_label[1] = 1  # Attack
            if life < 15:
                action_label[2] = 1  # Defensive play
            if mana >= 3:
                action_label[3] = 1  # Cast spell

            # Randomly add some complexity
            for i in range(4, 15):
                if random.random() < 0.1:  # 10% chance
                    action_label[i] = 1

            # Create outcome weight based on game outcome
            game_won = row.get('event_match_wins', 0) > row.get('event_match_losses', 0)
            outcome_weight = 0.8 if game_won else 0.4

            sample = {
                'state_tensor': state_vector,
                'action_label': action_label,
                'outcome_weight': outcome_weight,
                'game_id': game_id,
                'turn_number': row.get('turn_number', 0)
            }

            samples.append(sample)

            if len(samples) >= max_samples:
                return samples

    return samples

def process_17lands_data(data_file: str, output_file: str, max_samples: int = 1000):
    """
    Process 17lands data file and generate training dataset.

    Args:
        data_file: Path to 17lands CSV.gz file
        output_file: Path to output JSON file
        max_samples: Maximum number of samples to generate
    """
    logger.info(f"Processing {data_file}...")

    try:
        # Load data in chunks to manage memory
        chunk_size = 10000
        all_samples = []

        for chunk_df in pd.read_csv(data_file, chunksize=chunk_size):
            logger.info(f"Processing chunk with {len(chunk_df)} rows...")
            samples = extract_features_from_replay_data(chunk_df, max_samples // 2)
            all_samples.extend(samples)

            if len(all_samples) >= max_samples:
                break

            logger.info(f"Collected {len(all_samples)} samples so far...")

        # Limit to max_samples
        all_samples = all_samples[:max_samples]

        # Create training dataset structure
        training_data = {
            'metadata': {
                'source_file': data_file,
                'total_samples': len(all_samples),
                'state_tensor_dim': len(all_samples[0]['state_tensor']) if all_samples else 23,
                'num_action_types': 15,
                'created_by': 'scale_training_data.py'
            },
            'training_samples': all_samples
        }

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"✅ Generated {len(all_samples)} training samples")
        logger.info(f"📁 Saved to {output_file}")

        return len(all_samples)

    except Exception as e:
        logger.error(f"❌ Error processing {data_file}: {e}")
        return 0

def main():
    """Main function to scale training data."""
    logger.info("🚀 Scaling MTG AI Training Data")

    # Define input files (using smaller sets first for testing)
    input_files = [
        'data/17lands_data/replay_data_public.EOE.PremierDraft.csv.gz',  # 313M
        'data/17lands_data/replay_data_public.TDM.PremierDraft.csv.gz',  # 304M
    ]

    output_files = [
        'data/scaled_training_data_EOE.json',
        'data/scaled_training_data_TDM.json',
    ]

    total_samples = 0

    # Process each file
    for input_file, output_file in zip(input_files, output_files):
        if Path(input_file).exists():
            samples = process_17lands_data(input_file, output_file, max_samples=500)
            total_samples += samples
        else:
            logger.warning(f"⚠️ File not found: {input_file}")

    logger.info(f"🎉 Total samples generated: {total_samples}")

    # Create combined dataset
    if total_samples > 0:
        combined_samples = []
        for output_file in output_files:
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    combined_samples.extend(data['training_samples'])

        combined_data = {
            'metadata': {
                'source_files': input_files,
                'total_samples': len(combined_samples),
                'state_tensor_dim': 23,
                'num_action_types': 15,
                'created_by': 'scale_training_data.py'
            },
            'training_samples': combined_samples
        }

        combined_output = 'data/scaled_training_dataset_combined.json'
        with open(combined_output, 'w') as f:
            json.dump(combined_data, f, indent=2)

        logger.info(f"💾 Combined dataset saved: {combined_output}")
        logger.info(f"📊 Total combined samples: {len(combined_samples)}")

if __name__ == "__main__":
    main()