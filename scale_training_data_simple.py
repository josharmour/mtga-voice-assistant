#!/usr/bin/env python3
"""
Simple Training Data Scaling for MTG AI
Generate synthetic training samples based on game patterns for immediate scaling.
"""

import json
import random
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_samples(num_samples: int = 1000) -> list:
    """
    Generate synthetic training samples based on MTG game patterns.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of training samples
    """
    samples = []

    # Define game state patterns
    game_phases = ['early', 'mid', 'late']
    board_states = ['empty', 'developing', 'established', 'complex']

    for i in range(num_samples):
        # Generate realistic game state
        turn = random.randint(1, 20)
        phase = random.choice(game_phases)
        board_state = random.choice(board_states)

        # Create state tensor (23 dimensions)
        state_tensor = []

        # Basic game info (2 dims)
        state_tensor.append(turn / 20.0)  # Normalized turn number
        state_tensor.append(random.uniform(0, 1))  # Game progress indicator

        # Hand and mana (5 dims)
        if phase == 'early':
            hand_size = random.randint(5, 7)
            mana_available = random.randint(1, 3)
        elif phase == 'mid':
            hand_size = random.randint(3, 6)
            mana_available = random.randint(3, 6)
        else:  # late
            hand_size = random.randint(1, 4)
            mana_available = random.randint(4, 8)

        state_tensor.extend([
            hand_size / 7.0,           # Normalized hand size
            mana_available / 8.0,      # Normalized mana
            random.uniform(0, 1),      # Hand quality
            random.uniform(0, 1),      # Mana efficiency
            random.uniform(0, 1)       # Spell availability
        ])

        # Player state (3 dims)
        player_life = max(1, 20 - random.randint(0, 15))
        player_creatures = random.randint(0, 5)
        player_lands = mana_available

        state_tensor.extend([
            player_life / 20.0,
            player_creatures / 5.0,
            player_lands / 8.0
        ])

        # Opponent state (3 dims)
        oppo_life = max(1, 20 - random.randint(0, 15))
        oppo_creatures = random.randint(0, 5)
        oppo_lands = random.randint(1, 8)

        state_tensor.extend([
            oppo_life / 20.0,
            oppo_creatures / 5.0,
            oppo_lands / 8.0
        ])

        # Board complexity (4 dims)
        if board_state == 'empty':
            complexity = [0.1, 0.1, 0.1, 0.1]
        elif board_state == 'developing':
            complexity = [0.3, 0.4, 0.3, 0.2]
        elif board_state == 'established':
            complexity = [0.6, 0.7, 0.5, 0.4]
        else:  # complex
            complexity = [0.8, 0.9, 0.7, 0.6]

        state_tensor.extend(complexity)

        # Strategic context (4 dims)
        pressure = max(0, min(1, (20 - player_life) / 20.0))
        advantage = max(-1, min(1, (player_life - oppo_life) / 20.0))
        tempo = random.uniform(0, 1)
        value = random.uniform(0, 1)

        state_tensor.extend([pressure, advantage, tempo, value])

        # Ensure exactly 23 dimensions
        state_tensor = state_tensor[:23]

        # Generate action label based on game state (15 action types)
        action_label = [0] * 15

        # Define action types
        # 0: Play creature
        # 1: Attack
        # 2: Defensive play
        # 3: Cast spell
        # 4: Use ability
        # 5: Block
        # 6: Pass priority
        # 7: Play land
        # 8: Remove threat
        # 9: Card draw
        # 10: Combat trick
        # 11: Board wipe
        # 12: Counter spell
        # 13: Resource acceleration
        # 14: Strategic positioning

        # Rule-based action selection
        if hand_size > 0 and mana_available >= 1:
            action_label[0] = 1  # Play creature

        if mana_available >= 1 and player_creatures > 0:
            action_label[1] = 1  # Attack

        if pressure > 0.5 or oppo_creatures > player_creatures:
            action_label[2] = 1  # Defensive play

        if hand_size > 0 and mana_available >= 2:
            action_label[3] = 1  # Cast spell

        if player_creatures > 0:
            action_label[4] = 1  # Use ability

        if oppo_creatures > 0:
            action_label[5] = 1  # Block

        action_label[6] = 1  # Always can pass priority

        if player_lands < 6 and turn <= 8:
            action_label[7] = 1  # Play land

        if oppo_creatures > 2 and mana_available >= 3:
            action_label[8] = 1  # Remove threat

        if hand_size > 2:
            action_label[9] = 1  # Card draw

        if phase in ['mid', 'late'] and mana_available >= 2:
            action_label[10] = 1  # Combat trick

        if oppo_creatures > 3 and mana_available >= 4:
            action_label[11] = 1  # Board wipe

        if phase in ['mid', 'late'] and mana_available >= 2:
            action_label[12] = 1  # Counter spell

        if turn <= 5 and player_lands < 4:
            action_label[13] = 1  # Resource acceleration

        if board_state == 'complex':
            action_label[14] = 1  # Strategic positioning

        # Generate outcome weight
        # Higher weight for winning positions and optimal plays
        base_weight = 0.5
        if advantage > 0:  # Winning
            base_weight += 0.2
        if pressure < 0.3:  # Not under pressure
            base_weight += 0.1
        if tempo > 0.7:  # Good tempo
            base_weight += 0.1

        # Add some randomness
        outcome_weight = min(1.0, base_weight + random.uniform(-0.1, 0.1))

        sample = {
            'state_tensor': state_tensor,
            'action_label': action_label,
            'outcome_weight': outcome_weight,
            'game_id': f'synthetic_{i:04d}',
            'turn_number': turn,
            'phase': phase,
            'board_state': board_state
        }

        samples.append(sample)

    return samples

def create_scaled_dataset(num_samples: int = 1000, output_file: str = 'data/scaled_training_dataset.json'):
    """
    Create scaled training dataset with synthetic samples.

    Args:
        num_samples: Number of samples to generate
        output_file: Output file path
    """
    logger.info(f"🚀 Generating {num_samples} synthetic training samples...")

    samples = generate_synthetic_samples(num_samples)

    # Create training dataset structure
    training_data = {
        'metadata': {
            'source': 'synthetic_generation',
            'total_samples': len(samples),
            'state_tensor_dim': 23,
            'num_action_types': 15,
            'created_by': 'scale_training_data_simple.py',
            'data_quality': 'synthetic_patterns',
            'generation_strategy': 'rule_based_mtgs_patterns'
        },
        'training_samples': samples
    }

    # Save to JSON
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    logger.info(f"✅ Generated {len(samples)} training samples")
    logger.info(f"📁 Saved to {output_file}")

    # Print sample statistics
    action_counts = [0] * 15
    total_weight = 0

    for sample in samples:
        for i, action in enumerate(sample['action_label']):
            if action == 1:
                action_counts[i] += 1
        total_weight += sample['outcome_weight']

    logger.info(f"📊 Sample Statistics:")
    logger.info(f"  Average outcome weight: {total_weight / len(samples):.3f}")
    logger.info(f"  Action distribution: {action_counts}")

    return len(samples)

def main():
    """Main function."""
    logger.info("🎲 Creating Scaled MTG AI Training Dataset")

    # Generate scaled dataset
    num_samples = create_scaled_dataset(num_samples=2000, output_file='data/scaled_training_dataset.json')

    logger.info(f"🎉 Successfully created scaled dataset with {num_samples} samples!")
    logger.info("📈 Ready for improved model training!")

if __name__ == "__main__":
    main()