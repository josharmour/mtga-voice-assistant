#!/usr/bin/env python3
"""
Task 1.3: Decision Point Extraction and State Representation

Extract key decision moments from MTGA replay data and convert to
model-ready tensor representations for neural network training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging
from typing import Dict, List, Tuple, Optional
import torch

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MTGADecisionPointExtractor:
    """Extracts decision points from 17Lands MTGA replay data"""

    def __init__(self):
        self.decision_types = [
            'main_phase_cast',      # Cast creatures/spells
            'combat_attack',        # Attack decisions
            'combat_block',         # Blocking decisions
            'end_turn',             # End turn decisions
            'mulligan',             # Mulligan decisions
            'play_land',            # Land drop decisions
        ]

        # Key state components to extract
        self.state_components = {
            'hand_cards': [],           # Cards in hand
            'lands_in_play': [],        # Lands available
            'creatures_in_play': [],    # Creatures on battlefield
            'life_total': [],          # Life points
            'turn_number': [],         # Current turn
            'mana_available': [],      # Available mana
            'opponent_state': [],      # Opponent's board state
        }

    def extract_turn_decisions(self, game_data: pd.Series, max_turns: int = 30) -> List[Dict]:
        """Extract decision points from a single game"""
        decisions = []

        # Game metadata
        game_id = game_data.get('draft_id', 'unknown')
        on_play = game_data.get('on_play', True)
        won = game_data.get('won', False)
        num_turns = int(game_data.get('num_turns', 0) or 0)

        # Extract decisions turn by turn
        for turn_num in range(1, min(num_turns + 1, max_turns + 1)):
            try:
                turn_decisions = self._extract_turn_actions(game_data, turn_num, on_play)
                decisions.extend(turn_decisions)
            except Exception as e:
                logger.warning(f"Error extracting decisions for turn {turn_num}: {e}")
                continue

        return decisions

    def _extract_turn_actions(self, game_data: pd.Series, turn_num: int, on_play: bool) -> List[Dict]:
        """Extract all decision points from a specific turn"""
        decisions = []
        turn_prefix = f"user_turn_{turn_num}_"

        # Extract turn state information
        turn_state = self._get_turn_state(game_data, turn_num, on_play)

        # Skip if no meaningful actions this turn
        if not self._has_meaningful_actions(turn_state):
            return decisions

        # 1. Main Phase - Cast Creatures/Spells
        if turn_state.get('creatures_cast', '0') != '0' or turn_state.get('non_creatures_cast', '0') != '0':
            decision = {
                'type': 'main_phase_cast',
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'state': turn_state.copy(),
                'action': {
                    'creatures_cast': turn_state.get('creatures_cast', '0'),
                    'non_creatures_cast': turn_state.get('non_creatures_cast', '0'),
                    'instants_cast': turn_state.get('user_instants_sorceries_cast', '0'),
                }
            }
            decisions.append(decision)

        # 2. Combat - Attack Phase
        if turn_state.get('creatures_attacked', '0') != '0':
            decision = {
                'type': 'combat_attack',
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'state': turn_state.copy(),
                'action': {
                    'attackers': turn_state.get('creatures_attacked', '0'),
                    'damage_dealt': turn_state.get('user_combat_damage_taken', '0'),  # Damage opponent took
                }
            }
            decisions.append(decision)

        # 3. Combat - Block Phase
        if turn_state.get('creatures_blocking', '0') != '0':
            decision = {
                'type': 'combat_block',
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'state': turn_state.copy(),
                'action': {
                    'blockers': turn_state.get('creatures_blocking', '0'),
                    'blocked': turn_state.get('creatures_blocked', '0'),
                }
            }
            decisions.append(decision)

        # 4. Land Play
        if turn_state.get('lands_played', '0') != '0':
            decision = {
                'type': 'play_land',
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'state': turn_state.copy(),
                'action': {
                    'lands_played': turn_state.get('lands_played', '0'),
                }
            }
            decisions.append(decision)

        return decisions

    def _get_turn_state(self, game_data: pd.Series, turn_num: int, on_play: bool) -> Dict:
        """Extract complete game state for a given turn"""
        state = {}
        turn_prefix = f"user_turn_{turn_num}_"

        # Player's actions this turn
        for col in game_data.index:
            if col.startswith(turn_prefix):
                key = col.replace(turn_prefix, '')
                value = game_data[col]
                # Handle None/NaN values
                if pd.isna(value) or value is None:
                    state[key] = '0'
                else:
                    state[key] = str(value)

        # End of turn state (crucial for decision context)
        eot_prefix = f"user_turn_{turn_num}_eot_"
        for col in game_data.index:
            if col.startswith(eot_prefix):
                key = col.replace(eot_prefix, '')
                value = game_data[col]
                # Handle None/NaN values
                if pd.isna(value) or value is None:
                    state[f"eot_{key}"] = '0'
                else:
                    state[f"eot_{key}"] = str(value)

        # Add game context
        state['turn_number'] = turn_num
        state['on_play'] = on_play
        state['cards_drawn'] = state.get('cards_drawn', '0')
        state['mana_spent'] = state.get('user_mana_spent', 0.0)

        return state

    def _has_meaningful_actions(self, turn_state: Dict) -> bool:
        """Check if turn has actions worth learning from"""
        meaningful_actions = [
            'creatures_cast', 'non_creatures_cast', 'lands_played',
            'creatures_attacked', 'creatures_blocking', 'user_instants_sorceries_cast'
        ]

        for action in meaningful_actions:
            if turn_state.get(action, '0') != '0':
                return True
        return False

    def create_state_tensor(self, decision: Dict, max_hand_size: int = 10, max_board_size: int = 20) -> torch.Tensor:
        """Convert decision state to model-ready tensor"""

        state = decision['state']

        # Create feature vector (adjust size based on your model needs)
        features = []

        # Basic game info
        features.append(float(decision['turn']))                    # Turn number
        features.append(1.0 if decision['player'] == 'user' else 0.0)  # Player perspective
        features.append(float(state.get('eot_user_life', 20)))     # Player life
        features.append(float(state.get('eot_oppo_life', 20)))     # Opponent life

        # Mana and resources
        features.append(float(state.get('mana_spent', 0.0)))       # Mana spent this turn
        # Handle creature/land counts that might be card ID lists
        lands = state.get('eot_user_lands_in_play', '0')
        if isinstance(lands, str) and '|' in lands:
            land_count = len(lands.split('|'))
        else:
            land_count = float(lands or '0')
        features.append(land_count)  # Lands in play

        creatures = state.get('eot_user_creatures_in_play', '0')
        if isinstance(creatures, str) and '|' in creatures:
            creature_count = len(creatures.split('|'))
        else:
            creature_count = float(creatures or '0')
        features.append(creature_count)  # Creatures in play

        # Card advantage metrics (handle card ID lists)
        drawn = state.get('cards_drawn', '0')
        if isinstance(drawn, str) and '|' in drawn:
            drawn_count = len(drawn.split('|'))
        else:
            drawn_count = float(drawn or '0')
        features.append(drawn_count)  # Cards drawn this turn

        # Handle card lists (pipe-separated card IDs) - convert to count
        hand_cards = state.get('eot_user_cards_in_hand', '0')
        if isinstance(hand_cards, str) and '|' in hand_cards:
            card_count = len(hand_cards.split('|'))
        else:
            card_count = float(hand_cards or '0')
        features.append(card_count)  # Cards in hand EOT

        # Combat statistics (handle card ID lists)
        attackers = state.get('creatures_attacked', '0')
        if isinstance(attackers, str) and '|' in attackers:
            attacker_count = len(attackers.split('|'))
        else:
            attacker_count = float(attackers or '0')
        features.append(attacker_count)  # Attackers

        blockers = state.get('creatures_blocking', '0')
        if isinstance(blockers, str) and '|' in blockers:
            blocker_count = len(blockers.split('|'))
        else:
            blocker_count = float(blockers or '0')
        features.append(blocker_count)   # Blockers

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32)

        return tensor

    def extract_decisions_from_dataset(self, parquet_file: str, max_games: int = 1000) -> List[Dict]:
        """Extract decision points from a dataset file"""
        logger.info(f"Extracting decisions from {parquet_file}")

        # Load dataset
        df = pd.read_parquet(parquet_file)
        if max_games and len(df) > max_games:
            df = df.head(max_games)

        logger.info(f"Processing {len(df)} games")

        all_decisions = []
        successful_games = 0

        for idx, (_, game_data) in enumerate(df.iterrows()):
            try:
                decisions = self.extract_turn_decisions(game_data)
                if decisions:
                    # Add game metadata to each decision
                    for decision in decisions:
                        decision['game_id'] = game_data.get('draft_id', f'game_{idx}')
                        decision['expansion'] = game_data.get('expansion', 'unknown')
                        decision['game_outcome'] = game_data.get('won', False)
                        decision['num_turns'] = game_data.get('num_turns', 0)

                    all_decisions.extend(decisions)
                    successful_games += 1

                if idx % 100 == 0:
                    logger.info(f"Processed {idx} games, extracted {len(all_decisions)} decisions")

            except Exception as e:
                logger.warning(f"Error processing game {idx}: {e}")
                continue

        logger.info(f"✅ Extracted {len(all_decisions)} decisions from {successful_games} games")
        return all_decisions

def main():
    print("🧠 Task 1.3: Decision Point Extraction and State Representation")
    print("==========================================================")
    print("Extracting key decision moments from MTGA replay data...")
    print()

    # Initialize extractor
    extractor = MTGADecisionPointExtractor()

    # Process sample dataset
    sample_file = "preferred_mtg_data/combined_sample.parquet"

    if not Path(sample_file).exists():
        print(f"❌ Sample file not found: {sample_file}")
        return

    # Extract decisions
    decisions = extractor.extract_decisions_from_dataset(sample_file, max_games=100)

    # Analyze extracted decisions
    print(f"\n📊 Decision Analysis")
    print(f"====================")
    print(f"Total decisions extracted: {len(decisions)}")

    # Count by type
    decision_types = {}
    for decision in decisions:
        dtype = decision['type']
        decision_types[dtype] = decision_types.get(dtype, 0) + 1

    print(f"Decision breakdown:")
    for dtype, count in sorted(decision_types.items()):
        print(f"  {dtype}: {count} decisions")

    # Show sample decisions
    print(f"\n🎯 Sample Decisions")
    print(f"===================")
    for i, decision in enumerate(decisions[:5]):
        print(f"\nDecision {i+1}:")
        print(f"  Type: {decision['type']}")
        print(f"  Turn: {decision['turn']}")
        print(f"  Player: {decision['player']}")
        print(f"  Action: {decision['action']}")

        # Create state tensor
        tensor = extractor.create_state_tensor(decision)
        print(f"  State tensor shape: {tensor.shape}")
        print(f"  Sample features: {tensor[:6].tolist()}...")

    # Save extracted decisions
    if decisions:
        output_file = "extracted_decisions_sample.json"
        import json
        with open(output_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_decisions = []
            for decision in decisions:
                decision_copy = decision.copy()
                decision_copy['state_tensor'] = extractor.create_state_tensor(decision).tolist()
                serializable_decisions.append(decision_copy)
            json.dump(serializable_decisions[:100], f, indent=2)

        print(f"\n💾 Saved sample decisions to: {output_file}")
        print(f"   (First 100 decisions with state tensors)")

    print(f"\n🎉 Task 1.3 Implementation Complete!")
    print(f"Ready for: Task 1.4 (Outcome Weighting and Dataset Assembly)")

if __name__ == "__main__":
    main()