#!/usr/bin/env python3
"""
MTG AI Inference Engine
Real-time gameplay decision prediction using trained model.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from memory_safe_training import MemoryOptimizedMTGModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGInferenceEngine:
    """Real-time inference engine for MTG gameplay decisions."""

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path or "mtg_memory_optimized_model.pth"

        # Action type definitions for interpretation
        self.action_types = [
            "play_creature",      # 0
            "attack_creatures",   # 1
            "defensive_play",     # 2
            "cast_spell",         # 3
            "use_ability",        # 4
            "pass_priority",      # 5
            "block_creatures",    # 6
            "play_land",          # 7
            "hold_priority",      # 8
            "draw_card",          # 9
            "combat_trick",       # 10
            "board_wipe",         # 11
            "counter_spell",      # 12
            "resource_accel",     # 13
            "positioning"         # 14
        ]

        logger.info(f"🤖 MTG Inference Engine initialized on {self.device}")
        self.load_model()

    def load_model(self) -> bool:
        """Load the trained model from checkpoint."""
        try:
            logger.info(f"📂 Loading model from {self.model_path}")

            # Initialize model architecture
            self.model = MemoryOptimizedMTGModel(
                input_dim=21,
                d_model=64,
                num_actions=15
            ).to(self.device)

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ Model loaded successfully: {param_count:,} parameters")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def game_state_to_tensor(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert current game state to 21-dimension tensor.

        Args:
            game_state: Dictionary containing current game information

        Returns:
            Normalized tensor ready for model input
        """
        try:
            # Extract game state information with defaults
            turn_number = game_state.get('turn_number', 1)
            total_turns = game_state.get('estimated_total_turns', 20)
            on_play = game_state.get('on_play', True)

            # Hand and mana
            hand_size = game_state.get('hand_size', 5)
            mana_available = game_state.get('mana_available', 3)
            oppo_hand_size = game_state.get('opponent_hand_size', 5)

            # Life totals
            player_life = game_state.get('player_life', 20)
            oppo_life = game_state.get('opponent_life', 20)

            # Board state
            player_creatures = game_state.get('player_creatures', 0)
            oppo_creatures = game_state.get('opponent_creatures', 0)
            player_lands = game_state.get('player_lands', min(8, turn_number // 2 + 1))
            oppo_lands = game_state.get('opponent_lands', min(8, turn_number // 2))

            # Combat state
            creatures_attacking = game_state.get('creatures_attacking', 0)
            creatures_blocking = game_state.get('creatures_blocking', 0)

            # Game progress and strategy
            turn_progress = turn_number / max(20, total_turns)
            mana_spent = game_state.get('mana_spent_this_turn', 0)

            # Strategic context
            pressure_level = max(0, (20 - player_life) / 20.0)
            mana_development = min(1.0, mana_available / 5.0)
            board_presence = min(1.0, player_creatures / 3.0)
            life_diff = abs(player_life - oppo_life) / 20.0

            # Create 21-dimension state tensor
            state_tensor = [
                turn_number / 20.0,                    # Normalized turn number
                turn_progress,                          # Game progress
                hand_size / 7.0,                        # Hand size
                mana_available / 8.0,                   # Available mana
                player_life / 20.0,                     # Life total
                player_creatures / 5.0,                 # Creatures on board
                player_lands / 8.0,                     # Lands on board
                oppo_life / 20.0,                       # Opponent life
                oppo_creatures / 5.0,                   # Opponent creatures
                oppo_lands / 8.0,                       # Opponent lands
                # Board complexity features
                min(1.0, (player_creatures + oppo_creatures) / 10.0),  # Board complexity
                life_diff,                              # Life differential
                mana_spent / 10.0,                      # Mana efficiency
                hand_size / 10.0,                       # Card advantage
                turn_number / 15.0,                     # Timing importance
                # Strategic context
                pressure_level,                          # Pressure level
                mana_development,                        # Mana development
                board_presence,                          # Board presence
                1.0 if on_play else 0.5,                # Play/draw advantage
                1.0 if turn_number <= 3 else 0.5,       # Early game bonus
                1.0 if turn_number >= 10 else 0.5,      # Late game bonus
                np.random.uniform(0.3, 0.7)             # Random noise
            ]

            # Ensure exactly 21 dimensions
            state_tensor = state_tensor[:21]

            # Convert to tensor
            tensor = torch.tensor(state_tensor, dtype=torch.float32)

            logger.debug(f"🔢 Game state converted to tensor: {tensor.shape}")
            return tensor

        except Exception as e:
            logger.error(f"❌ Error converting game state to tensor: {e}")
            # Return default tensor
            return torch.zeros(21, dtype=torch.float32)

    def predict_actions(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal actions for current game state.

        Args:
            game_state: Current game state dictionary

        Returns:
            Dictionary with action predictions and confidence scores
        """
        try:
            start_time = time.time()

            # Convert game state to tensor
            state_tensor = self.game_state_to_tensor(game_state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                action_logits, value = self.model(state_tensor)

                # Apply sigmoid to get probabilities
                action_probs = torch.sigmoid(action_logits).squeeze(0)
                value_score = value.item()

            # Convert to numpy for easier handling
            probs = action_probs.cpu().numpy()

            # Create action predictions
            predictions = []
            for i, (action_type, prob) in enumerate(zip(self.action_types, probs)):
                predictions.append({
                    'action_type': action_type,
                    'action_index': i,
                    'confidence': float(prob),
                    'recommended': prob > 0.5  # Binary decision threshold
                })

            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)

            # Calculate inference time
            inference_time = time.time() - start_time

            result = {
                'predictions': predictions,
                'value_score': value_score,
                'inference_time_ms': inference_time * 1000,
                'model_device': str(self.device),
                'game_state_summary': {
                    'turn_number': game_state.get('turn_number', 1),
                    'player_life': game_state.get('player_life', 20),
                    'opponent_life': game_state.get('opponent_life', 20),
                    'hand_size': game_state.get('hand_size', 5),
                    'mana_available': game_state.get('mana_available', 3)
                }
            }

            logger.info(f"🎯 Prediction completed in {inference_time*1000:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return {
                'predictions': [],
                'value_score': 0.0,
                'inference_time_ms': 0.0,
                'error': str(e)
            }

    def get_top_actions(self, game_state: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-k recommended actions for current game state.

        Args:
            game_state: Current game state dictionary
            top_k: Number of top actions to return

        Returns:
            List of top recommended actions with confidence scores
        """
        predictions = self.predict_actions(game_state)
        return predictions['predictions'][:top_k]

    def explain_decision(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for the model's decision.

        Args:
            game_state: Current game state dictionary

        Returns:
            Explanation with reasoning and confidence
        """
        predictions = self.predict_actions(game_state)

        if not predictions['predictions']:
            return {
                'explanation': 'Unable to generate explanation due to prediction error',
                'confidence': 0.0,
                'recommended_actions': []
            }

        top_action = predictions['predictions'][0]
        confidence = top_action['confidence']

        # Generate contextual explanation
        turn = game_state.get('turn_number', 1)
        life = game_state.get('player_life', 20)
        opp_life = game_state.get('opponent_life', 20)

        explanation = f"Turn {turn}: "

        if top_action['action_type'] == 'play_creature':
            explanation += f"Recommended to play a creature (confidence: {confidence:.2f}). "
            if turn <= 3:
                explanation += "Early game board presence is crucial."
            else:
                explanation += "Board development continues to be important."

        elif top_action['action_type'] == 'attack_creatures':
            explanation += f"Recommended to attack with creatures (confidence: {confidence:.2f}). "
            if life > opp_life:
                explanation += "You have a life advantage to press."
            else:
                explanation += "Apply pressure to close the life gap."

        elif top_action['action_type'] == 'cast_spell':
            explanation += f"Recommended to cast a spell (confidence: {confidence:.2f}). "
            explanation += "Spell casting can provide immediate impact or board control."

        elif top_action['action_type'] == 'play_land':
            explanation += f"Recommended to play a land (confidence: {confidence:.2f}). "
            explanation += "Mana development is essential for future plays."

        elif top_action['action_type'] == 'defensive_play':
            explanation += f"Recommended defensive positioning (confidence: {confidence:.2f}). "
            explanation += "Conservative play may be warranted in this situation."

        else:
            explanation += f"Recommended to {top_action['action_type'].replace('_', ' ')} (confidence: {confidence:.2f})."

        return {
            'explanation': explanation,
            'confidence': confidence,
            'recommended_actions': predictions['predictions'][:3],
            'value_score': predictions['value_score'],
            'game_context': {
                'turn_phase': 'early' if turn <= 4 else 'mid' if turn <= 10 else 'late',
                'life_advantage': life - opp_life,
                'strategic_recommendation': 'aggressive' if life > opp_life else 'defensive' if life < 15 else 'balanced'
            }
        }

def main():
    """Main function to demonstrate inference engine."""
    logger.info("🚀 MTG AI Inference Engine Demo")
    logger.info("=" * 50)

    # Initialize inference engine
    engine = MTGInferenceEngine()

    # Test with sample game states
    sample_game_states = [
        {
            'turn_number': 3,
            'on_play': True,
            'hand_size': 6,
            'mana_available': 3,
            'player_life': 20,
            'opponent_life': 18,
            'player_creatures': 1,
            'opponent_creatures': 1,
            'player_lands': 3,
            'opponent_lands': 2,
            'creatures_attacking': 0,
            'creatures_blocking': 0,
            'mana_spent_this_turn': 0
        },
        {
            'turn_number': 8,
            'on_play': False,
            'hand_size': 4,
            'mana_available': 6,
            'player_life': 15,
            'opponent_life': 12,
            'player_creatures': 3,
            'opponent_creatures': 2,
            'player_lands': 6,
            'opponent_lands': 5,
            'creatures_attacking': 0,
            'creatures_blocking': 0,
            'mana_spent_this_turn': 1
        },
        {
            'turn_number': 15,
            'on_play': True,
            'hand_size': 2,
            'mana_available': 8,
            'player_life': 8,
            'opponent_life': 5,
            'player_creatures': 2,
            'opponent_creatures': 1,
            'player_lands': 8,
            'opponent_lands': 6,
            'creatures_attacking': 2,
            'creatures_blocking': 0,
            'mana_spent_this_turn': 3
        }
    ]

    # Test inference on each sample state
    for i, game_state in enumerate(sample_game_states):
        logger.info(f"\n📊 Sample Game State {i+1}:")
        logger.info(f"   Turn: {game_state['turn_number']}, Life: {game_state['player_life']}-{game_state['opponent_life']}")
        logger.info(f"   Hand: {game_state['hand_size']}, Mana: {game_state['mana_available']}")

        # Get predictions
        predictions = engine.predict_actions(game_state)

        # Show top 3 recommendations
        logger.info("   Top 3 Recommended Actions:")
        for j, action in enumerate(predictions['predictions'][:3]):
            status = "✅" if action['recommended'] else "💭"
            logger.info(f"   {j+1}. {status} {action['action_type'].replace('_', ' ').title()} "
                       f"(confidence: {action['confidence']:.3f})")

        # Get explanation
        explanation = engine.explain_decision(game_state)
        logger.info(f"   🧠 Explanation: {explanation['explanation']}")
        logger.info(f"   ⏱️  Inference time: {predictions['inference_time_ms']:.1f}ms")

    logger.info("\n✅ Inference engine demo completed successfully!")

if __name__ == "__main__":
    main()