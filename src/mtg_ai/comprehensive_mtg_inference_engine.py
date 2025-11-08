#!/usr/bin/env python3
"""
Comprehensive MTG AI Inference Engine
Real-time gameplay decision prediction using trained 282-dimensional comprehensive model.
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

from working_comprehensive_model import WorkingComprehensiveMTGModel
from safe_comprehensive_extractor import SafeComprehensiveMTGStateExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveMTGInferenceEngine:
    """Real-time inference engine for MTG gameplay decisions using comprehensive model."""

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the comprehensive inference engine.

        Args:
            model_path: Path to trained comprehensive model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path or "/home/joshu/logparser/working_comprehensive_mtg_model.pth"

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

        logger.info(f"🤖 Comprehensive MTG Inference Engine initialized on {self.device}")
        self.load_model()
        self.state_extractor = SafeComprehensiveMTGStateExtractor()

    def load_model(self) -> bool:
        """Load the trained comprehensive model from checkpoint."""
        try:
            logger.info(f"📂 Loading comprehensive model from {self.model_path}")

            # Initialize model architecture
            self.model = WorkingComprehensiveMTGModel(
                input_dim=282,
                d_model=256,
                num_actions=15
            ).to(self.device)

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ Comprehensive model loaded successfully: {param_count:,} parameters")
            logger.info(f"📊 Model processes 282-dimensional comprehensive board state")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def game_state_to_comprehensive_tensor(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert current game state to 282-dimensional comprehensive tensor.

        Args:
            game_state: Dictionary containing current game information

        Returns:
            Normalized tensor ready for comprehensive model input
        """
        try:
            # Create a mock row with the game state information
            # This simulates the 17Lands data structure our model was trained on

            mock_row_data = {
                # Core game info
                'draft_id': game_state.get('game_id', 'unknown'),
                'expansion': game_state.get('expansion', 'unknown'),
                'won': game_state.get('won', False),
                'num_turns': game_state.get('total_turns', 10),
                'on_play': game_state.get('on_play', True),
                'main_colors': ','.join(game_state.get('main_colors', ['R'])),
                'splash_colors': ','.join(game_state.get('splash_colors', [])),
                'num_mulligans': game_state.get('num_mulligans', 0),
                'opp_num_mulligans': game_state.get('opp_num_mulligans', 0),

                # Turn-specific state (for current turn)
                f'user_turn_{game_state["turn_number"]}_lands_played': game_state.get('lands_played_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_creatures_cast': game_state.get('creatures_cast_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_non_creatures_cast': game_state.get('spells_cast_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_user_instants_sorceries_cast': game_state.get('instants_cast_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_user_abilities': game_state.get('abilities_used_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_creatures_attacked': game_state.get('creatures_attacking', 0),
                f'user_turn_{game_state["turn_number"]}_creatures_blocked': game_state.get('creatures_blocking', 0),
                f'user_turn_{game_state["turn_number"]}_creatures_unblocked': game_state.get('creatures_unblocked', 0),
                f'user_turn_{game_state["turn_number"]}_creatures_blocking': game_state.get('creatures_blocking', 0),
                f'user_turn_{game_state["turn_number"]}_user_mana_spent': game_state.get('mana_spent_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_oppo_mana_spent': game_state.get('oppo_mana_spent_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_user_combat_damage_taken': game_state.get('damage_taken_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_oppo_combat_damage_taken': game_state.get('damage_dealt_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_user_creatures_killed_combat': game_state.get('creatures_lost_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_oppo_creatures_killed_combat': game_state.get('oppo_creatures_lost_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_user_creatures_killed_non_combat': game_state.get('creatures_removed_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_oppo_creatures_killed_non_combat': game_state.get('oppo_creatures_removed_this_turn', 0),

                # End of turn state
                f'user_turn_{game_state["turn_number"]}_cards_drawn': game_state.get('cards_drawn_this_turn', 1),
                f'user_turn_{game_state["turn_number"]}_cards_tutored': game_state.get('cards_tutored_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_cards_discarded': game_state.get('cards_discarded_this_turn', 0),
                f'user_turn_{game_state["turn_number"]}_eot_user_cards_in_hand': game_state.get('hand_size', game_state.get('hand_size', 5)),
                f'user_turn_{game_state["turn_number"]}_eot_oppo_cards_in_hand': game_state.get('opponent_hand_size', game_state.get('opponent_hand_size', 5)),
                f'user_turn_{game_state["turn_number"]}_eot_user_life': game_state.get('player_life', 20),
                f'user_turn_{game_state["turn_number"]}_eot_oppo_life': game_state.get('opponent_life', 20),
                f'user_turn_{game_state["turn_number"]}_eot_user_lands_in_play': game_state.get('lands_in_play', game_state.get('lands_played', 0)),
                f'user_turn_{game_state["turn_number"]}_eot_oppo_lands_in_play': game_state.get('opponent_lands_in_play', game_state.get('opponent_lands', 0)),
                f'user_turn_{game_state["turn_number"]}_eot_user_creatures_in_play': game_state.get('creatures_on_battlefield', game_state.get('creatures_in_play', 0)),
                f'user_turn_{game_state["turn_number"]}_eot_oppo_creatures_in_play': game_state.get('opponent_creatures_on_battlefield', game_state.get('opponent_creatures_in_play', 0)),
                f'user_turn_{game_state["turn_number"]}_eot_user_non_creatures_in_play': game_state.get('non_creatures_on_battlefield', game_state.get('non_creatures_in_play', 0)),
                f'user_turn_{game_state["turn_number"]}_eot_oppo_non_creatures_in_play': game_state.get('opponent_non_creatures_on_battlefield', game_state.get('opponent_non_creatures_in_play', 0)),
            }

            # Convert to pandas Series for compatibility with our extractor
            import pandas as pd
            mock_row = pd.Series(mock_row_data)

            # Extract comprehensive state tensor (282 dims)
            state_tensor = self.state_extractor.extract_comprehensive_state(mock_row, game_state["turn_number"])

            logger.debug(f"🔢 Comprehensive game state converted to tensor: {state_tensor.shape}")
            return torch.tensor(state_tensor, dtype=torch.float32)

        except Exception as e:
            logger.error(f"❌ Error converting game state to comprehensive tensor: {e}")
            # Return default tensor
            return torch.zeros(282, dtype=torch.float32)

    def _validate_action_in_context(self, action_type: str, game_state: Dict[str, Any]) -> bool:
        """
        Validate if an action type makes sense in the current game context.
        This prevents logically impossible actions like blocking when there are no attackers.

        Args:
            action_type: The action type to validate
            game_state: Current game state dictionary

        Returns:
            True if action is valid in this context, False otherwise
        """
        turn_number = game_state.get('turn_number', 1)
        player_creatures = game_state.get('creatures_in_play', 0)
        opponent_creatures = game_state.get('opponent_creatures_in_play', 0)
        creatures_attacking = game_state.get('creatures_attacking', 0)
        lands_in_play = game_state.get('lands_in_play', 0)
        hand_size = game_state.get('hand_size', 0)
        creatures_cast_this_turn = game_state.get('creatures_cast_this_turn', 0)
        lands_played_this_turn = game_state.get('lands_played_this_turn', 0)

        # Action-specific validation rules
        if action_type == "block_creatures":
            # Blocking is only valid if opponent has creatures that can attack
            # and we have reason to believe they're attacking
            return opponent_creatures > 0 and creatures_attacking > 0

        elif action_type == "attack_creatures":
            # Attacking is only valid if we have untapped creatures
            # and it's not the first turn (no summoning sickness)
            return player_creatures > 0 and turn_number > 1

        elif action_type == "play_creature":
            # Can only play creature if we have creatures in hand
            # and haven't exceeded reasonable creature limits for turn
            return hand_size > 0 and creatures_cast_this_turn < 3

        elif action_type == "cast_spell":
            # Can only cast spells if we have cards in hand and mana
            return hand_size > 0 and lands_in_play > 0

        elif action_type == "use_ability":
            # Can only use abilities if we have permanents on board
            return (player_creatures > 0 or lands_in_play > 0)

        elif action_type == "pass_priority":
            # Passing priority is always valid
            return True

        elif action_type == "defensive_play":
            # Defensive play is valid if we're behind or opponent has threats
            player_life = game_state.get('player_life', 20)
            opponent_life = game_state.get('opponent_life', 20)
            return (player_life < opponent_life or opponent_creatures > player_creatures)

        # Default: allow action
        return True

    def predict_comprehensive_actions(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal actions for current game state using comprehensive model.

        Args:
            game_state: Current game state dictionary

        Returns:
            Dictionary with action predictions and confidence scores
        """
        try:
            start_time = time.time()

            # Convert game state to comprehensive tensor
            state_tensor = self.game_state_to_comprehensive_tensor(game_state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                action_logits, value = self.model(state_tensor)

                # Apply sigmoid to get probabilities
                action_probs = torch.sigmoid(action_logits).squeeze(0)
                value_score = value.item()

            # Convert to numpy for easier handling
            probs = action_probs.cpu().numpy()

            # Create action predictions with context validation
            predictions = []
            for i, (action_type, prob) in enumerate(zip(self.action_types, probs)):
                # Validate action in current game context
                is_valid_in_context = self._validate_action_in_context(action_type, game_state)

                # Adjust confidence based on validity
                adjusted_confidence = float(prob) if is_valid_in_context else 0.0

                predictions.append({
                    'action_type': action_type,
                    'action_index': i,
                    'confidence': float(prob),
                    'adjusted_confidence': adjusted_confidence,
                    'valid_in_context': is_valid_in_context,
                    'recommended': is_valid_in_context and prob > 0.5  # Only recommend if valid and confident
                })

            # Sort by adjusted confidence (invalid actions go to bottom)
            predictions.sort(key=lambda x: x['adjusted_confidence'], reverse=True)

            # Calculate inference time
            inference_time = time.time() - start_time

            result = {
                'predictions': predictions,
                'value_score': value_score,
                'inference_time_ms': inference_time * 1000,
                'model_device': str(self.device),
                'model_type': 'comprehensive_282d',
                'game_state_summary': {
                    'turn_number': game_state.get('turn_number', 1),
                    'player_life': game_state.get('player_life', 20),
                    'opponent_life': game_state.get('opponent_life', 20),
                    'hand_size': game_state.get('hand_size', 5),
                    'lands_in_play': game_state.get('lands_in_play', 0),
                    'creatures_in_play': game_state.get('creatures_in_play', 0),
                    'opponent_creatures_in_play': game_state.get('opponent_creatures_in_play', 0),
                    'non_creatures_in_play': game_state.get('non_creatures_in_play', 0),
                    'opponent_non_creatures_in_play': game_state.get('opponent_non_creatures_in_play', 0),
                    'board_complexity': self._calculate_board_complexity(game_state)
                }
            }

            logger.info(f"🎯 Comprehensive prediction completed in {inference_time*1000:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ Error during comprehensive prediction: {e}")
            return {
                'predictions': [],
                'value_score': 0.0,
                'inference_time_ms': 0.0,
                'error': str(e)
            }

    def get_top_comprehensive_actions(self, game_state: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-k recommended actions for current game state.

        Args:
            game_state: Current game state dictionary
            top_k: Number of top actions to return

        Returns:
            List of top recommended actions with confidence scores
        """
        predictions = self.predict_comprehensive_actions(game_state)
        return predictions['predictions'][:top_k]

    def explain_comprehensive_decision(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for the model's comprehensive decision.

        Args:
            game_state: Current game state dictionary

        Returns:
            Explanation with reasoning and confidence
        """
        predictions = self.predict_comprehensive_actions(game_state)

        if not predictions['predictions']:
            return {
                'explanation': 'Unable to generate explanation due to prediction error',
                'confidence': 0.0,
                'recommended_actions': []
            }

        top_action = predictions['predictions'][0]
        confidence = top_action['confidence']

        # Generate contextual explanation based on comprehensive board state
        turn = game_state.get('turn_number', 1)
        life = game_state.get('player_life', 20)
        opp_life = game_state.get('opponent_life', 20)
        hand_size = game_state.get('hand_size', 5)
        lands = game_state.get('lands_in_play', 0)
        creatures = game_state.get('creatures_in_play', 0)
        oppo_creatures = game_state.get('opponent_creatures_in_play', 0)

        board_complexity = self._calculate_board_complexity(game_state)

        explanation = f"Turn {turn}: Comprehensive Analysis (Confidence: {confidence:.2f})\n"

        if top_action['action_type'] == 'play_creature':
            explanation += f"Recommended to play a creature. "
            if turn <= 3:
                explanation += "Early board development is crucial for establishing pressure."
            elif turn <= 8:
                explanation += "Mid-game creatures help maintain board presence and tempo."
            else:
                explanation += "Late-game creatures can close out the game or provide defensive options."

        elif top_action['action_type'] == 'attack_creatures':
            explanation += f"Recommended to attack with creatures. "
            if life > opp_life:
                explanation += "You have a life advantage to press the tempo."
            else:
                explanation += "Apply pressure to close the life gap."

        elif top_action['action_type'] == 'cast_spell':
            explanation += f"Recommended to cast a spell. "
            explanation += "Spell casting can provide immediate impact or board control."

        elif top_action['action_type'] == 'play_land':
            explanation += f"Recommended to play a land. "
            explanation += f"Mana development is essential (Current: {lands} lands)."

        elif top_action['action_type'] == 'block_creatures':
            explanation += f"Recommended to block creatures. "
            explanation += f"Defensive positioning to protect your life total."

        elif top_action['action_type'] == 'combat_trick':
            explanation += f"Recommended to use combat tricks. "
            explanation += "Instant-speed effects can swing combat in your favor."

        elif top_action['action_type'] == 'defensive_play':
            explanation += f"Recommended defensive positioning. "
            explanation += "Conservative play may be warranted given the board state."

        elif top_action['action_type'] == 'resource_accel':
            explanation += f"Recommended resource acceleration. "
            explanation += "Additional mana can provide future advantages."

        else:
            explanation += f"Recommended to {top_action['action_type'].replace('_', ' ')} (confidence: {confidence:.2f})."

        # Add comprehensive board analysis
        explanation += f"\n\n📊 Board Analysis:\n"
        explanation += f"  Life: {life} vs {opp_life} (Advantage: {'Player' if life > opp_life else 'Opponent' if life < opp_life else 'Even'})\n"
        explanation += f"  Hand: {hand_size} cards (Resource level: {'High' if hand_size >= 7 else 'Medium' if hand_size >= 4 else 'Low'})\n"
        explanation += f"  Board: {creatures} vs {oppo_creatures} creatures (Control: {'Player' if creatures > oppo_creatures else 'Opponent' if creatures < oppo_creatures else 'Even'})\n"
        explanation += f"  Lands: {lands} mana available\n"
        explanation += f"  Complexity: {board_complexity:.2f} ({'High' if board_complexity > 0.7 else 'Medium' if board_complexity > 0.4 else 'Low'})\n"

        # Strategic context
        explanation += f"🎯 Strategic Context:\n"
        if turn <= 4:
            explanation += "  Early Game: Focus on development and board presence\n"
        elif turn <= 10:
            explanation += "  Mid Game: Balance between aggression and defense\n"
        else:
            explanation += "  Late Game: Close out the game or establish inevitability\n"

        return {
            'explanation': explanation,
            'confidence': confidence,
            'recommended_actions': predictions['predictions'][:3],
            'value_score': predictions['value_score'],
            'game_context': {
                'turn_phase': 'early' if turn <= 4 else 'mid' if turn <= 10 else 'late',
                'life_advantage': life - opp_life,
                'board_control': creatures - oppo_creatures,
                'resource_status': 'strong' if lands >= turn//2 + 1 else 'average' if lands >= turn//3 else 'weak',
                'hand_advantage': hand_size - game_state.get('opponent_hand_size', 5),
                'complexity_level': board_complexity
            }
        }

    def _calculate_board_complexity(self, game_state: Dict[str, Any]) -> float:
        """Calculate board complexity score."""
        creatures = game_state.get('creatures_in_play', 0)
        oppo_creatures = game_state.get('opponent_creatures_in_play', 0)
        non_creatures = game_state.get('non_creatures_in_play', 0)
        oppo_non_creatures = game_state.get('opponent_non_creatures_in_play', 0)
        hand_size = game_state.get('hand_size', 5)

        # Calculate complexity based on board state
        total_permanents = creatures + oppo_creatures + non_creatures + oppo_non_creatures
        board_state_factor = min(1.0, total_permanents / 20.0)
        hand_factor = min(1.0, hand_size / 10.0)

        return (board_state_factor + hand_factor) / 2.0

def main():
    """Main function to demonstrate comprehensive inference engine."""
    logger.info("🚀 Comprehensive MTG AI Inference Engine Demo")
    logger.info("=" * 60)
    logger.info("🧠 Processing with full 282-dimensional board state understanding")

    # Initialize comprehensive inference engine
    engine = ComprehensiveMTGInferenceEngine()

    # Test with sample game states representing different scenarios
    sample_game_states = [
        {
            # Early game setup
            'turn_number': 3,
            'on_play': True,
            'hand_size': 6,
            'lands_in_play': 3,
            'creatures_in_play': 1,
            'opponent_creatures_in_play': 1,
            'player_life': 20,
            'opponent_life': 18,
            'lands_played_this_turn': 1,
            'creatures_cast_this_turn': 1,
            'game_id': 'early_setup_game',
            'expansion': 'STX',
            'won': True,
            'total_turns': 12
        },
        {
            # Mid-game combat
            'turn_number': 8,
            'on_play': False,
            'hand_size': 4,
            'lands_in_play': 6,
            'creatures_in_play': 3,
            'opponent_creatures_in_play': 2,
            'player_life': 15,
            'opponent_life': 12,
            'creatures_attacking': 2,
            'creatures_blocking': 0,
            'damage_dealt_this_turn': 4,
            'game_id': 'mid_game_combat',
            'expansion': 'STX',
            'won': True,
            'total_turns': 12
        },
        {
            # Late game board state
            'turn_number': 15,
            'on_play': True,
            'hand_size': 2,
            'lands_in_play': 8,
            'creatures_in_play': 2,
            'opponent_creatures_in_play': 1,
            'player_life': 8,
            'opponent_life': 5,
            'creatures_attacking': 1,
            'spells_cast_this_turn': 1,
            'mana_spent_this_turn': 7,
            'game_id': 'late_game_control',
            'expansion': 'STX',
            'won': True,
            'total_turns': 12
        },
        {
            # Defensive situation
            'turn_number': 6,
            'on_play': False,
            'hand_size': 5,
            'lands_in_play': 5,
            'creatures_in_play': 1,
            'opponent_creatures_in_play': 4,
            'player_life': 12,
            'opponent_life': 18,
            'creatures_blocking': 1,
            'damage_taken_this_turn': 6,
            'creatures_lost_this_turn': 0,
            'game_id': 'defensive_situation',
            'expansion': 'STX',
            'won': False,
            'total_turns': 12
        }
    ]

    # Test comprehensive inference on each sample state
    for i, game_state in enumerate(sample_game_states):
        logger.info(f"\n📊 Sample Game State {i+1}:")
        logger.info(f"   Game: {game_state['game_id']}")
        logger.info(f"   Turn: {game_state['turn_number']}, Life: {game_state['player_life']}-{game_state['opponent_life']}")
        logger.info(f"   Hand: {game_state['hand_size']}, Lands: {game_state['lands_in_play']}, Creatures: {game_state['creatures_in_play']}-{game_state['opponent_creatures_in_play']}")

        # Get comprehensive predictions
        predictions = engine.predict_comprehensive_actions(game_state)

        # Show top 3 recommendations
        logger.info("   Top 3 Recommended Actions:")
        if 'predictions' in predictions and predictions['predictions']:
            for j, action in enumerate(predictions['predictions'][:3]):
                status = "✅" if action.get('recommended', False) else "💭"
                logger.info(f"   {j+1}. {status} {action.get('action_type', 'unknown').replace('_', ' ').title()} "
                           f"(confidence: {action.get('confidence', 0.0):.3f})")

        # Get comprehensive explanation
        explanation = engine.explain_comprehensive_decision(game_state)
        logger.info(f"   🧠 Comprehensive Explanation:")

        # Format explanation for display
        if 'explanation' in explanation:
            explanation_lines = explanation['explanation'].split('\n')
            for line in explanation_lines[:10]:  # Show first 10 lines
                if line.strip():
                    logger.info(f"   {line}")

        logger.info(f"   ⏱️  Inference time: {predictions.get('inference_time_ms', 0):.1f}ms")
        logger.info(f"   🎯 Value Score: {predictions.get('value_score', 0):.3f}")
        logger.info(f"   📊 Model Type: {predictions.get('model_type', 'comprehensive_282d')}")

    logger.info("\n✅ Comprehensive inference engine demo completed successfully!")
    logger.info("🚀 Ready for real-time MTG gameplay decisions with full board state understanding!")

if __name__ == "__main__":
    main()