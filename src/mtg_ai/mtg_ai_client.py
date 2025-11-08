#!/usr/bin/env python3
"""
MTG AI Client - Easy Integration for Comprehensive MTG AI
Simple interface for the 282-dimensional comprehensive MTG AI system.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from comprehensive_mtg_inference_engine import ComprehensiveMTGInferenceEngine
except ImportError as e:
    logging.error(f"Failed to import inference engine: {e}")
    logging.error("Ensure all dependencies are installed: pip install torch pandas numpy")
    raise

class MTGAClient:
    """
    Simple client for MTG AI predictions.

    Usage:
        client = MTGAClient()
        recommendation = client.get_recommendation(game_state)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MTG AI client.

        Args:
            model_path: Path to the trained model file
        """
        self.logger = logging.getLogger(__name__)
        self.engine = None

        try:
            self.engine = ComprehensiveMTGInferenceEngine(model_path=model_path)
            self.logger.info("🤖 MTG AI Client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MTG AI Client: {e}")
            raise

    def get_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI recommendation for current game state.

        Args:
            game_state: Current game state dictionary

        Returns:
            Dictionary with top recommendation and confidence
        """
        if not self.engine:
            raise RuntimeError("MTG AI engine not initialized")

        try:
            # Get top action
            top_actions = self.engine.get_top_comprehensive_actions(game_state, top_k=1)

            if not top_actions:
                return {
                    'success': False,
                    'error': 'No actions predicted',
                    'recommendation': None
                }

            top_action = top_actions[0]

            # Get full explanation
            explanation = self.engine.explain_comprehensive_decision(game_state)

            return {
                'success': True,
                'recommendation': {
                    'action': top_action['action_type'],
                    'confidence': top_action['confidence'],
                    'action_index': top_action['action_index'],
                    'recommended': top_action['recommended']
                },
                'explanation': explanation.get('explanation', ''),
                'value_score': explanation.get('value_score', 0.0),
                'game_context': explanation.get('game_context', {}),
                'top_3_actions': self.engine.get_top_comprehensive_actions(game_state, top_k=3)
            }

        except Exception as e:
            self.logger.error(f"Error getting recommendation: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendation': None
            }

    def analyze_position(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current board position.

        Args:
            game_state: Current game state dictionary

        Returns:
            Position analysis with strengths and weaknesses
        """
        if not self.engine:
            raise RuntimeError("MTG AI engine not initialized")

        try:
            # Get comprehensive analysis
            explanation = self.engine.explain_comprehensive_decision(game_state)

            # Get all predictions for analysis
            predictions = self.engine.predict_comprehensive_actions(game_state)

            return {
                'position_evaluation': {
                    'value_score': predictions.get('value_score', 0.0),
                    'complexity': predictions.get('game_state_summary', {}).get('board_complexity', 0.0),
                    'turn_phase': explanation.get('game_context', {}).get('turn_phase', 'unknown'),
                    'life_advantage': explanation.get('game_context', {}).get('life_advantage', 0),
                    'board_control': explanation.get('game_context', {}).get('board_control', 0),
                    'resource_status': explanation.get('game_context', {}).get('resource_status', 'average')
                },
                'recommended_strategy': explanation.get('explanation', ''),
                'action_rankings': predictions.get('predictions', [])[:5],
                'key_considerations': self._extract_key_considerations(game_state, explanation)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing position: {e}")
            return {
                'error': str(e),
                'position_evaluation': None
            }

    def _extract_key_considerations(self, game_state: Dict[str, Any], explanation: Dict[str, Any]) -> List[str]:
        """Extract key strategic considerations from the analysis."""
        considerations = []

        try:
            turn = game_state.get('turn_number', 1)
            life = game_state.get('player_life', 20)
            opp_life = game_state.get('opponent_life', 20)
            hand_size = game_state.get('hand_size', 5)
            creatures = game_state.get('creatures_in_play', 0)
            oppo_creatures = game_state.get('opponent_creatures_in_play', 0)

            # Life considerations
            if life <= 10:
                considerations.append("Critical life total - prioritize defense")
            elif life - opp_life <= -10:
                considerations.append("Significant life disadvantage - consider defensive plays")
            elif life - opp_life >= 10:
                considerations.append("Life advantage - can pressure opponent")

            # Board considerations
            if creatures == 0 and turn >= 3:
                considerations.append("No creatures in play - prioritize board development")
            elif creatures >= oppo_creatures + 2:
                considerations.append("Board advantage - can apply pressure")
            elif oppo_creatures >= creatures + 3:
                considerations.append("Opponent has overwhelming board - defensive stance needed")

            # Hand considerations
            if hand_size >= 7:
                considerations.append("Large hand size - can afford to be selective")
            elif hand_size <= 2:
                considerations.append("Low hand size - conserve resources")

            # Turn considerations
            if turn <= 3:
                considerations.append("Early game - focus on development")
            elif turn >= 15:
                considerations.append("Late game - close out the game")

        except Exception as e:
            self.logger.warning(f"Error extracting considerations: {e}")

        return considerations

def create_sample_game_state(
    turn_number: int = 5,
    player_life: int = 20,
    opponent_life: int = 18,
    hand_size: int = 4,
    lands: int = 5,
    creatures: int = 2,
    oppo_creatures: int = 1
) -> Dict[str, Any]:
    """
    Create a sample game state for testing.

    Args:
        turn_number: Current turn number
        player_life: Player's life total
        opponent_life: Opponent's life total
        hand_size: Number of cards in hand
        lands: Lands in play
        creatures: Creatures in play
        oppo_creatures: Opponent creatures in play

    Returns:
        Complete game state dictionary
    """
    return {
        # Core game info
        'turn_number': turn_number,
        'on_play': True,
        'player_life': player_life,
        'opponent_life': opponent_life,
        'hand_size': hand_size,
        'opponent_hand_size': max(0, hand_size - 1),

        # Board state
        'lands_in_play': lands,
        'opponent_lands_in_play': max(0, lands - 1),
        'creatures_in_play': creatures,
        'opponent_creatures_in_play': oppo_creatures,
        'non_creatures_in_play': 0,
        'opponent_non_creatures_in_play': 0,

        # Turn actions (this turn)
        'lands_played_this_turn': 1 if turn_number <= 10 else 0,
        'creatures_cast_this_turn': 1 if creatures > 0 else 0,
        'spells_cast_this_turn': 0,
        'instants_cast_this_turn': 0,
        'abilities_used_this_turn': 0,
        'creatures_attacking': min(creatures, 1) if turn_number >= 3 else 0,
        'creatures_blocking': 0,
        'creatures_unblocked': 0,
        'mana_spent_this_turn': 3 if creatures > 0 else 0,
        'damage_taken_this_turn': 0,
        'damage_dealt_this_turn': 0,

        # Game metadata
        'game_id': 'sample_game',
        'expansion': 'STX',
        'total_turns': max(15, turn_number + 5),
        'won': player_life > opponent_life
    }

def main():
    """Demonstrate the MTG AI Client."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🚀 MTG AI Client Demo")
    logger.info("=" * 50)

    try:
        # Initialize client
        client = MTGAClient()

        # Test scenarios
        scenarios = [
            ("Early Game", create_sample_game_state(turn_number=3, hand_size=6, lands=3, creatures=1)),
            ("Mid Game", create_sample_game_state(turn_number=8, hand_size=4, lands=6, creatures=3, oppo_creatures=2)),
            ("Late Game", create_sample_game_state(turn_number=15, hand_size=2, lands=8, creatures=2, player_life=8, opponent_life=5)),
            ("Defensive", create_sample_game_state(turn_number=6, hand_size=5, lands=5, creatures=1, oppo_creatures=4, player_life=12, opponent_life=18))
        ]

        for scenario_name, game_state in scenarios:
            logger.info(f"\n📊 {scenario_name} Scenario:")
            logger.info(f"   Turn: {game_state['turn_number']}, Life: {game_state['player_life']}-{game_state['opponent_life']}")
            logger.info(f"   Board: {game_state['creatures_in_play']} vs {game_state['opponent_creatures_in_play']} creatures")
            logger.info(f"   Hand: {game_state['hand_size']} cards")

            # Get recommendation
            result = client.get_recommendation(game_state)

            if result['success']:
                rec = result['recommendation']
                logger.info(f"   🎯 Top Recommendation: {rec['action'].replace('_', ' ').title()}")
                logger.info(f"   💯 Confidence: {rec['confidence']:.3f}")
                logger.info(f"   🎪 Value Score: {result['value_score']:.3f}")
            else:
                logger.error(f"   ❌ Error: {result['error']}")

        logger.info("\n✅ MTG AI Client demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()