#!/usr/bin/env python3
"""
Enhanced MTGA Decision Point Extractor with Nuanced Decision Types

Extracts strategically meaningful decision moments with context-aware classification
for high-quality MTG AI training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging
from typing import Dict, List, Tuple, Optional
import torch
import json

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMTGADecisionExtractor:
    """Advanced extraction of strategically meaningful MTGA decisions"""

    def __init__(self):
        # Enhanced decision categories with strategic context
        self.strategic_decision_types = {
            # CREATURE PLAY STRATEGIES
            'Aggressive_Creature_Play': 'Play creature with power >= 4',
            'Defensive_Creature_Play': 'Play creature with toughness >= 4 and power <= 3',
            'Value_Creature_Play': 'Play creature with ETB (enter the battlefield) effects',
            'Ramp_Creature_Play': 'Play mana-producing creature',

            # SPELL STRATEGIES
            'Removal_Spell_Cast': 'Cast spell to destroy/remove threats',
            'Card_Advantage_Spell': 'Cast spell for card draw/filtering',
            'Combat_Trick_Cast': 'Cast instant to affect combat',
            'Counter_Spell_Cast': 'Cast spell to counter opponent action',
            'Tutor_Action': 'Search library for specific cards',

            # RESOURCE MANAGEMENT
            'Mana_Acceleration': 'Play additional mana sources',
            'Hand_Management': 'Discard, cycle, or filter hand cards',
            'Graveyard_Interaction': 'Utilize graveyard mechanics',

            # COMBAT STRATEGIES
            'All_In_Attack': 'Attack with most available creatures',
            'Cautious_Attack': 'Selective attack with subset',
            'Bluff_Attack': 'Attack with suboptimal creatures',
            'Strategic_Block': 'Block to preserve key pieces',
            'Chump_Block': 'Sacrifice block to prevent damage',
            'No_Block_Defense': 'Strategic choice to not block',
            'Trade_Block': 'Block for creature exchange',

            # GAME STATE CONTEXT
            'Comeback_Play': 'Action when significantly behind',
            'Tempo_Play': 'Fast pressure development',
            'Control_Setup': 'Prepare defensive board state',
            'Win_Consolidation': 'Secure winning position',
            'End_Turn_Value': 'End turn strategic setup'
        }

        # Strategic context factors
        self.context_factors = [
            'life_differential',           # Life difference
            'board_advantage',             # Creature count advantage
            'card_advantage',             # Hand size advantage
            'mana_advantage',             # Available mana
            'game_phase',                 # Early/mid/late game
            'turn_pressure',              # Urgency level
            'threat_density',             # Opponent threats on board
            'setup_potential'             # Future turn opportunities
        ]

    def extract_strategic_decisions(self, game_data: pd.Series, max_turns: int = 30) -> List[Dict]:
        """Extract strategically meaningful decisions with context"""
        decisions = []

        # Game context
        game_id = game_data.get('draft_id', 'unknown')
        on_play = game_data.get('on_play', True)
        won = game_data.get('won', False)
        num_turns = int(game_data.get('num_turns', 0) or 0)

        # Process each turn
        for turn_num in range(1, min(num_turns + 1, max_turns + 1)):
            try:
                turn_decisions = self._extract_strategic_turn_actions(
                    game_data, turn_num, on_play
                )
                decisions.extend(turn_decisions)
            except Exception as e:
                logger.warning(f"Error in turn {turn_num}: {e}")
                continue

        return decisions

    def _extract_strategic_turn_actions(self, game_data: pd.Series, turn_num: int, on_play: bool) -> List[Dict]:
        """Extract nuanced decisions from a turn with strategic analysis"""
        decisions = []
        turn_state = self._get_enhanced_turn_state(game_data, turn_num, on_play)

        if not self._has_meaningful_actions(turn_state):
            return decisions

        # Analyze creature plays with strategic context
        creature_decisions = self._analyze_creature_plays(turn_state, turn_num, on_play)
        decisions.extend(creature_decisions)

        # Analyze spell usage strategies
        spell_decisions = self._analyze_spell_strategies(turn_state, turn_num, on_play)
        decisions.extend(spell_decisions)

        # Analyze combat decisions
        combat_decisions = self._analyze_combat_strategy(turn_state, turn_num, on_play)
        decisions.extend(combat_decisions)

        # Analyze resource management
        resource_decisions = self._analyze_resource_management(turn_state, turn_num, on_play)
        decisions.extend(resource_decisions)

        # Add strategic context to all decisions
        for decision in decisions:
            decision['strategic_context'] = self._calculate_strategic_context(turn_state, decision)

        return decisions

    def _get_enhanced_turn_state(self, game_data: pd.Series, turn_num: int, on_play: bool) -> Dict:
        """Extract comprehensive game state with all relevant information"""
        state = {}
        turn_prefix = f"user_turn_{turn_num}_"

        # Player actions this turn
        for col in game_data.index:
            if col.startswith(turn_prefix):
                key = col.replace(turn_prefix, '')
                value = game_data[col]
                if pd.isna(value) or value is None:
                    state[key] = '0'
                else:
                    state[key] = str(value)

        # End of turn state
        eot_prefix = f"user_turn_{turn_num}_eot_"
        for col in game_data.index:
            if col.startswith(eot_prefix):
                key = col.replace(eot_prefix, '')
                value = game_data[col]
                if pd.isna(value) or value is None:
                    state[f"eot_{key}"] = '0'
                else:
                    state[f"eot_{key}"] = str(value)

        # Game context
        state['turn_number'] = turn_num
        state['on_play'] = on_play
        state['game_outcome'] = game_data.get('won', False)

        return state

    def _analyze_creature_plays(self, turn_state: Dict, turn_num: int, on_play: bool) -> List[Dict]:
        """Analyze creature plays with strategic intent"""
        decisions = []

        creatures_played = self._parse_card_list(turn_state.get('creatures_cast', '0'))
        if not creatures_played:
            return decisions

        # Classify creature play strategy based on context
        life_diff = self._get_life_differential(turn_state)
        board_state = self._parse_card_list(turn_state.get('eot_user_creatures_in_play', '0'))
        opponent_creatures = self._parse_card_list(turn_state.get('eot_oppo_creatures_in_play', '0'))

        for creature_id in creatures_played:
            # Strategic classification
            if life_diff < -10:  # Behind significantly
                strategy_type = 'Defensive_Creature_Play'
            elif len(board_state) >= 3:  # Established board
                strategy_type = 'Value_Creature_Play'
            elif len(opponent_creatures) == 0:  # Open board
                strategy_type = 'Aggressive_Creature_Play'
            else:
                strategy_type = 'Value_Creature_Play'  # Default

            decision = {
                'type': strategy_type,
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'action_details': {
                    'creature_id': creature_id,
                    'board_before': len(board_state),
                    'opp_creatures': len(opponent_creatures),
                    'life_diff': life_diff
                },
                'strategic_reasoning': self._get_creature_strategy_reasoning(
                    strategy_type, turn_state
                )
            }
            decisions.append(decision)

        return decisions

    def _analyze_spell_strategies(self, turn_state: Dict, turn_num: int, on_play: bool) -> List[Dict]:
        """Analyze spell usage with strategic classification"""
        decisions = []

        non_creatures = self._parse_card_list(turn_state.get('non_creatures_cast', '0'))
        instants = self._parse_card_list(turn_state.get('user_instants_sorceries_cast', '0'))
        abilities = self._parse_card_list(turn_state.get('user_abilities', '0'))

        # Combat damage indicators suggest combat tricks
        damage_taken = turn_state.get('user_combat_damage_taken', '0')
        damage_dealt = turn_state.get('oppo_combat_damage_taken', '0')

        if instants and (damage_taken != '0' or damage_dealt != '0'):
            # Likely combat tricks
            for spell_id in instants:
                decision = {
                    'type': 'Combat_Trick_Cast',
                    'turn': turn_num,
                    'player': 'user' if on_play else 'oppo',
                    'action_details': {
                        'spell_id': spell_id,
                        'combat_context': True
                    },
                    'strategic_reasoning': 'Used during combat for tactical advantage'
                }
                decisions.append(decision)

        # Card advantage spells
        if non_creatures:
            cards_drawn = turn_state.get('cards_drawn', '0')
            if self._is_card_advantage_spell(cards_drawn):
                for spell_id in non_creatures:
                    decision = {
                        'type': 'Card_Advantage_Spell',
                        'turn': turn_num,
                        'player': 'user' if on_play else 'oppo',
                        'action_details': {
                            'spell_id': spell_id,
                            'cards_generated': cards_drawn
                        },
                        'strategic_reasoning': 'Played for card advantage and resource generation'
                    }
                    decisions.append(decision)

        # Tutor actions
        tutored = self._parse_card_list(turn_state.get('cards_tutored', '0'))
        if tutored:
            for card_id in tutored:
                decision = {
                    'type': 'Tutor_Action',
                    'turn': turn_num,
                    'player': 'user' if on_play else 'oppo',
                    'action_details': {
                        'tutored_card': card_id
                    },
                    'strategic_reasoning': 'Searched library for specific answer or threat'
                }
                decisions.append(decision)

        return decisions

    def _analyze_combat_strategy(self, turn_state: Dict, turn_num: int, on_play: bool) -> List[Dict]:
        """Analyze combat decisions with strategic nuance"""
        decisions = []

        attackers = self._parse_card_list(turn_state.get('creatures_attacked', '0'))
        blockers = self._parse_card_list(turn_state.get('creatures_blocking', '0'))
        blocked = self._parse_card_list(turn_state.get('creatures_blocked', '0'))

        board_creatures = self._parse_card_list(turn_state.get('eot_user_creatures_in_play', '0'))
        opp_creatures = self._parse_card_list(turn_state.get('eot_oppo_creatures_in_play', '0'))

        # Attack strategy analysis
        if attackers:
            if len(attackers) == len(board_creatures) and len(board_creatures) > 2:
                attack_type = 'All_In_Attack'
                reasoning = 'Committed most creatures to attack'
            elif len(attackers) > len(board_creatures) // 2:
                attack_type = 'Cautious_Attack'
                reasoning = 'Selective attack maintaining board presence'
            else:
                attack_type = 'Bluff_Attack'
                reasoning = 'Minimal attack, possibly probing or bluffing'

            decision = {
                'type': attack_type,
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'action_details': {
                    'attackers': len(attackers),
                    'total_creatures': len(board_creatures),
                    'opp_creatures': len(opp_creatures)
                },
                'strategic_reasoning': reasoning
            }
            decisions.append(decision)

        # Block strategy analysis
        if blockers and opp_creatures:
            if len(blockers) == len(opp_creatures):
                block_type = 'Strategic_Block'
                reasoning = 'Blocked to control damage and preserve key creatures'
            elif len(blockers) < len(opp_creatures):
                block_type = 'Chump_Block'
                reasoning = 'Sacrificial blocking to prevent lethal damage'
            else:
                block_type = 'Trade_Block'
                reasoning = 'Looking for favorable creature exchanges'

            decision = {
                'type': block_type,
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'action_details': {
                    'blockers': len(blockers),
                    'blocked': len(blocked),
                    'attackers': len(opp_creatures)
                },
                'strategic_reasoning': reasoning
            }
            decisions.append(decision)

        # No block decision
        if not blockers and opp_creatures and attackers:
            decision = {
                'type': 'No_Block_Defense',
                'turn': turn_num,
                'player': 'user' if on_play else 'oppo',
                'action_details': {
                    'opp_attackers': len(opp_creatures),
                    'reason': 'strategic_choice'
                },
                'strategic_reasoning': 'Chose not to block, possibly to preserve creatures'
            }
            decisions.append(decision)

        return decisions

    def _analyze_resource_management(self, turn_state: Dict, turn_num: int, on_play: bool) -> List[Dict]:
        """Analyze resource management decisions"""
        decisions = []

        # Lands played
        lands_played = self._parse_card_list(turn_state.get('lands_played', '0'))
        if lands_played:
            current_turn = turn_num
            if current_turn <= 3:
                strategy = 'Mana_Acceleration'
                reasoning = 'Early game mana development'
            else:
                strategy = 'Mana_Acceleration'
                reasoning = 'Additional mana sources for late game'

            for land_id in lands_played:
                decision = {
                    'type': strategy,
                    'turn': turn_num,
                    'player': 'user' if on_play else 'oppo',
                    'action_details': {
                        'land_id': land_id,
                        'turn_played': current_turn
                    },
                    'strategic_reasoning': reasoning
                }
                decisions.append(decision)

        # Hand management
        discarded = self._parse_card_list(turn_state.get('cards_discarded', '0'))
        if discarded:
            for card_id in discarded:
                decision = {
                    'type': 'Hand_Management',
                    'turn': turn_num,
                    'player': 'user' if on_play else 'oppo',
                    'action_details': {
                        'discarded_card': card_id
                    },
                    'strategic_reasoning': 'Discarded for hand optimization or effect'
                }
                decisions.append(decision)

        return decisions

    def _calculate_strategic_context(self, turn_state: Dict, decision: Dict) -> Dict:
        """Calculate comprehensive strategic context for decision"""
        context = {}

        # Life differential
        user_life = float(turn_state.get('eot_user_life', 20))
        oppo_life = float(turn_state.get('eot_oppo_life', 20))
        context['life_differential'] = user_life - oppo_life

        # Board advantage
        user_creatures = len(self._parse_card_list(turn_state.get('eot_user_creatures_in_play', '0')))
        oppo_creatures = len(self._parse_card_list(turn_state.get('eot_oppo_creatures_in_play', '0')))
        context['board_advantage'] = user_creatures - oppo_creatures

        # Card advantage
        user_hand = len(self._parse_card_list(turn_state.get('eot_user_cards_in_hand', '0')))
        context['card_advantage'] = user_hand  # Would need opponent hand count for full analysis

        # Game phase
        turn_num = turn_state.get('turn_number', 1)
        if turn_num <= 4:
            context['game_phase'] = 'early'
        elif turn_num <= 10:
            context['game_phase'] = 'mid'
        else:
            context['game_phase'] = 'late'

        # Pressure level
        if context['life_differential'] < -10:
            context['pressure_level'] = 'high_defensive'
        elif context['life_differential'] > 10:
            context['pressure_level'] = 'aggressive_advantage'
        else:
            context['pressure_level'] = 'balanced'

        return context

    def _parse_card_list(self, card_string: str) -> List[str]:
        """Parse pipe-separated card IDs into list"""
        if not card_string or card_string == '0':
            return []
        return [card_id.strip() for card_id in str(card_string).split('|') if card_id.strip()]

    def _has_meaningful_actions(self, turn_state: Dict) -> bool:
        """Check if turn has strategically relevant actions"""
        action_indicators = [
            'creatures_cast', 'non_creatures_cast', 'lands_played',
            'creatures_attacked', 'creatures_blocking', 'user_instants_sorceries_cast',
            'cards_tutored', 'cards_discarded'
        ]

        for indicator in action_indicators:
            if turn_state.get(indicator, '0') != '0':
                return True
        return False

    def _get_life_differential(self, turn_state: Dict) -> float:
        """Calculate life differential (positive = winning)"""
        user_life = float(turn_state.get('eot_user_life', 20))
        oppo_life = float(turn_state.get('eot_oppo_life', 20))
        return user_life - oppo_life

    def _is_card_advantage_spell(self, cards_drawn: str) -> bool:
        """Simple heuristic for card advantage spells"""
        drawn_cards = self._parse_card_list(cards_drawn)
        return len(drawn_cards) > 1

    def _get_creature_strategy_reasoning(self, strategy_type: str, turn_state: Dict) -> str:
        """Generate reasoning for creature play strategy"""
        reasoning_map = {
            'Aggressive_Creature_Play': 'Played to apply early pressure',
            'Defensive_Creature_Play': 'Played to stabilize losing position',
            'Value_Creature_Play': 'Played for strategic board advantage',
            'Ramp_Creature_Play': 'Played to accelerate mana development'
        }
        return reasoning_map.get(strategy_type, 'Strategic creature deployment')

    def create_enhanced_state_tensor(self, decision: Dict) -> torch.Tensor:
        """Create comprehensive state tensor with strategic context"""
        context = decision.get('strategic_context', {})
        action_details = decision.get('action_details', {})

        features = []

        # Basic game info
        features.append(float(decision.get('turn', 1)))
        features.append(1.0 if decision.get('player') == 'user' else 0.0)

        # Strategic context
        features.append(float(context.get('life_differential', 0)))
        features.append(float(context.get('board_advantage', 0)))
        features.append(float(context.get('card_advantage', 0)))

        # Game phase encoding (early=0, mid=1, late=2)
        phase_encoding = {'early': 0.0, 'mid': 1.0, 'late': 2.0}
        features.append(phase_encoding.get(context.get('game_phase', 'mid'), 1.0))

        # Pressure level encoding (defensive=-1, balanced=0, aggressive=1)
        pressure_encoding = {'high_defensive': -1.0, 'balanced': 0.0, 'aggressive_advantage': 1.0}
        features.append(pressure_encoding.get(context.get('pressure_level', 'balanced'), 0.0))

        # Action-specific features
        if 'creature_id' in action_details:
            features.append(1.0)  # Is creature play
        else:
            features.append(0.0)

        if 'spell_id' in action_details:
            features.append(1.0)  # Is spell play
        else:
            features.append(0.0)

        # Combat features
        features.append(float(action_details.get('attackers', 0)))
        features.append(float(action_details.get('blockers', 0)))
        features.append(float(action_details.get('total_creatures', 0)))

        # Resource features
        features.append(float(action_details.get('lands_played', 0)))

        return torch.tensor(features, dtype=torch.float32)

    def extract_enhanced_decisions_from_dataset(self, parquet_file: str, max_games: int = 100) -> List[Dict]:
        """Extract enhanced strategic decisions from dataset"""
        logger.info(f"Extracting enhanced decisions from {parquet_file}")

        df = pd.read_parquet(parquet_file)
        if max_games and len(df) > max_games:
            df = df.head(max_games)

        logger.info(f"Processing {len(df)} games for enhanced analysis")

        all_decisions = []
        successful_games = 0

        for idx, (_, game_data) in enumerate(df.iterrows()):
            try:
                decisions = self.extract_strategic_decisions(game_data)
                if decisions:
                    # Add game metadata
                    for decision in decisions:
                        decision['game_id'] = game_data.get('draft_id', f'game_{idx}')
                        decision['expansion'] = game_data.get('expansion', 'unknown')
                        decision['game_outcome'] = game_data.get('won', False)
                        decision['num_turns'] = game_data.get('num_turns', 0)

                    all_decisions.extend(decisions)
                    successful_games += 1

                if idx % 20 == 0:
                    logger.info(f"Processed {idx} games, extracted {len(all_decisions)} enhanced decisions")

            except Exception as e:
                logger.warning(f"Error processing game {idx}: {e}")
                continue

        logger.info(f"✅ Extracted {len(all_decisions)} enhanced decisions from {successful_games} games")
        return all_decisions

def main():
    print("🧠 Enhanced MTGA Decision Point Extraction")
    print("==========================================")
    print("Extracting strategically nuanced decisions with context...")
    print()

    # Initialize enhanced extractor
    extractor = EnhancedMTGADecisionExtractor()

    # Process sample dataset
    sample_file = "preferred_mtg_data/combined_sample.parquet"

    if not Path(sample_file).exists():
        print(f"❌ Sample file not found: {sample_file}")
        return

    # Extract enhanced decisions
    decisions = extractor.extract_enhanced_decisions_from_dataset(sample_file, max_games=50)

    # Analysis
    print(f"\n📊 Enhanced Decision Analysis")
    print(f"=============================")
    print(f"Total enhanced decisions extracted: {len(decisions)}")

    # Count by strategic type
    decision_types = {}
    for decision in decisions:
        dtype = decision['type']
        decision_types[dtype] = decision_types.get(dtype, 0) + 1

    print(f"Strategic decision breakdown:")
    for dtype, count in sorted(decision_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dtype:<25} : {count:3d} decisions")

    # Show strategic context distribution
    contexts = {}
    for decision in decisions:
        pressure = decision.get('strategic_context', {}).get('pressure_level', 'unknown')
        contexts[pressure] = contexts.get(pressure, 0) + 1

    print(f"\nStrategic pressure distribution:")
    for pressure, count in contexts.items():
        print(f"  {pressure:<20} : {count} decisions")

    # Show sample enhanced decisions
    print(f"\n🎯 Sample Enhanced Decisions")
    print(f"=============================")

    for i, decision in enumerate(decisions[:5]):
        print(f"\nDecision {i+1}:")
        print(f"  Type: {decision['type']}")
        print(f"  Turn: {decision['turn']}")
        print(f"  Player: {decision['player']}")
        print(f"  Strategic Reasoning: {decision['strategic_reasoning']}")
        print(f"  Action Details: {decision['action_details']}")

        context = decision.get('strategic_context', {})
        print(f"  Context: {context['pressure_level']} game, {context['life_differential']:+.0f} life diff")

        # Create enhanced tensor
        tensor = extractor.create_enhanced_state_tensor(decision)
        print(f"  Enhanced tensor shape: {tensor.shape}")
        print(f"  Key features: turn={tensor[0]:.1f}, life_diff={tensor[2]:.1f}, pressure={tensor[5]:.1f}")

    # Save enhanced decisions
    if decisions:
        output_file = "enhanced_decisions_sample.json"
        serializable_decisions = []
        for decision in decisions[:100]:  # First 100 for sample
            decision_copy = decision.copy()
            decision_copy['enhanced_state_tensor'] = extractor.create_enhanced_state_tensor(decision).tolist()
            serializable_decisions.append(decision_copy)

        with open(output_file, 'w') as f:
            json.dump(serializable_decisions, f, indent=2)

        print(f"\n💾 Saved enhanced decisions to: {output_file}")
        print(f"   (First 100 enhanced decisions with strategic context)")

    print(f"\n🎉 Enhanced Task 1.3 Implementation Complete!")
    print(f"Strategic decision types: {len(decision_types)}")
    print(f"Ready for high-quality MTG AI model training!")

if __name__ == "__main__":
    main()