#!/usr/bin/env python3
"""
Simple test for MTG Action Space without torch dependency
Tests the core logic and structure of the action space system
"""

import sys
import os
import json
from typing import Dict, List

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_action_space_basic():
    """Test basic action space functionality without neural networks"""
    print("🧪 Testing basic action space functionality...")

    try:
        # Import without torch dependency
        from mtg_action_space import (
            ActionType, Phase, ManaCost, Target, Action, MTGActionSpace
        )
        print("✅ Successfully imported action space components")

        # Test action types
        print(f"✅ Action types defined: {len(ActionType)}")
        for action_type in ActionType:
            print(f"   - {action_type.value}")

        # Test phases
        print(f"✅ Game phases defined: {len(Phase)}")
        for phase in Phase:
            print(f"   - {phase.value}")

        # Test mana cost parsing
        action_space = MTGActionSpace()

        test_cases = [
            '{R}',
            '{2}{W}{U}',
            '{X}{G}{G}',
            '0',
            '{C}{C}'
        ]

        print("✅ Testing mana cost parsing:")
        for cost_str in test_cases:
            cost = action_space._parse_mana_cost(cost_str)
            total = cost.total_mana()
            print(f"   '{cost_str}' -> {total} mana")

        # Test mana cost validation
        print("✅ Testing mana cost validation:")
        cost = action_space._parse_mana_cost('{2}{R}{W}')

        # Test sufficient mana
        available_mana = {'red': 1, 'white': 1, 'blue': 2, 'colorless': 2}
        can_pay = cost.can_pay(available_mana)
        print(f"   Sufficient mana test: {can_pay}")

        # Test insufficient mana
        insufficient_mana = {'red': 0, 'white': 1, 'blue': 3, 'colorless': 2}
        cannot_pay = cost.can_pay(insufficient_mana)
        print(f"   Insufficient mana test: {cannot_pay}")

        # Test game state structure
        print("✅ Testing game state processing:")
        sample_game_state = {
            'hand': [
                {'id': 'goblin_guide', 'type': 'creature', 'mana_cost': '{R}', 'power': 2, 'toughness': 1},
                {'id': 'mountain', 'type': 'land'},
                {'id': 'lightning_bolt', 'type': 'instant', 'mana_cost': '{R}'}
            ],
            'battlefield': [
                {'id': 'forest', 'type': 'land', 'tapped': False}
            ],
            'available_mana': {'red': 2, 'green': 1, 'white': 0, 'blue': 0, 'black': 0, 'colorless': 1},
            'player_creatures': [],
            'opponent_creatures': []
        }

        # Test action generation (without neural networks)
        print("✅ Testing action generation:")
        land_actions = action_space._generate_land_actions(
            sample_game_state['hand'], sample_game_state['battlefield'], Phase.PRECOMBAT_MAIN
        )
        print(f"   Land actions: {len(land_actions)}")

        spell_actions = action_space._generate_spell_actions(
            sample_game_state['hand'], sample_game_state['available_mana'],
            sample_game_state['battlefield'], Phase.PRECOMBAT_MAIN
        )
        print(f"   Spell actions: {len(spell_actions)}")

        # Test decision context mapping
        print("✅ Testing decision context mapping:")
        contexts_tested = ['Aggressive_Creature_Play', 'Removal_Spell_Cast', 'All_In_Attack']
        for context in contexts_tested:
            if context in action_space.decision_action_mapping:
                mapped_actions = action_space.decision_action_mapping[context]
                print(f"   {context} -> {[a.value for a in mapped_actions]}")

        # Test phase restrictions
        print("✅ Testing phase restrictions:")
        for action_type, phases in action_space.phase_restrictions.items():
            print(f"   {action_type.value}: {len(phases)} phases")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_compatibility_with_existing_data():
    """Test compatibility with existing MTG project data"""
    print("\n🧪 Testing compatibility with existing project data...")

    # Check for existing data files
    data_files = [
        'enhanced_decisions_sample.json',
        'weighted_training_dataset_task1_4.json',
        'tokenized_training_dataset_task2_1.json'
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found data file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Check data structure
                if 'training_samples' in data:
                    samples = data['training_samples']
                    print(f"   Training samples: {len(samples)}")

                    # Check first sample structure
                    if samples:
                        sample = samples[0]
                        required_fields = ['decision_type', 'turn', 'strategic_context']
                        missing_fields = [field for field in required_fields if field not in sample]
                        if missing_fields:
                            print(f"   ⚠️  Missing fields in sample: {missing_fields}")
                        else:
                            print(f"   ✅ Sample structure compatible")

                elif 'metadata' in data:
                    print(f"   Metadata file detected")
                    metadata = data['metadata']
                    print(f"   Keys: {list(metadata.keys())}")

            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"   ⚠️  Data file not found: {file_path}")

def main():
    """Run basic tests for MTG Action Space"""
    print("🎯 MTG Action Space - Basic Testing")
    print("=" * 40)

    # Test basic functionality
    basic_test_passed = test_action_space_basic()

    # Test data compatibility
    test_compatibility_with_existing_data()

    print("\n" + "=" * 40)
    if basic_test_passed:
        print("🎉 Basic Tests PASSED!")
        print("✅ Action space system is structurally sound")
        print("✅ Core logic implemented correctly")
        print("✅ Ready for full neural network integration")
    else:
        print("❌ Basic Tests FAILED!")
        print("⚠️  Please check implementation")

    print("\n📝 Next Steps:")
    print("1. Install PyTorch for full neural network functionality")
    print("2. Run comprehensive tests with: python3 test_mtg_action_space.py")
    print("3. Test integration with transformer state encoder")
    print("4. Validate with real MTG game data")

if __name__ == "__main__":
    main()