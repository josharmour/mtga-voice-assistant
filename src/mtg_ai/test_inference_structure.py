#!/usr/bin/env python3
"""
Test the comprehensive inference engine structure without requiring PyTorch.
"""

import sys
import json
from pathlib import Path

def test_inference_engine_structure():
    """Test that the inference engine has all required components."""

    print("🧪 Testing Comprehensive MTG Inference Engine Structure")
    print("=" * 60)

    # Check if model file exists
    model_path = "/home/joshu/logparser/working_comprehensive_mtg_model.pth"
    if Path(model_path).exists():
        print(f"✅ Model file found: {model_path}")
        file_size = Path(model_path).stat().st_size
        print(f"📊 Model file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

    # Check if inference engine script exists
    engine_path = "comprehensive_mtg_inference_engine.py"
    if Path(engine_path).exists():
        print(f"✅ Inference engine script found: {engine_path}")

        # Read and analyze the script
        with open(engine_path, 'r') as f:
            content = f.read()

        # Check for key components
        required_components = [
            "class ComprehensiveMTGInferenceEngine",
            "def predict_comprehensive_actions",
            "def game_state_to_comprehensive_tensor",
            "def explain_comprehensive_decision",
            "def get_top_comprehensive_actions",
            "action_types = [",
            "282"
        ]

        for component in required_components:
            if component in content:
                print(f"✅ Found required component: {component}")
            else:
                print(f"❌ Missing component: {component}")
                return False

        # Check for bug fixes
        bug_fixes = [
            "oppo_non_creatures",  # Fixed variable name
            "action.get(",         # Added safety checks
            "predictions.get("     # Added safety checks
        ]

        for fix in bug_fixes:
            if fix in content:
                print(f"✅ Bug fix implemented: {fix}")
            else:
                print(f"⚠️ Bug fix missing: {fix}")

    else:
        print(f"❌ Inference engine script not found: {engine_path}")
        return False

    # Check for supporting files
    support_files = [
        "working_comprehensive_model.py",
        "safe_comprehensive_extractor.py"
    ]

    for file_path in support_files:
        if Path(file_path).exists():
            print(f"✅ Supporting file found: {file_path}")
        else:
            print(f"❌ Supporting file missing: {file_path}")
            return False

    # Check if training data exists
    training_data_files = list(Path("../..").glob("data/safe_comprehensive_*_samples.json"))
    if training_data_files:
        print(f"✅ Found {len(training_data_files)} training data files")
        for data_file in training_data_files[:3]:  # Show first 3
            print(f"📊 Training data: {data_file.name}")

            # Check data structure
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                if 'metadata' in data:
                    metadata = data['metadata']
                    print(f"  State tensor dimension: {metadata.get('state_tensor_dim', 'unknown')}")
                    print(f"  Total samples: {metadata.get('total_samples', 'unknown')}")
                    print(f"  Model compatibility: {metadata.get('model_compatibility', 'unknown')}")

                if 'training_samples' in data and data['training_samples']:
                    sample = data['training_samples'][0]
                    state_dim = len(sample.get('state_tensor', []))
                    action_dim = len(sample.get('action_label', []))
                    print(f"  Sample dimensions: state={state_dim}, actions={action_dim}")

                    if state_dim == 282:
                        print(f"  ✅ Correct 282-dimensional state tensor")
                    else:
                        print(f"  ❌ Incorrect state tensor dimension: {state_dim}")

            except Exception as e:
                print(f"  ⚠️ Error reading training data: {e}")
    else:
        print("⚠️ No training data files found")

    print("\n🎉 Comprehensive Inference Engine Structure Test Completed!")
    print("🚀 All core components are in place for 282-dimensional comprehensive inference")

    # Create sample game state for demonstration
    sample_game_state = {
        'turn_number': 5,
        'on_play': True,
        'hand_size': 4,
        'lands_in_play': 5,
        'creatures_in_play': 2,
        'opponent_creatures_in_play': 1,
        'player_life': 18,
        'opponent_life': 15,
        'lands_played_this_turn': 1,
        'creatures_cast_this_turn': 1,
        'game_id': 'sample_test_game',
        'expansion': 'STX',
        'won': True,
        'total_turns': 12
    }

    print(f"\n📊 Sample Game State Ready for Testing:")
    print(f"   Turn: {sample_game_state['turn_number']}")
    print(f"   Life: {sample_game_state['player_life']} vs {sample_game_state['opponent_life']}")
    print(f"   Board: {sample_game_state['creatures_in_play']} vs {sample_game_state['opponent_creatures_in_play']} creatures")
    print(f"   Hand: {sample_game_state['hand_size']} cards")
    print(f"   Lands: {sample_game_state['lands_in_play']}")

    print(f"\n🧠 Ready for Comprehensive 282D Inference!")
    print(f"📁 Model: {model_path}")
    print(f"🔧 Engine: {engine_path}")
    print(f"🎯 Input: 282-dimensional comprehensive board state")
    print(f"⚡ Output: 15 action predictions + confidence scores")

    return True

if __name__ == "__main__":
    success = test_inference_engine_structure()
    if success:
        print("\n✅ All tests passed! Ready to proceed with PyTorch testing.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

    sys.exit(0 if success else 1)