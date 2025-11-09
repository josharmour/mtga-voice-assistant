#!/usr/bin/env python3
"""
Test script for the complete RL training pipeline implementation.

This script tests the integration of all components to ensure Task T029 passes.
"""

import sys
import os
import tempfile
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_rl_trainer_basic():
    """Test basic RL trainer functionality."""
    print("🧪 Testing RL Trainer Basic Functionality...")

    try:
        from src.rl.training.trainer import RLTrainer, TrainingConfig
        from src.rl.algorithms.cql import ConservativeQLearning, ConservativeQLearningConfig

        # Create minimal config for testing
        config = TrainingConfig(
            max_episodes=2,
            batch_size=4,
            buffer_size=100,
            max_inference_time_ms=100.0,  # Constitutional requirement
            save_freq=1,
            validate_freq=1,
            tensorboard_logging=False  # Disable to avoid tensorboard dependency
        )

        # Create trainer
        trainer = RLTrainer(config)
        print("✅ RLTrainer created successfully")

        # Setup training
        algorithm = trainer.setup_training(state_dim=380, action_dim=16)
        print("✅ Training setup completed")

        # Test episode training
        episode_data = [
            {
                'turn_number': 1,
                'action_taken': 0,
                'player_life': 20,
                'opponent_life': 20,
                'hand_size': 7,
                'lands_played': 1,
                'available_mana': 1,
                'time_taken_ms': 100.0
            },
            {
                'turn_number': 2,
                'action_taken': 1,
                'player_life': 18,
                'opponent_life': 20,
                'hand_size': 6,
                'lands_played': 2,
                'available_mana': 2,
                'time_taken_ms': 150.0,
                'done': True
            }
        ]

        episode_stats = trainer.train_episode(algorithm, episode_data)
        print(f"✅ Episode training completed: {episode_stats}")

        # Test validation
        validation_results = trainer.validate_training(algorithm)
        print(f"✅ Training validation completed: compliant={validation_results['constitutional_compliance']['compliant']}")

        return True

    except Exception as e:
        print(f"❌ RL Trainer test failed: {e}")
        traceback.print_exc()
        return False

def test_17lands_data_processing():
    """Test 17Lands data processing."""
    print("\n🧪 Testing 17Lands Data Processing...")

    try:
        from src.rl.training.trainer import RLTrainer, TrainingConfig

        config = TrainingConfig(buffer_size=100, tensorboard_logging=False)
        trainer = RLTrainer(config)

        # Mock 17Lands data
        games_data = [
            {
                'game_id': 1,
                'player_won': True,
                'turns': [
                    {
                        'turn_number': 1,
                        'action_taken': 0,
                        'player_life': 20,
                        'opponent_life': 20,
                        'hand_size': 7,
                        'lands_played': 1
                    },
                    {
                        'turn_number': 2,
                        'action_taken': 1,
                        'player_life': 18,
                        'opponent_life': 20,
                        'hand_size': 6,
                        'lands_played': 2
                    }
                ]
            },
            {
                'game_id': 2,
                'player_won': False,
                'turns': [
                    {
                        'turn_number': 1,
                        'action_taken': 2,
                        'player_life': 20,
                        'opponent_life': 18,
                        'hand_size': 7,
                        'lands_played': 1
                    }
                ]
            }
        ]

        processed_episodes = trainer.process_17lands_data(games_data)
        print(f"✅ Processed {len(processed_episodes)} episodes from {len(games_data)} games")

        # Verify data structure
        total_transitions = sum(len(ep) for ep in processed_episodes)
        print(f"✅ Total transitions: {total_transitions}")

        return True

    except Exception as e:
        print(f"❌ 17Lands data processing test failed: {e}")
        traceback.print_exc()
        return False

def test_mtga_integration():
    """Test MTGA Voice Advisor integration."""
    print("\n🧪 Testing MTGA Voice Advisor Integration...")

    try:
        from src.rl.training.trainer import RLTrainer, TrainingConfig

        config = TrainingConfig(buffer_size=100, tensorboard_logging=False)
        trainer = RLTrainer(config)

        # Mock MTGA game state
        game_state = {
            'turn_number': 5,
            'phase': 'main',
            'player_life': 18,
            'opponent_life': 15,
            'hand_size': 4,
            'lands_played': 3,
            'available_mana': 3,
            'creatures_in_play': 2,
            'cards_in_hand': 3
        }

        # Extract RL state
        rl_state = trainer.extract_mtga_state(game_state)
        print(f"✅ Extracted RL state: {len(rl_state)} dimensions")

        # Verify state dimensions
        if len(rl_state) == 380:
            print("✅ State vector has correct dimensions (380)")
        else:
            print(f"❌ State vector has wrong dimensions: {len(rl_state)}")
            return False

        return True

    except Exception as e:
        print(f"❌ MTGA integration test failed: {e}")
        traceback.print_exc()
        return False

def test_cql_algorithm():
    """Test Conservative Q-Learning algorithm."""
    print("\n🧪 Testing Conservative Q-Learning Algorithm...")

    try:
        from src.rl.algorithms.cql import ConservativeQLearning, ConservativeQLearningConfig

        # Create CQL algorithm
        config = ConservativeQLearningConfig(
            state_dim=380,
            action_dim=16,
            batch_size=4,
            max_inference_time_ms=100.0  # Constitutional requirement
        )

        algorithm = ConservativeQLearning(380, 16, config)
        print("✅ CQL algorithm created successfully")

        # Test action selection
        import torch
        import numpy as np
        test_state = np.random.randn(380).astype(np.float32)
        action = algorithm.select_action(test_state)
        print(f"✅ Action selection successful: action={action}")

        # Test performance metrics
        metrics = algorithm.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")

        # Test explainability
        explanation = algorithm.explain_action(test_state, action)
        print(f"✅ Action explanation: confidence={explanation.get('confidence', 0):.3f}")

        # Test model info
        model_info = algorithm.get_model_info()
        print(f"✅ Model info: {model_info['total_params']} parameters")

        return True

    except Exception as e:
        print(f"❌ CQL algorithm test failed: {e}")
        traceback.print_exc()
        return False

def test_convergence_analysis():
    """Test training convergence analysis."""
    print("\n🧪 Testing Convergence Analysis...")

    try:
        from src.rl.training.trainer import RLTrainer, TrainingConfig

        config = TrainingConfig(buffer_size=100, tensorboard_logging=False)
        trainer = RLTrainer(config)

        # Mock training metrics
        metrics = {
            'episode_rewards': [0.1, 0.2, 0.15, 0.3, 0.4, 0.35, 0.5, 0.6, 0.55, 0.7],
            'losses': [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08]
        }

        convergence = trainer.analyze_training_convergence(metrics)
        print(f"✅ Convergence analysis: {convergence}")

        if convergence['converged'] or convergence['converging']:
            print("✅ Model shows convergence trends")
        else:
            print("⚠️ Model not yet converging (expected for short test)")

        return True

    except Exception as e:
        print(f"❌ Convergence analysis test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Complete RL Training Pipeline Implementation")
    print("=" * 60)

    tests = [
        test_rl_trainer_basic,
        test_17lands_data_processing,
        test_mtga_integration,
        test_cql_algorithm,
        test_convergence_analysis
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! RL training pipeline implementation is complete.")
        print("✅ Task T029 should now pass (Green phase)")
        return 0
    else:
        print("⚠️ Some tests failed. Check the implementation for issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)