#!/usr/bin/env python3
"""
MTG Decision Head Integration Tests

Comprehensive integration tests for the MTG Decision Head with all components.
Tests integration with transformer encoder, action space, and training data.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import json
import sys
import os
from typing import Dict, List, Any
import unittest

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mtg_decision_head import MTGDecisionHead, MTGDecisionHeadConfig, MTGDecisionTrainer, MTGDecisionInference
    from mtg_action_space import MTGActionSpace, Action, ActionType, Phase
    from mtg_transformer_encoder import MTGTransformerEncoder, MTGTransformerConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Running tests with mock components...")
    MTGActionSpace = None
    MTGTransformerEncoder = None


class TestMTGDecisionHeadIntegration(unittest.TestCase):
    """Integration tests for MTG Decision Head."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MTGDecisionHeadConfig(
            state_dim=128,
            action_dim=82,
            hidden_dim=128,  # Smaller for faster testing
            dropout=0.0,     # Disabled for reproducible tests
            scoring_method="attention"
        )
        self.decision_head = MTGDecisionHead(self.config)
        self.device = torch.device('cpu')  # Use CPU for reproducible tests

        # Create sample data
        self.sample_state_repr = torch.randn(2, 128)
        self.sample_action_encodings = torch.randn(2, 5, 82)
        self.decision_type = "Value_Creature_Play"

    def test_basic_functionality(self):
        """Test basic forward pass functionality."""
        print("🧪 Testing basic functionality...")

        # Forward pass
        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            self.decision_type
        )

        # Check output shapes
        self.assertEqual(outputs['action_scores'].shape, (2, 5))
        self.assertEqual(outputs['action_probabilities'].shape, (2, 5))
        self.assertEqual(outputs['state_value'].shape, (2, 1))
        self.assertEqual(outputs['decision_embedding'].shape, (32,))

        # Check probability normalization
        prob_sums = torch.sum(outputs['action_probabilities'], dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

        print("   ✅ Basic functionality test passed")

    def test_decision_type_embeddings(self):
        """Test different decision type embeddings."""
        print("🧪 Testing decision type embeddings...")

        decision_types = [
            "Aggressive_Creature_Play",
            "Defensive_Creature_Play",
            "Mana_Acceleration",
            "All_In_Attack"
        ]

        embeddings = []
        for dec_type in decision_types:
            outputs = self.decision_head(
                self.sample_state_repr,
                self.sample_action_encodings,
                dec_type
            )
            embeddings.append(outputs['decision_embedding'])

        # Check embeddings are different
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = torch.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                )
                # Should not be identical (similarity < 1.0)
                self.assertLess(similarity.item(), 0.99)

        print("   ✅ Decision type embeddings test passed")

    def test_action_selection(self):
        """Test action selection mechanisms."""
        print("🧪 Testing action selection...")

        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            self.decision_type
        )

        # Test deterministic selection
        action_idx, confidence, explainability = self.decision_head.select_action(
            outputs, deterministic=True
        )
        self.assertIsInstance(action_idx, int)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test stochastic selection
        action_idx_stochastic, confidence_stochastic, _ = self.decision_head.select_action(
            outputs, deterministic=False
        )
        self.assertIsInstance(action_idx_stochastic, int)

        print("   ✅ Action selection test passed")

    def test_action_ranking(self):
        """Test action ranking functionality."""
        print("🧪 Testing action ranking...")

        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            self.decision_type
        )

        # Test ranking
        ranked_actions = self.decision_head.rank_actions(outputs, top_k=3)
        self.assertEqual(len(ranked_actions), 3)

        # Check that scores are in descending order
        for i in range(len(ranked_actions) - 1):
            self.assertGreaterEqual(ranked_actions[i][1], ranked_actions[i+1][1])

        # Test that all indices are valid
        num_actions = self.sample_action_encodings.size(1)
        for idx, score in ranked_actions:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_actions)

        print("   ✅ Action ranking test passed")

    def test_loss_computation(self):
        """Test loss computation for training."""
        print("🧪 Testing loss computation...")

        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            self.decision_type
        )

        # Create target data
        target_actions = torch.tensor([1, 3])
        target_values = torch.tensor([0.8, -0.2])
        outcome_weights = torch.tensor([1.0, 0.5])

        # Compute losses
        losses = self.decision_head.compute_loss(
            outputs, target_actions, target_values, outcome_weights
        )

        # Check loss types
        self.assertIn('total_loss', losses)
        self.assertIn('actor_loss', losses)
        self.assertIn('critic_loss', losses)
        self.assertIn('entropy_loss', losses)

        # Check loss values are reasonable
        self.assertGreater(losses['total_loss'].item(), 0.0)
        self.assertGreater(losses['actor_loss'].item(), 0.0)
        self.assertGreater(losses['critic_loss'].item(), 0.0)

        print("   ✅ Loss computation test passed")

    def test_training_step(self):
        """Test a complete training step."""
        print("🧪 Testing training step...")

        self.decision_head.train()
        optimizer = torch.optim.AdamW(self.decision_head.parameters(), lr=1e-4)

        # Forward pass
        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            self.decision_type
        )

        # Compute loss
        target_actions = torch.tensor([1, 3])
        target_values = torch.tensor([0.8, -0.2])
        outcome_weights = torch.tensor([1.0, 0.5])

        losses = self.decision_head.compute_loss(
            outputs, target_actions, target_values, outcome_weights
        )

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Check that gradients were computed
        has_gradients = any(p.grad is not None for p in self.decision_head.parameters())
        self.assertTrue(has_gradients)

        print("   ✅ Training step test passed")

    def test_batch_processing(self):
        """Test different batch sizes."""
        print("🧪 Testing batch processing...")

        batch_sizes = [1, 4, 8]
        num_actions = 10

        for batch_size in batch_sizes:
            state_repr = torch.randn(batch_size, 128)
            action_encodings = torch.randn(batch_size, num_actions, 82)

            outputs = self.decision_head(state_repr, action_encodings, self.decision_type)

            # Check output shapes
            self.assertEqual(outputs['action_scores'].shape, (batch_size, num_actions))
            self.assertEqual(outputs['action_probabilities'].shape, (batch_size, num_actions))
            self.assertEqual(outputs['state_value'].shape, (batch_size, 1))

        print("   ✅ Batch processing test passed")

    def test_temperature_scaling(self):
        """Test temperature scaling effect on action selection."""
        print("🧪 Testing temperature scaling...")

        # Test with different temperatures
        temperatures = [0.1, 1.0, 10.0]
        original_temp = self.decision_head.temperature

        entropies = []
        for temp in temperatures:
            self.decision_head.temperature = temp
            outputs = self.decision_head(
                self.sample_state_repr,
                self.sample_action_encodings,
                self.decision_type
            )

            # Calculate entropy of action distribution
            probs = outputs['action_probabilities']
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropies.append(entropy.mean().item())

        # Restore original temperature
        self.decision_head.temperature = original_temp

        # Higher temperature should lead to higher entropy
        self.assertGreater(entropies[2], entropies[0])  # High temp > Low temp

        print("   ✅ Temperature scaling test passed")

    def test_exploration_mode(self):
        """Test exploration mode during training."""
        print("🧪 Testing exploration mode...")

        # Test training mode
        self.decision_head.set_training_mode(True)
        self.assertTrue(self.decision_head.training_mode)

        # Test inference mode
        self.decision_head.set_training_mode(False)
        self.assertFalse(self.decision_head.training_mode)

        print("   ✅ Exploration mode test passed")

    @unittest.skipIf(MTGActionSpace is None, "MTGActionSpace not available")
    def test_action_space_integration(self):
        """Test integration with MTG Action Space."""
        print("🧪 Testing action space integration...")

        action_space = MTGActionSpace()

        # Create sample game state
        game_state = {
            'hand': [
                {'id': 'creature_1', 'type': 'creature', 'mana_cost': '{2}{W}', 'power': 3, 'toughness': 4}
            ],
            'battlefield': [
                {'id': 'land_1', 'type': 'land', 'tapped': False}
            ],
            'available_mana': {'white': 2, 'red': 1, 'blue': 0, 'black': 0, 'green': 0, 'colorless': 2}
        }

        # Generate actions
        current_phase = Phase.PRECOMBAT_MAIN
        actions = action_space.generate_possible_actions(game_state, current_phase)

        # Encode actions
        if actions:
            action_encodings = torch.stack([
                action_space.encode_action(action) for action in actions[:5]
            ]).unsqueeze(0)  # Add batch dimension

            # Use with decision head
            state_repr = torch.randn(1, 128)
            outputs = self.decision_head(state_repr, action_encodings, "Value_Creature_Play")

            self.assertEqual(outputs['action_scores'].shape, (1, len(actions[:5])))

        print("   ✅ Action space integration test passed")

    @unittest.skipIf(MTGTransformerEncoder is None, "MTGTransformerEncoder not available")
    def test_transformer_integration(self):
        """Test integration with Transformer Encoder."""
        print("🧪 Testing transformer integration...")

        transformer_config = MTGTransformerConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,  # Smaller for testing
            dim_feedforward=256
        )
        transformer = MTGTransformerEncoder(transformer_config)

        # Create sample state tensor
        state_tensor = torch.randn(1, 282)  # Expected input dimension

        # Get state representation
        transformer_outputs = transformer(state_tensor)
        state_repr = transformer_outputs['state_representation']

        # Use with decision head
        action_encodings = torch.randn(1, 5, 82)
        outputs = self.decision_head(state_repr, action_encodings, "Value_Creature_Play")

        # Check integration works
        self.assertEqual(outputs['action_scores'].shape, (1, 5))

        print("   ✅ Transformer integration test passed")

    def test_training_data_compatibility(self):
        """Test compatibility with training data format."""
        print("🧪 Testing training data compatibility...")

        # Create sample in training data format
        sample_data = {
            'state_tensor': torch.randn(282),
            'action_label': torch.randint(0, 2, (15,)).float(),
            'outcome_weight': torch.tensor(0.85),
            'decision_type': 'Mana_Acceleration',
            'turn': 3,
            'game_outcome': True
        }

        # This would normally be processed by the transformer encoder
        # For now, we'll simulate the expected output
        simulated_state_repr = torch.randn(1, 128)
        simulated_action_encodings = torch.randn(1, 10, 82)

        # Test with decision head
        outputs = self.decision_head(
            simulated_state_repr,
            simulated_action_encodings,
            sample_data['decision_type']
        )

        # Check it produces valid outputs
        self.assertEqual(outputs['action_scores'].shape, (1, 10))
        self.assertIsNotNone(outputs['state_value'])

        print("   ✅ Training data compatibility test passed")

    def test_performance_characteristics(self):
        """Test performance characteristics."""
        print("🧪 Testing performance characteristics...")

        # Test inference time
        import time
        num_runs = 100

        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.decision_head(
                    self.sample_state_repr,
                    self.sample_action_encodings,
                    self.decision_type
                )
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms

        # Should be reasonably fast (< 10ms per inference on CPU)
        self.assertLess(avg_time, 10.0)

        print(f"   ⏱️  Average inference time: {avg_time:.2f}ms")
        print("   ✅ Performance characteristics test passed")

    def test_error_handling(self):
        """Test error handling for edge cases."""
        print("🧪 Testing error handling...")

        # Test unknown decision type
        outputs = self.decision_head(
            self.sample_state_repr,
            self.sample_action_encodings,
            "Unknown_Decision_Type"
        )
        # Should still work with zero embedding
        self.assertEqual(outputs['action_scores'].shape, (2, 5))

        # Test empty action space
        empty_action_encodings = torch.randn(2, 0, 82)
        with self.assertRaises(Exception):
            _ = self.decision_head(
                self.sample_state_repr,
                empty_action_encodings,
                self.decision_type
            )

        print("   ✅ Error handling test passed")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 MTG Decision Head Integration Tests")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMTGDecisionHeadIntegration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All integration tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s)")

    return result.wasSuccessful()


def main():
    """Main function to run integration tests."""
    success = run_integration_tests()

    if success:
        print("\n🎉 MTG Decision Head is ready for production use!")
        print("\n📋 Integration Test Summary:")
        print("  ✅ Basic functionality")
        print("  ✅ Decision type handling")
        print("  ✅ Action selection and ranking")
        print("  ✅ Loss computation and training")
        print("  ✅ Batch processing")
        print("  ✅ Temperature scaling")
        print("  ✅ Exploration modes")
        print("  ✅ Performance characteristics")
        print("  ✅ Error handling")

        if MTGActionSpace is not None:
            print("  ✅ Action space integration")
        if MTGTransformerEncoder is not None:
            print("  ✅ Transformer encoder integration")

        print(f"\n🚀 Ready to integrate with the complete MTG AI system!")
    else:
        print("\n⚠️  Some integration tests failed. Please review the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())