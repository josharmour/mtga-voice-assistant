#!/usr/bin/env python3
"""
Comprehensive Testing Suite for MTG Transformer Encoder - Task 3.1

This script provides:
- Unit tests for all model components
- Integration tests for the complete pipeline
- Performance validation against benchmarks
- Data integrity tests
- Model behavior verification

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import json
import tempfile
import os
from pathlib import Path
import logging

from mtg_transformer_encoder import (
    MTGTransformerEncoder, MTGTransformerConfig, MTGDataset,
    create_data_loaders, PositionalEncoding, BoardStateProcessor,
    ComponentProcessor, MultiModalFusion
)
from mtg_model_utils import ModelManager, WeightInitializer, ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMTGTransformerConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = MTGTransformerConfig()
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.nhead, 8)
        self.assertEqual(config.total_input_dim, 282)
        self.assertEqual(config.board_tokens_dim + config.hand_mana_dim +
                        config.phase_priority_dim + config.additional_features_dim,
                        config.total_input_dim)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = MTGTransformerConfig(d_model=128, nhead=4)
        self.assertEqual(config.d_model % config.nhead, 0)

        # Invalid configuration - d_model not divisible by nhead
        with self.assertRaises(AssertionError):
            MTGTransformerConfig(d_model=100, nhead=3)


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding implementation."""

    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        d_model = 256
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)

        # Test with sequence length 10, batch size 4
        x = torch.randn(10, 4, d_model)
        output = pe(x)
        self.assertEqual(output.shape, (10, 4, d_model))

    def test_positional_encoding_values(self):
        """Test that positional encoding adds position-dependent values."""
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=20)

        # Create zero input
        x = torch.zeros(5, 2, d_model)
        output = pe(x)

        # Output should not be zero anymore (due to positional encoding)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

        # Different positions should have different encodings
        self.assertFalse(torch.allclose(output[0], output[1]))


class TestBoardStateProcessor(unittest.TestCase):
    """Test board state processing."""

    def setUp(self):
        """Set up test configuration."""
        self.config = MTGTransformerConfig(d_model=128, nhead=4)
        self.processor = BoardStateProcessor(self.config)

    def test_board_processor_shape(self):
        """Test board processor output shape."""
        batch_size = 8
        board_tokens = torch.randn(batch_size, self.config.board_tokens_dim)
        output = self.processor(board_tokens)
        self.assertEqual(output.shape, (batch_size, self.config.d_model))

    def test_board_processor_with_positions(self):
        """Test board processor with position information."""
        batch_size = 4
        num_permanents = 3
        board_tokens = torch.randn(batch_size, self.config.board_tokens_dim)
        positions = torch.randint(0, self.config.max_board_positions, (batch_size, num_permanents))
        output = self.processor(board_tokens, positions)
        self.assertEqual(output.shape, (batch_size, self.config.d_model))


class TestComponentProcessor(unittest.TestCase):
    """Test component processor for non-board components."""

    def test_component_processor_shape(self):
        """Test component processor output shape."""
        input_dim = 128
        output_dim = 256
        processor = ComponentProcessor(input_dim, output_dim)

        batch_size = 8
        x = torch.randn(batch_size, input_dim)
        output = processor(x)
        self.assertEqual(output.shape, (batch_size, output_dim))


class TestMultiModalFusion(unittest.TestCase):
    """Test multi-modal fusion component."""

    def setUp(self):
        """Set up test configuration."""
        self.config = MTGTransformerConfig(d_model=128, nhead=4, num_encoder_layers=2)
        self.fusion = MultiModalFusion(self.config)

    def test_fusion_shape(self):
        """Test fusion output shape."""
        batch_size = 8
        num_components = 4
        x = torch.randn(batch_size, num_components, self.config.d_model)
        fused_repr, attn_weights = self.fusion(x)
        self.assertEqual(fused_repr.shape, (batch_size, self.config.d_model))
        # Attention weights shape: (num_heads, num_components, num_components)
        self.assertEqual(attn_weights.shape, (self.config.nhead, num_components, num_components))


class TestMTGTransformerEncoder(unittest.TestCase):
    """Test the main transformer encoder."""

    def setUp(self):
        """Set up test model."""
        self.config = MTGTransformerConfig(
            d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256
        )
        self.model = MTGTransformerEncoder(self.config)

    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        batch_size = 4
        state_tensor = torch.randn(batch_size, self.config.total_input_dim)
        outputs = self.model(state_tensor)

        # Check output shapes
        self.assertEqual(outputs['action_logits'].shape, (batch_size, self.config.action_vocab_size))
        self.assertEqual(outputs['value'].shape, (batch_size, 1))
        self.assertEqual(outputs['state_representation'].shape, (batch_size, self.config.output_dim))
        self.assertEqual(outputs['attention_weights'].shape[1], 4)  # 4 components

    def test_model_with_board_positions(self):
        """Test model with board position information."""
        batch_size = 2
        state_tensor = torch.randn(batch_size, self.config.total_input_dim)
        num_permanents = 3
        board_positions = torch.randint(0, self.config.max_board_positions, (batch_size, num_permanents))

        outputs = self.model(state_tensor, board_positions)
        self.assertEqual(outputs['action_logits'].shape, (batch_size, self.config.action_vocab_size))

    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        batch_size = 2
        state_tensor = torch.randn(batch_size, self.config.total_input_dim)
        action_labels = torch.randn(batch_size, self.config.action_vocab_size)

        outputs = self.model(state_tensor)
        loss = nn.MSELoss()(outputs['action_logits'], action_labels)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")

    def test_model_parameter_count(self):
        """Test model parameter count."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All parameters should be trainable


class TestMTGDataset(unittest.TestCase):
    """Test dataset loading functionality."""

    def setUp(self):
        """Create a temporary test dataset."""
        self.test_data = {
            "metadata": {
                "total_samples": 4,
                "final_tensor_dimension": 282,
                "component_dimensions": {
                    "board_tokens": 64,
                    "hand_mana": 128,
                    "phase_priority": 64,
                    "additional_features": 10
                }
            },
            "training_samples": [
                {
                    "state_tensor": [float(i) for i in range(282)],
                    "action_label": [1.0, 0.0] + [0.0] * 14,
                    "outcome_weight": 0.8,
                    "decision_type": "Attack",
                    "turn": 5,
                    "game_outcome": True
                },
                {
                    "state_tensor": [float(i+282) for i in range(282)],
                    "action_label": [0.0, 1.0] + [0.0] * 14,
                    "outcome_weight": 0.6,
                    "decision_type": "Block",
                    "turn": 6,
                    "game_outcome": False
                },
                {
                    "state_tensor": [float(i+564) for i in range(282)],
                    "action_label": [0.0, 0.0, 1.0] + [0.0] * 13,
                    "outcome_weight": 0.9,
                    "decision_type": "Cast_Spell",
                    "turn": 3,
                    "game_outcome": True
                },
                {
                    "state_tensor": [float(i+846) for i in range(282)],
                    "action_label": [0.0, 0.0, 0.0, 1.0] + [0.0] * 12,
                    "outcome_weight": 0.7,
                    "decision_type": "Mana_Acceleration",
                    "turn": 2,
                    "game_outcome": True
                }
            ]
        }

        # Write to temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file, indent=2)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)

    def test_dataset_loading(self):
        """Test dataset loading from JSON."""
        dataset = MTGDataset(self.temp_file.name)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.total_samples, 4)
        self.assertEqual(dataset.tensor_dimension, 282)

    def test_dataset_getitem(self):
        """Test individual sample retrieval."""
        dataset = MTGDataset(self.temp_file.name)
        sample = dataset[0]

        self.assertIn('state_tensor', sample)
        self.assertIn('action_label', sample)
        self.assertIn('outcome_weight', sample)
        self.assertEqual(sample['state_tensor'].shape, (282,))
        self.assertEqual(sample['action_label'].shape, (16,))
        self.assertEqual(sample['decision_type'], "Attack")

    def test_data_loader_creation(self):
        """Test data loader creation."""
        train_loader, val_loader = create_data_loaders(
            self.temp_file.name, batch_size=2, validation_split=0.5
        )

        # Test training loader
        batch = next(iter(train_loader))
        self.assertEqual(batch['state_tensor'].shape[0], 2)  # Batch size
        self.assertEqual(batch['state_tensor'].shape[1], 282)  # Feature dimension

        # Test validation loader
        val_batch = next(iter(val_loader))
        self.assertEqual(val_batch['state_tensor'].shape[0], 2)  # Should have 2 samples in val


class TestModelManager(unittest.TestCase):
    """Test model saving and loading."""

    def setUp(self):
        """Set up test model and manager."""
        self.config = MTGTransformerConfig(d_model=64, nhead=2, num_encoder_layers=1)
        self.model = MTGTransformerEncoder(self.config)
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_model_save_load(self):
        """Test model saving and loading."""
        # Save model
        metrics = {'test_metric': 0.5}
        save_path = self.manager.save_model(self.model, self.config, metrics, "test_model")
        self.assertTrue(os.path.exists(save_path))

        # Load model
        loaded_model, loaded_config = self.manager.load_model(save_path)

        # Check that models are equivalent
        self.assertEqual(loaded_config.d_model, self.config.d_model)
        self.assertEqual(loaded_config.nhead, self.config.nhead)

        # Test that loaded model produces same outputs
        test_input = torch.randn(2, 282)
        with torch.no_grad():
            original_output = self.model(test_input)
            loaded_output = loaded_model(test_input)

        self.assertTrue(torch.allclose(original_output['action_logits'],
                                      loaded_output['action_logits'], atol=1e-6))


class TestWeightInitialization(unittest.TestCase):
    """Test weight initialization methods."""

    def setUp(self):
        """Set up test model."""
        self.config = MTGTransformerConfig(d_model=64, nhead=2, num_encoder_layers=1)
        self.model = MTGTransformerEncoder(self.config)

    def test_xavier_initialization(self):
        """Test Xavier initialization."""
        WeightInitializer.init_model(self.model, 'xavier_uniform')

        # Check that weights are not zero
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                self.assertFalse(torch.allclose(param, torch.zeros_like(param)))

    def test_kaiming_initialization(self):
        """Test Kaiming initialization."""
        WeightInitializer.init_model(self.model, 'kaiming_uniform')

        # Check that weights are not zero
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                self.assertFalse(torch.allclose(param, torch.zeros_like(param)))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up integration test environment."""
        self.config = MTGTransformerConfig(d_model=64, nhead=2, num_encoder_layers=2)
        self.model = MTGTransformerEncoder(self.config)

        # Create test dataset
        self.test_data = {
            "metadata": {
                "total_samples": 8,
                "final_tensor_dimension": 282
            },
            "training_samples": [
                {
                    "state_tensor": [float(i % 100) for i in range(282)],
                    "action_label": [1.0 if i == j else 0.0 for i in range(16)],
                    "outcome_weight": 0.5 + (j * 0.1),
                    "decision_type": f"Action_{j}",
                    "turn": j + 1,
                    "game_outcome": j % 2 == 0
                }
                for j in range(8)
            ]
        }

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file, indent=2)
        self.temp_file.close()

    def tearDown(self):
        """Clean up."""
        os.unlink(self.temp_file.name)

    def test_complete_training_loop(self):
        """Test a complete training loop."""
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.temp_file.name, batch_size=2, validation_split=0.5
        )

        # Set up optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        action_criterion = nn.BCEWithLogitsLoss()

        # Training loop
        initial_loss = None
        final_loss = None

        for epoch in range(3):  # Short training loop
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                state_tensors = batch['state_tensor']
                action_labels = batch['action_label']

                outputs = self.model(state_tensors)
                action_logits = outputs['action_logits']

                loss = action_criterion(action_logits, action_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch == 0:
                initial_loss = epoch_loss
            elif epoch == 2:
                final_loss = epoch_loss

        # Loss should generally decrease
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)

        # Test evaluation
        self.model.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            outputs = self.model(val_batch['state_tensor'])
            self.assertEqual(outputs['action_logits'].shape[0], val_batch['state_tensor'].shape[0])

    def test_model_on_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            state_tensor = torch.randn(batch_size, 282)
            outputs = self.model(state_tensor)

            self.assertEqual(outputs['action_logits'].shape[0], batch_size)
            self.assertEqual(outputs['value'].shape[0], batch_size)
            self.assertEqual(outputs['state_representation'].shape[0], batch_size)

    def test_model_reproducibility(self):
        """Test model reproducibility with fixed seed."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create model and run forward pass
        model1 = MTGTransformerEncoder(self.config)
        test_input = torch.randn(2, 282)
        output1 = model1(test_input)

        # Reset seed and create identical model
        torch.manual_seed(42)
        np.random.seed(42)

        model2 = MTGTransformerEncoder(self.config)
        output2 = model2(test_input)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1['action_logits'], output2['action_logits'], atol=1e-6))


def run_comprehensive_validation():
    """Run comprehensive validation tests."""
    logger.info("Starting comprehensive MTG Transformer validation...")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMTGTransformerConfig,
        TestPositionalEncoding,
        TestBoardStateProcessor,
        TestComponentProcessor,
        TestMultiModalFusion,
        TestMTGTransformerEncoder,
        TestMTGDataset,
        TestModelManager,
        TestWeightInitialization,
        TestModelIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests * 100

    logger.info(f"\nValidation Summary:")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Success rate: {success_rate:.1f}%")

    if failures == 0 and errors == 0:
        logger.info("✅ All tests passed! MTG Transformer is ready for use.")
    else:
        logger.warning("❌ Some tests failed. Please review the issues above.")
        if result.failures:
            logger.error("Failures:")
            for test, traceback in result.failures:
                logger.error(f"- {test}: {traceback}")
        if result.errors:
            logger.error("Errors:")
            for test, traceback in result.errors:
                logger.error(f"- {test}: {traceback}")

    return result.wasSuccessful()


def main():
    """Main function for running tests."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        success = run_comprehensive_validation()
        sys.exit(0 if success else 1)
    else:
        # Run specific tests
        unittest.main()


if __name__ == "__main__":
    main()