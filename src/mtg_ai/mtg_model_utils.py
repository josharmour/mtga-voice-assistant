#!/usr/bin/env python3
"""
MTG Model Utilities - Supporting utilities for the MTG Transformer Encoder

This module provides:
- Model configuration management
- Weight initialization utilities
- Model saving/loading functionality
- Performance evaluation metrics
- Visualization utilities for attention weights
- Model architecture analysis tools

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from collections import defaultdict

from .mtg_transformer_encoder import MTGTransformerEncoder, MTGTransformerConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model saving, loading, and versioning."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            model_dir: Directory to store model checkpoints
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def save_model(self, model: MTGTransformerEncoder, config: MTGTransformerConfig,
                   metrics: Optional[Dict[str, Any]] = None,
                   model_name: str = "mtg_transformer",
                   epoch: Optional[int] = None) -> str:
        """
        Save model with configuration and metrics.

        Args:
            model: Model to save
            config: Model configuration
            metrics: Training metrics
            model_name: Base name for the model
            epoch: Training epoch (for checkpoint naming)

        Returns:
            Path to saved model
        """
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pth"
        else:
            filename = f"{model_name}_final.pth"

        checkpoint_path = self.checkpoints_dir / filename

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'metrics': metrics or {},
            'model_architecture': str(model),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        # Also save config separately for easy loading
        config_path = self.model_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)

        return str(checkpoint_path)

    def load_model(self, model_path: str, device: str = 'cpu') -> Tuple[MTGTransformerEncoder, MTGTransformerConfig]:
        """
        Load model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            device: Device to load model on

        Returns:
            Tuple of (model, config)
        """
        checkpoint = torch.load(model_path, map_location=device)

        # Reconstruct config
        config = MTGTransformerConfig(**checkpoint['config'])

        # Create model
        model = MTGTransformerEncoder(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model has {checkpoint['total_parameters']} total parameters")

        return model, config

    def list_checkpoints(self, model_name: str = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            model_name: Filter by model name

        Returns:
            List of checkpoint information
        """
        checkpoints = []

        for checkpoint_file in self.checkpoints_dir.glob("*.pth"):
            if model_name and not checkpoint_file.name.startswith(model_name):
                continue

            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'filename': checkpoint_file.name,
                    'total_parameters': checkpoint.get('total_parameters', 0),
                    'metrics': checkpoint.get('metrics', {})
                })
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x['filename'])


class WeightInitializer:
    """Utilities for advanced weight initialization."""

    @staticmethod
    def xavier_uniform_init(module: nn.Module, gain: float = 1.0):
        """Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def xavier_normal_init(module: nn.Module, gain: float = 1.0):
        """Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def kaiming_uniform_init(module: nn.Module, nonlinearity: str = 'relu'):
        """Kaiming uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def kaiming_normal_init(module: nn.Module, nonlinearity: str = 'relu'):
        """Kaiming normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def init_model(model: MTGTransformerEncoder, init_type: str = 'xavier_uniform'):
        """
        Initialize model weights.

        Args:
            model: Model to initialize
            init_type: Type of initialization
        """
        init_functions = {
            'xavier_uniform': WeightInitializer.xavier_uniform_init,
            'xavier_normal': WeightInitializer.xavier_normal_init,
            'kaiming_uniform': WeightInitializer.kaiming_uniform_init,
            'kaiming_normal': WeightInitializer.kaiming_normal_init
        }

        if init_type not in init_functions:
            raise ValueError(f"Unknown initialization type: {init_type}")

        init_fn = init_functions[init_type]
        model.apply(init_fn)

        # Special initialization for embeddings
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

        logger.info(f"Model initialized with {init_type} initialization")


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""

    def __init__(self, model: MTGTransformerEncoder, device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device for evaluation
        """
        self.model = model
        self.device = device

    def evaluate_model(self, data_loader, criterion_action=None, criterion_value=None) -> Dict[str, float]:
        """
        Comprehensive model evaluation.

        Args:
            data_loader: Data loader for evaluation
            criterion_action: Action loss criterion
            criterion_value: Value loss criterion

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        if criterion_action is None:
            criterion_action = nn.BCEWithLogitsLoss(reduction='none')
        if criterion_value is None:
            criterion_value = nn.MSELoss()

        metrics = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_values = []
        all_target_values = []

        with torch.no_grad():
            for batch in data_loader:
                state_tensors = batch['state_tensor'].to(self.device)
                action_labels = batch['action_label'].to(self.device)
                outcome_weights = batch['outcome_weight'].to(self.device)

                outputs = self.model(state_tensors)
                action_logits = outputs['action_logits']
                value_pred = outputs['value']
                value_target = outcome_weights.unsqueeze(1)

                # Compute losses
                action_loss = criterion_action(action_logits, action_labels)
                action_loss = (action_loss * outcome_weights.unsqueeze(1)).mean()
                value_loss = criterion_value(value_pred, value_target)

                # Store metrics
                metrics['action_loss'].append(action_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['total_loss'].append((action_loss + 0.5 * value_loss).item())

                # Compute accuracy
                predicted_actions = (torch.sigmoid(action_logits) > 0.5).float()
                batch_accuracy = (predicted_actions == action_labels).float().mean().item()
                metrics['accuracy'].append(batch_accuracy)

                # Store predictions for detailed analysis
                all_predictions.append(torch.sigmoid(action_logits).cpu())
                all_labels.append(action_labels.cpu())
                all_values.append(value_pred.cpu())
                all_target_values.append(value_target.cpu())

        # Aggregate metrics
        results = {}
        for key, values in metrics.items():
            results[key] = np.mean(values)
            results[f'{key}_std'] = np.std(values)

        # Compute additional metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_values = torch.cat(all_values)
        all_target_values = torch.cat(all_target_values)

        # Top-k accuracy
        for k in [1, 3, 5]:
            if all_predictions.shape[1] >= k:
                topk_pred = all_predictions.topk(k, dim=1).indices
                topk_labels = all_labels.topk(k, dim=1).indices
                topk_acc = (topk_pred == topk_labels).any(dim=1).float().mean().item()
                results[f'top_{k}_accuracy'] = topk_acc

        # Value prediction metrics
        results['value_mae'] = torch.mean(torch.abs(all_values - all_target_values)).item()
        results['value_rmse'] = torch.sqrt(torch.mean((all_values - all_target_values) ** 2)).item()
        results['value_r2'] = self._compute_r2(all_values, all_target_values)

        return dict(results)

    def _compute_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute R-squared score."""
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def analyze_attention_patterns(self, data_loader, num_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze attention patterns for interpretability.

        Args:
            data_loader: Data loader for samples
            num_samples: Number of samples to analyze

        Returns:
            Attention analysis results
        """
        self.model.eval()
        attention_patterns = []
        component_names = ['Board', 'Hand/Mana', 'Phase/Priority', 'Additional Features']

        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                state_tensors = batch['state_tensor'].to(self.device)
                outputs = self.model(state_tensors)
                attention_weights = outputs['attention_weights']

                # Average attention across heads and layers
                avg_attention = attention_weights.mean(dim=(0, 1))  # (num_components, num_components)

                attention_patterns.append(avg_attention.cpu().numpy())
                sample_count += 1

        # Aggregate attention patterns
        attention_patterns = np.array(attention_patterns)
        mean_attention = np.mean(attention_patterns, axis=0)
        std_attention = np.std(attention_patterns, axis=0)

        return {
            'mean_attention': mean_attention,
            'std_attention': std_attention,
            'component_names': component_names,
            'attention_patterns': attention_patterns
        }


class AttentionVisualizer:
    """Visualization utilities for attention weights."""

    @staticmethod
    def plot_attention_heatmap(attention_matrix: np.ndarray, component_names: List[str],
                              title: str = "Attention Weights", save_path: str = None) -> plt.Figure:
        """
        Plot attention heatmap.

        Args:
            attention_matrix: Attention weight matrix
            component_names: Names of components
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attention_matrix, xticklabels=component_names, yticklabels=component_names,
                   annot=True, cmap='Blues', ax=ax, fmt='.3f')
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_training_metrics(metrics: Dict[str, List[float]], save_path: str = None) -> plt.Figure:
        """
        Plot training metrics over epochs.

        Args:
            metrics: Dictionary of training metrics
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(metrics['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(metrics['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Loss difference
        loss_diff = np.array(metrics['val_loss']) - np.array(metrics['train_loss'])
        axes[1, 0].plot(loss_diff)
        axes[1, 0].set_title('Validation - Train Loss (Overfitting Indicator)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True)

        # Accuracy difference
        acc_diff = np.array(metrics['train_accuracy']) - np.array(metrics['val_accuracy'])
        axes[1, 1].plot(acc_diff)
        axes[1, 1].set_title('Train - Val Accuracy (Overfitting Indicator)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ModelProfiler:
    """Model profiling and analysis utilities."""

    @staticmethod
    def profile_model(model: MTGTransformerEncoder, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Profile model performance and memory usage.

        Args:
            model: Model to profile
            input_shape: Input tensor shape (batch_size, features)

        Returns:
            Profile results
        """
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB

        # Forward pass timing
        model.eval()
        with torch.no_grad():
            sample_input = torch.randn(input_shape)

            # Warm up
            for _ in range(10):
                _ = model(sample_input)

            # Time forward pass
            import time
            start_time = time.time()
            for _ in range(100):
                _ = model(sample_input)
            end_time = time.time()

            avg_forward_time = (end_time - start_time) / 100 * 1000  # ms

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory,
            'avg_forward_time_ms': avg_forward_time,
            'model_size_mb': param_memory,
            'flops_estimate': ModelProfiler._estimate_flops(model, input_shape)
        }

    @staticmethod
    def _estimate_flops(model: MTGTransformerEncoder, input_shape: Tuple[int, ...]) -> int:
        """
        Rough estimate of FLOPs for the model.

        Args:
            model: Model to analyze
            input_shape: Input tensor shape

        Returns:
            Estimated FLOPs
        """
        # This is a rough estimate - actual FLOPs would need more detailed analysis
        batch_size, features = input_shape
        config = model.config

        # Linear layers: input_features * output_features * batch_size
        flops = 0

        # Component processing layers
        flops += features * config.d_model * batch_size  # Board processor
        flops += config.hand_mana_dim * config.d_model * batch_size
        flops += config.phase_priority_dim * config.d_model * batch_size
        flops += config.additional_features_dim * config.d_model * batch_size

        # Multi-head attention (simplified)
        num_components = 4
        seq_len = num_components
        attention_flops = 2 * seq_len * seq_len * config.d_model * batch_size
        ffn_flops = 2 * seq_len * config.d_model * config.dim_feedforward * batch_size
        transformer_flops = (attention_flops + ffn_flops) * config.num_encoder_layers
        flops += transformer_flops

        # Output heads
        flops += config.d_model * config.dim_feedforward * batch_size * 2  # Two output heads

        return int(flops)


def create_standard_configs() -> Dict[str, MTGTransformerConfig]:
    """Create standard model configurations for different use cases."""
    configs = {
        'small': MTGTransformerConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            dim_feedforward=256,
            dropout=0.1
        ),
        'medium': MTGTransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=512,
            dropout=0.1
        ),
        'large': MTGTransformerConfig(
            d_model=512,
            nhead=16,
            num_encoder_layers=8,
            dim_feedforward=1024,
            dropout=0.1
        ),
        'tiny': MTGTransformerConfig(
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=128,
            dropout=0.2
        )
    }
    return configs


def main():
    """Main function for testing utilities."""
    logger.info("Testing MTG Model Utilities")

    # Create a model
    config = MTGTransformerConfig(d_model=256, nhead=8, num_encoder_layers=6)
    model = MTGTransformerEncoder(config)

    # Test model manager
    manager = ModelManager("test_models")
    save_path = manager.save_model(model, config, {}, "test_model")
    logger.info(f"Model saved to: {save_path}")

    # Test model loading
    loaded_model, loaded_config = manager.load_model(save_path)
    logger.info("Model loaded successfully")

    # Test weight initialization
    WeightInitializer.init_model(model, 'xavier_uniform')
    logger.info("Model reinitialized with Xavier uniform")

    # Test profiling
    profile = ModelProfiler.profile_model(model, (4, 282))
    logger.info(f"Model profile: {profile}")

    # Test visualization (without showing plots)
    dummy_metrics = {
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7],
        'train_accuracy': [0.5, 0.6, 0.7],
        'val_accuracy': [0.4, 0.5, 0.6]
    }
    fig = AttentionVisualizer.plot_training_metrics(dummy_metrics)
    plt.close(fig)
    logger.info("Visualization test completed")

    logger.info("MTG Model Utilities test completed successfully!")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()