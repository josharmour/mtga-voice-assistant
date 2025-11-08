#!/usr/bin/env python3
"""
MTG Training Pipeline - Task 3.4: Loss Function and Training Setup

Complete training pipeline for Magic: The Gathering AI with outcome-weighted loss functions,
multi-task learning, curriculum learning, and comprehensive evaluation metrics.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import json
import math
import os
import sys
import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import itertools
from pathlib import Path

# Import existing components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mtg_transformer_encoder import MTGTransformerEncoder, MTGTransformerConfig
    from mtg_action_space import MTGActionSpace, ActionType, Phase
except ImportError as e:
    logging.warning(f"Could not import some components: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mtg_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Model architecture
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0

    # Loss function weights
    action_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    outcome_weighted_loss: bool = True

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: [
        "basic_actions", "strategic_decisions", "complex_combat", "advanced_tactics"
    ])
    stage_transitions: Dict[str, int] = field(default_factory=lambda: {
        "basic_actions": 10,
        "strategic_decisions": 25,
        "complex_combat": 50,
        "advanced_tactics": 100
    })

    # Validation and early stopping
    validation_split: float = 0.2
    patience: int = 10
    min_delta: float = 1e-4

    # Hardware and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    keep_best: bool = True

    # Monitoring
    log_every: int = 10
    evaluate_every: int = 50
    use_tensorboard: bool = True


class MTGOutcomeWeightedLoss(nn.Module):
    """
    Outcome-weighted loss function for MTG AI training.

    Incorporates:
    - Game result importance weighting
    - Strategic decision importance
    - Class imbalance handling
    - Multi-task learning (action classification + value estimation)
    """

    def __init__(self,
                 action_loss_weight: float = 1.0,
                 value_loss_weight: float = 0.5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()

        self.action_loss_weight = action_loss_weight
        self.value_loss_weight = value_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # Class weights for handling imbalance (computed from data)
        self.register_buffer('class_weights', torch.ones(16))

    def compute_class_weights(self, dataset):
        """Compute class weights from dataset to handle imbalance."""
        action_counts = torch.zeros(16)
        total_samples = len(dataset)

        for sample in dataset:
            if 'action_label' in sample:
                action_tensor = torch.tensor(sample['action_label'])
                action_idx = torch.argmax(action_tensor).item()
                action_counts[action_idx] += 1

        # Compute inverse frequency weights
        class_weights = total_samples / (action_counts + 1e-6)
        class_weights = class_weights / class_weights.mean()

        self.class_weights = class_weights
        logger.info(f"Computed class weights: {class_weights}")

    def focal_loss(self, logits, targets, weights=None):
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Compute focal loss parameters
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        if weights is not None:
            focal_loss = focal_loss * weights

        return focal_loss.mean()

    def outcome_weighted_action_loss(self, logits, targets, outcome_weights):
        """
        Compute action classification loss with outcome weighting.

        Args:
            logits: Model predictions (batch_size, num_actions)
            targets: True actions (batch_size,)
            outcome_weights: Importance weights for each sample (batch_size,)
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = self._smooth_labels(targets, logits.size(1))

        # Compute focal loss with class weights
        class_weights = self.class_weights.to(logits.device)
        weighted_loss = self.focal_loss(logits, targets, class_weights)

        # Apply outcome weighting
        if outcome_weights is not None:
            outcome_weights = outcome_weights.to(logits.device)
            weighted_loss = weighted_loss * outcome_weights.mean()

        return self.action_loss_weight * weighted_loss

    def _smooth_labels(self, targets, num_classes):
        """Apply label smoothing to targets."""
        smoothed_targets = torch.zeros_like(targets).float()
        smoothed_targets.fill_(self.label_smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return smoothed_targets

    def value_loss(self, predicted_values, target_values, outcome_weights=None):
        """
        Compute value estimation loss (e.g., win probability).

        Args:
            predicted_values: Predicted game values (batch_size, 1)
            target_values: True game values (batch_size, 1)
            outcome_weights: Importance weights (batch_size,)
        """
        # Use MSE loss for value prediction
        value_loss = F.mse_loss(predicted_values, target_values, reduction='none')

        if outcome_weights is not None:
            outcome_weights = outcome_weights.to(predicted_values.device)
            value_loss = value_loss * outcome_weights.unsqueeze(1)

        return self.value_loss_weight * value_loss.mean()

    def forward(self, action_logits, value_logits, action_targets, value_targets, outcome_weights=None):
        """
        Compute total multi-task loss.

        Args:
            action_logits: Action classification predictions
            value_logits: Value estimation predictions
            action_targets: True action labels
            value_targets: True game values
            outcome_weights: Sample importance weights

        Returns:
            Dictionary of losses
        """
        # Convert targets to proper format
        if isinstance(action_targets, list):
            action_targets = torch.tensor(action_targets)
        if len(action_targets.shape) > 1:
            action_targets = torch.argmax(action_targets, dim=1)

        # Compute individual losses
        action_loss = self.outcome_weighted_action_loss(
            action_logits, action_targets, outcome_weights
        )

        value_loss = self.value_loss(
            value_logits, value_targets, outcome_weights
        )

        # Total loss
        total_loss = action_loss + value_loss

        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'value_loss': value_loss
        }


class MTGGameplayDecisionHead(nn.Module):
    """
    Gameplay decision head that combines state encoder with action scoring.

    This module takes encoded game states and produces:
    - Action probability distribution
    - Value estimation (win probability, board advantage, etc.)
    - Strategic importance scores
    """

    def __init__(self, config: MTGTransformerConfig):
        super().__init__()

        self.config = config
        self.action_vocab_size = config.action_vocab_size

        # Shared state processing layers
        self.state_processor = nn.Sequential(
            nn.Linear(config.output_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, self.action_vocab_size)
        )

        # Value estimation head (multiple value predictions)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3)  # win_prob, board_advantage, card_advantage
        )

        # Strategic importance scoring
        self.importance_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )

        # Attention mechanism for focusing on relevant game state components
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead // 2,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(self, encoded_states, attention_mask=None):
        """
        Forward pass through decision head.

        Args:
            encoded_states: Encoded game states (batch_size, seq_len, d_model)
            attention_mask: Attention mask for transformer

        Returns:
            Dictionary of predictions
        """
        # Process encoded states
        batch_size = encoded_states.size(0)

        # Apply attention to focus on relevant state components
        if attention_mask is not None:
            attended_states, _ = self.attention(
                encoded_states, encoded_states, encoded_states,
                key_padding_mask=~attention_mask.bool()
            )
        else:
            attended_states, _ = self.attention(
                encoded_states, encoded_states, encoded_states
            )

        # Global pooling (use mean for now, could be attention-weighted)
        pooled_states = attended_states.mean(dim=1)  # (batch_size, d_model)

        # Process through shared layers
        processed_states = self.state_processor(pooled_states)

        # Generate predictions
        action_logits = self.action_head(processed_states)
        values = self.value_head(processed_states)
        importance_scores = self.importance_head(processed_states)

        # Apply softmax to action logits for probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Apply sigmoid to value predictions for probabilities
        win_prob = torch.sigmoid(values[:, 0])
        board_advantage = torch.tanh(values[:, 1])  # Bounded between -1 and 1
        card_advantage = torch.tanh(values[:, 2])

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'win_probability': win_prob,
            'board_advantage': board_advantage,
            'card_advantage': card_advantage,
            'importance_scores': importance_scores,
            'processed_states': processed_states
        }


class MTGTrainingDataset(Dataset):
    """Dataset class for MTG training data."""

    def __init__(self, data_path: str, curriculum_stage: str = "all"):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON training data
            curriculum_stage: Current curriculum stage for filtering
        """
        self.data_path = data_path
        self.curriculum_stage = curriculum_stage
        self.data = []
        self._load_data()

    def _load_data(self):
        """Load training data from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)

            # Extract training samples
            if 'samples' in data:
                samples = data['samples']
            else:
                # If no 'samples' key, assume the whole file contains samples
                samples = data

            # Filter by curriculum stage
            if self.curriculum_stage != "all":
                samples = self._filter_by_curriculum_stage(samples)

            self.data = samples
            logger.info(f"Loaded {len(self.data)} training samples for stage '{self.curriculum_stage}'")

        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise

    def _filter_by_curriculum_stage(self, samples):
        """Filter samples based on curriculum stage."""
        filtered = []

        # Define difficulty levels for different decision types
        stage_difficulty = {
            "basic_actions": ["Mana_Acceleration", "Land_Play", "Basic_Combat"],
            "strategic_decisions": ["Card_Playing", "Removal_Usage", "Counter_Play"],
            "complex_combat": ["Complex_Combat", "Multi_Target_Combat", "Block_Decision"],
            "advanced_tactics": ["Combo_Play", "Bluff_Action", "Resource_Optimization", "Strategic_Sequencing"]
        }

        if self.curriculum_stage not in stage_difficulty:
            return samples

        allowed_types = set(stage_difficulty[self.curriculum_stage])

        for sample in samples:
            if 'decision_type' in sample and sample['decision_type'] in allowed_types:
                filtered.append(sample)

        return filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single training sample."""
        sample = self.data[idx]

        # Extract tensor data (handle different data formats)
        if 'tensor_data' in sample:
            state_tensor = torch.tensor(sample['tensor_data'], dtype=torch.float32)
        elif 'state_tensor' in sample:
            state_tensor = torch.tensor(sample['state_tensor'], dtype=torch.float32)
        else:
            # Fallback: create dummy tensor if no state data
            state_tensor = torch.zeros(282, dtype=torch.float32)

        # Extract action labels
        if 'action_label' in sample:
            action_label = torch.tensor(sample['action_label'], dtype=torch.long)
            action_target = torch.argmax(action_label).item()
        else:
            action_target = 0
            action_label = torch.zeros(16, dtype=torch.long)
            action_label[0] = 1

        # Extract outcome weight
        outcome_weight = sample.get('outcome_weight', 1.0)

        # Extract game outcome for value target
        game_outcome = sample.get('game_outcome', 0.0)
        if isinstance(game_outcome, bool):
            game_outcome = 1.0 if game_outcome else 0.0

        # Extract strategic context for additional value targets
        strategic_context = sample.get('strategic_context', {})
        board_advantage = strategic_context.get('board_advantage', 0.0) / 10.0  # Normalize
        card_advantage = strategic_context.get('card_advantage', 0.0) / 10.0   # Normalize

        value_target = torch.tensor([game_outcome, board_advantage, card_advantage], dtype=torch.float32)

        return {
            'state_tensor': state_tensor,
            'action_target': action_target,
            'action_label': action_label,
            'value_target': value_target,
            'outcome_weight': outcome_weight,
            'decision_type': sample.get('decision_type', 'unknown'),
            'turn': sample.get('turn', 0),
            'game_outcome': game_outcome
        }


class MTGCurriculumScheduler:
    """Curriculum learning scheduler for progressive training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage_idx = 0
        self.stages = config.curriculum_stages
        self.stage_transitions = config.stage_transitions
        self.current_epoch = 0

    def get_current_stage(self):
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def should_advance(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Determine if curriculum should advance to next stage.

        Args:
            epoch: Current epoch number
            metrics: Current training metrics

        Returns:
            True if should advance to next stage
        """
        if not self.config.curriculum_enabled:
            return False

        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at final stage

        current_stage = self.get_current_stage()
        required_epochs = self.stage_transitions.get(current_stage, 50)

        # Check epoch requirement
        if epoch < required_epochs:
            return False

        # Check performance requirement (if metrics available)
        if 'accuracy' in metrics:
            min_accuracy = 0.7 + (self.current_stage_idx * 0.05)  # Increasing requirements
            if metrics['accuracy'] < min_accuracy:
                return False

        return True

    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            new_stage = self.get_current_stage()
            logger.info(f"Advanced to curriculum stage: {new_stage}")
            return True
        return False


class MTGTrainer:
    """Main training class for MTG AI."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model components
        self._init_model()

        # Initialize loss function and optimizer
        self._init_training_components()

        # Initialize curriculum scheduler
        self.curriculum_scheduler = MTGCurriculumScheduler(config)

        # Training state
        self.current_epoch = 0
        self.best_validation_loss = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _init_model(self):
        """Initialize model components."""
        # Create transformer config
        transformer_config = MTGTransformerConfig(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout
        )

        # Initialize encoder and decision head
        self.state_encoder = MTGTransformerEncoder(transformer_config)
        self.decision_head = MTGGameplayDecisionHead(transformer_config)

        # Move to device
        self.state_encoder.to(self.device)
        self.decision_head.to(self.device)

        # Combine parameters for optimizer
        self.model_parameters = list(self.state_encoder.parameters()) + list(self.decision_head.parameters())

        logger.info(f"Initialized model with {sum(p.numel() for p in self.model_parameters)} parameters")

    def _init_training_components(self):
        """Initialize loss function, optimizer, and scheduler."""
        # Loss function
        self.criterion = MTGOutcomeWeightedLoss(
            action_loss_weight=self.config.action_loss_weight,
            value_loss_weight=self.config.value_loss_weight
        )

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_epochs * 100,  # Estimate total steps
            pct_start=self.config.warmup_steps / (self.config.max_epochs * 100),
            anneal_strategy='cos'
        )

    def prepare_data(self, data_path: str):
        """Prepare training and validation datasets."""
        # Load dataset with current curriculum stage
        full_dataset = MTGTrainingDataset(
            data_path,
            self.curriculum_scheduler.get_current_stage()
        )

        # Compute class weights for loss function
        self.criterion.compute_class_weights(full_dataset)

        # Split into train and validation
        val_size = int(len(full_dataset) * self.config.validation_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        logger.info(f"Prepared data: {train_size} training samples, {val_size} validation samples")

    def train_epoch(self):
        """Train for one epoch."""
        self.state_encoder.train()
        self.decision_head.train()

        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            state_tensors = batch['state_tensor'].to(self.device)
            action_targets = batch['action_target'].to(self.device)
            value_targets = batch['value_target'].to(self.device)
            outcome_weights = batch['outcome_weight'].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                # Encode states
                encoded_states = self.state_encoder(state_tensors)

                # Generate predictions
                predictions = self.decision_head(encoded_states)

                # Compute loss
                losses = self.criterion(
                    predictions['action_logits'],
                    predictions['win_probability'].unsqueeze(1),
                    action_targets,
                    value_targets[:, 0:1],  # Use win probability as value target
                    outcome_weights
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.config.gradient_clip_norm)
                self.optimizer.step()

            self.scheduler.step()

            # Update metrics
            for key, value in losses.items():
                epoch_metrics[key] += value.item()

            # Compute accuracy
            pred_actions = torch.argmax(predictions['action_logits'], dim=1)
            accuracy = (pred_actions == action_targets).float().mean().item()
            epoch_metrics['accuracy'] += accuracy

            # Log progress
            if batch_idx % self.config.log_every == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {losses['total_loss'].item():.4f}, "
                          f"Accuracy: {accuracy:.4f}")

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def validate_epoch(self):
        """Validate for one epoch."""
        self.state_encoder.eval()
        self.decision_head.eval()

        epoch_metrics = defaultdict(float)
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                state_tensors = batch['state_tensor'].to(self.device)
                action_targets = batch['action_target'].to(self.device)
                value_targets = batch['value_target'].to(self.device)
                outcome_weights = batch['outcome_weight'].to(self.device)

                # Forward pass
                with autocast(enabled=self.config.mixed_precision):
                    encoded_states = self.state_encoder(state_tensors)
                    predictions = self.decision_head(encoded_states)

                    losses = self.criterion(
                        predictions['action_logits'],
                        predictions['win_probability'].unsqueeze(1),
                        action_targets,
                        value_targets[:, 0:1],
                        outcome_weights
                    )

                # Update metrics
                for key, value in losses.items():
                    epoch_metrics[key] += value.item()

                # Compute accuracy
                pred_actions = torch.argmax(predictions['action_logits'], dim=1)
                accuracy = (pred_actions == action_targets).float().mean().item()
                epoch_metrics['accuracy'] += accuracy

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def train(self, data_path: str):
        """Main training loop."""
        logger.info("Starting training...")

        # Prepare data
        self.prepare_data(data_path)

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            self.train_metrics['epoch'].append(epoch)
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)

            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_metrics['epoch'].append(epoch)
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)

            # Log epoch results
            logger.info(f"Epoch {epoch} - "
                       f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}, "
                       f"Val Loss: {val_metrics['total_loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}")

            # Check for curriculum advancement
            if self.curriculum_scheduler.should_advance(epoch, val_metrics):
                if self.curriculum_scheduler.advance_stage():
                    # Reload data with new curriculum stage
                    self.prepare_data(data_path)

            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_metrics['total_loss'])

            # Early stopping
            if val_metrics['total_loss'] < self.best_validation_loss - self.config.min_delta:
                self.best_validation_loss = val_metrics['total_loss']
                self.patience_counter = 0

                if self.config.keep_best:
                    self.save_checkpoint(epoch, val_metrics['total_loss'], best=True)
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        logger.info("Training completed!")
        return self.train_metrics, self.val_metrics

    def save_checkpoint(self, epoch: int, loss: float, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'decision_head_state_dict': self.decision_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'curriculum_stage': self.curriculum_scheduler.current_stage_idx
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint at epoch {epoch}")

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.decision_head.load_state_dict(checkpoint['decision_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
        self.curriculum_scheduler.current_stage_idx = checkpoint['curriculum_stage']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def main():
    """Main training function."""
    # Initialize configuration
    config = TrainingConfig(
        batch_size=16,  # Smaller batch for dataset size
        max_epochs=20,
        learning_rate=1e-4,
        curriculum_enabled=True
    )

    # Initialize trainer
    trainer = MTGTrainer(config)

    # Start training
    data_path = "/home/joshu/logparser/complete_training_dataset_task2_4.json"
    train_metrics, val_metrics = trainer.train(data_path)

    # Final checkpoint
    trainer.save_checkpoint(trainer.current_epoch, trainer.best_validation_loss, best=True)

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()