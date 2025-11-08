#!/usr/bin/env python3
"""
Working Comprehensive MTG Model
Simple but effective model for 282-dimensional comprehensive board state.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class WorkingComprehensiveMTGModel(nn.Module):
    """Working comprehensive model for 282-dim input with proven architecture."""

    def __init__(self, input_dim=282, d_model=256, num_actions=15, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions

        # Simple but effective architecture
        self.encoder = nn.Sequential(
            # Input processing
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Hidden layers
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )

        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, state_tensor):
        """
        Forward pass through the working comprehensive model.

        Args:
            state_tensor: Input tensor of shape (batch_size, 282)

        Returns:
            action_logits: Action predictions (batch_size, num_actions)
            value: Value prediction (batch_size, 1)
        """
        # Encode the state
        encoded = self.encoder(state_tensor)

        # Generate predictions
        action_logits = self.action_head(encoded)
        value = self.value_head(encoded)

        return action_logits, value

class WorkingComprehensiveTrainer:
    """Trainer for the working comprehensive model."""

    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = WorkingComprehensiveMTGModel(
            input_dim=282,
            d_model=256,
            num_actions=15,
            dropout=0.1
        ).to(self.device)

        # Loss functions
        self.criterion_action = nn.BCEWithLogitsLoss()
        self.criterion_value = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,  # Standard learning rate
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.7
        )

        # Track metrics
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🧠 Working Comprehensive MTG Model initialized: {param_count:,} parameters")
        logger.info(f"🖥️  Device: {self.device}")

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (states, actions, weights) in enumerate(train_loader):
            states = states.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass
            action_logits, values = self.model(states)

            # Calculate losses
            action_loss = self.criterion_action(action_logits, actions)
            value_loss = self.criterion_value(values.squeeze(-1), weights)
            total_loss_batch = action_loss + 0.5 * value_loss

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += total_loss_batch.item()
            predicted = (torch.sigmoid(action_logits) > 0.5).float()
            total_correct += (predicted == actions).sum().item()
            total_samples += actions.numel()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for states, actions, weights in val_loader:
                states = states.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)

                action_logits, values = self.model(states)

                action_loss = self.criterion_action(action_logits, actions)
                value_loss = self.criterion_value(values.squeeze(-1), weights)
                total_loss_batch = action_loss + 0.5 * value_loss

                total_loss += total_loss_batch.item()
                predicted = (torch.sigmoid(action_logits) > 0.5).float()
                total_correct += (predicted == actions).sum().item()
                total_samples += actions.numel()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=20, patience=8):
        """Train the working comprehensive model."""
        logger.info("🚀 Starting Working Comprehensive MTG Model Training")
        logger.info(f"📊 Training samples: {len(train_loader.dataset)}")
        logger.info(f"📊 Validation samples: {len(val_loader.dataset)}")
        logger.info(f"🎯 Epochs: {epochs}, Patience: {patience}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Record metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:2d}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                       f"LR: {current_lr:.2e}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'working_comprehensive_mtg_model.pth')
                logger.info(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"⏹️  Early stopping triggered after {epoch+1} epochs")
                break

        logger.info(f"🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.training_history

def main():
    """Demonstrate the working comprehensive model."""
    logger.info("🧠 Working Comprehensive MTG Model Demo")
    logger.info("=" * 50)

    # Initialize model
    model = WorkingComprehensiveMTGModel(input_dim=282, d_model=256, num_actions=15)

    # Test with sample data
    batch_size = 4
    sample_input = torch.randn(batch_size, 282)

    # Forward pass
    action_logits, value = model(sample_input)

    logger.info(f"✅ Working model test successful!")
    logger.info(f"📊 Input shape: {sample_input.shape}")
    logger.info(f"📊 Action logits shape: {action_logits.shape}")
    logger.info(f"📊 Value shape: {value.shape}")
    logger.info(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()