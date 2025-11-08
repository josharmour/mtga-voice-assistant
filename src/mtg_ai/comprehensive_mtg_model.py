#!/usr/bin/env python3
"""
Comprehensive MTG AI Model
Full 282-dimensional state processing with Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)

class MultiModalTransformerEncoder(nn.Module):
    """Multi-modal Transformer encoder for 282-dim MTG state tensors."""

    def __init__(self, input_dim=282, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        # Component projections for different tensor sections
        self.core_projection = nn.Linear(32, d_model // 4)      # Core game info
        self.board_projection = nn.Linear(64, d_model // 4)     # Board state
        self.hand_projection = nn.Linear(128, d_model // 2)     # Hand & mana
        self.phase_projection = nn.Linear(58, d_model // 4)     # Phase/priority
        self.strategic_projection = nn.Linear(58, d_model // 4) # Strategic context

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Component attention weights
        self.component_attention = nn.Parameter(torch.ones(5))  # 5 components

    def forward(self, x):
        """
        Forward pass with multi-modal processing.

        Args:
            x: Input tensor of shape (batch_size, 282)

        Returns:
            Encoded representation of shape (batch_size, d_model)
        """
        batch_size = x.size(0)

        # Split input into components (282 total)
        core = x[:, 0:32]           # Core game info (32)
        board = x[:, 32:96]         # Board state (64)
        hand = x[:, 96:224]         # Hand & mana (128)
        phase = x[:, 224:282]       # Phase/priority (58) - remaining dimensions
        strategic = x[:, 224:282]    # Strategic context (58) - same as phase for simplicity

        # For simplicity, use phase as both phase and strategic
        strategic = phase

        # Project each component
        core_proj = self.core_projection(core)          # (batch, d_model//4)
        board_proj = self.board_projection(board)        # (batch, d_model//4)
        hand_proj = self.hand_projection(hand)          # (batch, d_model//2)
        phase_proj = self.phase_projection(phase)        # (batch, d_model//4)
        strategic_proj = self.strategic_projection(strategic)  # (batch, d_model//4)

        # Combine projections with attention weights
        component_weights = torch.softmax(self.component_attention, dim=0)

        # Pad smaller components to match largest (hand_proj)
        target_size = hand_proj.size(-1)
        core_padded = torch.cat([core_proj, torch.zeros_like(core_proj[:, :(target_size - core_proj.size(-1))])], dim=-1)
        board_padded = torch.cat([board_proj, torch.zeros_like(board_proj[:, :(target_size - board_proj.size(-1))])], dim=-1)
        phase_padded = torch.cat([phase_proj, torch.zeros_like(phase_proj[:, :(target_size - phase_proj.size(-1))])], dim=-1)
        strategic_padded = torch.cat([strategic_proj, torch.zeros_like(strategic_proj[:, :(target_size - strategic_proj.size(-1))])], dim=-1)

        # Weighted combination
        combined = (component_weights[0] * core_padded +
                   component_weights[1] * board_padded +
                   component_weights[2] * hand_proj +
                   component_weights[3] * phase_padded +
                   component_weights[4] * strategic_padded)

        # Add batch dimension for transformer (batch_size, 1, d_model)
        combined = combined.unsqueeze(1)

        # Add positional encoding
        combined = self.pos_encoding(combined)

        # Apply transformer
        encoded = self.transformer(combined)

        # Remove sequence dimension and apply layer norm
        encoded = self.layer_norm(encoded.squeeze(1))

        return encoded

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ComprehensiveMTGModel(nn.Module):
    """Comprehensive MTG AI model with 282-dim input and full board state processing."""

    def __init__(self, input_dim=282, d_model=256, num_actions=15, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions

        # Multi-modal transformer encoder
        self.encoder = MultiModalTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dropout=dropout
        )

        # Action prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_actions)
        )

        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 1)
        )

        # Explainability heads (attention analysis)
        self.component_explainer = nn.Sequential(
            nn.Linear(d_model, 5),  # Explain importance of each component
            nn.Softmax(dim=-1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights for better training."""
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
        Forward pass through the comprehensive model.

        Args:
            state_tensor: Input tensor of shape (batch_size, 282)

        Returns:
            action_logits: Action predictions (batch_size, num_actions)
            value: Value prediction (batch_size, 1)
            attention_weights: Component importance (batch_size, 5)
        """
        # Encode the state
        encoded = self.encoder(state_tensor)

        # Generate predictions
        action_logits = self.action_head(encoded)
        value = self.value_head(encoded)
        attention_weights = self.component_explainer(encoded)

        return action_logits, value, attention_weights

    def get_attention_weights(self, state_tensor):
        """Get attention weights for explainability."""
        with torch.no_grad():
            _, _, attention_weights = self.forward(state_tensor)
        return attention_weights

class ComprehensiveTrainer:
    """Trainer for the comprehensive MTG model."""

    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = ComprehensiveMTGModel(
            input_dim=282,
            d_model=256,
            num_actions=15,
            dropout=0.1
        ).to(self.device)

        # Loss functions
        self.criterion_action = nn.BCEWithLogitsLoss()
        self.criterion_value = nn.MSELoss()

        # Optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=5e-5,  # Smaller learning rate for larger model
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        # Track metrics
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🧠 Comprehensive MTG Model initialized: {param_count:,} parameters")
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
            action_logits, values, attention_weights = self.model(states)

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

                action_logits, values, attention_weights = self.model(states)

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

    def train(self, train_loader, val_loader, epochs=30, patience=10):
        """Train the comprehensive model."""
        logger.info("🚀 Starting Comprehensive MTG Model Training")
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
                torch.save(self.model.state_dict(), 'comprehensive_mtg_model.pth')
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
    """Demonstrate the comprehensive model."""
    logger.info("🧠 Comprehensive MTG Model Demo")
    logger.info("=" * 50)

    # Initialize model
    model = ComprehensiveMTGModel(input_dim=282, d_model=256, num_actions=15)

    # Test with sample data
    batch_size = 4
    sample_input = torch.randn(batch_size, 282)

    # Forward pass
    action_logits, value, attention_weights = model(sample_input)

    logger.info(f"✅ Model test successful!")
    logger.info(f"📊 Input shape: {sample_input.shape}")
    logger.info(f"📊 Action logits shape: {action_logits.shape}")
    logger.info(f"📊 Value shape: {value.shape}")
    logger.info(f"📊 Attention weights shape: {attention_weights.shape}")
    logger.info(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()