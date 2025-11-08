#!/usr/bin/env python3
"""
MTG Transformer State Encoder - Task 3.1

A Transformer-based neural network architecture for encoding Magic: The Gathering
game states into meaningful representations for decision-making.

This module implements:
- Multi-modal transformer encoder for 282-dimension game state tensors
- Component-specific processing for board, hand/mana, phase/priority features
- Position encoding for board permanents
- Multi-head attention mechanisms
- Regularization and normalization techniques
- Data loading utilities for JSON training dataset

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MTGTransformerConfig:
    """Configuration class for MTG Transformer Encoder."""

    # Model architecture parameters
    d_model: int = 256  # Hidden dimension of the transformer
    nhead: int = 8      # Number of attention heads
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5

    # Component-specific dimensions (updated to match actual dataset)
    board_tokens_dim: int = 64
    hand_mana_dim: int = 128
    phase_priority_dim: int = 64
    additional_features_dim: int = 10
    total_input_dim: int = 282  # Sum of all components (metadata value)

    # Actual observed dimensions in dataset
    observed_input_dim: int = 23  # Actual tensor dimension in data
    observed_action_dim: int = 15  # Actual action label dimension

    # Output dimensions
    action_vocab_size: int = 16
    output_dim: int = 128

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_seq_len: int = 1000  # For positional encoding

    # Board-specific parameters
    max_board_positions: int = 20  # Maximum permanents on board
    board_position_dim: int = 32   # Dimension for each board position embedding

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"

        # Calculate expected dimension and check if it matches
        expected_dim = (self.board_tokens_dim + self.hand_mana_dim +
                       self.phase_priority_dim + self.additional_features_dim)

        # Allow for actual data dimensions that may differ from expected
        if self.total_input_dim != expected_dim:
            print(f"Warning: total_input_dim ({self.total_input_dim}) != expected ({expected_dim})")
            print(f"Using observed_input_dim ({self.observed_input_dim}) instead of total_input_dim")
            self.total_input_dim = self.observed_input_dim


class PositionalEncoding(nn.Module):
    """Positional encoding for board permanents."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class BoardStateProcessor(nn.Module):
    """Processes board state tokens with positional encoding and attention."""

    def __init__(self, config: MTGTransformerConfig):
        super().__init__()
        self.config = config

        # Project board tokens to d_model dimension
        self.board_embedding = nn.Linear(config.total_input_dim, config.d_model)

        # Positional encoding for board positions
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_board_positions)

        # Board position embeddings (for up to max_board_positions permanents)
        self.position_embeddings = nn.Embedding(config.max_board_positions, config.board_position_dim)

        # Combine board token embeddings with position embeddings
        combined_input_dim = config.d_model + config.board_position_dim
        self.board_transform = nn.Linear(combined_input_dim, config.d_model)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, board_tokens: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process board state tokens.

        Args:
            board_tokens: Board state tensor of shape (batch_size, board_tokens_dim)
            positions: Optional position indices of shape (batch_size, num_permanents)

        Returns:
            Processed board representation of shape (batch_size, d_model)
        """
        batch_size = board_tokens.size(0)

        # Embed board tokens
        board_emb = self.board_embedding(board_tokens)  # (batch_size, d_model)

        if positions is not None:
            # Get position embeddings for each permanent
            pos_emb = self.position_embeddings(positions)  # (batch_size, num_permanents, board_position_dim)
            pos_emb = pos_emb.mean(dim=1)  # Aggregate position information

            # Combine board and position embeddings
            combined = torch.cat([board_emb, pos_emb], dim=-1)
            board_emb = self.board_transform(combined)

        # Add sequence dimension for positional encoding
        board_emb = board_emb.unsqueeze(0)  # (1, batch_size, d_model)
        board_emb = self.pos_encoder(board_emb)
        board_emb = board_emb.squeeze(0)  # (batch_size, d_model)

        # Apply normalization and dropout
        board_emb = self.layer_norm(board_emb)
        board_emb = self.dropout(board_emb)

        return board_emb


class ComponentProcessor(nn.Module):
    """Generic processor for non-board components (hand/mana, phase/priority, additional features)."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process component tensor."""
        x = self.projection(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiModalFusion(nn.Module):
    """Fuses representations from different game state components."""

    def __init__(self, config: MTGTransformerConfig):
        super().__init__()
        self.config = config

        # Multi-head attention for component interaction
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, component_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse component representations using transformer encoder.

        Args:
            component_embeddings: Tensor of shape (batch_size, num_components, d_model)

        Returns:
            Fused representation of shape (batch_size, d_model)
        """
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(
            component_embeddings, component_embeddings, component_embeddings
        )

        # Residual connection and normalization
        x = self.norm1(component_embeddings + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Residual connection and normalization
        x = self.norm2(x + ffn_output)

        # Global pooling (average over components)
        fused_repr = x.mean(dim=1)  # (batch_size, d_model)

        return fused_repr, attn_weights


class MTGTransformerEncoder(nn.Module):
    """
    Main Transformer Encoder for MTG game states.

    This model processes 282-dimension game state tensors consisting of:
    - Board tokens (64-dim): Permanent cards on battlefield
    - Hand/mana (128-dim): Hand contents and available mana
    - Phase/priority (64-dim): Game phase and priority information
    - Additional features (10-dim): Turn number, life totals, etc.
    """

    def __init__(self, config: MTGTransformerConfig):
        super().__init__()
        self.config = config

        # Component processors
        self.board_processor = BoardStateProcessor(config)
        self.hand_mana_processor = ComponentProcessor(
            config.hand_mana_dim, config.d_model, config.dropout
        )
        self.phase_priority_processor = ComponentProcessor(
            config.phase_priority_dim, config.d_model, config.dropout
        )
        self.additional_features_processor = ComponentProcessor(
            config.additional_features_dim, config.d_model, config.dropout
        )

        # Multi-modal fusion
        self.fusion = MultiModalFusion(config)

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.action_vocab_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 1)
        )

        # State representation head
        self.state_repr_head = nn.Sequential(
            nn.Linear(config.d_model, config.output_dim),
            nn.LayerNorm(config.output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, state_tensor: torch.Tensor,
                board_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MTG Transformer Encoder.

        Args:
            state_tensor: Input tensor of shape (batch_size, 282)
            board_positions: Optional board position indices of shape (batch_size, num_permanents)

        Returns:
            Dictionary containing:
            - 'action_logits': Action predictions of shape (batch_size, action_vocab_size)
            - 'value': State value estimate of shape (batch_size, 1)
            - 'state_representation': State embedding of shape (batch_size, output_dim)
            - 'attention_weights': Attention weights for explainability
            - 'component_embeddings': Individual component representations
        """
        batch_size = state_tensor.size(0)

        # Handle actual dataset dimensions vs expected dimensions
        input_dim = state_tensor.size(1)

        if input_dim == self.config.total_input_dim:
            # Full tensor with all components
            board_tokens = state_tensor[:, :self.config.board_tokens_dim]
            hand_mana = state_tensor[:,
                         self.config.board_tokens_dim:self.config.board_tokens_dim + self.config.hand_mana_dim]
            phase_priority = state_tensor[
                self.config.board_tokens_dim + self.config.hand_mana_dim:
                self.config.board_tokens_dim + self.config.hand_mana_dim + self.config.phase_priority_dim]
            additional_features = state_tensor[
                self.config.board_tokens_dim + self.config.hand_mana_dim + self.config.phase_priority_dim:
            ]
        elif input_dim == self.config.observed_input_dim:
            # Observed dataset dimensions (23) - treat as compressed representation
            # We'll split it proportionally based on original component ratios
            total_original = self.config.board_tokens_dim + self.config.hand_mana_dim + self.config.phase_priority_dim + self.config.additional_features_dim

            board_split = int(self.config.board_tokens_dim * input_dim / total_original)
            hand_mana_split = int(self.config.hand_mana_dim * input_dim / total_original)
            phase_priority_split = int(self.config.phase_priority_dim * input_dim / total_original)

            board_tokens = state_tensor[:, :board_split]
            hand_mana = state_tensor[:, board_split:board_split + hand_mana_split]
            phase_priority = state_tensor[:, board_split + hand_mana_split:board_split + hand_mana_split + phase_priority_split]
            additional_features = state_tensor[:, board_split + hand_mana_split + phase_priority_split:]
        else:
            # Unknown dimension - use equal split
            split_size = input_dim // 4
            board_tokens = state_tensor[:, :split_size]
            hand_mana = state_tensor[:, split_size:2*split_size]
            phase_priority = state_tensor[:, 2*split_size:3*split_size]
            additional_features = state_tensor[:, 3*split_size:]

        # Process each component
        board_emb = self.board_processor(board_tokens, board_positions)
        hand_mana_emb = self.hand_mana_processor(hand_mana)
        phase_priority_emb = self.phase_priority_processor(phase_priority)
        additional_features_emb = self.additional_features_processor(additional_features)

        # Stack component embeddings
        component_embeddings = torch.stack([
            board_emb, hand_mana_emb, phase_priority_emb, additional_features_emb
        ], dim=1)  # (batch_size, 4, d_model)

        # Fuse components using transformer
        fused_repr, attention_weights = self.fusion(component_embeddings)

        # Generate outputs
        action_logits = self.action_head(fused_repr)
        value = self.value_head(fused_repr)
        state_representation = self.state_repr_head(fused_repr)

        return {
            'action_logits': action_logits,
            'value': value,
            'state_representation': state_representation,
            'attention_weights': attention_weights,
            'component_embeddings': component_embeddings
        }


class MTGDataset(Dataset):
    """Dataset class for loading MTG training data from JSON format."""

    def __init__(self, json_file_path: str, transform=None):
        """
        Initialize dataset.

        Args:
            json_file_path: Path to JSON training data
            transform: Optional transform to apply to samples
        """
        self.json_file_path = json_file_path
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load training data from JSON file."""
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)

        training_samples = data.get('training_samples', [])
        logger.info(f"Loaded {len(training_samples)} training samples from {self.json_file_path}")

        # Extract metadata
        self.metadata = data.get('metadata', {})
        self.total_samples = self.metadata.get('total_samples', len(training_samples))
        self.tensor_dimension = self.metadata.get('final_tensor_dimension', 282)

        return training_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, int]]:
        """
        Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - state_tensor: Game state tensor
            - action_label: One-hot action labels
            - outcome_weight: Sample weight for training
            - decision_type: Type of decision (string)
            - turn: Turn number
            - game_outcome: Whether the game was won
        """
        sample = self.data[idx]

        # Convert to tensors
        state_tensor = torch.tensor(sample['state_tensor'], dtype=torch.float32)
        action_label = torch.tensor(sample['action_label'], dtype=torch.float32)
        outcome_weight = torch.tensor(sample['outcome_weight'], dtype=torch.float32)

        result = {
            'state_tensor': state_tensor,
            'action_label': action_label,
            'outcome_weight': outcome_weight,
            'decision_type': sample['decision_type'],
            'turn': sample['turn'],
            'game_outcome': sample['game_outcome']
        }

        if self.transform:
            result = self.transform(result)

        return result


def create_data_loaders(train_path: str, batch_size: int = 32,
                       validation_split: float = 0.2, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_path: Path to training JSON file
        batch_size: Batch size for data loaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = MTGDataset(train_path)

    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created data loaders: {train_size} training samples, {val_size} validation samples")

    return train_loader, val_loader


def train_model(model: MTGTransformerEncoder, train_loader: DataLoader,
                val_loader: DataLoader, num_epochs: int = 100,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                learning_rate: float = 1e-4) -> Dict[str, List[float]]:
    """
    Train the MTG Transformer model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Learning rate

    Returns:
        Dictionary containing training metrics
    """
    model = model.to(device)

    # Optimizer and loss functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    action_criterion = nn.BCEWithLogitsLoss(reduction='none')
    value_criterion = nn.MSELoss()

    # Training metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            state_tensors = batch['state_tensor'].to(device)
            action_labels = batch['action_label'].to(device)
            outcome_weights = batch['outcome_weight'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(state_tensors)
            action_logits = outputs['action_logits']
            value_pred = outputs['value']
            value_target = outcome_weights.unsqueeze(1)

            # Compute losses
            action_loss = action_criterion(action_logits, action_labels)
            action_loss = (action_loss * outcome_weights.unsqueeze(1)).mean()
            value_loss = value_criterion(value_pred, value_target)

            total_loss = action_loss + 0.5 * value_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += total_loss.item()
            predicted_actions = (torch.sigmoid(action_logits) > 0.5).float()
            train_correct += (predicted_actions == action_labels).sum().item()
            train_total += action_labels.numel()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                state_tensors = batch['state_tensor'].to(device)
                action_labels = batch['action_label'].to(device)
                outcome_weights = batch['outcome_weight'].to(device)

                outputs = model(state_tensors)
                action_logits = outputs['action_logits']
                value_pred = outputs['value']
                value_target = outcome_weights.unsqueeze(1)

                # Compute losses
                action_loss = action_criterion(action_logits, action_labels)
                action_loss = (action_loss * outcome_weights.unsqueeze(1)).mean()
                value_loss = value_criterion(value_pred, value_target)

                total_loss = action_loss + 0.5 * value_loss

                val_loss += total_loss.item()
                predicted_actions = (torch.sigmoid(action_logits) > 0.5).float()
                val_correct += (predicted_actions == action_labels).sum().item()
                val_total += action_labels.numel()

        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    return metrics


def main():
    """Main function for testing the MTG Transformer Encoder."""

    # Configuration
    config = MTGTransformerConfig(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=512,
        dropout=0.1
    )

    # Create model
    model = MTGTransformerEncoder(config)
    logger.info(f"Created MTG Transformer Encoder with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    sample_input = torch.randn(batch_size, 282)

    with torch.no_grad():
        outputs = model(sample_input)

    logger.info("Forward pass test successful!")
    logger.info(f"Action logits shape: {outputs['action_logits'].shape}")
    logger.info(f"Value shape: {outputs['value'].shape}")
    logger.info(f"State representation shape: {outputs['state_representation'].shape}")
    logger.info(f"Attention weights shape: {outputs['attention_weights'].shape}")

    # Test data loading
    try:
        train_loader, val_loader = create_data_loaders(
            '/home/joshu/logparser/complete_training_dataset_task2_4.json',
            batch_size=8
        )

        # Test training on a small batch
        for batch in train_loader:
            with torch.no_grad():
                outputs = model(batch['state_tensor'])
            logger.info("Data loading and batch processing test successful!")
            break

    except FileNotFoundError:
        logger.warning("Training data file not found. Skipping data loading test.")

    logger.info("MTG Transformer Encoder implementation complete!")


if __name__ == "__main__":
    main()