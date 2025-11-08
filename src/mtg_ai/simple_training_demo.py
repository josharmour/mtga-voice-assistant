#!/usr/bin/env python3
"""
Simple MTG AI Training Demo
A simplified version that works with the actual data structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mtg_transformer_encoder import MTGTransformerEncoder, MTGTransformerConfig

class SimpleMTGModel(nn.Module):
    """Simplified MTG model that works with actual data structure."""

    def __init__(self, input_dim=21, d_model=128, num_actions=15):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions

        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_actions)
        )

        # Value prediction head (for reinforcement learning)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, state_tensor):
        # Encode state
        encoded = self.encoder(state_tensor)

        # Predict action probabilities and value
        action_logits = self.action_head(encoded)
        value = self.value_head(encoded)

        return action_logits, value

class SimpleTrainer:
    """Simple trainer for MTG AI model."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleMTGModel().to(self.device)
        self.criterion_action = nn.BCEWithLogitsLoss()  # For multi-label classification
        self.criterion_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, data_path, epochs=20):
        """Train the model on the given data."""
        print(f"🤖 Starting MTG AI Training on {self.device}")

        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)

        samples = data['training_samples']
        print(f"Loaded {len(samples)} training samples")

        # Convert data to tensors
        states = torch.tensor([s['state_tensor'] for s in samples], dtype=torch.float32)
        actions = torch.tensor([s['action_label'] for s in samples], dtype=torch.float32)
        weights = torch.tensor([s['outcome_weight'] for s in samples], dtype=torch.float32)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        weights = weights.to(self.device)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, actions, weights)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (batch_states, batch_actions, batch_weights) in enumerate(train_loader):
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_weights = batch_weights.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                action_logits, values = self.model(batch_states)

                # Calculate losses
                action_loss = self.criterion_action(action_logits, batch_actions)
                value_loss = self.criterion_value(values.squeeze(), batch_weights)

                total_loss = action_loss + 0.5 * value_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                # Statistics (multi-label accuracy)
                train_loss += total_loss.item()
                predicted = (torch.sigmoid(action_logits) > 0.5).float()
                train_correct += (predicted == batch_actions.float()).sum().item()
                train_total += batch_actions.numel()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_states, batch_actions, batch_weights in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    batch_weights = batch_weights.to(self.device)

                    action_logits, values = self.model(batch_states)

                    action_loss = self.criterion_action(action_logits, batch_actions)
                    value_loss = self.criterion_value(values.squeeze(), batch_weights)

                    total_loss = action_loss + 0.5 * value_loss
                    val_loss += total_loss.item()

                    predicted = (torch.sigmoid(action_logits) > 0.5).float()
                    val_correct += (predicted == batch_actions.float()).sum().item()
                    val_total += batch_actions.numel()

            # Print progress
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

        print("🎉 Training completed!")
        print("💾 Model saved to mtg_simple_model.pth")

        # Save model
        torch.save(self.model.state_dict(), 'mtg_simple_model.pth')

        return {
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'epochs_trained': epochs
        }

def main():
    """Main training function."""
    print("🚀 MTG AI Simple Training Demo")
    print("=" * 50)

    # Initialize trainer
    trainer = SimpleTrainer()

    # Train model
    data_path = "/home/joshu/logparser/data/scaled_training_dataset.json"

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    try:
        metrics = trainer.train(data_path, epochs=20)
        print("\n✅ Training successful!")
        print(f"📊 Final accuracy: {metrics['validation_accuracy']:.2f}%")

        # Test the model
        print("\n🧪 Testing trained model...")
        trainer.model.eval()

        # Load a sample for testing
        with open(data_path, 'r') as f:
            data = json.load(f)

        test_sample = data['training_samples'][0]
        test_state = torch.tensor([test_sample['state_tensor']], dtype=torch.float32).to(trainer.device)

        with torch.no_grad():
            action_logits, value = trainer.model(test_state)

        predicted_action = torch.argmax(action_logits, dim=1).item()
        predicted_value = value.item()

        print(f"📝 Sample prediction: Action {predicted_action}, Value {predicted_value:.4f}")
        print(f"🎯 Actual action: {test_sample['action_label']}, Weight: {test_sample['outcome_weight']:.4f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()