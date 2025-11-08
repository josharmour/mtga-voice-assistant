#!/usr/bin/env python3
"""
Memory-Safe MTG AI Training
Optimized for 32GB DDR5 + 16GB GPU (RTX 5080) systems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
import psutil
import sys
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class MTGDataset(Dataset):
    """Memory-efficient dataset class that loads samples on-demand."""

    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.device = device

        # Load metadata and sample indices only
        print("📂 Loading dataset metadata...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.samples = self.data['training_samples']
        self.total_samples = len(self.samples)

        print(f"✅ Dataset loaded: {self.total_samples} samples")
        print(f"📊 Memory usage after loading metadata: {self._get_memory_usage():.1f}GB")

        # Pre-validate first sample
        if self.total_samples > 0:
            sample = self.samples[0]
            self.state_dim = len(sample['state_tensor'])
            self.action_dim = len(sample['action_label'])
            print(f"🔢 Sample dimensions: state={self.state_dim}, actions={self.action_dim}")

    def _get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors on-demand
        state = torch.tensor(sample['state_tensor'], dtype=torch.float32)
        action = torch.tensor(sample['action_label'], dtype=torch.float32)
        weight = torch.tensor(sample['outcome_weight'], dtype=torch.float32)

        return state, action, weight

class MemoryOptimizedMTGModel(nn.Module):
    """Memory-optimized MTG model with efficient parameter usage."""

    def __init__(self, input_dim=21, d_model=64, num_actions=15):  # Reduced d_model for memory
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_actions = num_actions

        # Efficient encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),  # More memory efficient than BatchNorm
            nn.ReLU(inplace=True),   # In-place operations save memory
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # Compact prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1)
        )

        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state_tensor):
        encoded = self.encoder(state_tensor)
        action_logits = self.action_head(encoded)
        value = self.value_head(encoded)
        return action_logits, value

class MemorySafeTrainer:
    """Memory-safe trainer with monitoring and cleanup."""

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        print(f"🖥️  Using device: {device}")

        # Model with reduced parameters for memory efficiency
        self.model = MemoryOptimizedMTGModel().to(device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 Model parameters: {param_count:,}")

        # Loss functions
        self.criterion_action = nn.BCEWithLogitsLoss()
        self.criterion_value = nn.MSELoss()

        # Optimizer with memory-efficient settings
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        print(f"💾 Initial GPU memory: {self._get_gpu_memory():.1f}GB" if device == 'cuda' else "💾 Using CPU")

    def _get_memory_usage(self):
        """Get system memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

    def _get_gpu_memory(self):
        """Get GPU memory usage in GB."""
        if self.device == 'cuda':
            return torch.cuda.memory_allocated() / 1024**3
        return 0

    def train(self, data_path, epochs=15, batch_size=8, max_samples=None):
        """Memory-safe training with monitoring."""
        print(f"🚀 Starting Memory-Safe MTG AI Training")
        print(f"📊 Initial memory usage: {self._get_memory_usage():.1f}GB")

        # Create dataset
        dataset = MTGDataset(data_path, device='cpu')  # Keep data on CPU initially

        # Limit samples if specified
        if max_samples:
            dataset.samples = dataset.samples[:max_samples]
            dataset.total_samples = len(dataset.samples)
            print(f"📊 Limited to {max_samples} samples")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(dataset)))

        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Memory-efficient data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Parallel loading
            pin_memory=True if self.device == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )

        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        print(f"📦 Batch size: {batch_size}")

        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop with memory management
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (states, actions, weights) in enumerate(train_loader):
                # Move batch to device
                states = states.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # Forward pass
                action_logits, values = self.model(states)

                # Calculate losses
                action_loss = self.criterion_action(action_logits, actions)
                value_loss = self.criterion_value(values.squeeze(-1), weights)
                total_loss = action_loss + 0.5 * value_loss

                # Backward pass
                total_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Statistics
                train_loss += total_loss.item()
                predicted = (torch.sigmoid(action_logits) > 0.5).float()
                train_correct += (predicted == actions).sum().item()
                train_total += actions.numel()

                # Memory cleanup every 10 batches
                if batch_idx % 10 == 0:
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

                # Progress reporting
                if batch_idx % 50 == 0:
                    current_mem = self._get_memory_usage()
                    gpu_mem = self._get_gpu_memory()
                    print(f"  Batch {batch_idx}/{len(train_loader)} - "
                          f"Loss: {total_loss.item():.4f} - "
                          f"Mem: {current_mem:.1f}GB - "
                          f"GPU: {gpu_mem:.1f}GB")

            # Validation phase
            val_loss, val_acc = self._validate(val_loader)

            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print epoch results
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Memory: {self._get_memory_usage():.1f}GB")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'mtg_memory_optimized_model.pth')
                print("💾 New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print("⏹️  Early stopping triggered")
                    break

            # Memory cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\n🎉 Training completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: mtg_memory_optimized_model.pth")

        return {
            'best_val_loss': best_val_loss,
            'final_memory': self._get_memory_usage(),
            'epochs_trained': epoch + 1
        }

    def _validate(self, val_loader):
        """Validation phase."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, actions, weights in val_loader:
                states = states.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)

                action_logits, values = self.model(states)

                action_loss = self.criterion_action(action_logits, actions)
                value_loss = self.criterion_value(values.squeeze(-1), weights)
                total_loss = action_loss + 0.5 * value_loss

                val_loss += total_loss.item()
                predicted = (torch.sigmoid(action_logits) > 0.5).float()
                val_correct += (predicted == actions).sum().item()
                val_total += actions.numel()

        return val_loss / len(val_loader), 100. * val_correct / val_total

def main():
    """Main training function with memory safety."""
    print("🧠 Memory-Safe MTG AI Training")
    print("=" * 50)
    print("💻 System: 32GB DDR5 + 16GB RTX 5080")
    print("🔒 Memory-safe training enabled")

    # Initialize trainer
    trainer = MemorySafeTrainer()

    # Train with memory-safe settings
    data_path = "/home/joshu/logparser/data/comprehensive_17lands_training_data_4522_samples.json"

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    try:
        # Memory-safe training parameters
        metrics = trainer.train(
            data_path,
            epochs=20,
            batch_size=32,  # Optimized for 16GB GPU
            max_samples=4000  # Use comprehensive samples
        )

        print("\n✅ Training successful!")
        print(f"📊 Final metrics: {metrics}")

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
        print(f"🎯 Actual action: {test_sample['action_label'][:5]}..., Weight: {test_sample['outcome_weight']:.4f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Memory cleanup completed")

if __name__ == "__main__":
    main()