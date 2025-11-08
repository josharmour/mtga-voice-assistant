#!/usr/bin/env python3
"""
Memory-Safe Comprehensive MTG Training Pipeline
Process 17Lands data with full 282-dimensional state representation and memory safety.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
import logging
import gc
import psutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from safe_comprehensive_extractor import SafeComprehensiveMTGStateExtractor
from comprehensive_mtg_model import ComprehensiveTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import official 17Lands dtypes
try:
    from official_17lands_replay_dtypes import get_dtypes
except ImportError:
    logger.warning("Official 17Lands dtypes not available, using defaults")
    get_dtypes = None

class SafeComprehensiveMTGDataset(Dataset):
    """Memory-safe dataset for comprehensive 282-dimensional MTG states."""

    def __init__(self, samples: List[Dict[str, Any]], device='cpu'):
        self.samples = samples
        self.device = device

        logger.info(f"📊 Safe dataset initialized with {len(samples)} samples")
        if len(samples) > 0:
            sample = samples[0]
            if 'state_tensor' in sample:
                state_dim = len(sample['state_tensor'])
                action_dim = len(sample['action_label'])
            else:
                state_dim = 282
                action_dim = 15
            logger.info(f"🔢 Sample dimensions: state={state_dim}, actions={action_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors with safety checks
        if 'state_tensor' in sample:
            state = torch.tensor(sample['state_tensor'], dtype=torch.float32)
        else:
            logger.warning(f"Sample {idx} missing state_tensor, using zeros")
            state = torch.zeros(282, dtype=torch.float32)

        action = torch.tensor(sample['action_label'], dtype=torch.float32)
        weight = torch.tensor(sample['outcome_weight'], dtype=torch.float32)

        return state, action, weight

class SafeComprehensiveDataProcessor:
    """Memory-safe processor for 17Lands data with comprehensive state extraction."""

    def __init__(self, memory_limit_gb: float = 10.0):
        self.memory_limit_gb = memory_limit_gb
        self.state_extractor = SafeComprehensiveMTGStateExtractor()
        self.processed_samples = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

    def process_replay_file(self, file_path: str, max_games: int = 500) -> int:
        """Process 17Lands replay file with memory-safe comprehensive extraction."""
        logger.info(f"📂 Processing comprehensive data from {file_path}")
        logger.info(f"🎯 Target: {max_games} games (memory-safe mode)")

        try:
            # Load dtypes if available
            if get_dtypes:
                logger.info("🔧 Loading official 17Lands dtypes...")
                dtypes = get_dtypes(file_path, print_missing=False)
                # Convert problematic dtypes
                safe_dtypes = {}
                for col, dtype in dtypes.items():
                    if dtype == 'float16':
                        safe_dtypes[col] = 'float32'
                    else:
                        safe_dtypes[col] = dtype
                logger.info(f"✅ Loaded {len(safe_dtypes)} column dtype definitions")
            else:
                safe_dtypes = None

            samples_extracted = 0
            games_processed = 0
            chunk_size = 100  # Very small chunks for memory safety

            # Process file in very small chunks
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size,
                                      dtype=safe_dtypes, low_memory=False):
                current_memory = self.get_memory_usage()
                logger.info(f"🔄 Processing tiny chunk with {len(chunk_df)} games - Memory: {current_memory:.1f}GB")

                chunk_samples = []
                for _, row in chunk_df.iterrows():
                    try:
                        # Extract multiple decision points per game (limited)
                        game_samples = self._extract_game_decisions_safe(row, str(file_path))
                        chunk_samples.extend(game_samples)
                        games_processed += 1

                        if games_processed >= max_games:
                            break

                    except Exception as e:
                        logger.warning(f"Error processing game {games_processed}: {e}")
                        continue

                self.processed_samples.extend(chunk_samples)
                samples_extracted += len(chunk_samples)

                logger.info(f"✅ Extracted {len(chunk_samples)} comprehensive samples from {games_processed} games")
                logger.info(f"📈 Running total: {len(self.processed_samples)} samples")

                # Aggressive memory management
                del chunk_df, chunk_samples
                gc.collect()

                if current_memory > self.memory_limit_gb:
                    logger.warning(f"⚠️ High memory usage: {current_memory:.1f}GB, forcing cleanup")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if games_processed >= max_games:
                    break

            logger.info(f"✅ Completed memory-safe comprehensive processing for {file_path}")
            return samples_extracted

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return 0

    def _extract_game_decisions_safe(self, row: pd.Series, file_path: str) -> List[Dict[str, Any]]:
        """Extract decision points with memory safety."""
        samples = []

        try:
            # Game metadata with safe extraction
            game_id = row.get('draft_id', 'unknown')
            expansion = row.get('expansion', 'unknown')
            won = row.get('won', False)
            num_turns = self.state_extractor._safe_int(row.get('num_turns', 10))
            on_play = row.get('on_play', True)

            # Extract decisions from limited turns for memory safety
            max_turns_to_extract = min(num_turns, 6)  # Only 6 decisions per game for safety

            for turn_num in range(1, max_turns_to_extract + 1):
                # Check if turn data exists
                turn_prefix = f'user_turn_{turn_num}_'
                if not any(col.startswith(turn_prefix) for col in row.index):
                    continue

                # Skip if no meaningful action this turn (except early turns)
                total_actions = (self.state_extractor._safe_float(row.get(f'{turn_prefix}lands_played', 0)) +
                               self.state_extractor._safe_float(row.get(f'{turn_prefix}creatures_cast', 0)) +
                               self.state_extractor._safe_float(row.get(f'{turn_prefix}non_creatures_cast', 0)) +
                               self.state_extractor._safe_float(row.get(f'{turn_prefix}user_instants_sorceries_cast', 0)))

                if total_actions == 0 and turn_num > 2:  # More restrictive skip
                    continue

                # Extract comprehensive state tensor (282 dims) safely
                state_tensor = self.state_extractor.extract_comprehensive_state(row, turn_num)

                # Extract action labels safely
                action_label = self.state_extractor.extract_game_actions(row, turn_num)

                # Calculate outcome weight
                outcome_weight = self._calculate_safe_outcome_weight(
                    won, turn_num, num_turns, total_actions, row, turn_prefix
                )

                sample = {
                    'state_tensor': state_tensor.tolist(),
                    'action_label': action_label,
                    'outcome_weight': outcome_weight,
                    'game_id': game_id,
                    'turn_number': turn_num,
                    'total_turns': num_turns,
                    'on_play': on_play,
                    'won': won,
                    'expansion': expansion,
                    'file_source': file_path,
                    'extraction_method': 'safe_comprehensive_282d',
                }

                samples.append(sample)

            return samples

        except Exception as e:
            logger.warning(f"Error extracting decisions from game {game_id}: {e}")
            return []

    def _calculate_safe_outcome_weight(self, won: bool, turn_num: int, total_turns: int,
                                     total_actions: int, row: pd.Series, turn_prefix: str) -> float:
        """Calculate outcome weight with safe operations."""
        try:
            # Base weight from game outcome
            base_weight = 0.8 if won else 0.4

            # Turn importance (early turns more important)
            turn_importance = 1.0 - (turn_num / max(20, total_turns)) * 0.3

            # Action complexity bonus
            complexity_bonus = min(0.2, total_actions * 0.05)

            # Simple board complexity
            user_creatures = self.state_extractor._safe_float(row.get(f'{turn_prefix}eot_user_creatures_in_play', 0))
            oppo_creatures = self.state_extractor._safe_float(row.get(f'{turn_prefix}eot_oppo_creatures_in_play', 0))
            board_complexity = min(0.15, (user_creatures + oppo_creatures) * 0.03)

            # Game length consideration
            game_length_bonus = 0.1 if total_turns >= 12 else 0.0

            total_weight = base_weight * turn_importance + complexity_bonus + board_complexity + game_length_bonus

            return min(1.0, max(0.1, total_weight))  # Ensure weight is in valid range

        except Exception as e:
            logger.warning(f"Error calculating outcome weight: {e}")
            return 0.5  # Default weight

    def save_safe_dataset(self, output_path: str):
        """Save the memory-safe comprehensive dataset."""
        logger.info(f"💾 Saving {len(self.processed_samples)} safe comprehensive samples to {output_path}")

        training_data = {
            'metadata': {
                'source': 'safe_comprehensive_17lands_replay_data',
                'total_samples': len(self.processed_samples),
                'state_tensor_dim': 282,
                'num_action_types': 15,
                'created_by': 'safe_comprehensive_training.py',
                'data_quality': 'memory_safe_full_board_state',
                'processing_method': '282d_safe_multi_modal_extraction',
                'model_compatibility': 'comprehensive_transformer_model',
                'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'memory_safe': True,
            },
            'training_samples': self.processed_samples
        }

        Path(output_path).parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"✅ Saved safe comprehensive dataset: {output_path}")
        self._print_safe_statistics()

    def _print_safe_statistics(self):
        """Print memory-safe processing statistics."""
        if not self.processed_samples:
            return

        logger.info("📊 Safe Comprehensive Processing Statistics:")

        # Basic stats
        logger.info(f"  Total samples: {len(self.processed_samples)}")
        logger.info(f"  State tensor dimension: 282 (memory-safe multi-modal)")

        # Action distribution
        action_counts = [0] * 15
        outcome_weights = []
        turn_distribution = {}

        for sample in self.processed_samples:
            for i, action in enumerate(sample['action_label']):
                if action == 1:
                    action_counts[i] += 1

            outcome_weights.append(sample['outcome_weight'])
            turn = sample.get('turn_number', 1)
            turn_distribution[turn] = turn_distribution.get(turn, 0) + 1

        import numpy as np
        logger.info(f"  Average outcome weight: {np.mean(outcome_weights):.3f}")
        logger.info(f"  Action distribution: {action_counts}")
        logger.info(f"  Turn range: {min(turn_distribution.keys())}-{max(turn_distribution.keys())}")

def main():
    """Main memory-safe comprehensive training pipeline."""
    logger.info("🚀 Memory-Safe Comprehensive MTG AI Training Pipeline")
    logger.info("=" * 70)
    logger.info("🎯 Processing real 17Lands data with full 282-dimensional board state")
    logger.info("💾 Memory-safe mode enabled for your system upgrade")

    # Initialize processor with conservative memory limits
    processor = SafeComprehensiveDataProcessor(memory_limit_gb=6.0)  # Conservative limit

    # Process PremierDraft files with memory safety
    data_dir = Path('data/17lands_data')
    premierdraft_files = list(data_dir.glob('replay_data_public.*.PremierDraft.csv.gz'))

    logger.info(f"📁 Found {len(premierdraft_files)} PremierDraft files for safe comprehensive processing")

    # Limit to small sample for memory safety
    files_to_process = premierdraft_files[:1]  # Only 1 file for memory safety
    samples_per_file = 200  # Small target for memory safety

    total_samples = 0
    start_time = time.time()

    for i, file_path in enumerate(files_to_process):
        logger.info(f"\n📊 Processing safe comprehensive file {i+1}/{len(files_to_process)}: {file_path.name}")

        samples = processor.process_replay_file(str(file_path), max_games=samples_per_file)
        total_samples += samples

        logger.info(f"📈 Running total: {len(processor.processed_samples)} safe comprehensive samples")

        # Aggressive memory cleanup between files
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    processing_time = time.time() - start_time
    logger.info(f"\n🎉 Memory-safe comprehensive processing completed in {processing_time:.1f} seconds")
    logger.info(f"📊 Total safe comprehensive samples: {len(processor.processed_samples)}")

    if len(processor.processed_samples) >= 50:  # Minimum samples for training
        # Save comprehensive dataset
        output_file = f'data/safe_comprehensive_282d_training_data_{len(processor.processed_samples)}_samples.json'
        processor.save_safe_dataset(output_file)

        # Create datasets for training
        logger.info("🔧 Creating safe datasets for training...")
        dataset = SafeComprehensiveMTGDataset(processor.processed_samples)

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create small data loaders for memory safety
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,  # Very small batch size
            shuffle=True,
            num_workers=1,  # Single worker for memory safety
            pin_memory=False  # Disable pin_memory for memory safety
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )

        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")

        # Initialize comprehensive trainer with memory-safe settings
        trainer = ComprehensiveTrainer()

        # Train the comprehensive model safely
        try:
            logger.info("🧠 Starting memory-safe comprehensive model training...")
            training_history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,  # Fewer epochs for memory safety
                patience=5
            )

            logger.info("\n✅ Memory-safe comprehensive training completed successfully!")
            logger.info(f"📁 Model saved to: comprehensive_mtg_model.pth")
            logger.info(f"🧠 Model now processes full 282-dimensional board state with memory safety")

            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Memory-safe comprehensive training failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        logger.error(f"❌ Insufficient samples for training ({len(processor.processed_samples)} < 50)")

if __name__ == "__main__":
    main()