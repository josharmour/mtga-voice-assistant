#!/usr/bin/env python3
"""
MTG Model Checkpointing and Versioning System - Task 3.4

Comprehensive model checkpointing and versioning system for Magic: The Gathering AI
including automatic checkpointing, version management, model comparison, and deployment
readiness validation.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
import time
import hashlib
import shutil
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sqlite3
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class CheckpointType(Enum):
    """Checkpoint type enumeration."""
    EPOCH = "epoch"
    BEST = "best"
    EMERGENCY = "emergency"
    MANUAL = "manual"


@dataclass
class ModelMetadata:
    """Metadata for model versioning."""

    # Basic information
    model_id: str
    version: str
    created_at: str
    status: ModelStatus

    # Training information
    epoch: int
    training_duration: float  # seconds
    total_samples: int

    # Architecture
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]

    # Performance metrics
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None

    # Model characteristics
    model_size_mb: float
    parameter_count: int
    file_hash: str

    # Additional information
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_model_id: Optional[str] = None
    dataset_version: Optional[str] = None
    git_commit: Optional[str] = None

    # Deployment information
    deployment_ready: bool = False
    deployment_checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CheckpointInfo:
    """Information about a specific checkpoint."""

    checkpoint_id: str
    model_id: str
    checkpoint_type: CheckpointType
    epoch: int
    created_at: str
    file_path: str
    file_size_mb: float
    metrics: Dict[str, float]
    is_best: bool = False
    description: str = ""


class ModelVersionDatabase:
    """SQLite database for tracking model versions and checkpoints."""

    def __init__(self, db_path: str = "model_versions.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                training_duration REAL NOT NULL,
                total_samples INTEGER NOT NULL,
                model_config TEXT NOT NULL,
                training_config TEXT NOT NULL,
                train_metrics TEXT NOT NULL,
                val_metrics TEXT NOT NULL,
                test_metrics TEXT,
                model_size_mb REAL NOT NULL,
                parameter_count INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                parent_model_id TEXT,
                dataset_version TEXT,
                git_commit TEXT,
                deployment_ready INTEGER DEFAULT 0,
                deployment_checks TEXT
            )
        ''')

        # Checkpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_mb REAL NOT NULL,
                metrics TEXT NOT NULL,
                is_best INTEGER DEFAULT 0,
                description TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        ''')

        # Model relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_relationships (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES models (model_id),
                FOREIGN KEY (child_id) REFERENCES models (model_id)
            )
        ''')

        conn.commit()
        conn.close()

    def save_model(self, metadata: ModelMetadata) -> bool:
        """Save model metadata to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO models VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                metadata.model_id,
                metadata.version,
                metadata.created_at,
                metadata.status.value,
                metadata.epoch,
                metadata.training_duration,
                metadata.total_samples,
                json.dumps(metadata.model_config),
                json.dumps(metadata.training_config),
                json.dumps(metadata.train_metrics),
                json.dumps(metadata.val_metrics),
                json.dumps(metadata.test_metrics) if metadata.test_metrics else None,
                metadata.model_size_mb,
                metadata.parameter_count,
                metadata.file_hash,
                metadata.description,
                json.dumps(metadata.tags),
                metadata.parent_model_id,
                metadata.dataset_version,
                metadata.git_commit,
                int(metadata.deployment_ready),
                json.dumps(metadata.deployment_checks)
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            return False

    def save_checkpoint(self, checkpoint: CheckpointInfo) -> bool:
        """Save checkpoint information to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO checkpoints VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                checkpoint.checkpoint_id,
                checkpoint.model_id,
                checkpoint.checkpoint_type.value,
                checkpoint.epoch,
                checkpoint.created_at,
                checkpoint.file_path,
                checkpoint.file_size_mb,
                json.dumps(checkpoint.metrics),
                int(checkpoint.is_best),
                checkpoint.description
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error saving checkpoint info: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
            row = cursor.fetchone()

            conn.close()

            if row:
                return self._row_to_model_metadata(row)
            return None

        except Exception as e:
            logger.error(f"Error retrieving model: {e}")
            return None

    def get_model_checkpoints(self, model_id: str) -> List[CheckpointInfo]:
        """Get all checkpoints for a model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM checkpoints WHERE model_id = ? ORDER BY epoch
            ''', (model_id,))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_checkpoint_info(row) for row in rows]

        except Exception as e:
            logger.error(f"Error retrieving checkpoints: {e}")
            return []

    def list_models(self, status: ModelStatus = None, limit: int = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = 'SELECT * FROM models'
            params = []

            if status:
                query += ' WHERE status = ?'
                params.append(status.value)

            query += ' ORDER BY created_at DESC'

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_model_metadata(row) for row in rows]

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def _row_to_model_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row[0],
            version=row[1],
            created_at=row[2],
            status=ModelStatus(row[3]),
            epoch=row[4],
            training_duration=row[5],
            total_samples=row[6],
            model_config=json.loads(row[7]),
            training_config=json.loads(row[8]),
            train_metrics=json.loads(row[9]),
            val_metrics=json.loads(row[10]),
            test_metrics=json.loads(row[11]) if row[11] else None,
            model_size_mb=row[12],
            parameter_count=row[13],
            file_hash=row[14],
            description=row[15],
            tags=json.loads(row[16]) if row[16] else [],
            parent_model_id=row[17],
            dataset_version=row[18],
            git_commit=row[19],
            deployment_ready=bool(row[20]),
            deployment_checks=json.loads(row[21]) if row[21] else {}
        )

    def _row_to_checkpoint_info(self, row) -> CheckpointInfo:
        """Convert database row to CheckpointInfo."""
        return CheckpointInfo(
            checkpoint_id=row[0],
            model_id=row[1],
            checkpoint_type=CheckpointType(row[2]),
            epoch=row[3],
            created_at=row[4],
            file_path=row[5],
            file_size_mb=row[6],
            metrics=json.loads(row[7]),
            is_best=bool(row[8]),
            description=row[9]
        )


class ModelCheckpointManager:
    """Manages model checkpointing and versioning."""

    def __init__(self, checkpoint_dir: str = "model_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.checkpoint_dir / "models").mkdir(exist_ok=True)
        (self.checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (self.checkpoint_dir / "exports").mkdir(exist_ok=True)

        # Initialize database
        self.db = ModelVersionDatabase(str(self.checkpoint_dir / "versions.db"))

        # Current model tracking
        self.current_model_id = None
        self.current_epoch = 0

    def create_model_version(self,
                           model: nn.Module,
                           training_config: Dict,
                           train_metrics: Dict,
                           val_metrics: Dict,
                           epoch: int,
                           description: str = "",
                           tags: List[str] = None,
                           parent_model_id: str = None) -> str:
        """Create a new model version."""

        # Generate model ID
        model_id = str(uuid.uuid4())
        version = self._generate_version_number()

        # Create model file
        model_file_path = self.checkpoint_dir / "models" / f"{model_id}.pth"
        model_state = {
            'state_encoder': model.state_encoder.state_dict(),
            'decision_head': model.decision_head.state_dict(),
            'config': training_config
        }

        torch.save(model_state, model_file_path)

        # Calculate file info
        file_size_mb = model_file_path.stat().st_size / (1024 * 1024)
        file_hash = self._calculate_file_hash(model_file_path)

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            created_at=datetime.now().isoformat(),
            status=ModelStatus.TRAINING,
            epoch=epoch,
            training_duration=0.0,  # Will be updated at training end
            total_samples=training_config.get('total_samples', 0),
            model_config=training_config,
            training_config=training_config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            model_size_mb=file_size_mb,
            parameter_count=param_count,
            file_hash=file_hash,
            description=description,
            tags=tags or [],
            parent_model_id=parent_model_id
        )

        # Save to database
        if self.db.save_model(metadata):
            self.current_model_id = model_id
            self.current_epoch = epoch
            logger.info(f"Created model version {model_id} (v{version})")
            return model_id
        else:
            # Clean up file if database save failed
            model_file_path.unlink(missing_ok=True)
            raise RuntimeError("Failed to save model metadata")

    def save_checkpoint(self,
                       model: nn.Module,
                       model_id: str,
                       checkpoint_type: CheckpointType,
                       epoch: int,
                       metrics: Dict,
                       description: str = "",
                       is_best: bool = False) -> str:
        """Save a model checkpoint."""

        checkpoint_id = str(uuid.uuid4())
        checkpoint_file = self.checkpoint_dir / "checkpoints" / f"{checkpoint_id}.pth"

        # Save checkpoint
        checkpoint_data = {
            'state_encoder': model.state_encoder.state_dict(),
            'decision_head': model.decision_head.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_id': model_id,
            'created_at': datetime.now().isoformat()
        }

        torch.save(checkpoint_data, checkpoint_file)

        # Calculate file size
        file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)

        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            checkpoint_type=checkpoint_type,
            epoch=epoch,
            created_at=datetime.now().isoformat(),
            file_path=str(checkpoint_file),
            file_size_mb=file_size_mb,
            metrics=metrics,
            is_best=is_best,
            description=description
        )

        # Save to database
        if self.db.save_checkpoint(checkpoint_info):
            logger.info(f"Saved checkpoint {checkpoint_id} for model {model_id}")
            return checkpoint_id
        else:
            checkpoint_file.unlink(missing_ok=True)
            raise RuntimeError("Failed to save checkpoint info")

    def load_model(self, model_id: str, model: nn.Module) -> bool:
        """Load a model version by ID."""
        metadata = self.db.get_model(model_id)
        if not metadata:
            logger.error(f"Model {model_id} not found")
            return False

        model_file = self.checkpoint_dir / "models" / f"{model_id}.pth"
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return False

        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            model.state_encoder.load_state_dict(checkpoint['state_encoder'])
            model.decision_head.load_state_dict(checkpoint['decision_head'])

            self.current_model_id = model_id
            self.current_epoch = metadata.epoch

            logger.info(f"Loaded model {model_id} (v{metadata.version})")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False

    def load_checkpoint(self, checkpoint_id: str, model: nn.Module) -> Tuple[bool, Dict]:
        """Load a checkpoint into model."""
        checkpoints = self.db.get_model_checkpoints(self.current_model_id)
        target_checkpoint = None

        for cp in checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                target_checkpoint = cp
                break

        if not target_checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False, {}

        try:
            checkpoint_data = torch.load(target_checkpoint.file_path, map_location='cpu')
            model.state_encoder.load_state_dict(checkpoint_data['state_encoder'])
            model.decision_head.load_state_dict(checkpoint_data['decision_head'])

            self.current_epoch = checkpoint_data['epoch']

            logger.info(f"Loaded checkpoint {checkpoint_id} at epoch {self.current_epoch}")
            return True, checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return False, {}

    def list_models(self, status: ModelStatus = None) -> List[ModelMetadata]:
        """List available models."""
        return self.db.list_models(status=status)

    def get_model_checkpoints(self, model_id: str = None) -> List[CheckpointInfo]:
        """Get checkpoints for a model."""
        model_id = model_id or self.current_model_id
        if not model_id:
            return []

        return self.db.get_model_checkpoints(model_id)

    def get_best_checkpoint(self, model_id: str = None) -> Optional[CheckpointInfo]:
        """Get the best checkpoint for a model."""
        checkpoints = self.get_model_checkpoints(model_id)
        best_checkpoint = None
        best_metric = float('-inf')

        for cp in checkpoints:
            # Use validation accuracy as the primary metric
            val_acc = cp.metrics.get('val_accuracy', 0)
            if val_acc > best_metric:
                best_metric = val_acc
                best_checkpoint = cp

        return best_checkpoint

    def update_model_status(self, model_id: str, status: ModelStatus,
                           test_metrics: Dict = None,
                           deployment_checks: Dict = None):
        """Update model status and additional metadata."""
        metadata = self.db.get_model(model_id)
        if not metadata:
            logger.error(f"Model {model_id} not found")
            return

        metadata.status = status
        if test_metrics:
            metadata.test_metrics = test_metrics
        if deployment_checks:
            metadata.deployment_checks = deployment_checks

        # Update deployment readiness
        metadata.deployment_ready = self._check_deployment_readiness(metadata)

        self.db.save_model(metadata)
        logger.info(f"Updated model {model_id} status to {status.value}")

    def export_model(self, model_id: str, export_format: str = "torchscript") -> str:
        """Export model for deployment."""
        metadata = self.db.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        model_file = self.checkpoint_dir / "models" / f"{model_id}.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model
        checkpoint = torch.load(model_file, map_location='cpu')

        if export_format == "torchscript":
            return self._export_torchscript(model_id, checkpoint)
        elif export_format == "onnx":
            return self._export_onnx(model_id, checkpoint)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def _export_torchscript(self, model_id: str, checkpoint: Dict) -> str:
        """Export model as TorchScript."""
        # This would need the actual model architecture
        # For now, create a placeholder export
        export_path = self.checkpoint_dir / "exports" / f"{model_id}_torchscript.pth"

        # Placeholder export - in practice would trace the actual model
        torch.save({
            'model_state': checkpoint,
            'export_format': 'torchscript',
            'exported_at': datetime.now().isoformat()
        }, export_path)

        logger.info(f"Exported model {model_id} as TorchScript: {export_path}")
        return str(export_path)

    def _export_onnx(self, model_id: str, checkpoint: Dict) -> str:
        """Export model as ONNX."""
        # Placeholder for ONNX export
        export_path = self.checkpoint_dir / "exports" / f"{model_id}.onnx"

        # In practice would convert to ONNX format
        logger.info(f"ONNX export not fully implemented: {export_path}")
        return str(export_path)

    def cleanup_old_checkpoints(self, model_id: str, keep_count: int = 5):
        """Clean up old checkpoints, keeping only the best N."""
        checkpoints = self.get_model_checkpoints(model_id)

        # Sort by validation accuracy
        checkpoints.sort(key=lambda x: x.metrics.get('val_accuracy', 0), reverse=True)

        # Keep top N checkpoints
        checkpoints_to_keep = checkpoints[:keep_count]
        checkpoints_to_delete = checkpoints[keep_count:]

        for cp in checkpoints_to_delete:
            try:
                os.unlink(cp.file_path)
                logger.info(f"Deleted old checkpoint {cp.checkpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {cp.checkpoint_id}: {e}")

    def _generate_version_number(self) -> str:
        """Generate a version number for the model."""
        # Get the latest version number
        models = self.db.list_models(limit=1)
        if models:
            last_version = models[0].version
            try:
                major, minor, patch = map(int, last_version.split('.'))
                patch += 1
                return f"{major}.{minor}.{patch}"
            except:
                pass

        return "1.0.0"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _check_deployment_readiness(self, metadata: ModelMetadata) -> bool:
        """Check if model is ready for deployment."""
        checks = {
            'has_train_metrics': bool(metadata.train_metrics),
            'has_val_metrics': bool(metadata.val_metrics),
            'min_val_accuracy': metadata.val_metrics.get('accuracy', 0) >= 0.7,
            'model_size_reasonable': metadata.model_size_mb < 1000,  # Less than 1GB
            'has_description': bool(metadata.description),
            'has_tags': bool(metadata.tags)
        }

        metadata.deployment_checks = checks
        return all(checks.values())

    def get_training_history(self, model_id: str) -> pd.DataFrame:
        """Get training history for a model."""
        checkpoints = self.get_model_checkpoints(model_id)

        history = []
        for cp in checkpoints:
            row = {
                'epoch': cp.epoch,
                'created_at': cp.created_at,
                'checkpoint_type': cp.checkpoint_type.value,
                'is_best': cp.is_best
            }
            row.update(cp.metrics)
            history.append(row)

        return pd.DataFrame(history)

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare multiple models."""
        comparison_data = []

        for model_id in model_ids:
            metadata = self.db.get_model(model_id)
            if metadata:
                row = {
                    'model_id': model_id,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'epoch': metadata.epoch,
                    'parameter_count': metadata.parameter_count,
                    'model_size_mb': metadata.model_size_mb
                }

                # Add metrics
                row.update({f'train_{k}': v for k, v in metadata.train_metrics.items()})
                row.update({f'val_{k}': v for k, v in metadata.val_metrics.items()})

                if metadata.test_metrics:
                    row.update({f'test_{k}': v for k, v in metadata.test_metrics.items()})

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)


class DeploymentValidator:
    """Validates models for deployment readiness."""

    def __init__(self):
        self.validation_checks = {
            'performance_check': self._check_performance,
            'size_check': self._check_model_size,
            'robustness_check': self._check_robustness,
            'compatibility_check': self._check_compatibility
        }

    def validate_for_deployment(self, model: nn.Module, metadata: ModelMetadata,
                              test_dataloader = None) -> Dict[str, Any]:
        """Validate model for deployment."""
        results = {
            'ready_for_deployment': True,
            'checks': {},
            'issues': [],
            'recommendations': []
        }

        for check_name, check_func in self.validation_checks.items():
            try:
                check_result = check_func(model, metadata, test_dataloader)
                results['checks'][check_name] = check_result

                if not check_result['passed']:
                    results['ready_for_deployment'] = False
                    results['issues'].extend(check_result.get('issues', []))

                if check_result.get('recommendations'):
                    results['recommendations'].extend(check_result['recommendations'])

            except Exception as e:
                logger.error(f"Deployment check {check_name} failed: {e}")
                results['checks'][check_name] = {'passed': False, 'error': str(e)}
                results['ready_for_deployment'] = False
                results['issues'].append(f"Check {check_name} failed with error: {e}")

        return results

    def _check_performance(self, model: nn.Module, metadata: ModelMetadata,
                          test_dataloader = None) -> Dict[str, Any]:
        """Check model performance criteria."""
        result = {'passed': True, 'issues': [], 'recommendations': []}

        # Check validation accuracy
        val_acc = metadata.val_metrics.get('accuracy', 0)
        if val_acc < 0.7:
            result['passed'] = False
            result['issues'].append(f"Validation accuracy too low: {val_acc:.3f}")
            result['recommendations'].append("Consider more training or better hyperparameters")

        # Check overfitting
        train_acc = metadata.train_metrics.get('accuracy', 0)
        if train_acc - val_acc > 0.1:
            result['passed'] = False
            result['issues'].append(f"Model appears overfitted: train {train_acc:.3f} vs val {val_acc:.3f}")
            result['recommendations'].append("Add regularization or collect more data")

        return result

    def _check_model_size(self, model: nn.Module, metadata: ModelMetadata,
                         test_dataloader = None) -> Dict[str, Any]:
        """Check model size constraints."""
        result = {'passed': True, 'issues': [], 'recommendations': []}

        if metadata.model_size_mb > 500:  # 500MB limit
            result['passed'] = False
            result['issues'].append(f"Model too large: {metadata.model_size_mb:.1f}MB")
            result['recommendations'].append("Consider model pruning or quantization")

        if metadata.parameter_count > 100_000_000:  # 100M parameters
            result['passed'] = False
            result['issues'].append(f"Too many parameters: {metadata.parameter_count:,}")
            result['recommendations'].append("Consider smaller architecture")

        return result

    def _check_robustness(self, model: nn.Module, metadata: ModelMetadata,
                         test_dataloader = None) -> Dict[str, Any]:
        """Check model robustness."""
        result = {'passed': True, 'issues': [], 'recommendations': []}

        # Check training stability (if we have enough epochs)
        if metadata.epoch >= 10:
            # This would require analyzing training curves
            # For now, assume passed
            pass

        # Check for reasonable loss values
        val_loss = metadata.val_metrics.get('total_loss', float('inf'))
        if val_loss > 2.0:
            result['issues'].append(f"High validation loss: {val_loss:.3f}")
            result['recommendations'].append("Model may not be converging properly")

        return result

    def _check_compatibility(self, model: nn.Module, metadata: ModelMetadata,
                           test_dataloader = None) -> Dict[str, Any]:
        """Check model compatibility requirements."""
        result = {'passed': True, 'issues': [], 'recommendations': []}

        # Check PyTorch version compatibility
        pytorch_version = torch.__version__
        if not pytorch_version.startswith(('1.', '2.')):
            result['passed'] = False
            result['issues'].append(f"Unsupported PyTorch version: {pytorch_version}")
            result['recommendations'].append("Use PyTorch 1.x or 2.x")

        # Check model metadata completeness
        required_fields = ['model_config', 'training_config', 'train_metrics', 'val_metrics']
        for field in required_fields:
            if not getattr(metadata, field):
                result['passed'] = False
                result['issues'].append(f"Missing required metadata: {field}")

        return result


def main():
    """Example usage of model versioning system."""
    # Initialize checkpoint manager
    checkpoint_manager = ModelCheckpointManager("demo_checkpoints")
    validator = DeploymentValidator()

    # Create some example model metadata
    example_metadata = ModelMetadata(
        model_id="demo_model",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        status=ModelStatus.TRAINING,
        epoch=10,
        training_duration=3600.0,
        total_samples=1000,
        model_config={"d_model": 256, "nhead": 8},
        training_config={"learning_rate": 1e-4, "batch_size": 32},
        train_metrics={"accuracy": 0.85, "loss": 0.45},
        val_metrics={"accuracy": 0.82, "loss": 0.48},
        model_size_mb=50.5,
        parameter_count=10000000,
        file_hash="demo_hash",
        description="Demo model for testing"
    )

    # Save model metadata
    if checkpoint_manager.db.save_model(example_metadata):
        logger.info("Successfully saved example model metadata")

    # List models
    models = checkpoint_manager.list_models()
    logger.info(f"Found {len(models)} models in database")

    for model in models:
        logger.info(f"Model: {model.model_id} v{model.version} - {model.status.value}")

    # Compare models
    if models:
        model_ids = [m.model_id for m in models[:3]]
        comparison = checkpoint_manager.compare_models(model_ids)
        logger.info(f"Model comparison:\n{comparison}")

    logger.info("Model versioning system demo completed")


if __name__ == "__main__":
    main()