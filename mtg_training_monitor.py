#!/usr/bin/env python3
"""
MTG Training Visualization and Monitoring Tools - Task 3.4

Comprehensive monitoring and visualization system for Magic: The Gathering AI training
including real-time progress tracking, learning curves, performance metrics, and interactive
dashboards.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
import sys
import time
import threading
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logging.warning("TensorBoard not available. Install with: pip install tensorboard")

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logging.warning("Weights & Biases not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring."""

    # Real-time monitoring
    enable_real_time: bool = True
    update_interval: int = 10  # seconds
    smoothing_window: int = 20

    # Visualization settings
    save_plots: bool = True
    plot_dir: str = "training_plots"
    interactive_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"

    # TensorBoard settings
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"

    # Weights & Biases settings
    use_wandb: bool = False
    wandb_project: str = "mtg-ai-training"
    wandb_entity: str = None

    # Metrics to track
    track_losses: bool = True
    track_accuracy: bool = True
    track_learning_rate: bool = True
    track_gradients: bool = True
    track_model_weights: bool = True
    track_decision_quality: bool = True

    # Alert settings
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_accuracy': 0.5,
        'max_loss_increase': 0.1,
        'min_lr': 1e-7,
        'max_grad_norm': 10.0
    })

    # Performance monitoring
    monitor_gpu_memory: bool = True
    monitor_training_time: bool = True
    log_system_metrics: bool = True


class MetricsCollector:
    """Collects and stores training metrics."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.timestamps = []
        self.start_time = None
        self.epoch_times = []

        # Smoothing buffers
        self.smoothed_metrics = defaultdict(lambda: deque(maxlen=config.smoothing_window))

        # Performance tracking
        self.gpu_memory_usage = []
        self.training_times = []

    def start_training(self):
        """Initialize metrics collection at start of training."""
        self.start_time = time.time()
        self.timestamps = [datetime.now()]

    def log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                         additional_metrics: Dict = None):
        """Log metrics for an epoch."""
        timestamp = datetime.now()
        self.timestamps.append(timestamp)

        # Log training metrics
        for key, value in train_metrics.items():
            self.metrics[f"train_{key}"].append(value)
            self.smoothed_metrics[f"train_{key}"].append(value)

        # Log validation metrics
        for key, value in val_metrics.items():
            self.metrics[f"val_{key}"].append(value)
            self.smoothed_metrics[f"val_{key}"].append(value)

        # Log additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.metrics[key].append(value)
                self.smoothed_metrics[key].append(value)

        # Log epoch time
        if len(self.epoch_times) > 0:
            epoch_time = timestamp - self.epoch_times[-1]
            self.metrics['epoch_time'].append(epoch_time.total_seconds())

        self.epoch_times.append(timestamp)

    def log_batch_metrics(self, batch_idx: int, batch_metrics: Dict):
        """Log metrics for a single batch."""
        for key, value in batch_metrics.items():
            self.metrics[f"batch_{key}"].append(value)

    def get_smoothed_metric(self, metric_name: str) -> List[float]:
        """Get smoothed version of a metric."""
        if metric_name in self.smoothed_metrics:
            return list(self.smoothed_metrics[metric_name])
        return []

    def get_metric_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame."""
        data = {
            'timestamp': self.timestamps,
            'epoch': list(range(len(self.timestamps)))
        }

        # Add all metrics
        for key, values in self.metrics.items():
            if len(values) == len(self.timestamps):
                data[key] = values
            elif len(values) > len(self.timestamps):
                data[key] = values[:len(self.timestamps)]
            else:
                data[key] = values + [np.nan] * (len(self.timestamps) - len(values))

        return pd.DataFrame(data)

    def check_alerts(self) -> List[str]:
        """Check for alert conditions."""
        alerts = []

        if not self.config.enable_alerts:
            return alerts

        # Check accuracy threshold
        if 'val_accuracy' in self.metrics and self.metrics['val_accuracy']:
            current_acc = self.metrics['val_accuracy'][-1]
            if current_acc < self.config.alert_thresholds['min_accuracy']:
                alerts.append(f"Low validation accuracy: {current_acc:.3f}")

        # Check loss increase
        if 'val_total_loss' in self.metrics and len(self.metrics['val_total_loss']) > 5:
            recent_losses = self.metrics['val_total_loss'][-5:]
            if len(recent_losses) >= 2:
                loss_increase = recent_losses[-1] - recent_losses[0]
                if loss_increase > self.config.alert_thresholds['max_loss_increase']:
                    alerts.append(f"Validation loss increasing: {loss_increase:.3f}")

        # Check learning rate
        if 'learning_rate' in self.metrics and self.metrics['learning_rate']:
            current_lr = self.metrics['learning_rate'][-1]
            if current_lr < self.config.alert_thresholds['min_lr']:
                alerts.append(f"Learning rate too low: {current_lr:.2e}")

        return alerts


class TensorBoardLogger:
    """TensorBoard logging functionality."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.writer = None

        if config.use_tensorboard and HAS_TENSORBOARD:
            log_dir = os.path.join(config.tensorboard_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging initialized: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = ""):
        """Log multiple scalar values."""
        if self.writer:
            for key, value in scalars.items():
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_learning_curve(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log learning curve data."""
        if self.writer:
            # Training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)

            # Validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)

    def log_model_graph(self, model, input_tensor):
        """Log model computation graph."""
        if self.writer:
            try:
                self.writer.add_graph(model, input_tensor)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")

    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logging functionality."""

    def __init__(self, config: MonitoringConfig, project_config: Dict = None):
        self.config = config
        self.run = None

        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=project_config or {},
                tags=["mtg-ai", "transformer"],
                reinit=True
            )
            self.run = wandb.run
            logger.info(f"W&B logging initialized: {self.run.url}")

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        """Log metrics to W&B."""
        if self.run:
            log_dict = {}
            for key, value in metrics.items():
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = value

            self.run.log(log_dict, step=step)

    def log_learning_curve(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log learning curve data."""
        if self.run:
            combined_metrics = {}
            for key, value in train_metrics.items():
                combined_metrics[f"train/{key}"] = value
            for key, value in val_metrics.items():
                combined_metrics[f"val/{key}"] = value

            self.run.log(combined_metrics, step=epoch)

    def log_model_checkpoint(self, model_path: str, epoch: int):
        """Log model checkpoint."""
        if self.run:
            artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Finish W&B run."""
        if self.run:
            self.run.finish()


class TrainingVisualizer:
    """Create training visualizations and plots."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.plot_dir = config.plot_dir

        # Create plot directory
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_learning_curves(self, metrics_collector: MetricsCollector, save: bool = True):
        """Create learning curve plots."""
        df = metrics_collector.get_metric_dataframe()

        if df.empty:
            logger.warning("No metrics data available for plotting")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MTG AI Training Progress', fontsize=16)

        # Loss curves
        if 'train_total_loss' in df.columns and 'val_total_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train_total_loss'], label='Training Loss', alpha=0.7)
            axes[0, 0].plot(df['epoch'], df['val_total_loss'], label='Validation Loss', alpha=0.7)

            # Add smoothed curves
            train_smooth = metrics_collector.get_smoothed_metric('train_total_loss')
            val_smooth = metrics_collector.get_smoothed_metric('val_total_loss')
            if train_smooth and val_smooth:
                axes[0, 0].plot(df['epoch'][:len(train_smooth)], train_smooth,
                               label='Training Loss (Smoothed)', linewidth=2)
                axes[0, 0].plot(df['epoch'][:len(val_smooth)], val_smooth,
                               label='Validation Loss (Smoothed)', linewidth=2)

            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        if 'train_accuracy' in df.columns and 'val_accuracy' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Training Accuracy', alpha=0.7)
            axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', alpha=0.7)

            # Add smoothed curves
            train_smooth = metrics_collector.get_smoothed_metric('train_accuracy')
            val_smooth = metrics_collector.get_smoothed_metric('val_accuracy')
            if train_smooth and val_smooth:
                axes[0, 1].plot(df['epoch'][:len(train_smooth)], train_smooth,
                               label='Training Accuracy (Smoothed)', linewidth=2)
                axes[0, 1].plot(df['epoch'][:len(val_smooth)], val_smooth,
                               label='Validation Accuracy (Smoothed)', linewidth=2)

            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)

        # Learning rate schedule
        if 'learning_rate' in df.columns:
            axes[1, 0].plot(df['epoch'], df['learning_rate'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Training time per epoch
        if 'epoch_time' in df.columns:
            axes[1, 1].plot(df['epoch'][1:], df['epoch_time'][1:])  # Skip first point
            axes[1, 1].set_title('Training Time per Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.plot_dir, f'learning_curves.{self.config.plot_format}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")

        plt.show()

    def create_interactive_dashboard(self, metrics_collector: MetricsCollector):
        """Create interactive Plotly dashboard."""
        df = metrics_collector.get_metric_dataframe()

        if df.empty:
            logger.warning("No metrics data available for interactive dashboard")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 'Learning Rate', 'Training Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Loss curves
        if 'train_total_loss' in df.columns and 'val_total_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_total_loss'],
                          mode='lines', name='Training Loss', line=dict(width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val_total_loss'],
                          mode='lines', name='Validation Loss', line=dict(width=2)),
                row=1, col=1
            )

        # Accuracy curves
        if 'train_accuracy' in df.columns and 'val_accuracy' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_accuracy'],
                          mode='lines', name='Training Accuracy', line=dict(width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val_accuracy'],
                          mode='lines', name='Validation Accuracy', line=dict(width=2)),
                row=1, col=2
            )

        # Learning rate
        if 'learning_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['learning_rate'],
                          mode='lines', name='Learning Rate', line=dict(width=2)),
                row=2, col=1
            )

        # Training time
        if 'epoch_time' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'][1:], y=df['epoch_time'][1:],
                          mode='lines+markers', name='Epoch Time', line=dict(width=2)),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title='MTG AI Training Dashboard',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Update y-axes
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type="log")
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)

        # Save interactive plot
        if self.config.interactive_plots:
            dashboard_path = os.path.join(self.plot_dir, 'training_dashboard.html')
            fig.write_html(dashboard_path)
            logger.info(f"Interactive dashboard saved to {dashboard_path}")

        fig.show()

    def create_confusion_matrix_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      class_names: List[str] = None, save: bool = True):
        """Create confusion matrix heatmap."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.plot_dir, f'confusion_matrix.{self.config.plot_format}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def create_gradient_norm_plot(self, gradient_norms: List[float], save: bool = True):
        """Create gradient norm monitoring plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(gradient_norms, alpha=0.7)
        plt.title('Gradient Norms During Training')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)

        # Add threshold line
        if hasattr(self.config, 'alert_thresholds') and 'max_grad_norm' in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds['max_grad_norm']
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
            plt.legend()

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.plot_dir, f'gradient_norms.{self.config.plot_format}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gradient norms plot saved to {save_path}")

        plt.show()


class TrainingMonitor:
    """Main training monitoring orchestrator."""

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.visualizer = TrainingVisualizer(self.config)

        # Initialize loggers
        self.tb_logger = TensorBoardLogger(self.config) if self.config.use_tensorboard else None
        self.wandb_logger = WandBLogger(self.config) if self.config.use_wandb else None

        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = False

    def start_training(self, model_config: Dict = None):
        """Initialize monitoring at start of training."""
        self.metrics_collector.start_training()

        # Initialize W&B with model config
        if self.wandb_logger and model_config:
            self.wandb_logger.run.config.update(model_config)

        # Start real-time monitoring thread
        if self.config.enable_real_time:
            self.start_real_time_monitoring()

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                  additional_metrics: Dict = None):
        """Log metrics for an epoch."""
        # Collect metrics
        self.metrics_collector.log_epoch_metrics(epoch, train_metrics, val_metrics, additional_metrics)

        # Log to external services
        if self.tb_logger:
            self.tb_logger.log_learning_curve(epoch, train_metrics, val_metrics)

        if self.wandb_logger:
            self.wandb_logger.log_learning_curve(epoch, train_metrics, val_metrics)

        # Check for alerts
        alerts = self.metrics_collector.check_alerts()
        for alert in alerts:
            logger.warning(f"Training Alert: {alert}")

    def log_batch(self, batch_idx: int, batch_metrics: Dict):
        """Log metrics for a batch."""
        self.metrics_collector.log_batch_metrics(batch_idx, batch_metrics)

    def log_gradients(self, model, step: int):
        """Log gradient statistics."""
        if not self.config.track_gradients:
            return

        total_norm = 0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Log individual parameter gradients to TensorBoard
                if self.tb_logger and step % 100 == 0:  # Log every 100 steps
                    self.tb_logger.log_histogram(f"gradients/{name}", param.grad.data, step)

        total_norm = total_norm ** (1. / 2)

        # Log total gradient norm
        if self.tb_logger:
            self.tb_logger.log_scalar("gradients/total_norm", total_norm, step)

        if self.wandb_logger:
            self.wandb_logger.log_metrics({"gradients/total_norm": total_norm}, step)

        # Add to metrics collector
        self.metrics_collector.metrics['gradient_norm'].append(total_norm)

    def log_model_weights(self, model, step: int):
        """Log model weight statistics."""
        if not self.config.track_model_weights:
            return

        for name, param in model.named_parameters():
            if param.data is not None:
                # Log weight statistics
                weight_stats = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }

                if self.tb_logger and step % 100 == 0:
                    self.tb_logger.log_histogram(f"weights/{name}", param.data, step)
                    for stat_name, stat_value in weight_stats.items():
                        self.tb_logger.log_scalar(f"weights/{name}/{stat_name}", stat_value, step)

    def create_plots(self):
        """Create all visualization plots."""
        logger.info("Creating training visualization plots...")

        # Learning curves
        self.visualizer.create_learning_curves(self.metrics_collector)

        # Interactive dashboard
        if self.config.interactive_plots:
            self.visualizer.create_interactive_dashboard(self.metrics_collector)

        # Gradient norms plot
        if 'gradient_norm' in self.metrics_collector.metrics:
            self.visualizer.create_gradient_norm_plot(
                self.metrics_collector.metrics['gradient_norm']
            )

        logger.info("Visualization plots created")

    def start_real_time_monitoring(self):
        """Start real-time monitoring thread."""
        def monitor_loop():
            while not self.stop_monitoring:
                try:
                    # Update real-time plots
                    if len(self.metrics_collector.metrics) > 0:
                        self.visualizer.create_learning_curves(
                            self.metrics_collector, save=False
                        )

                    time.sleep(self.config.update_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Real-time monitoring started")

    def stop_training(self, final_model_path: str = None):
        """Stop monitoring and finalize."""
        logger.info("Stopping training monitoring...")

        # Stop monitoring thread
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        # Create final plots
        self.create_plots()

        # Log final model checkpoint
        if final_model_path and self.wandb_logger:
            self.wandb_logger.log_model_checkpoint(final_model_path,
                                                 len(self.metrics_collector.timestamps))

        # Close loggers
        if self.tb_logger:
            self.tb_logger.close()

        if self.wandb_logger:
            self.wandb_logger.finish()

        logger.info("Training monitoring completed")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        df = self.metrics_collector.get_metric_dataframe()

        if df.empty:
            return {"error": "No training data available"}

        summary = {
            'total_epochs': len(df),
            'training_duration': None,
            'final_metrics': {},
            'best_metrics': {},
            'training_stability': {}
        }

        # Training duration
        if len(self.metrics_collector.timestamps) > 1:
            duration = self.metrics_collector.timestamps[-1] - self.metrics_collector.timestamps[0]
            summary['training_duration'] = str(duration)

        # Final metrics
        for column in df.columns:
            if column.startswith(('train_', 'val_')) and df[column].notna().any():
                summary['final_metrics'][column] = df[column].iloc[-1]

        # Best metrics
        if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
            best_acc_idx = df['val_accuracy'].idxmax()
            summary['best_metrics']['best_accuracy'] = df.loc[best_acc_idx, 'val_accuracy']
            summary['best_metrics']['best_accuracy_epoch'] = df.loc[best_acc_idx, 'epoch']

        if 'val_total_loss' in df.columns and df['val_total_loss'].notna().any():
            best_loss_idx = df['val_total_loss'].idxmin()
            summary['best_metrics']['best_loss'] = df.loc[best_loss_idx, 'val_total_loss']
            summary['best_metrics']['best_loss_epoch'] = df.loc[best_loss_idx, 'epoch']

        # Training stability (variance in recent metrics)
        recent_window = min(10, len(df))
        if recent_window > 1:
            recent_df = df.tail(recent_window)

            if 'val_accuracy' in recent_df.columns:
                acc_variance = recent_df['val_accuracy'].var()
                summary['training_stability']['accuracy_variance'] = acc_variance

            if 'val_total_loss' in recent_df.columns:
                loss_variance = recent_df['val_total_loss'].var()
                summary['training_stability']['loss_variance'] = loss_variance

        return summary


def main():
    """Example usage of training monitoring."""
    # Configuration
    config = MonitoringConfig(
        enable_real_time=False,  # Disable for demo
        save_plots=True,
        plot_dir="demo_monitoring",
        use_tensorboard=False,
        use_wandb=False
    )

    # Initialize monitor
    monitor = TrainingMonitor(config)

    # Simulate training
    monitor.start_training({"model_type": "transformer", "d_model": 256})

    for epoch in range(5):
        # Simulate metrics
        train_metrics = {
            'total_loss': 1.0 - epoch * 0.1 + np.random.normal(0, 0.05),
            'accuracy': 0.5 + epoch * 0.08 + np.random.normal(0, 0.02),
            'action_loss': 0.8 - epoch * 0.08 + np.random.normal(0, 0.03),
            'value_loss': 0.2 - epoch * 0.02 + np.random.normal(0, 0.01)
        }

        val_metrics = {
            'total_loss': 1.1 - epoch * 0.09 + np.random.normal(0, 0.04),
            'accuracy': 0.45 + epoch * 0.09 + np.random.normal(0, 0.03),
            'action_loss': 0.85 - epoch * 0.07 + np.random.normal(0, 0.04),
            'value_loss': 0.25 - epoch * 0.02 + np.random.normal(0, 0.01)
        }

        additional_metrics = {
            'learning_rate': 1e-4 * (0.95 ** epoch)
        }

        monitor.log_epoch(epoch, train_metrics, val_metrics, additional_metrics)

        # Simulate batch metrics
        for batch in range(10):
            batch_metrics = {
                'batch_loss': 1.0 - epoch * 0.1 + np.random.normal(0, 0.1)
            }
            monitor.log_batch(batch, batch_metrics)

    # Create plots and summary
    monitor.create_plots()
    summary = monitor.get_training_summary()

    print("Training Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    monitor.stop_training()


if __name__ == "__main__":
    main()