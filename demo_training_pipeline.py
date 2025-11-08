#!/usr/bin/env python3
"""
MTG AI Training Pipeline Demo - Task 3.4

Demonstration script showing the complete training pipeline integration.
This script showcases all components working together for MTG AI training.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
try:
    from mtg_training_pipeline import MTGTrainer, TrainingConfig
    from mtg_evaluation_metrics import MTGEvaluator, EvaluationConfig
    from mtg_training_monitor import TrainingMonitor, MonitoringConfig
    from mtg_model_versioning import ModelCheckpointManager
    from mtg_hyperparameter_optimization import HyperparameterOptimizer, HyperparameterConfig
except ImportError as e:
    logging.error(f"Could not import pipeline components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_training_data(num_samples: int = 100) -> str:
    """Create demo training data for demonstration."""
    import json

    # Create synthetic training data
    samples = []

    # Decision types for curriculum learning
    decision_types = [
        "Mana_Acceleration", "Land_Play", "Basic_Combat",
        "Card_Playing", "Removal_Usage", "Counter_Play",
        "Complex_Combat", "Multi_Target_Combat", "Block_Decision",
        "Combo_Play", "Bluff_Action", "Resource_Optimization", "Strategic_Sequencing"
    ]

    for i in range(num_samples):
        # Create synthetic state tensor (282 dimensions)
        state_tensor = np.random.randn(282).astype(np.float32)

        # Create synthetic action label (16 dimensions)
        action_label = np.zeros(16, dtype=int)
        action_idx = np.random.randint(0, 16)
        action_label[action_idx] = 1

        # Create synthetic outcome weight
        outcome_weight = np.random.uniform(0.1, 1.0)

        # Create synthetic game outcome
        game_outcome = np.random.random() > 0.5

        # Create synthetic strategic context
        strategic_context = {
            "life_differential": np.random.uniform(-20, 20),
            "board_advantage": np.random.randint(-5, 5),
            "card_advantage": np.random.randint(-10, 10),
            "game_phase": np.random.choice(["early", "mid", "late"]),
            "pressure_level": np.random.choice(["low", "balanced", "high"])
        }

        # Create weight components
        weight_components = {
            "base_outcome": 1.0,
            "strategic_importance": outcome_weight,
            "decision_rarity": np.random.uniform(0.5, 1.5),
            "game_impact": np.random.uniform(0.8, 1.2)
        }

        sample = {
            "tensor_data": state_tensor.tolist(),
            "action_label": action_label.tolist(),
            "outcome_weight": outcome_weight,
            "decision_type": np.random.choice(decision_types),
            "turn": np.random.randint(1, 15),
            "game_outcome": game_outcome,
            "strategic_context": strategic_context,
            "weight_components": weight_components
        }

        samples.append(sample)

    # Create metadata
    data = {
        "metadata": {
            "total_samples": num_samples,
            "final_tensor_dimension": 282,
            "pipeline_version": "2.4",
            "demo_data": True
        },
        "samples": samples
    }

    # Save to file
    demo_data_path = "demo_training_data.json"
    with open(demo_data_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created demo training data with {num_samples} samples: {demo_data_path}")
    return demo_data_path


def demo_basic_training():
    """Demonstrate basic training functionality."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Training Pipeline")
    logger.info("=" * 60)

    # Create demo data
    data_path = create_demo_training_data(100)

    # Configure training
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=3,  # Short demo
        curriculum_enabled=True,
        mixed_precision=True,
        save_every=1
    )

    # Initialize trainer
    trainer = MTGTrainer(config)

    # Configure monitoring
    monitor_config = MonitoringConfig(
        enable_real_time=False,  # Disable for demo
        save_plots=True,
        plot_dir="demo_plots",
        use_tensorboard=False
    )
    monitor = TrainingMonitor(monitor_config)

    # Start monitoring
    monitor.start_training(config.__dict__)

    try:
        # Train model
        train_metrics, val_metrics = trainer.train(data_path)

        # Log final metrics
        monitor.log_epoch(
            trainer.current_epoch,
            train_metrics,
            val_metrics,
            {"learning_rate": config.learning_rate}
        )

        logger.info("Basic training completed successfully!")
        logger.info(f"Final training accuracy: {train_metrics.get('accuracy', 'N/A')}")
        logger.info(f"Final validation accuracy: {val_metrics.get('accuracy', 'N/A')}")

    except Exception as e:
        logger.error(f"Training failed: {e}")

    finally:
        # Stop monitoring
        monitor.stop_training()

        # Clean up demo data
        if os.path.exists(data_path):
            os.remove(data_path)


def demo_evaluation():
    """Demonstrate model evaluation."""
    logger.info("=" * 60)
    logger.info("DEMO: Model Evaluation")
    logger.info("=" * 60)

    # Create demo data
    data_path = create_demo_training_data(50)

    # Configure and train a simple model
    config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-4,
        max_epochs=2,  # Very short for demo
        curriculum_enabled=False
    )

    trainer = MTGTrainer(config)

    try:
        # Train model
        trainer.train(data_path)

        # Configure evaluation
        eval_config = EvaluationConfig(
            save_results=True,
            output_dir="demo_evaluation",
            compute_calibration=True,
            analyze_by_decision_type=True,
            create_confusion_matrix=True
        )

        # Evaluate model
        evaluator = MTGEvaluator(eval_config)
        results = evaluator.evaluate_model(trainer, trainer.val_loader, trainer.device)

        # Display results
        logger.info("Evaluation completed!")
        logger.info(f"Overall accuracy: {results['overall_metrics'].get('accuracy', 'N/A')}")
        logger.info(f"F1-score (weighted): {results['overall_metrics'].get('f1_weighted', 'N/A')}")
        logger.info(f"Calibration error: {results['calibration_metrics'].get('expected_calibration_error', 'N/A')}")

        # Performance assessment
        assessment = results.get('performance_assessment', {})
        logger.info(f"Meets requirements: {assessment.get('meets_requirements', 'N/A')}")

        if assessment.get('strengths'):
            logger.info("Strengths:")
            for strength in assessment['strengths']:
                logger.info(f"  - {strength}")

        if assessment.get('issues'):
            logger.warning("Issues:")
            for issue in assessment['issues']:
                logger.warning(f"  - {issue}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    finally:
        # Clean up
        if os.path.exists(data_path):
            os.remove(data_path)


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization."""
    logger.info("=" * 60)
    logger.info("DEMO: Hyperparameter Optimization")
    logger.info("=" * 60)

    # Create demo data
    data_path = create_demo_training_data(30)

    # Configure optimization
    hyperopt_config = HyperparameterConfig(
        optimization_method="random",  # Use random search for demo
        n_trials=3,  # Very small for demo
        timeout_seconds=120,  # 2 minutes
        objective_metric="val_accuracy",
        objective_direction="maximize"
    )

    # Base configuration
    base_config = TrainingConfig(
        batch_size=8,
        max_epochs=2,  # Short training for optimization
        curriculum_enabled=False
    )

    eval_config = EvaluationConfig(save_results=False)

    try:
        # Run optimization
        optimizer = HyperparameterOptimizer(hyperopt_config)
        results = optimizer.optimize(data_path, base_config, eval_config)

        # Display results
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best parameters: {results.get('best_params', {})}")
        logger.info(f"Best objective value: {results.get('best_objective_value', 'N/A')}")
        logger.info(f"Total trials: {results.get('total_trials', 'N/A')}")

        # Parameter importance
        importance = results.get('parameter_importance', {})
        if importance:
            logger.info("Parameter importance:")
            for param, score in importance.items():
                logger.info(f"  {param}: {score:.3f}")

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")

    finally:
        # Clean up
        if os.path.exists(data_path):
            os.remove(data_path)


def demo_model_versioning():
    """Demonstrate model versioning and checkpointing."""
    logger.info("=" * 60)
    logger.info("DEMO: Model Versioning and Checkpointing")
    logger.info("=" * 60)

    # Initialize checkpoint manager
    checkpoint_manager = ModelCheckpointManager("demo_versioning")

    # Create demo data
    data_path = create_demo_training_data(20)

    # Configure training
    config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-4,
        max_epochs=3,
        checkpoint_dir="demo_versioning/checkpoints"
    )

    trainer = MTGTrainer(config)

    try:
        # Train model
        trainer.train(data_path)

        # Save model version
        model_id = checkpoint_manager.create_model_version(
            model=trainer,
            training_config=config.__dict__,
            train_metrics={"accuracy": 0.75, "loss": 0.5},
            val_metrics={"accuracy": 0.72, "loss": 0.55},
            epoch=trainer.current_epoch,
            description="Demo model for versioning",
            tags=["demo", "transformer", "mtg-ai"]
        )

        logger.info(f"Created model version: {model_id}")

        # List models
        models = checkpoint_manager.list_models()
        logger.info(f"Total models in database: {len(models)}")

        for model in models[:3]:  # Show first 3
            logger.info(f"  Model {model.model_id[:8]}... v{model.version} - {model.status.value}")

        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=trainer,
            model_id=model_id,
            checkpoint_type="epoch",
            epoch=trainer.current_epoch,
            metrics={"val_accuracy": 0.72, "val_loss": 0.55},
            description="Demo checkpoint"
        )

        logger.info(f"Created checkpoint: {checkpoint_id}")

        # Get best checkpoint
        best_checkpoint = checkpoint_manager.get_best_checkpoint(model_id)
        if best_checkpoint:
            logger.info(f"Best checkpoint: {best_checkpoint.checkpoint_id[:8]}... "
                       f"at epoch {best_checkpoint.epoch}")

        # Compare models
        if len(models) >= 2:
            model_ids = [m.model_id for m in models[:2]]
            comparison = checkpoint_manager.compare_models(model_ids)
            logger.info("Model comparison:")
            for _, row in comparison.iterrows():
                logger.info(f"  Model {row['model_id'][:8]}... - "
                           f"Val Acc: {row.get('val_accuracy', 'N/A')}")

    except Exception as e:
        logger.error(f"Model versioning demo failed: {e}")

    finally:
        # Clean up
        if os.path.exists(data_path):
            os.remove(data_path)


def demo_complete_pipeline():
    """Demonstrate complete training pipeline integration."""
    logger.info("=" * 60)
    logger.info("DEMO: Complete Training Pipeline Integration")
    logger.info("=" * 60)

    # Create demo data
    data_path = create_demo_training_data(50)

    # Initialize all components
    checkpoint_manager = ModelCheckpointManager("pipeline_demo")

    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=3,
        curriculum_enabled=True,
        mixed_precision=True,
        checkpoint_dir="pipeline_demo/checkpoints"
    )

    monitor_config = MonitoringConfig(
        enable_real_time=False,
        save_plots=True,
        plot_dir="pipeline_demo/plots",
        use_tensorboard=False
    )

    monitor = TrainingMonitor(monitor_config)
    eval_config = EvaluationConfig(save_results=False)

    trainer = MTGTrainer(config)

    try:
        # Start complete pipeline
        logger.info("Starting complete training pipeline...")

        # Start monitoring
        monitor.start_training(config.__dict__)

        # Training loop with comprehensive tracking
        for epoch in range(config.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.max_epochs}")

            # Train one epoch
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate_epoch()

            # Log to monitor
            monitor.log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=trainer,
                model_id="demo_pipeline",
                checkpoint_type="epoch",
                epoch=epoch,
                metrics=val_metrics,
                description=f"Epoch {epoch + 1} checkpoint"
            )

            logger.info(f"  Train Loss: {train_metrics.get('total_loss', 'N/A'):.4f}")
            logger.info(f"  Val Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")

        # Save final model version
        final_metrics = {
            'accuracy': val_metrics.get('accuracy', 0),
            'loss': val_metrics.get('total_loss', 0),
            'f1_score': val_metrics.get('f1_score', 0)
        }

        model_id = checkpoint_manager.create_model_version(
            model=trainer,
            training_config=config.__dict__,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=config.max_epochs,
            description="Complete pipeline demo model",
            tags=["demo", "complete-pipeline", "final"]
        )

        logger.info(f"Final model saved: {model_id}")

        # Final evaluation
        evaluator = MTGEvaluator(eval_config)
        results = evaluator.evaluate_model(trainer, trainer.val_loader, trainer.device)

        # Generate training summary
        summary = monitor.get_training_summary()
        logger.info("Training Summary:")
        for key, value in summary.items():
            if key != 'training_stability':  # Skip detailed stability metrics
                logger.info(f"  {key}: {value}")

        # Performance highlights
        logger.info("Performance Highlights:")
        logger.info(f"  Final validation accuracy: {results['overall_metrics'].get('accuracy', 'N/A'):.3f}")
        logger.info(f"  Final F1-score: {results['overall_metrics'].get('f1_weighted', 'N/A'):.3f}")
        logger.info(f"  Model size: {summary.get('model_size_mb', 'N/A')} MB")

        # Generate final plots
        monitor.create_plots()

        logger.info("Complete pipeline demo finished successfully!")

    except Exception as e:
        logger.error(f"Complete pipeline demo failed: {e}")

    finally:
        # Cleanup
        monitor.stop_training()
        if os.path.exists(data_path):
            os.remove(data_path)


def main():
    """Run all demonstrations."""
    logger.info("MTG AI Training Pipeline - Task 3.4 Demo")
    logger.info("=" * 60)
    logger.info("This demo showcases all components of the training pipeline:")
    logger.info("1. Basic training functionality")
    logger.info("2. Model evaluation and metrics")
    logger.info("3. Hyperparameter optimization")
    logger.info("4. Model versioning and checkpointing")
    logger.info("5. Complete pipeline integration")
    logger.info("=" * 60)

    demos = [
        ("Basic Training", demo_basic_training),
        ("Model Evaluation", demo_evaluation),
        ("Hyperparameter Optimization", demo_hyperparameter_optimization),
        ("Model Versioning", demo_model_versioning),
        ("Complete Pipeline", demo_complete_pipeline)
    ]

    for demo_name, demo_func in demos:
        try:
            logger.info(f"\n{'='*20} {demo_name} {'='*20}")
            demo_func()
            logger.info(f"✓ {demo_name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {demo_name} failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("All demonstrations completed!")
    logger.info("Check the generated directories for outputs:")
    logger.info("- demo_plots/: Training visualization plots")
    logger.info("- demo_evaluation/: Evaluation results")
    logger.info("- demo_versioning/: Model checkpoints and versions")
    logger.info("- pipeline_demo/: Complete pipeline outputs")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()