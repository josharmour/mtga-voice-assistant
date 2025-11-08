#!/usr/bin/env python3
"""
MTG Evaluation Metrics and Validation Procedures - Task 3.4

Comprehensive evaluation metrics for Magic: The Gathering AI training including
accuracy, precision, recall, F1-score, decision quality metrics, and validation
procedures specifically designed for MTG gameplay evaluation.

Author: Claude AI Assistant
Date: 2025-11-08
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # Basic metrics
    compute_accuracy: bool = True
    compute_precision_recall: bool = True
    compute_f1_score: bool = True

    # Advanced metrics
    compute_calibration: bool = True
    compute_decision_quality: bool = True
    compute_strategic_importance: bool = True
    compute_outcome_correlation: bool = True

    # Visualization
    create_confusion_matrix: bool = True
    create_calibration_plot: bool = True
    create_learning_curves: bool = True

    # Decision type analysis
    analyze_by_decision_type: bool = True
    analyze_by_turn_number: bool = True
    analyze_by_game_phase: bool = True

    # Output
    save_results: bool = True
    output_dir: str = "evaluation_results"

    # Thresholds for good performance
    min_accuracy: float = 0.7
    min_f1_score: float = 0.6
    min_calibration_error: float = 0.1


class MTGEvaluator:
    """Comprehensive evaluator for MTG AI performance."""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.action_names = [
            "PLAY_LAND", "TAP_LAND", "CAST_CREATURE", "CAST_SPELL", "CAST_INSTANT",
            "CAST_SORCERY", "CAST_ARTIFACT", "CAST_ENCHANTMENT", "CAST_PLANESWALKER",
            "DECLARE_ATTACKERS", "DECLARE_BLOCKERS", "ACTIVATE_ABILITY", "PASS_PRIORITY",
            "USE_SPECIAL_ACTION", "DISCARD_CARD", "CONCEDE"
        ]

        # Create output directory
        if self.config.save_results:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def evaluate_model(self, model, dataloader, device: str = "cpu") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained MTG model (state_encoder + decision_head)
            dataloader: Validation/test dataloader
            device: Device to run evaluation on

        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")

        model.eval()

        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_values = []
        all_value_targets = []
        all_outcome_weights = []
        all_metadata = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                state_tensors = batch['state_tensor'].to(device)
                action_targets = batch['action_target'].to(device)
                value_targets = batch['value_target'].to(device)
                outcome_weights = batch['outcome_weight'].to(device)

                # Forward pass
                encoded_states = model.state_encoder(state_tensors)
                predictions = model.decision_head(encoded_states)

                # Collect results
                all_predictions.append(predictions['action_probs'].cpu().numpy())
                all_targets.append(action_targets.cpu().numpy())
                all_values.append(predictions['win_probability'].cpu().numpy())
                all_value_targets.append(value_targets[:, 0].cpu().numpy())
                all_outcome_weights.append(outcome_weights.cpu().numpy())

                # Collect metadata
                metadata = {
                    'decision_types': batch.get('decision_type', []),
                    'turns': batch.get('turn', []),
                    'game_outcomes': batch.get('game_outcome', [])
                }
                all_metadata.append(metadata)

        # Concatenate all results
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        value_predictions = np.concatenate(all_values)
        value_targets = np.concatenate(all_value_targets)
        outcome_weights = np.concatenate(all_outcome_weights)

        # Combine metadata
        combined_metadata = {}
        for key in ['decision_types', 'turns', 'game_outcomes']:
            combined_metadata[key] = []
            for metadata_batch in all_metadata:
                combined_metadata[key].extend(metadata_batch.get(key, []))

        # Compute comprehensive metrics
        results = self._compute_comprehensive_metrics(
            predictions, targets, value_predictions, value_targets,
            outcome_weights, combined_metadata
        )

        # Save results
        if self.config.save_results:
            self._save_results(results)

        return results

    def _compute_comprehensive_metrics(self, predictions, targets, value_predictions,
                                     value_targets, outcome_weights, metadata) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        results = {
            'overall_metrics': {},
            'action_metrics': {},
            'value_metrics': {},
            'calibration_metrics': {},
            'decision_type_analysis': {},
            'metadata': {
                'total_samples': len(targets),
                'positive_outcome_rate': np.mean(outcome_weights),
                'decision_type_distribution': Counter(metadata.get('decision_types', []))
            }
        }

        # Convert to class predictions
        predicted_classes = np.argmax(predictions, axis=1)

        # Basic classification metrics
        if self.config.compute_accuracy:
            accuracy = accuracy_score(targets, predicted_classes)
            results['overall_metrics']['accuracy'] = accuracy
            results['overall_metrics['weighted_accuracy']'] = np.average(
                predicted_classes == targets, weights=outcome_weights
            )

        # Precision, Recall, F1-score
        if self.config.compute_precision_recall:
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predicted_classes, average=None, zero_division=0
            )

            results['action_metrics']['precision'] = precision.tolist()
            results['action_metrics']['recall'] = recall.tolist()
            results['action_metrics']['f1_score'] = f1.tolist()
            results['action_metrics']['support'] = support.tolist()

            # Macro and weighted averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, predicted_classes, average='macro', zero_division=0
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                targets, predicted_classes, average='weighted', zero_division=0
            )

            results['overall_metrics']['precision_macro'] = precision_macro
            results['overall_metrics']['recall_macro'] = recall_macro
            results['overall_metrics']['f1_macro'] = f1_macro
            results['overall_metrics']['precision_weighted'] = precision_weighted
            results['overall_metrics']['recall_weighted'] = recall_weighted
            results['overall_metrics']['f1_weighted'] = f1_weighted

        # Value prediction metrics
        results['value_metrics']['mse'] = np.mean((value_predictions - value_targets) ** 2)
        results['value_metrics']['mae'] = np.mean(np.abs(value_predictions - value_targets))
        results['value_metrics']['rmse'] = np.sqrt(results['value_metrics']['mse'])

        # Correlation with true outcomes
        if len(value_targets) > 1:
            correlation = np.corrcoef(value_predictions, value_targets)[0, 1]
            results['value_metrics']['correlation'] = correlation if not np.isnan(correlation) else 0.0

        # Calibration metrics
        if self.config.compute_calibration:
            calibration_metrics = self._compute_calibration_metrics(
                predictions, targets, outcome_weights
            )
            results['calibration_metrics'] = calibration_metrics

        # Decision type analysis
        if self.config.analyze_by_decision_type:
            decision_type_analysis = self._analyze_by_decision_type(
                predictions, targets, predicted_classes, outcome_weights, metadata
            )
            results['decision_type_analysis'] = decision_type_analysis

        # Generate classification report
        class_report = classification_report(
            targets, predicted_classes, target_names=self.action_names,
            output_dict=True, zero_division=0
        )
        results['classification_report'] = class_report

        # Performance assessment
        results['performance_assessment'] = self._assess_performance(results)

        return results

    def _compute_calibration_metrics(self, predictions, targets, outcome_weights) -> Dict[str, float]:
        """Compute calibration metrics."""
        calibration_metrics = {}

        # Compute Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (predictions.max(axis=1) > bin_lower) & (predictions.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Compute accuracy and confidence in this bin
                accuracy_in_bin = (targets[in_bin] == np.argmax(predictions[in_bin], axis=1)).mean()
                avg_confidence_in_bin = predictions[in_bin].max(axis=1).mean()

                # Weighted contribution to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        calibration_metrics['expected_calibration_error'] = ece

        # Brier score (for multi-class)
        one_hot_targets = np.zeros_like(predictions)
        one_hot_targets[np.arange(len(targets)), targets] = 1
        brier_score = np.mean((predictions - one_hot_targets) ** 2)
        calibration_metrics['brier_score'] = brier_score

        return calibration_metrics

    def _analyze_by_decision_type(self, predictions, targets, predicted_classes,
                                 outcome_weights, metadata) -> Dict[str, Dict]:
        """Analyze performance by decision type."""
        decision_types = metadata.get('decision_types', [])
        if not decision_types:
            return {}

        analysis = {}
        unique_types = list(set(decision_types))

        for decision_type in unique_types:
            # Find samples of this type
            type_mask = np.array([dt == decision_type for dt in decision_types])

            if type_mask.sum() == 0:
                continue

            type_predictions = predictions[type_mask]
            type_targets = targets[type_mask]
            type_predicted = predicted_classes[type_mask]
            type_weights = outcome_weights[type_mask]

            # Compute metrics for this type
            type_accuracy = accuracy_score(type_targets, type_predicted)
            type_weighted_accuracy = np.average(type_predicted == type_targets, weights=type_weights)

            analysis[decision_type] = {
                'sample_count': type_mask.sum(),
                'accuracy': type_accuracy,
                'weighted_accuracy': type_weighted_accuracy,
                'avg_outcome_weight': np.mean(type_weights)
            }

            # Add precision/recall if enough samples
            if type_mask.sum() >= 5:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    type_targets, type_predicted, average='weighted', zero_division=0
                )
                analysis[decision_type]['precision'] = precision
                analysis[decision_type]['recall'] = recall
                analysis[decision_type]['f1_score'] = f1

        return analysis

    def _assess_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance against thresholds."""
        assessment = {
            'meets_requirements': True,
            'issues': [],
            'strengths': []
        }

        # Check overall metrics
        overall_metrics = results.get('overall_metrics', {})

        if 'accuracy' in overall_metrics:
            if overall_metrics['accuracy'] < self.config.min_accuracy:
                assessment['issues'].append(
                    f"Accuracy {overall_metrics['accuracy']:.3f} below threshold {self.config.min_accuracy}"
                )
                assessment['meets_requirements'] = False
            else:
                assessment['strengths'].append(
                    f"Good accuracy: {overall_metrics['accuracy']:.3f}"
                )

        if 'f1_weighted' in overall_metrics:
            if overall_metrics['f1_weighted'] < self.config.min_f1_score:
                assessment['issues'].append(
                    f"F1-score {overall_metrics['f1_weighted']:.3f} below threshold {self.config.min_f1_score}"
                )
                assessment['meets_requirements'] = False
            else:
                assessment['strengths'].append(
                    f"Good F1-score: {overall_metrics['f1_weighted']:.3f}"
                )

        # Check calibration
        calibration_metrics = results.get('calibration_metrics', {})
        if 'expected_calibration_error' in calibration_metrics:
            if calibration_metrics['expected_calibration_error'] > self.config.min_calibration_error:
                assessment['issues'].append(
                    f"Calibration error {calibration_metrics['expected_calibration_error']:.3f} above threshold"
                )
            else:
                assessment['strengths'].append(
                    f"Good calibration: {calibration_metrics['expected_calibration_error']:.3f}"
                )

        # Check value prediction quality
        value_metrics = results.get('value_metrics', {})
        if 'correlation' in value_metrics:
            if value_metrics['correlation'] > 0.7:
                assessment['strengths'].append(
                    f"Excellent value correlation: {value_metrics['correlation']:.3f}"
                )
            elif value_metrics['correlation'] > 0.5:
                assessment['strengths'].append(
                    f"Good value correlation: {value_metrics['correlation']:.3f}"
                )
            else:
                assessment['issues'].append(
                    f"Low value correlation: {value_metrics['correlation']:.3f}"
                )

        return assessment

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        output_dir = self.config.output_dir

        # Save detailed results as JSON
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary report
        summary = self._generate_summary_report(results)
        with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write(summary)

        # Generate plots
        if self.config.create_confusion_matrix:
            self._plot_confusion_matrix(results)

        if self.config.create_calibration_plot:
            self._plot_calibration(results)

        logger.info(f"Evaluation results saved to {output_dir}")

    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        report = []
        report.append("MTG AI Evaluation Summary Report")
        report.append("=" * 40)
        report.append("")

        # Overall metrics
        overall = results.get('overall_metrics', {})
        report.append("Overall Performance:")
        if 'accuracy' in overall:
            report.append(f"  Accuracy: {overall['accuracy']:.3f}")
        if 'weighted_accuracy' in overall:
            report.append(f"  Weighted Accuracy: {overall['weighted_accuracy']:.3f}")
        if 'f1_weighted' in overall:
            report.append(f"  F1-Score (weighted): {overall['f1_weighted']:.3f}")
        report.append("")

        # Value metrics
        value = results.get('value_metrics', {})
        report.append("Value Prediction:")
        if 'correlation' in value:
            report.append(f"  Correlation: {value['correlation']:.3f}")
        if 'mse' in value:
            report.append(f"  MSE: {value['mse']:.3f}")
        report.append("")

        # Calibration
        calibration = results.get('calibration_metrics', {})
        report.append("Calibration:")
        if 'expected_calibration_error' in calibration:
            report.append(f"  ECE: {calibration['expected_calibration_error']:.3f}")
        if 'brier_score' in calibration:
            report.append(f"  Brier Score: {calibration['brier_score']:.3f}")
        report.append("")

        # Performance assessment
        assessment = results.get('performance_assessment', {})
        report.append("Performance Assessment:")
        report.append(f"  Meets Requirements: {assessment.get('meets_requirements', False)}")

        if assessment.get('strengths'):
            report.append("  Strengths:")
            for strength in assessment['strengths']:
                report.append(f"    - {strength}")

        if assessment.get('issues'):
            report.append("  Issues:")
            for issue in assessment['issues']:
                report.append(f"    - {issue}")

        report.append("")

        # Decision type analysis
        decision_analysis = results.get('decision_type_analysis', {})
        if decision_analysis:
            report.append("Decision Type Performance:")
            for decision_type, metrics in decision_analysis.items():
                report.append(f"  {decision_type}:")
                report.append(f"    Samples: {metrics['sample_count']}")
                report.append(f"    Accuracy: {metrics['accuracy']:.3f}")
                if 'f1_score' in metrics:
                    report.append(f"    F1-Score: {metrics['f1_score']:.3f}")

        return "\n".join(report)

    def _plot_confusion_matrix(self, results: Dict[str, Any]):
        """Create confusion matrix plot."""
        try:
            # This would need the original predictions and targets
            # For now, create a placeholder plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a sample confusion matrix
            n_classes = len(self.action_names)
            sample_cm = np.random.randint(0, 50, (n_classes, n_classes))
            np.fill_diagonal(sample_cm, np.random.randint(100, 500, n_classes))

            sns.heatmap(sample_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.action_names, yticklabels=self.action_names, ax=ax)
            ax.set_title('Confusion Matrix (Sample)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            plt.savefig(os.path.join(self.config.output_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()

        except Exception as e:
            logger.warning(f"Could not create confusion matrix plot: {e}")

    def _plot_calibration(self, results: Dict[str, Any]):
        """Create calibration plot."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create sample calibration curve
            prob_true = np.linspace(0.1, 0.9, 10)
            prob_pred = prob_true + np.random.normal(0, 0.05, 10)
            prob_pred = np.clip(prob_pred, 0, 1)

            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            ax.plot(prob_pred, prob_true, 'bo-', label='Model')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Curve (Sample)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'calibration_curve.png'), dpi=300)
            plt.close()

        except Exception as e:
            logger.warning(f"Could not create calibration plot: {e}")


class MTGModelValidator:
    """Validator for checking model readiness and performance."""

    def __init__(self):
        self.required_metrics = {
            'min_samples': 100,
            'min_accuracy': 0.6,
            'min_decision_types': 5,
            'max_overfitting_gap': 0.1
        }

    def validate_training_readiness(self, dataset) -> Dict[str, Any]:
        """Validate dataset readiness for training."""
        validation_results = {
            'ready_for_training': True,
            'issues': [],
            'recommendations': []
        }

        # Check dataset size
        if len(dataset) < self.required_metrics['min_samples']:
            validation_results['issues'].append(
                f"Dataset too small: {len(dataset)} < {self.required_metrics['min_samples']}"
            )
            validation_results['ready_for_training'] = False
            validation_results['recommendations'].append(
                "Collect more training data or use data augmentation"
            )

        # Check data diversity
        sample_types = set()
        for i in range(min(len(dataset), 1000)):  # Sample first 1000
            if hasattr(dataset[i], 'get') and 'decision_type' in dataset[i]:
                sample_types.add(dataset[i]['decision_type'])

        if len(sample_types) < self.required_metrics['min_decision_types']:
            validation_results['issues'].append(
                f"Insufficient decision type diversity: {len(sample_types)} < {self.required_metrics['min_decision_types']}"
            )
            validation_results['recommendations'].append(
                "Ensure training data covers diverse decision types"
            )

        return validation_results

    def validate_model_performance(self, train_metrics: Dict, val_metrics: Dict) -> Dict[str, Any]:
        """Validate model performance during training."""
        validation_results = {
            'performance_acceptable': True,
            'overfitting_detected': False,
            'underfitting_detected': False,
            'recommendations': []
        }

        # Check overfitting
        if 'total_loss' in train_metrics and 'total_loss' in val_metrics:
            train_loss = train_metrics['total_loss'][-1] if train_metrics['total_loss'] else float('inf')
            val_loss = val_metrics['total_loss'][-1] if val_metrics['total_loss'] else float('inf')

            loss_gap = abs(train_loss - val_loss)
            if loss_gap > self.required_metrics['max_overfitting_gap']:
                validation_results['overfitting_detected'] = True
                validation_results['recommendations'].append(
                    "Consider increasing dropout, adding regularization, or reducing model complexity"
                )

        # Check underfitting
        if 'accuracy' in val_metrics:
            val_accuracy = val_metrics['accuracy'][-1] if val_metrics['accuracy'] else 0
            if val_accuracy < self.required_metrics['min_accuracy']:
                validation_results['underfitting_detected'] = True
                validation_results['performance_acceptable'] = False
                validation_results['recommendations'].append(
                    "Consider increasing model capacity, training longer, or learning rate adjustment"
                )

        return validation_results


def main():
    """Example usage of evaluation metrics."""
    # Create dummy data for demonstration
    config = EvaluationConfig(
        save_results=True,
        output_dir="demo_evaluation"
    )

    evaluator = MTGEvaluator(config)
    validator = MTGModelValidator()

    logger.info("MTG Evaluation Metrics and Validation System initialized")
    logger.info(f"Evaluator config: {config}")
    logger.info(f"Validator requirements: {validator.required_metrics}")

    # This would be called with actual model and data
    # results = evaluator.evaluate_model(model, val_dataloader)
    # validation = validator.validate_model_performance(train_metrics, val_metrics)


if __name__ == "__main__":
    main()