#!/usr/bin/env python3
"""
MTG Hyperparameter Optimization Framework - Task 3.4

Comprehensive hyperparameter optimization system for Magic: The Gathering AI training
using grid search, random search, Bayesian optimization, and evolutionary strategies.

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
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import random
from pathlib import Path

# Hyperparameter optimization libraries
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logging.warning("Optuna not available. Install with: pip install optuna")

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logging.warning("scikit-optimize not available. Install with: pip install scikit-optimize")

# Import our training components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mtg_training_pipeline import MTGTrainer, TrainingConfig
    from mtg_evaluation_metrics import MTGEvaluator, EvaluationConfig
except ImportError as e:
    logging.warning(f"Could not import training components: {e}")

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""

    # Optimization method
    optimization_method: str = "optuna"  # "grid", "random", "optuna", "skopt", "evolutionary"

    # Search space
    param_ranges: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': (1e-5, 1e-3, 'log'),
        'batch_size': [8, 16, 32, 64],
        'dropout': (0.0, 0.5, 'uniform'),
        'weight_decay': (1e-6, 1e-3, 'log'),
        'd_model': [128, 256, 512],
        'nhead': [4, 8, 16],
        'num_encoder_layers': [2, 4, 6, 8],
        'dim_feedforward': [256, 512, 1024],
        'action_loss_weight': (0.5, 2.0, 'uniform'),
        'value_loss_weight': (0.1, 1.0, 'uniform'),
        'gradient_clip_norm': (0.5, 2.0, 'uniform')
    })

    # Optimization settings
    n_trials: int = 50
    timeout_seconds: int = 3600  # 1 hour
    n_jobs: int = 1

    # Early stopping for trials
    trial_patience: int = 5
    min_improvement: float = 1e-4

    # Objective function settings
    objective_metric: str = "val_accuracy"  # "val_loss", "val_accuracy", "val_f1"
    objective_direction: str = "maximize"  # "minimize" or "maximize"

    # Pruning settings (for Optuna)
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5

    # Output settings
    study_name: str = "mtg_hyperopt"
    output_dir: str = "hyperopt_results"
    save_all_trials: bool = True

    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 3


class HyperparameterSpace:
    """Define and manage hyperparameter search spaces."""

    def __init__(self, param_ranges: Dict[str, Any]):
        self.param_ranges = param_ranges
        self.param_types = {}

    def get_optuna_space(self):
        """Get Optuna search space."""
        if not HAS_OPTUNA:
            raise ImportError("Optuna not available")

        space = {}
        for param_name, param_config in self.param_ranges.items():
            if isinstance(param_config, tuple) and len(param_config) == 3:
                low, high, param_type = param_config
                if param_type == 'log':
                    space[param_name] = optuna.trial.suggest_float(
                        param_name, low, high, log=True
                    )
                elif param_type == 'uniform':
                    space[param_name] = optuna.trial.suggest_float(
                        param_name, low, high
                    )
                elif param_type == 'int':
                    space[param_name] = optuna.trial.suggest_int(
                        param_name, int(low), int(high)
                    )
            elif isinstance(param_config, list):
                space[param_name] = optuna.trial.suggest_categorical(
                    param_name, param_config
                )

        return space

    def get_skopt_space(self):
        """Get scikit-optimize search space."""
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize not available")

        dimensions = []
        param_names = []

        for param_name, param_config in self.param_ranges.items():
            param_names.append(param_name)

            if isinstance(param_config, tuple) and len(param_config) == 3:
                low, high, param_type = param_config
                if param_type == 'log':
                    dimensions.append(Real(low, high, prior='log', name=param_name))
                elif param_type == 'uniform':
                    dimensions.append(Real(low, high, name=param_name))
                elif param_type == 'int':
                    dimensions.append(Integer(int(low), int(high), name=param_name))
            elif isinstance(param_config, list):
                dimensions.append(Categorical(param_config, name=param_name))

        return dimensions, param_names

    def sample_random_params(self) -> Dict[str, Any]:
        """Sample random hyperparameters."""
        params = {}

        for param_name, param_config in self.param_ranges.items():
            if isinstance(param_config, tuple) and len(param_config) == 3:
                low, high, param_type = param_config
                if param_type == 'log':
                    value = 10 ** np.random.uniform(np.log10(low), np.log10(high))
                elif param_type == 'uniform':
                    value = np.random.uniform(low, high)
                elif param_type == 'int':
                    value = np.random.randint(int(low), int(high) + 1)
                else:
                    value = (low + high) / 2
            elif isinstance(param_config, list):
                value = random.choice(param_config)
            else:
                value = param_config

            params[param_name] = value

        return params

    def get_grid_params(self) -> List[Dict[str, Any]]:
        """Get grid search parameters."""
        grid_params = []
        param_lists = []

        for param_name, param_config in self.param_ranges.items():
            if isinstance(param_config, list):
                param_lists.append([(param_name, value) for value in param_config])
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                low, high, param_type = param_config
                if param_type == 'int':
                    values = list(range(int(low), int(high) + 1))
                    param_lists.append([(param_name, value) for value in values])
                else:
                    # Sample 5 values for continuous parameters
                    if param_type == 'log':
                        values = np.logspace(np.log10(low), np.log10(high), 5)
                    else:
                        values = np.linspace(low, high, 5)
                    param_lists.append([(param_name, float(value)) for value in values])

        # Generate all combinations
        for combination in itertools.product(*param_lists):
            params = {name: value for name, value in combination}
            grid_params.append(params)

        return grid_params


class ObjectiveFunction:
    """Objective function for hyperparameter optimization."""

    def __init__(self,
                 data_path: str,
                 base_config: TrainingConfig,
                 eval_config: EvaluationConfig,
                 hyperparam_config: HyperparameterConfig):
        self.data_path = data_path
        self.base_config = base_config
        self.eval_config = eval_config
        self.hyperparam_config = hyperparam_config
        self.trial_results = defaultdict(list)

    def __call__(self, trial_params: Dict[str, Any]) -> float:
        """
        Evaluate hyperparameters and return objective value.

        Args:
            trial_params: Dictionary of hyperparameters to test

        Returns:
            Objective value (to be minimized or maximized)
        """
        # Create trial configuration
        trial_config = self._create_trial_config(trial_params)

        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        try:
            # Train model with trial parameters
            trainer = MTGTrainer(trial_config)

            # Use reduced training for hyperparameter optimization
            original_max_epochs = trial_config.max_epochs
            trial_config.max_epochs = min(original_max_epochs, 10)  # Limit epochs for speed

            # Train model
            train_metrics, val_metrics = trainer.train(self.data_path)

            # Evaluate model
            evaluator = MTGEvaluator(self.eval_config)
            evaluation_results = evaluator.evaluate_model(
                trainer, trainer.val_loader, trainer.device
            )

            # Extract objective metric
            objective_value = self._extract_objective_value(evaluation_results)

            # Store trial results
            self.trial_results[str(trial_params)] = {
                'objective_value': objective_value,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'evaluation_results': evaluation_results,
                'config': trial_config
            }

            logger.info(f"Trial completed. Objective ({self.hyperparam_config.objective_metric}): {objective_value:.4f}")

            return objective_value

        except Exception as e:
            logger.error(f"Trial failed with params {trial_params}: {e}")
            # Return worst possible value
            if self.hyperparam_config.objective_direction == "maximize":
                return -float('inf')
            else:
                return float('inf')

    def _create_trial_config(self, trial_params: Dict[str, Any]) -> TrainingConfig:
        """Create training configuration for trial."""
        # Start with base config
        config_dict = self.base_config.__dict__.copy()

        # Update with trial parameters
        config_dict.update(trial_params)

        # Create new config object
        trial_config = TrainingConfig(**config_dict)

        return trial_config

    def _extract_objective_value(self, evaluation_results: Dict[str, Any]) -> float:
        """Extract objective value from evaluation results."""
        metric_name = self.hyperparam_config.objective_metric

        if metric_name == "val_loss":
            return evaluation_results.get('overall_metrics', {}).get('total_loss', float('inf'))
        elif metric_name == "val_accuracy":
            return evaluation_results.get('overall_metrics', {}).get('accuracy', 0.0)
        elif metric_name == "val_f1":
            return evaluation_results.get('overall_metrics', {}).get('f1_weighted', 0.0)
        else:
            logger.warning(f"Unknown objective metric: {metric_name}")
            return 0.0


class HyperparameterOptimizer:
    """Main hyperparameter optimization orchestrator."""

    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.space = HyperparameterSpace(config.param_ranges)
        self.results = defaultdict(list)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def optimize(self,
                 data_path: str,
                 base_config: TrainingConfig,
                 eval_config: EvaluationConfig) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            data_path: Path to training data
            base_config: Base training configuration
            eval_config: Evaluation configuration

        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.config.optimization_method}")

        # Create objective function
        objective = ObjectiveFunction(data_path, base_config, eval_config, self.config)

        if self.config.optimization_method == "grid":
            results = self._grid_search(objective)
        elif self.config.optimization_method == "random":
            results = self._random_search(objective)
        elif self.config.optimization_method == "optuna":
            results = self._optuna_search(objective)
        elif self.config.optimization_method == "skopt":
            results = self._skopt_search(objective)
        elif self.config.optimization_method == "evolutionary":
            results = self._evolutionary_search(objective)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")

        # Process and save results
        processed_results = self._process_results(results)
        self._save_results(processed_results)

        return processed_results

    def _grid_search(self, objective: ObjectiveFunction) -> Dict[str, Any]:
        """Grid search optimization."""
        logger.info("Running grid search...")
        param_combinations = self.space.get_grid_params()

        # Limit number of combinations for grid search
        max_combinations = min(len(param_combinations), self.config.n_trials)
        param_combinations = param_combinations[:max_combinations]

        results = []
        for i, params in enumerate(param_combinations):
            logger.info(f"Grid search trial {i+1}/{len(param_combinations)}: {params}")
            objective_value = objective(params)
            results.append({'params': params, 'objective_value': objective_value})

        return {'method': 'grid_search', 'trials': results}

    def _random_search(self, objective: ObjectiveFunction) -> Dict[str, Any]:
        """Random search optimization."""
        logger.info("Running random search...")
        results = []

        for i in range(self.config.n_trials):
            params = self.space.sample_random_params()
            logger.info(f"Random search trial {i+1}/{self.config.n_trials}")
            objective_value = objective(params)
            results.append({'params': params, 'objective_value': objective_value})

        return {'method': 'random_search', 'trials': results}

    def _optuna_search(self, objective: ObjectiveFunction) -> Dict[str, Any]:
        """Optuna Bayesian optimization."""
        if not HAS_OPTUNA:
            raise ImportError("Optuna not available")

        logger.info("Running Optuna optimization...")

        def optuna_objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_config in self.config.param_ranges.items():
                if isinstance(param_config, tuple) and len(param_config) == 3:
                    low, high, param_type = param_config
                    if param_type == 'log':
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    elif param_type == 'uniform':
                        params[param_name] = trial.suggest_float(param_name, low, high)
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)

            # Report intermediate values for pruning
            if self.config.enable_pruning:
                # This would need to be implemented in the training loop
                # For now, we'll just return the final objective value
                pass

            return objective(params)

        # Create study
        direction = self.config.objective_direction
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.config.pruning_warmup_steps,
                n_warmup_steps=self.config.pruning_warmup_steps
            ) if self.config.enable_pruning else optuna.pruners.NopPruner()
        )

        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds if self.config.timeout_seconds > 0 else None,
            n_jobs=self.config.n_jobs
        )

        # Process results
        results = []
        for trial in study.trials:
            results.append({
                'params': trial.params,
                'objective_value': trial.value,
                'state': trial.state.name,
                'number': trial.number
            })

        return {
            'method': 'optuna',
            'trials': results,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }

    def _skopt_search(self, objective: ObjectiveFunction) -> Dict[str, Any]:
        """scikit-optimize Bayesian optimization."""
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize not available")

        logger.info("Running scikit-optimize optimization...")

        dimensions, param_names = self.space.get_skopt_space()

        @use_named_args(dimensions)
        def skopt_objective(**params):
            return objective(params)

        # Run optimization
        if self.config.objective_direction == "maximize":
            result = gp_minimize(
                lambda *args, **kwargs: -skopt_objective(*args, **kwargs),
                dimensions,
                n_calls=self.config.n_trials,
                random_state=42
            )
            # Negate results back for maximization
            result.func_vals = [-val for val in result.func_vals]
            result.fun = -result.fun
        else:
            result = gp_minimize(skopt_objective, dimensions, n_calls=self.config.n_trials, random_state=42)

        # Process results
        trials = []
        for i, (params, value) in enumerate(zip(result.x_iters, result.func_vals)):
            param_dict = dict(zip(param_names, params))
            trials.append({
                'params': param_dict,
                'objective_value': value
            })

        best_params = dict(zip(param_names, result.x))

        return {
            'method': 'skopt',
            'trials': trials,
            'best_params': best_params,
            'best_value': result.fun,
            'n_trials': len(trials)
        }

    def _evolutionary_search(self, objective: ObjectiveFunction) -> Dict[str, Any]:
        """Evolutionary algorithm optimization."""
        logger.info("Running evolutionary optimization...")

        # Initialize population
        population_size = min(self.config.n_trials // 2, 10)
        population = [self.space.sample_random_params() for _ in range(population_size)]

        # Evaluate initial population
        evaluated_population = []
        for params in population:
            objective_value = objective(params)
            evaluated_population.append({'params': params, 'objective_value': objective_value})

        generations = self.config.n_trials // population_size
        all_trials = evaluated_population.copy()

        for generation in range(generations):
            logger.info(f"Evolutionary generation {generation + 1}/{generations}")

            # Select parents (tournament selection)
            parents = self._tournament_selection(evaluated_population, k=2)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1, parent2 = parents[i], parents[i + 1]
                    child = self._crossover(parent1['params'], parent2['params'])
                    child = self._mutate(child)
                    offspring.append(child)

            # Evaluate offspring
            for params in offspring:
                objective_value = objective(params)
                all_trials.append({'params': params, 'objective_value': objective_value})
                evaluated_population.append({'params': params, 'objective_value': objective_value})

            # Select next generation
            evaluated_population = self._select_next_generation(evaluated_population, population_size)

        return {'method': 'evolutionary', 'trials': all_trials}

    def _tournament_selection(self, population: List[Dict], k: int = 3) -> List[Dict]:
        """Tournament selection for evolutionary algorithm."""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, min(k, len(population)))
            winner = max(tournament, key=lambda x: x['objective_value'])
            selected.append(winner)
        return selected

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for evolutionary algorithm."""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, params: Dict) -> Dict:
        """Mutation operation for evolutionary algorithm."""
        mutated = params.copy()
        for param_name, param_config in self.config.param_ranges.items():
            if random.random() < 0.2:  # 20% mutation rate
                if isinstance(param_config, tuple) and len(param_config) == 3:
                    low, high, param_type = param_config
                    if param_type == 'log':
                        current = mutated[param_name]
                        # Small random change in log space
                        log_change = np.random.normal(0, 0.1)
                        new_value = current * (10 ** log_change)
                        mutated[param_name] = np.clip(new_value, low, high)
                    elif param_type == 'uniform':
                        current = mutated[param_name]
                        # Small random change
                        change = np.random.normal(0, (high - low) * 0.1)
                        new_value = current + change
                        mutated[param_name] = np.clip(new_value, low, high)
        return mutated

    def _select_next_generation(self, population: List[Dict], size: int) -> List[Dict]:
        """Select next generation for evolutionary algorithm."""
        # Sort by objective value (descending for maximization)
        sorted_population = sorted(population, key=lambda x: x['objective_value'], reverse=True)
        return sorted_population[:size]

    def _process_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw optimization results."""
        trials = raw_results.get('trials', [])

        if not trials:
            return raw_results

        # Sort trials by objective value
        is_maximize = self.config.objective_direction == "maximize"
        sorted_trials = sorted(trials, key=lambda x: x['objective_value'], reverse=is_maximize)

        # Extract best parameters
        best_trial = sorted_trials[0] if sorted_trials else None

        # Compute statistics
        objective_values = [trial['objective_value'] for trial in trials if trial['objective_value'] != float('inf') and trial['objective_value'] != -float('inf')]

        statistics = {
            'best_trial': best_trial,
            'best_objective_value': best_trial['objective_value'] if best_trial else None,
            'best_params': best_trial['params'] if best_trial else None,
            'mean_objective_value': np.mean(objective_values) if objective_values else None,
            'std_objective_value': np.std(objective_values) if objective_values else None,
            'total_trials': len(trials),
            'successful_trials': len(objective_values)
        }

        # Add parameter importance analysis
        param_importance = self._analyze_parameter_importance(trials)
        statistics['parameter_importance'] = param_importance

        # Combine with raw results
        processed_results = raw_results.copy()
        processed_results.update(statistics)

        return processed_results

    def _analyze_parameter_importance(self, trials: List[Dict]) -> Dict[str, float]:
        """Analyze importance of different hyperparameters."""
        if len(trials) < 10:
            return {}

        # Create DataFrame for analysis
        df_data = []
        for trial in trials:
            if trial['objective_value'] != float('inf') and trial['objective_value'] != -float('inf'):
                row = trial['params'].copy()
                row['objective_value'] = trial['objective_value']
                df_data.append(row)

        if not df_data:
            return {}

        df = pd.DataFrame(df_data)

        # Compute correlation with objective value
        importance = {}
        for param in df.columns:
            if param != 'objective_value':
                if df[param].dtype in ['int64', 'float64']:
                    correlation = df[param].corr(df['objective_value'])
                    importance[param] = abs(correlation) if not np.isnan(correlation) else 0.0

        return importance

    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results."""
        # Save as JSON
        results_path = os.path.join(self.config.output_dir, 'optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save as CSV for easy analysis
        if 'trials' in results:
            trials_data = []
            for trial in results['trials']:
                row = trial['params'].copy()
                row['objective_value'] = trial['objective_value']
                trials_data.append(row)

            if trials_data:
                df = pd.DataFrame(trials_data)
                csv_path = os.path.join(self.config.output_dir, 'trials_summary.csv')
                df.to_csv(csv_path, index=False)

        # Save best configuration
        if 'best_params' in results and results['best_params']:
            best_config_path = os.path.join(self.config.output_dir, 'best_config.json')
            with open(best_config_path, 'w') as f:
                json.dump(results['best_params'], f, indent=2)

        logger.info(f"Optimization results saved to {self.config.output_dir}")


def main():
    """Example usage of hyperparameter optimization."""
    # Configuration
    hyperparam_config = HyperparameterConfig(
        optimization_method="random",  # Start with random search for demo
        n_trials=5,  # Small number for demo
        timeout_seconds=300,  # 5 minutes
        objective_metric="val_accuracy",
        objective_direction="maximize"
    )

    base_config = TrainingConfig(
        batch_size=16,
        max_epochs=5,  # Short epochs for demo
        learning_rate=1e-4
    )

    eval_config = EvaluationConfig(
        save_results=False
    )

    # Run optimization
    optimizer = HyperparameterOptimizer(hyperparam_config)

    data_path = "/home/joshu/logparser/complete_training_dataset_task2_4.json"
    results = optimizer.optimize(data_path, base_config, eval_config)

    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best parameters: {results.get('best_params', {})}")
    logger.info(f"Best objective value: {results.get('best_objective_value', 'N/A')}")


if __name__ == "__main__":
    main()