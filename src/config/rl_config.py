"""
RL Configuration Management

Handles configuration for reinforcement learning components including
model parameters, training settings, and performance requirements.

Constitutional Requirements:
- Real-Time Responsiveness: Sub-100ms inference latency configuration
- Verifiable Testing: Test parameters and validation thresholds
- Graceful Degradation: Fallback mechanism configuration
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Default configuration paths
DEFAULT_CONFIG_PATH = "config/rl_config.json"
USER_CONFIG_PATH = os.path.expanduser("~/.mtga_advisor/rl_config.json")


@dataclass
class RLModelConfig:
    """Configuration for RL model architecture and parameters."""

    # Model architecture
    state_dim: int = 380  # Enhanced state representation dimensions
    action_dim: int = 64  # Number of possible actions
    hidden_dims: list = None  # Hidden layer sizes
    dueling_architecture: bool = True  # Use dueling DQN

    # Conservative Q-Learning parameters
    cql_alpha: float = 5.0  # Conservative Q-Learning weight
    cql_temp: float = 1.0   # Temperature parameter

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    buffer_size: int = 100000
    target_update_freq: int = 1000
    gamma: float = 0.99  # Discount factor

    # Network specifics
    activation: str = "relu"
    dropout_rate: float = 0.1
    layer_norm: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


@dataclass
class RLTrainingConfig:
    """Configuration for RL training process."""

    # Training schedule
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    save_freq: int = 1000  # Save model every N episodes

    # Curriculum learning
    curriculum_learning: bool = True
    curriculum_stages: int = 4
    difficulty_progression: list = None

    # Continual learning
    continual_learning: bool = True
    ewc_lambda: float = 0.4  # Elastic weight consolidation weight
    replay_buffer_ratio: float = 0.1  # Ratio for continual replay buffer

    # Data sources
    use_17lands_data: bool = True
    data_path: str = "data/17lands_data/"
    validation_split: float = 0.2

    # Performance optimization
    mixed_precision: bool = True
    gradient_clip_value: float = 1.0
    early_stopping_patience: int = 50

    # Logging and monitoring
    log_freq: int = 100
    tensorboard_log: bool = True
    tensorboard_path: str = "logs/rl_training"

    def __post_init__(self):
        if self.difficulty_progression is None:
            self.difficulty_progression = [0.1, 0.3, 0.6, 1.0]


@dataclass
class RLInferenceConfig:
    """Configuration for RL inference engine."""

    # Performance requirements (constitutional)
    max_inference_latency_ms: float = 100.0  # NON-NEGOTIABLE
    batch_inference: bool = True
    max_batch_size: int = 32

    # Device configuration
    prefer_gpu: bool = True
    fallback_to_cpu: bool = True
    warmup_iterations: int = 10

    # Model loading
    model_checkpoint_path: str = "models/rl/"
    model_version: str = "latest"
    confidence_threshold: float = 0.7  # Below this, use supervised fallback

    # Explainability
    enable_explainability: bool = True
    attention_visualization: bool = True
    decision_rationale: bool = True

    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 300  # 5 minutes


@dataclass
class RLValidationConfig:
    """Configuration for model validation and testing."""

    # Validation requirements (constitutional)
    min_win_rate_improvement: float = 0.25  # 25% minimum improvement
    confidence_level: float = 0.95  # 95% confidence interval

    # Test parameters
    test_games_count: int = 1000
    cross_validation_folds: int = 5
    statistical_significance: bool = True

    # Performance benchmarks
    latency_benchmark_samples: int = 1000
    memory_usage_limit_gb: float = 16.0

    # Generalization testing
    test_unseen_cards: bool = True
    test_new_mechanics: bool = True
    domain_adaptation_test: bool = True

    # Code coverage (constitutional)
    min_code_coverage_percent: float = 80.0


@dataclass
class RLSystemConfig:
    """Top-level RL system configuration."""

    # Sub-configurations
    model: RLModelConfig = None
    training: RLTrainingConfig = None
    inference: RLInferenceConfig = None
    validation: RLValidationConfig = None

    # System-wide settings
    enable_rl: bool = True
    debug_mode: bool = False
    seed: int = 42

    # Paths
    data_dir: str = "data/"
    models_dir: str = "models/rl/"
    logs_dir: str = "logs/"

    # Integration with existing system
    integration_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.model is None:
            self.model = RLModelConfig()
        if self.training is None:
            self.training = RLTrainingConfig()
        if self.inference is None:
            self.inference = RLInferenceConfig()
        if self.validation is None:
            self.validation = RLValidationConfig()
        if self.integration_config is None:
            self.integration_config = {
                "mtga_voice_advisor": {
                    "enable_rl_fallback": True,
                    "advice_confidence_threshold": 0.7,
                    "graceful_degradation": True
                }
            }


class RLConfigManager:
    """
    Manages RL configuration with validation and constitutional compliance checking.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Optional[RLSystemConfig] = None

        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

    def load_config(self, config_path: Optional[str] = None) -> RLSystemConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration
        """
        if config_path:
            self.config_path = config_path

        # Try user config first, then default config
        paths_to_try = [USER_CONFIG_PATH, self.config_path]

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config_dict = json.load(f)

                    self._config = self._dict_to_config(config_dict)
                    logger.info(f"Loaded RL config from {path}")
                    break

                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

        # Create default config if no valid config found
        if self._config is None:
            self._config = RLSystemConfig()
            self.save_config()
            logger.info("Created default RL configuration")

        # Validate constitutional requirements
        self._validate_constitutional_requirements()

        return self._config

    def save_config(self, config: Optional[RLSystemConfig] = None,
                   config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses current if None)
            config_path: Path to save configuration
        """
        if config:
            self._config = config

        if config_path:
            self.config_path = config_path

        if not self._config:
            raise ValueError("No configuration to save")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            config_dict = self._config_to_dict(self._config)

            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Saved RL config to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise

    def get_config(self) -> RLSystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        if self._config is None:
            self.load_config()

        # Apply updates (recursive)
        self._apply_updates(self._config, updates)

        # Validate updates
        self._validate_constitutional_requirements()

    def _apply_updates(self, obj: Any, updates: Dict[str, Any]) -> None:
        """Recursively apply updates to configuration object."""
        if isinstance(obj, dict):
            for key, value in updates.items():
                if key in obj:
                    if isinstance(value, dict) and isinstance(obj[key], (dict, object)):
                        self._apply_updates(obj[key], value)
                    else:
                        obj[key] = value
        elif hasattr(obj, '__dict__'):
            for key, value in updates.items():
                if hasattr(obj, key):
                    if isinstance(value, dict) and hasattr(getattr(obj, key), '__dict__'):
                        self._apply_updates(getattr(obj, key), value)
                    else:
                        setattr(obj, key, value)

    def _config_to_dict(self, config: RLSystemConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return asdict(config)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> RLSystemConfig:
        """Convert dictionary to configuration object."""
        # Convert nested dictionaries to dataclass instances
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = RLModelConfig(**config_dict['model'])

        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = RLTrainingConfig(**config_dict['training'])

        if 'inference' in config_dict and isinstance(config_dict['inference'], dict):
            config_dict['inference'] = RLInferenceConfig(**config_dict['inference'])

        if 'validation' in config_dict and isinstance(config_dict['validation'], dict):
            config_dict['validation'] = RLValidationConfig(**config_dict['validation'])

        return RLSystemConfig(**config_dict)

    def _validate_constitutional_requirements(self) -> None:
        """
        Validate that configuration meets constitutional requirements.

        Raises:
            ValueError: If constitutional requirements are violated
        """
        if not self._config:
            return

        violations = []

        # Real-Time Responsiveness: Sub-100ms inference latency (NON-NEGOTIABLE)
        if self._config.inference.max_inference_latency_ms > 100.0:
            violations.append(
                f"Inference latency {self._config.inference.max_inference_latency_ms}ms "
                f"exceeds constitutional requirement of 100ms"
            )

        # Verifiable Testing: 80%+ code coverage
        if self._config.validation.min_code_coverage_percent < 80.0:
            violations.append(
                f"Code coverage {self._config.validation.min_code_coverage_percent}% "
                f"below constitutional requirement of 80%"
            )

        # Graceful Degradation: Fallback must be enabled
        if not self._config.inference.fallback_to_cpu:
            violations.append(
                "CPU fallback disabled - violates graceful degradation requirement"
            )

        if not self._config.integration_config["mtga_voice_advisor"]["graceful_degradation"]:
            violations.append(
                "Graceful degradation disabled - violates constitutional requirement"
            )

        # Explainable AI: Explainability features must be enabled
        if not self._config.inference.enable_explainability:
            violations.append(
                "Explainability disabled - violates Explainable AI requirement"
            )

        # Data-Driven AI Development: Must use 17Lands data
        if not self._config.training.use_17lands_data:
            violations.append(
                "17Lands data usage disabled - violates data-driven AI requirement"
            )

        # Statistical validation requirements
        if self._config.validation.confidence_level < 0.95:
            violations.append(
                f"Confidence level {self._config.validation.confidence_level} "
                f"below constitutional requirement of 95%"
            )

        # Performance requirements
        if self._config.validation.min_win_rate_improvement < 0.25:
            violations.append(
                f"Win rate improvement requirement {self._config.validation.min_win_rate_improvement} "
                f"below constitutional minimum of 25%"
            )

        if violations:
            error_msg = "Constitutional requirements violated:\n" + "\n".join(f"  - {v}" for v in violations)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ All constitutional requirements satisfied")


# Global configuration manager instance
_config_manager = None


def get_rl_config_manager() -> RLConfigManager:
    """Get the global RL configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RLConfigManager()
    return _config_manager


def get_rl_config() -> RLSystemConfig:
    """Get the current RL configuration."""
    return get_rl_config_manager().get_config()


def setup_rl_config(config_path: Optional[str] = None) -> RLSystemConfig:
    """
    Setup RL configuration with validation.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Validated RL configuration
    """
    manager = get_rl_config_manager()
    config = manager.load_config(config_path)

    logger.info("RL configuration loaded and validated")
    logger.info(f"Model: {config.model.state_dim}D state, {config.model.action_dim}D action space")
    logger.info(f"Inference: Max {config.inference.max_inference_latency_ms}ms latency")
    logger.info(f"Training: Up to {config.training.max_episodes} episodes")

    return config


# Setup logging
import logging
logger = logging.getLogger(__name__)