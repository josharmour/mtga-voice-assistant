"""
Base RL Algorithm Classes

Provides foundational classes and interfaces for all RL algorithms,
ensuring constitutional compliance and consistent behavior across implementations.

Constitutional Requirements:
- Real-Time Responsiveness: Base classes designed for sub-100ms inference
- Verifiable Testing: Comprehensive validation and metric tracking
- Explainable AI: Base support for attention mechanisms and decision rationale
"""

import abc
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.config.rl_config import get_rl_config
from src.rl.utils.model_registry import get_model_registry

logger = logging.getLogger(__name__)


@dataclass
class RLAlgorithmConfig:
    """Configuration for RL algorithms with constitutional defaults."""

    # Performance requirements (constitutional)
    max_inference_time_ms: float = 100.0  # NON-NEGOTIABLE
    batch_processing: bool = True
    max_batch_size: int = 32

    # Training configuration
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    target_update_freq: int = 1000

    # Constitutional compliance
    enable_explainability: bool = True
    enable_performance_monitoring: bool = True
    enable_validation_checkpoints: bool = True

    # Safety and reliability
    gradient_clip_value: float = 1.0
    max_memory_usage_gb: float = 8.0


@dataclass
class RLAction:
    """Represents an RL action with metadata for explainability."""

    action_id: int
    action_type: str
    action_parameters: Dict[str, Any]
    confidence: float
    explanation: Optional[str] = None
    attention_weights: Optional[Dict[str, float]] = None
    q_value: Optional[float] = None
    uncertainty: Optional[float] = None


@dataclass
class RLState:
    """Represents an RL state with constitutional compliance tracking."""

    state_vector: torch.Tensor
    state_metadata: Dict[str, Any]
    timestamp: float
    phase: Optional[str] = None
    turn_number: Optional[int] = None
    valid_actions: Optional[List[int]] = None

    def __post_init__(self):
        if not isinstance(self.state_vector, torch.Tensor):
            self.state_vector = torch.tensor(self.state_vector, dtype=torch.float32)


@dataclass
class RLTransition:
    """Represents a single RL transition (state, action, reward, next_state)."""

    state: RLState
    action: RLAction
    reward: float
    next_state: Optional[RLState]
    done: bool
    episode_id: Optional[str] = None
    step_number: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class BaseRLAlgorithm(abc.ABC):
    """
    Abstract base class for all RL algorithms.

    Ensures constitutional compliance through:
    - Performance monitoring and validation
    - Explainability support
    - Graceful degradation mechanisms
    - Comprehensive testing support
    """

    def __init__(self, config: Optional[RLAlgorithmConfig] = None):
        """
        Initialize base RL algorithm.

        Args:
            config: Algorithm configuration with constitutional defaults
        """
        self.config = config or RLAlgorithmConfig()
        self.model_registry = get_model_registry()
        self.training_mode = False

        # Performance monitoring
        self.inference_times = []
        self.performance_violations = []

        # Constitutional compliance tracking
        self.last_validation_time = 0
        self.compliance_status = True

        # Explainability
        self.explainability_data = {}

        logger.info(f"Initialized {self.__class__.__name__} with constitutional compliance")

        # Validate configuration against requirements
        self._validate_configuration()

    @abc.abstractmethod
    def select_action(self, state: RLState, valid_actions: Optional[List[int]] = None) -> RLAction:
        """
        Select an action given the current state.

        Args:
            state: Current RL state
            valid_actions: List of valid action indices

        Returns:
            Selected action with metadata
        """
        pass

    @abc.abstractmethod
    def update(self, transitions: List[RLTransition]) -> Dict[str, float]:
        """
        Update the algorithm with transitions.

        Args:
            transitions: List of RL transitions

        Returns:
            Dictionary with training metrics
        """
        pass

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        pass

    def _validate_configuration(self) -> None:
        """
        Validate algorithm configuration against constitutional requirements.

        Raises:
            ValueError: If configuration violates requirements
        """
        violations = []

        # Real-Time Responsiveness check
        if self.config.max_inference_time_ms > 100.0:
            violations.append(f"Max inference time {self.config.max_inference_time_ms}ms exceeds 100ms requirement")

        # Explainability check
        if not self.config.enable_explainability:
            violations.append("Explainability disabled - violates Explainable AI requirement")

        # Performance monitoring check
        if not self.config.enable_performance_monitoring:
            violations.append("Performance monitoring disabled - violates Verifiable Testing requirement")

        if violations:
            error_msg = "Configuration violates constitutional requirements:\n" + "\n".join(f"  - {v}" for v in violations)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ Configuration satisfies all constitutional requirements")

    def _ensure_performance_requirements(self, inference_start_time: float) -> None:
        """
        Ensure inference meets performance requirements.

        Args:
            inference_start_time: Start time of inference
        """
        inference_time_ms = (time.time() - inference_start_time) * 1000
        self.inference_times.append(inference_time_ms)

        # Keep only recent measurements
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]

        # Check constitutional requirement
        if inference_time_ms > self.config.max_inference_time_ms:
            violation = {
                'timestamp': time.time(),
                'inference_time_ms': inference_time_ms,
                'limit_ms': self.config.max_inference_time_ms
            }
            self.performance_violations.append(violation)
            logger.warning(f"Performance violation: {inference_time_ms:.2f}ms > {self.config.max_inference_time_ms}ms")

        # Periodic validation
        if time.time() - self.last_validation_time > 60:  # Validate every minute
            self._validate_performance_trends()

    def _validate_performance_trends(self) -> None:
        """Validate performance trends and log compliance status."""
        if len(self.inference_times) < 10:
            return

        avg_inference_time = np.mean(self.inference_times[-100:])  # Last 100 inferences
        p95_inference_time = np.percentile(self.inference_times[-100:], 95)

        meets_requirement = (avg_inference_time <= self.config.max_inference_time_ms and
                           p95_inference_time <= self.config.max_inference_time_ms * 1.5)

        if meets_requirement:
            logger.debug(f"✅ Performance validation passed: avg={avg_inference_time:.2f}ms, p95={p95_inference_time:.2f}ms")
            self.compliance_status = True
        else:
            logger.error(f"❌ Performance validation failed: avg={avg_inference_time:.2f}ms, p95={p95_inference_time:.2f}ms")
            self.compliance_status = False

    def get_explainability_data(self) -> Dict[str, Any]:
        """
        Get explainability data for the last decision.

        Returns:
            Dictionary with explainability information
        """
        return {
            'algorithm_type': self.__class__.__name__,
            'explainability_enabled': self.config.enable_explainability,
            'data': self.explainability_data,
            'performance_stats': self.get_performance_stats()
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        if not self.inference_times:
            return {'status': 'No performance data available'}

        recent_times = self.inference_times[-100:]  # Last 100 inferences

        return {
            'avg_inference_time_ms': float(np.mean(recent_times)),
            'p95_inference_time_ms': float(np.percentile(recent_times, 95)),
            'p99_inference_time_ms': float(np.percentile(recent_times, 99)),
            'max_inference_time_ms': float(np.max(recent_times)),
            'min_inference_time_ms': float(np.min(recent_times)),
            'total_inferences': len(self.inference_times),
            'performance_violations': len(self.performance_violations),
            'constitutional_compliance': self.compliance_status,
            'target_max_time_ms': self.config.max_inference_time_ms
        }

    def validate_constitutional_compliance(self) -> Dict[str, Any]:
        """
        Validate constitutional compliance.

        Returns:
            Compliance validation results
        """
        violations = []

        # Check performance requirements
        if self.inference_times:
            avg_time = np.mean(self.inference_times[-100:])
            if avg_time > self.config.max_inference_time_ms:
                violations.append(f"Average inference time {avg_time:.2f}ms exceeds {self.config.max_inference_time_ms}ms requirement")

        # Check explainability
        if not self.config.enable_explainability:
            violations.append("Explainability not enabled")

        # Check performance monitoring
        if not self.config.enable_performance_monitoring:
            violations.append("Performance monitoring not enabled")

        # Check for recent violations
        recent_violations = [v for v in self.performance_violations
                           if time.time() - v['timestamp'] < 300]  # Last 5 minutes
        if recent_violations:
            violations.append(f"Recent performance violations: {len(recent_violations)}")

        compliance_results = {
            'algorithm': self.__class__.__name__,
            'compliant': len(violations) == 0,
            'violations': violations,
            'validation_time': time.time(),
            'performance_stats': self.get_performance_stats()
        }

        if compliance_results['compliant']:
            logger.info(f"✅ {self.__class__.__name__} constitutional compliance validated")
        else:
            logger.error(f"❌ {self.__class__.__name__} constitutional violations: {len(violations)}")

        return compliance_results

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking metrics."""
        self.inference_times.clear()
        self.performance_violations.clear()
        self.last_validation_time = time.time()
        logger.info("Performance tracking reset")

    def set_training_mode(self, training_mode: bool) -> None:
        """
        Set training mode.

        Args:
            training_mode: Whether to enable training mode
        """
        self.training_mode = training_mode
        logger.info(f"Training mode: {'enabled' if training_mode else 'disabled'}")

    def enable_explainability(self, enable: bool) -> None:
        """
        Enable or disable explainability features.

        Args:
            enable: Whether to enable explainability
        """
        if not enable and self.config.enable_explainability:
            logger.warning("Disabling explainability may violate constitutional requirements")

        # Note: In production, this would be controlled by configuration
        # and not dynamically changeable to ensure compliance

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get algorithm information for debugging and monitoring.

        Returns:
            Dictionary with algorithm information
        """
        return {
            'algorithm_name': self.__class__.__name__,
            'config': self.config.__dict__,
            'training_mode': self.training_mode,
            'compliance_status': self.compliance_status,
            'explainability_enabled': self.config.enable_explainability,
            'performance_monitoring_enabled': self.config.enable_performance_monitoring
        }


class BaseQAlgorithm(BaseRLAlgorithm):
    """
    Base class for Q-learning algorithms (DQN, CQL, etc.).

    Provides common functionality for Q-value based algorithms with
    constitutional compliance for real-time inference and explainability.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Optional[RLAlgorithmConfig] = None):
        """
        Initialize base Q-algorithm.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Algorithm configuration
        """
        super().__init__(config)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-network and target network
        self.q_network = None
        self.target_network = None
        self.optimizer = None

        # Training state
        self.training_step = 0
        self.last_target_update = 0

        logger.info(f"Initialized {self.__class__.__name__}: state_dim={state_dim}, action_dim={action_dim}")

    def get_q_values(self, state: RLState) -> torch.Tensor:
        """
        Get Q-values for a state.

        Args:
            state: RL state

        Returns:
            Q-values tensor
        """
        if self.q_network is None:
            raise RuntimeError("Q-network not initialized")

        with torch.no_grad():
            q_values = self.q_network(state.state_vector.unsqueeze(0))
            return q_values.squeeze(0)

    def _update_target_network(self) -> None:
        """Update target network with current Q-network parameters."""
        if self.target_network is not None:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug("Target network updated")

    def _should_update_target_network(self) -> bool:
        """Check if target network should be updated."""
        return (self.training_step - self.last_target_update) >= self.config.target_update_freq

    def _create_q_action(self, q_values: torch.Tensor, action_idx: int,
                       explanation: Optional[str] = None) -> RLAction:
        """
        Create an RLAction from Q-values.

        Args:
            q_values: Q-values tensor
            action_idx: Selected action index
            explanation: Optional explanation

        Returns:
            RLAction with explainability data
        """
        confidence = torch.softmax(q_values, dim=0)[action_idx].item()

        action = RLAction(
            action_id=action_idx,
            action_type=f"action_{action_idx}",
            action_parameters={},
            confidence=confidence,
            explanation=explanation,
            q_value=q_values[action_idx].item(),
            uncertainty=self._calculate_uncertainty(q_values, action_idx)
        )

        # Add explainability data
        if self.config.enable_explainability:
            self.explainability_data = {
                'q_values': q_values.tolist(),
                'selected_action': action_idx,
                'confidence': confidence,
                'uncertainty': action.uncertainty,
                'top_actions': torch.topk(q_values, 3).indices.tolist()
            }

        return action

    def _calculate_uncertainty(self, q_values: torch.Tensor, action_idx: int) -> float:
        """
        Calculate uncertainty for action selection.

        Args:
            q_values: Q-values tensor
            action_idx: Selected action index

        Returns:
            Uncertainty measure
        """
        # Simple uncertainty based on Q-value distribution
        q_std = torch.std(q_values).item()
        q_range = (torch.max(q_values) - torch.min(q_values)).item()

        # Normalize uncertainty
        uncertainty = q_std / (q_range + 1e-8)
        return min(uncertainty, 1.0)

    def select_action(self, state: RLState, valid_actions: Optional[List[int]] = None) -> RLAction:
        """
        Select action using Q-values with performance monitoring.

        Args:
            state: Current state
            valid_actions: List of valid action indices

        Returns:
            Selected action
        """
        start_time = time.time()

        try:
            q_values = self.get_q_values(state)

            # Filter valid actions if provided
            if valid_actions is not None:
                valid_mask = torch.full_like(q_values, float('-inf'))
                valid_mask[valid_actions] = 0
                q_values = q_values + valid_mask

            # Select action (epsilon-greedy during training, greedy during inference)
            if self.training_mode and np.random.random() < self.config.exploration_rate:
                action_idx = np.random.choice(valid_actions) if valid_actions else np.random.randint(self.action_dim)
                explanation = f"Random exploration (epsilon={self.config.exploration_rate:.2f})"
            else:
                action_idx = torch.argmax(q_values).item()
                explanation = f"Greedy selection: max Q-value"

            action = self._create_q_action(q_values, action_idx, explanation)

            # Ensure performance requirements
            self._ensure_performance_requirements(start_time)

            return action

        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Fallback to random action
            action_idx = np.random.randint(self.action_dim)
            return RLAction(
                action_id=action_idx,
                action_type="fallback_random",
                action_parameters={},
                confidence=0.0,
                explanation=f"Fallback due to error: {str(e)}"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for debugging.

        Returns:
            Dictionary with model information
        """
        info = self.get_algorithm_info()
        info.update({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'training_step': self.training_step,
            'q_network_initialized': self.q_network is not None,
            'target_network_initialized': self.target_network is not None
        })
        return info