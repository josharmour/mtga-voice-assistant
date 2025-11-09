"""
Conservative Q-Learning Algorithm Implementation

Implements Conservative Q-Learning for offline RL with 17Lands replay data.
Optimized for MTG gameplay with constitutional compliance.

Constitutional Requirements:
- Data-Driven AI Development: Optimized for 17Lands offline data
- Real-Time Responsiveness: Sub-100ms inference latency
- Explainable AI: Detailed attention and decision rationale
- Verifiable Testing: Comprehensive validation and monitoring
"""

import logging
import math
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import BaseQAlgorithm, RLAlgorithmConfig, RLState, RLAction, RLTransition
from ..data.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class CQLNetwork(nn.Module):
    """
    Q-Network architecture for Conservative Q-Learning.

    Designed for 380+ dimensional state vectors with attention mechanisms
    for explainability and constitutional compliance.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None,
                 dueling_architecture: bool = True, attention_heads: int = 4):
        """
        Initialize CQL network.

        Args:
            state_dim: State vector dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            dueling_architecture: Whether to use dueling DQN architecture
            attention_heads: Number of attention heads for explainability
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling_architecture = dueling_architecture
        self.attention_heads = attention_heads

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Shared feature extractor
        self.feature_layers = nn.ModuleList()
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            self.feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        self.feature_dim = input_dim

        # Attention mechanism for explainability
        if attention_heads > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=attention_heads,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.attention = None

        # Dueling architecture
        if dueling_architecture:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        else:
            # Standard Q-value stream
            self.q_stream = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"CQLNetwork initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"dueling={dueling_architecture}, attention_heads={attention_heads}")

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Tuple of (q_values, attention_weights)
        """
        # Ensure correct input shape
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Feature extraction
        x = state
        for layer in self.feature_layers:
            x = layer(x)

        # Apply attention if enabled
        attention_weights = None
        if self.attention is not None:
            x_reshaped = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
            attended_x, attention_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = attended_x.squeeze(1)  # [batch_size, feature_dim]

        # Compute Q-values
        if self.dueling_architecture:
            # Dueling architecture
            value = self.value_stream(x)  # [batch_size, 1]
            advantage = self.advantage_stream(x)  # [batch_size, action_dim]

            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard architecture
            q_values = self.q_stream(x)  # [batch_size, action_dim]

        return q_values, attention_weights

    def get_attention_importance(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get attention importance for explainability.

        Args:
            state: State tensor

        Returns:
            Attention importance scores
        """
        if self.attention is None:
            return torch.ones(state.shape[-1]) / state.shape[-1]

        with torch.no_grad():
            q_values, attention_weights = self.forward(state)
            if attention_weights is not None:
                # Average attention across heads and return importance per feature
                return attention_weights.mean(dim=1).squeeze()
            else:
                return torch.ones(state.shape[-1]) / state.shape[-1]


class ConservativeQLearningConfig:
    """Configuration for Conservative Q-Learning algorithm."""

    def __init__(self,
                 # Dimensions
                 state_dim: Optional[int] = None,
                 action_dim: Optional[int] = None,

                 # Base RL configuration
                 max_inference_time_ms: float = 100.0,
                 batch_processing: bool = True,
                 max_batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 0.1,
                 target_update_freq: int = 1000,
                 enable_explainability: bool = True,
                 enable_performance_monitoring: bool = True,
                 enable_validation_checkpoints: bool = True,
                 gradient_clip_value: float = 1.0,
                 max_memory_usage_gb: float = 8.0,

                 # Conservative Q-Learning parameters
                 cql_alpha: float = 5.0,  # Conservative weight
                 cql_temp: float = 1.0,   # Temperature for log-sum-exp
                 cql_lagrange: bool = True,  # Automatic alpha tuning

                 # Network parameters
                 hidden_dims: Optional[List[int]] = None,
                 dueling_architecture: bool = True,
                 attention_heads: int = 4,

                 # Training parameters
                 batch_size: int = 32,
                 buffer_size: int = 100000,

                 # Loss function weights
                 q_loss_weight: float = 1.0,
                 cql_loss_weight: float = 1.0):

        # Set dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Base RL configuration
        self.max_inference_time_ms = max_inference_time_ms
        self.batch_processing = batch_processing
        self.max_batch_size = max_batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.target_update_freq = target_update_freq
        self.enable_explainability = enable_explainability
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_validation_checkpoints = enable_validation_checkpoints
        self.gradient_clip_value = gradient_clip_value
        self.max_memory_usage_gb = max_memory_usage_gb

        # Conservative Q-Learning parameters
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.cql_lagrange = cql_lagrange

        # Network parameters
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128]
        self.dueling_architecture = dueling_architecture
        self.attention_heads = attention_heads

        # Training parameters
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Loss function weights
        self.q_loss_weight = q_loss_weight
        self.cql_loss_weight = cql_loss_weight

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """
        Validate configuration against constitutional requirements and constraints.

        Raises:
            ValueError: If configuration violates requirements
        """
        violations = []

        # Constitutional requirements
        if self.max_inference_time_ms > 100.0:
            violations.append(f"Max inference time {self.max_inference_time_ms}ms exceeds 100ms requirement")

        if not self.enable_explainability:
            violations.append("Explainability disabled - violates Explainable AI requirement")

        if not self.enable_performance_monitoring:
            violations.append("Performance monitoring disabled - violates Verifiable Testing requirement")

        # CQL-specific validation
        if self.cql_alpha <= 0:
            violations.append("CQL alpha must be positive")

        if self.cql_temp <= 0:
            violations.append("CQL temperature must be positive")

        if self.action_dim is not None and self.action_dim <= 0:
            violations.append("Action dimension must be positive")

        if self.state_dim is not None and self.state_dim <= 0:
            violations.append("State dimension must be positive")

        # Network validation
        if self.attention_heads < 0:
            violations.append("Attention heads must be non-negative")

        if self.batch_size <= 0:
            violations.append("Batch size must be positive")

        if self.learning_rate <= 0:
            violations.append("Learning rate must be positive")

        if not (0 <= self.discount_factor <= 1):
            violations.append("Discount factor must be between 0 and 1")

        if violations:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
            raise ValueError(error_msg)

        logger.debug("✅ ConservativeQLearningConfig validation passed")

    @property
    def __dict__(self):
        """Get configuration as dictionary for compatibility."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_inference_time_ms': self.max_inference_time_ms,
            'batch_processing': self.batch_processing,
            'max_batch_size': self.max_batch_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'target_update_freq': self.target_update_freq,
            'enable_explainability': self.enable_explainability,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_validation_checkpoints': self.enable_validation_checkpoints,
            'gradient_clip_value': self.gradient_clip_value,
            'max_memory_usage_gb': self.max_memory_usage_gb,
            'cql_alpha': self.cql_alpha,
            'cql_temp': self.cql_temp,
            'cql_lagrange': self.cql_lagrange,
            'hidden_dims': self.hidden_dims,
            'dueling_architecture': self.dueling_architecture,
            'attention_heads': self.attention_heads,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'q_loss_weight': self.q_loss_weight,
            'cql_loss_weight': self.cql_loss_weight
        }


class ConservativeQLearning(BaseQAlgorithm):
    """
    Conservative Q-Learning algorithm for offline RL with 17Lands data.

    Implements CQL with:
    - Conservative offline learning to avoid overestimation
    - Attention mechanisms for explainability
    - Performance monitoring for constitutional compliance
    - Real-time inference optimization
    """

    def __init__(self, state_dim_or_config=None, action_dim=None,
                 config: Optional[ConservativeQLearningConfig] = None):
        """
        Initialize Conservative Q-Learning algorithm.

        Supports multiple calling patterns:
        1. ConservativeQLearning(config)  # Config only
        2. ConservativeQLearning(state_dim, action_dim, config)  # Individual parameters

        Args:
            state_dim_or_config: State vector dimension or ConservativeQLearningConfig
            action_dim: Action space dimension (if state_dim_or_config is state_dim)
            config: Algorithm configuration
        """
        # Handle different calling patterns
        if isinstance(state_dim_or_config, ConservativeQLearningConfig):
            # Pattern 1: ConservativeQLearning(config)
            cql_config = state_dim_or_config
            state_dim = cql_config.state_dim
            action_dim = cql_config.action_dim
            if config is not None:
                # Override with provided config
                cql_config = config
        else:
            # Pattern 2: ConservativeQLearning(state_dim, action_dim, config)
            state_dim = state_dim_or_config
            if config is None:
                cql_config = ConservativeQLearningConfig(state_dim=state_dim, action_dim=action_dim)
            else:
                cql_config = config
                # Ensure config has correct dimensions
                if cql_config.state_dim is None:
                    cql_config.state_dim = state_dim
                if cql_config.action_dim is None:
                    cql_config.action_dim = action_dim

        # Validate dimensions
        if state_dim is None:
            raise ValueError("state_dim must be provided")
        if action_dim is None:
            raise ValueError("action_dim must be provided")

        super().__init__(state_dim, action_dim, cql_config)
        self.cql_config = self.config

        # Initialize networks
        self.q_network = CQLNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.cql_config.hidden_dims,
            dueling_architecture=self.cql_config.dueling_architecture,
            attention_heads=self.cql_config.attention_heads
        )

        self.target_network = CQLNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.cql_config.hidden_dims,
            dueling_architecture=self.cql_config.dueling_architecture,
            attention_heads=self.cql_config.attention_heads
        )

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.cql_config.learning_rate)

        # Training state
        self.cql_alpha = self.cql_config.cql_alpha
        self.log_alpha = torch.tensor(math.log(self.cql_alpha)).float()
        if self.cql_config.cql_lagrange:
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        # Loss tracking
        self.loss_history = []

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move networks to device
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # Training counters for target network updates
        self.updates_since_target = 0

        logger.info(f"ConservativeQLearning initialized: state_dim={state_dim}, action_dim={action_dim}")

    def to(self, device: torch.device) -> 'ConservativeQLearning':
        """
        Move algorithm to specified device.

        Args:
            device: Target device (cuda/cpu)

        Returns:
            Self for method chaining
        """
        self.device = device
        self.q_network.to(device)
        self.target_network.to(device)

        # Move log_alpha to device if it exists
        if hasattr(self, 'log_alpha'):
            self.log_alpha = self.log_alpha.to(device)

        return self

    def select_action(self, state: RLState, valid_actions: Optional[List[int]] = None) -> RLAction:
        """
        Select action with graceful degradation for network failures.

        Args:
            state: Current state
            valid_actions: List of valid action indices

        Returns:
            Selected action with fallback handling
        """
        start_time = time.time()

        try:
            # Get Q-values with potential graceful degradation
            q_values = self.get_q_values(state)

            # Check for NaN or infinite values
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                logger.warning("Q-values contain NaN or infinity, using fallback")
                return self._create_fallback_action(state, "invalid Q-values")

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
            logger.error(f"CQL action selection failed: {e}")
            # Return fallback action
            fallback_action = self._create_fallback_action(state, f"network error: {str(e)}")
            self._ensure_performance_requirements(start_time)
            return fallback_action

    def _create_fallback_action(self, state: RLState, reason: str) -> RLAction:
        """
        Create fallback action for graceful degradation.

        Args:
            state: Current state
            reason: Reason for fallback

        Returns:
            Fallback action with low confidence
        """
        # Choose a safe action (e.g., pass priority)
        action_id = 0  # Assume action 0 is always safe (pass priority)

        return RLAction(
            action_id=action_id,
            action_type="fallback_pass",
            action_parameters={},
            confidence=0.0,
            explanation=f"Fallback action due to {reason}. Defaulting to safe action.",
            attention_weights=None,
            q_value=0.0,
            uncertainty=1.0
        )

    def get_q_values(self, state: RLState) -> torch.Tensor:
        """
        Get Q-values for a state with graceful degradation.

        Args:
            state: RL state

        Returns:
            Q-values tensor
        """
        if self.q_network is None:
            raise RuntimeError("Q-network not initialized")

        try:
            with torch.no_grad():
                q_values, _ = self.q_network(state.state_vector.unsqueeze(0))
                return q_values.squeeze(0)
        except Exception as e:
            logger.error(f"Q-value computation failed: {e}")
            # Return zero Q-values for graceful degradation
            return torch.zeros(self.action_dim, dtype=torch.float32)

    def update(self, transitions: List[RLTransition]) -> Dict[str, float]:
        """
        Update algorithm with batch of transitions.

        Args:
            transitions: List of RL transitions

        Returns:
            Dictionary with training metrics
        """
        if len(transitions) < self.cql_config.batch_size:
            return {'error': 'Insufficient transitions for batch'}

        # Prepare batch
        try:
            batch = self._prepare_batch(transitions)
            metrics = self._train_step(batch)
            return metrics

        except Exception as e:
            logger.error(f"CQL update failed: {e}")
            return {'error': str(e)}

    def _prepare_batch(self, transitions: List[RLTransition]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        batch_size = self.cql_config.batch_size
        transitions = transitions[:batch_size]

        # Extract components
        states = torch.stack([t.state.state_vector for t in transitions])
        actions = torch.tensor([t.action.action_id if hasattr(t.action, 'action_id') else 0 for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)

        # Handle next states (None for terminal states)
        next_states = []
        for t in transitions:
            if t.next_state is not None:
                next_states.append(t.next_state.state_vector)
            else:
                # Create zero tensor for terminal states
                next_states.append(torch.zeros_like(states[0]))
        next_states = torch.stack(next_states)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()

        # Current Q-values
        current_q_values, _ = self.q_network(batch['states'])
        current_q_selected = current_q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_network(batch['next_states'])
            next_q_max = next_q_values.max(dim=1)[0]
            target_q = batch['rewards'] + (1 - batch['dones']) * self.cql_config.discount_factor * next_q_max

        # Q-learning loss
        q_loss = F.mse_loss(current_q_selected, target_q)

        # CQL conservative loss
        cql_loss = self._calculate_cql_loss(batch['states'], current_q_values)

        # Total loss
        total_loss = (self.cql_config.q_loss_weight * q_loss +
                     self.cql_config.cql_loss_weight * cql_loss * self.cql_alpha)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cql_config.gradient_clip_value)
        self.optimizer.step()

        # Update alpha if using Lagrange multiplier
        if self.cql_config.cql_lagrange:
            self._update_alpha(cql_loss)

        # Update target network
        if self.training_step % self.cql_config.target_update_freq == 0:
            self._update_target_network()

        self.training_step += 1

        # Track loss
        self.loss_history.append(total_loss.item())

        return {
            'total_loss': total_loss.item(),
            'q_loss': q_loss.item(),
            'cql_loss': cql_loss.item(),
            'alpha': self.cql_alpha,
            'current_q_mean': current_q_selected.mean().item(),
            'target_q_mean': target_q.mean().item()
        }

    def _calculate_cql_loss(self, states: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        """Calculate Conservative Q-Learning loss."""
        # Sample actions for CQL
        with torch.no_grad():
            # Sample random actions and current policy actions
            batch_size = states.shape[0]
            random_actions = torch.randint(0, self.action_dim, (batch_size,))
            current_actions = q_values.argmax(dim=1)

            # Combine actions
            sample_actions = torch.cat([random_actions, current_actions], dim=0)
            sample_states = states.repeat(2, 1)

            # Get Q-values for sampled actions
            sampled_q_values, _ = self.q_network(sample_states)
            sampled_q_selected = sampled_q_values.gather(1, sample_actions.unsqueeze(1)).squeeze()

        # Calculate log-sum-exp
        q_exp = q_values / self.cql_config.cql_temp
        log_sum_exp = torch.logsumexp(q_exp, dim=1)

        # CQL loss: difference between current Q-values and expected Q-values under current policy
        current_q_selected = q_values.gather(1, q_values.argmax(dim=1).unsqueeze(1)).squeeze()
        cql_loss = (log_sum_exp - current_q_selected).mean()

        return cql_loss

    def _update_alpha(self, cql_loss: torch.Tensor) -> None:
        """Update alpha parameter using Lagrange multiplier."""
        alpha_loss = -(self.log_alpha * (cql_loss.detach() - 10.0))  # Target margin of 10

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.cql_alpha = torch.exp(self.log_alpha).item()

    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'cql_alpha': self.cql_alpha,
            'config': self.cql_config.__dict__,
            'loss_history': self.loss_history[-1000:]  # Save last 1000 losses
        }

        if self.cql_config.cql_lagrange and hasattr(self, 'alpha_optimizer'):
            checkpoint['alpha_optimizer_state'] = self.alpha_optimizer.state_dict()
            checkpoint['log_alpha'] = self.log_alpha.item()

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.training_step = checkpoint['training_step']
        self.cql_alpha = checkpoint['cql_alpha']
        self.loss_history = checkpoint.get('loss_history', [])

        if self.cql_config.cql_lagrange and 'alpha_optimizer_state' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state'])
            self.log_alpha = torch.tensor(checkpoint['log_alpha']).float()

        logger.info(f"Model loaded from {path}")

    def get_explainability_data(self) -> Dict[str, Any]:
        """
        Get explainability data for last decision.

        Returns:
            Dictionary with explainability information
        """
        base_data = super().get_explainability_data()

        # Add CQL-specific explainability
        cql_data = {
            'algorithm_type': 'Conservative Q-Learning',
            'cql_alpha': self.cql_alpha,
            'training_step': self.training_step,
            'attention_heads': self.cql_config.attention_heads,
            'dueling_architecture': self.cql_config.dueling_architecture,
            'loss_trend': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            'data': self.explainability_data
        }

        base_data.update(cql_data)
        return base_data

    def get_attention_weights(self, state: RLState) -> Optional[torch.Tensor]:
        """
        Get attention weights for explainability.

        Args:
            state: RL state

        Returns:
            Attention weights tensor
        """
        if self.cql_config.attention_heads == 0:
            return None

        with torch.no_grad():
            state_tensor = state.state_vector.unsqueeze(0)
            _, attention_weights = self.q_network(state_tensor)
            return attention_weights

    def validate_constitutional_compliance(self) -> Dict[str, Any]:
        """
        Validate CQL algorithm against constitutional requirements.

        Returns:
            Compliance validation results
        """
        base_compliance = super().validate_constitutional_compliance()

        # CQL-specific validation
        violations = base_compliance.get('violations', []).copy()

        # Check CQL parameters
        if self.cql_alpha <= 0:
            violations.append("CQL alpha must be positive for conservative learning")

        # Check loss trends (should be decreasing for healthy training)
        if len(self.loss_history) > 100:
            recent_loss = np.mean(self.loss_history[-50:])
            older_loss = np.mean(self.loss_history[-100:-50])
            if recent_loss > older_loss * 1.5:  # Loss increased significantly
                violations.append("Training loss increasing - possible instability")

        # Check network configuration
        if self.cql_config.attention_heads == 0 and self.config.enable_explainability:
            violations.append("Attention disabled but explainability required")

        cql_compliance = {
            'algorithm': 'ConservativeQLearning',
            'compliant': len(violations) == 0,
            'violations': violations,
            'validation_time': time.time(),
            'cql_parameters': {
                'alpha': self.cql_alpha,
                'temp': self.cql_config.cql_temp,
                'lagrange': self.cql_config.cql_lagrange,
                'attention_heads': self.cql_config.attention_heads
            },
            'training_stats': {
                'training_step': self.training_step,
                'recent_loss': np.mean(self.loss_history[-50:]) if self.loss_history else 0.0,
                'loss_trend': 'decreasing' if len(self.loss_history) > 100 and recent_loss < older_loss else 'unknown'
            },
            'performance_stats': self.get_performance_stats()
        }

        if cql_compliance['compliant']:
            logger.info("✅ Conservative Q-Learning constitutional compliance validated")
        else:
            logger.error(f"❌ Conservative Q-Learning constitutional violations: {len(violations)}")

        return cql_compliance

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get CQL training statistics."""
        stats = {
            'training_step': self.training_step,
            'cql_alpha': self.cql_alpha,
            'total_training_losses': len(self.loss_history),
            'config': self.cql_config.__dict__
        }

        if self.loss_history:
            recent_losses = self.loss_history[-1000:]  # Last 1000 losses
            stats['loss_statistics'] = {
                'current_loss': self.loss_history[-1],
                'avg_loss_recent': np.mean(recent_losses),
                'loss_std_recent': np.std(recent_losses),
                'min_loss_recent': np.min(recent_losses),
                'max_loss_recent': np.max(recent_losses),
                'loss_trend': 'improving' if len(recent_losses) > 100 else 'insufficient_data'
            }

        return stats

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for constitutional compliance monitoring.

        Returns:
            Performance metrics including inference timing and accuracy
        """
        try:
            # Generate test state for performance measurement
            test_state = torch.randn(1, self.config.state_dim).to(self.device)

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                q_values = self.q_network(test_state)
                action = torch.argmax(q_values, dim=1).item()
            inference_time = (time.time() - start_time) * 1000

            # Calculate Q-value statistics
            with torch.no_grad():
                q_values_np = q_values.cpu().numpy().flatten()

            metrics = {
                'avg_q_value': float(np.mean(q_values_np)),
                'max_q_value': float(np.max(q_values_np)),
                'min_q_value': float(np.min(q_values_np)),
                'q_value_std': float(np.std(q_values_np)),
                'selected_action': int(action),
                'inference_time_ms': float(inference_time),
                'performance_compliant': inference_time <= self.config.max_inference_time_ms
            }

            # Add training statistics
            if self.loss_history:
                metrics['recent_loss'] = float(np.mean(self.loss_history[-10:]))

            return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                'error': str(e),
                'performance_compliant': False,
                'inference_time_ms': float('inf')
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for checkpointing and versioning.

        Returns:
            Model configuration and state information
        """
        try:
            model_info = {
                'algorithm': 'ConservativeQLearning',
                'state_dim': self.config.state_dim,
                'action_dim': self.config.action_dim,
                'network_config': {
                    'hidden_dims': self.cql_config.hidden_dims,
                    'dueling_architecture': self.cql_config.dueling_architecture,
                    'attention_heads': self.cql_config.attention_heads
                },
                'training_config': {
                    'learning_rate': self.config.learning_rate,
                    'cql_alpha': self.cql_alpha,
                    'cql_temp': self.cql_config.cql_temp,
                    'batch_size': self.config.batch_size
                },
                'training_state': {
                    'training_step': self.training_step,
                    'target_update_freq': self.config.target_update_freq,
                    'updates_since_target': self.updates_since_target
                },
                'model_parameters': {
                    'total_params': sum(p.numel() for p in self.q_network.parameters()),
                    'trainable_params': sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
                },
                'device': str(self.device),
                'timestamp': time.time()
            }

            # Add network state dict if available
            if hasattr(self.q_network, 'state_dict'):
                model_info['network_state'] = {
                    'q_network_keys': list(self.q_network.state_dict().keys()),
                    'target_network_keys': list(self.target_network.state_dict().keys()) if hasattr(self, 'target_network') else []
                }

            return model_info

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e), 'algorithm': 'ConservativeQLearning'}

    def explain_action(self, state: np.ndarray, action: int) -> Dict[str, Any]:
        """
        Explain why a specific action was chosen.

        Args:
            state: State vector (380-dim)
            action: Selected action (0-15)

        Returns:
            Explanation dictionary with reasoning and confidence
        """
        try:
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            if len(state) != self.config.state_dim:
                raise ValueError(f"State dimension mismatch: expected {self.config.state_dim}, got {len(state)}")

            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get Q-values and attention weights
            with torch.no_grad():
                if self.cql_config.attention_heads > 0:
                    q_values, attention_weights = self.q_network(state_tensor)
                else:
                    q_values = self.q_network(state_tensor)
                    attention_weights = None

            q_values_np = q_values.cpu().numpy().flatten()

            # Calculate action statistics
            all_q_values = q_values_np.tolist()
            selected_q_value = float(q_values_np[action])
            max_q_value = float(np.max(q_values_np))
            min_q_value = float(np.min(q_values_np))

            # Calculate confidence
            q_diff = max_q_value - min_q_value
            if q_diff > 0:
                confidence = (selected_q_value - min_q_value) / q_diff
            else:
                confidence = 0.5

            # Sort actions by Q-value
            sorted_actions = sorted(
                [(i, float(q)) for i, q in enumerate(all_q_values)],
                key=lambda x: x[1],
                reverse=True
            )

            # Create explanation
            explanation = {
                'action': int(action),
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'q_value': selected_q_value,
                'reasoning': {
                    'selected_action_rank': next(i for i, (act, _) in enumerate(sorted_actions) if act == action) + 1,
                    'q_value_range': {'min': min_q_value, 'max': max_q_value},
                    'advantage_over_best': max_q_value - selected_q_value,
                    'advantage_over_worst': selected_q_value - min_q_value
                },
                'alternative_actions': [
                    {'action': int(act), 'q_value': float(q)}
                    for act, q in sorted_actions[:5] if act != action
                ],
                'explainability_available': self.cql_config.attention_heads > 0,
                'timestamp': time.time()
            }

            # Add attention-based explanations if available
            if attention_weights is not None:
                try:
                    attention_np = attention_weights.cpu().numpy()
                    if len(attention_np.shape) >= 2:
                        # Average across attention heads and sequence positions
                        attention_importance = np.mean(attention_np, axis=(0, 1)) if len(attention_np.shape) > 2 else attention_np[0]

                        # Get most important state features
                        if len(attention_importance) >= self.config.state_dim:
                            top_features = np.argsort(attention_importance)[-5:]  # Top 5 features
                            explanation['attention_analysis'] = {
                                'top_features': [
                                    {
                                        'feature_index': int(idx),
                                        'importance': float(attention_importance[idx]),
                                        'state_value': float(state[idx])
                                    }
                                    for idx in top_features
                                ],
                                'total_attention_importance': float(np.sum(attention_importance))
                            }
                except Exception as e:
                    logger.warning(f"Failed to process attention weights: {e}")
                    explanation['attention_analysis'] = {'error': str(e)}

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain action: {e}")

            # Handle both int and RLAction types
            action_id = action
            if hasattr(action, 'action_id'):
                action_id = action.action_id
            elif hasattr(action, '__int__'):
                action_id = int(action)

            return {
                'action': int(action_id),
                'error': str(e),
                'confidence': 0.0,
                'explainability_available': False,
                'timestamp': time.time()
            }


# Factory function
def create_conservative_q_learning(state_dim: int, action_dim: int,
                                 config: Optional[ConservativeQLearningConfig] = None) -> ConservativeQLearning:
    """
    Create Conservative Q-Learning algorithm.

    Args:
        state_dim: State vector dimension
        action_dim: Action space dimension
        config: Algorithm configuration

    Returns:
        Initialized CQL algorithm
    """
    return ConservativeQLearning(state_dim, action_dim, config)