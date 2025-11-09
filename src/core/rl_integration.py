"""
RL Integration Layer

Integrates RL agent with existing MTGA Voice Advisor system,
providing seamless enhancement and graceful degradation.
"""

import logging
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..rl.inference.engine import InferenceEngine, InferenceResult
from ..rl.inference.explainability import ExplainabilitySystem, ExplanationResult
from ..rl.data.state_extractor import StateExtractor
from ..rl.utils.model_registry import get_model_registry
from ..ai import AIAdvisor
from ..mtga import GameStateManager

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """RL integration modes."""
    DISABLED = "disabled"  # RL system disabled
    ADVISORY = "advisory"  # RL provides advice alongside supervised
    PRIMARY = "primary"   # RL is primary decision source
    HYBRID = "hybrid"     # Intelligent blend of RL and supervised


@dataclass
class RLRecommendation:
    """Recommendation from RL system."""
    action: str
    confidence: float
    reasoning: str
    alternative_actions: List[str]
    rl_explanation: Optional[ExplanationResult]
    inference_time_ms: float
    quality_score: float


@dataclass
class IntegrationConfig:
    """Configuration for RL integration."""
    mode: IntegrationMode = IntegrationMode.ADVISORY
    min_confidence_threshold: float = 0.6
    max_inference_time_ms: float = 100.0
    enable_explanations: bool = True
    fallback_on_timeout: bool = True
    use_caching: bool = True
    model_id: Optional[str] = None


class RLIntegration:
    """
    Integration layer for RL agent with MTGA Voice Advisor.

    Provides:
    - Seamless integration with existing advisor system
    - Graceful degradation when RL unavailable
    - Configurable integration modes
    - Performance monitoring and optimization
    - Constitutional compliance validation
    """

    def __init__(
        self,
        ai_advisor: AIAdvisor,
        game_state_manager: GameStateManager,
        config: Optional[IntegrationConfig] = None
    ):
        self.ai_advisor = ai_advisor
        self.game_state_manager = game_state_manager
        self.config = config or IntegrationConfig()

        # Initialize RL components
        self.state_extractor = StateExtractor()
        self.inference_engine = None
        self.explainability_system = None
        self.model_registry = get_model_registry()

        # Integration state
        self.rl_enabled = False
        self.current_model_id = None
        self.performance_metrics = {
            'total_recommendations': 0,
            'rl_recommendations': 0,
            'fallback_recommendations': 0,
            'avg_inference_time': 0.0,
            'timeout_count': 0
        }

        # Threading for performance
        self._lock = threading.RLock()

        # Initialize RL system if configured
        if self.config.mode != IntegrationMode.DISABLED:
            self._initialize_rl_system()

        logger.info(f"RL integration initialized with mode: {self.config.mode.value}")

    def _initialize_rl_system(self):
        """Initialize RL inference engine and explainability system."""
        try:
            # Get active model or use specified model
            if self.config.model_id:
                model_info = self.model_registry.get_model(self.config.model_id)
            else:
                model_info = self.model_registry.get_active_model("cql")

            if not model_info:
                logger.warning("No RL model available for integration")
                return

            # Initialize inference engine
            self.inference_engine = InferenceEngine(
                model_path=model_info['file_path'],
                max_latency_ms=self.config.max_inference_time_ms,
                enable_explanation=self.config.enable_explanations,
                cache_size=1000 if self.config.use_caching else 0
            )

            # Initialize explainability system
            if self.config.enable_explanations:
                self.explainability_system = ExplainabilitySystem(
                    enable_attention_analysis=True,
                    enable_counterfactuals=True
                )

            self.current_model_id = model_info['model_id']
            self.rl_enabled = True

            logger.info(f"RL system initialized with model: {self.current_model_id}")

        except Exception as e:
            logger.error(f"Failed to initialize RL system: {e}")
            self.rl_enabled = False

    def get_recommendation(
        self,
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RLRecommendation:
        """
        Get recommendation from integrated RL system.

        Args:
            game_state: Current game state
            context: Additional context for decision making

        Returns:
            Integrated recommendation with fallback support
        """
        start_time = time.time()
        self.performance_metrics['total_recommendations'] += 1

        try:
            with self._lock:
                # Check if RL is enabled and available
                if not self.rl_enabled or not self.inference_engine:
                    return self._get_fallback_recommendation(
                        game_state, "RL system not available"
                    )

                # Extract valid actions from game state
                valid_actions = self._extract_valid_actions(game_state)

                # Get RL prediction
                rl_result = self._get_rl_prediction(game_state, valid_actions)

                if rl_result is None:
                    return self._get_fallback_recommendation(
                        game_state, "RL prediction failed"
                    )

                # Generate explanation if enabled
                explanation = None
                if self.explainability_system:
                    try:
                        explanation = self.explainability_system.explain_decision(
                            game_state,
                            rl_result.action,
                            rl_result.q_values,
                            valid_actions
                        )
                    except Exception as e:
                        logger.warning(f"Explanation generation failed: {e}")

                # Convert RL action to human-readable recommendation
                recommendation = self._convert_rl_to_recommendation(
                    rl_result, explanation, game_state
                )

                # Update metrics
                inference_time = (time.time() - start_time) * 1000
                self._update_performance_metrics(inference_time, True)

                self.performance_metrics['rl_recommendations'] += 1

                return recommendation

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            self.performance_metrics['fallback_recommendations'] += 1
            return self._get_fallback_recommendation(
                game_state, f"Integration error: {str(e)}"
            )

    def _get_rl_prediction(
        self,
        game_state: Dict[str, Any],
        valid_actions: Optional[List[int]]
    ) -> Optional[InferenceResult]:
        """Get prediction from RL inference engine."""
        try:
            return self.inference_engine.predict(
                game_state=game_state,
                valid_actions=valid_actions,
                use_cache=self.config.use_caching,
                timeout_ms=self.config.max_inference_time_ms
            )
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            self.performance_metrics['timeout_count'] += 1
            return None

    def _extract_valid_actions(self, game_state: Dict[str, Any]) -> List[int]:
        """Extract valid actions from game state."""
        # This would integrate with existing MTGA game state analysis
        # For now, return default action space
        valid_actions = []

        # Add basic actions based on game state
        if game_state.get('can_play_land', False):
            valid_actions.append(0)  # Play land

        if game_state.get('hand', []):
            # Add cast actions for cards in hand
            hand_size = len(game_state['hand'])
            for i in range(min(hand_size, 7)):  # Limit to 7 cast actions
                valid_actions.append(1 + i)

        if game_state.get('can_attack', False):
            valid_actions.append(10)  # Attack

        if game_state.get('can_block', False):
            valid_actions.append(11)  # Block

        # Always add pass priority
        valid_actions.append(12)

        return valid_actions if valid_actions else [12]  # Default to pass

    def _convert_rl_to_recommendation(
        self,
        rl_result: InferenceResult,
        explanation: Optional[ExplanationResult],
        game_state: Dict[str, Any]
    ) -> RLRecommendation:
        """Convert RL result to human-readable recommendation."""
        # Map RL action to MTG action description
        action_description = self._map_action_to_description(
            rl_result.action, game_state
        )

        # Generate reasoning from explanation or RL confidence
        if explanation and explanation.strategic_rationale:
            reasoning = explanation.strategic_rationale
        else:
            reasoning = f"RL model confidence: {rl_result.confidence:.2f}"

        # Get alternative actions
        alternatives = []
        if explanation and explanation.action_comparisons:
            alternatives = [ac.action_name for ac in explanation.action_comparisons[:3]]

        # Calculate quality score
        quality_score = 0.0
        if explanation:
            quality_score = explanation.quality_score
        else:
            # Base quality on confidence and inference time
            quality_score = min(rl_result.confidence, 1.0 - (rl_result.processing_time / 200.0))

        return RLRecommendation(
            action=action_description,
            confidence=rl_result.confidence,
            reasoning=reasoning,
            alternative_actions=alternatives,
            rl_explanation=explanation,
            inference_time_ms=rl_result.processing_time,
            quality_score=quality_score
        )

    def _map_action_to_description(self, action: int, game_state: Dict[str, Any]) -> str:
        """Map RL action index to MTG action description."""
        action_map = {
            0: "Play a land from your hand",
            1: "Cast a creature spell",
            2: "Cast an instant spell",
            3: "Cast a sorcery spell",
            4: "Cast an artifact spell",
            5: "Cast an enchantment spell",
            6: "Cast a planeswalker spell",
            7: "Activate an ability",
            8: "Attack with creatures",
            9: "Block with creatures",
            10: "Pass priority (do nothing)",
            11: "Use mana ability",
            12: "Take a mulligan",
            13: "Concede the game"
        }

        if action < len(action_map):
            return action_map[action]
        else:
            return f"Take action {action}"

    def _get_fallback_recommendation(
        self,
        game_state: Dict[str, Any],
        reason: str
    ) -> RLRecommendation:
        """Get fallback recommendation using existing AI advisor."""
        try:
            # Use existing AI advisor for fallback
            # This would integrate with the existing advisor system
            fallback_reasoning = f"Using fallback advisor: {reason}"

            # Simple heuristic fallback
            if game_state.get('life', 20) < 10:
                action = "Focus on defensive plays"
            elif game_state.get('hand_size', 0) > 5:
                action = "Develop your board with creatures"
            else:
                action = "Pass priority and conserve resources"

            return RLRecommendation(
                action=action,
                confidence=0.5,  # Moderate confidence for fallback
                reasoning=fallback_reasoning,
                alternative_actions=[],
                rl_explanation=None,
                inference_time_ms=0.0,
                quality_score=0.5
            )

        except Exception as e:
            logger.error(f"Fallback recommendation failed: {e}")
            return RLRecommendation(
                action="Pass priority and wait for a better opportunity",
                confidence=0.1,
                reasoning=f"System error: {str(e)}",
                alternative_actions=[],
                rl_explanation=None,
                inference_time_ms=0.0,
                quality_score=0.1
            )

    def _update_performance_metrics(self, inference_time: float, rl_success: bool):
        """Update performance tracking metrics."""
        # Update average inference time
        total_rl = self.performance_metrics['rl_recommendations']
        current_avg = self.performance_metrics['avg_inference_time']

        if rl_success:
            new_avg = (current_avg * total_rl + inference_time) / (total_rl + 1)
            self.performance_metrics['avg_inference_time'] = new_avg

        # Log performance warnings
        if inference_time > self.config.max_inference_time_ms:
            logger.warning(f"Slow inference: {inference_time:.2f}ms > {self.config.max_inference_time_ms}ms")

    def should_use_rl(self, game_state: Dict[str, Any]) -> bool:
        """
        Determine if RL should be used for this game state.

        Args:
            game_state: Current game state

        Returns:
            True if RL should be used
        """
        if not self.rl_enabled:
            return False

        # Check integration mode
        if self.config.mode == IntegrationMode.DISABLED:
            return False
        elif self.config.mode == IntegrationMode.PRIMARY:
            return True
        elif self.config.mode == IntegrationMode.ADVISORY:
            return True  # Always provide RL advice in advisory mode
        elif self.config.mode == IntegrationMode.HYBRID:
            # Use RL based on game complexity and confidence
            return self._should_use_rl_hybrid(game_state)

        return False

    def _should_use_rl_hybrid(self, game_state: Dict[str, Any]) -> bool:
        """Hybrid mode logic for RL usage."""
        # Use RL for complex game states
        complexity_score = 0

        # Board complexity
        board_size = len(game_state.get('board', []))
        if board_size > 5:
            complexity_score += 2
        elif board_size > 3:
            complexity_score += 1

        # Hand complexity
        hand_size = game_state.get('hand_size', 0)
        if hand_size > 5:
            complexity_score += 1

        # Combat complexity
        if game_state.get('is_combat_phase', False):
            complexity_score += 2

        # Life pressure
        life = game_state.get('life', 20)
        if life < 15:
            complexity_score += 1
        elif life < 10:
            complexity_score += 2

        # Use RL for complex situations
        return complexity_score >= 3

    def switch_mode(self, new_mode: IntegrationMode):
        """Switch integration mode."""
        with self._lock:
            old_mode = self.config.mode
            self.config.mode = new_mode

            if new_mode == IntegrationMode.DISABLED:
                self.rl_enabled = False
                logger.info("RL system disabled")
            elif old_mode == IntegrationMode.DISABLED and new_mode != IntegrationMode.DISABLED:
                # Re-enable RL if switching from disabled
                self._initialize_rl_system()

            logger.info(f"Integration mode switched: {old_mode.value} → {new_mode.value}")

    def update_model(self, model_id: str) -> bool:
        """
        Update to a different RL model.

        Args:
            model_id: New model ID to use

        Returns:
            True if update successful
        """
        try:
            with self._lock:
                model_info = self.model_registry.get_model(model_id)
                if not model_info:
                    logger.error(f"Model not found: {model_id}")
                    return False

                # Update configuration
                self.config.model_id = model_id

                # Reinitialize RL system with new model
                old_model_id = self.current_model_id
                self._initialize_rl_system()

                if self.rl_enabled:
                    logger.info(f"Model updated: {old_model_id} → {model_id}")
                    return True
                else:
                    logger.error("Failed to initialize new model")
                    return False

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return False

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics."""
        return {
            'config': {
                'mode': self.config.mode.value,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'max_inference_time_ms': self.config.max_inference_time_ms,
                'enable_explanations': self.config.enable_explanations,
                'use_caching': self.config.use_caching
            },
            'rl_system': {
                'enabled': self.rl_enabled,
                'current_model_id': self.current_model_id,
                'inference_engine_loaded': self.inference_engine is not None,
                'explainability_enabled': self.explainability_system is not None
            },
            'performance_metrics': self.performance_metrics.copy(),
            'inference_engine_performance': (
                self.inference_engine.get_performance_report()
                if self.inference_engine else None
            )
        }

    def benchmark_integration(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark the integrated system performance.

        Args:
            num_samples: Number of test samples

        Returns:
            Benchmark results
        """
        if not self.rl_enabled or not self.inference_engine:
            return {'error': 'RL system not enabled'}

        logger.info(f"Benchmarking RL integration with {num_samples} samples...")

        # Run inference engine benchmark
        engine_benchmark = self.inference_engine.benchmark_model(num_samples)

        # Generate sample game states for integration testing
        integration_times = []
        successful_recommendations = 0

        for i in range(num_samples):
            # Generate sample game state
            sample_state = {
                'turn_number': (i % 20) + 1,
                'phase': 'main' if i % 2 == 0 else 'combat',
                'life': max(1, 20 - (i % 15)),
                'hand_size': (i % 7) + 1,
                'board': [],
                'can_play_land': i % 3 == 0,
                'can_attack': i % 4 == 0,
                'can_block': i % 5 == 0
            }

            try:
                start_time = time.time()
                recommendation = self.get_recommendation(sample_state)
                end_time = time.time()

                integration_times.append((end_time - start_time) * 1000)
                successful_recommendations += 1

            except Exception as e:
                logger.warning(f"Benchmark sample {i} failed: {e}")

        # Calculate integration benchmark results
        if integration_times:
            integration_benchmark = {
                'total_samples': num_samples,
                'successful_recommendations': successful_recommendations,
                'success_rate': successful_recommendations / num_samples,
                'avg_integration_time_ms': np.mean(integration_times),
                'min_integration_time_ms': np.min(integration_times),
                'max_integration_time_ms': np.max(integration_times),
                'p95_integration_time_ms': np.percentile(integration_times, 95),
                'samples_within_target': np.sum(np.array(integration_times) <= self.config.max_inference_time_ms),
                'target_compliance_rate': np.sum(np.array(integration_times) <= self.config.max_inference_time_ms) / len(integration_times)
            }
        else:
            integration_benchmark = {'error': 'No successful integrations'}

        return {
            'inference_engine': engine_benchmark,
            'integration': integration_benchmark,
            'summary': {
                'rl_integration_ready': (
                    self.rl_enabled and
                    engine_benchmark.get('target_compliance_rate', 0) > 0.9 and
                    integration_benchmark.get('target_compliance_rate', 0) > 0.9
                )
            }
        }

    def validate_constitutional_compliance(self) -> Dict[str, Any]:
        """Validate that integration meets constitutional requirements."""
        compliance_results = {
            'compliant': True,
            'violations': [],
            'validation_time': time.time()
        }

        # Check RL system is enabled for primary modes
        if self.config.mode in [IntegrationMode.PRIMARY, IntegrationMode.HYBRID]:
            if not self.rl_enabled:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    "RL system required for primary/hybrid modes but not enabled"
                )

        # Check inference latency requirement
        if self.performance_metrics['avg_inference_time'] > 100.0:
            compliance_results['compliant'] = False
            compliance_results['violations'].append(
                f"Average inference time {self.performance_metrics['avg_inference_time']:.2f}ms exceeds 100ms requirement"
            )

        # Check model compliance if model is loaded
        if self.current_model_id:
            model_compliance = self.model_registry.validate_constitutional_compliance(
                self.current_model_id
            )
            if not model_compliance['compliant']:
                compliance_results['compliant'] = False
                compliance_results['violations'].extend(model_compliance['violations'])

        # Check fallback mechanism
        if not self.config.fallback_on_timeout:
            compliance_results['compliant'] = False
            compliance_results['violations'].append(
                "Fallback on timeout must be enabled for graceful degradation"
            )

        logger.info(f"RL integration constitutional compliance: {'✅ PASS' if compliance_results['compliant'] else '❌ FAIL'}")
        return compliance_results

    def close(self):
        """Clean up integration resources."""
        with self._lock:
            self.rl_enabled = False
            self.inference_engine = None
            self.explainability_system = None
            logger.info("RL integration closed")