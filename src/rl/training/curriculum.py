"""
Curriculum Learning System for MTG RL Training

Progressive training strategy that starts with simple game states
and gradually increases complexity based on learner performance.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages from simple to complex."""
    BASIC_ACTIONS = "basic_actions"
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    COMBAT_FOCUS = "combat_focus"
    ADVANCED_SYNERGIES = "advanced_synergies"
    FULL_GAME = "full_game"


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    stage: CurriculumStage
    difficulty_threshold: float  # Performance threshold to advance
    min_episodes: int  # Minimum episodes before advancement
    max_episodes: int  # Maximum episodes before forced advancement
    state_filter: Optional[Dict[str, Any]]  # State filtering criteria
    reward_multiplier: float  # Reward scaling for this stage
    description: str


class CurriculumLearning:
    """
    Curriculum learning system for progressive RL training.

    Implements 4-stage progressive training:
    1. Basic Actions - Fundamental game mechanics
    2. Single Turn - Individual turn optimization
    3. Multi-Turn - Short-term planning (2-3 turns)
    4. Full Game - Complete strategic understanding
    """

    def __init__(self):
        self.stages = self._define_stages()
        self.current_stage_idx = 0
        self.stage_metrics = {}
        self.episode_count = 0
        self.stage_performance = []

        logger.info("Curriculum learning system initialized")

    def _define_stages(self) -> List[StageConfig]:
        """Define curriculum learning stages."""
        return [
            StageConfig(
                stage=CurriculumStage.BASIC_ACTIONS,
                difficulty_threshold=0.65,  # 65% success rate to advance
                min_episodes=5000,
                max_episodes=15000,
                state_filter={
                    "turn_number": lambda x: x <= 5,  # Early game only
                    "board_complexity": lambda x: x <= 3,  # Few permanents
                    "hand_size": lambda x: 3 <= x <= 7  # Reasonable hand size
                },
                reward_multiplier=1.2,  # Boost rewards for learning
                description="Basic mechanics and fundamental actions"
            ),
            StageConfig(
                stage=CurriculumStage.SINGLE_TURN,
                difficulty_threshold=0.70,
                min_episodes=8000,
                max_episodes=20000,
                state_filter={
                    "turn_number": lambda x: 5 <= x <= 10,  # Early to mid game
                    "board_complexity": lambda x: x <= 5,
                    "combat_phase": True  # Include combat situations
                },
                reward_multiplier=1.1,
                description="Single turn optimization and tactical play"
            ),
            StageConfig(
                stage=CurriculumStage.MULTI_TURN,
                difficulty_threshold=0.75,
                min_episodes=12000,
                max_episodes=30000,
                state_filter={
                    "turn_number": lambda x: 8 <= x <= 15,  # Mid to late game
                    "board_complexity": lambda x: 3 <= x <= 8,
                    "strategic_depth": True  # Complex board states
                },
                reward_multiplier=1.0,
                description="Multi-turn planning and strategic positioning"
            ),
            StageConfig(
                stage=CurriculumStage.FULL_GAME,
                difficulty_threshold=0.80,
                min_episodes=20000,
                max_episodes=100000,  # No forced advancement
                state_filter=None,  # All game states
                reward_multiplier=1.0,
                description="Complete game understanding and mastery"
            )
        ]

    def get_current_stage(self) -> StageConfig:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def should_advance_stage(self, performance_metrics: Dict[str, float]) -> bool:
        """
        Determine if training should advance to the next stage.

        Args:
            performance_metrics: Current performance metrics

        Returns:
            True if ready to advance to next stage
        """
        current_stage = self.get_current_stage()

        # Check minimum episodes requirement
        if self.episode_count < current_stage.min_episodes:
            return False

        # Check performance threshold
        win_rate = performance_metrics.get("win_rate", 0.0)
        decision_quality = performance_metrics.get("decision_quality", 0.0)
        combined_performance = (win_rate + decision_quality) / 2.0

        if combined_performance >= current_stage.difficulty_threshold:
            logger.info(f"Stage advancement criteria met: {combined_performance:.3f} >= {current_stage.difficulty_threshold}")
            return True

        # Check maximum episodes (forced advancement)
        if self.episode_count >= current_stage.max_episodes:
            logger.warning(f"Forced advancement due to max episodes: {self.episode_count} >= {current_stage.max_episodes}")
            return True

        return False

    def advance_to_next_stage(self) -> bool:
        """
        Advance to the next curriculum stage.

        Returns:
            True if successfully advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            logger.info("Already at final curriculum stage")
            return False

        self.current_stage_idx += 1
        self.episode_count = 0  # Reset episode counter for new stage
        current_stage = self.get_current_stage()

        logger.info(f"Advanced to curriculum stage: {current_stage.stage.value}")
        logger.info(f"Stage description: {current_stage.description}")

        return True

    def filter_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Check if state meets current stage criteria.

        Args:
            state_data: Game state data

        Returns:
            True if state is appropriate for current stage
        """
        current_stage = self.get_current_stage()

        if current_stage.state_filter is None:
            return True  # No filtering for final stage

        for key, filter_func in current_stage.state_filter.items():
            if key not in state_data:
                continue

            if not filter_func(state_data[key]):
                return False

        return True

    def adjust_reward(self, base_reward: float, state_data: Dict[str, Any]) -> float:
        """
        Adjust reward based on curriculum stage.

        Args:
            base_reward: Original reward value
            state_data: Game state data

        Returns:
            Adjusted reward value
        """
        current_stage = self.get_current_stage()

        # Apply stage-specific multiplier
        adjusted_reward = base_reward * current_stage.reward_multiplier

        # Apply difficulty bonuses for early stages
        if self.current_stage_idx < 2:  # Basic or Single Turn stages
            complexity_bonus = self._calculate_complexity_bonus(state_data)
            adjusted_reward += complexity_bonus

        return adjusted_reward

    def _calculate_complexity_bonus(self, state_data: Dict[str, Any]) -> float:
        """Calculate complexity bonus for early learning stages."""
        bonus = 0.0

        # Bonus for handling complex board states
        board_complexity = state_data.get("board_complexity", 0)
        if board_complexity > 3:
            bonus += 0.1

        # Bonus for tactical decisions
        if state_data.get("is_combat_phase", False):
            bonus += 0.15

        # Bonus for resource management
        if state_data.get("mana_efficiency", 0) > 0.8:
            bonus += 0.1

        return bonus

    def update_metrics(self, reward: float, done: bool, info: Dict[str, Any]):
        """
        Update curriculum learning metrics.

        Args:
            reward: Episode reward
            done: Whether episode is finished
            info: Additional episode information
        """
        self.episode_count += 1

        current_stage = self.get_current_stage()
        stage_name = current_stage.stage.value

        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = {
                "episodes": 0,
                "total_reward": 0.0,
                "wins": 0,
                "decisions": 0,
                "good_decisions": 0
            }

        metrics = self.stage_metrics[stage_name]
        metrics["episodes"] += 1
        metrics["total_reward"] += reward
        metrics["decisions"] += info.get("decisions_made", 0)
        metrics["good_decisions"] += info.get("good_decisions", 0)

        if done:
            if info.get("won", False):
                metrics["wins"] += 1

        # Log progress periodically
        if self.episode_count % 1000 == 0:
            self._log_stage_progress()

    def _log_stage_progress(self):
        """Log current stage progress."""
        current_stage = self.get_current_stage()
        stage_name = current_stage.stage.value
        metrics = self.stage_metrics[stage_name]

        if metrics["episodes"] > 0:
            win_rate = metrics["wins"] / metrics["episodes"]
            avg_reward = metrics["total_reward"] / metrics["episodes"]
            decision_quality = metrics["good_decisions"] / max(metrics["decisions"], 1)

            logger.info(f"Stage {stage_name} Progress:")
            logger.info(f"  Episodes: {metrics['episodes']}")
            logger.info(f"  Win Rate: {win_rate:.3f}")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")
            logger.info(f"  Decision Quality: {decision_quality:.3f}")
            logger.info(f"  Progress: {self.episode_count}/{current_stage.max_episodes}")

    def get_stage_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of curriculum progress."""
        summary = {
            "current_stage": self.get_current_stage().stage.value,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_episodes": self.episode_count,
            "overall_metrics": {}
        }

        # Aggregate metrics across all stages
        total_episodes = sum(m["episodes"] for m in self.stage_metrics.values())
        total_wins = sum(m["wins"] for m in self.stage_metrics.values())
        total_reward = sum(m["total_reward"] for m in self.stage_metrics.values())

        if total_episodes > 0:
            summary["overall_metrics"] = {
                "total_episodes": total_episodes,
                "overall_win_rate": total_wins / total_episodes,
                "average_reward": total_reward / total_episodes,
                "stages_completed": len([m for m in self.stage_metrics.values() if m["episodes"] > 0])
            }

        return summary

    def reset(self):
        """Reset curriculum learning to first stage."""
        self.current_stage_idx = 0
        self.episode_count = 0
        self.stage_metrics = {}
        self.stage_performance = []
        logger.info("Curriculum learning reset to first stage")


# Convenience function for creating curriculum learning system
def create_curriculum() -> CurriculumLearning:
    """Create and configure curriculum learning system."""
    return CurriculumLearning()


# Stage-specific training utilities
def get_stage_training_params(stage: CurriculumStage) -> Dict[str, Any]:
    """Get training parameters optimized for specific curriculum stage."""

    base_params = {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995
    }

    stage_adjustments = {
        CurriculumStage.BASIC_ACTIONS: {
            "learning_rate": 2e-4,  # Higher LR for fast learning
            "batch_size": 32,  # Smaller batches for stable learning
            "epsilon_decay": 0.99  # Slower decay for more exploration
        },
        CurriculumStage.SINGLE_TURN: {
            "learning_rate": 1.5e-4,
            "batch_size": 48,
            "epsilon_decay": 0.992
        },
        CurriculumStage.MULTI_TURN: {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "epsilon_decay": 0.995
        },
        CurriculumStage.FULL_GAME: {
            "learning_rate": 5e-5,  # Lower LR for fine-tuning
            "batch_size": 128,  # Larger batches for stability
            "epsilon_decay": 0.998  # Faster decay for exploitation
        }
    }

    params = base_params.copy()
    if stage in stage_adjustments:
        params.update(stage_adjustments[stage])

    return params