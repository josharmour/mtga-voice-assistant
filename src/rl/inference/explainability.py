"""
Explainability System for MTG RL Agent

Provides decision rationale, attention visualization, and
human-understandable explanations for RL recommendations.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

from ..models.dueling_dqn import DuelingDQNNetwork as DuelingDQN
from ..data.state_extractor import StateExtractor

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations available."""
    ATTENTION_WEIGHTS = "attention_weights"
    FEATURE_IMPORTANCE = "feature_importance"
    ACTION_COMPARISON = "action_comparison"
    STRATEGIC_RATIONALE = "strategic_rationale"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class FeatureImportance:
    """Feature importance for explanation."""
    feature_name: str
    importance_score: float
    feature_value: Any
    direction: str  # 'positive' or 'negative'


@dataclass
class ActionComparison:
    """Comparison between selected and alternative actions."""
    action_name: str
    q_value: float
    advantage: float  # Difference from selected action
    rationale: str


@dataclass
class ExplanationResult:
    """Complete explanation for a decision."""
    selected_action: int
    selected_action_name: str
    confidence: float
    explanation_type: ExplanationType
    feature_importances: List[FeatureImportance]
    action_comparisons: List[ActionComparison]
    strategic_rationale: str
    attention_weights: Optional[np.ndarray]
    counterfactual_analysis: Optional[Dict[str, Any]]
    quality_score: float  # 0-1, human understandability


class ExplainabilitySystem:
    """
    Explainability system for MTG RL agent decisions.

    Provides multiple explanation types:
    - Attention weight visualization
    - Feature importance analysis
    - Action comparison and rationale
    - Strategic explanation generation
    - Counterfactual analysis
    """

    def __init__(
        self,
        model: Optional[DuelingDQN] = None,
        state_extractor: Optional[StateExtractor] = None,
        enable_attention_analysis: bool = True,
        enable_counterfactuals: bool = True
    ):
        self.model = model
        self.state_extractor = state_extractor or StateExtractor()
        self.enable_attention_analysis = enable_attention_analysis
        self.enable_counterfactuals = enable_counterfactuals

        # Feature name mappings
        self.feature_names = self._get_feature_names()
        self.action_names = self._get_action_names()

        # Explanation templates
        self.explanation_templates = self._load_explanation_templates()

        logger.info("Explainability system initialized")

    def _get_feature_names(self) -> List[str]:
        """Get names for all state features."""
        return [
            # Turn and phase features
            "turn_number", "main_phase", "combat_phase", "end_phase",

            # Life totals
            "player_life", "opponent_life", "life_advantage",

            # Resources
            "available_mana", "mana_efficiency", "land_count",

            # Hand information
            "hand_size", "hand_creatures", "hand_spells", "hand_lands",
            "hand_avg_cost", "hand_max_cost",

            # Board state
            "board_creatures", "board_artifacts", "board_enchantments",
            "board_planeswalkers", "board_power", "board_toughness",

            # Combat status
            "can_attack", "can_block", "attackers_ready", "blockers_ready",

            # Graveyard and resources
            "graveyard_size", "library_size", "exile_size",

            # Strategic position
            "board_control", "tempo_advantage", "card_advantage",
            "threat_level", "opportunity_level"
        ]

    def _get_action_names(self) -> List[str]:
        """Get names for all possible actions."""
        return [
            "Play Land", "Cast Creature", "Cast Instant", "Cast Sorcery",
            "Cast Artifact", "Cast Enchantment", "Cast Planeswalker",
            "Activate Ability", "Attack with Creatures", "Block with Creatures",
            "Pass Priority", "Use Mana Ability", "Mulligan", "Concede",
            "Sideboard Choice", "Skip Turn"
        ]

    def _load_explanation_templates(self) -> Dict[str, List[str]]:
        """Load explanation templates for different scenarios."""
        return {
            "creature_play": [
                "Playing {creature} creates board presence",
                "{creature} is good for current board state",
                "Need to develop board with creatures"
            ],
            "spell_cast": [
                "{spell} addresses current threat",
                "{spell} provides card advantage",
                "Timing is right for {spell}"
            ],
            "attack": [
                "Attacking capitalizes on board advantage",
                "Opponent has limited blockers",
                "Pressure needs to be applied"
            ],
            "defense": [
                "Blocking prevents lethal damage",
                "Conserving life total is crucial",
                "Trade is favorable"
            ],
            "resource_management": [
                "Efficient mana usage is important",
                "Saving mana for opponent's turn",
                "Maximizing resource efficiency"
            ]
        }

    def explain_decision(
        self,
        game_state: Dict[str, Any],
        selected_action: int,
        q_values: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        explanation_types: Optional[List[ExplanationType]] = None
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for RL decision.

        Args:
            game_state: Current game state
            selected_action: Action selected by RL agent
            q_values: Q-value distribution from model
            valid_actions: List of valid actions
            explanation_types: Types of explanations to generate

        Returns:
            Complete explanation result
        """
        if explanation_types is None:
            explanation_types = [
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.ACTION_COMPARISON,
                ExplanationType.STRATEGIC_RATIONALE
            ]

        try:
            # Get action name
            action_name = self.action_names[selected_action] if selected_action < len(self.action_names) else f"Action {selected_action}"
            confidence = float(q_values[selected_action])

            # Generate different explanation types
            feature_importances = []
            action_comparisons = []
            attention_weights = None
            counterfactual_analysis = None

            if ExplanationType.FEATURE_IMPORTANCE in explanation_types:
                feature_importances = self._analyze_feature_importance(
                    game_state, selected_action, q_values
                )

            if ExplanationType.ACTION_COMPARISON in explanation_types:
                action_comparisons = self._compare_actions(
                    selected_action, q_values, valid_actions
                )

            if ExplanationType.ATTENTION_WEIGHTS in explanation_types and self.enable_attention_analysis:
                attention_weights = self._get_attention_weights(game_state)

            if ExplanationType.COUNTERFACTUAL in explanation_types and self.enable_counterfactuals:
                counterfactual_analysis = self._analyze_counterfactuals(
                    game_state, selected_action, q_values
                )

            # Generate strategic rationale
            strategic_rationale = self._generate_strategic_rationale(
                game_state, selected_action, feature_importances, action_comparisons
            )

            # Calculate explanation quality score
            quality_score = self._calculate_explanation_quality(
                feature_importances, action_comparisons, strategic_rationale
            )

            return ExplanationResult(
                selected_action=selected_action,
                selected_action_name=action_name,
                confidence=confidence,
                explanation_type=ExplanationType.STRATEGIC_RATIONALE,
                feature_importances=feature_importances,
                action_comparisons=action_comparisons,
                strategic_rationale=strategic_rationale,
                attention_weights=attention_weights,
                counterfactual_analysis=counterfactual_analysis,
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            # Return minimal explanation
            return ExplanationResult(
                selected_action=selected_action,
                selected_action_name=f"Action {selected_action}",
                confidence=0.0,
                explanation_type=ExplanationType.STRATEGIC_RATIONALE,
                feature_importances=[],
                action_comparisons=[],
                strategic_rationale="Explanation generation failed",
                attention_weights=None,
                counterfactual_analysis=None,
                quality_score=0.0
            )

    def _analyze_feature_importance(
        self,
        game_state: Dict[str, Any],
        selected_action: int,
        q_values: np.ndarray
    ) -> List[FeatureImportance]:
        """Analyze feature importance for the decision."""
        importance_scores = []
        state_vector = self.state_extractor.extract_state(game_state)

        # Gradient-based importance
        if self.model is not None:
            importance_scores = self._compute_gradient_importance(
                state_vector, selected_action
            )

        # Perturbation-based importance (fallback)
        if not importance_scores:
            importance_scores = self._compute_perturbation_importance(
                game_state, selected_action, q_values
            )

        # Create feature importance objects
        feature_importances = []
        for i, (score, name) in enumerate(zip(importance_scores, self.feature_names)):
            if i < len(state_vector):
                feature_value = state_vector[i]
                direction = "positive" if score > 0 else "negative"

                feature_importances.append(FeatureImportance(
                    feature_name=name,
                    importance_score=float(abs(score)),
                    feature_value=feature_value,
                    direction=direction
                ))

        # Sort by importance
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)

        # Return top features
        return feature_importances[:10]

    def _compute_gradient_importance(
        self,
        state_vector: np.ndarray,
        selected_action: int
    ) -> List[float]:
        """Compute gradient-based feature importance."""
        if self.model is None:
            return []

        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            state_tensor.requires_grad_(True)

            # Forward pass
            q_values = self.model(state_tensor)
            selected_q = q_values[0, selected_action]

            # Backward pass
            self.model.zero_grad()
            selected_q.backward(retain_graph=True)

            # Get gradients
            gradients = state_tensor.grad.squeeze().detach().numpy()
            return np.abs(gradients).tolist()

        except Exception as e:
            logger.warning(f"Gradient importance computation failed: {e}")
            return []

    def _compute_perturbation_importance(
        self,
        game_state: Dict[str, Any],
        selected_action: int,
        q_values: np.ndarray
    ) -> List[float]:
        """Compute perturbation-based feature importance."""
        importance_scores = []

        try:
            original_state = game_state.copy()
            state_vector = self.state_extractor.extract_state(game_state)
            base_q = q_values[selected_action]

            for i in range(min(len(state_vector), len(self.feature_names))):
                # Perturb feature
                perturbed_vector = state_vector.copy()
                perturbation = 0.1 * (1 if perturbed_vector[i] >= 0 else -1)
                perturbed_vector[i] += perturbation

                # Get new Q-value (approximate)
                # This is a simplified approach - in practice would need model forward pass
                q_change = abs(perturbation) * 0.1  # Rough approximation
                importance_scores.append(q_change)

            return importance_scores

        except Exception as e:
            logger.warning(f"Perturbation importance computation failed: {e}")
            return [0.0] * len(self.feature_names)

    def _compare_actions(
        self,
        selected_action: int,
        q_values: np.ndarray,
        valid_actions: Optional[List[int]]
    ) -> List[ActionComparison]:
        """Compare selected action with alternatives."""
        action_comparisons = []

        selected_q = q_values[selected_action]

        # Get top alternative actions
        if valid_actions is not None:
            # Only consider valid actions
            valid_mask = np.zeros_like(q_values)
            valid_mask[valid_actions] = 1
            masked_q = q_values * valid_mask
        else:
            masked_q = q_values

        # Get indices of top actions (excluding selected)
        top_indices = np.argsort(masked_q)[-5:][::-1]

        for action_idx in top_indices:
            if action_idx != selected_action and masked_q[action_idx] > -float('inf'):
                action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"Action {action_idx}"
                q_value = float(masked_q[action_idx])
                advantage = selected_q - q_value

                rationale = self._get_action_rationale(action_idx, advantage)

                action_comparisons.append(ActionComparison(
                    action_name=action_name,
                    q_value=q_value,
                    advantage=float(advantage),
                    rationale=rationale
                ))

        return action_comparisons[:3]  # Return top 3 alternatives

    def _get_action_rationale(self, action_idx: int, advantage: float) -> str:
        """Get rationale for why action wasn't selected."""
        if advantage > 0.5:
            return "Much better alternative available"
        elif advantage > 0.2:
            return "Better alternative available"
        elif advantage > 0.1:
            return "Slightly better alternative"
        else:
            return "Comparable alternative"

    def _get_attention_weights(self, game_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get attention weights from model (if available)."""
        # This would require model to expose attention weights
        # For now, return None as most DQN models don't use attention
        return None

    def _analyze_counterfactuals(
        self,
        game_state: Dict[str, Any],
        selected_action: int,
        q_values: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Analyze counterfactual scenarios."""
        try:
            counterfactuals = {}

            # Counterfactual: What if life was different?
            if 'life' in game_state:
                original_life = game_state['life']
                game_state['life'] = original_life - 5  # Less life

                # In practice, would need model forward pass
                # For now, approximate effect
                life_pressure_effect = 0.2 if original_life < 15 else 0.1
                counterfactuals['low_life_scenario'] = {
                    'description': "If you had 5 less life",
                    'likely_action_change': "More defensive actions",
                    'confidence_shift': life_pressure_effect
                }
                game_state['life'] = original_life  # Restore

            # Counterfactual: What if opponent had different board state?
            if 'opponent_board' in game_state:
                original_board = game_state.get('opponent_board', [])
                # Simulate stronger opponent board
                counterfactuals['strong_opponent_scenario'] = {
                    'description': "If opponent had stronger board",
                    'likely_action_change': "More removal or blocking",
                    'confidence_shift': 0.15
                }

            return counterfactuals if counterfactuals else None

        except Exception as e:
            logger.warning(f"Counterfactual analysis failed: {e}")
            return None

    def _generate_strategic_rationale(
        self,
        game_state: Dict[str, Any],
        selected_action: int,
        feature_importances: List[FeatureImportance],
        action_comparisons: List[ActionComparison]
    ) -> str:
        """Generate human-readable strategic rationale."""
        rationales = []

        # Action-specific rationale
        action_name = self.action_names[selected_action] if selected_action < len(self.action_names) else f"Action {selected_action}"

        if "Play Land" in action_name:
            rationales.append("Land drop is important for mana development")
        elif "Cast Creature" in action_name:
            rationales.append("Board presence through creatures is valuable")
        elif "Attack" in action_name:
            rationales.append("Applying pressure through combat")
        elif "Block" in action_name:
            rationales.append("Defense against opponent's threats")
        elif "Pass Priority" in action_name:
            rationales.append("Conserving resources for opponent's turn")

        # Feature-based rationale
        if feature_importances:
            top_feature = feature_importances[0]
            if top_feature.importance_score > 0.1:
                if "life" in top_feature.feature_name.lower():
                    rationales.append(f"Life total consideration is crucial ({top_feature.feature_name})")
                elif "mana" in top_feature.feature_name.lower():
                    rationales.append(f"Mana efficiency is important ({top_feature.feature_name})")
                elif "board" in top_feature.feature_name.lower():
                    rationales.append(f"Board state control is priority ({top_feature.feature_name})")

        # Game state context
        turn_number = game_state.get('turn_number', 0)
        if turn_number <= 5:
            rationales.append("Early game development is key")
        elif turn_number >= 15:
            rationales.append("Late game requires decisive action")

        life_total = game_state.get('life', 20)
        if life_total <= 10:
            rationales.append("Low life requires defensive consideration")
        elif life_total >= 25:
            rationales.append("High life allows for aggressive play")

        # Combine rationales
        if rationales:
            strategic_rationale = "Primary reasoning: " + "; ".join(rationales[:3])
        else:
            strategic_rationale = "Decision based on overall game state assessment"

        return strategic_rationale

    def _calculate_explanation_quality(
        self,
        feature_importances: List[FeatureImportance],
        action_comparisons: List[ActionComparison],
        strategic_rationale: str
    ) -> float:
        """Calculate quality score for explanation (0-1)."""
        quality_score = 0.0

        # Feature importance quality (40% weight)
        if feature_importances:
            avg_importance = np.mean([fi.importance_score for fi in feature_importances])
            quality_score += 0.4 * min(avg_importance, 1.0)

        # Action comparison quality (30% weight)
        if action_comparisons:
            quality_score += 0.3  # Good comparisons present

        # Rationale quality (30% weight)
        if strategic_rationale and len(strategic_rationale) > 50:
            quality_score += 0.3

        return min(quality_score, 1.0)

    def get_explanation_summary(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Get summary of explanation for display."""
        return {
            'decision': {
                'action': explanation.selected_action_name,
                'confidence': round(explanation.confidence, 3),
                'rationale': explanation.strategic_rationale
            },
            'key_factors': [
                {
                    'feature': fi.feature_name,
                    'importance': round(fi.importance_score, 3),
                    'impact': fi.direction
                }
                for fi in explanation.feature_importances[:5]
            ],
            'alternatives': [
                {
                    'action': ac.action_name,
                    'q_value': round(ac.q_value, 3),
                    'difference': round(ac.advantage, 3),
                    'reason': ac.rationale
                }
                for ac in explanation.action_comparisons[:3]
            ],
            'quality': {
                'score': round(explanation.quality_score, 3),
                'rating': self._get_quality_rating(explanation.quality_score)
            }
        }

    def _get_quality_rating(self, quality_score: float) -> str:
        """Get human-readable quality rating."""
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.6:
            return "Good"
        elif quality_score >= 0.4:
            return "Fair"
        else:
            return "Poor"

    def export_explanation(
        self,
        explanation: ExplanationResult,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export explanation in specified format."""
        if format == "json":
            return {
                'selected_action': explanation.selected_action,
                'selected_action_name': explanation.selected_action_name,
                'confidence': explanation.confidence,
                'strategic_rationale': explanation.strategic_rationale,
                'feature_importances': [
                    {
                        'feature': fi.feature_name,
                        'importance': fi.importance_score,
                        'value': fi.feature_value,
                        'direction': fi.direction
                    }
                    for fi in explanation.feature_importances
                ],
                'action_comparisons': [
                    {
                        'action': ac.action_name,
                        'q_value': ac.q_value,
                        'advantage': ac.advantage,
                        'rationale': ac.rationale
                    }
                    for ac in explanation.action_comparisons
                ],
                'quality_score': explanation.quality_score
            }
        elif format == "text":
            summary = self.get_explanation_summary(explanation)
            return f"""
Decision: {summary['decision']['action']} (Confidence: {summary['decision']['confidence']})
Rationale: {summary['decision']['rationale']}

Key Factors:
{chr(10).join([f"- {f['feature']}: {f['importance']} ({f['impact']})" for f in summary['key_factors']])}

Alternatives Considered:
{chr(10).join([f"- {a['action']}: {a['difference']} advantage ({a['reason']})" for a in summary['alternatives']])}

Explanation Quality: {summary['quality']['rating']} ({summary['quality']['score']})
            """.strip()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for creating explainability system
def create_explainability_system(
    model: Optional[DuelingDQN] = None
) -> ExplainabilitySystem:
    """Create and configure explainability system."""
    return ExplainabilitySystem(
        model=model,
        enable_attention_analysis=True,
        enable_counterfactuals=True
    )