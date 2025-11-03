#!/usr/bin/env python3
"""
Advanced Prompt Builder with Grounded RAG Context

Builds prompts for the LLM with:
1. Board state with complete card information (abilities, costs, stats)
2. Relevant MTG rules (semantic search)
3. Win rate context from 17lands
4. All information with citations to prevent hallucinations
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GroundedPromptBuilder:
    """
    Builds LLM prompts with grounded, cited context.

    Uses CardRAG for card data and RulesVectorDB for rules,
    ensuring the LLM has factual information with sources.
    """

    def __init__(self, card_rag=None, rules_db=None, verbose: bool = False):
        """
        Initialize prompt builder.

        Args:
            card_rag: CardRagDatabase instance
            rules_db: RulesVectorDB instance
            verbose: Include more detailed information
        """
        self.card_rag = card_rag
        self.rules_db = rules_db
        self.verbose = verbose

    def build_board_state_section(
        self,
        player_hand: List[int],
        player_board: List[int],
        opponent_hand_count: int,
        opponent_board: List[int],
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build a grounded board state section.

        Args:
            player_hand: List of card grpIds in player's hand
            player_board: List of card grpIds on player's board
            opponent_hand_count: Number of cards opponent has
            opponent_board: List of card grpIds on opponent's board
            format_type: Draft format for statistics

        Returns:
            Formatted board state with full card information
        """
        if not self.card_rag:
            return ""

        lines = []
        lines.append("## YOUR BOARD STATE\n")

        # Your hand
        if player_hand:
            lines.append("### Your Hand")
            for grp_id in player_hand:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    # Format: Name (Cost) [Type] - Abilities
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        # Your board
        if player_board:
            lines.append("### Your Board")
            for grp_id in player_board:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        # Opponent's board
        if opponent_board:
            lines.append("### Opponent's Board")
            for grp_id in opponent_board:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        lines.append(f"Opponent's hand: {opponent_hand_count} unknown cards\n")

        return "\n".join(lines)

    def build_card_abilities_section(
        self,
        card_grp_ids: List[int],
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build detailed card abilities section with citations.

        Args:
            card_grp_ids: List of card grpIds
            format_type: Draft format for statistics

        Returns:
            Formatted card details section
        """
        if not self.card_rag or not card_grp_ids:
            return ""

        lines = []
        lines.append("## CARD ABILITIES AND DETAILS\n")

        for grp_id in card_grp_ids:
            card = self.card_rag.get_card_by_grpid(grp_id, format_type)
            if card:
                lines.append(card.to_rag_citation(include_stats=self.verbose))

        return "\n".join(lines)

    def build_rules_section(self, situation_query: str, top_k: int = 3) -> str:
        """
        Build relevant rules section using semantic search.

        Args:
            situation_query: Description of game situation
            top_k: Number of rules to retrieve

        Returns:
            Formatted rules section with citations
        """
        if not self.rules_db:
            return ""

        try:
            results = self.rules_db.query(situation_query, top_k=top_k)
            if not results:
                return ""

            lines = []
            lines.append("## RELEVANT MTG RULES\n")

            for rule in results:
                rule_id = rule.get('id', '')
                section = rule.get('section', '')
                text = rule.get('text', '')
                lines.append(f"**Rule {rule_id}** ({section}):")
                lines.append(f"{text}\n")

            lines.append("[Source: MTG Comprehensive Rules]\n")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Could not retrieve rules: {e}")
            return ""

    def build_win_rate_context(
        self,
        card_names: List[str],
        set_code: str,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build win rate context for cards.

        Args:
            card_names: List of card names
            set_code: Set code
            format_type: Draft format

        Returns:
            Formatted win rate context
        """
        if not self.card_rag:
            return ""

        lines = []
        high_winrate = []
        low_winrate = []

        for card_name in card_names:
            card = self.card_rag.get_card_by_name(card_name, set_code)
            if card and card.win_rate:
                if card.win_rate > 0.55:
                    high_winrate.append((card_name, card.win_rate, card.games_played))
                elif card.win_rate < 0.48:
                    low_winrate.append((card_name, card.win_rate, card.games_played))

        if high_winrate or low_winrate:
            lines.append("## PERFORMANCE CONTEXT\n")

            if high_winrate:
                lines.append("### Strong Cards (High Win Rate)")
                for name, wr, games in high_winrate:
                    if games and games >= 1000:
                        lines.append(f"- **{name}**: {wr*100:.1f}% WR ({games} games)")
                lines.append("")

            if low_winrate:
                lines.append("### Weaker Cards (Low Win Rate)")
                for name, wr, games in low_winrate:
                    if games and games >= 1000:
                        lines.append(f"- **{name}**: {wr*100:.1f}% WR ({games} games)")
                lines.append("")

            lines.append("[Source: 17lands.com]\n")

        return "\n".join(lines)

    def build_complete_prompt(
        self,
        base_objective: str,
        player_hand: List[int],
        player_board: List[int],
        opponent_hand_count: int,
        opponent_board: List[int],
        situation_query: str = None,
        card_names_for_stats: List[str] = None,
        set_code: str = None,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build a complete grounded prompt with all RAG context.

        Args:
            base_objective: What the player needs to do (e.g., "What should I attack with?")
            player_hand: Player's hand grpIds
            player_board: Player's board creatures grpIds
            opponent_hand_count: Number of cards opponent has
            opponent_board: Opponent's board creatures grpIds
            situation_query: Query for rules search (e.g., "combat damage timing")
            card_names_for_stats: Card names to include performance stats for
            set_code: Set being played
            format_type: Draft format

        Returns:
            Complete prompt with grounded context and citations
        """
        prompt_sections = []

        # Expert persona
        persona = (
            "You are an expert Magic: The Gathering strategist. "
            "Your advice must be grounded in the card abilities, rules, and statistics provided. "
            "Only reference information that is cited. Explain your reasoning step-by-step."
        )
        prompt_sections.append(persona)
        prompt_sections.append("")

        # Objective
        prompt_sections.append(f"## OBJECTIVE: {base_objective}\n")

        # Board state with complete card information
        board_section = self.build_board_state_section(
            player_hand,
            player_board,
            opponent_hand_count,
            opponent_board,
            format_type
        )
        if board_section:
            prompt_sections.append(board_section)

        # Card abilities and details
        all_grp_ids = player_hand + player_board + opponent_board
        if all_grp_ids:
            abilities_section = self.build_card_abilities_section(all_grp_ids, format_type)
            if abilities_section:
                prompt_sections.append(abilities_section)

        # Relevant rules
        if situation_query:
            rules_section = self.build_rules_section(situation_query, top_k=2)
            if rules_section:
                prompt_sections.append(rules_section)

        # Win rate context
        if card_names_for_stats and set_code:
            stats_section = self.build_win_rate_context(
                card_names_for_stats,
                set_code,
                format_type
            )
            if stats_section:
                prompt_sections.append(stats_section)

        # Citation footer
        prompt_sections.append("\n" + "="*70)
        prompt_sections.append("IMPORTANT: Base your advice only on the information above.")
        prompt_sections.append("Do not reference cards, rules, or statistics not listed.")
        prompt_sections.append("="*70)

        return "\n\n".join(prompt_sections)


class RagContextCache:
    """
    Caches RAG context to avoid redundant lookups.

    Since board state doesn't change every turn, caching
    card information reduces database queries.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache."""
        self.cache: Dict[int, Dict] = {}  # grpId → card info
        self.max_size = max_size

    def get(self, grp_id: int) -> Optional[Dict]:
        """Get cached card info."""
        return self.cache.get(grp_id)

    def put(self, grp_id: int, card_info: Dict):
        """Cache card info."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest (not implemented for simplicity)
            pass
        self.cache[grp_id] = card_info

    def clear(self):
        """Clear the cache."""
        self.cache.clear()


# Integration with existing RAGSystem
def enhance_existing_rag_system(rag_system, card_rag=None):
    """
    Enhance existing RAGSystem with grounded card context.

    This bridges the existing rag_advisor.py with the new card_rag.py
    to provide grounded, cited information.

    Args:
        rag_system: Existing RAGSystem instance
        card_rag: CardRagDatabase instance

    Returns:
        Enhanced enhance_prompt method
    """
    original_enhance = rag_system.enhance_prompt

    def enhanced_enhance_prompt(board_state: Dict, base_prompt: str) -> str:
        """Enhanced version with grounded card context."""

        # Get original enhancement
        enhanced = original_enhance(board_state, base_prompt)

        # Add grounded card information if available
        if card_rag:
            try:
                # Extract card grpIds from board state
                player_hand_grpids = board_state.get('player_hand_grpids', [])

                if player_hand_grpids:
                    # Add card abilities section
                    card_section = "\n\n## YOUR HAND DETAILS\n"
                    for grp_id in player_hand_grpids[:5]:  # Limit to 5 cards
                        card = card_rag.get_card_by_grpid(grp_id)
                        if card:
                            card_section += f"\n**{card.name}** ({card.mana_cost})\n"
                            card_section += f"Type: {card.type_line}\n"
                            if card.oracle_text:
                                card_section += f"Abilities: {card.oracle_text}\n"
                            if card.win_rate:
                                card_section += f"[Win Rate: {card.win_rate*100:.1f}%]\n"

                    enhanced += card_section

            except Exception as e:
                logger.debug(f"Could not enhance with card context: {e}")

        return enhanced

    return enhanced_enhance_prompt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Build a prompt without RAG (for testing structure)
    builder = GroundedPromptBuilder(verbose=True)

    prompt = builder.build_complete_prompt(
        base_objective="What should I attack with?",
        player_hand=[97934, 97856],  # Example grpIds
        player_board=[97886],
        opponent_hand_count=3,
        opponent_board=[71234],
        situation_query="combat damage and blocking",
        set_code="OM1",
        format_type="PremierDraft"
    )

    print("Generated Prompt Structure:")
    print(prompt)
    print("\nNote: This example shows the structure. With card_rag and rules_db,")
    print("actual card abilities, costs, and rules would be populated.")
