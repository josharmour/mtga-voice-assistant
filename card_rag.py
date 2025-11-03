#!/usr/bin/env python3
"""
Card RAG (Retrieval-Augmented Generation) System

Provides grounded, cited card information to the LLM without hallucinations.

This module bridges the unified card database with the RAG system to provide:
1. Card abilities (oracle text) with citations
2. Card costs and mana requirements
3. Card types and creature stats
4. Win rate context from 17lands
5. All information grounded in actual card data

The LLM can reference specific cards with confidence, knowing the data is
sourced from Arena's database and 17lands statistics.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CardInfo:
    """Complete card information for LLM context."""
    grpId: int
    name: str
    oracle_text: str
    mana_cost: str
    cmc: float
    type_line: str
    color_identity: str
    power: str
    toughness: str
    rarity: str
    set_code: str
    win_rate: Optional[float] = None
    gih_win_rate: Optional[float] = None
    avg_pick_position: Optional[float] = None
    games_played: Optional[int] = None

    def to_rag_citation(self, include_stats: bool = True) -> str:
        """
        Format card information as a grounded RAG citation.

        This provides the LLM with factual, verifiable information with sources.

        Args:
            include_stats: Include win rate statistics

        Returns:
            Formatted card information with citations
        """
        lines = []

        # Card name with ID (for verification)
        lines.append(f"**{self.name}** (grpId: {self.grpId}) [{self.set_code}]")

        # Mana cost and type
        lines.append(f"Cost: {self.mana_cost} | Type: {self.type_line}")

        # Power/toughness if creature
        if self.power or self.toughness:
            lines.append(f"Stats: {self.power}/{self.toughness}")

        # Color identity
        if self.color_identity:
            lines.append(f"Colors: {self.color_identity}")

        # Oracle text (card abilities) - with citation
        if self.oracle_text:
            lines.append(f"Abilities: {self.oracle_text}")
            lines.append(f"*[Source: Arena Card Database]*")

        # Win rate statistics - with citation
        if include_stats and self.win_rate is not None:
            lines.append("")
            lines.append("**Performance Statistics:**")
            if self.games_played and self.games_played >= 1000:
                lines.append(f"- Win Rate: {self.win_rate*100:.1f}% ({self.games_played} games)")
                lines.append(f"- GIH Win Rate: {self.gih_win_rate*100:.1f}%")
                if self.avg_pick_position:
                    lines.append(f"- Average Pick Position: {self.avg_pick_position:.1f}")
                lines.append(f"*[Source: 17lands.com]*")
            else:
                lines.append(f"- Limited data available ({self.games_played} games)")

        lines.append("")  # Empty line for separation
        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """
        Format card for inclusion in LLM prompt.

        Concise format for use in board state context.

        Returns:
            Formatted card context
        """
        parts = []

        # Name and cost
        if self.mana_cost:
            parts.append(f"{self.name} ({self.mana_cost})")
        else:
            parts.append(self.name)

        # Type
        if self.type_line:
            parts.append(f"[{self.type_line}]")

        # Power/toughness
        if self.power or self.toughness:
            parts.append(f"{self.power}/{self.toughness}")

        # Abilities summary
        if self.oracle_text:
            # Truncate long abilities
            abilities = self.oracle_text[:100]
            if len(self.oracle_text) > 100:
                abilities += "..."
            parts.append(f"Abilities: {abilities}")

        return " ".join(parts)


class CardRagDatabase:
    """
    Unified card database with RAG capabilities.

    Provides grounded card information to the LLM, combining:
    - Card metadata from Arena (names, abilities, costs)
    - Statistics from 17lands (win rates, pick positions)
    - All information with citations to prevent hallucinations
    """

    def __init__(
        self,
        unified_db: str = "data/unified_cards.db",
        stats_db: str = "data/card_stats.db"
    ):
        """
        Initialize the Card RAG database.

        Args:
            unified_db: Path to unified_cards.db (Arena data)
            stats_db: Path to card_stats.db (17lands data)
        """
        self.unified_db_path = Path(unified_db)
        self.stats_db_path = Path(stats_db)
        self.unified_conn = None
        self.stats_conn = None

        self._initialize()

    def _initialize(self):
        """Initialize database connections."""
        try:
            if self.unified_db_path.exists():
                self.unified_conn = sqlite3.connect(str(self.unified_db_path))
                self.unified_conn.row_factory = sqlite3.Row
                logger.info(f"Connected to {self.unified_db_path}")
            else:
                logger.warning(f"Unified database not found: {self.unified_db_path}")

            if self.stats_db_path.exists():
                self.stats_conn = sqlite3.connect(str(self.stats_db_path))
                self.stats_conn.row_factory = sqlite3.Row
                logger.info(f"Connected to {self.stats_db_path}")
            else:
                logger.debug(f"Stats database not found: {self.stats_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")

    def get_card_by_grpid(self, grp_id: int, format_type: str = "PremierDraft") -> Optional[CardInfo]:
        """
        Get complete card information by grpId with statistics.

        Args:
            grp_id: Arena graphics ID
            format_type: Draft format for statistics (e.g., "PremierDraft")

        Returns:
            CardInfo object with all data or None
        """
        if not self.unified_conn:
            logger.warning("Unified database not available")
            return None

        try:
            cursor = self.unified_conn.cursor()
            cursor.execute("""
                SELECT
                    grpId, name, oracle_text, mana_cost, cmc, type_line,
                    color_identity, power, toughness, rarity, set_code
                FROM cards
                WHERE grpId = ?
            """, (grp_id,))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Card not found: grpId {grp_id}")
                return None

            card_info = CardInfo(
                grpId=row['grpId'],
                name=row['name'],
                oracle_text=row['oracle_text'] or "",
                mana_cost=row['mana_cost'] or "",
                cmc=row['cmc'] or 0.0,
                type_line=row['type_line'] or "",
                color_identity=row['color_identity'] or "",
                power=row['power'] or "",
                toughness=row['toughness'] or "",
                rarity=row['rarity'] or "",
                set_code=row['set_code'] or ""
            )

            # Get statistics if available
            if self.stats_conn:
                card_info = self._add_statistics(card_info, format_type)

            return card_info

        except Exception as e:
            logger.error(f"Error fetching card {grp_id}: {e}")
            return None

    def get_card_by_name(self, card_name: str, set_code: str = None) -> Optional[CardInfo]:
        """
        Get card information by name and optional set.

        Args:
            card_name: Card name to search for
            set_code: Optional set code to narrow search

        Returns:
            CardInfo object or None
        """
        if not self.unified_conn:
            return None

        try:
            cursor = self.unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE name = ? AND set_code = ?
                    LIMIT 1
                """, (card_name, set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE name = ?
                    LIMIT 1
                """, (card_name,))

            row = cursor.fetchone()
            if not row:
                return None

            card_info = CardInfo(
                grpId=row['grpId'],
                name=row['name'],
                oracle_text=row['oracle_text'] or "",
                mana_cost=row['mana_cost'] or "",
                cmc=row['cmc'] or 0.0,
                type_line=row['type_line'] or "",
                color_identity=row['color_identity'] or "",
                power=row['power'] or "",
                toughness=row['toughness'] or "",
                rarity=row['rarity'] or "",
                set_code=row['set_code'] or ""
            )

            if self.stats_conn:
                card_info = self._add_statistics(card_info)

            return card_info

        except Exception as e:
            logger.error(f"Error fetching card {card_name}: {e}")
            return None

    def _add_statistics(self, card_info: CardInfo, format_type: str = "PremierDraft") -> CardInfo:
        """
        Add 17lands statistics to card information.

        Args:
            card_info: Base card information
            format_type: Draft format for stats

        Returns:
            CardInfo with statistics added
        """
        if not self.stats_conn:
            return card_info

        try:
            cursor = self.stats_conn.cursor()
            cursor.execute("""
                SELECT
                    win_rate, gih_win_rate, avg_taken_at, games_played
                FROM card_stats
                WHERE card_name = ? AND set_code = ? AND format = ?
                LIMIT 1
            """, (card_info.name, card_info.set_code, format_type))

            row = cursor.fetchone()
            if row:
                card_info.win_rate = row['win_rate']
                card_info.gih_win_rate = row['gih_win_rate']
                card_info.avg_pick_position = row['avg_taken_at']
                card_info.games_played = row['games_played']

        except Exception as e:
            logger.debug(f"Could not fetch stats for {card_info.name}: {e}")

        return card_info

    def get_board_state_context(
        self,
        card_grp_ids: List[int],
        include_stats: bool = True,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Generate complete board state context for LLM.

        Creates a formatted context of all cards with abilities, costs, and stats.

        Args:
            card_grp_ids: List of card grpIds on board
            include_stats: Include win rate statistics
            format_type: Draft format for statistics

        Returns:
            Formatted board state context with citations
        """
        if not self.unified_conn:
            return ""

        context_lines = []
        context_lines.append("=" * 70)
        context_lines.append("BOARD STATE WITH CARD INFORMATION")
        context_lines.append("=" * 70)
        context_lines.append("")

        for grp_id in card_grp_ids:
            card_info = self.get_card_by_grpid(grp_id, format_type)
            if card_info:
                context_lines.append(card_info.to_rag_citation(include_stats))

        context_lines.append("=" * 70)
        context_lines.append("[All information sourced from Arena database and 17lands.com]")
        context_lines.append("=" * 70)

        return "\n".join(context_lines)

    def search_cards_by_type(self, type_keyword: str, set_code: str = None) -> List[CardInfo]:
        """
        Search for cards by type (e.g., "creature", "instant").

        Useful for finding relevant cards when building prompts.

        Args:
            type_keyword: Type to search for (case-insensitive)
            set_code: Optional set code to limit search

        Returns:
            List of matching cards
        """
        if not self.unified_conn:
            return []

        try:
            cursor = self.unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE type_line LIKE ? AND set_code = ?
                    LIMIT 20
                """, (f"%{type_keyword}%", set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE type_line LIKE ?
                    LIMIT 20
                """, (f"%{type_keyword}%",))

            results = []
            for row in cursor.fetchall():
                card_info = CardInfo(
                    grpId=row['grpId'],
                    name=row['name'],
                    oracle_text=row['oracle_text'] or "",
                    mana_cost=row['mana_cost'] or "",
                    cmc=row['cmc'] or 0.0,
                    type_line=row['type_line'] or "",
                    color_identity=row['color_identity'] or "",
                    power=row['power'] or "",
                    toughness=row['toughness'] or "",
                    rarity=row['rarity'] or "",
                    set_code=row['set_code'] or ""
                )
                results.append(card_info)

            return results

        except Exception as e:
            logger.error(f"Error searching for {type_keyword}: {e}")
            return []

    def search_by_ability(self, ability_keyword: str, set_code: str = None) -> List[CardInfo]:
        """
        Search for cards by ability text (e.g., "draw a card").

        Args:
            ability_keyword: Ability keyword to search for
            set_code: Optional set code

        Returns:
            List of matching cards
        """
        if not self.unified_conn:
            return []

        try:
            cursor = self.unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE oracle_text LIKE ? AND set_code = ?
                    LIMIT 20
                """, (f"%{ability_keyword}%", set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE oracle_text LIKE ?
                    LIMIT 20
                """, (f"%{ability_keyword}%",))

            results = []
            for row in cursor.fetchall():
                card_info = CardInfo(
                    grpId=row['grpId'],
                    name=row['name'],
                    oracle_text=row['oracle_text'] or "",
                    mana_cost=row['mana_cost'] or "",
                    cmc=row['cmc'] or 0.0,
                    type_line=row['type_line'] or "",
                    color_identity=row['color_identity'] or "",
                    power=row['power'] or "",
                    toughness=row['toughness'] or "",
                    rarity=row['rarity'] or "",
                    set_code=row['set_code'] or ""
                )
                results.append(card_info)

            return results

        except Exception as e:
            logger.error(f"Error searching by ability: {e}")
            return []

    def close(self):
        """Close database connections."""
        if self.unified_conn:
            self.unified_conn.close()
        if self.stats_conn:
            self.stats_conn.close()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize
    card_rag = CardRagDatabase()

    # Get card by grpId
    card = card_rag.get_card_by_grpid(97934)  # Chizak, Apex Arachnosaur
    if card:
        print("Card Citation Format:")
        print(card.to_rag_citation())
        print("\nPrompt Context Format:")
        print(card.to_prompt_context())

    # Search by type
    creatures = card_rag.search_cards_by_type("creature", "OM1")
    print(f"\nFound {len(creatures)} creatures in OM1")

    card_rag.close()
