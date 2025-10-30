#!/usr/bin/env python3
"""
Draft Advisor for MTGA Voice Advisor

Provides draft pick recommendations using 17lands data and optional LLM analysis.
Clean table-based output inspired by python-mtga-helper.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
from termcolor import colored

logger = logging.getLogger(__name__)


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text (e.g., <nobr>, <sprite>)"""
    if not text:
        return text
    # Remove all HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text.strip()


@dataclass
class DraftCard:
    """Represents a card in a draft pack with stats"""
    arena_id: int
    name: str
    colors: str = ""
    rarity: str = ""
    types: List[str] = None
    win_rate: float = 0.0
    gih_win_rate: float = 0.0  # Games In Hand win rate
    iwd: float = 0.0  # Improvement When Drawn
    ever_drawn_game_count: int = 0
    grade: str = ""
    score: float = 0.0  # Percentile score 0-100

    def __post_init__(self):
        if self.types is None:
            self.types = []


class DraftAdvisor:
    """Generates draft pick recommendations using 17lands data"""

    # Grade thresholds (percentile scores)
    GRADE_THRESHOLDS = {
        "A+": 99,
        "A": 95,
        "A-": 90,
        "B+": 85,
        "B": 76,
        "B-": 68,
        "C+": 57,
        "C": 45,
        "C-": 36,
        "D+": 27,
        "D": 17,
        "D-": 5,
        "F": 0,
    }

    def __init__(self, card_db, rag_system=None, ollama_client=None):
        """
        Initialize DraftAdvisor

        Args:
            card_db: ArenaCardDatabase for card name resolution
            rag_system: Optional RAGSystem for 17lands stats
            ollama_client: Optional OllamaClient for LLM recommendations
        """
        self.card_db = card_db
        self.rag = rag_system
        self.ollama = ollama_client
        self.picked_cards = []
        self.current_pack_num = 0
        self.current_pick_num = 0
        self.current_set = ""

        # Initialize metadata DB connection
        self.metadata_db = None
        if self.rag and hasattr(self.rag, 'card_metadata'):
            self.metadata_db = self.rag.card_metadata
        else:
            try:
                from rag_advisor import CardMetadataDB
                self.metadata_db = CardMetadataDB('data/card_metadata.db')
            except Exception as e:
                logger.warning(f"Could not initialize CardMetadataDB: {e}")

    def recommend_pick(
        self,
        pack_arena_ids: List[int],
        pack_num: int,
        pick_num: int,
        event_name: str = ""
    ) -> Tuple[List[DraftCard], str]:
        """
        Generate pick recommendation for a draft pack

        Args:
            pack_arena_ids: List of Arena card IDs in the pack
            pack_num: Pack number (1-indexed)
            pick_num: Pick number (1-indexed)
            event_name: Event name like "PremierDraft_BLB_20250815"

        Returns:
            Tuple of (sorted pack cards, recommendation text)
        """
        self.current_pack_num = pack_num
        self.current_pick_num = pick_num

        # Extract set code from event name
        if event_name and "_" in event_name:
            parts = event_name.split("_")
            if len(parts) >= 2:
                self.current_set = parts[1].upper()

        logger.info(f"Analyzing Pack {pack_num}, Pick {pick_num} ({len(pack_arena_ids)} cards)")

        # 1. Resolve card names and get basic info
        pack_cards = self._resolve_cards(pack_arena_ids)

        if not pack_cards:
            logger.warning("Could not resolve any cards in pack")
            return [], "No cards found in pack"

        # 2. Enrich with metadata (colors, rarity, types)
        pack_cards = self._enrich_with_metadata(pack_cards)

        # 3. Enrich with 17lands stats if available
        if self.rag:
            pack_cards = self._enrich_with_stats(pack_cards)

        # 4. Calculate grades
        pack_cards = self._calculate_grades(pack_cards)

        # 5. Sort by grade/score (best first)
        pack_cards.sort(key=lambda c: (c.score if c.score else 0), reverse=True)

        # 6. Generate recommendation
        recommendation = self._generate_recommendation(pack_cards, pack_num, pick_num)

        return pack_cards, recommendation

    def _resolve_cards(self, arena_ids: List[int]) -> List[DraftCard]:
        """Resolve Arena IDs to card names"""
        cards = []

        for arena_id in arena_ids:
            try:
                # Try to get card name from database
                card_name = self.card_db.get_card_name(arena_id)

                if card_name:
                    # Strip HTML tags from card name
                    clean_name = strip_html_tags(card_name)

                    card = DraftCard(
                        arena_id=arena_id,
                        name=clean_name
                    )
                    cards.append(card)
                    logger.debug(f"Resolved {arena_id} -> {clean_name}")
                else:
                    logger.debug(f"Could not resolve arena ID {arena_id}")

            except Exception as e:
                logger.error(f"Error resolving arena ID {arena_id}: {e}")

        return cards

    def _enrich_with_metadata(self, cards: List[DraftCard]) -> List[DraftCard]:
        """
        Enrich cards with metadata (colors, rarity, types).
        This runs independently of stats loading to ensure colors always display.
        """
        if not self.metadata_db:
            logger.warning("Card metadata database not available.")
            return cards

        for card in cards:
            try:
                metadata = self.metadata_db.get_card_metadata(card.name)
                if metadata:
                    card.colors = metadata.get("color_identity", "")
                    card.rarity = metadata.get("rarity", "")
                    card.types = metadata.get("types", "").split() if metadata.get("types") else []
                    logger.debug(f"{card.name}: colors={card.colors}, rarity={card.rarity}")
            except Exception as e:
                logger.error(f"Error loading metadata for {card.name}: {e}")

        return cards

    def _enrich_with_stats(self, cards: List[DraftCard]) -> List[DraftCard]:
        """Enrich cards with 17lands statistics"""

        if not hasattr(self.rag, 'card_stats'):
            logger.warning("RAG system doesn't have card_stats")
            return cards

        for card in cards:
            try:
                stats = self.rag.card_stats.get_card_stats(card.name)

                if stats:
                    card.win_rate = stats.get("win_rate", 0.0) or 0.0
                    card.gih_win_rate = stats.get("gih_win_rate", 0.0) or 0.0
                    card.iwd = stats.get("iwd", 0.0) or 0.0
                    card.ever_drawn_game_count = stats.get("ever_drawn_game_count", 0) or 0

                    logger.debug(f"{card.name}: GIH WR={card.gih_win_rate:.3f}, IWD={card.iwd:.3f}")

            except Exception as e:
                logger.error(f"Error loading stats for {card.name}: {e}")

        return cards

    def _calculate_grades(self, cards: List[DraftCard]) -> List[DraftCard]:
        """
        Calculate percentile-based grades for cards

        Uses scipy normal distribution CDF to convert win rates to percentile scores,
        then maps to letter grades (A+ through F)
        """
        # Get all GIH win rates for normalization
        gih_win_rates = [c.gih_win_rate for c in cards if c.gih_win_rate > 0]

        if not gih_win_rates or len(gih_win_rates) < 3:
            # Not enough data for statistical grading, use simple thresholds
            for card in cards:
                if card.gih_win_rate > 0:
                    card.score = self._simple_score(card.gih_win_rate)
                    card.grade = self._score_to_grade(card.score)
            return cards

        # Calculate mean and std dev
        mean = float(np.mean(gih_win_rates))
        std = float(np.std(gih_win_rates, ddof=1))

        logger.debug(f"Grade calculation: mean={mean:.3f}, std={std:.3f}, n={len(gih_win_rates)}")

        # Calculate percentile scores using CDF
        for card in cards:
            if card.gih_win_rate > 0 and std > 0:
                # Calculate percentile using cumulative distribution function
                cdf = norm.cdf(card.gih_win_rate, loc=mean, scale=std)
                card.score = cdf * 100  # Convert to 0-100 scale
                card.grade = self._score_to_grade(card.score)

                logger.debug(f"{card.name}: GIH={card.gih_win_rate:.3f} -> score={card.score:.1f} -> {card.grade}")

        return cards

    def _simple_score(self, win_rate: float) -> float:
        """Simple fixed-threshold scoring when not enough data for CDF"""
        if win_rate >= 0.60:
            return 99
        elif win_rate >= 0.58:
            return 95
        elif win_rate >= 0.56:
            return 90
        elif win_rate >= 0.54:
            return 85
        elif win_rate >= 0.52:
            return 76
        elif win_rate >= 0.50:
            return 68
        elif win_rate >= 0.48:
            return 57
        else:
            return 45

    def _score_to_grade(self, score: float) -> str:
        """Convert percentile score to letter grade"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def _generate_recommendation(
        self,
        cards: List[DraftCard],
        pack_num: int,
        pick_num: int
    ) -> str:
        """Generate pick recommendation text"""

        if not cards:
            return "No cards available"

        # Get top card
        top_card = cards[0]

        if not top_card.grade:
            return f"Pick {top_card.name}"

        # Build recommendation
        rec = f"Pick {top_card.name}"

        if top_card.grade:
            rec += f" ({top_card.grade})"

        if top_card.gih_win_rate > 0:
            rec += f" - {top_card.gih_win_rate*100:.1f}% GIH WR"

        # Add context for early picks
        if pick_num <= 3 and len(cards) >= 2:
            second_card = cards[1]
            if second_card.grade and second_card.grade.startswith("A"):
                rec += f" | {second_card.name} ({second_card.grade}) also strong"

        return rec

    def record_pick(self, card_name: str):
        """Record a picked card"""
        self.picked_cards.append(card_name)
        logger.info(f"Recorded pick: {card_name} (total: {len(self.picked_cards)})")


def display_draft_pack(
    cards: List[DraftCard],
    pack_num: int,
    pick_num: int,
    recommendation: str,
    show_count: int = 15
):
    """
    Display draft pack using clean table format (for terminal/CLI mode)

    Args:
        cards: List of DraftCard objects (sorted by grade)
        pack_num: Pack number
        pick_num: Pick number
        recommendation: Recommendation text
        show_count: Max number of cards to display
    """
    print("\n" + "="*80)
    print(f"Pack {pack_num}, Pick {pick_num}")
    print("="*80 + "\n")

    if not cards:
        print("No cards in pack")
        return

    # Build table
    table = []
    for i, card in enumerate(cards[:show_count], 1):
        # Format color emoji
        color_emoji = format_color_emoji(card.colors)

        # Format rarity emoji
        rarity_emoji = format_rarity_emoji(card.rarity)

        # Color-coded grade
        grade_str = format_grade(card.grade) if card.grade else ""

        # Win rate
        win_rate_str = f"{card.gih_win_rate*100:.1f}%" if card.gih_win_rate > 0 else ""

        # Types
        types_str = " ".join(card.types[:2]) if card.types else ""  # Limit to first 2 types

        table.append([
            i,
            color_emoji,
            rarity_emoji,
            card.name,
            grade_str,
            win_rate_str,
            types_str
        ])

    print(tabulate(
        table,
        headers=["#", "", "", "Card", "Grade", "GIH WR", "Type"],
        tablefmt="simple",
        colalign=("right", "center", "center", "left", "center", "right", "left")
    ))

    print(f"\n💡 Recommendation: {recommendation}\n")


def format_draft_pack_for_gui(
    cards: List[DraftCard],
    pack_num: int,
    pick_num: int,
    recommendation: str,
    show_count: int = 15,
    picked_cards: List[str] = None,
    card_metadata_db = None,
    split_panes: bool = False
) -> tuple:
    """
    Format draft pack as list of strings for GUI display

    Args:
        cards: List of DraftCard objects (sorted by grade)
        pack_num: Pack number
        pick_num: Pick number
        recommendation: Recommendation text
        show_count: Max number of cards to display
        picked_cards: List of card names already picked (optional)
        card_metadata_db: CardMetadataDB for looking up colors of picked cards (optional)
        split_panes: If True, return (pack_lines, picked_lines) tuple; if False, return combined lines

    Returns:
        If split_panes=True: Tuple of (pack_lines, picked_lines)
        If split_panes=False: List of formatted strings ready for GUI display
    """
    lines = []
    lines.append("="*80)
    lines.append(f"DRAFT: Pack {pack_num}, Pick {pick_num}")
    lines.append("="*80)
    lines.append("")

    if not cards:
        lines.append("No cards in pack")
        return lines

    # Build card list
    for i, card in enumerate(cards[:show_count], 1):
        # Format color (use letter codes for GUI)
        color_str = format_color_emoji(card.colors, for_gui=True)

        # Format rarity (use letter for GUI: M/R/U/C)
        rarity_str = format_rarity_emoji(card.rarity, for_gui=True)

        # Grade (no color codes in GUI)
        grade_str = card.grade if card.grade else ""

        # Win rate
        win_rate_str = f"{card.gih_win_rate*100:.1f}%" if card.gih_win_rate > 0 else ""

        # Types
        types_str = " ".join(card.types[:2]) if card.types else ""

        # Format line
        line = f"{i:2}. {color_str} {rarity_str} {card.name:30} {grade_str:3} {win_rate_str:6} {types_str}"
        lines.append(line)

    lines.append("")
    lines.append(f"💡 RECOMMENDATION: {recommendation}")
    lines.append("")

    # Format picked cards separately if requested
    picked_lines = []
    if picked_cards and len(picked_cards) > 0:
        picked_lines.append("="*80)
        picked_lines.append(f"PICKED CARDS ({len(picked_cards)} total)")
        picked_lines.append("="*80)
        picked_lines.append("")

        # Group by color
        color_groups = {
            'W': [], 'U': [], 'B': [], 'R': [], 'G': [],
            'Multi': [], 'Colorless': []
        }

        for card_name in picked_cards:
            # Look up card color from metadata
            card_colors = ""
            if card_metadata_db:
                try:
                    metadata = card_metadata_db.get_card_metadata(card_name)
                    if metadata:
                        card_colors = metadata.get("color_identity", "")
                except Exception:
                    pass

            # Categorize by color
            if not card_colors:
                color_groups['Colorless'].append(card_name)
            elif len(card_colors) > 1:
                color_groups['Multi'].append(card_name)
            elif card_colors in ['W', 'U', 'B', 'R', 'G']:
                color_groups[card_colors].append(card_name)
            else:
                color_groups['Colorless'].append(card_name)

        # Display by color
        for color in ['W', 'U', 'B', 'R', 'G', 'Multi', 'Colorless']:
            if color_groups[color]:
                color_name = {
                    'W': 'White', 'U': 'Blue', 'B': 'Black',
                    'R': 'Red', 'G': 'Green', 'Multi': 'Multicolor',
                    'Colorless': 'Colorless'
                }[color]
                picked_lines.append(f"{color_name} ({len(color_groups[color])}):")
                for card in color_groups[color]:
                    picked_lines.append(f"  • {card}")
                picked_lines.append("")

    # Return based on split_panes setting
    if split_panes:
        return (lines, picked_lines)
    else:
        # Legacy mode: combine both into one list
        if picked_lines:
            lines.extend(picked_lines)
        return lines


def format_color_emoji(colors: str, for_gui: bool = False) -> str:
    """
    Convert color string to emoji or letter codes

    Args:
        colors: Color identity string (e.g., "W", "UR", "GWB")
        for_gui: If True, use letter codes instead of emoji (for fonts that don't support emoji)
    """
    if not colors:
        return "C" if for_gui else "⚪"  # Colorless

    if for_gui:
        # Use letter codes for GUI (Consolas font doesn't render emoji)
        return colors[:2] if len(colors) <= 2 else colors[:2]  # Show up to 2 colors
    else:
        # Use emoji for terminal (supports color emoji)
        emoji_map = {
            "W": "⚪",
            "U": "🔵",
            "B": "⚫",
            "R": "🔴",
            "G": "🟢"
        }

        # Handle multicolor
        if len(colors) > 1:
            return "".join(emoji_map.get(c, "") for c in colors[:2])

        return emoji_map.get(colors, "")


def format_rarity_emoji(rarity: str, for_gui: bool = False) -> str:
    """
    Convert rarity to emoji or letter codes

    Args:
        rarity: Rarity string (e.g., "mythic", "rare", "uncommon", "common")
        for_gui: If True, use letter codes instead of emoji (for fonts that don't support emoji)
    """
    if for_gui:
        # Use letter codes for GUI (Consolas font doesn't render emoji)
        rarity_map = {
            "mythic": "M",
            "rare": "R",
            "uncommon": "U",
            "common": "C"
        }
        return rarity_map.get(rarity.lower() if rarity else "", "")
    else:
        # Use emoji for terminal (supports emoji)
        emoji_map = {
            "mythic": "✨",
            "rare": "💎",
            "uncommon": "🔷",
            "common": "⬜"
        }
        return emoji_map.get(rarity.lower() if rarity else "", "")


def format_grade(grade: str) -> str:
    """Return color-coded grade string"""
    if not grade:
        return ""

    # Color mapping for grades
    if grade.startswith("A"):
        return colored(grade, "green", attrs=["bold"])
    elif grade.startswith("B"):
        return colored(grade, "cyan")
    elif grade.startswith("C"):
        return colored(grade, "yellow")
    elif grade.startswith("D"):
        return colored(grade, "red")
    else:  # F
        return colored(grade, "red", attrs=["bold"])
