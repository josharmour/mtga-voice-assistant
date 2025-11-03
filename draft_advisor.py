#!/usr/bin/env python3
"""
Draft Advisor for MTGA Voice Advisor

Provides draft pick recommendations using 17lands data and optional LLM analysis.
Clean table-based output inspired by python-mtga-helper.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
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
    ata: float = 0.0  # Average Taken At
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
        "A+": 99, "A": 95, "A-": 90, "B+": 85, "B": 76, "B-": 68,
        "C+": 57, "C": 45, "C-": 36, "D+": 27, "D": 17, "D-": 5, "F": 0,
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

        if event_name and "_" in event_name:
            self.current_set = event_name.split("_")[1].upper()

        logger.info(f"Analyzing Pack {pack_num}, Pick {pick_num} ({len(pack_arena_ids)} cards)")

        pack_cards = self._resolve_cards(pack_arena_ids)
        if not pack_cards:
            logger.warning("Could not resolve any cards in pack")
            return [], "No cards found in pack"

        self._enrich_pack(pack_cards)
        pack_cards = self._calculate_grades(pack_cards)
        pack_cards.sort(key=lambda c: c.score or 0, reverse=True)
        recommendation = self._generate_recommendation(pack_cards, pack_num, pick_num)

        return pack_cards, recommendation

    def _resolve_cards(self, arena_ids: List[int]) -> List[DraftCard]:
        """Resolve Arena IDs to card names"""
        cards = []
        for arena_id in arena_ids:
            try:
                card_name = self.card_db.get_card_name(arena_id)
                if card_name:
                    cards.append(DraftCard(arena_id=arena_id, name=strip_html_tags(card_name)))
            except Exception as e:
                logger.error(f"Error resolving arena ID {arena_id}: {e}")
        return cards

    def _enrich_pack(self, cards: List[DraftCard]):
        """Enrich all cards in a pack with metadata and stats concurrently."""
        with ThreadPoolExecutor() as executor:
            # Submit all enrichment tasks for all cards
            futures = [executor.submit(self._enrich_card, card) for card in cards]
            for future in futures:
                future.result()  # Wait for all tasks to complete

    def _enrich_card(self, card: DraftCard):
        """Fetches and sets metadata and stats for a single card."""
        self._enrich_card_with_metadata(card)
        if self.rag:
            self._enrich_card_with_stats(card)
        return card

    def _enrich_card_with_metadata(self, card: DraftCard):
        """Fetches and sets metadata for a single card."""
        if self.metadata_db:
            try:
                metadata = self.metadata_db.get_card_metadata(card.name)
                if metadata:
                    card.colors = metadata.get("color_identity", "")
                    card.rarity = metadata.get("rarity", "")
                    card.types = metadata.get("types", "").split() if metadata.get("types") else []
            except Exception as e:
                logger.error(f"Error loading metadata for {card.name}: {e}")

    def _enrich_card_with_stats(self, card: DraftCard):
        """Fetches and sets stats for a single card."""
        if hasattr(self.rag, 'card_stats'):
            try:
                stats = self.rag.card_stats.get_card_stats(card.name)
                if stats:
                    card.win_rate = stats.get("win_rate", 0.0) or 0.0
                    card.gih_win_rate = stats.get("gih_win_rate", 0.0) or 0.0
                    card.iwd = stats.get("iwd", 0.0) or 0.0
                    card.ata = stats.get("avg_taken_at", 0.0) or 0.0
                    card.ever_drawn_game_count = stats.get("ever_drawn_game_count", 0) or 0
            except Exception as e:
                logger.error(f"Error loading stats for {card.name}: {e}")

    def _calculate_grades(self, cards: List[DraftCard]) -> List[DraftCard]:
        """Calculate percentile-based grades for cards, incorporating ATA."""
        gih_win_rates = [c.gih_win_rate for c in cards if c.gih_win_rate > 0]
        atas = [c.ata for c in cards if c.ata > 0]

        if not gih_win_rates or len(gih_win_rates) < 3:
            for card in cards:
                if card.gih_win_rate > 0:
                    card.score = self._simple_score(card.gih_win_rate)
                    card.grade = self._score_to_grade(card.score)
            return cards

        gih_mean = np.mean(gih_win_rates)
        gih_std = np.std(gih_win_rates, ddof=1) if len(gih_win_rates) > 1 else 0
        ata_mean = np.mean(atas) if atas else 0
        ata_std = np.std(atas, ddof=1) if len(atas) > 1 else 0

        for card in cards:
            if card.gih_win_rate > 0 and gih_std > 0:
                gih_percentile = norm.cdf(card.gih_win_rate, loc=gih_mean, scale=gih_std)
                ata_percentile = 1 - norm.cdf(card.ata, loc=ata_mean, scale=ata_std) if card.ata > 0 and ata_std > 0 else 0.5
                card.score = ((gih_percentile * 0.7) + (ata_percentile * 0.3)) * 100
                card.grade = self._score_to_grade(card.score)
        return cards

    def _simple_score(self, win_rate: float) -> float:
        """Simple fixed-threshold scoring when not enough data for CDF."""
        if win_rate >= 0.60: return 99
        if win_rate >= 0.58: return 95
        if win_rate >= 0.56: return 90
        if win_rate >= 0.54: return 85
        if win_rate >= 0.52: return 76
        if win_rate >= 0.50: return 68
        if win_rate >= 0.48: return 57
        return 45

    def _score_to_grade(self, score: float) -> str:
        """Convert percentile score to letter grade."""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def _generate_recommendation(self, cards: List[DraftCard], pack_num: int, pick_num: int) -> str:
        """Generate pick recommendation text."""
        if not cards: return "No cards available"
        top_card = cards[0]
        rec = f"Pick {top_card.name}"
        if top_card.grade: rec += f" ({top_card.grade})"
        if top_card.ata > 0: rec += f" (ATA: {top_card.ata:.1f})"
        if pick_num <= 3 and len(cards) > 1 and cards[1].grade and cards[1].grade.startswith("A"):
            rec += f" | {cards[1].name} ({cards[1].grade}) also strong"
        return rec

    def record_pick(self, card_name: str):
        """Record a picked card."""
        self.picked_cards.append(card_name)
        logger.info(f"Recorded pick: {card_name} (total: {len(self.picked_cards)})")

def display_draft_pack(cards: List[DraftCard], pack_num: int, pick_num: int, recommendation: str, show_count: int = 15):
    """Display draft pack using clean table format (for terminal/CLI mode)."""
    print(f"\n{'='*80}\nPack {pack_num}, Pick {pick_num}\n{'='*80}\n")
    if not cards:
        print("No cards in pack")
        return

    table = [
        [
            i,
            format_color_emoji(c.colors),
            format_rarity_emoji(c.rarity),
            c.name,
            format_grade(c.grade) if c.grade else "",
            f"{c.gih_win_rate*100:.1f}%" if c.gih_win_rate > 0 else "",
            f"{c.ata:.1f}" if c.ata > 0 else "",
            " ".join(c.types[:2]) if c.types else ""
        ]
        for i, c in enumerate(cards[:show_count], 1)
    ]

    print(tabulate(table, headers=["#", "", "", "Card", "Grade", "GIH WR", "ATA", "Type"], tablefmt="simple", colalign=("right", "center", "center", "left", "center", "right", "right", "left")))
    print(f"\n💡 Recommendation: {recommendation}\n")

def format_draft_pack_for_gui(cards: List[DraftCard], pack_num: int, pick_num: int, recommendation: str, show_count: int = 15, picked_cards: Optional[List[str]] = None, card_metadata_db=None, split_panes: bool = False) -> tuple:
    """Format draft pack as list of strings for GUI display."""
    lines = [
        "="*80, f"DRAFT: Pack {pack_num}, Pick {pick_num}", "="*80,
        f"{'#':<3}{'C':<3}{'R':<2}{'Card Name':<30}{'Grade':<5}{'GIH WR':<8}{'ATA':<5}{'Type'}", "-"*80
    ]

    if not cards:
        lines.append("No cards in pack")
        return (lines, []) if split_panes else lines

    for i, card in enumerate(cards[:show_count], 1):
        lines.append(
            f"{i:<3}{format_color_emoji(card.colors, True):<3}{format_rarity_emoji(card.rarity, True):<2}"
            f"{card.name:<30}{card.grade or '':<5}{f'{card.gih_win_rate*100:.1f}%' if card.gih_win_rate > 0 else '':<8}"
            f"{f'{card.ata:.1f}' if card.ata > 0 else '':<5}{' '.join(card.types[:2]) if card.types else ''}"
        )

    lines.extend(["", f"💡 RECOMMENDATION: {recommendation}", ""])

    picked_lines = []
    if picked_cards:
        picked_lines.extend(["="*80, f"PICKED CARDS ({len(picked_cards)} total)", "="*80, ""])
        color_groups = {c: [] for c in ['W','U','B','R','G','Multi','Colorless']}
        for name in picked_cards:
            colors = ""
            if card_metadata_db:
                metadata = card_metadata_db.get_card_metadata(name)
                if metadata: colors = metadata.get("color_identity", "")

            if not colors: key = 'Colorless'
            elif len(colors) > 1: key = 'Multi'
            else: key = colors
            color_groups[key].append(name)

        color_map = {'W':'White','U':'Blue','B':'Black','R':'Red','G':'Green','Multi':'Multicolor','Colorless':'Colorless'}
        for color, group in color_groups.items():
            if group:
                picked_lines.append(f"{color_map[color]} ({len(group)}):")
                picked_lines.extend([f"  • {card}" for card in group])
                picked_lines.append("")

    return (lines, picked_lines) if split_panes else lines + picked_lines

def format_color_emoji(colors: str, for_gui: bool = False) -> str:
    """Convert color string to emoji or letter codes."""
    if not colors: return "C" if for_gui else "⚪"
    if for_gui: return colors[:2]
    emoji_map = {"W": "⚪", "U": "🔵", "B": "⚫", "R": "🔴", "G": "🟢"}
    return "".join(emoji_map.get(c, "") for c in colors[:2]) if len(colors) > 1 else emoji_map.get(colors, "")

def format_rarity_emoji(rarity: str, for_gui: bool = False) -> str:
    """Convert rarity to emoji or letter codes."""
    rarity_map = {"mythic": "M", "rare": "R", "uncommon": "U", "common": "C"} if for_gui else {"mythic": "✨", "rare": "💎", "uncommon": "🔷", "common": "⬜"}
    return rarity_map.get(rarity.lower() if rarity else "", "")

def format_grade(grade: str) -> str:
    """Return color-coded grade string."""
    if not grade: return ""
    color = "green" if grade.startswith("A") else "cyan" if grade.startswith("B") else "yellow" if grade.startswith("C") else "red"
    attrs = ["bold"] if grade.startswith(("A", "F")) else []
    return colored(grade, color, attrs=attrs)
