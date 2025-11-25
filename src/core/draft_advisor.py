#!/usr/bin/env python3
"""
Draft Advisor for MTGA Voice Advisor

Provides draft pick recommendations using 17lands data and optional AI analysis.
Clean table-based output inspired by python-mtga-helper.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm
from tabulate import tabulate
from termcolor import colored

from ..data.data_management import ScryfallClient, CardStatsDB
from ..data.arena_cards import ArenaCardDatabase

logger = logging.getLogger(__name__)


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
        "A+": 99, "A": 95, "A-": 90,
        "B+": 85, "B": 76, "B-": 68,
        "C+": 57, "C": 45, "C-": 36,
        "D+": 27, "D": 17, "D-": 5,
        "F": 0,
    }

    def __init__(self, scryfall_client: ScryfallClient, card_stats_db: CardStatsDB, ai_advisor=None, arena_card_db: ArenaCardDatabase = None):
        self.scryfall = scryfall_client
        self.stats_db = card_stats_db
        self.ai_advisor = ai_advisor
        self.arena_db = arena_card_db  # Local Arena card database (fast, always available)
        self.picked_cards = []
        self.current_pack_num = 0
        self.current_pick_num = 0
        self.current_set = ""

    def recommend_pick(
        self,
        pack_arena_ids: List[int],
        pack_num: int,
        pick_num: int,
        event_name: str = ""
    ) -> Tuple[List[DraftCard], str]:
        """
        Generate pick recommendation for a draft pack
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
            return [], "No cards found in pack"

        # 2. Enrich with stats
        pack_cards = self._enrich_with_stats(pack_cards)

        # 3. Calculate grades
        pack_cards = self._calculate_grades(pack_cards)

        # 4. Sort by grade/score (best first)
        pack_cards.sort(key=lambda c: (c.score if c.score else 0), reverse=True)

        # 5. Generate recommendation
        recommendation = self._generate_recommendation(pack_cards, pack_num, pick_num)

        return pack_cards, recommendation

    def _resolve_cards(self, arena_ids: List[int]) -> List[DraftCard]:
        """Resolve Arena IDs to card names using local Arena DB first, then Scryfall fallback"""
        cards = []
        for arena_id in arena_ids:
            try:
                card_data = None

                # Try local Arena database first (fast, always has latest cards)
                if self.arena_db:
                    local_data = self.arena_db.get_card_data(arena_id)
                    if local_data:
                        card_data = {
                            "name": local_data.get("name", f"Card {arena_id}"),
                            "colors": local_data.get("colors", ""),
                            "rarity": local_data.get("rarity", ""),
                            "type_line": local_data.get("type", ""),
                        }
                        logger.debug(f"Resolved {arena_id} -> {card_data['name']} (local DB)")

                # Fallback to Scryfall API if not in local DB
                if not card_data and self.scryfall:
                    logger.info(f"Card {arena_id} not in local DB, fetching from Scryfall...")
                    card_data = self.scryfall.get_card_by_arena_id(arena_id)

                if card_data:
                    card = DraftCard(
                        arena_id=arena_id,
                        name=card_data.get("name", f"Card {arena_id}"),
                        colors=card_data.get("colors", ""),
                        rarity=card_data.get("rarity", ""),
                        types=card_data.get("type_line", "").split() if card_data.get("type_line") else []
                    )
                    cards.append(card)
                else:
                    # Still create a card entry even if we can't resolve it
                    logger.warning(f"Could not resolve arena ID {arena_id}")
                    cards.append(DraftCard(arena_id=arena_id, name=f"Unknown Card {arena_id}"))
            except Exception as e:
                logger.error(f"Error resolving arena ID {arena_id}: {e}")
        return cards

    def _enrich_with_stats(self, cards: List[DraftCard]) -> List[DraftCard]:
        """Enrich cards with 17lands statistics"""
        if not self.stats_db:
            return cards

        for card in cards:
            try:
                # Try to get stats for current set, fallback to any set
                stats = self.stats_db.get_stats(card.name, self.current_set)
                if not stats and not self.current_set:
                     stats = self.stats_db.get_stats(card.name)

                if stats:
                    card.win_rate = stats.get("win_rate", 0.0) or 0.0
                    card.gih_win_rate = stats.get("gih_win_rate", 0.0) or 0.0
                    card.ata = stats.get("avg_taken_at", 0.0) or 0.0
                    card.ever_drawn_game_count = stats.get("games_played", 0) or 0
            except Exception as e:
                logger.error(f"Error loading stats for {card.name}: {e}")

        return cards

    def _calculate_grades(self, cards: List[DraftCard]) -> List[DraftCard]:
        """Calculate percentile-based grades"""
        gih_win_rates = [c.gih_win_rate for c in cards if c.gih_win_rate > 0]
        atas = [c.ata for c in cards if c.ata > 0]

        if not gih_win_rates or len(gih_win_rates) < 3:
            for card in cards:
                if card.gih_win_rate > 0:
                    card.score = self._simple_score(card.gih_win_rate)
                    card.grade = self._score_to_grade(card.score)
            return cards

        gih_mean = float(np.mean(gih_win_rates))
        gih_std = float(np.std(gih_win_rates, ddof=1))
        ata_mean = float(np.mean(atas)) if atas else 0
        ata_std = float(np.std(atas, ddof=1)) if atas else 0

        for card in cards:
            if card.gih_win_rate > 0 and gih_std > 0:
                gih_percentile = norm.cdf(card.gih_win_rate, loc=gih_mean, scale=gih_std)
                ata_percentile = 0.5
                if card.ata > 0 and ata_std > 0:
                    ata_percentile = 1 - norm.cdf(card.ata, loc=ata_mean, scale=ata_std)
                
                combined_score = (gih_percentile * 0.7) + (ata_percentile * 0.3)
                card.score = combined_score * 100
                card.grade = self._score_to_grade(card.score)

        return cards

    def _simple_score(self, win_rate: float) -> float:
        if win_rate >= 0.60: return 99
        elif win_rate >= 0.58: return 95
        elif win_rate >= 0.56: return 90
        elif win_rate >= 0.54: return 85
        elif win_rate >= 0.52: return 76
        elif win_rate >= 0.50: return 68
        elif win_rate >= 0.48: return 57
        else: return 45

    def _score_to_grade(self, score: float) -> str:
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def _generate_recommendation(self, cards: List[DraftCard], pack_num: int, pick_num: int) -> str:
        if not cards: return "No cards available"
        top_card = cards[0]
        rec = f"Pick {top_card.name}"
        if top_card.grade: rec += f" ({top_card.grade})"
        if top_card.ata > 0: rec += f" (ATA: {top_card.ata:.1f})"
        return rec

    def record_pick(self, card_name: str):
        self.picked_cards.append(card_name)


def display_draft_pack(cards: List[DraftCard], pack_num: int, pick_num: int, recommendation: str, show_count: int = 15):
    print("\n" + "="*80)
    print(f"Pack {pack_num}, Pick {pick_num}")
    print("="*80 + "\n")
    if not cards:
        print("No cards in pack")
        return

    table = []
    for i, card in enumerate(cards[:show_count], 1):
        table.append([
            i,
            format_color_emoji(card.colors),
            format_rarity_emoji(card.rarity),
            card.name,
            format_grade(card.grade) if card.grade else "",
            f"{card.gih_win_rate*100:.1f}%" if card.gih_win_rate > 0 else "",
            f"{card.ata:.1f}" if card.ata > 0 else "",
            " ".join(card.types[:2]) if card.types else ""
        ])

    print(tabulate(
        table,
        headers=["#", "", "", "Card", "Grade", "GIH WR", "ATA", "Type"],
        tablefmt="simple",
        colalign=("right", "center", "center", "left", "center", "right", "right", "left")
    ))
    print(f"\n💡 Recommendation: {recommendation}\n")


def format_draft_pack_for_gui(cards: List[DraftCard], pack_num: int, pick_num: int, recommendation: str, show_count: int = 15, picked_cards: List[str] = None, card_metadata_db = None, split_panes: bool = False) -> tuple:
    # Simplified GUI formatting
    lines = []
    lines.append(f"DRAFT: Pack {pack_num}, Pick {pick_num}")
    lines.append("-" * 80)
    
    for i, card in enumerate(cards[:show_count], 1):
        lines.append(f"{i}. {card.name} ({card.grade}) - WR: {card.gih_win_rate*100:.1f}%")
    
    lines.append(f"\nRec: {recommendation}")
    
    picked_lines = []
    if picked_cards:
        picked_lines.append("Picked Cards:")
        for p in picked_cards:
            picked_lines.append(f"- {p}")
            
    if split_panes:
        return (lines, picked_lines)
    return lines + picked_lines


def format_color_emoji(colors: str, for_gui: bool = False) -> str:
    if not colors: return "C" if for_gui else "⚪"
    if for_gui: return colors[:2]
    emoji_map = {"W": "⚪", "U": "🔵", "B": "⚫", "R": "🔴", "G": "🟢"}
    if len(colors) > 1: return "".join(emoji_map.get(c, "") for c in colors[:2])
    return emoji_map.get(colors, "")

def format_rarity_emoji(rarity: str, for_gui: bool = False) -> str:
    if for_gui: return rarity[0].upper() if rarity else ""
    emoji_map = {"mythic": "✨", "rare": "💎", "uncommon": "🔷", "common": "⬜"}
    return emoji_map.get(rarity.lower() if rarity else "", "")

def format_grade(grade: str) -> str:
    if not grade: return ""
    color = "green" if grade.startswith("A") else "cyan" if grade.startswith("B") else "yellow" if grade.startswith("C") else "red"
    return colored(grade, color)
