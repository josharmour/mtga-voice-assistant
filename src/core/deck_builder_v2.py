#!/usr/bin/env python3
"""
Deck Builder for MTGA Voice Advisor - API-based approach

Suggests deck configurations based on 17lands card ratings API data.
Uses GIHWR (Games In Hand Win Rate) and archetype constraints.
Similar approach to MTGA_Draft_17Lands project.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter
import sqlite3
import threading

logger = logging.getLogger(__name__)


@dataclass
class CardRating:
    """Card performance metrics from 17lands"""
    name: str
    color: str
    rarity: str
    gih_win_rate: float  # Games In Hand Win Rate
    avg_taken_at: float  # Average pick number
    cmc: int = 0  # Converted Mana Cost
    is_creature: bool = False


@dataclass
class DeckSuggestion:
    """Represents a suggested deck configuration"""
    archetype: str  # "Aggro", "Midrange", or "Control"
    main_colors: str
    color_pair_name: str
    maindeck: Dict[str, int]
    sideboard: Dict[str, int]
    lands: Dict[str, int]
    avg_gihwr: float  # Average GIHWR of maindeck
    penalty: float  # Penalty for unmet archetype requirements
    score: float  # Final score (avg_gihwr - penalty)


class DeckBuilderV2:
    """
    Builds deck suggestions using 17lands card ratings API data.

    This approach:
    - Uses card_stats.db populated from 17lands API
    - Works immediately for new sets (no GB downloads)
    - Builds Aggro/Midrange/Control archetypes
    - Scores based on GIHWR with archetype penalties
    """

    # Archetype constraints (from MTGA_Draft_17Lands)
    ARCHETYPES = {
        "Aggro": {
            "lands": 16,
            "min_creatures": 17,
            "max_avg_cmc": 2.40,
            "curve_requirements": {1: 4, 2: 8},  # min creatures at CMC
        },
        "Midrange": {
            "lands": 17,
            "min_creatures": 15,
            "max_avg_cmc": 3.04,
            "curve_requirements": {2: 6, 3: 5},
        },
        "Control": {
            "lands": 18,
            "min_creatures": 10,
            "max_avg_cmc": 3.68,
            "curve_requirements": {4: 3, 5: 2},
        },
    }

    COLOR_PAIR_NAMES = {
        "W": "Mono White", "U": "Mono Blue", "B": "Mono Black",
        "R": "Mono Red", "G": "Mono Green",
        "WU": "Azorius", "WB": "Orzhov", "WR": "Boros", "WG": "Selesnya",
        "UB": "Dimir", "UR": "Izzet", "UG": "Simic",
        "BR": "Rakdos", "BG": "Golgari", "RG": "Gruul",
    }

    BASIC_LANDS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}

    def __init__(self, db_path: Path = Path("data/card_stats.db")):
        """Initialize deck builder with card stats database"""
        self.db_path = db_path
        self._local = threading.local()

    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _get_card_info(self, card_name: str, set_code: str) -> Optional[CardRating]:
        """Get card rating info from database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT card_name, color, rarity, gih_win_rate, avg_taken_at
                FROM card_stats
                WHERE card_name = ? AND set_code = ?
            """, (card_name, set_code))

            row = cursor.fetchone()
            if row:
                return CardRating(
                    name=row['card_name'],
                    color=row['color'] or "",
                    rarity=row['rarity'] or "",
                    gih_win_rate=row['gih_win_rate'] or 0.0,
                    avg_taken_at=row['avg_taken_at'] or 99.0,
                )
            return None
        except Exception as e:
            logger.error(f"Error getting card info for {card_name}: {e}")
            return None

    def suggest_deck(
        self,
        drafted_cards: List[str],
        set_code: str,
        top_n: int = 3
    ) -> List[DeckSuggestion]:
        """
        Suggest deck configurations based on drafted cards.

        Args:
            drafted_cards: List of card names drafted
            set_code: Set code like "TLA", "FIN", etc.
            top_n: Number of suggestions to return

        Returns:
            List of DeckSuggestion objects sorted by score
        """
        if not drafted_cards:
            return []

        # Get ratings for all drafted cards
        card_ratings = {}
        for card_name in set(drafted_cards):
            if card_name in self.BASIC_LANDS:
                continue  # Skip basic lands
            rating = self._get_card_info(card_name, set_code.upper())
            if rating:
                card_ratings[card_name] = rating

        if not card_ratings:
            logger.warning(f"No card ratings found for {set_code}. Make sure card_stats.db is populated.")
            return []

        logger.info(f"Building decks from {len(card_ratings)} rated cards")

        # Count card copies
        card_counts = Counter(drafted_cards)

        # Determine viable color pairs based on drafted cards
        color_scores = Counter()
        for card_name, rating in card_ratings.items():
            for color in rating.color:
                color_scores[color] += card_counts[card_name] * rating.gih_win_rate

        # Get top color pairs
        viable_pairs = self._get_viable_color_pairs(color_scores)[:5]  # Top 5 color combinations

        suggestions = []

        # Try each archetype with each color pair
        for colors in viable_pairs:
            for archetype_name in ["Aggro", "Midrange", "Control"]:
                suggestion = self._build_archetype_deck(
                    card_counts=card_counts,
                    card_ratings=card_ratings,
                    colors=colors,
                    archetype_name=archetype_name
                )
                if suggestion:
                    suggestions.append(suggestion)

        # Sort by score (highest first)
        suggestions.sort(key=lambda s: s.score, reverse=True)

        return suggestions[:top_n]

    def _get_viable_color_pairs(self, color_scores: Counter) -> List[str]:
        """Determine viable color pairs from color scores"""
        pairs = []

        # Mono-color
        for color in "WUBRG":
            if color in color_scores:
                pairs.append(color)

        # Two-color (sorted)
        sorted_colors = [c for c, _ in color_scores.most_common(3)]
        for i, c1 in enumerate(sorted_colors):
            for c2 in sorted_colors[i+1:]:
                pairs.append(''.join(sorted([c1, c2])))

        # Sort by total score
        pairs.sort(key=lambda p: sum(color_scores.get(c, 0) for c in p), reverse=True)

        return pairs

    def _build_archetype_deck(
        self,
        card_counts: Counter,
        card_ratings: Dict[str, CardRating],
        colors: str,
        archetype_name: str
    ) -> Optional[DeckSuggestion]:
        """Build a deck for specific archetype and colors"""

        archetype = self.ARCHETYPES[archetype_name]

        # Filter cards to color pair
        available_cards = []
        for card_name, rating in card_ratings.items():
            # Check if card fits color identity
            if not rating.color or all(c in colors for c in rating.color):
                available_cards.append((card_name, rating))

        # Sort by GIHWR (descending)
        available_cards.sort(key=lambda x: x[1].gih_win_rate, reverse=True)

        # Build maindeck (23 non-land cards)
        maindeck = {}
        sideboard = {}
        total_cards = 0
        target_nonlands = 40 - archetype["lands"]

        for card_name, rating in available_cards:
            if total_cards >= target_nonlands:
                # Rest goes to sideboard
                sideboard[card_name] = card_counts[card_name]
                continue

            # Add copies we have
            copies = min(card_counts[card_name], target_nonlands - total_cards)
            if copies > 0:
                maindeck[card_name] = copies
                total_cards += copies

                # Remaining copies to sideboard
                remaining = card_counts[card_name] - copies
                if remaining > 0:
                    sideboard[card_name] = remaining

        if total_cards == 0:
            return None

        # Calculate lands
        lands = self._suggest_lands(colors, archetype["lands"])

        # Calculate metrics
        total_gihwr = sum(
            card_ratings[card].gih_win_rate * count
            for card, count in maindeck.items()
        )
        avg_gihwr = total_gihwr / total_cards if total_cards > 0 else 0.0

        # Calculate penalty for unmet requirements
        penalty = 0.0
        # TODO: Implement curve/creature count penalties

        score = avg_gihwr - penalty

        return DeckSuggestion(
            archetype=archetype_name,
            main_colors=colors,
            color_pair_name=self._get_color_pair_name(colors),
            maindeck=maindeck,
            sideboard=sideboard,
            lands=lands,
            avg_gihwr=avg_gihwr,
            penalty=penalty,
            score=score
        )

    def _suggest_lands(self, colors: str, total_lands: int) -> Dict[str, int]:
        """Suggest basic land distribution"""
        land_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}

        if len(colors) == 0:
            return {"Plains": total_lands}
        elif len(colors) == 1:
            return {land_map[colors]: total_lands}
        else:
            # Split evenly for multi-color
            per_color = total_lands // len(colors)
            lands = {}
            for i, color in enumerate(colors):
                if i == len(colors) - 1:
                    # Last color gets remainder
                    lands[land_map[color]] = total_lands - sum(lands.values())
                else:
                    lands[land_map[color]] = per_color
            return lands

    def _get_color_pair_name(self, colors: str) -> str:
        """Get human-readable color pair name"""
        return self.COLOR_PAIR_NAMES.get(colors, colors)
