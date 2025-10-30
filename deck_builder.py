#!/usr/bin/env python3
"""
Deck Builder for MTGA Voice Advisor

Suggests deck configurations based on 17lands winning deck data.
Matches drafted cards against successful deck archetypes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import csv

logger = logging.getLogger(__name__)


@dataclass
class DeckSuggestion:
    """Represents a suggested deck configuration"""
    main_colors: str  # e.g., "WU", "BR"
    splash_colors: str  # e.g., "" or "G"
    maindeck: Dict[str, int]  # {card_name: count}
    sideboard: Dict[str, int]  # {card_name: count}
    lands: Dict[str, int]  # {land_name: count}
    similarity_score: float  # 0-1 how well it matches drafted cards
    win_rate: float  # Average win rate of source decks
    num_source_decks: int  # How many winning decks this is based on
    color_pair_name: str  # Human-readable like "Azorius (WU)"


class DeckBuilder:
    """
    Builds deck suggestions based on 17lands winning deck data
    """

    # Standard limited deck composition
    DECK_SIZE = 40
    DEFAULT_NONLAND_COUNT = 23
    DEFAULT_LAND_COUNT = 17

    # Color pair names
    COLOR_PAIR_NAMES = {
        "W": "Mono White",
        "U": "Mono Blue",
        "B": "Mono Black",
        "R": "Mono Red",
        "G": "Mono Green",
        "WU": "Azorius",
        "WB": "Orzhov",
        "WR": "Boros",
        "WG": "Selesnya",
        "UB": "Dimir",
        "UR": "Izzet",
        "UG": "Simic",
        "BR": "Rakdos",
        "BG": "Golgari",
        "RG": "Gruul",
    }

    def __init__(self, data_dir: Path = Path("data")):
        """Initialize deck builder with path to 17lands data"""
        self.data_dir = data_dir
        self.winning_decks_cache = {}  # {set_code: List[deck_data]}

    def load_winning_decks(self, set_code: str, min_win_rate: float = 0.55, max_decks: int = 5000) -> List[Dict]:
        """
        Load winning decks from 17lands CSV data

        Args:
            set_code: Set code like "BLB", "FDN", etc.
            min_win_rate: Minimum win rate to consider (default: 55%)

        Returns:
            List of deck dictionaries with card counts and metadata
        """
        # Check cache
        cache_key = f"{set_code}_{min_win_rate}"
        if cache_key in self.winning_decks_cache:
            logger.debug(f"Using cached winning decks for {set_code}")
            return self.winning_decks_cache[cache_key]

        # Try multiple format types (PremierDraft, PickTwoDraft, QuickDraft)
        format_types = ["PremierDraft", "PickTwoDraft", "QuickDraft", "TradDraft"]
        csv_path = None

        for format_type in format_types:
            test_path = self.data_dir / f"17lands_{set_code.upper()}_{format_type}.csv"
            if test_path.exists():
                csv_path = test_path
                logger.debug(f"Found 17lands data: {csv_path.name}")
                break

        if not csv_path:
            logger.warning(f"17lands data not found for {set_code} (tried: {', '.join(format_types)})")
            return []

        logger.info(f"Loading winning decks from {csv_path.name}...")

        winning_decks = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Use csv.DictReader with a large field size limit
                csv.field_size_limit(1000000)
                reader = csv.DictReader(f)

                for row_num, row in enumerate(reader):
                    # Only process winning games
                    if row.get('won') != 'True':
                        continue

                    # Extract deck composition from deck_ columns
                    deck_cards = {}
                    for col_name, value in row.items():
                        if col_name.startswith('deck_') and value and value != '0':
                            card_name = col_name[5:]  # Remove 'deck_' prefix
                            try:
                                count = int(value)
                                if count > 0:
                                    deck_cards[card_name] = count
                            except ValueError:
                                continue

                    if not deck_cards:
                        continue

                    # Extract metadata
                    main_colors = row.get('main_colors', '')
                    splash_colors = row.get('splash_colors', '')

                    deck_data = {
                        'main_colors': main_colors,
                        'splash_colors': splash_colors,
                        'cards': deck_cards,
                        'won': True
                    }

                    winning_decks.append(deck_data)

                    # Limit to prevent memory issues
                    if len(winning_decks) >= max_decks:
                        break

        except Exception as e:
            logger.error(f"Error loading winning decks from {csv_path}: {e}")
            return []

        logger.info(f"Loaded {len(winning_decks)} winning decks from {set_code}")

        # Cache the results
        self.winning_decks_cache[cache_key] = winning_decks

        return winning_decks

    def suggest_deck(
        self,
        drafted_cards: List[str],
        set_code: str,
        top_n: int = 1
    ) -> List[DeckSuggestion]:
        """
        Suggest deck configurations based on drafted cards

        Args:
            drafted_cards: List of card names drafted
            set_code: Set code like "BLB"
            top_n: Number of suggestions to return

        Returns:
            List of DeckSuggestion objects sorted by similarity
        """
        # Load winning decks for this set
        winning_decks = self.load_winning_decks(set_code, max_decks=5000)

        if not winning_decks:
            logger.warning(f"No winning deck data available for {set_code}")
            return []

        # Convert drafted cards to set for faster lookup
        drafted_set = set(drafted_cards)
        drafted_counts = Counter(drafted_cards)

        # Score each winning deck by similarity to drafted pool
        deck_scores = []

        for deck_data in winning_decks:
            deck_cards = deck_data['cards']

            # Calculate similarity: how many drafted cards are in this deck
            overlap_cards = drafted_set & set(deck_cards.keys())
            overlap_count = sum(min(drafted_counts[card], deck_cards[card])
                              for card in overlap_cards)

            # Normalize by deck size
            total_nonlands = sum(count for card, count in deck_cards.items()
                                if not self._is_basic_land(card))

            if total_nonlands == 0:
                continue

            similarity = overlap_count / total_nonlands

            deck_scores.append({
                'deck_data': deck_data,
                'similarity': similarity,
                'overlap_cards': overlap_cards,
                'overlap_count': overlap_count
            })

        # Sort by similarity
        deck_scores.sort(key=lambda x: x['similarity'], reverse=True)

        # Group similar decks by color pair
        color_pair_groups = defaultdict(list)
        for score in deck_scores[:200]:  # Consider top 200 most similar decks
            main_colors = score['deck_data']['main_colors']
            color_pair_groups[main_colors].append(score)

        # Create suggestions for top color pairs
        suggestions = []

        for main_colors, decks_in_color in sorted(
            color_pair_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:top_n]:

            # Aggregate cards from similar decks in this color pair
            card_frequency = Counter()
            total_similarity = 0

            for deck_score in decks_in_color[:10]:  # Use top 10 decks
                deck_cards = deck_score['deck_data']['cards']
                for card, count in deck_cards.items():
                    if not self._is_basic_land(card):
                        card_frequency[card] += 1
                total_similarity += deck_score['similarity']

            avg_similarity = total_similarity / len(decks_in_color[:10])

            # Build suggested maindeck prioritizing:
            # 1. Cards we actually drafted
            # 2. Cards that appear frequently in winning decks
            suggested_maindeck = {}
            suggested_sideboard = {}

            for card in drafted_cards:
                if self._is_basic_land(card):
                    continue

                # Card in drafted pool - add to maindeck if it appears in winning decks
                if card in card_frequency:
                    # Use the count we drafted (up to what's typical)
                    count = drafted_counts[card]
                    suggested_maindeck[card] = count
                else:
                    # Card not common in this archetype - suggest sideboard
                    suggested_sideboard[card] = drafted_counts[card]

            # Calculate lands
            nonland_count = sum(suggested_maindeck.values())
            land_count = self.DECK_SIZE - nonland_count

            # Suggest basic land distribution based on color
            lands = self._suggest_lands(main_colors, land_count)

            # Create suggestion
            color_pair_name = self._get_color_pair_name(main_colors)

            suggestion = DeckSuggestion(
                main_colors=main_colors,
                splash_colors="",  # TODO: detect splashes
                maindeck=suggested_maindeck,
                sideboard=suggested_sideboard,
                lands=lands,
                similarity_score=avg_similarity,
                win_rate=0.0,  # TODO: calculate from source decks
                num_source_decks=len(decks_in_color),
                color_pair_name=color_pair_name
            )

            suggestions.append(suggestion)

        return suggestions

    def _is_basic_land(self, card_name: str) -> bool:
        """Check if card is a basic land"""
        basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
        return card_name in basics

    def _suggest_lands(self, main_colors: str, total_lands: int) -> Dict[str, int]:
        """Suggest basic land distribution"""
        land_map = {
            "W": "Plains",
            "U": "Island",
            "B": "Swamp",
            "R": "Mountain",
            "G": "Forest"
        }

        if not main_colors:
            return {"Plains": total_lands}  # Default

        if len(main_colors) == 1:
            # Mono-color
            land_type = land_map.get(main_colors, "Plains")
            return {land_type: total_lands}

        # Two-color: split evenly
        color1, color2 = main_colors[0], main_colors[1]
        land1 = land_map.get(color1, "Plains")
        land2 = land_map.get(color2, "Island")

        half = total_lands // 2
        return {
            land1: half,
            land2: total_lands - half
        }

    def _get_color_pair_name(self, colors: str) -> str:
        """Get human-readable color pair name"""
        return self.COLOR_PAIR_NAMES.get(colors, colors)


def display_deck_suggestion(suggestion: DeckSuggestion):
    """Display a deck suggestion using clean table format"""
    from tabulate import tabulate
    from termcolor import colored

    print("\n" + "="*80)
    print(f"Suggested Deck: {suggestion.color_pair_name} ({suggestion.main_colors})")
    print("="*80 + "\n")

    print(f"Based on {suggestion.num_source_decks} winning decks")
    print(f"Similarity to your draft: {suggestion.similarity_score*100:.1f}%")
    print()

    # Maindeck
    if suggestion.maindeck:
        print("MAINDECK (Spells):")
        print("-" * 40)

        # Sort by count (descending) then name
        sorted_cards = sorted(suggestion.maindeck.items(),
                            key=lambda x: (-x[1], x[0]))

        table = []
        for card_name, count in sorted_cards:
            table.append([count, card_name])

        print(tabulate(table, tablefmt="plain"))
        print()

    # Lands
    if suggestion.lands:
        print("LANDS:")
        print("-" * 40)
        for land_name, count in sorted(suggestion.lands.items()):
            print(f"{count:2} {land_name}")
        print()

    # Deck totals
    total_spells = sum(suggestion.maindeck.values())
    total_lands = sum(suggestion.lands.values())
    print(f"Total: {total_spells + total_lands} cards ({total_spells} spells + {total_lands} lands)")

    # Sideboard
    if suggestion.sideboard:
        print("\nSIDEBOARD:")
        print("-" * 40)

        sorted_sb = sorted(suggestion.sideboard.items(),
                          key=lambda x: (-x[1], x[0]))

        for card_name, count in sorted_sb[:10]:  # Show top 10
            print(f"{count:2} {card_name}")

        if len(suggestion.sideboard) > 10:
            print(f"   ... and {len(suggestion.sideboard) - 10} more cards")

    print()


def format_deck_suggestion_for_gui(suggestion: DeckSuggestion) -> List[str]:
    """
    Format deck suggestion as list of strings for GUI display

    Args:
        suggestion: DeckSuggestion object

    Returns:
        List of formatted strings ready for GUI display
    """
    lines = []
    lines.append("="*80)
    lines.append(f"DECK SUGGESTION: {suggestion.color_pair_name} ({suggestion.main_colors})")
    lines.append("="*80)
    lines.append("")

    lines.append(f"Based on {suggestion.num_source_decks} winning decks")
    lines.append(f"Similarity to your draft: {suggestion.similarity_score*100:.1f}%")
    lines.append("")

    # Maindeck
    if suggestion.maindeck:
        lines.append("MAINDECK (Spells):")
        lines.append("-" * 40)

        # Sort by count (descending) then name
        sorted_cards = sorted(suggestion.maindeck.items(),
                            key=lambda x: (-x[1], x[0]))

        for card_name, count in sorted_cards:
            lines.append(f"{count:2} {card_name}")

        lines.append("")

    # Lands
    if suggestion.lands:
        lines.append("LANDS:")
        lines.append("-" * 40)
        for land_name, count in sorted(suggestion.lands.items()):
            lines.append(f"{count:2} {land_name}")
        lines.append("")

    # Deck totals
    total_spells = sum(suggestion.maindeck.values())
    total_lands = sum(suggestion.lands.values())
    lines.append(f"Total: {total_spells + total_lands} cards ({total_spells} spells + {total_lands} lands)")
    lines.append("")

    # Sideboard
    if suggestion.sideboard:
        lines.append("SIDEBOARD:")
        lines.append("-" * 40)

        sorted_sb = sorted(suggestion.sideboard.items(),
                          key=lambda x: (-x[1], x[0]))

        for card_name, count in sorted_sb[:10]:  # Show top 10
            lines.append(f"{count:2} {card_name}")

        if len(suggestion.sideboard) > 10:
            lines.append(f"   ... and {len(suggestion.sideboard) - 10} more cards")
        lines.append("")

    return lines
