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
import gzip
import shutil
import requests
import threading

logger = logging.getLogger(__name__)


@dataclass
class DeckSuggestion:
    """Represents a suggested deck configuration.

    Attributes:
        main_colors: The primary color identity of the deck (e.g., "WU").
        splash_colors: Any splash colors in the deck (e.g., "G").
        maindeck: A dictionary mapping card names to their counts in the main deck.
        sideboard: A dictionary mapping card names to their counts in the sideboard.
        lands: A dictionary mapping basic land names to their counts.
        similarity_score: A score from 0 to 1 indicating how well the suggested
            deck matches the player's drafted cards.
        win_rate: The average win rate of the source decks this suggestion is based on.
        num_source_decks: The number of winning decks used to generate this suggestion.
        color_pair_name: The human-readable name for the color pair (e.g., "Azorius").
    """
    main_colors: str
    splash_colors: str
    maindeck: Dict[str, int]
    sideboard: Dict[str, int]
    lands: Dict[str, int]
    similarity_score: float
    win_rate: float
    num_source_decks: int
    color_pair_name: str


class DeckBuilder:
    """Builds deck suggestions for a pool of drafted cards.

    This class uses historical data of winning decks from 17lands to find
    archetypes that best match a player's drafted card pool. It then
    constructs a suggested decklist based on the most similar winning decks.

    Attributes:
        data_dir: The directory where 17lands CSV data is stored.
        winning_decks_cache: An in-memory cache for winning deck data to avoid
            re-reading files.
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
        self.cache_dir = data_dir / "17lands_cache"
        self.compressed_cache = self.cache_dir / "compressed"
        self.decompressed_cache = self.cache_dir / "decompressed"
        self.winning_decks_cache = {}  # {set_code: List[deck_data]}
        self.download_locks = {}  # {set_code: threading.Lock}

        # Create cache directories
        self.compressed_cache.mkdir(parents=True, exist_ok=True)
        self.decompressed_cache.mkdir(parents=True, exist_ok=True)

    def _ensure_game_data(self, set_code: str, format_type: str = "PremierDraft") -> Optional[Path]:
        """
        Ensure game data is available, downloading if necessary.

        Returns path to decompressed CSV, or None if unavailable.
        """
        set_code = set_code.upper()

        # Check decompressed cache first
        decompressed_path = self.decompressed_cache / f"{set_code}_{format_type}.csv"
        if decompressed_path.exists():
            logger.debug(f"Using cached data: {decompressed_path.name}")
            return decompressed_path

        # Check if we need to decompress
        compressed_path = self.compressed_cache / f"{set_code}_{format_type}.csv.gz"
        if compressed_path.exists():
            logger.info(f"Decompressing cached file: {compressed_path.name}")
            try:
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logger.info(f"Decompressed: {decompressed_path.name} ({decompressed_path.stat().st_size / (1024*1024):.1f} MB)")
                return decompressed_path
            except Exception as e:
                logger.error(f"Error decompressing {compressed_path}: {e}")
                return None

        # Need to download - use lock to prevent duplicate downloads
        if set_code not in self.download_locks:
            self.download_locks[set_code] = threading.Lock()

        with self.download_locks[set_code]:
            # Check again in case another thread downloaded while we waited
            if decompressed_path.exists():
                return decompressed_path

            # Download from 17lands
            url = f"https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{set_code}.{format_type}.csv.gz"

            logger.info(f"Downloading 17lands data for {set_code} from {url}")
            logger.info("This may take a few minutes for large datasets...")

            try:
                response = requests.get(url, stream=True, timeout=180)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                # Download to compressed cache
                with open(compressed_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (10*1024*1024) == 0:  # Log every 10MB
                            percent = (downloaded / total_size) * 100
                            logger.info(f"  Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)")

                logger.info(f"Downloaded: {compressed_path.name} ({compressed_path.stat().st_size / (1024*1024):.1f} MB)")

                # Decompress
                logger.info("Decompressing...")
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logger.info(f"Ready: {decompressed_path.name} ({decompressed_path.stat().st_size / (1024*1024):.1f} MB)")
                return decompressed_path

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403 or e.response.status_code == 404:
                    logger.warning(f"Game data not yet available for {set_code} (HTTP {e.response.status_code})")
                else:
                    logger.error(f"HTTP error downloading {set_code} data: {e}")
                return None
            except Exception as e:
                logger.error(f"Error downloading {set_code} data: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return None

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

        # First try to find existing files in old location (backward compatibility)
        for format_type in format_types:
            test_path = self.data_dir / f"17lands_{set_code.upper()}_{format_type}.csv"
            if test_path.exists():
                csv_path = test_path
                logger.debug(f"Found 17lands data: {csv_path.name}")
                break

        # If not found, try to download using new cache system
        if not csv_path:
            logger.info(f"Game data not in cache, attempting download for {set_code}...")
            for format_type in format_types:
                csv_path = self._ensure_game_data(set_code, format_type)
                if csv_path:
                    break

        if not csv_path:
            logger.warning(f"17lands data not available for {set_code} (tried: {', '.join(format_types)})")
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
