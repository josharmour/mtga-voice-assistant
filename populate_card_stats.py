#!/usr/bin/env python3
"""
Populate card_stats.db with 17lands card ratings using their API.

Much faster than processing CSV files (seconds vs hours).
"""

import sys
import logging
import time
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.data_management import CardStatsDB
from tools.test_17lands_api import SeventeenLandsAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_sets():
    """Fetch all available set codes from 17lands API."""
    try:
        url = "https://www.17lands.com/data/expansions"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        sets = response.json()
        # Filter out special formats (Cube, Chaos, etc.)
        exclude_sets = {'Cube', 'Cube - Powered', 'Chaos', 'CORE', 'Ravnica'}
        standard_sets = [s for s in sets if s not in exclude_sets]
        logger.info(f"Found {len(standard_sets)} sets from 17lands API")
        return sorted(standard_sets)

    except Exception as e:
        logger.error(f"Failed to fetch set list from API: {e}")
        # Fallback to recent sets
        return ["TLA", "FDN", "DSK", "MH3", "OTJ", "WOE"]

def populate_card_stats():
    """Fetch card ratings from 17lands API and populate database."""
    logger.info("="*70)
    logger.info("Populating card_stats.db with 17lands API data")
    logger.info("="*70)

    # Initialize
    client = SeventeenLandsAPIClient(cache_minutes=30)
    db = CardStatsDB()

    # Get all available sets dynamically
    all_sets = get_available_sets()
    logger.info(f"Will process {len(all_sets)} sets")

    total_cards = 0
    total_time = time.time()
    sets_with_data = 0

    for set_code in all_sets:
        logger.info(f"\n📦 Processing {set_code}...")

        try:
            # Fetch ratings from API
            start = time.time()
            ratings = client.get_card_ratings(set_code, format="PremierDraft")
            elapsed = time.time() - start

            if not ratings:
                logger.warning(f"  ⚠️  No data found for {set_code}")
                continue

            logger.info(f"  ✓ Fetched {len(ratings)} cards in {elapsed:.1f}s")

            # Convert to database format
            stats_batch = []
            for rating in ratings:
                # Skip cards with insufficient data
                if rating.game_count < 100:
                    continue

                stat = {
                    'card_name': rating.name,
                    'set_code': set_code,
                    'color': rating.color or '',
                    'rarity': rating.rarity or '',
                    'games_played': rating.game_count,
                    'win_rate': rating.win_rate or 0.0,
                    'avg_taken_at': 0.0,  # Not available from API
                    'games_in_hand': rating.game_count,  # Approximate
                    'gih_win_rate': rating.drawn_win_rate or rating.ever_drawn_win_rate or rating.win_rate or 0.0,
                    'opening_hand_win_rate': 0.0,  # Not available
                    'drawn_win_rate': rating.drawn_win_rate or 0.0,
                    'ever_drawn_win_rate': rating.ever_drawn_win_rate or 0.0,
                    'never_drawn_win_rate': 0.0,  # Not available
                    'alsa': 0.0,  # Not available
                    'ata': 0.0,  # Not available
                    'iwd': rating.drawn_improvement_win_rate or 0.0,
                    'format': 'PremierDraft',
                }
                stats_batch.append(stat)

            # Delete old data for this set
            db.delete_set_data(set_code)

            # Insert new data
            db.insert_card_stats(stats_batch)

            total_cards += len(stats_batch)
            sets_with_data += 1
            logger.info(f"  ✓ Inserted {len(stats_batch)} cards (>100 games)")

        except Exception as e:
            logger.error(f"  ✗ Error processing {set_code}: {e}")
            continue

        # Be polite to the API
        time.sleep(0.5)

    total_elapsed = time.time() - total_time

    logger.info("\n" + "="*70)
    logger.info(f"✅ Database populated successfully!")
    logger.info(f"   Sets with data: {sets_with_data}/{len(all_sets)}")
    logger.info(f"   Total cards: {total_cards}")
    logger.info(f"   Total time: {total_elapsed:.1f}s")
    logger.info(f"   Database: data/card_stats.db")
    logger.info("="*70)

if __name__ == "__main__":
    populate_card_stats()
