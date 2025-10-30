#!/usr/bin/env python3
"""
Download card data from 17lands card_data API endpoint.

This is MUCH faster than parsing CSV files - uses JSON API instead!

API returns data like:
- GIH WR (Games In Hand Win Rate)
- OH WR (Opening Hand Win Rate)
- GD WR (Games Drawn Win Rate)
- GP WR (Games Played Win Rate)
- ALSA (Average Last Seen At - draft pick)
- # Seen, # Picked, etc.

Usage:
    python3 download_card_data_api.py
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import time

from download_real_17lands_data import ALL_SETS, CURRENT_STANDARD
from rag_advisor import CardStatsDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 17lands card_data API endpoint
CARD_DATA_API = "https://www.17lands.com/card_data"


def download_card_data_api(
    set_code: str,
    format_type: str = "PremierDraft",
    start_date: str = None
) -> List[Dict]:
    """
    Download card data from 17lands API endpoint (JSON).

    Args:
        set_code: Set code (e.g., 'FDN', 'DSK')
        format_type: Format (PremierDraft, QuickDraft, Sealed)
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago

    Returns:
        List of card statistics dictionaries
    """
    if start_date is None:
        # Default to 30 days ago
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        'expansion': set_code,
        'format': format_type,
        'start': start_date,
    }

    logger.info(f"Downloading {set_code} ({ALL_SETS.get(set_code, set_code)}) via API...")
    logger.info(f"  Format: {format_type}")
    logger.info(f"  Start date: {start_date}")

    try:
        response = requests.get(CARD_DATA_API, params=params, timeout=30)
        response.raise_for_status()

        # Check if we got JSON
        if 'application/json' not in response.headers.get('Content-Type', ''):
            logger.error(f"  ✗ Not JSON response (got {response.headers.get('Content-Type')})")
            return []

        data = response.json()

        # API returns list of card objects
        if not isinstance(data, list):
            logger.error(f"  ✗ Unexpected response format")
            return []

        logger.info(f"  ✓ Retrieved {len(data)} cards")

        # Convert API format to our database format
        cards = []
        for card in data:
            # Skip cards with insufficient data
            if card.get('seen_count', 0) < 100:  # Min 100 games
                continue

            cards.append({
                'card_name': card.get('name', 'Unknown'),
                'set_code': set_code,
                'games_played': card.get('game_count', 0),
                'win_rate': card.get('gp_wr', 0.0),  # Games Played WR
                'gih_win_rate': card.get('gih_wr', 0.0),  # Games In Hand WR
                'opening_hand_win_rate': card.get('oh_wr', 0.0),  # Opening Hand WR
                'drawn_win_rate': card.get('gd_wr', 0.0),  # Games Drawn WR
                'iwd': card.get('iwd', 0.0),  # Improvement When Drawn
                'alsa': card.get('alsa', 0.0),  # Average Last Seen At (draft pick)
                'avg_taken_at': card.get('ata', 0.0),  # Average Taken At
                'seen_count': card.get('seen_count', 0),
                'pick_count': card.get('pick_count', 0),
                'color': card.get('color', ''),
                'rarity': card.get('rarity', ''),
                'last_updated': datetime.now().isoformat()
            })

        logger.info(f"  ✓ Parsed {len(cards)} cards (min 100 games)")
        return cards

    except requests.exceptions.RequestException as e:
        logger.error(f"  ✗ API request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return []


def download_multiple_sets(
    set_codes: List[str],
    format_type: str = "PremierDraft"
) -> int:
    """
    Download card data for multiple sets via API.

    Args:
        set_codes: List of set codes to download
        format_type: Format type

    Returns:
        Total number of cards added to database
    """
    db = CardStatsDB()
    total_cards = 0

    for i, set_code in enumerate(set_codes, 1):
        logger.info(f"\n[{i}/{len(set_codes)}] Processing {set_code}...")

        cards = download_card_data_api(set_code, format_type)

        if cards:
            # Remove old data for this set
            db.delete_set_data(set_code)

            # Insert new data
            db.insert_card_stats(cards)
            total_cards += len(cards)
            logger.info(f"  ✓ Added {len(cards)} cards to database")
        else:
            logger.warning(f"  ⚠️  No data available (set may be too new)")

        # Small delay between requests
        if i < len(set_codes):
            time.sleep(1)

    db.close()
    return total_cards


def main():
    logger.info("="*70)
    logger.info("17lands Card Data API Downloader (FAST)")
    logger.info("="*70)
    logger.info("\nThis uses the JSON API endpoint instead of CSV files.")
    logger.info("Much faster - downloads in seconds instead of minutes!\n")

    print("\nChoose download method:")
    print("  1. Quick test - single set (FDN, ~30 seconds)")
    print("  2. Current Standard sets (7 sets, ~5 minutes)")
    print(f"  3. All available sets ({len(ALL_SETS)} sets, ~15 minutes)")
    print("  4. Cancel")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == "1":
        # Quick test
        logger.info("\nDownloading FDN (test)...")
        cards = download_card_data_api("FDN", "PremierDraft")

        if cards:
            db = CardStatsDB()
            db.insert_card_stats(cards)
            db.close()
            logger.info(f"\n✓ Imported {len(cards)} cards!")

    elif choice == "2":
        # Current Standard
        logger.info(f"\nDownloading {len(CURRENT_STANDARD)} Standard sets via API...")
        logger.info(f"Sets: {', '.join(CURRENT_STANDARD)}")

        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Cancelled.")
            return

        total = download_multiple_sets(CURRENT_STANDARD)
        logger.info(f"\n✓ Imported {total:,} total cards!")

    elif choice == "3":
        # All sets
        logger.info(f"\nDownloading ALL {len(ALL_SETS)} sets via API...")

        confirm = input("\nThis will take ~15 minutes. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Cancelled.")
            return

        total = download_multiple_sets(list(ALL_SETS.keys()))
        logger.info(f"\n✓ Imported {total:,} total cards!")

    else:
        logger.info("Cancelled.")
        return

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
