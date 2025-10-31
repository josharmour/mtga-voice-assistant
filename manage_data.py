#!/usr/bin/env python3
"""
Unified Data Management Script for MTGA Voice Advisor

This script serves as a single point of control for downloading,
updating, and managing all external data required by the application,
including 17lands stats and Scryfall card data.

Usage:
    python manage_data.py --status
    python manage_data.py --update-17lands
    python manage_data.py --update-17lands --all-sets
    python manage_data.py --update-scryfall
"""

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import requests
from rag_advisor import CardStatsDB, ALL_SETS, CURRENT_STANDARD
from scryfall_db import ScryfallDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CARD_DATA_API = "https://www.17lands.com/card_data"

def download_card_data_api(
    set_code: str,
    format_type: str = "PremierDraft",
    start_date: str = None
) -> List[Dict]:
    """
    Download card data from 17lands API endpoint (JSON).
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        'expansion': set_code,
        'format': format_type,
        'start': start_date,
    }

    logger.info(f"Downloading {set_code} ({ALL_SETS.get(set_code, set_code)}) via API...")
    try:
        response = requests.get(CARD_DATA_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logger.error(f"  ✗ Unexpected response format")
            return []

        cards = []
        for card in data:
            if card.get('seen_count', 0) < 100:
                continue
            cards.append({
                'card_name': card.get('name', 'Unknown'),
                'set_code': set_code,
                'games_played': card.get('game_count', 0),
                'win_rate': card.get('gp_wr', 0.0),
                'gih_win_rate': card.get('gih_wr', 0.0),
                'opening_hand_win_rate': card.get('oh_wr', 0.0),
                'drawn_win_rate': card.get('gd_wr', 0.0),
                'iwd': card.get('iwd', 0.0),
                'alsa': card.get('alsa', 0.0),
                'avg_taken_at': card.get('ata', 0.0),
                'seen_count': card.get('seen_count', 0),
                'pick_count': card.get('pick_count', 0),
                'color': card.get('color', ''),
                'rarity': card.get('rarity', ''),
                'last_updated': datetime.now().isoformat()
            })
        return cards
    except requests.exceptions.RequestException as e:
        logger.error(f"  ✗ API request failed: {e}")
        return []

def download_multiple_sets(
    set_codes: List[str],
    format_type: str = "PremierDraft"
) -> int:
    """
    Download card data for multiple sets via API.
    """
    db = CardStatsDB()
    total_cards = 0
    for i, set_code in enumerate(set_codes, 1):
        logger.info(f"\n[{i}/{len(set_codes)}] Processing {set_code}...")
        cards = download_card_data_api(set_code, format_type)
        if cards:
            db.delete_set_data(set_code)
            db.insert_card_stats(cards)
            total_cards += len(cards)
            logger.info(f"  ✓ Added {len(cards)} cards to database")
        else:
            logger.warning(f"  ⚠️  No data available for {set_code}")
        if i < len(set_codes):
            time.sleep(1)
    db.close()
    return total_cards

def show_status():
    """Display the status of all data sources."""
    logger.info("="*70)
    logger.info("DATA STATUS REPORT")
    logger.info("="*70)
    
    # 17lands status
    db_path = Path("data/card_stats.db")
    if not db_path.exists():
        logger.warning("17lands database not found.")
    else:
        show_17lands_database_status()

    # Scryfall status
    scryfall_db_path = Path("data/scryfall_cache.db")
    if not scryfall_db_path.exists():
        logger.warning("Scryfall database not found.")
    else:
        with sqlite3.connect(scryfall_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            logger.info(f"Scryfall DB contains {count} cards.")
    logger.info("="*70)

def check_database_sets() -> Dict[str, datetime]:
    """
    Check which sets are in the database and when they were last updated.
    """
    db_path = Path("data/card_stats.db")
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT set_code, MAX(last_updated) FROM card_stats GROUP BY set_code")
    
    results = {}
    for set_code, last_updated_str in cursor.fetchall():
        try:
            results[set_code] = datetime.fromisoformat(last_updated_str)
        except (ValueError, TypeError):
            results[set_code] = datetime(2020, 1, 1)
    conn.close()
    return results

def show_17lands_database_status():
    """Show current 17lands database status."""
    db_sets = check_database_sets()
    if not db_sets:
        logger.info("No 17lands data found.")
        return

    logger.info("\n--- 17lands Database Status ---")
    conn = sqlite3.connect("data/card_stats.db")
    cursor = conn.cursor()
    total_cards = 0
    for set_code in sorted(db_sets.keys()):
        cursor.execute("SELECT COUNT(*) FROM card_stats WHERE set_code = ?", (set_code,))
        card_count = cursor.fetchone()[0]
        age_days = (datetime.now() - db_sets[set_code]).days
        set_name = ALL_SETS.get(set_code, "Unknown Set")
        status = "⚠️" if age_days > 90 else "✅"
        logger.info(f"  {status} {set_code:6s} | {card_count:5,} cards | {age_days:3d} days old | {set_name}")
        total_cards += card_count
    conn.close()
    logger.info(f"Total: {len(db_sets)} sets, {total_cards:,} cards")

def update_17lands_data(all_sets: bool, max_age: int):
    """Update card statistics from 17lands."""
    logger.info("="*70)
    logger.info("UPDATING 17LANDS DATA")
    logger.info("="*70)

    db_sets = check_database_sets()
    cutoff_date = datetime.now() - timedelta(days=max_age)
    sets_to_check = list(ALL_SETS.keys()) if all_sets else CURRENT_STANDARD
    needs_update = [
        s for s in sets_to_check if s not in db_sets or db_sets[s] < cutoff_date
    ]

    if not needs_update:
        logger.info("All 17lands data is up to date.")
        return

    logger.info(f"Found {len(needs_update)} sets to update: {', '.join(needs_update)}")
    download_multiple_sets(needs_update)
    logger.info("17lands data update complete.")

def update_scryfall_data():
    """
    Pre-populate the Scryfall database with cards found in the 17lands database.
    This is useful for ensuring that all relevant cards are cached locally.
    """
    logger.info("="*70)
    logger.info("UPDATING SCRYFALL CACHE")
    logger.info("="*70)

    scryfall_db = ScryfallDB()
    lands_db_path = Path("data/card_stats.db")

    if not lands_db_path.exists():
        logger.error("17lands database not found. Cannot update Scryfall cache.")
        return

    with sqlite3.connect(lands_db_path) as conn:
        res = conn.execute("SELECT DISTINCT card_name FROM card_stats")
        all_card_names = [row[0] for row in res.fetchall()]

    logger.info(f"Found {len(all_card_names)} unique card names in 17lands DB.")
    logger.info("Updating Scryfall cache... This may take a while.")

    cached_count = 0
    for i, name in enumerate(all_card_names):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(all_card_names)} cards...")
        
        # This will fetch and cache the card if it's not already in the DB
        if scryfall_db.get_card_by_name(name):
            cached_count += 1
        time.sleep(0.1)  # To be kind to the Scryfall API

    logger.info(f"Scryfall cache update complete. Cached {cached_count} cards.")

def main():
    parser = argparse.ArgumentParser(
        description="Unified data management script for MTGA Voice Advisor."
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show the status of all data sources.'
    )
    parser.add_argument(
        '--update-17lands',
        action='store_true',
        help='Update 17lands data for Standard sets.'
    )
    parser.add_argument(
        '--all-sets',
        action='store_true',
        help='Use with --update-17lands to update all sets, not just Standard.'
    )
    parser.add_argument(
        '--max-age',
        type=int,
        default=90,
        help='Max age in days for 17lands data before it is considered stale.'
    )
    parser.add_argument(
        '--update-scryfall',
        action='store_true',
        help='Pre-populate the Scryfall cache with cards from the 17lands database.'
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.update_17lands:
        update_17lands_data(all_sets=args.all_sets, max_age=args.max_age)
    elif args.update_scryfall:
        update_scryfall_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()