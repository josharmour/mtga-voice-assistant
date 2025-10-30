#!/usr/bin/env python3
"""
Intelligent Card Data Updater

Checks which sets need updating and downloads only what's needed.
Perfect for cron jobs or periodic maintenance.

Usage:
    python3 update_card_data.py               # Interactive mode
    python3 update_card_data.py --auto        # Auto mode (no prompts)
    python3 update_card_data.py --max-age 30  # Update sets older than 30 days
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set
import sqlite3

# Import from download script
from download_real_17lands_data import (
    ALL_SETS,
    CURRENT_STANDARD,
    download_card_data,
    parse_card_stats_from_csv,
)
from rag_advisor import CardStatsDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_sets() -> Dict[str, datetime]:
    """
    Check which sets are in the database and when they were last updated.

    Returns:
        Dict mapping set_code -> last_updated datetime
    """
    db_path = Path("data/card_stats.db")

    if not db_path.exists():
        logger.warning("Database not found - will create new one")
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get unique sets and their most recent update time
    cursor.execute("""
        SELECT set_code, MAX(last_updated)
        FROM card_stats
        GROUP BY set_code
    """)

    results = {}
    for set_code, last_updated_str in cursor.fetchall():
        try:
            # Parse ISO format datetime
            last_updated = datetime.fromisoformat(last_updated_str)
            results[set_code] = last_updated
        except (ValueError, TypeError):
            # If parsing fails, assume very old
            results[set_code] = datetime(2020, 1, 1)

    conn.close()
    return results


def check_needed_updates(max_age_days: int = 90, only_standard: bool = True) -> List[str]:
    """
    Determine which sets need updating.

    Args:
        max_age_days: Update sets older than this many days
        only_standard: Only check Standard-legal sets

    Returns:
        List of set codes that need updating
    """
    db_sets = check_database_sets()
    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    sets_to_check = CURRENT_STANDARD if only_standard else list(ALL_SETS.keys())
    needs_update = []

    logger.info(f"\nChecking {len(sets_to_check)} sets...")
    logger.info(f"Update threshold: {max_age_days} days (older than {cutoff_date.date()})")

    for set_code in sets_to_check:
        if set_code not in db_sets:
            logger.info(f"  ❌ {set_code}: Missing from database")
            needs_update.append(set_code)
        elif db_sets[set_code] < cutoff_date:
            age_days = (datetime.now() - db_sets[set_code]).days
            logger.info(f"  ⚠️  {set_code}: Outdated ({age_days} days old)")
            needs_update.append(set_code)
        else:
            age_days = (datetime.now() - db_sets[set_code]).days
            logger.info(f"  ✅ {set_code}: Up to date ({age_days} days old)")

    return needs_update


def update_sets(set_codes: List[str], interactive: bool = True) -> int:
    """
    Download and update specified sets.

    Args:
        set_codes: List of set codes to update
        interactive: If False, auto-confirm all prompts

    Returns:
        Number of sets successfully updated
    """
    if not set_codes:
        logger.info("\n✅ All sets are up to date!")
        return 0

    logger.info(f"\n📥 Need to update {len(set_codes)} set(s): {', '.join(set_codes)}")

    if interactive:
        confirm = input(f"\nDownload {len(set_codes)} set(s)? This may take a while. (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Update cancelled.")
            return 0

    db = CardStatsDB()
    updated_count = 0
    total_cards = 0

    for i, set_code in enumerate(set_codes, 1):
        logger.info(f"\n[{i}/{len(set_codes)}] Updating {set_code} ({ALL_SETS.get(set_code, 'Unknown')})")

        try:
            csv_path = download_card_data(set_code, "PremierDraft")

            if csv_path:
                # Delete old data for this set
                logger.info(f"  Removing old {set_code} data...")
                conn = sqlite3.connect("data/card_stats.db")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM card_stats WHERE set_code = ?", (set_code,))
                conn.commit()
                conn.close()

                # Insert new data
                cards = parse_card_stats_from_csv(csv_path, min_games=1000)
                if cards:
                    db.insert_card_stats(cards)
                    total_cards += len(cards)
                    updated_count += 1
                    logger.info(f"  ✅ {set_code}: Added {len(cards)} cards")
            else:
                logger.warning(f"  ⚠️  {set_code}: Download failed (may not have public data)")

        except Exception as e:
            logger.error(f"  ❌ {set_code}: Error - {e}")

        # Small delay between sets
        if i < len(set_codes):
            import time
            time.sleep(2)

    db.close()

    logger.info(f"\n{'='*70}")
    logger.info(f"Update Complete!")
    logger.info(f"  Sets updated: {updated_count}/{len(set_codes)}")
    logger.info(f"  Total cards: {total_cards:,}")
    logger.info(f"{'='*70}")

    return updated_count


def show_database_status():
    """Show current database status - sets, card counts, ages."""
    db_sets = check_database_sets()

    if not db_sets:
        logger.info("\n❌ No database found or empty database")
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"Current Database Status")
    logger.info(f"{'='*70}")

    # Get card counts per set
    conn = sqlite3.connect("data/card_stats.db")
    cursor = conn.cursor()

    total_cards = 0
    for set_code in sorted(db_sets.keys()):
        cursor.execute("SELECT COUNT(*) FROM card_stats WHERE set_code = ?", (set_code,))
        card_count = cursor.fetchone()[0]

        age_days = (datetime.now() - db_sets[set_code]).days
        set_name = ALL_SETS.get(set_code, "Unknown Set")

        # Status indicator
        if age_days > 90:
            status = "⚠️"
        else:
            status = "✅"

        logger.info(f"  {status} {set_code:6s} | {card_count:5,} cards | {age_days:3d} days old | {set_name}")
        total_cards += card_count

    conn.close()

    logger.info(f"{'='*70}")
    logger.info(f"Total: {len(db_sets)} sets, {total_cards:,} cards")
    logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent card data updater for MTGA Voice Advisor"
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto mode - no prompts, just update what\'s needed'
    )
    parser.add_argument(
        '--max-age',
        type=int,
        default=90,
        help='Update sets older than this many days (default: 90)'
    )
    parser.add_argument(
        '--all-sets',
        action='store_true',
        help='Check all sets, not just Standard'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show database status and exit'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("MTGA Card Data Updater")
    logger.info("="*70)

    # Show status mode
    if args.status:
        show_database_status()
        return

    # Check what needs updating
    needs_update = check_needed_updates(
        max_age_days=args.max_age,
        only_standard=not args.all_sets
    )

    # Update sets
    if needs_update:
        update_sets(needs_update, interactive=not args.auto)
    else:
        logger.info("\n✅ All sets are up to date!")
        logger.info("\nRun with --status to see database details")


if __name__ == "__main__":
    main()
