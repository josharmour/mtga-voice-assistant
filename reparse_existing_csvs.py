#!/usr/bin/env python3
"""
Re-parse existing downloaded CSVs with the fixed parser.

This script finds all downloaded CSVs and re-parses them
without re-downloading (saves hours of time).
"""

import logging
from pathlib import Path
from download_real_17lands_data import parse_card_stats_from_csv
from rag_advisor import CardStatsDB
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*70)
    logger.info("Re-parsing Existing CSV Files")
    logger.info("="*70)

    # Find all CSV files
    data_dir = Path("data")
    csv_files = list(data_dir.glob("17lands_*_*.csv"))

    if not csv_files:
        logger.error("No CSV files found in data/ directory")
        logger.error("Expected files like: data/17lands_OTJ_PremierDraft.csv")
        return

    logger.info(f"\nFound {len(csv_files)} CSV files to parse")

    # Clear existing database
    logger.info("\nClearing old database...")
    db_path = Path("data/card_stats.db")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM card_stats")
        conn.commit()
        conn.close()
        logger.info("✓ Cleared old data")

    # Parse each CSV
    db = CardStatsDB()
    total_cards = 0

    for i, csv_file in enumerate(sorted(csv_files), 1):
        logger.info(f"\n[{i}/{len(csv_files)}] Parsing {csv_file.name}...")

        try:
            cards = parse_card_stats_from_csv(str(csv_file), min_games=1000)

            if cards:
                db.insert_card_stats(cards)
                total_cards += len(cards)
                logger.info(f"  ✓ Added {len(cards)} cards to database")
            else:
                logger.warning(f"  ⚠️  No cards met minimum threshold (1000 games)")

        except Exception as e:
            logger.error(f"  ✗ Error parsing: {e}")

    db.close()

    logger.info("\n" + "="*70)
    logger.info("Re-parsing Complete!")
    logger.info("="*70)
    logger.info(f"Total cards imported: {total_cards:,}")
    logger.info(f"From {len(csv_files)} sets")
    logger.info("\nRun: python3 update_card_data.py --status")
    logger.info("="*70)


if __name__ == "__main__":
    main()
