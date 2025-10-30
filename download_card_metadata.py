#!/usr/bin/env python3
"""
Download Card Metadata from 17lands

These CSV files contain card attributes (colors, types, mana cost, rarity)
which are essential for the RAG system to provide intelligent advice.

Files downloaded:
- cards.csv: All card attributes (18,000+ cards)
- abilities.csv: Keyword abilities list

These complement the existing:
- card_stats.db: Win rates and performance data
- chromadb/: MTG comprehensive rules
"""

import requests
import csv
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CARDS_CSV_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv"
ABILITIES_CSV_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/cards/abilities.csv"


def download_cards_metadata() -> List[Dict]:
    """
    Download cards.csv with card attributes.

    Returns:
        List of card dictionaries with attributes
    """
    logger.info("Downloading cards.csv (18,000+ cards, ~5 MB)...")

    try:
        response = requests.get(CARDS_CSV_URL, timeout=60)
        response.raise_for_status()

        # Parse CSV
        cards = []
        reader = csv.DictReader(response.text.splitlines())

        for row in reader:
            cards.append({
                'card_id': row['id'],
                'expansion': row['expansion'],
                'name': row['name'],
                'rarity': row['rarity'],
                'color_identity': row['color_identity'],
                'mana_value': int(row['mana_value']) if row['mana_value'] else 0,
                'types': row['types'],
                'is_booster': row['is_booster'] == 'True'
            })

        logger.info(f"✓ Downloaded {len(cards):,} cards")
        return cards

    except Exception as e:
        logger.error(f"Failed to download cards.csv: {e}")
        return []


def download_abilities() -> List[Dict]:
    """
    Download abilities.csv with keyword abilities.

    Returns:
        List of ability dictionaries
    """
    logger.info("Downloading abilities.csv...")

    try:
        response = requests.get(ABILITIES_CSV_URL, timeout=30)
        response.raise_for_status()

        # Parse CSV
        abilities = []
        reader = csv.DictReader(response.text.splitlines())

        for row in reader:
            abilities.append({
                'ability_id': row['id'],
                'text': row['text']
            })

        logger.info(f"✓ Downloaded {len(abilities)} abilities")
        return abilities

    except Exception as e:
        logger.error(f"Failed to download abilities.csv: {e}")
        return []


def create_metadata_tables():
    """Create card_metadata and abilities tables in database."""
    db_path = Path("data/card_metadata.db")
    db_path.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Card metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS card_metadata (
            card_id INTEGER PRIMARY KEY,
            expansion TEXT,
            name TEXT,
            rarity TEXT,
            color_identity TEXT,
            mana_value INTEGER,
            types TEXT,
            is_booster BOOLEAN,
            UNIQUE(card_id)
        )
    """)

    # Abilities table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS abilities (
            ability_id INTEGER PRIMARY KEY,
            text TEXT,
            UNIQUE(ability_id)
        )
    """)

    # Index for fast card name lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_card_name
        ON card_metadata(name)
    """)

    # Index for expansion lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_expansion
        ON card_metadata(expansion)
    """)

    conn.commit()
    conn.close()

    logger.info("✓ Created metadata tables")


def insert_cards(cards: List[Dict]):
    """Insert card metadata into database."""
    db_path = Path("data/card_metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM card_metadata")

    # Insert cards
    for card in cards:
        cursor.execute("""
            INSERT INTO card_metadata
            (card_id, expansion, name, rarity, color_identity, mana_value, types, is_booster)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card['card_id'],
            card['expansion'],
            card['name'],
            card['rarity'],
            card['color_identity'],
            card['mana_value'],
            card['types'],
            card['is_booster']
        ))

    conn.commit()
    conn.close()

    logger.info(f"✓ Inserted {len(cards):,} cards into database")


def insert_abilities(abilities: List[Dict]):
    """Insert abilities into database."""
    db_path = Path("data/card_metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM abilities")

    # Insert abilities
    for ability in abilities:
        cursor.execute("""
            INSERT INTO abilities (ability_id, text)
            VALUES (?, ?)
        """, (ability['ability_id'], ability['text']))

    conn.commit()
    conn.close()

    logger.info(f"✓ Inserted {len(abilities)} abilities into database")


def show_examples():
    """Show example queries of the metadata."""
    db_path = Path("data/card_metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    logger.info("\n" + "="*70)
    logger.info("Example Queries")
    logger.info("="*70)

    # Example 1: Find a specific card
    cursor.execute("SELECT * FROM card_metadata WHERE name = 'Lightning Bolt' LIMIT 1")
    row = cursor.fetchone()
    if row:
        logger.info("\nExample 1: Lightning Bolt")
        logger.info(f"  Mana value: {row[5]}")
        logger.info(f"  Color: {row[4]}")
        logger.info(f"  Types: {row[6]}")
        logger.info(f"  Rarity: {row[3]}")

    # Example 2: All 2-mana blue creatures
    cursor.execute("""
        SELECT name, types, expansion
        FROM card_metadata
        WHERE mana_value = 2
        AND color_identity = 'U'
        AND types LIKE '%Creature%'
        LIMIT 5
    """)
    logger.info("\nExample 2: 2-mana blue creatures (first 5):")
    for row in cursor.fetchall():
        logger.info(f"  {row[0]} ({row[2]}) - {row[1]}")

    # Example 3: Count by rarity
    cursor.execute("""
        SELECT rarity, COUNT(*)
        FROM card_metadata
        GROUP BY rarity
    """)
    logger.info("\nExample 3: Cards by rarity:")
    for row in cursor.fetchall():
        logger.info(f"  {row[0]}: {row[1]:,} cards")

    conn.close()


def main():
    logger.info("="*70)
    logger.info("17lands Card Metadata Downloader")
    logger.info("="*70)
    logger.info("\nThis downloads card attributes (colors, types, mana cost, etc.)")
    logger.info("to enhance the RAG system's card knowledge.\n")

    # Create tables
    create_metadata_tables()

    # Download cards
    cards = download_cards_metadata()
    if cards:
        insert_cards(cards)

    # Download abilities
    abilities = download_abilities()
    if abilities:
        insert_abilities(abilities)

    if cards or abilities:
        logger.info("\n" + "="*70)
        logger.info("Success!")
        logger.info("="*70)
        logger.info(f"Database: data/card_metadata.db")
        logger.info(f"  Cards: {len(cards):,}")
        logger.info(f"  Abilities: {len(abilities)}")
        logger.info("\nThe RAG system can now access:")
        logger.info("  - Card colors and color identity")
        logger.info("  - Mana costs (CMC)")
        logger.info("  - Card types (Creature, Instant, etc.)")
        logger.info("  - Rarity")
        logger.info("  - Keyword abilities")

        # Show example queries
        show_examples()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
