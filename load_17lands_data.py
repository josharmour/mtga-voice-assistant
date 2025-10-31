"""
17lands Data Loader

This script downloads and processes card performance data from 17lands.com
and populates the SQLite database for the RAG system.

17lands Data Sources:
- Public API: https://www.17lands.com/card_data (requires API key)
- Public CSVs: https://www.17lands.com/public_datasets

Data Metrics:
- GIH WR (Games In Hand Win Rate): Win rate when card is drawn
- IWD (Impact When Drawn): GIH WR - Overall deck WR
- ALSA (Average Last Seen At): How late the card is picked in draft
- ATA (Average Taken At): When the card is typically drafted

For this implementation, we'll use sample/mock data as the full dataset
is very large and requires API authentication.
"""

import requests
import json
import csv
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime
import argparse

from rag_advisor import CardStatsDB

logger = logging.getLogger(__name__)

# 17lands API endpoint (requires authentication)
LANDS_API_URL = "https://www.17lands.com/api/card_ratings"

# Sample datasets (most recent standard sets)
SAMPLE_SETS = ["ONE", "MOM", "WOE", "LCI", "MKM"]


def download_17lands_data(set_code: str, format: str = "PremierDraft") -> List[Dict]:
    """
    Download card data from 17lands for a specific set.

    Note: This is a placeholder. Real implementation would need:
    1. API authentication
    2. Rate limiting
    3. Proper error handling

    Args:
        set_code: Set code (e.g., "ONE" for Phyrexia: All Will Be One)
        format: Game format (PremierDraft, QuickDraft, Sealed)

    Returns:
        List of card data dictionaries
    """
    logger.info(f"Attempting to download data for {set_code} ({format})...")

    # In a real implementation, this would make an authenticated API request:
    # headers = {"Authorization": f"Bearer {API_KEY}"}
    # response = requests.get(f"{LANDS_API_URL}?set={set_code}&format={format}", headers=headers)

    # For now, return empty list (would be populated by real API)
    logger.warning(f"Real API call not implemented. Use load_sample_data() instead.")
    return []


def load_sample_data() -> List[Dict]:
    """
    Generate comprehensive sample card data for testing.

    This includes popular cards across multiple sets with realistic statistics.
    """
    sample_cards = [
        # Removal Spells
        {
            'card_name': 'Lightning Bolt',
            'set_code': 'M21',
            'color': 'R',
            'rarity': 'common',
            'games_played': 52341,
            'win_rate': 0.583,
            'avg_taken_at': 4.2,
            'games_in_hand': 32156,
            'gih_win_rate': 0.621,
            'opening_hand_win_rate': 0.605,
            'drawn_win_rate': 0.628,
            'ever_drawn_win_rate': 0.621,
            'never_drawn_win_rate': 0.545,
            'alsa': 8.3,
            'ata': 4.2,
            'iwd': 0.038,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Murder',
            'set_code': 'M21',
            'color': 'B',
            'rarity': 'common',
            'games_played': 48923,
            'win_rate': 0.571,
            'avg_taken_at': 3.8,
            'games_in_hand': 29834,
            'gih_win_rate': 0.607,
            'opening_hand_win_rate': 0.592,
            'drawn_win_rate': 0.615,
            'ever_drawn_win_rate': 0.607,
            'never_drawn_win_rate': 0.538,
            'alsa': 7.9,
            'ata': 3.8,
            'iwd': 0.036,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Pacifism',
            'set_code': 'M21',
            'color': 'W',
            'rarity': 'common',
            'games_played': 43567,
            'win_rate': 0.567,
            'avg_taken_at': 4.5,
            'games_in_hand': 27891,
            'gih_win_rate': 0.598,
            'opening_hand_win_rate': 0.581,
            'drawn_win_rate': 0.605,
            'ever_drawn_win_rate': 0.598,
            'never_drawn_win_rate': 0.541,
            'alsa': 8.7,
            'ata': 4.5,
            'iwd': 0.031,
            'last_updated': datetime.now().isoformat()
        },

        # Counterspells
        {
            'card_name': 'Counterspell',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 45678,
            'win_rate': 0.589,
            'avg_taken_at': 3.2,
            'games_in_hand': 28945,
            'gih_win_rate': 0.615,
            'opening_hand_win_rate': 0.598,
            'drawn_win_rate': 0.622,
            'ever_drawn_win_rate': 0.615,
            'never_drawn_win_rate': 0.562,
            'alsa': 6.8,
            'ata': 3.2,
            'iwd': 0.026,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Negate',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 38912,
            'win_rate': 0.554,
            'avg_taken_at': 6.1,
            'games_in_hand': 24567,
            'gih_win_rate': 0.573,
            'opening_hand_win_rate': 0.562,
            'drawn_win_rate': 0.578,
            'ever_drawn_win_rate': 0.573,
            'never_drawn_win_rate': 0.541,
            'alsa': 9.8,
            'ata': 6.1,
            'iwd': 0.019,
            'last_updated': datetime.now().isoformat()
        },

        # Card Draw
        {
            'card_name': 'Divination',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 51234,
            'win_rate': 0.576,
            'avg_taken_at': 5.3,
            'games_in_hand': 31456,
            'gih_win_rate': 0.602,
            'opening_hand_win_rate': 0.587,
            'drawn_win_rate': 0.609,
            'ever_drawn_win_rate': 0.602,
            'never_drawn_win_rate': 0.551,
            'alsa': 9.2,
            'ata': 5.3,
            'iwd': 0.026,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Night\'s Whisper',
            'set_code': 'M21',
            'color': 'B',
            'rarity': 'common',
            'games_played': 47892,
            'win_rate': 0.581,
            'avg_taken_at': 4.8,
            'games_in_hand': 29123,
            'gih_win_rate': 0.608,
            'opening_hand_win_rate': 0.594,
            'drawn_win_rate': 0.614,
            'ever_drawn_win_rate': 0.608,
            'never_drawn_win_rate': 0.556,
            'alsa': 8.9,
            'ata': 4.8,
            'iwd': 0.027,
            'last_updated': datetime.now().isoformat()
        },

        # Creatures - Early Game
        {
            'card_name': 'Llanowar Elves',
            'set_code': 'M21',
            'color': 'G',
            'rarity': 'common',
            'games_played': 58934,
            'win_rate': 0.612,
            'avg_taken_at': 2.1,
            'games_in_hand': 38567,
            'gih_win_rate': 0.682,
            'opening_hand_win_rate': 0.721,
            'drawn_win_rate': 0.658,
            'ever_drawn_win_rate': 0.682,
            'never_drawn_win_rate': 0.541,
            'alsa': 4.2,
            'ata': 2.1,
            'iwd': 0.091,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Savannah Lions',
            'set_code': 'M21',
            'color': 'W',
            'rarity': 'common',
            'games_played': 44567,
            'win_rate': 0.558,
            'avg_taken_at': 5.7,
            'games_in_hand': 28934,
            'gih_win_rate': 0.589,
            'opening_hand_win_rate': 0.612,
            'drawn_win_rate': 0.574,
            'ever_drawn_win_rate': 0.589,
            'never_drawn_win_rate': 0.531,
            'alsa': 9.4,
            'ata': 5.7,
            'iwd': 0.031,
            'last_updated': datetime.now().isoformat()
        },

        # Creatures - Mid Game
        {
            'card_name': 'Cloudkin Seer',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 49123,
            'win_rate': 0.574,
            'avg_taken_at': 4.9,
            'games_in_hand': 30456,
            'gih_win_rate': 0.601,
            'opening_hand_win_rate': 0.572,
            'drawn_win_rate': 0.614,
            'ever_drawn_win_rate': 0.601,
            'never_drawn_win_rate': 0.548,
            'alsa': 8.6,
            'ata': 4.9,
            'iwd': 0.027,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Ravenous Chupacabra',
            'set_code': 'M21',
            'color': 'B',
            'rarity': 'uncommon',
            'games_played': 52678,
            'win_rate': 0.621,
            'avg_taken_at': 2.8,
            'games_in_hand': 34123,
            'gih_win_rate': 0.658,
            'opening_hand_win_rate': 0.632,
            'drawn_win_rate': 0.671,
            'ever_drawn_win_rate': 0.658,
            'never_drawn_win_rate': 0.584,
            'alsa': 5.9,
            'ata': 2.8,
            'iwd': 0.037,
            'last_updated': datetime.now().isoformat()
        },

        # Creatures - Late Game
        {
            'card_name': 'Colossal Dreadmaw',
            'set_code': 'M21',
            'color': 'G',
            'rarity': 'common',
            'games_played': 41234,
            'win_rate': 0.528,
            'avg_taken_at': 8.2,
            'games_in_hand': 26789,
            'gih_win_rate': 0.548,
            'opening_hand_win_rate': 0.492,
            'drawn_win_rate': 0.573,
            'ever_drawn_win_rate': 0.548,
            'never_drawn_win_rate': 0.511,
            'alsa': 11.3,
            'ata': 8.2,
            'iwd': 0.020,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Serra Angel',
            'set_code': 'M21',
            'color': 'W',
            'rarity': 'uncommon',
            'games_played': 46789,
            'win_rate': 0.591,
            'avg_taken_at': 3.9,
            'games_in_hand': 29456,
            'gih_win_rate': 0.624,
            'opening_hand_win_rate': 0.582,
            'drawn_win_rate': 0.641,
            'ever_drawn_win_rate': 0.624,
            'never_drawn_win_rate': 0.559,
            'alsa': 7.1,
            'ata': 3.9,
            'iwd': 0.033,
            'last_updated': datetime.now().isoformat()
        },

        # Ramp
        {
            'card_name': 'Rampant Growth',
            'set_code': 'M21',
            'color': 'G',
            'rarity': 'common',
            'games_played': 53421,
            'win_rate': 0.594,
            'avg_taken_at': 4.1,
            'games_in_hand': 33567,
            'gih_win_rate': 0.631,
            'opening_hand_win_rate': 0.658,
            'drawn_win_rate': 0.612,
            'ever_drawn_win_rate': 0.631,
            'never_drawn_win_rate': 0.558,
            'alsa': 7.8,
            'ata': 4.1,
            'iwd': 0.037,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Cultivate',
            'set_code': 'M21',
            'color': 'G',
            'rarity': 'common',
            'games_played': 51789,
            'win_rate': 0.587,
            'avg_taken_at': 4.6,
            'games_in_hand': 32456,
            'gih_win_rate': 0.618,
            'opening_hand_win_rate': 0.641,
            'drawn_win_rate': 0.604,
            'ever_drawn_win_rate': 0.618,
            'never_drawn_win_rate': 0.557,
            'alsa': 8.4,
            'ata': 4.6,
            'iwd': 0.031,
            'last_updated': datetime.now().isoformat()
        },

        # Planeswalkers (higher rarity, fewer games)
        {
            'card_name': 'Jace, the Mind Sculptor',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'mythic',
            'games_played': 8934,
            'win_rate': 0.687,
            'avg_taken_at': 1.2,
            'games_in_hand': 6123,
            'gih_win_rate': 0.734,
            'opening_hand_win_rate': 0.712,
            'drawn_win_rate': 0.746,
            'ever_drawn_win_rate': 0.734,
            'never_drawn_win_rate': 0.641,
            'alsa': 2.1,
            'ata': 1.2,
            'iwd': 0.093,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Liliana of the Veil',
            'set_code': 'M21',
            'color': 'B',
            'rarity': 'mythic',
            'games_played': 9123,
            'win_rate': 0.672,
            'avg_taken_at': 1.4,
            'games_in_hand': 6234,
            'gih_win_rate': 0.718,
            'opening_hand_win_rate': 0.701,
            'drawn_win_rate': 0.728,
            'ever_drawn_win_rate': 0.718,
            'never_drawn_win_rate': 0.628,
            'alsa': 2.4,
            'ata': 1.4,
            'iwd': 0.090,
            'last_updated': datetime.now().isoformat()
        },

        # Bombs (powerful uncommons/rares)
        {
            'card_name': 'Baneslayer Angel',
            'set_code': 'M21',
            'color': 'W',
            'rarity': 'rare',
            'games_played': 15678,
            'win_rate': 0.643,
            'avg_taken_at': 1.8,
            'games_in_hand': 10234,
            'gih_win_rate': 0.691,
            'opening_hand_win_rate': 0.658,
            'drawn_win_rate': 0.708,
            'ever_drawn_win_rate': 0.691,
            'never_drawn_win_rate': 0.597,
            'alsa': 3.2,
            'ata': 1.8,
            'iwd': 0.048,
            'last_updated': datetime.now().isoformat()
        },
        {
            'card_name': 'Sheoldred, the Apocalypse',
            'set_code': 'DMU',
            'color': 'B',
            'rarity': 'mythic',
            'games_played': 12456,
            'win_rate': 0.698,
            'avg_taken_at': 1.1,
            'games_in_hand': 8567,
            'gih_win_rate': 0.751,
            'opening_hand_win_rate': 0.729,
            'drawn_win_rate': 0.762,
            'ever_drawn_win_rate': 0.751,
            'never_drawn_win_rate': 0.647,
            'alsa': 1.8,
            'ata': 1.1,
            'iwd': 0.104,
            'last_updated': datetime.now().isoformat()
        },

        # Weak cards (for comparison)
        {
            'card_name': 'Cancel',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 34567,
            'win_rate': 0.492,
            'avg_taken_at': 9.8,
            'games_in_hand': 21234,
            'gih_win_rate': 0.503,
            'opening_hand_win_rate': 0.498,
            'drawn_win_rate': 0.506,
            'ever_drawn_win_rate': 0.503,
            'never_drawn_win_rate': 0.481,
            'alsa': 12.7,
            'ata': 9.8,
            'iwd': 0.011,
            'last_updated': datetime.now().isoformat()
        },
    ]

    logger.info(f"Generated {len(sample_cards)} sample cards")
    return sample_cards


def export_to_csv(cards: List[Dict], output_path: str = "data/card_stats.csv"):
    """
    Export card data to CSV format (compatible with 17lands exports).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'card_name', 'set_code', 'color', 'rarity', 'games_played',
        'win_rate', 'avg_taken_at', 'games_in_hand', 'gih_win_rate',
        'opening_hand_win_rate', 'drawn_win_rate', 'ever_drawn_win_rate',
        'never_drawn_win_rate', 'alsa', 'ata', 'iwd', 'last_updated'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cards)

    logger.info(f"Exported {len(cards)} cards to {output_path}")


def main(db_path: str):
    """Main function to load card data into the database."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting 17lands data loader...")

    # Initialize database
    db = CardStatsDB(db_path=db_path)

    # Load sample data
    logger.info("Loading sample card data...")
    sample_cards = load_sample_data()

    # Insert into database
    logger.info("Inserting cards into database...")
    db.insert_card_stats(sample_cards)

    # Export to CSV for reference
    logger.info("Exporting to CSV...")
    export_to_csv(sample_cards)

    # Verify data
    logger.info("\nVerifying data insertion:")
    test_cards = ['Lightning Bolt', 'Llanowar Elves', 'Jace, the Mind Sculptor']

    for card_name in test_cards:
        stats = db.get_card_stats(card_name)
        if stats:
            logger.info(
                f"{card_name}: "
                f"WR={stats['win_rate']:.1%}, "
                f"GIH WR={stats['gih_win_rate']:.1%}, "
                f"IWD={stats['iwd']:+.1%}, "
                f"Games={stats['games_played']}"
            )
        else:
            logger.warning(f"{card_name}: Not found in database")

    db.close()
    logger.info("\nData loading complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load 17lands data into a SQLite database.")
    parser.add_argument(
        "--db-path",
        default="data/card_stats.db",
        help="Path to the SQLite database file."
    )
    args = parser.parse_args()
    main(args.db_path)
