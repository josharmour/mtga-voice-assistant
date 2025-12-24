import argparse
import logging
from datetime import datetime
from typing import Dict, List

from src.data.data_management import CardStatsDB
from tools.test_17lands_api import SeventeenLandsAPIClient, CardRating

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_card_stats(set_code: str, format: str = "PremierDraft"):
    """
    Fetches card ratings from 17Lands API and populates the card_stats.db.
    """
    client = SeventeenLandsAPIClient()
    db = CardStatsDB()

    logger.info(f"Fetching card ratings for set '{set_code}' ({format})...")
    ratings: List[CardRating] = client.get_card_ratings(set_code, format)

    if not ratings:
        logger.warning(f"No ratings found for set '{set_code}' ({format}). Database not updated.")
        return

    stats_to_update = []
    for rating in ratings:
        # Map CardRating to the format expected by CardStatsDB.update_stats
        stats_to_update.append({
            "name": rating.name,
            "set_code": set_code.upper(), # Ensure set code is uppercase
            "win_rate": rating.win_rate,
            "gih_win_rate": rating.drawn_win_rate,
            "avg_taken_at": rating.avg_taken_at,
            "games_played": rating.game_count,
            "last_updated": datetime.now().isoformat()
        })
    
    logger.info(f"Updating database with {len(stats_to_update)} card stats for '{set_code}'...")
    db.update_stats(stats_to_update)
    logger.info(f"Successfully populated card_stats.db for set '{set_code}'.")
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate card_stats.db with 17Lands card ratings.")
    parser.add_argument("--set_code", required=True, help="The 3-letter set code (e.g., OTJ, MKM, TLA).")
    parser.add_argument("--format", default="PremierDraft", help="The draft format (e.g., PremierDraft, QuickDraft).")
    args = parser.parse_args()

    populate_card_stats(args.set_code, args.format)
