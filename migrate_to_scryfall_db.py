import json
import logging
from scryfall_db import ScryfallDB

logger = logging.getLogger(__name__)

def migrate():
    """
    Migrates card data from the old card_cache.json to the new
    scryfall_cache.db SQLite database.
    """
    try:
        with open("card_cache.json") as f:
            cache = json.load(f)
    except FileNotFoundError:
        logger.info("card_cache.json not found. No migration needed.")
        return
    except json.JSONDecodeError:
        logger.error("Could not decode card_cache.json. Aborting migration.")
        return

    db = ScryfallDB("data/scryfall_cache.db")

    # The old cache has a 'set' key, but the new db expects 'set_code'
    # The review also mentioned this. I will handle it here.
    for grp_id, card_data in cache.items():
        if 'set' in card_data:
            card_data['set_code'] = card_data.pop('set')
        db.update_card(int(grp_id), card_data)

    logger.info(f"Migrated {len(cache)} cards to scryfall_cache.db")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    migrate()