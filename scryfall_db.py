import sqlite3
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScryfallDB:
    """
    Manages a local SQLite database for Scryfall card data.
    This replaces the need for card_cache.json.
    """
    def __init__(self, db_path: str = "data/scryfall_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Allow SQLite to be used from multiple threads (thread-safe with locks in ArenaCardDatabase)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        self.create_table()

    def create_table(self):
        """Create the cards table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    grpId INTEGER PRIMARY KEY,
                    name TEXT,
                    set_code TEXT,
                    rarity TEXT,
                    types TEXT,
                    subtypes TEXT,
                    power INTEGER,
                    toughness INTEGER,
                    oracle_text TEXT,
                    type_line TEXT,
                    mana_cost TEXT,
                    last_updated TEXT
                )
            """)

    def get_card_by_grpId(self, grpId: int) -> Optional[Dict]:
        """
        Retrieve a card from the database by its Arena grpId.
        Fetches from Scryfall if not found in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grpId,))
        row = cursor.fetchone()

        if row:
            # Convert row to a dictionary
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        else:
            # Card not in DB, fetch from Scryfall
            return self.fetch_and_cache_card(grpId=grpId)

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieve a card from the database by its name.
        Fetches from Scryfall if not found in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        else:
            return self.fetch_and_cache_card(name=name)

    def fetch_and_cache_card(self, grpId: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch card data from Scryfall API and cache it in the database.
        Can fetch by either grpId or card name.
        """
        if grpId:
            logger.info(f"Fetching card data for grpId: {grpId} from Scryfall...")
            url = f"https://api.scryfall.com/cards/arena/{grpId}"
        elif name:
            logger.info(f"Fetching card data for name: {name} from Scryfall...")
            url = f"https://api.scryfall.com/cards/named?exact={name}"
        else:
            return None

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            card_data = response.json()
            
            # Arena ID might not be in the response if we searched by name
            arena_id = card_data.get("arena_id", grpId)
            if not arena_id:
                logger.warning(f"No Arena ID found for card: {name}")
                return None

            # Extract power and toughness safely
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            
            # Convert to integer if possible
            try:
                power = int(power) if power else None
            except (ValueError, TypeError):
                power = None
            
            try:
                toughness = int(toughness) if toughness else None
            except (ValueError, TypeError):
                toughness = None

            card_info = {
                "grpId": arena_id,
                "name": card_data.get("name"),
                "set_code": card_data.get("set"),
                "rarity": card_data.get("rarity"),
                "types": ", ".join(card_data.get("type_line", "").split(" — ")[0].split()),
                "subtypes": ", ".join(card_data.get("type_line", "").split(" — ")[1].split()) if " — " in card_data.get("type_line", "") else "",
                "power": power,
                "toughness": toughness,
                "oracle_text": card_data.get("oracle_text"),
                "type_line": card_data.get("type_line"),
                "mana_cost": card_data.get("mana_cost"),
                "last_updated": datetime.now().isoformat()
            }

            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO cards (
                        grpId, name, set_code, rarity, types, subtypes,
                        power, toughness, oracle_text, type_line, mana_cost, last_updated
                    ) VALUES (
                        :grpId, :name, :set_code, :rarity, :types, :subtypes,
                        :power, :toughness, :oracle_text, :type_line, :mana_cost, :last_updated
                    )
                """, card_info)
            
            logger.info(f"Successfully cached card: {card_info['name']}")
            return card_info

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch card data for grpId {grpId}: {e}")
            return None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

if __name__ == '__main__':
    # Example usage and testing
    db = ScryfallDB()
    
    # Test fetching a card
    test_grpId = 69172  # Fervent Champion
    print(f"Fetching card with grpId: {test_grpId}")
    card = db.get_card_by_grpId(test_grpId)
    if card:
        print("Card found:")
        for key, value in card.items():
            print(f"  {key}: {value}")
    else:
        print("Card not found.")

    # Test fetching again (should be from cache)
    print("\nFetching the same card again (should be faster)...")
    card = db.get_card_by_grpId(test_grpId)
    if card:
        print("Card found in cache.")
    
    db.close()
