import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ScryfallDB:
    def __init__(self, db_path: str = "data/scryfall_cache.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    grpId INTEGER PRIMARY KEY,
                    name TEXT,
                    set_code TEXT,
                    rarity TEXT,
                    oracle_text TEXT,
                    type_line TEXT,
                    mana_cost TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name)")
            self.conn.commit()
            logger.info(f"Initialized Scryfall DB at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Scryfall DB: {e}")
            self.conn = None

    def update_card(self, grp_id: int, card_data: Dict):
        if not self.conn:
            return

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO cards (grpId, name, set_code, rarity, oracle_text, type_line, mana_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                grp_id,
                card_data.get('name'),
                card_data.get('set_code'),
                card_data.get('rarity'),
                card_data.get('oracle_text'),
                card_data.get('type_line'),
                card_data.get('mana_cost'),
            ))

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        if not self.conn:
            return None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return None

        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()