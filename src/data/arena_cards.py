"""
ArenaCardDatabase - Local MTGA card database wrapper.
Maps Arena grpIds to card names and metadata.
"""
import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ArenaCardDatabase:
    """
    Provides access to the local unified_cards.db database.
    This database maps Arena grpIds to card names and metadata.

    To update this database when new sets release, run:
        python tools/build_unified_card_database.py
    """
    def __init__(self, db_path: str = "data/unified_cards.db"):
        self.db_path = Path(db_path)
        self._cache = {}  # In-memory cache: grpId -> dict

        if not self.db_path.exists():
            logger.warning(f"Arena card database not found at {db_path}. Run 'python tools/build_unified_card_database.py' to create it.")

        self._load_database()

    def _load_database(self):
        """Load the entire database into memory for fast lookups."""
        if not self.db_path.exists():
            return

        logger.info("Loading Arena card database into memory...")
        start_time = time.time()
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards")
            rows = cursor.fetchall()
            
            for row in rows:
                # Convert Row object to dict
                self._cache[row["grpId"]] = dict(row)
                
            conn.close()
            elapsed = time.time() - start_time
            logger.info(f"Loaded {len(self._cache)} cards in {elapsed:.4f}s")
        except Exception as e:
            logger.error(f"Failed to load card database: {e}")

    def get_card_name(self, grp_id: int) -> str:
        """Get card name by Arena grpId."""
        if grp_id == 0:
            return "Unknown Card 0"

        # Fast memory lookup
        if grp_id in self._cache:
            return self._cache[grp_id]["name"]
        
        # BLOCKING FALLBACK REMOVED: Rely on local DB to prevent performance issues.
        # if self.scryfall_client: ...
        
        logger.debug(f"Cache miss for grpId {grp_id}")
        return f"Unknown Card {grp_id}"

    def get_card_data(self, grp_id: int) -> Optional[Dict]:
        """Get full card data by Arena grpId."""
        if grp_id in self._cache:
            return self._cache[grp_id]
            
        # BLOCKING FALLBACK REMOVED
                
        return None

    def close(self):
        # No connection to close anymore
        pass
