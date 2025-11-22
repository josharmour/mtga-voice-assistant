"""
ArenaCardDatabase - Local MTGA card database wrapper.
Maps Arena grpIds to card names and metadata.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ArenaCardDatabase:
    """
    Provides access to the local unified_cards.db database.
    This database maps Arena grpIds to card names and metadata.
    """
    def __init__(self, db_path: str = "data/unified_cards.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            logger.warning(f"Arena card database not found at {db_path}")
        self.conn = None
        if self.db_path.exists():
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

    def get_card_name(self, grp_id: int) -> str:
        """Get card name by Arena grpId."""
        if not self.conn:
            return f"Unknown Card {grp_id}"
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM cards WHERE grpId = ?", (grp_id,))
        row = cursor.fetchone()
        
        if row:
            return row["name"]
        return f"Unknown Card {grp_id}"

    def get_card_data(self, grp_id: int) -> Optional[Dict]:
        """Get full card data by Arena grpId."""
        if not self.conn:
            return None
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

    def close(self):
        if self.conn:
            self.conn.close()
