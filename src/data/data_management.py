"""
Data management module for 17lands statistics.

Note: ScryfallClient was removed - use ArenaCardDatabase instead for card lookups.
"""
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CardStatsDB:
    """
    Manages 17lands statistics.
    Kept simple for now, can be expanded to fetch from 17lands API if needed.
    """
    def __init__(self, db_path: str = "data/card_stats.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread_local = threading.local()
        self._init_db()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.thread_local.conn.row_factory = sqlite3.Row
        return self.thread_local.conn

    def _init_db(self):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS card_stats (
                    card_name TEXT,
                    set_code TEXT,
                    win_rate REAL,
                    gih_win_rate REAL,
                    avg_taken_at REAL,
                    games_played INTEGER,
                    last_updated TEXT,
                    PRIMARY KEY (card_name, set_code)
                )
            """)

    def get_stats(self, card_name: str, set_code: str = None) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()

        if set_code:
            cursor.execute("SELECT * FROM card_stats WHERE card_name = ? AND set_code = ?", (card_name, set_code))
        else:
            # If no set code, get the one with most games played
            cursor.execute("SELECT * FROM card_stats WHERE card_name = ? ORDER BY games_played DESC LIMIT 1", (card_name,))

        row = cursor.fetchone()
        return dict(row) if row else None

    def update_stats(self, stats_list: List[Dict]):
        """Update stats from a list of dictionaries."""
        conn = self._get_conn()
        with conn:
            for stat in stats_list:
                conn.execute("""
                    INSERT OR REPLACE INTO card_stats (
                        card_name, set_code, win_rate, gih_win_rate,
                        avg_taken_at, games_played, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    stat.get("name"),
                    stat.get("set_code"),
                    stat.get("win_rate"),
                    stat.get("gih_win_rate"),
                    stat.get("avg_taken_at"),
                    stat.get("games_played"),
                    datetime.now().isoformat()
                ))

    def close(self):
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()
