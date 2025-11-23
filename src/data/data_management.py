import logging
import sqlite3
import threading
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class ScryfallClient:
    """
    Manages card data fetching from Scryfall API with SQLite caching.
    Replaces the need for a massive local card database.
    """
    def __init__(self, db_path: str = "data/scryfall_cache.db", arena_db=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread_local = threading.local()
        self.arena_db = arena_db  # Fallback for Arena-exclusive cards
        self._init_db()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.thread_local.conn.row_factory = sqlite3.Row
        return self.thread_local.conn

    def _init_db(self):
        """Initialize the cache database."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    id TEXT PRIMARY KEY,  -- Scryfall ID or Arena grpId (as string)
                    name TEXT,
                    set_code TEXT,
                    collector_number TEXT,
                    rarity TEXT,
                    mana_cost TEXT,
                    type_line TEXT,
                    oracle_text TEXT,
                    power TEXT,
                    toughness TEXT,
                    colors TEXT,
                    image_uri TEXT,
                    arena_id INTEGER,
                    last_updated TEXT
                )
            """)
            # Index for name searches
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON cards(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_id ON cards(arena_id)")

    def get_card_by_arena_id(self, arena_id: int) -> Optional[Dict]:
        """
        Get card data by Arena grpId.
        Checks cache first, then API.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE arena_id = ?", (arena_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)

        # Fetch from API
        return self._fetch_from_scryfall(arena_id=arena_id)

    def get_card_name(self, arena_id: int) -> str:
        """
        Compatibility method for GameStateManager.
        Returns card name or empty string if not found.
        """
        card_data = self.get_card_by_arena_id(arena_id)
        return card_data.get("name", "") if card_data else f"Unknown Card {arena_id}"

    def get_card_data(self, arena_id: int) -> Optional[Dict]:
        """
        Compatibility method for GameStateManager.
        Alias for get_card_by_arena_id.
        """
        return self.get_card_by_arena_id(arena_id)

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """
        Get card data by exact name.
        Checks cache first, then API.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row:
            return dict(row)

        # Fetch from API
        return self._fetch_from_scryfall(name=name)

    def _fetch_from_scryfall(self, arena_id: int = None, name: str = None) -> Optional[Dict]:
        """Fetch from Scryfall API and cache."""
        time.sleep(0.1)  # Rate limiting (100ms per request recommended by Scryfall)
        
        try:
            if arena_id:
                url = f"https://api.scryfall.com/cards/arena/{arena_id}"
                logger.info(f"Fetching from Scryfall (Arena ID: {arena_id})...")
            elif name:
                url = f"https://api.scryfall.com/cards/named?exact={name}"
                logger.info(f"Fetching from Scryfall (Name: {name})...")
            else:
                return None

            response = requests.get(url, timeout=10)
            
            if response.status_code == 404:
                # Suppress warning for common not founds to avoid log spam
                # logger.warning(f"Card not found on Scryfall: {arena_id or name}")
                return None
                
            response.raise_for_status()
            data = response.json()

            # Parse data
            card_data = {
                "id": data.get("id"),
                "name": data.get("name"),
                "set_code": data.get("set"),
                "collector_number": data.get("collector_number"),
                "rarity": data.get("rarity"),
                "mana_cost": data.get("mana_cost"),
                "type_line": data.get("type_line"),
                "oracle_text": data.get("oracle_text"),
                "power": data.get("power"),
                "toughness": data.get("toughness"),
                "colors": "".join(data.get("colors", [])),
                "image_uri": data.get("image_uris", {}).get("normal"),
                "arena_id": data.get("arena_id", arena_id), # Use provided arena_id if not in response
                "last_updated": datetime.now().isoformat()
            }

            # Cache it
            self._cache_card(card_data)
            return card_data

        except Exception as e:
            logger.error(f"Error fetching from Scryfall: {e}")
            return None

    def _cache_card(self, card_data: Dict):
        """Save card data to cache."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO cards (
                    id, name, set_code, collector_number, rarity, mana_cost,
                    type_line, oracle_text, power, toughness, colors,
                    image_uri, arena_id, last_updated
                ) VALUES (
                    :id, :name, :set_code, :collector_number, :rarity, :mana_cost,
                    :type_line, :oracle_text, :power, :toughness, :colors,
                    :image_uri, :arena_id, :last_updated
                )
            """, card_data)

    def close(self):
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()


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
