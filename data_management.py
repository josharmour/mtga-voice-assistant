
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests
import argparse
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

# Content of scryfall_db.py
class ScryfallDB:
    """
    Manages a local SQLite database for Scryfall card data.
    This replaces the need for card_cache.json.
    """
    def __init__(self, db_path: str = "data/scryfall_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread_local = threading.local()
        self.create_table()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.thread_local.conn.row_factory = sqlite3.Row
        return self.thread_local.conn

    def create_table(self):
        """Create the cards table if it doesn't exist."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
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
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grpId,))
        row = cursor.fetchone()

        if row:
            # Convert row to a dictionary
            return dict(row)
        else:
            # Card not in DB, fetch from Scryfall
            return self.fetch_and_cache_card(grpId=grpId)

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieve a card from the database by its name.
        Fetches from Scryfall if not found in the database.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        else:
            return self.fetch_and_cache_card(name=name)

    def fetch_and_cache_card(self, grpId: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch card data from Scryfall API and cache it in the database.
        Can fetch by either grpId or card name.
        """
        if grpId:
            logging.info(f"Fetching card data for grpId: {grpId} from Scryfall...")
            url = f"https://api.scryfall.com/cards/arena/{grpId}"
        elif name:
            logging.info(f"Fetching card data for name: {name} from Scryfall...")
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
                logging.warning(f"No Arena ID found for card: {name}")
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

            conn = self._get_conn()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cards (
                        grpId, name, set_code, rarity, types, subtypes,
                        power, toughness, oracle_text, type_line, mana_cost, last_updated
                    ) VALUES (
                        :grpId, :name, :set_code, :rarity, :types, :subtypes,
                        :power, :toughness, :oracle_text, :type_line, :mana_cost, :last_updated
                    )
                """, card_info)
            
            logging.info(f"Successfully cached card: {card_info['name']}")
            return card_info

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch card data for grpId {grpId}: {e}")
            return None

    def close(self):
        """Close the database connection."""
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()


# Thread-safe CardStatsDB for 17lands statistics
class CardStatsDB:
    """
    Manages an SQLite database for storing and retrieving 17lands card statistics.
    Thread-safe implementation using thread-local storage.
    """

    def __init__(self, db_path: str = "data/card_stats.db"):
        """Initialize the CardStatsDB with thread-safe connections."""
        self.db_path = Path(db_path)
        self.thread_local = threading.local()
        self._initialize_db()

    def _get_conn(self):
        """Get thread-local database connection."""
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.thread_local.conn.row_factory = sqlite3.Row
        return self.thread_local.conn

    def _initialize_db(self):
        """Create database and tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS card_stats (
                    card_name TEXT PRIMARY KEY,
                    set_code TEXT,
                    color TEXT,
                    rarity TEXT,
                    games_played INTEGER,
                    win_rate REAL,
                    avg_taken_at REAL,
                    games_in_hand INTEGER,
                    gih_win_rate REAL,
                    opening_hand_win_rate REAL,
                    drawn_win_rate REAL,
                    ever_drawn_win_rate REAL,
                    never_drawn_win_rate REAL,
                    alsa REAL,
                    ata REAL,
                    iwd REAL,
                    format TEXT,
                    last_updated TEXT
                )
            """)
            conn.commit()
            logging.info(f"Initialized card stats database: {self.db_path}")

        except Exception as e:
            logging.error(f"Failed to initialize card stats DB: {e}")

    def insert_card_stats(self, stats: List[Dict]):
        """Insert or update card statistics."""
        if not stats:
            return

        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            for stat in stats:
                cursor.execute("""
                    INSERT OR REPLACE INTO card_stats VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    stat.get('card_name'),
                    stat.get('set_code'),
                    stat.get('color', ''),
                    stat.get('rarity', ''),
                    stat.get('games_played', 0),
                    stat.get('win_rate', 0.0),
                    stat.get('avg_taken_at', 0.0),
                    stat.get('games_in_hand', 0),
                    stat.get('gih_win_rate', 0.0),
                    stat.get('opening_hand_win_rate', 0.0),
                    stat.get('drawn_win_rate', 0.0),
                    stat.get('ever_drawn_win_rate', 0.0),
                    stat.get('never_drawn_win_rate', 0.0),
                    stat.get('alsa', 0.0),
                    stat.get('ata', 0.0),
                    stat.get('iwd', 0.0),
                    stat.get('format', 'PremierDraft'),
                    stat.get('last_updated', datetime.now().isoformat())
                ))

            conn.commit()
            logging.info(f"Inserted/updated {len(stats)} card statistics")

        except Exception as e:
            logging.error(f"Error inserting card stats: {e}")

    def get_card_stats(self, card_name: str) -> Optional[Dict]:
        """Get statistics for a specific card."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM card_stats WHERE card_name = ?", (card_name,))
            row = cursor.fetchone()

            if not row:
                return None

            return dict(row)

        except Exception as e:
            logging.error(f"Error fetching card stats: {e}")
            return None

    def delete_set_data(self, set_code: str):
        """Delete all statistics for a given set."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM card_stats WHERE set_code = ?", (set_code,))
            conn.commit()
            logging.info(f"Deleted old data for set: {set_code}")

        except Exception as e:
            logging.error(f"Error deleting set data: {e}")

    def search_by_name(self, pattern: str, limit: int = 10) -> List[Dict]:
        """Search for cards by name pattern."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM card_stats
                WHERE card_name LIKE ?
                ORDER BY games_played DESC
                LIMIT ?
            """, (pattern, limit))

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logging.error(f"Error searching cards: {e}")
            return []

    def close(self):
        """Close database connection."""
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()


# Content of card_rag.py
@dataclass
class CardInfo:
    """Complete card information for LLM context."""
    grpId: int
    name: str
    oracle_text: str
    mana_cost: str
    cmc: float
    type_line: str
    color_identity: str
    power: str
    toughness: str
    rarity: str
    set_code: str
    win_rate: Optional[float] = None
    gih_win_rate: Optional[float] = None
    avg_pick_position: Optional[float] = None
    games_played: Optional[int] = None

    def to_rag_citation(self, include_stats: bool = True) -> str:
        """
        Format card information as a grounded RAG citation.

        This provides the LLM with factual, verifiable information with sources.

        Args:
            include_stats: Include win rate statistics

        Returns:
            Formatted card information with citations
        """
        lines = []

        # Card name with ID (for verification)
        lines.append(f"**{self.name}** (grpId: {self.grpId}) [{self.set_code}]")

        # Mana cost and type
        lines.append(f"Cost: {self.mana_cost} | Type: {self.type_line}")

        # Power/toughness if creature
        if self.power or self.toughness:
            lines.append(f"Stats: {self.power}/{self.toughness}")

        # Color identity
        if self.color_identity:
            lines.append(f"Colors: {self.color_identity}")

        # Oracle text (card abilities) - with citation
        if self.oracle_text:
            lines.append(f"Abilities: {self.oracle_text}")
            lines.append(f"*[Source: Arena Card Database]*")

        # Win rate statistics - with citation
        if include_stats and self.win_rate is not None:
            lines.append("")
            lines.append("**Performance Statistics:**")
            if self.games_played and self.games_played >= 1000:
                lines.append(f"- Win Rate: {self.win_rate*100:.1f}% ({self.games_played} games)")
                lines.append(f"- GIH Win Rate: {self.gih_win_rate*100:.1f}%")
                if self.avg_pick_position:
                    lines.append(f"- Average Pick Position: {self.avg_pick_position:.1f}")
                lines.append(f"*[Source: 17lands.com]*")
            else:
                lines.append(f"- Limited data available ({self.games_played} games)")

        lines.append("")  # Empty line for separation
        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """
        Format card for inclusion in LLM prompt.

        Concise format for use in board state context.

        Returns:
            Formatted card context
        """
        parts = []

        # Name and cost
        if self.mana_cost:
            parts.append(f"{self.name} ({self.mana_cost})")
        else:
            parts.append(self.name)

        # Type
        if self.type_line:
            parts.append(f"[{self.type_line}]")

        # Power/toughness
        if self.power or self.toughness:
            parts.append(f"{self.power}/{self.toughness}")

        # Abilities summary
        if self.oracle_text:
            # Truncate long abilities
            abilities = self.oracle_text[:100]
            if len(self.oracle_text) > 100:
                abilities += "..."
            parts.append(f"Abilities: {abilities}")

        return " ".join(parts)


class CardRagDatabase:
    """
    Unified card database with RAG capabilities.

    Provides grounded card information to the LLM, combining:
    - Card metadata from Arena (names, abilities, costs)
    - Statistics from 17lands (win rates, pick positions)
    - All information with citations to prevent hallucinations
    """

    def __init__(
        self,
        unified_db: str = "data/unified_cards.db",
        stats_db: str = "data/card_stats.db"
    ):
        """
        Initialize the Card RAG database.

        Args:
            unified_db: Path to unified_cards.db (Arena data)
            stats_db: Path to card_stats.db (17lands data)
        """
        self.unified_db_path = Path(unified_db)
        self.stats_db_path = Path(stats_db)
        self.thread_local = threading.local()
        self._initialize()

    def _get_unified_conn(self):
        if not hasattr(self.thread_local, "unified_conn"):
            self.thread_local.unified_conn = sqlite3.connect(str(self.unified_db_path))
            self.thread_local.unified_conn.row_factory = sqlite3.Row
        return self.thread_local.unified_conn

    def _get_stats_conn(self):
        if not hasattr(self.thread_local, "stats_conn"):
            self.thread_local.stats_conn = sqlite3.connect(str(self.stats_db_path))
            self.thread_local.stats_conn.row_factory = sqlite3.Row
        return self.thread_local.stats_conn

    def _initialize(self):
        """Initialize database connections."""
        try:
            if self.unified_db_path.exists():
                logging.info(f"Connected to {self.unified_db_path}")
            else:
                logging.warning(f"Unified database not found: {self.unified_db_path}")

            if self.stats_db_path.exists():
                logging.info(f"Connected to {self.stats_db_path}")
            else:
                logging.debug(f"Stats database not found: {self.stats_db_path}")

        except Exception as e:
            logging.error(f"Failed to initialize databases: {e}")

    def get_card_by_grpid(self, grp_id: int, format_type: str = "PremierDraft") -> Optional[CardInfo]:
        """
        Get complete card information by grpId with statistics.

        Args:
            grp_id: Arena graphics ID
            format_type: Draft format for statistics (e.g., "PremierDraft")

        Returns:
            CardInfo object with all data or None
        """
        try:
            unified_conn = self._get_unified_conn()
            cursor = unified_conn.cursor()
            cursor.execute("""
                SELECT
                    grpId, name, oracle_text, mana_cost, cmc, type_line,
                    color_identity, power, toughness, rarity, set_code
                FROM cards
                WHERE grpId = ?
            """, (grp_id,))

            row = cursor.fetchone()
            if not row:
                logging.warning(f"Card not found: grpId {grp_id}")
                return None

            card_info = CardInfo(
                grpId=row['grpId'],
                name=row['name'],
                oracle_text=row['oracle_text'] or "",
                mana_cost=row['mana_cost'] or "",
                cmc=row['cmc'] or 0.0,
                type_line=row['type_line'] or "",
                color_identity=row['color_identity'] or "",
                power=row['power'] or "",
                toughness=row['toughness'] or "",
                rarity=row['rarity'] or "",
                set_code=row['set_code'] or ""
            )

            # Get statistics if available
            stats_conn = self._get_stats_conn()
            if stats_conn:
                card_info = self._add_statistics(card_info, format_type)

            return card_info

        except Exception as e:
            logging.error(f"Error fetching card {grp_id}: {e}")
            return None

    def get_card_by_name(self, card_name: str, set_code: str = None) -> Optional[CardInfo]:
        """
        Get card information by name and optional set.

        Args:
            card_name: Card name to search for
            set_code: Optional set code to narrow search

        Returns:
            CardInfo object or None
        """
        try:
            unified_conn = self._get_unified_conn()
            cursor = unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE name = ? AND set_code = ?
                    LIMIT 1
                """, (card_name, set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE name = ?
                    LIMIT 1
                """, (card_name,))

            row = cursor.fetchone()
            if not row:
                return None

            card_info = CardInfo(
                grpId=row['grpId'],
                name=row['name'],
                oracle_text=row['oracle_text'] or "",
                mana_cost=row['mana_cost'] or "",
                cmc=row['cmc'] or 0.0,
                type_line=row['type_line'] or "",
                color_identity=row['color_identity'] or "",
                power=row['power'] or "",
                toughness=row['toughness'] or "",
                rarity=row['rarity'] or "",
                set_code=row['set_code'] or ""
            )

            stats_conn = self._get_stats_conn()
            if stats_conn:
                card_info = self._add_statistics(card_info)

            return card_info

        except Exception as e:
            logging.error(f"Error fetching card {card_name}: {e}")
            return None

    def _add_statistics(self, card_info: CardInfo, format_type: str = "PremierDraft") -> CardInfo:
        """
        Add 17lands statistics to card information.

        Args:
            card_info: Base card information
            format_type: Draft format for stats

        Returns:
            CardInfo with statistics added
        """
        try:
            stats_conn = self._get_stats_conn()
            cursor = stats_conn.cursor()
            cursor.execute("""
                SELECT
                    win_rate, gih_win_rate, avg_taken_at, games_played
                FROM card_stats
                WHERE card_name = ? AND set_code = ? AND format = ?
                LIMIT 1
            """, (card_info.name, card_info.set_code, format_type))

            row = cursor.fetchone()
            if row:
                card_info.win_rate = row['win_rate']
                card_info.gih_win_rate = row['gih_win_rate']
                card_info.avg_pick_position = row['avg_taken_at']
                card_info.games_played = row['games_played']

        except Exception as e:
            logging.debug(f"Could not fetch stats for {card_info.name}: {e}")

        return card_info

    def get_board_state_context(
        self,
        card_grp_ids: List[int],
        include_stats: bool = True,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Generate complete board state context for LLM.

        Creates a formatted context of all cards with abilities, costs, and stats.

        Args:
            card_grp_ids: List of card grpIds on board
            include_stats: Include win rate statistics
            format_type: Draft format for statistics

        Returns:
            Formatted board state context with citations
        """
        try:
            self._get_unified_conn()  # Verify connection available
        except Exception:
            return ""

        context_lines = []
        context_lines.append("=" * 70)
        context_lines.append("BOARD STATE WITH CARD INFORMATION")
        context_lines.append("=" * 70)
        context_lines.append("")

        for grp_id in card_grp_ids:
            card_info = self.get_card_by_grpid(grp_id, format_type)
            if card_info:
                context_lines.append(card_info.to_rag_citation(include_stats))

        context_lines.append("=" * 70)
        context_lines.append("[All information sourced from Arena database and 17lands.com]")
        context_lines.append("=" * 70)

        return "\n".join(context_lines)

    def search_cards_by_type(self, type_keyword: str, set_code: str = None) -> List[CardInfo]:
        """
        Search for cards by type (e.g., "creature", "instant").

        Useful for finding relevant cards when building prompts.

        Args:
            type_keyword: Type to search for (case-insensitive)
            set_code: Optional set code to limit search

        Returns:
            List of matching cards
        """
        try:
            unified_conn = self._get_unified_conn()
            cursor = unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE type_line LIKE ? AND set_code = ?
                    LIMIT 20
                """, (f"%{type_keyword}%", set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE type_line LIKE ?
                    LIMIT 20
                """, (f"%{type_keyword}%",))

            results = []
            for row in cursor.fetchall():
                card_info = CardInfo(
                    grpId=row['grpId'],
                    name=row['name'],
                    oracle_text=row['oracle_text'] or "",
                    mana_cost=row['mana_cost'] or "",
                    cmc=row['cmc'] or 0.0,
                    type_line=row['type_line'] or "",
                    color_identity=row['color_identity'] or "",
                    power=row['power'] or "",
                    toughness=row['toughness'] or "",
                    rarity=row['rarity'] or "",
                    set_code=row['set_code'] or ""
                )
                results.append(card_info)

            return results

        except Exception as e:
            logging.error(f"Error searching for {type_keyword}: {e}")
            return []

    def search_by_ability(self, ability_keyword: str, set_code: str = None) -> List[CardInfo]:
        """
        Search for cards by ability text (e.g., "draw a card").

        Args:
            ability_keyword: Ability keyword to search for
            set_code: Optional set code

        Returns:
            List of matching cards
        """
        try:
            unified_conn = self._get_unified_conn()
            cursor = unified_conn.cursor()

            if set_code:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE oracle_text LIKE ? AND set_code = ?
                    LIMIT 20
                """, (f"%{ability_keyword}%", set_code))
            else:
                cursor.execute("""
                    SELECT
                        grpId, name, oracle_text, mana_cost, cmc, type_line,
                        color_identity, power, toughness, rarity, set_code
                    FROM cards
                    WHERE oracle_text LIKE ?
                    LIMIT 20
                """, (f"%{ability_keyword}%",))

            results = []
            for row in cursor.fetchall():
                card_info = CardInfo(
                    grpId=row['grpId'],
                    name=row['name'],
                    oracle_text=row['oracle_text'] or "",
                    mana_cost=row['mana_cost'] or "",
                    cmc=row['cmc'] or 0.0,
                    type_line=row['type_line'] or "",
                    color_identity=row['color_identity'] or "",
                    power=row['power'] or "",
                    toughness=row['toughness'] or "",
                    rarity=row['rarity'] or "",
                    set_code=row['set_code'] or ""
                )
                results.append(card_info)

            return results

        except Exception as e:
            logging.error(f"Error searching by ability: {e}")
            return []

    def close(self):
        """Close database connections."""
        if hasattr(self.thread_local, "unified_conn"):
            self.thread_local.unified_conn.close()
        if hasattr(self.thread_local, "stats_conn"):
            self.thread_local.stats_conn.close()

# Content of src/database.py
class ArenaCardDatabase:
    """
    Unified card database using unified_cards.db (17,000+ cards).

    No API fallbacks needed - all MTGA cards are in the database.
    """
    def __init__(self, db_path: str = "data/unified_cards.db"):
        self.db_path = Path(db_path)
        self.thread_local = threading.local()
        self.cache: Dict[int, dict] = {}

        if not self.db_path.exists():
            logging.error(f"✗ Card database not found at {db_path}")
            logging.error("  Run: python build_unified_card_database.py")
            return

        try:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row

            # Get stats
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cards")
            total = cursor.fetchone()[0]

            logging.info(f"✓ Unified card database loaded ({total:,} cards)")

        except Exception as e:
            logging.error(f"Failed to load card database: {e}")

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Set row factory so dict(row) works properly
            self.thread_local.conn.row_factory = sqlite3.Row
        return self.thread_local.conn

    @property
    def conn(self):
        """Property for backward compatibility - returns thread-local connection."""
        return self._get_conn()

    def get_card_name(self, grp_id: int) -> str:
        """Get card name from grpId."""
        # Validate input
        if not grp_id or grp_id == 0:
            logging.warning(f"get_card_name called with invalid grpId: {grp_id}")
            return f"Unknown({grp_id})"

        # Check cache
        if grp_id in self.cache:
            card = self.cache[grp_id]
            name = card.get("printed_name") or card.get("name", f"Unknown({grp_id})")
            return name

        # Query database
        try:
            conn = self._get_conn()
            if not conn:
                logging.error(f"No database connection available for grpId {grp_id}")
                return f"Unknown({grp_id})"

            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
            row = cursor.fetchone()

            if row:
                card = dict(row)
                self.cache[grp_id] = card
                name = card.get("printed_name") or card.get("name", f"Unknown({grp_id})")
                return name
            else:
                logging.debug(f"Card {grp_id} not found in database")
                return f"Unknown({grp_id})"

        except Exception as e:
            logging.error(f"Database error for grpId {grp_id}: {e}")
            return f"Unknown({grp_id})"

    def get_card_data(self, grp_id: int) -> Optional[dict]:
        """Get full card data from grpId."""
        # Check cache
        if grp_id in self.cache:
            return self.cache[grp_id]

        # Query database
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
            row = cursor.fetchone()

            if row:
                card = dict(row)
                self.cache[grp_id] = card
                return card

            return None

        except Exception as e:
            logging.error(f"Database error for grpId {grp_id}: {e}")
            return None

    def get_oracle_text(self, grp_id: int) -> str:
        """Get oracle text for a card."""
        card = self.get_card_data(grp_id)
        return card.get("oracle_text", "") if card else ""

    def get_mana_cost(self, grp_id: int) -> str:
        """Get mana cost for a card (e.g., '{2}{U}{U}')."""
        card = self.get_card_data(grp_id)
        return card.get("mana_cost", "") if card else ""

    def get_cmc(self, grp_id: int) -> Optional[float]:
        """Get converted mana cost for a card."""
        card = self.get_card_data(grp_id)
        cmc = card.get("cmc") if card else None
        return float(cmc) if cmc is not None else None

    def get_type_line(self, grp_id: int) -> str:
        """Get card type line (e.g., 'Creature - Elf Wizard')."""
        card = self.get_card_data(grp_id)
        return card.get("type_line", "") if card else ""

# Content of manage_data.py
CARD_DATA_API = "https://www.17lands.com/card_data"

def download_card_data_api(
    set_code: str,
    format_type: str = "PremierDraft",
    start_date: str = None
) -> List[Dict]:
    """
    Download card data from 17lands API endpoint (JSON).
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        'expansion': set_code,
        'format': format_type,
        'start': start_date,
    }

    logging.info(f"Downloading {set_code} via API...")
    try:
        response = requests.get(CARD_DATA_API, params=params, timeout=30)
        response.raise_for_status()
        
        if not response.text:
            logging.warning(f"  ✗ API returned empty response for {set_code}")
            return []
            
        data = response.json()
        if not isinstance(data, list):
            logging.error(f"  ✗ Unexpected response format")
            return []

        cards = []
        for card in data:
            if card.get('seen_count', 0) < 100:
                continue
            cards.append({
                'card_name': card.get('name', 'Unknown'),
                'set_code': set_code,
                'games_played': card.get('game_count', 0),
                'win_rate': card.get('gp_wr', 0.0),
                'gih_win_rate': card.get('gih_wr', 0.0),
                'opening_hand_win_rate': card.get('oh_wr', 0.0),
                'drawn_win_rate': card.get('gd_wr', 0.0),
                'iwd': card.get('iwd', 0.0),
                'alsa': card.get('alsa', 0.0),
                'avg_taken_at': card.get('ata', 0.0),
                'seen_count': card.get('seen_count', 0),
                'pick_count': card.get('pick_count', 0),
                'color': card.get('color', ''),
                'rarity': card.get('rarity', ''),
                'last_updated': datetime.now().isoformat()
            })
        return cards
    except requests.exceptions.RequestException as e:
        logging.error(f"  ✗ API request failed: {e}")
        return []

def download_multiple_sets(
    set_codes: List[str],
    format_type: str = "PremierDraft"
) -> int:
    """
    Download card data for multiple sets via API.
    """
    db = CardStatsDB()
    total_cards = 0
    for i, set_code in enumerate(set_codes, 1):
        logging.info(f"\n[{i}/{len(set_codes)}] Processing {set_code}...")
        cards = download_card_data_api(set_code, format_type)
        if cards:
            db.delete_set_data(set_code)
            db.insert_card_stats(cards)
            total_cards += len(cards)
            logging.info(f"  ✓ Added {len(cards)} cards to database")
        else:
            logging.warning(f"  ⚠️  No data available for {set_code}")
        if i < len(set_codes):
            time.sleep(1)
    db.close()
    return total_cards

def show_status():
    """Display the status of all data sources."""
    logging.info("="*70)
    logging.info("DATA STATUS REPORT")
    logging.info("="*70)
    
    # 17lands status
    db_path = Path("data/card_stats.db")
    if not db_path.exists():
        logging.warning("17lands database not found.")
    else:
        show_17lands_database_status()

    # Scryfall status
    scryfall_db_path = Path("data/scryfall_cache.db")
    if not scryfall_db_path.exists():
        logging.warning("Scryfall database not found.")
    else:
        with sqlite3.connect(scryfall_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            logging.info(f"Scryfall DB contains {count} cards.")
    logging.info("="*70)

def check_database_sets() -> Dict[str, datetime]:
    """
    Check which sets are in the database and when they were last updated.
    """
    db_path = Path("data/card_stats.db")
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT set_code, MAX(last_updated) FROM card_stats GROUP BY set_code")
    
    results = {}
    for set_code, last_updated_str in cursor.fetchall():
        try:
            results[set_code] = datetime.fromisoformat(last_updated_str)
        except (ValueError, TypeError):
            results[set_code] = datetime(2020, 1, 1)
    conn.close()
    return results

def show_17lands_database_status():
    """Show current 17lands database status."""
    db_sets = check_database_sets()
    if not db_sets:
        logging.info("No 17lands data found.")
        return

    logging.info("\n--- 17lands Database Status ---")
    conn = sqlite3.connect("data/card_stats.db")
    cursor = conn.cursor()
    total_cards = 0
    for set_code in sorted(db_sets.keys()):
        cursor.execute("SELECT COUNT(*) FROM card_stats WHERE set_code = ?", (set_code,))
        card_count = cursor.fetchone()[0]
        age_days = (datetime.now() - db_sets[set_code]).days
        status = "⚠️" if age_days > 90 else "✅"
        logging.info(f"  {status} {set_code:6s} | {card_count:5,} cards | {age_days:3d} days old")
        total_cards += card_count
    conn.close()
    logging.info(f"Total: {len(db_sets)} sets, {total_cards:,} cards")

def update_17lands_data(all_sets: bool, max_age: int):
    """Update card statistics from 17lands."""
    logging.info("="*70)
    logging.info("UPDATING 17LANDS DATA")
    logging.info("="*70)

    db_sets = check_database_sets()
    cutoff_date = datetime.now() - timedelta(days=max_age)
    sets_to_check = ALL_SETS if all_sets else CURRENT_STANDARD
    needs_update = [
        s for s in sets_to_check if s not in db_sets or db_sets[s] < cutoff_date
    ]

    if not needs_update:
        logging.info("All 17lands data is up to date.")
        return

    logging.info(f"Found {len(needs_update)} sets to update: {', '.join(needs_update)}")
    download_multiple_sets(needs_update)
    logging.info("17lands data update complete.")

def update_scryfall_data():
    """
    Pre-populate the Scryfall database with cards found in the 17lands database.
    This is useful for ensuring that all relevant cards are cached locally.
    """
    logging.info("="*70)
    logging.info("UPDATING SCRYFALL CACHE")
    logging.info("="*70)

    scryfall_db = ScryfallDB()
    lands_db_path = Path("data/card_stats.db")

    if not lands_db_path.exists():
        logging.error("17lands database not found. Cannot update Scryfall cache.")
        return

    with sqlite3.connect(lands_db_path) as conn:
        res = conn.execute("SELECT DISTINCT card_name FROM card_stats")
        all_card_names = [row[0] for row in res.fetchall()]

    logging.info(f"Found {len(all_card_names)} unique card names in 17lands DB.")
    logging.info("Updating Scryfall cache... This may take a while.")

    cached_count = 0
    for i, name in enumerate(all_card_names):
        if i % 100 == 0:
            logging.info(f"Processed {i}/{len(all_card_names)} cards...")
        
        # This will fetch and cache the card if it's not already in the DB
        if scryfall_db.get_card_by_name(name):
            cached_count += 1
        time.sleep(0.1)  # To be kind to the Scryfall API

    logging.info(f"Scryfall cache update complete. Cached {cached_count} cards.")


def check_and_update_card_database() -> bool:
    """
    Check if the unified card database exists and is initialized.
    Used by app.py to ensure the database is available before starting.

    Returns:
        True if database exists and is accessible, False otherwise
    """
    db_path = Path("data/unified_cards.db")

    if not db_path.exists():
        logging.error(f"Card database not found at {db_path}")
        logging.error("Please run: python3 tools/build_unified_card_database.py")
        return False

    try:
        # Verify database is accessible
        db = ArenaCardDatabase(str(db_path))
        # Check if we can query it
        _ = db.get_card_name(1)  # Query a card
        logging.info("Card database verified and ready")
        return True

    except Exception as e:
        logging.error(f"Failed to verify card database: {e}")
        return False

if __name__ == "__main__":
    from constants import ALL_SETS, CURRENT_STANDARD
    from tools.download_17lands_data import download_17lands_data

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Manage MTGA Advisor data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status command
    parser_status = subparsers.add_parser("status", help="Show data status")

    # 17lands command
    parser_17lands = subparsers.add_parser("update-17lands", help="Update 17lands data")
    parser_17lands.add_argument("--all-sets", action="store_true", help="Download all sets, not just current standard")
    parser_17lands.add_argument("--max-age", type=int, default=30, help="Max age of data in days before re-downloading")

    # Scryfall command
    parser_scryfall = subparsers.add_parser("update-scryfall", help="Update Scryfall cache")

    # Download command
    parser_download = subparsers.add_parser("download", help="Download 17lands data")
    parser_download.add_argument("--set-codes", type=str, nargs='+', required=True, help="One or more MTG set codes (e.g., 'MKM' 'LCI')")
    parser_download.add_argument("--draft-type", type=str, default="PremierDraft", help="Draft type (e.g., 'PremierDraft')")
    parser_download.add_argument("--data-type", type=str, default="replay_data", help="Data type to download (e.g., 'draft_data', 'game_data', 'replay_data')")
    parser_download.add_argument("--output-dir", type=Path, default=Path("data/17lands"), help="Directory to save the data")

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "update-17lands":
        update_17lands_data(args.all_sets, args.max_age)
    elif args.command == "update-scryfall":
        update_scryfall_data()
    elif args.command == "download":
        for set_code in args.set_codes:
            download_17lands_data(set_code, args.draft_type, args.data_type, args.output_dir)
