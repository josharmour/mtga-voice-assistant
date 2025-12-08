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

    # Threshold for warning about unknown cards
    UNKNOWN_CARD_WARNING_THRESHOLD = 5

    # Maximum valid grpId for actual cards (ability objects have much higher IDs)
    # As of Dec 2025, the highest card grpId is around 102112
    # IDs above 110000 are typically abilities, tokens, or special game objects
    MAX_VALID_CARD_GRPID = 200000

    def __init__(self, db_path: str = "data/unified_cards.db"):
        self.db_path = Path(db_path)
        self._cache = {}  # In-memory cache: grpId -> dict
        self._unknown_cards = set()  # Track unique unknown grpIds
        self._unknown_card_callback = None  # Callback when threshold exceeded
        self._tracking_enabled = False  # Only track after caught up with logs

        if not self.db_path.exists():
            logger.warning(f"Arena card database not found at {db_path}. Run 'python tools/build_unified_card_database.py' to create it.")

        self._load_database()

    def set_unknown_card_callback(self, callback):
        """Set a callback function to be called when unknown card threshold is exceeded.

        The callback receives the count of unknown cards as an argument.
        """
        self._unknown_card_callback = callback

    @property
    def unknown_card_count(self) -> int:
        """Return the number of unique unknown cards encountered."""
        return len(self._unknown_cards)

    def get_unknown_cards(self) -> set:
        """Return the set of unknown grpIds."""
        return self._unknown_cards.copy()

    def clear_unknown_cards(self):
        """Clear the unknown cards tracking."""
        self._unknown_cards.clear()

    def enable_tracking(self):
        """Enable unknown card tracking. Call after caught up with logs."""
        self._tracking_enabled = True
        logger.info("Unknown card tracking enabled")

    def disable_tracking(self):
        """Disable unknown card tracking."""
        self._tracking_enabled = False

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

        # Track unknown card
        self._track_unknown_card(grp_id)
        return f"Unknown Card {grp_id}"

    def get_card_data(self, grp_id: int) -> Optional[Dict]:
        """Get full card data by Arena grpId."""
        if grp_id in self._cache:
            return self._cache[grp_id]

        # Track unknown card
        if grp_id and grp_id != 0:
            self._track_unknown_card(grp_id)

        return None

    def _track_unknown_card(self, grp_id: int):
        """Track an unknown card and trigger callback if threshold exceeded."""
        if not self._tracking_enabled:
            return  # Don't track during log catch-up

        # Don't track IDs that are too high - these are ability objects, not cards
        if grp_id > self.MAX_VALID_CARD_GRPID:
            logger.debug(f"Ignoring high grpId {grp_id} (likely ability/token object, not a card)")
            return

        if grp_id in self._unknown_cards:
            return  # Already tracked

        self._unknown_cards.add(grp_id)
        logger.warning(f"Unknown card grpId {grp_id} - total unknown: {len(self._unknown_cards)}")

        # Trigger callback if threshold exceeded
        if (self._unknown_card_callback and
            len(self._unknown_cards) >= self.UNKNOWN_CARD_WARNING_THRESHOLD):
            try:
                self._unknown_card_callback(len(self._unknown_cards))
            except Exception as e:
                logger.error(f"Error in unknown card callback: {e}")

        # Log to file for diagnostics
        self._log_unknown_card_to_file(grp_id)

    def _log_unknown_card_to_file(self, grp_id: int):
        """Append unknown card ID to a log file for analysis."""
        try:
            log_path = Path("logs/unknown_cards.log")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - Unknown grpId: {grp_id}\n")
        except Exception as e:
            logger.error(f"Failed to write to unknown cards log: {e}")

    def close(self):
        # No connection to close anymore
        pass
