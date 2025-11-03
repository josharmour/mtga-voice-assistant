import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

def check_and_update_card_database() -> bool:
    """
    Check if card database needs update and offer to update it.

    This now uses the AutoUpdater for intelligent updates:
    - Detects Arena database changes
    - Downloads 17lands data for new sets
    - Updates based on what format you're playing

    Returns True if database is ready to use, False if update failed.
    """
    from datetime import datetime, timedelta
    from pathlib import Path
    import sqlite3
    import sys
    import os

    # Add parent directory to path to import auto_updater
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from auto_updater import AutoUpdater
    except ImportError:
        # Fallback to old behavior if auto_updater isn't available
        logging.warning("Auto-updater not found, using legacy update check")
        return _legacy_check_and_update()

    db_path = Path("data/unified_cards.db")

    # Use auto-updater for intelligent updates
    updater = AutoUpdater(auto_mode=True)  # Auto mode for seamless startup

    # If database doesn't exist, build it
    if not db_path.exists():
        print("\n" + "="*70)
        print("INITIAL SETUP REQUIRED")
        print("="*70)
        print("Setting up card databases for first use.")
        print("This will:")
        print("  1. Extract card data from Arena installation")
        print("  2. Download current win rates from 17lands")
        print("  3. Create optimized local databases")
        print()
        print("Expected time: 2-5 minutes")
        print()

        response = input("Continue with setup? (y/n): ").strip().lower()
        if response != 'y':
            print("Cannot proceed without card database. Exiting.")
            return False

        # Build the database
        if not updater.rebuild_unified_database():
            print("✗ Database build failed.")
            return False

        print("✓ Card database built successfully!")

        # Also get 17lands data for current sets
        print("\nDownloading performance statistics...")
        updater.update_17lands_data()

        return True

    # Database exists - check for updates (daily)
    last_check_file = Path("data/.last_update_check")
    should_check = True

    if last_check_file.exists():
        try:
            with open(last_check_file, 'r') as f:
                last_check = datetime.fromisoformat(f.read().strip())
                hours_since = (datetime.now() - last_check).total_seconds() / 3600
                if hours_since < 24:
                    should_check = False
                    logging.info(f"Skipping update check (last check {hours_since:.1f} hours ago)")
        except Exception:
            pass

    if should_check:
        # Run update check
        updated = updater.check_and_update_all()

        # Save check timestamp
        last_check_file.parent.mkdir(parents=True, exist_ok=True)
        with open(last_check_file, 'w') as f:
            f.write(datetime.now().isoformat())

        if updated:
            print("✅ Databases updated successfully!")

    return True


def _legacy_check_and_update() -> bool:
    """Legacy update check for backward compatibility."""
    from datetime import datetime
    from pathlib import Path
    import subprocess

    db_path = Path("data/unified_cards.db")

    if not db_path.exists():
        print("\nCard database not found. Building...")
        result = subprocess.run(["python3", "build_unified_card_database.py"],
                              capture_output=False)
        return result.returncode == 0

    return True


class ArenaCardDatabase:
    """
    Unified card database using unified_cards.db (17,000+ cards with reskin support).

    No API fallbacks needed - all MTGA cards are in the database.
    """
    def __init__(self, db_path: str = "data/unified_cards.db", show_reskin_names: bool = False):
        self.db_path = Path(db_path)
        self.conn = None
        self.cache: Dict[int, dict] = {}
        self.show_reskin_names = show_reskin_names  # Toggle for Spider-Man reskins

        if not self.db_path.exists():
            logging.error(f"✗ Card database not found at {db_path}")
            logging.error("  Run: python build_unified_card_database.py")
            return

        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Get stats
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cards")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM cards WHERE is_reskin = 1")
            reskins = cursor.fetchone()[0]

            logging.info(f"✓ Unified card database loaded ({total:,} cards, {reskins} reskins)")

        except Exception as e:
            logging.error(f"Failed to load card database: {e}")
            self.conn = None

    def get_card_name(self, grp_id: int) -> str:
        """Get card name from grpId. Shows reskin names if show_reskin_names is True."""
        if not grp_id or not self.conn:
            return f"Unknown({grp_id})" if grp_id else "Unknown"

        # Check cache
        if grp_id in self.cache:
            card = self.cache[grp_id]
            if self.show_reskin_names and card.get("is_reskin"):
                # Show Spider-Man reskin name
                return card.get("name", f"Unknown({grp_id})")
            else:
                # Show OmenPaths canonical name (printed name)
                return card.get("printed_name") or card.get("name", f"Unknown({grp_id})")

        # Query database
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
            row = cursor.fetchone()

            if row:
                card = dict(row)
                self.cache[grp_id] = card

                if self.show_reskin_names and card.get("is_reskin"):
                    # Show Spider-Man reskin name
                    return card.get("name", f"Unknown({grp_id})")
                else:
                    # Show OmenPaths canonical name (printed name)
                    return card.get("printed_name") or card.get("name", f"Unknown({grp_id})")
            else:
                logging.debug(f"Card {grp_id} not in database")
                return f"Unknown({grp_id})"

        except Exception as e:
            logging.error(f"Database error for grpId {grp_id}: {e}")
            return f"Unknown({grp_id})"

    def get_card_data(self, grp_id: int) -> Optional[dict]:
        """Get full card data from grpId."""
        if not grp_id or not self.conn:
            return None

        # Check cache
        if grp_id in self.cache:
            return self.cache[grp_id]

        # Query database
        try:
            cursor = self.conn.cursor()
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

    def get_keywords(self, grp_id: int) -> List[str]:
        """Get ability keywords for a card (e.g., ['Flying', 'Lifelink'])."""
        card = self.get_card_data(grp_id)
        if not card:
            return []
        keywords_str = card.get("keywords", "")
        if not keywords_str:
            return []
        # Parse comma-separated keywords
        return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
