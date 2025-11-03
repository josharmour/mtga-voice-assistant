import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

def check_and_update_card_database() -> bool:
    """
    Check if card database needs update and offer to update it.

    Returns True if database is ready to use, False if update failed.
    """
    from datetime import datetime, timedelta
    from pathlib import Path
    import sqlite3

    db_path = Path("data/unified_cards.db")

    # If database doesn't exist, we need to build it
    if not db_path.exists():
        print("\n" + "="*70)
        print("CARD DATABASE NOT FOUND")
        print("="*70)
        print("The unified card database needs to be built first.")
        print("This is a one-time setup that downloads ~492MB from Scryfall.")
        print("Expected time: 5-10 minutes depending on your connection.")
        print()

        response = input("Build database now? (y/n): ").strip().lower()
        if response != 'y':
            print("Cannot proceed without card database. Exiting.")
            return False

        print("\nBuilding database...")
        import subprocess
        result = subprocess.run(["python3", "build_unified_card_database.py"],
                              capture_output=False)

        if result.returncode != 0:
            print("✗ Database build failed.")
            return False

        print("✓ Database built successfully!")
        return True

    # Database exists - check if it's stale (>7 days old)
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'default_cards_updated_at'")
        row = cursor.fetchone()
        conn.close()

        if not row:
            logging.warning("No update timestamp in database")
            return True  # Use database anyway

        last_update_str = row[0]
        last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00').replace('+00:00', ''))
        age = datetime.now() - last_update

        if age.days >= 7:
            print(f"\n⚠ Card database is {age.days} days old (updated {last_update_str[:10]})")
            response = input("Update now? (y/n - update recommended weekly): ").strip().lower()

            if response == 'y':
                print("Updating database...")
                import subprocess
                result = subprocess.run(["python3", "update_card_database.py", "--quick"],
                                      capture_output=False)

                if result.returncode == 0:
                    print("✓ Database updated successfully!")
                else:
                    print("⚠ Update failed, using existing database.")
        else:
            logging.info(f"Card database is current ({age.days} days old)")

        return True

    except Exception as e:
        logging.warning(f"Could not check database age: {e}")
        return True  # Use database anyway


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
