#!/usr/bin/env python3
"""
Automatic Database Update Manager for MTGA Voice Advisor

This module handles all automatic updates:
1. Detects when Arena's Raw_CardDatabase has new cards
2. Automatically downloads 17lands data for new sets
3. Updates data daily based on format being played
4. Shows progress bars for all downloads
"""

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressBar:
    """Simple progress bar for console output."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        self._display()

    def _display(self):
        """Display the progress bar."""
        if self.total == 0:
            return

        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        sys.stdout.write(f'\r{self.desc}: [{bar}] {percent*100:.1f}%')
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write('\n')


class AutoUpdater:
    """Manages automatic database updates for MTGA Voice Advisor."""

    def __init__(self, auto_mode: bool = True):
        """
        Initialize the auto-updater.

        Args:
            auto_mode: If True, updates happen automatically without prompts
        """
        self.auto_mode = auto_mode
        self.update_state_file = Path("data/.update_state.json")
        self.unified_db_path = Path("data/unified_cards.db")
        self.stats_db_path = Path("data/card_stats.db")
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load the update state from disk."""
        if self.update_state_file.exists():
            try:
                with open(self.update_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load update state: {e}")

        return {
            "last_arena_check": None,
            "last_17lands_check": None,
            "arena_db_hash": None,
            "known_sets": [],
            "format_last_played": {},
            "last_full_update": None
        }

    def _save_state(self):
        """Save the update state to disk."""
        self.update_state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.update_state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def find_arena_database(self) -> Optional[Path]:
        """Find MTGA's Raw_CardDatabase file."""
        search_paths = [
            # Steam on Linux
            Path.home() / ".local/share/Steam/steamapps/common/MTGA/MTGA_Data/Downloads/Raw",
            # Windows paths
            Path("C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw"),
            Path("C:/Program Files (x86)/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw"),
            # Wine/Bottles/Lutris paths
            Path.home() / ".wine/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw",
        ]

        for search_path in search_paths:
            if search_path.exists():
                import glob
                pattern = str(search_path / "Raw_CardDatabase_*.mtga")
                files = glob.glob(pattern)
                if files:
                    return Path(max(files, key=os.path.getmtime))

        # Try system-wide search
        try:
            result = subprocess.run(
                ["find", str(Path.home()), "-name", "Raw_CardDatabase*.mtga", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                files = result.stdout.strip().split('\n')
                if files:
                    return Path(files[0])
        except Exception:
            pass

        return None

    def get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_arena_database_updated(self) -> bool:
        """Check if Arena's Raw_CardDatabase has been updated."""
        arena_db = self.find_arena_database()
        if not arena_db:
            logger.warning("Could not find Arena database")
            return False

        current_hash = self.get_file_hash(arena_db)

        if self.state.get("arena_db_hash") != current_hash:
            logger.info(f"Arena database has been updated (new hash: {current_hash[:8]}...)")
            self.state["arena_db_hash"] = current_hash
            self._save_state()
            return True

        return False

    def rebuild_unified_database(self) -> bool:
        """Rebuild unified_cards.db from Arena's database."""
        logger.info("Rebuilding unified card database...")

        if not self.auto_mode:
            response = input("\nArena database has been updated. Rebuild card database? (y/n): ")
            if response.lower() != 'y':
                return False

        try:
            result = subprocess.run(
                [sys.executable, "build_unified_card_database.py"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("✅ Card database rebuilt successfully")
                return True
            else:
                logger.error(f"Failed to rebuild database: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error rebuilding database: {e}")
            return False

    def detect_current_format(self) -> Optional[Tuple[str, str]]:
        """
        Detect what format is currently being played by checking recent logs.

        Returns:
            Tuple of (set_code, format_type) or None
        """
        # Check the most recent player log for event information
        log_paths = [
            Path.home() / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            Path.home() / "AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
        ]

        for log_path in log_paths:
            if not log_path.exists():
                continue

            try:
                # Read last 10KB of log to find recent event
                with open(log_path, 'rb') as f:
                    f.seek(0, 2)  # Go to end
                    file_size = f.tell()
                    f.seek(max(0, file_size - 10240))  # Go back 10KB
                    content = f.read().decode('utf-8', errors='ignore')

                # Look for Event_Join patterns
                import re
                pattern = r'"InternalEventName":"(PremierDraft|QuickDraft|TradDraft|PickTwoDraft|Sealed)_([A-Z0-9]+)_'
                matches = re.findall(pattern, content)

                if matches:
                    format_type, set_code = matches[-1]  # Get most recent
                    return (set_code, format_type)

            except Exception as e:
                logger.debug(f"Could not read log {log_path}: {e}")

        return None

    def get_17lands_sets_needing_update(self) -> List[Tuple[str, str]]:
        """
        Determine which sets need 17lands data updates.

        Returns:
            List of (set_code, format_type) tuples that need updating
        """
        needs_update = []

        # Get current MTGA sets from constants (which fetches from Scryfall)
        from constants import CURRENT_STANDARD
        current_sets = set(CURRENT_STANDARD)
        logger.info(f"Current MTGA Standard sets: {sorted(current_sets)}")

        # Check what's in our database
        if not self.stats_db_path.exists():
            # No stats database, need everything
            for set_code in current_sets:
                needs_update.append((set_code, "PremierDraft"))
            return needs_update

        try:
            conn = sqlite3.connect(self.stats_db_path)
            cursor = conn.cursor()

            # Check which sets we have and when they were updated
            cursor.execute("""
                SELECT set_code, MAX(last_updated) as last_update
                FROM card_stats
                GROUP BY set_code
            """)

            existing_sets = {}
            for set_code, last_update_str in cursor.fetchall():
                try:
                    last_update = datetime.fromisoformat(last_update_str)
                    existing_sets[set_code] = last_update
                except:
                    pass

            conn.close()

            # Determine what needs updating
            for set_code in current_sets:
                if set_code not in existing_sets:
                    # Don't have this set at all
                    needs_update.append((set_code, "PremierDraft"))
                elif (datetime.now() - existing_sets[set_code]).days > 7:
                    # Data is more than a week old
                    needs_update.append((set_code, "PremierDraft"))

            # Also check for format-specific updates based on play history
            current_format = self.detect_current_format()
            if current_format:
                set_code, format_type = current_format
                if format_type != "PremierDraft":
                    # Playing a different format, might need that data too
                    needs_update.append((set_code, format_type))

        except Exception as e:
            logger.error(f"Error checking existing 17lands data: {e}")

        return needs_update

    def download_17lands_data(self, set_code: str, format_type: str = "PremierDraft") -> bool:
        """
        Download 17lands data for a specific set and format.

        Args:
            set_code: The set code (e.g., "BLB")
            format_type: The draft format (e.g., "PremierDraft", "QuickDraft")

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Downloading 17lands data for {set_code} ({format_type})...")

        try:
            # Use the manage_data.py functionality
            from manage_data import download_card_data_api
            from rag_advisor import CardStatsDB

            # Download with progress indication
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            cards = download_card_data_api(set_code, format_type, start_date)

            if not cards:
                logger.warning(f"No data available for {set_code} {format_type}")
                return False

            # Insert into database
            db = CardStatsDB()
            db.delete_set_data(set_code)  # Remove old data
            db.insert_card_stats(cards)
            db.close()

            logger.info(f"✅ Downloaded {len(cards)} cards for {set_code}")
            return True

        except Exception as e:
            logger.error(f"Error downloading 17lands data: {e}")
            return False

    def update_17lands_data(self) -> bool:
        """Update all necessary 17lands data."""
        sets_to_update = self.get_17lands_sets_needing_update()

        if not sets_to_update:
            logger.info("17lands data is up to date")
            return True

        if not self.auto_mode:
            print(f"\n17lands data needs updating for {len(sets_to_update)} set(s):")
            for set_code, format_type in sets_to_update:
                print(f"  - {set_code} ({format_type})")
            response = input("\nUpdate now? (y/n): ")
            if response.lower() != 'y':
                return False

        success_count = 0
        progress = ProgressBar(len(sets_to_update), "Updating 17lands data")

        for set_code, format_type in sets_to_update:
            if self.download_17lands_data(set_code, format_type):
                success_count += 1
            progress.update()
            time.sleep(1)  # Be nice to 17lands API

        logger.info(f"Updated {success_count}/{len(sets_to_update)} sets successfully")
        return success_count > 0

    def check_and_update_all(self) -> bool:
        """
        Main update function - checks and updates everything as needed.

        This should be called:
        1. On application startup
        2. Daily (or when specified time has passed)
        3. When entering a draft with a new set

        Returns:
            True if any updates were performed
        """
        updated = False

        print("\n" + "="*60)
        print("  Checking for updates...")
        print("="*60 + "\n")

        # Check if unified_cards.db exists
        if not self.unified_db_path.exists():
            print("⚠️  Card database not found. Building...")
            if self.rebuild_unified_database():
                updated = True
            else:
                logger.error("Failed to build initial card database")
                return False

        # Check if Arena database has been updated
        last_check = self.state.get("last_arena_check")
        if not last_check or (datetime.now() - datetime.fromisoformat(last_check)).total_seconds() > 86400:
            if self.check_arena_database_updated():
                print("🔄 Arena has new cards, rebuilding database...")
                if self.rebuild_unified_database():
                    updated = True
            self.state["last_arena_check"] = datetime.now().isoformat()
            self._save_state()

        # Check 17lands data updates
        last_17lands = self.state.get("last_17lands_check")
        if not last_17lands or (datetime.now() - datetime.fromisoformat(last_17lands)).total_seconds() > 86400:
            print("📊 Checking for 17lands updates...")
            if self.update_17lands_data():
                updated = True
            self.state["last_17lands_check"] = datetime.now().isoformat()
            self._save_state()

        if not updated:
            print("✅ All databases are up to date")
        else:
            print("\n✅ Updates complete!")

        print("="*60 + "\n")
        return updated

    def update_for_draft(self, set_code: str, format_type: str) -> bool:
        """
        Ensure we have data for a specific draft format.

        This is called when entering a draft to ensure we have the right data.

        Args:
            set_code: The set being drafted (e.g., "BLB")
            format_type: The draft format (e.g., "QuickDraft")

        Returns:
            True if data is available (either already present or downloaded)
        """
        # Check if we already have recent data
        if self.stats_db_path.exists():
            try:
                conn = sqlite3.connect(self.stats_db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(last_updated)
                    FROM card_stats
                    WHERE set_code = ?
                """, (set_code,))
                result = cursor.fetchone()
                conn.close()

                if result and result[0]:
                    last_update = datetime.fromisoformat(result[0])
                    if (datetime.now() - last_update).days < 7:
                        logger.info(f"Recent data exists for {set_code}")
                        return True
            except Exception as e:
                logger.warning(f"Could not check existing data: {e}")

        # Need to download data
        if self.auto_mode:
            print(f"\n📥 Downloading data for {set_code} {format_type}...")
            return self.download_17lands_data(set_code, format_type)
        else:
            response = input(f"\nNeed data for {set_code} {format_type}. Download now? (y/n): ")
            if response.lower() == 'y':
                return self.download_17lands_data(set_code, format_type)
            return False


def integrate_with_startup():
    """
    Integration code to add to advisor.py or database.py startup.

    This should be called when the application starts.
    """
    from pathlib import Path

    # Check if it's been more than 24 hours since last check
    last_check_file = Path("data/.last_update_check")
    should_check = True

    if last_check_file.exists():
        try:
            with open(last_check_file, 'r') as f:
                last_check = datetime.fromisoformat(f.read().strip())
                if (datetime.now() - last_check).total_seconds() < 86400:  # 24 hours
                    should_check = False
        except:
            pass

    if should_check:
        updater = AutoUpdater(auto_mode=True)  # Auto mode for seamless updates
        updater.check_and_update_all()

        # Update last check time
        last_check_file.parent.mkdir(parents=True, exist_ok=True)
        with open(last_check_file, 'w') as f:
            f.write(datetime.now().isoformat())


if __name__ == "__main__":
    # Manual run for testing
    import argparse

    parser = argparse.ArgumentParser(description="MTGA Voice Advisor Auto-Updater")
    parser.add_argument("--auto", action="store_true", help="Run in automatic mode (no prompts)")
    parser.add_argument("--force", action="store_true", help="Force update all databases")
    args = parser.parse_args()

    updater = AutoUpdater(auto_mode=args.auto)

    if args.force:
        # Force rebuild everything
        print("Forcing complete update...")
        updater.state = {}  # Clear state to force updates
        updater.check_and_update_all()
    else:
        updater.check_and_update_all()