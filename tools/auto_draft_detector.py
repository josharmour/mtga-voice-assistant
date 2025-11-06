#!/usr/bin/env python3
"""
Auto Draft Detection and Data Download

Monitors MTGA log for draft entry and automatically downloads missing card data.
Can be run standalone or integrated into the main advisor.
"""

import re
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple
from download_real_17lands_data import download_card_data, parse_card_stats_from_csv, ALL_SETS

logger = logging.getLogger(__name__)


class DraftDetector:
    """Detects draft entry from MTGA log and manages data downloads"""

    # Regex patterns for draft event detection
    EVENT_JOIN_PATTERN = re.compile(r'"Event_Join".*?"InternalEventName":"([^"]+)"')
    DRAFT_PACK_PATTERN = re.compile(r'DraftStatus')

    # Draft format keywords
    DRAFT_FORMATS = ['PremierDraft', 'QuickDraft', 'TradDraft', 'Sealed', 'PickTwoDraft']

    def __init__(self, db_path: str = "data/card_stats.db", auto_download: bool = True):
        """
        Initialize DraftDetector

        Args:
            db_path: Path to card stats database
            auto_download: If True, automatically download missing data
        """
        self.db_path = Path(db_path)
        self.auto_download = auto_download
        self.current_draft_set = None
        self.last_event_name = None

    def check_set_in_database(self, set_code: str) -> bool:
        """
        Check if a set exists in the database with reasonable data.

        Args:
            set_code: Set code to check (e.g., 'BLB', 'OTJ')

        Returns:
            True if set has data, False otherwise
        """
        if not self.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM card_stats WHERE set_code = ? AND games_played >= 1000",
                (set_code,)
            )

            count = cursor.fetchone()[0]
            conn.close()

            # Consider set present if it has at least 100 cards with 1000+ games
            return count >= 100

        except Exception as e:
            logger.error(f"Error checking database for {set_code}: {e}")
            return False

    def extract_set_from_event_name(self, event_name: str) -> Optional[Tuple[str, str]]:
        """
        Extract set code and format from InternalEventName.

        Examples:
            "PremierDraft_BLB_20240801" -> ('BLB', 'PremierDraft')
            "QuickDraft_OTJ_20240405" -> ('OTJ', 'QuickDraft')
            "Sealed_MKM_20240209" -> ('MKM', 'Sealed')
            "PickTwoDraft_OM1_20251001" -> ('OM1', 'PickTwoDraft')

        Args:
            event_name: InternalEventName from MTGA log

        Returns:
            Tuple of (set_code, format) or None if not a draft event
        """
        # Check if it's a draft format
        for draft_format in self.DRAFT_FORMATS:
            if event_name.startswith(draft_format):
                # Extract set code (usually 3 letters after format)
                # Format: {DraftFormat}_{SET}_{DATE}
                parts = event_name.split('_')

                if len(parts) >= 3:
                    set_code = parts[1].upper()

                    # Validate set code
                    if set_code in ALL_SETS:
                        return (set_code, draft_format)

                    # Try parts[2] in case format has underscore
                    if len(parts) >= 4:
                        set_code = parts[2].upper()
                        if set_code in ALL_SETS:
                            return (set_code, draft_format)

        return None

    def process_log_line(self, line: str) -> Optional[str]:
        """
        Process a single log line to detect draft entry.

        Args:
            line: Line from Player.log

        Returns:
            Set code if draft detected and data is missing, None otherwise
        """
        # Look for Event_Join with InternalEventName
        match = self.EVENT_JOIN_PATTERN.search(line)

        if match:
            event_name = match.group(1)

            # Skip if same event as last time
            if event_name == self.last_event_name:
                return None

            self.last_event_name = event_name
            logger.info(f"Detected event join: {event_name}")

            # Extract set and format
            result = self.extract_set_from_event_name(event_name)

            if result:
                set_code, draft_format = result
                set_name = ALL_SETS.get(set_code, 'Unknown')

                logger.info(f"Draft detected: {set_name} ({set_code}) - {draft_format}")

                # Check if we have data for this set
                if not self.check_set_in_database(set_code):
                    logger.warning(f"Missing data for {set_code}")
                    self.current_draft_set = set_code
                    return set_code
                else:
                    logger.info(f"Data already available for {set_code}")
                    self.current_draft_set = set_code

        return None

    def download_missing_data(self, set_code: str, interactive: bool = True) -> bool:
        """
        Download data for a missing set.

        Args:
            set_code: Set code to download
            interactive: If True, ask user for permission

        Returns:
            True if download succeeded, False otherwise
        """
        set_name = ALL_SETS.get(set_code, 'Unknown')

        if interactive:
            print()
            print("=" * 70)
            print(f"⚠️  Missing Draft Data: {set_name} ({set_code})")
            print("=" * 70)
            print(f"The advisor needs 17lands statistics for {set_name}.")
            print(f"This will download ~500MB-2GB of data and take 5-15 minutes.")
            print()

            response = input("Download now? (y/n): ").strip().lower()

            if response != 'y':
                print("Skipping download. Advisor will work with limited data.")
                return False

        try:
            logger.info(f"Downloading card data for {set_code}...")
            print(f"\n📥 Downloading {set_name} ({set_code})...")

            # Determine format (most use PremierDraft, OM1 uses PickTwoDraft)
            from download_real_17lands_data import SET_FORMATS
            event_format = SET_FORMATS.get(set_code, "PremierDraft")

            csv_path = download_card_data(set_code, event_format)

            if not csv_path:
                logger.error(f"Download failed for {set_code}")
                print(f"❌ Download failed for {set_code}")
                return False

            # Parse and insert into database
            print(f"📊 Parsing card statistics...")
            cards = parse_card_stats_from_csv(csv_path, min_games=1000)

            if not cards:
                logger.error(f"No cards parsed from {csv_path}")
                print(f"❌ No card data found in downloaded file")
                return False

            # Import into database
            from rag_advisor import CardStatsDB
            db = CardStatsDB()
            db.insert_card_stats(cards)
            db.close()

            print(f"✅ Successfully imported {len(cards)} cards for {set_name}")
            logger.info(f"Successfully imported {len(cards)} cards for {set_code}")

            return True

        except Exception as e:
            logger.error(f"Error downloading data for {set_code}: {e}")
            print(f"❌ Error: {e}")
            return False


def test_detector():
    """Test the draft detector with sample log lines"""
    print("=" * 70)
    print("Draft Detector Test")
    print("=" * 70)

    detector = DraftDetector(auto_download=False)

    # Test cases
    test_lines = [
        '{"Event_Join":{"InternalEventName":"PremierDraft_BLB_20240801"}}',
        '{"Event_Join":{"InternalEventName":"QuickDraft_OTJ_20240405"}}',
        '{"Event_Join":{"InternalEventName":"Sealed_MKM_20240209"}}',
        '{"Event_Join":{"InternalEventName":"PickTwoDraft_OM1_20251001"}}',
        '{"Event_Join":{"InternalEventName":"Constructed_Ranked_20251020"}}',
        '{"Event_Join":{"InternalEventName":"PremierDraft_DSK_20241001"}}',
        '{"Event_Join":{"InternalEventName":"PremierDraft_FDN_20241115"}}',
    ]

    print("\nTesting event detection:\n")

    for line in test_lines:
        result = detector.process_log_line(line)

        # Extract event name for display
        match = detector.EVENT_JOIN_PATTERN.search(line)
        if match:
            event_name = match.group(1)
            parsed = detector.extract_set_from_event_name(event_name)

            if parsed:
                set_code, draft_format = parsed
                set_name = ALL_SETS.get(set_code, 'Unknown')
                has_data = detector.check_set_in_database(set_code)

                status = "✅ HAS DATA" if has_data else "❌ MISSING"
                print(f"{event_name:40s} -> {set_code} ({draft_format:15s}) {status}")
            else:
                print(f"{event_name:40s} -> Not a draft event")

    print()
    print("=" * 70)
    print("Database Status:")
    print("=" * 70)

    # Check which sets have data
    for set_code in ['BLB', 'OTJ', 'MKM', 'OM1', 'DSK', 'FDN', 'LCI', 'WOE']:
        has_data = detector.check_set_in_database(set_code)
        set_name = ALL_SETS.get(set_code, 'Unknown')
        status = "✅" if has_data else "❌"
        print(f"{status} {set_code}: {set_name}")

    print()


def main():
    """Main function for standalone testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Auto Draft Detector for MTGA")
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--check', type=str, help='Check if a set code has data')
    parser.add_argument('--download', type=str, help='Download data for a set code')

    args = parser.parse_args()

    if args.test:
        test_detector()
    elif args.check:
        detector = DraftDetector()
        has_data = detector.check_set_in_database(args.check)
        set_name = ALL_SETS.get(args.check, 'Unknown')
        print(f"{args.check} ({set_name}): {'✅ Has data' if has_data else '❌ Missing data'}")
    elif args.download:
        detector = DraftDetector()
        success = detector.download_missing_data(args.download, interactive=True)
        exit(0 if success else 1)
    else:
        print("MTGA Auto Draft Detector")
        print()
        print("Usage:")
        print("  --test              Run test mode")
        print("  --check SET         Check if set has data (e.g., BLB)")
        print("  --download SET      Download data for set")
        print()
        print("Example:")
        print("  python auto_draft_detector.py --test")
        print("  python auto_draft_detector.py --check BLB")
        print("  python auto_draft_detector.py --download OTJ")


if __name__ == "__main__":
    main()
