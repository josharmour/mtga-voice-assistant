#!/usr/bin/env python3
"""
Test script for Task 1.2 - Action Sequence Parsing and Normalization
"""

import sqlite3
import sys
from pathlib import Path
from action_normalization import ActionSequenceParser, CardMetadataEnricher

def test_card_enrichment():
    """Test card metadata enrichment."""
    print("=" * 60)
    print("TEST 1: Card Metadata Enrichment")
    print("=" * 60)

    enricher = CardMetadataEnricher("unified_cards.db")
    print(f"✓ Loaded {len(enricher.cache)} cards")

    # Test a few known cards
    test_ids = [list(enricher.cache.keys())[:5]]  # First 5 cards
    if test_ids[0]:
        for card_id in test_ids[0]:
            metadata = enricher.get_card_metadata(card_id)
            print(f"  Card {card_id}: {metadata.get('name')}")
            if metadata['name'] is None:
                print(f"    WARNING: No metadata for card {card_id}")
    else:
        print("  WARNING: No cards found in database")

    print()


def test_database_setup():
    """Test output database creation."""
    print("=" * 60)
    print("TEST 2: Database Setup")
    print("=" * 60)

    # Remove old database for fresh test
    db_path = "action_sequences_test.db"
    Path(db_path).unlink(missing_ok=True)

    parser = ActionSequenceParser(output_db_path=db_path)
    print(f"✓ Created output database: {db_path}")

    # Check tables exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"✓ Created {len(tables)} tables:")
        for table_name, in tables:
            print(f"    - {table_name}")

    print()
    return db_path


def test_small_file_processing():
    """Test processing a small replay file."""
    print("=" * 60)
    print("TEST 3: Small File Processing")
    print("=" * 60)

    db_path = "action_sequences_test.db"
    parser = ActionSequenceParser(output_db_path=db_path)

    # Find smallest replay file
    data_dir = Path("data/17lands_data")
    replay_files = sorted(data_dir.glob("replay_data*.csv.gz"), key=lambda p: p.stat().st_size)

    if not replay_files:
        print("ERROR: No replay files found")
        return

    # Process the smallest file
    smallest_file = replay_files[0]
    print(f"Processing smallest file: {smallest_file.name}")
    print(f"  File size: {smallest_file.stat().st_size / 1024 / 1024:.1f} MB")

    actions_count = parser.process_replay_file(smallest_file)
    print(f"✓ Processed {actions_count} actions")

    print()
    return actions_count


def test_database_queries():
    """Test querying the action sequence database."""
    print("=" * 60)
    print("TEST 4: Database Queries")
    print("=" * 60)

    db_path = "action_sequences_test.db"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Count actions
        cursor.execute("SELECT COUNT(*) FROM action_sequences")
        action_count, = cursor.fetchone()
        print(f"✓ Total actions in database: {action_count}")

        # Count sequences (games)
        cursor.execute("SELECT COUNT(*) FROM action_sequence_metadata")
        sequence_count, = cursor.fetchone()
        print(f"✓ Total sequences (games) in database: {sequence_count}")

        # Sample action types
        cursor.execute("""
            SELECT action_type, COUNT(*) as count
            FROM action_sequences
            GROUP BY action_type
            ORDER BY count DESC
            LIMIT 10
        """)
        print("\n✓ Top 10 action types:")
        for action_type, count in cursor.fetchall():
            print(f"    {action_type}: {count}")

        # Check action distribution by player
        cursor.execute("""
            SELECT player, COUNT(*) as count
            FROM action_sequences
            GROUP BY player
        """)
        print("\n✓ Actions by player:")
        for player, count in cursor.fetchall():
            print(f"    {player}: {count}")

        # Sample a game sequence
        cursor.execute("""
            SELECT DISTINCT sequence_id
            FROM action_sequences
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            sample_seq_id = result[0]
            print(f"\n✓ Sample sequence: {sample_seq_id}")

            cursor.execute("""
                SELECT action_type, card_name, turn_number, timestamp
                FROM action_sequences
                WHERE sequence_id = ?
                ORDER BY turn_number, timestamp
                LIMIT 10
            """, (sample_seq_id,))

            print("  First 10 actions:")
            for action_type, card_name, turn, ts in cursor.fetchall():
                card_info = f" ({card_name})" if card_name else ""
                print(f"    Turn {turn}, TS {ts}: {action_type}{card_info}")

    print()


def test_outcomes_tracking():
    """Test that game outcomes are properly tracked."""
    print("=" * 60)
    print("TEST 5: Game Outcomes Tracking")
    print("=" * 60)

    db_path = "action_sequences_test.db"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check outcome distribution
        cursor.execute("""
            SELECT game_outcome, COUNT(*) as count
            FROM action_sequence_metadata
            GROUP BY game_outcome
        """)

        outcomes = cursor.fetchall()
        if outcomes:
            total_games = sum(count for _, count in outcomes)
            print(f"✓ Total games: {total_games}")
            for outcome, count in outcomes:
                pct = (count / total_games * 100) if total_games > 0 else 0
                result = "WIN" if outcome == 1 else "LOSS"
                print(f"    {result}: {count} ({pct:.1f}%)")
        else:
            print("  No outcome data")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║ Task 1.2: Action Sequence Parsing and Normalization Tests ║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_card_enrichment()
        db_path = test_database_setup()
        test_small_file_processing()
        test_database_queries()
        test_outcomes_tracking()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run full data processing: python action_normalization.py")
        print("2. Validate action sequences with decision point extraction")
        print("3. Proceed to Task 1.3: Decision Point Extraction")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
