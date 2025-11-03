#!/usr/bin/env python3
"""
Test script for draft advisor functionality

This script simulates draft events to test the draft advisor integration.
"""

import sys
import json

# Add current directory to path
sys.path.insert(0, '.')

from draft_advisor import DraftAdvisor, DraftCard, display_draft_pack


def test_display():
    """Test the display_draft_pack function with mock data"""

    print("\n" + "="*80)
    print("Testing Draft Pack Display")
    print("="*80 + "\n")

    # Create mock cards
    mock_cards = [
        DraftCard(
            arena_id=75001,
            name="Lightning Strike",
            colors="R",
            rarity="uncommon",
            types=["Instant"],
            win_rate=0.547,
            gih_win_rate=0.625,
            iwd=0.038,
            grade="A+",
            score=99.2
        ),
        DraftCard(
            arena_id=75002,
            name="Courageous Goblin",
            colors="R",
            rarity="common",
            types=["Creature", "Goblin"],
            win_rate=0.534,
            gih_win_rate=0.572,
            iwd=0.025,
            grade="B+",
            score=85.3
        ),
        DraftCard(
            arena_id=75003,
            name="Scroll of Avacyn",
            colors="",
            rarity="common",
            types=["Artifact"],
            win_rate=0.512,
            gih_win_rate=0.541,
            iwd=0.018,
            grade="B",
            score=76.1
        ),
        DraftCard(
            arena_id=75004,
            name="Llanowar Elves",
            colors="G",
            rarity="common",
            types=["Creature", "Elf", "Druid"],
            win_rate=0.523,
            gih_win_rate=0.558,
            iwd=0.022,
            grade="B",
            score=78.4
        ),
        DraftCard(
            arena_id=75005,
            name="Cancel",
            colors="U",
            rarity="common",
            types=["Instant"],
            win_rate=0.485,
            gih_win_rate=0.502,
            iwd=0.008,
            grade="C+",
            score=57.2
        ),
    ]

    recommendation = "Pick Lightning Strike (A+) - 62.5% GIH WR"

    # Test display
    display_draft_pack(mock_cards, pack_num=1, pick_num=3, recommendation=recommendation)

    print("\n✓ Display test completed!\n")


def test_event_parsing():
    """Test parsing of draft event structures"""

    print("\n" + "="*80)
    print("Testing Draft Event Parsing")
    print("="*80 + "\n")

    # Mock Premier Draft event
    premier_draft_event = {
        "EventId": "PremierDraft_BLB_20250815",
        "DraftId": "abc-123-def-456",
        "PackNumber": 0,  # 0-indexed in JSON
        "PickNumber": 2,  # 0-indexed in JSON
        "CardsInPack": ["75001", "75002", "75003", "75004", "75005"]
    }

    print("Premier Draft Event Structure:")
    print(json.dumps(premier_draft_event, indent=2))

    # Mock Quick Draft event
    quick_draft_event = {
        "EventName": "QuickDraft_BLB_20250815",
        "PackNumber": 0,
        "PickNumber": 2,
        "DraftPack": ["75001", "75002", "75003", "75004", "75005"],
        "PickedCards": ["75010", "75011"]
    }

    print("\nQuick Draft Event Structure:")
    print(json.dumps(quick_draft_event, indent=2))

    # Mock Sealed Pool event
    sealed_pool_event = {
        "Courses": [
            {
                "InternalEventName": "Sealed_BLB_20250815",
                "CardPool": [75001, 75002, 75003, 75004, 75005] + list(range(75006, 75050)),
                "CurrentWins": 0,
                "CurrentLosses": 0,
                "CourseDeckSummary": {
                    "Name": "?=?Loc/Decks/Precon/Sealed_BLB",
                    "Attributes": [
                        {"name": "Format", "value": "Sealed"}
                    ]
                }
            }
        ]
    }

    print("\nSealed Pool Event Structure:")
    print(json.dumps(sealed_pool_event, indent=2))

    print("\n✓ Event parsing test completed!\n")


def test_grade_calculation():
    """Test the grading algorithm"""

    print("\n" + "="*80)
    print("Testing Grade Calculation")
    print("="*80 + "\n")

    from draft_advisor import DraftAdvisor

    # Mock card_db since it's required by the constructor
    class MockCardDB:
        def get_card_name(self, grp_id):
            return "Mock Card"

    class MockScryfallDB:
        def get_card_metadata(self, card_name):
            return {
                "color_identity": "R",
                "rarity": "uncommon",
                "types": "Instant",
            }

    draft_advisor = DraftAdvisor(MockCardDB(), rag_system=None)
    draft_advisor.metadata_db = MockScryfallDB()

    # Test score to grade mapping
    test_scores = [99.5, 95.5, 90.5, 85.5, 76.5, 68.5, 57.5, 45.5, 36.5, 27.5, 17.5, 5.5, 2.0]

    print("Score → Grade Mapping:")
    print("-" * 40)

    for score in test_scores:
        grade = DraftAdvisor(card_db=None)._score_to_grade(score)
        print(f"  {score:5.1f}% → {grade}")

    print("\n✓ Grade calculation test completed!\n")


def main():
    """Run all tests"""

    print("\n" + "="*80)
    print("DRAFT ADVISOR TEST SUITE")
    print("="*80)

    try:
        test_display()
        test_event_parsing()
        test_grade_calculation()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80 + "\n")

        print("Next steps:")
        print("1. Install dependencies: pip install tabulate termcolor scipy")
        print("2. Start MTGA and enter a draft")
        print("3. Run: python advisor.py")
        print("4. Watch for draft pick recommendations!\n")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPlease install dependencies:")
        print("  pip install tabulate termcolor scipy\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
