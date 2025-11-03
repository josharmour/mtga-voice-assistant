#!/usr/bin/env python3
"""
Test script for draft advisor functionality

This script simulates draft events to test the draft advisor integration.
"""

import sys
import json
import unittest
from unittest.mock import MagicMock

# Add current directory to path
sys.path.insert(0, '.')

from draft_advisor import DraftAdvisor, DraftCard, display_draft_pack


class TestDraftAdvisor(unittest.TestCase):

    def test_display(self):
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
        ]

        recommendation = "Pick Lightning Strike (A+) - 62.5% GIH WR"

        # Test display
        display_draft_pack(mock_cards, pack_num=1, pick_num=3, recommendation=recommendation)

        print("\n✓ Display test completed!\n")

    def test_event_parsing(self):
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
        print("\n✓ Event parsing test completed!\n")

    def test_grade_calculation(self):
        """Test the grading algorithm"""
        print("\n" + "="*80)
        print("Testing Grade Calculation")
        print("="*80 + "\n")

        # Mock card_db since it's required by the constructor
        mock_card_db = MagicMock()
        mock_card_db.get_card_name.return_value = "Mock Card"

        draft_advisor = DraftAdvisor(mock_card_db)

        # Test score to grade mapping
        test_scores = [99.5, 95.5, 90.5, 85.5, 76.5, 68.5, 57.5, 45.5, 36.5, 27.5, 17.5, 5.5, 2.0]

        print("Score → Grade Mapping:")
        print("-" * 40)

        for score in test_scores:
            grade = draft_advisor._score_to_grade(score)
            print(f"  {score:5.1f}% → {grade}")

        print("\n✓ Grade calculation test completed!\n")


if __name__ == "__main__":
    unittest.main()
