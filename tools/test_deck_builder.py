#!/usr/bin/env python3
"""
Test script for deck builder functionality

Tests deck building suggestions based on drafted cards.
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, '.')

from deck_builder import DeckBuilder, display_deck_suggestion

class TestDeckBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.builder = DeckBuilder()
        self.drafted_cards = [
            # Red cards (12)
            "Lightning Strike", "Lightning Strike",  # 2 copies
            "Emberheart Challenger",
            "Heartfire Hero",
            "Flamecache Gecko",
            "Cindering Cutthroat",
            "Steampath Charger",
            "Frilled Sparkshooter",
            "Kindlespark Duo",
            "Fireglass Mentor",
            "Conduct Electricity",
            "Festival of Embers",

            # White cards (10)
            "Salvation Swan",
            "Moonrise Cleric",
            "Starscape Cleric",
            "Plumecreed Escort",
            "Pawpatch Recruit",
            "Seasoned Warrenguard",
            "Intrepid Rabbit",
            "Repel Calamity",
            "Sugar Coat",
            "Dawn's Truce",

            # Blue cards (8)
            "Shoreline Looter",
            "Pearl of Wisdom",
            "Dazzling Denial",
            "Sonar Strike",
            "Splash Lasher",
            "Eddymurk Crab",
            "Long River's Pull",
            "Stargaze",

            # Green cards (10)
            "Brambleguard Captain",
            "Galewind Moose",
            "Burrowguard Mentor",
            "Seedpod Squire",
            "Nettle Guard",
            "Bushy Bodyguard",
            "High Stride",
            "Longstalk Brawl",
            "Feed the Cycle",
            "Druid of the Spade",

            # Artifacts/Colorless (5)
            "Patchwork Banner",
            "Short Bow",
            "Carrot Cake",
            "Fountainport Bell",
            "Feather of Flight",
        ]
        self.set_code = "BLB"

    def test_deck_suggestion(self):
        """Test deck suggestion generation"""
        print("\n" + "="*80)
        print("Testing Deck Builder")
        print("="*80 + "\n")

        print(f"Drafted {len(self.drafted_cards)} cards")
        print()

        print(f"Checking for 17lands data: data/17lands_{self.set_code}_PremierDraft.csv")

        try:
            # Generate deck suggestions
            suggestions = self.builder.suggest_deck(self.drafted_cards, self.set_code, top_n=3)

            # Only assert if suggestions were expected
            if self.builder.winning_decks_cache:
                self.assertIsNotNone(suggestions)
                self.assertIsInstance(suggestions, list)
                self.assertGreater(len(suggestions), 0, "No suggestions generated")
            else:
                # If no data, we expect no suggestions
                self.assertEqual(len(suggestions), 0, "Suggestions generated without data")

            print(f"\n✅ Generated {len(suggestions)} deck suggestions!")
            print()

            # Display all suggestions
            for i, suggestion in enumerate(suggestions, 1):
                print()
                if i == 1:
                    print("="*80)
                    print(f"TOP SUGGESTION #{i}")
                    print("="*80)
                else:
                    print("-"*80)
                    print(f"ALTERNATIVE #{i}")
                    print("-"*80)

                display_deck_suggestion(suggestion)

            print("\n✅ Deck builder test completed successfully!")

        except FileNotFoundError as e:
            self.fail(f"File not found: {e}. Download 17lands data for {self.set_code} set. Run: python3 download_real_17lands_data.py")
        except Exception as e:
            self.fail(f"Test failed: {e}")

    def test_color_detection(self):
        """Test color pair detection"""
        print("\n" + "="*80)
        print("Testing Color Pair Detection")
        print("="*80 + "\n")

        test_colors = ["W", "U", "B", "R", "G", "WU", "BR", "GW", "UR", "BG"]

        print("Color Pair Names:")
        print("-" * 40)
        for colors in test_colors:
            name = self.builder._get_color_pair_name(colors)
            print(f"  {colors:4} → {name}")
            self.assertIsNotNone(name)
            self.assertIsInstance(name, str)

        print("\n✅ Color detection test completed!")

if __name__ == "__main__":
    unittest.main()