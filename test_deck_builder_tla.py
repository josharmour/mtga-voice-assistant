#!/usr/bin/env python3
"""
Quick test to generate deck suggestions for your current TLA draft.
This will work even if the advisor wasn't running during the draft.
"""

from src.core.deck_builder_v2 import DeckBuilderV2
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize deck builder
db = DeckBuilderV2()

# TODO: Replace this with your actual picked cards
# You can find them in the MTGA log or just list what you remember
picked_cards = [
    # Add your picked cards here, for example:
    # "Aang, the Last Airbender",
    # "Katara, Waterbending Warrior",
    # etc.
]

if not picked_cards:
    print("\n⚠️  No cards specified!")
    print("Edit this file and add your picked cards to the 'picked_cards' list")
    print(f"File: {__file__}")
    exit(1)

print(f"\nGenerating deck suggestions for {len(picked_cards)} TLA cards...\n")

# Generate suggestions
suggestions = db.suggest_deck(picked_cards, "TLA", top_n=3)

if not suggestions:
    print("❌ No suggestions generated")
    print("Make sure card_stats.db has TLA data")
    exit(1)

# Display suggestions
print("="*80)
print("🎯 DECK SUGGESTIONS")
print("="*80)

for i, s in enumerate(suggestions, 1):
    print(f"\n{i}. {s.archetype} {s.color_pair_name} ({s.main_colors})")
    print(f"   Score: {s.score:.3f} (Avg GIHWR: {s.avg_gihwr:.3f}, Penalty: {s.penalty:.3f})")
    print(f"   Maindeck: {sum(s.maindeck.values())} cards")
    print(f"   Lands: {sum(s.lands.values())} ({', '.join(f'{c}x {l}' for l, c in s.lands.items())})")

    if i == 1:
        # Show full decklist for top suggestion
        print("\n   MAINDECK:")
        for card, count in sorted(s.maindeck.items(), key=lambda x: -x[1]):
            print(f"     {count}x {card}")

        if s.sideboard:
            print(f"\n   SIDEBOARD: {sum(s.sideboard.values())} cards")

print("\n" + "="*80)
