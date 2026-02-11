#!/usr/bin/env python3
"""
Script to trigger debug export programmatically.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.ai_debug_interface import AIDebugInterface
from src.core.mtga import GameStateManager
from src.data.arena_cards import ArenaCardDatabase

print("Initializing card database...")
card_db = ArenaCardDatabase()

print("Creating game state manager...")
gsm = GameStateManager(card_db)

print("Creating debug interface...")
debug = AIDebugInterface(game_state_mgr=gsm)

print("Exporting debug bundle (this may take 30-60 seconds for screenshots)...")
bundle_path = debug.export_debug_bundle(overwrite=True)

print(f"\n[OK] Debug bundle exported to: {bundle_path}")
print("\nQuick Summary:")
print("=" * 60)

# Print current state
import json
with open(bundle_path, 'r', encoding='utf-8') as f:
    bundle = json.load(f)

current_state = bundle.get('current_state', {})
match = current_state.get('match', {})
print(f"Match Status: {match.get('status')}")

if match.get('status') == 'active_match':
    print(f"Turn: {match.get('turn')}")
    print(f"Phase: {match.get('phase')}")
    print(f"Your Turn: {match.get('is_your_turn')}")
    print(f"Has Priority: {match.get('has_priority')}")
    print(f"Life: {match.get('life', {}).get('yours')} vs {match.get('life', {}).get('opponent')}")

    zones = match.get('zones', {})
    print(f"\nYour Zones:")
    print(f"  Hand: {zones.get('hand', {}).get('count')} cards")
    print(f"  Battlefield: {zones.get('battlefield', {}).get('count')} permanents")
    print(f"  Graveyard: {zones.get('graveyard', {}).get('count')} cards")
    print(f"  Library: {zones.get('library', {}).get('count')} cards")

# Print recent errors
errors = bundle.get('recent_errors', [])
if errors:
    print(f"\n[!] Recent Errors ({len(errors)}):")
    for err in errors[:3]:
        print(f"  {err.get('line', '')[:100]}")

# Print recent warnings
warnings = bundle.get('recent_warnings', [])
if warnings:
    print(f"\n[!] Recent Warnings ({len(warnings)}):")
    for warn in warnings[:3]:
        print(f"  {warn.get('line', '')[:100]}")

print("\n" + "=" * 60)
print("Full analysis in debug_bundle.json")
