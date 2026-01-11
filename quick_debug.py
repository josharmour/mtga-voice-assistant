#!/usr/bin/env python3
"""
QUICK DEBUG EXPORT

Run this script when something goes wrong to instantly export a debug bundle.

Usage:
    python quick_debug.py

This creates a JSON file with:
- Current match state
- Recent logs (last 5 minutes)
- Recent errors
- Match timeline
- System health

Perfect for sharing with an AI assistant!
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.ai_debug_interface import AIDebugInterface
except ImportError as e:
    print(f"Error importing debug interface: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def main():
    print("=" * 60)
    print("QUICK DEBUG EXPORT")
    print("=" * 60)
    print()

    # Create debug interface (works without running app)
    debug = AIDebugInterface()

    print("🔍 Collecting debug data...")
    print("  - Scanning logs...")
    print("  - Checking system state...")
    print("  - Gathering recent errors...")
    print()

    try:
        # Export bundle
        bundle_path = debug.export_debug_bundle()

        print("✅ SUCCESS!")
        print()
        print(f"📁 Debug bundle exported to:")
        print(f"   {bundle_path}")
        print()
        print("📋 What's in the bundle:")
        print("  - Current match state (turn, cards, life)")
        print("  - Recent logs (last 5 minutes)")
        print("  - All recent errors")
        print("  - Match event timeline")
        print("  - System health check")
        print()
        print("💡 Next steps:")
        print("  1. Share this JSON file with your AI assistant")
        print("  2. Tell them: 'I exported a debug bundle, can you analyze it?'")
        print("  3. They can see exactly what's happening!")
        print()

        # Quick preview
        with open(bundle_path, 'r') as f:
            data = json.load(f)

        # Show error count
        error_count = len(data.get('recent_errors', []))
        warning_count = len(data.get('recent_warnings', []))

        if error_count > 0 or warning_count > 0:
            print("⚠️  Quick Health Check:")
            if error_count > 0:
                print(f"   - {error_count} recent errors found")
            if warning_count > 0:
                print(f"   - {warning_count} recent warnings found")
            print()

        # Show match state
        match_state = data.get('current_state', {}).get('match', {})
        if match_state.get('status') == 'active_match':
            turn = match_state.get('turn', 'unknown')
            phase = match_state.get('phase', 'unknown')
            print(f"🎮 Active Match Detected:")
            print(f"   Turn {turn} - {phase}")
            print()

        return 0

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Debug export failed. Check logs/advisor.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
