#!/usr/bin/env python3
"""
Quick debugging CLI for AI assistants.

Usage:
    python debug_cli.py status           # Get current state
    python debug_cli.py errors           # Show recent errors
    python debug_cli.py timeline         # Show match timeline
    python debug_cli.py logs --pattern "ERROR"  # Query logs
    python debug_cli.py export           # Export debug bundle

This lets you quickly check what's happening without digging through logs.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ai_debug_interface import AIDebugInterface


def pretty_print(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, default=str))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    # Create debug interface (no app instance needed for log queries)
    debug = AIDebugInterface()

    if command == "status":
        print("=== CURRENT STATE ===")
        state = debug.get_current_state()
        pretty_print(state)

    elif command == "errors":
        print("=== RECENT ERRORS ===")
        errors = debug.get_recent_errors(count=10)
        for error in errors:
            print(f"{error['timestamp']}: {error['line']}")

    elif command == "warnings":
        print("=== RECENT WARNINGS ===")
        warnings = debug.get_recent_warnings(count=10)
        for warning in warnings:
            print(f"{warning['timestamp']}: {warning['line']}")

    elif command == "timeline":
        print("=== MATCH TIMELINE ===")
        timeline = debug.get_match_timeline(max_events=30)
        for event in timeline:
            print(f"{event['timestamp']}: {event['event']}")

    elif command == "logs":
        # Parse additional args
        pattern = None
        level = None
        minutes = 5

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--pattern" and i + 1 < len(sys.argv):
                pattern = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--level" and i + 1 < len(sys.argv):
                level = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--minutes" and i + 1 < len(sys.argv):
                minutes = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1

        print(f"=== LOGS (last {minutes} min) ===")
        logs = debug.query_logs(
            level=level,
            pattern=pattern,
            last_minutes=minutes,
            max_lines=50
        )
        for log in logs:
            print(log['line'])

    elif command == "export":
        print("=== EXPORTING DEBUG BUNDLE ===")
        overwrite = "--overwrite" in sys.argv
        bundle_path = debug.export_debug_bundle(overwrite=overwrite)
        print(f"Debug bundle exported to: {bundle_path}")
        print("\nYou can now share this file with an AI assistant for analysis.")

    elif command == "health":
        print("=== SYSTEM HEALTH CHECK ===")
        health = debug._check_health()
        pretty_print(health)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
