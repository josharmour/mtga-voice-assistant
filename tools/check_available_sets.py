#!/usr/bin/env python3
"""
Check which sets actually have public data available on 17lands S3.

This script probes the S3 bucket to find which sets can be downloaded.
"""

import requests
from download_real_17lands_data import ALL_SETS, S3_BASE

def check_set_availability(set_code: str, event_type: str = "PremierDraft") -> bool:
    """Check if a set's data is available on S3."""
    url = f"{S3_BASE}/game_data/game_data_public.{set_code}.{event_type}.csv.gz"

    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def main():
    print("=" * 70)
    print("Checking 17lands S3 for available datasets...")
    print("=" * 70)
    print()

    available_sets = []
    unavailable_sets = []

    for set_code, set_name in sorted(ALL_SETS.items()):
        print(f"Checking {set_code:6s} ({set_name})...", end=" ", flush=True)

        if check_set_availability(set_code):
            print("✅ AVAILABLE")
            available_sets.append(set_code)
        else:
            print("❌ NOT AVAILABLE")
            unavailable_sets.append(set_code)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Available:     {len(available_sets)} sets")
    print(f"Not available: {len(unavailable_sets)} sets")
    print()

    if available_sets:
        print("✅ Available sets:")
        for code in available_sets:
            print(f"   {code}: {ALL_SETS[code]}")
        print()

    if unavailable_sets:
        print("❌ Unavailable sets (too new or no public data):")
        for code in unavailable_sets:
            print(f"   {code}: {ALL_SETS[code]}")
        print()

    print("=" * 70)
    print("Python code for CURRENT_STANDARD:")
    print("=" * 70)
    print(f"CURRENT_STANDARD = {available_sets[:8]}")  # First 8 available sets
    print()


if __name__ == "__main__":
    main()
