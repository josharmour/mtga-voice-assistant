#!/usr/bin/env python3
"""
Comprehensive check of ALL sets visible on 17lands public_datasets page.
Based on screenshots from 2025-10-28.
"""

import requests
import time

S3_BASE = "https://17lands-public.s3.amazonaws.com/analysis_data"

# Complete list from 17lands public_datasets page screenshots
ALL_VISIBLE_SETS = [
    # 2025 Sets
    'FOE', 'FIN', 'Y25EOE', 'TDM', 'Y25TDM', 'DFT', 'Y25DFT', 'PIO',

    # 2024-2025 Sets
    'OM1', 'FDN', 'DSK', 'Y25DSK', 'BLB', 'Y25BLB', 'MH3',
    'OTJ', 'Y24OTJ', 'MKM', 'Y24MKM', 'LCI', 'Y24LCI',

    # 2023-2024 Sets
    'WOE', 'Y24WOE', 'LTR', 'MOM', 'MAT', 'SIR',
    'ONE', 'Y23ONE', 'BRO', 'Y23BRO', 'DMU', 'Y23DMU',

    # 2022-2023 Sets
    'HBG', 'SNC', 'Y22SNC', 'NEO', 'DBL', 'VOW', 'RAVM', 'MID',

    # 2021-2022 Sets
    'AFR', 'STX', 'CORE', 'KHM', 'KLR', 'ZNR', 'AKR',

    # Older Sets
    'M21', 'M20', 'IKO', 'THB', 'ELD', 'WAR', 'M19',
    'DOM', 'RIX', 'GRN', 'RNA', 'KTK', 'XLN',

    # Special
    'Ravnica', 'Cube', 'Chaos',
]

def check_set(set_code: str) -> bool:
    """Check if set has PremierDraft data."""
    url = f"{S3_BASE}/game_data/game_data_public.{set_code}.PremierDraft.csv.gz"
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 80)
    print("Comprehensive 17lands Dataset Availability Check")
    print("=" * 80)
    print(f"\nChecking {len(ALL_VISIBLE_SETS)} sets from public_datasets page...\n")

    available = []
    unavailable = []

    for i, set_code in enumerate(ALL_VISIBLE_SETS, 1):
        print(f"[{i:2d}/{len(ALL_VISIBLE_SETS)}] {set_code:10s} ... ", end='', flush=True)

        if check_set(set_code):
            print("✅ AVAILABLE")
            available.append(set_code)
        else:
            print("❌ NOT AVAILABLE")
            unavailable.append(set_code)

        time.sleep(0.2)  # Small delay to be nice to S3

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Available:     {len(available):2d} sets")
    print(f"❌ Not available: {len(unavailable):2d} sets")

    print("\n" + "=" * 80)
    print("AVAILABLE SETS (for download_real_17lands_data.py)")
    print("=" * 80)

    # Categorize available sets
    standard_sets = []
    alchemy_sets = []
    special_sets = []

    for code in available:
        if code.startswith('Y'):
            alchemy_sets.append(code)
        elif code in ['Cube', 'Chaos', 'Ravnica', 'CORE']:
            special_sets.append(code)
        else:
            standard_sets.append(code)

    print("\nStandard/Regular Sets:")
    for code in standard_sets:
        print(f"    '{code}',")

    print("\nAlchemy Sets (Arena-only):")
    for code in alchemy_sets:
        print(f"    '{code}',")

    print("\nSpecial Sets:")
    for code in special_sets:
        print(f"    '{code}',")

    print("\n" + "=" * 80)
    print("UNAVAILABLE SETS (listed on page but no data)")
    print("=" * 80)
    for code in unavailable:
        print(f"    {code}")

    print("\n" + "=" * 80)
    print("Python Dictionary (ALL_SETS) - Standard Sets Only:")
    print("=" * 80)
    print("ALL_SETS = {")
    for code in standard_sets:
        print(f"    '{code}': 'Set Name',  # TODO: Add full name")
    print("}")

if __name__ == "__main__":
    main()
