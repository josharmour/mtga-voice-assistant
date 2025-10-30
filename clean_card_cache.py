#!/usr/bin/env python3
"""
Clean HTML tags from existing card_cache.json file.

This script removes HTML tags (like <nobr>) from card names in the cache file.
Run this once to clean your existing cache, then the updated advisor.py will
keep it clean going forward.
"""

import json
import re
from pathlib import Path


def clean_card_name(name: str) -> str:
    """Remove HTML tags from card names"""
    if not name:
        return name
    return re.sub(r'<[^>]+>', '', name)


def clean_cache_file(cache_path: str = "card_cache.json"):
    """Clean HTML tags from card names in cache file"""
    cache_file = Path(cache_path)

    if not cache_file.exists():
        print(f"Cache file not found: {cache_path}")
        return

    print(f"Loading cache from {cache_path}...")
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    print(f"Found {len(cache)} cards in cache")

    # Track changes
    cleaned_count = 0

    # Clean all card names
    for grp_id, card_data in cache.items():
        if isinstance(card_data, dict) and "name" in card_data:
            original_name = card_data["name"]
            cleaned_name = clean_card_name(original_name)

            if original_name != cleaned_name:
                card_data["name"] = cleaned_name
                cleaned_count += 1
                print(f"Cleaned: {original_name} -> {cleaned_name}")

    if cleaned_count > 0:
        # Backup original
        backup_path = cache_file.with_suffix('.json.backup')
        print(f"\nBacking up original to {backup_path}...")
        cache_file.rename(backup_path)

        # Save cleaned cache
        print(f"Saving cleaned cache with {cleaned_count} updated names...")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Cleaned {cleaned_count} card names")
        print(f"✓ Original cache backed up to {backup_path}")
    else:
        print("\n✓ No HTML tags found in card names - cache is already clean!")


if __name__ == "__main__":
    clean_cache_file()
