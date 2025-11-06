#!/usr/bin/env python3
"""
Data Coverage Analysis for MTGA Voice Advisor

Analyzes the current database to verify complete coverage for all draftable sets.
Provides recommendations for missing data and auto-download capabilities.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from download_real_17lands_data import ALL_SETS, CURRENT_STANDARD, SET_FORMATS

# Current Standard sets (October 2025) based on rotation
# Rotation happened in July 2025, these sets are legal until mid-2026
STANDARD_SETS_2025 = [
    'FDN',  # Foundations (permanent through 2029)
    'DFT',  # Aetherdrift (2025)
    'TDM',  # Tarkir: Dragonstorm (2025)
    'FIN',  # Final Fantasy (2025)
    'EOE',  # Edge of Eternities (2025)
    'BLB',  # Bloomburrow (2024)
    'OTJ',  # Outlaws of Thunder Junction (2024)
    'MKM',  # Murders at Karlov Manor (2024)
    'LCI',  # The Lost Caverns of Ixalan (2023)
]

# Recently rotated out (July 2025) - still useful for flashback drafts
ROTATED_OUT_2025 = [
    'WOE',  # Wilds of Eldraine
    'MOM',  # March of the Machine
    'ONE',  # Phyrexia: All Will Be One
    'BRO',  # The Brothers' War
]

# Popular flashback/special draft sets
SPECIAL_DRAFT_SETS = [
    'MH3',  # Modern Horizons 3
    'LTR',  # Lord of the Rings
    'DMU',  # Dominaria United
    'SNC',  # Streets of New Capenna
    'NEO',  # Kamigawa: Neon Dynasty
    'VOW',  # Innistrad: Crimson Vow
    'MID',  # Innistrad: Midnight Hunt
    'AFR',  # Adventures in the Forgotten Realms
    'STX',  # Strixhaven
    'KHM',  # Kaldheim
    'SIR',  # Shadows over Innistrad Remastered
    'PIO',  # Pioneer Masters
    'KTK',  # Khans of Tarkir
    'HBG',  # Alchemy Horizons: Baldur's Gate
    'OM1',  # Through the Omenpaths
]


def get_database_stats() -> Dict[str, Dict]:
    """
    Get detailed statistics for all sets in the database.

    Returns:
        Dict mapping set_code to stats dict with:
            - cards: number of cards
            - avg_games: average games per card
            - min_games: minimum games for any card
            - max_games: maximum games for any card
            - last_updated: datetime of last update
    """
    db_path = Path("data/card_stats.db")

    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            set_code,
            COUNT(*) as cards,
            CAST(AVG(games_played) AS INT) as avg_games,
            MIN(games_played) as min_games,
            MAX(games_played) as max_games,
            MAX(last_updated) as last_updated
        FROM card_stats
        GROUP BY set_code
    """)

    results = {}
    for row in cursor.fetchall():
        set_code, cards, avg_games, min_games, max_games, last_updated_str = row

        try:
            last_updated = datetime.fromisoformat(last_updated_str)
        except (ValueError, TypeError):
            last_updated = datetime(2020, 1, 1)

        results[set_code] = {
            'cards': cards,
            'avg_games': avg_games,
            'min_games': min_games,
            'max_games': max_games,
            'last_updated': last_updated,
            'age_days': (datetime.now() - last_updated).days
        }

    conn.close()
    return results


def analyze_coverage():
    """Main analysis function"""
    print("=" * 120)
    print("MTGA VOICE ADVISOR - DATA COVERAGE ANALYSIS")
    print("=" * 120)
    print()

    # Get current database state
    db_stats = get_database_stats()

    if not db_stats:
        print("❌ ERROR: No database found at data/card_stats.db")
        print("   Run: python download_real_17lands_data.py")
        return

    # ===== SECTION 1: Database Overview =====
    print("📊 DATABASE OVERVIEW")
    print("-" * 120)

    total_cards = sum(s['cards'] for s in db_stats.values())
    total_sets = len(db_stats)
    avg_age = sum(s['age_days'] for s in db_stats.values()) / len(db_stats)

    print(f"Total Sets:     {total_sets}")
    print(f"Total Cards:    {total_cards:,}")
    print(f"Average Age:    {avg_age:.1f} days")
    print()

    # ===== SECTION 2: Current Standard Sets =====
    print("✅ CURRENT STANDARD SETS (October 2025)")
    print("-" * 120)
    print(f"{'Set':6s} | {'Full Name':40s} | {'Cards':>6s} | {'Avg Games':>10s} | {'Age':>8s} | {'Status':8s}")
    print("-" * 120)

    missing_standard = []
    for set_code in STANDARD_SETS_2025:
        set_name = ALL_SETS.get(set_code, 'Unknown')[:40]

        if set_code in db_stats:
            stats = db_stats[set_code]
            status = "✅ GOOD" if stats['age_days'] < 90 else "⚠️ OLD"
            print(f"{set_code:6s} | {set_name:40s} | {stats['cards']:6,} | {stats['avg_games']:10,} | {stats['age_days']:3d} days | {status}")
        else:
            print(f"{set_code:6s} | {set_name:40s} | {'N/A':>6s} | {'N/A':>10s} | {'N/A':>8s} | ❌ MISSING")
            missing_standard.append(set_code)

    print()

    # ===== SECTION 3: Recently Rotated (Still Useful) =====
    print("📜 RECENTLY ROTATED SETS (Still in Flashback Drafts)")
    print("-" * 120)
    print(f"{'Set':6s} | {'Full Name':40s} | {'Cards':>6s} | {'Avg Games':>10s} | {'Age':>8s} | {'Status':8s}")
    print("-" * 120)

    for set_code in ROTATED_OUT_2025:
        set_name = ALL_SETS.get(set_code, 'Unknown')[:40]

        if set_code in db_stats:
            stats = db_stats[set_code]
            status = "✅ GOOD" if stats['age_days'] < 90 else "⚠️ OLD"
            print(f"{set_code:6s} | {set_name:40s} | {stats['cards']:6,} | {stats['avg_games']:10,} | {stats['age_days']:3d} days | {status}")
        else:
            print(f"{set_code:6s} | {set_name:40s} | {'N/A':>6s} | {'N/A':>10s} | {'N/A':>8s} | ❌ MISSING")

    print()

    # ===== SECTION 4: Special/Flashback Draft Sets =====
    print("🎲 SPECIAL & FLASHBACK DRAFT SETS")
    print("-" * 120)
    print(f"{'Set':6s} | {'Full Name':40s} | {'Cards':>6s} | {'Avg Games':>10s} | {'Age':>8s} | {'Status':8s}")
    print("-" * 120)

    missing_special = []
    for set_code in SPECIAL_DRAFT_SETS:
        set_name = ALL_SETS.get(set_code, 'Unknown')[:40]

        if set_code in db_stats:
            stats = db_stats[set_code]
            status = "✅ GOOD" if stats['age_days'] < 90 else "⚠️ OLD"
            print(f"{set_code:6s} | {set_name:40s} | {stats['cards']:6,} | {stats['avg_games']:10,} | {stats['age_days']:3d} days | {status}")
        else:
            print(f"{set_code:6s} | {set_name:40s} | {'N/A':>6s} | {'N/A':>10s} | {'N/A':>8s} | ❌ MISSING")
            missing_special.append(set_code)

    print()

    # ===== SECTION 5: Data Quality Assessment =====
    print("🔍 DATA QUALITY ASSESSMENT")
    print("-" * 120)

    # Check for sets with low game counts
    low_data_sets = []
    for set_code, stats in db_stats.items():
        if stats['avg_games'] < 10000:
            low_data_sets.append((set_code, stats['avg_games']))

    if low_data_sets:
        print("⚠️  Sets with low average game counts (< 10,000 games):")
        for set_code, avg_games in sorted(low_data_sets, key=lambda x: x[1]):
            set_name = ALL_SETS.get(set_code, 'Unknown')
            print(f"   {set_code}: {avg_games:,} games - {set_name}")
        print()
    else:
        print("✅ All sets have adequate game data (> 10,000 games average)")
        print()

    # Check for outdated sets
    old_sets = []
    for set_code, stats in db_stats.items():
        if stats['age_days'] > 90:
            old_sets.append((set_code, stats['age_days']))

    if old_sets:
        print("⚠️  Sets with outdated data (> 90 days old):")
        for set_code, age in sorted(old_sets, key=lambda x: x[1], reverse=True):
            set_name = ALL_SETS.get(set_code, 'Unknown')
            print(f"   {set_code}: {age} days old - {set_name}")
        print()
    else:
        print("✅ All sets have recent data (< 90 days old)")
        print()

    # ===== SECTION 6: Missing Data Summary =====
    print("📋 MISSING DATA SUMMARY")
    print("-" * 120)

    all_missing = missing_standard + missing_special

    if all_missing:
        print(f"❌ Found {len(all_missing)} missing sets:")
        for set_code in all_missing:
            set_name = ALL_SETS.get(set_code, 'Unknown')
            priority = "HIGH" if set_code in STANDARD_SETS_2025 else "MEDIUM"
            print(f"   [{priority:6s}] {set_code}: {set_name}")
        print()
    else:
        print("✅ All important sets have data!")
        print()

    # ===== SECTION 7: Recommendations =====
    print("💡 RECOMMENDATIONS")
    print("-" * 120)

    if missing_standard:
        print("❗ CRITICAL: Missing current Standard sets")
        print(f"   Run: python update_card_data.py --auto")
        print(f"   Missing: {', '.join(missing_standard)}")
        print()

    if old_sets:
        print("⚠️  RECOMMENDED: Update outdated sets")
        print(f"   Run: python update_card_data.py --max-age 90")
        print(f"   Will update {len(old_sets)} sets")
        print()

    if low_data_sets:
        print("ℹ️  INFO: Some sets have low data volume")
        print(f"   This is normal for newer or special sets")
        print(f"   Affected: {len(low_data_sets)} sets")
        print()

    # Data freshness check
    if avg_age < 7:
        print("✅ EXCELLENT: Database is very fresh (< 7 days old)")
    elif avg_age < 30:
        print("✅ GOOD: Database is reasonably fresh (< 30 days old)")
    elif avg_age < 90:
        print("⚠️  FAIR: Database is getting old (< 90 days old)")
    else:
        print("❌ POOR: Database needs updating (> 90 days old)")

    print()

    # ===== SECTION 8: Auto-Download Capability =====
    print("🤖 AUTO-DOWNLOAD CAPABILITY")
    print("-" * 120)
    print("Current implementation status:")
    print()
    print("✅ Manual download:     download_real_17lands_data.py")
    print("✅ Update checker:      update_card_data.py --status")
    print("✅ Auto updater:        update_card_data.py --auto")
    print("❌ Draft detection:     NOT IMPLEMENTED")
    print("❌ Auto-fetch on enter: NOT IMPLEMENTED")
    print()
    print("To implement auto-download when entering draft:")
    print("1. Detect Event_Join message in Player.log")
    print("2. Extract set code from InternalEventName (e.g., 'PremierDraft_BLB_20240801')")
    print("3. Check if set data exists in database")
    print("4. Auto-download if missing (with user permission)")
    print()

    # ===== SECTION 9: Coverage Score =====
    print("📈 COVERAGE SCORE")
    print("-" * 120)

    # Calculate coverage score
    standard_coverage = (len(STANDARD_SETS_2025) - len(missing_standard)) / len(STANDARD_SETS_2025) * 100
    special_coverage = (len(SPECIAL_DRAFT_SETS) - len(missing_special)) / len(SPECIAL_DRAFT_SETS) * 100
    overall_coverage = (standard_coverage + special_coverage) / 2

    print(f"Standard Sets:  {standard_coverage:5.1f}% ({len(STANDARD_SETS_2025) - len(missing_standard)}/{len(STANDARD_SETS_2025)} sets)")
    print(f"Special Sets:   {special_coverage:5.1f}% ({len(SPECIAL_DRAFT_SETS) - len(missing_special)}/{len(SPECIAL_DRAFT_SETS)} sets)")
    print(f"Overall:        {overall_coverage:5.1f}%")
    print()

    if overall_coverage >= 95:
        print("🏆 EXCELLENT COVERAGE!")
    elif overall_coverage >= 80:
        print("✅ GOOD COVERAGE")
    elif overall_coverage >= 60:
        print("⚠️  FAIR COVERAGE - Consider downloading missing sets")
    else:
        print("❌ POOR COVERAGE - Many sets missing")

    print()
    print("=" * 120)


def main():
    analyze_coverage()


if __name__ == "__main__":
    main()
