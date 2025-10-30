#!/usr/bin/env python3
"""
Download Real 17lands Card Data (No API Key Required!)

17lands provides public datasets via direct download from AWS S3.
No authentication or API keys needed.

URL format:
https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{SET}.{EVENT}.csv.gz

Available sets (recent): OTJ, MKM, LCI, WOE, MOM, ONE
Available events: PremierDraft, QuickDraft, TradDraft, Sealed
"""

import requests
import gzip
import csv
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

from rag_advisor import CardStatsDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All sets with publicly available data on 17lands S3
# Verified: October 2025 - Complete list from 17lands public_datasets page
# Total: 29 sets with game_data available
ALL_SETS = {
    # 2025 Sets (Most Recent)
    'EOE': 'Edge of Eternities',
    'FIN': 'Final Fantasy',
    'TDM': 'Tarkir: Dragonstorm',
    'DFT': 'Aetherdrift',

    # 2024-2025 Sets
    'OM1': 'Through the Omenpaths',
    'FDN': 'Foundations',
    'DSK': 'Duskmourn: House of Horror',
    'BLB': 'Bloomburrow',
    'MH3': 'Modern Horizons 3',
    'OTJ': 'Outlaws of Thunder Junction',
    'MKM': 'Murders at Karlov Manor',

    # 2023-2024 Sets
    'LCI': 'The Lost Caverns of Ixalan',
    'WOE': 'Wilds of Eldraine',
    'LTR': 'The Lord of the Rings: Tales of Middle-earth',
    'MOM': 'March of the Machine',
    'ONE': 'Phyrexia: All Will Be One',

    # 2022-2023 Sets
    'BRO': 'The Brothers\' War',
    'DMU': 'Dominaria United',
    'SNC': 'Streets of New Capenna',
    'NEO': 'Kamigawa: Neon Dynasty',
    'VOW': 'Innistrad: Crimson Vow',
    'MID': 'Innistrad: Midnight Hunt',

    # 2021-2022 Sets
    'AFR': 'Adventures in the Forgotten Realms',
    'STX': 'Strixhaven: School of Mages',
    'KHM': 'Kaldheim',

    # Special/Remastered/Masters Sets
    'SIR': 'Shadows over Innistrad Remastered',
    'PIO': 'Pioneer Masters',
    'HBG': 'Alchemy Horizons: Baldur\'s Gate',
    'KTK': 'Khans of Tarkir',
}

# Sets with non-standard formats (most sets use PremierDraft by default)
SET_FORMATS = {
    'OM1': 'PickTwoDraft',  # Through the Omenpaths uses new Pick-Two draft method
}

# Sets that exist but don't have public data:
# Y25EOE, Y25TDM, Y25DFT, Y25DSK, Y25BLB (Alchemy versions - no data)
# Y24OTJ, Y24MKM, Y24LCI, Y24WOE (Alchemy versions - no data)
# Y23ONE, Y23BRO, Y23DMU, Y22SNC (Alchemy versions - no data)
# MAT, DBL, RAVM, CORE, KLR, ZNR, AKR (too old or no data)
# M21, M20, IKO, THB, ELD, WAR, M19, DOM, RIX, GRN, RNA, XLN (too old)
# Ravnica, Cube, Chaos (special formats, no data)

# Current Standard (as of October 2025)
CURRENT_STANDARD = ['OM1', 'FDN', 'DSK', 'BLB', 'OTJ', 'MKM', 'LCI', 'WOE']

# 17lands S3 base URL
S3_BASE = "https://17lands-public.s3.amazonaws.com/analysis_data"


def download_card_data(set_code: str, event_type: str = None) -> str:
    """
    Download 17lands dataset for a specific set.

    Args:
        set_code: Set code (e.g., 'OTJ', 'MKM')
        event_type: Event type (PremierDraft, PickTwoDraft, etc.)
                   If None, uses SET_FORMATS mapping or defaults to PremierDraft

    Returns:
        Path to downloaded CSV file
    """
    # Use format mapping for sets with special formats
    if event_type is None:
        event_type = SET_FORMATS.get(set_code, "PremierDraft")

    url = f"{S3_BASE}/game_data/game_data_public.{set_code}.{event_type}.csv.gz"

    logger.info(f"Downloading {ALL_SETS.get(set_code, set_code)} ({event_type})...")
    logger.info(f"URL: {url}")

    try:
        # Download with progress
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Save compressed file
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        gz_path = data_dir / f"17lands_{set_code}_{event_type}.csv.gz"
        csv_path = data_dir / f"17lands_{set_code}_{event_type}.csv"

        # Download
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"  Progress: {pct:.1f}% ({downloaded:,} / {total_size:,} bytes)")

        logger.info(f"✓ Downloaded to {gz_path}")

        # Decompress
        logger.info("Decompressing...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                f_out.write(f_in.read())

        logger.info(f"✓ Decompressed to {csv_path}")

        # Clean up compressed file
        gz_path.unlink()

        return str(csv_path)

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return None


def parse_card_stats_from_csv(csv_path: str, min_games: int = 1000) -> List[Dict]:
    """
    Parse 17lands CSV and aggregate card statistics.

    17lands CSVs have one row per game, with columns like:
    - won: True/False (game result)
    - deck_CardName: count in deck
    - opening_hand_CardName: count in opening hand
    - drawn_CardName: count drawn during game

    Args:
        csv_path: Path to CSV file
        min_games: Minimum games played to include card

    Returns:
        List of card statistics dictionaries
    """
    logger.info(f"Parsing {csv_path}...")

    # Aggregate stats by card name
    card_stats = defaultdict(lambda: {
        'games_in_deck': 0,  # Games where card was in deck
        'wins_in_deck': 0,
        'gih_games': 0,      # Games in hand (opening hand OR drawn)
        'gih_wins': 0,
        'oh_games': 0,       # Opening hand games
        'oh_wins': 0,
    })

    rows_processed = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Check column names
        if not reader.fieldnames:
            logger.error("CSV has no columns!")
            return []

        # Extract unique card names from column headers
        # Columns are like: deck_CardName, opening_hand_CardName, drawn_CardName
        card_names = set()
        for col in reader.fieldnames:
            if col.startswith('deck_'):
                card_name = col[5:]  # Remove 'deck_' prefix
                card_names.add(card_name)

        logger.info(f"Found {len(card_names)} unique cards in CSV")

        for row in reader:
            rows_processed += 1

            if rows_processed % 100000 == 0:
                logger.info(f"  Processed {rows_processed:,} rows...")

            # Get game result
            won = row.get('won', 'False') == 'True'

            # Process each card
            for card_name in card_names:
                # Check if card was in deck
                deck_col = f'deck_{card_name}'
                if deck_col in row and row[deck_col] and int(float(row[deck_col] or 0)) > 0:
                    card_stats[card_name]['games_in_deck'] += 1
                    if won:
                        card_stats[card_name]['wins_in_deck'] += 1

                    # Check if in opening hand
                    oh_col = f'opening_hand_{card_name}'
                    if oh_col in row and row[oh_col] and int(float(row[oh_col] or 0)) > 0:
                        card_stats[card_name]['oh_games'] += 1
                        if won:
                            card_stats[card_name]['oh_wins'] += 1

                    # Check if drawn (opening hand OR drawn during game)
                    drawn_col = f'drawn_{card_name}'
                    in_opening_hand = oh_col in row and row[oh_col] and int(float(row[oh_col] or 0)) > 0
                    drawn_later = drawn_col in row and row[drawn_col] and int(float(row[drawn_col] or 0)) > 0

                    if in_opening_hand or drawn_later:
                        card_stats[card_name]['gih_games'] += 1
                        if won:
                            card_stats[card_name]['gih_wins'] += 1

    logger.info(f"✓ Processed {rows_processed:,} rows")

    # Convert to card list
    cards = []
    for card_name, stats in card_stats.items():
        if stats['games_in_deck'] < min_games:
            continue

        # Calculate win rates
        win_rate = stats['wins_in_deck'] / stats['games_in_deck'] if stats['games_in_deck'] > 0 else 0.0
        gih_wr = stats['gih_wins'] / stats['gih_games'] if stats['gih_games'] > 0 else win_rate
        oh_wr = stats['oh_wins'] / stats['oh_games'] if stats['oh_games'] > 0 else win_rate

        cards.append({
            'card_name': card_name,
            'set_code': Path(csv_path).stem.split('_')[1],  # Extract from filename
            'games_played': stats['games_in_deck'],
            'win_rate': win_rate,
            'gih_win_rate': gih_wr,
            'opening_hand_win_rate': oh_wr,
            'iwd': gih_wr - win_rate,
            'last_updated': datetime.now().isoformat()
        })

    logger.info(f"✓ Parsed {len(cards)} cards (min {min_games} games)")
    return cards


def quick_download_sample(set_code: str = "OTJ", max_rows: int = 10000) -> List[Dict]:
    """
    Quick sample download - just get first N rows for testing.

    This is much faster than processing entire dataset.
    """
    url = f"{S3_BASE}/game_data/game_data_public.{set_code}.PremierDraft.csv.gz"

    logger.info(f"Quick sample download: {set_code} (first {max_rows:,} rows)")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Stream and decompress on-the-fly
        card_stats = defaultdict(lambda: {'games': 0, 'wins': 0})
        rows = 0

        with gzip.GzipFile(fileobj=response.raw) as gz:
            reader = csv.DictReader(line.decode('utf-8') for line in gz)

            for row in reader:
                rows += 1
                if rows > max_rows:
                    break

                # Simple aggregation
                if 'card_name' in row:
                    card = row['card_name']
                    card_stats[card]['games'] += 1
                    if row.get('won') == '1':
                        card_stats[card]['wins'] += 1

        # Convert to card list
        cards = []
        for card_name, stats in card_stats.items():
            if stats['games'] >= 10:  # Low threshold for sample
                cards.append({
                    'card_name': card_name,
                    'set_code': set_code,
                    'games_played': stats['games'],
                    'win_rate': stats['wins'] / stats['games'],
                    'gih_win_rate': stats['wins'] / stats['games'],  # Simplified
                    'iwd': 0.0,
                    'last_updated': datetime.now().isoformat()
                })

        logger.info(f"✓ Sample: {len(cards)} cards from {rows} rows")
        return cards

    except Exception as e:
        logger.error(f"Sample download failed: {e}")
        return []


def main():
    """Main function - download and import 17lands data"""
    import time

    logger.info("="*70)
    logger.info("17lands Real Data Downloader (No API Key Required!)")
    logger.info("="*70)

    # Ask user which method
    print("\nChoose download method:")
    print("  1. Quick sample (10k rows, ~100 cards, 1 minute)")
    print("  2. Full download - single set (1M+ rows, 1000+ cards, 10-30 minutes)")
    print(f"  3. Full download - current Standard sets ({len(CURRENT_STANDARD)} sets, ~60-180 minutes)")
    print(f"  4. Full download - all available sets ({len(ALL_SETS)} sets, several hours)")
    print("  5. Cancel")

    choice = input("\nEnter choice (1/2/3/4/5): ").strip()

    if choice == "1":
        # Quick sample
        logger.info("\nDownloading quick sample...")
        cards = quick_download_sample(set_code="OTJ", max_rows=10000)

        if cards:
            db = CardStatsDB()
            db.insert_card_stats(cards)
            db.close()
            logger.info(f"\n✓ Imported {len(cards)} cards to database!")

    elif choice == "2":
        # Full download - single set
        print("\nAvailable set codes:")
        for code, name in sorted(ALL_SETS.items()):
            print(f"  {code}: {name}")

        set_code = input("\nEnter set code [OTJ]: ").strip().upper() or "OTJ"

        if set_code not in ALL_SETS:
            logger.error(f"Invalid set code: {set_code}")
            return

        csv_path = download_card_data(set_code)  # Auto-detects format

        if csv_path:
            cards = parse_card_stats_from_csv(csv_path, min_games=1000)

            if cards:
                db = CardStatsDB()
                db.insert_card_stats(cards)
                db.close()
                logger.info(f"\n✓ Imported {len(cards)} cards to database!")

    elif choice == "3":
        # Full download - current Standard
        logger.info(f"\nDownloading {len(CURRENT_STANDARD)} current Standard sets...")
        logger.info(f"Sets: {', '.join(CURRENT_STANDARD)}")

        confirm = input("\nThis will take 60-180 minutes. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Cancelled.")
            return

        all_cards = []
        db = CardStatsDB()

        for i, set_code in enumerate(CURRENT_STANDARD, 1):
            logger.info(f"\n[{i}/{len(CURRENT_STANDARD)}] Processing {set_code} ({ALL_SETS[set_code]})...")

            csv_path = download_card_data(set_code)  # Auto-detects format

            if csv_path:
                cards = parse_card_stats_from_csv(csv_path, min_games=1000)
                if cards:
                    db.insert_card_stats(cards)
                    all_cards.extend(cards)
                    logger.info(f"✓ {set_code}: Added {len(cards)} cards")

            # Small delay between downloads to be nice to S3
            if i < len(CURRENT_STANDARD):
                time.sleep(2)

        db.close()
        logger.info(f"\n✓ Imported {len(all_cards)} total cards from {len(CURRENT_STANDARD)} sets!")

    elif choice == "4":
        # Full download - all sets
        logger.info(f"\nDownloading ALL {len(ALL_SETS)} available sets...")
        logger.info("This will download:")
        for code, name in sorted(ALL_SETS.items()):
            logger.info(f"  • {code}: {name}")

        confirm = input(f"\nThis will take SEVERAL HOURS and download gigabytes of data. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Cancelled.")
            return

        all_cards = []
        db = CardStatsDB()

        set_codes = sorted(ALL_SETS.keys())

        for i, set_code in enumerate(set_codes, 1):
            logger.info(f"\n[{i}/{len(set_codes)}] Processing {set_code} ({ALL_SETS[set_code]})...")

            csv_path = download_card_data(set_code)  # Auto-detects format

            if csv_path:
                cards = parse_card_stats_from_csv(csv_path, min_games=1000)
                if cards:
                    db.insert_card_stats(cards)
                    all_cards.extend(cards)
                    logger.info(f"✓ {set_code}: Added {len(cards)} cards")
            else:
                logger.warning(f"⚠ {set_code}: Download failed (set may not have public data)")

            # Small delay between downloads
            if i < len(set_codes):
                time.sleep(2)

        db.close()
        logger.info(f"\n✓ Imported {len(all_cards)} total cards from {len(set_codes)} sets!")

    else:
        logger.info("Cancelled.")
        return

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
