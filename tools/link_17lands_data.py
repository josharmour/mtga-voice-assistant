#!/usr/bin/env python3
"""
Link existing 17lands game data from ~/ArenaBot/data/17Lands_data/
to the format expected by the deck builder.

Expected source format: game_data_public.{SET}.PremierDraft.csv.gz
Target format: data/17lands_{SET}_PremierDraft.csv (decompressed)
"""

import gzip
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_17lands_data(source_dir: Path, target_dir: Path):
    """Process and link 17lands game data files"""

    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all game_data files
    pattern = "game_data_public.*.PremierDraft.csv.gz"
    game_data_files = list(source_dir.glob(pattern))

    if not game_data_files:
        logging.warning(f"No game_data files found matching {pattern}")
        return

    logging.info(f"Found {len(game_data_files)} game_data files to process\n")

    processed = 0
    skipped = 0
    errors = 0

    for source_file in sorted(game_data_files):
        # Extract set code from filename
        # Format: game_data_public.SET.PremierDraft.csv.gz
        parts = source_file.stem.split('.')
        if len(parts) < 3:
            logging.warning(f"Skipping unexpected filename format: {source_file.name}")
            continue

        set_code = parts[1]
        target_file = target_dir / f"17lands_{set_code}_PremierDraft.csv"

        # Skip if already exists and is newer than source
        if target_file.exists():
            source_mtime = source_file.stat().st_mtime
            target_mtime = target_file.stat().st_mtime
            if target_mtime >= source_mtime:
                logging.info(f"✓ {set_code:6s} - Already up to date ({target_file.stat().st_size / (1024*1024):.1f} MB)")
                skipped += 1
                continue

        try:
            logging.info(f"Processing {set_code}...")

            # Decompress file
            with gzip.open(source_file, 'rb') as f_in:
                with open(target_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            size_mb = target_file.stat().st_size / (1024*1024)
            logging.info(f"✅ {set_code:6s} - Decompressed {size_mb:.1f} MB → {target_file.name}")
            processed += 1

        except Exception as e:
            logging.error(f"❌ {set_code:6s} - Error: {e}")
            errors += 1

    logging.info(f"\n{'='*60}")
    logging.info(f"Summary:")
    logging.info(f"  Processed: {processed}")
    logging.info(f"  Skipped (already up to date): {skipped}")
    logging.info(f"  Errors: {errors}")
    logging.info(f"  Total available sets: {processed + skipped}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    source_dir = Path.home() / "ArenaBot" / "data" / "17lands_data"
    target_dir = Path(__file__).parent.parent / "data"

    logging.info(f"17Lands Data Processor")
    logging.info(f"{'='*60}")
    logging.info(f"Source: {source_dir}")
    logging.info(f"Target: {target_dir}")
    logging.info(f"{'='*60}\n")

    process_17lands_data(source_dir, target_dir)
