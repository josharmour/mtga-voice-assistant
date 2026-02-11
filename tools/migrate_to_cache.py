#!/usr/bin/env python3
"""
Migrate existing 17lands data from ~/ArenaBot/data/17lands_data/ to the cache structure.

This creates symlinks in the cache pointing to the source compressed files,
avoiding duplication while making the data available to the deck builder.
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def migrate_to_cache():
    """Create symlinks from cache to ArenaBot data directory"""

    source_dir = Path.home() / "ArenaBot" / "data" / "17lands_data"
    cache_compressed = Path("data/17lands_cache/compressed")

    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        return

    cache_compressed.mkdir(parents=True, exist_ok=True)

    # Find all Premier Draft game_data files
    pattern = "game_data_public.*.PremierDraft.csv.gz"
    game_data_files = list(source_dir.glob(pattern))

    if not game_data_files:
        logging.warning(f"No game_data files found in {source_dir}")
        return

    logging.info(f"Found {len(game_data_files)} game_data files\n")

    linked = 0
    skipped = 0

    for source_file in sorted(game_data_files):
        # Extract set code
        parts = source_file.stem.split('.')
        if len(parts) < 3:
            continue

        set_code = parts[1]
        format_type = parts[2]

        # Target filename in cache
        cache_file = cache_compressed / f"{set_code}_{format_type}.csv.gz"

        if cache_file.exists():
            if cache_file.is_symlink():
                logging.info(f"✓ {set_code:6s} - Already linked")
            else:
                logging.info(f"✓ {set_code:6s} - Already exists (real file)")
            skipped += 1
            continue

        try:
            # Create symlink
            os.symlink(source_file.absolute(), cache_file)
            size_mb = source_file.stat().st_size / (1024*1024)
            logging.info(f"✅ {set_code:6s} - Linked ({size_mb:.1f} MB)")
            linked += 1
        except Exception as e:
            logging.error(f"❌ {set_code:6s} - Error: {e}")

    logging.info(f"\n{'='*60}")
    logging.info(f"Summary:")
    logging.info(f"  Linked: {linked}")
    logging.info(f"  Skipped (already exists): {skipped}")
    logging.info(f"  Total sets available: {linked + skipped}")
    logging.info(f"{'='*60}")
    logging.info(f"\nNote: Compressed files are symlinked, not copied.")
    logging.info(f"Decompressed versions will be created in cache on first use.")

if __name__ == "__main__":
    logging.info("17Lands Data Migration Tool")
    logging.info("="*60)
    logging.info("Creating symlinks from cache to ArenaBot data...")
    logging.info("="*60 + "\n")

    migrate_to_cache()
