#!/usr/bin/env python3
"""
Build unified_cards.db from MTGA's Raw_CardDatabase and 17lands data.

This script:
1. Finds MTGA's Raw_CardDatabase SQLite file
2. Extracts grpId → card name mappings
3. Enriches with Scryfall data (if available)
4. Creates unified_cards.db for the application
"""

import sqlite3
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_raw_card_database() -> Optional[Path]:
    """Find MTGA's Raw_CardDatabase file."""

    search_paths = [
        # Steam on Linux
        Path.home() / ".local/share/Steam/steamapps/common/MTGA/MTGA_Data/Downloads/Raw",
        # Windows paths
        Path("C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw"),
        Path("C:/Program Files (x86)/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw"),
        # Wine/Bottles/Lutris paths
        Path.home() / ".wine/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw",
        Path.home() / ".var/app/com.usebottles.bottles/data/bottles/bottles/MTGA/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw",
    ]

    for search_path in search_paths:
        if search_path.exists():
            pattern = str(search_path / "Raw_CardDatabase_*.mtga")
            files = glob.glob(pattern)
            if files:
                # Return the most recent file
                return Path(max(files, key=os.path.getmtime))

    # Try a system-wide search as fallback
    logger.info("Searching for Raw_CardDatabase file system-wide...")
    try:
        result = subprocess.run(
            ["find", str(Path.home()), "-name", "Raw_CardDatabase*.mtga", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split('\n')
            if files:
                return Path(files[0])
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"System-wide search failed: {e}")

    return None

def extract_arena_cards(raw_db_path: Path) -> Dict[int, Dict]:
    """Extract card data from MTGA's Raw_CardDatabase."""

    logger.info(f"Extracting cards from {raw_db_path}")

    conn = sqlite3.connect(raw_db_path)
    cursor = conn.cursor()

    # Get all cards with their English names
    cursor.execute("""
        SELECT
            c.GrpId,
            l.Loc as name,
            c.Rarity,
            c.Colors,
            c.ColorIdentity,
            c.Types,
            c.Subtypes,
            c.Supertypes,
            c.Power,
            c.Toughness,
            c.CollectorNumber,
            c.ExpansionCode,
            c.IsToken
        FROM Cards c
        LEFT JOIN Localizations_enUS l ON c.TitleId = l.LocId
        WHERE c.GrpId IS NOT NULL
    """)

    cards = {}
    for row in cursor.fetchall():
        grp_id = row[0]
        name = row[1] or f"Unknown({grp_id})"

        # Clean up HTML tags in names
        name = name.replace("<nobr>", "").replace("</nobr>", "")

        cards[grp_id] = {
            "grpId": grp_id,
            "name": name,
            "printed_name": name,  # For compatibility
            "rarity": row[2],
            "colors": row[3] or "",
            "color_identity": row[4] or "",
            "types": row[5] or "",
            "subtypes": row[6] or "",
            "supertypes": row[7] or "",
            "power": row[8],
            "toughness": row[9],
            "collector_number": row[10],
            "set_code": row[11],
            "is_token": bool(row[12])
        }

    conn.close()
    logger.info(f"Extracted {len(cards)} cards from Arena database")
    return cards

def load_scryfall_cache() -> Dict[str, Dict]:
    """Load Scryfall data from cache if available."""

    cache_path = Path("card_cache.json")
    if not cache_path.exists():
        logger.info("No Scryfall cache found, will use Arena data only")
        return {}

    logger.info("Loading Scryfall cache...")
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Index by name for easy lookup
            scryfall_by_name = {}
            for card_data in data:
                if isinstance(card_data, dict) and 'name' in card_data:
                    scryfall_by_name[card_data['name'].lower()] = card_data
            logger.info(f"Loaded {len(scryfall_by_name)} cards from Scryfall cache")
            return scryfall_by_name
    except Exception as e:
        logger.error(f"Failed to load Scryfall cache: {e}")
        return {}

def enrich_with_scryfall(arena_cards: Dict[int, Dict], scryfall_data: Dict[str, Dict]) -> Dict[int, Dict]:
    """Enrich Arena cards with Scryfall data where available."""

    if not scryfall_data:
        return arena_cards

    logger.info("Enriching cards with Scryfall data...")
    enriched_count = 0

    for grp_id, card in arena_cards.items():
        card_name_lower = card['name'].lower()

        if card_name_lower in scryfall_data:
            scryfall_card = scryfall_data[card_name_lower]

            # Add Scryfall fields that Arena doesn't have
            if 'oracle_text' in scryfall_card:
                card['oracle_text'] = scryfall_card['oracle_text']
            if 'mana_cost' in scryfall_card:
                card['mana_cost'] = scryfall_card['mana_cost']
            if 'cmc' in scryfall_card:
                card['cmc'] = scryfall_card['cmc']
            if 'type_line' in scryfall_card:
                card['type_line'] = scryfall_card['type_line']
            if 'keywords' in scryfall_card:
                card['keywords'] = ','.join(scryfall_card['keywords'])

            enriched_count += 1

    logger.info(f"Enriched {enriched_count} cards with Scryfall data")
    return arena_cards

def convert_rarity(rarity_num: Optional[int]) -> str:
    """Convert Arena's numeric rarity to string."""
    rarity_map = {
        0: "common",
        1: "common",
        2: "uncommon",
        3: "rare",
        4: "mythic",
        5: "special"
    }
    return rarity_map.get(rarity_num, "unknown")

def convert_colors(color_str: Optional[str]) -> str:
    """Convert Arena's color format to standard WUBRG."""
    if not color_str:
        return ""

    color_map = {
        "1": "W",
        "2": "U",
        "3": "B",
        "4": "R",
        "5": "G"
    }

    colors = []
    for char in str(color_str).split(','):
        if char.strip() in color_map:
            colors.append(color_map[char.strip()])

    return "".join(colors)

def create_unified_database(cards: Dict[int, Dict], output_path: Path):
    """Create the unified_cards.db SQLite database."""

    logger.info(f"Creating unified database at {output_path}")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old database if it exists
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create table with all fields we might need
    cursor.execute("""
        CREATE TABLE cards (
            grpId INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            printed_name TEXT,
            oracle_text TEXT,
            mana_cost TEXT,
            cmc REAL,
            type_line TEXT,
            color_identity TEXT,
            colors TEXT,
            keywords TEXT,
            power TEXT,
            toughness TEXT,
            rarity TEXT,
            set_code TEXT,
            collector_number TEXT,
            types TEXT,
            subtypes TEXT,
            supertypes TEXT,
            is_token BOOLEAN,
            is_reskin BOOLEAN DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indices for common queries
    cursor.execute("CREATE INDEX idx_name ON cards(name)")
    cursor.execute("CREATE INDEX idx_set ON cards(set_code)")
    cursor.execute("CREATE INDEX idx_rarity ON cards(rarity)")

    # Insert cards
    for grp_id, card in cards.items():
        # Convert numeric values to appropriate formats
        rarity = convert_rarity(card.get('rarity'))
        colors = convert_colors(card.get('colors'))
        color_identity = convert_colors(card.get('color_identity', card.get('colors')))

        cursor.execute("""
            INSERT INTO cards (
                grpId, name, printed_name, oracle_text, mana_cost, cmc,
                type_line, color_identity, colors, keywords,
                power, toughness, rarity, set_code, collector_number,
                types, subtypes, supertypes, is_token, is_reskin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grp_id,
            card['name'],
            card.get('printed_name', card['name']),
            card.get('oracle_text', ''),
            card.get('mana_cost', ''),
            card.get('cmc', 0),
            card.get('type_line', ''),
            color_identity,
            colors,
            card.get('keywords', ''),
            card.get('power', ''),
            card.get('toughness', ''),
            rarity,
            card.get('set_code', ''),
            card.get('collector_number', ''),
            card.get('types', ''),
            card.get('subtypes', ''),
            card.get('supertypes', ''),
            card.get('is_token', False),
            0  # is_reskin - default to not a reskin
        ))

    # Add metadata table for version tracking
    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cursor.execute("INSERT INTO metadata (key, value) VALUES ('version', '1.0')")
    cursor.execute("INSERT INTO metadata (key, value) VALUES ('card_count', ?)", (len(cards),))

    conn.commit()
    conn.close()

    logger.info(f"Created database with {len(cards)} cards")

def update_17lands_data():
    """Update 17lands statistics data."""
    logger.info("Updating 17lands data...")
    try:
        result = subprocess.run(
            [sys.executable, "manage_data.py", "--update-17lands"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("17lands data updated successfully")
        else:
            logger.warning(f"Failed to update 17lands data: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not update 17lands data: {e}")

def build_database():
    """Main function to build the unified card database."""

    print("\n" + "="*60)
    print("  Building Unified Card Database")
    print("="*60 + "\n")

    # Step 1: Find Raw_CardDatabase
    print("Step 1: Locating MTGA card database...")
    raw_db_path = find_raw_card_database()

    if not raw_db_path:
        print("\n❌ ERROR: Could not find MTGA's Raw_CardDatabase file.")
        print("\nPlease ensure Magic: The Gathering Arena is installed.")
        print("Expected locations:")
        print("  - Linux: ~/.local/share/Steam/steamapps/common/MTGA/")
        print("  - Windows: C:/Program Files/Wizards of the Coast/MTGA/")
        print("\nYou can also manually specify the path by editing this script.")
        return 1

    print(f"✅ Found: {raw_db_path}")
    print(f"   Size: {raw_db_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 2: Extract Arena cards
    print("\nStep 2: Extracting card data from Arena...")
    arena_cards = extract_arena_cards(raw_db_path)
    print(f"✅ Extracted {len(arena_cards)} cards")

    # Step 3: Load and apply Scryfall data (optional)
    print("\nStep 3: Loading Scryfall enrichment data...")
    scryfall_data = load_scryfall_cache()
    if scryfall_data:
        arena_cards = enrich_with_scryfall(arena_cards, scryfall_data)
        print(f"✅ Enriched with Scryfall data")
    else:
        print("⚠️  No Scryfall cache found, using Arena data only")

    # Step 4: Create unified database
    print("\nStep 4: Creating unified database...")
    output_path = Path("data/unified_cards.db")
    create_unified_database(arena_cards, output_path)
    print(f"✅ Created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 5: Update 17lands data (optional)
    print("\nStep 5: Updating 17lands statistics...")
    update_17lands_data()

    print("\n" + "="*60)
    print("  ✅ Database build complete!")
    print("="*60)
    print("\nYou can now run the advisor application:")
    print("  python advisor.py")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(build_database())
