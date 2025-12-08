#!/usr/bin/env python3
"""
Diagnose Arena Card Database.
Checks if the unified_cards.db exists, is readable, and contains data.
"""
import sqlite3
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("data/unified_cards.db")

def check_database():
    if not DB_PATH.exists():
        logger.error(f"❌ Database not found at {DB_PATH}")
        return

    logger.info(f"✅ Database found at {DB_PATH}")
    logger.info(f"   Size: {DB_PATH.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check card count
        cursor.execute("SELECT count(*) FROM cards")
        count = cursor.fetchone()[0]
        logger.info(f"✅ Total cards in database: {count}")

        # Check metadata
        try:
            cursor.execute("SELECT value FROM metadata WHERE key='version'")
            version = cursor.fetchone()
            if version:
                logger.info(f"   Database Version: {version[0]}")
            
            cursor.execute("SELECT value FROM metadata WHERE key='last_updated'") # Hypothetical key, might not exist
            last_updated = cursor.fetchone()
            if last_updated:
               logger.info(f"   Last Updated: {last_updated[0]}")
        except sqlite3.OperationalError:
            logger.warning("⚠️  Metadata table might be missing or incomplete.")

        # Check for a known common card (e.g., 'Mountain')
        cursor.execute("SELECT grpId, name, set_code FROM cards WHERE name='Mountain' LIMIT 1")
        card = cursor.fetchone()
        if card:
            logger.info(f"✅ Connectivity check passed (Found '{card['name']}' from '{card['set_code']}', ID: {card['grpId']})")
        else:
            logger.warning("⚠️  Could not find 'Mountain' in database - this is unusual.")

        conn.close()

    except sqlite3.Error as e:
        logger.error(f"❌ Database error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("-" * 40)
    print("  Arena Card Database Diagnostic")
    print("-" * 40)
    check_database()
    print("-" * 40)
