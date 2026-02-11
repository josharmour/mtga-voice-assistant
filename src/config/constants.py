"""
Magic: The Gathering set constants for MTGA Voice Advisor.

These sets are fetched dynamically from Scryfall API where possible,
with fallback to hardcoded values if the API is unavailable.
"""

import logging
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)

# Fallback sets (update these manually if Scryfall API is unavailable)
# Last updated: November 2025 - please verify current Standard sets at https://scryfall.com/docs/api/sets
_FALLBACK_CURRENT_STANDARD = [
    "ONE", "MOM", "MAT", "LTR", "WOE", "LCI", "MKM", "OTJ", "BLB", "MH3"
]

_FALLBACK_ALL_SETS = [
    "ONE", "MOM", "MAT", "LTR", "WOE", "LCI", "MKM", "OTJ", "BLB", "MH3"
]


@lru_cache(maxsize=1)
def _fetch_standard_sets_from_scryfall():
    """
    Fetch current Standard legal sets from Scryfall API.

    Note: This fetches PAPER Standard. MTGA Standard may differ.
    For the most accurate MTGA-specific set list, check:
    - MTGA in-game collection/deck builder
    - https://scryfall.com/ and filter by "MTGA" format
    - 17lands.com for current draft sets

    Returns list of set codes currently legal in Standard format.
    Falls back to hardcoded list if API is unavailable.
    """
    try:
        response = requests.get("https://api.scryfall.com/sets", timeout=5)
        response.raise_for_status()

        standard_sets = []
        for set_obj in response.json().get("data", []):
            # Only include sets that are:
            # 1. Legal in Standard format
            # 2. Supported in MTGA (digital = True means it's in MTGA)
            legalities = set_obj.get("legalities", {})
            is_digital = set_obj.get("digital", False)

            if legalities.get("standard") == "legal" and is_digital:
                standard_sets.append(set_obj.get("code", "").upper())

        if standard_sets:
            logger.debug(f"Fetched {len(standard_sets)} MTGA Standard sets from Scryfall: {sorted(standard_sets)}")
            return sorted(standard_sets)
    except Exception as e:
        logger.warning(f"Could not fetch Standard sets from Scryfall API: {e}. Using fallback.")

    return _FALLBACK_CURRENT_STANDARD


# Dynamically fetch current Standard sets, fall back to hardcoded if needed
try:
    CURRENT_STANDARD = _fetch_standard_sets_from_scryfall()
except Exception:
    CURRENT_STANDARD = _FALLBACK_CURRENT_STANDARD
    logger.warning("Using fallback Standard set list. Please update constants.py manually.")

# All known MTGA sets (broader than just current Standard)
# Used for historical analysis and data management
ALL_SETS = _FALLBACK_ALL_SETS