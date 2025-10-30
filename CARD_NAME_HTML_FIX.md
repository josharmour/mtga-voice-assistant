# Card Name HTML Tag Fix

## Problem

Card names retrieved from Arena's database contained HTML tags that were being displayed in the GUI and TUI:

- `<nobr>Full-Throttle</nobr> Fanatic` instead of `Full-Throttle Fanatic`
- `<nobr>Bane-Marked</nobr> Leonin` instead of `Bane-Marked Leonin`
- `<sprite="SpriteSheet_MiscIcons" name="arena_a">Demilich` instead of `Demilich`

## Solution

Created a utility function `clean_card_name()` that strips all HTML tags from card names using regex, and applied it at the source where card names are loaded into the cache.

## Changes Made

### 1. Added Utility Function (advisor.py, lines 67-91)

```python
def clean_card_name(name: str) -> str:
    """
    Remove HTML tags from card names.

    Some card names from Arena's database contain HTML tags like <nobr> and </nobr>
    that need to be stripped for proper display and matching.
    """
    if not name:
        return name
    return re.sub(r'<[^>]+>', '', name)
```

### 2. Applied Cleaning in ArenaCardDatabase._load_all_cards_into_cache() (line 876)

Changed:
```python
"name": row[1],
```

To:
```python
"name": clean_card_name(row[1]),
```

### 3. Applied Cleaning in ArenaCardDatabase._fetch_from_scryfall() (line 966)

Changed:
```python
"name": card_data.get("name", f"Unknown({grp_id})"),
```

To:
```python
"name": clean_card_name(card_data.get("name", f"Unknown({grp_id})")),
```

## Cleanup Scripts

### clean_card_cache.py

A one-time cleanup script that:
- Loads the existing `card_cache.json`
- Strips HTML tags from all card names
- Backs up the original cache to `card_cache.json.backup`
- Saves the cleaned cache

**Usage:**
```bash
python3 clean_card_cache.py
```

**Results:**
- Cleaned 674 card names
- Removed `<nobr>`, `</nobr>`, `<sprite>`, and `<i>` tags
- Preserved all other card data

### test_clean_card_names.py

Unit tests for the `clean_card_name()` function:
- Tests various HTML tag patterns
- Verifies edge cases (empty strings, None, normal names)
- All tests pass

**Usage:**
```bash
python3 test_clean_card_names.py
```

## Impact

### What's Fixed:
- ✅ Card names display cleanly in GUI (Tkinter)
- ✅ Card names display cleanly in TUI (Curses)
- ✅ Card names display cleanly in CLI mode
- ✅ Draft advisor card lists show clean names
- ✅ Deck builder card lists show clean names
- ✅ All future card lookups will be automatically cleaned

### What Still Works:
- ✅ Card database lookups (by grpId)
- ✅ Scryfall API fallback
- ✅ Card statistics from RAG system
- ✅ Card metadata queries
- ✅ All existing functionality unchanged

## Files Modified

1. **advisor.py** - Added `clean_card_name()` function and applied it in two methods
2. **card_cache.json** - Cleaned by running `clean_card_cache.py` (backed up to `.backup`)

## Files Created

1. **clean_card_cache.py** - One-time cache cleanup script
2. **test_clean_card_names.py** - Unit tests for the cleaning function
3. **CARD_NAME_HTML_FIX.md** - This documentation

## Verification

The fix was verified by:

1. Running unit tests - all passed
2. Cleaning the existing cache - 674 names cleaned
3. Checking for remaining HTML tags - none found
4. Verifying specific problematic cards:
   - ✅ `Full-Throttle Fanatic` (was `<nobr>Full-Throttle</nobr> Fanatic`)
   - ✅ `Bane-Marked Leonin` (was `<nobr>Bane-Marked</nobr> Leonin`)
   - ✅ `Half-Elf Monk` (was `<nobr>Half-Elf</nobr> Monk`)

## Future Maintenance

No ongoing maintenance required. The fix is applied at the source:

1. When loading from Arena's database → names are cleaned
2. When fetching from Scryfall API → names are cleaned
3. When saving to cache → clean names are saved
4. When loading from cache → names are already clean

All future card additions will automatically have clean names.
