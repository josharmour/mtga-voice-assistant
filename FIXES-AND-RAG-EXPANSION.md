# Fixes and RAG Database Expansion Guide

**Date**: 2025-10-28

---

## Issue 1: TUI Color System Fixed ✅

### The Problem

The TUI was crashing with:
```python
TypeError: 'str' object cannot be interpreted as an integer
```

**Root cause:** The `_output()` method was passing string color names ("green", "blue", etc.) to the TUI's `add_message()` method, but `curses.color_pair()` expects integer color pair IDs.

### The Fix

Added a color mapping system to the `AdvisorTUI` class:

```python
# In AdvisorTUI.__init__()
self.color_map = {
    "green": 1,
    "cyan": 2,
    "yellow": 3,
    "red": 4,
    "blue": 5,
    "white": 6,
}

# In add_message()
if isinstance(color, str):
    color = self.color_map.get(color, 6)  # Default to white
```

**Files modified:**
- `advisor.py` lines 1803-1871

**Status:** ✅ Fixed and tested

---

## Issue 2: RAG Database Expansion

### Current Situation

The RAG system currently has only **~25 sample cards** for demonstration purposes. The card statistics database (`data/card_stats.db`) is populated by `load_17lands_data.py`, which contains sample data for common cards like:

- Lightning Bolt
- Murder
- Llanowar Elves
- Serra Angel
- Jace, the Mind Sculptor
- ~20 more

### Why Only 25 Cards?

The `load_17lands_data.py` file uses **sample/mock data** because:

1. **17lands.com requires API authentication** - Real data needs an API key
2. **Dataset is massive** - Full 17lands data is 100MB+ per set
3. **Rate limiting** - API has request limits
4. **Sample is sufficient for testing** - Demonstrates the RAG system works

### How to Expand the Database

You have **three options** to add more cards:

---

## Option 1: Add More Sample Cards Manually (Easy)

Edit `load_17lands_data.py` and add more cards to the `sample_cards` list in `load_sample_data()`:

```python
def load_sample_data() -> List[Dict]:
    sample_cards = [
        # ... existing cards ...

        # Add your cards here:
        {
            'card_name': 'Counterspell',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 45000,
            'win_rate': 0.575,
            'avg_taken_at': 3.5,
            'games_in_hand': 28000,
            'gih_win_rate': 0.615,
            'opening_hand_win_rate': 0.600,
            'drawn_win_rate': 0.620,
            'ever_drawn_win_rate': 0.615,
            'never_drawn_win_rate': 0.535,
            'alsa': 7.2,
            'ata': 3.5,
            'iwd': 0.040,
            'last_updated': datetime.now().isoformat()
        },
        # Add more cards...
    ]
    return sample_cards
```

**Steps:**

1. Edit `load_17lands_data.py`
2. Add card entries (copy the format above)
3. Run: `python3 load_17lands_data.py`
4. Cards are inserted into `data/card_stats.db`

**Pro:** Simple, no API keys needed
**Con:** Tedious for many cards, stats are made up

---

## Option 2: Use Real 17lands Data (Advanced)

### Step 1: Get API Access

1. Go to https://www.17lands.com/
2. Create an account
3. Request API access (may require Patreon subscription)
4. Get your API key

### Step 2: Modify load_17lands_data.py

Update the `download_17lands_data()` function:

```python
def download_17lands_data(set_code: str, format: str = "PremierDraft") -> List[Dict]:
    """Download real card data from 17lands API"""

    API_KEY = "your_api_key_here"  # Add your key

    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{LANDS_API_URL}?set={set_code}&format={format}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    # Transform API response to card stats format
    cards = []
    for card in data['cards']:
        cards.append({
            'card_name': card['name'],
            'set_code': set_code,
            'color': card['color'],
            'rarity': card['rarity'],
            'games_played': card['games_played'],
            'win_rate': card['win_rate'],
            'gih_win_rate': card['gih_win_rate'],
            'iwd': card['improvement_when_drawn'],
            # ... map other fields
        })

    return cards
```

### Step 3: Download Multiple Sets

```python
# In main()
sets = ["ONE", "MOM", "WOE", "LCI", "MKM"]  # Recent sets

all_cards = []
for set_code in sets:
    logger.info(f"Downloading {set_code}...")
    cards = download_17lands_data(set_code)
    all_cards.extend(cards)
    time.sleep(2)  # Rate limiting

db.insert_card_stats(all_cards)
```

**Pro:** Real competitive data, thousands of cards
**Con:** Requires API key, may cost money, large downloads

---

## Option 3: Use Public CSV Datasets (Intermediate)

17lands provides public CSV datasets without authentication:

### Step 1: Download CSV

Visit: https://www.17lands.com/public_datasets

Download CSVs for recent sets (e.g., `game_data_public.ONE.PremierDraft.csv`)

### Step 2: Parse CSV

Add this function to `load_17lands_data.py`:

```python
def load_from_csv(csv_path: str) -> List[Dict]:
    """Load card data from 17lands public CSV"""
    import csv
    from collections import defaultdict

    # Aggregate stats by card name
    card_stats = defaultdict(lambda: {
        'games_played': 0,
        'wins': 0,
        'gih_games': 0,
        'gih_wins': 0,
    })

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_name = row['card_name']
            card_stats[card_name]['games_played'] += 1
            if row['won'] == '1':
                card_stats[card_name]['wins'] += 1
            if row['drawn'] == '1':
                card_stats[card_name]['gih_games'] += 1
                if row['won'] == '1':
                    card_stats[card_name]['gih_wins'] += 1

    # Convert to card list
    cards = []
    for card_name, stats in card_stats.items():
        if stats['games_played'] > 100:  # Minimum threshold
            win_rate = stats['wins'] / stats['games_played']
            gih_wr = stats['gih_wins'] / stats['gih_games'] if stats['gih_games'] > 0 else win_rate

            cards.append({
                'card_name': card_name,
                'set_code': 'UNKNOWN',  # Parse from CSV filename
                'games_played': stats['games_played'],
                'win_rate': win_rate,
                'gih_win_rate': gih_wr,
                'iwd': gih_wr - win_rate,
                # ... other fields
            })

    return cards
```

### Step 3: Run Import

```python
# In main()
csv_files = [
    'data/game_data_public.ONE.PremierDraft.csv',
    'data/game_data_public.MOM.PremierDraft.csv',
]

all_cards = []
for csv_file in csv_files:
    logger.info(f"Loading {csv_file}...")
    cards = load_from_csv(csv_file)
    all_cards.extend(cards)

db.insert_card_stats(all_cards)
```

**Pro:** Free, no API key, large dataset
**Con:** Need to download/process large CSVs manually

---

## Current Database Status

Check what's in the database:

```bash
# Count cards in database
sqlite3 data/card_stats.db "SELECT COUNT(*) FROM card_stats;"

# List all cards
sqlite3 data/card_stats.db "SELECT card_name, games_played, win_rate FROM card_stats LIMIT 20;"

# Check specific card
sqlite3 data/card_stats.db "SELECT * FROM card_stats WHERE card_name='Lightning Bolt';"
```

---

## RAG System Architecture

The RAG system has **two databases**:

### 1. Card Statistics Database (SQLite)
- **File:** `data/card_stats.db`
- **Purpose:** Card win rates, pick rates, performance metrics
- **Populated by:** `load_17lands_data.py`
- **Current size:** ~25 cards (sample)
- **Can expand to:** Thousands of cards

### 2. Rules Vector Database (ChromaDB)
- **Directory:** `data/chromadb/`
- **Purpose:** Semantic search over MTG comprehensive rules
- **Populated by:** `test_rag.py` (calls `RAGSystem.initialize_rules()`)
- **Current size:** ~3000 rules from `data/MagicCompRules.txt`
- **Status:** Complete (all comprehensive rules)

**The rules database is already fully populated!** You only need to expand the card statistics database.

---

## Quick Expansion Steps

### For 50-100 Common Cards (5 minutes)

1. Open `load_17lands_data.py`
2. Copy-paste the card entry format
3. Add your favorite cards (make up reasonable stats)
4. Run: `python3 load_17lands_data.py`

### For 1000+ Cards with Real Data (1 hour)

1. Download public CSVs from 17lands.com
2. Implement the CSV parser (Option 3 above)
3. Run import script
4. Verify with `sqlite3`

### For Complete Database (Requires API)

1. Get 17lands API key
2. Implement API downloader (Option 2 above)
3. Download all recent sets
4. Import to database

---

## Checking RAG Enhancement

To see if cards are being enhanced by RAG:

```python
from rag_advisor import RAGSystem, CardStatsDB

# Check card database
db = CardStatsDB()
stats = db.get_card_stats("Lightning Bolt")
if stats:
    print(f"✓ Lightning Bolt in database")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  GIH WR: {stats['gih_win_rate']:.1%}")
else:
    print("✗ Lightning Bolt not in database")
db.close()

# Check rules database
rag = RAGSystem()
rules = rag.query_rules("combat damage", top_k=3)
if rules:
    print(f"\n✓ Rules database working ({len(rules)} results)")
    for r in rules:
        print(f"  Rule {r['id']}: {r['text'][:60]}...")
else:
    print("✗ Rules database not initialized")
rag.close()
```

---

## What Gets Enhanced by RAG?

When RAG is enabled, the AI advisor gets:

1. **Card Statistics** (from SQLite):
   ```
   Lightning Bolt:
   - Win rate: 58.3%
   - GIH win rate: 62.1%
   - IWD: +3.8% (high impact)
   ```

2. **Relevant Rules** (from ChromaDB):
   ```
   Rule 510.1: Combat damage step - creatures deal damage
   Rule 704.5g: State-based action for lethal damage
   ```

3. **Strategic Context**:
   - High win rate cards are prioritized
   - Rules relevant to current phase are included
   - Counter-play suggestions based on game state

---

## Recommendations

### For Testing/Development (Now)

**Use the current 25-card sample** - It's sufficient to:
- Test RAG integration
- Verify prompt enhancement
- Develop advisor features

### For Production Use (Later)

**Option 1 (Easiest):** Add 50-100 popular cards manually to `load_17lands_data.py`

**Option 2 (Best):** Download public CSVs and import 5000+ cards with real stats

**Option 3 (Most Complete):** Get API access and import full database

---

## Summary

✅ **TUI color fix:** Implemented and working
✅ **RAG rules database:** Complete (3000+ rules)
⚠️ **RAG card database:** Only 25 cards (sample)

**To expand card database:**
1. Easy: Edit `load_17lands_data.py` (add more sample cards)
2. Medium: Download CSV, parse, import
3. Hard: Get API key, download all sets

**Current database is fine for testing.** Expand when you want more accurate card performance data in advisor recommendations.

---

END OF DOCUMENT
