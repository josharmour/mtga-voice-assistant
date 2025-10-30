# Complete RAG System Architecture

**Date**: 2025-10-28
**Status**: ✅ COMPLETE - Three-database RAG system

---

## Overview

The MTGA Voice Advisor now has **three complementary databases** for intelligent card advice:

1. **Card Statistics** (game performance)
2. **Card Metadata** (attributes & types)
3. **Rules Knowledge** (MTG comprehensive rules)

---

## Three-Database Architecture

### 1. Card Statistics DB (`card_stats.db`)

**Purpose**: Game performance data from 17lands

**Contains**:
- Win rates (overall, GIH, opening hand)
- Games played
- Draft pick data (ALSA, ATA)
- Improvement When Drawn (IWD)

**Example Query**:
```python
from rag_advisor import CardStatsDB

db = CardStatsDB()
stats = db.get_card_stats("Lightning Bolt")
print(f"Win rate: {stats['win_rate']:.1%}")
print(f"GIH WR: {stats['gih_win_rate']:.1%}")
print(f"IWD: {stats['iwd']:.1%}")
db.close()
```

**Output**:
```
Win rate: 58.3%
GIH WR: 62.1%
IWD: +3.8%
```

### 2. Card Metadata DB (`card_metadata.db`) **← NEW!**

**Purpose**: Card attributes from 17lands

**Contains**:
- Card colors (R, U, B, G, W, colorless)
- Mana cost (CMC)
- Card types (Creature, Instant, Sorcery, etc.)
- Rarity (common, uncommon, rare, mythic)
- Keyword abilities

**Example Query**:
```python
from rag_advisor import CardMetadataDB

db = CardMetadataDB()
meta = db.get_card_metadata("Lightning Bolt")
print(f"Mana value: {meta['mana_value']}")
print(f"Color: {meta['color_identity']}")
print(f"Types: {meta['types']}")
print(f"Rarity: {meta['rarity']}")
db.close()
```

**Output**:
```
Mana value: 1
Color: R
Types: Instant
Rarity: rare
```

**Advanced Search**:
```python
# Find all 2-mana blue creatures
creatures = db.search_cards(
    mana_value=2,
    color='U',
    types='Creature',
    limit=10
)
for card in creatures:
    print(f"{card['name']} - {card['types']}")
```

**Output**:
```
Merfolk Looter - Creature - Merfolk Rogue
Dragon's Eye Savants - Creature - Human Wizard
Jeskai Elder - Creature - Human Monk
...
```

### 3. Rules Vector DB (`chromadb/`)

**Purpose**: MTG Comprehensive Rules for rules lookups

**Contains**:
- 3000+ rules from MagicCompRules.txt
- Semantic vector embeddings
- Fast similarity search

**Example Query**:
```python
from rag_advisor import RAGSystem

rag = RAGSystem()
rules = rag.query_rules("combat damage", top_k=3)
for rule in rules:
    print(f"Rule {rule['id']}: {rule['text'][:60]}...")
```

**Output**:
```
Rule 510.1: Combat damage step - creatures deal damage...
Rule 704.5g: State-based action for lethal damage...
Rule 510.2: Combat damage assignment order...
```

---

## How They Work Together

When the AI advisor encounters a card, it queries **all three databases**:

### Example: "Lightning Bolt"

**Query 1 - Statistics**:
```python
stats = card_stats_db.get_card_stats("Lightning Bolt")
```
Returns:
- Win rate: 58.3%
- GIH WR: 62.1%
- IWD: +3.8% (high impact)

**Query 2 - Metadata**:
```python
meta = card_metadata_db.get_card_metadata("Lightning Bolt")
```
Returns:
- Mana value: 1
- Color: R (red)
- Types: Instant
- Rarity: rare

**Query 3 - Rules**:
```python
rules = rag_system.query_rules("instant damage")
```
Returns:
- Instant spells can be cast any time
- Damage resolves immediately
- Can target creatures or players

**Combined AI Response**:
> "Lightning Bolt is a 1-mana red instant with a 58% win rate. It's highly impactful (IWD +3.8%) because you can cast it instantly at any time to deal 3 damage. Consider using it to remove their 2 or 3-toughness creatures, or save it for direct player damage in the late game."

---

## Setup Instructions

### Step 1: Download Card Metadata (one-time, 30 seconds)

```bash
python3 download_card_metadata.py
```

**Result**: `data/card_metadata.db` with 22,509 cards

### Step 2: Download Card Statistics (60-180 minutes)

```bash
python3 download_real_17lands_data.py
# Choose option 3 (Current Standard)
```

**Result**: `data/card_stats.db` with ~6000-8000 cards

### Step 3: Initialize Rules Database (one-time, 5 minutes)

```bash
python3 test_rag.py
```

**Result**: `data/chromadb/` with 3000+ rules

---

## Database File Sizes

| Database | File | Size | Records | Download Time |
|----------|------|------|---------|---------------|
| Card Metadata | `card_metadata.db` | ~8 MB | 22,509 cards | 30 seconds |
| Card Statistics | `card_stats.db` | ~2 MB | 6,000-8,000 cards | 60-180 min |
| Rules Vectors | `chromadb/` | ~50 MB | 3,000+ rules | 5 minutes |

**Total**: ~60 MB, setup time ~2-3 hours (mostly card statistics)

---

## Integration with Advisor

The advisor automatically queries all three databases when enabled:

```python
# In advisor.py
if rag_enabled:
    # Get card stats
    stats = card_stats_db.get_card_stats(card_name)

    # Get card metadata
    meta = card_metadata_db.get_card_metadata(card_name)

    # Get relevant rules
    rules = rag_system.query_rules(card_name + " " + meta['types'])

    # Combine into AI prompt
    prompt += f"\nCard: {card_name}"
    prompt += f"\n- Type: {meta['types']}, CMC: {meta['mana_value']}"
    prompt += f"\n- Win Rate: {stats['win_rate']:.1%}, IWD: {stats['iwd']:+.1%}"
    prompt += f"\n- Relevant Rules: {rules}"
```

---

## Example AI Prompts (Before vs After)

### Before (No RAG):

```
Current board state:
- Your cards: Lightning Bolt, Mountain, Island
- Opponent: Unknown creatures

What should I do?
```

### After (With Full RAG):

```
Current board state:
- Your cards:
  * Lightning Bolt (1-mana red instant, 58% WR, +3.8% IWD)
  * Mountain (basic land, produces R)
  * Island (basic land, produces U)

- Opponent creatures:
  * Unknown 3-toughness creature

Relevant rules:
- Instant spells can be cast during opponent's turn
- Combat damage uses the stack
- 3 damage kills creatures with 3 toughness

Strategic context:
- Lightning Bolt is highly impactful when drawn
- Can remove 3-toughness threats efficiently
- Consider timing: combat trick vs removal

What should I do?
```

**Result**: Much more intelligent, context-aware advice!

---

## Maintenance

### Card Metadata
**Update frequency**: Rarely (only when new sets add card types)
```bash
# Check for updates (optional)
python3 download_card_metadata.py
```

### Card Statistics
**Update frequency**: Quarterly (when new sets release)
```bash
# Check status
python3 update_card_data.py --status

# Update outdated sets
python3 update_card_data.py --auto
```

### Rules Database
**Update frequency**: Annually (when comprehensive rules change)
```bash
# Re-initialize if rules file updated
python3 test_rag.py
```

---

## Benefits of Three-Database System

### 1. **Richer Context**
- Statistics tell you WHAT performs well
- Metadata tells you WHY (colors, types, mana cost)
- Rules tell you HOW mechanics work

### 2. **Better Advice**
- "This 2-mana blue flyer has high tempo"
- "Save your instant-speed removal for their bomb"
- "You need more creatures in the 2-3 mana range"

### 3. **Strategic Depth**
- Mana curve analysis (via metadata)
- Win rate trends (via statistics)
- Rules interactions (via vector search)

### 4. **Flexible Queries**
- Search by mana cost: "Show me all 2-drops"
- Search by performance: "What's the best removal?"
- Search by rules: "How does trample work?"

---

## API Reference

### CardMetadataDB

```python
from rag_advisor import CardMetadataDB

db = CardMetadataDB()

# Get single card
meta = db.get_card_metadata("Card Name")

# Search cards
cards = db.search_cards(
    mana_value=2,
    color='U',
    types='Creature',
    rarity='common',
    limit=10
)

db.close()
```

### CardStatsDB

```python
from rag_advisor import CardStatsDB

db = CardStatsDB()

# Get single card stats
stats = db.get_card_stats("Card Name")

# Search top performers
top_cards = db.search_by_win_rate(min_games=1000, limit=10)

db.close()
```

### RAGSystem

```python
from rag_advisor import RAGSystem

rag = RAGSystem()

# Query rules
rules = rag.query_rules("combat damage", top_k=5)

# Get card stats (if available)
stats = rag.get_card_stats("Card Name")

rag.close()
```

---

## Summary

✅ **Three databases working together**:
1. Card Statistics (performance)
2. Card Metadata (attributes) **← NEW!**
3. Rules Knowledge (mechanics)

✅ **22,509 cards** with metadata (colors, types, mana cost)
✅ **6,000-8,000 cards** with statistics (win rates, performance)
✅ **3,000+ rules** with semantic search

✅ **Integration complete** - Ready to use!

**Next steps**:
1. Run `python3 download_card_metadata.py` (30 seconds)
2. Run `python3 download_real_17lands_data.py` (choose option 3)
3. The advisor now has complete card knowledge!

---

END OF DOCUMENT
