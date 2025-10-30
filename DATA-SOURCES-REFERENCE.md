# Data Sources Reference Guide

**Complete field-by-field documentation of all data sources and their contribution to the MTGA Voice Advisor**

**Date**: 2025-10-28
**Version**: 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Data Source 1: Card Statistics](#data-source-1-card-statistics)
3. [Data Source 2: Card Metadata](#data-source-2-card-metadata)
4. [Data Source 3: Keyword Abilities](#data-source-3-keyword-abilities)
5. [Data Source 4: MTG Rules](#data-source-4-mtg-rules)
6. [Integration Flow](#integration-flow)
7. [Field Usage Examples](#field-usage-examples)

---

## Overview

The MTGA Voice Advisor uses **four data sources** to provide intelligent tactical advice:

| Data Source | Origin | Records | Purpose |
|-------------|--------|---------|---------|
| **Card Statistics** | 17lands game_data CSVs | 6,000-8,000 | Performance metrics (win rates) |
| **Card Metadata** | 17lands cards.csv | 22,509 | Card attributes (colors, types, mana cost) |
| **Keyword Abilities** | 17lands abilities.csv | 18,353 | MTG keyword abilities |
| **MTG Rules** | MagicCompRules.txt | 3,000+ | Comprehensive rules |

---

## Data Source 1: Card Statistics

### Origin
- **Source**: 17lands public game_data CSV files
- **URL Pattern**: `https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{SET}.{FORMAT}.csv.gz`
- **Download Script**: `download_real_17lands_data.py`
- **Database**: `data/card_stats.db` (SQLite)

### Schema

```sql
CREATE TABLE card_stats (
    card_name TEXT PRIMARY KEY,
    set_code TEXT,
    color TEXT,
    rarity TEXT,
    games_played INTEGER,
    win_rate REAL,
    avg_taken_at REAL,
    games_in_hand INTEGER,
    gih_win_rate REAL,
    opening_hand_win_rate REAL,
    drawn_win_rate REAL,
    ever_drawn_win_rate REAL,
    never_drawn_win_rate REAL,
    alsa REAL,
    ata REAL,
    iwd REAL,
    last_updated TEXT
);
```

### Field Descriptions

#### Core Identification
| Field | Type | Description | Example | Advisor Use |
|-------|------|-------------|---------|-------------|
| `card_name` | TEXT | Card name (primary key) | "Lightning Bolt" | Match cards in game log |
| `set_code` | TEXT | Set code | "OTJ", "MKM" | Track data freshness |
| `color` | TEXT | Card color(s) | "R", "UB", "W" | Color identity analysis |
| `rarity` | TEXT | Rarity tier | "common", "rare", "mythic" | Pick priority (draft) |

#### Game Performance Metrics
| Field | Type | Description | Range | Advisor Use |
|-------|------|-------------|-------|-------------|
| `games_played` | INTEGER | Total games recorded | 1,000-100,000+ | Data reliability indicator |
| `win_rate` | REAL | Overall win rate | 0.0-1.0 (45%-60% typical) | Base card power level |
| `gih_win_rate` | REAL | Win rate when card was in hand (Games In Hand) | 0.0-1.0 | **Primary card strength metric** |
| `opening_hand_win_rate` | REAL | Win rate with card in opening hand | 0.0-1.0 | Mulligan decisions |
| `drawn_win_rate` | REAL | Win rate when drawn later | 0.0-1.0 | Late-game value |
| `ever_drawn_win_rate` | REAL | Win rate in games where drawn (any time) | 0.0-1.0 | Overall impact when accessed |
| `never_drawn_win_rate` | REAL | Win rate in games where never drawn | 0.0-1.0 | Deck baseline without card |

#### Draft Performance
| Field | Type | Description | Range | Advisor Use |
|-------|------|-------------|-------|-------------|
| `alsa` | REAL | Average Last Seen At (draft pick position) | 1.0-15.0 | Draft pick priority |
| `ata` | REAL | Average Taken At (when card was picked) | 1.0-15.0 | Community evaluation |
| `avg_taken_at` | REAL | Synonym for ata | 1.0-15.0 | Pick timing |

#### Derived Metrics
| Field | Type | Description | Range | Advisor Use |
|-------|------|-------------|-------|-------------|
| `iwd` | REAL | **Improvement When Drawn** (gih_wr - win_rate) | -0.1 to +0.2 | **Card impact when drawn** |
| `games_in_hand` | INTEGER | Games where card was in hand | 500-50,000+ | Sample size for GIH WR |

#### Metadata
| Field | Type | Description | Example | Advisor Use |
|-------|------|-------------|---------|-------------|
| `last_updated` | TEXT | Timestamp of data import | "2025-10-28T10:00:00" | Data freshness |

### How Advisor Uses Card Statistics

#### 1. **Card Power Assessment**
```python
if stats['gih_win_rate'] > 0.58:
    advice = "Strong card - prioritize playing it"
elif stats['gih_win_rate'] < 0.48:
    advice = "Weak card - consider sideboarding out"
```

#### 2. **Impact Analysis** (Most Important)
```python
iwd = stats['iwd']
if iwd > 0.04:
    advice = "High impact when drawn - keep in opening hand"
elif iwd < 0.0:
    advice = "Negative impact - only good in specific decks"
```

**IWD Interpretation**:
- `+0.04 or higher`: Bomb/game-winner (Lightning Bolt, premium removal)
- `+0.02 to +0.04`: Strong playable
- `0.00 to +0.02`: Average card
- `Negative`: Situational or build-around card

#### 3. **Draft Pick Recommendations**
```python
if stats['alsa'] < 3.0:
    advice = "High pick - usually taken early (picks 1-3)"
elif stats['alsa'] < 8.0:
    advice = "Mid pick - solid playable (picks 4-8)"
else:
    advice = "Late pick or filler"
```

#### 4. **Sample Size Warnings**
```python
if stats['games_played'] < 1000:
    warning = "Limited data - take statistics with caution"
```

---

## Data Source 2: Card Metadata

### Origin
- **Source**: 17lands cards.csv (master card database)
- **URL**: `https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv`
- **Download Script**: `download_card_metadata.py`
- **Database**: `data/card_metadata.db` (SQLite)

### Schema

```sql
CREATE TABLE card_metadata (
    card_id INTEGER PRIMARY KEY,
    expansion TEXT,
    name TEXT,
    rarity TEXT,
    color_identity TEXT,
    mana_value INTEGER,
    types TEXT,
    is_booster BOOLEAN
);
```

### Field Descriptions

| Field | Type | Description | Example | Advisor Use |
|-------|------|-------------|---------|-------------|
| `card_id` | INTEGER | Unique 17lands card ID | 65591 | Cross-reference with other datasets |
| `expansion` | TEXT | Set code | "OTJ", "MKM", "HOU" | Set identification |
| `name` | TEXT | Card name | "Lightning Bolt" | **Primary lookup key** |
| `rarity` | TEXT | Card rarity | "common", "uncommon", "rare", "mythic", "basic", "token" | Draft value assessment |
| `color_identity` | TEXT | Mana colors | "R", "U", "WU", "BRG", "" (colorless) | **Mana base analysis** |
| `mana_value` | INTEGER | Converted mana cost (CMC) | 0-16 (typically 0-7) | **Curve analysis** |
| `types` | TEXT | Card types and subtypes | "Instant", "Creature - Human Wizard", "Enchantment - Aura" | **Card type identification** |
| `is_booster` | BOOLEAN | Available in draft boosters | TRUE/FALSE | Draft relevance |

### Color Identity Values

| Value | Meaning | Example Cards |
|-------|---------|---------------|
| `""` (empty) | Colorless | Ornithopter, artifacts |
| `W` | White | Serra Angel |
| `U` | Blue | Counterspell |
| `B` | Black | Murder |
| `R` | Red | Lightning Bolt |
| `G` | Green | Llanowar Elves |
| `WU` | White-Blue | Azorius cards |
| `BR` | Black-Red | Rakdos cards |
| `GW` | Green-White | Selesnya cards |
| `URG` | Blue-Red-Green | Tri-color cards |

### Card Types

**Major Types**:
- `Artifact` - Artifacts
- `Creature` - Creatures (most common)
- `Enchantment` - Enchantments
- `Instant` - Instant-speed spells
- `Land` - Lands (mana sources)
- `Planeswalker` - Planeswalkers
- `Sorcery` - Sorcery-speed spells

**With Subtypes**:
- `Creature - Human Wizard`
- `Artifact - Equipment`
- `Enchantment - Aura`
- `Land - Mountain`

### How Advisor Uses Card Metadata

#### 1. **Mana Curve Analysis**
```python
curve = {}
for card in deck:
    meta = card_metadata_db.get_card_metadata(card)
    curve[meta['mana_value']] = curve.get(meta['mana_value'], 0) + 1

if curve[2] < 5:
    advice = "Curve is too high - need more 2-drops"
```

#### 2. **Color Fixing**
```python
colors = set()
for card in deck:
    meta = card_metadata_db.get_card_metadata(card)
    for color in meta['color_identity']:
        colors.add(color)

if len(colors) > 2 and dual_lands < 3:
    advice = "Three-color deck needs better color fixing"
```

#### 3. **Card Type Distribution**
```python
creatures = sum(1 for card in deck
                if 'Creature' in card_metadata_db.get_card_metadata(card)['types'])

if creatures < 13:
    advice = "Low creature count - may struggle on board"
```

#### 4. **Contextual Advice**
```python
meta = card_metadata_db.get_card_metadata("Lightning Bolt")
advice = f"{meta['name']} is a {meta['mana_value']}-mana {meta['color_identity']} {meta['types']}"
# Output: "Lightning Bolt is a 1-mana R Instant"
```

#### 5. **Rarity Assessment**
```python
if meta['rarity'] == 'mythic':
    advice = "Rare bomb - high value target for removal"
elif meta['rarity'] == 'common':
    advice = "Common - easily replaceable"
```

---

## Data Source 3: Keyword Abilities

### Origin
- **Source**: 17lands abilities.csv
- **URL**: `https://17lands-public.s3.amazonaws.com/analysis_data/cards/abilities.csv`
- **Download Script**: `download_card_metadata.py`
- **Database**: `data/card_metadata.db` (SQLite)

### Schema

```sql
CREATE TABLE abilities (
    ability_id INTEGER PRIMARY KEY,
    text TEXT
);
```

### Field Descriptions

| Field | Type | Description | Example | Advisor Use |
|-------|------|-------------|---------|-------------|
| `ability_id` | INTEGER | Unique ability ID | 1, 2, 3 | Cross-reference |
| `text` | TEXT | Ability keyword | "Flying", "Deathtouch", "Haste" | Keyword identification |

### Common Abilities (Partial List)

| ID | Ability | Meaning | Tactical Impact |
|----|---------|---------|-----------------|
| 1 | Deathtouch | Kills any creature it damages | Combat advantage |
| 2 | Defender | Cannot attack | Defensive only |
| 3 | Double strike | Deals damage twice | High damage output |
| 6 | First strike | Deals damage before normal combat | Combat advantage |
| 7 | Flash | Can be cast at instant speed | Surprise blocker |
| 8 | Flying | Can only be blocked by flyers/reach | Evasion |
| 9 | Haste | Can attack immediately | Tempo advantage |
| 10 | Hexproof | Cannot be targeted by opponent | Protection |
| 13 | Lifelink | Damage also gains life | Stabilization |
| 15 | Menace | Must be blocked by 2+ creatures | Evasion |
| 17 | Reach | Can block flying creatures | Anti-air |
| 19 | Trample | Excess damage to player | Push through blockers |
| 20 | Vigilance | Doesn't tap when attacking | Attack + block |

### How Advisor Uses Abilities

#### 1. **Combat Math**
```python
if 'Flying' in card_abilities and not opponent_has_flyers:
    advice = "Unblockable - safe to attack"

if 'Deathtouch' in card_abilities:
    advice = "Trades with any creature - excellent blocker"
```

#### 2. **Threat Assessment**
```python
if 'Haste' in card_abilities:
    advice = "Attacks immediately - remove before it hits"

if 'Hexproof' in card_abilities:
    advice = "Cannot target - use board wipes instead"
```

#### 3. **Blocking Decisions**
```python
if attacker_has('Trample') and blocker_toughness < attacker_power:
    advice = "Trample damage goes through - consider chump blocking"

if attacker_has('First strike') and blocker_lacks('First strike'):
    advice = "Dies before dealing damage - bad trade"
```

---

## Data Source 4: MTG Rules

### Origin
- **Source**: MagicCompRules.txt (Wizards of the Coast Comprehensive Rules)
- **URL**: https://media.wizards.com/images/magic/tcg/resources/rules/MagicCompRules.txt
- **Download Script**: Manual download or `test_rag.py`
- **Database**: `data/chromadb/` (ChromaDB vector store)

### Structure

Rules are parsed into **3,000+ chunks**, each containing:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | STRING | Rule number | "510.1", "704.5g" |
| `text` | STRING | Rule text | "First, the active player declares attackers..." |
| `category` | STRING | Rule section | "Combat", "State-Based Actions" |
| `embedding` | VECTOR | 384-dim semantic vector | [0.42, -0.18, ...] |

### Major Rule Categories

1. **Game Concepts** (Rules 100-199)
   - Turn structure, priority, zones

2. **Parts of a Card** (Rules 200-299)
   - Mana costs, types, abilities

3. **Card Types** (Rules 300-399)
   - Artifacts, creatures, enchantments, instants, lands, planeswalkers, sorceries

4. **Zones** (Rules 400-499)
   - Library, hand, battlefield, graveyard, stack, exile, command

5. **Turn Structure** (Rules 500-599)
   - Beginning phase, main phase, combat phase, ending phase

6. **Spells, Abilities, and Effects** (Rules 600-699)
   - Casting spells, activated/triggered abilities, static abilities

7. **State-Based Actions** (Rules 700-799)
   - Damage, death, sacrifice, counters

8. **Additional Rules** (Rules 800-899)
   - Multiplayer, special mechanics

### How Advisor Uses Rules

#### 1. **Semantic Search**
```python
# User asks: "Can I cast Lightning Bolt during combat?"
rules = rag.query_rules("instant combat timing", top_k=3)

# Returns:
# Rule 307.1: "Instants can be cast any time you have priority"
# Rule 506.4: "Combat damage step: players get priority"
```

#### 2. **Mechanic Explanations**
```python
# User plays card with Trample
rules = rag.query_rules("trample damage assignment", top_k=2)

# Returns:
# Rule 702.19b: "Trample lets excess damage be dealt to player"
```

#### 3. **Rules Clarification**
```python
# User confused about priority
rules = rag.query_rules("priority passing", top_k=3)

# Returns rules about:
# - When priority is gained
# - When priority is passed
# - How priority works in combat
```

#### 4. **Interaction Resolution**
```python
# User: "Does First Strike kill Deathtouch creature first?"
rules = rag.query_rules("first strike deathtouch combat", top_k=4)

# Returns rules about:
# - First strike damage step
# - Deathtouch application
# - State-based actions timing
```

---

## Integration Flow

### How All Data Sources Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    Game Event Detected                       │
│              (e.g., "Lightning Bolt cast")                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Query 1: Card Metadata                       │
│          card_metadata_db.get_card_metadata()                │
│                                                              │
│  Returns: {                                                  │
│    name: "Lightning Bolt",                                  │
│    mana_value: 1,                                           │
│    color_identity: "R",                                     │
│    types: "Instant"                                         │
│  }                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Query 2: Card Statistics                     │
│            card_stats_db.get_card_stats()                    │
│                                                              │
│  Returns: {                                                  │
│    gih_win_rate: 0.621,                                     │
│    iwd: 0.038,                                              │
│    games_played: 45000                                      │
│  }                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query 3: Rules Search                      │
│              rag.query_rules("instant timing")               │
│                                                              │
│  Returns: [                                                  │
│    "Rule 307.1: Instants can be cast...",                   │
│    "Rule 117.1a: Priority timing..."                        │
│  ]                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Synthesize Advice                         │
│                                                              │
│  "Lightning Bolt (1-mana red instant) has 62% win rate      │
│   when in hand (IWD +3.8% - high impact). Cast it at        │
│   instant speed during combat to remove their blocker       │
│   or save it for direct damage. Strong removal that         │
│   trades efficiently with any 3-toughness threat."          │
└─────────────────────────────────────────────────────────────┘
```

### Data Source Priority

1. **Card Metadata** (Required)
   - Must know: What type of card? What does it cost?
   - Without this: Cannot give contextual advice

2. **Card Statistics** (Highly Important)
   - Tells us: Is this card good? When is it good?
   - Without this: Can still give basic advice but no performance data

3. **Rules Knowledge** (Contextual)
   - Explains: How does this mechanic work?
   - Without this: Can still give advice but may miss interaction details

---

## Field Usage Examples

### Example 1: Draft Pick Decision

**Scenario**: Player picks 1-4 in draft

**Data Gathered**:
```python
# Card A: Lightning Bolt
metadata_A = {
    'mana_value': 1,
    'color_identity': 'R',
    'types': 'Instant',
    'rarity': 'rare'
}
stats_A = {
    'gih_win_rate': 0.621,
    'iwd': 0.038,
    'alsa': 2.8,
    'ata': 3.2,
    'games_played': 45000
}

# Card B: Hill Giant
metadata_B = {
    'mana_value': 4,
    'color_identity': 'R',
    'types': 'Creature - Giant',
    'rarity': 'common'
}
stats_B = {
    'gih_win_rate': 0.523,
    'iwd': 0.015,
    'alsa': 9.2,
    'ata': 9.8,
    'games_played': 38000
}
```

**Advisor Analysis**:
```
Card A (Lightning Bolt):
- High impact (IWD +3.8%)
- Early pick (ALSA 2.8)
- Premium removal
- Recommendation: PICK

Card B (Hill Giant):
- Average playable (IWD +1.5%)
- Late pick (ALSA 9.2)
- Curve filler
- Recommendation: PASS
```

### Example 2: Mulligan Decision

**Scenario**: Opening hand evaluation

**Data Gathered**:
```python
hand = ["Lightning Bolt", "Mountain", "Mountain", "Shock", "Lava Axe", "Inferno Titan", "Chandra"]

for card in hand:
    meta = card_metadata_db.get_card_metadata(card)
    stats = card_stats_db.get_card_stats(card)

    # Calculate curve
    curve[meta['mana_value']] += 1

    # Check opening hand performance
    if stats and stats['opening_hand_win_rate']:
        oh_wr_total += stats['opening_hand_win_rate']
```

**Advisor Analysis**:
```
Curve Analysis:
- 1-mana: 1 card (Lightning Bolt)
- 2-mana: 1 card (Shock)
- 3-mana: 0 cards
- 4-mana: 0 cards
- 5-mana: 1 card (Lava Axe)
- 6-mana: 1 card (Inferno Titan)
- Lands: 2 (need 3-4 for this curve)

Opening Hand Win Rates:
- Lightning Bolt: 60.2% (good)
- Shock: 53.1% (okay)
- Lava Axe: 48.2% (bad)
- Inferno Titan: 55.8% (good but uncastable)

Recommendation: MULLIGAN
- Too few lands (need 3-4)
- No 3-4 drops (curve gap)
- High CMC cards uncastable early
```

### Example 3: Combat Decision

**Scenario**: Should I attack with my 2/2 into their 2/3?

**Data Gathered**:
```python
# My creature
attacker = {
    'name': "Grizzly Bears",
    'power': 2,
    'toughness': 2,
    'abilities': [],
    'gih_win_rate': 0.525
}

# Their creature
blocker = {
    'name': "Hill Giant",
    'power': 3,
    'toughness': 3,
    'abilities': [],
    'gih_win_rate': 0.530
}

# Query rules
rules = rag.query_rules("combat damage assignment", top_k=2)
```

**Advisor Analysis**:
```
Combat Math:
- Grizzly Bears (2/2) attacks
- Hill Giant (3/3) blocks
- Grizzly Bears deals 2 damage to Hill Giant (survives as 3/1)
- Hill Giant deals 3 damage to Grizzly Bears (dies)

Trade Analysis:
- You lose: 2/2 creature (GIH WR 52.5%)
- They damage: 3/3 creature to 3/1 (still alive)

Rules Context:
- Rule 510.1: Combat damage happens simultaneously
- Rule 704.5g: Lethal damage destroys creatures

Recommendation: DO NOT ATTACK
- Bad trade (you lose creature, they keep theirs)
- Consider: Do you have instant-speed removal for Hill Giant?
- Consider: Can you add another attacker to kill it?
```

---

## Data Quality Indicators

### Card Statistics Reliability

| Games Played | Reliability | Note |
|--------------|-------------|------|
| 10,000+ | Excellent | Highly reliable |
| 5,000-10,000 | Good | Reliable |
| 1,000-5,000 | Acceptable | Use with caution |
| 500-1,000 | Poor | High variance |
| <500 | Very Poor | Not included in database |

### Data Freshness

Check `last_updated` field:
- **< 1 month old**: Current meta, fully reliable
- **1-3 months old**: Slightly dated, still good
- **3-6 months old**: May miss new cards/balance changes
- **> 6 months old**: Outdated, update recommended

---

## Summary Table: All Fields and Their Uses

| Field | Database | Type | Primary Use | Secondary Use |
|-------|----------|------|-------------|---------------|
| `card_name` | card_stats | TEXT | Card identification | Lookup key |
| `set_code` | card_stats | TEXT | Data tracking | Set rotation |
| `color` | card_stats | TEXT | Color identity | Mana base |
| `rarity` | card_stats | TEXT | Pick priority | Draft value |
| `games_played` | card_stats | INT | Data reliability | Sample size |
| `win_rate` | card_stats | REAL | Base power level | Comparison baseline |
| `gih_win_rate` | card_stats | REAL | **Primary strength metric** | Keep/sideboard decisions |
| `opening_hand_win_rate` | card_stats | REAL | Mulligan decisions | Opening hand value |
| `iwd` | card_stats | REAL | **Card impact** | Bomb identification |
| `alsa` | card_stats | REAL | Draft pick priority | Community valuation |
| `card_id` | card_metadata | INT | Cross-reference | Unique ID |
| `expansion` | card_metadata | TEXT | Set identification | Legality |
| `name` | card_metadata | TEXT | Card lookup | Display |
| `color_identity` | card_metadata | TEXT | **Mana requirements** | Color fixing |
| `mana_value` | card_metadata | INT | **Curve analysis** | Tempo assessment |
| `types` | card_metadata | TEXT | **Card type** | Deck composition |
| `is_booster` | card_metadata | BOOL | Draft relevance | Format legality |
| `ability_id` | abilities | INT | Ability lookup | Cross-reference |
| `text` | abilities | TEXT | **Keyword identification** | Mechanic explanation |
| `rule_id` | chromadb | STRING | Rule reference | Citation |
| `rule_text` | chromadb | STRING | **Rules explanation** | Interaction resolution |

**Bold fields** = Most frequently used by advisor

---

## Update Procedures

### When to Update Each Source

| Data Source | Update Frequency | Trigger | Command |
|-------------|------------------|---------|---------|
| Card Statistics | Quarterly | New set release | `python3 update_card_data.py` |
| Card Metadata | Rarely | New mechanics | `python3 download_card_metadata.py` |
| Abilities | Rarely | New keywords | `python3 download_card_metadata.py` |
| Rules | Annually | Rules update | Download new MagicCompRules.txt |

---

## Integration Code Example

```python
def get_card_advice(card_name: str) -> str:
    """Generate advice using all data sources."""

    # 1. Get card metadata (required)
    meta = card_metadata_db.get_card_metadata(card_name)
    if not meta:
        return f"Unknown card: {card_name}"

    # 2. Get card statistics (optional but important)
    stats = card_stats_db.get_card_stats(card_name)

    # 3. Query relevant rules (optional)
    rules = rag_system.query_rules(f"{card_name} {meta['types']}", top_k=2)

    # 4. Synthesize advice
    advice = f"{card_name} is a {meta['mana_value']}-mana {meta['color_identity']} {meta['types']}"

    if stats:
        advice += f"\n- Win rate when in hand: {stats['gih_win_rate']:.1%}"
        advice += f"\n- Impact when drawn: {stats['iwd']:+.1%}"

        if stats['iwd'] > 0.03:
            advice += "\n- High impact card - strong playable"

    if rules:
        advice += "\n\nRelevant rules:"
        for rule in rules:
            advice += f"\n- {rule['id']}: {rule['text'][:60]}..."

    return advice
```

---

END OF DOCUMENT

