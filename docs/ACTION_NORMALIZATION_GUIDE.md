# Action Sequence Parsing & Normalization - Developer Guide

## Overview

This guide covers the complete implementation of **Task 1.2: Action Sequence Parsing and Normalization**, which transforms raw 17Lands replay data into normalized, queryable action sequences ready for model training.

## Architecture

```
Raw 17Lands Data (CSV)
    ↓
ActionSequenceParser
    ├─ RawActionMapper: Normalize action types
    ├─ CardMetadataEnricher: Enrich with card data
    └─ BatchProcessor: Efficient database insertion
    ↓
action_sequences.db (SQLite)
    ├─ action_sequences table (normalized actions)
    └─ action_sequence_metadata table (game metadata)
    ↓
Ready for Task 1.3 (Decision Point Extraction)
```

## Core Classes

### 1. ActionType (Enum)

Defines 21 normalized action types covering all MTG gameplay:

```python
class ActionType(Enum):
    # Card drawing/tutoring
    CARD_DRAWN = "card_drawn"
    CARD_TUTORED = "card_tutored"
    CARD_DISCARDED = "card_discarded"

    # Casting
    CREATURE_CAST = "creature_cast"
    NON_CREATURE_SPELL_CAST = "non_creature_spell_cast"
    INSTANT_SORCERY_CAST = "instant_sorcery_cast"

    # Combat
    CREATURE_ATTACKED = "creature_attacked"
    CREATURE_BLOCKED = "creature_blocked"
    # ... and 13 more
```

**Usage:**
```python
action_type = ActionType.CREATURE_CAST
action_value = action_type.value  # "creature_cast"
```

### 2. NormalizedAction (Dataclass)

Represents a single action in a normalized format:

```python
@dataclass
class NormalizedAction:
    action_id: int                  # Auto-assigned
    sequence_id: str                # game_id_turn_player
    action_type: ActionType         # Enum
    player: PlayerPerspective       # USER or OPPONENT
    card_grp_id: Optional[int]      # Arena card ID
    card_name: Optional[str]        # Enriched
    card_type_line: Optional[str]   # Enriched
    card_mana_cost: Optional[str]   # Enriched
    turn_number: int                # 1-based
    phase: Optional[str]            # Main, Combat, etc.
    timestamp: Optional[int]        # Order within turn
    game_outcome: Optional[bool]    # True=user won
    opposing_outcome: Optional[bool]# True=opponent won
```

**Conversion to Database Record:**
```python
action = NormalizedAction(...)
db_dict = action.to_dict()  # Dictionary ready for INSERT
```

### 3. CardMetadataEnricher

Loads card database into memory for fast lookups:

```python
enricher = CardMetadataEnricher("unified_cards.db")
metadata = enricher.get_card_metadata(card_grp_id=67352)
# Returns: {'name': 'Island', 'type_line': 'Land', 'mana_cost': ''}
```

**Benefits:**
- O(1) lookup time
- Entire database cached in memory
- Graceful degradation if DB unavailable
- Ready for integration when DB is populated

### 4. RawActionMapper

Static mapping between 17Lands raw columns and normalized types:

```python
ActionType, is_creature = RawActionMapper.get_action_type("creatures_cast")
# Returns: (ActionType.CREATURE_CAST, True)
```

**Mapping Examples:**
| Raw Column | Normalized Type | Creature Action |
|------------|-----------------|-----------------|
| creatures_cast | CREATURE_CAST | True |
| cards_drawn | CARD_DRAWN | False |
| lands_played | LAND_PLAYED | False |
| creatures_attacked | CREATURE_ATTACKED | True |
| user_abilities | ACTIVATED_ABILITY | False |

### 5. ActionSequenceParser

Main engine for processing replay data:

```python
parser = ActionSequenceParser(
    cards_db_path="unified_cards.db",
    output_db_path="action_sequences.db"
)

# Process single file
actions_count = parser.process_replay_file(
    Path("data/17lands_data/replay_data_public.PIO.PremierDraft.csv.gz"),
    batch_size=5000  # Rows per batch
)

# Or process all files
total_actions = parser.process_all_files(
    data_dir=Path("data/17lands_data")
)
```

## Database Schema

### action_sequences Table

Contains normalized action records:

```sql
CREATE TABLE action_sequences (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    player TEXT NOT NULL,
    card_grp_id INTEGER,
    card_name TEXT,
    card_type_line TEXT,
    card_mana_cost TEXT,
    turn_number INTEGER NOT NULL,
    phase TEXT,
    timestamp INTEGER,
    game_outcome INTEGER,
    opposing_outcome INTEGER,
    FOREIGN KEY (sequence_id) REFERENCES action_sequence_metadata(sequence_id)
);
```

**Index Strategy:**
- Implicit: action_sequences.action_id (PRIMARY KEY)
- Consider adding: `CREATE INDEX idx_sequence ON action_sequences(sequence_id)`
- Consider adding: `CREATE INDEX idx_action_type ON action_sequences(action_type)`
- Consider adding: `CREATE INDEX idx_outcome ON action_sequences(game_outcome)`

### action_sequence_metadata Table

Game-level metadata for context:

```sql
CREATE TABLE action_sequence_metadata (
    sequence_id TEXT PRIMARY KEY,
    game_id TEXT,
    expansion TEXT,           -- "PIO", "TDM", etc.
    event_type TEXT,          -- "PremierDraft", "Sealed"
    rank TEXT,                -- "mythic", "gold", etc.
    opp_rank TEXT,
    main_colors TEXT,         -- "WU", "BR", etc.
    splash_colors TEXT,
    on_play INTEGER,          -- 1 if user played first
    num_mulligans INTEGER,
    opp_num_mulligans INTEGER,
    opp_colors TEXT,
    num_turns INTEGER,
    game_outcome INTEGER,     -- 1 for win, 0 for loss
    action_count INTEGER      -- Total actions in game
);
```

## Usage Examples

### Processing Data

**Quick Start - Recent Data:**
```bash
python process_action_sequences.py
```

Processes 5 recent sets with stable data structure (~15-20M actions).

**Full Processing - All Data:**
```bash
source venv/bin/activate
python action_normalization.py
```

Processes all 60+ sets (~250-300M actions, requires 4-6 hours).

### Querying Data

**Example 1: Get all actions in a game**
```python
import sqlite3

conn = sqlite3.connect("action_sequences.db")
cursor = conn.cursor()

game_id = "36e49f99254d42ddb8a8a2a6c268042a_1_1"
cursor.execute("""
    SELECT action_type, card_name, turn_number, player, timestamp
    FROM action_sequences
    WHERE sequence_id LIKE ?
    ORDER BY turn_number, timestamp
""", (f"{game_id}%",))

for row in cursor.fetchall():
    print(row)
```

**Example 2: Win rate by action type (user perspective)**
```python
cursor.execute("""
    SELECT
        action_type,
        COUNT(*) as total,
        SUM(game_outcome) as wins,
        ROUND(100.0 * SUM(game_outcome) / COUNT(*), 1) as win_pct
    FROM action_sequences
    WHERE player = 'user'
    GROUP BY action_type
    ORDER BY total DESC
""")

for action_type, total, wins, pct in cursor.fetchall():
    print(f"{action_type:30} {pct:5.1f}% ({wins}/{total})")
```

**Example 3: Actions per turn distribution**
```python
cursor.execute("""
    SELECT turn_number, COUNT(*) as action_count
    FROM action_sequences
    GROUP BY turn_number
    ORDER BY turn_number
""")

for turn, count in cursor.fetchall():
    print(f"Turn {turn:2d}: {count:6d} actions")
```

**Example 4: Early game action analysis (turns 1-3)**
```python
cursor.execute("""
    SELECT
        action_type,
        player,
        COUNT(*) as frequency
    FROM action_sequences
    WHERE turn_number <= 3 AND game_outcome = 1
    GROUP BY action_type, player
    ORDER BY frequency DESC
    LIMIT 10
""")

print("Actions in winning games (early game):")
for action_type, player, freq in cursor.fetchall():
    print(f"  {action_type:30} ({player:4}): {freq}")
```

**Example 5: Creature cast cost analysis**
```python
cursor.execute("""
    SELECT
        card_name,
        card_mana_cost,
        COUNT(*) as cast_count,
        SUM(game_outcome) as wins
    FROM action_sequences
    WHERE action_type = 'creature_cast'
      AND player = 'user'
      AND card_name IS NOT NULL
    GROUP BY card_grp_id
    HAVING cast_count >= 20
    ORDER BY cast_count DESC
    LIMIT 20
""")

for name, mana, casts, wins in cursor.fetchall():
    if casts > 0:
        pct = 100.0 * wins / casts
        print(f"{name:25} {mana:6} {pct:5.1f}% ({wins}/{casts})")
```

## Integration with Next Steps

### Task 1.3: Decision Point Extraction

Action sequences form the input for identifying decision points:

```python
# In Task 1.3, you'll build decision points from these sequences
def extract_decision_points(sequence_id):
    """Extract critical game decisions from action sequence"""
    actions = query_action_sequence(sequence_id)
    # Filter to meaningful decision moments
    # (e.g., when player has choices)
    decision_points = []
    for action in actions:
        if is_decision_point(action):
            decision_points.append(action)
    return decision_points
```

### Task 1.4: Outcome Weighting

Use game_outcome to weight training data:

```python
# In Task 1.4, weight actions by their outcomes
cursor.execute("""
    SELECT
        action_type,
        game_outcome,
        COUNT(*) as count
    FROM action_sequences
    WHERE player = 'user'
    GROUP BY action_type, game_outcome
""")

# Create weighted training pairs:
# (action_sequence) -> (outcome_weight)
```

### Phase 2: State Encoding

Use card metadata to build game state:

```python
# In Phase 2, convert actions to state tensors
# card_grp_id + card_type_line + card_mana_cost
# Enable computation of:
# - Board state at each turn
# - Hand composition
# - Mana curve
# - Card type distribution
```

## Performance Characteristics

### Processing Speed
- **Small file (1MB):** ~1-2 seconds
- **Medium file (50MB):** ~1-2 minutes
- **Large file (500MB+):** ~10-20 minutes
- **All files (60GB+):** ~4-6 hours

### Database Size
- **Sample (PIO.TradSealed, 1,687 games):** ~30 MB
- **Full recent sets:** ~3-5 GB
- **All historical data:** ~15-20 GB

### Memory Usage
- **Processing:** 2-4 GB RAM
- **Database queries:** <500 MB for typical queries
- **Card enrichment cache:** ~100 MB (with full DB)

### Query Performance
- **Point queries (single game):** <1 ms
- **Range queries (action type):** 10-100 ms
- **Aggregate queries:** 100-500 ms
- **Complex joins:** 1-5 seconds

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
source venv/bin/activate
pip install pandas
```

### Issue: CSV parsing errors for older files

**Cause:** Some early 17Lands files have inconsistent column counts

**Solution:**
- Automatic: Script handles with `on_bad_lines='skip'`
- Manual: Process only recent files (PIO onwards)

### Issue: Out of memory during processing

**Solution 1:** Reduce batch size
```python
parser.process_replay_file(filepath, batch_size=1000)
```

**Solution 2:** Process files separately
```python
for file in recent_files:
    parser.process_replay_file(Path(f"data/17lands_data/{file}"))
```

### Issue: Empty card metadata

**Expected:** unified_cards.db is empty by default

**To populate:**
1. Install MTGA
2. Run: `python tools/build_unified_card_database.py`
3. Script will find MTGA's Raw_CardDatabase and populate unified_cards.db

## Testing

Run comprehensive tests:

```bash
source venv/bin/activate
python test_action_normalization.py
```

**Test Coverage:**
1. Card metadata enrichment
2. Database setup and schema
3. Small file processing
4. Database queries
5. Outcome tracking

Expected output: All tests pass with statistics.

## File Structure

```
logparser/
├── action_normalization.py           # Main implementation
├── test_action_normalization.py      # Comprehensive tests
├── process_action_sequences.py       # Quick processing script
├── ACTION_NORMALIZATION_GUIDE.md     # This file
├── TASK_1_2_COMPLETION_SUMMARY.md    # Detailed summary
└── data/
    └── 17lands_data/                 # Raw replay CSV files
        └── replay_data_public.*.csv.gz
```

## API Reference

### ActionSequenceParser

```python
parser = ActionSequenceParser(
    cards_db_path: str = "unified_cards.db",
    output_db_path: str = "action_sequences.db"
)

# Process a single file
actions: int = parser.process_replay_file(
    filepath: Path,
    batch_size: int = 1000
) -> int

# Process all files in directory
total: int = parser.process_all_files(
    data_dir: Path = Path("data/17lands_data")
) -> int
```

### NormalizedAction

```python
action = NormalizedAction(
    action_id=0,
    sequence_id="draft_id_1_1_1_user",
    action_type=ActionType.CREATURE_CAST,
    player=PlayerPerspective.USER,
    card_grp_id=12345,
    card_name="Mountain",
    card_type_line="Land",
    card_mana_cost="",
    turn_number=1,
    phase="Main",
    timestamp=0,
    game_outcome=True,
    opposing_outcome=False
)

db_dict: Dict = action.to_dict()
```

### CardMetadataEnricher

```python
enricher = CardMetadataEnricher(
    db_path: str = "unified_cards.db"
)

metadata: Dict = enricher.get_card_metadata(
    card_grp_id: int
)
# Returns: {'name': str, 'type_line': str, 'mana_cost': str}
```

## Contributing

To extend the normalization schema:

1. Add new ActionType to enum
2. Add mapping in RawActionMapper.RAW_ACTION_MAPPING
3. Update database schema if needed
4. Add test cases to test_action_normalization.py

## References

- Task Spec: `JULES_TASK_LIST_NON_DRAFT.md`
- Related Code: `replay_data_analysis.py`, `parse_replay_data.py`
- Data: 17Lands Public Replay Data
- Format: CSV (gzipped)
- License: See project LICENSE

---

*Last Updated: November 7, 2025*
*Task 1.2: Action Sequence Parsing and Normalization*
