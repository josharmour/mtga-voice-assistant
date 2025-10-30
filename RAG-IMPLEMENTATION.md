# RAG Database Implementation Plan for MTGA Voice Advisor

**Date**: 2025-10-27
**Purpose**: Add Retrieval Augmented Generation (RAG) for MTG rules and card statistics

---

## Overview

Add two optional RAG databases to enhance tactical advice quality:

1. **Rules RAG Database**: Retrieve relevant MTG rules based on game state
2. **Card Statistics RAG Database**: Retrieve 17lands.com win rate data for cards in play

**Benefits**:
- **Precision**: Only retrieve rules/stats relevant to current situation
- **Context efficiency**: Don't bloat prompt with entire rulebook
- **Data-driven advice**: Ground recommendations in actual win rate data
- **Scalability**: Easy to update as new sets/data become available

---

## Architecture

```
Game State
    ↓
Query Builder (extracts keywords/card names)
    ↓
    ├─→ Rules RAG DB ──→ Relevant rule snippets
    ↓
    └─→ Card Stats RAG DB ──→ Win rate data
    ↓
Enhanced Prompt → LLM → Tactical Advice
```

### Tech Stack
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2, 384 dim, fast)
- **Vector Store**: ChromaDB (local, persistent, no external dependencies)
- **Chunking**: LangChain TextSplitter
- **Optional**: Keep static rules summary as fallback

---

## Part 1: Rules RAG Database

### Data Source
- **Primary**: MagicCompRules 20250919.txt (80,000+ words)
- **Structure**: Numbered rules (e.g., "702.1 Deathtouch", "306 Planeswalkers")

### Indexing Strategy

**Chunk by rule sections:**
```python
# Example chunks:
{
  "rule_id": "702.1",
  "rule_name": "Deathtouch",
  "text": "702.1. Deathtouch is a static ability...",
  "keywords": ["deathtouch", "damage", "destroy", "combat"],
  "metadata": {"category": "keyword_ability"}
}

{
  "rule_id": "306",
  "rule_name": "Planeswalkers",
  "text": "306.1. A player who has priority may cast...",
  "keywords": ["planeswalker", "loyalty", "activate"],
  "metadata": {"category": "card_type"}
}
```

**Rule categories to index:**
- Keyword abilities (Flying, Deathtouch, Trample, etc.) - Section 702
- Card types (Creatures, Planeswalkers, Enchantments) - Sections 300-313
- Combat rules - Section 506-510
- Phase structure - Section 500-514
- Stack and priority - Section 117, 405
- State-based actions - Section 704
- Zone transitions - Section 121, 400

### Query Strategy

**Query based on game state:**

```python
def build_rules_queries(board_state: BoardState) -> List[str]:
    queries = []

    # 1. Current phase rules
    queries.append(f"{board_state.current_phase} phase rules")

    # 2. Card type rules (if relevant cards in play)
    card_types = set()
    for card in board_state.your_battlefield + board_state.opponent_battlefield:
        if "Planeswalker" in card.types:
            card_types.add("planeswalker")
        if "Enchantment" in card.types:
            card_types.add("enchantment")
    queries.extend(card_types)

    # 3. Keyword abilities on cards
    keywords = extract_keywords_from_cards(board_state.your_hand + board_state.your_battlefield)
    queries.extend(keywords)  # e.g., ["flying", "deathtouch", "lifelink"]

    # 4. Recent game events
    if board_state.history:
        if board_state.history.died_this_turn:
            queries.append("creature death graveyard triggers")
        if board_state.history.damage_dealt:
            queries.append("combat damage rules")

    # 5. Zone-specific rules
    if board_state.your_graveyard or board_state.opponent_graveyard:
        queries.append("graveyard recursion mechanics")
    if board_state.stack:
        queries.append("stack resolution priority")

    return queries
```

**Retrieval:**
- Query vector DB with each query string
- Retrieve top 2-3 most relevant rule chunks per query
- Deduplicate by rule_id
- Limit to 5-7 total rule chunks (keep prompt concise)

### Example Retrieved Context

**Scenario**: Turn 3, Combat phase, opponent has creature with Flying

**Retrieved rules:**
```
702.9. Flying
702.9a Flying is an evasion ability. A creature with flying can't be blocked
except by creatures with flying and/or reach. A creature with flying can block
a creature with or without flying.

506. Declare Attackers Step
506.4c An attacking creature with flying can be blocked only by creatures
defending player controls that have flying or reach.

506.4d If a creature is attacking a planeswalker, the planeswalker's controller...
```

---

## Part 2: Card Statistics RAG Database

### Data Source: 17lands.com Public Datasets

**URL**: https://www.17lands.com/public_datasets
**Format**: Compressed CSV (.csv.gz)
**Update Frequency**: Updated per MTG set release
**Access**: Direct S3 bucket download (no API key required)

---

## 17Lands Dataset Analysis (2025)

### Available Dataset Types

17lands provides three main dataset types, hosted on AWS S3:

1. **game_data_public.<SET>.<FORMAT>.csv.gz**
   - Game-level statistics (win rates, card performance)
   - Size: ~800k-1M rows × 1,475 columns per set
   - Example: `game_data_public.LCI.PremierDraft.csv.gz`

2. **draft_data_public.<SET>.<FORMAT>.csv.gz**
   - Pick-by-pick draft data
   - Includes pack composition, pick order, pool state
   - Smaller than game_data

3. **replay_data_public.<SET>.<FORMAT>.csv.gz**
   - Turn-by-turn game actions
   - Largest files (detailed play-by-play)
   - Includes cast timings, combat decisions, damage tracking

### Formats Available
- **PremierDraft** (ranked best-of-1)
- **TradDraft** (traditional best-of-3)
- **QuickDraft** (bot draft)
- **Sealed**

### Sets Available
All recent sets from 2023+: LCI (Lost Caverns of Ixalan), MKM (Murders at Karlov Manor), OTJ (Outlaws of Thunder Junction), BLB (Bloomburrow), DSK (Duskmourn), FDN (Foundations), etc.

---

## Game Data Schema (Primary for RAG)

**File example**: `game_data_public.FDN.PremierDraft.csv.gz`
**Typical size**: 823,614 rows × 1,475 columns (~200-500MB compressed)

### Column Structure

**Metadata Columns (~18):**
```python
{
  "expansion": "FDN",               # Set code
  "event_type": "PremierDraft",     # Format
  "draft_id": "abc123...",          # Unique draft ID
  "draft_time": "2025-01-15 10:30", # Timestamp
  "game_time": "2025-01-15 11:05",  # When game played
  "build_index": 0,                 # Deck version (0=original)
  "match_number": 1,                # Match in event
  "game_number": 1,                 # Game in match
  "rank": "Gold",                   # Player rank
  "opp_rank": "Platinum",           # Opponent rank
  "main_colors": "UB",              # Deck colors (Dimir)
  "splash_colors": "",              # Splash colors if any
  "on_play": true,                  # True = going first
  "num_mulligans": 0,               # Player mulligans
  "opp_num_mulligans": 1,           # Opponent mulligans
  "opp_colors": "WR",               # Opponent colors (Boros)
  "num_turns": 12,                  # Game length
  "won": true                       # Did player win?
}
```

**User Privacy Columns (2):**
```python
{
  "user_n_games_bucket": 500,      # Games played (bucketed: 1,5,10,50,100,500,1000)
  "user_game_win_rate_bucket": 0.58 # Overall WR (2% buckets: 0.50, 0.52, 0.54...)
}
```

**Card Columns (~1,455 = 291 cards × 5 prefixes):**

For EACH card in the set, there are 5 columns:

```python
# Example: "Llanowar Elves"
{
  "deck_Llanowar Elves": 2,           # Count in main deck (0-4)
  "sideboard_Llanowar Elves": 0,      # Count in sideboard
  "opening_hand_Llanowar Elves": 1,   # Was in opening hand? (0/1)
  "drawn_Llanowar Elves": 1,          # Drawn during game? (0/1)
  "tutored_Llanowar Elves": 0         # Tutored/searched? (0/1)
}
```

**Data Types (from 17lands schema):**
- Strings: expansion, event_type, draft_id, draft_time, game_time, rank, opp_rank, main_colors, splash_colors, opp_colors
- Booleans: on_play, won
- int8: build_index, match_number, game_number, num_mulligans, opp_num_mulligans, num_turns, deck_*, sideboard_*, opening_hand_*, drawn_*, tutored_*
- int16: user_n_games_bucket
- float16: user_game_win_rate_bucket

---

## Key Metrics We Can Calculate

From the game_data CSV, we can derive all these win rate statistics:

### 1. Game In Hand Win Rate (GIH WR)
**Definition**: Win rate when card was drawn/in opening hand
**Formula**:
```python
games_with_card = (opening_hand_CardName == 1) | (drawn_CardName == 1)
gih_wr = games_with_card['won'].mean()
```

### 2. Opening Hand Win Rate (OH WR)
**Definition**: Win rate when card was in opening hand
```python
opening_hand_games = (opening_hand_CardName == 1)
oh_wr = opening_hand_games['won'].mean()
```

### 3. Game Draw Win Rate (GD WR)
**Definition**: Win rate when card was drawn (not in opening hand)
```python
drawn_games = (drawn_CardName == 1) & (opening_hand_CardName == 0)
gd_wr = drawn_games['won'].mean()
```

### 4. Game Not Seen Win Rate (GNS WR)
**Definition**: Win rate when card was in deck but never seen
```python
not_seen = (deck_CardName > 0) & (opening_hand_CardName == 0) & (drawn_CardName == 0)
gns_wr = not_seen['won'].mean()
```

### 5. Improvement When Drawn (IWD)
**Definition**: How much better you do when you draw the card
**Formula**: `IWD = GIH_WR - GNS_WR`
**Example**: If Sheoldred has 62% GIH WR and 54% GNS WR, IWD = +8%

### 6. Archetype Performance
**Definition**: Win rate by color combination
```python
archetype_wr = df.groupby('main_colors')['won'].mean()
```

### 7. Cards Played (GP)
**Definition**: Number of games card was in deck
```python
games_played = (deck_CardName > 0).sum()
```

---

## RAG Indexing Strategy for 17Lands Data

### Approach 1: Pre-Aggregated Card Stats (Recommended)

**Why**: Game data CSVs are huge (500MB each). We don't need every game - just aggregate statistics per card.

**Process**:
1. Download game_data CSV for latest set(s)
2. Calculate metrics for each card (GIH WR, IWD, etc.)
3. Store aggregated stats in vector DB
4. Re-index when new set releases

**Aggregated Document Example**:
```python
{
  "card_name": "Llanowar Elves",
  "set": "FDN",
  "format": "PremierDraft",
  "color": "G",
  "stats": {
    "gih_wr": 58.2,        # Game In Hand Win Rate
    "oh_wr": 59.1,         # Opening Hand Win Rate
    "iwd": 5.3,            # Improvement When Drawn
    "gns_wr": 52.9,        # Game Not Seen WR
    "games_played": 45219, # Sample size
    "alsa": 3.2,           # Average Last Seen At (turn played)
    "copies_in_deck": 1.8  # Avg copies per deck that runs it
  },
  "archetype_performance": {
    "GW": 61.2,  # 61.2% WR in Selesnya
    "GU": 57.8,  # 57.8% WR in Simic
    "GB": 56.5   # etc.
  },
  "timing": {
    "turn_1_cast_wr": 63.5,  # WR when cast turn 1
    "turn_2_cast_wr": 60.2,
    "turn_3_cast_wr": 57.1
  }
}
```

**Vector DB Document Text (for embeddings)**:
```
Llanowar Elves (FDN, Green)
Win rate: 58.2% when drawn
Improvement: +5.3% when seen
Best in Selesnya (61.2% WR)
Optimal turn 1 play (63.5% WR)
Sample size: 45,219 games
```

### Approach 2: Query Raw CSVs (Alternative)

For real-time analysis or very recent data, could query CSVs directly:
- Use DuckDB for SQL queries on CSV.gz
- Calculate stats on-demand
- Slower but always current

---

## Implementation Plan for 17Lands RAG

### Step 1: Download & Process Data

```python
# indexers/download_17lands.py
import pandas as pd
import requests
from pathlib import Path

SETS = ["FDN", "DSK", "BLB", "OTJ"]  # Recent sets
FORMATS = ["PremierDraft"]
BASE_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/game_data"

def download_17lands_data(output_dir: Path):
    """Download game_data CSVs for specified sets"""
    for set_code in SETS:
        for format_name in FORMATS:
            filename = f"game_data_public.{set_code}.{format_name}.csv.gz"
            url = f"{BASE_URL}/{filename}"

            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)

            output_path = output_dir / filename
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✓ Downloaded {filename} ({output_path.stat().st_size / 1e6:.1f} MB)")
```

### Step 2: Aggregate Card Statistics

```python
# indexers/aggregate_card_stats.py
def calculate_card_metrics(df: pd.DataFrame, card_name: str) -> dict:
    """Calculate all win rate metrics for a single card"""

    # Filter to games where card was in deck
    has_card = df[f'deck_{card_name}'] > 0
    card_games = df[has_card]

    if len(card_games) < 100:  # Require minimum sample size
        return None

    # Game In Hand (drawn or in opening hand)
    gih = card_games[
        (card_games[f'opening_hand_{card_name}'] == 1) |
        (card_games[f'drawn_{card_name}'] == 1)
    ]
    gih_wr = gih['won'].mean() if len(gih) > 0 else None

    # Opening Hand
    oh = card_games[card_games[f'opening_hand_{card_name}'] == 1]
    oh_wr = oh['won'].mean() if len(oh) > 0 else None

    # Not Seen
    gns = card_games[
        (card_games[f'opening_hand_{card_name}'] == 0) &
        (card_games[f'drawn_{card_name}'] == 0)
    ]
    gns_wr = gns['won'].mean() if len(gns) > 0 else None

    # Improvement When Drawn
    iwd = (gih_wr - gns_wr) if (gih_wr and gns_wr) else None

    # Archetype performance
    archetype_wr = card_games.groupby('main_colors')['won'].mean().to_dict()

    return {
        "card_name": card_name,
        "games_played": len(card_games),
        "gih_wr": gih_wr * 100 if gih_wr else None,
        "oh_wr": oh_wr * 100 if oh_wr else None,
        "gns_wr": gns_wr * 100 if gns_wr else None,
        "iwd": iwd * 100 if iwd else None,
        "archetype_wr": {k: v * 100 for k, v in archetype_wr.items()}
    }

def aggregate_all_cards(csv_path: Path) -> list:
    """Process entire CSV and extract stats for all cards"""
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path, compression='gzip')

    # Find all card columns (they start with "deck_")
    card_columns = [col for col in df.columns if col.startswith('deck_')]
    card_names = [col.replace('deck_', '') for col in card_columns]

    print(f"Found {len(card_names)} cards, calculating metrics...")

    all_stats = []
    for i, card_name in enumerate(card_names):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(card_names)} cards...")

        stats = calculate_card_metrics(df, card_name)
        if stats:
            all_stats.append(stats)

    print(f"✓ Calculated stats for {len(all_stats)} cards")
    return all_stats
```

### Step 3: Index into Vector DB

```python
# indexers/index_17lands.py (updated)
def index_card_stats(stats_list: List[dict], db_path: Path):
    """Index aggregated card statistics into ChromaDB"""

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection("card_stats")
    except:
        pass

    collection = client.create_collection(
        name="card_stats",
        metadata={"description": "17lands card win rate statistics"}
    )

    documents = []
    metadatas = []
    ids = []

    for stats in stats_list:
        # Create searchable text
        doc_text = f"""
{stats['card_name']}
Win rate when drawn: {stats['gih_wr']:.1f}%
Improvement when drawn: {stats['iwd']:+.1f}%
Games played: {stats['games_played']}
Best archetypes: {', '.join([f"{k} ({v:.1f}%)" for k, v in sorted(stats['archetype_wr'].items(), key=lambda x: x[1], reverse=True)[:3]])}
        """.strip()

        documents.append(doc_text)
        metadatas.append(stats)
        ids.append(f"{stats['set']}_{stats['card_name']}")

    # Add in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print(f"✓ Indexed {len(documents)} cards into vector DB")
```

---

## Storage Requirements

**Raw CSVs (compressed)**:
- 4 sets × 1 format = 4 files × 400MB = ~1.6GB

**Aggregated Stats (JSON)**:
- ~300 cards/set × 4 sets = 1,200 cards
- ~2KB per card = ~2.4MB total

**Vector DB (embeddings)**:
- 1,200 cards × 384 dimensions × 4 bytes = ~1.8MB
- Plus metadata: ~5MB total

**Total**: ~1.6GB raw data + ~10MB processed data

**Recommendation**: Keep raw CSVs for re-processing, use aggregated stats for RAG queries

---

## Example RAG Query Results

**Query**: "Llanowar Elves win rate optimal timing"

**Retrieved Document**:
```
Llanowar Elves (FDN, Green)
Win rate when drawn: 58.2%
Improvement when drawn: +5.3%
Games played: 45,219
Best archetypes: GW (61.2%), GU (57.8%), GB (56.5%)
Optimal turn 1 play (63.5% WR)
```

**AI Advice Enhancement**:
> "Play Llanowar Elves now - 58% win rate when drawn, +5% improvement over not drawing. Turn 1 play has 63% win rate. Best in Selesnya decks."

---

## Summary: 17Lands Integration

**Data Available**: ✅ Yes (public S3 bucket)
**Format**: ✅ CSV.gz (standardized)
**Metrics**: ✅ Complete (GIH WR, IWD, archetype performance)
**Sample Size**: ✅ Huge (800k+ games per set)
**Update Frequency**: ✅ Per set release
**Cost**: ✅ Free (no API key)

### Indexing Strategy

**Index by card + context:**

```python
# Card document for embedding
doc_text = f"""
{card_name} ({mana_cost})
Type: {card_types}
Win Rate: {gih_wr}% (when in hand)
Best in: {best_archetype}
Synergies: {synergies}
Optimal turn: {optimal_turn}
"""
```

**Metadata to store:**
- Card name (for exact matching)
- Set/format (for filtering)
- Win rates (all variants)
- Archetype data
- Synergy tags
- Color identity

### Query Strategy

**Query based on cards in play:**

```python
def build_card_stats_queries(board_state: BoardState) -> List[str]:
    queries = []

    # 1. Cards in your hand (should I play this?)
    for card in board_state.your_hand:
        queries.append(f"{card.name} win rate play patterns")

    # 2. Cards on your battlefield (how to use effectively?)
    for card in board_state.your_battlefield:
        queries.append(f"{card.name} synergies optimal usage")

    # 3. Opponent's threats (how to respond?)
    for card in board_state.opponent_battlefield:
        queries.append(f"{card.name} weakness counters")

    # 4. Archetype detection (what's the game plan?)
    your_colors = detect_colors(board_state.your_battlefield)
    queries.append(f"{your_colors} archetype win conditions")

    return queries
```

**Retrieval:**
- Query vector DB with card names + context
- Retrieve top 3-5 most relevant cards
- Include win rate stats, synergies, and timing data
- Format for LLM consumption

### Example Retrieved Context

**Scenario**: You have Sheoldred in hand, opponent at 15 life, turn 4

**Retrieved stats:**
```
Sheoldred, the Apocalypse
- Win rate when played: 62.3% (top 5% in set)
- Optimal turn to play: Turn 4 (current turn!)
- Best when opponent life < 20 (current: 15 ✓)
- Synergies: Card draw effects, life drain
- Weakness: Immediate removal (hold up protection if able)
- Average game length when played: 8.2 turns

Black/Blue Control archetype
- Win rate: 58.7%
- Game plan: Stabilize early, win with card advantage
- Key turns: Turn 4 (deploy threat), Turn 6+ (lock game)
```

---

## Part 3: Implementation Details

### File Structure

```
logparser/
├── advisor.py                    # Main application (add RAG integration)
├── rag_engine.py                 # NEW: RAG query engine
├── indexers/
│   ├── index_rules.py            # NEW: Index MTG rules
│   └── index_17lands.py          # NEW: Index card stats
├── data/
│   ├── MagicCompRules.txt        # Download from Wizards
│   ├── 17lands_data/             # Download CSVs from 17lands
│   │   ├── game_data_public.DMU.csv
│   │   └── ...
│   └── vector_stores/            # ChromaDB persistent storage
│       ├── rules_db/
│       └── card_stats_db/
└── requirements.txt              # Add: chromadb, sentence-transformers, langchain
```

### Code: RAG Engine

```python
# rag_engine.py
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

class RAGEngine:
    """Retrieval Augmented Generation engine for MTG rules and card stats"""

    def __init__(self, data_dir: Path, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            logging.info("RAG engine disabled - using static context")
            return

        self.data_dir = data_dir
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB clients
        self.rules_client = chromadb.PersistentClient(
            path=str(data_dir / "vector_stores" / "rules_db")
        )
        self.stats_client = chromadb.PersistentClient(
            path=str(data_dir / "vector_stores" / "card_stats_db")
        )

        try:
            self.rules_collection = self.rules_client.get_collection("mtg_rules")
            logging.info(f"✓ Loaded rules DB: {self.rules_collection.count()} chunks")
        except:
            logging.warning("⚠ Rules DB not found - run indexers/index_rules.py first")
            self.rules_collection = None

        try:
            self.stats_collection = self.stats_client.get_collection("card_stats")
            logging.info(f"✓ Loaded card stats DB: {self.stats_collection.count()} cards")
        except:
            logging.warning("⚠ Card stats DB not found - run indexers/index_17lands.py first")
            self.stats_collection = None

    def retrieve_relevant_rules(self, board_state, max_chunks: int = 5) -> List[Dict]:
        """Retrieve relevant MTG rules based on current game state"""
        if not self.enabled or not self.rules_collection:
            return []

        queries = self._build_rules_queries(board_state)
        logging.debug(f"Rules queries: {queries}")

        # Query vector DB for each query
        all_results = []
        for query in queries[:3]:  # Limit to top 3 queries
            results = self.rules_collection.query(
                query_texts=[query],
                n_results=2  # Top 2 per query
            )
            if results['documents']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    all_results.append({
                        'rule_id': metadata.get('rule_id', ''),
                        'rule_name': metadata.get('rule_name', ''),
                        'text': doc
                    })

        # Deduplicate by rule_id
        seen = set()
        unique_results = []
        for result in all_results:
            rule_id = result['rule_id']
            if rule_id not in seen:
                seen.add(rule_id)
                unique_results.append(result)
                if len(unique_results) >= max_chunks:
                    break

        logging.info(f"Retrieved {len(unique_results)} relevant rules")
        return unique_results

    def retrieve_card_stats(self, board_state, max_cards: int = 5) -> List[Dict]:
        """Retrieve win rate stats for cards in play"""
        if not self.enabled or not self.stats_collection:
            return []

        card_names = self._get_relevant_cards(board_state)
        logging.debug(f"Card stat queries: {card_names}")

        # Query by exact card name match first (metadata filter)
        results = []
        for card_name in card_names[:max_cards]:
            try:
                matches = self.stats_collection.query(
                    query_texts=[card_name],
                    n_results=1,
                    where={"card_name": card_name}  # Exact match
                )
                if matches['documents'] and matches['metadatas']:
                    results.append(matches['metadatas'][0][0])
            except:
                logging.debug(f"No stats found for {card_name}")

        logging.info(f"Retrieved stats for {len(results)} cards")
        return results

    def _build_rules_queries(self, board_state) -> List[str]:
        """Build search queries based on game state"""
        queries = []

        # Current phase
        queries.append(f"{board_state.current_phase} phase rules")

        # Card types on battlefield
        types_seen = set()
        for card in board_state.your_battlefield + board_state.opponent_battlefield:
            # This is simplified - in real implementation, parse card.types
            if hasattr(card, 'card_types'):
                for ctype in card.card_types:
                    if ctype not in types_seen:
                        types_seen.add(ctype)
                        queries.append(f"{ctype} rules")

        # Zone-based queries
        if board_state.stack:
            queries.append("stack resolution priority")
        if board_state.your_graveyard or board_state.opponent_graveyard:
            queries.append("graveyard zone triggers")

        # Game history events
        if board_state.history and board_state.history.died_this_turn:
            queries.append("death triggers state-based actions")

        return queries[:5]  # Limit queries

    def _get_relevant_cards(self, board_state) -> List[str]:
        """Get card names to query stats for"""
        cards = []

        # Priority: Cards in hand (deciding whether to play)
        for card in board_state.your_hand[:3]:  # Top 3 in hand
            cards.append(card.name)

        # Your battlefield threats
        for card in board_state.your_battlefield[:2]:
            cards.append(card.name)

        # Opponent's threats (how to deal with)
        for card in board_state.opponent_battlefield[:2]:
            cards.append(card.name)

        return cards

    def format_for_prompt(self, rules: List[Dict], stats: List[Dict]) -> str:
        """Format retrieved data for LLM prompt"""
        lines = []

        if rules:
            lines.append("== RELEVANT RULES ==")
            for rule in rules:
                lines.append(f"{rule['rule_id']} {rule['rule_name']}")
                lines.append(rule['text'][:300])  # Truncate long rules
                lines.append("")

        if stats:
            lines.append("== CARD PERFORMANCE DATA ==")
            for stat in stats:
                card_name = stat.get('card_name', 'Unknown')
                gih_wr = stat.get('gih_wr', 0)
                optimal_turn = stat.get('optimal_turn', '?')
                lines.append(f"{card_name}: {gih_wr}% win rate, optimal turn {optimal_turn}")
            lines.append("")

        return "\n".join(lines)
```

### Code: Update AIAdvisor to use RAG

```python
# In advisor.py - modify AIAdvisor class

class AIAdvisor:
    SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor.

You will receive:
1. Current game state (board, hand, life totals, etc.)
2. Relevant MTG rules for this situation
3. Win rate data for cards in play from 17lands

Use this information to provide data-driven tactical advice.

CRITICAL RULES:
1. Ground advice in the provided win rate data
2. Reference specific rules when explaining interactions
3. Do NOT recap the board state - give actionable tactics only
4. Only reference cards explicitly listed
5. Consider turn timing and optimal play patterns from data

Give ONLY tactical advice in 1-2 short sentences."""

    def __init__(self, ollama_host: str = "http://localhost:11434",
                 model: str = "gemma3:270m",
                 enable_rag: bool = True,
                 data_dir: Path = Path.home() / "logparser" / "data"):
        self.client = OllamaClient(host=ollama_host, model=model)

        # Initialize RAG engine
        self.rag = RAGEngine(data_dir, enabled=enable_rag)

    def get_tactical_advice(self, board_state: BoardState) -> Optional[str]:
        # Retrieve relevant context via RAG
        relevant_rules = self.rag.retrieve_relevant_rules(board_state, max_chunks=5)
        card_stats = self.rag.retrieve_card_stats(board_state, max_cards=5)

        # Build enhanced prompt
        prompt = self._build_prompt(board_state, relevant_rules, card_stats)
        advice = self.client.generate(f"{self.SYSTEM_PROMPT}\n\n{prompt}")

        if advice:
            logging.debug(f"AI advice (RAG-enhanced): {advice[:500]}...")
        return advice

    def _build_prompt(self, board_state: BoardState,
                      rules: List[Dict], stats: List[Dict]) -> str:
        """Build prompt with RAG-retrieved context"""
        lines = []

        # Add RAG context FIRST (if available)
        if rules or stats:
            rag_context = self.rag.format_for_prompt(rules, stats)
            lines.append(rag_context)
            lines.append("="*70)
            lines.append("")

        # Then add game state (existing code)
        lines.append(f"== GAME STATE: Turn {board_state.current_turn}, {board_state.current_phase} ==")
        # ... rest of existing _build_prompt code ...

        return "\n".join(lines)
```

### Code: Rules Indexer

```python
# indexers/index_rules.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
from typing import List, Dict

def parse_magic_rules(rules_file: Path) -> List[Dict]:
    """Parse MagicCompRules.txt into structured chunks"""
    with open(rules_file, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []

    # Split by rule numbers (e.g., "100.1", "702.15")
    # This regex finds patterns like "702.1. Deathtouch"
    pattern = r'(\d+\.\d+[a-z]?)\.\s+([^\n]+)'

    for match in re.finditer(pattern, content):
        rule_id = match.group(1)
        rule_name = match.group(2).strip()

        # Extract rule text (everything until next rule number)
        start_pos = match.end()
        next_match = re.search(r'\d+\.\d+[a-z]?\.', content[start_pos:])
        end_pos = start_pos + next_match.start() if next_match else len(content)

        rule_text = content[start_pos:end_pos].strip()

        # Categorize rule
        category = categorize_rule(rule_id, rule_name)

        chunks.append({
            'rule_id': rule_id,
            'rule_name': rule_name,
            'text': f"{rule_id}. {rule_name}\n{rule_text}",
            'category': category,
            'keywords': extract_keywords(rule_name + " " + rule_text)
        })

    return chunks

def categorize_rule(rule_id: str, rule_name: str) -> str:
    """Categorize rule by ID range"""
    major_id = int(rule_id.split('.')[0])

    if major_id >= 702:
        return "keyword_ability"
    elif 300 <= major_id < 314:
        return "card_type"
    elif 500 <= major_id < 514:
        return "turn_structure"
    elif 506 <= major_id < 511:
        return "combat"
    else:
        return "general"

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from rule text"""
    keywords = []
    important_terms = [
        'flying', 'deathtouch', 'lifelink', 'trample', 'haste',
        'vigilance', 'reach', 'first strike', 'double strike',
        'hexproof', 'indestructible', 'menace', 'defender',
        'combat', 'damage', 'destroy', 'exile', 'graveyard',
        'stack', 'priority', 'trigger', 'activate', 'cast'
    ]

    text_lower = text.lower()
    for term in important_terms:
        if term in text_lower:
            keywords.append(term)

    return keywords

def index_rules(rules_file: Path, db_path: Path):
    """Index MTG rules into ChromaDB"""
    print(f"Parsing rules from {rules_file}...")
    chunks = parse_magic_rules(rules_file)
    print(f"Found {len(chunks)} rule chunks")

    # Initialize embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print(f"Creating vector database at {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))

    # Create collection (delete if exists)
    try:
        client.delete_collection("mtg_rules")
    except:
        pass

    collection = client.create_collection(
        name="mtg_rules",
        metadata={"description": "Magic: The Gathering Comprehensive Rules"}
    )

    # Add documents in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        documents = [chunk['text'] for chunk in batch]
        metadatas = [{
            'rule_id': chunk['rule_id'],
            'rule_name': chunk['rule_name'],
            'category': chunk['category']
        } for chunk in batch]
        ids = [chunk['rule_id'] for chunk in batch]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Indexed {min(i+batch_size, len(chunks))}/{len(chunks)} rules...")

    print(f"✓ Successfully indexed {len(chunks)} rules")

if __name__ == "__main__":
    rules_file = Path(__file__).parent.parent / "data" / "MagicCompRules.txt"
    db_path = Path(__file__).parent.parent / "data" / "vector_stores" / "rules_db"

    if not rules_file.exists():
        print(f"ERROR: Rules file not found at {rules_file}")
        print("Download from: https://magic.wizards.com/en/rules")
        exit(1)

    index_rules(rules_file, db_path)
```

### Code: 17lands Indexer

```python
# indexers/index_17lands.py
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import csv
from typing import Dict, List

def parse_17lands_csv(csv_file: Path) -> List[Dict]:
    """Parse 17lands game data CSV"""
    cards = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse relevant columns from 17lands data
            card_data = {
                'card_name': row.get('name', ''),
                'set': row.get('expansion', ''),
                'color': row.get('color', ''),
                'rarity': row.get('rarity', ''),
                'gih_wr': float(row.get('ever_drawn_win_rate', 0)) * 100,
                'oh_wr': float(row.get('opening_hand_win_rate', 0)) * 100,
                'gd_wr': float(row.get('drawn_win_rate', 0)) * 100,
                'iwd': float(row.get('drawn_improvement_win_rate', 0)) * 100,
                'games_played': int(row.get('# games_played', 0)),
                'avg_seen': float(row.get('avg_seen', 0)),
            }

            # Create searchable document text
            doc_text = f"""
{card_data['card_name']}
Color: {card_data['color']}
Win rate when in hand: {card_data['gih_wr']:.1f}%
Opening hand win rate: {card_data['oh_wr']:.1f}%
Games played: {card_data['games_played']}
            """.strip()

            card_data['doc_text'] = doc_text
            cards.append(card_data)

    return cards

def index_card_stats(csv_files: List[Path], db_path: Path):
    """Index 17lands card statistics into ChromaDB"""
    print("Parsing 17lands data...")
    all_cards = []

    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...")
        cards = parse_17lands_csv(csv_file)
        all_cards.extend(cards)

    print(f"Found {len(all_cards)} card entries")

    # Initialize embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print(f"Creating vector database at {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))

    # Create collection
    try:
        client.delete_collection("card_stats")
    except:
        pass

    collection = client.create_collection(
        name="card_stats",
        metadata={"description": "17lands card win rate statistics"}
    )

    # Add documents in batches
    batch_size = 100
    for i in range(0, len(all_cards), batch_size):
        batch = all_cards[i:i+batch_size]

        documents = [card['doc_text'] for card in batch]
        metadatas = [{
            'card_name': card['card_name'],
            'set': card['set'],
            'color': card['color'],
            'rarity': card['rarity'],
            'gih_wr': card['gih_wr'],
            'oh_wr': card['oh_wr'],
            'games_played': card['games_played']
        } for card in batch]
        ids = [f"{card['set']}_{card['card_name']}_{i+j}" for j, card in enumerate(batch)]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Indexed {min(i+batch_size, len(all_cards))}/{len(all_cards)} cards...")

    print(f"✓ Successfully indexed {len(all_cards)} card statistics")

if __name__ == "__main__":
    # Find all 17lands CSV files
    data_dir = Path(__file__).parent.parent / "data" / "17lands_data"
    csv_files = list(data_dir.glob("game_data_public.*.csv"))

    if not csv_files:
        print(f"ERROR: No 17lands CSV files found in {data_dir}")
        print("Download from: https://www.17lands.com/public_datasets")
        exit(1)

    db_path = Path(__file__).parent.parent / "data" / "vector_stores" / "card_stats_db"
    index_card_stats(csv_files, db_path)
```

---

## Part 4: Configuration and Usage

### requirements.txt additions

```
# Add to existing requirements:
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain>=0.1.0
```

### Environment Variables / Config

```python
# Add to advisor.py or config file
RAG_ENABLED = os.getenv("MTGA_RAG_ENABLED", "true").lower() == "true"
RAG_DATA_DIR = Path(os.getenv("MTGA_RAG_DATA_DIR", Path.home() / ".mtga_advisor" / "data"))
RAG_RULES_MAX_CHUNKS = int(os.getenv("MTGA_RAG_RULES_CHUNKS", "5"))
RAG_STATS_MAX_CARDS = int(os.getenv("MTGA_RAG_STATS_CARDS", "5"))
```

### Setup Instructions

```bash
# 1. Install dependencies
pip install chromadb sentence-transformers langchain

# 2. Download data sources
mkdir -p data/17lands_data
cd data

# Download MTG rules
wget https://media.wizards.com/2025/downloads/MagicCompRules%2020250919.txt -O MagicCompRules.txt

# Download 17lands data (visit https://www.17lands.com/public_datasets)
# Download game_data_public.<SET>.csv files to data/17lands_data/

# 3. Index the data
python indexers/index_rules.py      # Takes ~5 minutes
python indexers/index_17lands.py    # Takes ~10 minutes per set

# 4. Run advisor with RAG enabled (default)
python advisor.py

# 5. Run advisor WITHOUT RAG (fallback)
MTGA_RAG_ENABLED=false python advisor.py
```

---

## Part 5: Testing and Validation

### Test Scenarios

**Test 1: Rules Retrieval**
```python
# Game state: Combat phase, opponent has flying creature
# Expected: Retrieve flying rules (702.9) and combat rules (506.4)

board_state = BoardState(...)
board_state.current_phase = "Combat_Main"
# opponent has creature with Flying

rules = rag.retrieve_relevant_rules(board_state)
assert any("Flying" in r['rule_name'] for r in rules)
assert any("Declare Attackers" in r['text'] for r in rules)
```

**Test 2: Card Stats Retrieval**
```python
# Game state: Sheoldred in hand, turn 4
# Expected: Retrieve Sheoldred stats showing 62% win rate, optimal turn 4

card = GameObject(name="Sheoldred, the Apocalypse", grp_id=12345)
board_state.your_hand.append(card)
board_state.current_turn = 4

stats = rag.retrieve_card_stats(board_state)
sheoldred_stats = next(s for s in stats if s['card_name'] == "Sheoldred, the Apocalypse")
assert sheoldred_stats['gih_wr'] > 60
assert sheoldred_stats['optimal_turn'] == 4
```

**Test 3: Advice Quality**
```bash
# Manual test: Play a match with RAG enabled
python advisor.py

# Expected improvements:
# - Advice references specific rules ("According to rule 702.9, flying...")
# - Advice mentions win rates ("Sheoldred has 62% win rate, strong play")
# - Advice is data-driven ("This line wins 8% more games")
```

### Metrics to Track

1. **Retrieval quality**:
   - Are retrieved rules relevant? (manual review)
   - Are retrieved cards correct? (exact match test)

2. **Performance**:
   - RAG query latency: Target <100ms
   - Total advice generation: Target <3s (including LLM)
   - Memory usage: Target <500MB for vector DBs

3. **Advice quality**:
   - Does advice reference rules appropriately?
   - Does advice use win rate data?
   - User feedback: Is advice more helpful?

---

## Part 6: Future Enhancements

### Phase 2 Features (Later)

1. **Archetype detection**:
   - Detect your deck archetype from battlefield/graveyard
   - Retrieve archetype-specific strategies from 17lands

2. **Synergy detection**:
   - Find cards that win together (17lands co-occurrence data)
   - Suggest combo pieces to play

3. **Opponent modeling**:
   - Track opponent's cards/plays
   - Retrieve data on common deck archetypes
   - Predict likely cards in opponent's hand

4. **Meta-game awareness**:
   - Track current meta from recent 17lands data
   - Adjust advice for popular archetypes

5. **Mulligan advisor**:
   - Use opening hand win rate data
   - Advise keep/mulligan decisions

6. **Draft pick advisor**:
   - Use 17lands draft data
   - Suggest picks based on win rates + synergies

---

## Summary

**Implementation Effort**:
- Rules indexer: ~200 lines
- Card stats indexer: ~200 lines
- RAG engine: ~300 lines
- Integration: ~100 lines (modify existing AIAdvisor)
- **Total**: ~800 lines of new code

**Data Requirements**:
- MagicCompRules.txt: ~5MB
- 17lands CSVs: ~50-100MB per set
- Vector DB storage: ~200-500MB total
- RAM usage: ~300-500MB

**Performance Impact**:
- First run: 15 minutes (indexing)
- Subsequent runs: +100ms per advice query (RAG retrieval)
- Worth it: Yes - significantly better advice quality

**Benefits**:
✅ Context-aware rule explanations
✅ Data-driven tactical recommendations
✅ No context window bloat (only retrieve relevant data)
✅ Easy to update (just re-index new sets)
✅ Offline-first (all local, no API calls)

**Next Steps**:
1. Create `rag_engine.py` with basic RAG class
2. Create `indexers/index_rules.py`
3. Create `indexers/index_17lands.py`
4. Integrate RAG into `AIAdvisor`
5. Test with actual game data
6. Iterate based on advice quality

---

END OF DOCUMENT
