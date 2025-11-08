# MTG Drafting AI - Draft Project Memory

**Last Updated:** November 7, 2025
**Project Status:** Phase 1 Underway - Task 1.2 Ready to Start
**Current Phase:** Data Acquisition (Phase 1)

---

## Quick Reference

### Project Goal
Build an AI model that learns from 17Lands draft data to predict and recommend optimal card picks during live Magic: The Gathering drafting.

### Current Status
- ✅ **Task 1.1 Complete** - Data ingestion system (download scripts ready)
- ⏳ **Task 1.2 Next** - Data parsing and cleaning
- ✅ **Task 1.3 Obsolete** - Replaced by MTGJSON-based unified_cards.db
- ⏳ **Phase 2+ Pending** - Feature engineering and model development

### Key Asset
✅ `unified_cards.db` - Comprehensive offline card database built from MTGJSON
- Location: `/home/joshu/logparser/unified_cards.db`
- Build script: `/home/joshu/logparser/tools/build_unified_card_database.py`
- Much more robust than Scryfall API approach

---

## Phase 1: Data Acquisition and Processing Pipeline

### Task 1.1: Data Ingestion System ✅ COMPLETE

**Status:** DONE
**Objective:** Develop scripts to automatically download public datasets from 17Lands

**What Was Built:**
- Download scripts for draft data
- Download scripts for game data
- Automatic versioning system
- Support for multiple formats

**Key Files:**
- `tools/download_17lands_data.py` - Main download script
- `/home/joshu/logparser/data/17lands_data/` - Data storage directory
- Data includes draft_data and game_data files (gzipped CSV)

**Data Available:**
- Draft data: Multiple sets (AFR, BLB, BRO, DMU, DSK, EOE, FDN, FIN, etc.)
- Game data: Same set coverage
- Coverage: ~2.5+ years of data
- Formats: PremierDraft, TradDraft, Sealed

---

### Task 1.2: Data Parsing, Cleaning, and Filtering ⏳ NEXT

**Status:** NOT STARTED
**Assigned to:** Data Processing Agent
**Objective:** Process raw logs into structured format with quality assurance

**Specific Requirements:**
1. **Extract for every pick:**
   - `draft_id` - Unique draft identifier
   - `player_rank` - Player's rank at time of draft
   - `pack_number` - Which pack (1, 2, or 3)
   - `pick_number` - Which pick within pack (1-15)
   - `pool` - Cards currently in player's pool
   - `pack` - Cards available in current pack
   - `pick_made` - Which card was picked
   - `win_count` - Wins in draft (from game results)
   - `loss_count` - Losses in draft (from game results)

2. **Filter for high-ranking players only:**
   - Include: Platinum, Diamond, Mythic ranks
   - Exclude: Gold and lower
   - Reasoning: Higher quality picks to learn from

3. **Data Quality Checks:**
   - Remove incomplete drafts
   - Remove malformed data
   - Validate card IDs against unified_cards.db
   - Ensure consistency between draft and game data

**Will Use:** `unified_cards.db` for all card metadata lookups
- This replaces the old Scryfall API approach (much more efficient)
- Contains grpId → card name mappings
- Pre-populated with all card data

**Estimated Scope:**
- Input: Raw draft_data CSV files (100s of MB)
- Output: Structured pick database (normalized format)
- Expected: 10M-50M+ individual picks from all games

---

### Task 1.3: Card Metadata Integration ✅ OBSOLETE

**Status:** OBSOLETE (Replaced by better approach)
**Previous Objective:** Gather comprehensive card data from Scryfall API

**Why Obsolete:**
- Original plan: Fetch card data from Scryfall API per request
- Problem: Inefficient, rate-limited, requires network calls
- Solution: Built `unified_cards.db` using MTGJSON (all cards offline)

**The Better Solution:**
- **File:** `tools/build_unified_card_database.py`
- **Source:** MTGJSON AllPrintings.json (complete card database)
- **Method:**
  1. Extract Arena cards from MTGA's Raw_CardDatabase
  2. Enrich with MTGJSON data (oracle text, type line, mana cost)
  3. Create unified_cards.db with all fields
  4. Pre-populated with grpId → card properties mapping

**Database Schema (unified_cards.db):**
```sql
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
    is_reskin BOOLEAN,
    last_updated TIMESTAMP
);
```

**Current Status of unified_cards.db:**
- Location: `/home/joshu/logparser/unified_cards.db`
- Currently empty (needs MTGA installation to populate)
- When populated: Contains 15K-20K cards with full metadata
- Used by: Task 1.2 (parsing), Non-Draft gameplay AI (Task 1.2)

---

## Phase 2: Feature Engineering and Representation

**Status:** Not yet started (placeholder)
**Coming:** After Task 1.2 complete

---

## Project Structure

### Draft Data Format (17Lands)
- **Source:** 17Lands public datasets
- **Format:** Gzipped CSV files
- **Naming:** `draft_data_public.{SET}.{FORMAT}.csv.gz`
- **Example Sets:** AFR, BLB, BRO, DMU, DSK, EOE, FDN, FIN, etc.

### Task 1.2 Expected Input
Raw draft CSV with columns like:
- draft_id, set_number, build_index, event_type, draft_time
- pack_card_*.NN (card options)
- pick_card (picked card)
- game_win_rate (result tracking)
- ... many more columns

### Task 1.2 Expected Output
Structured pick database with:
- One row per pick
- Normalized format: `[draft_id, pack_num, pick_num, pool, pack, pick_made, rank, wins, losses]`
- Validated against card database
- High-rank players only
- Clean, malformed entries removed

---

## Key Technical Details

### unified_cards.db
- **Location:** `/home/joshu/logparser/unified_cards.db`
- **Builder:** `tools/build_unified_card_database.py`
- **Source Data:**
  - MTGA's Raw_CardDatabase (card IDs and English names)
  - MTGJSON AllPrintings.json (comprehensive card data)
- **Primary Key:** grpId (Arena card ID)
- **Usage in Task 1.2:** Validate all card IDs and enrich with metadata

**To Populate unified_cards.db:**
```bash
python tools/build_unified_card_database.py
```
(Requires MTGA installation)

### Data Locations
```
/home/joshu/logparser/
├── data/
│   └── 17lands_data/
│       ├── draft_data_public.*.csv.gz
│       └── game_data_public.*.csv.gz
├── tools/
│   ├── download_17lands_data.py
│   └── build_unified_card_database.py
└── unified_cards.db
```

---

## Architecture Decisions Made

### Decision 1: Unified Card Database Instead of Scryfall API
- **Why:** Offline, no rate limits, comprehensive, more efficient
- **Implementation:** MTGJSON-based build script
- **File:** `tools/build_unified_card_database.py`
- **Status:** Ready to use (currently empty, awaits population)

### Decision 2: High-Rank Player Filtering
- **Why:** Learn from the best (Platinum+)
- **Filters:** Platinum, Diamond, Mythic ranks
- **Exclude:** Gold and lower
- **Reduces noise:** Lower-ranked picks less reliable

### Decision 3: Pick-Level Granularity
- **Why:** ML needs to learn pick-by-pick decisions
- **Extract:** Every pick with full context
- **Context:** Current pool, pack options, pick result, rank, performance

---

## Known Status & Notes

### What's Ready
✅ Download scripts (Task 1.1 complete)
✅ Data repository (60+ GB available)
✅ Card database infrastructure (unified_cards.db)
✅ Task specifications (clear requirements for 1.2)

### What's Next
⏳ **Task 1.2:** Parse draft data into pick-level records
- Requires implementing data processor
- Will use unified_cards.db for validation
- Output: Structured pick database
- Estimated time: 1-2 weeks

### Architecture Refinements
- Task 1.3 obsoleted and replaced
- unified_cards.db is a major improvement
- Makes card lookups offline and efficient
- Ready for use across both draft and non-draft projects

---

## Comparison: Draft vs Non-Draft Projects

### Shared Resources
- ✅ `unified_cards.db` - Used by both
- ✅ 17Lands data - Downloaded for both
- ✅ `tools/build_unified_card_database.py` - Build script for both

### Separate Pipelines
- **Draft:** Focus on pick decisions (what to pick from limited pool)
- **Non-Draft:** Focus on gameplay decisions (what to play/attack with)
- **Draft data:** draft_data_public.*.csv.gz
- **Gameplay data:** replay_data_public.*.csv.gz

### Status Difference
- **Draft Project:** Phase 1 in progress (data acquisition)
- **Non-Draft Project:** Phase 1 complete (data → training dataset)

---

## Next Steps for Draft Project

### Immediate (Start Task 1.2)
1. **Data Parser:** Read draft CSV files
2. **Pick Extractor:** Extract pick-level records
3. **Validation:** Check cards exist in unified_cards.db
4. **Filtering:** Keep only Platinum+ players
5. **Quality Checks:** Remove incomplete/malformed data
6. **Output:** Structured pick database

### Short Term (Phase 2)
- Feature engineering for draft picks
- State representation for draft context
- Model architecture for pick prediction

### Medium Term
- Training data preparation
- Model training on pick history
- Evaluation and tuning

---

## Important Files & References

### Main Files
- **Draft Task List:** `/home/joshu/Desktop/JULES_TASK_LIST_DRAFT.md`
- **Data Download:** `tools/download_17lands_data.py`
- **Card Database:** `tools/build_unified_card_database.py`
- **Card DB:** `unified_cards.db`

### Data Locations
- **Raw Draft Data:** `data/17lands_data/draft_data_public.*.csv.gz`
- **Raw Game Data:** `data/17lands_data/game_data_public.*.csv.gz`

### Related Projects
- **Non-Draft Project:** `/home/joshu/logparser/PROJECT_MEMORY.md`
- **Shared Assets:** `unified_cards.db`, download scripts

---

## Summary for Future Claude

**Where We Are:**
- ✅ Task 1.1 complete (download scripts ready)
- ✅ Major architecture improvement (unified_cards.db)
- ⏳ Task 1.2 ready to start (all specs clear)

**What's Done:**
- Download infrastructure
- Data availability confirmed
- Card database approach finalized
- Data requirements specified

**What's Next:**
- Implement Task 1.2 (parse and structure pick data)
- Extract 10M-50M picks from raw data
- Validate against card database
- Filter for high-rank players

**Status: Phase 1 (Data Acquisition) in progress**
**Task 1.1: ✅ Done | Task 1.2: ⏳ Ready to Start**

---

**Generated:** November 7, 2025
**Last Review:** November 7, 2025
**Next Review:** Before starting Task 1.2
