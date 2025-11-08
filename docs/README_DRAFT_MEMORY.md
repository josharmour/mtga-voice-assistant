# Claude Memory Index - MTG Draft AI Project

**Quick Links for Draft Project Reference:**

## 🎯 Start Here
- **DRAFT_PROJECT_MEMORY.md** ← **START HERE** - Complete draft project status
- **JULES_TASK_LIST_DRAFT.md** - Official task list (44 lines)

## 📊 Project Status
- **Phase:** 1 (Data Acquisition)
- **Progress:** Task 1.1 complete, Task 1.2 ready to start
- **Estimated Timeline:** Phase 1: 1-2 weeks, Full project: TBD

## 🔗 Related Resources
- **Non-Draft Project:** `/home/joshu/logparser/PROJECT_MEMORY.md`
- **Shared Database:** `unified_cards.db` (used by both projects)
- **Shared Scripts:** `tools/download_17lands_data.py`

## 📂 Key Files
- **Draft Task List:** `/home/joshu/Desktop/JULES_TASK_LIST_DRAFT.md`
- **Data Download:** `tools/download_17lands_data.py`
- **Card Database:** `tools/build_unified_card_database.py`
- **Raw Data:** `data/17lands_data/draft_data_public.*.csv.gz`

## 🎯 Current Task: Task 1.2

**Status:** Ready to start
**Requirements:**
- Parse draft data CSV files
- Extract pick-level records (draft_id, pack_num, pick_num, pool, pack, pick_made, rank, wins, losses)
- Validate card IDs against unified_cards.db
- Filter for high-rank players only (Platinum, Diamond, Mythic)
- Remove incomplete/malformed drafts

**Input:** Raw draft CSV from 17Lands
**Output:** Structured pick database
**Estimated Time:** 1-2 weeks

## 💾 Key Database
**unified_cards.db**
- Location: `/home/joshu/logparser/unified_cards.db`
- Built from: MTGJSON (all-cards offline database)
- Primary Key: grpId (Arena card ID)
- Used for: Card validation and metadata lookup
- Status: Currently empty (needs MTGA to populate)
- Builder: `tools/build_unified_card_database.py`

## 📋 Task 1.2 Specifics

**Extract per pick:**
```
draft_id        - Draft identifier
player_rank     - Platinum/Diamond/Mythic only
pack_number     - 1, 2, or 3
pick_number     - 1-15 within pack
pool            - Player's current cards
pack            - Available cards
pick_made       - Card picked
win_count       - Wins from game data
loss_count      - Losses from game data
```

**Filters:**
- ✓ High-rank only (Platinum+)
- ✓ Complete drafts only
- ✓ Valid card IDs only
- ✓ Consistent data only

## 🔄 Phase Overview

**Phase 1:** Data Acquisition (In Progress)
- Task 1.1: Data download ✅ DONE
- Task 1.2: Data parsing ⏳ NEXT
- Task 1.3: Card metadata ✅ OBSOLETE (replaced by unified_cards.db)

**Phase 2+:** Feature Engineering (Not started)

## 📈 Draft vs Non-Draft

Both projects use:
- ✓ Same 17Lands data source
- ✓ Same unified_cards.db
- ✓ Same download infrastructure

Different focus:
- **Draft:** Pick decisions (what to pick from 15 cards)
- **Non-Draft:** Gameplay decisions (what to play in-game)

## 🚀 Quick Start (When Ready for Phase 2)

```bash
# Download data
python tools/download_17lands_data.py

# Build card database (if needed)
python tools/build_unified_card_database.py

# Parse draft data (Task 1.2)
# Implement data parser for pick extraction
```

## 📝 Architecture Decision

**Major Change:** Unified Card Database
- Old approach: Scryfall API per request
- New approach: MTGJSON offline database
- Reason: More efficient, no rate limits, complete
- File: `tools/build_unified_card_database.py`
- Status: Ready, just needs population

## ⚠️ Important Notes

✓ Task 1.2 specifications are clear and detailed
✓ unified_cards.db replaces old Scryfall approach
✓ Both draft and non-draft projects can share infrastructure
✓ High-rank filtering (Platinum+) for data quality
✓ 10M-50M picks expected after parsing

## 📞 For Questions

- **Status:** See `DRAFT_PROJECT_MEMORY.md`
- **Tasks:** See `/home/joshu/Desktop/JULES_TASK_LIST_DRAFT.md`
- **Task 1.2:** See "Task 1.2: Data Parsing" section in DRAFT_PROJECT_MEMORY.md
- **Database:** See "unified_cards.db" section
- **Architecture:** See "Architecture Decisions Made" section

## 🔄 Comparison with Non-Draft

| Aspect | Non-Draft | Draft |
|--------|-----------|-------|
| Project Status | 40% (Phase 2 ready) | 5% (Task 1.2 ready) |
| Current Phase | Phase 1 Complete | Phase 1 In Progress |
| Next Task | Phase 2 Encoding | Task 1.2 Parsing |
| Data Type | Gameplay (replay_data) | Draft picks (draft_data) |
| Model Focus | In-game decisions | Pick decisions |

## 📋 Memory Update Checklist

When starting Task 1.2:
- [ ] Update "Current Task" section
- [ ] Update progress percentage
- [ ] Add any technical decisions
- [ ] Document data format findings
- [ ] Update estimated timelines

When Phase 2 starts:
- [ ] Create DRAFT_PHASE2_MEMORY.md
- [ ] Document feature engineering approach
- [ ] Update main DRAFT_PROJECT_MEMORY.md
- [ ] Link to Phase 2 documentation

---

**Last Updated:** November 7, 2025
**Status:** Phase 1 - Task 1.2 Ready to Start
**Next Review:** Before Task 1.2 implementation
