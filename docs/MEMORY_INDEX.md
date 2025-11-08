# Claude Memory Master Index

**Last Updated:** November 7, 2025
**Total Memory Files:** 8 (Non-Draft + Draft + Index)

---

## 🎯 Quick Navigation

### For Non-Draft Project (Gameplay AI)
**Status:** 40% Complete (Phase 1 Done, Phase 2 Pending)

📖 **Start:** `README_CLAUDE_MEMORY.md` (2-3 min read)
📖 **Main:** `PROJECT_MEMORY.md` (10-15 min read)
📖 **Details:** Task-specific docs (listed below)

**Quick Links:**
- Overview: `PROJECT_MEMORY.md`
- Phase 1: `PHASE_1_COMPLETION_SUMMARY.md`
- Task 1.2: `TASK_1_2_COMPLETION_SUMMARY.md`
- Task 1.3-1.4: `TASK_1_3_1_4_COMPLETION.md`
- API Reference: `ACTION_NORMALIZATION_GUIDE.md`

---

### For Draft Project (Drafting AI)
**Status:** 5% Complete (Task 1.1 Done, Task 1.2 Ready)

📖 **Start:** `README_DRAFT_MEMORY.md` (2-3 min read)
📖 **Main:** `DRAFT_PROJECT_MEMORY.md` (5-10 min read)

**Quick Links:**
- Overview: `DRAFT_PROJECT_MEMORY.md`
- Official Tasks: `/home/joshu/Desktop/JULES_TASK_LIST_DRAFT.md`

---

## 📊 Project Comparison

| Aspect | Non-Draft | Draft |
|--------|-----------|-------|
| **Goal** | Gameplay AI | Drafting AI |
| **Data Focus** | Gameplay (replay_data) | Draft picks (draft_data) |
| **Model Task** | Predict in-game decisions | Predict pick decisions |
| **Status** | 40% (Phase 1 complete) | 5% (Task 1.1 complete) |
| **Current Phase** | Phase 2 Ready | Task 1.2 Ready |
| **Timeline** | 27-45 weeks remaining | TBD |
| **Memory File** | `PROJECT_MEMORY.md` | `DRAFT_PROJECT_MEMORY.md` |

---

## 📂 Complete File Structure

```
/home/joshu/logparser/

MEMORY FILES:
├── MEMORY_INDEX.md ← YOU ARE HERE
│
├── NON-DRAFT PROJECT MEMORY:
│   ├── README_CLAUDE_MEMORY.md (Quick index)
│   ├── PROJECT_MEMORY.md (Main memory - 3000+ lines)
│   ├── PHASE_1_COMPLETION_SUMMARY.md
│   ├── TASK_1_2_COMPLETION_SUMMARY.md
│   ├── TASK_1_3_1_4_COMPLETION.md
│   ├── ACTION_NORMALIZATION_GUIDE.md
│   └── TASK_1_2_DELIVERABLES.txt
│
├── DRAFT PROJECT MEMORY:
│   ├── README_DRAFT_MEMORY.md (Quick index)
│   └── DRAFT_PROJECT_MEMORY.md (Main memory - 1500+ lines)
│
CODE MODULES (Non-Draft):
├── action_normalization.py (410 lines)
├── decision_point_extraction.py (550 lines)
├── outcome_weighting_dataset.py (450 lines)
├── process_action_sequences.py (95 lines)
├── test_action_normalization.py (202 lines)
├── test_decision_and_weighting.py (320 lines)
│
SHARED TOOLS:
├── tools/
│   ├── download_17lands_data.py
│   └── build_unified_card_database.py
│
DATABASES:
├── action_sequences.db (127K+ actions)
├── decision_points.db (144+ decision points)
├── training_dataset.db (weighted examples)
└── unified_cards.db (card metadata - used by both)

DATA:
└── data/17lands_data/
    ├── draft_data_public.*.csv.gz (60+ files)
    ├── game_data_public.*.csv.gz (60+ files)
    └── replay_data_public.*.csv.gz (60+ files)
```

---

## 🎓 How to Use This Memory System

### Scenario 1: Continuing Non-Draft Project (Phase 2)
1. Read `README_CLAUDE_MEMORY.md` (2-3 min)
2. Review "What's Ready" section of `PROJECT_MEMORY.md`
3. Check "Next Phase" section for Phase 2 details
4. Start Phase 2 implementation

### Scenario 2: Starting Draft Project (Task 1.2)
1. Read `README_DRAFT_MEMORY.md` (2-3 min)
2. Review Task 1.2 specs in `DRAFT_PROJECT_MEMORY.md`
3. Check "Current Task" section for exact requirements
4. Start Task 1.2 implementation

### Scenario 3: Learning Project Architecture
1. Start with `MEMORY_INDEX.md` (this file)
2. Read non-draft `PROJECT_MEMORY.md` quick reference
3. Read draft `DRAFT_PROJECT_MEMORY.md` quick reference
4. Review shared resources section

### Scenario 4: Understanding Data Pipeline
1. Non-Draft: See "Data Flow" in `PROJECT_MEMORY.md`
2. Draft: See "Project Structure" in `DRAFT_PROJECT_MEMORY.md`
3. Both: See "Shared Resources" in this file

---

## 🔗 Shared Resources

### unified_cards.db
- **Location:** `/home/joshu/logparser/unified_cards.db`
- **Purpose:** Offline card metadata database
- **Source:** MTGJSON AllPrintings.json
- **Builder:** `tools/build_unified_card_database.py`
- **Used By:** Both projects (Non-Draft Task 1.2, Draft Task 1.2)
- **Primary Key:** grpId (Arena card ID)
- **Status:** Currently empty (needs MTGA installation)

### Download Scripts
- **File:** `tools/download_17lands_data.py`
- **Purpose:** Download replay, draft, and game data
- **Formats:** CSV (gzipped)
- **Coverage:** 60+ sets, 2+ years of data
- **Used By:** Both projects (Phase 1 data acquisition)

### Data Directories
- **Location:** `/home/joshu/logparser/data/17lands_data/`
- **Contents:**
  - `draft_data_public.*.csv.gz` (Draft picks)
  - `game_data_public.*.csv.gz` (Game stats)
  - `replay_data_public.*.csv.gz` (Full gameplay logs)

---

## 📈 Project Progress Summary

### Non-Draft Project: 40% Complete
✅ **Phase 0:** Data Exploration (Complete)
✅ **Phase 1:** Gameplay Data Parsing (Complete)
  - Task 1.1: Data download ✅
  - Task 1.2: Action normalization ✅
  - Task 1.3: Decision points ✅
  - Task 1.4: Outcome weighting ✅

⏳ **Phase 2:** State Encoding (0% - Next)
⏳ **Phase 3:** Model Architecture (0%)
⏳ **Phase 4:** Training (0%)
⏳ **Phase 5:** Deployment (0%)

### Draft Project: 5% Complete
✅ **Phase 1:** Data Acquisition (In Progress)
  - Task 1.1: Data ingestion ✅
  - Task 1.2: Data parsing ⏳ NEXT
  - Task 1.3: Card metadata ✅ OBSOLETE

⏳ **Phase 2+:** Feature Engineering (0%)

---

## ⏱️ Timeline Estimates

### Non-Draft Project
- Completed: 2 weeks (Phase 0-1)
- Remaining: 27-45 weeks (Phase 2-5)
- **Next Phase 2:** 5-8 weeks

### Draft Project
- In Progress: 1-2 weeks (Phase 1)
- Total Estimate: TBD (depends on Phase 1 completion time)
- **Next Task 1.2:** 1-2 weeks

---

## 🔧 Technical Details

### Non-Draft Data Pipeline
```
17Lands Replay Data (60GB+)
  → action_normalization.py
  → action_sequences.db (127K+ actions)
  → decision_point_extraction.py
  → decision_points.db (144+ decision points)
  → outcome_weighting_dataset.py
  → training_dataset.db (weighted examples)
  → Phase 2: State Encoding
```

### Draft Data Pipeline
```
17Lands Draft Data (60GB+)
  → [Task 1.2: Data Parser - NOT YET IMPLEMENTED]
  → Structured pick database (10M-50M picks)
  → Phase 2: Feature Engineering
```

### Shared Card Database
```
MTGA Raw_CardDatabase + MTGJSON AllPrintings.json
  → build_unified_card_database.py
  → unified_cards.db (All cards with metadata)
  → Used by both projects for card lookup
```

---

## 📋 Key Metrics

### Non-Draft Project
- **Code:** 3,500+ lines (production + tests)
- **Tests:** 40+ (100% passing)
- **Databases:** 3 production DBs
- **Data:** 127K+ actions, 144+ decision points
- **Documentation:** 1,100+ lines
- **Status:** Phase 1 complete, Phase 2 ready

### Draft Project
- **Code:** Not yet written for Task 1.2
- **Tests:** Not yet written
- **Data:** 60+ GB available, 10M-50M picks estimated
- **Documentation:** 1,500+ lines (memory + specs)
- **Status:** Task 1.1 complete, Task 1.2 specifications ready

---

## 🎯 Decision Points

### Non-Draft: Importance-Scaled Weighting ✓
- Recommended strategy for outcome weighting
- Weighs hard wins (0.95-1.0) > easy losses (0.05-0.3)
- Enables better learning from important decisions

### Draft: High-Rank Player Filtering ✓
- Only include Platinum, Diamond, Mythic
- Filter gold and lower ranks
- Improves data quality for AI training

### Both: Unified Card Database ✓
- Replaces Scryfall API approach
- MTGJSON-based offline database
- More efficient and complete

---

## 🚀 Next Actions

### For Non-Draft (Phase 2 Start)
1. Read `PROJECT_MEMORY.md` section "Next Phase"
2. Review Phase 2 task descriptions
3. Check "Ready for Phase 2" section
4. Begin state encoding implementation

### For Draft (Task 1.2 Start)
1. Read `DRAFT_PROJECT_MEMORY.md` section "Task 1.2"
2. Review "Specific Requirements"
3. Check "Task 1.2 Expected Input/Output"
4. Begin data parser implementation

---

## 💡 Pro Tips for Future Claude

1. **Always start with the quick index file** (`README_CLAUDE_MEMORY.md` or `README_DRAFT_MEMORY.md`)
2. **Then read relevant sections** of the main memory file as needed
3. **Cross-reference between projects** to understand shared resources
4. **Keep memory files updated** as you complete tasks
5. **Document decisions** in memory for future reference
6. **Use the "Data Flow" diagrams** to understand pipelines
7. **Check "Known Issues"** sections before starting work

---

## 📞 Quick Reference

### Non-Draft Status
- **Progress:** 40% (2/5 phases)
- **Current:** Phase 1 complete
- **Next:** Phase 2 - State Encoding
- **Memory:** `PROJECT_MEMORY.md`

### Draft Status
- **Progress:** 5% (1/3 tasks)
- **Current:** Task 1.1 complete
- **Next:** Task 1.2 - Data Parsing
- **Memory:** `DRAFT_PROJECT_MEMORY.md`

### Shared
- **Database:** `unified_cards.db`
- **Scripts:** `tools/download_17lands_data.py`
- **Data:** `data/17lands_data/`

---

## 📝 Memory Maintenance

### When to Update Memory Files
- [ ] After completing a task
- [ ] Before starting a new phase
- [ ] When making architecture decisions
- [ ] When encountering blockers
- [ ] When discovering new information
- [ ] Weekly review of progress

### Files to Update
1. `MEMORY_INDEX.md` (this file) - Overall progress
2. Project-specific memory - Task-specific details
3. Project quick index - Links and status

### How to Update
1. Add new sections as needed
2. Update progress percentages
3. Add discovered issues/solutions
4. Update timelines if needed
5. Link to new documentation

---

**Master Index Created:** November 7, 2025
**Total Memory Documentation:** 8 comprehensive files
**Total Memory Size:** 8,000+ lines
**Coverage:** Complete project status for both projects

Next Claude: Start with the quick index file for your project, then dive into the main memory as needed!
