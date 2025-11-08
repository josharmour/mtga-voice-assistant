# Claude Memory Index - MTG Gameplay AI Project

**Quick Links for Claude Instances:**

## 🎯 Start Here
- **PROJECT_MEMORY.md** ← **START HERE** - Complete project status and memory
- **PHASE_1_COMPLETION_SUMMARY.md** - Detailed Phase 1 overview

## 📊 Task Documentation
- **TASK_1_2_COMPLETION_SUMMARY.md** - Action normalization (Task 1.2)
- **TASK_1_3_1_4_COMPLETION.md** - Decision points & weighting (Tasks 1.3-1.4)
- **ACTION_NORMALIZATION_GUIDE.md** - Full API reference with examples

## 💾 Databases
- `action_sequences.db` - 127K+ normalized actions
- `decision_points.db` - 144+ decision points with state vectors
- `training_dataset.db` - Weighted training examples

## 🐍 Code Modules
- `action_normalization.py` (410 lines) - Action parsing
- `decision_point_extraction.py` (550 lines) - Decision extraction
- `outcome_weighting_dataset.py` (450 lines) - Dataset assembly
- Test files: `test_action_normalization.py`, `test_decision_and_weighting.py`

## 📈 Current Status
- **Progress:** 40% (2 of 5 phases complete)
- **Phase 1:** ✅ Complete - Data parsing & dataset
- **Phase 2:** ⏳ Next - State and action encoding (5-8 weeks)
- **Phases 3-5:** ⏳ Pending - Model, training, deployment

## 🚀 Quick Start
```bash
# Navigate to project
cd /home/joshu/logparser

# Activate venv
source venv/bin/activate

# Run tests
python test_decision_and_weighting.py

# Extract decision points
python -c 'from decision_point_extraction import DecisionPointExtractor; DecisionPointExtractor().extract_from_all_games(limit=1000)'

# Assemble dataset
python -c 'from outcome_weighting_dataset import DatasetAssembler; assembler = DatasetAssembler(); examples = assembler.assemble_training_examples(); assembler.save_training_examples(examples)'
```

## 📋 Key Facts
- **21 action types** normalized from 100+ raw columns
- **12-feature state vectors** (numeric + color encoding)
- **4 weighting strategies** (importance-scaled recommended)
- **144+ decision points** extracted per 5 games (28.8 average)
- **70/15/15 train/val/test splits** precomputed
- **3,500+ lines** of production code
- **40+ tests** all passing

## 🔄 Data Flow
```
17Lands Replay Data (60GB+)
    ↓ [Task 1.1]
replay_data_public.*.csv.gz
    ↓ [Task 1.2: action_normalization.py]
action_sequences.db (127K+ actions)
    ↓ [Task 1.3: decision_point_extraction.py]
decision_points.db (144+ decision points)
    ↓ [Task 1.4: outcome_weighting_dataset.py]
training_dataset.db (weighted examples)
    ↓ [Phase 2: State Encoding - START HERE]
Fixed-shape tensors
    ↓ [Phase 3: Model Architecture]
Neural Network
    ↓ [Phase 4: Training]
Trained Model
    ↓ [Phase 5: Deployment]
Live Gameplay AI
```

## 🎓 When Starting New Work

1. **Read:** `PROJECT_MEMORY.md` (5-10 min overview)
2. **Check:** Relevant task completion summary
3. **Reference:** API guide for code you'll touch
4. **Run:** Tests to validate setup
5. **Proceed:** With task

## 📞 For Questions

- **Architecture questions:** See `PHASE_1_COMPLETION_SUMMARY.md`
- **Code/API questions:** See `ACTION_NORMALIZATION_GUIDE.md`
- **Status questions:** See `PROJECT_MEMORY.md`
- **Task details:** See `TASK_1_X_COMPLETION_SUMMARY.md`

## ⚠️ Important Notes

- **unified_cards.db:** Currently empty - populate if card enrichment needed
- **Full dataset:** 60GB+ not fully processed yet - test data validated
- **Phase 2 blocker:** Need fixed-shape tensors before Phase 3
- **GPU needed:** For training (Phase 4)

## 📝 Keep This Updated

When starting work on a new phase or making significant changes:
1. Update this file with new links
2. Update `PROJECT_MEMORY.md` with status
3. Create task completion summary
4. Document any blockers or lessons learned

---

**Last Updated:** November 7, 2025
**Status:** Phase 1 Complete, Ready for Phase 2
**Next Review:** Before Phase 2 begins
