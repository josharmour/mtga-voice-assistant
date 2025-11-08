# MTG Arena Voice Advisor - New Directory Structure

## Overview

This document describes the reorganized directory structure that improves maintainability and separates production code from research components.

## Directory Structure

```
mtga-voice-assistant/
├── main.py                    # 🎯 Main entry point (NEW)
├── src/                       # 📦 All source code (NEW)
│   ├── __init__.py
│   ├── core/                   # 🎮 Main application
│   │   ├── __init__.py
│   │   ├── app.py             # Main orchestrator
│   │   ├── mtga.py            # Log parsing & game state
│   │   ├── ai.py              # LLM integration & RAG
│   │   ├── ui.py              # GUI/TUI/CLI interfaces
│   │   ├── draft_advisor.py   # Draft recommendations
│   │   └── deck_builder.py    # Deck building utilities
│   ├── mtg_ai/                 # 🤖 MTG AI research components
│   │   ├── __init__.py
│   │   ├── mtg_transformer_encoder.py
│   │   ├── mtg_action_space.py
│   │   ├── mtg_decision_head.py
│   │   ├── mtg_training_pipeline.py
│   │   ├── mtg_evaluation_metrics.py
│   │   ├── mtg_hyperparameter_optimization.py
│   │   ├── mtg_training_monitor.py
│   │   └── mtg_model_versioning.py
│   ├── data/                   # 📊 Data management
│   │   ├── __init__.py
│   │   ├── data_management.py
│   │   └── card_rag.py
│   └── config/                 # ⚙️ Configuration
│       ├── __init__.py
│       ├── config_manager.py
│       └── constants.py
├── data/                      # 📂 Training data & databases (gitignored)
├── archive/                   # 📦 Development artifacts (gitignored)
│   └── task_scripts/          # 40+ Phase 1-4 development scripts
├── docs/                      # 📚 Documentation (consolidated from local-docs)
├── tools/                     # 🔧 Utility scripts
├── tests/                     # 🧪 Test files
└── venv/                      # 🐍 Virtual environment
```

## Key Changes

### ✅ **Before Reorganization**
- 19 Python files cluttered in root directory
- Mixed production and research code
- 38 task scripts in gitignored folder
- 42 documentation files scattered in local-docs/

### ✅ **After Reorganization**
- Clean root directory with only essential files
- **`src/`** package structure with clear separation
- **Production code** in `src/core/`
- **Research components** in `src/mtg_ai/`
- **Archived development work** in `archive/`
- **Consolidated documentation** in `docs/`

## Launch Instructions

### 🚀 **New Way (Recommended)**
```bash
# GUI mode (recommended)
python3 main.py

# TUI mode (terminal)
python3 main.py --tui

# CLI mode (simple output)
python3 main.py --cli
```

### 🔄 **Alternative (Direct from source)**
```bash
# Run from source directory
python3 src/core/app.py --tui
```

## Benefits

1. **🧹 Clean Root Directory** - Only essential files visible at project root
2. **📦 Proper Package Structure** - Python packages with `__init__.py` files
3. **🎯 Clear Separation** - Production vs research code clearly separated
4. **📁 Better Organization** - Related files grouped together
5. **🔧 Maintainability** - Easier to navigate and understand structure
6. **📚 Documentation** - Consolidated and organized
7. **📦 Development Artifacts** - Properly archived, not cluttering main code

## Import Structure

The new package structure uses relative imports:

### **Within src/core/:**
```python
from .mtga import LogFollower
from .ai import AIAdvisor
from ..data.data_management import ArenaCardDatabase
```

### **Within src/mtg_ai/:**
```python
from .mtg_transformer_encoder import MTGTransformerEncoder
from .mtg_action_space import MTGActionSpace
```

### **Cross-package imports:**
```python
# In core/app.py
from ..data.data_management import ArenaCardDatabase
from ..config.config_manager import UserPreferences
```

## Migration Notes

- ✅ All imports updated automatically
- ✅ All functionality preserved
- ✅ Application tested and working
- ✅ Documentation updated
- ✅ Git history preserved

## Future Development

- New features should follow the package structure
- Add new files to appropriate `src/` subdirectories
- Keep `archive/` for development artifacts
- Maintain separation between production and research code