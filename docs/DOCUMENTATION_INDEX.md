# Documentation Index

Quick reference for all documentation files.

---

## Core Documentation

### 1. **README.md** (25 KB)
**Main project documentation**

Covers:
- Quick start instructions
- Module-by-module documentation (11 Python modules)
- Data structures and workflows
- Database architecture
- Common tasks
- Troubleshooting guide
- Performance notes

**Read this:** First thing to understand the overall project structure.

---

### 2. **SETUP.md** (11 KB)
**Installation and configuration guide**

Covers:
- System requirements (hardware & software)
- Step-by-step installation
  - Python setup
  - Ollama installation
  - Repository setup
  - Virtual environment
  - Dependencies
  - Card data download
  - MTGA configuration
- First run verification
- Troubleshooting setup issues
- Advanced configuration
- Environment variables
- Performance tips
- Uninstallation

**Read this:** When setting up the application for the first time.

---

### 3. **API_REFERENCE.md** (23 KB)
**Complete API documentation**

Covers:
- 7 main modules:
  - `mtga.py` - LogFollower, BoardState, GameStateManager
  - `ai.py` - OllamaClient, RAGSystem, AIAdvisor
  - `ui.py` - TextToSpeech, AdvisorTUI, AdvisorGUI
  - `draft_advisor.py` - DraftAdvisor, DraftCard
  - `data_management.py` - All database classes
  - `config_manager.py` - UserPreferences
  - `card_rag.py` - CardInfo

For each class/function:
- Purpose and usage
- Method signatures with type hints
- Parameter descriptions
- Return values
- Examples
- Common patterns

**Read this:** When developing or integrating with specific modules.

---

### 4. **DEVELOPMENT.md** (16 KB)
**Development and contribution guide**

Covers:
- Development setup
- Project structure
- Architecture overview
- Common development tasks
  - Adding database classes
  - Adding LLM models
  - Adding UI modes
  - Adding features
- Testing guide
  - Running tests
  - Writing tests
  - Test categories
- Debugging techniques
- Performance profiling
- Adding features checklist
- Code style guidelines
- Database migrations
- Contributing guidelines
- PR checklist
- Troubleshooting development

**Read this:** When modifying code or adding new features.

---

### 5. **CLAUDE.md** (in repo)
**Claude Code guidance**

Covers:
- Project overview and tech stack
- Common development commands
- High-level architecture
- Core components and responsibilities
- Key architectural patterns
- Game-domain details
- Important quirks and edge cases

**Read this:** When working with Claude Code or need quick reference.

---

## Quick Navigation by Task

### I want to...

#### **Get Started**
1. Read: README.md (Quick Start section)
2. Follow: SETUP.md (step-by-step)
3. Test: Run `python3 app.py`

#### **Understand How It Works**
1. Read: README.md (Module Documentation section)
2. Review: CLAUDE.md (Architecture Overview)
3. Study: API_REFERENCE.md (specific modules)

#### **Develop a Feature**
1. Read: DEVELOPMENT.md (Common Tasks)
2. Reference: API_REFERENCE.md (relevant modules)
3. Test: Write tests in tests/ directory
4. Submit: Follow PR guidelines in DEVELOPMENT.md

#### **Debug an Issue**
1. Check: README.md (Troubleshooting)
2. Or: SETUP.md (Setup issues)
3. Or: DEVELOPMENT.md (Debugging section)
4. Logs: Check `logs/advisor.log`

#### **Integrate with Specific Module**
1. Find module in API_REFERENCE.md
2. Study class signatures and examples
3. Review examples section at bottom of API_REFERENCE.md

#### **Install/Configure**
1. Follow: SETUP.md (step-by-step)
2. Verify: "Step 9: Verify Everything Works"

#### **Optimize Performance**
1. Read: README.md (Performance Notes)
2. Profile: DEVELOPMENT.md (Performance Profiling)
3. Optimize: DEVELOPMENT.md (Optimization Checklist)

#### **Contribute Code**
1. Read: DEVELOPMENT.md (Contributing section)
2. Follow: Code style guidelines
3. Run: Quality checks and tests
4. Submit: PR with checklist completed

---

## Documentation Statistics

| File | Size | Topics | Purpose |
|------|------|--------|---------|
| README.md | 25 KB | 11 modules | Comprehensive overview |
| SETUP.md | 11 KB | Installation | Getting started |
| API_REFERENCE.md | 23 KB | 7 modules | Developer reference |
| DEVELOPMENT.md | 16 KB | Development | Contributing |
| CLAUDE.md | ~15 KB | Architecture | Claude Code |
| **Total** | **~90 KB** | **Complete** | **Full docs** |

---

## Key Sections by Module

### `app.py` - Main Orchestrator
- README: "Module 1: app.py"
- API: "app.py" section
- Dev: "Adding new UI mode"

### `mtga.py` - Log Parsing
- README: "Module 2: mtga.py"
- API: "mtga.py" section with examples
- Dev: "Debugging log parsing"

### `ai.py` - LLM & RAG
- README: "Module 3: ai.py"
- API: "ai.py" section with examples
- Dev: "Adding new LLM model"

### `ui.py` - User Interfaces
- README: "Module 4: ui.py"
- API: "ui.py" section
- Dev: "Adding new UI mode"

### `draft_advisor.py` - Draft Picks
- README: "Module 5: draft_advisor.py"
- API: "draft_advisor.py" section

### `data_management.py` - Databases
- README: "Module 6: data_management.py"
- API: "data_management.py" section
- Dev: "Adding database class"

### `config_manager.py` - Preferences
- README: "Module 7: config_manager.py"
- API: "config_manager.py" section

### `card_rag.py` - Card Formatting
- README: "Module 8: card_rag.py"
- API: "card_rag.py" section

---

## Development Workflow

```
Start Development
    ↓
Read README.md (understand project)
    ↓
Read DEVELOPMENT.md (setup environment)
    ↓
Identify relevant module(s)
    ↓
Read API_REFERENCE.md (module details)
    ↓
Write code + tests
    ↓
Reference DEVELOPMENT.md (quality checks)
    ↓
Submit PR per guidelines
```

---

## User Workflow

```
First Time
    ↓
Read README.md (Quick Start)
    ↓
Follow SETUP.md (installation)
    ↓
Run application
    ↓
Play MTGA
    ↓
Enjoy advice!
```

---

## Search Tips

### For Specific Classes
Find in API_REFERENCE.md:
- Search for class name (e.g., "AIAdvisor")
- Review all methods with examples

### For Specific Functions
Find in README.md module sections:
- Each module lists key methods
- API_REFERENCE.md has full signatures

### For Troubleshooting
1. Check README.md "Troubleshooting" section
2. Or SETUP.md "Troubleshooting Setup Issues"
3. Or DEVELOPMENT.md "Debugging" section

### For Installation Help
1. SETUP.md "Step-by-step"
2. SETUP.md "Troubleshooting Setup"

### For Thread Safety
1. API_REFERENCE.md "data_management.py"
2. DEVELOPMENT.md "Thread Safety"

### For Performance
1. README.md "Performance Notes"
2. DEVELOPMENT.md "Performance Profiling"

---

## Documentation Maintenance

### Last Updated
- All documentation regenerated and up-to-date as of current session
- Reflects current codebase state
- All modules documented

### Coverage
- ✅ All 11 Python modules documented
- ✅ All major classes documented
- ✅ All key methods documented with examples
- ✅ Installation process documented
- ✅ API fully referenced
- ✅ Development guide included
- ✅ Troubleshooting covered

### How to Update Documentation
1. Edit relevant .md file
2. Keep sections organized
3. Update table of contents if adding sections
4. Run through linter if available
5. Commit with clear message

---

## Quick Reference Card

**Quick Start:**
```bash
source venv/bin/activate
python3 app.py
```

**Update Data:**
```bash
python3 manage_data.py --update-17lands
```

**Run Tests:**
```bash
pytest
```

**Profile Code:**
```bash
python3 -m cProfile -s cumtime app.py
```

**View Logs:**
```bash
tail -f logs/advisor.log
```

**Check Database:**
```bash
python3 -c "from data_management import ArenaCardDatabase; \
db = ArenaCardDatabase(); print(db.get_card_name(1))"
```

---

## Additional Resources

### External Documentation
- Ollama: https://github.com/ollama/ollama
- Scryfall: https://scryfall.com/docs/api
- 17lands: https://www.17lands.com/
- Python 3.12: https://docs.python.org/3.12/
- SQLite: https://www.sqlite.org/docs.html
- MTG Rules: https://magic.wizards.com/en/rules

### Local Resources
- `logs/advisor.log` - Application logs
- `data/*.db` - SQLite databases
- `tests/` - Test suite with examples
- `tools/` - Development utilities

---

## Contact & Support

For issues or questions:
1. Check appropriate documentation above
2. Review troubleshooting section
3. Check application logs
4. Examine test cases for examples

