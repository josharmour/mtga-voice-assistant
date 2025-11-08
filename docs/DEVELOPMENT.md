# Development Guide

Guide for developers working on MTGA Voice Advisor.

---

## Table of Contents
1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Common Tasks](#common-tasks)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Performance Profiling](#performance-profiling)
8. [Adding Features](#adding-features)
9. [Database Migrations](#database-migrations)
10. [Contributing](#contributing)

---

## Development Setup

### Prerequisites
- Python 3.12
- Git
- Virtual environment
- Ollama running
- All optional dependencies installed

### Initial Setup

```bash
# Clone repository
git clone https://github.com/joshu/logparser.git
cd logparser

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install all dependencies (including dev tools)
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install optional packages for full functionality
pip install kokoro-onnx chromadb sentence-transformers numpy scipy

# Verify installation
python3 -c "from app import CLIVoiceAdvisor; print('OK')"
```

### IDE Setup

**VSCode:**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

**PyCharm:**
1. Open project
2. File → Settings → Project → Python Interpreter
3. Select existing environment: `venv/bin/python`

---

## Project Structure

```
logparser/
├── app.py                      # Main orchestrator (entry point)
├── mtga.py                     # Log parsing
├── ai.py                       # LLM & RAG
├── ui.py                       # User interfaces
├── draft_advisor.py            # Draft recommendations
├── data_management.py          # Database operations
├── config_manager.py           # Preferences
├── card_rag.py                 # Card formatting
├── constants.py                # MTG constants
├── advisor.py                  # Launcher
├── launch_advisor.sh           # Shell launcher
├── requirements.txt            # Dependencies
├── tests/                      # Test suite
│   ├── test_mtga.py
│   ├── test_ai.py
│   ├── test_ui.py
│   ├── test_data.py
│   └── conftest.py             # Pytest fixtures
├── tools/                      # Development tools
│   ├── build_unified_card_database.py
│   ├── auto_updater.py
│   └── benchmark_rag.py
├── data/                       # Runtime data
│   ├── unified_cards.db
│   ├── card_stats.db
│   ├── scryfall_cache.db
│   └── chromadb/
├── logs/                       # Runtime logs
├── docs/                       # Documentation
│   ├── README.md
│   ├── SETUP.md
│   ├── API_REFERENCE.md
│   ├── DEVELOPMENT.md
│   └── CLAUDE.md
└── .git/                       # Git repository
```

---

## Architecture Overview

### Data Flow

```
Game → Player.log → LogFollower → MatchScanner → BoardState
        ↓
    GameStateManager → app.py._on_new_line()
        ↓
    GroundedPromptBuilder + RAGSystem
        ↓
    OllamaClient (LLM)
        ↓
    AIAdvisor (analysis)
        ↓
    TextToSpeech + UI
        ↓
    User (advice + voice)
```

### Module Dependencies

```
app.py (main)
├── mtga.py (log parsing)
├── ai.py (LLM)
├── ui.py (output)
├── data_management.py (databases)
├── config_manager.py (preferences)
└── draft_advisor.py (picks)

ai.py
├── data_management.py (card info)
├── card_rag.py (formatting)
└── constants.py (sets)

ui.py (standalone, no deps)

data_management.py (sqlite3)

config_manager.py (json)
```

---

## Common Tasks

### Adding a New Database Class

1. Create class with thread-local storage:

```python
# In data_management.py
import threading

class MyNewDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.thread_local = threading.local()
        self._initialize_db()

    def _get_conn(self):
        """Get thread-local connection."""
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
        return self.thread_local.conn

    def _initialize_db(self):
        """Create tables if needed."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS my_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        conn.commit()

    def query_something(self) -> List[Dict]:
        """Query database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM my_table")
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close connection."""
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()
```

2. Register in app.py if needed

3. Write tests in tests/test_data.py

### Adding a New LLM Model

1. Pull model in Ollama:

```bash
ollama pull modelname:tag
```

2. Update default in ai.py:

```python
class OllamaClient:
    def __init__(self, model: str = "modelname:tag", ...):
        ...
```

3. Or make configurable:

```bash
export OLLAMA_MODEL="modelname:tag"
python3 app.py
```

### Adding New UI Mode

1. Create class in ui.py:

```python
class MyNewUI:
    def __init__(self, board_state: BoardState = None):
        self.board_state = board_state

    def display_board_state(self, board_state: BoardState):
        # Render board state
        pass

    def display_advice(self, advice: str):
        # Show advice
        pass

    def get_command(self) -> str:
        # Wait for user input
        return ""
```

2. Register in app.py:

```python
if use_my_mode:
    ui = MyNewUI()
    ui.display_board_state(board_state)
```

3. Test thoroughly

### Adding New AI Analysis Feature

1. Add method to AIAdvisor:

```python
class AIAdvisor:
    def analyze_something(self, board_state: BoardState) -> str:
        """New analysis feature."""
        prompt = self.prompt_builder.build_analysis_prompt(board_state)
        response = self.ollama.query(prompt)
        return self._parse_response(response)
```

2. Call from app.py._on_new_line():

```python
def _on_new_line(self, line: str):
    # ... existing code ...
    analysis = self.ai_advisor.analyze_something(board_state)
    self.ui.display_analysis(analysis)
```

3. Write unit tests

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mtga.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_mtga.py::test_log_follower
```

### Writing Tests

Create test file in tests/ directory:

```python
# tests/test_myfeature.py
import pytest
from mymodule import MyClass

@pytest.fixture
def my_fixture():
    """Provide test data."""
    return MyClass()

class TestMyFeature:
    def test_basic_functionality(self, my_fixture):
        """Test that feature works."""
        result = my_fixture.do_something()
        assert result is not None

    def test_error_handling(self, my_fixture):
        """Test error cases."""
        with pytest.raises(ValueError):
            my_fixture.do_something_bad()
```

### Test Categories

**Unit Tests:** Test individual functions
```python
def test_clean_card_name():
    from ai import clean_card_name
    assert clean_card_name("Opt<nobr>") == "Opt"
```

**Integration Tests:** Test module interactions
```python
def test_board_state_update():
    from mtga import GameStateManager
    mgr = GameStateManager(log_path)
    # ... test full pipeline
```

**Regression Tests:** Prevent old bugs
```python
def test_reskin_handling():
    # Ensure Spider-Man reskins work correctly
    pass
```

---

## Debugging

### Enable Debug Logging

```python
# At top of script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOGLEVEL=DEBUG
python3 app.py
```

### Debug Specific Module

```bash
# Test card database
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from data_management import ArenaCardDatabase
db = ArenaCardDatabase()
print(db.get_card_name(1))
"
```

### Breakpoint Debugging

```python
# Use Python debugger
def my_function():
    x = 5
    breakpoint()  # Pause execution here
    return x * 2

# Or use pdb
import pdb; pdb.set_trace()
```

### Profiling Execution

```bash
# Profile function calls
python3 -m cProfile -s cumtime app.py --cli

# Memory profiling
pip install memory-profiler
python3 -m memory_profiler app.py
```

### Log File Analysis

```bash
# View logs
tail -f logs/advisor.log

# Search for errors
grep -i error logs/advisor.log

# Count log levels
grep ERROR logs/advisor.log | wc -l
grep WARNING logs/advisor.log | wc -l
```

---

## Performance Profiling

### Identify Bottlenecks

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
board_state = game_manager.get_current_board_state()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Memory Profiling

```bash
pip install memory-profiler

@profile
def my_function():
    large_list = [i for i in range(1000000)]
    return sum(large_list)

python3 -m memory_profiler script.py
```

### Database Performance

```python
import time

# Time database query
start = time.time()
card = db.get_card_name(1)
elapsed = time.time() - start
print(f"Query took {elapsed:.3f}s")

# Check table size
cursor = db._get_conn().cursor()
cursor.execute("SELECT COUNT(*) FROM cards")
count = cursor.fetchone()[0]
print(f"Table has {count} rows")
```

---

## Adding Features

### Feature Development Workflow

1. **Plan:**
   - Understand requirements
   - Design solution
   - Check dependencies

2. **Implement:**
   - Write code
   - Add logging
   - Handle errors

3. **Test:**
   - Unit tests
   - Integration tests
   - Manual testing

4. **Document:**
   - Code comments
   - Docstrings
   - Update README

5. **Review:**
   - Code quality
   - Performance
   - Thread safety

### Code Style Guidelines

**Follow PEP 8:**

```python
# Good
def calculate_win_rate(games_won: int, total_games: int) -> float:
    """Calculate win rate percentage.

    Args:
        games_won: Number of games won
        total_games: Total games played

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if total_games == 0:
        return 0.0
    return games_won / total_games

# Bad
def calcWR(w,t):
    if t==0:return 0
    return w/t
```

**Type Hints:**

```python
from typing import Dict, List, Optional

def get_cards(names: List[str]) -> Dict[str, dict]:
    """Get card info by names."""
    ...

def find_card(name: str) -> Optional[dict]:
    """Find card or None."""
    ...
```

**Docstrings:**

```python
def method(param: str) -> int:
    """One-line summary.

    Longer description if needed.

    Args:
        param: Parameter description

    Returns:
        Return value description

    Raises:
        ValueError: When param is invalid

    Example:
        >>> method("test")
        42
    """
```

### Quality Checks

```bash
# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py

# All together
black . && flake8 . && mypy .
```

---

## Database Migrations

### Adding New Column

```python
# In data_management.py
def migrate_v2():
    """Add new column to card_stats."""
    db = CardStatsDB()
    conn = db._get_conn()
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(card_stats)")
    columns = {row[1] for row in cursor.fetchall()}

    if "new_field" not in columns:
        cursor.execute("""
            ALTER TABLE card_stats
            ADD COLUMN new_field TEXT DEFAULT ''
        """)
        conn.commit()
        logging.info("Migration complete")

    db.close()

# Call during startup
if __name__ == "__main__":
    migrate_v2()
    # ... rest of app
```

### Backing Up Database

```bash
# Before making changes
cp data/card_stats.db data/card_stats.db.backup

# Restore if needed
cp data/card_stats.db.backup data/card_stats.db
```

### Verifying Data Integrity

```bash
sqlite3 data/card_stats.db "PRAGMA integrity_check;"
# Should output: ok
```

---

## Contributing

### Before Submitting PR

1. **Code Quality:**
   ```bash
   black .
   flake8 .
   mypy .
   ```

2. **Tests:**
   ```bash
   pytest --cov=.
   ```

3. **Documentation:**
   - Update docstrings
   - Update README if needed
   - Add examples

4. **Performance:**
   - Profile critical paths
   - Check memory usage
   - Verify thread safety

5. **Compatibility:**
   - Test on Windows/macOS/Linux
   - Verify Python 3.12
   - Test with Ollama models

### PR Checklist

- [ ] Code follows PEP 8
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Performance acceptable
- [ ] Thread-safe if needed
- [ ] Handles errors gracefully

### Commit Message Format

```
Type: Brief description (max 50 chars)

Longer explanation if needed (max 72 chars per line).
- Bullet points OK
- Reference issues: #123

[Optional: Fixes #456]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code reorganization
- `perf:` Performance improvement
- `test:` Test additions
- `docs:` Documentation
- `chore:` Maintenance

### Review Process

1. Automated tests run
2. Code review by maintainer
3. Performance review if needed
4. Merge to main
5. Deployment considerations

---

## Troubleshooting Development

### "Import Error" After Code Changes

```bash
# Python caches compiled files
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Reinstall in development mode
pip install -e .
```

### Thread Safety Issues

```python
# Always use thread-local storage for SQLite
class MyClass:
    def __init__(self):
        self.thread_local = threading.local()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(
                db_path,
                check_same_thread=False
            )
        return self.thread_local.conn
```

### Ollama Connection Issues

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check port
netstat -an | grep 11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows

# Restart Ollama
killall ollama  # macOS/Linux
# Windows: Restart from System Tray
```

### Database Lock Errors

```python
# Ensure connections are properly closed
try:
    db = CardStatsDB()
    # ... do work
finally:
    db.close()  # Always close

# Or use context manager if implemented
with CardStatsDB() as db:
    # ... work
    # Auto-closes
```

---

## Performance Optimization Checklist

- [ ] Profile bottlenecks with cProfile
- [ ] Cache frequently accessed data
- [ ] Use database indexes efficiently
- [ ] Minimize API calls
- [ ] Batch operations when possible
- [ ] Use appropriate data structures
- [ ] Avoid unnecessary string concatenation
- [ ] Test with realistic data volumes

---

## Resources

- Python Docs: https://docs.python.org/3.12/
- SQLite: https://www.sqlite.org/docs.html
- Ollama Docs: https://github.com/ollama/ollama
- PEP 8 Style Guide: https://pep8.org/
- MTG Rules: https://magic.wizards.com/en/rules
- 17lands: https://www.17lands.com/

