# MTGA Voice Advisor

A real-time tactical advisor for Magic: The Gathering Arena that analyzes game logs and provides strategic recommendations through multiple interfaces (GUI, TUI, CLI). The system combines local LLM analysis, retrieval-augmented generation, and text-to-speech to deliver actionable game advice.

**Key Features:**
- Real-time turn-by-turn tactical advice during gameplay
- Draft pick recommendations using 17lands statistics
- Voice-enabled output with multiple TTS engines
- Multiple UI modes: GUI (Tkinter), TUI (curses), CLI
- Fully local processing (no cloud dependencies)
- Thread-safe SQLite database access

---

## Quick Start

### Prerequisites
- Python 3.12
- Ollama running locally with a model (e.g., `ollama pull mistral:7b`)
- MTGA with detailed logs enabled (Options → Account → "Detailed Logs (Plugin Support)")
- Virtual environment with dependencies installed

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download initial data
python3 manage_data.py --update-17lands
python3 manage_data.py --update-scryfall
```

### Run Application

```bash
# GUI mode (default)
python3 app.py

# TUI mode (terminal)
python3 app.py --tui

# CLI mode (simple output)
python3 app.py --cli

# Or use launcher script
./launch_advisor.sh
```

---

## Module Documentation

### 1. `app.py` - Main Application Orchestrator (1,963 LOC)

**Purpose:** Central entry point that coordinates all subsystems and manages the event loop.

**Key Classes:**
- `CLIVoiceAdvisor` - Main application controller

**Key Methods:**
```python
__init__()              # Initialize all subsystems
run()                   # Start event loop
run_gui()               # GUI mode
run_tui()               # TUI mode
run_cli()               # CLI mode
_on_new_line()          # Handle each log entry
_parse_game_action()    # Extract game state from log
```

**Responsibilities:**
- Auto-detect Arena installation and Player.log path (Windows/macOS/Linux)
- Initialize database, LLM client, TTS, UI
- Manage three UI modes (user selects at startup)
- Main event loop - calls `_on_new_line()` for each log entry
- Gracefully handle Ollama not running
- Manage user preferences persistence

**Example Usage:**
```python
advisor = CLIVoiceAdvisor(use_tui=False, use_gui=True)
advisor.run()
```

---

### 2. `mtga.py` - Log Parsing & Game State (1,187 LOC)

**Purpose:** Monitor MTGA Player.log file and reconstruct current board state.

**Key Classes:**

#### LogFollower
Monitors the Player.log file indefinitely, detecting new lines and handling log rotation.

```python
def follow(callback: Callable[[str], None])
    # Monitors file, calls callback for each new line
    # Handles log rotation via inode detection
```

**Features:**
- Auto-reconnects if file disappears
- Detects inode changes (log rotation)
- Yields lines via callback (observer pattern)

#### BoardState
Data structure representing current game state.

```python
@dataclass
class BoardState:
    your_life: int
    opponent_life: int
    your_hand: List[GameObject]
    your_battlefield: List[GameObject]
    opponent_battlefield: List[GameObject]
    stack: List[GameObject]
    mana_pools: Dict[str, int]
    combat_state: CombatHistory
    library_state: Dict[str, List[str]]
```

#### MatchScanner
Parses raw GRE (Game Rules Engine) log messages into structured game objects.

```python
def parse_gre_to_client_event(line: str) -> Optional[ClientEvent]
    # Parses GameRoomStateNotification, ChangesNotification, etc.
    # Extracts game objects, zones, player state
    # Returns ClientEvent or None
```

**Key Pattern:** Incremental updates - each log line updates BoardState

#### GameStateManager
Orchestrates LogFollower and MatchScanner, maintains game history.

```python
def __init__(log_path: str)
    # Initialize log follower and scanner

def get_current_board_state() -> BoardState
    # Return latest board state

def get_game_history() -> List[BoardState]
    # Return all historical board states
```

**Important Details:**
- Handles both Arena ID and GRE instance ID mapping
- Tracks combat state (attackers, blockers, damage assignments)
- Detects reskins and alternate card printings
- Strips HTML tags from card names (e.g., "Opt<nobr>")

---

### 3. `ai.py` - LLM Integration & RAG System (1,537 LOC)

**Purpose:** Generate tactical advice using local LLM with retrieval-augmented generation.

**Key Classes:**

#### RulesParser
Parses MTG Comprehensive Rules text file into hierarchical chunks.

```python
def parse_rules(rules_file: str) -> List[RuleEntry]
    # Chunks rules hierarchically with parent context
    # Returns (parent_rule, sub_rule) pairs for better context
```

#### RulesVectorDB
Vector database for semantic search over embedded MTG rules.

```python
def __init__(db_path: str = "data/chromadb/")
    # Initialize ChromaDB with embeddings

def query_rules(question: str, top_k: int = 5) -> List[RuleEntry]
    # Semantic search on embedded rules
    # Returns most relevant rules
```

#### CardStatsDB
Manages 17lands card statistics (thread-safe).

```python
def __init__(db_path: str = "data/card_stats.db")
    # Initialize thread-local SQLite connection

def insert_card_stats(stats: List[Dict])
    # Insert or update card statistics

def get_card_stats(card_name: str) -> Optional[Dict]
    # Get win rate, games played, avg pick position
```

**Thread-Safety:** Uses `threading.local()` for per-thread SQLite connections

#### CardMetadataDB
Aggregates card metadata from multiple sources.

```python
def get_card_by_id(grp_id: int) -> Optional[CardInfo]
    # Get complete card info (name, types, abilities, costs)

def get_cards_by_type(type_keyword: str) -> List[CardInfo]
    # Search by card type
```

#### RAGSystem
Retrieval-Augmented Generation combining rules and statistics.

```python
def query_context(board_state: BoardState, question: str) -> str
    # Query RulesVectorDB for relevant rules
    # Query CardStatsDB for card performance
    # Return grounded, cited context
```

**Key Feature:** All information is cited - "[Source: 17lands.com]" - reduces hallucinations

#### GroundedPromptBuilder
Constructs LLM prompts with complete context.

```python
def build_prompt(board_state: BoardState) -> str
    # Serialize board state to readable text
    # Include win rate statistics
    # Embed relevant MTG rules
    # Format with source citations
```

#### OllamaClient
Interface to local Ollama LLM running on localhost:11434.

```python
def __init__(model: str = "mistral:7b")
    # Connect to Ollama

def query(prompt: str) -> str
    # Send prompt, receive analysis

def is_running() -> bool
    # Check if Ollama is available
```

#### AIAdvisor
Main tactical analysis engine.

```python
def get_advice(board_state: BoardState, use_rag: bool = True) -> str
    # Generate turn-by-turn recommendations
    # Apply safety rules (e.g., don't destroy key lands)
    # Return formatted advice
```

**Safety Rules:**
- Won't suggest destroying all lands
- Validates creature counts before combat
- Checks mana availability
- Enforces game rules

---

### 4. `ui.py` - User Interfaces (1,831 LOC)

**Purpose:** Provide three interactive interfaces for displaying advice and game state.

**Key Classes:**

#### TextToSpeech
Converts text to audio with dual-engine support.

```python
def __init__(engine: str = "kokoro")
    # Primary: Kokoro (fast, ONNX-based)
    # Fallback: BarkTTS (transformer-based)

def speak(text: str, volume: int = 100)
    # Generate audio and play
    # Handles fallback if primary fails

def set_voice(voice_id: str)
    # Change voice

def set_playback_speed(speed: float)
    # Adjust speech rate
```

**Dual-Engine Strategy:**
1. Try Kokoro (fast, reliable)
2. Fall back to BarkTTS if Kokoro fails
3. Disable TTS with warning if both fail

#### AdvisorTUI
Terminal User Interface using curses library.

```python
def __init__(board_state: BoardState)
    # Initialize curses, set up windows

def display_board_state(board_state: BoardState)
    # Show formatted game state in terminal

def display_advice(advice: str)
    # Show LLM advice with colors

def get_command() -> str
    # Read user input (/help, /quit, etc.)
```

**Keyboard Commands:**
- `/help` - Show available commands
- `/quit` - Exit application
- `/status` - Show game status
- `/voice` - Toggle voice output

#### AdvisorGUI
Graphical User Interface using Tkinter.

```python
def __init__(root: tk.Tk)
    # Create windows and widgets

def display_board_state(board_state: BoardState)
    # Show formatted game state

def display_advice(advice: str)
    # Display advice in text widget

def show_settings_dialog()
    # User preferences (voice, theme, etc.)
```

**Widgets:**
- Scrolled text for advice display
- Status bar showing game info
- Settings button (voice, theme, volume)
- Always-on-top option

---

### 5. `draft_advisor.py` - Draft Recommendations (543 LOC)

**Purpose:** Recommend optimal draft picks using 17lands data.

**Key Classes:**

#### DraftCard
Represents a card in draft pack.

```python
@dataclass
class DraftCard:
    name: str
    grp_id: int
    win_rate: float
    games_in_hand: int
    improvement: float  # win_rate impact when drawn
    avg_pick_position: float
```

#### DraftAdvisor
Scores draft picks and provides recommendations.

```python
def __init__(stats_db: CardStatsDB)
    # Initialize with 17lands data

def score_draft_picks(pack: List[str], format_type: str = "PremierDraft") -> List[Recommendation]
    # Analyze pack, return scored picks
    # Assigns grades: A+ through F

def detect_format(board_state: BoardState) -> str
    # Detect if in draft/sealed from board state
```

**Grading System:**
- A+ : Top 10% win rate
- A : Top 25%
- B : Top 50%
- C : Top 75%
- D : Below 75%
- F : Unplayed cards

**Output Format:**
```
Pick Recommendations (PremierDraft):
┌─────────────────────────────────┬──────┬────────┬─────────┐
│ Card Name                       │Grade │Win Rate│Pick Pos │
├─────────────────────────────────┼──────┼────────┼─────────┤
│ Counterspell               [A+] │ 65%  │ 2.3    │
│ Removal Spell              [A]  │ 60%  │ 4.1    │
```

---

### 6. `data_management.py` - Database Management (1,141 LOC)

**Purpose:** Manage all card databases and data persistence with thread-safe connections.

**Key Classes:**

#### ScryfallDB
Local cache of Scryfall card data.

```python
def __init__(db_path: str = "data/scryfall_cache.db")
    # Thread-local SQLite connection

def get_card_by_grpId(grp_id: int) -> Optional[Dict]
    # Get card from cache or fetch from Scryfall API

def get_card_by_name(name: str) -> Optional[Dict]
    # Get card by name

def fetch_and_cache_card(grpId: int = None, name: str = None) -> Optional[Dict]
    # Fetch from Scryfall, cache locally
```

**Thread-Safety:** Each thread gets its own SQLite connection via `threading.local()`

#### CardRagDatabase
Unified card database combining Arena data and 17lands statistics.

```python
def __init__(unified_db: str = "data/unified_cards.db",
             stats_db: str = "data/card_stats.db")
    # Initialize dual thread-local connections

def get_card_by_grpid(grp_id: int) -> Optional[CardInfo]
    # Get complete card info with statistics

def get_card_by_name(card_name: str, set_code: str = None) -> Optional[CardInfo]
    # Get card info by name

def get_board_state_context(card_grp_ids: List[int]) -> str
    # Generate LLM context for all cards on board
```

#### ArenaCardDatabase
Access to unified_cards.db with caching.

```python
def __init__(db_path: str = "data/unified_cards.db",
             show_reskin_names: bool = False)
    # Load 22,700+ Arena cards with reskin support

@property
def conn
    # Thread-local connection property (backward compatible)

def get_card_name(grp_id: int) -> str
    # Get card name, showing reskin names if enabled

def get_card_data(grp_id: int) -> Optional[dict]
    # Get complete card data

def get_oracle_text(grp_id: int) -> str
def get_mana_cost(grp_id: int) -> str
def get_cmc(grp_id: int) -> Optional[float]
def get_type_line(grp_id: int) -> str
```

**Caching:** LRU cache for frequently accessed cards

#### CardStatsDB
17lands card performance statistics (thread-safe).

```python
def __init__(db_path: str = "data/card_stats.db")
    # Thread-local SQLite connection

def insert_card_stats(stats: List[Dict])
    # Bulk insert statistics

def get_card_stats(card_name: str) -> Optional[Dict]
    # Get win rate, games played, etc.

def delete_set_data(set_code: str)
    # Remove old data for a set

def search_by_name(pattern: str, limit: int = 10) -> List[Dict]
    # Search by card name pattern
```

**Thread-Safety:** Per-thread connections via `threading.local()`

#### Data Functions

```python
def download_card_data_api(set_code: str) -> List[Dict]
    # Download from 17lands API

def download_multiple_sets(set_codes: List[str]) -> int
    # Batch download multiple sets

def check_and_update_card_database() -> bool
    # Verify database exists and is accessible

def show_status()
    # Display database status report

def update_17lands_data(all_sets: bool, max_age: int)
    # Update statistics from 17lands

def update_scryfall_data()
    # Pre-populate Scryfall cache
```

---

### 7. `config_manager.py` - User Preferences (229 LOC)

**Purpose:** Persist user settings and preferences.

**Key Classes:**

#### UserPreferences
Dataclass for storing user configuration.

```python
@dataclass
class UserPreferences:
    # Window geometry
    window_geometry: Tuple[int, int] = (1200, 800)
    window_position: Tuple[int, int] = (100, 100)

    # UI settings
    ui_theme: str = "dark"  # dark or light

    # Voice settings
    voice_engine: str = "kokoro"  # kokoro or bark
    voice_volume: int = 100
    voice_playback_speed: float = 1.0
    voice_name: str = "am_adam"

    # Model settings
    ollama_model: str = "mistral:7b"
    use_rag: bool = True

    # Display options
    show_reskin_names: bool = False

    # API keys (for bug reporting)
    github_token: str = ""
    imgbb_api_key: str = ""

    @staticmethod
    def load() -> "UserPreferences"
        # Load from ~/.mtga_advisor/preferences.json

    def save() -> None
        # Save to ~/.mtga_advisor/preferences.json
```

**Storage Location:** `~/.mtga_advisor/preferences.json`

**Auto-Creation:** Directory and file created on first run

---

### 8. `card_rag.py` - Card-Specific RAG (519 LOC)

**Purpose:** Format card information for LLM context with citations.

**Key Classes:**

#### CardInfo
Complete card information for LLM.

```python
@dataclass
class CardInfo:
    grpId: int
    name: str
    oracle_text: str
    mana_cost: str
    cmc: float
    type_line: str
    color_identity: str
    power: str
    toughness: str
    rarity: str
    set_code: str
    win_rate: Optional[float] = None
    gih_win_rate: Optional[float] = None
    avg_pick_position: Optional[float] = None
    games_played: Optional[int] = None

def to_rag_citation(include_stats: bool = True) -> str
    # Format with citations: "[Source: 17lands.com]"

def to_prompt_context() -> str
    # Concise format for LLM prompts
```

---

### 9. `constants.py` - MTG Set Constants (71 LOC)

**Purpose:** Define Magic sets and fetch from Scryfall.

```python
ALL_SETS: List[str]           # All known sets
CURRENT_STANDARD: List[str]   # Currently legal Standard sets

def get_all_sets() -> List[str]
    # Fetch live from Scryfall API, cache results

def get_current_standard_sets() -> List[str]
    # Fetch live from Scryfall API
```

**Caching:** LRU cache with 15-minute expiration

---

### 10. `advisor.py` - Launcher Wrapper (147 LOC)

**Purpose:** Wrapper script for launching the application with environment setup.

```bash
./advisor.py
# or
python3 advisor.py --gui
python3 advisor.py --tui
python3 advisor.py --cli
```

**Responsibilities:**
- Activate virtual environment
- Set up Python path
- Handle command-line arguments
- Launch appropriate UI mode

---

### 11. `launch_advisor.sh` - Shell Launcher (15 LOC)

**Purpose:** Bash script for launching on Unix-like systems.

```bash
#!/bin/bash
export PATH="$HOME/ollama/bin:$PATH"
cd /home/joshu/logparser
source venv/bin/activate
python3 app.py
```

---

## Data Structures

### BoardState
Represents complete game state at any moment.

```python
@dataclass
class BoardState:
    your_life: int
    opponent_life: int
    your_hand: List[GameObject]
    your_battlefield: List[GameObject]
    opponent_battlefield: List[GameObject]
    stack: List[GameObject]
    mana_pools: Dict[str, int]           # {color: amount}
    combat_state: CombatHistory
    library_state: Dict[str, List[str]]

@dataclass
class GameObject:
    instance_id: int
    grp_id: int                          # Arena graphics ID
    name: str
    zone: str                             # hand, battlefield, etc.
    owner: str                            # you or opponent
    power: Optional[int]
    toughness: Optional[int]
    abilities: List[str]
    is_tapped: bool

@dataclass
class CombatHistory:
    current_attackers: List[int]         # Instance IDs
    current_blockers: Dict[int, int]     # {attacker_id: blocker_id}
    damage_assignments: Dict[int, int]   # {instance_id: damage}
```

### CardInfo
Complete card information with statistics.

```python
@dataclass
class CardInfo:
    grpId: int
    name: str
    oracle_text: str
    mana_cost: str
    type_line: str
    win_rate: Optional[float]            # 17lands
    gih_win_rate: Optional[float]        # Games in hand
    avg_pick_position: Optional[float]   # Average taken at
    games_played: Optional[int]
```

---

## Database Architecture

### SQLite Databases

| Database | Size | Purpose | Thread-Safe |
|----------|------|---------|-------------|
| `unified_cards.db` | ~3.7 MB | 22,700 Arena cards | ✅ Yes |
| `card_stats.db` | ~100 MB | 17lands statistics | ✅ Yes |
| `scryfall_cache.db` | Variable | Scryfall API cache | ✅ Yes |

### Vector Database

| Database | Size | Purpose |
|----------|------|---------|
| `chromadb/` | ~20 MB | Embedded MTG rules (optional) |

**Thread-Safety Implementation:**
Each class uses `threading.local()` to store per-thread SQLite connections:

```python
class DatabaseClass:
    def __init__(self):
        self.thread_local = threading.local()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(db_path, check_same_thread=False)
        return self.thread_local.conn
```

---

## Workflow: Turn-by-Turn Advice

```
1. Player makes move in MTGA
   ↓
2. Player.log updated with game event
   ↓
3. LogFollower detects new line
   ↓
4. app.py._on_new_line() called
   ↓
5. GameStateManager parses event
   - MatchScanner extracts game objects
   - BoardState updated
   ↓
6. Decision: Ask LLM?
   - Check if state significantly changed
   - Avoid spam
   ↓
7. GroundedPromptBuilder builds prompt
   - Serializes BoardState to text
   - RAGSystem queries:
     * RulesVectorDB for relevant rules
     * CardStatsDB for win rates
   - All sources cited
   ↓
8. AIAdvisor sends to Ollama
   - Generates advice
   - Applies safety rules
   ↓
9. TextToSpeech converts to audio
   - Kokoro (primary)
   - BarkTTS (fallback)
   ↓
10. UI displays advice
    - GUI: Rich text widget
    - TUI: Terminal output
    - CLI: Stdout
    ↓
11. Audio plays to user
```

---

## Common Tasks

### Running in Different Modes

```bash
# GUI (default, recommended)
python3 app.py

# TUI (terminal, no X11 needed)
python3 app.py --tui

# CLI (simple output)
python3 app.py --cli
```

### Updating Card Data

```bash
# Update 17lands statistics (current sets only)
python3 manage_data.py --update-17lands

# Update all sets (not just Standard)
python3 manage_data.py --update-17lands --all-sets

# Update Scryfall cache
python3 manage_data.py --update-scryfall

# Show status
python3 manage_data.py --status

### Downloading 17Lands Data
To download raw data from 17Lands, use the `download` command. This data is saved in a versioned directory structure to facilitate analysis of specific sets and time periods. You can specify the data type to download using the `--data-type` argument.

```bash
# Download replay data for a single set
python3 data_management.py download --set-codes MKM --data-type replay_data

# Download draft data for multiple sets
python3 data_management.py download --set-codes MKM LCI --data-type draft_data

# Download game data for a single set
python3 data_management.py download --set-codes MKM --data-type game_data
```

**Data Versioning:** The downloaded data is organized in the `data/17lands` directory, with subdirectories for each set and the date of download (e.g., `data/17lands/MKM/2025-11-06`). For more advanced data versioning and management, consider using tools like [DVC](https://dvc.org/) or [Git LFS](https://git-lfs.github.com/).
```

### Testing Components

```bash
# Test card database
python3 -c "from data_management import ArenaCardDatabase; db = ArenaCardDatabase(); print(db.get_card_name(1))"

# Test Ollama connection
python3 -c "from ai import OllamaClient; c = OllamaClient(); print(c.is_running())"

# Test log parser
python3 -c "from mtga import LogFollower; print('LogFollower OK')"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **App won't start** | Check `logs/advisor.log` for errors |
| **"Card database not found"** | Run `python3 tools/build_unified_card_database.py` |
| **Ollama not found** | Ensure Ollama is running: `ollama serve` |
| **No voice output** | Install TTS: `pip install kokoro-onnx transformers torch` |
| **No RAG context** | Install RAG: `pip install chromadb sentence-transformers` |
| **"Can't find Player.log"** | Enable detailed logs in MTGA Options |
| **Thread safety errors** | Verify SQLite check_same_thread=False is set |

---

## Architecture Highlights

### Graceful Degradation
- Ollama optional: shows warnings if not running, continues without advice
- RAG optional: works without ChromaDB/sentence-transformers
- TTS optional: Kokoro falls back to BarkTTS, then disables
- Draft advisor optional: works without numpy/scipy

### Thread Safety
- All SQLite connections use thread-local storage
- Each thread gets its own connection via `threading.local()`
- `check_same_thread=False` allows concurrent safe access

### Producer-Consumer Pipeline
- LogFollower (producer) yields lines
- GameStateManager (consumer) parses them
- AIAdvisor (consumer) analyzes state
- UI (consumer) displays results

### Data Grounding
- All LLM context is cited with sources
- Rules from official MTG Comprehensive Rules
- Statistics from 17lands
- Reduces hallucinations through verifiable sources

---

## Development

### Adding New LLM Model

Edit `ai.py:OllamaClient.__init__()`:
```python
def __init__(self, model: str = "mistral:7b"):
    self.model = model  # Change default or accept parameter
```

### Adding New TTS Engine

Edit `ui.py:TextToSpeech.__init__()`:
```python
def __init__(self, engine: str = "kokoro"):
    # Add new engine initialization
    # Update _init_engine() method
    # Add fallback chain
```

### Adding New Database

Edit `data_management.py`:
```python
class NewDatabase:
    def __init__(self):
        self.thread_local = threading.local()

    def _get_conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(...)
        return self.thread_local.conn
```

---

## Performance Notes

- **Card lookups:** Cached in-memory (LRU cache)
- **Board state:** Single BoardState per game, updated incrementally
- **Log parsing:** ~10ms per entry
- **LLM queries:** ~1-3s per analysis (local Ollama)
- **Database queries:** <1ms (local SQLite)
- **Memory usage:** ~500MB-1GB with all data loaded

---

## Files in Repository

```
logparser/
├── app.py                    # Main orchestrator
├── mtga.py                   # Log parsing & game state
├── ai.py                     # LLM & RAG system
├── ui.py                     # GUI/TUI/CLI interfaces
├── draft_advisor.py          # Draft pick recommendations
├── data_management.py        # Database management (thread-safe)
├── config_manager.py         # User preferences
├── card_rag.py               # Card formatting for RAG
├── constants.py              # MTG set constants
├── advisor.py                # Launcher wrapper
├── launch_advisor.sh         # Shell launcher
├── requirements.txt          # Python dependencies
├── CLAUDE.md                 # Claude Code guidance
├── README.md                 # This file
├── data/                     # SQLite DBs, ChromaDB, CSVs
│   ├── unified_cards.db      # Arena cards
│   ├── card_stats.db         # 17lands stats
│   ├── scryfall_cache.db     # Scryfall cache
│   ├── chromadb/             # Vector embeddings
│   └── 17lands_*.csv         # Raw statistics
├── tests/                    # Test suite
├── tools/                    # Development utilities
│   ├── build_unified_card_database.py
│   └── auto_updater.py
└── logs/                     # Runtime logs
```

---

## License & Credits

This project analyzes Magic: The Gathering Arena gameplay to provide tactical advice.

**External Data Sources:**
- 17lands.com - Draft statistics
- Scryfall.com - Card metadata
- MTG Comprehensive Rules - Official rules text

**Dependencies:**
- Ollama - Local LLM backend
- ChromaDB - Vector database
- Tkinter/Curses - UI rendering
- Kokoro/BarkTTS - Voice synthesis
