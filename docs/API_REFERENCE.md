# API Reference

Complete API documentation for all public classes and methods.

---

## Table of Contents
1. [mtga.py](#mtgapy) - Log Parsing & Game State
2. [ai.py](#aipy) - LLM & RAG System
3. [ui.py](#uipy) - User Interfaces
4. [draft_advisor.py](#draft_advisorpy) - Draft Recommendations
5. [data_management.py](#data_managementpy) - Database Management
6. [config_manager.py](#config_managerpy) - Preferences
7. [card_rag.py](#card_ragpy) - Card Formatting

---

## mtga.py

### LogFollower

Monitors MTGA Player.log file for new entries.

```python
class LogFollower:
    def __init__(self, log_path: str):
        """Initialize log follower.

        Args:
            log_path: Full path to Player.log

        Raises:
            FileNotFoundError: If log file doesn't exist
        """

    def follow(self, callback: Callable[[str], None]) -> None:
        """Monitor log and call callback for each new line.

        Args:
            callback: Function called with each new line

        Note:
            Blocks indefinitely until file stops existing
            Handles log rotation via inode detection

        Example:
            follower = LogFollower("/path/to/Player.log")
            follower.follow(lambda line: print(line))
        """

    def get_inode(self) -> int:
        """Get current file inode (for rotation detection)."""
```

### BoardState

Represents complete game state.

```python
@dataclass
class BoardState:
    your_life: int
    opponent_life: int
    your_hand: List['GameObject']
    your_battlefield: List['GameObject']
    opponent_battlefield: List['GameObject']
    stack: List['GameObject']
    mana_pools: Dict[str, int]
    combat_state: 'CombatHistory'
    library_state: Dict[str, List[str]]
    turn_count: int
    active_player: str  # "you" or "opponent"
    phase: str  # "main", "combat", "end", etc.
```

### GameObject

Represents a card instance in the game.

```python
@dataclass
class GameObject:
    instance_id: int           # Unique in game
    grp_id: int                # Arena graphics ID
    name: str
    zone: str                  # "hand", "battlefield", "graveyard", etc.
    owner: str                 # "you" or "opponent"
    power: Optional[int]
    toughness: Optional[int]
    abilities: List[str]
    is_tapped: bool
    is_attacking: bool = False
    is_blocking: bool = False
```

### CombatHistory

Tracks combat state.

```python
@dataclass
class CombatHistory:
    current_attackers: List[int]         # Instance IDs
    current_blockers: Dict[int, int]     # {attacker_id: blocker_id}
    damage_assignments: Dict[int, int]   # {instance_id: damage}
    combat_damage_assignments: Dict[int, int]
```

### MatchScanner

Parses raw GRE log messages.

```python
class MatchScanner:
    def __init__(self):
        """Initialize match scanner."""

    def parse_gre_to_client_event(self, line: str) -> Optional['ClientEvent']:
        """Parse GRE log line into client event.

        Args:
            line: Raw log line from Player.log

        Returns:
            ClientEvent if parseable, None otherwise

        Example:
            scanner = MatchScanner()
            event = scanner.parse_gre_to_client_event(log_line)
            if event:
                print(f"Game event: {event.type}")
        """
```

### GameStateManager

Orchestrates log following and state management.

```python
class GameStateManager:
    def __init__(self, log_path: str):
        """Initialize game state manager.

        Args:
            log_path: Path to Player.log
        """

    def get_current_board_state(self) -> Optional[BoardState]:
        """Get latest board state.

        Returns:
            Current BoardState or None if not in game
        """

    def get_game_history(self) -> List[BoardState]:
        """Get all historical board states.

        Returns:
            List of BoardState from start of game
        """

    def start_following(self, callback: Callable[[BoardState], None]) -> None:
        """Start monitoring log with callbacks.

        Args:
            callback: Called each time BoardState changes
        """
```

---

## ai.py

### OllamaClient

Interface to local Ollama LLM.

```python
class OllamaClient:
    def __init__(self, model: str = "mistral:7b", host: str = "localhost:11434"):
        """Initialize Ollama client.

        Args:
            model: Model name (e.g., "mistral:7b", "llama2:7b")
            host: Ollama server address
        """

    def is_running(self) -> bool:
        """Check if Ollama is running.

        Returns:
            True if reachable, False otherwise
        """

    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """Send prompt to LLM.

        Args:
            prompt: Input text
            temperature: Creativity (0.0=deterministic, 1.0=random)

        Returns:
            LLM response

        Raises:
            ConnectionError: If Ollama not running

        Example:
            client = OllamaClient()
            response = client.query("What should I do next?")
            print(response)
        """

    def list_available_models(self) -> List[str]:
        """Get list of available models.

        Returns:
            List of model names installed in Ollama
        """
```

### RulesParser

Parses MTG Comprehensive Rules.

```python
class RulesParser:
    def __init__(self, rules_file: str = "data/rules.txt"):
        """Initialize rules parser.

        Args:
            rules_file: Path to Comprehensive Rules text file
        """

    def parse(self) -> List[Dict[str, str]]:
        """Parse rules file into structured entries.

        Returns:
            List of {rule_number, text, parent_rule} dicts
        """
```

### RulesVectorDB

Vector database for rules semantic search.

```python
class RulesVectorDB:
    def __init__(self, db_path: str = "data/chromadb/"):
        """Initialize vector database.

        Args:
            db_path: ChromaDB directory path
        """

    def query_rules(self, question: str, top_k: int = 5) -> List[str]:
        """Semantic search for relevant rules.

        Args:
            question: Query text
            top_k: Number of results to return

        Returns:
            List of relevant rule texts

        Example:
            db = RulesVectorDB()
            rules = db.query_rules("Can I draw cards in combat?")
            for rule in rules:
                print(rule)
        """
```

### CardStatsDB

Thread-safe access to 17lands statistics.

```python
class CardStatsDB:
    def __init__(self, db_path: str = "data/card_stats.db"):
        """Initialize card stats database.

        Args:
            db_path: Path to SQLite database
        """

    def insert_card_stats(self, stats: List[Dict[str, Any]]) -> None:
        """Insert or update card statistics.

        Args:
            stats: List of stat dicts with keys:
                - card_name (str)
                - set_code (str)
                - win_rate (float)
                - games_played (int)
                - avg_taken_at (float)

        Example:
            db = CardStatsDB()
            db.insert_card_stats([{
                'card_name': 'Lightning Bolt',
                'set_code': 'MIR',
                'win_rate': 0.65,
                'games_played': 5000
            }])
        """

    def get_card_stats(self, card_name: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific card.

        Args:
            card_name: Card name to look up

        Returns:
            Dict with card stats or None if not found

        Example:
            db = CardStatsDB()
            stats = db.get_card_stats("Lightning Bolt")
            if stats:
                print(f"Win rate: {stats['win_rate'] * 100:.1f}%")
        """

    def delete_set_data(self, set_code: str) -> None:
        """Delete all stats for a set.

        Args:
            set_code: Set code (e.g., "MIR")
        """

    def search_by_name(self, pattern: str, limit: int = 10) -> List[Dict]:
        """Search cards by name pattern.

        Args:
            pattern: SQL LIKE pattern (e.g., "%Bolt%")
            limit: Max results

        Returns:
            List of matching card stats
        """

    def close(self) -> None:
        """Close database connection."""
```

### GroundedPromptBuilder

Constructs LLM prompts with context.

```python
class GroundedPromptBuilder:
    def __init__(self, board_state: 'BoardState', card_db: 'CardRagDatabase'):
        """Initialize prompt builder.

        Args:
            board_state: Current game state
            card_db: Database for card info
        """

    def build_prompt(self, question: str = None) -> str:
        """Build complete prompt for LLM.

        Args:
            question: Optional specific question

        Returns:
            Formatted prompt with context and sources

        Example:
            builder = GroundedPromptBuilder(board_state, card_db)
            prompt = builder.build_prompt("What should I do next?")
            print(prompt)
        """
```

### AIAdvisor

Main tactical analysis engine.

```python
class AIAdvisor:
    def __init__(self, card_db: 'CardRagDatabase', model: str = "mistral:7b", use_rag: bool = True):
        """Initialize AI advisor.

        Args:
            card_db: Card database
            model: LLM model name
            use_rag: Enable RAG system
        """

    def get_advice(self, board_state: 'BoardState') -> str:
        """Generate tactical advice.

        Args:
            board_state: Current game state

        Returns:
            Advice text with recommendations

        Example:
            advisor = AIAdvisor(card_db)
            advice = advisor.get_advice(board_state)
            print(advice)
        """

    def analyze_mulligan(self, hand: List['GameObject']) -> str:
        """Analyze opening hand.

        Args:
            hand: Cards in opening hand

        Returns:
            Mulligan recommendation
        """
```

---

## ui.py

### TextToSpeech

Converts text to audio.

```python
class TextToSpeech:
    def __init__(self, engine: str = "kokoro", voice: str = "am_adam"):
        """Initialize TTS engine.

        Args:
            engine: "kokoro" (primary) or "bark" (fallback)
            voice: Voice name (depends on engine)
        """

    def speak(self, text: str, volume: int = 100) -> None:
        """Convert text to speech and play.

        Args:
            text: Text to speak
            volume: Volume level (0-100)

        Example:
            tts = TextToSpeech()
            tts.speak("Your opponent is attacking!")
        """

    def set_voice(self, voice: str) -> None:
        """Change voice.

        Args:
            voice: Voice identifier
        """

    def set_playback_speed(self, speed: float) -> None:
        """Set speech rate.

        Args:
            speed: Playback speed (0.5=half speed, 2.0=double speed)
        """

    def set_volume(self, volume: int) -> None:
        """Set volume.

        Args:
            volume: Volume (0-100)
        """
```

### AdvisorTUI

Terminal User Interface.

```python
class AdvisorTUI:
    def __init__(self, board_state: 'BoardState' = None):
        """Initialize TUI.

        Args:
            board_state: Initial board state (optional)
        """

    def display_board_state(self, board_state: 'BoardState') -> None:
        """Display formatted game state.

        Args:
            board_state: Game state to display
        """

    def display_advice(self, advice: str) -> None:
        """Display tactical advice.

        Args:
            advice: Advice text
        """

    def get_command(self) -> str:
        """Wait for and return user command.

        Returns:
            User input (e.g., "/quit", "/help")
        """
```

### AdvisorGUI

Graphical User Interface.

```python
class AdvisorGUI:
    def __init__(self, root: 'tk.Tk'):
        """Initialize GUI.

        Args:
            root: Tkinter root window
        """

    def display_board_state(self, board_state: 'BoardState') -> None:
        """Display game state in GUI.

        Args:
            board_state: Game state to display
        """

    def display_advice(self, advice: str) -> None:
        """Display advice in text widget.

        Args:
            advice: Advice text
        """

    def show_settings_dialog(self) -> None:
        """Show user preferences dialog."""

    def update_status_bar(self, status: str) -> None:
        """Update status bar.

        Args:
            status: Status message
        """
```

---

## draft_advisor.py

### DraftCard

Represents a card in draft.

```python
@dataclass
class DraftCard:
    name: str
    grp_id: int
    win_rate: float
    games_in_hand: int
    improvement: float              # Impact when drawn
    avg_pick_position: float
```

### DraftAdvisor

Draft pick recommendations.

```python
class DraftAdvisor:
    def __init__(self, stats_db: 'CardStatsDB'):
        """Initialize draft advisor.

        Args:
            stats_db: Card statistics database
        """

    def score_draft_picks(self, pack: List[str],
                         current_picks: List[str] = None,
                         format_type: str = "PremierDraft") -> List['Recommendation']:
        """Score draft pack.

        Args:
            pack: List of card names in pack
            current_picks: Cards already drafted (optional)
            format_type: Draft format

        Returns:
            List of recommendations with grades

        Example:
            advisor = DraftAdvisor(stats_db)
            recs = advisor.score_draft_picks(["Card A", "Card B"])
            for rec in recs:
                print(f"{rec.name}: {rec.grade}")
        """

    def detect_format(self, board_state: 'BoardState') -> Optional[str]:
        """Detect draft format from board state.

        Args:
            board_state: Current game state

        Returns:
            Format name or None
        """
```

---

## data_management.py

### ScryfallDB

Scryfall card data cache.

```python
class ScryfallDB:
    def __init__(self, db_path: str = "data/scryfall_cache.db"):
        """Initialize Scryfall cache.

        Args:
            db_path: Path to SQLite database
        """

    def get_card_by_grpId(self, grpId: int) -> Optional[Dict]:
        """Get card by Arena graphics ID.

        Args:
            grpId: Arena graphics ID

        Returns:
            Card dict or None if not found
        """

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """Get card by name.

        Args:
            name: Card name

        Returns:
            Card dict or None
        """

    def fetch_and_cache_card(self, grpId: int = None, name: str = None) -> Optional[Dict]:
        """Fetch from Scryfall and cache.

        Args:
            grpId: Arena graphics ID (use one)
            name: Card name (use one)

        Returns:
            Card dict or None
        """

    def close(self) -> None:
        """Close database connection."""
```

### CardRagDatabase

Unified card database with RAG.

```python
class CardRagDatabase:
    def __init__(self, unified_db: str = "data/unified_cards.db",
                 stats_db: str = "data/card_stats.db"):
        """Initialize card RAG database.

        Args:
            unified_db: Arena cards database
            stats_db: 17lands statistics database
        """

    def get_card_by_grpid(self, grp_id: int) -> Optional['CardInfo']:
        """Get card with statistics.

        Args:
            grp_id: Arena graphics ID

        Returns:
            CardInfo or None
        """

    def get_card_by_name(self, card_name: str, set_code: str = None) -> Optional['CardInfo']:
        """Get card by name.

        Args:
            card_name: Card name
            set_code: Optional set code to narrow search

        Returns:
            CardInfo or None
        """

    def get_board_state_context(self, card_grp_ids: List[int]) -> str:
        """Generate context for board state.

        Args:
            card_grp_ids: List of grpIds on board

        Returns:
            Formatted context with sources
        """

    def search_cards_by_type(self, type_keyword: str,
                            set_code: str = None) -> List['CardInfo']:
        """Search by card type.

        Args:
            type_keyword: Type to search for
            set_code: Optional set filter

        Returns:
            List of matching cards
        """

    def search_by_ability(self, ability_keyword: str,
                         set_code: str = None) -> List['CardInfo']:
        """Search by ability text.

        Args:
            ability_keyword: Ability to search for
            set_code: Optional set filter

        Returns:
            List of matching cards
        """

    def close(self) -> None:
        """Close database connections."""
```

### ArenaCardDatabase

Arena unified cards database.

```python
class ArenaCardDatabase:
    def __init__(self, db_path: str = "data/unified_cards.db",
                 show_reskin_names: bool = False):
        """Initialize Arena card database.

        Args:
            db_path: Path to unified_cards.db
            show_reskin_names: Show reskin names instead of canonical
        """

    @property
    def conn:
        """Thread-local database connection."""

    def get_card_name(self, grp_id: int) -> str:
        """Get card name.

        Args:
            grp_id: Arena graphics ID

        Returns:
            Card name or "Unknown(grp_id)"
        """

    def get_card_data(self, grp_id: int) -> Optional[dict]:
        """Get full card data.

        Args:
            grp_id: Arena graphics ID

        Returns:
            Card dict or None
        """

    def get_oracle_text(self, grp_id: int) -> str:
        """Get card abilities.

        Args:
            grp_id: Arena graphics ID

        Returns:
            Oracle text
        """

    def get_mana_cost(self, grp_id: int) -> str:
        """Get mana cost.

        Args:
            grp_id: Arena graphics ID

        Returns:
            Mana cost (e.g., "{2}{U}{U}")
        """

    def get_cmc(self, grp_id: int) -> Optional[float]:
        """Get converted mana cost.

        Args:
            grp_id: Arena graphics ID

        Returns:
            CMC or None
        """

    def get_type_line(self, grp_id: int) -> str:
        """Get card type.

        Args:
            grp_id: Arena graphics ID

        Returns:
            Type line (e.g., "Creature - Elf Wizard")
        """
```

### Management Functions

```python
def check_and_update_card_database() -> bool:
    """Verify database exists and is accessible.

    Returns:
        True if OK, False if missing or error
    """

def download_card_data_api(set_code: str) -> List[Dict]:
    """Download stats from 17lands API.

    Args:
        set_code: Set code

    Returns:
        List of card stat dicts
    """

def download_multiple_sets(set_codes: List[str]) -> int:
    """Download multiple sets.

    Args:
        set_codes: List of set codes

    Returns:
        Total cards downloaded
    """

def update_17lands_data(all_sets: bool = False, max_age: int = 30):
    """Update 17lands database.

    Args:
        all_sets: Update all sets or just Standard
        max_age: Max age in days before updating
    """

def update_scryfall_data():
    """Update Scryfall cache."""

def show_status():
    """Display database status report."""
```

---

## config_manager.py

### UserPreferences

User configuration.

```python
@dataclass
class UserPreferences:
    # Window settings
    window_geometry: Tuple[int, int] = (1200, 800)
    window_position: Tuple[int, int] = (100, 100)

    # UI
    ui_theme: str = "dark"

    # Voice
    voice_engine: str = "kokoro"
    voice_volume: int = 100
    voice_playback_speed: float = 1.0
    voice_name: str = "am_adam"

    # Model
    ollama_model: str = "mistral:7b"
    use_rag: bool = True

    # Display
    show_reskin_names: bool = False

    # API keys
    github_token: str = ""
    imgbb_api_key: str = ""

    @staticmethod
    def load() -> 'UserPreferences':
        """Load preferences from disk.

        Returns:
            UserPreferences or default if file not found
        """

    def save(self) -> None:
        """Save preferences to disk."""
```

---

## card_rag.py

### CardInfo

Complete card information.

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

    def to_rag_citation(self, include_stats: bool = True) -> str:
        """Format for RAG context.

        Returns:
            Formatted text with citations
        """

    def to_prompt_context(self) -> str:
        """Format for LLM prompt.

        Returns:
            Concise formatted text
        """
```

---

## Common Patterns

### Using Databases with Threading

All databases are thread-safe. Each thread gets its own connection:

```python
from data_management import ArenaCardDatabase
import threading

db = ArenaCardDatabase()

def worker():
    # Each thread gets its own connection automatically
    card_name = db.get_card_name(1)
    print(f"Card: {card_name}")

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

db.close()  # Clean up
```

### Building Custom Prompts

```python
from ai import GroundedPromptBuilder, RAGSystem
from data_management import CardRagDatabase

card_db = CardRagDatabase()
rag = RAGSystem(card_db)
builder = GroundedPromptBuilder(board_state, card_db)

prompt = builder.build_prompt("What is the best move?")
print(prompt)  # Includes rules, stats, sources
```

### Monitoring Game State

```python
from mtga import GameStateManager

def on_state_change(board_state):
    print(f"Your life: {board_state.your_life}")
    print(f"Opponent life: {board_state.opponent_life}")

mgr = GameStateManager("/path/to/Player.log")
mgr.start_following(on_state_change)
```

---

## Error Handling

All methods include proper error handling. Common exceptions:

- `FileNotFoundError` - Missing database or log file
- `ConnectionError` - Ollama not running
- `sqlite3.DatabaseError` - Database corruption
- `requests.RequestException` - Network error

Recommended error handling:

```python
try:
    advisor = AIAdvisor(card_db)
    advice = advisor.get_advice(board_state)
except ConnectionError:
    print("Error: Ollama not running")
except Exception as e:
    print(f"Error: {e}")
    logging.error(f"Failed to get advice", exc_info=True)
```

