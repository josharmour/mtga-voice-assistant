# MTGA Voice Advisor - Optimization & Refactor TODO

This document outlines optimization opportunities identified through codebase analysis. Issues are prioritized and organized by area.

---

## Priority Legend

- **P0** - Critical performance issues, high impact, should be addressed first
- **P1** - Important optimizations, medium impact
- **P2** - Architectural improvements, high effort but significant long-term benefit
- **P3** - Nice-to-have improvements, low priority

---

## 1. Log Parsing (`src/core/mtga.py`)

### P1: Compiled Regex for Game State Detection
**Location:** `src/core/app.py:696-705`

**Current Issue:**
```python
has_game_state_change = any(indicator in line for indicator in [
    'GameStateMessage',
    'ActionsAvailableReq',
    'turnInfo',
    # ...7 more strings
])
```
This runs on EVERY log line, performing up to 9 substring searches.

**Solution:**
```python
import re
GAME_STATE_PATTERN = re.compile(
    r'GameStateMessage|ActionsAvailableReq|turnInfo|priorityPlayer|gameObjects|zones|GameStage_Start|mulligan'
)
has_game_state_change = GAME_STATE_PATTERN.search(line) is not None
```

**Files:** `src/core/app.py`

---

### P3: Streaming JSON Parsing
**Location:** `src/core/mtga.py:1044-1093`

**Current Issue:**
Manual brace counting for JSON detection is error-prone (doesn't handle strings containing braces) and buffers entire JSON object in memory before parsing.

**Solution:**
- Consider using `ijson` for streaming JSON parsing of large GRE messages
- Add validation for malformed JSON earlier in the pipeline

**Files:** `src/core/mtga.py`

---

### P1: Timestamp Parsing Optimization
**Location:** `src/core/mtga.py:238-254`

**Current Issue:**
`parse_timestamp()` is called for EVERY line processed, using string operations and `datetime.strptime()` even when not needed.

**Solution:**
- Only parse timestamps when actually needed (e.g., for freshness checks)
- Cache the last parsed timestamp to avoid redundant parsing
- Move timestamp parsing out of the hot path

**Files:** `src/core/mtga.py`

---

### P2: Consolidate Draft Event Detection
**Location:** `src/core/mtga.py:928-1041`

**Current Issue:**
Draft event detection is spread across multiple regex patterns and conditional blocks, making it hard to maintain and optimize.

**Solution:**
Create a dedicated `DraftEventParser` class with a single entry point:
```python
class DraftEventParser:
    def __init__(self, callbacks: Dict[str, Callable]):
        self.callbacks = callbacks
        self._patterns = self._compile_patterns()

    def parse(self, line: str) -> bool:
        """Returns True if line was a draft event."""
        # Consolidated draft event handling
```

**Files:** `src/core/mtga.py` (extract to new file `src/core/draft_parser.py`)

---

## 2. Board State Tracking (`src/core/mtga.py`)

### P0: Zone-Based Object Caching
**Location:** `src/core/mtga.py:1139-1210`

**Current Issue:**
Every call to `get_current_board_state()` iterates ALL game objects and performs multiple string comparisons per object. This is O(n) on every board state request.

**Solution:**
Maintain pre-sorted zone lists that update incrementally:
```python
class MatchScanner:
    def __init__(self):
        # Zone-based object caches
        self._zone_objects: Dict[int, Set[int]] = {}  # zone_id -> set of instance_ids

    def _update_object_zone(self, obj: GameObject, old_zone: int, new_zone: int):
        if old_zone in self._zone_objects:
            self._zone_objects[old_zone].discard(obj.instance_id)
        if new_zone not in self._zone_objects:
            self._zone_objects[new_zone] = set()
        self._zone_objects[new_zone].add(obj.instance_id)
```

**Files:** `src/core/mtga.py`

---

### P0: Pre-Resolve Card Names on Object Creation
**Location:** `src/core/mtga.py:1144-1148`

**Current Issue:**
```python
for obj in self.scanner.game_objects.values():
    if not obj.name or obj.name.startswith("Unknown Card"):
        obj.name = self.card_lookup.get_card_name(obj.grp_id)
```
This iterates all objects on every board state request.

**Solution:**
- Resolve card names ONCE when the object is first created or when `grp_id` changes
- Move name resolution to `_parse_game_objects()` instead of `get_current_board_state()`
- Add a flag to track if name has been resolved

**Files:** `src/core/mtga.py`

---

### P1: Zone Type Enum Instead of String Comparisons
**Location:** Throughout `src/core/mtga.py`

**Current Issue:**
```python
if "Hand" in zone_type:
if "Battlefield" in zone_type:
if "Graveyard" in zone_type:
```
String substring checks are slow and fragile.

**Solution:**
```python
from enum import Enum, auto

class ZoneType(Enum):
    UNKNOWN = auto()
    HAND = auto()
    BATTLEFIELD = auto()
    GRAVEYARD = auto()
    EXILE = auto()
    LIBRARY = auto()
    STACK = auto()
    COMMAND = auto()

# In MatchScanner:
ZONE_TYPE_MAP = {
    "ZoneType_Hand": ZoneType.HAND,
    "ZoneType_Battlefield": ZoneType.BATTLEFIELD,
    # etc.
}
```

**Files:** `src/core/mtga.py`

---

### P2: BoardState Diff-Based Updates
**Location:** `src/core/mtga.py:1118-1137`

**Current Issue:**
A new `BoardState` object is created from scratch on every call, allocating new lists even when they're empty.

**Solution:**
- Cache the last `BoardState` and only update changed fields
- Implement a change detection mechanism to know when zones have been modified
- Consider using `__slots__` on `BoardState` for memory efficiency

**Files:** `src/core/mtga.py`

---

## 3. UI Update Patterns (`src/core/ui.py`, `src/core/app.py`)

### P1: Batch UI Updates with Dirty Flags
**Location:** `src/core/ui.py:1048-1113`

**Current Issue:**
Every UI update creates a new closure and schedules it via `after(0, ...)`:
```python
def set_status(self, text: str):
    def _update():
        self.status_label.config(text=text)
    self.root.after(0, _update)
```
This creates excessive scheduler overhead.

**Solution:**
```python
class AdvisorGUI:
    def __init__(self):
        self._pending_updates = {}
        self._update_scheduled = False

    def _schedule_update(self, key: str, value):
        self._pending_updates[key] = value
        if not self._update_scheduled:
            self._update_scheduled = True
            self.root.after(16, self._flush_updates)  # ~60fps

    def _flush_updates(self):
        self._update_scheduled = False
        for key, value in self._pending_updates.items():
            self._apply_update(key, value)
        self._pending_updates.clear()
```

**Files:** `src/core/ui.py`

---

### P1: Adaptive Log Queue Processing
**Location:** `src/core/ui.py:701-727`

**Current Issue:**
During startup, thousands of log lines flood the queue, causing UI starvation. Fixed batch size of 100 lines every 200ms.

**Solution:**
Implement adaptive batch sizing based on queue depth:
```python
def _process_log_queue(self):
    queue_depth = len(self.log_queue)

    # Adaptive batch size: process more when backlogged
    if queue_depth > 5000:
        batch_size = 1000  # Aggressive catch-up
    elif queue_depth > 1000:
        batch_size = 500
    else:
        batch_size = min(50, queue_depth)

    # Process batch...
```

**Files:** `src/core/ui.py`

---

### P2: Diff-Based Text Widget Updates
**Location:** `src/core/secondary_window.py:36-55`

**Current Issue:**
```python
def update_text(self, lines):
    self.text_area.delete(1.0, tk.END)  # Deletes ALL content
    for line in lines:
        # Re-inserts everything
```
Full content replacement on every update is inefficient.

**Solution:**
- Implement diff-based updates (only modify changed lines)
- Pre-compute line formatting in the caller, pass `(line, tag)` tuples
- Consider using a virtual list that only renders visible lines

**Files:** `src/core/secondary_window.py`

---

### P2: Extract BoardStateFormatter Class
**Location:** `src/core/app.py:356-479`

**Current Issue:**
Board state formatting logic (`_format_board_state_for_display`) is embedded in the main app class, mixing presentation with business logic.

**Solution:**
Extract a dedicated formatter:
```python
class BoardStateFormatter:
    def format_for_display(self, board_state: BoardState) -> List[str]:
        """Format board state as lines for display."""

    def format_card(self, card: GameObject) -> str:
        """Format a single card for display."""
```

**Files:** `src/core/app.py` (extract to new file `src/core/formatters.py`)

---

## 4. AI Context & Prompt Generation (`src/core/llm/`)

### P1: Shared PromptBuilder Class
**Location:** `src/core/llm/google_advisor.py` vs `src/core/llm/ollama_advisor.py`

**Current Issue:**
GeminiAdvisor builds rich context with card text:
```python
context = self._build_context(board_state)
```

OllamaAdvisor sends raw dict dump:
```python
content=f"Here is the board state:\n{board_state}\n..."  # Raw dict!
```

**Solution:**
Create shared prompt building infrastructure:
```python
# src/core/llm/prompt_builder.py
class MTGPromptBuilder:
    def __init__(self, scryfall: ScryfallClient):
        self.scryfall = scryfall

    def build_tactical_prompt(self, board_state: Dict) -> str:
        """Build rich context string for any LLM."""
        context = self._build_context(board_state)
        return f"""
        You are a Magic: The Gathering expert advisor...
        {context}
        """

    def _build_context(self, board_state: Dict) -> str:
        # Shared implementation from GeminiAdvisor._build_context
```

**Files:**
- Create `src/core/llm/prompt_builder.py`
- Update `src/core/llm/google_advisor.py`
- Update `src/core/llm/ollama_advisor.py`
- Update `src/core/llm/openai_advisor.py`
- Update `src/core/llm/anthropic_advisor.py`

---

### P0: Pre-Cache Card Text in GameObjects
**Location:** `src/core/llm/google_advisor.py:129-131`

**Current Issue:**
```python
if grp_id:
    card_data = self.scryfall.get_card_by_arena_id(grp_id)
```
For each card in prompt building, a potential Scryfall lookup occurs.

**Solution:**
Add card text directly to `GameObject` during state building:
```python
@dataclasses.dataclass
class GameObject:
    # ... existing fields
    oracle_text: str = ""  # Pre-fetched card text
    type_line: str = ""
    mana_cost: str = ""
```

Resolve in `_parse_game_objects()` or `get_current_board_state()` once, then reuse in prompt building.

**Files:**
- `src/core/mtga.py` (add fields to GameObject, populate during parsing)
- `src/core/llm/google_advisor.py` (use pre-cached data)

---

### P2: Prompt Token Budget Management
**Location:** `src/core/llm/google_advisor.py:50-66`

**Current Issue:**
The prompt includes full oracle text for every card, deck list summary, with no token budget management. Large board states may exceed context limits.

**Solution:**
```python
class MTGPromptBuilder:
    MAX_PROMPT_TOKENS = 4000  # Configurable

    def build_tactical_prompt(self, board_state: Dict) -> str:
        context = self._build_context(board_state)

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(context) // 4

        if estimated_tokens > self.MAX_PROMPT_TOKENS:
            context = self._compress_context(context, board_state)

        return prompt

    def _compress_context(self, context: str, board_state: Dict) -> str:
        # Summarize oracle text for common cards
        # Only include relevant deck cards
        # Use shorthand for basic lands
```

**Files:** `src/core/llm/prompt_builder.py`

---

### P2: Prompt Caching for Unchanged States
**Location:** All LLM advisors

**Current Issue:**
If the board state hasn't changed significantly, the same prompt is rebuilt from scratch.

**Solution:**
```python
class MTGPromptBuilder:
    def __init__(self):
        self._last_board_hash = None
        self._cached_prompt = None

    def build_tactical_prompt(self, board_state: Dict) -> str:
        current_hash = self._compute_board_hash(board_state)

        if current_hash == self._last_board_hash:
            return self._cached_prompt

        prompt = self._build_prompt_impl(board_state)
        self._last_board_hash = current_hash
        self._cached_prompt = prompt
        return prompt

    def _compute_board_hash(self, board_state: Dict) -> int:
        # Hash relevant state fields
```

**Files:** `src/core/llm/prompt_builder.py`

---

## 5. Architectural Improvements

### P2: Event-Driven Architecture
**Location:** Throughout codebase

**Current Issue:**
Polling-based log following with callback chain creates tight coupling between components.

**Solution:**
Introduce an event bus pattern:
```python
# src/core/events.py
from typing import Callable, Dict, List, Any

class EventBus:
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, data: Any):
        for handler in self._handlers.get(event_type, []):
            handler(data)

# Event types:
# - "board_state_changed" -> BoardState
# - "priority_gained" -> {turn, phase}
# - "draft_pack_opened" -> DraftPackData
# - "game_started" -> None
# - "game_ended" -> GameResult
```

**Files:** Create `src/core/events.py`, update `src/core/app.py`, `src/core/mtga.py`

---

### P2: Separate Domain Model from Presentation
**Location:** `src/core/mtga.py` (BoardState)

**Current Issue:**
`BoardState` contains both data and implicit formatting concerns, mixing domain logic with presentation.

**Solution:**
```
GameState (domain) → BoardStateView (UI presentation) → GUI/TUI/API
```

```python
# src/core/domain/game_state.py
@dataclass
class GameState:
    """Pure domain model - no presentation concerns."""
    turn: int
    phase: Phase
    active_player: PlayerId
    # etc.

# src/core/views/board_state_view.py
class BoardStateView:
    """Presentation layer for GameState."""
    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def to_display_lines(self) -> List[str]:
        # Formatting logic here
```

**Files:** Create `src/core/domain/` directory structure

---

### P3: Performance Monitoring System
**Location:** `src/core/mtga.py:1273-1274` (has basic timing)

**Current Issue:**
Only `get_current_board_state()` has timing. No comprehensive performance monitoring.

**Solution:**
```python
# src/core/monitoring.py
from contextlib import contextmanager
from collections import defaultdict
import time

class PerformanceMonitor:
    _instance = None

    def __init__(self):
        self.metrics = defaultdict(list)
        self.enabled = True

    @classmethod
    def get(cls) -> 'PerformanceMonitor':
        if cls._instance is None:
            cls._instance = PerformanceMonitor()
        return cls._instance

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.metrics[name].append(elapsed)

    def report(self) -> Dict[str, Dict]:
        return {
            name: {
                "count": len(times),
                "avg_ms": sum(times) / len(times) * 1000,
                "max_ms": max(times) * 1000,
                "total_ms": sum(times) * 1000
            }
            for name, times in self.metrics.items()
        }

# Usage:
with PerformanceMonitor.get().measure("parse_gre_message"):
    self._parse_game_state_message(message)
```

**Files:** Create `src/core/monitoring.py`, add instrumentation throughout

---

### P3: Configuration-Driven LLM Selection
**Location:** `src/core/ai.py`, `src/core/llm/*.py`

**Current Issue:**
Each advisor is a separate class with slightly different interfaces.

**Solution:**
Define a common protocol and configuration:
```python
# src/core/llm/base.py
from typing import Protocol, Dict, List

class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 500
    temperature: float = 0.7
    api_key: Optional[str] = None

class LLMAdapter(Protocol):
    def complete(self, prompt: str) -> str: ...

class BaseMTGAdvisor:
    def __init__(self, llm: LLMAdapter, prompt_builder: MTGPromptBuilder):
        self.llm = llm
        self.prompt_builder = prompt_builder

    def get_tactical_advice(self, board_state: Dict) -> str:
        prompt = self.prompt_builder.build_tactical_prompt(board_state)
        return self.llm.complete(prompt)
```

**Files:**
- Create `src/core/llm/base.py`
- Refactor all advisor classes to use common base

---

## Implementation Order

### Phase 1: Quick Wins (P0 + easy P1)
1. [ ] Pre-resolve card names on object creation (`mtga.py`)
2. [ ] Compiled regex for game state detection (`app.py`)
3. [ ] Create shared PromptBuilder class (`llm/prompt_builder.py`)
4. [ ] Fix OllamaAdvisor to use proper context building

### Phase 2: Performance (remaining P1)
5. [ ] Zone-based object caching (`mtga.py`)
6. [ ] Timestamp parsing optimization (`mtga.py`)
7. [ ] Batch UI updates with dirty flags (`ui.py`)
8. [ ] Adaptive log queue processing (`ui.py`)
9. [ ] Zone type enum (`mtga.py`)

### Phase 3: Architecture (P2)
10. [ ] Event-driven architecture (`events.py`)
11. [ ] Extract BoardStateFormatter (`formatters.py`)
12. [ ] Diff-based text widget updates (`secondary_window.py`)
13. [ ] Prompt token budget management
14. [ ] Prompt caching for unchanged states
15. [ ] BoardState diff-based updates
16. [ ] Consolidate draft event detection

### Phase 4: Polish (P3)
17. [ ] Streaming JSON parsing
18. [ ] Performance monitoring system
19. [ ] Configuration-driven LLM selection
20. [ ] Separate domain model from presentation

---

## Testing Recommendations

For each optimization:
1. Add unit tests for the specific component being modified
2. Run performance benchmarks before/after
3. Test with large log files (startup scenario)
4. Test during active gameplay (real-time scenario)
5. Verify no regressions in AI advice quality

---

## Notes

- All file paths are relative to repository root
- Priority rankings are based on impact/effort ratio
- Some optimizations may require updating multiple files
- Consider feature flags for risky changes during development
