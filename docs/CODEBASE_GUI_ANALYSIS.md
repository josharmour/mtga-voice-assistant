# MTGA Voice Advisor - Codebase GUI & Logging Analysis

## Executive Summary
This document provides a detailed analysis of the current GUI structure, logging mechanisms, and card database lookups in the MTGA Voice Advisor codebase. It identifies existing patterns that can be leveraged for implementing a live game log display feature.

---

## 1. GUI PANE STRUCTURE & CREATION

### 1.1 GUI Frameworks Used
The codebase supports **two UI modes**:
- **TUI (Terminal UI)**: Uses `curses` library (AdvisorTUI)
- **GUI (Graphical UI)**: Uses `tkinter` (AdvisorGUI)

### 1.2 AdvisorTUI - Curses-Based Terminal UI
**File**: `/home/joshu/logparser/ui.py` (lines 239-679)

#### Pane Structure (Curses)
```python
class AdvisorTUI:
    def _create_windows(self):
        """Create and layout windows"""
        height, width = self.stdscr.getmaxyx()

        # Status bar: 1 line at top
        self.status_win = curses.newwin(1, width, 0, 0)

        # Board state: 70% of available height
        available_height = height - 3
        board_height = max(10, int(available_height * 0.7))
        self.board_win = curses.newwin(board_height, width, 1, 0)
        self.board_win.scrollok(True)

        # Messages: remaining 30% of space
        msg_y = 1 + board_height
        msg_height = height - msg_y - 2
        self.msg_win = curses.newwin(msg_height, width, msg_y, 0)
        self.msg_win.scrollok(True)

        # Input prompt: 1 line at bottom
        self.input_win = curses.newwin(1, width, height - 1, 0)
```

**Key Points**:
- Uses `curses.newwin(height, width, y, x)` for pane creation
- All panes support `scrollok(True)` for scrolling
- Layout is responsive to terminal resize via `curses.update_lines_cols()`
- Refresh methods: `_refresh_board()`, `_refresh_messages()`, `_refresh_input()`

#### Color System (Curses)
```python
def __init__(self, stdscr):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # Green
    curses.init_pair(2, curses.COLOR_CYAN, -1)    # Cyan
    curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Yellow
    curses.init_pair(4, curses.COLOR_RED, -1)     # Red
    curses.init_pair(5, curses.COLOR_BLUE, -1)    # Blue
    curses.init_pair(6, curses.COLOR_WHITE, -1)   # White

    self.color_map = {
        "green": 1, "cyan": 2, "yellow": 3,
        "red": 4, "blue": 5, "white": 6,
    }
```

**Line**: `/home/joshu/logparser/ui.py:270-288`

---

### 1.3 AdvisorGUI - Tkinter-Based GUI
**File**: `/home/joshu/logparser/ui.py` (lines 687-1786)

#### Pane Structure (Tkinter)
```python
class AdvisorGUI:
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # Settings panel (left side)
        settings_frame = tk.Frame(self.root, bg=self.bg_color, width=250)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        settings_frame.pack_propagate(False)
        
        # Main content area (right side)
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Board state area (top)
        self.board_text = scrolledtext.ScrolledText(
            content_frame,
            height=15,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.board_text.pack(fill=tk.BOTH, expand=True)
        self.board_text.config(state=tk.DISABLED)
        
        # Advisor messages area (middle)
        self.messages_text = scrolledtext.ScrolledText(
            content_frame,
            height=15,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        
        # RAG References panel (bottom)
        self.rag_text = scrolledtext.ScrolledText(
            content_frame,
            height=6,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.rag_text.pack(fill=tk.BOTH, expand=False)
```

**Lines**: `/home/joshu/logparser/ui.py:739-1088`

**Key Features**:
- Uses `scrolledtext.ScrolledText` for auto-scrolling text display
- All text panes start with `state=tk.DISABLED` for read-only
- Layout uses `pack()` with side/fill/expand parameters
- Panes are collapsible (RAG panel has expand/collapse toggle)

#### Color System (Tkinter)
```python
# Board text color tags (by card color)
self.board_text.tag_config('color_w', foreground='#ffff99')   # White
self.board_text.tag_config('color_u', foreground='#55aaff')   # Blue
self.board_text.tag_config('color_b', foreground='#aaaaaa')   # Black
self.board_text.tag_config('color_r', foreground='#ff5555')   # Red
self.board_text.tag_config('color_g', foreground='#00ff88')   # Green
self.board_text.tag_config('color_c', foreground='#cccccc')   # Colorless
self.board_text.tag_config('color_multi', foreground='#ffdd44') # Multicolor

# Message text color tags
self.messages_text.tag_config('green', foreground='#00ff88')
self.messages_text.tag_config('blue', foreground='#55aaff')
self.messages_text.tag_config('cyan', foreground='#00ffff')
self.messages_text.tag_config('yellow', foreground='#ffff00')
self.messages_text.tag_config('red', foreground='#ff5555')
self.messages_text.tag_config('white', foreground='#ffffff')

# RAG panel tags
self.rag_text.tag_config('rule', foreground='#ffff00', font=('Consolas', 8, 'bold'))
self.rag_text.tag_config('card', foreground='#00ff88', font=('Consolas', 8, 'bold'))
self.rag_text.tag_config('query', foreground='#55aaff', font=('Consolas', 8, 'italic'))
self.rag_text.tag_config('stats', foreground='#ff88ff')
```

**Lines**: `/home/joshu/logparser/ui.py:1005-1087`

---

## 2. HOW LOGS ARE CURRENTLY DISPLAYED & TAPPABLE

### 2.1 Log Following System
**File**: `/home/joshu/logparser/mtga.py` (lines 12-92)

```python
class LogFollower:
    """Follows the Arena Player.log file and yields new lines as they're added."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self.inode = None
        self.offset = 0
        self.first_open = True  # Track if first time opening

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
        print(f"[DEBUG] LogFollower.follow() started! Watching: {self.log_path}")
        logging.info(f"LogFollower.follow() started! Watching: {self.log_path}")
        while True:
            try:
                current_inode = None
                try:
                    current_inode = os.stat(self.log_path).st_ino
                except FileNotFoundError:
                    logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                    time.sleep(5)
                    continue

                if self.inode is None or self.inode != current_inode:
                    if self.file:
                        self.file.close()
                    self.file = open(self.log_path, 'r', encoding='utf-8', errors='replace')
                    self.inode = current_inode

                    # On first open, seek to end to ignore old matches
                    # On log rotation, start from beginning of new file
                    if self.first_open:
                        self.file.seek(0, 2)  # Seek to end of file
                        self.offset = self.file.tell()
                        self.first_open = False
                        logging.info("Log file opened - starting from end (ignoring old matches).")
                    else:
                        # Log rotation - start from beginning of new file
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning of new file.")

                self.file.seek(self.offset)
                line_count = 0
                while True:
                    line = self.file.readline()
                    if not line:
                        break
                    line_count += 1
                    self.offset = self.file.tell()
                    stripped_line = line.strip()
                    
                    # Debug: show what we're reading
                    if "Draft" in stripped_line or "BotDraft" in stripped_line:
                        print(f"[DEBUG] Draft-related line: {stripped_line[:150]}")
                    
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Read line: {stripped_line[:100]}...")
                    
                    callback(stripped_line)  # Pass line to callback
```

**Key Mechanisms**:
1. **Inode tracking**: Detects log file rotation
2. **Non-blocking**: Uses `time.sleep(0.05)` between reads
3. **Callback pattern**: Each line passed to callback function
4. **Smart seeking**: First open goes to end, rotations start from beginning

### 2.2 Log Line Processing & Event Parsing
**File**: `/home/joshu/logparser/mtga.py` (lines 817-932)

```python
def parse_log_line(self, line: str) -> bool:
    """Parse individual log lines and trigger callbacks"""
    logging.debug(f"Full log line received by GameStateManager: {line}")

    # DRAFT EVENTS: Check if this is the JSON line after an end event marker
    if self._next_line_event:
        event_type = self._next_line_event
        self._next_line_event = None
        
        try:
            json_start = line.find("{")
            if json_start != -1:
                parsed_data = json.loads(line[json_start:])
                logging.info(f"Parsed draft event: {event_type}")
                
                # Check if data is wrapped in a Payload field
                if "Payload" in parsed_data and isinstance(parsed_data["Payload"], str):
                    try:
                        inner_data = json.loads(parsed_data["Payload"])
                        logging.debug(f"Unpacked Payload for {event_type}")
                        parsed_data = inner_data
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse Payload JSON: {e}")
                
                # Call the registered callback if it exists
                if event_type in self._draft_callbacks:
                    print(f"[DEBUG] Calling callback for {event_type}")
                    self._draft_callbacks[event_type](parsed_data)
```

**Event Types Detected**:
- `Draft.Notify`: Premier Draft picks
- `LogBusinessEvents`: Draft pick tracking
- `<== EventName(uuid)`: End event markers
- `[UnityCrossThreadLogger]==> EventName {...}`: Start events

---

## 3. WHERE grpId/ARENA IDs ARE RESOLVED TO CARD NAMES

### 3.1 Card Information Data Class
**File**: `/home/joshu/logparser/card_rag.py` (lines 27-45)

```python
@dataclass
class CardInfo:
    """Complete card information for LLM context."""
    grpId: int              # Arena graphics ID (primary lookup key)
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
```

### 3.2 grpId to Card Name Resolution
**File**: `/home/joshu/logparser/card_rag.py` (lines 180-232)

```python
def get_card_by_grpid(self, grp_id: int, format_type: str = "PremierDraft") -> Optional[CardInfo]:
    """
    Get complete card information by grpId with statistics.
    
    Args:
        grp_id: Arena graphics ID
        format_type: Draft format for statistics (e.g., "PremierDraft")
    
    Returns:
        CardInfo object with all data or None
    """
    if not self.unified_conn:
        logger.warning("Unified database not available")
        return None
    
    try:
        cursor = self.unified_conn.cursor()
        cursor.execute("""
            SELECT
                grpId, name, oracle_text, mana_cost, cmc, type_line,
                color_identity, power, toughness, rarity, set_code
            FROM cards
            WHERE grpId = ?
        """, (grp_id,))
        
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Card not found: grpId {grp_id}")
            return None
        
        card_info = CardInfo(
            grpId=row['grpId'],
            name=row['name'],
            oracle_text=row['oracle_text'] or "",
            mana_cost=row['mana_cost'] or "",
            cmc=row['cmc'] or 0.0,
            type_line=row['type_line'] or "",
            color_identity=row['color_identity'] or "",
            power=row['power'] or "",
            toughness=row['toughness'] or "",
            rarity=row['rarity'] or "",
            set_code=row['set_code'] or ""
        )
        
        # Get statistics if available
        if self.stats_conn:
            card_info = self._add_statistics(card_info, format_type)
        
        return card_info
```

**Key Points**:
- Primary lookup key: `grpId` (Arena graphics ID)
- Database: `unified_cards.db` (contains full card data)
- Also retrieves statistics from `card_stats.db` (17lands data)
- Handles missing cards gracefully with warning log

### 3.3 Card Name to grpId Reverse Lookup
**File**: `/home/joshu/logparser/card_rag.py` (lines 234-275)

```python
def get_card_by_name(self, card_name: str, set_code: str = None) -> Optional[CardInfo]:
    """
    Get card information by name and optional set.
    
    Args:
        card_name: Card name to search for
        set_code: Optional set code to narrow search
    
    Returns:
        CardInfo object or None
    """
    if not self.unified_conn:
        return None
    
    try:
        cursor = self.unified_conn.cursor()
        
        if set_code:
            # Search by name and set
            cursor.execute("""
                SELECT
                    grpId, name, oracle_text, mana_cost, cmc, type_line,
                    color_identity, power, toughness, rarity, set_code
                FROM cards
                WHERE name = ? AND set_code = ?
            """, (card_name, set_code))
        else:
            # Search by name only (any set)
            cursor.execute("""
                SELECT
                    grpId, name, oracle_text, mana_cost, cmc, type_line,
                    color_identity, power, toughness, rarity, set_code
                FROM cards
                WHERE name = ?
            """, (card_name,))
        
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Card not found: {card_name}")
            return None
        
        card_info = CardInfo(
            grpId=row['grpId'],
            name=row['name'],
            # ... rest of CardInfo initialization
        )
```

### 3.4 GameObject Data Class (In-Game Card Reference)
**File**: `/home/joshu/logparser/mtga.py` (lines 95-108)

```python
@dataclasses.dataclass
class GameObject:
    instance_id: int          # Unique instance on board
    grp_id: int              # Arena graphics ID (used for lookup)
    zone_id: int
    owner_seat_id: int
    name: str = ""           # Resolved from grp_id
    power: Optional[int] = None
    toughness: Optional[int] = None
    is_tapped: bool = False
    is_attacking: bool = False
    summoning_sick: bool = False
    counters: Dict[str, int] = dataclasses.field(default_factory=dict)
    attached_to: Optional[int] = None
    visibility: str = "public"
```

---

## 4. CARD DATABASE LOOKUP METHODS

### 4.1 Unified Card Database Structure
**File**: `/home/joshu/logparser/card_rag.py` (lines 142-179)

```python
class CardRAGSystem:
    """
    Manages card lookup from unified database and statistics.
    
    Attributes:
        unified_db: Path to unified_cards.db (Arena data)
        stats_db: Path to card_stats.db (17lands data)
    """
    
    def __init__(self, unified_db: str = "data/unified_cards.db", 
                 stats_db: str = "data/card_stats.db"):
        self.unified_db_path = Path(unified_db)
        self.stats_db_path = Path(stats_db)
        self.unified_conn = None
        self.stats_conn = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize database connections."""
        try:
            if self.unified_db_path.exists():
                self.unified_conn = sqlite3.connect(str(self.unified_db_path))
                self.unified_conn.row_factory = sqlite3.Row
                logger.info(f"Connected to {self.unified_db_path}")
            
            if self.stats_db_path.exists():
                self.stats_conn = sqlite3.connect(str(self.stats_db_path))
                self.stats_conn.row_factory = sqlite3.Row
                logger.info(f"Connected to {self.stats_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
```

### 4.2 Lookup Methods
1. **By grpId (Primary)**
   - Method: `get_card_by_grpid(grp_id: int, format_type: str)`
   - Returns: Complete CardInfo with statistics
   
2. **By Name (Secondary)**
   - Method: `get_card_by_name(card_name: str, set_code: str = None)`
   - Returns: CardInfo with optional set filtering
   
3. **By Type/Keyword**
   - Method: `search_cards_by_type(type_keyword: str, set_code: str = None)`
   - Returns: List[CardInfo] matching type filter

### 4.3 Statistics Addition
**File**: `/home/joshu/logparser/card_rag.py` (lines 297-342)

```python
def _add_statistics(self, card_info: CardInfo, format_type: str = "PremierDraft") -> CardInfo:
    """Add statistics from 17lands database to card info"""
    if not self.stats_conn:
        return card_info
    
    try:
        cursor = self.stats_conn.cursor()
        cursor.execute("""
            SELECT win_rate, gih_win_rate, avg_pick_position, games_played
            FROM card_statistics
            WHERE card_name = ? AND format_type = ?
        """, (card_info.name, format_type))
        
        row = cursor.fetchone()
        if row:
            card_info.win_rate = row['win_rate']
            card_info.gih_win_rate = row['gih_win_rate']
            card_info.avg_pick_position = row['avg_pick_position']
            card_info.games_played = row['games_played']
    except Exception as e:
        logger.warning(f"Failed to fetch statistics for {card_info.name}: {e}")
    
    return card_info
```

---

## 5. GAME EVENT LOGGING & LINE PROCESSING

### 5.1 Event Flow
```
Player.log file
    ↓
LogFollower.follow() [non-blocking callback pattern]
    ↓
App.main_loop() calls game_state_mgr.parse_log_line()
    ↓
GameStateManager.parse_log_line() detects event type
    ↓
Calls registered callback for event type
    ↓
Updates GameState / BoardState
    ↓
UI updated via set_board_state() / add_message()
```

### 5.2 Event Detection Patterns
**File**: `/home/joshu/logparser/mtga.py` (lines 857-932)

```python
# DRAFT EVENTS: Draft.Notify messages
if "[UnityCrossThreadLogger]Draft.Notify" in line:
    match = re.search(r'\[UnityCrossThreadLogger\]Draft\.Notify (.+)', line)
    if match:
        json_str = match.group(1)
        draft_data = json.loads(json_str)
        pack_num = draft_data.get("SelfPack", 1)
        pick_num = draft_data.get("SelfPick", 1)
        pack_arena_ids = [int(card_id) for card_id in draft_data.get("PackCards", "").split(",")]
        logging.info(f"Draft.Notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

# DRAFT EVENTS: Premier Draft picks
if line.startswith("[UnityCrossThreadLogger]==>"):
    match = re.search(r'\[UnityCrossThreadLogger\]==> (\w+) (.*)', line)
    if match:
        event_type = match.group(1)
        outer_json_str = match.group(2)
        outer_json = json.loads(outer_json_str)
        if "request" in outer_json:
            inner_json = json.loads(outer_json["request"])
            logging.info(f"Detected start event: {event_type}")

# DRAFT EVENTS: End event markers (JSON on next line)
if line.startswith("<=="):
    match = re.search(r'<== (\w+)\(([a-f0-9-]+)\)', line)
    if match:
        event_type = match.group(1)
        self._next_line_event = event_type
        logging.debug(f"Detected end event marker: {event_type}")
```

---

## 6. EXISTING LOG DISPLAY & MONITORING FUNCTIONALITY

### 6.1 Board State Display Implementation
**File**: `/home/joshu/logparser/app.py` (lines 1487-1605)

```python
def _format_card_display(self, card: "GameObject") -> str:
    """Format card display with name, type, P/T, and status indicators"""
    if "Unknown" in card.name:
        return f"{card.name} ⚠️"
    
    status_parts = []
    
    # Add tapped status
    if card.is_tapped:
        status_parts.append("🔄")
    
    # Add power/toughness for creatures
    if card.power is not None and card.toughness is not None:
        power_val = card.power.get("value") if isinstance(card.power, dict) else card.power
        tough_val = card.toughness.get("value") if isinstance(card.toughness, dict) else card.toughness
        status_parts.append(f"{power_val}/{tough_val}")
    
    # Add summoning sickness indicator
    if card.summoning_sick:
        status_parts.append("😴")
    
    # Add attacking status
    if card.is_attacking:
        status_parts.append("⚡")
    
    # Add any counters
    if card.counters:
        counter_str = ", ".join([f"{count}x {ctype}" for ctype, count in card.counters.items()])
        status_parts.append(f"[{counter_str}]")
    
    if status_parts:
        return f"{card.name} ({', '.join(status_parts)})"
    else:
        return card.name

def _display_board_state(self, board_state: "BoardState"):
    """Display comprehensive visual representation of current board state"""
    lines = []
    lines.append("")
    lines.append("="*70)
    if board_state.in_mulligan_phase:
        lines.append("🎴 MULLIGAN PHASE - Opening Hand")
    else:
        lines.append(f"TURN {board_state.current_turn} - {board_state.current_phase}")
    lines.append("="*70)
    
    # Game History - what happened this turn
    if board_state.history and board_state.history.turn_number == board_state.current_turn:
        history = board_state.history
        if history.cards_played_this_turn or history.died_this_turn:
            lines.append("")
            lines.append("📜 THIS TURN:")
            if history.cards_played_this_turn:
                played_names = [c.name for c in history.cards_played_this_turn]
                lines.append(f"   ⚡ Played: {', '.join(played_names)}")
            if history.lands_played_this_turn > 0:
                lines.append(f"   🌍 Lands: {history.lands_played_this_turn}")
            if history.died_this_turn:
                lines.append(f"   💀 Died: {', '.join(history.died_this_turn)}")
    
    # Opponent info
    lines.append("")
    lines.append("─"*70)
    opponent_lib = board_state.opponent_library_count if board_state.opponent_library_count > 0 else "?"
    lines.append(f"OPPONENT: ❤️  {board_state.opponent_life} life | 🃏 {board_state.opponent_hand_count} cards | 📖 {opponent_lib} library")
    
    lines.append("")
    lines.append(f"  ⚔️  Battlefield ({len(board_state.opponent_battlefield)}):")
    if board_state.opponent_battlefield:
        for card in board_state.opponent_battlefield:
            card_info = self._format_card_display(card)
            lines.append(f"      • {card_info}")
    else:
        lines.append("      (empty)")
    
    # ... similar for your side ...
    
    # Update GUI
    if self.gui:
        self.gui.set_board_state(lines)
    if self.tui:
        self.tui.set_board_state(lines)
```

### 6.2 Message Display System
**File**: `/home/joshu/logparser/ui.py` (lines 357-366 for TUI, 1755-1785 for GUI)

#### TUI Add Message
```python
def add_message(self, msg: str, color = 0):
    """Add message to message log. Color can be string name or int pair ID."""
    timestamp = time.strftime("%H:%M:%S")
    # Convert color string to int if needed
    if isinstance(color, str):
        color = self.color_map.get(color, 6)  # Default to white
    self.messages.append((timestamp, msg, color))
    # Auto-scroll to bottom
    self.msg_scroll = max(0, len(self.messages) - self.msg_win.getmaxyx()[0])
    self._refresh_messages()
```

#### GUI Add Message
```python
def add_message(self, msg: str, color=None):
    """Add message to the advisor messages display."""
    try:
        if not hasattr(self, 'messages_text'):
            return
        
        import time
        timestamp = time.strftime("%H:%M:%S")
        
        self.messages_text.config(state=tk.NORMAL)
        
        # Add timestamp
        self.messages_text.insert(tk.END, f"[{timestamp}] ")
        
        # Add message with color tag if specified
        if color and isinstance(color, str):
            color_tag = color.lower()
            if color_tag in ['green', 'blue', 'cyan', 'yellow', 'red', 'white']:
                self.messages_text.insert(tk.END, msg + '\n', color_tag)
            else:
                self.messages_text.insert(tk.END, msg + '\n')
        else:
            self.messages_text.insert(tk.END, msg + '\n')
        
        self.messages_text.config(state=tk.DISABLED)
        # Auto-scroll to bottom
        self.messages_text.see(tk.END)
```

### 6.3 Board State Update
**File**: `/home/joshu/logparser/ui.py` (lines 352-355 for TUI, 1735-1753 for GUI)

#### TUI Set Board State
```python
def set_board_state(self, lines: List[str]):
    """Update board state display"""
    self.board_state_lines = lines
    self._refresh_board()
```

#### GUI Set Board State
```python
def set_board_state(self, lines: list):
    """Update board state display."""
    try:
        if not hasattr(self, 'board_text'):
            return
        
        self.board_text.config(state=tk.NORMAL)
        self.board_text.delete(1.0, tk.END)
        
        # Add each line to the board display
        for line in lines:
            self.board_text.insert(tk.END, line + '\n')
        
        self.board_text.config(state=tk.DISABLED)
        # Auto-scroll to bottom
        self.board_text.see(tk.END)
```

---

## 7. BOARD STATE PANE - IMPLEMENTATION PATTERN

### 7.1 Pattern Summary
The board state pane follows this pattern:

1. **Data Collection**: `_display_board_state()` builds list of formatted lines
2. **Display Update**: Calls `set_board_state(lines)` with list of strings
3. **Rendering**:
   - TUI: Stores in `self.board_state_lines`, refreshes via `_refresh_board()`
   - GUI: Clears widget, inserts all lines with optional color tags
4. **Interaction**:
   - TUI: UP/DOWN arrows scroll, Page Up/Down for messages pane
   - GUI: Mouse wheel/scrollbar built-in to ScrolledText widget
5. **Auto-scroll**: Both TUI and GUI auto-scroll to bottom on update

### 7.2 Extensibility Points
1. **New Pane Addition (Tkinter)**:
```python
self.game_log_text = scrolledtext.ScrolledText(
    content_frame,
    height=10,
    bg='#1a1a1a',
    fg=self.fg_color,
    font=('Consolas', 8),
    relief=tk.FLAT,
    padx=10,
    pady=10
)
self.game_log_text.pack(fill=tk.BOTH, expand=True)
self.game_log_text.config(state=tk.DISABLED)

# Add color tags
self.game_log_text.tag_config('event', foreground='#ffff88')
self.game_log_text.tag_config('action', foreground='#88ff88')
self.game_log_text.tag_config('warning', foreground='#ff8888')
```

2. **New Pane Addition (Curses)**:
```python
# In _create_windows():
log_height = max(5, int(available_height * 0.2))
self.log_win = curses.newwin(log_height, width, msg_y + msg_height, 0)
self.log_win.scrollok(True)
```

---

## 8. COLOR CODING & TEXT FORMATTING

### 8.1 Tkinter Color Scheme
**Theme**: Dark theme with neon accents

```python
# Predefined colors
self.bg_color = '#2b2b2b'          # Dark background
self.fg_color = '#ffffff'          # White text
self.accent_color = '#00ff88'      # Neon green
self.warning_color = '#ff5555'     # Neon red
self.info_color = '#55aaff'        # Neon blue

# Card color mapping
'color_w': '#ffff99'               # White → pale yellow
'color_u': '#55aaff'               # Blue → cyan
'color_b': '#aaaaaa'               # Black → gray
'color_r': '#ff5555'               # Red → bright red
'color_g': '#00ff88'               # Green → bright green
'color_c': '#cccccc'               # Colorless → light gray
'color_multi': '#ffdd44'           # Multicolor → orange-yellow

# Message types
'green': '#00ff88'                 # Success/positive
'cyan': '#00ffff'                  # Info
'yellow': '#ffff00'                # Warning
'red': '#ff5555'                   # Error
'blue': '#55aaff'                  # Debug/technical
'white': '#ffffff'                 # Default
```

### 8.2 Curses Color Scheme
```python
curses.init_pair(1, curses.COLOR_GREEN, -1)      # Green
curses.init_pair(2, curses.COLOR_CYAN, -1)       # Cyan (headers)
curses.init_pair(3, curses.COLOR_YELLOW, -1)     # Yellow
curses.init_pair(4, curses.COLOR_RED, -1)        # Red
curses.init_pair(5, curses.COLOR_BLUE, -1)       # Blue
curses.init_pair(6, curses.COLOR_WHITE, -1)      # White
```

### 8.3 Status Indicators & Emojis
Used throughout display:
- `🔄` Tapped card
- `❤️` Life total
- `🃏` Cards in hand/library
- `📖` Library size
- `⚔️` Battlefield
- `⚰️` Graveyard
- `🚫` Exile
- `📋` Stack
- `⚡` Attacking/Action
- `😴` Summoning sickness
- `🎴` Mulligan phase
- `📜` Turn history
- `🌍` Land
- `💀` Death
- `🔮` Scry/Reveal

---

## 9. KEY IMPLEMENTATION INSIGHTS

### 9.1 Thread-Safe Updates (Non-Blocking Log Following)
- **Log Follower**: Runs on thread, calls callback for each line
- **Game State Manager**: Processes callbacks synchronously
- **GUI Updates**: Queued through main Tkinter loop
- **Pattern**: Callback → Event Processing → Message Queue → UI Update

### 9.2 Database Access Pattern
```
grpId from log event
    ↓
CardRAGSystem.get_card_by_grpid(grp_id)
    ↓
Query unified_cards.db
    ↓
Optionally add stats from card_stats.db
    ↓
Return CardInfo object with all data
    ↓
Can be formatted for display or LLM context
```

### 9.3 Pane Refresh Pattern
```python
# Update data model
data_lines = format_display(game_state)

# Update all UI representations
if self.gui:
    self.gui.set_board_state(data_lines)
if self.tui:
    self.tui.set_board_state(data_lines)

# Message display (separate from board)
self.gui.add_message(msg, color)
self.tui.add_message(msg, color)
```

### 9.4 Event Detection Hierarchy
1. **Line-level**: Regex pattern matching for event markers
2. **Payload-level**: JSON parsing for nested data
3. **Field-level**: Extraction of specific data (grpIds, positions, etc.)
4. **Callback-level**: Registered handler processes event
5. **State-level**: Updates game state and triggers UI refresh

---

## 10. RECOMMENDED PATTERNS FOR NEW LOG DISPLAY PANE

Based on the existing codebase patterns, a new game log display pane should:

1. **Use ScrolledText widget** (Tkinter) or `curses.newwin()` (Curses)
2. **Implement same message pattern**:
   ```python
   def add_log_entry(self, msg: str, event_type: str):
       # event_type: 'draw', 'play', 'attack', 'block', 'damage', 'cast', etc.
       timestamp = time.strftime("%H:%M:%S")
       tag = self.event_type_tags.get(event_type, 'default')
       self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n", tag)
       self.log_text.see(tk.END)
   ```

3. **Define event type color tags**:
   ```python
   self.log_text.tag_config('draw', foreground='#00ffff')
   self.log_text.tag_config('play', foreground='#ffff88')
   self.log_text.tag_config('attack', foreground='#ff5555')
   self.log_text.tag_config('block', foreground='#55aaff')
   self.log_text.tag_config('cast', foreground='#00ff88')
   self.log_text.tag_config('damage', foreground='#ff8888')
   ```

4. **Auto-scroll on insert**: Use `widget.see(tk.END)` or `widget.yview(tk.END)`

5. **Follow existing naming**: `game_log_win` (TUI), `game_log_text` (GUI)

6. **Integrate with event callbacks**: Hook into `GameStateManager.parse_log_line()`

---

