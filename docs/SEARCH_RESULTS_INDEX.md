# Codebase Search Results - Complete Index

## Overview
This document indexes all findings from the comprehensive codebase search covering GUI structure, logging, card database lookups, and event processing.

Two detailed files have been generated:
1. **CODEBASE_GUI_ANALYSIS.md** - Complete 10-section analysis with code examples
2. **QUICK_REFERENCE.txt** - Quick lookup table for common patterns

---

## 1. GUI Pane Structure - File Locations & Line Numbers

### Tkinter GUI Implementation
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| AdvisorGUI Class | `ui.py` | 687-1786 | Main GUI class |
| _create_widgets() | `ui.py` | 739-1088 | All widget creation |
| Settings Frame | `ui.py` | 759-927 | Left sidebar panel |
| Content Frame | `ui.py` | 966-1082 | Main content area |
| Board State Widget | `ui.py` | 992-1003 | ScrolledText for board |
| Messages Widget | `ui.py` | 1024-1035 | ScrolledText for messages |
| RAG References Widget | `ui.py` | 1070-1081 | ScrolledText for RAG |
| set_board_state() | `ui.py` | 1735-1753 | Update board display |
| add_message() | `ui.py` | 1755-1785 | Add timestamped message |

### Curses TUI Implementation
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| AdvisorTUI Class | `ui.py` | 239-679 | Main TUI class |
| _create_windows() | `ui.py` | 297-321 | Curses window layout |
| Status Window | `ui.py` | 302 | 1-line status bar |
| Board Window | `ui.py` | 304-308 | 70% of height |
| Message Window | `ui.py` | 311-314 | 30% of height |
| Input Window | `ui.py` | 317 | 1-line input |
| _refresh_board() | `ui.py` | 368-393 | Draw board pane |
| _refresh_messages() | `ui.py` | 395-421 | Draw messages pane |
| add_message() | `ui.py` | 357-366 | Add message with color |
| set_board_state() | `ui.py` | 352-355 | Update board |

---

## 2. Log Display & Tapping

### Log File Following System
| Component | File | Lines | Key Feature |
|-----------|------|-------|-------------|
| LogFollower Class | `mtga.py` | 12-92 | Main log follower |
| __init__() | `mtga.py` | 14-19 | Initialize with path |
| follow() | `mtga.py` | 21-87 | Main loop, callback pattern |
| Inode Tracking | `mtga.py` | 29-46 | Detects log rotation |
| File Seeking | `mtga.py` | 50-60 | Smart positioning |
| Line Reading | `mtga.py` | 62-79 | Non-blocking read loop |

### Event Parsing System
| Component | File | Lines | Detects |
|-----------|------|-------|---------|
| parse_log_line() | `mtga.py` | 817-932 | All event types |
| Draft.Notify Events | `mtga.py` | 859-891 | Premier Draft picks |
| Start Events | `mtga.py` | 895-916 | LogBusinessEvents |
| End Event Markers | `mtga.py` | 920-931 | Next line JSON |
| Callback System | `mtga.py` | 843-846 | Registered handlers |
| JSON Parsing | `mtga.py` | 826-840 | Payload extraction |

### Existing Display Methods
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| _display_board_state() | `app.py` | 1526-1605 | Formats board text |
| _format_card_display() | `app.py` | 1487-1524 | Formats individual cards |
| Display Updates | `app.py` | 1618-1620 | Calls set_board_state() |

---

## 3. grpId to Card Name Resolution

### Card Data Structures
| Class | File | Lines | Purpose |
|-------|------|-------|---------|
| CardInfo | `card_rag.py` | 27-45 | Complete card data |
| GameObject | `mtga.py` | 94-108 | In-game card instance |

### Card Lookup Methods
| Method | File | Lines | Input | Returns |
|--------|------|-------|-------|---------|
| get_card_by_grpid() | `card_rag.py` | 180-232 | grpId (int) | CardInfo |
| get_card_by_name() | `card_rag.py` | 234-275 | name (str), set_code (optional) | CardInfo |
| search_cards_by_type() | `card_rag.py` | 372-475 | type_keyword (str), set_code (optional) | List[CardInfo] |

### Database Connection
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| CardRAGSystem.__init__() | `card_rag.py` | 143-157 | Initialize DB connections |
| _initialize() | `card_rag.py` | 160-178 | Open SQLite connections |
| unified_cards.db | `card_rag.py` | 164-166 | Arena card database |
| card_stats.db | `card_rag.py` | 170-173 | 17lands statistics |

---

## 4. Card Database Lookup Methods

### SQL Tables
**unified_cards.db - cards table**
```
Columns: grpId (PK), name, oracle_text, mana_cost, cmc, type_line,
         color_identity, power, toughness, rarity, set_code
```

**card_stats.db - card_statistics table**
```
Columns: card_name, format_type, win_rate, gih_win_rate,
         avg_pick_position, games_played
```

### Lookup Patterns
| Pattern | Method | File | Lines |
|---------|--------|------|-------|
| Primary (grpId) | get_card_by_grpid(grp_id) | card_rag.py | 180-232 |
| By Name | get_card_by_name(name, set_code=None) | card_rag.py | 234-275 |
| By Type | search_cards_by_type(type_keyword, set_code=None) | card_rag.py | 372-475 |
| Statistics | _add_statistics(card_info, format_type) | card_rag.py | 297-342 |

---

## 5. Game Event Logging & Line Processing

### Event Flow
```
Arena Player.log file
    ↓
LogFollower.follow(callback)
    ↓
GameStateManager.parse_log_line(line)
    ↓
Registered callback for event type
    ↓
Update GameState / BoardState
    ↓
GUI/TUI refresh via set_board_state() / add_message()
```

### Event Detection Patterns
| Pattern | File | Lines | Detects |
|---------|------|-------|---------|
| Draft.Notify | mtga.py | 859-891 | Premier Draft pick |
| Start Events | mtga.py | 895-916 | LogBusinessEvents |
| End Markers | mtga.py | 920-931 | JSON on next line |
| Zone Changes | (GRE parsing) | - | Card moves zones |
| Damage Events | (GRE parsing) | 608-625 | Damage dealt |
| Combat Events | (GRE parsing) | 627-648 | Attack/block |

---

## 6. Existing Log Display Functionality

### Board State Display
| Method | File | Lines | Functionality |
|--------|------|-------|---|
| _display_board_state() | app.py | 1526-1605 | Builds board text lines |
| _format_card_display() | app.py | 1487-1524 | Formats individual cards |
| Updates | app.py | 1618-1620 | Sends to GUI/TUI |

### Message Display
| Method | File | Lines | Type | Notes |
|--------|------|-------|------|-------|
| add_message() TUI | ui.py | 357-366 | Timestamp + text | Auto-scroll |
| add_message() GUI | ui.py | 1755-1785 | Timestamp + text + tag | Color support |
| set_board_state() TUI | ui.py | 352-355 | Replace lines | Responsive |
| set_board_state() GUI | ui.py | 1735-1753 | Replace text | Auto-scroll |

---

## 7. Board State Pane Pattern

### Pattern Components
1. **Data Collection**: Build list of formatted text lines
2. **Display Update**: Call set_board_state(lines) or add_message(msg, color)
3. **Rendering**: TUI uses curses, GUI uses ScrolledText
4. **Interaction**: TUI has keyboard navigation, GUI has mouse support
5. **Auto-scroll**: Both implementations scroll to bottom on update

### Extensibility Points for New Pane
**Tkinter Addition**:
```python
self.game_log_text = scrolledtext.ScrolledText(content_frame, ...)
self.game_log_text.tag_config('event', foreground='#ffff88')
```

**Curses Addition**:
```python
self.log_win = curses.newwin(height, width, y, x)
self.log_win.scrollok(True)
```

---

## 8. Color Coding & Text Formatting

### Tkinter Color Scheme
| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark | #2b2b2b |
| Panels | Darker | #1a1a1a |
| Text | White | #ffffff |
| Success | Green | #00ff88 |
| Info | Cyan | #55aaff |
| Warning | Yellow | #ffff00 |
| Error | Red | #ff5555 |
| White Mana | Pale Yellow | #ffff99 |
| Black Mana | Gray | #aaaaaa |
| Multicolor | Orange-Yellow | #ffdd44 |
| Colorless | Light Gray | #cccccc |

### Status Indicators
| Indicator | Meaning |
|-----------|---------|
| 🔄 | Tapped |
| ❤️ | Life total |
| 🃏 | Cards |
| ⚔️ | Battlefield |
| ⚡ | Attacking/Action |
| 😴 | Summoning sickness |
| 💀 | Death |
| 🎴 | Mulligan/Draft |
| 🌍 | Land |
| 📜 | History |
| 🔮 | Scry/Reveal |
| ⚰️ | Graveyard |
| 🚫 | Exile |

---

## 9. Implementation Patterns & Best Practices

### Thread-Safe Updates
- **Pattern**: LogFollower runs on thread, processes callbacks synchronously
- **File**: `mtga.py` lines 21-87
- **Notes**: GUI updates happen from main thread only

### Database Lookups
- **Pattern**: grpId → unified_cards.db → CardInfo
- **File**: `card_rag.py` lines 180-232
- **Performance**: Direct SQL query with optional stats lookup

### Pane Refresh Pattern
```python
# Update data
data_lines = format_display(game_state)
# Update UIs
if self.gui:
    self.gui.set_board_state(data_lines)
if self.tui:
    self.tui.set_board_state(data_lines)
```

---

## 10. Key Files Summary

| File | Size | Purpose | Key Classes |
|------|------|---------|-------------|
| ui.py | 1786 lines | All UI (TUI & GUI) | AdvisorTUI, AdvisorGUI |
| mtga.py | 950+ lines | Log parsing & game state | LogFollower, GameStateManager |
| card_rag.py | 500+ lines | Card database & RAG | CardRAGSystem, CardInfo |
| app.py | 1650+ lines | Main application | CLIVoiceAdvisor |
| CODEBASE_GUI_ANALYSIS.md | Detailed | Full analysis with code | Reference document |
| QUICK_REFERENCE.txt | Reference | Quick lookups | Pattern examples |

---

## Quick Links to Common Tasks

**Add a message to both UIs**:
- File: `app.py` - look for `self.gui.add_message()` or `self.tui.add_message()`
- Example: `/home/joshu/logparser/app.py:264-266`

**Resolve grpId to card name**:
- File: `card_rag.py` - method `get_card_by_grpid()`
- Example: `/home/joshu/logparser/card_rag.py:180-232`

**Handle game events**:
- File: `mtga.py` - method `parse_log_line()`
- Example: `/home/joshu/logparser/mtga.py:817-932`

**Display board state**:
- File: `app.py` - method `_display_board_state()`
- Example: `/home/joshu/logparser/app.py:1526-1605`

**Create GUI pane**:
- File: `ui.py` - method `AdvisorGUI._create_widgets()`
- Example: `/home/joshu/logparser/ui.py:739-1088`

---

## Files Generated

1. **CODEBASE_GUI_ANALYSIS.md**
   - 10 comprehensive sections
   - Full code examples
   - Line-by-line references
   - Pattern documentation

2. **QUICK_REFERENCE.txt**
   - Quick lookup tables
   - Color scheme reference
   - Common patterns
   - Threading notes

3. **SEARCH_RESULTS_INDEX.md** (this file)
   - Consolidated index
   - File/line cross-reference
   - Quick navigation

