# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MTGA Voice Advisor** is a voice-enabled tactical advisor for Magic: The Gathering Arena that provides real-time game analysis. It monitors MTGA's game log, analyzes board state, queries a local Ollama LLM for tactical advice, and speaks recommendations using text-to-speech.

**Technology Stack:**
- Python 3.10+
- Ollama (local LLM)
- Optional RAG system (ChromaDB, sentence-transformers, torch)
- TTS engines: Kokoro ONNX (primary), pyttsx3 (fallback)
- Three UI modes: Tkinter GUI, Curses TUI, CLI

## Development Commands

```bash
# Install minimal dependencies (core functionality only)
pip install requests urllib3 pyttsx3

# Install full dependencies (with RAG system - ~1.5GB)
pip install -r requirements.txt

# Run application (default: GUI mode if Tkinter available)
python advisor.py

# Run with TUI (terminal interface)
python advisor.py --tui

# Run with CLI only (no curses)
python advisor.py --cli

# Test RAG system and initialize databases
python test_rag.py

# Download card metadata database (~30 seconds)
python download_card_metadata.py

# Download card statistics from 17lands (~60-180 minutes)
python download_real_17lands_data.py

# Update existing card data
python update_card_data.py --status
python update_card_data.py --auto

# Check available 17lands sets
python check_available_sets.py

# View logs in real-time
tail -f logs/advisor.log
```

**Prerequisites:**
- Ollama running locally on `http://localhost:11434`
- MTGA installed with **Detailed Logs (Plugin Support)** enabled in Options → Account
- Python 3.10+

## Architecture Overview

### Single-File Architecture with 10 Parts

The main application (`advisor.py`, ~3930 lines) is organized into 10 major parts:

1. **Arena Log Detection** (lines 55-123) - Cross-platform MTGA log path detection
2. **Real-Time Log Parsing** (`LogFollower`, lines 125-194) - Tails Player.log with inode tracking
3. **Game State Tracking** (`MatchScanner`, lines 196-759) - Parses GreToClientEvent JSON
4. **Card ID Resolution** (`ArenaCardDatabase`, lines 761-962) - grpId → card name mapping
5. **Board State Building** (`GameStateManager`, lines 964-1369) - Constructs AI-friendly state
6. **AI Advice Generation** (`OllamaClient`, `AIAdvisor`, lines 1371-1817) - LLM integration
7. **Text-to-Speech** (`TextToSpeech`, lines 1819-2029) - Multi-engine TTS with fallback
8. **TUI** (`AdvisorTUI`, lines 2031-2481) - Curses-based terminal interface
9. **GUI** (`AdvisorGUI`, lines 2483-2944) - Tkinter-based graphical interface
10. **Main CLI Loop** (`CLIVoiceAdvisor`, lines 2946-end) - Core orchestrator

### Three-Database RAG System

Optional enhancement system (`rag_advisor.py`) with three complementary databases:

1. **Card Statistics DB** (`data/card_stats.db`)
   - 17lands.com performance data (win rates, GIH, IWD)
   - ~6000-8000 cards per format
   - Downloaded via `download_real_17lands_data.py`

2. **Card Metadata DB** (`data/card_metadata.db`)
   - Card attributes: colors, mana cost, types, rarity
   - ~22,509 cards from 17lands API
   - Downloaded via `download_card_metadata.py`

3. **Rules Vector DB** (`data/chromadb/`)
   - MTG Comprehensive Rules with semantic search
   - ~3000+ rules with embeddings
   - Initialized via `test_rag.py`

### Data Flow

```
Player.log → LogFollower (inode tracking)
                ↓
          MatchScanner (JSON parsing)
                ↓
          GameStateManager (state building)
                ↓
          ArenaCardDatabase (grpId → name) ← card_cache.json (Scryfall)
                ↓
          BoardState representation
                ↓
          RAGSystem (optional) ← [card_stats.db, card_metadata.db, chromadb]
                ↓
          AIAdvisor (prompt building)
                ↓
          OllamaClient (LLM query)
                ↓
          TextToSpeech (voice output)
                ↓
          UI (GUI/TUI/CLI)
```

### Key Design Patterns

**Thread-Safe Advice Generation:**
- Separate worker thread for AI queries to avoid blocking log monitoring
- Thread-safe state updates via locks
- Non-blocking TUI input handling with `timeout(0)` mode

**Multi-Level Card Resolution:**
1. Arena card database (local binary file)
2. Scryfall cache (`card_cache.json`, ~2.8MB)
3. Scryfall API (fallback with rate limiting)

**File Rotation Detection:**
- Inode tracking in `LogFollower` detects log rotation
- Gracefully handles MTGA log file replacement

**Priority-Based Advice Triggers:**
- AI advice generated when `priorityPlayer == systemSeatId`
- Detects key decision points (beginning of turn, declare attackers, etc.)
- Debouncing to avoid advice spam

**Graceful Degradation:**
- RAG system optional (imports wrapped in try/except)
- GUI → TUI → CLI fallback chain
- Multiple TTS engines with automatic fallback

## Data Models

**Core Classes:**
- `GameObject` - Individual card with grpId, instanceId, zoneId, controller
- `PlayerState` - Life total, hand size, priority status, deck/graveyard counts
- `GameHistory` - Recent game events and turn tracking
- `BoardState` - Complete game state representation for AI

**RAG Classes:**
- `RulesParser` - Parses MagicCompRules.txt into chunks
- `CardStatsDB` - SQLite wrapper for 17lands statistics
- `CardMetadataDB` - SQLite wrapper for card attributes
- `RAGSystem` - Unified interface with ChromaDB vector store

## Configuration and Runtime Behavior

**Dynamic Configuration:**
- Ollama models discovered via API (`/api/tags`) at startup
- TTS voices discovered from available engines
- Slash commands for runtime switching: `/model`, `/voice`, `/volume`

**Critical Paths:**
- `card_cache.json` - Scryfall card data cache in project root
- `logs/advisor.log` - Application logs with DEBUG level
- `data/` - RAG databases (optional)

**Platform Detection:**
- **Windows:** `C:/Users/{user}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log`
- **macOS:** `~/Library/Logs/Wizards Of The Coast/MTGA/Player.log`
- **Linux:** Multiple paths checked (Steam, Bottles, Lutris)

## Slash Commands (In-App)

Available during runtime (CLI/TUI mode):
- `/help` - Show all commands
- `/models` - List available Ollama models
- `/model <name>` - Switch LLM model
- `/voices` - List available TTS voices
- `/voice <name>` - Switch TTS voice
- `/volume <0-100>` - Adjust TTS volume
- `/mute` - Toggle voice output
- `/status` - Show current configuration
- `/quit` or `/exit` - Exit application

## Testing and Initialization

**RAG System Initialization:**
```bash
# One-time setup (5 minutes)
python test_rag.py
```

Tests and initializes:
1. Rules parsing (MagicCompRules.txt → ChromaDB)
2. Card statistics database connectivity
3. Card metadata database connectivity
4. Semantic search functionality

**RAG System Updates:**
- Card metadata: Rarely (only when new card types added)
- Card statistics: Quarterly (new set releases)
- Rules database: Annually (comprehensive rules updates)

## Important Implementation Notes

**Priority Detection:**
- Key design decision: advice triggered when local player has priority
- Not a multiplayer analysis tool - single player perspective only
- `priorityPlayer` field in game state determines trigger timing

**Card Database Sources:**
1. Arena's local `Raw_CardDatabase_*.mtga` (preferred, most accurate)
2. Scryfall API (fallback with caching)
3. 17lands API (for statistics and metadata)

**TUI Input Handling:**
- Non-blocking with `stdscr.timeout(0)`
- Special keys: ↑↓ (board scroll), PgUp/PgDn (message scroll)
- Terminal resize handling with curses signals

**JSON Parsing:**
- MTGA log contains multi-line JSON objects
- Buffer accumulation until complete `GreToClientEvent` parsed
- Regex-based event detection: `r'\[UnityCrossThreadLogger\]GreToClientEvent'`

## Code Organization Conventions

- **Section Markers:** Search for `# Part N:` comments (10 major sections)
- **Type Hints:** Comprehensive throughout codebase
- **Logging:** File gets DEBUG, console gets WARNING+ only
- **Error Handling:** Try/except with logging, graceful degradation
- **Dataclasses:** Used for all data models (`GameObject`, `PlayerState`, etc.)
