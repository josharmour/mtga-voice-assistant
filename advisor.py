#!/usr/bin/env python3
import dataclasses
import json
import logging
from pathlib import Path
import re
import requests
import urllib.request
import sqlite3
import time
import os
import threading
from typing import Dict, List, Optional, Callable
import subprocess
import tempfile
import curses
from collections import deque
import sys

# Add src directory to path for refactored modules
sys.path.append(str(Path(__file__).parent / "src"))

from log_parser import LogFollower
from game_state import (
    GameObject, PlayerState, GameHistory, BoardState, MatchScanner, GameStateManager
)
from ai_advisor import OllamaClient, AIAdvisor
from tts import TextToSpeech
from ui import AdvisorTUI, AdvisorGUI
from database import ArenaCardDatabase, check_and_update_card_database

# Import configuration manager for user preferences
try:
    from config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")

# Import RAG system (optional - will gracefully degrade if not available)
try:
    from rag_advisor import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available. Install dependencies with: pip install chromadb sentence-transformers torch")

# Import draft advisor (requires tabulate, termcolor, scipy)
try:
    from draft_advisor import DraftAdvisor, display_draft_pack, format_draft_pack_for_gui
    from deck_builder import DeckBuilder, display_deck_suggestion, format_deck_suggestion_for_gui
    DRAFT_ADVISOR_AVAILABLE = True
except ImportError as e:
    DRAFT_ADVISOR_AVAILABLE = False
    logging.warning(f"Draft advisor not available: {e}. Install with: pip install tabulate termcolor scipy")

# Import Tkinter (optional - for GUI mode)
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available. GUI mode disabled.")

# Ensure the logs directory exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging - file gets all, console gets only errors/warnings
log_file_path = log_dir / "advisor.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Console handler
    ]
)
# Set console handler to WARNING to hide debug/info noise
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.WARNING)

# ----------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------

def clean_card_name(name: str) -> str:
    """
    Remove HTML tags from card names.

    Some card names from Arena's database contain HTML tags like <nobr> and </nobr>
    that need to be stripped for proper display and matching.

    Args:
        name: Raw card name potentially containing HTML tags

    Returns:
        Clean card name with all HTML tags removed

    Examples:
        "<nobr>Full-Throttle</nobr> Fanatic" -> "Full-Throttle Fanatic"
        "<nobr>Bane-Marked</nobr> Leonin" -> "Bane-Marked Leonin"
    """
    if not name:
        return name

    # Remove all HTML tags using regex
    # This handles <nobr>, </nobr>, and any other HTML tags
    clean_name = re.sub(r'<[^>]+>', '', name)

    return clean_name

# ----------------------------------------------------------------------------------
# Part 1: Arena Log Detection and Path Handling
# ----------------------------------------------------------------------------------

def detect_player_log_path():
    """
    Detect the Arena Player.log file based on OS and installation method.
    Returns the path as a string, or None if not found.
    """
    home = Path.home()
    # Windows
    if os.name == 'nt':
        username = os.getenv('USERNAME')
        drive = os.getenv('USERPROFILE')[0]
        windows_path = f"{drive}:/Users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(windows_path):
            return windows_path
    # macOS
    elif os.name == 'posix' and os.uname().sysname == 'Darwin':
        macos_path = home / "Library/Logs/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(macos_path):
            return str(macos_path)
    # Linux
    elif os.name == 'posix':
        username = os.getenv('USER')
        paths = [
            home / f".var/app/com.usebottles.bottles/data/bottles/bottles/MTG-Arena/drive_c/users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / f"Games/magic-the-gathering-arena/drive_c/users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / ".local/share/Steam/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
        ]
        for path in paths:
            if path.exists():
                return str(path)
    return None



class CLIVoiceAdvisor:
    # Available voices in Kokoro v1.0
    AVAILABLE_VOICES = ["af_alloy", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
                        "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo",
                        "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable",
                        "bm_george", "bm_lewis", "ef_dora", "em_alex", "ff_siwis", "hf_alpha",
                        "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha",
                        "pf_dora", "pm_alex", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]

    # Available BarkTTS voices (English speakers)
    BARK_VOICES = ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
                   "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
                   "v2/en_speaker_8", "v2/en_speaker_9"]

    def __init__(self, use_tui: bool = False, use_gui: bool = False):
        self.use_tui = use_tui
        self.use_gui = use_gui
        self.tui = None
        self.gui = None
        self.tk_root = None
        self.previous_board_state = None  # Track previous state for importance detection

        # Load user preferences for persistent settings across sessions
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()
            if not use_tui:
                logging.debug("User preferences loaded successfully")

        # Set continuous monitoring from preferences or use default
        if self.prefs:
            self.continuous_monitoring = self.prefs.opponent_turn_alerts
        else:
            self.continuous_monitoring = True  # Enable continuous advisory mode

        self.last_alert_time = 0  # Timestamp of last critical alert (for rate limiting)

        self.log_path = detect_player_log_path()
        if not self.log_path:
            if not use_tui:
                print("ERROR: Could not find Arena Player.log. Please ensure the game is installed and you have run it at least once.")
            exit(1)

        # Initialize card database and show status
        if not use_tui:
            print("Loading card database...")

        # Pass reskin preference to card database if available
        show_reskin_names = self.prefs.reskin_names if self.prefs else False
        card_db = ArenaCardDatabase(show_reskin_names=show_reskin_names)
        if not use_tui:
            if card_db.conn:
                cursor = card_db.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cards")
                total = cursor.fetchone()[0]
                print(f"✓ Loaded unified card database ({total:,} cards)")
            else:
                print(f"⚠ Card database not found - cards will show as Unknown")
                print("  Run: python build_unified_card_database.py")

        # Check if Ollama is running before initializing AI advisor
        ollama_test = OllamaClient()
        if not ollama_test.is_running():
            if not use_tui:
                print("\n⚠ Ollama service is not running!")
                print("Attempting to start Ollama...")

            if ollama_test.start_ollama():
                if not use_tui:
                    print("✓ Ollama service started successfully")
            else:
                if not use_tui:
                    print("\n❌ Failed to start Ollama automatically.")
                    print("Please start Ollama manually with: ollama serve")
                    print("Then restart this advisor.\n")
                exit(1)
        elif not use_tui:
            print("✓ Ollama service is running")

        self.game_state_mgr = GameStateManager(card_db)
        self.ai_advisor = AIAdvisor(card_db=card_db)
        self.tts = TextToSpeech(voice="am_adam", volume=1.0)
        self.log_follower = LogFollower(self.log_path)

        # Initialize draft advisor if available
        self.draft_advisor = None
        self.deck_builder = None
        if DRAFT_ADVISOR_AVAILABLE:
            try:
                rag_system = self.ai_advisor.rag_system if hasattr(self.ai_advisor, 'rag_system') else None
                ollama_client = self.ai_advisor.client if hasattr(self.ai_advisor, 'client') else None
                self.draft_advisor = DraftAdvisor(card_db, rag_system, ollama_client)
                self.deck_builder = DeckBuilder()

                # Register draft event callbacks with GameStateManager
                self.game_state_mgr.register_draft_callback("EventGetCoursesV2", self._on_draft_pool)
                self.game_state_mgr.register_draft_callback("LogBusinessEvents", self._on_premier_draft_pick)
                self.game_state_mgr.register_draft_callback("Draft.Notify", self._on_draft_notify)
                self.game_state_mgr.register_draft_callback("BotDraftDraftStatus", self._on_quick_draft_status)
                self.game_state_mgr.register_draft_callback("BotDraftDraftPick", self._on_quick_draft_pick)

                if not use_tui:
                    print("✓ Draft advisor enabled")
                    print("✓ Deck builder enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize draft advisor: {e}")
                self.draft_advisor = None
                self.deck_builder = None

        self.last_turn_advised = -1
        self.advice_thread = None
        self.first_turn_detected = False
        self.cli_thread = None
        self.running = True
        self._deck_suggestions_generated = False  # Track if deck suggestions shown
        self._last_announced_pick = None  # Deduplication: track (pack_num, pick_num) of last TTS announcement

        # Fetch available Ollama models
        self.available_models = self._fetch_ollama_models()

    def _output(self, message: str, color: str = "white"):
        """Output message to either CLI, TUI, or GUI"""
        if self.use_gui and self.gui:
            self.gui.add_message(message, color)
        elif self.use_tui and self.tui:
            self.tui.add_message(message, color)
        else:
            print(message)

    def _update_status(self, board_state: BoardState = None):
        """Update status bar for TUI or GUI"""
        if board_state:
            status = f"Turn {board_state.current_turn} | Model: {self.ai_advisor.client.model} | Voice: {self.tts.voice} | Vol: {int(self.tts.volume * 100)}%"
        else:
            status = f"Model: {self.ai_advisor.client.model} | Voice: {self.tts.voice} | Vol: {int(self.tts.volume * 100)}%"

        if self.use_gui and self.gui:
            self.gui.set_status(status)
        elif self.use_tui and self.tui:
            self.tui.set_status(status)

    def get_last_rag_references(self) -> Optional[Dict]:
        """Proxy method to get RAG references from AI advisor"""
        if hasattr(self, 'ai_advisor') and hasattr(self.ai_advisor, 'get_last_rag_references'):
            return self.ai_advisor.get_last_rag_references()
        return None

    # GUI Callback Methods
    def _on_gui_model_change(self, model):
        """Handle model change from GUI with validation and auto-pull"""
        model = model.strip()
        if not model:
            return

        # Check if model is already available locally
        if model in self.available_models:
            self.ai_advisor.client.model = model
            self._update_status()
            logging.info(f"Model changed to: {model}")
            self._output(f"✓ Model changed to: {model}", "green")
            return

        # Model not found locally - try to pull it
        self._output(f"⏳ Model '{model}' not found locally. Attempting to pull from Ollama...", "yellow")
        logging.info(f"Attempting to pull model: {model}")

        try:
            # Use Ollama API to pull the model
            import subprocess
            result = subprocess.run(
                ['ollama', 'pull', model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Refresh available models
                self.available_models = self._fetch_ollama_models()

                # Update GUI dropdown
                if self.gui:
                    self.gui.model_dropdown['values'] = self.available_models

                # Set the new model
                self.ai_advisor.client.model = model
                self._update_status()
                logging.info(f"Successfully pulled and switched to model: {model}")
                self._output(f"✓ Successfully pulled and loaded: {model}", "green")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logging.error(f"Failed to pull model {model}: {error_msg}")
                self._output(f"✗ Failed to pull model '{model}': {error_msg}", "red")
                self._output(f"Available models: {', '.join(self.available_models)}", "cyan")
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout while pulling model: {model}")
            self._output(f"✗ Timeout while pulling model '{model}' (>5 minutes)", "red")
        except FileNotFoundError:
            logging.error("Ollama CLI not found in PATH")
            self._output(f"✗ Ollama CLI not found. Please install Ollama first.", "red")
        except Exception as e:
            logging.error(f"Error pulling model {model}: {e}")
            self._output(f"✗ Error pulling model '{model}': {str(e)}", "red")

    def _on_gui_voice_change(self, voice):
        """Handle voice change from GUI"""
        self.tts.voice = voice
        self._update_status()
        logging.info(f"Voice changed to: {voice}")

    def _on_gui_tts_engine_change(self, engine):
        """Handle TTS engine change from GUI"""
        old_volume = self.tts.volume
        if engine == "bark":
            new_voice = self.BARK_VOICES[0]
        else:
            new_voice = "am_adam"
        self.tts = TextToSpeech(voice=new_voice, volume=old_volume, force_engine=engine)
        self._update_status()
        # Update GUI with new voice list
        if self.gui:
            self.gui.update_settings(
                self.available_models,
                self.AVAILABLE_VOICES,
                self.BARK_VOICES,
                self.ai_advisor.client.model,
                self.tts.voice,
                int(self.tts.volume * 100),
                self.tts.tts_engine
            )
        logging.info(f"TTS engine changed to: {engine}")

    def _on_gui_volume_change(self, volume):
        """Handle volume change from GUI"""
        self.tts.volume = volume / 100.0
        logging.debug(f"Volume set to: {volume}%")

    def _fetch_ollama_models(self) -> list:
        """Query Ollama API for locally installed models"""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = data.get('models', [])
                # Extract model names and sort them
                model_names = sorted([m['name'] for m in models])
                logging.info(f"Found {len(model_names)} Ollama models: {model_names}")
                return model_names
        except Exception as e:
            logging.warning(f"Could not fetch Ollama models: {e}")
            # Return default model as fallback
            return ["llama3.2"]

    # Draft event callbacks
    def _on_draft_pool(self, data: dict):
        """Handle EventGetCoursesV2 - sealed/draft pool event"""
        if not self.draft_advisor:
            return

        try:
            courses = data.get("Courses", [])
            if not courses:
                return

            for course in courses:
                event_name = course.get("InternalEventName", "")
                card_pool = course.get("CardPool", [])
                current_module = course.get("CurrentModule", "")

                # Check if this is a limited event
                if any(fmt in event_name for fmt in ["Sealed", "Draft"]) and card_pool:
                    logging.info(f"Draft pool detected: {event_name} with {len(card_pool)} cards, CurrentModule={current_module}")

                    # Store the most recent card pool and event info for manual deck suggestions
                    self._last_card_pool = card_pool
                    self._last_draft_event_name = event_name
                    self._last_draft_module = current_module

                    # Suppress output - this event fires for all available drafts (including completed ones)
                    # Only show if we actually add Sealed deck building feature
                    # self._output(f"\n📦 Draft Pool Detected: {event_name}", "cyan")
                    # self._output(f"   {len(card_pool)} cards in pool", "cyan")

        except Exception as e:
            logging.error(f"Error handling draft pool event: {e}")

    def _on_draft_notify(self, data: dict):
        """Handle Draft.Notify - Premier Draft pack display"""
        if not self.draft_advisor:
            return

        try:
            # Extract pack information (already 1-indexed from parsing)
            pack_num = data.get("PackNumber", 1)
            pick_num = data.get("PickNumber", 1)
            pack_arena_ids = data.get("PackCards", [])
            draft_id = data.get("DraftId", "")

            if not pack_arena_ids:
                return

            logging.info(f"Draft.Notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")
            print(f"[DEBUG] Processing Draft.Notify: Pack {pack_num}, Pick {pick_num}")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Track previous pack to infer picked card
            # If we have a previous pack and we're now on a new pick, infer what was picked
            if hasattr(self, '_last_draft_pack') and hasattr(self, '_last_draft_recommendation'):
                last_pack_num = getattr(self, '_last_draft_pack_num', 0)
                last_pick_num = getattr(self, '_last_draft_pick_num', 0)

                # Check if we advanced to next pick (same pack, next pick OR next pack, pick 1)
                pick_advanced = (pack_num == last_pack_num and pick_num == last_pick_num + 1) or \
                               (pack_num == last_pack_num + 1 and pick_num == 1)

                if pick_advanced and self._last_draft_recommendation:
                    # Record the recommended card as picked (user likely followed recommendation)
                    # Note: This is an inference - we don't have explicit pick data for Premier Draft
                    picked_card = self._last_draft_recommendation
                    if picked_card and picked_card not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(picked_card)
                        logging.info(f"Inferred pick: {picked_card} (Pack {last_pack_num}, Pick {last_pick_num})")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, draft_id
            )

            # Store current pack info for next iteration
            self._last_draft_pack = pack_arena_ids
            self._last_draft_pack_num = pack_num
            self._last_draft_pick_num = pick_num
            self._last_draft_recommendation = pack_cards[0].name if pack_cards else None

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                from draft_advisor import format_draft_pack_for_gui
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            elif self.use_tui and self.tui:
                # TUI display
                from draft_advisor import format_draft_pack_for_gui
                lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db
                )
                self.tui.set_board_state(lines)
            else:
                # Terminal display
                from draft_advisor import display_draft_pack
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Check if draft is complete
            # Standard draft: Pack 3, Pick 15 or >= 45 cards
            # Pick Two draft: >= 21 cards (7 picks × 2 cards × ~1.5 packs)
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if pack_num == 3 and pick_num == 15 and not pick_two_mode:
                # Standard draft final pick detected
                logging.info("Premier Draft final pick detected - waiting for completion")
            elif len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                # We've tracked enough picks - draft complete!
                logging.info(f"Draft complete! {len(self.draft_advisor.picked_cards)} cards picked (min: {min_cards_for_completion})")
                self._generate_deck_suggestions(draft_id)

        except Exception as e:
            logging.error(f"Error handling Draft.Notify: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _on_premier_draft_pick(self, data: dict):
        """Handle LogBusinessEvents - Premier Draft pick"""
        if not self.draft_advisor:
            return

        try:
            # Extract pack information
            event_id = data.get("EventId", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            cards_in_pack = data.get("CardsInPack", [])

            if not cards_in_pack:
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in cards_in_pack]

            logging.info(f"Premier Draft Pick: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Track previous pack to infer picked card (similar to Draft.Notify)
            if hasattr(self, '_last_premier_pack') and hasattr(self, '_last_premier_recommendation'):
                last_pack_num = getattr(self, '_last_premier_pack_num', 0)
                last_pick_num = getattr(self, '_last_premier_pick_num', 0)

                # Check if we advanced to next pick
                pick_advanced = (pack_num == last_pack_num and pick_num == last_pick_num + 1) or \
                               (pack_num == last_pack_num + 1 and pick_num == 1)

                if pick_advanced and self._last_premier_recommendation:
                    picked_card = self._last_premier_recommendation
                    if picked_card and picked_card not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(picked_card)
                        logging.info(f"Inferred pick: {picked_card} (Pack {last_pack_num}, Pick {last_pick_num})")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_id
            )

            # Store current pack info for next iteration
            self._last_premier_pack = pack_arena_ids
            self._last_premier_pack_num = pack_num
            self._last_premier_pick_num = pick_num
            self._last_premier_recommendation = pack_cards[0].name if pack_cards else None

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Check if draft is complete
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                logging.info(f"Draft complete! {len(self.draft_advisor.picked_cards)} cards picked")
                self._generate_deck_suggestions(event_id)

        except Exception as e:
            logging.error(f"Error handling Premier Draft pick: {e}")

    def _on_quick_draft_status(self, data: dict):
        """Handle BotDraftDraftStatus - Quick Draft status (pack + pool)"""
        print(f"[DEBUG] _on_quick_draft_status called!")
        if not self.draft_advisor:
            print(f"[DEBUG] No draft advisor available!")
            return

        try:
            event_name = data.get("EventName", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            draft_pack = data.get("DraftPack", [])
            picked_cards = data.get("PickedCards", [])

            if not draft_pack:
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in draft_pack]

            logging.info(f"Quick Draft Status: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Update draft advisor's picked cards list
            for card_id in picked_cards:
                try:
                    card_name = self.game_state_mgr.card_lookup.get_card_name(int(card_id))
                    if card_name and card_name not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(card_name)
                except Exception as e:
                    logging.debug(f"Error recording picked card: {e}")

            # Check if draft is complete
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                self._generate_deck_suggestions(event_name)

        except Exception as e:
            logging.error(f"Error handling Quick Draft status: {e}")

    def _on_quick_draft_pick(self, data: dict):
        """Handle BotDraftDraftPick - Quick Draft pick confirmation AND next pack"""
        print(f"[DEBUG] _on_quick_draft_pick called! Data keys: {list(data.keys())}")

        # BotDraftDraftPick response includes the NEXT pack to pick from!
        # So we need to process it here, not wait for BotDraftDraftStatus
        if not self.draft_advisor:
            print(f"[DEBUG] No draft advisor available!")
            return

        try:
            event_name = data.get("EventName", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            draft_pack = data.get("DraftPack", [])
            picked_cards = data.get("PickedCards", [])

            print(f"[DEBUG] Processing: Pack {pack_num}, Pick {pick_num}, {len(draft_pack)} cards in pack")

            if not draft_pack:
                print(f"[DEBUG] No draft pack in this response")
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in draft_pack]

            logging.info(f"Quick Draft Pick: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            print(f"[DEBUG] Got {len(pack_cards)} cards, recommendation: {recommendation}")

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
                print(f"[DEBUG] Sent to GUI!")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Update draft advisor's picked cards list
            for card_id in picked_cards:
                try:
                    card_name = self.game_state_mgr.card_lookup.get_card_name(int(card_id))
                    if card_name and card_name not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(card_name)
                except Exception as e:
                    logging.debug(f"Error recording picked card: {e}")

            # Check if draft is complete (use lower thresholds since pick inference isn't perfect)
            # Standard draft: 40+ cards (out of 45), Pick Two: 18+ cards (out of 21)
            num_cards_to_pick = data.get("NumCardsToPick", 1)
            total_picks_needed = 18 if num_cards_to_pick == 2 else 40

            if len(self.draft_advisor.picked_cards) >= total_picks_needed:
                self._generate_deck_suggestions(event_name)

        except Exception as e:
            logging.error(f"Error handling Quick Draft pick: {e}")
            import traceback
            traceback.print_exc()

    def _generate_deck_suggestions(self, event_name: str):
        """Generate and display deck building suggestions"""
        if not self.deck_builder or not self.draft_advisor:
            return

        # Prevent duplicate suggestions - only generate once per draft
        if hasattr(self, '_deck_suggestions_generated') and self._deck_suggestions_generated:
            logging.debug("Deck suggestions already generated for this draft")
            return

        try:
            # Extract set code and format from event name
            set_code = None
            format_type = "PremierDraft"  # default
            if event_name and "_" in event_name:
                parts = event_name.split("_")
                if len(parts) >= 2:
                    format_type = parts[0]  # e.g., "QuickDraft", "PremierDraft"
                    set_code = parts[1].upper()

            if not set_code:
                logging.warning("Could not determine set code from event name")
                return

            # Auto-download 17lands data if needed
            try:
                from auto_updater import AutoUpdater
                updater = AutoUpdater(auto_mode=True)  # Auto mode for seamless experience
                if not updater.update_for_draft(set_code, format_type):
                    logging.warning(f"Could not ensure data for {set_code} {format_type}")
            except Exception as e:
                logging.debug(f"Auto-updater not available: {e}")

            drafted_cards = self.draft_advisor.picked_cards

            # Minimum card check: PickTwoDraft has 21 picks, standard draft has 45
            # We need at least 20 cards to build a reasonable deck
            if not drafted_cards or len(drafted_cards) < 20:
                logging.warning(f"Not enough cards drafted: {len(drafted_cards)}")
                return

            logging.info(f"🏗️  Generating deck suggestions for {len(drafted_cards)} drafted cards from {set_code}...")

            # Generate suggestions (top 3 color pairs)
            suggestions = self.deck_builder.suggest_deck(
                drafted_cards,
                set_code,
                top_n=3
            )

            if not suggestions:
                self._output("⚠️  No deck suggestions available (17lands data may be missing)", "yellow")
                return

            # Display based on mode
            if self.use_gui and self.gui:
                # Format all suggestions for GUI
                all_lines = []
                all_lines.append("="*80)
                all_lines.append("DRAFT COMPLETE - Deck Suggestions")
                all_lines.append("="*80)
                all_lines.append("")

                for i, suggestion in enumerate(suggestions, 1):
                    if i == 1:
                        all_lines.append(f"🏆 BEST MATCH: {suggestion.color_pair_name}")
                    else:
                        all_lines.append(f"📊 ALTERNATIVE #{i-1}: {suggestion.color_pair_name}")
                    all_lines.append("")

                    # Add formatted deck suggestion
                    deck_lines = format_deck_suggestion_for_gui(suggestion)
                    all_lines.extend(deck_lines)

                all_lines.append("ℹ️  Copy the suggested deck into MTGA, then return here for gameplay advice!")
                all_lines.append("="*80)

                # Display in GUI
                self.gui.reset_pane_labels()  # Reset to "BOARD STATE" and "ADVISOR"
                self.gui.set_board_state(all_lines)
                self.gui.add_message(f"Draft complete! Suggested: {suggestions[0].color_pair_name}", "green")

                # Speak the top suggestion
                if self.tts:
                    self.tts.speak(f"Suggested deck: {suggestions[0].color_pair_name}")

            else:
                # Terminal display
                self._output("\n" + "="*80, "cyan")
                self._output("DRAFT COMPLETE - Building Deck Suggestions...", "cyan")
                self._output("="*80 + "\n", "cyan")

                for i, suggestion in enumerate(suggestions, 1):
                    if i == 1:
                        self._output(f"\n🏆 BEST MATCH: {suggestion.color_pair_name}", "green")
                    else:
                        self._output(f"\n📊 ALTERNATIVE #{i-1}: {suggestion.color_pair_name}", "cyan")

                    display_deck_suggestion(suggestion)

                    # Speak the top suggestion
                    if i == 1 and self.tts:
                        self.tts.speak(f"Suggested deck: {suggestion.color_pair_name}")

                self._output("\nℹ️  Copy the suggested deck into MTGA, then return here for gameplay advice!", "cyan")
                self._output("="*80 + "\n", "cyan")

            # Mark suggestions as generated to prevent duplicates
            self._deck_suggestions_generated = True

        except Exception as e:
            logging.error(f"Error generating deck suggestions: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Start the advisor with background log monitoring and interactive CLI"""
        if self.use_gui:
            self.run_gui()
            return
        if self.use_tui:
            self.run_tui()
            return

        print("\n" + "="*60)
        print("MTGA Voice Advisor Started")
        print("="*60)
        print(f"Log: {self.log_path}")
        print(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)")
        print(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%")
        print("\nWaiting for a match... (Enable Detailed Logs in MTGA settings)")
        print("Type /help for commands\n")

        # Speak startup message so user knows audio is working
        startup_msg = "All systems online. Waiting for match."
        self.tts.speak(startup_msg)

        # Start log follower in background thread
        log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
        log_thread.start()

        # Interactive CLI loop
        self._run_cli_loop()

    def run_tui(self):
        """Start the advisor with TUI interface"""
        def _tui_main(stdscr):
            # Initialize TUI
            self.tui = AdvisorTUI(stdscr)

            # Display startup message
            self._output(f"✓ MTGA Voice Advisor Started", "green")
            self._output(f"Log: {self.log_path}", "blue")
            self._output(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)", "blue")
            self._output(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%", "blue")
            self._output(f"✓ Ollama service connected", "green")
            self._output("", "white")
            self._output("Waiting for a match... (Enable Detailed Logs in MTGA settings)", "cyan")
            self._output("Type /help for commands", "cyan")

            # Update status bar
            self._update_status()

            # Speak startup message so user knows audio is working
            startup_msg = "All systems online. Waiting for match."
            self.tts.speak(startup_msg)

            # Start log follower in background thread
            log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
            log_thread.start()

            # Set up input callback
            def on_input(user_input: str):
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_query(user_input)

            # TUI event loop
            try:
                while self.running:
                    # Handle input (non-blocking)
                    self.tui.get_input(on_input)
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.05)
            except KeyboardInterrupt:
                self._output("\nShutting down...", "yellow")
                self.running = False
                self.log_follower.close()
            finally:
                self.tui.cleanup()

                # Save preferences before exiting
                if self.prefs and CONFIG_MANAGER_AVAILABLE:
                    self.prefs.save()
                    logging.info("User preferences saved")

                # Clean up database connection
                if hasattr(self.game_state_mgr.card_lookup, 'close'):
                    self.game_state_mgr.card_lookup.close()
                    logging.info("Card database connection closed")

        curses.wrapper(_tui_main)

    def run_gui(self):
        """Start the advisor with Tkinter GUI interface"""
        if not TKINTER_AVAILABLE:
            print("ERROR: Tkinter is not available. Install with: sudo apt-get install python3-tk")
            exit(1)

        # Create Tk root
        self.tk_root = tk.Tk()
        self.gui = AdvisorGUI(self.tk_root, self)

        # Initialize settings
        self.gui.update_settings(
            self.available_models,
            self.AVAILABLE_VOICES,
            self.BARK_VOICES,
            self.ai_advisor.client.model,
            self.tts.voice,
            int(self.tts.volume * 100),
            self.tts.tts_engine
        )

        # Display startup messages
        self._output(f"✓ MTGA Voice Advisor Started", "green")
        self._output(f"Log: {self.log_path}", "cyan")
        self._output(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)", "cyan")
        self._output(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%", "cyan")
        self._output(f"✓ Ollama service connected", "green")
        self._output("", "white")
        self._output("Waiting for a match... (Enable Detailed Logs in MTGA settings)", "cyan")

        # Update status bar
        self._update_status()

        # Speak startup message so user knows audio is working
        startup_msg = "All systems online. Waiting for match."
        self.tts.speak(startup_msg)

        # Start log follower in background thread
        log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
        log_thread.start()

        # Run Tk main loop
        try:
            self.tk_root.mainloop()
        except KeyboardInterrupt:
            self._output("\nShutting down...", "yellow")
        finally:
            self.running = False
            if hasattr(self, 'log_follower'):
                self.log_follower.close()
            if hasattr(self.game_state_mgr.card_lookup, 'close'):
                self.game_state_mgr.card_lookup.close()
                logging.info("Card database connection closed")
            if self.gui:
                self.gui.cleanup()

    def _run_log_monitor(self):
        """Monitor Arena log in background"""
        try:
            self.log_follower.follow(self.on_line)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error(f"Log monitor error: {e}")

    def _run_cli_loop(self):
        """Interactive command prompt for user input"""
        try:
            while self.running:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        self._handle_command(user_input)
                    else:
                        # Treat as free-form query to the AI about the current board
                        self._handle_query(user_input)
                except EOFError:
                    # Running in background without stdin - just sleep and continue
                    logging.info("Running in background mode (no stdin available)")
                    while self.running:
                        time.sleep(1)
                    break
                except KeyboardInterrupt:
                    print("\n\nShutting down...")
                    self.running = False
                    self.log_follower.close()
                    break
        except Exception as e:
            logging.error(f"CLI error: {e}")
        finally:
            # Save preferences before exiting
            if self.prefs and CONFIG_MANAGER_AVAILABLE:
                self.prefs.save()
                logging.info("User preferences saved")

            # Clean up database connection
            if hasattr(self.game_state_mgr.card_lookup, 'close'):
                self.game_state_mgr.card_lookup.close()
                logging.info("Card database connection closed")

    def _handle_command(self, command: str):
        """Handle slash commands"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._show_help()
        elif cmd == "/quit" or cmd == "/exit":
            self._output("Exiting...", "yellow")
            self.running = False
            if self.use_tui and self.tui:
                self.tui.running = False
        elif cmd == "/clear":
            if self.use_tui and self.tui:
                self.tui.messages.clear()
                self.tui._refresh_messages()
                logging.info("Message history cleared")
            else:
                print("\n" * 50)  # Clear console
        elif cmd == "/settings":
            if self.use_tui and self.tui:
                self._show_settings_interactive()
            else:
                self._show_settings()
        elif cmd == "/voice":
            if len(parts) > 1:
                voice = parts[1].lower()
                if voice in self.AVAILABLE_VOICES:
                    self.tts.set_voice(voice)
                    self._output(f"✓ Voice changed to: {voice}", "green")
                    self._update_status()
                else:
                    self._output(f"✗ Unknown voice. Available: {', '.join(self.AVAILABLE_VOICES[:5])}...", "red")
            else:
                self._output(f"✓ Current voice: {self.tts.voice}", "green")
        elif cmd == "/volume":
            if len(parts) > 1:
                try:
                    vol = float(parts[1]) / 100.0
                    self.tts.set_volume(vol)
                    self._output(f"✓ Volume set to: {int(vol * 100)}%", "green")
                    self._update_status()
                except ValueError:
                    self._output("✗ Volume must be a number (0-100)", "red")
            else:
                self._output(f"✓ Current volume: {int(self.tts.volume * 100)}%", "green")
        elif cmd == "/status":
            board_state = self.game_state_mgr.get_current_board_state()
            if board_state:
                self._output(f"Turn: {board_state.current_turn} | Your Turn: {board_state.is_your_turn} | Has Priority: {board_state.has_priority}", "cyan")
                self._output(f"Your Hand: {len(board_state.your_hand)} cards", "blue")
                self._output(f"Your Battlefield: {len(board_state.your_battlefield)} permanents", "blue")
                self._output(f"Opponent Battlefield: {len(board_state.opponent_battlefield)} permanents", "blue")
            else:
                self._output("No match in progress", "yellow")
        elif cmd == "/tts":
            if self.tts.tts_engine:
                engine_name = "Kokoro" if self.tts.tts_engine == "kokoro" else "BarkTTS"
                self._output(f"✓ TTS Engine: {engine_name} ({self.tts.tts_engine})", "green")
                if self.tts.tts_engine == "kokoro":
                    self._output(f"  Voice: {self.tts.voice}", "blue")
                else:
                    self._output(f"  Using Bark's built-in voice presets", "blue")
            else:
                self._output("✗ No TTS engine initialized", "red")
        elif cmd == "/models":
            model_lines = [
                f"Available Ollama models ({len(self.available_models)}):",
                ""
            ]
            for i, model in enumerate(self.available_models, 1):
                marker = " (current)" if model == self.ai_advisor.client.model else ""
                model_lines.append(f"  {i}. {model}{marker}")
            model_lines.append("")
            model_lines.append("Use '/model <name>' to switch models")

            if self.use_tui and self.tui:
                self.tui.show_popup(model_lines, "Available Models")
            else:
                for line in model_lines:
                    print(line)
        elif cmd == "/model":
            if len(parts) > 1:
                model_name = parts[1]
                # Check if model exists in available models (support partial match)
                matching_models = [m for m in self.available_models if model_name in m]
                if matching_models:
                    if len(matching_models) == 1:
                        selected_model = matching_models[0]
                        self.ai_advisor.client.model = selected_model
                        self._output(f"✓ Model changed to: {selected_model}", "green")
                        self._update_status()
                        logging.info(f"Ollama model switched to: {selected_model}")
                    else:
                        self._output(f"✗ Ambiguous model name. Matching models: {', '.join(matching_models)}", "red")
                        self._output(f"  Please be more specific.", "red")
                else:
                    self._output(f"✗ Model not found: {model_name}", "red")
                    self._output(f"  Available models: {', '.join(self.available_models[:5])}{'...' if len(self.available_models) > 5 else ''}", "yellow")
            else:
                self._output(f"✓ Current model: {self.ai_advisor.client.model}", "green")
        elif cmd == "/continuous":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.continuous_monitoring = True
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = True
                    self._output("✓ Continuous monitoring ENABLED - AI will alert you of critical changes anytime", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.continuous_monitoring = False
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = False
                    self._output("✓ Continuous monitoring DISABLED - advice only when you have priority", "yellow")
                else:
                    self._output("✗ Use '/continuous on' or '/continuous off'", "red")
            else:
                status = "ENABLED" if self.continuous_monitoring else "DISABLED"
                self._output(f"✓ Continuous monitoring: {status}", "green")
        elif cmd == "/opponent-alerts":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.continuous_monitoring = True
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = True
                    self._output("✓ Opponent Turn Alerts: ON", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.continuous_monitoring = False
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = False
                    self._output("✓ Opponent Turn Alerts: OFF", "yellow")
                else:
                    self._output("✗ Use '/opponent-alerts on' or '/opponent-alerts off'", "red")
            else:
                status = "ON" if self.continuous_monitoring else "OFF"
                self._output(f"Opponent Turn Alerts: {status}", "green")
        elif cmd == "/reskins":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.game_state_mgr.card_lookup.show_reskin_names = True
                    if self.prefs:
                        self.prefs.reskin_names = True
                    self._output("✓ Reskin names: ON (Spider-Man variants enabled)", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.game_state_mgr.card_lookup.show_reskin_names = False
                    if self.prefs:
                        self.prefs.reskin_names = False
                    self._output("✓ Reskin names: OFF (canonical names)", "yellow")
                else:
                    self._output("✗ Use '/reskins on' or '/reskins off'", "red")
            else:
                status = "ON" if self.game_state_mgr.card_lookup.show_reskin_names else "OFF"
                self._output(f"Reskin names: {status}", "green")
        else:
            self._output(f"Unknown command: {cmd}. Type /help for commands.", "red")

    def _handle_query(self, query: str):
        """Handle free-form queries to the AI about current board state"""
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state or not board_state.current_turn:
            self._output("No match in progress. Start a game first.", "yellow")
            return

        self._output("\nThinking...", "cyan")
        # Use the AI to answer the query in context of current board state
        prompt = f"""
The user is asking about their current board state in Magic: The Gathering Arena.

Current Board State:
- Current Turn: {board_state.current_turn}
- Your Hand: {', '.join([c.name for c in board_state.your_hand]) if board_state.your_hand else 'Empty'}
- Your Battlefield: {', '.join([c.name for c in board_state.your_battlefield]) if board_state.your_battlefield else 'Empty'}
- Opponent Battlefield: {', '.join([c.name for c in board_state.opponent_battlefield]) if board_state.opponent_battlefield else 'Empty'}

User's Question: {query}

Provide a concise answer (1-2 sentences) based on the board state.
"""
        try:
            response = self.ai_advisor.client.generate(prompt)
            if response:
                # Parse thinking from answer
                thinking, answer = self._parse_reasoning_response(response)

                # Display thinking (not spoken)
                if thinking:
                    self._output("💭 Thinking:", "blue")
                    for line in thinking.split("\n"):
                        if line.strip():
                            self._output(f"   {line.strip()}", "blue")
                    self._output("", "white")

                # Display and speak answer
                final_answer = answer if answer else response
                self._output(f"Advisor: {final_answer}\n", "green")
                clean_response = self._strip_markdown(final_answer)
                self.tts.speak(clean_response)
            else:
                self._output("No response from AI.\n", "yellow")
        except Exception as e:
            self._output(f"Error getting response: {e}\n", "red")
            logging.error(f"Query error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _show_help(self):
        """Display help menu"""
        help_lines = [
            "",
            "Commands:",
            "  /help                    - Show this help menu",
            "  /settings                - Interactive settings menu (TUI: ↑↓ Enter ESC)",
            "  /clear                   - Clear message history",
            "  /quit or /exit           - Exit the advisor",
            "  /opponent-alerts [on/off] - Toggle opponent turn alerts (alerts for critical changes anytime)",
            "  /reskins [on/off]        - Show Spider-Man reskin names (e.g., /reskins on)",
            "  /tts                     - Show active TTS engine (Kokoro or BarkTTS)",
            "  /voice [name]            - Change voice (e.g., /voice bella, /voice v2/en_speaker_3)",
            "  /volume [0-100]          - Set volume (e.g., /volume 80)",
            "  /models                  - List available Ollama models",
            "  /model [name]            - Change AI model (e.g., /model llama3.2, /model qwen)",
            "  /status                  - Show current board state",
            "",
            "Interactive Settings (TUI mode):",
            "  ↑/↓ arrows      - Navigate settings",
            "  Enter/Space     - Cycle through options (Model, Voice, TTS Engine)",
            "  +/-             - Adjust volume",
            "  ESC or Q        - Close settings menu",
            "",
            "Free-form queries:",
            "  Type any question about your board state and the advisor will answer.",
            "",
            "Model Selection:",
            "  Smaller models = faster but less smart (e.g., llama3.2:1b)",
            "  Larger models = slower but smarter (e.g., llama3.2:70b)",
            "",
            "TTS Engines:",
            "  Kokoro: Fast, high-quality, 30+ voices (primary)",
            "  BarkTTS: Fallback engine with built-in voices",
            "  Toggle between them in /settings menu",
            "",
        ]

        if self.use_tui and self.tui:
            self.tui.show_popup(help_lines, "Help")
        else:
            for line in help_lines:
                print(line)

    def _show_settings(self):
        """Show current settings (non-interactive)"""
        self._output("", "white")
        self._output("Current Settings:", "cyan")
        self._output(f"  AI Model: {self.ai_advisor.client.model}", "white")

        # TTS Engine info
        if self.tts.tts_engine:
            engine_name = "Kokoro" if self.tts.tts_engine == "kokoro" else "BarkTTS"
            self._output(f"  TTS:      {engine_name}", "white")
            if self.tts.tts_engine == "kokoro":
                self._output(f"  Voice:    {self.tts.voice}", "white")
        else:
            self._output(f"  TTS:      None (initialization failed)", "white")

        self._output(f"  Volume:   {int(self.tts.volume * 100)}%", "white")
        self._output(f"  Log:      {self.log_path}", "white")
        self._output("", "white")

    def _show_settings_interactive(self):
        """Show interactive settings menu (TUI only)"""
        def settings_callback(action, value):
            if action == "get_values":
                return (
                    self.available_models,
                    self.AVAILABLE_VOICES,
                    self.BARK_VOICES,
                    self.ai_advisor.client.model,
                    self.tts.voice,
                    int(self.tts.volume * 100),
                    self.tts.tts_engine
                )
            elif action == "model":
                self.ai_advisor.client.model = value
                self._update_status()
                # Don't spam message log - just update status bar
                logging.info(f"Model changed to: {value}")
            elif action == "voice":
                self.tts.voice = value
                self._update_status()
                # Don't spam message log - just update status bar
                logging.info(f"Voice changed to: {value}")
            elif action == "volume":
                self.tts.volume = value / 100.0
                self._update_status()
                # Don't spam message log - just update status bar
                logging.debug(f"Volume set to: {value}%")
            elif action == "tts_engine":
                # Reinitialize TTS with new engine and appropriate default voice
                old_volume = self.tts.volume
                # Set default voice for the new engine
                if value == "bark":
                    new_voice = self.BARK_VOICES[0]  # Default to first bark voice
                else:
                    new_voice = "am_adam"  # Default kokoro voice
                self.tts = TextToSpeech(voice=new_voice, volume=old_volume, force_engine=value)
                self._update_status()
                engine_name = "Kokoro" if value == "kokoro" else "BarkTTS"
                # Don't spam message log - just update status bar
                logging.info(f"TTS engine changed to: {value} with voice {new_voice}")

        self.tui.show_settings_menu(settings_callback)

    def on_line(self, line: str):
        """Parse log line and update game state"""
        logging.debug(f"Received line in on_line: {line[:100]}...")
        state_changed = self.game_state_mgr.parse_log_line(line)
        if state_changed:
            logging.debug("Game state changed. Checking for decision point.")
            self._check_for_decision_point()

    def _check_for_decision_point(self):
        """Check if we should give automatic advice or important updates"""
        logging.debug("Checking for decision point...")
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state:
            logging.debug("No board state available yet.")
            return

        # Check for mulligan phase
        logging.debug(f"Board state in_mulligan_phase: {board_state.in_mulligan_phase}, game_stage: {board_state.game_stage}")

        if board_state.in_mulligan_phase:
            logging.info("🎴 IN MULLIGAN PHASE - will display hand and give advice")
            # Display board state with opening hand
            self._display_board_state(board_state)

            # Only generate mulligan advice once
            if not hasattr(self, 'mulligan_advice_given') or not self.mulligan_advice_given:
                logging.info("🎴 Mulligan phase detected - generating mulligan advice")
                self.mulligan_advice_given = True

                if self.advice_thread and self.advice_thread.is_alive():
                    logging.info("Still processing previous advice request.")
                    return

                self.advice_thread = threading.Thread(target=self._generate_mulligan_advice, args=(board_state,))
                self.advice_thread.start()
            return
        else:
            # Reset mulligan flag when we're out of mulligan phase
            if hasattr(self, 'mulligan_advice_given') and self.mulligan_advice_given:
                logging.info("Resetting mulligan advice flag")
            self.mulligan_advice_given = False

        if board_state.current_turn is None:
            logging.debug("Current turn not yet determined.")
            return

        # On first turn detection, sync to current turn so we only advise FUTURE turns
        if not self.first_turn_detected:
            self.last_turn_advised = board_state.current_turn - 1
            self.first_turn_detected = True
            self.previous_board_state = board_state
            logging.info(f"First turn detected (Turn {board_state.current_turn}). Will advise starting from Turn {board_state.current_turn + 1}")
            return

        # Check for critical updates even during opponent's turn (continuous monitoring)
        if self.continuous_monitoring and self.previous_board_state:
            if self.advice_thread and self.advice_thread.is_alive():
                logging.debug("Advice thread still running, skipping continuous check")
            else:
                # Rate limit: Only check for critical updates every 10 seconds minimum
                current_time = time.time()
                if current_time - self.last_alert_time >= 10:
                    # Check if something critical happened
                    critical_advice = self.ai_advisor.check_important_updates(board_state, self.previous_board_state)
                    if critical_advice:
                        logging.info("Critical update detected - speaking immediately")
                        self._speak_advice(f"Alert: {critical_advice}")
                        self.last_alert_time = current_time
                else:
                    logging.debug(f"Rate limit active: {int(10 - (current_time - self.last_alert_time))}s until next alert check")

        # Update previous state
        self.previous_board_state = board_state

        # Standard priority-based advice for your turn
        is_new_turn_for_player = board_state.is_your_turn and board_state.current_turn > self.last_turn_advised

        # Check if opponent is attacking and you need blocking advice
        opponent_is_attacking = board_state.history and len(board_state.history.current_attackers) > 0
        is_blocking_step = opponent_is_attacking and board_state.has_priority and not board_state.is_your_turn

        # Track if we've given blocking advice for this combat
        if not hasattr(self, '_last_combat_advised'):
            self._last_combat_advised = None

        # Create a unique identifier for this combat (turn + attackers)
        if opponent_is_attacking:
            combat_id = (board_state.current_turn, tuple(sorted(board_state.history.current_attackers)))
        else:
            combat_id = None

        need_blocking_advice = is_blocking_step and combat_id and combat_id != self._last_combat_advised

        if board_state.has_priority and (is_new_turn_for_player or need_blocking_advice):
            if self.advice_thread and self.advice_thread.is_alive():
                logging.info("Still processing previous advice request.")
                return

            if is_new_turn_for_player:
                self.last_turn_advised = board_state.current_turn
                logging.info(f"Generating advice for new turn: {board_state.current_turn}")

            if need_blocking_advice:
                self._last_combat_advised = combat_id
                logging.info(f"⚔️ Opponent attacking with {len(board_state.history.current_attackers)} creatures - generating blocking advice")

            self.advice_thread = threading.Thread(target=self._generate_and_speak_advice, args=(board_state,))
            self.advice_thread.start()

    def _format_card_display(self, card: GameObject) -> str:
        """Format card display with name, type, P/T, and status indicators"""
        # Check if card is unknown
        if "Unknown" in card.name:
            return f"{card.name} ⚠️"

        # Build status indicators
        status_parts = []

        # Add tapped status
        if card.is_tapped:
            status_parts.append("🔄")

        # Add power/toughness for creatures
        if card.power is not None and card.toughness is not None:
            # Handle dict format {'value': int} just in case
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

        # Combine all parts
        if status_parts:
            return f"{card.name} ({', '.join(status_parts)})"
        else:
            return card.name

    def _display_board_state(self, board_state: BoardState):
        """Display a comprehensive visual representation of the current board state"""
        # Build board state lines
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
            if history.cards_played_this_turn or history.died_this_turn or history.lands_played_this_turn:
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

        if board_state.opponent_graveyard:
            recent = board_state.opponent_graveyard[-5:]
            lines.append(f"  ⚰️ Graveyard ({len(board_state.opponent_graveyard)}): {', '.join([c.name for c in recent])}")

        if board_state.opponent_exile:
            lines.append(f"  🚫 Exile ({len(board_state.opponent_exile)}): {', '.join([c.name for c in board_state.opponent_exile])}")

        # Stack (shared)
        if board_state.stack:
            lines.append("")
            lines.append("─"*70)
            lines.append(f"📋 STACK ({len(board_state.stack)}):")
            for card in board_state.stack:
                lines.append(f"   ⚡ {card.name}")

        # Your info
        lines.append("")
        lines.append("─"*70)
        your_lib = board_state.your_library_count if board_state.your_library_count > 0 else "?"
        lines.append(f"YOU: ❤️  {board_state.your_life} life | 🃏 {board_state.your_hand_count} cards | 📖 {your_lib} library")

        lines.append("")
        lines.append(f"  🃏 Hand ({len(board_state.your_hand)}):")
        if board_state.your_hand:
            for card in board_state.your_hand:
                card_info = self._format_card_display(card)
                lines.append(f"      • {card_info}")
        else:
            lines.append("      (empty)")

        lines.append("")
        lines.append(f"  ⚔️  Battlefield ({len(board_state.your_battlefield)}):")
        if board_state.your_battlefield:
            for card in board_state.your_battlefield:
                card_info = self._format_card_display(card)
                lines.append(f"      • {card_info}")
        else:
            lines.append("      (empty)")

        if board_state.your_graveyard:
            recent = board_state.your_graveyard[-5:]
            lines.append(f"  ⚰️ Graveyard ({len(board_state.your_graveyard)}): {', '.join([c.name for c in recent])}")

        if board_state.your_exile:
            lines.append(f"  🚫 Exile ({len(board_state.your_exile)}): {', '.join([c.name for c in board_state.your_exile])}")

        lines.append("")
        lines.append("="*70)

        # Output board state
        if self.use_gui and self.gui:
            self.gui.set_board_state(lines)
        elif self.use_tui and self.tui:
            self.tui.set_board_state(lines)
        else:
            for line in lines:
                print(line)

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting for TTS (asterisks, hashtags, etc.)"""
        # Remove bold/italic asterisks
        text = re.sub(r'\*+', '', text)
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove bullet points
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
        return text

    def _parse_reasoning_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to separate thinking/reasoning from final answer.
        Returns (thinking, answer) tuple.
        """
        # DeepSeek-R1 and similar models use <think> tags
        if "<think>" in response and "</think>" in response:
            think_start = response.find("<think>")
            think_end = response.find("</think>") + len("</think>")
            thinking = response[think_start+7:think_end-8].strip()  # Extract content between tags
            answer = response[think_end:].strip()
            return (thinking, answer)

        # Some models use "Thought:" or "Reasoning:" prefixes
        if response.startswith("Thought:") or response.startswith("Reasoning:"):
            lines = response.split("\n", 1)
            if len(lines) > 1:
                return (lines[0], lines[1].strip())

        # No thinking detected, return empty thinking and full response as answer
        return ("", response)

    def _speak_advice(self, advice_text: str):
        """Speak advice text (helper for critical updates)"""
        if advice_text:
            # Parse thinking from advice for critical alerts too
            thinking, answer = self._parse_reasoning_response(advice_text)

            # Display thinking (not spoken)
            if thinking:
                self._output("🔴 Critical Alert - Thinking:", "red")
                for line in thinking.split("\n")[:3]:  # Limit to 3 lines for critical alerts
                    if line.strip():
                        self._output(f"   {line.strip()}", "blue")

            # Speak and display the answer
            final_text = answer if answer else advice_text
            clean_advice = self._strip_markdown(final_text)
            self.tts.speak(clean_advice)
            self._output(f"🔴 Alert: {final_text}", "red")

    def _generate_mulligan_advice(self, board_state: BoardState):
        """Generate and speak advice for mulligan decision, enhanced with 17lands data."""
        self._output("\n🎴 MULLIGAN DECISION", "cyan")
        self._output("Analyzing your opening hand with 17lands data...\n", "cyan")

        if not board_state.your_hand:
            self._output("⚠ No hand visible yet - waiting for cards...", "yellow")
            return

        hand_cards_with_stats = []
        total_oh_wr = 0
        cards_with_stats_count = 0

        for card in board_state.your_hand:
            stats = self.ai_advisor.rag_system.card_stats.get_card_stats(card.name) if self.ai_advisor.rag_system else None
            oh_wr = stats.get('opening_hand_win_rate', 0.0) if stats else 0.0
            if oh_wr > 0:
                total_oh_wr += oh_wr
                cards_with_stats_count += 1
            hand_cards_with_stats.append(f"{card.name} (OH WR: {oh_wr:.1%})")
        
        avg_oh_wr = total_oh_wr / cards_with_stats_count if cards_with_stats_count > 0 else 0.0

        deck_strategy = ""
        if board_state.your_decklist:
            total_lands = sum(count for name, count in board_state.your_decklist.items() if "Land" in name or any(lt in name for lt in ["Forest", "Plains", "Island", "Mountain", "Swamp"]))
            total_spells = sum(board_state.your_decklist.values()) - total_lands
            key_cards = [f"{name} (x{count})" for name, count in list(board_state.your_decklist.items())[:5] if "Land" not in name]
            deck_strategy = f"Your deck has {total_lands} lands and {total_spells} spells. Key cards: {', '.join(key_cards)}."

        mulligan_prompt = f"""You are analyzing an opening hand in Magic: The Gathering.

OPENING HAND ({len(board_state.your_hand)} cards):
{', '.join(hand_cards_with_stats)}

HAND ANALYSIS:
- Average Opening Hand Win Rate (OH WR): {avg_oh_wr:.1%} (based on 17lands data)
- A good hand is typically >55% OH WR. A hand below 50% is often a mulligan.

DECK INFORMATION:
{deck_strategy or "Deck information not available."}

MULLIGAN DECISION CRITERIA:
1. Land count: Is it between 2-4 for a 7-card hand?
2. Data-driven advice: Is the average OH WR acceptable?
3. Curve & Synergy: Can you make plays in the early turns?

Based on all this information, should the player mulligan? Respond with:
- KEEP: [Explain why in 1-2 sentences]
- MULLIGAN: [Explain why in 1-2 sentences]"""

        try:
            advice = self.ai_advisor.client.generate(mulligan_prompt)

            if advice:
                # Parse thinking vs answer
                thinking, answer = self._parse_reasoning_response(advice)

                # Display thinking in dim color (not spoken)
                if thinking:
                    self._output("💭 Thinking:", "blue")
                    self._output(thinking, "blue")
                    self._output("", "reset")

                # Speak and display the final answer
                final_text = answer if answer else advice
                self._output(f"Advisor: {final_text}", "green")
                clean_advice = self._strip_markdown(final_text)
                self.tts.speak(clean_advice)
            else:
                self._output("Failed to get mulligan advice.", "red")

        except Exception as e:
            logging.error(f"Error generating mulligan advice: {e}")
            self._output(f"Error getting mulligan advice: {e}", "red")

    def _generate_and_speak_advice(self, board_state: BoardState):
        """Generate and speak advice for the current turn"""
        # Validate before sending to LLM
        if not self.game_state_mgr.validate_board_state(board_state):
            logging.warning("Skipping advice generation due to invalid board state")
            self._output(f"\n>>> Turn {board_state.current_turn}: Waiting for complete board state...", "yellow")
            return

        # Display board state
        self._display_board_state(board_state)

        # Update status bar
        self._update_status(board_state)

        self._output(f"\n>>> Turn {board_state.current_turn}: Your move!", "cyan")
        self._output("Getting advice from the master...\n", "cyan")

        advice = self.ai_advisor.get_tactical_advice(board_state)

        if advice:
            # Parse thinking vs answer
            thinking, answer = self._parse_reasoning_response(advice)

            # Display thinking in dim color (not spoken)
            if thinking:
                self._output("💭 Thinking:", "blue")
                # Split thinking into lines for readability
                for line in thinking.split("\n"):
                    if line.strip():
                        self._output(f"   {line.strip()}", "blue")
                self._output("", "white")  # Blank line

            # Display and speak the answer
            if answer:
                self._output(f"Advisor: {answer}\n", "green")
                logging.info(f"ADVICE:\n{answer}")
                # Strip markdown before speaking
                clean_advice = self._strip_markdown(answer)
                self.tts.speak(clean_advice)
            else:
                # If no clear answer, speak the full advice
                self._output(f"Advisor: {advice}\n", "green")
                logging.info(f"ADVICE:\n{advice}")
                clean_advice = self._strip_markdown(advice)
                self.tts.speak(clean_advice)
        else:
            logging.warning("No advice was generated.")

if __name__ == "__main__":
    import argparse
    import dataclasses

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MTGA Voice Advisor - Real-time tactical advice for Magic: The Gathering Arena")
    parser.add_argument("--tui", action="store_true", help="Use Text User Interface (TUI) mode with curses")
    parser.add_argument("--cli", action="store_true", help="Use basic Command Line Interface (CLI) mode")
    args = parser.parse_args()

    # Check and update card database before starting
    if not check_and_update_card_database():
        print("Cannot start without card database.")
        sys.exit(1)

    # Default to GUI unless TUI or CLI specified
    use_gui = not args.tui and not args.cli and TKINTER_AVAILABLE
    use_tui = args.tui

    # Create and run advisor
    advisor = CLIVoiceAdvisor(use_tui=use_tui, use_gui=use_gui)
    advisor.run()
