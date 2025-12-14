
import dataclasses
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Callable

import requests

from .mtga import LogFollower, GameStateManager
from .ai import AIAdvisor
from .ui import TextToSpeech, AdvisorGUI
from .formatters import BoardStateFormatter
from .advice_triggers import AdviceTriggerManager, TriggerType, TriggerEvent
from ..data.data_management import CardStatsDB
from ..data.arena_cards import ArenaCardDatabase
from .version import get_version

# Import configuration manager for user preferences
try:
    from ..config.config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")

# Import draft advisor
try:
    from .draft_advisor import DraftAdvisor, display_draft_pack, format_draft_pack_for_gui
    from .deck_builder_v2 import DeckBuilderV2
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

# Configure logging
log_file_path = log_dir / "advisor.log"

# Standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Set console handler to WARNING to hide debug/info noise on the console if desired,
# but basicConfig sets it for root. We can adjust specific handlers if needed.
# For now, keeping it simple.


# ----------------------------------------------------------------------------------
# Performance: Compiled Regex Patterns
# ----------------------------------------------------------------------------------

# Compiled regex for efficient game state change detection
# This runs on EVERY log line, so using compiled regex instead of multiple 'in' checks
# improves performance significantly (single regex search vs. up to 9 substring searches)
GAME_STATE_CHANGE_PATTERN = re.compile(
    r'GameStateMessage|ActionsAvailableReq|turnInfo|priorityPlayer|gameObjects|zones|GameStage_Start|mulligan'
)

# Compiled regex for filtering spammy UI/Hover messages from GUI display
SPAM_FILTER_PATTERN = re.compile(
    r'ClientToGreuimessage|GREMessageType_UIMessage|onHover|ClientToMatchServiceMessageType_ClientToGREUIMessage'
)


# ----------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------

def clean_card_name(name: str) -> str:
    """Remove HTML tags from card names."""
    if not name:
        return name
    return re.sub(r'<[^>]+>', '', name)

def remove_emojis(text: str) -> str:
    """Remove or replace emojis for Windows console compatibility."""
    # Replace common emojis with ASCII equivalents
    replacements = {
        '🤖': '[AI]',
        '🎲': '[DICE]',
        '⚔️': '[ATTACK]',
        '🛡️': '[DEFEND]',
        '✨': '[SPARKLE]',
        '💎': '[GEM]',
        '🏆': '[TROPHY]',
        '⚠️': '[WARNING]',
        '✅': '[OK]',
        '❌': '[X]',
        '⏳': '[WAIT]',
        '📦': '[BOX]',
        '🎮': '[GAME]',
        '🃏': '[CARD]',
        '🔍': '[SEARCH]',
        '⚡': '[EVENT]',
        '💪': '[POWER]',
        '🎴': '[MULLIGAN]',
        '🔮': '[SCRY]',
        '📋': '[DECK]',
        '💀': '[DEATH]',
        '💥': '[DAMAGE]',
        '🐛': '[BUG]',
        '📸': '[SNAP]',
        '📤': '[UPLOAD]',
        '🔄': '[RESTART]',
        '📝': '[NOTE]',
        '═══': '===',
    }

    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)

    # Remove any remaining emojis (Unicode range for emojis)
    import re
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    result = emoji_pattern.sub('', result)

    return result

def detect_player_log_path():
    """Detect the Arena Player.log file based on OS."""
    home = Path.home()
    if os.name == 'nt':
        username = os.getenv('USERNAME')
        drive = os.getenv('USERPROFILE')[0]
        windows_path = f"{drive}:/Users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(windows_path):
            return windows_path
    elif os.name == 'posix' and os.uname().sysname == 'Darwin':
        macos_path = home / "Library/Logs/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(macos_path):
            return str(macos_path)
    elif os.name == 'posix':
        possible_paths = [
            # Windows (WSL)
            Path("/mnt/c/Users/joshu/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"),
            # Linux (Steam)
            Path.home() / ".steam/steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            # Mac
            Path.home() / "Library/Logs/Wizards Of The Coast/MTGA/Player.log",
            # Windows (Default)
            Path.home() / "AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            # Linux (Bottles/Lutris - generic user)
            home / f".var/app/com.usebottles.bottles/data/bottles/bottles/MTG-Arena/drive_c/users/{os.getenv('USER')}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / f"Games/magic-the-gathering-arena/drive_c/users/{os.getenv('USER')}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            # Linux (Steam - older/alternative paths)
            home / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / ".local/share/Steam/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
    return None


class CLIVoiceAdvisor:
    # Available voices in Kokoro v1.0 (American and British only)
    # Voice ID -> Display Name mapping
    VOICE_DISPLAY_NAMES = {
        # American Female
        "af_alloy": "Female Alloy",
        "af_bella": "Female Bella",
        "af_heart": "Female Heart",
        "af_jessica": "Female Jessica",
        "af_kore": "Female Kore",
        "af_nicole": "Female Nicole",
        "af_nova": "Female Nova",
        "af_river": "Female River",
        "af_sarah": "Female Sarah",
        "af_sky": "Female Sky",
        # American Male
        "am_adam": "Male Adam",
        "am_echo": "Male Echo",
        "am_eric": "Male Eric",
        "am_fenrir": "Male Fenrir",
        "am_liam": "Male Liam",
        "am_michael": "Male Michael",
        "am_onyx": "Male Onyx",
        "am_puck": "Male Puck",
        # British Female
        "bf_alice": "Female Alice (UK)",
        "bf_emma": "Female Emma (UK)",
        "bf_isabella": "Female Isabella (UK)",
        "bf_lily": "Female Lily (UK)",
        # British Male
        "bm_daniel": "Male Daniel (UK)",
        "bm_fable": "Male Fable (UK)",
        "bm_george": "Male George (UK)",
        "bm_lewis": "Male Lewis (UK)",
    }

    # List of voice IDs for internal use
    AVAILABLE_VOICES = list(VOICE_DISPLAY_NAMES.keys())

    BARK_VOICES = []

    def __init__(self, use_gui: bool = True):
        self.use_gui = use_gui
        self.gui = None
        self.tk_root = None
        self.previous_board_state = None

        # Voice announcement state tracking
        self._announced_ready = False
        self._announced_catching_up = False
        self._last_announced_game_state = None  # Track: "match", "draft", "deck_building"

        # Available models for the GUI dropdown (updated December 2025)
        self.available_models = [
            # Google Gemini
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            # OpenAI
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            # Anthropic
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            # Local (Ollama)
            "llama3",
            "mistral"
        ]

        # Load user preferences
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()

        self.continuous_monitoring = self.prefs.opponent_turn_alerts if self.prefs else True
        self.last_alert_time = 0

        # PERFORMANCE FIX: Debounce GUI updates to prevent overwhelming the UI
        self._last_gui_update = 0
        self._gui_update_interval = 0.25  # Update GUI max 4x per second (was 0.5)
        self._pending_board_state = None

        self.log_path = detect_player_log_path()
        if not self.log_path:
            print("ERROR: Could not find Arena Player.log.")
            exit(1)

        # Initialize card stats database for 17lands data
        self.card_stats = CardStatsDB()

        # Initialize Arena Card Database (for local card name lookups)
        # Uses unified_cards.db built from MTGA's Raw_CardDatabase
        print("Initializing Arena card database...")
        self.arena_db = ArenaCardDatabase()

        # Initialize board state formatter with card database
        self.formatter = BoardStateFormatter(card_db=self.arena_db)

        # Initialize GameStateManager with Arena card database
        # ArenaCardDatabase has get_card_name and get_card_data methods
        self.game_state_mgr = GameStateManager(self.arena_db)

        # Initialize AI advisor with card database for land detection
        self.ai_advisor = AIAdvisor(card_db=self.arena_db, prefs=self.prefs)

        # Initialize TTS asynchronously to prevent blocking startup
        self.tts = None
        self.tts_loading = True
        threading.Thread(target=self._init_tts_async, daemon=True).start()



        self.log_follower = LogFollower(self.log_path)

        # Initialize draft advisor
        self.draft_advisor = None
        self.deck_builder = None
        if DRAFT_ADVISOR_AVAILABLE:
            try:
                # Pass CardStatsDB and ArenaCardDatabase to DraftAdvisor
                self.draft_advisor = DraftAdvisor(self.card_stats, self.ai_advisor, self.arena_db)
                self.deck_builder = DeckBuilderV2()

                self.game_state_mgr.register_draft_callback("EventGetCoursesV2", self._on_draft_pool)
                self.game_state_mgr.register_draft_callback("LogBusinessEvents", self._on_premier_draft_pick)
                self.game_state_mgr.register_draft_callback("Draft.Notify", self._on_draft_notify)
                self.game_state_mgr.register_draft_callback("BotDraftDraftStatus", self._on_quick_draft_status)
                self.game_state_mgr.register_draft_callback("BotDraftDraftPick", self._on_quick_draft_pick)
                self.game_state_mgr.register_draft_callback("EventPlayerDraftMakePick", self._on_draft_make_pick)
                self.game_state_mgr.register_draft_callback("SceneChange_DraftToDeckBuilder", self._on_deck_builder_entered)

                # NOTE: Draft state recovery moved to _follow_log() to avoid blocking startup
                # self.game_state_mgr.recover_draft_state(self.log_follower)

                # Future: Subscribe to events for decoupled communication
                # Example event subscriptions (replacing direct callbacks):
                # def on_board_state_changed(event: Event):
                #     self._update_ui(event.data)
                # get_event_bus().subscribe(EventType.BOARD_STATE_CHANGED, on_board_state_changed)
                # get_event_bus().subscribe(EventType.DRAFT_PACK_OPENED, self._handle_draft_pack)
                # get_event_bus().subscribe(EventType.ADVICE_READY, self._display_advice)

                print("[OK] Draft advisor enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize draft advisor: {e}")

        # Register game event callbacks (always, not just when draft advisor is available)
        self.game_state_mgr.register_game_callback("match_started", self._on_match_started)

        # Initialize advice trigger manager for smart advice timing
        self.advice_trigger_mgr = AdviceTriggerManager(prefs=self.prefs)

        self.last_turn_advised = ""
        self.advice_thread = None
        self.first_turn_detected = False
        self.cli_thread = None
        self.running = True
        self._last_announced_pick = None
        self._current_board_state = None  # Cache for manual advice requests
        self.last_advice_packet = None   # Cache for feedback mechanism
        
        self.version = get_version()
        print(f"MTGA Voice Advisor v{self.version}")



    def _output(self, message: str, color: str = "white"):
        if self.use_gui and self.gui:
            self.gui.add_message(message, color)
        else:
            # Remove emojis for Windows console compatibility
            clean_message = remove_emojis(message) if os.name == 'nt' else message
            # Encode with error handling for Windows console
            try:
                print(clean_message)
            except UnicodeEncodeError:
                # Fallback: print with replacement characters using utf-8 logic
                try:
                    print(clean_message.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
                except:
                    # Ultimate fallback
                    print(clean_message.encode('ascii', errors='ignore').decode('ascii'))

    def _announce(self, message: str):
        """Speak a voice announcement (non-blocking)."""
        if self.tts and not self.tts_loading:
            threading.Thread(target=lambda: self.tts.speak(message), daemon=True).start()

    def _maybe_update_gui(self):
        """Debounced GUI update - max 2x per second to prevent performance issues"""
        if not self._pending_board_state:
            return

        current_time = time.time()
        if current_time - self._last_gui_update > self._gui_update_interval:
            self._update_status(self._pending_board_state)
            self._last_gui_update = current_time
            self._pending_board_state = None

    def _update_status(self, board_state: "BoardState" = None):
        """Update status display (GUI)"""
        model_name = self.ai_advisor.advisor.model_name if self.ai_advisor.advisor else 'N/A'
        
        if board_state:
            status_text = (
                f"Turn {board_state.current_turn} | "
                f"Phase: {board_state.current_phase} | "
                f"Life: {board_state.your_life}/{board_state.opponent_life} | "
                f"Model: {model_name}"
            )
        elif not self.log_follower.is_caught_up:
            status_text = f"Catching up logs... | Model: {model_name}"
        else:
            status_text = f"Ready - Waiting for match... | Model: {model_name}"

        if self.use_gui and self.gui:
            self.gui.set_status(status_text)
            if board_state:
                # Format and update board state in GUI
                lines = self._format_board_state_for_display(board_state)
                self.gui.set_board_state(lines)
                
                # Update Library/Deck Window
                if board_state.your_decklist:
                    deck_lines = []
                    total_cards = sum(board_state.your_decklist.values())
                    deck_lines.append(f"LIBRARY REMAINING: {board_state.your_library_count}")
                    deck_lines.append("=" * 40)
                    
                    # Calculate approximate probabilities if library count > 0
                    remaining_lib = board_state.your_library_count
                    
                    # Sort by name
                    sorted_deck = sorted(board_state.your_decklist.items())
                    
                    for card_name, count in sorted_deck:
                        # Simple heuristic: subtract copies seen in other zones to estimate remaining
                        # This is imperfect without strict card-counting logic (e.g. tracking every drawn card)
                        # But GameStateManager should ideally track "cards_seen" or similar.
                        # For now, we just list the deck composition.
                        # TODO: Implement accurate "remaining in deck" tracking in GameStateManager
                        
                        prob_str = ""
                        if remaining_lib > 0:
                            # Rough probability based on original count (inaccurate as game progresses but better than nothing)
                            pct = (count / remaining_lib) * 100
                            prob_str = f" ({pct:.1f}%)"
                            
                        deck_lines.append(f"{count}x {card_name}{prob_str}")
                        
                    self.gui.set_deck_content(deck_lines)
                else:
                    self.gui.set_deck_content(["Waiting for deck data...", "(Deck list not yet parsed)"])

    def _format_board_state_for_display(self, board_state: "BoardState") -> list:
        """
        Format board state as list of strings for terminal/GUI display.

        Delegates to BoardStateFormatter for actual formatting logic.
        """
        return self.formatter.format_for_display(board_state)

    # GUI Callback Methods
    def change_ai_model(self, provider: str, model: str):
        """
        Hot-swap the AI model and provider.
        Called from GUI when settings change.
        """
        model = model.strip()
        provider = provider.strip()
        if not model or not provider: return
        
        # Get API key/config from prefs based on provider
        api_key = None
        extra_config = {}
        if self.prefs:
            if provider.lower() == "google":
                api_key = self.prefs.google_api_key
            elif provider.lower() == "openai":
                api_key = self.prefs.openai_api_key
            elif provider.lower() == "anthropic":
                api_key = self.prefs.anthropic_api_key
            elif provider.lower() == "llama.cpp":
                extra_config["server_url"] = self.prefs.llamacpp_server_url

        # Attempt hot-swap
        success = self.ai_advisor.set_model(provider, model, api_key, **extra_config)
        
        if success:
            if self.prefs:
                self.prefs.set_model(model, provider=provider)
            self._update_status()
            self._output(f"✓ AI Hot-swapped to: {provider} / {model}", "green")
        else:
            self._output(f"✗ Failed to swap to: {provider} / {model}", "red")

    def _on_gui_voice_change(self, voice):
        self.tts.voice = voice
        self._update_status()
        if self.prefs:
            self.prefs.set_voice(voice)

    def _on_gui_volume_change(self, volume):
        self.tts.volume = volume / 100.0
        if self.prefs:
            self.prefs.set_volume(volume)



    # Game event callbacks
    def _on_match_started(self, data: dict):
        """Handle match started event - announce via voice."""
        # Only announce if caught up (not processing old logs)
        if self.log_follower.is_caught_up and self._last_announced_game_state != "match":
            self._last_announced_game_state = "match"
            self._announce("Match started")
            logging.info("Voice announcement: Match started")

    # Draft event callbacks (Simplified for brevity, logic remains similar but uses new DraftAdvisor)
    def _on_draft_pool(self, data: dict):
        pass # Pool handling logic if needed

    def _on_draft_notify(self, data: dict):
        """Handle Draft.Notify event - show pick recommendations"""
        logging.info(f"_on_draft_notify callback triggered with data keys: {data.keys()}")
        if not self.draft_advisor: return
        try:
            pack_num = data.get("PackNumber", 1)
            pick_num = data.get("PickNumber", 1)
            pack_arena_ids = data.get("PackCards", [])
            draft_id = data.get("DraftId", "")
            logging.info(f"Processing draft notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            if not pack_arena_ids: return

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._last_announced_pick = None
                self.draft_advisor.reset_draft()
                # Announce draft start (only if caught up)
                if self.log_follower.is_caught_up and self._last_announced_game_state != "draft":
                    self._last_announced_game_state = "draft"
                    self._announce("Draft started")
                    logging.info("Voice announcement: Draft started")

            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, draft_id
            )

            self._display_draft_recommendation(pack_cards, pack_num, pick_num, recommendation)
            
        except Exception as e:
            logging.error(f"Error handling Draft.Notify: {e}")

    def _on_premier_draft_pick(self, data: dict):
        if not self.draft_advisor: return
        try:
            pack_num = data.get("PackNumber", 0) + 1
            pick_num = data.get("PickNumber", 0) + 1
            cards_in_pack = data.get("CardsInPack", [])
            event_id = data.get("EventId", "")

            if not cards_in_pack: return
            pack_arena_ids = [int(card_id) for card_id in cards_in_pack]

            if pack_num == 1 and pick_num == 1:
                self._last_announced_pick = None
                self.draft_advisor.reset_draft()

            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_id
            )

            self._display_draft_recommendation(pack_cards, pack_num, pick_num, recommendation)

        except Exception as e:
            logging.error(f"Error handling Premier Draft pick: {e}")

    def _on_quick_draft_status(self, data: dict):
        if not self.draft_advisor: return
        try:
            pack_num = data.get("PackNumber", 0) + 1
            pick_num = data.get("PickNumber", 0) + 1
            draft_pack = data.get("DraftPack", [])
            event_name = data.get("EventName", "")

            if not draft_pack: return
            pack_arena_ids = [int(card_id) for card_id in draft_pack]

            if pack_num == 1 and pick_num == 1:
                self._last_announced_pick = None
                self.draft_advisor.reset_draft()

            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            self._display_draft_recommendation(pack_cards, pack_num, pick_num, recommendation)

        except Exception as e:
            logging.error(f"Error handling Quick Draft status: {e}")

    def _on_quick_draft_pick(self, data: dict):
        # Handle user pick to update picked_cards list
        pass

    def _on_draft_make_pick(self, data: dict):
        """Handle EventPlayerDraftMakePick - track the card(s) the user picked"""
        if not self.draft_advisor:
            return
        try:
            grp_ids = data.get("GrpIds", [])
            pack_num = data.get("Pack", 0)
            pick_num = data.get("Pick", 0)

            if grp_ids and self.arena_db:
                for grp_id in grp_ids:
                    card_data = self.arena_db.get_card_data(grp_id)
                    if card_data:
                        card_name = card_data.get("name", f"Card {grp_id}")
                        card_colors = card_data.get("colors", "")
                        # Use record_pick to track both name and colors
                        self.draft_advisor.record_pick(card_name, card_colors)
                        logging.info(f"Tracked pick: {card_name} [{card_colors}] (Pack {pack_num}, Pick {pick_num})")
                        self._output(f"Picked: {card_name}", "green")
                    else:
                        logging.warning(f"Could not resolve picked card ID: {grp_id}")

        except Exception as e:
            logging.error(f"Error tracking draft pick: {e}")

    def _on_deck_builder_entered(self, data: dict):
        """Handle transition from Draft to DeckBuilder - suggest a deck"""
        # Announce deck building phase (only if caught up)
        if self.log_follower.is_caught_up and self._last_announced_game_state != "deck_building":
            self._last_announced_game_state = "deck_building"
            self._announce("Deck building")
            logging.info("Voice announcement: Deck building")

        if not self.draft_advisor or not self.deck_builder:
            return

        try:
            picked_cards = self.draft_advisor.picked_cards
            if not picked_cards:
                self._output("No picked cards tracked - cannot suggest deck", "yellow")
                return

            self._output(f"Draft complete! Building deck from {len(picked_cards)} cards...", "cyan")

            # Get set code from draft advisor
            set_code = self.draft_advisor.current_set or "TLA"

            # Get deck suggestions
            suggestions = self.deck_builder.suggest_deck(picked_cards, set_code, top_n=3)

            if suggestions:
                self._display_deck_suggestions(suggestions, picked_cards)
            else:
                self._output("Could not generate deck suggestions (no card stats found)", "yellow")

        except Exception as e:
            logging.error(f"Error generating deck suggestions: {e}")
            self._output(f"Error building deck: {e}", "red")

    def _display_deck_suggestions(self, suggestions, picked_cards):
        """Display deck building suggestions"""
        self._output("=" * 60, "white")
        self._output("DECK BUILDING SUGGESTIONS", "cyan")
        self._output("=" * 60, "white")

        for i, suggestion in enumerate(suggestions, 1):
            self._output(f"\nOption {i}: {suggestion.color_pair_name} {suggestion.archetype}", "green")
            self._output(f"  Average GIHWR: {suggestion.avg_gihwr*100:.1f}%", "white")
            self._output(f"  Score: {suggestion.score:.2f}", "white")

            # Show maindeck (top cards)
            self._output(f"\n  Maindeck ({sum(suggestion.maindeck.values())} cards):", "white")
            sorted_cards = sorted(suggestion.maindeck.items(), key=lambda x: -x[1])
            for card_name, count in sorted_cards[:10]:
                self._output(f"    {count}x {card_name}", "white")
            if len(sorted_cards) > 10:
                self._output(f"    ... and {len(sorted_cards) - 10} more cards", "white")

            # Show lands
            self._output(f"\n  Lands ({sum(suggestion.lands.values())}):", "white")
            for land, count in suggestion.lands.items():
                self._output(f"    {count}x {land}", "white")

        # TTS announcement
        if self.tts and suggestions:
            best = suggestions[0]
            self.tts.speak(f"Recommended: {best.color_pair_name} {best.archetype}")

    def _display_draft_recommendation(self, pack_cards, pack_num, pick_num, recommendation):
        """Display draft pick recommendation to user"""
        logging.info(f"Displaying draft recommendation: Pack {pack_num}, Pick {pick_num} - {len(pack_cards)} cards, rec: {recommendation}")

        # Display logic for GUI/CLI
        if self.use_gui and self.gui:
            from .draft_advisor import format_draft_pack_for_gui
            pack_lines, picked_lines = format_draft_pack_for_gui(
                pack_cards, pack_num, pick_num, recommendation,
                picked_cards=self.draft_advisor.picked_cards,
                split_panes=True
            )
            self.gui.set_draft_panes(pack_lines, picked_lines, len(self.draft_advisor.picked_cards), 45)
            self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
        else:
            from .draft_advisor import display_draft_pack
            display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

        # TTS - only announce if caught up to live events (avoid spam on startup)
        if self.tts and pack_cards and self.log_follower.is_caught_up:
            current_pick = (pack_num, pick_num)
            if current_pick != self._last_announced_pick:
                self._last_announced_pick = current_pick
                self.tts.speak(f"Pick {pack_cards[0].name}")

    def _init_tts_async(self):
        """Initialize TTS in a background thread."""
        try:
            saved_voice = self.prefs.current_voice if self.prefs else "am_adam"
            saved_volume = (self.prefs.volume if self.prefs else 100) / 100.0
            print(f"Initializing TTS engine (background)...")
            self.tts = TextToSpeech(voice=saved_voice, volume=saved_volume)
            self.tts_loading = False
            print(f"[OK] TTS engine ready")

            # Announce ready (only once)
            if not self._announced_ready:
                self._announced_ready = True
                self.tts.speak("Ready")

            # Update GUI if it's running
            if self.gui:
                self.gui.update_settings(
                    models=self.available_models,
                    voices=self.AVAILABLE_VOICES,
                    voice_display_names=self.VOICE_DISPLAY_NAMES,
                    bark_voices=self.BARK_VOICES,
                    current_model=self.prefs.current_model if self.prefs else "gemini-1.5-flash",
                    current_voice=saved_voice,
                    volume=int(saved_volume * 100),
                    tts_engine=self.tts.tts_engine
                )
        except Exception as e:
            logging.error(f"Failed to initialize TTS: {e}")
            self.tts = None
            self.tts_loading = False

    def run(self):
        """Main application loop"""
        if self.use_gui:
            self.tk_root = tk.Tk()
            self.gui = AdvisorGUI(self.tk_root, self, version=self.version)

            # Set up unknown card warning callback
            if self.arena_db:
                self.arena_db.set_unknown_card_callback(
                    lambda count: self.gui.show_unknown_cards_warning(count)
                )

            # Set up push-to-talk callback for manual advice
            self.gui.set_push_to_talk_callback(self._on_push_to_talk)

            # Initialize GUI with current settings
            self.gui.update_settings(
                models=self.available_models,
                voices=self.AVAILABLE_VOICES,
                voice_display_names=self.VOICE_DISPLAY_NAMES,
                bark_voices=self.BARK_VOICES,
                current_model=self.ai_advisor.advisor.model_name if self.ai_advisor.advisor else "gemini-3-pro-preview",
                current_voice=self.tts.voice if self.tts else (self.prefs.current_voice if self.prefs else "am_adam"),
                volume=int(self.tts.volume * 100) if self.tts else (self.prefs.volume if self.prefs else 80),
                tts_engine=self.tts.tts_engine if self.tts else "kokoro"
            )

            # Set initial status
            self._update_status()

            # Start log following in a separate thread
            log_thread = threading.Thread(target=self._follow_log, daemon=True)
            log_thread.start()

            self.tk_root.mainloop()
        else:
            # CLI mode
            print(f"Listening to log: {self.log_path}")
            print("Press Ctrl+C to exit")
            try:
                self._follow_log()
            except KeyboardInterrupt:
                print("\nExiting...")

    def _on_push_to_talk(self, user_query: str = None):
        """Handle push-to-talk request for manual advice."""
        logging.info(f"Push-to-talk requested (query: {user_query})")

        # Get current board state
        board_state = self._current_board_state
        if not board_state:
            # Try to get current state from game manager
            try:
                board_state = self.game_state_mgr.get_current_board_state()
            except Exception as e:
                logging.warning(f"Could not get board state: {e}")

        if not board_state:
            self._output("⚠ No game state available. Start or join a match first.", "yellow")
            return

        # Create manual trigger event
        trigger_event = self.advice_trigger_mgr.trigger_manual(user_query)

        # Get advice
        self._get_advice_for_trigger(board_state, trigger_event)

    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS pronunciation."""
        import re
        clean_advice = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean_advice = re.sub(r'\*([^*]+)\*', r'\1', clean_advice)
        clean_advice = re.sub(r'`([^`]+)`', r'\1', clean_advice)
        clean_advice = re.sub(r'^(Recommendation|Advice|Suggestion|Note|Tip):\s*', '', clean_advice, flags=re.IGNORECASE)
        # Remove markdown headers
        clean_advice = re.sub(r'#+\s', '', clean_advice)
        return clean_advice.strip()

    def _get_advice_for_trigger(self, board_state, trigger_event: TriggerEvent):
        """Get and output advice for a trigger event using streaming."""
        def get_advice():
            try:
                logging.info(f"Getting advice for trigger: {trigger_event.trigger_type.name}")
                
                # Show thinking indicator in UI
                if self.use_gui and self.gui:
                    self.gui.show_thinking()

                # Convert BoardState dataclass to dict for AI advisor
                board_state_dict = dataclasses.asdict(board_state)

                # Add trigger context to help AI understand the situation
                if trigger_event.user_query:
                    board_state_dict['user_query'] = trigger_event.user_query
                board_state_dict['trigger_type'] = trigger_event.trigger_type.name
                
                # Streaming Output Setup
                advice_full_text = ""
                sentence_buffer = ""
                verbose_speech = True
                if self.gui and hasattr(self.gui, 'verbose_speech_var'):
                    verbose_speech = self.gui.verbose_speech_var.get()
                
                is_first_sentence = True
                
                # Start the stream
                stream = self.ai_advisor.get_tactical_advice_stream(board_state_dict)
                
                import re
                
                for chunk in stream:
                    advice_full_text += chunk
                    sentence_buffer += chunk

                    # Regex to split sentences (including delimiter)
                    # Split by .!? followed by space, newline, or end of string
                    if re.search(r'[.!?](\s+|$)', sentence_buffer):
                        parts = re.split(r'([.!?](?:\s+|$))', sentence_buffer)
                        
                        # Process complete sentences (parts[0::2] are text, parts[1::2] are delimiters)
                        # We need pairs.
                        while len(parts) > 2:
                            sentence = parts.pop(0) + parts.pop(0)
                            
                            clean_sent = self._clean_for_tts(sentence)
                            if clean_sent.strip():
                                # Speak sentence
                                if verbose_speech or is_first_sentence:
                                    if self.tts:
                                        self.tts.speak(clean_sent)
                                    is_first_sentence = False
                                    
                                # Print to console if in CLI mode
                                if not self.use_gui:
                                    print(sentence, end="", flush=True)

                        # Reassemble remaining buffer
                        sentence_buffer = "".join(parts)

                # Handle remaining buffer
                if sentence_buffer.strip():
                     clean_sent = self._clean_for_tts(sentence_buffer)
                     if clean_sent and (verbose_speech or is_first_sentence):
                        if self.tts:
                             self.tts.speak(clean_sent)
                     if not self.use_gui:
                         print(sentence_buffer, flush=True)
                
                if not self.use_gui:
                    print() # Newline after stream

                if advice_full_text:
                    if self.use_gui and self.gui:
                        self.gui.add_message(f"🤖 Advisor: {advice_full_text}", "green")
                    elif not self.use_gui:
                        # Already printed to console
                        pass

                    # Capture context for feedback
                    self.last_advice_packet = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "board_state": board_state_dict,
                        "advice": advice_full_text,
                        "trigger_type": trigger_event.trigger_type.name,
                        "model": self.ai_advisor.advisor.model_name if self.ai_advisor.advisor else "N/A"
                    }

            except Exception as e:
                error_msg = remove_emojis(str(e)) if os.name == 'nt' else str(e)
                logging.error(f"Error getting advice: {error_msg}")
                self._output(f"Error: {error_msg}", "red")
            finally:
                # Hide thinking indicator
                if self.use_gui and self.gui:
                    self.gui.hide_thinking()

        threading.Thread(target=get_advice, daemon=True).start()

    def _follow_log(self):
        """Follow log and process lines"""
        # Recover draft state if app is started mid-draft (moved here from __init__ to avoid blocking startup)
        if DRAFT_ADVISOR_AVAILABLE and self.draft_advisor:
            try:
                self.game_state_mgr.recover_draft_state(self.log_follower)
                
                # Fix for Bug #52: Check if the recovered draft event is fresh
                # If the log contains an old draft (e.g., from yesterday), we don't want to shout advice on startup
                last_ts = self.game_state_mgr.scanner.last_event_timestamp
                if last_ts:
                    now = datetime.datetime.now()
                    diff_hours = (now - last_ts).total_seconds() / 3600
                    if diff_hours > 2.0:
                        logging.info(f"Recovered draft state is stale ({diff_hours:.1f} hours old). Ignoring.")
                        self.draft_advisor.reset_draft()
            except Exception as e:
                logging.warning(f"Failed to recover draft state: {e}")

        def callback(line):
            # Announce "catching up" once when we detect we're processing old logs
            if not self.log_follower.is_caught_up and not self._announced_catching_up:
                self._announced_catching_up = True
                self._announce("Catching up")
                logging.info("Voice announcement: Catching up (processing historical logs)")

            # Stream log to GUI if enabled
            # OPTIMIZATION: Only show logs in GUI if we are caught up to live events.
            # This prevents flooding the UI with thousands of historical log lines on startup,
            # which causes the application to freeze/hang.
            if self.use_gui and self.gui and self.log_follower.is_caught_up:
                # Filter out spammy UI/Hover messages from the GUI display to reduce noise
                if not SPAM_FILTER_PATTERN.search(line):
                    self.gui.append_log(line)

            # Process with GameStateManager
            self.game_state_mgr.parse_log_line(line)

            # CRITICAL OPTIMIZATION:
            # Do not recalculate board state or update UI if we are not caught up with the log.
            # Doing so for every historical line causes massive startup delays and freezing.
            if not self.log_follower.is_caught_up:
                return

            # Enable unknown card tracking once we're caught up
            # This prevents false warnings from old log entries before a database update
            if self.arena_db and not getattr(self, '_unknown_tracking_enabled', False):
                self._unknown_tracking_enabled = True
                self.arena_db.enable_tracking()

            # PERFORMANCE FIX: Only get board state when game state changes
            # This prevents rebuilding the board state thousands of times per second
            # Check if line contains game state indicators using compiled regex for efficiency
            has_game_state_change = GAME_STATE_CHANGE_PATTERN.search(line) is not None

            if has_game_state_change:
                # Get the board state for display and advice checking
                board_state = self.game_state_mgr.get_current_board_state()

                # Queue debounced GUI update
                if board_state:
                    self._pending_board_state = board_state
                    self._maybe_update_gui()
            else:
                # No game state change - skip board state calculation entirely
                board_state = None

            # Cache board state for manual (push-to-talk) advice requests
            if board_state:
                self._current_board_state = board_state

            # Use AdviceTriggerManager for smart advice timing
            # CRITICAL: Only trigger advice if we are caught up to live events!
            if board_state and self.log_follower.is_caught_up:
                # Check for event freshness to avoid advising on old log data
                is_fresh = True
                if self.game_state_mgr.scanner.last_event_timestamp:
                    now = datetime.datetime.now()
                    diff = (now - self.game_state_mgr.scanner.last_event_timestamp).total_seconds()
                    if diff > 30:
                        logging.debug(f"Skipping advice for stale event (delay: {diff:.1f}s)")
                        is_fresh = False

                if is_fresh:
                    # Check if any automatic trigger should fire
                    trigger_event = self.advice_trigger_mgr.check_triggers(board_state)

                    if trigger_event:
                        current_turn = board_state.current_turn
                        current_phase = board_state.current_phase
                        print(f"\n[{trigger_event.trigger_type.name}] Analyzing Turn {current_turn} - {current_phase}...")
                        self._get_advice_for_trigger(board_state, trigger_event)

        self.log_follower.follow(callback)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTGA Voice Advisor")
    parser.add_argument("--cli", action="store_true", help="Use Command Line Interface (Headless)")
    args = parser.parse_args()

    # Default to GUI unless CLI is requested
    use_gui = not args.cli

    app = CLIVoiceAdvisor(use_gui=use_gui)
    app.run()

if __name__ == "__main__":
    main()
