
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
from collections import deque
import sys

from .mtga import LogFollower, GameStateManager
from .ai import AIAdvisor
from .ui import TextToSpeech, AdvisorGUI
from ..data.data_management import ScryfallClient, CardStatsDB
from ..data.arena_cards import ArenaCardDatabase

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
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
    # Available voices in Kokoro v1.0
    AVAILABLE_VOICES = ["af_alloy", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
                        "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo",
                        "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable",
                        "bm_george", "bm_lewis", "ef_dora", "em_alex", "ff_siwis", "hf_alpha",
                        "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha",
                        "pf_dora", "pm_alex", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]

    BARK_VOICES = []

    def __init__(self, use_gui: bool = True):
        self.use_gui = use_gui
        self.gui = None
        self.tk_root = None
        self.previous_board_state = None

        # Load user preferences
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()

        self.continuous_monitoring = self.prefs.opponent_turn_alerts if self.prefs else True
        self.last_alert_time = 0

        self.log_path = detect_player_log_path()
        if not self.log_path:
            print("ERROR: Could not find Arena Player.log.")
            exit(1)

        # Initialize Arena Card Database (for local card name lookups)
        print("Initializing Arena card database...")
        self.arena_db = ArenaCardDatabase()

        # Initialize Scryfall Client (for AI advisor context - card text, oracle, etc.)
        print("Initializing Scryfall client...")
        self.scryfall = ScryfallClient()
        self.card_stats = CardStatsDB()

        # Initialize GameStateManager with Arena card database
        # ArenaCardDatabase has get_card_name and get_card_data methods
        self.game_state_mgr = GameStateManager(self.arena_db)

        # Initialize AI advisor (Gemini)
        saved_model = self.prefs.current_model if self.prefs else "gemini-1.5-flash"
        self.ai_advisor = AIAdvisor(model=saved_model)

        # Available Gemini models
        self.available_models = [
            "gemini-3-pro-preview",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]

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
                # Pass ScryfallClient and CardStatsDB to DraftAdvisor
                self.draft_advisor = DraftAdvisor(self.scryfall, self.card_stats, self.ai_advisor)
                self.deck_builder = DeckBuilderV2()

                self.game_state_mgr.register_draft_callback("EventGetCoursesV2", self._on_draft_pool)
                self.game_state_mgr.register_draft_callback("LogBusinessEvents", self._on_premier_draft_pick)
                self.game_state_mgr.register_draft_callback("Draft.Notify", self._on_draft_notify)
                self.game_state_mgr.register_draft_callback("BotDraftDraftStatus", self._on_quick_draft_status)
                self.game_state_mgr.register_draft_callback("BotDraftDraftPick", self._on_quick_draft_pick)

                print("[OK] Draft advisor enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize draft advisor: {e}")

        self.last_turn_advised = ""
        self.advice_thread = None
        self.first_turn_detected = False
        self.cli_thread = None
        self.running = True
        self._last_announced_pick = None



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
                print(clean_message.encode('ascii', errors='ignore').decode('ascii'))

    def _update_status(self, board_state: "BoardState" = None):
        """Update status display (GUI)"""
        if board_state:
            status_text = (
                f"Turn {board_state.current_turn} | "
                f"Phase: {board_state.current_phase} | "
                f"Life: {board_state.your_life}/{board_state.opponent_life} | "
                f"Model: {self.ai_advisor.advisor.model_name if hasattr(self.ai_advisor, 'advisor') else 'N/A'}"
            )
        else:
            status_text = f"Waiting for game... | Model: {self.ai_advisor.advisor.model_name if hasattr(self.ai_advisor, 'advisor') else 'N/A'}"

        if self.use_gui and self.gui:
            self.gui.set_status(status_text)
            if board_state:
                # Format and update board state in GUI
                lines = self._format_board_state_for_display(board_state)
                self.gui.set_board_state(lines)

    def _format_board_state_for_display(self, board_state: "BoardState") -> list:
        """Format board state as list of strings for terminal/GUI display"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"TURN {board_state.current_turn} - {board_state.current_phase}")
        lines.append("=" * 70)
        lines.append("")
        
        # Life totals
        lines.append(f"Your Life: {board_state.your_life}  |  Opponent Life: {board_state.opponent_life}")
        lines.append("")
        
        # Helper to format card line
        def format_card(card):
            pt = ""
            if card.power is not None and card.toughness is not None:
                pt = f" ({card.power}/{card.toughness})"
            
            counters = ""
            if card.counters:
                # Format counters like [+1/+1, -1/-1]
                c_list = []
                for c_type, c_count in card.counters.items():
                    if c_count > 0:
                        c_list.append(f"{c_count} {c_type}")
                if c_list:
                    counters = f" [{', '.join(c_list)}]"
            
            return f"  • {card.name}{pt}{counters}"

        # Your hand
        hand_count = len(board_state.your_hand) if board_state.your_hand else board_state.your_hand_count
        lines.append(f"YOUR HAND ({hand_count} cards):")
        if board_state.your_hand:
            for card in board_state.your_hand:
                lines.append(format_card(card))
        else:
            lines.append("  (Hidden or empty)")
        lines.append("")
        
        # Separate battlefield cards into lands and non-lands
        your_lands = []
        your_nonlands = []
        
        for card in board_state.your_battlefield:
            # Heuristic: if it has P/T, it's likely a creature. If it's a basic land, it's a land.
            if card.name in ["Plains", "Island", "Swamp", "Mountain", "Forest"]:
                your_lands.append(card)
            elif card.power is not None:
                your_nonlands.append(card)
            elif "Land" in card.name:
                your_lands.append(card)
            else:
                your_nonlands.append(card)

        lines.append(f"YOUR BATTLEFIELD ({len(your_nonlands)} non-lands):")
        for card in your_nonlands:
            lines.append(format_card(card))
            
        # Group lands
        if your_lands:
            import collections
            land_counts = collections.Counter([c.name for c in your_lands])
            lines.append(f"YOUR LANDS ({len(your_lands)}):")
            for name, count in land_counts.items():
                lines.append(f"  • {count}x {name}")
        lines.append("")

        # Opponent battlefield
        opp_lands = []
        opp_nonlands = []
        for card in board_state.opponent_battlefield:
            if card.name in ["Plains", "Island", "Swamp", "Mountain", "Forest"]:
                opp_lands.append(card)
            elif card.power is not None:
                opp_nonlands.append(card)
            elif "Land" in card.name:
                opp_lands.append(card)
            else:
                opp_nonlands.append(card)

        lines.append(f"OPPONENT'S BATTLEFIELD ({len(opp_nonlands)} non-lands):")
        for card in opp_nonlands:
            lines.append(format_card(card))
            
        if opp_lands:
            import collections
            land_counts = collections.Counter([c.name for c in opp_lands])
            lines.append(f"OPPONENT'S LANDS ({len(opp_lands)}):")
            for name, count in land_counts.items():
                lines.append(f"  • {count}x {name}")
        lines.append("")
        
        # Graveyards
        lines.append(f"YOUR GRAVEYARD ({len(board_state.your_graveyard)} cards):")
        if board_state.your_graveyard:
            for card in board_state.your_graveyard[-5:]:  # Last 5
                lines.append(f"  • {card.name}")
        lines.append("")
        
        lines.append(f"OPPONENT'S GRAVEYARD ({len(board_state.opponent_graveyard)} cards):")
        if board_state.opponent_graveyard:
            for card in board_state.opponent_graveyard[-5:]:  # Last 5
                lines.append(f"  • {card.name}")
        lines.append("")
        
        return lines

    # GUI Callback Methods
    def _on_gui_model_change(self, model):
        model = model.strip()
        if not model: return
        
        # Update AI advisor model
        self.ai_advisor.advisor.model_name = model
        if self.prefs:
            self.prefs.set_model(model)
        
        self._update_status()
        self._output(f"✓ Model changed to: {model}", "green")

    def _on_gui_voice_change(self, voice):
        self.tts.voice = voice
        self._update_status()
        if self.prefs:
            self.prefs.set_voice(voice)

    def _on_gui_volume_change(self, volume):
        self.tts.volume = volume / 100.0
        if self.prefs:
            self.prefs.set_volume(volume)

    # Draft event callbacks (Simplified for brevity, logic remains similar but uses new DraftAdvisor)
    def _on_draft_pool(self, data: dict):
        pass # Pool handling logic if needed

    def _on_draft_notify(self, data: dict):
        if not self.draft_advisor: return
        try:
            pack_num = data.get("PackNumber", 1)
            pick_num = data.get("PickNumber", 1)
            pack_arena_ids = data.get("PackCards", [])
            draft_id = data.get("DraftId", "")

            if not pack_arena_ids: return

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._last_announced_pick = None
                self.draft_advisor.picked_cards = []

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
                self.draft_advisor.picked_cards = []

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
                self.draft_advisor.picked_cards = []

            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            self._display_draft_recommendation(pack_cards, pack_num, pick_num, recommendation)

        except Exception as e:
            logging.error(f"Error handling Quick Draft status: {e}")

    def _on_quick_draft_pick(self, data: dict):
        # Handle user pick to update picked_cards list
        pass

    def _display_draft_recommendation(self, pack_cards, pack_num, pick_num, recommendation):
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

        # TTS
        if self.tts and pack_cards:
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
            
            # Update GUI if it's running
            if self.gui:
                self.gui.update_settings(
                    models=self.available_models,
                    voices=self.AVAILABLE_VOICES,
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
            self.gui = AdvisorGUI(self.tk_root, self)
            
            # Initialize GUI with current settings
            self.gui.update_settings(
                models=self.available_models,
                voices=self.AVAILABLE_VOICES,
                bark_voices=self.BARK_VOICES,
                current_model=self.ai_advisor.advisor.model_name if hasattr(self.ai_advisor, 'advisor') else "gemini-3-pro-preview",
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

    def _follow_log(self):
        """Follow log and process lines"""
        def callback(line):
            # Stream log to GUI if enabled
            # OPTIMIZATION: Only show logs in GUI if we are caught up to live events.
            # This prevents flooding the UI with thousands of historical log lines on startup,
            # which causes the application to freeze/hang.
            if self.use_gui and self.gui and self.log_follower.is_caught_up:
                self.gui.append_log(line)

            # Process with GameStateManager
            self.game_state_mgr.parse_log_line(line)
            
            # Check for tactical advice trigger
            board_state = self.game_state_mgr.get_current_board_state()
            
            # DEBUG: Print trigger status
            if board_state:
                # print(f"DEBUG: Turn={board_state.current_turn}, Phase={board_state.current_phase}, Priority={board_state.has_priority}, LastAdvised={self.last_turn_advised}")
                # Update UI continuously
                self._update_status(board_state)
            else:
                # print("DEBUG: No board state")
                pass
            
            # Trigger advice if player has priority OR is in mulligan phase
            # CRITICAL: Only trigger advice if we are caught up to live events!
            # Otherwise, we will spam advice for every historical turn processed on startup.
            should_trigger = (
                self.log_follower.is_caught_up and 
                ((board_state and board_state.has_priority) or (board_state and board_state.in_mulligan_phase))
            )
            
            if should_trigger:
                # Trigger advice if needed
                current_turn = board_state.current_turn
                current_phase = board_state.current_phase
                
                # Advice key: Turn + Phase (advise once per phase)
                if board_state.in_mulligan_phase:
                     # Use a specific key for mulligan to ensure it triggers even if turn/phase strings are generic
                     # Append hand count to key so it re-triggers if they take a mulligan and get a new hand?
                     # Actually, usually you mulligan once per decision.
                     # Let's just use "Mulligan" combined with hand count to allow advice for each new hand.
                     advice_key = f"Mulligan_{len(board_state.your_hand)}"
                else:
                    advice_key = f"{current_turn}_{current_phase}"
                
                if advice_key != self.last_turn_advised:
                    self.last_turn_advised = advice_key
                    
                    print(f"\n[Thinking] Analyzing board state for Turn {current_turn} - {current_phase}...")
                    
                    # self._update_status(board_state) # Already updated above
                    
                    # Run in thread to not block log parsing
                    def get_advice():
                        try:
                            # Convert BoardState dataclass to dict for AI advisor
                            board_state_dict = dataclasses.asdict(board_state)
                            # Debug: Calling AI Advisor for advice
                            advice = self.ai_advisor.get_tactical_advice(board_state_dict)
                            # Debug: AI Advisor returned advice
                            if advice:
                                self._output(f"\n🤖 Advisor: {advice}", "green")
                                
                                # Strip markdown formatting before TTS (remove ** and other markdown)
                                import re
                                clean_advice = re.sub(r'\*\*([^*]+)\*\*', r'\1', advice)  # Remove **bold**
                                clean_advice = re.sub(r'\*([^*]+)\*', r'\1', clean_advice)  # Remove *italic*
                                clean_advice = re.sub(r'`([^`]+)`', r'\1', clean_advice)  # Remove `code`
                                
                                self.tts.speak(clean_advice)
                        except Exception as e:
                            # Remove emojis from error message for Windows console
                            error_msg = remove_emojis(str(e)) if os.name == 'nt' else str(e)
                            logging.error(f"Error getting advice: {error_msg}")
                            self._output(f"Error: {error_msg}", "red")

                    threading.Thread(target=get_advice, daemon=True).start()

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
