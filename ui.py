
import logging
import os
import subprocess
import tempfile
from pathlib import Path
import curses
import time
from collections import deque
from typing import List, Callable

# Content of src/tts.py
class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0, force_engine: str = None):
        """
        Initialize TTS with Kokoro as primary, BarkTTS as fallback.

        Args:
            voice: Voice name
            volume: Volume (0.0-1.0)
            force_engine: Force specific engine ("kokoro" or "bark"), or None for auto-fallback
        """
        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))  # Clamp volume to 0.0-1.0
        self.tts_engine = None  # Will be "kokoro" or "bark"
        self.tts = None
        self.bark_processor = None
        self.bark_model = None

        if force_engine == "bark":
            # Force BarkTTS
            logging.info(f"Forcing BarkTTS engine")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return
            logging.error("❌ Failed to initialize BarkTTS")
        elif force_engine == "kokoro":
            # Force Kokoro
            logging.info(f"Forcing Kokoro engine with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return
            logging.error("❌ Failed to initialize Kokoro TTS")
        else:
            # Try Kokoro first (primary), then fall back
            logging.info(f"Attempting to initialize Kokoro TTS (primary) with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return

            # Fall back to BarkTTS
            logging.warning("Kokoro TTS failed, falling back to BarkTTS (secondary)")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return

            # No TTS available
            logging.error("❌ Failed to initialize any TTS engine (Kokoro and Bark both failed)")

    def _init_kokoro(self) -> bool:
        """Try to initialize Kokoro TTS. Returns True on success."""
        try:
            from kokoro_onnx import Kokoro
            import numpy as np
            from pathlib import Path
            self.np = np

            # Use downloaded models from ~/.local/share/kokoro/
            models_dir = Path.home() / '.local' / 'share' / 'kokoro'
            model_path = str(models_dir / 'kokoro-v1.0.onnx')
            voices_path = str(models_dir / 'voices-v1.0.bin')

            self.tts = Kokoro(model_path=model_path, voices_path=voices_path)
            self.tts_engine = "kokoro"
            return True
        except Exception as e:
            logging.debug(f"Kokoro initialization failed: {e}")
            return False

    def _init_bark(self) -> bool:
        """Try to initialize BarkTTS. Returns True on success."""
        try:
            from transformers import AutoProcessor, BarkModel
            import numpy as np
            import torch

            self.np = np
            self.torch = torch

            # Load Bark model and processor
            logging.info("Loading BarkTTS model (this may take a moment)...")
            self.bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.bark_model = BarkModel.from_pretrained("suno/bark-small")

            # Move to GPU if available
            if torch.cuda.is_available():
                self.bark_model = self.bark_model.to("cuda")
                logging.info("BarkTTS using GPU acceleration")

            self.tts_engine = "bark"
            return True
        except Exception as e:
            logging.debug(f"BarkTTS initialization failed: {e}")
            return False

    def set_voice(self, voice: str):
        """Change voice dynamically"""
        self.voice = voice
        logging.info(f"Voice changed to: {voice}")

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        logging.info(f"Volume changed to: {self.volume}")

    def speak(self, text: str):
        """Speak text using available TTS engine (Kokoro or Bark)"""
        if not text:
            logging.debug("No text provided to speak.")
            return

        if not self.tts_engine:
            logging.error("No TTS engine initialized, cannot speak.")
            return

        # Route to appropriate TTS engine
        if self.tts_engine == "kokoro":
            self._speak_kokoro(text)
        elif self.tts_engine == "bark":
            self._speak_bark(text)

    def _speak_kokoro(self, text: str):
        """Speak using Kokoro TTS"""
        logging.info(f"Speaking with Kokoro ({self.voice}): {text[:100]}...")
        try:
            # Generate audio using Kokoro
            audio_array, sample_rate = self.tts.create(text, voice=self.voice, speed=1.0)

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "Kokoro")
            logging.debug("Successfully spoke text with Kokoro.")
        except Exception as e:
            logging.error(f"Kokoro TTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _speak_bark(self, text: str):
        """Speak using BarkTTS"""
        logging.info(f"Speaking with BarkTTS ({self.voice}): {text[:100]}...")
        try:
            # Process text input
            inputs = self.bark_processor(text, voice_preset=self.voice)

            # Move inputs to same device as model
            if self.torch.cuda.is_available():
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate audio
            with self.torch.no_grad():
                audio_array = self.bark_model.generate(**inputs)

            # Convert to numpy and get sample rate
            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = self.bark_model.generation_config.sample_rate

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "BarkTTS")
            logging.debug("Successfully spoke text with BarkTTS.")
        except Exception as e:
            logging.error(f"BarkTTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _save_and_play_audio(self, audio_array, sample_rate: int, engine_name: str):
        """Save audio to temp file and play it"""
        import scipy.io.wavfile as wavfile

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, sample_rate, (audio_array * 32767).astype(self.np.int16))

        logging.info(f"Generated audio saved to {tmp_path}, playing...")

        # Try different audio players
        played = False
        players = [
            (["aplay", tmp_path], "aplay"),
            (["paplay", tmp_path], "paplay"),
            (["ffplay", "-nodisp", "-autoexit", tmp_path], "ffplay")
        ]

        for cmd, player_name in players:
            try:
                subprocess.run(cmd, check=True, timeout=120,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                played = True
                logging.info(f"Audio played with {player_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.debug(f"{player_name} error: {e}")
                continue

        if not played:
            logging.error("No audio player found (aplay, paplay, or ffplay). Cannot play audio.")

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

# Content of src/ui.py
# Import Tkinter (optional - for GUI mode)
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available. GUI mode disabled.")

# Import config manager (optional)
try:
    from config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")

class AdvisorTUI:
    """
    Text User Interface for MTGA Voice Advisor using curses.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Status Bar: Turn 5 | Model: llama3.2 | Voice: am_adam     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ Board State Window (scrollable)                            │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ Messages Window (scrollable)                               │
    │ - Advisor responses                                        │
    │ - Game events                                              │
    │ - Command feedback                                         │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │ You: _                                                      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.board_state_lines = []
        self.messages = deque(maxlen=100)  # Keep last 100 messages
        self.input_buffer = ""
        self.input_callback = None

        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Green
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Cyan
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Yellow
        curses.init_pair(4, curses.COLOR_RED, -1)     # Red
        curses.init_pair(5, curses.COLOR_BLUE, -1)    # Blue
        curses.init_pair(6, curses.COLOR_WHITE, -1)   # White

        # Color name to pair ID mapping
        self.color_map = {
            "green": 1,
            "cyan": 2,
            "yellow": 3,
            "red": 4,
            "blue": 5,
            "white": 6,
        }

        # Configure stdscr
        self.stdscr.keypad(True)
        curses.curs_set(1)  # Show cursor

        # Create windows
        self._create_windows()

    def _create_windows(self):
        """Create and layout windows"""
        height, width = self.stdscr.getmaxyx()

        # Status bar: 1 line at top
        self.status_win = curses.newwin(1, width, 0, 0)

        # Board state: 70% of available height (more space for full board display)
        available_height = height - 3  # Minus status, separator, input
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

        # Scroll positions
        self.board_scroll = 0
        self.msg_scroll = 0

    def resize(self):
        """Handle terminal resize"""
        try:
            # Update curses internal tracking of terminal size
            curses.update_lines_cols()

            # Clear and refresh main screen
            self.stdscr.clear()
            self.stdscr.refresh()

            # Recreate windows with new dimensions
            self._create_windows()

            # Redraw everything
            self.refresh_all()
        except Exception as e:
            # Silently handle resize errors
            pass

    def set_status(self, text: str):
        """Update status bar"""
        try:
            self.status_win.clear()
            self.status_win.addstr(0, 0, text[:self.status_win.getmaxyx()[1]-1],
                                  curses.color_pair(1) | curses.A_BOLD)
            self.status_win.refresh()
        except curses.error:
            pass

    def set_board_state(self, lines: List[str]):
        """Update board state display"""
        self.board_state_lines = lines
        self._refresh_board()

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

    def _refresh_board(self):
        """Redraw board state window"""
        try:
            self.board_win.clear()
            height, width = self.board_win.getmaxyx()

            # Draw border
            try:
                self.board_win.addstr(0, 0, "═" * (width-1), curses.color_pair(2))
                self.board_win.addstr(0, 2, " BOARD STATE ", curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass

            # Draw visible lines
            visible_lines = self.board_state_lines[self.board_scroll:self.board_scroll + height - 1]
            for i, line in enumerate(visible_lines):
                try:
                    # Truncate to fit width
                    display_line = line[:width-1]
                    self.board_win.addstr(i + 1, 0, display_line)
                except curses.error:
                    pass

            self.board_win.refresh()
        except curses.error:
            pass

    def _refresh_messages(self):
        """Redraw messages window"""
        try:
            self.msg_win.clear()
            height, width = self.msg_win.getmaxyx()

            # Draw border
            try:
                self.msg_win.addstr(0, 0, "═" * (width-1), curses.color_pair(2))
                self.msg_win.addstr(0, 2, " MESSAGES ", curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass

            # Draw visible messages
            visible_msgs = list(self.messages)[self.msg_scroll:self.msg_scroll + height - 1]
            for i, (timestamp, msg, color) in enumerate(visible_msgs):
                try:
                    # Format: [HH:MM:SS] message
                    display_line = f"[{timestamp}] {msg}"[:width-1]
                    attr = curses.color_pair(color) if color else 0
                    self.msg_win.addstr(i + 1, 0, display_line, attr)
                except curses.error:
                    pass

            self.msg_win.refresh()
        except curses.error:
            pass

    def _refresh_input(self):
        """Redraw input prompt"""
        try:
            self.input_win.clear()
            width = self.input_win.getmaxyx()[1]

            # Show prompt and input buffer
            prompt = "You: "
            display = prompt + self.input_buffer

            # Truncate if too long
            if len(display) >= width:
                display = prompt + "..." + self.input_buffer[-(width-len(prompt)-4):]

            self.input_win.addstr(0, 0, display)
            self.input_win.refresh()
        except curses.error:
            pass

    def refresh_all(self):
        """Refresh all windows"""
        try:
            self.status_win.refresh()
        except:
            pass
        self._refresh_board()
        self._refresh_messages()
        self._refresh_input()

    def get_input(self, callback: Callable[[str], None]):
        """
        Get user input (non-blocking with callback).
        Call this in a loop to handle input.
        """
        self.input_callback = callback
        self._refresh_input()

        try:
            # Non-blocking input
            self.stdscr.timeout(100)  # 100ms timeout
            ch = self.stdscr.getch()

            if ch == -1:  # No input
                return True

            if ch == curses.KEY_RESIZE:
                self.resize()
            elif ch == ord('\n') or ch == curses.KEY_ENTER or ch == 10:
                # Enter key - submit input
                if self.input_buffer.strip():
                    user_input = self.input_buffer
                    self.input_buffer = ""
                    self._refresh_input()
                    if self.input_callback:
                        self.input_callback(user_input)
            elif ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
                # Backspace
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                    self._refresh_input()
            elif ch == curses.KEY_UP:
                # Scroll board up
                self.board_scroll = max(0, self.board_scroll - 1)
                self._refresh_board()
            elif ch == curses.KEY_DOWN:
                # Scroll board down
                max_scroll = max(0, len(self.board_state_lines) - self.board_win.getmaxyx()[0] + 1)
                self.board_scroll = min(max_scroll, self.board_scroll + 1)
                self._refresh_board()
            elif ch == curses.KEY_PPAGE:  # Page Up
                # Scroll messages up
                self.msg_scroll = max(0, self.msg_scroll - 5)
                self._refresh_messages()
            elif ch == curses.KEY_NPAGE:  # Page Down
                # Scroll messages down
                max_scroll = max(0, len(self.messages) - self.msg_win.getmaxyx()[0] + 1)
                self.msg_scroll = min(max_scroll, self.msg_scroll + 5)
                self._refresh_messages()
            elif 32 <= ch <= 126:  # Printable ASCII
                self.input_buffer += chr(ch)
                self._refresh_input()

            return self.running

        except KeyboardInterrupt:
            self.running = False
            return False

    def show_popup(self, lines: List[str], title: str = ""):
        """Show a temporary popup overlay (press any key to dismiss)"""
        try:
            if not lines:
                return

            height, width = self.stdscr.getmaxyx()

            # Calculate popup dimensions (80% of screen)
            popup_height = min(len(lines) + 4, int(height * 0.8))
            max_line_len = max((len(line) for line in lines), default=20)
            popup_width = min(max_line_len + 4, int(width * 0.8))

            # Center the popup
            y = (height - popup_height) // 2
            x = (width - popup_width) // 2

            # Create popup window with border
            popup = curses.newwin(popup_height, popup_width, y, x)
            popup.box()

            # Add title if provided
            if title:
                popup.addstr(0, 2, f" {title} ", curses.color_pair(2) | curses.A_BOLD)

            # Add content (scrollable if needed)
            max_content_lines = popup_height - 3
            for i, line in enumerate(lines[:max_content_lines]):
                try:
                    popup.addstr(i + 1, 2, line[:popup_width - 4])
                except curses.error:
                    pass

            # Add footer
            footer = "Press any key to close"
            popup.addstr(popup_height - 1, (popup_width - len(footer)) // 2,
                        footer, curses.color_pair(3))

            popup.refresh()

            # Wait for keypress (blocking)
            self.stdscr.timeout(-1)  # Blocking mode
            self.stdscr.getch()
            self.stdscr.timeout(100)  # Back to non-blocking

            # Clear popup and refresh screen
            del popup
            self.stdscr.touchwin()
            self.refresh_all()

        except Exception as e:
            pass

    def show_settings_menu(self, settings_callback):
        """
        Show interactive settings menu.

        Args:
            settings_callback: Function to call with (setting_name, new_value)

        Returns tuple of (models_list, kokoro_voices_list, bark_voices_list, current_model, current_voice, current_volume, current_tts)
        """
        try:
            height, width = self.stdscr.getmaxyx()

            # Create popup (60% of screen)
            popup_height = min(20, int(height * 0.6))
            popup_width = min(70, int(width * 0.7))
            y = (height - popup_height) // 2
            x = (width - popup_width) // 2

            popup = curses.newwin(popup_height, popup_width, y, x)
            popup.keypad(True)

            # Get initial values from callback
            result = settings_callback("get_values", None)
            models, kokoro_voices, bark_voices, current_model, current_voice, current_volume, current_tts = result

            selected_idx = 0
            settings_items = ["AI Model", "Voice", "Volume", "TTS Engine"]

            while True:
                popup.clear()
                popup.box()
                popup.addstr(0, 2, " Settings ", curses.color_pair(2) | curses.A_BOLD)

                # Display settings with selection
                for i, item in enumerate(settings_items):
                    line_y = i + 2

                    # Highlight selected item
                    attr = curses.A_REVERSE if i == selected_idx else 0

                    if item == "AI Model":
                        value = current_model
                        hint = " (Enter to cycle)"
                    elif item == "Voice":
                        value = current_voice
                        hint = " (Enter to cycle)"
                    elif item == "Volume":
                        value = f"{current_volume}%"
                        hint = " (+/- to adjust)"
                    elif item == "TTS Engine":
                        value = "Kokoro" if current_tts == "kokoro" else "BarkTTS"
                        hint = " (Enter to toggle)"

                    display = f"  {item:15} {value}{hint}"
                    try:
                        popup.addstr(line_y, 2, display[:popup_width-4], attr)
                    except curses.error:
                        pass

                # Footer
                footer_y = popup_height - 2
                popup.addstr(footer_y, 2, "↑↓: Navigate  Enter: Change  ESC/Q: Close", curses.color_pair(3))

                popup.refresh()

                # Handle input
                ch = popup.getch()

                if ch == 27 or ch == ord('q') or ch == ord('Q'):  # ESC or Q
                    break
                elif ch == curses.KEY_UP:
                    selected_idx = (selected_idx - 1) % len(settings_items)
                elif ch == curses.KEY_DOWN:
                    selected_idx = (selected_idx + 1) % len(settings_items)
                elif ch == ord('\n') or ch == ord(' '):  # Enter or Space
                    setting = settings_items[selected_idx]

                    if setting == "AI Model":
                        # Cycle to next model
                        current_idx = models.index(current_model) if current_model in models else 0
                        new_idx = (current_idx + 1) % len(models)
                        current_model = models[new_idx]
                        settings_callback("model", current_model)

                    elif setting == "Voice":
                        # Cycle to next voice (use appropriate voice list)
                        voice_list = kokoro_voices if current_tts == "kokoro" else bark_voices
                        current_idx = voice_list.index(current_voice) if current_voice in voice_list else 0
                        new_idx = (current_idx + 1) % len(voice_list)
                        current_voice = voice_list[new_idx]
                        settings_callback("voice", current_voice)

                    elif setting == "TTS Engine":
                        # Toggle TTS engine
                        new_tts = "bark" if current_tts == "kokoro" else "kokoro"
                        current_tts = new_tts
                        settings_callback("tts_engine", new_tts)

                elif ch == ord('+') or ch == ord('='):  # Volume up
                    if settings_items[selected_idx] == "Volume":
                        current_volume = min(100, current_volume + 10)
                        settings_callback("volume", current_volume)

                elif ch == ord('-') or ch == ord('_'):  # Volume down
                    if settings_items[selected_idx] == "Volume":
                        current_volume = max(0, current_volume - 10)
                        settings_callback("volume", current_volume)

            # Cleanup
            del popup
            self.stdscr.touchwin()
            self.refresh_all()

        except Exception as e:
            pass

    def cleanup(self):
        """Cleanup curses"""
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()


class LogHighlighter:
    """Highlights and color-codes MTGA log lines based on detected game items."""

    def __init__(self, card_db=None):
        """Initialize log highlighter with optional card database."""
        self.card_db = card_db

        # Color scheme
        self.colors = {
            "card_detected": "#00ff88",      # Green - card found and resolved
            "grpid": "#55aaff",               # Blue - grpId/Arena ID detected
            "draft_event": "#ffff00",         # Yellow - draft-related event
            "game_event": "#88ffff",          # Cyan - game event
            "error": "#ff5555",               # Red - error or unknown
            "default": "#ffffff",             # White - normal text
        }

        # Pattern detection
        self.patterns = {
            "grpid": r"\b(\d{6})\b",         # 6-digit grpId
            "arena_id": r"ArenaID:(\d+)",    # ArenaID:NNNN format
            "card_name": r"name:([^,\]]+)",  # name:CardName format
        }

    def get_detected_items(self, log_line: str) -> List[dict]:
        """
        Extract and resolve detected items from a log line.

        Returns list of dicts with: {text, type, resolved_name, color, position}
        """
        import re
        detected = []

        # Check for grpIds (4-6 digit numbers - Arena card IDs)
        # Most cards are 5-digit (10035-99795), some 4-digit (6873-9475), few 6-digit (100069-102112)
        for match in re.finditer(r'\b(\d{4,6})\b', log_line):
            grp_id = int(match.group(1))

            # Validate this looks like a card ID (skip small numbers like counts, turns, seats)
            # Valid card IDs: 6+ (minimum), 4873-102112 (practical range)
            if grp_id < 6873 and grp_id > 9:
                continue  # Skip numbers that are too small to be card IDs but too large to ignore

            resolved_name = None
            color = self.colors["grpid"]

            # Try to resolve using card database
            if self.card_db:
                try:
                    card_name = self.card_db.get_card_name(grp_id)
                    # Only set resolved_name if it's NOT an "Unknown()" response
                    if card_name and not card_name.startswith("Unknown("):
                        resolved_name = card_name
                        color = self.colors["card_detected"]  # Green only if found
                except Exception:
                    pass

            detected.append({
                "text": match.group(1),
                "type": "grpid",
                "resolved_name": resolved_name,
                "color": color,
                "position": (match.start(), match.end()),
            })

        # Check for arena IDs
        for match in re.finditer(r'ArenaID:(\d+)', log_line):
            detected.append({
                "text": match.group(0),
                "type": "arena_id",
                "resolved_name": None,
                "color": self.colors["grpid"],
                "position": (match.start(), match.end()),
            })

        # Detect draft events
        draft_keywords = ["Draft", "PackNumber", "PickNumber", "DraftId", "DraftPack"]
        if any(keyword in log_line for keyword in draft_keywords):
            detected.append({
                "text": "DRAFT_EVENT",
                "type": "draft_event",
                "resolved_name": None,
                "color": self.colors["draft_event"],
                "position": (0, 0),  # Metadata only, not positioned
            })

        # Detect game events
        game_keywords = ["GameStage", "GRE_", "Zone", "PlayerState", "Turn"]
        if any(keyword in log_line for keyword in game_keywords):
            detected.append({
                "text": "GAME_EVENT",
                "type": "game_event",
                "resolved_name": None,
                "color": self.colors["game_event"],
                "position": (0, 0),  # Metadata only, not positioned
            })

        return detected

    def get_event_summary(self, log_line: str, detected: List[dict]) -> str:
        """Generate a summary of detected events for status display."""
        summary_parts = []

        if any(d["type"] == "draft_event" for d in detected):
            summary_parts.append("📦 Draft")

        if any(d["type"] == "game_event" for d in detected):
            summary_parts.append("🎮 Game")

        cards_found = [d for d in detected if d["resolved_name"]]
        if cards_found:
            summary_parts.append(f"🃏 {len(cards_found)} card(s)")

        grpids_found = [d for d in detected if d["type"] == "grpid"]
        if grpids_found and not cards_found:
            summary_parts.append(f"🔍 {len(grpids_found)} ID(s)")

        return " | ".join(summary_parts) if summary_parts else ""


class AdvisorGUI:
    def __init__(self, root, advisor_ref):
        self.root = root
        self.advisor_ref = advisor_ref  # Used by event handlers and update loop
        self.advisor = advisor_ref  # Keep for backward compatibility

        # Load user preferences for GUI settings persistence
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()
            logging.debug("User preferences loaded for GUI mode")

        # Configure root window
        self.root.title("MTGA Voice Advisor")

        # Apply window geometry from prefs (or use default)
        if self.prefs:
            geometry = self.prefs.window_geometry
        else:
            geometry = "900x700"
        self.root.geometry(geometry)

        # Apply always_on_top setting (default True)
        always_on_top = self.prefs.always_on_top if self.prefs else True
        self.root.attributes('-topmost', always_on_top)

        self.root.configure(bg='#2b2b2b')

        # Color scheme
        self.bg_color = '#2b2b2b'
        self.fg_color = '#ffffff'
        self.accent_color = '#00ff88'
        self.success_color = '#00ff88'    # Green for success/logs
        self.warning_color = '#ff5555'
        self.info_color = '#55aaff'

        self._create_widgets()

        # Message queue for thread-safe updates
        self.message_queue = deque(maxlen=100)
        self.board_state_lines = []
        self.rag_panel_expanded = False  # Track RAG panel expansion state

        # Bind F12 for bug reports
        self.root.bind('<F12>', lambda e: self._capture_bug_report())

        # Bind window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start update loop
        self.running = True
        self._update_loop()

    def _create_widgets(self):
        """Create all GUI widgets"""

        # Top status bar
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=30)
        status_frame.pack(side=tk.TOP, fill=tk.X)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Initializing...",
            bg='#1a1a1a',
            fg=self.accent_color,
            font=('Consolas', 10, 'bold'),
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Settings panel (left side)
        settings_frame = tk.Frame(self.root, bg=self.bg_color, width=250)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        settings_frame.pack_propagate(False)

        tk.Label(
            settings_frame,
            text="⚙ SETTINGS",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 12, 'bold')
        ).pack(pady=(0, 10))

        # Model selection
        tk.Label(settings_frame, text="AI Model:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(settings_frame, textvariable=self.model_var, width=25)
        self.model_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.model_dropdown.bind('<<ComboboxSelected>>', self._on_model_change)
        self.model_dropdown.bind('<Return>', self._on_model_change)  # Allow Enter key to confirm

        # Voice selection
        tk.Label(settings_frame, text="Voice:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.voice_var = tk.StringVar()
        self.voice_dropdown = ttk.Combobox(settings_frame, textvariable=self.voice_var, state='readonly', width=25)
        self.voice_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.voice_dropdown.bind('<<ComboboxSelected>>', self._on_voice_change)

        # TTS Engine toggle
        tk.Label(settings_frame, text="TTS Engine:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.tts_engine_var = tk.StringVar(value="Kokoro")
        tts_frame = tk.Frame(settings_frame, bg=self.bg_color)
        tts_frame.pack(pady=(0, 10), fill=tk.X)

        tk.Radiobutton(
            tts_frame,
            text="Kokoro",
            variable=self.tts_engine_var,
            value="Kokoro",
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            command=self._on_tts_engine_change
        ).pack(side=tk.LEFT)

        tk.Radiobutton(
            tts_frame,
            text="BarkTTS",
            variable=self.tts_engine_var,
            value="BarkTTS",
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            command=self._on_tts_engine_change
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Volume slider
        tk.Label(settings_frame, text="Volume:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        volume_frame = tk.Frame(settings_frame, bg=self.bg_color)
        volume_frame.pack(pady=(0, 10), fill=tk.X)

        self.volume_var = tk.IntVar(value=100)
        self.volume_slider = tk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=self._on_volume_change,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            troughcolor='#1a1a1a'
        )
        self.volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.volume_label = tk.Label(volume_frame, text="100%", bg=self.bg_color, fg=self.fg_color, width=5)
        self.volume_label.pack(side=tk.RIGHT)

        # Opponent Turn Alerts checkbox (renamed from "Continuous Monitoring")
        opponent_alerts_default = self.prefs.opponent_turn_alerts if self.prefs else True
        self.continuous_var = tk.BooleanVar(value=opponent_alerts_default)
        tk.Checkbutton(
            settings_frame,
            text="Opponent Turn Alerts",
            variable=self.continuous_var,
            command=self._on_continuous_toggle,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Show thinking checkbox
        show_thinking_default = self.prefs.show_thinking if self.prefs else True
        self.show_thinking_var = tk.BooleanVar(value=show_thinking_default)
        tk.Checkbutton(
            settings_frame,
            text="Show AI Thinking",
            variable=self.show_thinking_var,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Always on top checkbox
        always_on_top_default = self.prefs.always_on_top if self.prefs else True
        self.always_on_top_var = tk.BooleanVar(value=always_on_top_default)
        tk.Checkbutton(
            settings_frame,
            text="Always on Top",
            variable=self.always_on_top_var,
            command=self._on_always_on_top_toggle,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Pick Two Draft checkbox
        self.pick_two_draft_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            settings_frame,
            text="Pick Two Draft",
            variable=self.pick_two_draft_var,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Buttons
        tk.Button(
            settings_frame,
            text="Clear Messages",
            command=self._clear_messages,
            bg='#3a3a3a',
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=(20, 5), fill=tk.X)

        tk.Button(
            settings_frame,
            text="🐛 Bug Report (F12)",
            command=self._capture_bug_report,
            bg='#5555ff',
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=5, fill=tk.X)

        tk.Button(
            settings_frame,
            text="Exit",
            command=self._on_exit,
            bg=self.warning_color,
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=5, fill=tk.X)

        # Chat/Prompt input area
        chat_label = tk.Label(
            settings_frame,
            text="📝 SEND PROMPT",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        chat_label.pack(pady=(15, 5), anchor=tk.W)

        self.prompt_text = tk.Text(
            settings_frame,
            height=4,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=5,
            pady=5,
            wrap=tk.WORD
        )
        self.prompt_text.pack(pady=(0, 5), fill=tk.BOTH, expand=False)
        self.prompt_text.bind('<Control-Return>', self._on_prompt_send)

        send_btn = tk.Button(
            settings_frame,
            text="Send [Ctrl+Enter]",
            command=self._on_prompt_send,
            bg=self.info_color,
            fg='#1a1a1a',
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        send_btn.pack(pady=5, fill=tk.X)

        # Main content area (right side)
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Board state area (top)
        board_header_frame = tk.Frame(content_frame, bg=self.bg_color)
        board_header_frame.pack(pady=(0, 5), fill=tk.X)

        self.board_label = tk.Label(
            board_header_frame,
            text="═══ BOARD STATE ═══",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        self.board_label.pack(side=tk.LEFT, expand=True)

        # Draft card counter (hidden by default)
        self.draft_counter_label = tk.Label(
            board_header_frame,
            text="",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 9, 'bold')
        )
        self.draft_counter_label.pack(side=tk.RIGHT, padx=10)

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

        # Configure color tags for draft pack display (matching messages_text colors)
        self.board_text.tag_config('color_w', foreground='#ffff99')   # White - pale yellow
        self.board_text.tag_config('color_u', foreground='#55aaff')   # Blue
        self.board_text.tag_config('color_b', foreground='#aaaaaa')   # Black - gray
        self.board_text.tag_config('color_r', foreground='#ff5555')   # Red
        self.board_text.tag_config('color_g', foreground='#00ff88')   # Green
        self.board_text.tag_config('color_c', foreground='#cccccc')   # Colorless - light gray
        self.board_text.tag_config('color_multi', foreground='#ffdd44') # Multicolor - orange-yellow

        # Advisor messages area (bottom)
        self.advisor_label = tk.Label(
            content_frame,
            text="═══ ADVISOR ═══",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        self.advisor_label.pack(pady=(10, 5))

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
        self.messages_text.config(state=tk.DISABLED)

        # Configure text tags for colors
        self.messages_text.tag_config('green', foreground='#00ff88')
        self.messages_text.tag_config('blue', foreground='#55aaff')
        self.messages_text.tag_config('cyan', foreground='#00ffff')
        self.messages_text.tag_config('yellow', foreground='#ffff00')
        self.messages_text.tag_config('red', foreground='#ff5555')
        self.messages_text.tag_config('white', foreground='#ffffff')

        # MTGA Logs panel
        logs_header_frame = tk.Frame(content_frame, bg=self.bg_color)
        logs_header_frame.pack(pady=(10, 5), fill=tk.X)

        self.logs_label = tk.Label(
            logs_header_frame,
            text="═══ MTGA LOGS (COLOR-CODED) ═══",
            bg=self.bg_color,
            fg=self.success_color,
            font=('Consolas', 10, 'bold')
        )
        self.logs_label.pack(side=tk.LEFT, expand=True)

        # Toggle button for logs
        self.logs_toggle_btn = tk.Button(
            logs_header_frame,
            text="[Collapse]",
            bg='#1a1a1a',
            fg=self.success_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            command=self._toggle_logs_panel
        )
        self.logs_toggle_btn.pack(side=tk.RIGHT, padx=5)

        self.logs_text = scrolledtext.ScrolledText(
            content_frame,
            height=8,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        self.logs_text.config(state=tk.DISABLED)

        # Configure color tags for logs
        self.logs_text.tag_config('card_detected', foreground='#00ff88')  # Green
        self.logs_text.tag_config('grpid', foreground='#55aaff')         # Blue
        self.logs_text.tag_config('draft_event', foreground='#ffff00')    # Yellow
        self.logs_text.tag_config('game_event', foreground='#88ffff')     # Cyan
        self.logs_text.tag_config('error', foreground='#ff5555')          # Red
        self.logs_text.tag_config('default', foreground='#ffffff')        # White

        # Initialize log highlighter
        self.log_highlighter = LogHighlighter(card_db=None)  # Will be set in __init__

        # RAG References panel (bottom)
        rag_header_frame = tk.Frame(content_frame, bg=self.bg_color)
        rag_header_frame.pack(pady=(10, 5), fill=tk.X)

        self.rag_label = tk.Label(
            rag_header_frame,
            text="═══ RAG REFERENCES ═══",
            bg=self.bg_color,
            fg=self.info_color,
            font=('Consolas', 10, 'bold')
        )
        self.rag_label.pack(side=tk.LEFT, expand=True)

        # Toggle button for RAG references
        self.rag_toggle_btn = tk.Button(
            rag_header_frame,
            text="[Expand]",
            bg='#1a1a1a',
            fg=self.info_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            command=self._toggle_rag_panel
        )
        self.rag_toggle_btn.pack(side=tk.RIGHT, padx=5)

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
        self.rag_text.config(state=tk.DISABLED)

        # Tag configurations for RAG panel
        self.rag_text.tag_config('rule', foreground='#ffff00', font=('Consolas', 8, 'bold'))
        self.rag_text.tag_config('card', foreground='#00ff88', font=('Consolas', 8, 'bold'))
        self.rag_text.tag_config('query', foreground='#55aaff', font=('Consolas', 8, 'italic'))
        self.rag_text.tag_config('stats', foreground='#ff88ff')

    def _on_model_change(self, event=None):
        """Handle model selection change."""
        try:
            model = self.model_var.get()
            if model and CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_model(model)
                logging.debug(f"Model changed to: {model}")
        except Exception as e:
            logging.error(f"Error changing model: {e}")

    def _on_voice_change(self, event=None):
        """Handle voice selection change."""
        try:
            voice = self.voice_var.get()
            if voice and CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_voice_name(voice)
                logging.debug(f"Voice changed to: {voice}")
        except Exception as e:
            logging.error(f"Error changing voice: {e}")

    def _toggle_rag_panel(self):
        """Toggle RAG references panel visibility."""
        try:
            if self.rag_text.winfo_viewable():
                self.rag_text.pack_forget()
                self.rag_toggle_btn.config(text="[Expand]")
            else:
                self.rag_text.pack(fill=tk.BOTH, expand=False)
                self.rag_toggle_btn.config(text="[Collapse]")
        except Exception as e:
            logging.error(f"Error toggling RAG panel: {e}")

    def _toggle_logs_panel(self):
        """Toggle MTGA logs panel visibility."""
        try:
            if self.logs_text.winfo_viewable():
                self.logs_text.pack_forget()
                self.logs_toggle_btn.config(text="[Expand]")
            else:
                self.logs_text.pack(fill=tk.BOTH, expand=True)
                self.logs_toggle_btn.config(text="[Collapse]")
        except Exception as e:
            logging.error(f"Error toggling logs panel: {e}")

    def _on_exit(self):
        """Handle exit button click."""
        try:
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.save()
            self.root.quit()
        except Exception as e:
            logging.error(f"Error on exit: {e}")
            self.root.quit()

    def _prompt_for_credentials(self):
        """Show dialog to collect GitHub and ImgBB API credentials."""
        if not TKINTER_AVAILABLE:
            return None

        dialog = tk.Toplevel(self.root)
        dialog.title("Bug Report API Credentials")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"600x400+{x}+{y}")

        result = {}

        # Main frame with padding
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="API Credentials for Bug Reporting",
            font=("TkDefaultFont", 12, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Info label
        info_label = ttk.Label(
            main_frame,
            text="These credentials will be saved locally for future bug reports.\n"
                 "Leave blank to skip uploading to GitHub and ImgBB.",
            justify=tk.LEFT,
            wraplength=550
        )
        info_label.pack(pady=(0, 15))

        # Create a scrollable frame for the form
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # GitHub credentials
        github_label = ttk.Label(scrollable_frame, text="GitHub Credentials:", font=("TkDefaultFont", 10, "bold"))
        github_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        ttk.Label(scrollable_frame, text="GitHub Owner (username):").grid(row=1, column=0, sticky="w", pady=2)
        github_owner_entry = ttk.Entry(scrollable_frame, width=50)
        github_owner_entry.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))

        ttk.Label(scrollable_frame, text="GitHub Repo (repository name):").grid(row=2, column=0, sticky="w", pady=2)
        github_repo_entry = ttk.Entry(scrollable_frame, width=50)
        github_repo_entry.grid(row=2, column=1, sticky="ew", pady=2, padx=(10, 0))

        ttk.Label(scrollable_frame, text="GitHub Token (personal access token):").grid(row=3, column=0, sticky="w", pady=2)
        github_token_entry = ttk.Entry(scrollable_frame, width=50, show="*")
        github_token_entry.grid(row=3, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Separator
        ttk.Separator(scrollable_frame, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=15)

        # ImgBB credentials
        imgbb_label = ttk.Label(scrollable_frame, text="ImgBB Credentials:", font=("TkDefaultFont", 10, "bold"))
        imgbb_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 5))

        ttk.Label(scrollable_frame, text="ImgBB API Key:").grid(row=6, column=0, sticky="w", pady=2)
        imgbb_key_entry = ttk.Entry(scrollable_frame, width=50, show="*")
        imgbb_key_entry.grid(row=6, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Help text
        help_text = ttk.Label(
            scrollable_frame,
            text="How to get these credentials:\n\n"
                 "GitHub Token: Settings → Developer settings → Personal access tokens → Generate new token\n"
                 "  (Required scope: 'repo' for creating issues)\n\n"
                 "ImgBB API Key: https://api.imgbb.com/ → Get API key (free)",
            justify=tk.LEFT,
            foreground="gray",
            wraplength=550
        )
        help_text.grid(row=7, column=0, columnspan=2, sticky="w", pady=(15, 0))

        scrollable_frame.columnconfigure(1, weight=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))

        def on_save():
            result["github_owner"] = github_owner_entry.get().strip()
            result["github_repo"] = github_repo_entry.get().strip()
            result["github_token"] = github_token_entry.get().strip()
            result["imgbb_api_key"] = imgbb_key_entry.get().strip()
            dialog.destroy()

        def on_cancel():
            result["cancelled"] = True
            dialog.destroy()

        save_btn = ttk.Button(button_frame, text="Save & Continue", command=on_save)
        save_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Pre-fill with existing values or project defaults
        if CONFIG_MANAGER_AVAILABLE and self.prefs:
            # Use stored values if available, otherwise use project defaults
            github_owner_entry.insert(0, self.prefs.github_owner or "josharmour")
            github_repo_entry.insert(0, self.prefs.github_repo or "mtga-voice-assistant")
            if self.prefs.github_token:
                github_token_entry.insert(0, self.prefs.github_token)
            if self.prefs.imgbb_api_key:
                imgbb_key_entry.insert(0, self.prefs.imgbb_api_key)
        else:
            # No preferences available, use project defaults
            github_owner_entry.insert(0, "josharmour")
            github_repo_entry.insert(0, "mtga-voice-assistant")

        # Wait for dialog to close
        dialog.wait_window()

        return result if result and not result.get("cancelled") else None

    def _capture_bug_report(self):
        """Capture bug report with screenshot, logs, and board state"""
        import threading
        import subprocess
        import os
        import time
        import base64
        import requests

        # Ask user if they want to add a title and description (before background thread)
        add_details = False
        issue_title = None
        user_description = None
        should_upload = False
        credentials_ready = False

        try:
            from tkinter import messagebox, simpledialog
            if self.root and self.root.winfo_exists():
                add_details = messagebox.askyesno(
                    "Bug Report Details",
                    "Do you want to add a title and description to this bug report?",
                    parent=self.root
                )
                if add_details:
                    # Prompt for title
                    title_prompt = simpledialog.askstring(
                        "Bug Report Title",
                        "Enter a title for the bug report (or leave blank for default):",
                        parent=self.root
                    )
                    if title_prompt and title_prompt.strip():
                        issue_title = title_prompt.strip()

                    # Prompt for description
                    desc_prompt = simpledialog.askstring(
                        "Bug Report Description",
                        "Please describe the bug:",
                        parent=self.root
                    )
                    if desc_prompt and desc_prompt.strip():
                        user_description = desc_prompt.strip()
        except (ImportError, Exception) as e:
            logging.debug(f"GUI not available for bug report details: {e}")

        # Notify user we're starting the capture
        self.add_message("📸 Capturing bug report...", "cyan")

        # Define a callback to handle upload decision and credentials AFTER local save
        def handle_upload_decision():
            """Called from background thread when local save is complete."""
            nonlocal should_upload, credentials_ready

            # Ask user if they want to upload to GitHub (must be in main thread)
            try:
                from tkinter import messagebox
                if self.root and self.root.winfo_exists():
                    # Use after() to run in main thread
                    result = []
                    def ask_upload():
                        result.append(messagebox.askyesno(
                            "Upload Bug Report?",
                            "Bug report saved locally!\n\nWould you like to upload it to GitHub?",
                            parent=self.root
                        ))
                    self.root.after(0, ask_upload)
                    # Wait for result
                    import time
                    timeout = 30  # 30 second timeout
                    elapsed = 0
                    while not result and elapsed < timeout:
                        time.sleep(0.1)
                        elapsed += 0.1

                    if result and result[0]:
                        should_upload = True

                        # Check for cached credentials
                        needs_credentials = False
                        if CONFIG_MANAGER_AVAILABLE and self.prefs:
                            if not self.prefs.has_github_credentials() or not self.prefs.imgbb_api_key:
                                needs_credentials = True
                        else:
                            needs_credentials = True

                        if needs_credentials:
                            # Prompt for credentials in main thread
                            cred_result = []
                            def ask_credentials():
                                cred_result.append(self._prompt_for_credentials())
                            self.root.after(0, ask_credentials)

                            # Wait for credentials
                            elapsed = 0
                            while not cred_result and elapsed < timeout:
                                time.sleep(0.1)
                                elapsed += 0.1

                            if cred_result and cred_result[0] and not cred_result[0].get("cancelled"):
                                credentials = cred_result[0]
                                # Save the credentials
                                if CONFIG_MANAGER_AVAILABLE and self.prefs:
                                    self.prefs.set_api_keys(
                                        github_token=credentials.get("github_token", ""),
                                        github_owner=credentials.get("github_owner", ""),
                                        github_repo=credentials.get("github_repo", ""),
                                        imgbb_api_key=credentials.get("imgbb_api_key", "")
                                    )
                                    logging.info("API credentials saved to user preferences")
                                    credentials_ready = True
                            else:
                                should_upload = False
                        else:
                            credentials_ready = True
            except (ImportError, Exception) as e:
                logging.debug(f"GUI not available for upload prompt: {e}")

        def upload_to_imgbb(image_path):
            """Upload image to ImgBB using stored credentials."""
            if not CONFIG_MANAGER_AVAILABLE or not self.prefs or not self.prefs.imgbb_api_key:
                return None, "ImgBB API key not configured."

            try:
                with open(image_path, "rb") as file:
                    payload = {
                        "key": self.prefs.imgbb_api_key,
                        "image": base64.b64encode(file.read()),
                    }
                    response = requests.post("https://api.imgbb.com/1/upload", payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data["data"]["url"], None
                    else:
                        return None, f"ImgBB API error: {response.text}"
            except Exception as e:
                return None, f"ImgBB upload failed: {str(e)}"

        def create_github_issue(title, body):
            """Create GitHub issue using stored credentials."""
            if not CONFIG_MANAGER_AVAILABLE or not self.prefs:
                return None, "Config manager not available."

            if not self.prefs.has_github_credentials():
                return None, "GitHub credentials not configured."

            try:
                url = f"https://api.github.com/repos/{self.prefs.github_owner}/{self.prefs.github_repo}/issues"
                headers = {
                    "Authorization": f"token {self.prefs.github_token}",
                    "Accept": "application/vnd.github.v3+json",
                }
                data = {"title": title, "body": body}
                response = requests.post(url, json=data, headers=headers)
                if response.status_code == 201:
                    return response.json()["html_url"], None
                else:
                    return None, f"GitHub API error: {response.text}"
            except Exception as e:
                return None, f"GitHub issue creation failed: {str(e)}"

        def capture_in_background():
            try:
                # Create bug_reports directory if it doesn't exist
                bug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bug_reports")
                os.makedirs(bug_dir, exist_ok=True)

                # Generate timestamp for this report
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                report_file = os.path.join(bug_dir, f"bug_report_{timestamp}.txt")
                screenshot_file = os.path.join(bug_dir, f"screenshot_{timestamp}.png")

                # Use title and description from parent scope (already prompted before thread started)
                final_title = issue_title if issue_title else f"Bug Report: {timestamp}"
                final_description = user_description if user_description else "No description provided."

                # Take screenshot using gnome-screenshot or scrot
                try:
                    subprocess.run(['gnome-screenshot', '-f', screenshot_file],
                                 timeout=2, check=False, capture_output=True)
                except:
                    try:
                        subprocess.run(['scrot', screenshot_file],
                                     timeout=2, check=False, capture_output=True)
                    except:
                        screenshot_file = "Screenshot failed (install gnome-screenshot or scrot)"

                # Collect current state
                board_state_text = "\n".join(self.board_state_lines) if self.board_state_lines else "No board state"

                # Read recent logs (last 300 lines)
                recent_logs = ""
                log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "advisor.log")
                try:
                    with open(log_path, "r") as f:
                        lines = f.readlines()
                        recent_logs = "".join(lines[-300:])
                except Exception as e:
                    recent_logs = f"Failed to read logs: {e}"

                # Get current settings
                settings = f"""Model: {self.model_var.get() if hasattr(self, 'model_var') else 'N/A'}
Voice: {self.voice_var.get() if hasattr(self, 'voice_var') else 'N/A'}
TTS Engine: {self.tts_engine_var.get() if hasattr(self, 'tts_engine_var') else 'N/A'}
Volume: {self.volume_var.get() if hasattr(self, 'volume_var') else 'N/A'}%
Continuous Monitoring: {self.continuous_var.get() if hasattr(self, 'continuous_var') else 'N/A'}
Show AI Thinking: {self.show_thinking_var.get() if hasattr(self, 'show_thinking_var') else 'N/A'}
"""

                # Write bug report to local file
                with open(report_file, "w") as f:
                    f.write("="*70 + "\n")
                    f.write(f"BUG REPORT: {final_title}\n")
                    f.write("="*70 + "\n\n")

                    f.write("USER DESCRIPTION:\n")
                    f.write(f"{final_description}\n\n")

                    f.write("SCREENSHOT:\n")
                    f.write(f"{screenshot_file}\n\n")

                    f.write("="*70 + "\n")
                    f.write("CURRENT SETTINGS:\n")
                    f.write("="*70 + "\n")
                    f.write(settings + "\n")

                    f.write("="*70 + "\n")
                    f.write("CURRENT BOARD STATE:\n")
                    f.write("="*70 + "\n")
                    f.write(board_state_text + "\n\n")

                    f.write("="*70 + "\n")
                    f.write("RECENT LOGS (last 300 lines):\n")
                    f.write("="*70 + "\n")
                    f.write(recent_logs + "\n")

                self.add_message(f"✓ Bug report saved locally: {report_file}", "green")
                logging.info(f"Bug report captured locally: {report_file}")

                # Handle upload decision and credentials (runs in main thread via after())
                handle_upload_decision()

                # Only attempt upload if credentials_ready flag is set (from main thread)
                if not should_upload:
                    self.add_message("✓ Bug report saved locally only", "green")
                    return

                if not credentials_ready:
                    self.add_message("⚠ Upload cancelled - credentials not configured", "yellow")
                    return

                # Upload screenshot to Imgbb
                screenshot_url, error = upload_to_imgbb(screenshot_file)
                if error:
                    self.add_message(f"⚠ Screenshot upload failed: {error}", "yellow")
                    logging.warning(f"Screenshot upload failed: {error}")

                    # If ImgBB API key is invalid, suggest re-entering credentials
                    if "Invalid API" in str(error) or "Bad" in str(error):
                        self.add_message("💡 Tip: Your ImgBB API key may be invalid. Delete ~/.mtga_advisor/preferences.json to re-enter credentials.", "cyan")

                    # Continue without screenshot URL
                    screenshot_url = screenshot_file

                # Create GitHub issue
                issue_body = f"""**Description:**
{final_description}

**Screenshot:**
![Screenshot]({screenshot_url})

**Settings:**
```
{settings}
```

**Board State:**
```
{board_state_text}
```

**Recent Logs:**
```
{recent_logs}
```
"""
                issue_url, error = create_github_issue(final_title, issue_body)
                if error:
                    self.add_message(f"⚠ GitHub upload failed: {error}", "yellow")
                    logging.info(f"GitHub issue not created: {error}")

                    # If credentials are bad, suggest re-entering them
                    if "Bad credentials" in str(error) or "401" in str(error):
                        self.add_message("💡 Tip: Your GitHub token may be invalid. Delete ~/.mtga_advisor/preferences.json to re-enter credentials.", "cyan")
                else:
                    self.add_message(f"✓ Bug report uploaded to GitHub: {issue_url}", "green")
                    logging.info(f"Bug report uploaded to GitHub: {issue_url}")

            except Exception as e:
                self.add_message(f"✗ Bug report failed: {e}", "red")
                logging.error(f"Failed to capture bug report: {e}")

        # Run in background thread so it doesn't freeze the UI
        threading.Thread(target=capture_in_background, daemon=True).start()
        self.add_message("📸 Capturing bug report...", "cyan")

    def _on_tts_engine_change(self):
        """Handle TTS engine selection change."""
        try:
            engine = self.tts_engine_var.get()
            if engine and CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_voice_settings(engine=engine)
                logging.debug(f"TTS engine changed to: {engine}")
        except Exception as e:
            logging.error(f"Error changing TTS engine: {e}")

    def _on_volume_change(self, value):
        """Handle volume slider change."""
        try:
            volume = int(value)
            # Update volume label
            self.volume_label.config(text=f"{volume}%")
            # Save preference
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_voice_settings(volume=volume)
                logging.debug(f"Volume changed to: {volume}%")
        except Exception as e:
            logging.error(f"Error changing volume: {e}")

    def _on_continuous_toggle(self):
        """Handle opponent turn alerts toggle."""
        try:
            enabled = self.continuous_var.get()
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_game_preferences(opponent_alerts=enabled)
                logging.debug(f"Opponent turn alerts: {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logging.error(f"Error toggling opponent alerts: {e}")

    def _on_always_on_top_toggle(self):
        """Handle always on top toggle."""
        try:
            enabled = self.always_on_top_var.get()
            if self.root:
                self.root.attributes('-topmost', enabled)
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.always_on_top = enabled
                self.prefs.save()
                logging.debug(f"Always on top: {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logging.error(f"Error toggling always on top: {e}")

    def _clear_messages(self):
        """Clear the messages display."""
        try:
            self.messages_text.config(state=tk.NORMAL)
            self.messages_text.delete(1.0, tk.END)
            self.messages_text.config(state=tk.DISABLED)
            logging.info("Messages cleared")
        except Exception as e:
            logging.error(f"Error clearing messages: {e}")

    def _on_prompt_send(self, event=None):
        """Handle prompt send button or Ctrl+Enter."""
        try:
            prompt = self.prompt_text.get(1.0, tk.END).strip()
            if not prompt:
                return

            # Clear the prompt text
            self.prompt_text.delete(1.0, tk.END)

            # Send to advisor if available
            if self.advisor_ref:
                logging.info(f"Custom prompt sent: {prompt[:50]}...")
                # The advisor would process this custom prompt
                # This is a placeholder for the actual implementation

        except Exception as e:
            logging.error(f"Error sending prompt: {e}")

    def on_closing(self):
        """Handle window close event."""
        try:
            # Save window geometry
            if self.root:
                geometry = self.root.geometry()
                if geometry and CONFIG_MANAGER_AVAILABLE and self.prefs:
                    # Parse geometry string: "WIDTHxHEIGHT+X+Y"
                    parts = geometry.split('+')
                    if len(parts) >= 3:
                        size_part = parts[0]
                        x = int(parts[1])
                        y = int(parts[2])
                        if 'x' in size_part:
                            w, h = map(int, size_part.split('x'))
                            self.prefs.set_window_geometry(w, h, x, y)

                # Close the window
                self.root.destroy()
        except Exception as e:
            logging.error(f"Error during window close: {e}")
            if self.root:
                self.root.destroy()

    def cleanup(self):
        """Placeholder cleanup method for GUI. Actual cleanup is handled by on_closing and Tkinter's destruction."""
        logging.debug("AdvisorGUI cleanup method called.")
        # No specific cleanup needed here as on_closing handles window destruction

    def _update_loop(self):
        """Main GUI update loop."""
        try:
            # Update board state display
            if self.advisor_ref and hasattr(self.advisor_ref, 'board_state'):
                board_state = self.advisor_ref.board_state
                if board_state:
                    # Update board display (placeholder)
                    pass

            # Schedule next update
            if self.root and self.root.winfo_exists():
                self.root.after(100, self._update_loop)

        except Exception as e:
            logging.error(f"Error in update loop: {e}")
            # Try to reschedule despite error
            if self.root and self.root.winfo_exists():
                self.root.after(100, self._update_loop)

    def update_settings(self, models, voices, bark_voices, current_model, current_voice, volume, tts_engine):
        """Update GUI settings with current values from advisor."""
        try:
            # Update model dropdown
            if hasattr(self, 'model_dropdown'):
                self.model_dropdown['values'] = models
                self.model_var.set(current_model)

            # Update voice dropdown
            if hasattr(self, 'voice_dropdown'):
                all_voices = list(voices) + list(bark_voices)
                self.voice_dropdown['values'] = all_voices
                self.voice_var.set(current_voice)

            # Update volume slider
            if hasattr(self, 'volume_slider'):
                self.volume_var.set(volume)
                self.volume_label.config(text=f"{volume}%")

            # Update TTS engine radio buttons
            if hasattr(self, 'tts_engine_var'):
                self.tts_engine_var.set(tts_engine)

            logging.debug(f"GUI settings updated: model={current_model}, voice={current_voice}, volume={volume}, engine={tts_engine}")

        except Exception as e:
            logging.error(f"Error updating GUI settings: {e}")

    def set_card_database(self, card_db):
        """Set the card database for log highlighting."""
        try:
            if hasattr(self, 'log_highlighter'):
                self.log_highlighter.card_db = card_db
        except Exception as e:
            logging.error(f"Error setting card database: {e}")

    def set_status(self, text: str):
        """Update status display (currently a no-op for GUI, but required for compatibility)."""
        try:
            # Could update window title or a status bar if we add one
            logging.debug(f"Status: {text}")
        except Exception as e:
            logging.error(f"Error setting status: {e}")

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

        except Exception as e:
            logging.error(f"Error setting board state: {e}")

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
                # Map common color names to tags
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

        except Exception as e:
            logging.error(f"Error adding message: {e}")

    def add_log_line(self, log_line: str, detected_items: List = None):
        """
        Add a log line to the logs display with color-coded detected items.

        Args:
            log_line: The raw log line text
            detected_items: List of detected items with color/type info (from LogHighlighter)
        """
        try:
            if not hasattr(self, 'logs_text'):
                return

            import time
            timestamp = time.strftime("%H:%M:%S")

            self.logs_text.config(state=tk.NORMAL)

            # Add timestamp
            self.logs_text.insert(tk.END, f"[{timestamp}] ")

            # If no detected items, just add the line normally
            if not detected_items:
                self.logs_text.insert(tk.END, log_line + '\n')
                self.logs_text.config(state=tk.DISABLED)
                self.logs_text.see(tk.END)
                return

            # Process detected items - need to add text with color tags intelligently
            # For now, add the full line with a summary tag if items were detected
            has_card = any(d.get("resolved_name") for d in detected_items)
            has_draft = any(d.get("type") == "draft_event" for d in detected_items)
            has_game = any(d.get("type") == "game_event" for d in detected_items)

            # Determine the primary tag based on what was detected
            tag = None
            if has_card:
                tag = "card_detected"
            elif has_draft:
                tag = "draft_event"
            elif has_game:
                tag = "game_event"
            elif detected_items:
                tag = "grpid"

            # Add the log line with appropriate tag
            if tag:
                self.logs_text.insert(tk.END, log_line + '\n', tag)
            else:
                self.logs_text.insert(tk.END, log_line + '\n')

            # Add summary of what was detected (as a comment)
            if detected_items:
                summary = self.log_highlighter.get_event_summary(log_line, detected_items)
                if summary:
                    self.logs_text.insert(tk.END, f"  └─ {summary}\n", "default")

            self.logs_text.config(state=tk.DISABLED)

            # Auto-scroll to bottom
            self.logs_text.see(tk.END)

            # Keep only last 500 lines to prevent memory issues
            line_count = int(self.logs_text.index(tk.END).split('.')[0])
            if line_count > 500:
                self.logs_text.config(state=tk.NORMAL)
                self.logs_text.delete("1.0", "50.0")
                self.logs_text.config(state=tk.DISABLED)

        except Exception as e:
            logging.error(f"Error adding log line: {e}")
