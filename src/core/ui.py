
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
import time
from collections import deque
from typing import List, Callable
from .secondary_window import SecondaryWindow
import json
import threading
import multiprocessing
from .monitoring import get_monitor

# ----------------------------------------------------------------------------------
# Performance: Compiled Regex Patterns for Log Event Detection
# ----------------------------------------------------------------------------------

# Compiled regex for efficient draft event detection in log filtering
DRAFT_EVENT_PATTERN = re.compile(r'Draft|PackNumber|PickNumber|DraftId|DraftPack')

# Compiled regex for efficient game event detection in log filtering
GAME_EVENT_PATTERN = re.compile(r'GameStage|GRE_|Zone|PlayerState|Turn')


# ----------------------------------------------------------------------------------
# Content of src/tts.py
# ----------------------------------------------------------------------------------

def _tts_worker_process(queue: multiprocessing.Queue, voice: str, volume: float, force_engine: str):
    """
    Worker process for TTS generation.
    Runs in a separate process to avoid blocking the main UI thread.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='[TTS Worker] %(message)s')

    tts_engine = None
    tts = None
    np = None
    bark_processor = None
    bark_model = None
    torch_module = None

    def init_kokoro():
        nonlocal tts_engine, tts, np
        try:
            from kokoro_onnx import Kokoro
            import numpy
            np = numpy

            from pathlib import Path as P
            models_dir = P.home() / '.local' / 'share' / 'kokoro'
            model_path = str(models_dir / 'kokoro-v1.0.onnx')
            voices_path = str(models_dir / 'voices-v1.0.bin')

            tts = Kokoro(model_path=model_path, voices_path=voices_path)
            tts_engine = "kokoro"
            logging.info("Kokoro TTS initialized in worker process")
            return True
        except Exception as e:
            logging.debug(f"Kokoro init failed: {e}")
            return False

    def init_bark():
        nonlocal tts_engine, bark_processor, bark_model, np, torch_module
        try:
            from transformers import AutoProcessor, BarkModel
            import numpy
            import torch

            np = numpy
            torch_module = torch

            bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            bark_model = BarkModel.from_pretrained("suno/bark-small")

            if torch.cuda.is_available():
                bark_model = bark_model.to("cuda")

            tts_engine = "bark"
            logging.info("BarkTTS initialized in worker process")
            return True
        except Exception as e:
            logging.debug(f"BarkTTS init failed: {e}")
            return False

    def play_audio(audio_array, sample_rate):
        """Play audio using sounddevice or fallback"""
        try:
            import sounddevice as sd
            sd.play(audio_array, sample_rate)
            sd.wait()  # Wait for playback to finish
        except ImportError:
            # Fallback to file-based playback
            import scipy.io.wavfile as wavfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                wavfile.write(tmp_path, sample_rate, (audio_array * 32767).astype(np.int16))

            players = [
                ["aplay", tmp_path],
                ["paplay", tmp_path],
                ["ffplay", "-nodisp", "-autoexit", tmp_path]
            ]
            for cmd in players:
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except:
                    continue
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            logging.error(f"Audio playback error: {e}")

    def speak_kokoro(text, current_voice, current_volume):
        """Generate and play TTS using Kokoro"""
        try:
            clean_text = text.replace('\n', ' ').replace('\r', '').strip()
            audio_array, sample_rate = tts.create(clean_text, voice=current_voice, speed=1.0)
            audio_array = audio_array * current_volume
            play_audio(audio_array, sample_rate)
        except Exception as e:
            logging.error(f"Kokoro TTS error: {e}")

    def speak_bark(text, current_voice, current_volume):
        """Generate and play TTS using BarkTTS"""
        try:
            inputs = bark_processor(text, voice_preset=current_voice)
            if torch_module.cuda.is_available():
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch_module.no_grad():
                audio_array = bark_model.generate(**inputs)

            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = bark_model.generation_config.sample_rate
            audio_array = audio_array * current_volume
            play_audio(audio_array, sample_rate)
        except Exception as e:
            logging.error(f"BarkTTS error: {e}")

    # Initialize TTS engine
    current_voice = voice
    current_volume = volume

    if force_engine == "bark":
        init_bark()
    elif force_engine == "kokoro":
        init_kokoro()
    else:
        if not init_kokoro():
            init_bark()

    if not tts_engine:
        logging.error("No TTS engine available in worker process")
        return

    logging.info(f"TTS worker ready with engine: {tts_engine}")

    # Process queue messages
    while True:
        try:
            msg = queue.get()

            if msg is None:  # Shutdown signal
                logging.info("TTS worker shutting down")
                break

            if isinstance(msg, dict):
                cmd = msg.get("cmd")
                if cmd == "speak":
                    text = msg.get("text", "")
                    if text and tts_engine:
                        if tts_engine == "kokoro":
                            speak_kokoro(text, current_voice, current_volume)
                        elif tts_engine == "bark":
                            speak_bark(text, current_voice, current_volume)
                elif cmd == "set_voice":
                    current_voice = msg.get("voice", current_voice)
                    logging.info(f"Voice changed to: {current_voice}")
                elif cmd == "set_volume":
                    current_volume = max(0.0, min(1.0, msg.get("volume", current_volume)))
                    logging.info(f"Volume changed to: {current_volume}")
        except Exception as e:
            logging.error(f"TTS worker error: {e}")


class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0, force_engine: str = None):
        """
        Initialize TTS with a separate worker process to avoid blocking the UI.

        Args:
            voice: Voice name
            volume: Volume (0.0-1.0)
            force_engine: Force specific engine ("kokoro" or "bark"), or None for auto-fallback
        """
        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))  # Clamp volume to 0.0-1.0
        self.force_engine = force_engine
        self.tts_engine = "worker"  # Using worker process

        # Multiprocessing components
        self._queue = None
        self._process = None
        self._started = False

        # Start the worker process
        self._start_worker()

    def _start_worker(self):
        """Start the TTS worker process"""
        if self._started:
            return

        try:
            # Use spawn context for Windows compatibility
            ctx = multiprocessing.get_context('spawn')
            self._queue = ctx.Queue()
            self._process = ctx.Process(
                target=_tts_worker_process,
                args=(self._queue, self.voice, self.volume, self.force_engine),
                daemon=True
            )
            self._process.start()
            self._started = True
            logging.info(f"✓ TTS worker process started (PID: {self._process.pid})")
        except Exception as e:
            logging.error(f"❌ Failed to start TTS worker process: {e}")
            self._started = False

    def set_voice(self, voice: str):
        """Change voice dynamically by notifying the worker process"""
        self.voice = voice
        if self._started and self._queue:
            self._queue.put({"cmd": "set_voice", "voice": voice})
        logging.info(f"Voice changed to: {voice}")

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0) by notifying the worker process"""
        self.volume = max(0.0, min(1.0, volume))
        if self._started and self._queue:
            self._queue.put({"cmd": "set_volume", "volume": self.volume})
        logging.info(f"Volume changed to: {self.volume}")

    def speak(self, text: str):
        """Queue text for speaking in the worker process (non-blocking)"""
        if not text:
            logging.debug("No text provided to speak.")
            return

        if not self._started:
            logging.error("TTS worker not started, cannot speak.")
            return

        # Queue the speak request (non-blocking)
        try:
            self._queue.put({"cmd": "speak", "text": text})
            logging.info(f"Queued TTS: {text[:100]}...")
        except Exception as e:
            logging.error(f"Failed to queue TTS: {e}")

    def shutdown(self):
        """Shut down the TTS worker process"""
        if self._started and self._queue and self._process:
            try:
                self._queue.put(None)  # Send shutdown signal
                self._process.join(timeout=5)  # Wait up to 5 seconds
                if self._process.is_alive():
                    self._process.terminate()
                logging.info("TTS worker process shut down")
            except Exception as e:
                logging.error(f"Error shutting down TTS worker: {e}")
            finally:
                self._started = False

    def __del__(self):
        """Clean up on destruction"""
        self.shutdown()

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
    from ..config.config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")


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

        # Detect draft events using compiled regex for efficiency
        if DRAFT_EVENT_PATTERN.search(log_line):
            detected.append({
                "text": "DRAFT_EVENT",
                "type": "draft_event",
                "resolved_name": None,
                "color": self.colors["draft_event"],
                "position": (0, 0),  # Metadata only, not positioned
            })

        # Detect game events using compiled regex for efficiency
        if GAME_EVENT_PATTERN.search(log_line):
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
    def __init__(self, root, advisor_ref, version="Unknown"):
        self.root = root
        self.advisor_ref = advisor_ref
        self.advisor = advisor_ref
        self.version = version

        self._last_issue_title = None
        self._last_timestamp = None

        self.prefs = UserPreferences.load() if CONFIG_MANAGER_AVAILABLE else None

        self.root.title(f"MTGA Voice Advisor v{self.version}")

        # Set custom taskbar/window icon
        self._set_window_icon()

        geometry = self.prefs.window_geometry if self.prefs else "900x700"
        self.root.geometry(geometry)
        always_on_top = self.prefs.always_on_top if self.prefs else True
        self.root.attributes('-topmost', always_on_top)
        self.root.configure(bg='#2b2b2b')

        self.bg_color = '#2b2b2b'
        self.fg_color = '#ffffff'
        self.accent_color = '#00ff88'
        self.success_color = '#00ff88'
        self.warning_color = '#ff5555'
        self.info_color = '#55aaff'

        self._create_widgets()
        self._initialize_secondary_windows()

        self.message_queue = deque(maxlen=100)
        self.board_state_lines = ["="*70, "⏳ WAITING FOR MATCH...", "="*70]
        self.rag_panel_expanded = False

        # Performance: Batched UI update system with dirty flags
        self._pending_updates = {}  # key -> value for pending updates
        self._update_scheduled = False

        self.root.bind('<F12>', lambda e: self._capture_bug_report())
        # Bind push-to-talk hotkey from preferences (default: space)
        # Convert display name to Tkinter binding format
        hotkey_display = self.prefs.push_to_talk_key if self.prefs else "space"
        hotkey_binding = self._display_to_binding(hotkey_display)
        self._current_hotkey_binding = hotkey_binding
        self.root.bind(hotkey_binding, self._on_push_to_talk)
        logging.info(f"Push-to-talk bound to: {hotkey_display} (binding: {hotkey_binding})")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Push-to-talk callback (set by app.py)
        self._push_to_talk_callback = None

        self.running = True
        self._update_loop()
        self._process_log_queue()

        self.root.after(100, self._initial_ui_setup)
        self.root.after(500, self._ensure_windows_visible)

    def _set_window_icon(self):
        """Set the window and taskbar icon."""
        try:
            from pathlib import Path
            # Find the icon file relative to this script
            script_dir = Path(__file__).parent.parent.parent  # Go up from src/core to project root
            icon_path = script_dir / "assets" / "icon.ico"

            # On Windows, set AppUserModelID to show custom icon in taskbar
            if os.name == 'nt':
                try:
                    import ctypes
                    # Set a unique AppUserModelID for this application
                    app_id = 'MTGAVoiceAdvisor.App.1.0'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
                    logging.debug(f"Set Windows AppUserModelID: {app_id}")
                except Exception as e:
                    logging.warning(f"Could not set AppUserModelID: {e}")

            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
                logging.debug(f"Window icon set from {icon_path}")
            else:
                # Try PNG as fallback (works on some platforms)
                png_path = script_dir / "assets" / "icon.png"
                if png_path.exists():
                    from PIL import Image, ImageTk
                    img = Image.open(png_path)
                    photo = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, photo)
                    self._icon_photo = photo  # Keep reference to prevent garbage collection
                    logging.debug(f"Window icon set from {png_path}")
                else:
                    logging.debug(f"No icon found at {icon_path} or {png_path}")
        except Exception as e:
            logging.warning(f"Could not set window icon: {e}")

    def _initial_ui_setup(self):
        """Set initial UI state from preferences."""
        if self.prefs:
            # Set the saved model first so _on_provider_change doesn't override it
            self.model_var.set(self.prefs.current_model)
            self.provider_var.set(self.prefs.model_provider)
            self._on_provider_change(skip_model_reset=True)  # Don't reset model to default

            api_key = ""
            provider = self.prefs.model_provider.lower()
            if provider == 'google':
                api_key = self.prefs.google_api_key
            elif provider == 'openai':
                api_key = self.prefs.openai_api_key
            elif provider == 'anthropic':
                api_key = self.prefs.anthropic_api_key
            self.api_key_var.set(api_key)

            # Also set voice from preferences
            if hasattr(self, 'voice_var') and self.prefs.current_voice:
                # We'll set this after voice_dropdown is populated by update_settings
                self._pending_voice = self.prefs.current_voice

            logging.info(f"UI loaded preferences: provider={self.prefs.model_provider}, model={self.prefs.current_model}")

    def _initialize_secondary_windows(self):
        """Create and configure secondary windows with close handlers."""
        # Track which windows are popped out (True = separate window, False = docked/embedded)
        # Default to docked (False) - load from preferences
        self._board_popped_out = not (self.prefs.board_window_docked if self.prefs and hasattr(self.prefs, 'board_window_docked') else True)
        self._deck_popped_out = not (self.prefs.deck_window_docked if self.prefs and hasattr(self.prefs, 'deck_window_docked') else True)
        self._log_popped_out = not (self.prefs.log_window_docked if self.prefs and hasattr(self.prefs, 'log_window_docked') else True)

        self.board_window = SecondaryWindow(self.root, "Board State",
            self.prefs.board_window_geometry if self.prefs and hasattr(self.prefs, 'board_window_geometry') else "600x800+50+50",
            on_close=self._on_board_window_close)
        self.deck_window = SecondaryWindow(self.root, "Library",
            self.prefs.deck_window_geometry if self.prefs and hasattr(self.prefs, 'deck_window_geometry') else "400x600+700+50",
            on_close=self._on_deck_window_close)
        self.log_window = SecondaryWindow(self.root, "MTGA Logs",
            self.prefs.log_window_geometry if self.prefs and hasattr(self.prefs, 'log_window_geometry') else "800x200+50+700",
            on_close=self._on_log_window_close)

    def _on_board_window_close(self):
        """Handle board window close - save geometry and show embedded version."""
        if self.board_window and self.board_window.winfo_exists():
            # Save geometry before hiding
            if self.prefs:
                self.prefs.board_window_geometry = self.board_window.geometry()
                self.prefs.board_window_docked = True  # Save docked state
            self.board_window.withdraw()
            self._board_popped_out = False
            # Show embedded board panel in paned window
            if hasattr(self, '_embedded_board_frame') and hasattr(self, '_content_paned'):
                self._content_paned.add(self._embedded_board_frame, stretch="always", minsize=80)
                # Copy current content to embedded view
                if self.board_state_lines:
                    self._update_embedded_board(self.board_state_lines)

    def _on_deck_window_close(self):
        """Handle deck window close - save geometry and show embedded version."""
        if self.deck_window and self.deck_window.winfo_exists():
            if self.prefs:
                self.prefs.deck_window_geometry = self.deck_window.geometry()
                self.prefs.deck_window_docked = True  # Save docked state
            self.deck_window.withdraw()
            self._deck_popped_out = False
            if hasattr(self, '_embedded_deck_frame') and hasattr(self, '_content_paned'):
                self._content_paned.add(self._embedded_deck_frame, stretch="always", minsize=80)

    def _on_log_window_close(self):
        """Handle log window close - save geometry and show embedded version."""
        if self.log_window and self.log_window.winfo_exists():
            if self.prefs:
                self.prefs.log_window_geometry = self.log_window.geometry()
                self.prefs.log_window_docked = True  # Save docked state
            self.log_window.withdraw()
            self._log_popped_out = False
            if hasattr(self, '_embedded_log_frame') and hasattr(self, '_content_paned'):
                self._content_paned.add(self._embedded_log_frame, stretch="never", minsize=60)

    def _pop_out_board(self):
        """Pop out the board panel to a separate window."""
        if hasattr(self, '_embedded_board_frame') and hasattr(self, '_content_paned'):
            try:
                self._content_paned.forget(self._embedded_board_frame)
            except tk.TclError:
                pass  # Already removed
        self._board_popped_out = True
        if self.prefs:
            self.prefs.board_window_docked = False  # Save popped out state
        if self.board_window:
            self.board_window.deiconify()
            self.board_window.lift()

    def _pop_out_deck(self):
        """Pop out the deck panel to a separate window."""
        if hasattr(self, '_embedded_deck_frame') and hasattr(self, '_content_paned'):
            try:
                self._content_paned.forget(self._embedded_deck_frame)
            except tk.TclError:
                pass  # Already removed
        self._deck_popped_out = True
        if self.prefs:
            self.prefs.deck_window_docked = False  # Save popped out state
        if self.deck_window:
            self.deck_window.deiconify()
            self.deck_window.lift()

    def _pop_out_log(self):
        """Pop out the log panel to a separate window."""
        if hasattr(self, '_embedded_log_frame') and hasattr(self, '_content_paned'):
            try:
                self._content_paned.forget(self._embedded_log_frame)
            except tk.TclError:
                pass  # Already removed
        self._log_popped_out = True
        if self.prefs:
            self.prefs.log_window_docked = False  # Save popped out state
        if self.log_window:
            self.log_window.deiconify()
            self.log_window.lift()

    def _ensure_windows_visible(self):
        """Show windows based on saved docked state preferences."""
        try:
            # Only show popped-out windows if they were previously popped out
            if self._board_popped_out and self.board_window:
                self.board_window.deiconify()
            elif not self._board_popped_out:
                # Dock the board window
                if self.board_window:
                    self.board_window.withdraw()
                if hasattr(self, '_embedded_board_frame') and hasattr(self, '_content_paned'):
                    self._content_paned.add(self._embedded_board_frame, stretch="always", minsize=80)

            if self._deck_popped_out and self.deck_window:
                self.deck_window.deiconify()
            elif not self._deck_popped_out:
                # Dock the deck window
                if self.deck_window:
                    self.deck_window.withdraw()
                if hasattr(self, '_embedded_deck_frame') and hasattr(self, '_content_paned'):
                    self._content_paned.add(self._embedded_deck_frame, stretch="always", minsize=80)

            if self._log_popped_out and self.log_window:
                # Keep log window hidden by default - user must click "Logs" button to show
                self.log_window.withdraw()
            elif not self._log_popped_out:
                # Dock the log window
                if self.log_window:
                    self.log_window.withdraw()
                if hasattr(self, '_embedded_log_frame') and hasattr(self, '_content_paned'):
                    self._content_paned.add(self._embedded_log_frame, stretch="never", minsize=60)
        except Exception as e:
            logging.error(f"Error setting up window visibility: {e}")

    def _create_widgets(self):
        """Create all GUI widgets"""
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=30)
        status_frame.pack(side=tk.TOP, fill=tk.X)
        status_frame.pack_propagate(False)
        self.status_label = tk.Label(status_frame, text="Initializing...", bg='#1a1a1a', fg=self.accent_color, font=('Consolas', 10, 'bold'), anchor=tk.W, padx=10)
        self.status_label.pack(fill=tk.BOTH, expand=True)

        settings_frame = tk.Frame(self.root, bg=self.bg_color, width=250)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        settings_frame.pack_propagate(False)

        tk.Label(settings_frame, text="⚙ SETTINGS", bg=self.bg_color, fg=self.accent_color, font=('Consolas', 12, 'bold')).pack(pady=(0, 10))

        # --- AI Provider and Model Selection ---
        tk.Label(settings_frame, text="AI Provider:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.provider_var = tk.StringVar()
        self.provider_dropdown = ttk.Combobox(settings_frame, textvariable=self.provider_var, values=["Google", "OpenAI", "Anthropic", "Ollama"], width=25)
        self.provider_dropdown.pack(pady=(0, 5), fill=tk.X)
        self.provider_dropdown.bind('<<ComboboxSelected>>', self._on_provider_change)

        self.api_key_frame = tk.Frame(settings_frame, bg=self.bg_color)
        tk.Label(self.api_key_frame, text="API Key:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = tk.Entry(self.api_key_frame, textvariable=self.api_key_var, show="*", width=27)
        self.api_key_entry.pack(pady=(0, 5), fill=tk.X)
        self.api_key_entry.bind('<KeyRelease>', self._on_api_key_change)

        self.ollama_frame = tk.Frame(settings_frame, bg=self.bg_color)
        self.check_ollama_btn = tk.Button(self.ollama_frame, text="Check Ollama & Refresh Models", command=self._check_ollama, bg='#3a3a3a', fg='white', relief=tk.FLAT)
        self.check_ollama_btn.pack(pady=2, fill=tk.X)

        tk.Label(settings_frame, text="AI Model:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(settings_frame, textvariable=self.model_var, width=25)
        self.model_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.model_dropdown.bind('<<ComboboxSelected>>', self._on_model_change)
        self.model_dropdown.bind('<Return>', self._on_model_change)

        # Voice selection
        tk.Label(settings_frame, text="Voice:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.voice_var = tk.StringVar()
        self.voice_dropdown = ttk.Combobox(settings_frame, textvariable=self.voice_var, state='readonly', width=25)
        self.voice_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.voice_dropdown.bind('<<ComboboxSelected>>', self._on_voice_change)

        # Volume slider
        tk.Label(settings_frame, text="Volume:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        volume_frame = tk.Frame(settings_frame, bg=self.bg_color)
        volume_frame.pack(pady=(0, 10), fill=tk.X)

        self.volume_var = tk.IntVar(value=100)
        self.volume_slider = tk.Scale(volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.volume_var, command=self._on_volume_change, bg=self.bg_color, fg=self.fg_color, highlightthickness=0, troughcolor='#1a1a1a')
        self.volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.volume_label = tk.Label(volume_frame, text="100%", bg=self.bg_color, fg=self.fg_color, width=5)
        self.volume_label.pack(side=tk.RIGHT)

        # Checkboxes
        always_on_top_default = self.prefs.always_on_top if self.prefs else True
        self.always_on_top_var = tk.BooleanVar(value=always_on_top_default)
        tk.Checkbutton(settings_frame, text="Always on Top", variable=self.always_on_top_var, command=self._on_always_on_top_toggle, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        # Voice Toggles
        self.verbose_speech_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Verbose Spoken Advice", variable=self.verbose_speech_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        self.mute_all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Mute All Audio", variable=self.mute_all_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        # --- Hotkey Settings ---
        tk.Label(settings_frame, text="Request Advice (Hotkey):", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W, pady=(10, 0))
        hotkey_frame = tk.Frame(settings_frame, bg=self.bg_color)
        hotkey_frame.pack(fill=tk.X, pady=2)

        # Get current hotkey from preferences
        current_hotkey = self.prefs.push_to_talk_key if self.prefs else "space"
        self.hotkey_var = tk.StringVar(value=current_hotkey)
        self.hotkey_display = tk.Label(hotkey_frame, textvariable=self.hotkey_var, bg='#2a2a2a', fg=self.accent_color,
                                       font=('Consolas', 10, 'bold'), width=12, relief=tk.SUNKEN, padx=5)
        self.hotkey_display.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(hotkey_frame, text="Set", command=self._set_hotkey, bg='#3a3a3a', fg='white', relief=tk.FLAT, width=5).pack(side=tk.LEFT)

        # Audio output device selection (speakers)
        tk.Label(settings_frame, text="Speakers:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W, pady=(5, 0))
        self.audio_output_var = tk.StringVar(value=self.prefs.audio_output_device if self.prefs and self.prefs.audio_output_device else "System Default")
        self.audio_output_dropdown = ttk.Combobox(settings_frame, textvariable=self.audio_output_var, width=25, state='readonly')
        self.audio_output_dropdown.pack(pady=2, fill=tk.X)
        self.audio_output_dropdown.bind('<<ComboboxSelected>>', self._on_audio_output_change)

        # Populate audio devices on startup
        self._populate_audio_devices()

        # --- Remaining widgets ---
        # (This part is condensed for brevity, assuming it remains largely the same)
        tk.Label(settings_frame, text="Windows:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W, pady=(10, 0))
        windows_frame = tk.Frame(settings_frame, bg=self.bg_color)
        windows_frame.pack(fill=tk.X, pady=5)
        tk.Button(windows_frame, text="Board", command=self._pop_out_board, bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(windows_frame, text="Deck", command=self._pop_out_deck, bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(windows_frame, text="Logs", command=self._pop_out_log, bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(settings_frame, text="Clear Messages", command=self._clear_messages, bg='#3a3a3a', fg=self.fg_color, relief=tk.FLAT, padx=10, pady=5).pack(pady=(20, 5), fill=tk.X)
        tk.Button(settings_frame, text="🐛 Bug Report (F12)", command=self._capture_bug_report, bg='#5555ff', fg=self.fg_color, relief=tk.FLAT, padx=10, pady=5).pack(pady=5, fill=tk.X)

        # Unknown cards warning frame (initially hidden)
        self._unknown_cards_frame = tk.Frame(settings_frame, bg='#553300')
        self._unknown_cards_label = tk.Label(
            self._unknown_cards_frame,
            text="⚠ Unknown cards detected!",
            bg='#553300', fg='#ffaa00',
            font=('Consolas', 9, 'bold'),
            wraplength=220
        )
        self._unknown_cards_label.pack(pady=5, padx=5)
        tk.Button(
            self._unknown_cards_frame,
            text="📥 Update Card Database",
            command=self._update_card_database,
            bg='#ffaa00', fg='#1a1a1a',
            relief=tk.FLAT, padx=10, pady=5,
            font=('Consolas', 9, 'bold')
        ).pack(pady=(0, 5), fill=tk.X, padx=5)
        # Don't pack yet - shown when unknown cards are detected

        button_frame = tk.Frame(settings_frame, bg=self.bg_color)
        button_frame.pack(pady=5, fill=tk.X)
        tk.Button(button_frame, text="🔄 Restart App", command=self._on_restart, bg=self.info_color, fg='#1a1a1a', relief=tk.FLAT, padx=10, pady=5, font=('Consolas', 9, 'bold')).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Exit", command=self._on_exit, bg=self.warning_color, fg=self.fg_color, relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Version label
        tk.Label(settings_frame, text=f"v{self.version}", bg=self.bg_color, fg='#666666', font=('Consolas', 8)).pack(side=tk.BOTTOM, pady=5)

        # Main content area with resizable panes
        self._content_paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg=self.bg_color,
            sashwidth=6, sashrelief=tk.RAISED, sashpad=2)
        self._content_paned.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Messages pane (always visible)
        messages_frame = tk.Frame(self._content_paned, bg=self.bg_color)
        self.advisor_label = tk.Label(messages_frame, text="═══ ADVISOR MESSAGES ═══", bg=self.bg_color, fg=self.accent_color, font=('Consolas', 10, 'bold'))
        self.advisor_label.pack(pady=(0, 5))
        self.messages_text = scrolledtext.ScrolledText(messages_frame, height=20, bg='#1a1a1a', fg=self.fg_color, font=('Consolas', 9), relief=tk.FLAT, padx=10, pady=10)
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        self.messages_text.config(state=tk.DISABLED)
        self.messages_text.tag_config('green', foreground='#00ff88')
        self.messages_text.tag_config('blue', foreground='#55aaff')
        self.messages_text.tag_config('cyan', foreground='#00ffff')
        self.messages_text.tag_config('yellow', foreground='#ffff00')
        self.messages_text.tag_config('red', foreground='#ff5555')
        self.messages_text.tag_config('white', foreground='#ffffff')
        self._content_paned.add(messages_frame, stretch="always", minsize=100)

        # Create embedded panels (initially hidden - added to paned window when windows are closed)
        self._create_embedded_panels()

        self.log_highlighter = LogHighlighter(card_db=None)

    def _create_embedded_panels(self):
        """Create embedded panels that show when secondary windows are closed."""
        # Embedded Board State Panel
        self._embedded_board_frame = tk.LabelFrame(self._content_paned, text="📋 Board State (click 'Board' to pop out)",
            bg='#1a1a1a', fg=self.accent_color, font=('Consolas', 9, 'bold'))
        self._embedded_board_text = scrolledtext.ScrolledText(self._embedded_board_frame, height=8,
            bg='#1a1a1a', fg=self.fg_color, font=('Consolas', 8), relief=tk.FLAT, padx=5, pady=5)
        self._embedded_board_text.pack(fill=tk.BOTH, expand=True)
        self._embedded_board_text.config(state=tk.DISABLED)
        # Don't add to paned window yet - will be added when board window is closed

        # Embedded Deck/Library Panel
        self._embedded_deck_frame = tk.LabelFrame(self._content_paned, text="📚 Library (click 'Deck' to pop out)",
            bg='#1a1a1a', fg=self.accent_color, font=('Consolas', 9, 'bold'))
        self._embedded_deck_text = scrolledtext.ScrolledText(self._embedded_deck_frame, height=6,
            bg='#1a1a1a', fg=self.fg_color, font=('Consolas', 8), relief=tk.FLAT, padx=5, pady=5)
        self._embedded_deck_text.pack(fill=tk.BOTH, expand=True)
        self._embedded_deck_text.config(state=tk.DISABLED)
        # Don't add yet

        # Embedded Log Panel
        self._embedded_log_frame = tk.LabelFrame(self._content_paned, text="📜 MTGA Logs (click 'Logs' to pop out)",
            bg='#1a1a1a', fg=self.accent_color, font=('Consolas', 9, 'bold'))
        self._embedded_log_text = scrolledtext.ScrolledText(self._embedded_log_frame, height=4,
            bg='#1a1a1a', fg=self.fg_color, font=('Consolas', 8), relief=tk.FLAT, padx=5, pady=5)
        self._embedded_log_text.pack(fill=tk.BOTH, expand=True)
        self._embedded_log_text.config(state=tk.DISABLED)
        # Don't add yet

    def _update_embedded_board(self, lines):
        """Update the embedded board state panel."""
        if hasattr(self, '_embedded_board_text'):
            self._embedded_board_text.config(state=tk.NORMAL)
            self._embedded_board_text.delete(1.0, tk.END)
            self._embedded_board_text.insert(tk.END, '\n'.join(lines))
            self._embedded_board_text.config(state=tk.DISABLED)

    def _update_embedded_deck(self, lines):
        """Update the embedded deck panel."""
        if hasattr(self, '_embedded_deck_text'):
            self._embedded_deck_text.config(state=tk.NORMAL)
            self._embedded_deck_text.delete(1.0, tk.END)
            self._embedded_deck_text.insert(tk.END, '\n'.join(lines))
            self._embedded_deck_text.config(state=tk.DISABLED)

    def _update_embedded_log(self, text):
        """Update the embedded log panel."""
        if hasattr(self, '_embedded_log_text'):
            self._embedded_log_text.config(state=tk.NORMAL)
            self._embedded_log_text.delete(1.0, tk.END)
            self._embedded_log_text.insert(tk.END, text)
            self._embedded_log_text.see(tk.END)
            self._embedded_log_text.config(state=tk.DISABLED)

    def _on_provider_change(self, event=None, skip_model_reset=False):
        """Handle provider selection change.

        Args:
            event: Tkinter event (from combobox selection)
            skip_model_reset: If True, don't reset model to default (used when loading from prefs)
        """
        provider = self.provider_var.get()
        if not provider:
            return

        # Hide all conditional widgets first
        self.api_key_frame.pack_forget()
        self.ollama_frame.pack_forget()

        # Default model lists (updated December 2025)
        model_lists = {
            "Google": [
                "gemini-3-pro-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ],
            "OpenAI": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "Anthropic": [
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
            ],
            "Ollama": []
        }

        if provider == "Ollama":
            self.ollama_frame.pack(pady=2, fill=tk.X)
            self.model_dropdown['values'] = model_lists[provider]
            if not skip_model_reset and self.model_var.get() not in model_lists[provider]:
                 self.model_var.set("")
        else:
            self.api_key_frame.pack(pady=2, fill=tk.X)
            self.model_dropdown['values'] = model_lists[provider]
            if not skip_model_reset and self.model_var.get() not in model_lists[provider]:
                 self.model_var.set(model_lists[provider][0])

        # Only save and notify on user-initiated changes (not initial load)
        if not skip_model_reset:
            if self.prefs:
                self.prefs.set_model(self.model_var.get(), provider=provider)
            logging.info(f"AI Provider changed to: {provider}")
            self.add_message(f"AI Provider changed to: {provider}. Restart required to apply changes.", "green")

    def _on_api_key_change(self, event=None):
        """Handle API key entry change."""
        if not self.prefs: return
        provider = self.provider_var.get().lower()
        key = self.api_key_var.get()

        key_map = {
            "google": {"google_api_key": key},
            "openai": {"openai_api_key": key},
            "anthropic": {"anthropic_api_key": key},
        }
        if provider in key_map:
            self.prefs.set_api_keys(**key_map[provider])
            logging.debug(f"Updated API key for {provider}")

    def _check_ollama(self):
        """Check for Ollama installation and list available models."""
        self.add_message("Checking for Ollama installation...", "cyan")
        threading.Thread(target=self._check_ollama_thread, daemon=True).start()

    def _check_ollama_thread(self):
        """Background thread to check for Ollama."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            models = sorted([model['name'] for model in data.get('models', [])])

            def update_ui():
                if models:
                    self.add_message(f"Ollama found! {len(models)} models available.", "green")
                    self.model_dropdown['values'] = models
                    if self.prefs.current_model in models:
                        self.model_var.set(self.prefs.current_model)
                    else:
                        self.model_var.set(models[0])
                        self._on_model_change()
                else:
                    self.add_message("Ollama found, but no models installed.", "yellow")
                    self.add_message("Run 'ollama pull llama3.2' in your terminal.", "yellow")
                    self.model_dropdown['values'] = []
                    self.model_var.set("")
            self.root.after(0, update_ui)

        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: self.add_message("Ollama not found. Is it running at http://localhost:11434?", "red"))
        except Exception as e:
            self.root.after(0, lambda: self.add_message(f"Error checking Ollama: {e}", "red"))
            logging.error(f"Error checking Ollama: {e}")

    def _on_model_change(self, event=None):
        """Handle model selection change."""
        model = self.model_var.get()
        provider = self.provider_var.get()
        if model and self.prefs:
            self.prefs.set_model(model, provider=provider)
            logging.info(f"✓ AI Model changed to: {model}")
            self.add_message(f"AI Model changed to: {model}. Restart required to apply changes.", "green")

    def _on_voice_change(self, event=None):
        """Handle voice selection change"""
        try:
            display_name = self.voice_var.get()
            # Convert display name back to voice ID
            voice_id = getattr(self, '_voice_id_from_display', {}).get(display_name, display_name)
            logging.info(f"Voice changed to: {voice_id} ({display_name})")
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                self.advisor_ref.tts.set_voice(voice_id)
            if self.prefs:
                self.prefs.set_voice_name(voice_id)
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                # Use display name for the spoken message
                threading.Thread(target=lambda: self.advisor_ref.tts.speak(f"Voice changed to {display_name}"), daemon=True).start()
        except Exception as e:
            logging.error(f"Error changing voice: {e}")

    def append_log(self, text: str):
        """Append text to the log queue (thread-safe)"""
        if not hasattr(self, 'log_queue'):
            # PERFORMANCE FIX: Increase queue size 10x to prevent dropping log lines
            self.log_queue = deque(maxlen=10000)
        self.log_queue.append(text)

    def _process_log_queue(self):
        """Process queued log messages with adaptive batch sizing."""
        monitor = get_monitor()
        with monitor.measure("ui.process_log_queue"):
            try:
                if hasattr(self, 'log_queue') and self.log_queue:
                    queue_depth = len(self.log_queue)

                    # Adaptive batch size based on backlog
                    if queue_depth > 5000:
                        batch_size = 2000  # Aggressive catch-up mode
                        next_interval = 50  # Process faster when backlogged
                    elif queue_depth > 2000:
                        batch_size = 1000
                        next_interval = 100
                    elif queue_depth > 500:
                        batch_size = 500
                        next_interval = 150
                    else:
                        batch_size = min(100, queue_depth)
                        next_interval = 200  # Normal rate when caught up

                    # Show catching-up indicator for large backlogs
                    if queue_depth > 1000:
                        self.set_status(f"Processing logs... ({queue_depth} remaining)")
                        # Log performance metrics for debugging
                        if queue_depth % 1000 == 0:
                            logging.debug(f"Log queue depth: {queue_depth}, batch_size: {batch_size}")

                    if self.log_window and self.log_window.winfo_exists():
                        lines = []
                        for _ in range(min(batch_size, len(self.log_queue))):
                            lines.append(self.log_queue.popleft())

                        if lines:
                            self.log_window.append_batch(lines)

                        # Clear status when caught up
                        if queue_depth > 1000 and len(self.log_queue) <= 1000:
                            self.set_status("Ready")
                    else:
                        # Window doesn't exist - prevent unbounded growth
                        if queue_depth > 9000:
                            self.log_queue.clear()
                            logging.debug("Cleared log queue - window not available")

                    if self.root and self.root.winfo_exists():
                        self.root.after(next_interval, self._process_log_queue)
                else:
                    # No queue or empty - check again later at normal rate
                    if self.root and self.root.winfo_exists():
                        self.root.after(200, self._process_log_queue)

            except Exception as e:
                logging.error(f"Error processing log queue: {e}")
                if self.root and self.root.winfo_exists():
                    self.root.after(500, self._process_log_queue)

    def show_unknown_cards_warning(self, count: int):
        """Show the unknown cards warning in the UI.

        Called by ArenaCardDatabase when unknown card threshold is exceeded.
        Must be called from the main thread via root.after().
        """
        def _show():
            if hasattr(self, '_unknown_cards_frame'):
                self._unknown_cards_label.config(
                    text=f"⚠ {count} unknown cards!\nUpdate database for new sets."
                )
                self._unknown_cards_frame.pack(pady=5, fill=tk.X, before=self._get_button_frame())

        # Schedule on main thread if called from callback
        if self.root and self.root.winfo_exists():
            self.root.after(0, _show)

    def _get_button_frame(self):
        """Get the button frame widget for positioning."""
        # Find the frame containing Restart/Exit buttons
        for child in self._unknown_cards_frame.master.winfo_children():
            if isinstance(child, tk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, tk.Button) and "Restart" in str(subchild.cget('text')):
                        return child
        return None

    def hide_unknown_cards_warning(self):
        """Hide the unknown cards warning."""
        if hasattr(self, '_unknown_cards_frame'):
            self._unknown_cards_frame.pack_forget()

    def _update_card_database(self):
        """Run the card database update script."""
        import sys
        from tkinter import messagebox

        # Confirm with user
        result = messagebox.askyesno(
            "Update Card Database",
            "This will update the card database from your MTGA installation.\n\n"
            "Make sure MTGA is installed and has been run at least once.\n\n"
            "The app will restart after the update.\n\n"
            "Continue?",
            parent=self.root
        )

        if not result:
            return

        self.add_message("📥 Starting card database update...", "cyan")
        logging.info("User initiated card database update")

        def run_update():
            try:
                # Run the build script
                script_path = Path(__file__).parent.parent.parent / "tools" / "build_unified_card_database.py"

                if not script_path.exists():
                    self.root.after(0, lambda: self.add_message(f"❌ Build script not found at {script_path}", "red"))
                    return

                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    self.root.after(0, lambda: self.add_message("✅ Card database updated successfully!", "green"))
                    self.root.after(0, lambda: self.add_message("🔄 Restarting to load new cards...", "cyan"))
                    # Hide the warning
                    self.root.after(0, self.hide_unknown_cards_warning)
                    # Restart after a brief delay
                    self.root.after(1500, self._on_restart)
                else:
                    error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    self.root.after(0, lambda: self.add_message(f"❌ Update failed: {error_msg}", "red"))
                    logging.error(f"Card database update failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self.add_message("❌ Update timed out after 5 minutes", "red"))
            except Exception as e:
                self.root.after(0, lambda: self.add_message(f"❌ Update error: {e}", "red"))
                logging.error(f"Card database update error: {e}")

        # Run in background thread
        threading.Thread(target=run_update, daemon=True).start()

    def _on_push_to_talk(self, event=None):
        """Handle push-to-talk hotkey press for manual advice."""
        # Ignore if focus is in a text entry widget
        focused = self.root.focus_get()
        if focused and (isinstance(focused, tk.Entry) or
                       isinstance(focused, tk.Text) or
                       (hasattr(focused, 'winfo_class') and focused.winfo_class() in ('Entry', 'Text', 'TEntry'))):
            return  # Let the entry handle the keypress

        hotkey = self.hotkey_var.get() if hasattr(self, 'hotkey_var') else "space"
        logging.info(f"Push-to-talk triggered ({hotkey})")
        self.add_message("🎤 Requesting advice...", "cyan")

        if self._push_to_talk_callback:
            try:
                self._push_to_talk_callback()
            except Exception as e:
                logging.error(f"Push-to-talk callback error: {e}")
                self.add_message(f"❌ Error getting advice: {e}", "red")
        else:
            self.add_message("⚠ Advice system not ready", "yellow")

    def set_push_to_talk_callback(self, callback):
        """Set the callback for push-to-talk advice requests."""
        self._push_to_talk_callback = callback

    def _display_to_binding(self, display_name: str) -> str:
        """
        Convert a display name like 'F5' or 'Control+space' to Tkinter binding format.
        e.g., 'F5' -> '<F5>', 'Control+space' -> '<Control-space>'
        """
        if not display_name:
            return '<space>'

        # If it already looks like a binding, return as-is
        if display_name.startswith('<') and display_name.endswith('>'):
            return display_name

        # Handle modifier+key format (e.g., "Control+space", "Alt+F5")
        if '+' in display_name:
            parts = display_name.split('+')
            # Tkinter uses hyphen between modifiers and key
            return f"<{'-'.join(parts)}>"
        else:
            # Simple key
            return f"<{display_name}>"

    def _set_hotkey(self):
        """Open dialog to set a new push-to-talk hotkey."""
        # Create a small dialog to capture the new hotkey
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Push-to-Talk Hotkey")
        dialog.geometry("320x140")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 160
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 70
        dialog.geometry(f"+{x}+{y}")

        tk.Label(dialog, text="Press a key (F1-F12 recommended):", bg=self.bg_color, fg=self.fg_color,
                 font=('Consolas', 11)).pack(pady=10)
        tk.Label(dialog, text="Avoid Alt/Ctrl combos (conflict with menus)", bg=self.bg_color, fg='#888888',
                 font=('Consolas', 9)).pack()

        key_label = tk.Label(dialog, text="Waiting...", bg='#2a2a2a', fg=self.accent_color,
                             font=('Consolas', 14, 'bold'), width=20, relief=tk.SUNKEN)
        key_label.pack(pady=8)

        captured_key = [None]
        captured_binding = [None]  # The actual Tkinter binding string

        def on_key(event):
            # Ignore modifier-only keypresses
            if event.keysym in ('Control_L', 'Control_R', 'Alt_L', 'Alt_R',
                               'Shift_L', 'Shift_R', 'Meta_L', 'Meta_R'):
                return

            key = event.keysym

            # For function keys (F1-F12), ignore any modifier state
            # This prevents accidental Alt+F5 when user just wants F5
            is_function_key = key.startswith('F') and key[1:].isdigit()

            if is_function_key:
                # Function keys - no modifiers, just the key
                binding = f"<{key}>"
                display = key
            else:
                # For other keys, check modifiers
                # But be strict - only include if the modifier key is actually held
                modifiers = []

                # Check for actual modifier keys being held (not residual state)
                # On Windows, event.state can have spurious bits set
                # Only add modifiers if it's clearly intentional (non-function key + modifier)
                if event.state & 0x4:  # Control
                    modifiers.append("Control")
                if event.state & 0x20000:  # Alt on Windows (high bit)
                    modifiers.append("Alt")
                if event.state & 0x1 and key not in ('space', 'Return', 'Tab'):  # Shift
                    modifiers.append("Shift")

                # Build binding string (Tkinter format: <Modifier-key>)
                if modifiers:
                    binding = f"<{'-'.join(modifiers)}-{key}>"
                    display = f"{'+'.join(modifiers)}+{key}"
                else:
                    binding = f"<{key}>"
                    display = key

            captured_key[0] = display
            captured_binding[0] = binding
            key_label.config(text=display)

        def confirm():
            if captured_binding[0]:
                # Unbind old hotkey
                old_binding = getattr(self, '_current_hotkey_binding', '<space>')
                try:
                    self.root.unbind(old_binding)
                except:
                    pass

                # Update display and binding
                new_binding = captured_binding[0]
                new_display = captured_key[0]
                self.hotkey_var.set(new_display)
                self._current_hotkey_binding = new_binding

                # Bind new hotkey
                self.root.bind(new_binding, self._on_push_to_talk)

                # Save display name to preferences (for UI)
                if self.prefs:
                    self.prefs.set_push_to_talk_key(new_display)

                self.add_message(f"Hotkey set to: {new_display}", "green")
                logging.info(f"Push-to-talk hotkey changed to: {new_display} (binding: {new_binding})")

            dialog.destroy()

        dialog.bind('<Key>', on_key)

        btn_frame = tk.Frame(dialog, bg=self.bg_color)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Confirm", command=confirm, bg='#3a3a3a', fg='white', relief=tk.FLAT, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg='#3a3a3a', fg='white', relief=tk.FLAT, width=10).pack(side=tk.LEFT, padx=5)

        dialog.focus_set()

    def _populate_audio_devices(self):
        """Populate the audio device dropdown with available output devices."""
        output_devices = ["System Default"]

        def query_devices():
            """Query audio devices in background to avoid blocking UI."""
            nonlocal output_devices
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    # Include output devices (speakers)
                    if device['max_output_channels'] > 0:
                        output_devices.append(f"{device['name']}")
            except ImportError:
                logging.debug("sounddevice not available, using system default only")
            except Exception as e:
                logging.warning(f"Error querying audio devices: {e}")

            # Update dropdown on main thread
            def update_dropdowns():
                self.audio_output_dropdown['values'] = output_devices
                # Set current selection from preferences
                if self.prefs and self.prefs.audio_output_device:
                    if self.prefs.audio_output_device in output_devices:
                        self.audio_output_var.set(self.prefs.audio_output_device)

            self.root.after(0, update_dropdowns)

        # Run device query in background thread to avoid blocking UI
        threading.Thread(target=query_devices, daemon=True).start()

        # Set initial values (will be updated when thread completes)
        self.audio_output_dropdown['values'] = output_devices

    def _on_audio_output_change(self, event=None):
        """Handle audio output (speakers) device change."""
        device = self.audio_output_var.get()
        if device == "System Default":
            device = ""

        if self.prefs:
            self.prefs.set_audio_devices(output_device=device)

        self.add_message(f"✓ Speakers: {device or 'System Default'}", "green")
        logging.info(f"Audio output device changed to: {device or 'System Default'}")

    def _on_restart(self):
        """Handle restart button click - restarts the application."""
        import sys
        from pathlib import Path
        try:
            if self.prefs: self.prefs.save()
            self.add_message("🔄 Restarting application...", "cyan")
            logging.info("User initiated application restart")
            self.root.quit()

            # Use absolute paths for robust restart
            python_executable = Path(sys.executable).resolve()
            script_path = Path(sys.argv[0]).resolve()

            # Get the project root directory (where main.py lives)
            project_root = script_path.parent
            if script_path.name != 'main.py':
                # If running from a different entry point, find main.py
                possible_main = project_root / 'main.py'
                if possible_main.exists():
                    script_path = possible_main

            args = [str(python_executable), str(script_path)] + sys.argv[1:]
            logging.info(f"Restarting with: {' '.join(args)}")
            logging.info(f"Working directory: {project_root}")

            # Start new process with explicit working directory
            subprocess.Popen(args, cwd=str(project_root))
        except Exception as e:
            logging.error(f"Error restarting app: {e}")
            self.add_message(f"❌ Failed to restart: {e}", "red")

    def _on_exit(self):
        """Handle exit button click."""
        try:
            if self.prefs: self.prefs.save()
            self.root.quit()
        except Exception as e:
            logging.error(f"Error on exit: {e}")
            self.root.quit()

    def _capture_bug_report(self):
        """Capture bug report with screenshot, logs, and board state to local file only"""
        import threading
        import subprocess
        import time
        import os

        # Ask user for title only (required), description is optional and can be submitted via TTS
        issue_title = None
        user_description = None

        try:
            from tkinter import messagebox, simpledialog
            if self.root and self.root.winfo_exists():
                # Prompt for title (required)
                title_prompt = simpledialog.askstring(
                    "Bug Report Title",
                    "Enter a title for the bug report:",
                    parent=self.root
                )
                if title_prompt and title_prompt.strip():
                    issue_title = title_prompt.strip()
                elif title_prompt is not None:  # User clicked OK but left blank
                    # Use default if they explicitly want to proceed with no title
                    messagebox.showwarning(
                        "Title Required",
                        "A title is required for the bug report. Using default.",
                        parent=self.root
                    )
                else:  # User clicked Cancel
                    self.add_message("Bug report cancelled", "yellow")
                    return
        except (ImportError, Exception) as e:
            logging.debug(f"GUI not available for bug report details: {e}")

        # Notify user we're starting the capture
        self.add_message("📸 Capturing bug report...", "cyan")

        def capture_in_background():
            nonlocal user_description
            try:
                # Create bug_reports directory if it doesn't exist
                base_bug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bug_reports")
                os.makedirs(base_bug_dir, exist_ok=True)

                # Find next available Bug Report ID
                existing_reports = [d for d in os.listdir(base_bug_dir) if os.path.isdir(os.path.join(base_bug_dir, d)) and d.startswith("Bug Report #")]
                max_id = 0
                for report in existing_reports:
                    try:
                        report_id = int(report.split("#")[1])
                        if report_id > max_id:
                            max_id = report_id
                    except (IndexError, ValueError):
                        continue
                
                next_id = max_id + 1
                folder_name = f"Bug Report #{next_id}"
                
                # Create specific folder for this report
                report_dir = os.path.join(base_bug_dir, folder_name)
                os.makedirs(report_dir, exist_ok=True)

                report_file = os.path.join(report_dir, "bug_report.txt")
                log_file_copy = os.path.join(report_dir, "advisor.log")

                # Use title from user input
                final_title = issue_title if issue_title else f"Bug Report #{next_id}"
                final_description = user_description if user_description else "No description provided."

                # --- SCREENSHOT CAPTURE ---
                import pyautogui
                try:
                    import pygetwindow as gw
                except ImportError:
                    gw = None
                    logging.warning("pygetwindow not available, window-specific screenshots might fail")

                # 1. Capture Full Desktop (Overview)
                try:
                    full_screenshot = pyautogui.screenshot()
                    full_screenshot.save(os.path.join(report_dir, "desktop_full.png"))
                    logging.info("Captured full desktop screenshot")
                except Exception as e:
                    logging.error(f"Failed to capture full desktop: {e}")

                # 2. Capture Advisor Main Window
                try:
                    if self.root and self.root.winfo_exists():
                        x = self.root.winfo_rootx()
                        y = self.root.winfo_rooty()
                        w = self.root.winfo_width()
                        h = self.root.winfo_height()
                        # Add a small buffer/border
                        region = (x, y, w, h)
                        main_shot = pyautogui.screenshot(region=region)
                        main_shot.save(os.path.join(report_dir, "advisor_main.png"))
                except Exception as e:
                    logging.error(f"Failed to capture main window: {e}")

                # 3. Capture Secondary Windows (Board, Deck, Logs)
                for win_name, win_obj in [("board", self.board_window), ("deck", self.deck_window), ("logs", self.log_window)]:
                    try:
                        if win_obj and win_obj.winfo_exists() and win_obj.winfo_viewable():
                            x = win_obj.winfo_rootx()
                            y = win_obj.winfo_rooty()
                            w = win_obj.winfo_width()
                            h = win_obj.winfo_height()
                            region = (x, y, w, h)
                            shot = pyautogui.screenshot(region=region)
                            shot.save(os.path.join(report_dir, f"advisor_{win_name}.png"))
                    except Exception as e:
                        logging.error(f"Failed to capture {win_name} window: {e}")

                # 4. Capture MTGA Window
                mtga_found = False
                if gw:
                    try:
                        # Try common titles for MTGA
                        mtga_windows = gw.getWindowsWithTitle('Magic: The Gathering Arena')
                        if not mtga_windows:
                            mtga_windows = gw.getWindowsWithTitle('MTGA')
                        
                        if mtga_windows:
                            mtga_win = mtga_windows[0]
                            # Ensure it's not minimized (restore if needed? No, might disrupt user. Just check.)
                            if not mtga_win.isMinimized:
                                region = (mtga_win.left, mtga_win.top, mtga_win.width, mtga_win.height)
                                mtga_shot = pyautogui.screenshot(region=region)
                                mtga_shot.save(os.path.join(report_dir, "mtga_game.png"))
                                mtga_found = True
                                logging.info("Captured MTGA window")
                            else:
                                logging.warning("MTGA window found but is minimized")
                    except Exception as e:
                        logging.error(f"Failed to capture MTGA window: {e}")
                
                if not mtga_found:
                    logging.warning("MTGA window not found or could not be captured")


                # Collect current state
                board_state_text = "\n".join(self.board_state_lines) if self.board_state_lines else "No board state"

                # Read recent logs - optimized for context window limits
                # Full log saved to file, but summary in report is limited to ~100 lines
                recent_logs = ""
                log_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "advisor.log"),
                ]

                for log_path in log_paths:
                    try:
                        if os.path.exists(log_path):
                            with open(log_path, "r", encoding='utf-8', errors='replace') as f:
                                lines = f.readlines()
                                # Filter to only include ERROR and WARNING for summary
                                # Plus last 50 lines for context
                                error_lines = [l for l in lines if ' - ERROR - ' in l or ' - WARNING - ' in l]
                                tail_lines = lines[-500:]  # Save more to file for debugging

                                # For the text summary: errors/warnings + last 50 lines
                                summary_lines = error_lines[-30:] + ["\n--- LAST 50 LINES ---\n"] + lines[-50:]
                                recent_logs = "".join(summary_lines)

                            # Write tail to the full log file for detailed debugging
                            with open(log_file_copy, "w", encoding='utf-8') as f:
                                f.writelines(tail_lines)
                            break
                    except Exception as e:
                        logging.error(f"Error reading/copying log file: {e}")
                        continue

                if not recent_logs:
                    recent_logs = "(No logs found)"

                # Read recent MTGA logs (last 50 lines) for context
                mtga_log_snippet = ""
                mtga_log_path = None
                
                # Try to get log path from advisor reference
                if hasattr(self, 'advisor_ref') and self.advisor_ref:
                    if hasattr(self.advisor_ref, 'log_follower') and self.advisor_ref.log_follower:
                        mtga_log_path = self.advisor_ref.log_follower.log_path
                    elif hasattr(self.advisor_ref, 'log_path'):
                        mtga_log_path = self.advisor_ref.log_path
                
                if mtga_log_path and os.path.exists(mtga_log_path):
                    try:
                         with open(mtga_log_path, "r", encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                            # Only include last 50 lines of MTGA logs for context
                            mtga_log_snippet = "".join(lines[-50:])
                    except Exception as e:
                        mtga_log_snippet = f"(Error reading MTGA log: {e})"
                else:
                    mtga_log_snippet = "(MTGA log path not available)"

                # Get current settings
                def safe_get_var(var_obj):
                    try:
                        if hasattr(var_obj, 'get'):
                            return str(var_obj.get())
                    except:
                        return "N/A"
                    return "N/A"

                settings = f"""Version: {self.version}
Model: {safe_get_var(self.model_var) if hasattr(self, 'model_var') else 'N/A'}
Voice: {safe_get_var(self.voice_var) if hasattr(self, 'voice_var') else 'N/A'}
Volume: {safe_get_var(self.volume_var) if hasattr(self, 'volume_var') else 'N/A'}%
"""

                # Write summary report
                with open(report_file, "w", encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write(f"BUG REPORT: {final_title}\n")
                    f.write("="*70 + "\n\n")
                    f.write("USER DESCRIPTION:\n")
                    f.write(f"{final_description}\n\n")
                    f.write("="*70 + "\n")
                    f.write("CURRENT SETTINGS:\n")
                    f.write(settings + "\n")
                    f.write("="*70 + "\n")
                    f.write("CURRENT BOARD STATE:\n")
                    f.write(board_state_text + "\n\n")
                    f.write("="*70 + "\n")
                    f.write("RECENT LOGS (Snippet):\n")
                    f.write(recent_logs + "\n")
                    f.write("="*70 + "\n")
                    f.write("MTGA LOGS (Snippet):\n")
                    f.write(mtga_log_snippet + "\n")

                self.add_message(f"✓ Bug report saved to: {folder_name}", "green")
                logging.info(f"Bug report saved to {report_dir}")

                # Offer to upload to GitHub
                if self.root and self.root.winfo_exists():
                    self.root.after(0, lambda: self._offer_upload_to_github(report_dir, folder_name, final_title, final_description))

            except Exception as e:
                self.add_message(f"✗ Bug report failed: {e}", "red")
                logging.error(f"Failed to capture bug report: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # Run in background thread
        threading.Thread(target=capture_in_background, daemon=True).start()

    def _offer_upload_to_github(self, report_dir, folder_name, title, body):
        """Offer to upload the bug report to GitHub."""
        from tkinter import messagebox
        import shutil
        import webbrowser
        
        # Compress the report directory
        zip_path = f"{report_dir}.zip"
        shutil.make_archive(report_dir, 'zip', report_dir)
        
        msg = f"Bug report saved locally to:\n{folder_name}\n\nDo you want to open a GitHub issue for this?"
        if messagebox.askyesno("Upload to GitHub", msg, parent=self.root):
            
            # Check for GitHub CLI (gh)
            gh_available = False
            try:
                subprocess.run(["gh", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                gh_available = True
            except:
                pass

            if gh_available:
                # Use gh CLI to create issue
                self.add_message("🚀 Uploading to GitHub...", "cyan")
                
                def upload_with_gh():
                    try:
                        # Note: gh issue create doesn't support attaching files directly easily via CLI flags 
                        # without web interaction or using 'gh release upload' trickery.
                        # Best practice for CLI automation is usually just opening the web form OR
                        # creating the text issue and telling user to drag file.
                        # However, we can try to be helpful.
                        
                        # We will create the issue with the body text
                        # Then open the issue in browser so user can drag zip
                        
                        cmd = [
                            "gh", "issue", "create",
                            "--title", title,
                            "--body", f"{body}\n\n**Version:** {self.version}\n\n*(Please drag the generated zip file '{os.path.basename(zip_path)}' into this issue to attach screenshots and logs)*",
                            "--web"  # Open in browser to let user attach files
                        ]
                        subprocess.run(cmd, check=True)
                        
                        # Open folder so user can find the zip
                        if os.name == 'nt':
                            os.startfile(os.path.dirname(report_dir))
                        elif os.name == 'posix':
                             subprocess.run(['xdg-open', os.path.dirname(report_dir)])
                             
                    except Exception as e:
                        logging.error(f"GitHub upload failed: {e}")
                        self.root.after(0, lambda: self.add_message(f"GitHub CLI failed, opening browser...", "yellow"))
                        # Fallback to manual
                        self.root.after(0, lambda: self._manual_github_upload(title, body, zip_path))
                
                threading.Thread(target=upload_with_gh, daemon=True).start()
            else:
                # Fallback: Open browser + folder
                self._manual_github_upload(title, body, zip_path)

    def _manual_github_upload(self, title, body, zip_path):
        """Handle manual GitHub upload flow."""
        import urllib.parse
        import webbrowser
        
        repo_url = "https://github.com/josharmour/mtga-voice-assistant/issues/new"
        params = {
            "title": title,
            "body": f"{body}\n\n**Version:** {self.version}\n\n*(Please drag the attached zip file into this area)*"
        }
        query_string = urllib.parse.urlencode(params)
        full_url = f"{repo_url}?{query_string}"
        
        webbrowser.open(full_url)
        
        # Open file explorer to the zip location
        try:
            folder = os.path.dirname(zip_path)
            if os.name == 'nt':
                os.startfile(folder)
            elif os.name == 'posix':
                subprocess.run(['xdg-open', folder])
            elif os.name == 'mac':
                subprocess.run(['open', folder])
        except Exception as e:
            logging.error(f"Could not open file explorer: {e}")
            
        self.add_message("📂 Drag the zip file into the browser window", "green")


    def _on_volume_change(self, value):
        """Handle volume slider change."""
        volume = int(value)
        self.volume_label.config(text=f"{volume}%")
        if self.prefs: self.prefs.set_voice_settings(volume=volume)

    def _on_always_on_top_toggle(self):
        """Handle always on top toggle."""
        enabled = self.always_on_top_var.get()
        self.root.attributes('-topmost', enabled)
        if self.prefs:
            self.prefs.always_on_top = enabled
            self.prefs.save()

    def _clear_messages(self):
        """Clear the messages display."""
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete(1.0, tk.END)
        self.messages_text.config(state=tk.DISABLED)

    def on_closing(self):
        """Handle window close event."""
        try:
            if self.root and self.prefs:
                self.prefs.window_geometry = self.root.geometry()
                if self.deck_window and self.deck_window.winfo_exists(): self.prefs.deck_window_geometry = self.deck_window.geometry()
                if self.board_window and self.board_window.winfo_exists(): self.prefs.board_window_geometry = self.board_window.geometry()
                if self.log_window and self.log_window.winfo_exists(): self.prefs.log_window_geometry = self.log_window.geometry()
                self.prefs.save()
            self.root.destroy()
        except Exception as e:
            logging.error(f"Error during window close: {e}")
            if self.root: self.root.destroy()

    def _update_loop(self):
        """Main GUI update loop."""
        if self.root and self.root.winfo_exists():
            self.root.after(100, self._update_loop)

    # ----------------------------------------------------------------------------------
    # Performance: Batched UI Update System
    # ----------------------------------------------------------------------------------

    def _schedule_update(self, key: str, value):
        """
        Queue an update to be applied in the next batch.

        This method batches UI updates to reduce scheduler overhead from excessive
        root.after() calls. Updates are coalesced by key and flushed at ~60fps (16ms).

        Args:
            key: Update identifier (e.g., "status", "board_state", "deck_content")
            value: The value to update (type depends on key)
        """
        self._pending_updates[key] = value
        if not self._update_scheduled:
            self._update_scheduled = True
            # Flush updates at ~60fps (16ms) for smooth UI
            self.root.after(16, self._flush_updates)

    def _flush_updates(self):
        """
        Apply all pending updates in one batch.

        This method processes all queued updates and applies them to the UI,
        then clears the pending updates dictionary. Called automatically by
        _schedule_update() after a brief delay to batch rapid successive updates.
        """
        monitor = get_monitor()
        with monitor.measure("ui.flush_updates"):
            self._update_scheduled = False
            updates = self._pending_updates.copy()
            self._pending_updates.clear()

            for key, value in updates.items():
                self._apply_update(key, value)

    def _apply_update(self, key: str, value):
        """
        Apply a single update by key.

        This method maps update keys to their corresponding UI update operations.
        All updates are executed on the main UI thread.

        Args:
            key: Update identifier
            value: The value to apply (type depends on key)
        """
        try:
            if key == "status":
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.config(text=value)

            elif key == "board_state":
                # Update secondary window if popped out
                if self._board_popped_out and self.board_window and self.board_window.winfo_exists():
                    self.board_window.update_text(value)
                # Also update embedded panel if visible
                elif not self._board_popped_out:
                    self._update_embedded_board(value)

            elif key == "deck_content":
                # Update secondary window if popped out
                # Use force_full_update to avoid diff issues when switching from draft picks to library
                if self._deck_popped_out and self.deck_window and self.deck_window.winfo_exists():
                    self.deck_window.force_full_update(value)
                # Also update embedded panel if visible
                elif not self._deck_popped_out:
                    self._update_embedded_deck(value)

            elif key == "deck_window_title":
                if self.deck_window and self.deck_window.winfo_exists():
                    self.deck_window.title(value)

            elif key == "draft_panes":
                # value is a tuple: (pack_lines, picked_lines, picked_count, total_needed)
                pack_lines, picked_lines, picked_count, total_needed = value
                # Update board/pack window
                if self._board_popped_out and self.board_window and self.board_window.winfo_exists():
                    self.board_window.update_text(pack_lines)
                elif not self._board_popped_out:
                    self._update_embedded_board(pack_lines)
                # Update deck/picked window
                if picked_lines:
                    deck_lines = [f"=== PICKED CARDS ({picked_count}/{total_needed}) ==="] + picked_lines
                    if self._deck_popped_out and self.deck_window and self.deck_window.winfo_exists():
                        self.deck_window.update_text(deck_lines)
                    elif not self._deck_popped_out:
                        self._update_embedded_deck(deck_lines)

            elif key == "messages":
                # value is a list of (msg, color) tuples
                if not hasattr(self, 'messages_text'):
                    return
                self.messages_text.config(state=tk.NORMAL)
                for msg, color in value:
                    timestamp = time.strftime("%H:%M:%S")
                    self.messages_text.insert(tk.END, f"[{timestamp}] ")
                    if color and isinstance(color, str):
                        color_tag = color.lower()
                        if color_tag in ['green', 'blue', 'cyan', 'yellow', 'red', 'white']:
                            self.messages_text.insert(tk.END, msg + '\n', color_tag)
                        else:
                            self.messages_text.insert(tk.END, msg + '\n')
                    else:
                        self.messages_text.insert(tk.END, msg + '\n')
                self.messages_text.config(state=tk.DISABLED)
                self.messages_text.see(tk.END)

            elif key == "settings":
                # value is a dict with: models, voices, voice_display_names, bark_voices, current_model, current_voice, volume, tts_engine
                # Store voice mappings for lookup
                self._voice_display_names = value.get('voice_display_names', {})
                self._voice_id_from_display = {v: k for k, v in self._voice_display_names.items()}

                # Build display names list for dropdown
                display_names = [self._voice_display_names.get(v, v) for v in value['voices']]
                display_names += list(value['bark_voices'])  # bark voices use their IDs as-is
                self.voice_dropdown['values'] = display_names

                # Set current voice using display name
                current_voice_id = value['current_voice']
                current_display = self._voice_display_names.get(current_voice_id, current_voice_id)
                self.voice_var.set(current_display)

                self.volume_var.set(value['volume'])
                self.volume_label.config(text=f"{value['volume']}%")
                logging.debug(f"GUI settings updated from advisor.")

        except Exception as e:
            logging.error(f"Error applying update for key '{key}': {e}")

    # ----------------------------------------------------------------------------------
    # End of Batched UI Update System
    # ----------------------------------------------------------------------------------

    def update_settings(self, models, voices, voice_display_names, bark_voices, current_model, current_voice, volume, tts_engine):
        """Update GUI settings with current values from advisor (batched, thread-safe)."""
        self._schedule_update("settings", {
            'models': models,
            'voices': voices,
            'voice_display_names': voice_display_names,
            'bark_voices': bark_voices,
            'current_model': current_model,
            'current_voice': current_voice,
            'volume': volume,
            'tts_engine': tts_engine
        })

    def set_status(self, text: str):
        """Update status label (batched, thread-safe)."""
        self._schedule_update("status", text)

    def set_card_database(self, card_db):
        """Set the card database for log highlighting."""
        if hasattr(self, 'log_highlighter'):
            self.log_highlighter.card_db = card_db

    def set_board_state(self, lines: list):
        """Update board state display in external window (batched, thread-safe)."""
        self._schedule_update("board_state", lines)

    def set_draft_panes(self, pack_lines, picked_lines=None, picked_count=0, total_needed=45):
        """Update draft display (batched, thread-safe)."""
        self._schedule_update("draft_panes", (pack_lines, picked_lines, picked_count, total_needed))

    def set_deck_window_title(self, title: str):
        """Update the title of the Deck/Library window (batched, thread-safe)."""
        self._schedule_update("deck_window_title", title)

    def set_deck_content(self, lines: list):
        """Update the content of the Deck/Library window (batched, thread-safe)."""
        self._schedule_update("deck_content", lines)

    def add_message(self, msg: str, color=None):
        """Add message to the advisor messages display (batched, thread-safe)."""
        # Batch messages together - append to existing messages list or create new one
        if "messages" not in self._pending_updates:
            self._pending_updates["messages"] = []
        self._pending_updates["messages"].append((msg, color))

        # Schedule flush if not already scheduled
        if not self._update_scheduled:
            self._update_scheduled = True
            self.root.after(16, self._flush_updates)
