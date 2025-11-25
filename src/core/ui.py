
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
    def __init__(self, root, advisor_ref):
        self.root = root
        self.advisor_ref = advisor_ref
        self.advisor = advisor_ref

        self._last_issue_title = None
        self._last_timestamp = None

        self.prefs = UserPreferences.load() if CONFIG_MANAGER_AVAILABLE else None

        self.root.title("MTGA Voice Advisor")
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
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.running = True
        self._update_loop()
        self._process_log_queue()

        self.root.after(100, self._initial_ui_setup)
        self.root.after(500, self._ensure_windows_visible)

    def _initial_ui_setup(self):
        """Set initial UI state from preferences."""
        if self.prefs:
            self.provider_var.set(self.prefs.model_provider)
            self._on_provider_change()
            self.model_var.set(self.prefs.current_model)

            api_key = ""
            provider = self.prefs.model_provider.lower()
            if provider == 'google':
                api_key = self.prefs.google_api_key
            elif provider == 'openai':
                api_key = self.prefs.openai_api_key
            elif provider == 'anthropic':
                api_key = self.prefs.anthropic_api_key
            self.api_key_var.set(api_key)

    def _initialize_secondary_windows(self):
        """Create and configure secondary windows with close handlers."""
        # Track which windows are popped out (True = separate window, False = embedded)
        self._board_popped_out = True
        self._deck_popped_out = True
        self._log_popped_out = True

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
            self.deck_window.withdraw()
            self._deck_popped_out = False
            if hasattr(self, '_embedded_deck_frame') and hasattr(self, '_content_paned'):
                self._content_paned.add(self._embedded_deck_frame, stretch="always", minsize=80)

    def _on_log_window_close(self):
        """Handle log window close - save geometry and show embedded version."""
        if self.log_window and self.log_window.winfo_exists():
            if self.prefs:
                self.prefs.log_window_geometry = self.log_window.geometry()
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
        if self.log_window:
            self.log_window.deiconify()
            self.log_window.lift()

    def _ensure_windows_visible(self):
        """Ensure secondary windows are visible on startup."""
        try:
            if self.board_window: self.board_window.deiconify()
            if self.deck_window: self.deck_window.deiconify()
            if self.log_window: self.log_window.deiconify()
        except Exception as e:
            logging.error(f"Error showing secondary windows: {e}")

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
        opponent_alerts_default = self.prefs.opponent_turn_alerts if self.prefs else True
        self.continuous_var = tk.BooleanVar(value=opponent_alerts_default)
        tk.Checkbutton(settings_frame, text="Opponent Turn Alerts", variable=self.continuous_var, command=self._on_continuous_toggle, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        show_thinking_default = self.prefs.show_thinking if self.prefs else True
        self.show_thinking_var = tk.BooleanVar(value=show_thinking_default)
        tk.Checkbutton(settings_frame, text="Show AI Thinking", variable=self.show_thinking_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        always_on_top_default = self.prefs.always_on_top if self.prefs else True
        self.always_on_top_var = tk.BooleanVar(value=always_on_top_default)
        tk.Checkbutton(settings_frame, text="Always on Top", variable=self.always_on_top_var, command=self._on_always_on_top_toggle, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        self.pick_two_draft_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Pick Two Draft", variable=self.pick_two_draft_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        # Voice Toggles
        self.verbose_speech_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Verbose Spoken Advice", variable=self.verbose_speech_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

        self.mute_all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Mute All Audio", variable=self.mute_all_var, bg=self.bg_color, fg=self.fg_color, selectcolor='#1a1a1a', activebackground=self.bg_color, activeforeground=self.fg_color).pack(anchor=tk.W, pady=2)

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
        button_frame = tk.Frame(settings_frame, bg=self.bg_color)
        button_frame.pack(pady=5, fill=tk.X)
        tk.Button(button_frame, text="🔄 Restart App", command=self._on_restart, bg=self.info_color, fg='#1a1a1a', relief=tk.FLAT, padx=10, pady=5, font=('Consolas', 9, 'bold')).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(button_frame, text="Exit", command=self._on_exit, bg=self.warning_color, fg=self.fg_color, relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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

    def _on_provider_change(self, event=None):
        """Handle provider selection change."""
        provider = self.provider_var.get()
        if not provider:
            return

        # Hide all conditional widgets first
        self.api_key_frame.pack_forget()
        self.ollama_frame.pack_forget()

        # Default model lists
        model_lists = {
            "Google": ["gemini-3-pro-preview", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
            "OpenAI": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "Ollama": []
        }

        if provider == "Ollama":
            self.ollama_frame.pack(pady=2, fill=tk.X)
            self.model_dropdown['values'] = model_lists[provider]
            if not self.model_var.get() in model_lists[provider]:
                 self.model_var.set("")
        else:
            self.api_key_frame.pack(pady=2, fill=tk.X)
            self.model_dropdown['values'] = model_lists[provider]
            if self.model_var.get() not in model_lists[provider]:
                 self.model_var.set(model_lists[provider][0])

        if self.prefs:
            self.prefs.set_model(self.model_var.get(), provider=provider)

        logging.info(f"✓ AI Provider changed to: {provider}")
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
            new_voice = self.voice_var.get()
            logging.info(f"Voice changed to: {new_voice}")
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                self.advisor_ref.tts.set_voice(new_voice)
            if self.prefs:
                self.prefs.set_voice_name(new_voice)
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                threading.Thread(target=lambda: self.advisor_ref.tts.speak(f"Voice changed to {new_voice.replace('_', ' ')}"), daemon=True).start()
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

    def _on_restart(self):
        """Handle restart button click - restarts the application."""
        import sys
        try:
            if self.prefs: self.prefs.save()
            self.add_message("🔄 Restarting application...", "cyan")
            logging.info("User initiated application restart")
            self.root.quit()
            python_executable = sys.executable
            script_path = os.path.abspath(sys.argv[0])
            args = [python_executable, script_path] + sys.argv[1:]
            logging.info(f"Restarting with: {' '.join(args)}")
            subprocess.Popen(args)
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

                # Read recent logs (last 1000 lines for better context)
                recent_logs = ""
                log_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "advisor.log"),
                ]

                for log_path in log_paths:
                    try:
                        if os.path.exists(log_path):
                            # Copy the full log file
                            import shutil
                            shutil.copy2(log_path, log_file_copy)
                            
                            # Also read for the summary text file
                            with open(log_path, "r", encoding='utf-8', errors='replace') as f:
                                lines = f.readlines()
                                recent_logs = "".join(lines[-1000:])
                            break
                    except Exception as e:
                        logging.error(f"Error reading/copying log file: {e}")
                        continue

                if not recent_logs:
                    recent_logs = "(No logs found)"

                # Read recent MTGA logs (last 200 lines) for context
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
                            mtga_log_snippet = "".join(lines[-200:])
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

                settings = f"""Model: {safe_get_var(self.model_var) if hasattr(self, 'model_var') else 'N/A'}
Voice: {safe_get_var(self.voice_var) if hasattr(self, 'voice_var') else 'N/A'}
Volume: {safe_get_var(self.volume_var) if hasattr(self, 'volume_var') else 'N/A'}%
Continuous Monitoring: {safe_get_var(self.continuous_var) if hasattr(self, 'continuous_var') else 'N/A'}
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

            except Exception as e:
                self.add_message(f"✗ Bug report failed: {e}", "red")
                logging.error(f"Failed to capture bug report: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # Run in background thread
        threading.Thread(target=capture_in_background, daemon=True).start()


    def _on_volume_change(self, value):
        """Handle volume slider change."""
        volume = int(value)
        self.volume_label.config(text=f"{volume}%")
        if self.prefs: self.prefs.set_voice_settings(volume=volume)

    def _on_continuous_toggle(self):
        """Handle opponent turn alerts toggle."""
        enabled = self.continuous_var.get()
        if self.prefs: self.prefs.set_game_preferences(opponent_alerts=enabled)

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
                if self._deck_popped_out and self.deck_window and self.deck_window.winfo_exists():
                    self.deck_window.update_text(value)
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
                # value is a dict with: models, voices, bark_voices, current_model, current_voice, volume, tts_engine
                self.voice_dropdown['values'] = list(value['voices']) + list(value['bark_voices'])
                self.voice_var.set(value['current_voice'])
                self.volume_var.set(value['volume'])
                self.volume_label.config(text=f"{value['volume']}%")
                logging.debug(f"GUI settings updated from advisor.")

        except Exception as e:
            logging.error(f"Error applying update for key '{key}': {e}")

    # ----------------------------------------------------------------------------------
    # End of Batched UI Update System
    # ----------------------------------------------------------------------------------

    def update_settings(self, models, voices, bark_voices, current_model, current_voice, volume, tts_engine):
        """Update GUI settings with current values from advisor (batched, thread-safe)."""
        self._schedule_update("settings", {
            'models': models,
            'voices': voices,
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
