
import logging
import os
import subprocess
import tempfile
from pathlib import Path
import time
from collections import deque
from typing import List, Callable
from .secondary_window import SecondaryWindow

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
        """Play audio directly using sounddevice (better for WSL)"""
        try:
            import sounddevice as sd
            
            logging.info(f"Playing audio with sounddevice ({engine_name})...")
            
            # Play audio directly (non-blocking)
            sd.play(audio_array, sample_rate)
            
            logging.info(f"Audio playback started successfully")
            
        except ImportError:
            logging.error("sounddevice not installed. Install with: pip install sounddevice")
            # Fallback to system players
            self._save_and_play_audio_fallback(audio_array, sample_rate, engine_name)
        except Exception as e:
            logging.error(f"sounddevice playback error: {e}")
            # Fallback to system players
            self._save_and_play_audio_fallback(audio_array, sample_rate, engine_name)
    
    def _save_and_play_audio_fallback(self, audio_array, sample_rate: int, engine_name: str):
        """Fallback: Save audio to temp file and play with system commands"""
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
                # Use Popen instead of run to avoid blocking the UI thread
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                played = True
                logging.info(f"Audio playing with {player_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.debug(f"{player_name} error: {e}")
                continue

        if not played:
            logging.error("No audio player found (aplay, paplay, or ffplay). Cannot play audio.")
            # Clean up temp file immediately if we didn't play
            try:
                os.unlink(tmp_path)
            except:
                pass
        else:
            # Clean up temp file after a delay (audio should be done by then)
            # Use a background thread to avoid blocking
            import threading
            def cleanup_audio_file():
                import time
                time.sleep(10)  # Wait 10 seconds for audio to finish
                try:
                    os.unlink(tmp_path)
                    logging.debug(f"Cleaned up temp audio file: {tmp_path}")
                except:
                    pass
            threading.Thread(target=cleanup_audio_file, daemon=True).start()

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

        # Track bug report state to prevent UI freezing
        self._last_issue_title = None
        self._last_timestamp = None

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

        # Initialize secondary windows
        self.board_window = SecondaryWindow(self.root, "Board State", 
            self.prefs.board_window_geometry if self.prefs and hasattr(self.prefs, 'board_window_geometry') else "600x800")
        
        self.deck_window = SecondaryWindow(self.root, "My Deck",
            self.prefs.deck_window_geometry if self.prefs and hasattr(self.prefs, 'deck_window_geometry') else "400x600")
            
        self.log_window = SecondaryWindow(self.root, "MTGA Logs",
            self.prefs.log_window_geometry if self.prefs and hasattr(self.prefs, 'log_window_geometry') else "800x200")

        # Message queue for thread-safe updates
        self.message_queue = deque(maxlen=100)
        # Initialize with a helpful message
        self.board_state_lines = [
            "=" * 70,
            "⏳ WAITING FOR MATCH...",
            "=" * 70,
            "",
            "Board state will appear here when you:",
            "  1. Start a new game in MTGA",
            "  2. Are in mulligan phase or during gameplay",
            "  3. Game events are being captured",
            "",
            "The advisor will display:",
            "  • Your cards in hand",
            "  • Cards on the battlefield",
            "  • Opponent's board state",
            "  • Stack (spells being cast)",
            "  • Life totals and library counts",
        ]
        self.rag_panel_expanded = False  # Track RAG panel expansion state

        # Bind F12 for bug reports
        self.root.bind('<F12>', lambda e: self._capture_bug_report())

        # Bind window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start update loop
        self.running = True
        self._update_loop()
        self._process_log_queue()

        # Ensure secondary windows are visible by default
        self.root.after(500, self._ensure_windows_visible)

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

        # TTS Engine (Kokoro only - BarkTTS removed)
        # No need for radio buttons since there's only one option

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

        # Windows toggles
        tk.Label(settings_frame, text="Windows:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W, pady=(10, 0))
        
        windows_frame = tk.Frame(settings_frame, bg=self.bg_color)
        windows_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(windows_frame, text="Board", command=lambda: self.board_window.deiconify(), bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(windows_frame, text="Deck", command=lambda: self.deck_window.deiconify(), bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(windows_frame, text="Logs", command=lambda: self.log_window.deiconify(), bg='#3a3a3a', fg='white', relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

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

        # Button frame for restart and exit side by side
        button_frame = tk.Frame(settings_frame, bg=self.bg_color)
        button_frame.pack(pady=5, fill=tk.X)

        tk.Button(
            button_frame,
            text="🔄 Restart App",
            command=self._on_restart,
            bg=self.info_color,
            fg='#1a1a1a',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=('Consolas', 9, 'bold')
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        tk.Button(
            button_frame,
            text="Exit",
            command=self._on_exit,
            bg=self.warning_color,
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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

        # Advisor messages area
        self.advisor_label = tk.Label(
            content_frame,
            text="═══ ADVISOR MESSAGES ═══",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        self.advisor_label.pack(pady=(0, 5))

        self.messages_text = scrolledtext.ScrolledText(
            content_frame,
            height=20,
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

        # Initialize log highlighter
        self.log_highlighter = LogHighlighter(card_db=None)  # Will be set in __init__

        # RAG References panel removed

    def _on_model_change(self, event=None):
        """Handle model selection change."""
        try:
            model = self.model_var.get()
            if model and CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.set_model(model)
                logging.info(f"✓ AI Model changed to: {model}")
                self.add_message(f"AI Model changed to: {model}", "green")

                # Apply immediately if we have access to AI advisor
                if hasattr(self, 'ai_advisor') and self.ai_advisor:
                    if hasattr(self.ai_advisor, 'client') and self.ai_advisor.client:
                        self.ai_advisor.client.model = model
                        logging.debug("Applied model change to active AI client")
        except Exception as e:
            logging.error(f"Error changing model: {e}")

    def _on_voice_change(self, event=None):
        """Handle voice selection change"""
        try:
            new_voice = self.voice_var.get()
            logging.info(f"Voice changed to: {new_voice}")
            
            # Update TTS engine voice via advisor reference
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                if hasattr(self.advisor_ref.tts, 'set_voice'):
                    self.advisor_ref.tts.set_voice(new_voice)
            
            # Save preference
            if self.prefs:
                self.prefs.set_voice_name(new_voice)
            
            # Test the new voice in a background thread
            if hasattr(self.advisor_ref, 'tts') and self.advisor_ref.tts:
                import threading
                threading.Thread(target=lambda: self.advisor_ref.tts.speak(f"Voice changed to {new_voice.replace('_', ' ')}"), daemon=True).start()
        except Exception as e:
            logging.error(f"Error changing voice: {e}")

    # _toggle_logs_panel removed (dead code referring to non-existent self.logs_text)

    def append_log(self, text: str):
        """Append text to the log queue (thread-safe)"""
        # Use a queue to prevent flooding the main thread with update events
        if not hasattr(self, 'log_queue'):
            self.log_queue = deque(maxlen=1000)
        self.log_queue.append(text)

    def _process_log_queue(self):
        """Process queued log messages in batches."""
        try:
            if hasattr(self, 'log_queue') and self.log_queue:
                # Process up to 50 lines at a time to keep UI responsive
                lines = []
                for _ in range(min(50, len(self.log_queue))):
                    lines.append(self.log_queue.popleft())
                
                if lines and self.log_window and self.log_window.winfo_exists():
                    # Join lines for a single update
                    text_block = "".join(lines) # lines already have newlines or will be added by append_text?
                    # append_log in app.py passes raw line. append_text adds newline.
                    # Let's handle it carefully.
                    for line in lines:
                        self.log_window.append_text(line)
            
            # Schedule next check
            if self.root and self.root.winfo_exists():
                self.root.after(100, self._process_log_queue)
                
        except Exception as e:
            logging.error(f"Error processing log queue: {e}")
            if self.root and self.root.winfo_exists():
                self.root.after(100, self._process_log_queue)


    def _on_restart(self):
        """Handle restart button click - restarts the application."""
        import subprocess
        import sys
        import os

        try:
            # Save preferences before restarting
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.save()

            # Show restart message
            self.add_message("🔄 Restarting application...", "cyan")
            logging.info("User initiated application restart")

            # Close the current app
            self.root.quit()

            # Restart the application by re-executing the same Python script
            # Get the original command that started this app
            if hasattr(self, 'advisor_ref') and self.advisor_ref:
                # If we have access to the advisor, we could potentially preserve state
                # For now, just cleanly restart
                pass

            # Re-execute the current Python process
            # This will restart the app with the same arguments
            python_executable = sys.executable
            script_path = os.path.abspath(sys.argv[0])

            # Pass through any original arguments (excluding the script name)
            args = [python_executable, script_path] + sys.argv[1:]

            logging.info(f"Restarting with: {' '.join(args)}")
            subprocess.Popen(args)

        except Exception as e:
            logging.error(f"Error restarting app: {e}")
            self.add_message(f"❌ Failed to restart: {e}", "red")

    def _on_exit(self):
        """Handle exit button click."""
        try:
            if CONFIG_MANAGER_AVAILABLE and self.prefs:
                self.prefs.save()
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

    def _prompt_for_credentials_async(self):
        """Prompt for credentials asynchronously."""
        from tkinter import simpledialog

        def ask_creds():
            credentials = self._prompt_for_credentials()
            if credentials and not credentials.get("cancelled"):
                # Save the credentials
                if CONFIG_MANAGER_AVAILABLE and self.prefs:
                    self.prefs.set_api_keys(
                        github_token=credentials.get("github_token", ""),
                        github_owner=credentials.get("github_owner", ""),
                        github_repo=credentials.get("github_repo", ""),
                        imgbb_api_key=credentials.get("imgbb_api_key", "")
                    )
                    logging.info("API credentials saved to user preferences")

                # Proceed with upload
                self._execute_upload(self._last_issue_title, self._last_timestamp)

        self.root.after(0, ask_creds)

    def _execute_upload(self, issue_title, timestamp):
        """Execute the actual upload process."""
        import threading

        def do_upload():
            try:
                # Add upload logic here (reuse existing upload functions)
                self.add_message("📤 Uploading to GitHub...", "cyan")
                # ... upload implementation ...
                self.add_message("✓ Upload completed!", "green")
            except Exception as e:
                self.add_message(f"✗ Upload failed: {e}", "red")

        threading.Thread(target=do_upload, daemon=True).start()

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



    # Removed duplicate non-thread-safe methods - thread-safe versions are below

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
            if self.root and CONFIG_MANAGER_AVAILABLE and self.prefs:
                # Save main window
                self.prefs.window_geometry = self.root.geometry()
                
                # Save secondary windows if open
                if self.deck_window and self.deck_window.winfo_exists():
                    self.prefs.deck_window_geometry = self.deck_window.geometry()
                
                if self.board_window and self.board_window.winfo_exists():
                    self.prefs.board_window_geometry = self.board_window.geometry()
                
                if self.log_window and self.log_window.winfo_exists():
                    self.prefs.log_window_geometry = self.log_window.geometry()
                
                self.prefs.save()

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
        """Update GUI settings with current values from advisor (thread-safe)."""
        def _update():
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
        
        self.root.after(0, _update)

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
        """Update board state display in external window (thread-safe)."""
        def _update():
            if self.board_window and self.board_window.winfo_exists():
                self.board_window.update_text(lines)
        self.root.after(0, _update)

    def set_draft_panes(self, pack_lines, picked_lines=None, picked_count=0, total_needed=45):
        """Update draft display (thread-safe)."""
        def _update():
            if self.board_window and self.board_window.winfo_exists():
                self.board_window.update_text(pack_lines)
                
            if self.deck_window and self.deck_window.winfo_exists() and picked_lines:
                deck_lines = [f"=== PICKED CARDS ({picked_count}/{total_needed}) ==="] + picked_lines
                self.deck_window.update_text(deck_lines)
        self.root.after(0, _update)

    def add_message(self, msg: str, color=None):
        """Add message to the advisor messages display (thread-safe)."""
        def _update():
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
        
        self.root.after(0, _update)


