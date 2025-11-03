import curses
import logging
import time
from collections import deque
from typing import List, Callable

# Import Tkinter (optional - for GUI mode)
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available. GUI mode disabled.")

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

class AdvisorGUI:
    def __init__(self, root, advisor_ref):
        self.root = root
        self.advisor = advisor_ref

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

        # Show Spider-Man Reskins checkbox
        reskin_default = self.prefs.reskin_names if self.prefs else False
        self.reskin_var = tk.BooleanVar(value=reskin_default)
        tk.Checkbutton(
            settings_frame,
            text="Show Spider-Man Reskins",
            variable=self.reskin_var,
            command=self._on_reskin_toggle,
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
            text="🏗️ Make Deck Suggestion",
            command=self._manual_deck_suggestion,
            bg='#3a3a3a',
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
