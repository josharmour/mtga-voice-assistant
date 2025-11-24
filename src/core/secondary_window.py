import tkinter as tk
from tkinter import scrolledtext
from typing import List, Optional

class SecondaryWindow(tk.Toplevel):
    def __init__(self, parent, title, geometry=None, on_close=None):
        super().__init__(parent)
        self.title(title)
        self.configure(bg='#2b2b2b')
        if geometry:
            self.geometry(geometry)

        # Default behavior is to withdraw (hide) instead of destroy
        self.protocol("WM_DELETE_WINDOW", on_close if on_close else self.withdraw)

        self.text_area = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            bg='#1a1a1a',
            fg='#ffffff',
            font=('Consolas', 10),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.config(state=tk.DISABLED)

        # Define tags for coloring
        self.text_area.tag_config("color_w", foreground="#ffffff", background="#333333") # White
        self.text_area.tag_config("color_u", foreground="#55aaff") # Blue
        self.text_area.tag_config("color_b", foreground="#aaaaaa") # Black
        self.text_area.tag_config("color_r", foreground="#ff5555") # Red
        self.text_area.tag_config("color_g", foreground="#55ff55") # Green
        self.text_area.tag_config("color_multi", foreground="#ffff55") # Multicolor

        # Track previous content for diff-based updates
        self._previous_lines: List[str] = []
        self._diff_update_enabled = True  # Can disable for full refresh

    def update_text(self, lines: List[str]):
        """Update text content with optional diff-based optimization."""
        if not self._diff_update_enabled or not self._previous_lines:
            # Full update for first render or when diff disabled
            self._full_update(lines)
        else:
            # Try diff-based update
            self._diff_update(lines)

        self._previous_lines = list(lines)  # Store copy for next diff

    def _full_update(self, lines: List[str]):
        """Full content replacement (original behavior)."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)

        for line in lines:
            tag = self._get_line_tag(line)
            if tag:
                self.text_area.insert(tk.END, line + "\n", tag)
            else:
                self.text_area.insert(tk.END, line + "\n")

        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def _diff_update(self, new_lines: List[str]):
        """Update only changed lines using diff."""
        old_lines = self._previous_lines

        # Quick check: if lengths differ significantly, do full update
        if abs(len(new_lines) - len(old_lines)) > len(old_lines) * 0.5:
            self._full_update(new_lines)
            return

        self.text_area.config(state=tk.NORMAL)

        # Find changed lines and update them
        max_len = max(len(old_lines), len(new_lines))
        changes_made = 0

        for i in range(max_len):
            old_line = old_lines[i] if i < len(old_lines) else None
            new_line = new_lines[i] if i < len(new_lines) else None

            if old_line != new_line:
                changes_made += 1
                line_start = f"{i + 1}.0"
                line_end = f"{i + 1}.end"

                if new_line is None:
                    # Line was removed - delete it
                    self.text_area.delete(line_start, f"{i + 2}.0")
                elif old_line is None:
                    # Line was added - insert at end
                    tag = self._get_line_tag(new_line)
                    if tag:
                        self.text_area.insert(tk.END, new_line + "\n", tag)
                    else:
                        self.text_area.insert(tk.END, new_line + "\n")
                else:
                    # Line was modified - replace it
                    self.text_area.delete(line_start, line_end)
                    tag = self._get_line_tag(new_line)
                    if tag:
                        self.text_area.insert(line_start, new_line, tag)
                    else:
                        self.text_area.insert(line_start, new_line)

        # If too many changes, just do full update next time
        if changes_made > len(new_lines) * 0.3:
            # More than 30% changed - full update would have been faster
            pass  # Log this for performance tuning

        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def _get_line_tag(self, line: str) -> Optional[str]:
        """Determine the color tag for a line based on content."""
        if "⚪" in line: return "color_w"
        elif "🔵" in line: return "color_u"
        elif "⚫" in line: return "color_b"
        elif "🔴" in line: return "color_r"
        elif "🟢" in line: return "color_g"
        elif "🌈" in line: return "color_multi"
        return None

    def force_full_update(self, lines: List[str]):
        """Force a full update, bypassing diff logic."""
        self._full_update(lines)
        self._previous_lines = list(lines)

    def clear(self):
        """Clear all content and reset diff state."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)
        self._previous_lines = []

    def append_text(self, text, tags=None):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def append_batch(self, lines):
        """Append multiple lines efficiently."""
        self.text_area.config(state=tk.NORMAL)
        for line in lines:
            self.text_area.insert(tk.END, line + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)
