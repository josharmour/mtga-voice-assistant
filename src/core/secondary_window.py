import tkinter as tk
from tkinter import scrolledtext

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

    def update_text(self, lines):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        
        for line in lines:
            tag = None
            if "⚪" in line: tag = "color_w"
            elif "🔵" in line: tag = "color_u"
            elif "⚫" in line: tag = "color_b"
            elif "🔴" in line: tag = "color_r"
            elif "🟢" in line: tag = "color_g"
            elif "🌈" in line: tag = "color_multi"
            
            if tag:
                self.text_area.insert(tk.END, line + "\n", tag)
            else:
                self.text_area.insert(tk.END, line + "\n")
                
        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

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
