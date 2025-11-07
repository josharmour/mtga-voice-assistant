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

from mtga import (
    LogFollower,
    GameObject,
    PlayerState,
    GameHistory,
    BoardState,
    MatchScanner,
    GameStateManager,
)
from ai import OllamaClient, AIAdvisor
from ui import TextToSpeech, AdvisorTUI, AdvisorGUI
from card_rag import CardRagDatabase as ArenaCardDatabase

# Import configuration manager for user preferences
try:
    from config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")

# Import RAG system (optional - will gracefully degrade if not available)
try:
    from ai import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available. Install dependencies with: pip install chromadb sentence-transformers torch")

# Import draft advisor (requires tabulate, termcolor, scipy)
try:
    from draft_advisor import DraftAdvisor, display_draft_pack, format_draft_pack_for_gui
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