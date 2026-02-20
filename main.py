#!/usr/bin/env python3
"""
MTGA Voice Advisor - Main Entry Point

Usage:
    python main.py           # GUI mode (default)
    python main.py --cli     # Command line / Headless mode
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# On Windows, set AppUserModelID BEFORE any GUI imports
# This is required for the taskbar to show our custom icon
if os.name == 'nt':
    try:
        import ctypes
        app_id = 'MTGAVoiceAdvisor.App.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Ignore errors, icon will just show default

print("DEBUG: main.py top-level executing...")

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
print(f"DEBUG: main.py setting up path: {src_path}")
sys.path.insert(0, str(src_path))

# Import and run the main application
print("DEBUG: main.py importing app...")
from src.core.app import main

if __name__ == "__main__":
    print("DEBUG: main.py calling app.main()...")
    main()