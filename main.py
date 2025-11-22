#!/usr/bin/env python3
"""
MTGA Voice Advisor - Main Entry Point

Usage:
    python main.py           # GUI mode (default)
    python main.py --cli     # Command line / Headless mode
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
from src.core.app import main

if __name__ == "__main__":
    main()