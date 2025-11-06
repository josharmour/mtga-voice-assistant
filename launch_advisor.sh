#!/bin/bash
# Launch MTGA Voice Advisor (defaults to GUI)

# Add Ollama to PATH
export PATH="$HOME/ollama/bin:$PATH"

# Change to the logparser directory
cd /home/joshu/logparser

# Activate virtual environment
source venv/bin/activate

# Launch the advisor (GUI is default)
python3 app.py
