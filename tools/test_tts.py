
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    from src.core.ui import TextToSpeech
    print("Successfully imported TextToSpeech")
except ImportError as e:
    print(f"Failed to import TextToSpeech: {e}")
    sys.exit(1)

print("\n--- Testing Kokoro TTS Initialization ---")
tts = TextToSpeech(force_engine="kokoro")
if tts.tts_engine == "kokoro":
    print("✅ Kokoro initialized successfully")
else:
    print("❌ Kokoro failed to initialize")

print("\n--- Testing Bark TTS Initialization ---")
tts = TextToSpeech(force_engine="bark")
if tts.tts_engine == "bark":
    print("✅ Bark initialized successfully")
else:
    print("❌ Bark failed to initialize")
