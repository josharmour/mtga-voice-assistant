#!/usr/bin/env python3
"""Test if sounddevice can play audio in WSL"""

import numpy as np
import sounddevice as sd

print("Testing sounddevice audio playback in WSL...")
print(f"sounddevice version: {sd.__version__}")

# List available audio devices
print("\nAvailable audio devices:")
print(sd.query_devices())

# Generate a simple 440 Hz tone (A note)
fs = 48000  # Sample rate
duration = 1.0  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)

print(f"\nPlaying a {duration}s test tone (440 Hz)...")
try:
    sd.play(audio, fs)
    sd.wait()  # Wait for playback to finish
    print("✓ Audio playback successful!")
except Exception as e:
    print(f"✗ Audio playback failed: {e}")
    import traceback
    traceback.print_exc()
