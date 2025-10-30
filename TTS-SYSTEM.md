# MTGA Voice Advisor - TTS System

**Date**: 2025-10-28
**Status**: ✅ IMPLEMENTED - Kokoro primary, BarkTTS fallback

---

## Overview

The MTGA Voice Advisor uses a **fallback TTS system** that prioritizes quality and speed:

1. **Primary**: Kokoro (fast, high-quality, multiple voices)
2. **Fallback**: BarkTTS (if Kokoro fails to initialize)

This ensures the advisor always has voice output, even if one TTS engine fails.

---

## TTS Engine Priority

### 1. Kokoro (Primary)

**Why Primary:**
- ✅ Fast generation (~200-500ms per sentence)
- ✅ High audio quality
- ✅ Multiple voices (80+ voices)
- ✅ ONNX-based (CPU-optimized)
- ✅ Small models (~100MB)

**Installation:**
```bash
pip install kokoro-onnx scipy
```

**Models Location:**
```
~/.local/share/kokoro/
├── kokoro-v1.0.onnx    # Main model
└── voices-v1.0.bin     # Voice data
```

**Voice Selection:**
- Use `/voice <name>` command
- Example voices: `am_adam`, `am_michael`, `af_bella`, `af_sarah`
- See full list: [Kokoro Voices](https://github.com/thewh1teagle/kokoro-onnx)

### 2. BarkTTS (Fallback)

**Why Fallback:**
- ⚠️ Slower generation (~3-5 seconds per sentence)
- ⚠️ Larger models (~1GB)
- ⚠️ Requires PyTorch
- ✅ More expressive (emotions, intonations)
- ✅ Handles longer text better

**Installation:**
```bash
pip install transformers torch
```

**Models:**
- Automatically downloads from Hugging Face
- Model: `suno/bark-small` (~1GB)
- Stored in: `~/.cache/huggingface/`

**Voice Presets:**
- Fixed preset: `v2/en_speaker_6`
- Cannot be changed via `/voice` command

---

## Implementation Details

### Initialization Flow

```python
class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0):
        # Try Kokoro first
        if self._init_kokoro():
            self.tts_engine = "kokoro"
            return

        # Fall back to BarkTTS
        if self._init_bark():
            self.tts_engine = "bark"
            return

        # No TTS available
        self.tts_engine = None
```

### Speak Flow

```python
def speak(self, text: str):
    if self.tts_engine == "kokoro":
        self._speak_kokoro(text)
    elif self.tts_engine == "bark":
        self._speak_bark(text)
```

### Audio Playback

Both engines use the same playback method:

1. Generate audio (numpy array)
2. Apply volume adjustment
3. Save to temporary WAV file
4. Try audio players in order:
   - `aplay` (ALSA)
   - `paplay` (PulseAudio)
   - `ffplay` (FFmpeg)
5. Clean up temp file

---

## Commands

### `/tts` - Show Active TTS Engine

Check which TTS engine is currently active:

```bash
You: /tts
✓ TTS Engine: Kokoro (kokoro)
  Voice: am_adam
```

Or if using fallback:

```bash
You: /tts
✓ TTS Engine: BarkTTS (bark)
  Using Bark's built-in voice presets
```

### `/voice [name]` - Change Voice (Kokoro Only)

Change the voice used by Kokoro:

```bash
You: /voice bella
✓ Voice changed to: bella
```

**Note:** This only works with Kokoro. BarkTTS uses fixed voice presets.

### `/volume [0-100]` - Set Volume

Adjust TTS volume (works with both engines):

```bash
You: /volume 80
✓ Volume set to: 80%
```

### `/settings` - Show Current Settings

Display all settings including TTS engine:

```bash
You: /settings

Current Settings:
  AI Model: llama3.2:latest
  TTS:      Kokoro
  Voice:    am_adam
  Volume:   100%
  Log:      /home/user/.local/share/Steam/.../Player.log
```

---

## Kokoro Voices

### Male Voices (American)

| Voice ID | Description |
|----------|-------------|
| `am_adam` | Default male voice |
| `am_michael` | Alternative male voice |

### Female Voices (American)

| Voice ID | Description |
|----------|-------------|
| `af_bella` | Default female voice |
| `af_sarah` | Alternative female voice |
| `af_nicole` | Another female option |

### Other Regions

- British: `bf_emma`, `bm_george`
- Australian: `af_skye`
- Indian: `if_sara`

**Full voice list:** Run `/voice` without arguments to see available voices, or check [Kokoro documentation](https://github.com/thewh1teagle/kokoro-onnx).

---

## Performance Comparison

| Feature | Kokoro | BarkTTS |
|---------|--------|---------|
| **Speed** | ~300ms | ~4s |
| **Quality** | High | Very High |
| **Model Size** | ~100MB | ~1GB |
| **Voices** | 80+ | Few presets |
| **CPU Usage** | Low | Medium-High |
| **GPU Support** | No (ONNX) | Yes (PyTorch) |
| **Expressiveness** | Neutral | Emotional |

---

## Troubleshooting

### Issue: "No TTS engine initialized"

**Symptoms:**
```
❌ Failed to initialize any TTS engine (Kokoro and Bark both failed)
```

**Solution 1: Install Kokoro**
```bash
pip install kokoro-onnx scipy

# Download models (first run)
python3 -c "from kokoro_onnx import Kokoro; Kokoro()"
```

**Solution 2: Install BarkTTS**
```bash
pip install transformers torch
```

### Issue: "Kokoro models not found"

**Symptoms:**
```
Kokoro initialization failed: [Errno 2] No such file or directory: '~/.local/share/kokoro/kokoro-v1.0.onnx'
```

**Solution:**
```bash
# Manually download models
mkdir -p ~/.local/share/kokoro
cd ~/.local/share/kokoro

# Download from Kokoro releases
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/v1.0/voices-v1.0.bin
```

Or let the library download them automatically:
```bash
python3 -c "from kokoro_onnx import Kokoro; k = Kokoro(); print('Models downloaded!')"
```

### Issue: "No audio player found"

**Symptoms:**
```
No audio player found (aplay, paplay, or ffplay). Cannot play audio.
```

**Solution (Linux):**
```bash
# Install ALSA utils (aplay)
sudo apt-get install alsa-utils

# Or install PulseAudio utils (paplay)
sudo apt-get install pulseaudio-utils

# Or install FFmpeg (ffplay)
sudo apt-get install ffmpeg
```

### Issue: BarkTTS is slow

**Solution 1: Use GPU**
```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 2: Switch to Kokoro**
```bash
pip install kokoro-onnx scipy
# Restart advisor
python3 advisor.py
```

---

## Example Usage Session

### Startup with Kokoro

```bash
$ python3 advisor.py

Loading card database...
✓ Loaded 21477 cards from Arena database

Attempting to initialize Kokoro TTS (primary) with voice: adam, volume: 1.0
✓ Kokoro TTS initialized successfully

============================================================
MTGA Voice Advisor Started
============================================================
Log: /home/user/.local/share/Steam/.../Player.log
AI Model: llama3.2:latest (3 available)
Voice: adam | Volume: 100%
```

### Startup with BarkTTS Fallback

```bash
$ python3 advisor.py

Loading card database...
✓ Loaded 21477 cards from Arena database

Attempting to initialize Kokoro TTS (primary) with voice: adam, volume: 1.0
Kokoro initialization failed: No module named 'kokoro_onnx'
Kokoro TTS failed, falling back to BarkTTS (secondary)
Loading BarkTTS model (this may take a moment)...
✓ BarkTTS initialized successfully

============================================================
MTGA Voice Advisor Started
============================================================
Log: /home/user/.local/share/Steam/.../Player.log
AI Model: llama3.2:latest (3 available)
TTS: BarkTTS (fallback)
```

### Checking TTS Status

```bash
You: /tts
✓ TTS Engine: Kokoro (kokoro)
  Voice: am_adam

You: /voice bella
✓ Voice changed to: bella

You: /tts
✓ TTS Engine: Kokoro (kokoro)
  Voice: af_bella

You: /settings

Current Settings:
  AI Model: llama3.2:latest
  TTS:      Kokoro
  Voice:    af_bella
  Volume:   100%
  Log:      /home/user/.local/share/Steam/.../Player.log
```

---

## Technical Architecture

### Class Structure

```python
class TextToSpeech:
    # Attributes
    tts_engine: str           # "kokoro" or "bark"
    tts: Kokoro              # Kokoro instance (if using Kokoro)
    bark_processor: Processor # BarkTTS processor (if using Bark)
    bark_model: Model        # BarkTTS model (if using Bark)
    voice: str               # Current voice (Kokoro only)
    volume: float            # Volume (0.0-1.0)

    # Methods
    __init__()               # Initialize with fallback logic
    _init_kokoro() -> bool   # Try Kokoro initialization
    _init_bark() -> bool     # Try BarkTTS initialization
    speak(text: str)         # Route to correct engine
    _speak_kokoro(text)      # Kokoro-specific TTS
    _speak_bark(text)        # BarkTTS-specific TTS
    _save_and_play_audio()   # Shared playback method
```

### Dependencies

**For Kokoro:**
```
kokoro-onnx>=0.1.0
scipy>=1.11.0
numpy>=1.24.0
```

**For BarkTTS:**
```
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

**System (Linux):**
```
aplay (alsa-utils)
# OR
paplay (pulseaudio-utils)
# OR
ffplay (ffmpeg)
```

---

## Design Decisions

### Why Kokoro as Primary?

1. **Speed**: 10x faster than BarkTTS
2. **Quality**: ONNX models are highly optimized
3. **Voices**: 80+ voices vs Bark's ~10 presets
4. **Size**: 100MB vs 1GB for Bark
5. **Dependencies**: Lighter (no PyTorch required)

### Why BarkTTS as Fallback?

1. **Availability**: Easy to install via pip
2. **Reliability**: Mature Hugging Face integration
3. **Quality**: Still produces good audio
4. **Expressiveness**: Better emotional range than Kokoro

### Why Not pyttsx3?

- ❌ Poor Linux support (depends on espeak)
- ❌ Robotic voice quality
- ❌ Limited customization
- ✅ Replaced with Kokoro/Bark

---

## Future Enhancements

### Potential Additions

1. **Runtime Engine Switching**
   ```bash
   /tts switch bark  # Switch to BarkTTS
   /tts switch kokoro # Switch back to Kokoro
   ```

2. **Voice Preview**
   ```bash
   /voice preview bella  # Hear sample before switching
   ```

3. **Custom Bark Voice Presets**
   ```bash
   /bark-voice v2/en_speaker_9  # Change Bark preset
   ```

4. **TTS Speed Control**
   ```bash
   /speed 1.2  # 20% faster
   /speed 0.8  # 20% slower
   ```

5. **Audio Effects**
   ```bash
   /pitch 1.1  # Slightly higher pitch
   /reverb on  # Add reverb effect
   ```

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| advisor.py | Lines 1568-1756 | Complete TTS rewrite |
| - TextToSpeech.__init__ | Lines 1569-1595 | Fallback initialization |
| - _init_kokoro | Lines 1597-1615 | Kokoro setup (NEW) |
| - _init_bark | Lines 1617-1641 | BarkTTS setup (NEW) |
| - speak | Lines 1653-1667 | Engine routing (MODIFIED) |
| - _speak_kokoro | Lines 1669-1685 | Kokoro-specific TTS (NEW) |
| - _speak_bark | Lines 1687-1715 | BarkTTS-specific TTS (NEW) |
| - _save_and_play_audio | Lines 1717-1756 | Shared playback (NEW) |
| - /tts command | Lines 2244-2253 | Show TTS status (NEW) |
| - /help update | Lines 2333-2336 | Document TTS (MODIFIED) |
| - /settings update | Lines 2344-2351 | Show TTS engine (MODIFIED) |
| requirements.txt | Lines 5-10 | Update TTS deps (MODIFIED) |

---

## Conclusion

**TTS fallback system implemented!** ✅

The advisor now uses:
- ✅ **Kokoro** as primary TTS (fast, high-quality)
- ✅ **BarkTTS** as fallback (if Kokoro fails)
- ✅ Automatic engine selection
- ✅ New `/tts` command to check engine status
- ✅ Updated help and settings displays

**Benefits:**
- Guaranteed voice output (one engine will always work)
- Best-in-class performance with Kokoro
- Reliable fallback with BarkTTS
- Easy to check which engine is active

**Usage:**
```bash
# Start advisor (auto-selects best TTS)
python3 advisor.py

# Check which TTS is active
You: /tts

# Change voice (Kokoro only)
You: /voice bella
```

---

END OF DOCUMENT
