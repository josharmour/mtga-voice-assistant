# MTGA Voice Advisor - Installation Guide

## Quick Start

### Minimum Installation (Core Features Only)

```bash
# Install core dependencies
pip3 install requests urllib3 pyttsx3

# On Linux, also install espeak system package
sudo apt-get install espeak espeak-ng libespeak1
```

This gives you:
- ✅ Real-time game tracking
- ✅ AI tactical advice (via Ollama)
- ✅ Voice output (basic TTS)
- ✅ CLI and TUI modes
- ✅ Model selector
- ✅ Voice selector

### Full Installation (All Features)

```bash
# Install all dependencies
pip3 install -r requirements.txt

# On Linux, install espeak
sudo apt-get install espeak espeak-ng libespeak1
```

This adds:
- ✅ High-quality TTS (Kokoro ONNX)
- ✅ RAG system with MTG rule knowledge
- ✅ Enhanced AI advice

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# 1. System packages for TTS
sudo apt-get update
sudo apt-get install espeak espeak-ng libespeak1

# 2. Python packages (minimum)
pip3 install requests urllib3 pyttsx3

# 3. Python packages (full, optional)
pip3 install -r requirements.txt
```

### Linux (Arch/Manjaro)

```bash
# 1. System packages for TTS
sudo pacman -S espeak-ng

# 2. Python packages
pip3 install -r requirements.txt
```

### macOS

```bash
# macOS has built-in TTS, no system packages needed

# Install Python packages
pip3 install -r requirements.txt
```

### Windows

```bash
# Windows needs additional packages for curses and TTS

# 1. Install core dependencies
pip install requests urllib3 pyttsx3

# 2. Install Windows-specific packages
pip install windows-curses pywin32

# 3. Optional: full features
pip install -r requirements.txt
```

---

## Dependency Breakdown

### Required (Core Functionality)

| Package | Purpose | Size |
|---------|---------|------|
| `requests` | HTTP client for Ollama API | ~200KB |
| `urllib3` | HTTP library | ~300KB |
| `pyttsx3` | Text-to-speech engine | ~50KB |

**Total: ~500KB**

### Optional (Enhanced Features)

| Package | Purpose | Size |
|---------|---------|------|
| `kokoro-onnx` | High-quality TTS voices | ~100MB |
| `scipy` | Scientific computing (for Kokoro) | ~50MB |
| `chromadb` | Vector database (RAG) | ~20MB |
| `sentence-transformers` | Embeddings (RAG) | ~500MB |
| `torch` | Machine learning (RAG) | ~800MB |

**Total: ~1.5GB** (only if you want RAG features)

---

## What Packages Did You Need to Install?

Based on your question, you likely needed one or more of:

1. **urllib3** - May not have been installed
2. **espeak** (Linux system package) - Required for pyttsx3 on Linux
3. **requests** - If not already installed

To check what's missing:

```bash
# Check Python packages
python3 -c "import requests; import urllib3; import pyttsx3; print('All core packages installed!')"

# Check espeak (Linux only)
which espeak
```

---

## Minimal Requirements (No RAG, Basic TTS)

If you want the absolute minimum to get started:

```bash
# Only install core packages
pip3 install requests urllib3 pyttsx3

# Linux: Install espeak
sudo apt-get install espeak-ng
```

Then edit `requirements.txt` to remove the RAG packages:

```bash
# Create minimal requirements file
cat > requirements-minimal.txt << EOF
requests>=2.31.0
urllib3>=2.0.0
pyttsx3>=2.90
EOF

pip3 install -r requirements-minimal.txt
```

---

## Verifying Installation

### Check Python Dependencies

```bash
python3 << EOF
import sys
packages = ['requests', 'urllib3', 'pyttsx3', 'curses', 'json', 'logging']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - MISSING")
        missing.append(pkg)

if missing:
    print(f"\nInstall missing packages: pip3 install {' '.join(missing)}")
else:
    print("\n✅ All core dependencies installed!")
EOF
```

### Check Optional Dependencies

```bash
python3 << EOF
optional = ['chromadb', 'sentence_transformers', 'torch']
for pkg in optional:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"⚠️  {pkg} - not installed (optional)")
EOF
```

### Test the Advisor

```bash
# Test help (should show usage)
python3 advisor.py --help

# Test startup (Ctrl+C to exit)
python3 advisor.py
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'requests'"

**Solution:**
```bash
pip3 install requests urllib3
```

### Issue: "ModuleNotFoundError: No module named '_curses'"

**Solution (Windows):**
```bash
pip install windows-curses
```

**Solution (Linux/macOS):** Curses is built-in, this shouldn't happen

### Issue: "No TTS engine found" or "espeak not found"

**Solution (Linux):**
```bash
sudo apt-get install espeak espeak-ng libespeak1
```

**Solution (macOS):** Should work out of the box with built-in TTS

**Solution (Windows):**
```bash
pip install pywin32
```

### Issue: "ChromaDB not available"

**Solution:** This is optional. If you don't need RAG features, ignore it.

To install RAG:
```bash
pip3 install chromadb sentence-transformers torch
```

### Issue: Large download size

**Solution:** Skip RAG packages for now
```bash
# Only install core packages
pip3 install requests urllib3 pyttsx3

# Comment out RAG packages in requirements.txt
sed -i 's/^chromadb/#chromadb/' requirements.txt
sed -i 's/^sentence-transformers/#sentence-transformers/' requirements.txt
sed -i 's/^torch/#torch/' requirements.txt
```

---

## RAG System Setup

The RAG system is **optional** and provides enhanced AI advice using MTG rule knowledge.

### Has RAG Been Initialized?

Check if the vector database exists:
```bash
ls -la rag_*.db chroma_db/ 2>/dev/null || echo "RAG not initialized"
```

### Initialize RAG (First Time)

If you want to use RAG features:

1. Install dependencies:
```bash
pip3 install chromadb sentence-transformers torch
```

2. Run the initialization script:
```bash
python3 test_rag.py
```

This will:
- Download MTG comprehensive rules
- Create vector embeddings
- Build the ChromaDB database
- Test query functionality

**Note:** This is a one-time operation that takes ~5-10 minutes and requires ~2GB of disk space.

### RAG Status Check

```bash
python3 -c "
from rag_advisor import RAGSystem
rag = RAGSystem()
print('✅ RAG system ready!')
" 2>&1 | grep -E "(✅|Error|not available)"
```

---

## What You Told Me You Needed

You mentioned you "had to install a couple pip packages." The most likely candidates are:

1. **urllib3** - Often not installed by default
   ```bash
   pip3 install urllib3
   ```

2. **requests** - HTTP library for Ollama
   ```bash
   pip3 install requests
   ```

If you're on Linux and pyttsx3 wasn't working:

3. **espeak system package**
   ```bash
   sudo apt-get install espeak-ng
   ```

---

## Recommended Installation Order

### Step 1: Minimal Setup (5 minutes)

```bash
# Install core packages
pip3 install requests urllib3 pyttsx3

# Linux only
sudo apt-get install espeak-ng

# Test
python3 advisor.py --help
```

### Step 2: Try It Out

```bash
# Start in CLI mode
python3 advisor.py

# Or start in TUI mode
python3 advisor.py --tui
```

### Step 3: Optional Enhancement (30 minutes)

Only if you want RAG features:

```bash
# Install large packages
pip3 install chromadb sentence-transformers torch

# Initialize RAG database
python3 test_rag.py
```

---

## Summary

**Minimum to run advisor:**
- `requests`, `urllib3`, `pyttsx3`
- On Linux: `espeak-ng` system package

**Current status:**
- ❓ RAG database: Not initialized (no database files found)
- ✅ Core advisor: Should work with minimal dependencies
- ✅ TUI mode: Available (curses is built-in on Linux)
- ✅ Model selector: Working
- ✅ Voice selector: Working

**What packages did you actually need to install?** Please let me know so I can update this guide!
