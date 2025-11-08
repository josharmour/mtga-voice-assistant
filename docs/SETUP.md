# Installation & Setup Guide

Complete step-by-step guide for installing and configuring MTGA Voice Advisor.

---

## System Requirements

### Hardware
- **CPU:** Intel i5/Ryzen 5 or better (for local LLM)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB for card databases and LLM model
- **Disk Speed:** SSD recommended for database access

### Software
- **OS:** Windows 10/11, macOS 10.15+, or Linux
- **Python:** 3.12 or higher
- **MTGA:** Latest version with detailed logs enabled
- **Ollama:** Latest version (https://ollama.ai)

---

## Step 1: Install Python

### Windows
1. Download from https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH" during installation
3. Open Command Prompt and verify:
   ```cmd
   python --version
   ```

### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.12

# Or download from https://www.python.org/downloads/
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip

# Verify
python3.12 --version
```

---

## Step 2: Install Ollama

Ollama provides the local LLM backend - entirely private, no cloud services.

### Windows
1. Download from https://ollama.ai/download
2. Run installer
3. Open PowerShell and pull a model:
   ```powershell
   ollama pull mistral:7b
   ```
4. Verify Ollama is running:
   ```powershell
   # Ollama should start automatically on login
   # Check: Start > Ollama
   ```

### macOS
```bash
# Install
brew install ollama

# Pull model
ollama pull mistral:7b

# Start Ollama (runs in background)
ollama serve

# In another terminal, verify:
curl http://localhost:11434/api/tags
```

### Linux
```bash
# Download and install
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull mistral:7b

# Start Ollama
ollama serve

# Verify in another terminal:
curl http://localhost:11434/api/tags
```

**Model Recommendations:**
- **mistral:7b** (default) - 26GB RAM, fast, good quality
- **llama2:7b** - 20GB RAM, good for basic advice
- **mistral:4b** - 10GB RAM, lightweight
- **neural-chat:7b** - 26GB RAM, specialized for conversation

Check available models: https://ollama.ai/library

---

## Step 3: Clone/Download Repository

### Option A: Git Clone
```bash
git clone https://github.com/joshu/logparser.git
cd logparser
```

### Option B: Manual Download
1. Download repository from GitHub
2. Extract to desired location
3. Open terminal/Command Prompt in that directory

---

## Step 4: Set Up Virtual Environment

A virtual environment isolates Python packages for this project.

### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux (Terminal)
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**Verify activation:**
- Prompt should show `(venv)` at the start
- Run `python --version` - should show Python 3.12

---

## Step 5: Install Dependencies

### Core Dependencies (Required)
```bash
pip install -r requirements.txt
```

### Optional Dependencies

**For TTS (Voice Output):**
```bash
# Kokoro (recommended - fast)
pip install kokoro-onnx

# BarkTTS (fallback)
pip install transformers torch
```

**For RAG (Better Advice):**
```bash
pip install chromadb sentence-transformers
```

**For Draft Advisor:**
```bash
pip install numpy scipy tabulate termcolor
```

**All Optional:**
```bash
pip install kokoro-onnx chromadb sentence-transformers numpy scipy tabulate termcolor
```

**Verify Installation:**
```bash
python -c "import sqlite3, requests, argparse; print('Core OK')"
```

---

## Step 6: Download Card Data

The app requires card databases for functionality.

```bash
# Update 17lands statistics (current sets)
python3 manage_data.py --update-17lands

# Update all sets (not just Standard)
python3 manage_data.py --update-17lands --all-sets

# Update Scryfall cache (optional but recommended)
python3 manage_data.py --update-scryfall

# Check status
python3 manage_data.py --status
```

**First Run:**
- 17lands download: ~5-10 minutes (depends on internet)
- Scryfall cache: ~10-20 minutes (many API calls)
- Both create files in `data/` directory

**Verify:**
```bash
ls -lh data/
# Should show:
# - unified_cards.db (~3.7 MB)
# - card_stats.db (~100 MB)
# - scryfall_cache.db (variable)
```

---

## Step 7: Configure MTGA

The app monitors MTGA's Player.log file for game events.

### Enable Detailed Logs

1. Open MTGA
2. Go to **Options** → **Account**
3. Check **"Detailed Logs (Plugin Support)"**
4. Restart MTGA

**Verify:** The app will auto-detect `Player.log` on startup.

---

## Step 8: First Run

### Make Sure Ollama is Running

```bash
# Terminal 1: Start Ollama
ollama serve
# Should show: Listening on 127.0.0.1:11434

# Terminal 2: Start the app (see below)
```

### Launch the Application

**GUI Mode (Recommended):**
```bash
python3 app.py
```

**TUI Mode (Terminal):**
```bash
python3 app.py --tui
```

**CLI Mode (Simple Output):**
```bash
python3 app.py --cli
```

### Expected Output

```
WARNING:root:ChromaDB not available. Install with: pip install chromadb
WARNING:root:sentence-transformers not available. Install with: pip install sentence-transformers
Loading card database...
✓ Loaded unified card database (22,719 cards)
✓ Ollama service is running

============================================================
MTGA Voice Advisor Started
============================================================
Log: /path/to/Player.log
AI Model: mistral:7b (3 available)
Voice: am_adam | Volume: 100%

Waiting for a match... (Enable Detailed Logs in MTGA settings)
Type /help for commands

You:
```

---

## Step 9: Verify Everything Works

### Test Card Database
```bash
python3 -c "from data_management import ArenaCardDatabase; \
db = ArenaCardDatabase(); \
name = db.get_card_name(1); \
print(f'Card 1: {name}')"
```

### Test Ollama Connection
```bash
python3 -c "from ai import OllamaClient; \
c = OllamaClient(); \
print(f'Ollama running: {c.is_running()}')"
```

### Test Game State Parser
```bash
python3 -c "from mtga import LogFollower, GameStateManager; \
print('Game state parser OK')"
```

### Test UI
```bash
# GUI test (requires X11/display)
python3 -c "from ui import AdvisorGUI; print('GUI OK')"

# TUI test
python3 -c "from ui import AdvisorTUI; print('TUI OK')"
```

---

## Troubleshooting Setup

### Issue: "Python not found"
**Solution:** Python not in PATH. Reinstall and check "Add to PATH" option.

### Issue: "venv activation failed"
**Windows:**
```cmd
# Try absolute path
C:\Users\YourName\logparser\venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Try full path
/home/user/logparser/venv/bin/activate
```

### Issue: "Module not found" after pip install
**Solution:**
```bash
# Verify virtual environment is activated
python --version  # Should show Python 3.12

# Reinstall all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Ollama won't start
**Windows:** Check Start Menu for Ollama application
**macOS:** Try `ollama serve` in Terminal
**Linux:** Check firewall allows port 11434
```bash
# Verify Ollama port
lsof -i :11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows
```

### Issue: "Can't find Player.log"
**Possible Locations:**
- **Windows:** `C:\Users\[YourName]\AppData\LocalLow\Wizards Of The Coast\MTGA\`
- **macOS:** `~/Library/Application Support/Wizards Of The Coast/MTGA/`
- **Linux (Proton):** `~/.local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/`

### Issue: "Card database not found"
**Solution:**
```bash
python3 manage_data.py --update-17lands
python3 manage_data.py --update-scryfall
```

### Issue: "No TTS output"
**Solution:** Install TTS packages
```bash
pip install kokoro-onnx transformers torch scipy
```

---

## Advanced Configuration

### Using Different LLM Model

Edit `app.py` line ~228:
```python
self.ai_advisor = AIAdvisor(card_db=card_db, model="neural-chat:7b")
```

Or at runtime:
```bash
# Set environment variable
export OLLAMA_MODEL="llama2:7b"
python3 app.py
```

### Using Different TTS Engine

Edit `app.py` line ~170:
```python
self.tts = TextToSpeech(engine="bark")  # Use BarkTTS instead
```

### Custom Preferences

Edit `~/.mtga_advisor/preferences.json`:
```json
{
  "window_geometry": [1200, 800],
  "ui_theme": "dark",
  "voice_engine": "kokoro",
  "voice_volume": 100,
  "ollama_model": "mistral:7b",
  "use_rag": true
}
```

---

## Environment Variables

Optional environment variables for configuration:

```bash
# Ollama settings
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODEL=mistral:7b

# Application settings
export MTGA_ADVISOR_MODE=cli  # gui, tui, cli
export MTGA_LOG_PATH=/custom/path/to/Player.log

# Performance
export MTGA_ADVISOR_WORKERS=4  # Number of threads
```

---

## Next Steps

1. **Launch the app:** `python3 app.py`
2. **Start a match in MTGA** (Draft, Sealed, or Ranked)
3. **Watch for advice** in the terminal/GUI
4. **Customize preferences** via settings dialog (GUI mode)
5. **Update data** weekly: `python3 manage_data.py --update-17lands`

---

## Performance Tips

### For Slower Systems
- Use **CLI mode** instead of GUI (less resource-intensive)
- Use **mistral:4b** model instead of 7b
- Disable RAG system: `use_rag=False` in preferences
- Close other applications

### For Faster Systems
- Use GPU: `ollama pull mistral:7b-gpu`
- Enable RAG system for better advice
- Use GUI mode for better UX

### Database Optimization
```bash
# Rebuild indices (run weekly)
python3 -c "from data_management import CardStatsDB; \
db = CardStatsDB(); \
conn = db._get_conn(); \
conn.execute('VACUUM'); \
conn.close(); \
print('Optimized')"
```

---

## Uninstallation

To completely remove:

```bash
# Deactivate virtual environment
deactivate

# Remove directory
rm -rf logparser/  # macOS/Linux
rmdir /s logparser  # Windows

# Remove user preferences
rm -rf ~/.mtga_advisor/  # macOS/Linux
rmdir %APPDATA%\.mtga_advisor  # Windows
```

---

## Support

If you encounter issues:

1. Check `logs/advisor.log` for detailed errors
2. Run diagnostics:
   ```bash
   python3 -c "from data_management import ArenaCardDatabase; \
   db = ArenaCardDatabase(); print('✓ Database OK')"
   ```
3. Verify Ollama is running: `curl http://localhost:11434/api/tags`
4. Check MTGA detailed logs are enabled in Options

