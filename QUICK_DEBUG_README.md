# Quick Debug - For AI-Assisted Troubleshooting

## When Something Goes Wrong...

Instead of trying to explain the problem or paste logs, just run:

```bash
python quick_debug.py
```

This instantly creates a complete debug snapshot that you can share with an AI assistant (like me!).

## Three Ways to Export

### 1. **GUI Button** (Easiest - During Match)
- Press **F12** key
- Or click "🔍 Debug Export" in the settings panel
- Folder opens automatically with the JSON file

### 2. **Quick Command** (Fastest - From Terminal)
```bash
python quick_debug.py
```
No arguments needed! Just run and share the file.

### 3. **Detailed CLI** (For Specific Queries)
```bash
python debug_cli.py status    # See current state
python debug_cli.py errors    # See recent errors
python debug_cli.py export    # Full export
```

## Example Workflow

**You**: "Hey, the AI just told me to play a Swamp but I'm not playing black!"

**You do**: Press F12 or run `python quick_debug.py`

**You say**: "I just exported a debug bundle, can you check debug_bundle_20260110_195234.json?"

**AI**: *Analyzes the JSON and sees:*
- Your hand only had Mountains
- Decklist had no Swamps
- AI prompt mistakenly included old match data
- **Root cause identified in 30 seconds!**

## What the AI Can See

From the exported JSON file, an AI can instantly see:

✅ **Match State**
- Turn number, phase, who has priority
- Your hand cards (by name)
- Battlefield cards (yours and opponent's)
- Life totals, mana pools

✅ **Your Deck**
- Complete decklist (if loaded)
- Cards in each zone

✅ **Recent Activity**
- Last 5 minutes of logs
- All errors with full context
- Match timeline (turn-by-turn events)

✅ **System Health**
- What's running
- What files exist
- Any connection issues

## File Location

Debug bundles are saved to:
```
logs/debug_bundle_YYYYMMDD_HHMMSS.json
```

Example: `logs/debug_bundle_20260110_195234.json`

## Sharing with AI

When sharing with an AI assistant:

1. **Run the export** (F12, or `python quick_debug.py`)
2. **Locate the file** in the `logs/` folder
3. **Upload or paste** the JSON content
4. **Ask your question** - the AI has full context!

### Good Questions to Ask:

- "Why did the AI suggest the wrong card?"
- "Why isn't my deck showing in the library panel?"
- "The app froze during turn 5, what happened?"
- "Why didn't I get advice on my turn?"
- "Is there a bug in the card detection?"

## Privacy Note

The debug bundle contains:
- Your current match data (cards, life, etc.)
- Recent advisor logs
- No personal information
- No API keys (those are never logged)

Safe to share with AI for debugging!

## Troubleshooting the Debug Tool

If `python quick_debug.py` fails:

1. **Make sure you're in the project root**:
   ```bash
   cd /path/to/mtga-voice-assistant
   ```

2. **Check Python is working**:
   ```bash
   python --version
   ```

3. **Try the absolute path**:
   ```bash
   python C:/Users/joshu/mtga-voice-assistant/quick_debug.py
   ```

4. **Still not working?** The logs themselves are in:
   ```
   logs/advisor.log
   ```
   You can share the last 100 lines of that file as a fallback.

## Advanced: What's In The JSON?

The exported file has this structure:

```json
{
  "export_time": "2026-01-10T19:52:34",
  "current_state": {
    "match": {
      "turn": 5,
      "phase": "Phase_Main1",
      "zones": {
        "hand": {"cards": [...], "count": 7},
        "battlefield": {"cards": [...]},
        ...
      }
    },
    "ui": {...},
    "system": {...}
  },
  "recent_logs": [...],
  "recent_errors": [...],
  "match_timeline": [...]
}
```

An AI can parse this instantly and pinpoint issues!
