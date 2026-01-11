# AI-Assisted Debugging Guide

## Quick Start - When Something Goes Wrong

### METHOD 1: One-Click GUI Export (EASIEST)
1. Press **F12** (or click "🔍 Debug Export" button in settings)
2. Wait 2 seconds
3. Share the exported JSON file with your AI assistant

### METHOD 2: Quick Command Line
```bash
python quick_debug.py
```
That's it! Creates a debug bundle instantly.

### METHOD 3: Detailed CLI Tools
For more control:

```bash
# Quick status check
python debug_cli.py status

# Check errors
python debug_cli.py errors

# Export full bundle
python debug_cli.py export
```

## What Gets Exported

The debug bundle (JSON file) contains everything an AI needs:
- Current match state (turn, phase, life, cards in hand/battlefield)
- Recent logs (last 5 minutes)
- All recent errors with context
- Match event timeline
- System health check
- UI state (what you're seeing)

**No more pasting logs!** Just share one JSON file.

### 4. Query Specific Logs
```bash
# Find all errors in last 10 minutes
python debug_cli.py logs --level ERROR --minutes 10

# Search for specific pattern
python debug_cli.py logs --pattern "Swamp" --minutes 30

# Find warnings
python debug_cli.py logs --level WARNING
```

## What's Been Created

### 1. Structured Logger (`src/core/debug_logger.py`)
New logging system with:
- Component tags: `[MTGA_PARSER]`, `[AI_ADVISOR]`, etc.
- Operation boundaries: `▶ START` / `◀ END` markers
- State snapshots: `📸 STATE` markers
- Event markers: `⚡ EVENT`
- Error context: `❌ ERROR` with full details
- Correlation IDs to track related operations

**Usage:**
```python
from src.core.debug_logger import get_logger

logger = get_logger("MY_COMPONENT")

with logger.operation("do_something", param1="value"):
    # Your code here
    logger.state_snapshot("current_data", my_data)
    logger.event("important_thing_happened", detail="info")
```

### 2. AI Debug Interface (`src/core/ai_debug_interface.py`)
API for AI agents to query state and logs:
```python
from src.core.ai_debug_interface import create_debug_interface

debug = create_debug_interface(app_instance)

# Get complete current state
state = debug.get_current_state()

# Query logs
errors = debug.query_logs(level="ERROR", last_minutes=5)

# Export for analysis
bundle = debug.export_debug_bundle()
```

### 3. Debug CLI (`debug_cli.py`)
Command-line tool for quick checks:
```bash
python debug_cli.py status      # Current state
python debug_cli.py errors      # Recent errors
python debug_cli.py timeline    # Match events
python debug_cli.py export      # Full debug bundle
```

## Debugging Workflows

### "Why did the AI suggest the wrong card?"

1. **Export debug bundle**:
   ```bash
   python debug_cli.py export
   ```

2. **Search the bundle for the advice**:
   ```bash
   grep "Swamp" logs/debug_bundle_*.json
   ```

3. **Check what data AI received**:
   Look for `"current_state"` → `"match"` → `"zones"` → `"hand"`

4. **Verify decklist**:
   Look for `"deck"` → `"has_decklist"` and `"unique_cards"`

5. **Check the prompt** (in advisor.log):
   ```bash
   python debug_cli.py logs --pattern "AI PROMPT PREVIEW" --minutes 30
   ```

### "The app froze during a match"

1. **Check recent errors**:
   ```bash
   python debug_cli.py errors
   ```

2. **Look for failed operations**:
   ```bash
   python debug_cli.py logs --pattern "✗ FAILED" --minutes 10
   ```

3. **Check health**:
   ```bash
   python debug_cli.py health
   ```

### "Cards aren't being detected"

1. **Check current state**:
   ```bash
   python debug_cli.py status | grep -A 20 "hand"
   ```

2. **Look for parse errors**:
   ```bash
   python debug_cli.py logs --pattern "[MTGA_PARSER].*ERROR" --minutes 30
   ```

3. **Check zone updates**:
   ```bash
   python debug_cli.py logs --pattern "Found zones" --minutes 5
   ```

## Integration with App

To enable real-time debugging in the main app, add this to `app.py`:

```python
from src.core.ai_debug_interface import create_debug_interface

class MTGAAdvisorApp:
    def __init__(self):
        # ... existing init code ...

        # Create debug interface
        self.debug_interface = create_debug_interface(self)

        # Optional: Auto-export on errors
        self._setup_error_export()

    def _setup_error_export(self):
        """Auto-export debug bundle when critical errors occur."""
        import logging

        class ErrorExportHandler(logging.Handler):
            def __init__(self, debug_interface):
                super().__init__(logging.ERROR)
                self.debug_interface = debug_interface
                self.last_export = 0

            def emit(self, record):
                # Rate limit: only export once per minute
                import time
                now = time.time()
                if now - self.last_export > 60:
                    self.last_export = now
                    self.debug_interface.export_debug_bundle()

        error_handler = ErrorExportHandler(self.debug_interface)
        logging.getLogger().addHandler(error_handler)
```

## For AI Assistants

When a user asks you to debug, request they run:
```bash
python debug_cli.py export
```

Then ask them to share the exported JSON file. You can analyze:

1. **Current match state**: `current_state.match`
2. **Recent operations**: `recent_logs` with `▶ START` and `◀ END`
3. **Error context**: `recent_errors` with full details
4. **Timeline**: `match_timeline` to see event sequence
5. **System health**: `current_state.system`

## Benefits

✅ **Instant State Visibility** - No more "can you paste your logs?"
✅ **Structured Data** - AI can parse JSON easily
✅ **Full Context** - Everything needed to diagnose issues
✅ **Timeline Tracking** - See exactly when things happened
✅ **Easy Sharing** - Single JSON file with all info

## Next Steps

1. **Integrate into app.py** - Add debug interface initialization
2. **Migrate logging** - Start using structured logger in key components
3. **Test workflows** - Try debugging a real issue with the tools
4. **Refine** - Adjust based on actual debugging experience

## Migration Example

**Before:**
```python
logging.info(f"Processing game state")
# ... code ...
logging.debug(f"Hand: {hand_cards}")
logging.info(f"Done processing")
```

**After:**
```python
logger = get_logger("GAME_STATE")

with logger.operation("process_game_state", turn=5):
    # ... code ...
    logger.state_snapshot("hand", hand_cards, count=len(hand_cards))
    # Operation auto-logs success/failure and duration
```

**Benefits:**
- Clear operation boundaries
- Automatic timing
- Structured data
- Easy to filter in logs
