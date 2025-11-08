# Auto-Download Integration Guide

This guide explains how to integrate auto-download functionality into the MTGA Voice Advisor.

## Current Status

✅ **Manual download**: `download_real_17lands_data.py`
✅ **Update checker**: `update_card_data.py --status`
✅ **Auto updater**: `update_card_data.py --auto`
✅ **Draft detector**: `auto_draft_detector.py` (NEW)
✅ **Coverage analyzer**: `analyze_data_coverage.py` (NEW)

## Integration Steps

### 1. Import the DraftDetector

Add to `advisor.py` (around line 2950, near other imports):

```python
from auto_draft_detector import DraftDetector
```

### 2. Initialize in CLIVoiceAdvisor.__init__()

Add to the `__init__()` method (around line 2960):

```python
# Auto draft detection and data download
self.draft_detector = DraftDetector(
    db_path="data/card_stats.db",
    auto_download=True  # Set to False for manual mode
)
```

### 3. Process Log Lines in LogFollower

Modify the `MatchScanner.process_log_line()` method (around line 300) to detect draft entry:

```python
def process_log_line(self, line: str):
    """Process a single line from the log file."""

    # EXISTING CODE: detect GreToClientEvent
    if "[UnityCrossThreadLogger]GreToClientEvent" in line:
        self.buffer.append(line)
        return

    # NEW CODE: detect draft entry
    if "Event_Join" in line:
        missing_set = self.draft_detector.process_log_line(line)

        if missing_set and self.draft_detector.auto_download:
            # Show notification in UI
            logger.warning(f"Missing data for {missing_set} - downloading...")

            # Download in background thread to avoid blocking
            import threading
            download_thread = threading.Thread(
                target=self.draft_detector.download_missing_data,
                args=(missing_set, True),  # interactive=True
                daemon=True
            )
            download_thread.start()

    # EXISTING CODE: continue with normal processing
    if self.buffer:
        # ... rest of method
```

### 4. Alternative: Startup Data Check

For a simpler integration, add a startup check before the main loop:

```python
def main():
    """Main entry point"""

    # NEW CODE: Check data coverage on startup
    print("\n🔍 Checking card data coverage...")

    from auto_draft_detector import DraftDetector
    from analyze_data_coverage import STANDARD_SETS_2025

    detector = DraftDetector()
    missing_sets = []

    for set_code in STANDARD_SETS_2025:
        if not detector.check_set_in_database(set_code):
            missing_sets.append(set_code)

    if missing_sets:
        print(f"\n⚠️  Missing data for {len(missing_sets)} Standard sets:")
        for set_code in missing_sets:
            from download_real_17lands_data import ALL_SETS
            print(f"   • {set_code}: {ALL_SETS.get(set_code, 'Unknown')}")

        print("\nRun this command to download missing data:")
        print("   python update_card_data.py --auto")
        print()

        response = input("Download now? (y/n): ").strip().lower()
        if response == 'y':
            from update_card_data import update_sets
            update_sets(missing_sets, interactive=False)
    else:
        print("✅ All Standard sets have data\n")

    # EXISTING CODE: start advisor
    advisor = CLIVoiceAdvisor()
    advisor.run()
```

## Integration Options

### Option A: Aggressive Auto-Download (Recommended)
- Detects draft entry from `Event_Join` message
- Automatically downloads missing data in background
- Shows progress notification in UI
- User can continue playing while download happens

**Pros**: Seamless experience, always has data when needed
**Cons**: Unexpected downloads (5-15 min, 500MB-2GB)

### Option B: Startup Check + Manual Download
- Checks for missing Standard sets on startup
- Prompts user to download before starting
- User chooses what to download

**Pros**: User has full control, no surprises
**Cons**: Might miss flashback draft sets

### Option C: Notification Only
- Detects draft entry
- Warns user about missing data
- Provides command to run manually

**Pros**: Non-intrusive, user fully in control
**Cons**: User has to stop and run download command

## Implementation Recommendation

For best user experience, use a **hybrid approach**:

1. **Startup check** for Standard sets (Option B)
2. **Detection + notification** for flashback sets (Option C)
3. **Auto-download** only if user enables it in config

### Example Configuration

Add to a config file or as command-line flag:

```python
# Config options
auto_download_standard = True   # Auto-download Standard sets on startup
auto_download_flashback = False # Prompt for flashback sets
notify_missing_data = True      # Show warnings when data is missing
```

## Testing the Integration

### Test Draft Detection

```bash
# Test the detector standalone
python auto_draft_detector.py --test

# Check specific sets
python auto_draft_detector.py --check BLB
python auto_draft_detector.py --check OTJ

# Test download (with prompt)
python auto_draft_detector.py --download KTK
```

### Verify Coverage

```bash
# Full coverage report
python analyze_data_coverage.py

# Quick status check
python update_card_data.py --status
```

## Performance Considerations

### Download Times
- **Quick sets** (KTK, HBG): 5-10 minutes, 500MB
- **Standard sets** (BLB, OTJ): 10-20 minutes, 1-3GB
- **Large sets** (DMU, MOM): 20-30 minutes, 3-5GB

### Background Download
- Downloads happen in separate thread
- Parsing is CPU-intensive (can cause lag)
- Consider showing progress bar in TUI/GUI

### Database Size
- Current: 1.1 MB (29 sets, 7,917 cards)
- Growth: ~40 KB per set
- Full coverage: ~1.5 MB (all available sets)

## Future Enhancements

1. **Incremental updates**: Only download new game data (17lands provides daily updates)
2. **Smart caching**: Keep only recent/popular sets, auto-remove old data
3. **Cloud sync**: Share database across devices
4. **Predictive download**: Pre-download upcoming sets based on Arena schedule
5. **Background refresh**: Auto-update stale data (>90 days old)

## Troubleshooting

### "Download failed for SET"
- Check internet connection
- Verify set is available on 17lands S3
- Try running: `python check_available_sets.py`

### "Database locked"
- Close other instances of advisor
- Check for stale lock files

### "No cards parsed"
- CSV might be corrupted
- Delete and re-download: `rm data/17lands_SET_*.csv`

## Quick Reference Commands

```bash
# Check coverage
python analyze_data_coverage.py

# Download all Standard sets
python download_real_17lands_data.py  # Choose option 3

# Update outdated sets
python update_card_data.py --auto

# Test draft detection
python auto_draft_detector.py --test

# Check specific set
python auto_draft_detector.py --check BLB

# Download specific set
python auto_draft_detector.py --download OTJ
```
