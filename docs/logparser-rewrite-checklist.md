# MTGA Voice Advisor - Complete Rewrite Checklist

**Date**: 2025-10-27
**Purpose**: Guide for Claude Code to rewrite `/mnt/synology/repos/logparser/advisor.py`
**Primary References**: `logparser-fixes.md` + `card-database-optimization.md`

---

## Context: What This Project Is

Single-file (945 lines) Python voice advisor for MTG Arena that:
1. Monitors `Player.log` in real-time
2. Parses GRE (Game Rules Engine) protocol messages
3. Builds board state representation
4. Queries local Ollama LLM for tactical advice
5. Speaks advice using Kokoro TTS

**User's main complaint**: Inaccurate board state → bad LLM context → wrong voice advice

---

## Critical Fixes Required (From External Docs)

### 1. Zone Transfer Parsing ⭐ (FROM: logparser-fixes.md)

**Location**: `advisor.py` line 202-222 (`parse_gre_to_client_event()`)

**Add**:
```python
elif msg_type == "GREMessageType_Annotation":
    state_changed |= self._parse_annotations(message)
```

**Plus**: Add entire `_parse_annotations()` method (see logparser-fixes.md for complete code)

**Impact**: Fixes board state accuracy from 60% → 95%

### 2. Card Database Replacement ⭐ (FROM: card-database-optimization.md)

**Location**: Replace entire `ScryfallClient` class (lines 354-392)

**Replace with**: `ArenaCardDatabase` class (see card-database-optimization.md for complete code)

**Changes needed**:
- Line 735: `ScryfallClient()` → `ArenaCardDatabase()`
- Line 73: Add `detect_card_database_path()` function
- Line 789: Add database cleanup in shutdown

**Impact**: Eliminates 100-300ms API delays, no more `Unknown(grpId)`

---

## Additional Issues to Fix (Not in External Docs)

### Issue #3: Hardcoded Zone IDs (Wrong)

**Location**: Lines 172-188

**Current code** (WRONG):
```python
class MatchScanner:
    ZONE_HAND = 2
    ZONE_BATTLEFIELD = 3
    ZONE_GRAVEYARD = 4
    ZONE_STACK = 1
    ZONE_LIBRARY = 0

    ZONE_TYPE_MAP = {
        'ZoneType_Hand': 2,
        'ZoneType_Battlefield': 3,
        # ... etc (all hardcoded)
    }
```

**Problem**: Zone IDs are **assigned dynamically per match** by Arena. From logs:
```
Zone mapping: ZoneType_Hand -> zoneId 31 (not 2!)
Zone mapping: ZoneType_Battlefield -> zoneId 28 (not 3!)
```

**Fix**: These constants are unused because `_parse_zones()` already builds `zone_type_to_ids` dynamically (line 304). **Delete lines 172-188 entirely.**

### Issue #4: Zone Filtering Logic (Inefficient)

**Location**: Lines 516-536 (`get_current_board_state()`)

**Current approach**:
```python
# Discovers zone IDs every time
for zone_type_str, zone_id in self.scanner.zone_type_to_ids.items():
    if "Hand" in zone_type_str:
        hand_zone_id = zone_id
```

**Better approach** (after zone transfers are added):
```python
# Zone IDs are stable after game start, cache them
if not hasattr(self, '_cached_hand_zone_id'):
    self._cached_hand_zone_id = next(
        (zid for ztype, zid in self.scanner.zone_type_to_ids.items() if "Hand" in ztype),
        None
    )
hand_zone_id = self._cached_hand_zone_id
```

**Impact**: Minor performance improvement, cleaner code

### Issue #5: Debug Logging Spam (Performance)

**Location**: Lines 119, 265, 502-503

**Current code**:
```python
logging.debug(f"Read line: {stripped_line[:100]}...")  # EVERY LINE!
logging.info(f"  GameObject: instanceId={instance_id}, grpId={grp_id}...")  # EVERY OBJECT!
```

**Problem**: With DEBUG enabled, writes thousands of lines per second to disk

**Fix**:
```python
# Only log at DEBUG when explicitly enabled
if logging.getLogger().isEnabledFor(logging.DEBUG):
    logging.debug(f"Read line: {stripped_line[:100]}...")

# Change GameObject logging to DEBUG instead of INFO
logging.debug(f"  GameObject: instanceId={instance_id}...")  # Was INFO
```

**Impact**: 10-20x faster log processing in production

---

## Code Structure to Keep

### ✅ Keep These (They Work Well):

1. **LogFollower class** (lines 79-131)
   - Inode tracking for log rotation
   - File seek/tell for efficient tailing
   - **Keep as-is**

2. **JSON buffering** (lines 421-467)
   - Handles multi-line GRE messages correctly
   - Depth tracking with `{` and `}` counting
   - **Keep as-is**

3. **GameStateManager._find_gre_event()** (lines 405-419)
   - Recursive search for greToClientEvent
   - Handles nested structures
   - **Keep as-is**

4. **OllamaClient** (lines 545-565)
   - Simple HTTP client for LLM
   - **Keep as-is**

5. **TextToSpeech (Kokoro)** (lines 605-713)
   - TTS with voice selection
   - **Keep as-is**

6. **CLIVoiceAdvisor orchestration** (lines 718-939)
   - Thread management
   - CLI command handling
   - **Keep structure, improve edge cases**

### ⚠️ Modify These:

1. **MatchScanner** (lines 171-348)
   - ✅ Keep structure
   - ⭐ Add `_parse_annotations()` method
   - ❌ Remove hardcoded zone constants
   - ✅ Keep `_parse_game_state_message()`
   - ✅ Keep `_parse_zones()`

2. **ScryfallClient** → **ArenaCardDatabase** (lines 354-392)
   - ❌ Delete entire class
   - ⭐ Replace with new implementation

3. **BoardState building** (lines 469-539)
   - ✅ Keep structure
   - ⚠️ Add validation (from logparser-fixes.md)
   - ⚠️ Optimize zone lookups

---

## Integration Points Between Fixes

### Fix #1 (Zone Transfers) + Fix #2 (Arena DB)

**How they work together**:

```python
# Zone transfer detected
def _parse_annotations(self, message: dict) -> bool:
    for instance_id in affected_ids:
        if instance_id in self.game_objects:
            obj = self.game_objects[instance_id]
            obj.zone_id = zone_dest  # ← Zone updated

            # Get card name from Arena DB (Fix #2)
            card_name = self.card_lookup.get_card_name(obj.grp_id)  # ← Fast!

            logging.info(f"⚡ Zone transfer: {card_name} {zone_src} → {zone_dest}")
```

**Key point**: Zone transfers update `zone_id` in real-time, and Arena DB provides instant card names for logging/debugging.

### Board State Building

**After both fixes**:

```python
def get_current_board_state(self) -> Optional[BoardState]:
    # ... existing setup ...

    for obj in self.scanner.game_objects.values():
        # Fix #2: Instant card name lookup (was 100-300ms)
        obj.name = self.card_lookup.get_card_name(obj.grp_id)

        # Fix #1: Accurate zone_id (from zone transfers)
        if obj.zone_id == hand_zone_id:  # ← Always accurate now
            board_state.your_hand.append(obj)

    # NEW: Validate before returning
    if not self.validate_board_state(board_state):
        logging.warning("Board state validation failed")
        return None

    return board_state
```

---

## Testing Checklist

### Test #1: Zone Transfers Working
```bash
python advisor.py
# Start MTGA match, draw a card
tail -f logs/advisor.log | grep "Zone transfer"
```
**Expected**: `⚡ Zone transfer: Plains Library → Hand (Draw)`

### Test #2: Arena DB Working
```bash
python advisor.py
```
**Expected startup log**:
```
✓ Connected to Arena card database: /path/to/Raw_CardDatabase_*.mtga
✓ Loaded 21481 cards from Arena database
```

**Expected during game**:
```
Board State Summary: Your Hand: ['Swamp', 'Forsaken Miner'], Your Battlefield: ['Infestation Sage']
```
**No more `Unknown(grpId)`!**

### Test #3: LLM Context Accuracy
```bash
# Play a creature, check what LLM received
grep "Your Battlefield:" logs/advisor.log | tail -1
```
**Expected**: Lists actual card names, not `Unknown(grpId)`

### Test #4: TTS Output Quality
**Play a match, listen to voice advice**

**Expected**: TTS speaks actual card names, not "Unknown card 87485"

---

## Rewrite Strategy

### Approach 1: Minimal Changes (Recommended)
1. Add zone transfer parsing (50 lines)
2. Replace ScryfallClient with ArenaCardDatabase (150 lines)
3. Remove hardcoded zone constants (delete 20 lines)
4. Add board state validation (30 lines)
5. Fix debug logging levels (10 changes)

**Total changes**: ~250 lines modified/added, 20 deleted

**Risk**: Low - most code stays the same

### Approach 2: Full Rewrite
1. Reorganize into multiple files
2. Add type hints everywhere
3. Improve error handling
4. Add unit tests
5. Better separation of concerns

**Total changes**: Entire file restructured

**Risk**: High - might introduce new bugs

**Recommendation**: Use Approach 1. The existing code works, just needs the two critical fixes.

---

## Order of Implementation

### Phase 1: Critical Fixes (Do First)
1. ⭐ Add `detect_card_database_path()` function
2. ⭐ Replace `ScryfallClient` with `ArenaCardDatabase` class
3. ⭐ Update instantiation (line 735)
4. ⭐ Test: Verify no more `Unknown(grpId)`

### Phase 2: Accuracy Fix (Do Second)
5. ⭐ Add `elif msg_type == "GREMessageType_Annotation"` (line ~220)
6. ⭐ Add `_parse_annotations()` method
7. ⭐ Test: Verify zone transfer logs appear

### Phase 3: Polish (Do Third)
8. ✅ Add board state validation
9. ✅ Remove hardcoded zone constants
10. ✅ Fix debug logging levels
11. ✅ Add database cleanup

### Phase 4: Documentation (Optional)
12. Add docstrings to new methods
13. Update inline comments
14. Add usage examples

---

## Known Limitations (Don't Try to Fix)

### 1. Ollama LLM Latency (2-5 seconds)
**Not fixable in this rewrite.** This is model inference time.

**Possible future optimizations**:
- Use streaming responses
- Use smaller model (llama3.2:1b)
- Cache similar board states

**But don't include in this rewrite** - focus on accuracy first.

### 2. Log Polling Interval (100ms)
**Already optimal.** Could reduce to 50ms but minimal benefit.

### 3. TTS Generation Time (200-500ms)
**Not fixable in this rewrite.** Kokoro TTS is fast enough.

---

## What Success Looks Like

### Before Rewrite:
```
Board State Summary: Your Hand: ['Swamp', 'Unknown(87485)'],
Your Battlefield: ['Unknown(96697)', 'Swamp', 'Unknown(189222)']

LLM Advice: "Consider playing your unknown cards strategically"
TTS: 🔊 "Play unknown card eighty seven thousand..."
```

### After Rewrite:
```
Board State Summary: Your Hand: ['Swamp', 'Forsaken Miner'],
Your Battlefield: ['Infestation Sage', 'Swamp', 'Greedy Freebooter']

LLM Advice: "Attack with Infestation Sage, hold Forsaken Miner as blocker"
TTS: 🔊 "Attack with Infestation Sage for early pressure"
```

**Key metrics**:
- ✅ 0 `Unknown(grpId)` entries
- ✅ Board state matches MTGA UI exactly
- ✅ LLM advice references actual cards
- ✅ TTS speaks useful tactical advice
- ✅ <1s latency from game action → board state update

---

## Files to Provide to Claude

### Required:
1. **logparser-fixes.md** - Zone transfer implementation
2. **card-database-optimization.md** - Arena DB implementation
3. **This file** - Integration guidance and additional fixes

### Optional:
4. **advisor.py** - Current implementation (for context)
5. **logs/advisor.log** - Sample logs showing current issues

---

## Prompt Template for Claude

```
I need you to rewrite /mnt/synology/repos/logparser/advisor.py to fix critical
accuracy and performance issues.

Requirements:
1. Add zone transfer parsing (see logparser-fixes.md section "Fix #1")
2. Replace Scryfall API with Arena SQLite database (see card-database-optimization.md)
3. Remove hardcoded zone constants (lines 172-188)
4. Add board state validation before sending to LLM
5. Keep existing code structure where possible (minimal changes approach)

Testing criteria:
- No more "Unknown(grpId)" in logs
- Zone transfer logs appear: "⚡ Zone transfer: CardName ZoneA → ZoneB"
- Board state matches MTGA UI exactly
- Startup shows: "✓ Loaded 21481 cards from Arena database"

Please show me the complete rewritten advisor.py file.
```

---

## Common Pitfalls to Avoid

### ❌ Don't:
1. Add memory scanning (not needed, slower than logs)
2. Use OCR or UI automation (doesn't work with Unity)
3. Rewrite LogFollower (it works perfectly)
4. Change JSON buffering logic (complex but correct)
5. Optimize LLM inference (out of scope)
6. Add multi-threading for log parsing (already async)

### ✅ Do:
1. Keep single-file structure (it's intentional)
2. Preserve thread-based AI advice generation
3. Maintain CLI command interface
4. Keep platform detection for cross-OS support
5. Test on actual MTGA matches

---

## Summary: What Claude Needs

**Primary Documents**:
- ✅ `logparser-fixes.md` - Has complete zone transfer code
- ✅ `card-database-optimization.md` - Has complete Arena DB code
- ✅ This checklist - Integration guidance

**Additional Context**:
- Current file: `/mnt/synology/repos/logparser/advisor.py`
- Logs: `/mnt/synology/repos/logparser/logs/advisor.log`
- Test environment: Live MTGA matches on Linux

**Expected outcome**: Single rewritten `advisor.py` file that fixes both accuracy and performance issues with minimal changes to working code.

---

END OF DOCUMENT
