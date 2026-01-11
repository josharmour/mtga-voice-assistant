# Logging Overhaul Plan - AI-Assisted Debugging

## Goals
1. **Structured Output** - Consistent format that's easy to parse
2. **Easy Searching** - Clear patterns for grep/search
3. **Context Tracking** - Follow operations across components
4. **State Visibility** - See what data is being processed
5. **Performance Monitoring** - Track slow operations
6. **Error Context** - Full context when things fail

## New Logging System

### Features

#### 1. Component Tags
Every log has a clear component identifier:
```
[MTGA_PARSER] ▶ START: parse_game_state | turn=5 phase=Main1
[AI_ADVISOR] ⚡ EVENT: advice_requested | trigger=TURN_START
[DECK_BUILDER] 📸 STATE: decklist | size=60
```

#### 2. Operation Boundaries
Clear start/end markers with timing:
```
[GAME_STATE] ▶ START: get_board_state | turn=3
[GAME_STATE] ◀ END (✓ SUCCESS): get_board_state | duration_ms=45.23
```

#### 3. Correlation IDs
Track related operations across components:
```
[MTGA_PARSER] [CORR:match_abc123] ▶ START: parse_log_line
[AI_ADVISOR] [CORR:match_abc123] ⚡ EVENT: advice_requested
[LLM_CLIENT] [CORR:match_abc123] ◀ END (✓ SUCCESS): api_call | duration_ms=1205.42
```

#### 4. State Snapshots
Critical state at key points:
```
[GAME_STATE] 📸 STATE: hand | preview=['Mountain', 'Lightning Bolt', '...'] | size=7
[DECK_LOADER] 📸 STATE: decklist | size=60 | unique_cards=40
```

#### 5. Event Markers
Important events clearly marked:
```
[MATCH_SCANNER] ⚡ EVENT: match_started | match_id=abc123
[MATCH_SCANNER] ⚡ EVENT: match_ended | winner=player | turns=12
```

#### 6. Error Context
Errors with full contextual information:
```
[CARD_DB] ❌ ERROR: card_lookup_failed | card_id=12345 | exception_type=KeyError
```

## Migration Strategy

### Phase 1: Core Components (Week 1)
- [x] Create `debug_logger.py` module
- [ ] Migrate `mtga.py` (log parser)
  - Add operation boundaries for parsing
  - State snapshots for game state
  - Correlation IDs per match
- [ ] Migrate `ai.py` (AI advisor)
  - Log AI requests with full context
  - Track API call performance
  - Log prompts and responses

### Phase 2: Support Components (Week 2)
- [ ] Migrate `app.py` (main orchestrator)
- [ ] Migrate `ui.py` (GUI)
- [ ] Migrate `draft_advisor.py`
- [ ] Migrate `deck_builder.py`

### Phase 3: LLM Adapters (Week 3)
- [ ] Migrate all `llm/*.py` files
- [ ] Add request/response logging
- [ ] Track API performance metrics

## Usage Examples

### Before (Old Style):
```python
logging.info(f"Parsing game state message")
logging.debug(f"Hand cards: {hand_cards}")
logging.info(f"Board state updated")
```

### After (New Style):
```python
logger = get_logger("MTGA_PARSER")

with logger.operation("parse_game_state", turn=5, phase="Main1"):
    # Do parsing work
    logger.state_snapshot("hand", hand_cards, count=len(hand_cards))
    logger.event("board_state_updated", changed_zones=["hand", "battlefield"])
```

## Debugging Workflows

### Find Why AI Gave Bad Advice
```bash
# 1. Find the advice request
grep "⚡ EVENT: advice_requested" advisor.log | tail -n 1

# 2. Get the correlation ID from that line
# [AI_ADVISOR] [CORR:advice_xyz] ⚡ EVENT: advice_requested

# 3. See all related operations
grep "CORR:advice_xyz" advisor.log

# 4. Check the board state data
grep "CORR:advice_xyz" advisor.log | grep "📸 STATE"

# 5. See the AI prompt
grep "CORR:advice_xyz" advisor.log | grep "AI PROMPT"
```

### Find Slow Operations
```bash
# Find all operations that took > 100ms
grep "duration_ms=" advisor.log | awk -F'duration_ms=' '{if ($2+0 > 100) print}'

# Find slowest operations
grep "duration_ms=" advisor.log | awk -F'duration_ms=' '{print $2, $0}' | sort -n -r | head -n 10
```

### Track a Match From Start to Finish
```bash
# Find match start
grep "⚡ EVENT: match_started" advisor.log | tail -n 1
# Get match_id: match_abc123

# See all match operations
grep "match_abc123" advisor.log

# See turn-by-turn timeline
grep "match_abc123" advisor.log | grep "turn="
```

### Debug Parse Errors
```bash
# Find all parsing errors
grep "\[MTGA_PARSER\].*❌ ERROR" advisor.log

# See context around an error
grep -B 10 -A 10 "parse.*ERROR" advisor.log | tail -n 30
```

## Benefits for AI Debugging

1. **Pattern Matching**: AI can easily identify log patterns
2. **Context Extraction**: Structured format makes data extraction simple
3. **Timeline Reconstruction**: Correlation IDs link related events
4. **State Comparison**: State snapshots show data at decision points
5. **Performance Analysis**: Metrics help identify bottlenecks
6. **Error Diagnosis**: Full context in error messages

## Implementation Priority

**HIGH PRIORITY** (Do first):
1. `mtga.py` - Most complex, most bugs
2. `ai.py` - Critical for debugging advice
3. `app.py` - Main orchestration

**MEDIUM PRIORITY**:
4. `ui.py` - User-facing issues
5. `draft_advisor.py` - Draft-specific bugs

**LOW PRIORITY** (Nice to have):
6. LLM adapters - Already have basic logging
7. Utility modules - Less critical

## Rollout Plan

1. **Week 1**: Implement Phase 1, test with real matches
2. **Week 2**: Gather feedback, refine format
3. **Week 3**: Implement Phases 2-3
4. **Week 4**: Full migration complete

## Success Metrics

- ✅ Time to diagnose issues reduced by 50%
- ✅ Can trace any operation end-to-end with correlation IDs
- ✅ AI assistant can answer "why did X happen?" questions
- ✅ Performance bottlenecks easily identified
- ✅ Error context sufficient to fix bugs without reproduction
