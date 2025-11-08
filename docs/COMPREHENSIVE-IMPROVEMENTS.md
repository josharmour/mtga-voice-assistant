# MTGA Voice Advisor - Comprehensive Improvement Plan

**Date**: 2025-10-27
**Goal**: Transform advisor from basic board state tracking to full MTG strategic analysis

---

## Current Status vs. Target

### ✅ Currently Implemented
- Zone transfer parsing (accurate real-time tracking)
- Arena card database (instant card lookups)
- Hand and battlefield tracking
- Basic AI prompting
- TTS output

### ❌ Missing Critical Features
1. **Graveyard tracking** - needed for recursion, reanimation, flashback
2. **Exile tracking** - needed for impulse draw, foretell, adventure cards
3. **Library/deck analysis** - needed for draw probability, tutor targets
4. **Stack tracking** - needed for instant-speed interaction advice
5. **Game history** - needed for "this turn" context
6. **Mana tracking** - needed for "what can I cast?" analysis
7. **MTG rules context** - AI needs to know mechanics

---

## Phase 1: Expand Zone Tracking

### 1A. Add Graveyard & Exile to BoardState

**File**: `advisor.py` line 186-201

**Current BoardState**:
```python
@dataclasses.dataclass
class BoardState:
    your_seat_id: int
    opponent_seat_id: int
    your_life: int = 20
    your_hand_count: int = 0
    your_hand: List[GameObject] = dataclasses.field(default_factory=list)
    your_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_life: int = 20
    opponent_hand_count: int = 0
    opponent_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    current_turn: int = 0
    current_phase: str = ""
    is_your_turn: bool = False
    has_priority: bool = False
```

**Enhanced BoardState**:
```python
@dataclasses.dataclass
class BoardState:
    # Player identification
    your_seat_id: int
    opponent_seat_id: int

    # Life totals
    your_life: int = 20
    opponent_life: int = 20

    # Zone: Hand
    your_hand_count: int = 0
    your_hand: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_hand_count: int = 0

    # Zone: Battlefield
    your_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_battlefield: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Graveyard (NEW!)
    your_graveyard: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_graveyard: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Exile (NEW!)
    your_exile: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_exile: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Library (NEW!)
    your_library_count: int = 0
    opponent_library_count: int = 0

    # Zone: Stack (NEW!)
    stack: List[GameObject] = dataclasses.field(default_factory=list)

    # Turn tracking
    current_turn: int = 0
    current_phase: str = ""
    is_your_turn: bool = False
    has_priority: bool = False
```

### 1B. Update Zone Filtering in get_current_board_state()

**File**: `advisor.py` line 698-738

**Current filtering** (only hand + battlefield):
```python
for obj in self.scanner.game_objects.values():
    obj.name = self.card_lookup.get_card_name(obj.grp_id)

    if obj.owner_seat_id == your_seat_id:
        if hand_zone_id and obj.zone_id == hand_zone_id:
            board_state.your_hand.append(obj)
        elif battlefield_zone_id and obj.zone_id == battlefield_zone_id:
            board_state.your_battlefield.append(obj)
```

**Enhanced filtering** (all zones):
```python
# Discover all zone IDs
hand_zone_id = None
battlefield_zone_id = None
graveyard_zone_id = None
exile_zone_id = None
library_zone_id = None
stack_zone_id = None

for zone_type_str, zone_id in self.scanner.zone_type_to_ids.items():
    if "Hand" in zone_type_str:
        hand_zone_id = zone_id
    elif "Battlefield" in zone_type_str:
        battlefield_zone_id = zone_id
    elif "Graveyard" in zone_type_str:
        graveyard_zone_id = zone_id
    elif "Exile" in zone_type_str:
        exile_zone_id = zone_id
    elif "Library" in zone_type_str:
        library_zone_id = zone_id
    elif "Stack" in zone_type_str:
        stack_zone_id = zone_id

# Filter objects by zone
for obj in self.scanner.game_objects.values():
    obj.name = self.card_lookup.get_card_name(obj.grp_id)

    if obj.owner_seat_id == your_seat_id:
        if hand_zone_id and obj.zone_id == hand_zone_id:
            board_state.your_hand.append(obj)
        elif battlefield_zone_id and obj.zone_id == battlefield_zone_id:
            board_state.your_battlefield.append(obj)
        elif graveyard_zone_id and obj.zone_id == graveyard_zone_id:
            board_state.your_graveyard.append(obj)
        elif exile_zone_id and obj.zone_id == exile_zone_id:
            board_state.your_exile.append(obj)
    elif obj.owner_seat_id == opponent_seat_id:
        if battlefield_zone_id and obj.zone_id == battlefield_zone_id:
            board_state.opponent_battlefield.append(obj)
        elif graveyard_zone_id and obj.zone_id == graveyard_zone_id:
            board_state.opponent_graveyard.append(obj)
        elif exile_zone_id and obj.zone_id == exile_zone_id:
            board_state.opponent_exile.append(obj)

    # Stack is shared between both players
    if stack_zone_id and obj.zone_id == stack_zone_id:
        board_state.stack.append(obj)

# Count library cards
for obj in self.scanner.game_objects.values():
    if library_zone_id and obj.zone_id == library_zone_id:
        if obj.owner_seat_id == your_seat_id:
            board_state.your_library_count += 1
        elif obj.owner_seat_id == opponent_seat_id:
            board_state.opponent_library_count += 1
```

---

## Phase 2: Add Game History Tracking

### 2A. Create GameHistory Class

**File**: `advisor.py` - Add after GameObject class

```python
@dataclasses.dataclass
class GameHistory:
    """Tracks important events from current turn for tactical context"""
    turn_number: int

    # Cards played this turn
    cards_played_this_turn: List[GameObject] = dataclasses.field(default_factory=list)

    # Creatures that attacked this turn
    attackers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)

    # Creatures that blocked this turn
    blockers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)

    # Damage dealt this turn
    damage_dealt: Dict[int, int] = dataclasses.field(default_factory=dict)  # {instanceId: damage}

    # Cards that died this turn
    died_this_turn: List[str] = dataclasses.field(default_factory=list)  # Card names

    # Mana spent this turn
    mana_spent: Dict[str, int] = dataclasses.field(default_factory=dict)  # {"W": 2, "U": 1, ...}

    # Lands played this turn
    lands_played_this_turn: int = 0
```

### 2B. Update MatchScanner to Track History

**File**: `advisor.py` - MatchScanner.__init__

```python
def __init__(self):
    self.game_objects: Dict[int, GameObject] = {}
    self.players: Dict[int, PlayerState] = {}
    self.current_turn = 0
    self.current_phase = ""
    self.active_player_seat: Optional[int] = None
    self.priority_player_seat: Optional[int] = None
    self.local_player_seat_id: Optional[int] = None
    self.zone_type_to_ids: Dict[str, int] = {}
    self.observed_zone_ids: set = set()
    self.zone_id_to_type: Dict[int, str] = {}

    # NEW: Game history tracking
    self.game_history: GameHistory = GameHistory(turn_number=0)
```

### 2C. Update Zone Transfer Parsing to Track History

**File**: `advisor.py` - _parse_annotations method

**Add after zone transfer detection**:
```python
# Update game objects with new zones
for instance_id in affected_ids:
    if instance_id in self.game_objects:
        obj = self.game_objects[instance_id]
        old_zone = obj.zone_id

        if zone_dest is not None:
            obj.zone_id = zone_dest

            # Get zone names
            zone_src_name = self.zone_id_to_type.get(zone_src, f"Zone{zone_src}")
            zone_dest_name = self.zone_id_to_type.get(zone_dest, f"Zone{zone_dest}")

            # NEW: Track history events
            if "Battlefield" in zone_dest_name:
                # Card was played/entered battlefield
                self.game_history.cards_played_this_turn.append(obj)

            if "Graveyard" in zone_dest_name and "Battlefield" in zone_src_name:
                # Creature died
                card_name = self.game_objects[instance_id].name if hasattr(self.game_objects[instance_id], 'name') else f"Card{instance_id}"
                self.game_history.died_this_turn.append(card_name)

            logging.info(f"⚡ Zone transfer: Card {instance_id} (grpId:{obj.grp_id}) "
                       f"{zone_src_name} → {zone_dest_name} ({category})")

            state_changed = True
```

### 2D. Add Turn Reset

**File**: `advisor.py` - _parse_turn_info method

```python
def _parse_turn_info(self, turn_info: dict) -> bool:
    state_changed = False
    if self.priority_player_seat != turn_info.get("priorityPlayer"):
        self.priority_player_seat = turn_info.get("priorityPlayer")
        state_changed = True

    # NEW: Reset history on new turn
    new_turn = turn_info.get("turnNumber")
    if self.current_turn != new_turn:
        self.current_turn = new_turn
        self.game_history = GameHistory(turn_number=new_turn)
        state_changed = True

    self.current_phase = turn_info.get("phase", self.current_phase)
    self.active_player_seat = turn_info.get("activePlayer", self.active_player_seat)
    return state_changed
```

---

## Phase 3: Add Deck List Tracking

### 3A. Parse Deck from Player.log

The Player.log contains deck lists in `ClientToMatchServiceMessageType_ClientToGREUIMessage` events.

**Add new method to MatchScanner**:

```python
def _parse_deck_submission(self, message: dict) -> bool:
    """
    Parse deck list from ClientToGREMessage when game starts.
    This gives us the complete 60-card (or 40-card) deck list.
    """
    if "payload" not in message:
        return False

    payload = message["payload"]

    # Look for deck submission or game start messages
    if "deckCards" in payload:
        deck_cards = payload["deckCards"]

        # Store the deck list
        self.deck_list = {}  # {grpId: count}
        for card_entry in deck_cards:
            grp_id = card_entry.get("grpId")
            count = card_entry.get("quantity", 1)
            if grp_id:
                self.deck_list[grp_id] = count

        logging.info(f"📋 Deck loaded: {len(self.deck_list)} unique cards")
        return True

    return False
```

### 3B. Calculate Remaining Deck Composition

**Add method to BoardState**:

```python
def get_remaining_deck_stats(self, known_cards: Dict[int, int], cards_drawn: List[GameObject]) -> Dict[str, int]:
    """
    Calculate what's left in library based on deck list and cards seen.

    Returns color distribution of remaining cards for draw probability analysis.
    """
    remaining = known_cards.copy()

    # Subtract cards we've drawn/seen
    for card in cards_drawn:
        if card.grp_id in remaining:
            remaining[card.grp_id] = max(0, remaining[card.grp_id] - 1)

    # Calculate color distribution
    color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}

    for grp_id, count in remaining.items():
        # Get card data to determine colors
        # (would need card_lookup integration here)
        pass

    return color_counts
```

---

## Phase 4: Enhanced AI Prompt

### 4A. Include MTG Rules Context

**File**: `advisor.py` - AIAdvisor class

**Current system prompt**:
```python
SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor.

CRITICAL RULES:
1. Do NOT recap the board state, life totals, or list cards. The player can already see this.
2. ONLY reference cards that are explicitly listed in the board state. Do NOT invent cards.
3. If a card shows as "Unknown", you cannot know what it is.

Give ONLY tactical advice in 1-2 short sentences. Start directly with your recommendation."""
```

**Enhanced system prompt with MTG rules**:
```python
SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor with comprehensive rules knowledge.

MTG RULES CONTEXT:
- Phases: Untap → Upkeep → Draw → Main1 → Combat (Begin → Declare Attackers → Declare Blockers → Damage) → Main2 → End
- Priority: Players can cast instants/activate abilities when they have priority
- Stack: Spells/abilities resolve in reverse order (last in, first out)
- Combat: Attackers attack players/planeswalkers. Defenders assign blockers. Damage is dealt simultaneously.
- Graveyard: Dead creatures, used spells, discarded cards
- Exile: Permanently removed cards (some can return via specific effects)
- Library: Remaining deck. Empty library = loss on next draw.

CRITICAL RULES FOR ADVICE:
1. Do NOT recap board state - give actionable tactics only
2. ONLY reference cards explicitly listed below
3. Consider ALL zones (hand, battlefield, graveyard, exile, stack)
4. Account for game history (cards played this turn, combat this turn)
5. Evaluate mana efficiency and card advantage
6. Consider deck composition when advising tutors/draw

Give 1-2 sentence tactical recommendation focused on optimal play sequence."""
```

### 4B. Comprehensive Prompt Builder

**File**: `advisor.py` - AIAdvisor._build_prompt method

**Current prompt**:
```python
def _build_prompt(self, board_state: BoardState) -> str:
    lines = [
        f"Turn {board_state.current_turn}, {board_state.current_phase} Phase.",
        f"You have {board_state.your_life} life.",
        f"Opponent has {board_state.opponent_life} life.",
        "",
        "Your Hand (X cards):",
        *[f"{i}. {card.name}" for i, card in enumerate(board_state.your_hand, 1)],
        "",
        "Your Battlefield (X permanents):",
        *[f"{i}. {card.name}" for i, card in enumerate(board_state.your_battlefield, 1)],
        "",
        "Opponent's Battlefield (X permanents):",
        *[f"{i}. {card.name}" for i, card in enumerate(board_state.opponent_battlefield, 1)],
        "",
        "Using ONLY the cards listed above, what is the optimal play sequence?"
    ]
    return "\n".join(lines)
```

**Enhanced prompt with all zones**:
```python
def _build_prompt(self, board_state: BoardState, game_history: GameHistory = None) -> str:
    lines = [
        f"=== GAME STATE: Turn {board_state.current_turn} ===",
        f"Phase: {board_state.current_phase}",
        f"Your Life: {board_state.your_life} | Opponent Life: {board_state.opponent_life}",
        f"Priority: {'YOU' if board_state.has_priority else 'Opponent'}",
        "",
    ]

    # YOUR ZONES
    if board_state.your_hand:
        lines.append(f"YOUR HAND ({len(board_state.your_hand)} cards):")
        for card in board_state.your_hand:
            lines.append(f"  • {card.name}")
        lines.append("")

    if board_state.your_battlefield:
        lines.append(f"YOUR BATTLEFIELD ({len(board_state.your_battlefield)} permanents):")
        for card in board_state.your_battlefield:
            status = []
            if hasattr(card, 'is_tapped') and card.is_tapped:
                status.append("TAPPED")
            if hasattr(card, 'has_summoning_sickness') and card.has_summoning_sickness:
                status.append("SICK")
            status_str = f" [{', '.join(status)}]" if status else ""
            lines.append(f"  • {card.name}{status_str}")
        lines.append("")

    if board_state.your_graveyard:
        lines.append(f"YOUR GRAVEYARD ({len(board_state.your_graveyard)} cards):")
        # Show last 5 cards (most recent)
        for card in board_state.your_graveyard[-5:]:
            lines.append(f"  • {card.name}")
        if len(board_state.your_graveyard) > 5:
            lines.append(f"  ... and {len(board_state.your_graveyard) - 5} more")
        lines.append("")

    if board_state.your_exile:
        lines.append(f"YOUR EXILE ({len(board_state.your_exile)} cards):")
        for card in board_state.your_exile:
            lines.append(f"  • {card.name}")
        lines.append("")

    # OPPONENT ZONES
    if board_state.opponent_battlefield:
        lines.append(f"OPPONENT BATTLEFIELD ({len(board_state.opponent_battlefield)} permanents):")
        for card in board_state.opponent_battlefield:
            lines.append(f"  • {card.name}")
        lines.append("")

    if board_state.opponent_graveyard:
        lines.append(f"OPPONENT GRAVEYARD ({len(board_state.opponent_graveyard)} cards):")
        for card in board_state.opponent_graveyard[-5:]:
            lines.append(f"  • {card.name}")
        if len(board_state.opponent_graveyard) > 5:
            lines.append(f"  ... and {len(board_state.opponent_graveyard) - 5} more")
        lines.append("")

    # STACK
    if board_state.stack:
        lines.append(f"STACK (resolves top-down):")
        for i, spell in enumerate(reversed(board_state.stack), 1):
            lines.append(f"  {i}. {spell.name}")
        lines.append("")

    # LIBRARY INFO
    lines.append(f"Library: You have {board_state.your_library_count} cards remaining")
    lines.append(f"Library: Opponent has {board_state.opponent_library_count} cards remaining")
    lines.append("")

    # GAME HISTORY (THIS TURN)
    if game_history:
        if game_history.cards_played_this_turn:
            lines.append("THIS TURN:")
            lines.append("  Played:")
            for card in game_history.cards_played_this_turn:
                lines.append(f"    • {card.name}")

        if game_history.died_this_turn:
            lines.append("  Died:")
            for card_name in game_history.died_this_turn:
                lines.append(f"    • {card_name}")

        if game_history.attackers_this_turn:
            lines.append("  Attacked:")
            for card in game_history.attackers_this_turn:
                lines.append(f"    • {card.name}")
        lines.append("")

    lines.append("QUESTION: What is the optimal play sequence right now?")

    return "\n".join(lines)
```

---

## Phase 5: Display Board State with All Zones

### Update _display_board_state method

**File**: `advisor.py` - CLIVoiceAdvisor._display_board_state

**Enhanced display**:
```python
def _display_board_state(self, board_state: BoardState):
    """Display a comprehensive visual representation of game state"""
    print("\n" + "="*70)
    print(f"TURN {board_state.current_turn} - {board_state.current_phase}")
    print("="*70)

    # OPPONENT INFO
    print(f"\n🔴 OPPONENT: {board_state.opponent_life} life, {board_state.opponent_hand_count} cards in hand")

    if board_state.opponent_battlefield:
        print(f"  Battlefield ({len(board_state.opponent_battlefield)}):")
        for card in board_state.opponent_battlefield:
            print(f"    • {card.name}")

    if board_state.opponent_graveyard:
        print(f"  Graveyard ({len(board_state.opponent_graveyard)}): ", end="")
        print(", ".join(c.name for c in board_state.opponent_graveyard[-3:]))

    if board_state.opponent_exile:
        print(f"  Exile ({len(board_state.opponent_exile)}): ", end="")
        print(", ".join(c.name for c in board_state.opponent_exile))

    print(f"\n{'─'*70}")

    # STACK
    if board_state.stack:
        print(f"\n⚡ STACK (resolves top → bottom):")
        for i, spell in enumerate(reversed(board_state.stack), 1):
            print(f"  {i}. {spell.name}")

    print(f"\n{'─'*70}")

    # YOUR INFO
    print(f"\n🟢 YOU: {board_state.your_life} life")
    print(f"  Library: {board_state.your_library_count} cards")

    if board_state.your_hand:
        print(f"\n  Hand ({len(board_state.your_hand)}):")
        for card in board_state.your_hand:
            status = "⚠ UNKNOWN" if "Unknown" in card.name else ""
            print(f"    • {card.name} {status}")

    if board_state.your_battlefield:
        print(f"\n  Battlefield ({len(board_state.your_battlefield)}):")
        for card in board_state.your_battlefield:
            status = "⚠ UNKNOWN" if "Unknown" in card.name else ""
            print(f"    • {card.name} {status}")

    if board_state.your_graveyard:
        print(f"\n  Graveyard ({len(board_state.your_graveyard)}):")
        for card in board_state.your_graveyard[-5:]:
            print(f"    • {card.name}")
        if len(board_state.your_graveyard) > 5:
            print(f"    ... and {len(board_state.your_graveyard) - 5} more")

    if board_state.your_exile:
        print(f"\n  Exile ({len(board_state.your_exile)}):")
        for card in board_state.your_exile:
            print(f"    • {card.name}")

    print("\n" + "="*70)
```

---

## Implementation Priority

### Critical (Do First):
1. ✅ Zone transfer parsing (already done)
2. ✅ Arena card database (already done)
3. ✅ Delete objects (already done)
4. ⚠️ **Add graveyard tracking** (Phase 1)
5. ⚠️ **Add exile tracking** (Phase 1)

### High Priority:
6. ⚠️ **Add stack tracking** (Phase 1)
7. ⚠️ **Add game history** (Phase 2)
8. ⚠️ **Enhance AI prompt** (Phase 4)

### Medium Priority:
9. ⚠️ Deck list parsing (Phase 3)
10. ⚠️ Library composition analysis (Phase 3)

### Low Priority (Polish):
11. MTG rules in system prompt (Phase 4A)
12. Visual improvements (Phase 5)
13. Performance optimizations

---

## Expected Impact

### Before Improvements:
```
AI Advice: "Play your creature"
(Generic, doesn't consider graveyard recursion or exile interactions)
```

### After Improvements:
```
AI Advice: "Hold Forsaken Miner - opponent's Tragic Trajectory in graveyard means
they can flashback removal. Play land and pass, force them to use it on their turn."

(Context-aware, considers graveyard, uses full game state)
```

---

## Testing Checklist

After implementing each phase:

### Phase 1 Tests:
- [ ] Graveyard shows cards that died
- [ ] Exile shows foretell/impulse cards
- [ ] Stack shows spells being cast
- [ ] Library counts decrease correctly

### Phase 2 Tests:
- [ ] Game history resets each turn
- [ ] "Cards played this turn" accurate
- [ ] "Creatures that died this turn" tracked

### Phase 3 Tests:
- [ ] Deck list loaded at game start
- [ ] Remaining deck composition calculated
- [ ] Draw probability advice appears

### Phase 4 Tests:
- [ ] AI references graveyard in advice
- [ ] AI considers exile zone
- [ ] AI accounts for cards played this turn
- [ ] TTS speaks comprehensive advice

---

## Code Size Estimate

- Phase 1 (Zones): ~100 lines
- Phase 2 (History): ~80 lines
- Phase 3 (Deck): ~120 lines
- Phase 4 (Prompt): ~150 lines
- Phase 5 (Display): ~60 lines

**Total**: ~510 new lines of code

**Result**: Transform from basic advisor to comprehensive MTG AI assistant!

---

END OF DOCUMENT
