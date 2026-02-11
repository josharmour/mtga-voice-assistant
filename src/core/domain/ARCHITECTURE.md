# Domain Model Architecture

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  formatters.py (BoardStateFormatter)                      │  │
│  │  - format_for_display()                                   │  │
│  │  - format_compact()                                       │  │
│  │  - format_hand(), format_battlefield()                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               ↑                                  │
└───────────────────────────────┼──────────────────────────────────┘
                                │ Uses
                                │
┌───────────────────────────────┼──────────────────────────────────┐
│                      Domain Layer                                │
│  ┌────────────────────────────┴──────────────────────────────┐  │
│  │  domain/game_state.py                                      │  │
│  │                                                            │  │
│  │  GameState (Aggregate Root)                               │  │
│  │  ├── turn_number: int                                     │  │
│  │  ├── phase: Phase                                         │  │
│  │  ├── local_player: PlayerGameState                        │  │
│  │  ├── opponent: PlayerGameState                            │  │
│  │  ├── local_battlefield: List[Permanent]                   │  │
│  │  ├── opponent_battlefield: List[Permanent]                │  │
│  │  ├── combat: CombatState                                  │  │
│  │  └── history: TurnHistory                                 │  │
│  │                                                            │  │
│  │  Rich domain methods:                                     │  │
│  │  • is_combat_phase() -> bool                              │  │
│  │  • get_creatures_that_can_attack() -> List[Permanent]     │  │
│  │  • get_permanent_by_id(id) -> Optional[Permanent]         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Supporting Domain Models                                  │  │
│  │                                                            │  │
│  │  Permanent (Entity)                                        │  │
│  │  ├── identity: CardIdentity                               │  │
│  │  ├── power/toughness: Optional[int]                       │  │
│  │  ├── tapped, attacking, blocking: bool/Optional[int]      │  │
│  │  └── can_attack(), can_block() -> bool                    │  │
│  │                                                            │  │
│  │  CardIdentity (Value Object - Frozen)                     │  │
│  │  ├── grp_id: int                                           │  │
│  │  ├── instance_id: int                                      │  │
│  │  └── name: str                                             │  │
│  │                                                            │  │
│  │  PlayerGameState (Aggregate)                              │  │
│  │  ├── life_total: int                                       │  │
│  │  ├── hand: ZoneCollection                                 │  │
│  │  ├── graveyard: ZoneCollection                            │  │
│  │  └── hand_size, total_mana_available (properties)         │  │
│  │                                                            │  │
│  │  Phase (Enum)                                             │  │
│  │  ├── MAIN_1, MAIN_2, COMBAT_ATTACKERS, etc.              │  │
│  │  └── is_combat_phase, is_main_phase (properties)          │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                                ↑
                                │ Converts
                                │
┌───────────────────────────────┼──────────────────────────────────┐
│                      Adapter Layer (Transition)                  │
│  ┌────────────────────────────┴──────────────────────────────┐  │
│  │  domain/adapters.py (BoardStateAdapter)                   │  │
│  │                                                            │  │
│  │  to_game_state(board_state) -> GameState                  │  │
│  │  from_game_state(game_state) -> BoardState                │  │
│  │                                                            │  │
│  │  Bidirectional conversion for migration period            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               ↑                                  │
└───────────────────────────────┼──────────────────────────────────┘
                                │
┌───────────────────────────────┼──────────────────────────────────┐
│                      Infrastructure Layer                        │
│  ┌────────────────────────────┴──────────────────────────────┐  │
│  │  mtga.py (Legacy - To be migrated)                        │  │
│  │                                                            │  │
│  │  BoardState (Current)                                     │  │
│  │  ├── your_life, opponent_life                             │  │
│  │  ├── your_battlefield: List[GameObject]                   │  │
│  │  ├── current_phase: str  (string-based, not type-safe)    │  │
│  │  └── ... 30+ mixed domain/presentation fields             │  │
│  │                                                            │  │
│  │  GameObject (Legacy)                                      │  │
│  │  ├── instance_id, grp_id, zone_id                         │  │
│  │  ├── power, toughness, is_tapped, is_attacking            │  │
│  │  └── No domain methods (anemic model)                     │  │
│  │                                                            │  │
│  │  GameStateManager                                         │  │
│  │  └── get_current_board_state() -> BoardState              │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Domain Model Relationships

```
GameState (Aggregate Root)
    │
    ├── Phase (Enum)
    │   ├── UNTAP, UPKEEP, DRAW
    │   ├── MAIN_1, MAIN_2
    │   ├── COMBAT_BEGIN, COMBAT_ATTACKERS, COMBAT_BLOCKERS, ...
    │   └── END, CLEANUP
    │
    ├── PlayerGameState (Aggregate) × 2 (local + opponent)
    │   ├── life_total: int
    │   ├── energy: int
    │   ├── mana_pool: Dict[str, int]
    │   ├── hand: ZoneCollection
    │   │   └── cards: List[CardIdentity]
    │   ├── graveyard: ZoneCollection
    │   │   └── cards: List[CardIdentity]
    │   ├── exile: ZoneCollection
    │   │   └── cards: List[CardIdentity]
    │   └── library_count: int
    │
    ├── local_battlefield: List[Permanent]
    │   └── Permanent (Entity)
    │       ├── identity: CardIdentity (frozen value object)
    │       │   ├── grp_id: int
    │       │   ├── instance_id: int
    │       │   └── name: str
    │       ├── controller_id: int
    │       ├── owner_id: int
    │       ├── power/toughness: Optional[int]
    │       ├── tapped, attacking, blocking: bool/int
    │       ├── counters: Dict[str, int]
    │       └── can_attack(), can_block() (domain methods)
    │
    ├── opponent_battlefield: List[Permanent]
    │
    ├── stack: ZoneCollection
    │   └── cards: List[CardIdentity]
    │
    ├── combat: CombatState
    │   ├── attackers: Set[int]
    │   ├── blockers: Dict[int, List[int]]
    │   └── damage_assignments: Dict[int, int]
    │
    └── history: TurnHistory
        ├── cards_played: List[CardIdentity]
        ├── lands_played_count: int
        ├── creatures_that_attacked: List[int]
        └── creatures_died: List[str]
```

## Data Flow

### Current (Legacy)

```
MTGA Arena Log File
        ↓
LogFollower (follows file)
        ↓
GameStateManager.parse_log_line()
        ↓
MatchScanner (parses GRE events)
        ↓
BoardState (mixed domain + presentation)
        ↓
        ├→ BoardStateFormatter.format_for_display()
        │          ↓
        │      UI Display
        │
        └→ AIAdvisor.analyze()
                   ↓
              LLM Analysis
```

### Future (After Migration)

```
MTGA Arena Log File
        ↓
LogFollower (follows file)
        ↓
GameStateManager.parse_log_line()
        ↓
MatchScanner (parses GRE events)
        ↓
GameState (pure domain model)
        ↓
        ├→ BoardStateFormatter.format_for_display()
        │          ↓
        │      UI Display
        │
        └→ AIAdvisor.analyze()
                   ↓
              LLM Analysis
```

### Transition Period (Both Coexist)

```
MTGA Arena Log File
        ↓
LogFollower (follows file)
        ↓
GameStateManager.parse_log_line()
        ↓
MatchScanner (parses GRE events)
        ↓
BoardState (legacy)
        ↓
BoardStateAdapter.to_game_state()
        ↓
GameState (domain model)
        ↓
        ├→ BoardStateFormatter.format_for_display()
        │          ↓
        │      UI Display
        │
        └→ AIAdvisor.analyze()
                   ↓
              LLM Analysis
```

## Domain Model Design Patterns

### Value Objects (Immutable)
```python
@dataclass(frozen=True)
class CardIdentity:
    grp_id: int
    instance_id: int
    name: str
```

**Characteristics:**
- Immutable (frozen)
- Equality by value
- No identity
- Thread-safe

### Entities
```python
@dataclass
class Permanent:
    identity: CardIdentity  # Value object
    controller_id: int
    tapped: bool
    # ... mutable state
```

**Characteristics:**
- Has identity (instance_id)
- Mutable state
- Equality by identity
- Lifecycle

### Aggregates
```python
@dataclass
class PlayerGameState:
    player_id: int
    hand: ZoneCollection
    graveyard: ZoneCollection
    # ... related entities
```

**Characteristics:**
- Groups related entities
- Consistency boundary
- Access through root

### Aggregate Root
```python
@dataclass
class GameState:
    local_player: PlayerGameState
    opponent: PlayerGameState
    local_battlefield: List[Permanent]
    # ... complete game state
```

**Characteristics:**
- Top-level aggregate
- Entry point for all queries
- Maintains invariants
- Coordinates all domain objects

## Comparison: Before vs After

### Phase Handling

**Before (String-based):**
```python
# BoardState
current_phase: str = "Phase_Combat_Attackers"

# Usage - error-prone
if board_state.current_phase == "Phase_Combat_Attackers":  # Typo risk
    # Combat logic

if board_state.current_phase in [
    "Phase_Combat_Begin",
    "Phase_Combat_Attackers",
    "Phase_Combat_Blockers",
    "Phase_Combat_Damage",
    "Phase_Combat_End"
]:  # Verbose, error-prone
    # Combat phase logic
```

**After (Enum-based):**
```python
# GameState
phase: Phase = Phase.COMBAT_ATTACKERS

# Usage - type-safe, clear
if game_state.phase == Phase.COMBAT_ATTACKERS:  # IDE autocomplete
    # Combat logic

if game_state.is_combat_phase:  # Rich domain method
    # Combat phase logic
```

### Creature Attack Eligibility

**Before (Scattered Logic):**
```python
# Logic scattered in application code
creature = board_state.your_battlefield[0]

if (creature.power is not None and
    not creature.is_tapped and
    not creature.summoning_sick and
    board_state.is_your_turn):
    # Can attack
```

**After (Centralized Domain Logic):**
```python
# Business rule in domain model
creature = game_state.local_battlefield[0]

if creature.can_attack and game_state.is_local_player_turn:
    # Can attack
```

### Hand Size Calculation

**Before (Manual Counting):**
```python
# BoardState
your_hand: List[GameObject] = []
your_hand_count: int = 0  # Redundant, can get out of sync

# Usage
hand_size = len(board_state.your_hand)
# Or
hand_size = board_state.your_hand_count  # Which one is correct?
```

**After (Computed Property):**
```python
# PlayerGameState
hand: ZoneCollection = ZoneCollection()

@property
def hand_size(self) -> int:
    return len(self.hand)

# Usage
hand_size = game_state.local_player.hand_size  # Always correct
```

## Benefits Summary

### 1. Type Safety
- Enums instead of strings
- Explicit types everywhere
- IDE autocomplete
- Compile-time error detection

### 2. Business Logic Centralization
- Domain methods (can_attack, is_combat_phase)
- No duplication across codebase
- Single source of truth

### 3. Testability
- Pure domain models
- No external dependencies
- Fast unit tests
- Easy to mock

### 4. Clarity
- Self-documenting code
- Rich domain semantics
- Ubiquitous language

### 5. Maintainability
- Separation of concerns
- Clear boundaries
- Easy to evolve

## Testing Architecture

```
Domain Models (game_state.py)
        ↓
        ↓ No dependencies
        ↓
Unit Tests (test_domain_models.py)
        │
        ├─ TestPhase
        ├─ TestCardIdentity
        ├─ TestPermanent
        ├─ TestZoneCollection
        ├─ TestPlayerGameState
        ├─ TestCombatState
        └─ TestGameState

Adapter (adapters.py)
        ↓
        ↓ Depends on domain + mtga
        ↓
Integration Tests (future)
        │
        ├─ TestBoardStateToGameState
        └─ TestGameStateToBoardState
```

## Migration Phases Diagram

```
Phase 1: Foundation
┌────────────────────┐
│ Create domain      │
│ models + adapters  │  ✅ COMPLETE
│ + tests + docs     │
└────────────────────┘

Phase 2: Adapter Testing
┌────────────────────┐
│ Integration tests  │  ⏳ NEXT
│ for adapters       │
└────────────────────┘

Phase 3: Formatters
┌────────────────────┐
│ Update formatters  │
│ to use GameState   │
└────────────────────┘

Phase 4: AI Module
┌────────────────────┐
│ Update AI to use   │
│ GameState          │
└────────────────────┘

Phase 5: Direct Construction
┌────────────────────┐
│ Build GameState    │
│ directly in        │
│ GameStateManager   │
└────────────────────┘

Phase 6: Cleanup
┌────────────────────┐
│ Remove BoardState  │
│ Remove adapters    │  Migration complete!
└────────────────────┘
```

## Conclusion

The domain model architecture provides:

1. **Clean Separation**: Domain logic isolated from presentation
2. **Type Safety**: Enums, explicit types, frozen value objects
3. **Rich Semantics**: Business logic in domain methods
4. **Testability**: Pure models with no dependencies
5. **Maintainability**: Clear boundaries, single responsibility

The architecture is ready for gradual migration while maintaining full backwards compatibility.
