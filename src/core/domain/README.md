# Domain Models Package

## Overview

This package contains pure domain models for the MTGA Voice Advisor, implementing a clean separation between business logic and presentation concerns.

## Philosophy

The domain models follow Domain-Driven Design (DDD) principles:

1. **Rich Domain Models**: Models contain business logic, not just data
2. **Ubiquitous Language**: Model names reflect MTG terminology
3. **Separation of Concerns**: No UI, formatting, or infrastructure code
4. **Type Safety**: Explicit types with validation
5. **Immutability**: Value objects are immutable (frozen dataclasses)

## Package Structure

```
src/core/domain/
├── __init__.py           # Package exports
├── game_state.py         # Core domain models
├── adapters.py           # Conversion between legacy and domain models
├── MIGRATION.md          # Migration guide
├── README.md             # This file
└── test_domain_models.py # Unit tests
```

## Core Models

### CardIdentity (Value Object)
Immutable representation of a card instance.

```python
from src.core.domain import CardIdentity

card = CardIdentity(
    grp_id=12345,           # Arena card type ID
    instance_id=67890,      # Unique instance ID
    name="Lightning Bolt"
)

# Immutable - this will raise an error
# card.grp_id = 99999
```

### Permanent (Entity)
Represents a permanent on the battlefield.

```python
from src.core.domain import Permanent, CardIdentity

creature = Permanent(
    identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
    controller_id=1,
    owner_id=1,
    power=2,
    toughness=2,
    tapped=False,
    summoning_sick=False,
)

# Rich domain logic
if creature.can_attack:
    print(f"{creature.identity.name} can attack!")
```

### Phase (Enum)
Type-safe representation of game phases.

```python
from src.core.domain import Phase

phase = Phase.COMBAT_ATTACKERS

# Rich domain logic built-in
if phase.is_combat_phase:
    print("Combat is happening!")

# Convert from Arena strings
phase = Phase.from_arena_string("Phase_Main1")
assert phase == Phase.MAIN_1
```

### PlayerGameState (Aggregate)
Complete state for one player.

```python
from src.core.domain import PlayerGameState

player = PlayerGameState(
    player_id=1,
    life_total=18,
    has_priority=True,
    is_active_player=True,
)

# Domain methods
print(f"Hand size: {player.hand_size}")
print(f"Total mana: {player.total_mana_available}")
```

### GameState (Aggregate Root)
Complete game state.

```python
from src.core.domain import GameState, Phase

game = GameState(
    turn_number=5,
    phase=Phase.COMBAT_ATTACKERS,
)

# Rich domain queries
if game.is_combat_phase and game.is_local_player_turn:
    attackers = game.get_creatures_that_can_attack()
    print(f"{len(attackers)} creatures can attack")
```

## Usage Examples

### Basic Game State Inspection

```python
from src.core.domain import GameState, Phase

# Assume we have a game state (from adapter or built directly)
game_state = GameState(
    turn_number=3,
    phase=Phase.MAIN_1,
)

# Type-safe phase checking
if game_state.is_main_phase:
    print("It's a main phase - you can cast sorceries")

# Clear ownership semantics
if game_state.local_has_priority:
    print("You have priority - you can take an action")
```

### Combat Logic

```python
# Check if we can attack
if game_state.is_local_player_turn and game_state.phase == Phase.COMBAT_ATTACKERS:
    attackers = game_state.get_creatures_that_can_attack()

    for creature in attackers:
        print(f"{creature.identity.name} ({creature.power}/{creature.toughness}) can attack")
```

### Zone Management

```python
# Add card to hand
from src.core.domain import CardIdentity

card = CardIdentity(grp_id=123, instance_id=456, name="Lightning Bolt")
game_state.local_player.hand.add_card(card)

# Check hand size
print(f"Hand size: {game_state.local_player.hand_size}")

# Access graveyard (ordered zone)
top_of_gy = game_state.local_player.graveyard.top_cards(3)
for card in top_of_gy:
    print(f"In graveyard: {card.name}")
```

## Migration from BoardState

The domain models are designed to eventually replace the legacy `BoardState` class. During the transition:

1. **Use Adapters**: Convert between BoardState and GameState
2. **Gradual Migration**: Update one module at a time
3. **Backwards Compatible**: Both models coexist during migration

Example migration:

```python
# Legacy code
board_state = game_state_manager.get_current_board_state()
if board_state.current_phase == "Phase_Combat_Attackers":
    # Combat logic...

# Migrated code
from src.core.domain.adapters import BoardStateAdapter

board_state = game_state_manager.get_current_board_state()
game_state = BoardStateAdapter.to_game_state(board_state)

if game_state.phase == Phase.COMBAT_ATTACKERS:
    # Combat logic - cleaner and type-safe!
```

See [MIGRATION.md](MIGRATION.md) for the complete migration plan.

## Testing

Domain models are easy to test because they have no external dependencies:

```python
from src.core.domain import Permanent, CardIdentity

def test_tapped_creatures_cannot_attack():
    creature = Permanent(
        identity=CardIdentity(grp_id=1, instance_id=1, name="Test"),
        controller_id=1,
        owner_id=1,
        power=2,
        toughness=2,
        tapped=True,  # Tapped!
    )

    assert creature.can_attack == False  # Business rule verified
```

Run tests:

```bash
# Run all domain tests
pytest src/core/domain/test_domain_models.py -v

# Run specific test
pytest src/core/domain/test_domain_models.py::TestPermanent::test_can_attack_untapped -v
```

## Design Decisions

### Why Frozen Dataclasses for Value Objects?

Value objects like `CardIdentity` are immutable (frozen) because:
- A card's identity never changes
- Immutability prevents bugs from accidental modification
- Enables safe sharing across threads
- Makes code easier to reason about

### Why Separate Permanent from CardIdentity?

A `Permanent` is the *battlefield representation* of a card, while `CardIdentity` is the *essence* of the card. This separation allows:
- Same card identity in multiple zones (hand, graveyard, exile)
- Battlefield-specific state (tapped, attacking) separate from identity
- Clear distinction between "what the card is" vs "where it is and what state it's in"

### Why Rich Domain Models?

Instead of anemic data containers, domain models contain business logic:

```python
# Anemic model (bad)
class Creature:
    tapped: bool
    summoning_sick: bool

# Check in application code
if not creature.tapped and not creature.summoning_sick:
    # Can attack

# Rich model (good)
class Permanent:
    tapped: bool
    summoning_sick: bool

    @property
    def can_attack(self) -> bool:
        return self.is_creature and not self.tapped and not self.summoning_sick

# Check using domain method
if permanent.can_attack:
    # Can attack
```

Benefits:
- Business rules are centralized in domain models
- No duplication of logic across codebase
- Self-documenting code
- Easier to test

### Why Enums for Phases?

Type-safe enums provide:
- IDE autocomplete
- Compile-time error detection
- No typos (`Phase.MAIN_1` vs `"Phase_Main1"`)
- Built-in domain logic (`phase.is_combat_phase`)

## Best Practices

### 1. Keep Domain Models Pure

```python
# Good - Pure domain logic
class Permanent:
    def can_attack(self) -> bool:
        return self.is_creature and not self.tapped

# Bad - Mixing presentation
class Permanent:
    def display_string(self) -> str:
        return f"{self.name} ({self.power}/{self.toughness})"
```

### 2. Use Type Hints

```python
# Good - Explicit types
def get_permanent_by_id(self, instance_id: int) -> Optional[Permanent]:
    ...

# Bad - No type information
def get_permanent_by_id(self, instance_id):
    ...
```

### 3. Add Domain Methods

```python
# Good - Domain method
if game_state.is_combat_phase:
    ...

# Bad - Checking internal state
if game_state.phase in [Phase.COMBAT_BEGIN, Phase.COMBAT_ATTACKERS, ...]:
    ...
```

### 4. Prefer Immutability

```python
# Good - Immutable value object
@dataclass(frozen=True)
class CardIdentity:
    grp_id: int
    instance_id: int

# Bad - Mutable value object
@dataclass
class CardIdentity:
    grp_id: int  # Can be changed!
    instance_id: int  # Can be changed!
```

## Future Enhancements

Potential improvements to domain models:

1. **Validation Rules**: Add `__post_init__` validation for invariants
2. **Domain Events**: Emit events when state changes (for decoupled communication)
3. **More Rich Logic**: Add methods like `can_cast_spell()`, `sufficient_mana()`
4. **Card Type Handling**: Parse and expose type line (Creature, Artifact, etc.)
5. **Advanced Combat**: Multi-block, first strike, damage assignment

## References

- [Domain-Driven Design by Eric Evans](https://www.domainlanguage.com/ddd/)
- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Martin Fowler - Anemic Domain Model](https://martinfowler.com/bliki/AnemicDomainModel.html)
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)

## Contributing

When adding new domain models:

1. Keep them pure (no UI, no I/O)
2. Add type hints
3. Include docstrings
4. Write unit tests
5. Add domain methods where appropriate
6. Consider immutability for value objects
7. Update this README with examples

## Questions?

See [MIGRATION.md](MIGRATION.md) for migration questions, or consult the main project documentation in [CLAUDE.md](../../../CLAUDE.md).
