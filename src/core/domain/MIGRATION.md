# Domain Model Migration Guide

## Overview

This document outlines the migration path from the existing `BoardState` dataclass in `mtga.py` to the new clean domain models in `src/core/domain/`.

## Why Migrate?

The current `BoardState` class mixes domain logic with presentation concerns:
- Contains data that's used for both business logic AND display
- Formatting logic is scattered across multiple files
- Hard to test domain logic in isolation
- Violates Single Responsibility Principle

## New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Domain Layer                              │
│  src/core/domain/game_state.py                              │
│  - Pure business logic                                       │
│  - No presentation concerns                                  │
│  - Rich domain semantics                                     │
│  - Type-safe, immutable where appropriate                    │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                            │ Uses
                            │
┌─────────────────────────────────────────────────────────────┐
│                 Presentation Layer                           │
│  src/core/formatters.py                                     │
│  - Converts domain models to display strings                │
│  - UI/TUI/CLI specific formatting                           │
│  - No business logic                                         │
└─────────────────────────────────────────────────────────────┘
```

## Migration Phases

### Phase 1: Create Domain Models ✅ COMPLETE

- [x] Create `src/core/domain/` package
- [x] Define pure domain models (`GameState`, `Permanent`, etc.)
- [x] Add rich domain semantics (properties, methods)
- [x] Document architecture and migration path

### Phase 2: Create Adapter Layer (NEXT)

Create adapters to convert between old and new models:

```python
# src/core/domain/adapters.py

from typing import Optional
from ..mtga import BoardState, GameObject
from .game_state import GameState, Permanent, CardIdentity, PlayerGameState

class BoardStateAdapter:
    """Converts BoardState to GameState."""

    @staticmethod
    def to_game_state(board_state: BoardState) -> GameState:
        """
        Convert legacy BoardState to domain GameState.

        Args:
            board_state: Legacy BoardState from mtga.py

        Returns:
            Domain model GameState
        """
        game_state = GameState(
            turn_number=board_state.current_turn,
            phase=Phase.from_arena_string(board_state.current_phase),
            in_mulligan=board_state.in_mulligan_phase,
            game_stage=board_state.game_stage,
        )

        # Convert local player
        game_state.local_player = PlayerGameState(
            player_id=board_state.your_seat_id,
            life_total=board_state.your_life,
            energy=board_state.your_energy,
            has_priority=board_state.has_priority,
            is_active_player=board_state.is_your_turn,
            mana_pool=board_state.your_mana_pool.copy(),
            library_count=board_state.your_library_count,
            decklist=board_state.your_decklist.copy(),
            library_top_known=board_state.library_top_known.copy(),
        )

        # Convert hand
        for card in board_state.your_hand:
            card_id = CardIdentity(
                grp_id=card.grp_id,
                instance_id=card.instance_id,
                name=card.name,
            )
            game_state.local_player.hand.add_card(card_id)

        # Convert battlefield
        for game_obj in board_state.your_battlefield:
            perm = BoardStateAdapter._to_permanent(game_obj)
            game_state.local_battlefield.append(perm)

        # ... similar for opponent, graveyards, etc.

        return game_state

    @staticmethod
    def _to_permanent(game_obj: GameObject) -> Permanent:
        """Convert GameObject to Permanent."""
        identity = CardIdentity(
            grp_id=game_obj.grp_id,
            instance_id=game_obj.instance_id,
            name=game_obj.name,
        )

        return Permanent(
            identity=identity,
            controller_id=game_obj.owner_seat_id,
            owner_id=game_obj.owner_seat_id,
            tapped=game_obj.is_tapped,
            attacking=game_obj.is_attacking,
            summoning_sick=game_obj.summoning_sick,
            power=game_obj.power,
            toughness=game_obj.toughness,
            counters=game_obj.counters.copy(),
            attached_to=game_obj.attached_to,
            type_line=game_obj.type_line,
            color_identity=game_obj.color_identity,
        )

    @staticmethod
    def from_game_state(game_state: GameState) -> BoardState:
        """
        Convert domain GameState back to legacy BoardState.

        This is needed during the transition period to support
        existing code that expects BoardState.

        Args:
            game_state: Domain model GameState

        Returns:
            Legacy BoardState
        """
        # Implementation...
        pass
```

### Phase 3: Update Consumers

Update code that uses BoardState to work with both models:

```python
# Example: Update AI module
class AIAdvisor:
    def analyze_game_state(self, board_state: BoardState) -> str:
        # Convert to domain model
        game_state = BoardStateAdapter.to_game_state(board_state)

        # Use domain model for logic
        if game_state.is_combat_phase:
            return self._analyze_combat(game_state)

        # ...
```

### Phase 4: Update Formatters

Formatters should work with domain models:

```python
# src/core/formatters.py

class BoardStateFormatter:
    """Formats GameState for display."""

    def format_for_display(self, game_state: GameState) -> List[str]:
        """
        Format complete game state as lines for display.

        Args:
            game_state: Domain model GameState

        Returns:
            List of formatted strings ready for display
        """
        lines = []

        # Header with turn and phase
        lines.append("=" * 70)
        lines.append(f"TURN {game_state.turn_number} - {game_state.phase.name}")
        lines.append("=" * 70)

        # Life totals
        lines.append(
            f"Your Life: {game_state.local_player.life_total}  |  "
            f"Opponent Life: {game_state.opponent.life_total}"
        )

        # ... rest of formatting

        return lines
```

### Phase 5: Direct Construction

Update `GameStateManager` to build domain models directly:

```python
class GameStateManager:
    def get_current_board_state(self) -> Optional[BoardState]:
        """Legacy method - deprecated."""
        game_state = self.get_current_game_state()
        if game_state:
            return BoardStateAdapter.from_game_state(game_state)
        return None

    def get_current_game_state(self) -> Optional[GameState]:
        """New method - returns domain model."""
        # Build GameState directly from parsed data
        game_state = GameState(...)

        # No conversion needed
        return game_state
```

### Phase 6: Deprecation and Cleanup

Once all consumers are updated:

1. Mark `BoardState` as deprecated
2. Add deprecation warnings
3. Update all remaining usages
4. Eventually remove `BoardState` entirely

## Benefits After Migration

### 1. Separation of Concerns
- Domain logic isolated in domain models
- Presentation logic isolated in formatters
- Each layer has single responsibility

### 2. Testability
```python
# Easy to test domain logic without UI
def test_can_attack():
    perm = Permanent(
        identity=CardIdentity(grp_id=1, instance_id=1, name="Test"),
        controller_id=1,
        owner_id=1,
        power=2,
        toughness=2,
    )
    assert perm.can_attack == True

    perm.tapped = True
    assert perm.can_attack == False
```

### 3. Rich Domain Semantics
```python
# Before (BoardState)
if board_state.current_phase in ["Phase_Combat_Begin", "Phase_Combat_Attackers", ...]:
    # Combat logic

# After (GameState)
if game_state.is_combat_phase:
    # Combat logic - much clearer!
```

### 4. Type Safety
```python
# Before - string-based phases
phase: str = "Phase_Main1"  # Easy to typo

# After - enum-based phases
phase: Phase = Phase.MAIN_1  # Type-safe, autocomplete works
```

### 5. Immutability Where Appropriate
```python
# CardIdentity is frozen - can't be accidentally modified
card_id = CardIdentity(grp_id=1, instance_id=1, name="Lightning Bolt")
# card_id.grp_id = 2  # Error! Frozen dataclass
```

## Example Usage

### Before (Current)
```python
# In app.py
board_state = self.game_state_manager.get_current_board_state()

# Formatting mixed with data access
print(f"Turn {board_state.current_turn}")
for card in board_state.your_battlefield:
    print(f"  {card.name} ({card.power}/{card.toughness})")
```

### After (Domain Models)
```python
# In app.py
game_state = self.game_state_manager.get_current_game_state()

# Domain logic is clean
if game_state.is_combat_phase:
    attackers = game_state.get_creatures_that_can_attack()
    for creature in attackers:
        print(f"Can attack: {creature.identity.name}")

# Formatting is separate
formatter = BoardStateFormatter()
display_lines = formatter.format_for_display(game_state)
for line in display_lines:
    print(line)
```

## Backwards Compatibility

During migration, maintain backwards compatibility:

```python
class GameStateManager:
    """Supports both old and new APIs."""

    def get_current_board_state(self) -> Optional[BoardState]:
        """Legacy API - still works but deprecated."""
        warnings.warn(
            "get_current_board_state is deprecated, use get_current_game_state",
            DeprecationWarning
        )
        game_state = self.get_current_game_state()
        if game_state:
            return BoardStateAdapter.from_game_state(game_state)
        return None

    def get_current_game_state(self) -> Optional[GameState]:
        """New API - returns domain model."""
        # Implementation...
        pass
```

## Testing Strategy

### Unit Tests for Domain Models
```python
# tests/test_domain_models.py

def test_permanent_can_attack():
    """Test creature attack eligibility logic."""
    creature = Permanent(
        identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
        controller_id=1,
        owner_id=1,
        power=2,
        toughness=2,
    )

    assert creature.can_attack == True

    creature.tapped = True
    assert creature.can_attack == False

    creature.tapped = False
    creature.summoning_sick = True
    assert creature.can_attack == False

def test_phase_detection():
    """Test phase categorization."""
    assert Phase.COMBAT_ATTACKERS.is_combat_phase == True
    assert Phase.MAIN_1.is_main_phase == True
    assert Phase.UPKEEP.is_combat_phase == False

def test_game_state_priority():
    """Test priority detection."""
    game_state = GameState()
    game_state.local_player.has_priority = True

    assert game_state.local_has_priority == True
    assert game_state.priority_player == game_state.local_player
```

### Integration Tests
```python
# tests/test_adapters.py

def test_board_state_to_game_state_conversion():
    """Test adapter converts BoardState to GameState correctly."""
    # Create legacy BoardState
    board_state = BoardState(
        your_seat_id=1,
        opponent_seat_id=2,
        current_turn=5,
        your_life=18,
        opponent_life=20,
    )

    # Convert to domain model
    game_state = BoardStateAdapter.to_game_state(board_state)

    # Verify conversion
    assert game_state.turn_number == 5
    assert game_state.local_player.life_total == 18
    assert game_state.opponent.life_total == 20
```

## Timeline

- **Phase 1** (Complete): Domain models created
- **Phase 2** (1-2 days): Create adapter layer
- **Phase 3** (3-5 days): Update formatters and AI to use domain models
- **Phase 4** (2-3 days): Update UI components
- **Phase 5** (3-5 days): Refactor GameStateManager to build domain models
- **Phase 6** (1-2 days): Deprecate and clean up BoardState

**Total estimated time**: 2-3 weeks for complete migration

## Rollback Plan

If issues arise during migration:

1. Keep adapter layer functional
2. Revert consumers to use legacy BoardState
3. Domain models remain in codebase for future use
4. No breaking changes since both APIs coexist

## Questions and Answers

### Q: Why not just refactor BoardState in place?
**A**: BoardState is deeply entangled with mtga.py parsing logic. Creating clean domain models allows incremental migration without breaking existing functionality.

### Q: Will this impact performance?
**A**: Minimal impact. Adapter conversions are O(n) where n is number of cards, which is typically small (<100). Domain models use similar data structures to current BoardState.

### Q: Can we use both models simultaneously?
**A**: Yes! The adapter layer allows seamless conversion. During migration, some code can use BoardState while other code uses GameState.

### Q: What about backwards compatibility?
**A**: Legacy `get_current_board_state()` remains functional and uses adapter to convert GameState -> BoardState internally.

## References

- Martin Fowler - Domain-Driven Design
- Clean Architecture - Robert C. Martin
- Existing `formatters.py` for presentation layer patterns
- Existing `mtga.py` for current data structures

## Conclusion

This migration establishes a clean separation between domain logic and presentation logic, making the codebase:

- More maintainable
- Easier to test
- More type-safe
- Better organized
- Following SOLID principles

The migration can happen incrementally without breaking existing functionality, with both old and new models coexisting during the transition period.
