"""
Domain Models for MTGA Voice Advisor.

This package contains pure domain models representing the game state
and related entities. These models are independent of presentation logic
and focus solely on representing the business domain accurately.

Key principles:
- No presentation logic (formatting, display, etc.)
- No UI concerns (colors, layouts, etc.)
- Pure data structures with business meaning
- Immutable where possible
- Rich with domain semantics

The domain layer is separated from:
- Presentation layer (formatters.py)
- Application layer (app.py)
- Infrastructure layer (mtga.py log parsing)
"""

from .game_state import (
    Phase,
    CardIdentity,
    Permanent,
    ZoneCollection,
    PlayerGameState,
    GameState,
    CombatState,
    TurnHistory,
)

__all__ = [
    "Phase",
    "CardIdentity",
    "Permanent",
    "ZoneCollection",
    "PlayerGameState",
    "GameState",
    "CombatState",
    "TurnHistory",
]
