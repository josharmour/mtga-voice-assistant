"""
Pure domain models for MTG game state.

This module contains domain models that represent the true essence of a Magic: The Gathering
game state without any presentation or UI concerns. These models are:

- Pure data structures (dataclasses)
- Independent of formatting/display logic
- Rich with domain semantics
- Immutable where appropriate
- Type-safe with explicit type hints

Migration Path:
    The existing BoardState in mtga.py should gradually be migrated to use these
    domain models. The migration can happen incrementally:

    Phase 1: Create domain models (this file)
    Phase 2: Create adapters to convert BoardState -> GameState
    Phase 3: Update consumers to use domain models
    Phase 4: Deprecate and remove old BoardState
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum, auto


class Phase(Enum):
    """Game phase enumeration following MTG turn structure."""

    UNKNOWN = auto()

    # Beginning phase
    UNTAP = auto()
    UPKEEP = auto()
    DRAW = auto()

    # Main phase
    MAIN_1 = auto()

    # Combat phase
    COMBAT_BEGIN = auto()
    COMBAT_ATTACKERS = auto()
    COMBAT_BLOCKERS = auto()
    COMBAT_DAMAGE = auto()
    COMBAT_END = auto()

    # Second main phase
    MAIN_2 = auto()

    # Ending phase
    END = auto()
    CLEANUP = auto()

    @classmethod
    def from_arena_string(cls, phase_str: str) -> "Phase":
        """
        Convert Arena phase string to Phase enum.

        Args:
            phase_str: Phase string from Arena logs (e.g., "Phase_Main1")

        Returns:
            Corresponding Phase enum value
        """
        phase_map = {
            "Phase_Beginning_Untap": cls.UNTAP,
            "Phase_Beginning_Upkeep": cls.UPKEEP,
            "Phase_Beginning_Draw": cls.DRAW,
            "Phase_Main1": cls.MAIN_1,
            "Phase_Combat_Begin": cls.COMBAT_BEGIN,
            "Phase_Combat_Attackers": cls.COMBAT_ATTACKERS,
            "Phase_Combat_Blockers": cls.COMBAT_BLOCKERS,
            "Phase_Combat_Damage": cls.COMBAT_DAMAGE,
            "Phase_Combat_End": cls.COMBAT_END,
            "Phase_Main2": cls.MAIN_2,
            "Phase_Ending_End": cls.END,
            "Phase_Ending_Cleanup": cls.CLEANUP,
        }
        return phase_map.get(phase_str, cls.UNKNOWN)

    @property
    def is_combat_phase(self) -> bool:
        """Check if this is a combat phase."""
        return self in {
            Phase.COMBAT_BEGIN,
            Phase.COMBAT_ATTACKERS,
            Phase.COMBAT_BLOCKERS,
            Phase.COMBAT_DAMAGE,
            Phase.COMBAT_END,
        }

    @property
    def is_main_phase(self) -> bool:
        """Check if this is a main phase."""
        return self in {Phase.MAIN_1, Phase.MAIN_2}


@dataclass(frozen=True)
class CardIdentity:
    """
    Immutable card identification - represents the essence of a card instance.

    This is pure domain data with no presentation concerns.
    """

    grp_id: int  # Arena's card type identifier (shared across instances)
    instance_id: int  # Unique identifier for this specific card instance
    name: str = ""  # Human-readable card name

    def __post_init__(self):
        """Validate card identity invariants."""
        if self.instance_id < 0:
            raise ValueError(f"Invalid instance_id: {self.instance_id}")


@dataclass
class Permanent:
    """
    Domain model for a permanent on the battlefield.

    This represents the current state of a card that exists on the battlefield.
    No presentation logic - pure domain state.
    """

    identity: CardIdentity
    controller_id: int  # Player who controls this permanent
    owner_id: int  # Player who owns this permanent

    # Combat and tap state
    tapped: bool = False
    attacking: bool = False
    blocking: Optional[int] = None  # Instance ID of attacker being blocked (if blocking)
    summoning_sick: bool = False

    # Creature stats (None if not a creature)
    power: Optional[int] = None
    toughness: Optional[int] = None

    # Counters and attachments
    counters: Dict[str, int] = field(default_factory=dict)  # counter_type -> count
    attached_to: Optional[int] = None  # Instance ID of permanent this is attached to

    # Metadata
    type_line: str = ""  # e.g., "Creature - Human Warrior"
    color_identity: str = ""  # e.g., "W", "UB", "WUBRG"

    @property
    def is_creature(self) -> bool:
        """Check if this permanent is a creature."""
        return self.power is not None and self.toughness is not None

    @property
    def is_tapped(self) -> bool:
        """Check if this permanent is tapped."""
        return self.tapped

    @property
    def can_attack(self) -> bool:
        """Check if this permanent can attack (heuristic)."""
        return self.is_creature and not self.tapped and not self.summoning_sick

    @property
    def can_block(self) -> bool:
        """Check if this permanent can block (heuristic)."""
        return self.is_creature and not self.tapped


@dataclass
class ZoneCollection:
    """
    Collection of cards in a specific game zone.

    This is a domain concept representing a logical grouping of cards.
    No presentation logic - just the cards and their order.
    """

    cards: List[CardIdentity] = field(default_factory=list)

    # Metadata
    zone_name: str = ""  # e.g., "Hand", "Library", "Graveyard"
    owner_id: Optional[int] = None  # Which player owns this zone
    is_ordered: bool = False  # Whether card order matters (library, graveyard)
    is_hidden: bool = False  # Whether contents are hidden from opponent

    def __len__(self) -> int:
        """Get count of cards in zone."""
        return len(self.cards)

    def is_empty(self) -> bool:
        """Check if zone is empty."""
        return len(self.cards) == 0

    def top_cards(self, n: int = 1) -> List[CardIdentity]:
        """
        Get top N cards from zone.

        Args:
            n: Number of cards to get from top

        Returns:
            List of up to N cards from the top of the zone
        """
        return self.cards[:n] if self.is_ordered else []

    def add_card(self, card: CardIdentity, position: Optional[int] = None):
        """
        Add a card to the zone.

        Args:
            card: Card to add
            position: Optional position (0 = top). If None, adds to end.
        """
        if position is None:
            self.cards.append(card)
        else:
            self.cards.insert(position, card)

    def remove_card(self, instance_id: int) -> Optional[CardIdentity]:
        """
        Remove a card by instance ID.

        Args:
            instance_id: Instance ID of card to remove

        Returns:
            Removed card, or None if not found
        """
        for i, card in enumerate(self.cards):
            if card.instance_id == instance_id:
                return self.cards.pop(i)
        return None


@dataclass
class PlayerGameState:
    """
    Domain model for a player's state in the game.

    This represents everything about one player's current game state,
    independent of how it's displayed.
    """

    player_id: int  # Seat ID

    # Life and resources
    life_total: int = 20
    energy: int = 0
    poison_counters: int = 0

    # Mana
    mana_pool: Dict[str, int] = field(default_factory=dict)  # color -> amount

    # Priority and control
    has_priority: bool = False
    is_active_player: bool = False  # Whose turn it is

    # Zones
    library_count: int = 0  # Usually hidden, only count known
    hand: ZoneCollection = field(default_factory=lambda: ZoneCollection(
        zone_name="Hand", is_hidden=True
    ))
    graveyard: ZoneCollection = field(default_factory=lambda: ZoneCollection(
        zone_name="Graveyard", is_ordered=True
    ))
    exile: ZoneCollection = field(default_factory=lambda: ZoneCollection(
        zone_name="Exile"
    ))

    # Deck tracking (if available)
    decklist: Dict[str, int] = field(default_factory=dict)  # card_name -> count
    library_top_known: List[str] = field(default_factory=list)  # Known top cards

    @property
    def hand_size(self) -> int:
        """Get current hand size."""
        return len(self.hand)

    @property
    def graveyard_size(self) -> int:
        """Get current graveyard size."""
        return len(self.graveyard)

    @property
    def exile_size(self) -> int:
        """Get current exile size."""
        return len(self.exile)

    @property
    def total_mana_available(self) -> int:
        """Get total mana in pool across all colors."""
        return sum(self.mana_pool.values())


@dataclass
class CombatState:
    """
    Domain model for combat state.

    This represents the current combat situation, including attackers,
    blockers, and damage assignments.
    """

    # Attackers (instance IDs of attacking creatures)
    attackers: Set[int] = field(default_factory=set)

    # Blocker assignments: attacker_instance_id -> list of blocker_instance_ids
    blockers: Dict[int, List[int]] = field(default_factory=dict)

    # Damage assignments: instance_id -> damage_amount
    damage_assignments: Dict[int, int] = field(default_factory=dict)

    def is_in_combat(self) -> bool:
        """Check if combat is currently happening."""
        return len(self.attackers) > 0

    def get_blockers_for(self, attacker_id: int) -> List[int]:
        """
        Get all blockers assigned to an attacker.

        Args:
            attacker_id: Instance ID of attacking creature

        Returns:
            List of blocker instance IDs
        """
        return self.blockers.get(attacker_id, [])

    def is_blocked(self, attacker_id: int) -> bool:
        """Check if an attacker is blocked."""
        return attacker_id in self.blockers and len(self.blockers[attacker_id]) > 0

    def clear(self):
        """Clear all combat state (called when combat ends)."""
        self.attackers.clear()
        self.blockers.clear()
        self.damage_assignments.clear()


@dataclass
class TurnHistory:
    """
    Domain model for tracking important events during the current turn.

    This provides tactical context for decision-making.
    """

    turn_number: int = 0

    # Cards played this turn
    cards_played: List[CardIdentity] = field(default_factory=list)
    lands_played_count: int = 0

    # Combat history
    creatures_that_attacked: List[int] = field(default_factory=list)  # Instance IDs
    creatures_that_blocked: List[int] = field(default_factory=list)  # Instance IDs

    # Damage and destruction
    damage_dealt: Dict[int, int] = field(default_factory=dict)  # instance_id -> damage
    creatures_died: List[str] = field(default_factory=list)  # Card names

    def reset_for_new_turn(self, turn_num: int):
        """Reset history for a new turn."""
        self.turn_number = turn_num
        self.cards_played.clear()
        self.lands_played_count = 0
        self.creatures_that_attacked.clear()
        self.creatures_that_blocked.clear()
        self.damage_dealt.clear()
        self.creatures_died.clear()


@dataclass
class GameState:
    """
    Pure domain model representing complete MTG game state.

    This is the top-level domain model that represents everything about
    the current game. It is completely independent of presentation concerns.

    Key design decisions:
    - Immutable where possible (using frozen dataclasses for value objects)
    - Rich domain semantics (methods that express game concepts)
    - No formatting or display logic
    - Type-safe with explicit types
    - Self-documenting through clear names

    Migration from BoardState:
        1. Create adapter layer that converts BoardState -> GameState
        2. Update consumers (AI, UI) to work with GameState
        3. Gradually migrate BoardState construction to use GameState
        4. Eventually deprecate BoardState
    """

    # Turn and phase
    turn_number: int = 0
    phase: Phase = Phase.UNKNOWN

    # Players
    local_player: PlayerGameState = field(default_factory=lambda: PlayerGameState(player_id=0))
    opponent: PlayerGameState = field(default_factory=lambda: PlayerGameState(player_id=0))

    # Battlefield (shared zone)
    local_battlefield: List[Permanent] = field(default_factory=list)
    opponent_battlefield: List[Permanent] = field(default_factory=list)

    # Stack (shared zone, ordered)
    stack: ZoneCollection = field(default_factory=lambda: ZoneCollection(
        zone_name="Stack", is_ordered=True
    ))

    # Combat state (only relevant during combat)
    combat: CombatState = field(default_factory=CombatState)

    # Turn history
    history: TurnHistory = field(default_factory=TurnHistory)

    # Game stage
    in_mulligan: bool = False
    game_stage: str = ""  # e.g., "GameStage_Play"

    # Computed properties for common queries

    @property
    def active_player(self) -> PlayerGameState:
        """Get the active player (whose turn it is)."""
        if self.local_player.is_active_player:
            return self.local_player
        return self.opponent

    @property
    def priority_player(self) -> PlayerGameState:
        """Get the player with priority."""
        if self.local_player.has_priority:
            return self.local_player
        return self.opponent

    @property
    def is_local_player_turn(self) -> bool:
        """Check if it's the local player's turn."""
        return self.local_player.is_active_player

    @property
    def local_has_priority(self) -> bool:
        """Check if local player has priority."""
        return self.local_player.has_priority

    @property
    def is_combat_phase(self) -> bool:
        """Check if currently in combat phase."""
        return self.phase.is_combat_phase

    @property
    def is_main_phase(self) -> bool:
        """Check if currently in main phase."""
        return self.phase.is_main_phase

    @property
    def total_permanents(self) -> int:
        """Get total number of permanents on battlefield."""
        return len(self.local_battlefield) + len(self.opponent_battlefield)

    def get_permanent_by_id(self, instance_id: int) -> Optional[Permanent]:
        """
        Find a permanent by instance ID.

        Args:
            instance_id: Instance ID to search for

        Returns:
            Permanent if found, None otherwise
        """
        for perm in self.local_battlefield + self.opponent_battlefield:
            if perm.identity.instance_id == instance_id:
                return perm
        return None

    def get_permanents_owned_by(self, player_id: int) -> List[Permanent]:
        """
        Get all permanents owned by a specific player.

        Args:
            player_id: Player ID to filter by

        Returns:
            List of permanents owned by that player
        """
        if player_id == self.local_player.player_id:
            return self.local_battlefield
        elif player_id == self.opponent.player_id:
            return self.opponent_battlefield
        return []

    def get_creatures_that_can_attack(self) -> List[Permanent]:
        """
        Get all creatures controlled by active player that can attack.

        Returns:
            List of attackable creatures
        """
        if not self.is_local_player_turn:
            return []

        return [p for p in self.local_battlefield if p.can_attack]

    def get_creatures_that_can_block(self) -> List[Permanent]:
        """
        Get all creatures controlled by non-active player that can block.

        Returns:
            List of creatures that can block
        """
        if self.is_local_player_turn:
            return []

        return [p for p in self.local_battlefield if p.can_block]

    def reset_for_new_turn(self, turn_num: int):
        """
        Reset transient state for a new turn.

        Args:
            turn_num: New turn number
        """
        self.turn_number = turn_num
        self.history.reset_for_new_turn(turn_num)
        self.combat.clear()

    def reset_for_new_game(self):
        """Reset all state for a new game."""
        self.turn_number = 0
        self.phase = Phase.UNKNOWN
        self.local_player = PlayerGameState(player_id=self.local_player.player_id)
        self.opponent = PlayerGameState(player_id=self.opponent.player_id)
        self.local_battlefield.clear()
        self.opponent_battlefield.clear()
        self.stack = ZoneCollection(zone_name="Stack", is_ordered=True)
        self.combat.clear()
        self.history = TurnHistory()
        self.in_mulligan = False
        self.game_stage = ""
