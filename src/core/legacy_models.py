"""
Legacy data models used during the transition to the new domain models.
"""
import dataclasses
from typing import Any, Dict, List, Optional

# Constants for counter types
P1P1_COUNTERS_KEY = "P1P1" # Represents +1/+1 counters

@dataclasses.dataclass
class GameObject:
    instance_id: int
    grp_id: int
    zone_id: int
    owner_seat_id: int
    name: str = ""
    color_identity: str = ""
    base_power: Optional[int] = None
    base_toughness: Optional[int] = None
    is_tapped: bool = False
    is_attacking: bool = False
    summoning_sick: bool = False
    counters: Dict[str, int] = dataclasses.field(default_factory=dict)
    attached_to: Optional[int] = None
    visibility: str = "public"
    type_line: str = ""

    @property
    def effective_power(self) -> Optional[int]:
        if self.base_power is None:
            return None
        p1p1_count = self.counters.get(P1P1_COUNTERS_KEY, 0)
        return self.base_power + p1p1_count

    @property
    def effective_toughness(self) -> Optional[int]:
        if self.base_toughness is None:
            return None
        p1p1_count = self.counters.get(P1P1_COUNTERS_KEY, 0)
        return self.base_toughness + p1p1_count

@dataclasses.dataclass
class PlayerState:
    seat_id: int
    life_total: int = 20
    hand_count: int = 0
    has_priority: bool = False
    mana_pool: Dict[str, int] = dataclasses.field(default_factory=dict)
    energy: int = 0

@dataclasses.dataclass
class GameHistory:
    turn_number: int = 0
    cards_played_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    attackers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    blockers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    damage_dealt: Dict[int, int] = dataclasses.field(default_factory=dict)
    died_this_turn: List[str] = dataclasses.field(default_factory=list)
    lands_played_this_turn: int = 0
    current_attackers: List[int] = dataclasses.field(default_factory=list)
    current_blockers: Dict[int, int] = dataclasses.field(default_factory=dict)
    combat_damage_assignments: Dict[int, int] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class BoardState:
    your_seat_id: int
    opponent_seat_id: int
    your_life: int = 20
    opponent_life: int = 20
    your_mana_pool: Dict[str, int] = dataclasses.field(default_factory=dict)
    your_energy: int = 0
    opponent_energy: int = 0
    your_hand_count: int = 0
    your_hand: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_hand_count: int = 0
    your_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    your_graveyard: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_graveyard: List[GameObject] = dataclasses.field(default_factory=list)
    your_exile: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_exile: List[GameObject] = dataclasses.field(default_factory=list)
    your_library_count: int = 0
    opponent_library_count: int = 0
    stack: List[GameObject] = dataclasses.field(default_factory=list)
    current_turn: int = 0
    current_phase: str = ""
    is_your_turn: bool = False
    has_priority: bool = False
    history: Optional[GameHistory] = None
    your_decklist: Dict[str, int] = dataclasses.field(default_factory=dict)
    your_deck_remaining: int = 0
    library_top_known: List[str] = dataclasses.field(default_factory=list)
    scry_info: Optional[str] = None
    in_mulligan_phase: bool = False
    game_stage: str = ""
    pending_decision: Optional[str] = None
    decision_context: Dict[str, Any] = dataclasses.field(default_factory=dict)
