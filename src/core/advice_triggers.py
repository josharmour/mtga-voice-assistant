"""
Advice Trigger System for MTGA Voice Advisor

Manages when tactical advice is triggered based on:
1. Automatic triggers (turn start, mulligan, draft)
2. Optional phase-based triggers (combat, main phases, etc.)
3. Manual triggers (push-to-talk)

Based on MTG timing and priority rules:
- Players receive priority at the beginning of most steps/phases
- After spell/ability resolution, active player gets priority
- Instants can be cast any time you have priority
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of advice triggers"""
    # Automatic (always on by default)
    TURN_START = auto()          # Beginning of your turn
    MULLIGAN = auto()            # Mulligan decision
    DRAFT_PICK = auto()          # Draft pick decision

    # Optional (user configurable)
    MAIN_PHASE_1 = auto()        # First main phase
    DECLARE_ATTACKERS = auto()   # Declare attackers step
    DECLARE_BLOCKERS = auto()    # Declare blockers step (opponent attacks)
    MAIN_PHASE_2 = auto()        # Second main phase
    OPPONENT_END_STEP = auto()   # Opponent's end step
    RESPONSE_WINDOW = auto()     # Opponent cast spell (counter opportunity)

    # Manual
    PUSH_TO_TALK = auto()        # User pressed spacebar
    VOICE_QUERY = auto()         # User asked a question


@dataclass
class TriggerEvent:
    """Represents a triggered advice event"""
    trigger_type: TriggerType
    turn: int
    phase: str
    is_your_turn: bool
    user_query: Optional[str] = None  # For voice queries


class AdviceTriggerManager:
    """
    Manages advice trigger logic based on game state and user preferences.

    Tracks state transitions to detect:
    - Turn changes (for turn start advice)
    - Phase changes (for phase-specific advice)
    - Priority changes (for response windows)
    """

    # Phase name patterns from MTGA (may vary, using substring matching)
    PHASE_PATTERNS = {
        'upkeep': ['Upkeep', 'Phase_Upkeep'],
        'draw': ['Draw', 'Phase_Draw'],
        'main1': ['Main', 'Precombat', 'Phase_Main1', 'FirstMain'],
        'combat_begin': ['BeginCombat', 'Phase_Combat_Begin'],
        'attackers': ['DeclareAttackers', 'Phase_Combat_Declare_Attackers'],
        'blockers': ['DeclareBlockers', 'Phase_Combat_Declare_Blockers'],
        'damage': ['CombatDamage', 'Phase_Combat_Damage'],
        'combat_end': ['EndCombat', 'Phase_Combat_End'],
        'main2': ['Main', 'Postcombat', 'Phase_Main2', 'SecondMain'],
        'end': ['End', 'Phase_End', 'Ending'],
        'cleanup': ['Cleanup', 'Phase_Cleanup'],
    }

    def __init__(self, prefs=None):
        """
        Initialize the trigger manager.

        Args:
            prefs: UserPreferences instance with trigger settings
        """
        self.prefs = prefs

        # State tracking for transition detection
        self._last_turn: int = 0
        self._last_phase: str = ""
        self._last_is_your_turn: bool = False
        self._last_has_priority: bool = False
        self._last_opponent_spell: bool = False

        # Deduplication: track which triggers have fired this turn/phase
        self._fired_triggers: Set[str] = set()

        # Callback for when advice should be triggered
        self._advice_callback: Optional[Callable[[TriggerEvent], None]] = None

    def set_advice_callback(self, callback: Callable[[TriggerEvent], None]):
        """Set the callback to invoke when advice should be triggered."""
        self._advice_callback = callback

    def check_triggers(self, board_state) -> Optional[TriggerEvent]:
        """
        Check if any automatic/optional triggers should fire based on game state.

        Args:
            board_state: Current BoardState from game

        Returns:
            TriggerEvent if advice should be triggered, None otherwise
        """
        if not board_state:
            return None

        turn = board_state.current_turn
        phase = board_state.current_phase or ""
        is_your_turn = board_state.is_your_turn
        has_priority = board_state.has_priority
        in_mulligan = board_state.in_mulligan_phase

        trigger_event = None

        # Check mulligan (highest priority)
        if in_mulligan and self._should_trigger(TriggerType.MULLIGAN):
            trigger_key = f"mulligan_{len(board_state.your_hand)}"
            if trigger_key not in self._fired_triggers:
                self._fired_triggers.add(trigger_key)
                trigger_event = TriggerEvent(
                    trigger_type=TriggerType.MULLIGAN,
                    turn=turn,
                    phase="Mulligan",
                    is_your_turn=True
                )

        # Check turn start (your turn just started)
        elif self._detect_turn_start(turn, is_your_turn, has_priority):
            if self._should_trigger(TriggerType.TURN_START):
                trigger_key = f"turn_start_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    trigger_event = TriggerEvent(
                        trigger_type=TriggerType.TURN_START,
                        turn=turn,
                        phase=phase,
                        is_your_turn=True
                    )

        # Check phase-specific triggers (only if we have priority)
        elif has_priority:
            phase_trigger = self._check_phase_triggers(turn, phase, is_your_turn)
            if phase_trigger:
                trigger_event = phase_trigger

        # Update state tracking
        self._last_turn = turn
        self._last_phase = phase
        self._last_is_your_turn = is_your_turn
        self._last_has_priority = has_priority

        # Clear fired triggers on turn change
        if turn != self._last_turn:
            self._fired_triggers.clear()

        return trigger_event

    def _detect_turn_start(self, turn: int, is_your_turn: bool, has_priority: bool) -> bool:
        """Detect if this is the start of your turn."""
        # Turn start = turn number changed AND it's now your turn AND you have priority
        turn_changed = turn != self._last_turn and turn > 0
        became_your_turn = is_your_turn and not self._last_is_your_turn

        return (turn_changed or became_your_turn) and is_your_turn and has_priority

    def _check_phase_triggers(self, turn: int, phase: str, is_your_turn: bool) -> Optional[TriggerEvent]:
        """Check for phase-specific optional triggers."""
        phase_lower = phase.lower()

        # Main Phase 1 (your turn, pre-combat)
        if self._is_phase(phase, 'main1') and is_your_turn:
            if self._should_trigger(TriggerType.MAIN_PHASE_1):
                trigger_key = f"main1_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    return TriggerEvent(TriggerType.MAIN_PHASE_1, turn, phase, is_your_turn)

        # Declare Attackers (your turn)
        elif self._is_phase(phase, 'attackers') and is_your_turn:
            if self._should_trigger(TriggerType.DECLARE_ATTACKERS):
                trigger_key = f"attackers_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    return TriggerEvent(TriggerType.DECLARE_ATTACKERS, turn, phase, is_your_turn)

        # Declare Blockers (opponent's turn - you're blocking)
        elif self._is_phase(phase, 'blockers') and not is_your_turn:
            if self._should_trigger(TriggerType.DECLARE_BLOCKERS):
                trigger_key = f"blockers_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    return TriggerEvent(TriggerType.DECLARE_BLOCKERS, turn, phase, is_your_turn)

        # Main Phase 2 (your turn, post-combat)
        elif self._is_phase(phase, 'main2') and is_your_turn:
            if self._should_trigger(TriggerType.MAIN_PHASE_2):
                trigger_key = f"main2_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    return TriggerEvent(TriggerType.MAIN_PHASE_2, turn, phase, is_your_turn)

        # Opponent's End Step (instant window before your turn)
        elif self._is_phase(phase, 'end') and not is_your_turn:
            if self._should_trigger(TriggerType.OPPONENT_END_STEP):
                trigger_key = f"opp_end_{turn}"
                if trigger_key not in self._fired_triggers:
                    self._fired_triggers.add(trigger_key)
                    return TriggerEvent(TriggerType.OPPONENT_END_STEP, turn, phase, is_your_turn)

        return None

    def _is_phase(self, current_phase: str, phase_key: str) -> bool:
        """Check if current phase matches a phase pattern."""
        if not current_phase:
            return False

        patterns = self.PHASE_PATTERNS.get(phase_key, [])
        current_lower = current_phase.lower()

        for pattern in patterns:
            if pattern.lower() in current_lower:
                return True
        return False

    def _should_trigger(self, trigger_type: TriggerType) -> bool:
        """Check if a trigger type is enabled in preferences."""
        if not self.prefs:
            # Default behavior: only automatic triggers
            return trigger_type in (
                TriggerType.TURN_START,
                TriggerType.MULLIGAN,
                TriggerType.DRAFT_PICK,
                TriggerType.PUSH_TO_TALK,
                TriggerType.VOICE_QUERY
            )

        # Map trigger types to preference attributes
        pref_map = {
            TriggerType.TURN_START: 'advice_on_turn_start',
            TriggerType.MULLIGAN: 'advice_on_mulligan',
            TriggerType.DRAFT_PICK: 'advice_on_draft',
            TriggerType.MAIN_PHASE_1: 'advice_on_main_phase_1',
            TriggerType.DECLARE_ATTACKERS: 'advice_on_declare_attackers',
            TriggerType.DECLARE_BLOCKERS: 'advice_on_declare_blockers',
            TriggerType.MAIN_PHASE_2: 'advice_on_main_phase_2',
            TriggerType.OPPONENT_END_STEP: 'advice_on_opponent_end_step',
            TriggerType.RESPONSE_WINDOW: 'advice_on_response_window',
            TriggerType.PUSH_TO_TALK: True,  # Always enabled
            TriggerType.VOICE_QUERY: True,   # Always enabled
        }

        pref_attr = pref_map.get(trigger_type)
        if pref_attr is True:
            return True
        if pref_attr and hasattr(self.prefs, pref_attr):
            return getattr(self.prefs, pref_attr)

        return False

    def trigger_manual(self, user_query: Optional[str] = None) -> TriggerEvent:
        """
        Create a manual trigger event (push-to-talk or voice query).

        Args:
            user_query: Optional question from user (None = just get current advice)

        Returns:
            TriggerEvent for manual advice
        """
        trigger_type = TriggerType.VOICE_QUERY if user_query else TriggerType.PUSH_TO_TALK

        return TriggerEvent(
            trigger_type=trigger_type,
            turn=self._last_turn,
            phase=self._last_phase,
            is_your_turn=self._last_is_your_turn,
            user_query=user_query
        )

    def reset(self):
        """Reset state tracking for a new game."""
        self._last_turn = 0
        self._last_phase = ""
        self._last_is_your_turn = False
        self._last_has_priority = False
        self._last_opponent_spell = False
        self._fired_triggers.clear()
        logger.info("Advice trigger manager reset for new game")
