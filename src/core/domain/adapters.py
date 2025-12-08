"""
Adapters for converting between legacy BoardState and domain GameState.

This module provides the bridge between the old data structures in mtga.py
and the new clean domain models. During the migration period, both models
can coexist and be converted as needed.

Usage:
    # Converting legacy to domain model
    board_state = game_state_manager.get_current_board_state()  # Old API
    game_state = BoardStateAdapter.to_game_state(board_state)

    # Converting domain back to legacy (for backwards compat)
    legacy_state = BoardStateAdapter.from_game_state(game_state)
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

from .game_state import (
    GameState,
    PlayerGameState,
    Permanent,
    CardIdentity,
    Phase,
    ZoneCollection,
    CombatState,
    TurnHistory,
)

logger = logging.getLogger(__name__)


class BoardStateAdapter:
    """
    Adapter for converting between BoardState and GameState.

    This adapter maintains a clean separation between:
    - Legacy mtga.py data structures (BoardState, GameObject)
    - New domain models (GameState, Permanent, etc.)

    The adapter is bidirectional to support gradual migration.
    """

    @staticmethod
    def to_game_state(board_state) -> GameState:
        """
        Convert legacy BoardState to domain GameState.

        This method extracts data from the legacy BoardState structure
        and constructs a clean domain model.

        Args:
            board_state: Legacy BoardState from mtga.py

        Returns:
            Domain model GameState

        Example:
            >>> board_state = game_state_manager.get_current_board_state()
            >>> game_state = BoardStateAdapter.to_game_state(board_state)
            >>> if game_state.is_combat_phase:
            ...     print("Combat!")
        """
        # Convert phase
        phase = Phase.from_arena_string(board_state.current_phase)

        # Create game state with basic info
        game_state = GameState(
            turn_number=board_state.current_turn,
            phase=phase,
            in_mulligan=board_state.in_mulligan_phase,
            game_stage=board_state.game_stage,
        )

        # Convert local player
        game_state.local_player = BoardStateAdapter._to_player_state(
            player_id=board_state.your_seat_id,
            life=board_state.your_life,
            energy=board_state.your_energy,
            mana_pool=board_state.your_mana_pool,
            hand_cards=board_state.your_hand,
            graveyard_cards=board_state.your_graveyard,
            exile_cards=board_state.your_exile,
            library_count=board_state.your_library_count,
            decklist=board_state.your_decklist,
            library_top_known=board_state.library_top_known,
            has_priority=board_state.has_priority,
            is_active=board_state.is_your_turn,
        )

        # Convert opponent
        game_state.opponent = BoardStateAdapter._to_player_state(
            player_id=board_state.opponent_seat_id,
            life=board_state.opponent_life,
            energy=board_state.opponent_energy,
            mana_pool={},  # Opponent mana pool not tracked in BoardState
            hand_cards=[],  # Opponent hand is hidden
            graveyard_cards=board_state.opponent_graveyard,
            exile_cards=board_state.opponent_exile,
            library_count=board_state.opponent_library_count,
            decklist={},  # Opponent decklist unknown
            library_top_known=[],
            has_priority=not board_state.has_priority,  # Inverse of local player
            is_active=not board_state.is_your_turn,  # Inverse of local player
        )

        # Set opponent hand count (we don't have the cards, just the count)
        game_state.opponent.hand = ZoneCollection(zone_name="Hand", is_hidden=True)
        # Note: We can't populate actual cards since opponent hand is hidden
        # The hand count is available via len(game_state.opponent.hand) after we
        # add placeholder cards in the migration logic (if needed)

        # Convert battlefield
        game_state.local_battlefield = [
            BoardStateAdapter._to_permanent(obj, board_state.your_seat_id)
            for obj in board_state.your_battlefield
        ]

        game_state.opponent_battlefield = [
            BoardStateAdapter._to_permanent(obj, board_state.opponent_seat_id)
            for obj in board_state.opponent_battlefield
        ]

        # Convert stack
        game_state.stack = ZoneCollection(zone_name="Stack", is_ordered=True)
        for stack_obj in board_state.stack:
            card_id = CardIdentity(
                grp_id=stack_obj.grp_id,
                instance_id=stack_obj.instance_id,
                name=stack_obj.name,
            )
            game_state.stack.add_card(card_id)

        # Convert combat state (from history)
        if board_state.history:
            game_state.combat = BoardStateAdapter._to_combat_state(board_state.history)
            game_state.history = BoardStateAdapter._to_turn_history(board_state.history)

        return game_state

    @staticmethod
    def _to_player_state(
        player_id: int,
        life: int,
        energy: int,
        mana_pool: dict,
        hand_cards: list,
        graveyard_cards: list,
        exile_cards: list,
        library_count: int,
        decklist: dict,
        library_top_known: list,
        has_priority: bool,
        is_active: bool,
    ) -> PlayerGameState:
        """
        Helper to convert player data to PlayerGameState.

        Args:
            player_id: Player seat ID
            life: Life total
            energy: Energy counters
            mana_pool: Mana pool dict
            hand_cards: List of GameObject in hand
            graveyard_cards: List of GameObject in graveyard
            exile_cards: List of GameObject in exile
            library_count: Number of cards in library
            decklist: Deck composition
            library_top_known: Known top library cards
            has_priority: Whether player has priority
            is_active: Whether it's player's turn

        Returns:
            PlayerGameState domain model
        """
        player = PlayerGameState(
            player_id=player_id,
            life_total=life,
            energy=energy,
            mana_pool=mana_pool.copy() if mana_pool else {},
            has_priority=has_priority,
            is_active_player=is_active,
            library_count=library_count,
            decklist=decklist.copy() if decklist else {},
            library_top_known=library_top_known.copy() if library_top_known else [],
        )

        # Convert hand
        player.hand = ZoneCollection(zone_name="Hand", is_hidden=True)
        for card in hand_cards:
            card_id = CardIdentity(
                grp_id=card.grp_id,
                instance_id=card.instance_id,
                name=card.name,
            )
            player.hand.add_card(card_id)

        # Convert graveyard
        player.graveyard = ZoneCollection(zone_name="Graveyard", is_ordered=True)
        for card in graveyard_cards:
            card_id = CardIdentity(
                grp_id=card.grp_id,
                instance_id=card.instance_id,
                name=card.name,
            )
            player.graveyard.add_card(card_id)

        # Convert exile
        player.exile = ZoneCollection(zone_name="Exile")
        for card in exile_cards:
            card_id = CardIdentity(
                grp_id=card.grp_id,
                instance_id=card.instance_id,
                name=card.name,
            )
            player.exile.add_card(card_id)

        return player

    @staticmethod
    def _to_permanent(game_obj, owner_id: int) -> Permanent:
        """
        Convert GameObject to Permanent.

        Args:
            game_obj: GameObject from mtga.py
            owner_id: Owner seat ID

        Returns:
            Permanent domain model
        """
        identity = CardIdentity(
            grp_id=game_obj.grp_id,
            instance_id=game_obj.instance_id,
            name=game_obj.name,
        )

        # Use effective stats if available (GameObject property)
        # Fallback to base stats or None
        power = getattr(game_obj, 'effective_power', getattr(game_obj, 'base_power', None))
        toughness = getattr(game_obj, 'effective_toughness', getattr(game_obj, 'base_toughness', None))

        return Permanent(
            identity=identity,
            controller_id=owner_id,  # Assume controller = owner for now
            owner_id=owner_id,
            tapped=game_obj.is_tapped,
            attacking=game_obj.is_attacking,
            blocking=None,  # Not tracked in GameObject currently
            summoning_sick=game_obj.summoning_sick,
            power=power,
            toughness=toughness,
            counters=game_obj.counters.copy() if game_obj.counters else {},
            attached_to=game_obj.attached_to,
            type_line=game_obj.type_line,
            color_identity=game_obj.color_identity,
        )

    @staticmethod
    def _to_combat_state(history) -> CombatState:
        """
        Convert GameHistory to CombatState.

        Args:
            history: GameHistory from mtga.py

        Returns:
            CombatState domain model
        """
        combat = CombatState()

        # Convert attackers
        if hasattr(history, 'current_attackers'):
            combat.attackers = set(history.current_attackers)

        # Convert blockers
        if hasattr(history, 'current_blockers'):
            for attacker_id, blocker_id in history.current_blockers.items():
                combat.blockers[attacker_id] = [blocker_id]  # Convert to list

        # Convert damage assignments
        if hasattr(history, 'combat_damage_assignments'):
            combat.damage_assignments = history.combat_damage_assignments.copy()

        return combat

    @staticmethod
    def _to_turn_history(history) -> TurnHistory:
        """
        Convert GameHistory to TurnHistory.

        Args:
            history: GameHistory from mtga.py

        Returns:
            TurnHistory domain model
        """
        turn_hist = TurnHistory(turn_number=history.turn_number)

        # Convert cards played this turn
        if hasattr(history, 'cards_played_this_turn'):
            turn_hist.cards_played = [
                CardIdentity(
                    grp_id=card.grp_id,
                    instance_id=card.instance_id,
                    name=card.name,
                )
                for card in history.cards_played_this_turn
            ]

        # Lands played
        if hasattr(history, 'lands_played_this_turn'):
            turn_hist.lands_played_count = history.lands_played_this_turn

        # Combat history
        if hasattr(history, 'attackers_this_turn'):
            turn_hist.creatures_that_attacked = [
                card.instance_id for card in history.attackers_this_turn
            ]

        if hasattr(history, 'blockers_this_turn'):
            turn_hist.creatures_that_blocked = [
                card.instance_id for card in history.blockers_this_turn
            ]

        # Damage dealt
        if hasattr(history, 'damage_dealt'):
            turn_hist.damage_dealt = history.damage_dealt.copy()

        # Creatures died
        if hasattr(history, 'died_this_turn'):
            turn_hist.creatures_died = history.died_this_turn.copy()

        return turn_hist

    @staticmethod
    def from_game_state(game_state: GameState):
        """
        Convert domain GameState back to legacy BoardState.

        This is needed during the transition period to support
        existing code that expects BoardState.

        Note: This is a temporary backwards compatibility shim.
        Eventually this method will be removed when migration is complete.

        Args:
            game_state: Domain model GameState

        Returns:
            Legacy BoardState (requires import of mtga module)

        Example:
            >>> game_state = GameState(...)
            >>> board_state = BoardStateAdapter.from_game_state(game_state)
            >>> legacy_function(board_state)  # Works with old code
        """
        # Import here to avoid circular dependency
        # This is intentional - adapter is the bridge between layers
        from ..mtga import BoardState, GameObject, GameHistory

        # Create legacy BoardState
        board_state = BoardState(
            your_seat_id=game_state.local_player.player_id,
            opponent_seat_id=game_state.opponent.player_id,
            your_life=game_state.local_player.life_total,
            opponent_life=game_state.opponent.life_total,
            current_turn=game_state.turn_number,
            current_phase=BoardStateAdapter._phase_to_arena_string(game_state.phase),
            is_your_turn=game_state.local_player.is_active_player,
            has_priority=game_state.local_player.has_priority,
            your_mana_pool=game_state.local_player.mana_pool.copy(),
            your_energy=game_state.local_player.energy,
            opponent_energy=game_state.opponent.energy,
            in_mulligan_phase=game_state.in_mulligan,
            game_stage=game_state.game_stage,
        )

        # Convert hand
        board_state.your_hand = [
            BoardStateAdapter._to_game_object(card, game_state.local_player.player_id)
            for card in game_state.local_player.hand.cards
        ]
        board_state.your_hand_count = len(board_state.your_hand)
        board_state.opponent_hand_count = len(game_state.opponent.hand)

        # Convert battlefield
        board_state.your_battlefield = [
            BoardStateAdapter._permanent_to_game_object(perm)
            for perm in game_state.local_battlefield
        ]

        board_state.opponent_battlefield = [
            BoardStateAdapter._permanent_to_game_object(perm)
            for perm in game_state.opponent_battlefield
        ]

        # Convert graveyards
        board_state.your_graveyard = [
            BoardStateAdapter._to_game_object(card, game_state.local_player.player_id)
            for card in game_state.local_player.graveyard.cards
        ]

        board_state.opponent_graveyard = [
            BoardStateAdapter._to_game_object(card, game_state.opponent.player_id)
            for card in game_state.opponent.graveyard.cards
        ]

        # Convert exile zones
        board_state.your_exile = [
            BoardStateAdapter._to_game_object(card, game_state.local_player.player_id)
            for card in game_state.local_player.exile.cards
        ]

        board_state.opponent_exile = [
            BoardStateAdapter._to_game_object(card, game_state.opponent.player_id)
            for card in game_state.opponent.exile.cards
        ]

        # Convert stack
        board_state.stack = [
            BoardStateAdapter._to_game_object(card, 0)  # Stack owner unknown
            for card in game_state.stack.cards
        ]

        # Library counts
        board_state.your_library_count = game_state.local_player.library_count
        board_state.opponent_library_count = game_state.opponent.library_count

        # Deck tracking
        board_state.your_decklist = game_state.local_player.decklist.copy()
        board_state.library_top_known = game_state.local_player.library_top_known.copy()
        board_state.your_deck_remaining = game_state.local_player.library_count

        # Convert history
        if game_state.history:
            board_state.history = BoardStateAdapter._to_game_history(
                game_state.history, game_state.combat
            )

        return board_state

    @staticmethod
    def _phase_to_arena_string(phase: Phase) -> str:
        """Convert Phase enum to Arena phase string."""
        phase_map = {
            Phase.UNTAP: "Phase_Beginning_Untap",
            Phase.UPKEEP: "Phase_Beginning_Upkeep",
            Phase.DRAW: "Phase_Beginning_Draw",
            Phase.MAIN_1: "Phase_Main1",
            Phase.COMBAT_BEGIN: "Phase_Combat_Begin",
            Phase.COMBAT_ATTACKERS: "Phase_Combat_Attackers",
            Phase.COMBAT_BLOCKERS: "Phase_Combat_Blockers",
            Phase.COMBAT_DAMAGE: "Phase_Combat_Damage",
            Phase.COMBAT_END: "Phase_Combat_End",
            Phase.MAIN_2: "Phase_Main2",
            Phase.END: "Phase_Ending_End",
            Phase.CLEANUP: "Phase_Ending_Cleanup",
        }
        return phase_map.get(phase, "")

    @staticmethod
    def _to_game_object(card: CardIdentity, owner_seat_id: int):
        """Convert CardIdentity to GameObject."""
        from ..mtga import GameObject

        return GameObject(
            instance_id=card.instance_id,
            grp_id=card.grp_id,
            zone_id=0,  # Zone ID not available in CardIdentity
            owner_seat_id=owner_seat_id,
            name=card.name,
        )

    @staticmethod
    def _permanent_to_game_object(perm: Permanent):
        """Convert Permanent to GameObject."""
        from ..mtga import GameObject

        return GameObject(
            instance_id=perm.identity.instance_id,
            grp_id=perm.identity.grp_id,
            zone_id=0,  # Zone ID not tracked in Permanent
            owner_seat_id=perm.owner_id,
            name=perm.identity.name,
            color_identity=perm.color_identity,
            base_power=perm.power,
            base_toughness=perm.toughness,
            is_tapped=perm.tapped,
            is_attacking=perm.attacking,
            summoning_sick=perm.summoning_sick,
            counters=perm.counters.copy(),
            attached_to=perm.attached_to,
            type_line=perm.type_line,
        )

    @staticmethod
    def _to_game_history(history: TurnHistory, combat: CombatState):
        """Convert TurnHistory and CombatState to GameHistory."""
        from ..mtga import GameHistory, GameObject

        game_history = GameHistory(turn_number=history.turn_number)

        # Convert cards played
        game_history.cards_played_this_turn = [
            BoardStateAdapter._to_game_object(card, 0)  # Owner unknown
            for card in history.cards_played
        ]

        game_history.lands_played_this_turn = history.lands_played_count

        # Combat state
        game_history.current_attackers = list(combat.attackers)
        game_history.current_blockers = {
            attacker: blockers[0] if blockers else 0
            for attacker, blockers in combat.blockers.items()
        }
        game_history.combat_damage_assignments = combat.damage_assignments.copy()

        # Damage and deaths
        game_history.damage_dealt = history.damage_dealt.copy()
        game_history.died_this_turn = history.creatures_died.copy()

        return game_history
