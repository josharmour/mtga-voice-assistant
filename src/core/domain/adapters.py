"""
Adapters to convert between legacy and domain models.
"""
from src.core.domain.game_state import GameState, PlayerGameState, CardIdentity, Permanent, Phase, ZoneCollection
from src.core.legacy_models import BoardState, GameObject

class BoardStateAdapter:
    """
    Adapter to convert legacy BoardState to the new GameState domain model.
    """

    @staticmethod
    def to_game_state(board_state: BoardState) -> GameState:
        """
        Convert a legacy BoardState object to a new GameState domain model.
        """
        if not board_state:
            return None

        local_player = PlayerGameState(
            player_id=board_state.your_seat_id,
            life_total=board_state.your_life,
            has_priority=board_state.has_priority,
            is_active_player=board_state.is_your_turn,
            mana_pool=board_state.your_mana_pool,
            energy=board_state.your_energy,
            library_count=board_state.your_library_count,
            hand=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.your_hand]
            ),
            graveyard=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.your_graveyard]
            ),
            exile=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.your_exile]
            )
        )

        opponent = PlayerGameState(
            player_id=board_state.opponent_seat_id,
            life_total=board_state.opponent_life,
            is_active_player=not board_state.is_your_turn,
            energy=board_state.opponent_energy,
            library_count=board_state.opponent_library_count,
            graveyard=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.opponent_graveyard]
            ),
            exile=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.opponent_exile]
            )
        )

        game_state = GameState(
            turn_number=board_state.current_turn,
            phase=Phase.from_arena_string(board_state.current_phase),
            local_player=local_player,
            opponent=opponent,
            local_battlefield=[BoardStateAdapter._to_permanent(card) for card in board_state.your_battlefield],
            opponent_battlefield=[BoardStateAdapter._to_permanent(card) for card in board_state.opponent_battlefield],
            stack=ZoneCollection(
                cards=[BoardStateAdapter._to_card_identity(card) for card in board_state.stack]
            )
        )
        return game_state

    @staticmethod
    def _to_card_identity(game_object: GameObject) -> CardIdentity:
        return CardIdentity(
            grp_id=game_object.grp_id,
            instance_id=game_object.instance_id,
            name=game_object.name
        )

    @staticmethod
    def _to_permanent(game_object: GameObject) -> Permanent:
        return Permanent(
            identity=BoardStateAdapter._to_card_identity(game_object),
            controller_id=game_object.owner_seat_id,
            owner_id=game_object.owner_seat_id,
            tapped=game_object.is_tapped,
            attacking=game_object.is_attacking,
            summoning_sick=game_object.summoning_sick,
            power=game_object.base_power,
            toughness=game_object.base_toughness
        )
