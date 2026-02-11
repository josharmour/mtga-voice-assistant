
import pytest
from src.core.legacy_models import BoardState, GameObject
from src.core.domain.adapters import BoardStateAdapter

def test_board_state_adapter():
    board_state = BoardState(
        your_seat_id=1,
        opponent_seat_id=2,
        your_life=20,
        opponent_life=18,
        your_mana_pool={'R': 2, 'G': 1},
        your_energy=3,
        opponent_energy=1,
        your_library_count=40,
        opponent_library_count=38,
        your_hand=[GameObject(instance_id=101, grp_id=1001, zone_id=1, owner_seat_id=1, name='Card A')],
        your_graveyard=[GameObject(instance_id=102, grp_id=1002, zone_id=2, owner_seat_id=1, name='Card B')],
        your_exile=[GameObject(instance_id=103, grp_id=1003, zone_id=3, owner_seat_id=1, name='Card C')],
        opponent_graveyard=[GameObject(instance_id=201, grp_id=2001, zone_id=4, owner_seat_id=2, name='Card X')],
        opponent_exile=[GameObject(instance_id=202, grp_id=2002, zone_id=5, owner_seat_id=2, name='Card Y')],
        current_phase='Phase_Main1'
    )

    game_state = BoardStateAdapter.to_game_state(board_state)

    assert game_state.local_player.player_id == 1
    assert game_state.opponent.player_id == 2
    assert game_state.local_player.life_total == 20
    assert game_state.opponent.life_total == 18
    assert game_state.local_player.mana_pool == {'R': 2, 'G': 1}
    assert game_state.local_player.energy == 3
    assert game_state.opponent.energy == 1
    assert game_state.local_player.library_count == 40
    assert game_state.opponent.library_count == 38
    assert len(game_state.local_player.hand.cards) == 1
    assert game_state.local_player.hand.cards[0].name == 'Card A'
    assert len(game_state.local_player.graveyard.cards) == 1
    assert game_state.local_player.graveyard.cards[0].name == 'Card B'
    assert len(game_state.local_player.exile.cards) == 1
    assert game_state.local_player.exile.cards[0].name == 'Card C'
    assert len(game_state.opponent.graveyard.cards) == 1
    assert game_state.opponent.graveyard.cards[0].name == 'Card X'
    assert len(game_state.opponent.exile.cards) == 1
    assert game_state.opponent.exile.cards[0].name == 'Card Y'
    assert game_state.phase.name == 'MAIN_1'
