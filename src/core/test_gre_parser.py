"""
Unit tests for GREParser.
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.core.gre_parser import GREParser
from src.core.zone_manager import ZoneManager
from src.core.legacy_models import GameObject, PlayerState, GameHistory


class MockMatchScanner:
    """Mock MatchScanner for testing."""

    def __init__(self):
        self.game_objects = {}
        self.players = {}
        self.local_player_seat_id = None
        self._pending_decision = None
        self._decision_context = {}
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat = None
        self.priority_player_seat = None
        self.game_history = GameHistory(turn_number=0)

    def reset_match_state(self):
        """Reset match state for new game."""
        self.game_objects = {}
        self.players = {}
        self.current_turn = 0
        self.game_history = GameHistory(turn_number=0)


@pytest.fixture
def mock_scanner():
    """Fixture providing a mock MatchScanner."""
    return MockMatchScanner()


@pytest.fixture
def zone_manager():
    """Fixture providing a ZoneManager."""
    return ZoneManager()


@pytest.fixture
def gre_parser(mock_scanner, zone_manager):
    """Fixture providing a GREParser with mocked dependencies."""
    card_lookup = Mock()
    return GREParser(mock_scanner, zone_manager, card_lookup)


def test_gre_parser_initialization(mock_scanner, zone_manager):
    """Test GREParser initializes correctly."""
    card_lookup = Mock()
    parser = GREParser(mock_scanner, zone_manager, card_lookup)

    assert parser.scanner is mock_scanner
    assert parser.zone_manager is zone_manager
    assert parser.card_lookup is card_lookup


def test_parse_empty_event(gre_parser):
    """Test parsing an event without greToClientEvent."""
    event_data = {}
    result = gre_parser.parse_gre_to_client_event(event_data)
    assert result is False


def test_parse_event_without_messages(gre_parser):
    """Test parsing an event with no messages."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent"
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)
    assert result is False


def test_parse_ui_only_messages(gre_parser):
    """Test parsing UI-only messages returns False."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {"type": "GREMessageType_UIMessage"},
                {"type": "GREMessageType_UIMessage"}
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)
    assert result is False


def test_set_local_player_seat_id(gre_parser, mock_scanner):
    """Test setting local player seat ID from systemSeatIds."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "systemSeatIds": [1],
                    "gameStateMessage": {}
                }
            ]
        }
    }
    gre_parser.parse_gre_to_client_event(event_data)
    assert mock_scanner.local_player_seat_id == 1


def test_parse_actions_available_req(gre_parser, mock_scanner):
    """Test parsing ActionsAvailableReq message."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_ActionsAvailableReq",
                    "actionsAvailableReq": {"someAction": "data"}
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert mock_scanner._pending_decision == "actions_available"
    assert mock_scanner._decision_context["type"] == "priority"


def test_parse_game_objects(gre_parser, mock_scanner, zone_manager):
    """Test parsing game objects from GameStateMessage."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "gameObjects": [
                            {
                                "instanceId": 100,
                                "grpId": 1000,
                                "zoneId": 1,
                                "ownerSeatId": 1
                            }
                        ]
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert 100 in mock_scanner.game_objects
    assert mock_scanner.game_objects[100].grp_id == 1000
    assert mock_scanner.game_objects[100].zone_id == 1


def test_parse_game_objects_zone_transfer(gre_parser, mock_scanner, zone_manager):
    """Test game object zone transfer."""
    # First add an object in zone 1
    mock_scanner.game_objects[100] = GameObject(
        instance_id=100,
        grp_id=1000,
        zone_id=1,
        owner_seat_id=1
    )
    zone_manager.transfer_card(100, new_zone_id=1)

    # Now move it to zone 2
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "gameObjects": [
                            {
                                "instanceId": 100,
                                "grpId": 1000,
                                "zoneId": 2,
                                "ownerSeatId": 1
                            }
                        ]
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert mock_scanner.game_objects[100].zone_id == 2
    assert 100 in zone_manager.get_zone_contents(2)
    assert 100 not in zone_manager.get_zone_contents(1)


def test_parse_zones(gre_parser, mock_scanner, zone_manager):
    """Test parsing zones from GameStateMessage."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "zones": [
                            {
                                "zoneId": 1,
                                "type": "ZoneType_Hand",
                                "ownerSeatId": 1,
                                "objectInstanceIds": [100, 101]
                            }
                        ]
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert zone_manager.zone_id_to_type[1] == "ZoneType_Hand"
    assert 100 in zone_manager.get_zone_contents(1)
    assert 101 in zone_manager.get_zone_contents(1)


def test_parse_players(gre_parser, mock_scanner):
    """Test parsing player data from GameStateMessage."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "players": [
                            {
                                "systemSeatNumber": 1,
                                "lifeTotal": 20
                            },
                            {
                                "systemSeatNumber": 2,
                                "lifeTotal": 18
                            }
                        ]
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert 1 in mock_scanner.players
    assert 2 in mock_scanner.players
    assert mock_scanner.players[1].life_total == 20
    assert mock_scanner.players[2].life_total == 18


def test_parse_players_life_total_update(gre_parser, mock_scanner):
    """Test updating player life total."""
    # Initialize player
    mock_scanner.players[1] = PlayerState(seat_id=1, life_total=20)

    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "players": [
                            {
                                "systemSeatNumber": 1,
                                "lifeTotal": 15
                            }
                        ]
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert mock_scanner.players[1].life_total == 15


def test_parse_turn_info(gre_parser, mock_scanner):
    """Test parsing turn information."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "turnInfo": {
                            "turnNumber": 1,
                            "phase": "Phase_Main1",
                            "activePlayer": 1,
                            "priorityPlayer": 1
                        }
                    }
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert mock_scanner.current_turn == 1
    assert mock_scanner.current_phase == "Phase_Main1"
    assert mock_scanner.active_player_seat == 1
    assert mock_scanner.priority_player_seat == 1


def test_parse_turn_info_new_game_detection(gre_parser, mock_scanner):
    """Test that turn 1 after turn > 1 triggers reset."""
    # Simulate being in turn 5
    mock_scanner.current_turn = 5

    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "turnInfo": {
                            "turnNumber": 1
                        }
                    }
                }
            ]
        }
    }

    # Add a spy to verify reset was called
    reset_called = False
    original_reset = mock_scanner.reset_match_state

    def spy_reset():
        nonlocal reset_called
        reset_called = True
        original_reset()

    mock_scanner.reset_match_state = spy_reset

    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert reset_called is True
    assert mock_scanner.current_turn == 1


def test_parse_annotations_zone_transfer(gre_parser, mock_scanner, zone_manager):
    """Test parsing zone transfer annotations."""
    # Setup existing game object in zone 1
    mock_scanner.game_objects[100] = GameObject(
        instance_id=100,
        grp_id=1000,
        zone_id=1,
        owner_seat_id=1
    )
    zone_manager.transfer_card(100, new_zone_id=1)

    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_Annotation",
                    "annotations": [
                        {
                            "type": ["AnnotationType_ZoneTransfer"],
                            "affectedIds": [100],
                            "details": [
                                {
                                    "key": "zone_dest",
                                    "valueInt32": [2]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert mock_scanner.game_objects[100].zone_id == 2
    assert 100 in zone_manager.get_zone_contents(2)
    assert 100 not in zone_manager.get_zone_contents(1)


def test_parse_game_state_message_json_string(gre_parser, mock_scanner):
    """Test parsing gameStateMessage as JSON string."""
    import json

    game_state = {
        "players": [
            {"systemSeatNumber": 1, "lifeTotal": 20}
        ]
    }

    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": json.dumps(game_state)
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is True
    assert 1 in mock_scanner.players
    assert mock_scanner.players[1].life_total == 20


def test_parse_game_state_message_invalid_json(gre_parser):
    """Test handling of invalid JSON in gameStateMessage."""
    event_data = {
        "greToClientEvent": {
            "type": "SomeEvent",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": "invalid json {{"
                }
            ]
        }
    }
    result = gre_parser.parse_gre_to_client_event(event_data)

    assert result is False


def test_parse_mana_pool_list_format(gre_parser):
    """Test parsing mana pool in list format."""
    mana_data = [
        {"color": ["ManaColor_Red"], "count": 2},
        {"color": ["ManaColor_Green"], "count": 1},
        {"color": ["ManaColor_Blue"], "count": 3}
    ]

    result = gre_parser._parse_mana_pool(mana_data)

    assert result == {"R": 2, "G": 1, "U": 3}


def test_parse_mana_pool_dict_format(gre_parser):
    """Test parsing mana pool in dict format."""
    mana_data = {
        "Red": 2,
        "White": 1,
        "Colorless": 3
    }

    result = gre_parser._parse_mana_pool(mana_data)

    assert result == {"R": 2, "W": 1, "C": 3}


def test_parse_players_with_mana_pool(gre_parser, mock_scanner):
    """Test parsing players with mana pool data."""
    players_data = [
        {
            "systemSeatNumber": 1,
            "lifeTotal": 20,
            "pendingMana": [
                {"color": ["ManaColor_Red"], "count": 2},
                {"color": ["ManaColor_Green"], "count": 1}
            ]
        }
    ]

    result = gre_parser._parse_players(players_data)

    assert result == True
    assert 1 in mock_scanner.players
    assert mock_scanner.players[1].mana_pool == {"R": 2, "G": 1}


def test_mana_pool_cleared_on_phase_change(gre_parser, mock_scanner):
    """Test that mana pool is cleared on major phase changes."""
    from src.core.legacy_models import PlayerState

    # Setup initial state
    mock_scanner.current_phase = "Phase_Main1"
    mock_scanner.players[1] = PlayerState(seat_id=1, mana_pool={"R": 2, "G": 1})
    mock_scanner.players[2] = PlayerState(seat_id=2, mana_pool={"U": 3})

    # Simulate phase change to combat
    turn_info = {
        "turnNumber": 1,
        "phase": "Phase_Combat",
        "activePlayer": 1,
        "priorityPlayer": 1
    }

    result = gre_parser._parse_turn_info(turn_info)

    assert result == True
    assert mock_scanner.current_phase == "Phase_Combat"
    assert mock_scanner.players[1].mana_pool == {}
    assert mock_scanner.players[2].mana_pool == {}


def test_should_clear_mana_on_phase_change(gre_parser):
    """Test the mana clearing logic."""
    # Major phase transitions should clear mana
    assert gre_parser._should_clear_mana_on_phase_change("Phase_Main1", "Phase_Combat") == True
    assert gre_parser._should_clear_mana_on_phase_change("Phase_Combat", "Phase_Main2") == True

    # Same phase should not clear mana
    assert gre_parser._should_clear_mana_on_phase_change("Phase_Main1", "Phase_Main1") == False

    # Non-major phases should not trigger clearing
    assert gre_parser._should_clear_mana_on_phase_change("Phase_Unknown", "Phase_Main1") == False
