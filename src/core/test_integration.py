"""
Integration tests for the refactored game state system.

These tests verify that GREParser, ZoneManager, and MatchScanner work together correctly.
"""
import pytest
from unittest.mock import Mock
from src.core.mtga import GameStateManager
from src.core.gre_parser import GREParser
from src.core.zone_manager import ZoneManager
from src.core.legacy_models import GameObject, PlayerState


@pytest.fixture
def mock_card_lookup():
    """Fixture providing a mock card lookup."""
    return Mock()


def test_basic_game_flow_integration(mock_card_lookup):
    """Test a basic game flow with player setup and turn progression."""
    # Create game state manager
    gsm = GameStateManager(card_lookup=mock_card_lookup)

    # Simulate GRE event with player initialization
    gre_event = {
        "greToClientEvent": {
            "type": "GREMessageType_ConnectResp",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "systemSeatIds": [1],
                    "gameStateMessage": {
                        "players": [
                            {
                                "systemSeatNumber": 1,
                                "lifeTotal": 20
                            },
                            {
                                "systemSeatNumber": 2,
                                "lifeTotal": 20
                            }
                        ],
                        "turnInfo": {
                            "turnNumber": 1,
                            "phase": "Phase_Beginning_Upkeep",
                            "activePlayer": 1,
                            "priorityPlayer": 1
                        }
                    }
                }
            ]
        }
    }

    # Process the event
    state_changed = gsm._on_gre_to_client_event(gre_event)

    # Verify state was updated
    assert state_changed is True
    assert gsm.match_scanner.local_player_seat_id == 1
    assert gsm.match_scanner.current_turn == 1
    assert 1 in gsm.match_scanner.players
    assert 2 in gsm.match_scanner.players
    assert gsm.match_scanner.players[1].life_total == 20
    assert gsm.match_scanner.players[2].life_total == 20


def test_card_movement_integration(mock_card_lookup):
    """Test card movement between zones."""
    # Create game state manager
    gsm = GameStateManager(card_lookup=mock_card_lookup)

    # First, set up initial state with a card in hand
    initial_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "systemSeatIds": [1],
                    "gameStateMessage": {
                        "gameObjects": [
                            {
                                "instanceId": 100,
                                "grpId": 1000,
                                "zoneId": 1,
                                "ownerSeatId": 1
                            }
                        ],
                        "zones": [
                            {
                                "zoneId": 1,
                                "type": "ZoneType_Hand",
                                "ownerSeatId": 1,
                                "objectInstanceIds": [100]
                            }
                        ]
                    }
                }
            ]
        }
    }

    gsm._on_gre_to_client_event(initial_event)

    # Verify card is in hand
    assert 100 in gsm.match_scanner.game_objects
    assert gsm.match_scanner.game_objects[100].zone_id == 1
    assert 100 in gsm.zone_manager.get_zone_contents(1)

    # Now move the card to battlefield (zone 2)
    transfer_event = {
        "greToClientEvent": {
            "type": "GREMessageType_ZoneTransfer",
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

    state_changed = gsm._on_gre_to_client_event(transfer_event)

    # Verify card moved to battlefield
    assert state_changed is True
    assert gsm.match_scanner.game_objects[100].zone_id == 2
    assert 100 in gsm.zone_manager.get_zone_contents(2)
    assert 100 not in gsm.zone_manager.get_zone_contents(1)


def test_life_total_updates_integration(mock_card_lookup):
    """Test player life total updates through the system."""
    gsm = GameStateManager(card_lookup=mock_card_lookup)

    # Initialize players
    init_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
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
                                "lifeTotal": 20
                            }
                        ]
                    }
                }
            ]
        }
    }

    gsm._on_gre_to_client_event(init_event)

    # Damage event - reduce player 2's life
    damage_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "players": [
                            {
                                "systemSeatNumber": 2,
                                "lifeTotal": 17
                            }
                        ]
                    }
                }
            ]
        }
    }

    state_changed = gsm._on_gre_to_client_event(damage_event)

    # Verify life total changed
    assert state_changed is True
    assert gsm.match_scanner.players[1].life_total == 20  # Unchanged
    assert gsm.match_scanner.players[2].life_total == 17  # Damaged


def test_turn_progression_integration(mock_card_lookup):
    """Test turn progression through the system."""
    gsm = GameStateManager(card_lookup=mock_card_lookup)

    # Start at turn 1
    turn1_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
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

    gsm._on_gre_to_client_event(turn1_event)
    assert gsm.match_scanner.current_turn == 1

    # Progress to turn 2
    turn2_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "turnInfo": {
                            "turnNumber": 2,
                            "phase": "Phase_Beginning_Upkeep",
                            "activePlayer": 2,
                            "priorityPlayer": 2
                        }
                    }
                }
            ]
        }
    }

    state_changed = gsm._on_gre_to_client_event(turn2_event)

    # Verify turn progression
    assert state_changed is True
    assert gsm.match_scanner.current_turn == 2
    assert gsm.match_scanner.active_player_seat == 2
    assert gsm.match_scanner.priority_player_seat == 2


def test_zone_metadata_integration(mock_card_lookup):
    """Test that zone metadata is properly tracked."""
    gsm = GameStateManager(card_lookup=mock_card_lookup)

    zones_event = {
        "greToClientEvent": {
            "type": "GREMessageType_GameStateMessage",
            "greToClientMessages": [
                {
                    "type": "GREMessageType_GameStateMessage",
                    "gameStateMessage": {
                        "zones": [
                            {
                                "zoneId": 1,
                                "type": "ZoneType_Hand",
                                "ownerSeatId": 1
                            },
                            {
                                "zoneId": 2,
                                "type": "ZoneType_Battlefield",
                                "ownerSeatId": 1
                            },
                            {
                                "zoneId": 3,
                                "type": "ZoneType_Graveyard",
                                "ownerSeatId": 1
                            }
                        ]
                    }
                }
            ]
        }
    }

    gsm._on_gre_to_client_event(zones_event)

    # Verify zone metadata
    from src.core.domain.game_state import ZoneType
    assert gsm.zone_manager.get_zone_enum(1) == ZoneType.HAND
    assert gsm.zone_manager.get_zone_enum(2) == ZoneType.BATTLEFIELD
    assert gsm.zone_manager.get_zone_enum(3) == ZoneType.GRAVEYARD
