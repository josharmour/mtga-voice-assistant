"""
Unit tests for ZoneManager.
"""
import pytest
from src.core.zone_manager import ZoneManager
from src.core.domain.game_state import ZoneType


def test_zone_manager_initialization():
    """Test that ZoneManager initializes with empty state."""
    zone_manager = ZoneManager()

    assert len(zone_manager.zone_id_to_type) == 0
    assert len(zone_manager.zone_id_to_enum) == 0
    assert len(zone_manager.zone_id_to_owner) == 0
    assert len(zone_manager._zone_objects) == 0


def test_update_zone_metadata():
    """Test updating zone metadata."""
    zone_manager = ZoneManager()

    zone_manager.update_zone_metadata(1, "ZoneType_Hand", 1)

    assert zone_manager.zone_id_to_type[1] == "ZoneType_Hand"
    assert zone_manager.zone_id_to_enum[1] == ZoneType.HAND
    assert zone_manager.zone_id_to_owner[1] == 1


def test_update_zone_metadata_without_owner():
    """Test updating zone metadata without an owner."""
    zone_manager = ZoneManager()

    zone_manager.update_zone_metadata(2, "ZoneType_Stack")

    assert zone_manager.zone_id_to_type[2] == "ZoneType_Stack"
    assert zone_manager.zone_id_to_enum[2] == ZoneType.STACK
    assert 2 not in zone_manager.zone_id_to_owner


def test_transfer_card_new_zone():
    """Test transferring a card to a new zone."""
    zone_manager = ZoneManager()

    zone_manager.transfer_card(instance_id=100, new_zone_id=1)

    assert 100 in zone_manager.get_zone_contents(1)
    assert len(zone_manager.get_zone_contents(1)) == 1


def test_transfer_card_between_zones():
    """Test transferring a card from one zone to another."""
    zone_manager = ZoneManager()

    # Add card to zone 1
    zone_manager.transfer_card(instance_id=100, new_zone_id=1)
    assert 100 in zone_manager.get_zone_contents(1)

    # Move card from zone 1 to zone 2
    zone_manager.transfer_card(instance_id=100, new_zone_id=2, old_zone_id=1)
    assert 100 not in zone_manager.get_zone_contents(1)
    assert 100 in zone_manager.get_zone_contents(2)


def test_transfer_multiple_cards():
    """Test transferring multiple cards to the same zone."""
    zone_manager = ZoneManager()

    zone_manager.transfer_card(instance_id=100, new_zone_id=1)
    zone_manager.transfer_card(instance_id=101, new_zone_id=1)
    zone_manager.transfer_card(instance_id=102, new_zone_id=1)

    zone_contents = zone_manager.get_zone_contents(1)
    assert len(zone_contents) == 3
    assert 100 in zone_contents
    assert 101 in zone_contents
    assert 102 in zone_contents


def test_transfer_card_cleans_up_empty_zones():
    """Test that transferring the last card out of a zone removes the zone."""
    zone_manager = ZoneManager()

    # Add card to zone 1
    zone_manager.transfer_card(instance_id=100, new_zone_id=1)
    assert 1 in zone_manager._zone_objects

    # Move card to zone 2
    zone_manager.transfer_card(instance_id=100, new_zone_id=2, old_zone_id=1)
    assert 1 not in zone_manager._zone_objects
    assert 2 in zone_manager._zone_objects


def test_get_zone_contents_empty():
    """Test getting contents of a non-existent zone returns empty set."""
    zone_manager = ZoneManager()

    contents = zone_manager.get_zone_contents(999)
    assert len(contents) == 0
    assert isinstance(contents, set)


def test_get_zone_enum():
    """Test getting zone enum by zone ID."""
    zone_manager = ZoneManager()

    zone_manager.update_zone_metadata(1, "ZoneType_Battlefield")

    assert zone_manager.get_zone_enum(1) == ZoneType.BATTLEFIELD


def test_get_zone_enum_unknown():
    """Test getting zone enum for unknown zone returns UNKNOWN."""
    zone_manager = ZoneManager()

    assert zone_manager.get_zone_enum(999) == ZoneType.UNKNOWN


def test_clear():
    """Test that clear() resets all zone information."""
    zone_manager = ZoneManager()

    # Add some data
    zone_manager.update_zone_metadata(1, "ZoneType_Hand", 1)
    zone_manager.transfer_card(instance_id=100, new_zone_id=1)

    # Verify data exists
    assert len(zone_manager.zone_id_to_type) > 0
    assert len(zone_manager._zone_objects) > 0

    # Clear and verify everything is empty
    zone_manager.clear()
    assert len(zone_manager.zone_id_to_type) == 0
    assert len(zone_manager.zone_id_to_enum) == 0
    assert len(zone_manager.zone_id_to_owner) == 0
    assert len(zone_manager._zone_objects) == 0


def test_multiple_zones_with_cards():
    """Test managing multiple zones with various cards."""
    zone_manager = ZoneManager()

    # Set up zones
    zone_manager.update_zone_metadata(1, "ZoneType_Hand", 1)
    zone_manager.update_zone_metadata(2, "ZoneType_Battlefield", 1)
    zone_manager.update_zone_metadata(3, "ZoneType_Graveyard", 1)

    # Add cards to different zones
    zone_manager.transfer_card(100, new_zone_id=1)
    zone_manager.transfer_card(101, new_zone_id=1)
    zone_manager.transfer_card(102, new_zone_id=2)
    zone_manager.transfer_card(103, new_zone_id=3)

    # Verify zone contents
    assert len(zone_manager.get_zone_contents(1)) == 2
    assert len(zone_manager.get_zone_contents(2)) == 1
    assert len(zone_manager.get_zone_contents(3)) == 1


def test_clear_zone():
    """Test clearing a specific zone."""
    zone_manager = ZoneManager()

    # Add cards to zone
    zone_manager.transfer_card(100, new_zone_id=1)
    zone_manager.transfer_card(101, new_zone_id=1)
    assert len(zone_manager.get_zone_contents(1)) == 2

    # Clear the zone
    zone_manager.clear_zone(1)
    assert len(zone_manager.get_zone_contents(1)) == 0

    # Clearing non-existent zone should not error
    zone_manager.clear_zone(999)


def test_reconcile_zone_add_cards():
    """Test reconciling zone by adding cards."""
    zone_manager = ZoneManager()

    # Start with empty zone
    assert len(zone_manager.get_zone_contents(1)) == 0

    # Reconcile to have 3 cards
    zone_manager.reconcile_zone(1, {100, 101, 102})
    assert zone_manager.get_zone_contents(1) == {100, 101, 102}


def test_reconcile_zone_remove_cards():
    """Test reconciling zone by removing cards."""
    zone_manager = ZoneManager()

    # Start with 3 cards
    zone_manager.transfer_card(100, new_zone_id=1)
    zone_manager.transfer_card(101, new_zone_id=1)
    zone_manager.transfer_card(102, new_zone_id=1)
    assert len(zone_manager.get_zone_contents(1)) == 3

    # Reconcile to have only 1 card
    zone_manager.reconcile_zone(1, {100})
    assert zone_manager.get_zone_contents(1) == {100}


def test_reconcile_zone_replace_cards():
    """Test reconciling zone by replacing cards."""
    zone_manager = ZoneManager()

    # Start with cards A, B
    zone_manager.transfer_card(100, new_zone_id=1)
    zone_manager.transfer_card(101, new_zone_id=1)
    assert zone_manager.get_zone_contents(1) == {100, 101}

    # Reconcile to have cards B, C (remove A, add C, keep B)
    zone_manager.reconcile_zone(1, {101, 102})
    assert zone_manager.get_zone_contents(1) == {101, 102}


def test_reconcile_zone_to_empty():
    """Test reconciling zone to empty."""
    zone_manager = ZoneManager()

    # Start with 2 cards
    zone_manager.transfer_card(100, new_zone_id=1)
    zone_manager.transfer_card(101, new_zone_id=1)
    assert len(zone_manager.get_zone_contents(1)) == 2

    # Reconcile to empty
    zone_manager.reconcile_zone(1, set())
    assert len(zone_manager.get_zone_contents(1)) == 0
