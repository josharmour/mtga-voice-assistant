"""
Centralized zone management for MTG game state.
"""
import logging
from typing import Dict, Set, Optional

from src.core.domain.game_state import ZoneType, get_zone_type

class ZoneManager:
    """
    Manages all game zones and card movements between them.
    This class centralizes zone tracking logic previously scattered
    across MatchScanner._parse_zones and _parse_annotations.
    """

    def __init__(self):
        # Zone metadata
        self.zone_id_to_type: Dict[int, str] = {}
        self.zone_id_to_enum: Dict[int, ZoneType] = {}
        self.zone_id_to_owner: Dict[int, int] = {}

        # Zone contents (zone_id -> set of instance_ids)
        self._zone_objects: Dict[int, Set[int]] = {}

    def clear(self):
        """Reset all zone information for a new match."""
        self.zone_id_to_type.clear()
        self.zone_id_to_enum.clear()
        self.zone_id_to_owner.clear()
        self._zone_objects.clear()
        logging.info("ZoneManager cleared for new match.")

    def clear_zone(self, zone_id: int):
        """
        Clear all cards from a specific zone.
        Used when a zone is explicitly reported as empty.
        """
        if zone_id in self._zone_objects:
            del self._zone_objects[zone_id]
            logging.debug(f"Zone {zone_id} cleared")

    def reconcile_zone(self, zone_id: int, card_ids: Set[int]):
        """
        Reconcile a zone to exactly match the provided set of card IDs.
        This is the authoritative zone update - any cards not in card_ids
        are removed, and any new cards are added.

        Args:
            zone_id: The zone to reconcile
            card_ids: The complete set of instance IDs that should be in this zone
        """
        old_cards = self._zone_objects.get(zone_id, set())
        new_cards = card_ids

        # Find differences
        removed_cards = old_cards - new_cards
        added_cards = new_cards - old_cards

        if removed_cards or added_cards:
            # Update the zone
            if new_cards:
                self._zone_objects[zone_id] = set(new_cards)
            else:
                # Zone is now empty
                if zone_id in self._zone_objects:
                    del self._zone_objects[zone_id]

            if removed_cards:
                logging.debug(f"Zone {zone_id}: removed {len(removed_cards)} cards: {removed_cards}")
            if added_cards:
                logging.debug(f"Zone {zone_id}: added {len(added_cards)} cards: {added_cards}")

    def update_zone_metadata(self, zone_id: int, zone_type_str: str, owner_seat_id: Optional[int] = None):
        """
        Update metadata for a given zone (type, owner).
        """
        self.zone_id_to_type[zone_id] = zone_type_str
        self.zone_id_to_enum[zone_id] = get_zone_type(zone_type_str)
        if owner_seat_id:
            self.zone_id_to_owner[zone_id] = owner_seat_id

    def transfer_card(self, instance_id: int, new_zone_id: int, old_zone_id: Optional[int] = None):
        """
        Move a card from one zone to another.
        This is the primary method for handling card movement from annotations.
        """
        # Remove from old zone if it exists
        if old_zone_id is not None and old_zone_id in self._zone_objects:
            self._zone_objects[old_zone_id].discard(instance_id)
            if not self._zone_objects[old_zone_id]:
                del self._zone_objects[old_zone_id]

        # Add to new zone
        if new_zone_id not in self._zone_objects:
            self._zone_objects[new_zone_id] = set()
        self._zone_objects[new_zone_id].add(instance_id)

    def get_zone_contents(self, zone_id: int) -> Set[int]:
        """Get the set of instance IDs in a given zone."""
        return self._zone_objects.get(zone_id, set())

    def get_zone_enum(self, zone_id: int) -> ZoneType:
        """Get the ZoneType enum for a given zone ID."""
        return self.zone_id_to_enum.get(zone_id, ZoneType.UNKNOWN)
