"""
Decoupled GRE message parser.
"""
import logging
import json
from typing import Any, Dict, TYPE_CHECKING

from src.core.zone_manager import ZoneManager
from src.core.legacy_models import GameObject, PlayerState, GameHistory
from src.core.domain.game_state import ZoneType, get_zone_type
from src.core.monitoring import get_monitor

if TYPE_CHECKING:
    from src.core.mtga import MatchScanner

class GREParser:
    """
    Parses GRE messages and updates game state accordingly.
    This class extracts parsing logic from MatchScanner, acting as a router
    that delegates to other components like ZoneManager.
    """

    def __init__(self, scanner: 'MatchScanner', zone_manager: ZoneManager, card_lookup: Any):
        self.scanner = scanner
        self.zone_manager = zone_manager
        self.card_lookup = card_lookup

    def parse_gre_to_client_event(self, event_data: dict) -> bool:
        if "greToClientEvent" not in event_data: return False
        gre_event = event_data["greToClientEvent"]

        messages = gre_event.get("greToClientMessages", [])
        is_ui_only = all(m.get("type") == "GREMessageType_UIMessage" for m in messages)

        if is_ui_only:
            logging.debug(f"GREToClientEvent received (UI Messages only) - type: {gre_event.get('type', 'N/A')}")
        else:
            logging.info(f"GREToClientEvent received - type: {gre_event.get('type', 'N/A')}")

        if not messages: return False

        if not is_ui_only:
            logging.info(f"Processing {len(messages)} messages")

        state_changed = False
        for message in messages:
            msg_type = message.get("type", "")

            if msg_type == "GREMessageType_UIMessage":
                logging.debug(f"Message type: {msg_type}")
            else:
                logging.info(f"Message type: {msg_type}")

            if "systemSeatIds" in message and self.scanner.local_player_seat_id is None:
                self.scanner.local_player_seat_id = message["systemSeatIds"][0]
                logging.info(f"Set local player seat ID to: {self.scanner.local_player_seat_id} (via systemSeatIds)")

            if msg_type == "GREMessageType_GameStateMessage":
                state_changed |= self._parse_game_state_message(message)
            elif msg_type == "GREMessageType_ActionsAvailableReq":
                logging.info("ActionsAvailableReq - player has priority")
                self.scanner._pending_decision = "actions_available"
                self.scanner._decision_context = {"type": "priority", "actions": message.get("actionsAvailableReq", {})}
                state_changed = True
            elif msg_type == "GREMessageType_Annotation":
                state_changed |= self._parse_annotations(message)

        return state_changed

    def _parse_game_state_message(self, message: dict) -> bool:
        monitor = get_monitor()
        with monitor.measure("gre_parser.parse_game_state_message"):
            if "gameStateMessage" not in message:
                return False

            game_state_raw = message["gameStateMessage"]
            game_state = None

            if isinstance(game_state_raw, str):
                try:
                    clean_json = game_state_raw.strip()
                    if "{" in clean_json:
                        clean_json = clean_json[clean_json.find("{"):]
                    game_state = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse GameStateMessage JSON: {e}")
                    return False
            elif isinstance(game_state_raw, dict):
                game_state = game_state_raw
            else:
                logging.warning(f"Unknown GameStateMessage format: {type(game_state_raw)}")
                return False

            if not game_state:
                return False

            logging.info(f"GameStateMessage received")
            state_changed = False

            if "gameObjects" in game_state:
                state_changed |= self._parse_game_objects(game_state["gameObjects"])
            if "zones" in game_state:
                state_changed |= self._parse_zones(game_state["zones"])
            if "annotations" in game_state:
                state_changed |= self._parse_annotations(game_state["annotations"])
            if "players" in game_state:
                state_changed |= self._parse_players(game_state["players"])
            if "turnInfo" in game_state:
                state_changed |= self._parse_turn_info(game_state["turnInfo"])
            if "actions" in game_state:
                state_changed |= self._parse_actions(game_state["actions"])

            return state_changed

    def _parse_game_objects(self, game_objects: list) -> bool:
        state_changed = False
        for obj_data in game_objects:
            instance_id = obj_data.get("instanceId")
            if not instance_id: continue

            if instance_id not in self.scanner.game_objects:
                self.scanner.game_objects[instance_id] = GameObject(
                    instance_id=instance_id,
                    grp_id=obj_data.get("grpId"),
                    zone_id=obj_data.get("zoneId"),
                    owner_seat_id=obj_data.get("ownerSeatId")
                )
                state_changed = True

            # Update existing object
            game_obj = self.scanner.game_objects[instance_id]

            # CRITICAL FIX: Upgrade placeholder with real grpId
            # When cards are first seen in zones, they're created as placeholders with grp_id=0
            # Later gameObjects messages provide the real grp_id - we must upgrade them here
            grp_id = obj_data.get("grpId")
            if game_obj.grp_id == 0 and grp_id and grp_id != 0:
                logging.info(f"    -> Upgrading placeholder {instance_id} with real grpId {grp_id}")
                game_obj.grp_id = grp_id
                # Resolve card name and metadata now that we have the real grpId
                self.scanner._resolve_card_metadata(game_obj)
                state_changed = True

            # Update zone if changed
            if obj_data.get("zoneId") is not None and game_obj.zone_id != obj_data.get("zoneId"):
                self.zone_manager.transfer_card(instance_id, obj_data.get("zoneId"), game_obj.zone_id)
                game_obj.zone_id = obj_data.get("zoneId")
                state_changed = True

            # Update tap state if changed
            # Check for explicit isTapped field or overlayGrpId (which indicates tapped state)
            is_tapped = obj_data.get("isTapped", False)
            overlay_grp_id = obj_data.get("overlayGrpId")
            if overlay_grp_id is not None and overlay_grp_id > 0:
                is_tapped = True

            if game_obj.is_tapped != is_tapped:
                game_obj.is_tapped = is_tapped
                logging.debug(f"  Card {instance_id} ({game_obj.name}) tap state changed to: {is_tapped}")
                state_changed = True

        return state_changed

    def _parse_zones(self, zones) -> bool:
        """
        Parse zone information from game state.

        CRITICAL: Zones are AUTHORITATIVE - when a zone is reported,
        we reconcile it to exactly match the reported contents.
        Missing objectInstanceIds is treated as an empty zone.
        """
        state_changed = False
        for zone_obj in zones:
            if not isinstance(zone_obj, dict): continue

            zone_id = zone_obj.get("zoneId")
            zone_type_str = zone_obj.get("type")
            owner_seat_id = zone_obj.get("ownerSeatId")

            if zone_id and zone_type_str:
                self.zone_manager.update_zone_metadata(zone_id, zone_type_str, owner_seat_id)

            # CRITICAL FIX: Treat missing objectInstanceIds as empty list
            # Zone reports are authoritative - if a zone is reported, we must reconcile it
            object_instance_ids = zone_obj.get("objectInstanceIds")
            if object_instance_ids is None:
                object_instance_ids = []

            # Create GameObject entries for any cards we haven't seen before
            for card_id in object_instance_ids:
                if card_id not in self.scanner.game_objects:
                    self.scanner.game_objects[card_id] = GameObject(
                        instance_id=card_id,
                        grp_id=0,
                        zone_id=zone_id,
                        owner_seat_id=owner_seat_id,
                        name=f"Unknown Card {card_id}"
                    )
                    state_changed = True
                else:
                    # Update zone_id in GameObject if it changed
                    card = self.scanner.game_objects[card_id]
                    if card.zone_id != zone_id:
                        card.zone_id = zone_id
                        state_changed = True

            # CRITICAL FIX: Reconcile zone contents authoritatively
            # This ensures the zone manager exactly matches what MTGA reported
            old_contents = self.zone_manager.get_zone_contents(zone_id)
            new_contents = set(object_instance_ids)

            if old_contents != new_contents:
                self.zone_manager.reconcile_zone(zone_id, new_contents)
                state_changed = True

        return state_changed

    def _parse_players(self, players: list) -> bool:
        state_changed = False
        for player_data in players:
            seat_id = player_data.get("systemSeatNumber")
            if not seat_id: continue

            if seat_id not in self.scanner.players:
                self.scanner.players[seat_id] = PlayerState(seat_id=seat_id)
                state_changed = True

            player = self.scanner.players[seat_id]

            # Parse life total
            if "lifeTotal" in player_data and player.life_total != player_data["lifeTotal"]:
                player.life_total = player_data["lifeTotal"]
                state_changed = True

            # Parse mana pool - check for pendingMana or manaPool fields
            mana_data = player_data.get("pendingMana") or player_data.get("manaPool")
            if mana_data:
                new_mana_pool = self._parse_mana_pool(mana_data)
                if new_mana_pool != player.mana_pool:
                    player.mana_pool = new_mana_pool
                    state_changed = True
                    logging.info(f"  Player {seat_id} mana pool updated: {new_mana_pool}")

            # Parse energy
            if "energy" in player_data and player.energy != player_data["energy"]:
                player.energy = player_data["energy"]
                state_changed = True

        return state_changed

    def _parse_mana_pool(self, mana_data) -> dict:
        """
        Parse mana pool data from GRE message.
        Mana data can be in different formats:
        - List of dicts with "color" and "count": [{"color": ["ManaColor_Red"], "count": 2}]
        - Direct dict: {"Red": 2, "Blue": 1}
        """
        mana_pool = {}

        if isinstance(mana_data, list):
            # Format: [{"color": ["ManaColor_Red"], "count": 2}]
            for mana_entry in mana_data:
                colors = mana_entry.get("color", [])
                count = mana_entry.get("count", 0)
                for color_str in colors:
                    # Convert "ManaColor_Red" to "R"
                    color_code = self._mana_color_to_code(color_str)
                    if color_code:
                        mana_pool[color_code] = mana_pool.get(color_code, 0) + count
        elif isinstance(mana_data, dict):
            # Format: {"Red": 2, "Blue": 1} or similar
            for color_key, count in mana_data.items():
                color_code = self._mana_color_to_code(color_key)
                if color_code and count > 0:
                    mana_pool[color_code] = count

        return mana_pool

    def _mana_color_to_code(self, color_str: str) -> str:
        """Convert MTGA color string to single-letter code."""
        color_map = {
            "ManaColor_White": "W",
            "ManaColor_Blue": "U",
            "ManaColor_Black": "B",
            "ManaColor_Red": "R",
            "ManaColor_Green": "G",
            "ManaColor_Colorless": "C",
            "ManaColor_Generic": "C",
            "White": "W",
            "Blue": "U",
            "Black": "B",
            "Red": "R",
            "Green": "G",
            "Colorless": "C",
            "Generic": "C",
        }
        return color_map.get(color_str, "")

    def _parse_turn_info(self, turn_info: dict) -> bool:
        state_changed = False
        new_turn = turn_info.get("turnNumber", 0)
        if new_turn == 1 and self.scanner.current_turn > 1:
            self.scanner.reset_match_state()
            state_changed = True

        if self.scanner.current_turn != new_turn:
            self.scanner.current_turn = new_turn
            self.scanner.game_history = GameHistory(turn_number=new_turn)
            state_changed = True

        # Check for phase change
        new_phase = turn_info.get("phase")
        if new_phase and new_phase != self.scanner.current_phase:
            old_phase = self.scanner.current_phase
            self.scanner.current_phase = new_phase

            # Clear mana pools on phase/step change (mana empties between steps/phases)
            # Exception: mana doesn't empty during main phase steps or combat steps
            should_clear_mana = self._should_clear_mana_on_phase_change(old_phase, new_phase)
            if should_clear_mana:
                for player in self.scanner.players.values():
                    if player.mana_pool:
                        logging.debug(f"  Clearing mana pool for player {player.seat_id} on phase change {old_phase} -> {new_phase}")
                        player.mana_pool = {}
                        state_changed = True
        else:
            self.scanner.current_phase = turn_info.get("phase", self.scanner.current_phase)

        self.scanner.active_player_seat = turn_info.get("activePlayer", self.scanner.active_player_seat)
        self.scanner.priority_player_seat = turn_info.get("priorityPlayer", self.scanner.priority_player_seat)
        return state_changed

    def _should_clear_mana_on_phase_change(self, old_phase: str, new_phase: str) -> bool:
        """
        Determine if mana pool should be cleared on phase change.
        By MTG rules, mana empties at end of each step and phase.

        However, for simplicity and to avoid clearing mana when it shouldn't be,
        we only clear on major phase transitions.
        """
        # List of major phase transitions where mana should definitely be cleared
        major_phases = ["Phase_Beginning", "Phase_Main1", "Phase_Combat", "Phase_Main2", "Phase_Ending"]

        # If transitioning between major phases, clear mana
        if old_phase in major_phases and new_phase in major_phases and old_phase != new_phase:
            return True

        return False

    def _parse_annotations(self, data) -> bool:
        annotations_list = []
        if isinstance(data, list):
            annotations_list = data
        elif isinstance(data, dict):
            annotations_list = data.get("annotations", [])

        if not annotations_list:
            return False

        state_changed = False
        for annotation in annotations_list:
            ann_type = annotation.get("type", [])
            affected_ids = annotation.get("affectedIds", [])
            details = annotation.get("details", [])

            if "AnnotationType_ZoneTransfer" in ann_type:
                zone_dest = None
                for detail in details:
                    if detail.get("key") == "zone_dest":
                        zone_dest = detail.get("valueInt32", [None])[0]

                for instance_id in affected_ids:
                    if instance_id in self.scanner.game_objects:
                        obj = self.scanner.game_objects[instance_id]
                        if zone_dest is not None and obj.zone_id != zone_dest:
                            self.zone_manager.transfer_card(instance_id, zone_dest, obj.zone_id)
                            obj.zone_id = zone_dest
                            state_changed = True
        return state_changed

    def _parse_actions(self, actions: list) -> bool:
        """
        Parse actions from game state message.
        Actions can include mana generation (ActionType_Activate_Mana) which helps track mana pool.

        NOTE: This is a partial implementation. MTGA may not always send complete mana pool state,
        and mana tracking from actions may be unreliable because:
        1. Mana empties at end of steps/phases (not always sent)
        2. Actions may be consumed before we see them
        3. The actual mana pool state should ideally come from the players array

        This method primarily logs mana-related actions for debugging purposes.
        """
        if not actions:
            return False

        state_changed = False
        for action_data in actions:
            if not isinstance(action_data, dict):
                continue

            seat_id = action_data.get("seatId")
            action = action_data.get("action", {})
            action_type = action.get("actionType", "")

            # Log mana activation actions for debugging
            if "ActionType_Activate_Mana" in action_type:
                instance_id = action.get("instanceId")
                ability_grp_id = action.get("abilityGrpId")

                # Try to identify what kind of mana was produced
                # Note: We don't actually know the colors without more context
                logging.debug(f"  Mana action: Seat {seat_id} activated mana ability {ability_grp_id} on object {instance_id}")
                # We could potentially track this, but without color info it's not very useful

        return state_changed
