"""
Decoupled GRE message parser.
"""
import logging
import json
from typing import Any, Dict

from src.core.zone_manager import ZoneManager
from src.core.mtga import MatchScanner, GameObject, PlayerState, GameHistory, get_zone_type, ZoneType
from src.core.monitoring import get_monitor

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
            if obj_data.get("zoneId") is not None and game_obj.zone_id != obj_data.get("zoneId"):
                self.zone_manager.transfer_card(instance_id, obj_data.get("zoneId"), game_obj.zone_id)
                game_obj.zone_id = obj_data.get("zoneId")
                state_changed = True

        return state_changed

    def _parse_zones(self, zones) -> bool:
        state_changed = False
        for zone_obj in zones:
            if not isinstance(zone_obj, dict): continue

            zone_id = zone_obj.get("zoneId")
            zone_type_str = zone_obj.get("type")
            owner_seat_id = zone_obj.get("ownerSeatId")

            if zone_id and zone_type_str:
                self.zone_manager.update_zone_metadata(zone_id, zone_type_str, owner_seat_id)

            object_instance_ids = zone_obj.get("objectInstanceIds")
            if object_instance_ids is not None:
                for card_id in object_instance_ids:
                    if card_id not in self.scanner.game_objects:
                        self.scanner.game_objects[card_id] = GameObject(
                            instance_id=card_id,
                            grp_id=0,
                            zone_id=zone_id,
                            owner_seat_id=owner_seat_id,
                            name=f"Unknown Card {card_id}"
                        )
                        self.zone_manager.transfer_card(card_id, zone_id)
                        state_changed = True
                    else:
                        card = self.scanner.game_objects[card_id]
                        if card.zone_id != zone_id:
                            self.zone_manager.transfer_card(card_id, zone_id, card.zone_id)
                            card.zone_id = zone_id
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
            if "lifeTotal" in player_data and player.life_total != player_data["lifeTotal"]:
                player.life_total = player_data["lifeTotal"]
                state_changed = True
        return state_changed

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

        self.scanner.current_phase = turn_info.get("phase", self.scanner.current_phase)
        self.scanner.active_player_seat = turn_info.get("activePlayer", self.scanner.active_player_seat)
        self.scanner.priority_player_seat = turn_info.get("priorityPlayer", self.scanner.priority_player_seat)
        return state_changed

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
