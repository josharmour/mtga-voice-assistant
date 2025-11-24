
import logging
import os
import time
from typing import Callable
import dataclasses
import json
import re
from typing import Dict, List, Optional, Callable

# Content of src/log_parser.py
class LogFollower:
    """Follows the Arena Player.log file and yields new lines as they're added."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self.inode = None
        self.offset = 0
        self.first_open = True  # Track if this is first time opening
        self.is_caught_up = False # Track if we have processed the backlog

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
        # Debug output disabled for cleaner console
        logging.info(f"LogFollower.follow() started! Watching: {self.log_path}")
        while True:
            try:
                current_inode = None
                try:
                    current_inode = os.stat(self.log_path).st_ino
                except FileNotFoundError:
                    logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                    time.sleep(5)
                    continue # Skip to next iteration if file not found
                except Exception as e:
                    logging.error(f"Error getting inode for {self.log_path}: {e}")
                    time.sleep(1)
                    continue

                logging.debug(f"LogFollower: current_inode={current_inode}, self.inode={self.inode}")
                if self.inode is None or self.inode != current_inode:
                    # Debug: Opening log file (inode changed or first open)
                    if self.file:
                        self.file.close()
                    self.file = open(self.log_path, 'r', encoding='utf-8', errors='replace')
                    self.inode = current_inode
                    # Debug: File opened successfully

                    # On first open, seek to end to ignore old matches
                    # On log rotation, start from beginning of new file
                    if self.first_open:
                        # PERFORMANCE FIX: Read from beginning to get card definitions,
                        # but we won't trigger expensive board state calculations until caught up
                        # (handled in app.py callback with is_caught_up flag)
                        self.file.seek(0, 0)
                        self.offset = 0
                        self.first_open = False
                        logging.info("Log file opened - starting from beginning to rebuild state.")
                    else:
                        # Log rotation - start from beginning of new file
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning of new file.")

                self.file.seek(self.offset)
                line_count = 0
                while True:
                    line = self.file.readline()
                    if not line:
                        self.is_caught_up = True
                        break
                    line_count += 1
                    self.offset = self.file.tell()
                    stripped_line = line.strip()

                    # Debug: draft-related lines (disabled for cleaner console)
                    # if "Draft" in stripped_line or "BotDraft" in stripped_line:
                    #     print(f"[DEBUG] Draft-related line: {stripped_line[:150]}")

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Read line: {stripped_line[:100]}...") # Log first 100 chars to avoid spam
                    callback(stripped_line)
                    
                    # Yield thread execution to keep UI responsive during heavy parsing
                    # This is critical when reading large log files on startup
                    if line_count % 200 == 0:
                        time.sleep(0.001)

                # if line_count > 0:
                #     print(f"[DEBUG] Processed {line_count} lines")
                time.sleep(0.1)  # Poll every 100ms
            except FileNotFoundError:
                logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error following log file: {e}")
                time.sleep(1)

    def close(self):
        if self.file:
            self.file.close()

# Content of src/game_state.py
@dataclasses.dataclass
class GameObject:
    instance_id: int
    grp_id: int
    zone_id: int
    owner_seat_id: int
    name: str = ""
    color_identity: str = ""  # Card color(s): W, U, B, R, G, or combinations
    power: Optional[int] = None
    toughness: Optional[int] = None
    is_tapped: bool = False
    is_attacking: bool = False
    summoning_sick: bool = False
    counters: Dict[str, int] = dataclasses.field(default_factory=dict)  # {counter_type: count}
    attached_to: Optional[int] = None  # Instance ID of attached permanent
    visibility: str = "public"  # "public", "private", "revealed"
    type_line: str = ""  # Added to prevent AttributeError in some contexts

@dataclasses.dataclass
class PlayerState:
    seat_id: int
    life_total: int = 20
    hand_count: int = 0
    has_priority: bool = False
    mana_pool: Dict[str, int] = dataclasses.field(default_factory=dict)  # {"W": 2, "U": 1, etc.}
    energy: int = 0

@dataclasses.dataclass
class GameHistory:
    """Tracks important events from current turn for tactical context"""
    turn_number: int = 0
    cards_played_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    attackers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    blockers_this_turn: List[GameObject] = dataclasses.field(default_factory=list)
    damage_dealt: Dict[int, int] = dataclasses.field(default_factory=dict)
    died_this_turn: List[str] = dataclasses.field(default_factory=list)
    lands_played_this_turn: int = 0

    # Current combat state (during combat phase)
    current_attackers: List[int] = dataclasses.field(default_factory=list)  # Instance IDs
    current_blockers: Dict[int, int] = dataclasses.field(default_factory=dict)  # {attacker_id: blocker_id}
    combat_damage_assignments: Dict[int, int] = dataclasses.field(default_factory=dict)  # {instance_id: damage}

@dataclasses.dataclass
class BoardState:
    your_seat_id: int
    opponent_seat_id: int

    # Life totals
    your_life: int = 20
    opponent_life: int = 20

    # Mana and energy
    your_mana_pool: Dict[str, int] = dataclasses.field(default_factory=dict)
    your_energy: int = 0
    opponent_energy: int = 0

    # Zone: Hand
    your_hand_count: int = 0
    your_hand: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_hand_count: int = 0

    # Zone: Battlefield
    your_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_battlefield: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Graveyard
    your_graveyard: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_graveyard: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Exile
    your_exile: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_exile: List[GameObject] = dataclasses.field(default_factory=list)

    # Zone: Library
    your_library_count: int = 0
    opponent_library_count: int = 0

    # Zone: Stack
    stack: List[GameObject] = dataclasses.field(default_factory=list)

    # Turn tracking
    current_turn: int = 0
    current_phase: str = ""
    is_your_turn: bool = False
    has_priority: bool = False

    # Game history
    history: Optional[GameHistory] = None

    # Deck tracking (Phase 3)
    your_decklist: Dict[str, int] = dataclasses.field(default_factory=dict)  # {card_name: count}
    your_deck_remaining: int = 0  # Cards left in library

    # Known cards tracking (Phase 3)
    library_top_known: List[str] = dataclasses.field(default_factory=list)  # Card names on top of library
    scry_info: Optional[str] = None  # "Top 2: Lightning Bolt, Forest"

    # Mulligan phase tracking
    in_mulligan_phase: bool = False
    game_stage: str = ""  # "GameStage_Start" or "GameStage_Play"

import datetime

class MatchScanner:
    """
    Parses GRE messages to track game state.
    """

    def __init__(self, card_lookup=None):
        self.game_objects: Dict[int, GameObject] = {}
        self.players: Dict[int, PlayerState] = {}
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat: Optional[int] = None
        self.priority_player_seat: Optional[int] = None
        self.local_player_seat_id: Optional[int] = None
        self.zone_type_to_ids: Dict[str, int] = {}
        self.observed_zone_ids: set = set()
        self.zone_id_to_type: Dict[int, str] = {}
        self.game_history: GameHistory = GameHistory()
        self.last_event_timestamp: Optional[datetime.datetime] = None # Track when the last event happened

        # Phase 3: Deck tracking
        self.submitted_decklist: Dict[int, int] = {}
        self.cards_seen: Dict[int, int] = {}

        # Phase 3: Known library cards
        self.library_top_known: List[str] = []
        self.scry_info: Optional[str] = None

        # Mulligan tracking
        self.game_stage: str = ""
        self.in_mulligan_phase: bool = False

        # P0 Performance: Zone-based object caching
        # Maps zone_id -> set of instance_ids in that zone
        # This eliminates O(n) iteration in get_current_board_state()
        self._zone_objects: Dict[int, set] = {}

        # P0 Performance: Card name resolution
        # Resolve card names once on object creation, not repeatedly
        self.card_lookup = card_lookup

    def parse_timestamp(self, line: str):
        """Extract timestamp from log line if present."""
        # Format: [UnityCrossThreadLogger]11/22/2025 10:20:17 PM: ...
        try:
            if "[UnityCrossThreadLogger]" in line:
                # Find the timestamp part
                start = line.find("]") + 1
                end = line.find(": Match") 
                if end == -1: end = line.find(": GRE")
                if end == -1: end = line.find(": ")
                
                if start > 0 and end > start:
                    ts_str = line[start:end].strip()
                    # Try parsing "11/22/2025 10:20:17 PM"
                    self.last_event_timestamp = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
        except Exception:
            pass # Ignore parsing errors, timestamp is optional hint

    def _update_object_zone(self, instance_id: int, old_zone_id: Optional[int], new_zone_id: int):
        """
        Update zone membership cache for an object.

        This maintains the _zone_objects index which maps zone_id -> set of instance_ids.
        This cache eliminates O(n) iteration in get_current_board_state().

        Args:
            instance_id: The instance ID of the game object
            old_zone_id: The previous zone ID (None if object is new)
            new_zone_id: The new zone ID
        """
        # Remove from old zone if it exists
        if old_zone_id is not None and old_zone_id in self._zone_objects:
            self._zone_objects[old_zone_id].discard(instance_id)
            # Clean up empty zone sets to save memory
            if not self._zone_objects[old_zone_id]:
                del self._zone_objects[old_zone_id]

        # Add to new zone
        if new_zone_id not in self._zone_objects:
            self._zone_objects[new_zone_id] = set()
        self._zone_objects[new_zone_id].add(instance_id)

    def reset_match_state(self):
        """Clear all game state when a new match starts"""
        logging.info("🔄 NEW MATCH DETECTED - Clearing all previous match state")
        self.game_objects.clear()
        self.players.clear()
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat = None
        self.priority_player_seat = None
        # Keep local_player_seat_id - it persists across matches
        self.zone_type_to_ids.clear()
        self.observed_zone_ids.clear()
        self.zone_id_to_type.clear()
        self.game_history = GameHistory()
        self.cards_seen.clear()
        self.library_top_known.clear()
        self.scry_info = None
        self.game_stage = ""
        self.in_mulligan_phase = False
        self._zone_objects.clear()
        # Note: submitted_decklist is set by _parse_deck_submission, so don't clear it here

    def _resolve_card_metadata(self, game_obj: GameObject) -> None:
        """
        P0 Performance optimization: Resolve card name and color once on object creation.
        This eliminates the need to iterate all objects in get_current_board_state().
        """
        if not self.card_lookup or game_obj.grp_id == 0:
            return

        # Resolve name if not already set or is a placeholder
        if not game_obj.name or game_obj.name.startswith("Unknown Card"):
            game_obj.name = self.card_lookup.get_card_name(game_obj.grp_id)

        # Resolve color identity if not already set
        if not game_obj.color_identity:
            try:
                card_data = self.card_lookup.get_card_data(game_obj.grp_id)
                if card_data:
                    game_obj.color_identity = card_data.get("color_identity", "")
            except Exception as e:
                logging.debug(f"Could not fetch color for {game_obj.grp_id}: {e}")

    def parse_gre_to_client_event(self, event_data: dict) -> bool:
        if "greToClientEvent" not in event_data: return False
        gre_event = event_data["greToClientEvent"]
        logging.info(f"GREToClientEvent received - type: {gre_event.get('type', 'N/A')}")
        if "greToClientMessages" not in gre_event: return False
        logging.info(f"Processing {len(gre_event['greToClientMessages'])} messages")
        state_changed = False
        for message in gre_event["greToClientMessages"]:
            msg_type = message.get("type", "")
            logging.info(f"Message type: {msg_type}")
            if "systemSeatIds" in message and not self.local_player_seat_id:
                self.local_player_seat_id = message["systemSeatIds"][0]
                logging.info(f"Set local player seat ID to: {self.local_player_seat_id}")

            if msg_type == "GREMessageType_GameStateMessage":
                state_changed |= self._parse_game_state_message(message)
            elif msg_type == "GREMessageType_ActionsAvailableReq":
                logging.info("ActionsAvailableReq - player has priority")
                state_changed = True # This is a key decision point
            elif msg_type == "GREMessageType_Annotation":
                state_changed |= self._parse_annotations(message)

        return state_changed

    def _parse_game_state_message(self, message: dict) -> bool:
        """
        Parse the massive GameStateMessage.
        This is the primary source of truth for the board state.
        """
        if "gameStateMessage" not in message:
            return False

        # Handle potential JSON string within the message
        game_state_raw = message["gameStateMessage"]
        game_state = None

        if isinstance(game_state_raw, str):
            try:
                # Clean up potential leading/trailing garbage characters if any
                clean_json = game_state_raw.strip()
                # Some logs have weird prefixes like "ClientToGREMessage "
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
        logging.info(f"Game State keys: {list(game_state.keys()) if game_state else 'empty'}")
        state_changed = False

        # Parse game stage (for mulligan detection)
        if "gameInfo" in game_state:
            game_info = game_state["gameInfo"]
            old_stage = self.game_stage
            self.game_stage = game_info.get("stage", self.game_stage)
            if old_stage != self.game_stage:
                logging.info(f"🎮 Game stage changed: {old_stage} → {self.game_stage}")
                state_changed = True

        # Handle deleted objects FIRST before processing new ones
        if "diffDeletedInstanceIds" in game_state:
            deleted_ids = game_state["diffDeletedInstanceIds"]
            logging.debug(f"Removing {len(deleted_ids)} deleted objects")
            for obj_id in deleted_ids:
                if obj_id in self.game_objects:
                    # P0 Performance: Remove from zone cache before deleting
                    obj = self.game_objects[obj_id]
                    old_zone_id = obj.zone_id
                    if old_zone_id in self._zone_objects:
                        self._zone_objects[old_zone_id].discard(obj_id)
                        # Clean up empty zone sets to save memory
                        if not self._zone_objects[old_zone_id]:
                            del self._zone_objects[old_zone_id]

                    del self.game_objects[obj_id]
                    state_changed = True

        if "gameObjects" in game_state:
            logging.info(f"Found gameObjects with {len(game_state['gameObjects'])} items")
            state_changed |= self._parse_game_objects(game_state["gameObjects"])
        else:
            logging.info("No gameObjects in game state message")
        if "zones" in game_state:
            logging.info(f"Found zones - THIS IS WHERE CARDS ARE! {type(game_state['zones'])}")
            state_changed |= self._parse_zones(game_state["zones"])
        else:
            logging.info("No zones in game state message")
        if "players" in game_state:
            logging.info(f"Found players with {len(game_state['players'])} items")
            state_changed |= self._parse_players(game_state["players"])
        else:
            logging.info("No players in game state message")
        if "turnInfo" in game_state:
            logging.info(f"Found turnInfo: {game_state['turnInfo']}")
            
            # Detect new match: if turn number goes from high to 1, it's a new game
            new_turn = game_state['turnInfo'].get('turnNumber', 0)
            if new_turn == 1 and self.current_turn is not None and self.current_turn > 1:
                logging.info(f"🔄 NEW MATCH DETECTED (turn reset from {self.current_turn} to 1)")
                self.reset_match_state()
            
            state_changed |= self._parse_turn_info(game_state["turnInfo"])
        else:
            logging.info("No turnInfo in game state message")
        return state_changed

    def _parse_game_objects(self, game_objects: list) -> bool:
        state_changed = False
        logging.info(f"Parsing {len(game_objects)} game objects")
        for obj_data in game_objects:
            instance_id = obj_data.get("instanceId")
            if not instance_id: continue

            zone_id = obj_data.get("zoneId")
            grp_id = obj_data.get("grpId")
            owner_seat_id = obj_data.get("ownerSeatId")

            # Skip objects without valid grpId (e.g., tokens, placeholder objects)
            if not grp_id or grp_id == 0:
                logging.debug(f"  Skipping GameObject with invalid grpId: instanceId={instance_id}, grpId={grp_id}")
                continue

            if zone_id is not None:
                self.observed_zone_ids.add(zone_id)

            # Parse new tactical fields
            is_tapped = obj_data.get("isTapped", False)
            is_attacking = obj_data.get("isAttacking", False)
            summoning_sick = obj_data.get("summoningSickness", False)
            counters = obj_data.get("counters", {})
            attached_to = obj_data.get("attachedTo")
            visibility = obj_data.get("visibility", "public")

            # Extract power value (can be int or {'value': int})
            power = obj_data.get("power")
            if isinstance(power, dict):
                power = power.get("value")

            # Extract toughness value (can be int or {'value': int})
            toughness = obj_data.get("toughness")
            if isinstance(toughness, dict):
                toughness = toughness.get("value")

            logging.debug(f"  GameObject: instanceId={instance_id}, grpId={grp_id}, zoneId={zone_id}, ownerSeatId={owner_seat_id}, tapped={is_tapped}")

            # CRITICAL FIX: Ensure player exists if we see their cards
            # This handles cases where we join mid-match and missed the player definition
            if owner_seat_id and owner_seat_id not in self.players:
                logging.info(f"Inferred existence of Player {owner_seat_id} from object {instance_id}")
                self.players[owner_seat_id] = PlayerState(seat_id=owner_seat_id)

            if instance_id not in self.game_objects:
                self.game_objects[instance_id] = GameObject(
                    instance_id=instance_id,
                    grp_id=grp_id,
                    zone_id=zone_id,
                    owner_seat_id=owner_seat_id,
                    is_tapped=is_tapped,
                    is_attacking=is_attacking,
                    summoning_sick=summoning_sick,
                    counters=counters if isinstance(counters, dict) else {},
                    attached_to=attached_to,
                    visibility=visibility,
                    power=power,
                    toughness=toughness
                )
                # P0 Performance: Resolve card name and color once on creation
                self._resolve_card_metadata(self.game_objects[instance_id])
                self._update_object_zone(instance_id, None, zone_id)
                logging.info(f"    -> Created new GameObject")
                state_changed = True
            else:
                # Update existing object
                game_obj = self.game_objects[instance_id]

                # UPGRADE PLACEHOLDER: If we have a real grpId and the object was a placeholder (grpId=0)
                if game_obj.grp_id == 0 and grp_id and grp_id != 0:
                    logging.info(f"    -> Upgrading placeholder {instance_id} with real grpId {grp_id}")
                    game_obj.grp_id = grp_id
                    # P0 Performance: Resolve card name and color once on upgrade
                    self._resolve_card_metadata(game_obj)
                    state_changed = True

                if zone_id is not None and game_obj.zone_id != zone_id:
                    logging.info(f"    -> Zone changed from {game_obj.zone_id} to {zone_id}")

                    # P0 Performance: Update zone cache when zone changes
                    old_zone_id = game_obj.zone_id
                    self._update_object_zone(instance_id, old_zone_id, zone_id)

                    game_obj.zone_id = zone_id
                    state_changed = True

                # Update tactical state
                if game_obj.is_tapped != is_tapped:
                    game_obj.is_tapped = is_tapped
                    state_changed = True

                # Track combat state changes
                if game_obj.is_attacking != is_attacking:
                    game_obj.is_attacking = is_attacking
                    if is_attacking:
                        # Creature declared as attacker
                        if instance_id not in self.game_history.current_attackers:
                            self.game_history.current_attackers.append(instance_id)
                            logging.info(f"⚔️ Creature {instance_id} declared as attacker")
                    else:
                        # Creature no longer attacking (combat ended or removed)
                        if instance_id in self.game_history.current_attackers:
                            self.game_history.current_attackers.remove(instance_id)
                    state_changed = True

                if game_obj.summoning_sick != summoning_sick:
                    game_obj.summoning_sick = summoning_sick
                    state_changed = True
                if counters and game_obj.counters != counters:
                    game_obj.counters = counters if isinstance(counters, dict) else {}
                    state_changed = True
                if attached_to and game_obj.attached_to != attached_to:
                    game_obj.attached_to = attached_to
                    state_changed = True
                if power is not None and game_obj.power != power:
                    logging.info(f"💪 Power change for {game_obj.name} ({instance_id}): {game_obj.power} -> {power}")
                    game_obj.power = power
                    state_changed = True
                if toughness is not None and game_obj.toughness != toughness:
                    logging.info(f"🛡️ Toughness change for {game_obj.name} ({instance_id}): {game_obj.toughness} -> {toughness}")
                    game_obj.toughness = toughness
                    state_changed = True

        return state_changed
        
    def _parse_zones(self, zones) -> bool:
        """Parse the zones structure which contains cards in hand, battlefield, etc."""
        state_changed = False

        if not isinstance(zones, list):
            logging.debug(f"Zones is not a list as expected. Type: {type(zones)}")
            return False

        for idx, zone_obj in enumerate(zones):
            if not isinstance(zone_obj, dict):
                continue

            zone_type_str = zone_obj.get("type")  # e.g., "ZoneType_Hand"
            zone_id = zone_obj.get("zoneId")
            owner_seat_id = zone_obj.get("ownerSeatId")
            object_instance_ids = zone_obj.get("objectInstanceIds", [])

            if zone_type_str and zone_id:
                # Map zone type string to zone ID
                if zone_type_str not in self.zone_type_to_ids or self.zone_type_to_ids[zone_type_str] != zone_id:
                    logging.info(f"Zone mapping found: {zone_type_str} -> zoneId {zone_id} (owner: {owner_seat_id})")
                
                self.zone_type_to_ids[zone_type_str] = zone_id
                self.zone_id_to_type[zone_id] = zone_type_str
                
                # Update all cards in this zone with the correct zone ID
                for card_id in object_instance_ids:
                    if card_id not in self.game_objects:
                        # If a card is in a zone but not yet in game_objects, create a minimal placeholder.
                        self.game_objects[card_id] = GameObject(
                            instance_id=card_id,
                            grp_id=0, # Placeholder grpId
                            zone_id=zone_id,
                            owner_seat_id=owner_seat_id,
                            name=f"Unknown Card {card_id}", # Temporary name
                            visibility="private" if "Hand" in zone_type_str or "Library" in zone_type_str else "public"
                        )
                        logging.debug(f"    -> Created minimal placeholder GameObject {card_id} for zone {zone_id} ({zone_type_str})")

                        # P0 Performance: Update zone cache for new placeholder
                        self._update_object_zone(card_id, None, zone_id)

                        state_changed = True

                    card = self.game_objects[card_id]
                    if card.zone_id != zone_id:
                        logging.debug(f"Updating card {card_id} zone from {card.zone_id} to {zone_id} ({zone_type_str})")

                        # P0 Performance: Update zone cache when zone changes
                        old_zone_id = card.zone_id
                        self._update_object_zone(card_id, old_zone_id, zone_id)

                        card.zone_id = zone_id
                        state_changed = True

            elif zone_id:
                # Zone with ID but no type?
                logging.warning(f"Zone with ID {zone_id} has no type! Owner: {owner_seat_id}")

        return state_changed

    def _parse_players(self, players: list) -> bool:
        state_changed = False
        for player_data in players:
            seat_id = player_data.get("systemSeatNumber")
            if not seat_id: continue

            if seat_id not in self.players:
                self.players[seat_id] = PlayerState(seat_id=seat_id)
                state_changed = True

            player = self.players[seat_id]

            # Parse life total
            if "lifeTotal" in player_data and player.life_total != player_data["lifeTotal"]:
                player.life_total = player_data["lifeTotal"]
                state_changed = True

            # Parse hand count
            if "handCardCount" in player_data and player.hand_count != player_data["handCardCount"]:
                player.hand_count = player_data["handCardCount"]
                state_changed = True

            # Parse mana pool (e.g., {"W": 2, "U": 1, "G": 0})
            if "manaPool" in player_data:
                mana_pool = player_data["manaPool"]
                if isinstance(mana_pool, dict) and mana_pool != player.mana_pool:
                    player.mana_pool = mana_pool
                    logging.debug(f"  Player {seat_id} mana pool: {mana_pool}")
                    state_changed = True

            # Parse energy counters
            if "energy" in player_data and player.energy != player_data["energy"]:
                player.energy = player_data["energy"]
                logging.debug(f"  Player {seat_id} energy: {player.energy}")
                state_changed = True

            # Detect mulligan phase
            pending_msg = player_data.get("pendingMessageType", "")
            logging.debug(f"Player {seat_id} pendingMessageType: {pending_msg}, local_player: {self.local_player_seat_id}")

            if pending_msg == "ClientMessageType_MulliganResp" and seat_id == self.local_player_seat_id:
                if not self.in_mulligan_phase:
                    self.in_mulligan_phase = True
                    logging.info(f"🎴 MULLIGAN PHASE DETECTED for player {seat_id}")
                    state_changed = True
            elif self.in_mulligan_phase and pending_msg != "ClientMessageType_MulliganResp":
                # Mulligan phase ended
                logging.info("🎴 Mulligan phase ended")
                self.in_mulligan_phase = False
                state_changed = True

        return state_changed

    def _parse_turn_info(self, turn_info: dict) -> bool:
        state_changed = False
        if self.priority_player_seat != turn_info.get("priorityPlayer"):
            self.priority_player_seat = turn_info.get("priorityPlayer")
            state_changed = True

        # Reset game history on new turn
        new_turn = turn_info.get("turnNumber")
        if self.current_turn != new_turn:
            self.current_turn = new_turn
            self.game_history = GameHistory(turn_number=new_turn)
            logging.info(f"🔄 New turn {new_turn} - resetting game history")
            state_changed = True

        # Clear combat state when exiting combat phases
        new_phase = turn_info.get("phase", self.current_phase)
        if self.current_phase != new_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase

            # Clear combat data when moving from combat to post-combat
            if "Combat" in old_phase and "Combat" not in new_phase:
                if self.game_history.current_attackers:
                    logging.info(f"Combat ended - clearing {len(self.game_history.current_attackers)} attackers")
                    self.game_history.current_attackers.clear()
                    self.game_history.current_blockers.clear()
                    self.game_history.combat_damage_assignments.clear()
                    state_changed = True

        self.active_player_seat = turn_info.get("activePlayer", self.active_player_seat)
        return state_changed

    def _parse_annotations(self, message: dict) -> bool:
        """
        Parse annotation messages for zone transfers, damage, abilities.

        Zone transfers are THE authoritative source for card movement.
        This fixes the board state accuracy issues for LLM context.
        """
        if "annotations" not in message:
            return False

        state_changed = False

        for annotation in message["annotations"]:
            ann_type = annotation.get("type", [])

            # ZONE TRANSFERS - THE CRITICAL ANNOTATION TYPE
            if "AnnotationType_ZoneTransfer" in ann_type:
                affected_ids = annotation.get("affectedIds", [])
                details = annotation.get("details", [])

                # Parse source/dest zones
                zone_src = None
                zone_dest = None
                category = None

                for detail in details:
                    key = detail.get("key")
                    if key == "zone_src":
                        zone_src = detail.get("valueInt32", [None])[0]
                    elif key == "zone_dest":
                        zone_dest = detail.get("valueInt32", [None])[0]
                    elif key == "category":
                        category = detail.get("valueString", [None])[0]

                # Update game objects with new zones
                for instance_id in affected_ids:
                    if instance_id in self.game_objects:
                        obj = self.game_objects[instance_id]
                        old_zone = obj.zone_id

                        if zone_dest is not None:
                            obj.zone_id = zone_dest

                            # Get zone names from mapping
                            zone_src_name = self.zone_id_to_type.get(zone_src, f"Zone{zone_src}")
                            zone_dest_name = self.zone_id_to_type.get(zone_dest, f"Zone{zone_dest}")

                            # Track game history events
                            if "Battlefield" in zone_dest_name:
                                # Card entered battlefield (played/put into play)
                                self.game_history.cards_played_this_turn.append(obj)
                                if hasattr(obj, 'card_types') and "CardType_Land" in str(obj.card_types):
                                    self.game_history.lands_played_this_turn += 1

                            if "Graveyard" in zone_dest_name and "Battlefield" in zone_src_name:
                                # Creature died
                                card_name = getattr(obj, 'name', f"Card{instance_id}")
                                self.game_history.died_this_turn.append(card_name)
                                logging.debug(f"💀 {card_name} died this turn")

                            logging.info(f"⚡ Zone transfer: Card {instance_id} (grpId:{obj.grp_id}) "
                                       f"{zone_src_name} → {zone_dest_name} ({category})")

                            state_changed = True
                    else:
                        # Card not in game_objects yet - will be created in next GameStateMessage
                        logging.debug(f"Zone transfer for unknown instance {instance_id} "
                                    f"(will be created shortly)")

            # OTHER ANNOTATION TYPES (optional but useful)
            elif "AnnotationType_DamageDealt" in ann_type:
                # Track damage for better tactical advice
                affected_ids = annotation.get("affectedIds", [])
                details = annotation.get("details", [])

                damage_amount = 0
                for detail in details:
                    if detail.get("key") == "damage":
                        damage_amount = detail.get("valueInt32", [0])[0]

                for instance_id in affected_ids:
                    self.game_history.damage_dealt[instance_id] = damage_amount

                logging.debug(f"💥 Damage dealt: {damage_amount} to {affected_ids}")

            elif "AnnotationType_ObjectIdChanged" in ann_type:
                # Card transformed (e.g., daybound/nightbound)
                logging.debug(f"Card transformed: {annotation.get('affectedIds', [])}")

            elif "AnnotationType_BlockerAssigned" in ann_type or "AnnotationType_Blocking" in ann_type:
                # Track blocker assignments
                affected_ids = annotation.get("affectedIds", [])
                details = annotation.get("details", [])

                attacker_id = None
                blocker_id = None

                # Parse blocker assignment details
                for detail in details:
                    key = detail.get("key")
                    if key == "attacker" or key == "attackedTarget":
                        attacker_id = detail.get("valueInt32", [None])[0]
                    elif key == "blocker" or key == "blockingCreature":
                        blocker_id = detail.get("valueInt32", [None])[0]

                # If we found blocker assignment, record it
                if attacker_id and blocker_id:
                    self.game_history.current_blockers[attacker_id] = blocker_id
                    logging.info(f"🛡️ Blocker assigned: creature {blocker_id} blocks attacker {attacker_id}")
                    state_changed = True

            elif "AnnotationType_CombatDamage" in ann_type or "AnnotationType_DamageAssigned" in ann_type:
                # Track combat damage assignments
                affected_ids = annotation.get("affectedIds", [])
                details = annotation.get("details", [])

                damage_amount = 0
                for detail in details:
                    if detail.get("key") == "damage" or detail.get("key") == "amount":
                        damage_amount = detail.get("valueInt32", [0])[0]

                # Record damage for each affected creature
                for instance_id in affected_ids:
                    self.game_history.combat_damage_assignments[instance_id] = damage_amount
                    logging.debug(f"⚔️ Combat damage: {damage_amount} assigned to creature {instance_id}")

        return state_changed

class GameStateManager:
    def __init__(self, card_lookup: "ArenaCardDatabase"):
        self.scanner = MatchScanner(card_lookup=card_lookup)
        self.card_lookup = card_lookup
        self._json_buffer: str = ""
        self._json_depth: int = 0

        # Draft event detection
        self._next_line_event: Optional[str] = None  # For <== events with JSON on next line
        self._draft_callbacks: Dict[str, Callable] = {}  # Event type -> callback function

    def register_draft_callback(self, event_type: str, callback: Callable):
        """Register a callback for a specific draft event type"""
        self._draft_callbacks[event_type] = callback
        logging.info(f"Registered draft callback for event type: {event_type}")

    def _find_gre_event(self, data: dict) -> Optional[dict]:
        """Recursively searches for 'greToClientEvent' within a dictionary."""
        if isinstance(data, dict):
            if "greToClientEvent" in data:
                return data
            for key, value in data.items():
                result = self._find_gre_event(value)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_gre_event(item)
                if result:
                    return result
        return None

    def _parse_deck_submission(self, parsed_data: dict) -> bool:
        """
        Parse deck submission from GREMessageType_ConnectResp or similar messages.

        Expected format (may vary):
        {
            "greToClientEvent": {
                "greToClientMessages": [
                    {
                        "type": "GREMessageType_ConnectResp",
                        "connectResp": {
                            "deckMessage": {
                                "deckCards": [grpId1, grpId2, ...]
                            }
                        }
                    }
                ]
            }
        }
        """
        try:
            deck_cards = None

            # Pattern 1: GREMessageType_ConnectResp with deckMessage (most common)
            if "greToClientEvent" in parsed_data:
                gre_event = parsed_data["greToClientEvent"]
                if "greToClientMessages" in gre_event:
                    for message in gre_event["greToClientMessages"]:
                        if message.get("type") == "GREMessageType_ConnectResp":
                            connect_resp = message.get("connectResp", {})
                            deck_message = connect_resp.get("deckMessage", {})
                            if "deckCards" in deck_message:
                                deck_cards = deck_message["deckCards"]
                                break

            # Pattern 2: params.deckCards
            if not deck_cards and "params" in parsed_data and "deckCards" in parsed_data["params"]:
                deck_cards = parsed_data["params"]["deckCards"]

            # Pattern 3: payload with nested JSON string
            if not deck_cards and "payload" in parsed_data:
                payload_str = parsed_data["payload"]
                if isinstance(payload_str, str):
                    payload_data = json.loads(payload_str)
                    if "CourseDeck" in payload_data:
                        deck_cards = payload_data["CourseDeck"].get("deckCards", [])
                elif isinstance(payload_str, dict):
                    if "CourseDeck" in payload_str:
                        deck_cards = payload_str["CourseDeck"].get("deckCards", [])

            # Pattern 4: Direct deckCards key
            if not deck_cards and "deckCards" in parsed_data:
                deck_cards = parsed_data["deckCards"]

            if deck_cards and isinstance(deck_cards, list) and len(deck_cards) > 0:
                # New match detected - clear all previous state
                self.scanner.reset_match_state()

                # Count occurrences of each grpId
                deck_composition = {}
                for grp_id in deck_cards:
                    deck_composition[grp_id] = deck_composition.get(grp_id, 0) + 1

                self.scanner.submitted_decklist = deck_composition
                self.scanner.cards_seen = {}  # Reset cards seen

                total_cards = len(deck_cards)
                unique_cards = len(deck_composition)
                logging.info(f"📋 Deck submission parsed: {total_cards} cards, {unique_cards} unique")
                logging.debug(f"Deck composition (grpIds): {deck_composition}")

                return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.debug(f"Failed to parse deck submission: {e}")

        return False

    def _parse_ux_event(self, parsed_data: dict) -> bool:
        """
        Parse UX event data, particularly scry/surveil results.

        Example message structure:
        {
            "MessageType": "UXEventData.ScryResultData",
            "PlayerId": "you",
            "CardsToTop": [103, 105],
            "CardsToBottom": [104]
        }
        """
        try:
            # Look for ScryResultData or similar UX events
            message_type = parsed_data.get("MessageType", "")

            # Check for scry/surveil events
            if "ScryResultData" in str(parsed_data) or "Scry" in message_type:
                # Try to extract scry information from various possible structures
                cards_to_top = parsed_data.get("CardsToTop", [])
                cards_to_bottom = parsed_data.get("CardsToBottom", [])
                player_id = parsed_data.get("PlayerId")

                # Check if this scry was performed by the local player
                if player_id == "you" or player_id == self.scanner.local_player_seat_id:
                    # Convert instance IDs to card names
                    top_card_names = []
                    for instance_id in cards_to_top:
                        if instance_id in self.scanner.game_objects:
                            card_obj = self.scanner.game_objects[instance_id]
                            card_name = card_obj.name if card_obj.name else f"Card{card_obj.grp_id}"
                            top_card_names.append(card_name)

                    if top_card_names:
                        self.scanner.library_top_known = top_card_names
                        self.scanner.scry_info = f"Scried: top {len(top_card_names)} card(s) known"
                        logging.info(f"🔮 Scry detected: {len(top_card_names)} cards on top - {', '.join(top_card_names)}")
                        return True

        except (KeyError, TypeError, AttributeError) as e:
            logging.debug(f"Failed to parse UX event: {e}")

        return False

    def parse_log_line(self, line: str) -> bool:
        logging.debug(f"Full log line received by GameStateManager: {line}")
        
        # Try to extract timestamp from the line to track event freshness
        self.scanner.parse_timestamp(line)

        # DRAFT EVENTS: Check if this is the JSON line after an end event marker
        if self._next_line_event:
            event_type = self._next_line_event
            self._next_line_event = None  # Clear the flag

            # Try to parse the JSON on this line
            try:
                json_start = line.find("{")
                if json_start != -1:
                    parsed_data = json.loads(line[json_start:])
                    logging.info(f"Parsed draft event: {event_type}")

                    # Check if data is wrapped in a Payload field (common for BotDraft events)
                    if "Payload" in parsed_data and isinstance(parsed_data["Payload"], str):
                        try:
                            # Parse the escaped JSON in Payload field
                            payload_str = parsed_data["Payload"]
                            if payload_str and payload_str.strip():
                                inner_data = json.loads(payload_str)
                                logging.debug(f"Unpacked Payload for {event_type}")
                                parsed_data = inner_data
                        except json.JSONDecodeError as e:
                            # Suppress warning for common empty/invalid payloads
                            logging.debug(f"Failed to parse Payload JSON: {e}")

                    # Call the registered callback if it exists
                    if event_type in self._draft_callbacks:
                        # Debug: Calling callback for event
                        self._draft_callbacks[event_type](parsed_data)
                    else:
                        # Debug: No callback registered for draft event
                        logging.debug(f"No callback registered for draft event: {event_type}")
                else:
                    # Some events like LogBusinessEvents return a status string, not JSON
                    logging.debug(f"No JSON found in line after {event_type}, might be status string: {line[:100]}")
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON for draft event {event_type}: {e}")

            return False  # Draft events don't change game state

        # DRAFT EVENTS: Check for Draft.Notify messages (Premier Draft)
        # Format: [UnityCrossThreadLogger]Draft.Notify {"draftId":"...","SelfPick":5,"SelfPack":1,"PackCards":"..."}
        if "[UnityCrossThreadLogger]Draft.Notify" in line:
            match = re.search(r'\[UnityCrossThreadLogger\]Draft\.Notify (.+)', line)
            if match:
                json_str = match.group(1)
                try:
                    draft_data = json.loads(json_str)

                    # Extract pack and pick numbers (1-indexed)
                    pack_num = draft_data.get("SelfPack", 1)
                    pick_num = draft_data.get("SelfPick", 1)

                    # Parse PackCards string (comma-separated card IDs)
                    pack_cards_str = draft_data.get("PackCards", "")
                    if pack_cards_str:
                        pack_arena_ids = [int(card_id) for card_id in pack_cards_str.split(",")]

                        logging.info(f"Draft.Notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")
                        # Debug: Draft.Notify detected

                        # Call Draft.Notify callback if registered
                        if "Draft.Notify" in self._draft_callbacks:
                            callback_data = {
                                "PackNumber": pack_num,
                                "PickNumber": pick_num,
                                "PackCards": pack_arena_ids,
                                "DraftId": draft_data.get("draftId", "")
                            }
                            self._draft_callbacks["Draft.Notify"](callback_data)

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse Draft.Notify JSON: {e}")

            return False  # Draft events don't change game state

        # DRAFT EVENTS: Check for start events with JSON in same line
        # Format: [UnityCrossThreadLogger]==> EventName {...}
        if line.startswith("[UnityCrossThreadLogger]==>"):
            match = re.search(r'\[UnityCrossThreadLogger\]==> (\w+) (.*)', line)
            if match:
                event_type = match.group(1)
                outer_json_str = match.group(2)

                try:
                    outer_json = json.loads(outer_json_str)
                    if "request" in outer_json:
                        inner_json = json.loads(outer_json["request"])

                        logging.info(f"Detected start event: {event_type}")

                        # Handle LogBusinessEvents (Premier Draft picks)
                        if event_type == "LogBusinessEvents" and "DraftId" in inner_json:
                            logging.info(f"Premier Draft pick detected")
                            if "LogBusinessEvents" in self._draft_callbacks:
                                self._draft_callbacks["LogBusinessEvents"](inner_json)
                except json.JSONDecodeError as e:
                    logging.debug(f"Failed to parse start event JSON: {e}")

            return False  # Draft events don't change game state

        # DRAFT EVENTS: Check for end event markers
        # Format: <== EventName(uuid)
        if line.startswith("<=="):
            match = re.search(r'<== (\w+)\(([a-f0-9-]+)\)', line)
            if match:
                event_type = match.group(1)
                # event_id = match.group(2)  # UUID, not currently used

                # Mark that the next line contains the JSON for this event
                self._next_line_event = event_type
                # Debug: Detected end event marker
                logging.debug(f"Detected end event marker: {event_type}, expecting JSON on next line")

            return False  # Draft events don't change game state

        # If we are not building a JSON object, look for the start
        if self._json_depth == 0:
            json_start_index = line.find('{')
            if json_start_index == -1:
                return False  # Not in an object and no new one starts, so skip.

            # Start of a new JSON object
            line_content = line[json_start_index:]
            self._json_buffer = line_content
            self._json_depth = line_content.count('{') - line_content.count('}')
        else:
            # We are in the middle of a JSON object, append the new line
            self._json_buffer += line
            self._json_depth += line.count('{') - line.count('}')

        # Check for malformed JSON (more closing than opening braces)
        if self._json_depth < 0:
            logging.warning(f"JSON depth is negative ({self._json_depth}). Buffer corrupted, resetting.")
            self._json_buffer = ""
            self._json_depth = 0
            return False

        # If we have a complete object, parse it
        if self._json_depth == 0 and self._json_buffer:
            # Make a copy to parse and clear the instance buffer immediately
            json_to_parse = self._json_buffer
            self._json_buffer = ""

            try:
                parsed_data = json.loads(json_to_parse)

                # Parse deck submission (but don't return - let GRE event processing continue)
                # This is important because GRE event processing sets local_player_seat_id
                self._parse_deck_submission(parsed_data)

                # Parse UX events (but don't return - let GRE event processing continue)
                self._parse_ux_event(parsed_data)

                # Original GRE event parsing (this sets local_player_seat_id from systemSeatIds)
                gre_event_data = self._find_gre_event(parsed_data)

                if gre_event_data:
                    logging.debug("Successfully found and parsed GreToClientEvent JSON.")
                    return self.scanner.parse_gre_to_client_event(gre_event_data)
                else:
                    logging.debug("Parsed JSON but 'greToClientEvent' not found within the object.")
                    return False
            except json.JSONDecodeError as e:
                logging.debug(f"JSON parsing failed for buffered content. Error: {e}. Content: {json_to_parse[:200]}...")
                # Buffer is already cleared, so we are ready for the next object.
                return False

        return False

    def get_current_board_state(self) -> Optional[BoardState]:
        start_time = time.time()
        # logging.debug("Attempting to get current board state.")
        if not self.scanner.local_player_seat_id:
            # print("DEBUG: No local player seat ID")
            logging.debug("No local player seat ID found yet.")
            return None

        your_seat_id = self.scanner.local_player_seat_id
        opponent_seat_id = next((seat for seat in self.scanner.players if seat != your_seat_id), None)
        if opponent_seat_id is None:
            logging.debug(f"No opponent seat ID found. Known players: {list(self.scanner.players.keys())}")
            return None

        your_player = self.scanner.players.get(your_seat_id)
        opponent_player = self.scanner.players.get(opponent_seat_id)
        if not your_player or not opponent_player:
            logging.debug(f"Missing player data. Your player: {your_player}, Opponent player: {opponent_player}")
            return None

        # print(f"DEBUG: Priority check: priority_seat={self.scanner.priority_player_seat}, your_seat={your_seat_id}")
        board_state = BoardState(
            your_seat_id=your_seat_id,
            opponent_seat_id=opponent_seat_id,
            your_life=your_player.life_total,
            your_hand_count=your_player.hand_count,
            opponent_life=opponent_player.life_total,
            # Use the tracked hand count from PlayerState, which is updated via game state messages
            opponent_hand_count=opponent_player.hand_count,
            current_turn=self.scanner.current_turn,
            current_phase=self.scanner.current_phase,
            is_your_turn=(self.scanner.active_player_seat == your_seat_id),
            has_priority=(self.scanner.priority_player_seat == your_seat_id),
            your_mana_pool=your_player.mana_pool.copy() if your_player.mana_pool else {},
            your_energy=your_player.energy,
            opponent_energy=opponent_player.energy,
            library_top_known=self.scanner.library_top_known.copy(),
            scry_info=self.scanner.scry_info,
            in_mulligan_phase=self.scanner.in_mulligan_phase,
            game_stage=self.scanner.game_stage
        )

        for obj_id, obj in self.scanner.game_objects.items():
            logging.debug(f"  Object {obj_id}: grpId={obj.grp_id}, zoneId={obj.zone_id}, owner={obj.owner_seat_id}")

        # P0 Performance: Card names/colors are now resolved once on object creation.
        # This minimal fallback handles only edge cases (e.g., objects created before card_lookup was available).
        fallback_count = 0
        for obj in self.scanner.game_objects.values():
            # Fallback: Only resolve if still missing (should be rare)
            if obj.grp_id != 0 and (not obj.name or obj.name.startswith("Unknown Card")):
                obj.name = self.card_lookup.get_card_name(obj.grp_id)
                fallback_count += 1
                logging.debug(f"Fallback resolution for {obj.grp_id}: {obj.name}")

            if obj.grp_id != 0 and not obj.color_identity:
                try:
                    card_data = self.card_lookup.get_card_data(obj.grp_id)
                    if card_data:
                        obj.color_identity = card_data.get("color_identity", "")
                        fallback_count += 1
                except Exception as e:
                    logging.debug(f"Could not fetch color for {obj.grp_id}: {e}")

        if fallback_count > 0:
            logging.info(f"⚠️ Fallback resolution needed for {fallback_count} objects (should be rare)")

        # P0 Performance: Use zone-based lookups instead of iterating all objects
        # This changes from O(n) to O(zones * objects_per_zone), which is much faster

        # Helper function to process objects in a zone
        def process_zone(zone_type_str: str, owner_filter: Optional[int] = None):
            """Get all objects in a specific zone type, optionally filtered by owner."""
            # Find all zone IDs that match this zone type
            zone_ids = [zid for zid, ztype in self.scanner.zone_id_to_type.items() if zone_type_str in ztype]

            objects = []
            for zone_id in zone_ids:
                if zone_id in self.scanner._zone_objects:
                    for instance_id in self.scanner._zone_objects[zone_id]:
                        obj = self.scanner.game_objects.get(instance_id)
                        if obj:
                            # Filter by owner if specified
                            if owner_filter is None or obj.owner_seat_id == owner_filter:
                                objects.append(obj)
                                logging.debug(f"Card {obj.grp_id} ({obj.name}), color={obj.color_identity}, zone={zone_id} ({zone_type_str}), owner={obj.owner_seat_id}")
            return objects

        # Process each zone type efficiently
        board_state.your_hand = process_zone("Hand", your_seat_id)
        board_state.your_battlefield = process_zone("Battlefield", your_seat_id)
        board_state.your_graveyard = process_zone("Graveyard", your_seat_id)
        board_state.your_exile = process_zone("Exile", your_seat_id)
        board_state.opponent_battlefield = process_zone("Battlefield", opponent_seat_id)
        board_state.opponent_graveyard = process_zone("Graveyard", opponent_seat_id)
        board_state.opponent_exile = process_zone("Exile", opponent_seat_id)
        board_state.stack = process_zone("Stack")  # Stack is shared, no owner filter

        # Count library cards (we don't need the full list, just the count)
        your_library_objects = process_zone("Library", your_seat_id)
        board_state.your_library_count = len(your_library_objects)

        opponent_library_objects = process_zone("Library", opponent_seat_id)
        board_state.opponent_library_count = len(opponent_library_objects)

        # Log zone contents
        logging.debug(f"  -> Your hand: {[obj.name for obj in board_state.your_hand]}")
        logging.debug(f"  -> Your battlefield: {[obj.name for obj in board_state.your_battlefield]}")
        logging.debug(f"  -> Opponent battlefield: {[obj.name for obj in board_state.opponent_battlefield]}")

        # CRITICAL FIX: Sync hand count
        # If we detected actual cards in hand, trust that over the potentially stale server reported count.
        real_hand_count = len(board_state.your_hand)
        if real_hand_count > board_state.your_hand_count:
            logging.info(f"Correcting hand count from {board_state.your_hand_count} to {real_hand_count} based on detected cards.")
            board_state.your_hand_count = real_hand_count

        # Add game history to board state
        board_state.history = self.scanner.game_history

        # Phase 3: Add deck tracking to board state
        if self.scanner.submitted_decklist:
            # Convert grpId-based deck to card names with counts
            deck_with_names = {}
            total_deck_size = 0
            for grp_id, count in self.scanner.submitted_decklist.items():
                card_name = self.card_lookup.get_card_name(grp_id)
                deck_with_names[card_name] = count
                total_deck_size += count

            board_state.your_decklist = deck_with_names

            # Only calculate library count from deck size if we don't already have it from zone tracking
            # Zone tracking is more accurate as it counts actual game objects in library
            if board_state.your_library_count == 0:
                cards_seen = (len(board_state.your_hand) +
                             len(board_state.your_battlefield) +
                             len(board_state.your_graveyard) +
                             len(board_state.your_exile))
                board_state.your_library_count = max(0, total_deck_size - cards_seen)
                logging.debug(f"Calculated library count from deck size: {board_state.your_library_count} "
                             f"(deck size {total_deck_size} - seen {cards_seen})")
            else:
                logging.debug(f"Using library count from zone tracking: {board_state.your_library_count}")

            board_state.your_deck_remaining = board_state.your_library_count

            logging.debug(f"Deck tracking: {len(deck_with_names)} unique cards, deck size {total_deck_size}, "
                         f"{board_state.your_library_count} remaining in library")

        # Fill in missing hand cards with placeholders if count > detected
        if board_state.your_hand_count > len(board_state.your_hand):
            missing_count = board_state.your_hand_count - len(board_state.your_hand)
            logging.info(f"Hand count mismatch: Arena says {board_state.your_hand_count}, detected {len(board_state.your_hand)}. Adding {missing_count} placeholders.")
            for i in range(missing_count):
                # Create a placeholder object
                placeholder = GameObject(
                    instance_id=-1, 
                    grp_id=0, 
                    zone_id=0, 
                    owner_seat_id=your_seat_id,
                    name="Unknown Card",
                    visibility="private"
                )
                board_state.your_hand.append(placeholder)

        logging.info(f"Board State Summary: Hand: {len(board_state.your_hand)} (Count: {board_state.your_hand_count}), "
                    f"Battlefield: {len(board_state.your_battlefield)}, "
                    f"Graveyard: {len(board_state.your_graveyard)}, "
                    f"Exile: {len(board_state.your_exile)}, "
                    f"Library: {board_state.your_library_count}")
        
        elapsed = time.time() - start_time
        logging.debug(f"PERFORMANCE: get_current_board_state took {elapsed:.4f} seconds")
        
        return board_state

    def validate_board_state(self, board_state: BoardState) -> bool:
        """
        Validate that board state makes sense before sending to LLM.
        Returns True if valid, False if something is wrong.

        Note: We're lenient with hand count mismatches because:
        1. Arena's reported hand_count can lag behind actual card detection
        2. Unknown cards (not in database) are still playable cards
        """
        issues = []
        warnings = []

        # Check hand count - but only warn if dramatically different
        # Arena sometimes reports 0 while cards are being parsed
        if board_state.your_hand_count > 0:
            if len(board_state.your_hand) != board_state.your_hand_count:
                warnings.append(f"Hand count mismatch: detected {len(board_state.your_hand)}, "
                              f"Arena reports {board_state.your_hand_count}")

        # Check for unknown cards - warn but don't block
        unknown_count = sum(1 for card in board_state.your_hand if "Unknown" in card.name)
        if unknown_count > 0:
            warnings.append(f"{unknown_count} unknown cards in hand")

        # Check battlefield for unknowns
        unknown_bf = sum(1 for card in board_state.your_battlefield if "Unknown" in card.name)
        if unknown_bf > 0:
            warnings.append(f"{unknown_bf} unknown cards on battlefield")

        # Card count consistency validation (from PR review)
        if board_state.your_decklist:
            total_deck_size = sum(board_state.your_decklist.values())

            cards_accounted_for = (
                len(board_state.your_hand) +
                len(board_state.your_battlefield) +
                len(board_state.your_graveyard) +
                len(board_state.your_exile) +
                board_state.your_library_count +
                sum(1 for card in board_state.stack if card.owner_seat_id == board_state.your_seat_id)
            )

            # Allow for a small discrepancy, but flag major differences
            if abs(total_deck_size - cards_accounted_for) > 2:
                issues.append(f"Major card count mismatch: Deck has {total_deck_size} cards, but {cards_accounted_for} are accounted for.")

        # Only fail validation for critical issues
        if issues:
            logging.warning(f"Board state validation FAILED: {', '.join(issues)}")
            return False

        if warnings:
            logging.debug(f"Board state warnings (non-critical): {', '.join(warnings)}")

        logging.debug("Board state validation passed ✓")
        return True
