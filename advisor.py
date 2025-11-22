#!/usr/bin/env python3
import dataclasses
import json
import logging
from pathlib import Path
import requests
import urllib.request
import time
import os
import threading
from typing import Dict, List, Optional, Callable
import subprocess
import tempfile


import logging
import os
from pathlib import Path

# Ensure the logs directory exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging - file gets all, console gets only errors/warnings
log_file_path = log_dir / "advisor.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Console handler
    ]
)
# Set console handler to WARNING to hide debug/info noise
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.WARNING)

# ----------------------------------------------------------------------------------
# Part 1: Arena Log Detection and Path Handling
# ----------------------------------------------------------------------------------

def detect_player_log_path():
    """
    Detect the Arena Player.log file based on OS and installation method.
    Returns the path as a string, or None if not found.
    """
    home = Path.home()
    # Windows
    if os.name == 'nt':
        username = os.getenv('USERNAME')
        drive = os.getenv('USERPROFILE')[0]
        windows_path = f"{drive}:/Users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(windows_path):
            return windows_path
    # macOS
    elif os.name == 'posix' and os.uname().sysname == 'Darwin':
        macos_path = home / "Library/Logs/Wizards Of The Coast/MTGA/Player.log"
        if os.path.exists(macos_path):
            return str(macos_path)
    # Linux
    elif os.name == 'posix':
        username = os.getenv('USER')
        paths = [
            home / f".var/app/com.usebottles.bottles/data/bottles/bottles/MTG-Arena/drive_c/users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / f"Games/magic-the-gathering-arena/drive_c/users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
            home / ".local/share/Steam/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log",
        ]
        for path in paths:
            if path.exists():
                return str(path)
    return None

def detect_card_database_path():
    """
    Find Arena's card database across platforms.
    Returns path to Raw_CardDatabase_*.mtga file.
    """
    home = Path.home()

    # Windows
    if os.name == 'nt':
        arena_data = Path("C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/")
        if arena_data.exists():
            db_files = list(arena_data.glob("Raw_CardDatabase_*.mtga"))
            if db_files:
                return str(db_files[0])

    # Linux (Steam/Bottles)
    elif os.name == 'posix':
        paths = [
            home / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/",
            home / ".var/app/com.usebottles.bottles/data/bottles/bottles/MTG-Arena/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/",
        ]
        for base_path in paths:
            if base_path.exists():
                db_files = list(base_path.glob("Raw_CardDatabase_*.mtga"))
                if db_files:
                    return str(db_files[0])

    logging.warning("Could not find Arena card database - will fallback to Scryfall API")
    return None

# ----------------------------------------------------------------------------------
# Part 2: Real-Time Log Parsing
# ----------------------------------------------------------------------------------

class LogFollower:
    """Follows the Arena Player.log file and yields new lines as they're added."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self.inode = None
        self.offset = 0

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
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
                    if self.file:
                        self.file.close()
                    self.file = open(self.log_path, 'r', encoding='utf-8', errors='replace')
                    self.inode = current_inode
                    self.offset = 0
                    logging.info("Log file (re)opened.")
                
                self.file.seek(self.offset)
                while True:
                    line = self.file.readline()
                    if not line:
                        break
                    self.offset = self.file.tell()
                    stripped_line = line.strip()
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Read line: {stripped_line[:100]}...") # Log first 100 chars to avoid spam
                    callback(stripped_line)
                time.sleep(0.1)
            except FileNotFoundError:
                logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error following log file: {e}")
                time.sleep(1)

    def close(self):
        if self.file:
            self.file.close()

# ----------------------------------------------------------------------------------
# Part 3: Game State Tracking from GreToClientEvent
# ----------------------------------------------------------------------------------

@dataclasses.dataclass
class GameObject:
    instance_id: int
    grp_id: int
    zone_id: int
    owner_seat_id: int
    name: str = ""
    power: Optional[int] = None
    toughness: Optional[int] = None
    is_tapped: bool = False
    is_attacking: bool = False

@dataclasses.dataclass
class PlayerState:
    seat_id: int
    life_total: int = 20
    hand_count: int = 0

@dataclasses.dataclass
class BoardState:
    your_seat_id: int
    opponent_seat_id: int
    your_life: int = 20
    your_hand_count: int = 0
    your_hand: List[GameObject] = dataclasses.field(default_factory=list)
    your_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    opponent_life: int = 20
    opponent_hand_count: int = 0
    opponent_battlefield: List[GameObject] = dataclasses.field(default_factory=list)
    current_turn: int = 0
    current_phase: str = ""
    is_your_turn: bool = False
    has_priority: bool = False

class MatchScanner:
    """
    Parses GRE messages to track game state.

    Note: Zone IDs are assigned dynamically per match by Arena.
    We discover them from GameStateMessage zones[] arrays and track them
    in zone_type_to_ids and zone_id_to_type dictionaries.
    """

    def __init__(self):
        self.game_objects: Dict[int, GameObject] = {}
        self.players: Dict[int, PlayerState] = {}
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat: Optional[int] = None
        self.priority_player_seat: Optional[int] = None
        self.local_player_seat_id: Optional[int] = None
        self.zone_type_to_ids: Dict[str, int] = {}  # Maps zone types to their zone IDs
        self.observed_zone_ids: set = set()  # Track all zone IDs we see
        self.zone_id_to_type: Dict[int, str] = {}  # Reverse mapping

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
        logging.info(f"GameStateMessage received")
        game_state = message.get("gameStateMessage", {})
        logging.info(f"Game State keys: {list(game_state.keys()) if game_state else 'empty'}")
        state_changed = False
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

            if zone_id is not None:
                self.observed_zone_ids.add(zone_id)

            logging.debug(f"  GameObject: instanceId={instance_id}, grpId={grp_id}, zoneId={zone_id}, ownerSeatId={owner_seat_id}")

            if instance_id not in self.game_objects:
                self.game_objects[instance_id] = GameObject(
                    instance_id=instance_id,
                    grp_id=grp_id,
                    zone_id=zone_id,
                    owner_seat_id=owner_seat_id
                )
                logging.info(f"    -> Created new GameObject")
                state_changed = True

            game_obj = self.game_objects[instance_id]
            if zone_id is not None and game_obj.zone_id != zone_id:
                logging.info(f"    -> Zone changed from {game_obj.zone_id} to {zone_id}")
                game_obj.zone_id = zone_id
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
                self.zone_type_to_ids[zone_type_str] = zone_id
                self.zone_id_to_type[zone_id] = zone_type_str
                logging.debug(f"Zone mapping: {zone_type_str} -> zoneId {zone_id} (owner: {owner_seat_id}, cards: {len(object_instance_ids)})")

                # Update all cards in this zone with the correct zone ID
                for card_id in object_instance_ids:
                    if card_id in self.game_objects:
                        card = self.game_objects[card_id]
                        if card.zone_id != zone_id:
                            logging.debug(f"Updating card {card_id} zone from {card.zone_id} to {zone_id} ({zone_type_str})")
                            card.zone_id = zone_id
                            state_changed = True

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
            if "lifeTotal" in player_data and player.life_total != player_data["lifeTotal"]:
                player.life_total = player_data["lifeTotal"]
                state_changed = True
            if "handCardCount" in player_data and player.hand_count != player_data["handCardCount"]:
                player.hand_count = player_data["handCardCount"]
                state_changed = True
        return state_changed

    def _parse_turn_info(self, turn_info: dict) -> bool:
        state_changed = False
        if self.priority_player_seat != turn_info.get("priorityPlayer"):
            self.priority_player_seat = turn_info.get("priorityPlayer")
            state_changed = True
        if self.current_turn != turn_info.get("turnNumber"):
            self.current_turn = turn_info.get("turnNumber")
            state_changed = True
        self.current_phase = turn_info.get("phase", self.current_phase)
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
                logging.debug(f"Damage dealt: {annotation.get('details', [])}")

            elif "AnnotationType_ObjectIdChanged" in ann_type:
                # Card transformed (e.g., daybound/nightbound)
                logging.debug(f"Card transformed: {annotation.get('affectedIds', [])}")

        return state_changed

# ----------------------------------------------------------------------------------
# Part 4: Card ID Resolution (grpId to Card Name)
# ----------------------------------------------------------------------------------

class ArenaCardDatabase:
    """
    Fast card lookup using Arena's local SQLite database.
    Falls back to Scryfall API if database not found.
    """

    def __init__(self, db_path: str = None, cache_file: str = "card_cache.json"):
        self.db_path = db_path or detect_card_database_path()
        self.cache_file = cache_file
        self.cache: Dict[int, dict] = {}
        self.conn = None

        # Try to connect to Arena database
        if self.db_path and os.path.exists(self.db_path):
            try:
                import sqlite3
                self.conn = sqlite3.connect(self.db_path)
                logging.info(f"✓ Connected to Arena card database: {self.db_path}")
                self._load_all_cards_into_cache()
            except Exception as e:
                logging.warning(f"Failed to open Arena database: {e}")
                self.conn = None

        # Fallback: Load from cache file
        if not self.conn:
            logging.warning("Arena database not available - using cache/Scryfall fallback")
            self.load_cache()

    def _load_all_cards_into_cache(self):
        """
        Pre-load ALL cards from Arena database into memory cache.
        This is FAST - 21k cards load in ~100ms with 10MB RAM.
        """
        if not self.conn:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    c.GrpId,
                    l.Loc as Name,
                    c.ExpansionCode,
                    c.Rarity,
                    c.Types,
                    c.Subtypes,
                    c.Power,
                    c.Toughness
                FROM Cards c
                JOIN Localizations_enUS l ON c.TitleId = l.LocId AND l.Formatted = 1
                WHERE c.IsToken = 0
            """)

            for row in cursor.fetchall():
                grp_id = row[0]
                self.cache[grp_id] = {
                    "name": row[1],
                    "set": row[2],
                    "rarity": row[3],
                    "types": row[4],
                    "subtypes": row[5],
                    "power": row[6],
                    "toughness": row[7]
                }

            logging.info(f"✓ Loaded {len(self.cache)} cards from Arena database")

            # Save to cache file for fast startup next time
            self._save_cache()

        except Exception as e:
            logging.error(f"Error loading cards from database: {e}")

    def load_cache(self):
        """Load cards from cache file (fallback if DB not available)"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = {int(k): v for k, v in json.load(f).items()}
                logging.info(f"Loaded {len(self.cache)} cards from cache file")
            except Exception as e:
                logging.warning(f"Could not load cache file: {e}")

    def get_card_name(self, grp_id: int) -> str:
        """Get card name from grpId - instant lookup"""
        if not grp_id:
            return "Unknown Card"

        if grp_id in self.cache:
            return self.cache[grp_id].get("name", f"Unknown({grp_id})")

        # Fallback to Scryfall API only if not in Arena DB
        logging.warning(f"Card {grp_id} not in Arena database - fetching from Scryfall")
        return self._fetch_from_scryfall(grp_id)

    def get_card_data(self, grp_id: int) -> Optional[dict]:
        """Get full card data including types, P/T, etc."""
        if grp_id in self.cache:
            return self.cache[grp_id]
        return None

    def _fetch_from_scryfall(self, grp_id: int) -> str:
        """Fallback to Scryfall API (original implementation)"""
        try:
            response = requests.get(f"https://api.scryfall.com/cards/arena/{grp_id}", timeout=5)
            if response.status_code == 200:
                card_data = response.json()
                self.cache[grp_id] = {
                    "name": card_data.get("name", f"Unknown({grp_id})"),
                    "set": card_data.get("set", ""),
                    "rarity": card_data.get("rarity", ""),
                }
                self._save_cache()
                return self.cache[grp_id]["name"]
        except Exception as e:
            logging.error(f"Scryfall API error for grpId {grp_id}: {e}")

        return f"Unknown({grp_id})"

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

    def close(self):
        """Clean up database connection"""
        if self.conn:
            self.conn.close()

# ----------------------------------------------------------------------------------
# Part 5: Building Board State for AI
# ----------------------------------------------------------------------------------

class GameStateManager:
    def __init__(self, card_lookup: ArenaCardDatabase):
        self.scanner = MatchScanner()
        self.card_lookup = card_lookup
        self._line_buffer: List[str] = []
        self._json_depth: int = 0

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

    def parse_log_line(self, line: str) -> bool:
        logging.debug(f"Full log line received by GameStateManager: {line}")

        # If a line contains "GreToClientEvent", it's the start of a new event.
        # Clear the buffer and reset depth, then process this line.
        if "GreToClientEvent" in line:
            self._line_buffer = []
            self._json_depth = 0
            logging.debug("Detected 'GreToClientEvent' in line. Resetting buffer.")

        # Append the current line to the buffer
        self._line_buffer.append(line)

        # Update JSON depth
        self._json_depth += line.count('{')
        self._json_depth -= line.count('}')

        # If depth is 0 and buffer is not empty, we might have a complete JSON object
        if self._json_depth == 0 and self._line_buffer:
            full_json_str = "".join(self._line_buffer)
            self._line_buffer = [] # Clear buffer after attempting to process

            # Find the start of the JSON object
            json_start = full_json_str.find("{")
            if json_start == -1:
                logging.debug("No JSON object start found in buffered lines.")
                return False

            try:
                parsed_data = json.loads(full_json_str[json_start:])
                gre_event_data = self._find_gre_event(parsed_data)

                if gre_event_data:
                    logging.debug("Successfully found and parsed GreToClientEvent JSON.")
                    return self.scanner.parse_gre_to_client_event(gre_event_data)
                else:
                    logging.debug("Parsed JSON but 'greToClientEvent' not found within the object.")
                    return False
            except json.JSONDecodeError as e:
                logging.debug(f"JSON parsing failed for buffered lines. Error: {e}. Content: {full_json_str[:200]}...")
                return False
        elif self._json_depth < 0:
            logging.warning(f"JSON depth went negative. Resetting buffer. Current depth: {self._json_depth}")
            self._line_buffer = []
            self._json_depth = 0
        
        return False

    def get_current_board_state(self) -> Optional[BoardState]:
        logging.debug("Attempting to get current board state.")
        if not self.scanner.local_player_seat_id:
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

        board_state = BoardState(
            your_seat_id=your_seat_id,
            opponent_seat_id=opponent_seat_id,
            your_life=your_player.life_total,
            your_hand_count=your_player.hand_count,
            opponent_life=opponent_player.life_total,
            opponent_hand_count=opponent_player.hand_count,
            current_turn=self.scanner.current_turn,
            current_phase=self.scanner.current_phase,
            is_your_turn=(self.scanner.active_player_seat == your_seat_id),
            has_priority=(self.scanner.priority_player_seat == your_seat_id)
        )

        # Debug: Log all game objects before filtering
        logging.debug(f"Total game objects: {len(self.scanner.game_objects)}")
        for obj_id, obj in self.scanner.game_objects.items():
            logging.debug(f"  Object {obj_id}: grpId={obj.grp_id}, zoneId={obj.zone_id}, owner={obj.owner_seat_id}")

        # Get the actual zone IDs from the mappings discovered during parsing
        hand_zone_id = None
        battlefield_zone_id = None
        for zone_type_str, zone_id in self.scanner.zone_type_to_ids.items():
            if "Hand" in zone_type_str:
                hand_zone_id = zone_id
            elif "Battlefield" in zone_type_str:
                battlefield_zone_id = zone_id

        logging.debug(f"Using zone mappings: Hand={hand_zone_id}, Battlefield={battlefield_zone_id}")

        for obj in self.scanner.game_objects.values():
            obj.name = self.card_lookup.get_card_name(obj.grp_id)
            logging.debug(f"Card {obj.grp_id} ({obj.name}), zone={obj.zone_id}, owner={obj.owner_seat_id}")

            if obj.owner_seat_id == your_seat_id:
                if hand_zone_id and obj.zone_id == hand_zone_id:
                    logging.debug(f"  -> Added to your hand: {obj.name}")
                    board_state.your_hand.append(obj)
                elif battlefield_zone_id and obj.zone_id == battlefield_zone_id:
                    logging.debug(f"  -> Added to your battlefield: {obj.name}")
                    board_state.your_battlefield.append(obj)
                else:
                    logging.debug(f"  -> Skipped your card (zone {obj.zone_id})")
            elif obj.owner_seat_id == opponent_seat_id:
                if battlefield_zone_id and obj.zone_id == battlefield_zone_id:
                    logging.debug(f"  -> Added to opponent battlefield: {obj.name}")
                    board_state.opponent_battlefield.append(obj)
                else:
                    logging.debug(f"  -> Skipped opponent card (zone {obj.zone_id})")
            else:
                logging.debug(f"  -> Skipped card (owner {obj.owner_seat_id} not a player)")

        logging.info(f"Board State Summary: Your Hand: {[card.name for card in board_state.your_hand]}, Your Battlefield: {[card.name for card in board_state.your_battlefield]}, Opponent Battlefield: {[card.name for card in board_state.opponent_battlefield]}")
        return board_state

    def validate_board_state(self, board_state: BoardState) -> bool:
        """
        Validate that board state makes sense before sending to LLM.
        Returns True if valid, False if something is wrong.
        """
        issues = []

        # Check hand count
        if len(board_state.your_hand) != board_state.your_hand_count:
            issues.append(f"Hand mismatch: found {len(board_state.your_hand)}, "
                         f"expected {board_state.your_hand_count}")

        # Check for unknown cards
        unknown_count = sum(1 for card in board_state.your_hand if "Unknown" in card.name)
        if unknown_count > 0:
            issues.append(f"{unknown_count} unknown cards in hand")

        # Check battlefield
        unknown_bf = sum(1 for card in board_state.your_battlefield if "Unknown" in card.name)
        if unknown_bf > 0:
            issues.append(f"{unknown_bf} unknown cards on battlefield")

        if issues:
            logging.warning(f"Board state validation failed: {', '.join(issues)}")
            return False

        logging.debug("Board state validation passed ✓")
        return True

# ----------------------------------------------------------------------------------
# Part 6: AI Advice Generation
# ----------------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.host = host
        self.model = model

    def generate(self, prompt: str) -> Optional[str]:
        logging.debug(f"Ollama prompt: {prompt[:500]}...")
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                logging.debug(f"Ollama raw response: {result}")
                return result.get('response', '').strip()
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return None

class AIAdvisor:
    SYSTEM_PROMPT = "You are an expert Magic: The Gathering tactical advisor. Analyze the board state and provide a concise, actionable turn plan. Focus on the best sequence of plays."

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.client = OllamaClient(host=ollama_host, model=model)

    def get_tactical_advice(self, board_state: BoardState) -> Optional[str]:
        prompt = self._build_prompt(board_state)
        advice = self.client.generate(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        if advice:
            logging.debug(f"AI generated advice: {advice[:500]}...")
        else:
            logging.debug("AI did not generate any advice.")
        return advice

    def _build_prompt(self, board_state: BoardState) -> str:
        lines = [
            f"Turn {board_state.current_turn} - {board_state.current_phase} Phase.",
            f"You: {board_state.your_life} Life, {board_state.your_hand_count} cards in hand.",
            f"Opponent: {board_state.opponent_life} Life, {board_state.opponent_hand_count} cards in hand.",
            "",
            "Your Hand:",
            *[f"- {card.name}" for card in board_state.your_hand],
            "",
            "Your Battlefield:",
            *[f"- {card.name}" for card in board_state.your_battlefield],
            "",
            "Opponent's Battlefield:",
            *[f"- {card.name}" for card in board_state.opponent_battlefield],
            "",
            "What is the optimal play sequence for this turn?"
        ]
        return "\n".join(lines)

# ----------------------------------------------------------------------------------
# Part 7: Text-to-Speech Output
# ----------------------------------------------------------------------------------

class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0):
        """Initialize Kokoro text-to-speech with specified voice (default: adam) and volume (0.0-1.0)"""
        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))  # Clamp volume to 0.0-1.0
        logging.info(f"Initializing Kokoro TTS with voice: {voice}, volume: {self.volume}")
        try:
            from kokoro_onnx import Kokoro
            import numpy as np
            from pathlib import Path
            self.np = np

            # Use downloaded models from ~/.local/share/kokoro/
            models_dir = Path.home() / '.local' / 'share' / 'kokoro'
            model_path = str(models_dir / 'kokoro-v1.0.onnx')
            voices_path = str(models_dir / 'voices-v1.0.bin')

            self.tts = Kokoro(model_path=model_path, voices_path=voices_path)
            logging.info(f"Kokoro TTS initialized successfully with {voice} voice")
        except Exception as e:
            logging.error(f"Failed to initialize Kokoro TTS: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.tts = None

    def set_voice(self, voice: str):
        """Change voice dynamically"""
        self.voice = voice
        logging.info(f"Voice changed to: {voice}")

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        logging.info(f"Volume changed to: {self.volume}")

    def speak(self, text: str):
        if not text:
            logging.debug("No text provided to speak.")
            return

        if not self.tts:
            logging.error("Kokoro TTS not initialized, cannot speak.")
            return

        logging.info(f"Speaking with Kokoro ({self.voice}): {text[:100]}...")
        try:
            # Generate audio using Kokoro
            # create() returns (audio_array, sample_rate)
            audio_array, sample_rate = self.tts.create(text, voice=self.voice, speed=1.0)

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Convert to bytes for WAV file
            import scipy.io.wavfile as wavfile

            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                wavfile.write(tmp_path, sample_rate, (audio_array * 32767).astype(self.np.int16))

            logging.info(f"Generated audio saved to {tmp_path}, playing...")

            # Play the audio using system audio player
            played = False
            try:
                subprocess.run(["aplay", tmp_path], check=True, timeout=120)
                played = True
                logging.info("Audio played with aplay")
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.debug(f"aplay error: {e}")

            if not played:
                try:
                    subprocess.run(["paplay", tmp_path], check=True, timeout=120)
                    played = True
                    logging.info("Audio played with paplay")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logging.debug(f"paplay error: {e}")

            if not played:
                try:
                    subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=True, timeout=120)
                    played = True
                    logging.info("Audio played with ffplay")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logging.debug(f"ffplay error: {e}")

            if not played:
                logging.error("No audio player found (aplay, paplay, or ffplay). Cannot play audio.")

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

            logging.debug("Successfully spoke text with Kokoro.")
        except Exception as e:
            logging.error(f"Kokoro TTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

# ----------------------------------------------------------------------------------
# Part 8: Main CLI Loop
# ----------------------------------------------------------------------------------

class CLIVoiceAdvisor:
    # Available voices in Kokoro v1.0
    AVAILABLE_VOICES = ["af_alloy", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
                        "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo",
                        "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable",
                        "bm_george", "bm_lewis", "ef_dora", "em_alex", "ff_siwis", "hf_alpha",
                        "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha",
                        "pf_dora", "pm_alex", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]

    def __init__(self):
        self.log_path = detect_player_log_path()
        if not self.log_path:
            print("ERROR: Could not find Arena Player.log. Please ensure the game is installed and you have run it at least once.")
            exit(1)

        self.game_state_mgr = GameStateManager(ArenaCardDatabase())
        self.ai_advisor = AIAdvisor()
        self.tts = TextToSpeech(voice="am_adam", volume=1.0)
        self.log_follower = LogFollower(self.log_path)

        self.last_turn_advised = -1
        self.advice_thread = None
        self.first_turn_detected = False
        self.cli_thread = None
        self.running = True

    def run(self):
        """Start the advisor with background log monitoring and interactive CLI"""
        print("\n" + "="*60)
        print("MTGA Voice Advisor Started")
        print("="*60)
        print(f"Log: {self.log_path}")
        print(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%")
        print("\nWaiting for a match... (Enable Detailed Logs in MTGA settings)")
        print("Type /help for commands\n")

        # Start log follower in background thread
        log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
        log_thread.start()

        # Interactive CLI loop
        self._run_cli_loop()

    def _run_log_monitor(self):
        """Monitor Arena log in background"""
        try:
            self.log_follower.follow(self.on_line)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error(f"Log monitor error: {e}")

    def _run_cli_loop(self):
        """Interactive command prompt for user input"""
        try:
            while self.running:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        self._handle_command(user_input)
                    else:
                        # Treat as free-form query to the AI about the current board
                        self._handle_query(user_input)
                except KeyboardInterrupt:
                    print("\n\nShutting down...")
                    self.running = False
                    self.log_follower.close()
                    break
        except Exception as e:
            logging.error(f"CLI error: {e}")
        finally:
            # Clean up database connection
            if hasattr(self.game_state_mgr.card_lookup, 'close'):
                self.game_state_mgr.card_lookup.close()
                logging.info("Card database connection closed")

    def _handle_command(self, command: str):
        """Handle slash commands"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._show_help()
        elif cmd == "/settings":
            self._show_settings()
        elif cmd == "/voice":
            if len(parts) > 1:
                voice = parts[1].lower()
                if voice in self.AVAILABLE_VOICES:
                    self.tts.set_voice(voice)
                    print(f"✓ Voice changed to: {voice}")
                else:
                    print(f"✗ Unknown voice. Available: {', '.join(self.AVAILABLE_VOICES[:5])}...")
            else:
                print(f"✓ Current voice: {self.tts.voice}")
        elif cmd == "/volume":
            if len(parts) > 1:
                try:
                    vol = float(parts[1]) / 100.0
                    self.tts.set_volume(vol)
                    print(f"✓ Volume set to: {int(vol * 100)}%")
                except ValueError:
                    print("✗ Volume must be a number (0-100)")
            else:
                print(f"✓ Current volume: {int(self.tts.volume * 100)}%")
        elif cmd == "/status":
            board_state = self.game_state_mgr.get_current_board_state()
            if board_state:
                print(f"Turn: {board_state.current_turn} | Your Turn: {board_state.is_your_turn} | Has Priority: {board_state.has_priority}")
                print(f"Your Hand: {len(board_state.your_hand)} cards")
                print(f"Your Battlefield: {len(board_state.your_battlefield)} permanents")
                print(f"Opponent Battlefield: {len(board_state.opponent_battlefield)} permanents")
            else:
                print("No match in progress")
        else:
            print(f"Unknown command: {cmd}. Type /help for commands.")

    def _handle_query(self, query: str):
        """Handle free-form queries to the AI about current board state"""
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state or not board_state.current_turn:
            print("No match in progress. Start a game first.")
            return

        print("\nThinking...")
        # Use the AI to answer the query in context of current board state
        prompt = f"""
The user is asking about their current board state in Magic: The Gathering Arena.

Current Board State:
- Current Turn: {board_state.current_turn}
- Your Hand: {', '.join(board_state.your_hand) if board_state.your_hand else 'Empty'}
- Your Battlefield: {', '.join(board_state.your_battlefield) if board_state.your_battlefield else 'Empty'}
- Opponent Battlefield: {', '.join(board_state.opponent_battlefield) if board_state.opponent_battlefield else 'Empty'}

User's Question: {query}

Provide a concise answer (1-2 sentences) based on the board state.
"""
        try:
            response = self.ai_advisor.query_ollama(prompt)
            print(f"Advisor: {response}\n")
            # Optionally speak the response
            # self.tts.speak(response)
        except Exception as e:
            print(f"Error getting response: {e}\n")

    def _show_help(self):
        """Display help menu"""
        print("""
Commands:
  /help          - Show this help menu
  /settings      - Show current settings
  /voice [name]  - Change voice (e.g., /voice bella)
  /volume [0-100] - Set volume (e.g., /volume 80)
  /status        - Show current board state

Free-form queries:
  Type any question about your board state and the advisor will answer.
""")

    def _show_settings(self):
        """Show current settings"""
        print(f"""
Current Settings:
  Voice:   {self.tts.voice}
  Volume:  {int(self.tts.volume * 100)}%
  Log:     {self.log_path}
""")

    def on_line(self, line: str):
        """Parse log line and update game state"""
        logging.debug(f"Received line in on_line: {line[:100]}...")
        state_changed = self.game_state_mgr.parse_log_line(line)
        if state_changed:
            logging.debug("Game state changed. Checking for decision point.")
            self._check_for_decision_point()

    def _check_for_decision_point(self):
        """Check if we should give automatic advice"""
        logging.debug("Checking for decision point...")
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state:
            logging.debug("No board state available yet.")
            return

        if board_state.current_turn is None:
            logging.debug("Current turn not yet determined.")
            return

        # On first turn detection, sync to current turn so we only advise FUTURE turns
        if not self.first_turn_detected:
            self.last_turn_advised = board_state.current_turn - 1
            self.first_turn_detected = True
            logging.info(f"First turn detected (Turn {board_state.current_turn}). Will advise starting from Turn {board_state.current_turn + 1}")
            return

        is_new_turn_for_player = board_state.is_your_turn and board_state.current_turn > self.last_turn_advised

        if board_state.has_priority and is_new_turn_for_player:
            if self.advice_thread and self.advice_thread.is_alive():
                logging.info("Still processing previous advice request.")
                return

            self.last_turn_advised = board_state.current_turn
            self.advice_thread = threading.Thread(target=self._generate_and_speak_advice, args=(board_state,))
            self.advice_thread.start()

    def _generate_and_speak_advice(self, board_state: BoardState):
        """Generate and speak advice for the current turn"""
        # Validate before sending to LLM
        if not self.game_state_mgr.validate_board_state(board_state):
            logging.warning("Skipping advice generation due to invalid board state")
            print(f"\n>>> Turn {board_state.current_turn}: Waiting for complete board state...")
            return

        print(f"\n>>> Turn {board_state.current_turn}: Your move!")
        print("Getting advice from the master...\n")

        advice = self.ai_advisor.get_tactical_advice(board_state)

        if advice:
            print(f"Advisor: {advice}\n")
            logging.info(f"ADVICE:\n{advice}")
            self.tts.speak(advice)
        else:
            logging.warning("No advice was generated.")

if __name__ == "__main__":
    # Add dataclasses import for older python versions
    import dataclasses
    advisor = CLIVoiceAdvisor()
    advisor.run()
