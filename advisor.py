#!/usr/bin/env python3
import dataclasses
import json
import logging
from pathlib import Path
import re
import requests
import urllib.request
import sqlite3
import time
import os
import threading
from typing import Dict, List, Optional, Callable
import subprocess
import tempfile
import curses
from collections import deque

# Import configuration manager for user preferences
try:
    from config_manager import UserPreferences
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logging.warning("Config manager not available. User preferences will not persist.")

# Import RAG system (optional - will gracefully degrade if not available)
try:
    from rag_advisor import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available. Install dependencies with: pip install chromadb sentence-transformers torch")

# Import draft advisor (requires tabulate, termcolor, scipy)
try:
    from draft_advisor import DraftAdvisor, display_draft_pack, format_draft_pack_for_gui
    from deck_builder import DeckBuilder, display_deck_suggestion, format_deck_suggestion_for_gui
    DRAFT_ADVISOR_AVAILABLE = True
except ImportError as e:
    DRAFT_ADVISOR_AVAILABLE = False
    logging.warning(f"Draft advisor not available: {e}. Install with: pip install tabulate termcolor scipy")

# Import Tkinter (optional - for GUI mode)
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available. GUI mode disabled.")

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
# Utility Functions
# ----------------------------------------------------------------------------------

def clean_card_name(name: str) -> str:
    """
    Remove HTML tags from card names.

    Some card names from Arena's database contain HTML tags like <nobr> and </nobr>
    that need to be stripped for proper display and matching.

    Args:
        name: Raw card name potentially containing HTML tags

    Returns:
        Clean card name with all HTML tags removed

    Examples:
        "<nobr>Full-Throttle</nobr> Fanatic" -> "Full-Throttle Fanatic"
        "<nobr>Bane-Marked</nobr> Leonin" -> "Bane-Marked Leonin"
    """
    if not name:
        return name

    # Remove all HTML tags using regex
    # This handles <nobr>, </nobr>, and any other HTML tags
    clean_name = re.sub(r'<[^>]+>', '', name)

    return clean_name

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
        # Path is AppData/LocalLow, but LOCALAPPDATA gives AppData/Local.
        # So, we get the parent of LOCALAPPDATA and append "LocalLow".
        local_appdata = os.getenv('LOCALAPPDATA')
        if local_appdata:
            appdata_dir = Path(local_appdata).parent
            windows_path = appdata_dir / "LocalLow" / "Wizards Of The Coast" / "MTGA" / "Player.log"
            if windows_path.exists():
                return str(windows_path)
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
            # Steam common directory (where MTGA is actually installed)
            home / ".local/share/Steam/steamapps/common/MTGA/MTGA_Data/Downloads/Raw/",
            # Steam proton prefix (old path, kept for compatibility)
            home / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/",
            # Bottles installation
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
        self.first_open = True  # Track if this is first time opening

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
        print(f"[DEBUG] LogFollower.follow() started! Watching: {self.log_path}")
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
                    print(f"[DEBUG] Opening log file (inode changed or first open)")
                    if self.file:
                        self.file.close()
                    self.file = open(self.log_path, 'r', encoding='utf-8', errors='replace')
                    self.inode = current_inode
                    print(f"[DEBUG] File opened successfully, inode={self.inode}")

                    # On first open, seek to end to ignore old matches
                    # On log rotation, start from beginning of new file
                    if self.first_open:
                        # First time opening - go to end
                        self.file.seek(0, 2)  # Seek to end of file
                        self.offset = self.file.tell()
                        self.first_open = False
                        print(f"[DEBUG] First open - seeked to end, offset={self.offset}")
                        logging.info("Log file opened - starting from end (ignoring old matches).")
                    else:
                        # Log rotation - start from beginning of new file
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning of new file.")
                
                self.file.seek(self.offset)
                line_count = 0
                while True:
                    line = self.file.readline()
                    if not line:
                        break
                    line_count += 1
                    self.offset = self.file.tell()
                    stripped_line = line.strip()

                    # Debug: show what we're reading
                    if "Draft" in stripped_line or "BotDraft" in stripped_line or "<==" in stripped_line or "==>" in stripped_line:
                        print(f"[DEBUG] Draft-related line: {stripped_line[:150]}")

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Read line: {stripped_line[:100]}...") # Log first 100 chars to avoid spam
                    callback(stripped_line)
                if line_count > 0:
                    print(f"[DEBUG] Processed {line_count} lines")
                time.sleep(0.05)
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
    summoning_sick: bool = False
    counters: Dict[str, int] = dataclasses.field(default_factory=dict)  # {counter_type: count}
    attached_to: Optional[int] = None  # Instance ID of attached permanent
    visibility: str = "public"  # "public", "private", "revealed"

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
        self.game_history: GameHistory = GameHistory()  # Track events this turn

        # Phase 3: Deck tracking
        self.submitted_decklist: Dict[int, int] = {}  # {grpId: count} - original 60/40 card deck
        self.cards_seen: Dict[int, int] = {}  # {grpId: count} - cards drawn/mulligan'd

        # Phase 3: Known library cards (from scry/surveil)
        self.library_top_known: List[str] = []  # Card names on top of library
        self.scry_info: Optional[str] = None  # Description of last scry

        # Mulligan tracking
        self.game_stage: str = ""  # "GameStage_Start" during mulligan, "GameStage_Play" after
        self.in_mulligan_phase: bool = False

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
        # Note: submitted_decklist is set by _parse_deck_submission, so don't clear it here

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
                logging.info(f"    -> Created new GameObject")
                state_changed = True
            else:
                # Update existing object
                game_obj = self.game_objects[instance_id]

                if zone_id is not None and game_obj.zone_id != zone_id:
                    logging.info(f"    -> Zone changed from {game_obj.zone_id} to {zone_id}")
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
                    game_obj.power = power
                    state_changed = True
                if toughness is not None and game_obj.toughness != toughness:
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

# ----------------------------------------------------------------------------------
# Part 4: Card ID Resolution (grpId to Card Name) - UNIFIED DATABASE
# ----------------------------------------------------------------------------------

def check_and_update_card_database() -> bool:
    """
    Check if card database needs update and offer to update it.

    Returns True if database is ready to use, False if update failed.
    """
    from datetime import datetime, timedelta
    from pathlib import Path
    import sqlite3

    db_path = Path("data/unified_cards.db")

    # If database doesn't exist, we need to build it
    if not db_path.exists():
        print("\n" + "="*70)
        print("CARD DATABASE NOT FOUND")
        print("="*70)
        print("The unified card database needs to be built first.")
        print("This is a one-time setup that downloads ~492MB from Scryfall.")
        print("Expected time: 5-10 minutes depending on your connection.")
        print()

        response = input("Build database now? (y/n): ").strip().lower()
        if response != 'y':
            print("Cannot proceed without card database. Exiting.")
            return False

        print("\nBuilding database...")
        import subprocess
        result = subprocess.run(["python3", "build_unified_card_database.py"],
                              capture_output=False)

        if result.returncode != 0:
            print("✗ Database build failed.")
            return False

        print("✓ Database built successfully!")
        return True

    # Database exists - check if it's stale (>7 days old)
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'default_cards_updated_at'")
        row = cursor.fetchone()
        conn.close()

        if not row:
            logging.warning("No update timestamp in database")
            return True  # Use database anyway

        last_update_str = row[0]
        last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00').replace('+00:00', ''))
        age = datetime.now() - last_update

        if age.days >= 7:
            print(f"\n⚠ Card database is {age.days} days old (updated {last_update_str[:10]})")
            response = input("Update now? (y/n - update recommended weekly): ").strip().lower()

            if response == 'y':
                print("Updating database...")
                import subprocess
                result = subprocess.run(["python3", "update_card_database.py", "--quick"],
                                      capture_output=False)

                if result.returncode == 0:
                    print("✓ Database updated successfully!")
                else:
                    print("⚠ Update failed, using existing database.")
        else:
            logging.info(f"Card database is current ({age.days} days old)")

        return True

    except Exception as e:
        logging.warning(f"Could not check database age: {e}")
        return True  # Use database anyway


class ArenaCardDatabase:
    """
    Unified card database using unified_cards.db (17,000+ cards with reskin support).

    No API fallbacks needed - all MTGA cards are in the database.
    """
    def __init__(self, db_path: str = "data/unified_cards.db", show_reskin_names: bool = False):
        self.db_path = Path(db_path)
        self.conn = None
        self.cache: Dict[int, dict] = {}
        self.show_reskin_names = show_reskin_names  # Toggle for Spider-Man reskins

        if not self.db_path.exists():
            logging.error(f"✗ Card database not found at {db_path}")
            logging.error("  Run: python build_unified_card_database.py")
            return

        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Get stats
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cards")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM cards WHERE is_reskin = 1")
            reskins = cursor.fetchone()[0]

            logging.info(f"✓ Unified card database loaded ({total:,} cards, {reskins} reskins)")

        except Exception as e:
            logging.error(f"Failed to load card database: {e}")
            self.conn = None

    def get_card_name(self, grp_id: int) -> str:
        """Get card name from grpId. Shows reskin names if show_reskin_names is True."""
        if not grp_id or not self.conn:
            return f"Unknown({grp_id})" if grp_id else "Unknown"

        # Check cache
        if grp_id in self.cache:
            card = self.cache[grp_id]
            if self.show_reskin_names and card.get("is_reskin"):
                # Show Spider-Man reskin name
                return card.get("name", f"Unknown({grp_id})")
            else:
                # Show OmenPaths canonical name (printed name)
                return card.get("printed_name") or card.get("name", f"Unknown({grp_id})")

        # Query database
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
            row = cursor.fetchone()

            if row:
                card = dict(row)
                self.cache[grp_id] = card

                if self.show_reskin_names and card.get("is_reskin"):
                    # Show Spider-Man reskin name
                    return card.get("name", f"Unknown({grp_id})")
                else:
                    # Show OmenPaths canonical name (printed name)
                    return card.get("printed_name") or card.get("name", f"Unknown({grp_id})")
            else:
                logging.debug(f"Card {grp_id} not in database")
                return f"Unknown({grp_id})"

        except Exception as e:
            logging.error(f"Database error for grpId {grp_id}: {e}")
            return f"Unknown({grp_id})"

    def get_card_data(self, grp_id: int) -> Optional[dict]:
        """Get full card data from grpId."""
        if not grp_id or not self.conn:
            return None

        # Check cache
        if grp_id in self.cache:
            return self.cache[grp_id]

        # Query database
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
            row = cursor.fetchone()

            if row:
                card = dict(row)
                self.cache[grp_id] = card
                return card

            return None

        except Exception as e:
            logging.error(f"Database error for grpId {grp_id}: {e}")
            return None

    def get_oracle_text(self, grp_id: int) -> str:
        """Get oracle text for a card."""
        card = self.get_card_data(grp_id)
        return card.get("oracle_text", "") if card else ""

    def get_mana_cost(self, grp_id: int) -> str:
        """Get mana cost for a card (e.g., '{2}{U}{U}')."""
        card = self.get_card_data(grp_id)
        return card.get("mana_cost", "") if card else ""

    def get_cmc(self, grp_id: int) -> Optional[float]:
        """Get converted mana cost for a card."""
        card = self.get_card_data(grp_id)
        cmc = card.get("cmc") if card else None
        return float(cmc) if cmc is not None else None

    def get_type_line(self, grp_id: int) -> str:
        """Get card type line (e.g., 'Creature - Elf Wizard')."""
        card = self.get_card_data(grp_id)
        return card.get("type_line", "") if card else ""

    def get_keywords(self, grp_id: int) -> List[str]:
        """Get ability keywords for a card (e.g., ['Flying', 'Lifelink'])."""
        card = self.get_card_data(grp_id)
        if not card:
            return []
        keywords_str = card.get("keywords", "")
        if not keywords_str:
            return []
        # Parse comma-separated keywords
        return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# ----------------------------------------------------------------------------------
# Part 5: Building Board State for AI
# ----------------------------------------------------------------------------------

class GameStateManager:
    def __init__(self, card_lookup: ArenaCardDatabase):
        self.scanner = MatchScanner()
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
                            inner_data = json.loads(parsed_data["Payload"])
                            logging.debug(f"Unpacked Payload for {event_type}")
                            parsed_data = inner_data
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse Payload JSON: {e}")

                    # Call the registered callback if it exists
                    if event_type in self._draft_callbacks:
                        print(f"[DEBUG] Calling callback for {event_type}")
                        self._draft_callbacks[event_type](parsed_data)
                    else:
                        print(f"[DEBUG] No callback registered for draft event: {event_type}")
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
                        print(f"[DEBUG] Draft.Notify detected: Pack {pack_num}, Pick {pick_num}")

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
                print(f"[DEBUG] Detected end event marker: {event_type}, expecting JSON on next line")
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
            has_priority=(self.scanner.priority_player_seat == your_seat_id),
            your_mana_pool=your_player.mana_pool.copy() if your_player.mana_pool else {},
            your_energy=your_player.energy,
            opponent_energy=opponent_player.energy,
            library_top_known=self.scanner.library_top_known.copy(),
            scry_info=self.scanner.scry_info,
            in_mulligan_phase=self.scanner.in_mulligan_phase,
            game_stage=self.scanner.game_stage
        )

        # Debug: Log all game objects before filtering
        logging.debug(f"Total game objects: {len(self.scanner.game_objects)}")
        for obj_id, obj in self.scanner.game_objects.items():
            logging.debug(f"  Object {obj_id}: grpId={obj.grp_id}, zoneId={obj.zone_id}, owner={obj.owner_seat_id}")

        # Get the actual zone IDs from the mappings discovered during parsing
        hand_zone_id = None
        battlefield_zone_id = None
        graveyard_zone_id = None
        exile_zone_id = None
        library_zone_id = None
        stack_zone_id = None

        for zone_type_str, zone_id in self.scanner.zone_type_to_ids.items():
            if "Hand" in zone_type_str:
                hand_zone_id = zone_id
            elif "Battlefield" in zone_type_str:
                battlefield_zone_id = zone_id
            elif "Graveyard" in zone_type_str:
                graveyard_zone_id = zone_id
            elif "Exile" in zone_type_str:
                exile_zone_id = zone_id
            elif "Library" in zone_type_str:
                library_zone_id = zone_id
            elif "Stack" in zone_type_str:
                stack_zone_id = zone_id

        logging.debug(f"Using zone mappings: Hand={hand_zone_id}, Battlefield={battlefield_zone_id}, "
                     f"Graveyard={graveyard_zone_id}, Exile={exile_zone_id}, Stack={stack_zone_id}")

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
                elif graveyard_zone_id and obj.zone_id == graveyard_zone_id:
                    logging.debug(f"  -> Added to your graveyard: {obj.name}")
                    board_state.your_graveyard.append(obj)
                elif exile_zone_id and obj.zone_id == exile_zone_id:
                    logging.debug(f"  -> Added to your exile: {obj.name}")
                    board_state.your_exile.append(obj)
                elif library_zone_id and obj.zone_id == library_zone_id:
                    board_state.your_library_count += 1
                else:
                    logging.debug(f"  -> Skipped your card (zone {obj.zone_id})")
            elif obj.owner_seat_id == opponent_seat_id:
                if battlefield_zone_id and obj.zone_id == battlefield_zone_id:
                    logging.debug(f"  -> Added to opponent battlefield: {obj.name}")
                    board_state.opponent_battlefield.append(obj)
                elif graveyard_zone_id and obj.zone_id == graveyard_zone_id:
                    logging.debug(f"  -> Added to opponent graveyard: {obj.name}")
                    board_state.opponent_graveyard.append(obj)
                elif exile_zone_id and obj.zone_id == exile_zone_id:
                    logging.debug(f"  -> Added to opponent exile: {obj.name}")
                    board_state.opponent_exile.append(obj)
                elif library_zone_id and obj.zone_id == library_zone_id:
                    board_state.opponent_library_count += 1
                else:
                    logging.debug(f"  -> Skipped opponent card (zone {obj.zone_id})")
            else:
                logging.debug(f"  -> Skipped card (owner {obj.owner_seat_id} not a player)")

            # Stack is shared between players
            if stack_zone_id and obj.zone_id == stack_zone_id:
                logging.debug(f"  -> Added to stack: {obj.name}")
                board_state.stack.append(obj)

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

            # Calculate library count based on deck size minus known zones
            # This is more accurate than relying on game_objects library count
            cards_seen = (len(board_state.your_hand) +
                         len(board_state.your_battlefield) +
                         len(board_state.your_graveyard) +
                         len(board_state.your_exile))
            board_state.your_library_count = max(0, total_deck_size - cards_seen)
            board_state.your_deck_remaining = board_state.your_library_count

            logging.debug(f"Deck tracking: {len(deck_with_names)} unique cards, deck size {total_deck_size}, "
                         f"seen {cards_seen} cards, {board_state.your_library_count} remaining in library")

        logging.info(f"Board State Summary: Hand: {len(board_state.your_hand)}, "
                    f"Battlefield: {len(board_state.your_battlefield)}, "
                    f"Graveyard: {len(board_state.your_graveyard)}, "
                    f"Exile: {len(board_state.your_exile)}, "
                    f"Library: {board_state.your_library_count}")
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

# ----------------------------------------------------------------------------------
# Part 6: AI Advice Generation
# ----------------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "mistral:7b"):
        self.host = host
        self.model = model

    def is_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_ollama(self) -> bool:
        """Try to start Ollama service"""
        try:
            import subprocess
            # Try to start ollama serve in the background
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait a moment for it to start
            import time
            time.sleep(2)
            return self.is_running()
        except Exception as e:
            logging.error(f"Failed to start Ollama: {e}")
            return False

    def generate(self, prompt: str) -> Optional[str]:
        logging.debug(f"Ollama prompt: {prompt[:500]}...")
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            logging.debug(f"Ollama raw response: {result}")
            return result.get('response', '').strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama error: {e}")
            return None

class AIAdvisor:
    SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor.

CRITICAL RULES:
1. ONLY reference cards explicitly listed in "YOUR HAND" or "YOUR BATTLEFIELD"
2. You CANNOT destroy lands (Forest, Plains, Swamp, Mountain, Island) - lands are permanent
3. You can only cast spells from YOUR HAND during your main phase
4. Creatures can attack if they've been on battlefield since your last turn
5. If you see "Unknown" cards, say "Wait for card identification"

FORBIDDEN ACTIONS:
- Do NOT mention cards not listed in the board state
- Do NOT suggest destroying/removing lands
- Do NOT invent card names

Give ONLY tactical advice in 1-2 short sentences. Start directly with your recommendation."""

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "mistral:7b", use_rag: bool = True, card_db: Optional[ArenaCardDatabase] = None):
        self.client = OllamaClient(host=ollama_host, model=model)
        self.use_rag = use_rag and RAG_AVAILABLE
        self.rag_system = None
        self.card_db = card_db  # Store card database for oracle text lookups
        self.last_rag_references = None  # Track most recent RAG references used

        # Initialize RAG system if enabled
        if self.use_rag:
            try:
                logging.info("Initializing RAG system...")
                self.rag_system = RAGSystem()
                # Only initialize rules if ChromaDB and embeddings are available
                if hasattr(self.rag_system.rules_db, 'client') and self.rag_system.rules_db.client:
                    logging.info("RAG system initialized with rules database")
                else:
                    logging.info("RAG system initialized (rules search disabled - install chromadb and sentence-transformers)")
            except Exception as e:
                logging.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
                self.use_rag = False
        else:
            if not RAG_AVAILABLE:
                logging.info("RAG system disabled (dependencies not installed)")
            else:
                logging.info("RAG system disabled by configuration")

    def get_tactical_advice(self, board_state: BoardState) -> Optional[str]:
        prompt = self._build_prompt(board_state)

        # Enhance prompt with RAG context if available
        if self.use_rag and self.rag_system:
            try:
                # Convert BoardState to dict format for RAG system
                board_dict = self._board_state_to_dict(board_state)

                # Use the enhanced method that returns references
                if hasattr(self.rag_system, 'enhance_prompt_with_references'):
                    prompt, self.last_rag_references = self.rag_system.enhance_prompt_with_references(board_dict, prompt)
                    logging.debug(f"Prompt enhanced with RAG context. References: {self.last_rag_references}")
                else:
                    # Fallback to old method if the new method isn't available
                    prompt = self.rag_system.enhance_prompt(board_dict, prompt)
                    logging.debug("Prompt enhanced with RAG context (references not tracked)")
            except Exception as e:
                logging.warning(f"Failed to enhance prompt with RAG: {e}")
                self.last_rag_references = None

        advice = self.client.generate(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        if advice:
            logging.debug(f"AI generated advice: {advice[:500]}...")
        else:
            logging.debug("AI did not generate any advice.")
        return advice

    def get_last_rag_references(self) -> Optional[Dict]:
        """Get the RAG references from the last tactical advice generation."""
        return self.last_rag_references

    def check_important_updates(self, board_state: BoardState, previous_board_state: Optional[BoardState]) -> Optional[str]:
        """
        Check if there are important changes that warrant immediate notification.
        Returns advice if important, None if not worth speaking.
        """
        if not previous_board_state:
            return None

        # Build a prompt asking the model to evaluate importance
        evaluation_prompt = f"""You are a tactical advisor monitoring a Magic: The Gathering game in progress.

PREVIOUS STATE (just before):
- Turn {previous_board_state.current_turn}, {previous_board_state.current_phase}
- Your life: {previous_board_state.your_life} | Opponent: {previous_board_state.opponent_life}
- Your battlefield: {len(previous_board_state.your_battlefield)} | Opponent: {len(previous_board_state.opponent_battlefield)}

CURRENT STATE (right now):
- Turn {board_state.current_turn}, {board_state.current_phase}
- Your life: {board_state.your_life} | Opponent: {board_state.opponent_life}
- Your battlefield: {len(board_state.your_battlefield)} | Opponent: {board_state.opponent_battlefield}

WHAT JUST HAPPENED:"""

        changes_detected = False
        # Add detected changes
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn:
                evaluation_prompt += f"\n- Cards played: {', '.join([c.name for c in history.cards_played_this_turn])}"
                changes_detected = True
            if history.died_this_turn:
                evaluation_prompt += f"\n- Creatures died: {', '.join(history.died_this_turn)}"
                changes_detected = True

        # Check life total changes (only significant ones)
        life_change = board_state.your_life - previous_board_state.your_life
        if life_change < -5:  # Only care about losing 5+ life
            evaluation_prompt += f"\n- Your life dropped by {abs(life_change)}"
            changes_detected = True
        elif life_change != 0:
            # Track it but don't necessarily alert
            evaluation_prompt += f"\n- Your life: {life_change:+d}"
            changes_detected = True

        opponent_life_change = board_state.opponent_life - previous_board_state.opponent_life
        if opponent_life_change < -5:  # Only care if opponent losing significant life
            evaluation_prompt += f"\n- Opponent life dropped by {abs(opponent_life_change)}"
            changes_detected = True

        # If no significant changes, don't even query
        if not changes_detected:
            return None

        evaluation_prompt += """

IMPORTANT: Most game events are NOT critical. Only alert if this is URGENT and the player must act NOW.

Is this TRULY critical? (Answer NO for 95% of changes)
- YES = Immediate lethal threat, opponent about to win, must counter/respond this instant
- NO = Everything else (normal plays, small damage, regular creatures, incremental advantage)

Examples of NOT critical:
- Opponent played a creature (unless it's lethal next turn)
- Lost 3-4 life (that's normal)
- Opponent gained some life
- A single creature died

Examples of CRITICAL:
- Opponent has exact lethal damage on board ready to attack
- Opponent played a game-ending combo piece
- You're at 2 life and they have burn spell

Respond in EXACTLY this format:
- If critical: "ALERT: [one sentence warning]"
- If not critical: "NO"

Your response:"""

        response = self.client.generate(evaluation_prompt)
        if response:
            response = response.strip()
            logging.debug(f"Importance check response: {response}")

            if response.startswith("ALERT:"):
                # Extract the advice part after "ALERT:"
                advice = response[6:].strip()
                logging.info(f"Critical update detected: {advice}")
                return advice
            elif "ALERT:" in response:
                # Handle case where model adds extra text before ALERT:
                alert_start = response.find("ALERT:")
                advice = response[alert_start + 6:].strip()
                # Remove any trailing quotes or punctuation artifacts
                advice = advice.strip('"\'.,!?')
                logging.info(f"Critical update detected (extracted): {advice}")
                return advice

        return None

    def _board_state_to_dict(self, board_state: BoardState) -> Dict:
        """Convert BoardState to dictionary format for RAG system."""
        return {
            'phase': board_state.current_phase,
            'turn': board_state.current_turn,
            'battlefield': {
                'player': [{'name': card.name} for card in board_state.your_battlefield],
                'opponent': [{'name': card.name} for card in board_state.opponent_battlefield]
            },
            'hand': [{'name': card.name} for card in board_state.your_hand],
            'graveyard': {
                'player': [{'name': card.name} for card in board_state.your_graveyard],
                'opponent': [{'name': card.name} for card in board_state.opponent_graveyard]
            },
            'stack': [{'name': card.name} for card in board_state.stack],
            'stack_size': len(board_state.stack)
        }

    def _build_prompt(self, board_state: BoardState) -> str:
        """Build comprehensive prompt with all zones and game history"""
        lines = [
            f"== GAME STATE: Turn {board_state.current_turn}, {board_state.current_phase} Phase ==",
            f"Your life: {board_state.your_life} | Opponent life: {board_state.opponent_life}",
            f"Your library: {board_state.your_library_count} cards | Opponent library: {board_state.opponent_library_count} cards",
            "",
        ]

        # Game History - what happened this turn
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn or history.died_this_turn or history.lands_played_this_turn:
                lines.append("== THIS TURN ==")
                if history.cards_played_this_turn:
                    played_names = [c.name for c in history.cards_played_this_turn]
                    lines.append(f"Cards played: {', '.join(played_names)}")
                if history.lands_played_this_turn > 0:
                    lines.append(f"Lands played: {history.lands_played_this_turn}")
                if history.died_this_turn:
                    lines.append(f"Creatures died: {', '.join(history.died_this_turn)}")
                lines.append("")

        # Hand - with mana cost, type, and oracle text
        if board_state.your_hand:
            lines.append(f"== YOUR HAND ({len(board_state.your_hand)}) ==")
            for card in board_state.your_hand:
                card_info = f"• {card.name}"

                # Add card details if available
                if self.card_db and card.grp_id:
                    mana_cost = self.card_db.get_mana_cost(card.grp_id)
                    type_line = self.card_db.get_type_line(card.grp_id)
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)

                    # Format: Name {COST} (Type)
                    if mana_cost or type_line:
                        details = []
                        if mana_cost:
                            details.append(mana_cost)
                        if type_line:
                            details.append(f"({type_line})")
                        card_info += f" {' '.join(details)}"

                    # Add oracle text on separate line
                    if oracle_text:
                        card_info += f"\n  Rules: {oracle_text}"

                lines.append(card_info)
            lines.append("")
        else:
            lines.append("== YOUR HAND == (empty)")
            lines.append("")

        # Battlefield - with tapped/untapped status, tactical details, and oracle text
        if board_state.your_battlefield:
            lines.append(f"== YOUR BATTLEFIELD ({len(board_state.your_battlefield)}) ==")
            for card in board_state.your_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.summoning_sick:
                    status_flags.append("summoning sick")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")
                if card.attached_to:
                    status_flags.append(f"attached to instance {card.attached_to}")

                status_text = f" ({', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                card_line = f"• {card.name}{power_toughness}{status_text}"

                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_line += f"\n  ({oracle_text})"

                lines.append(card_line)
            lines.append("")
        else:
            lines.append("== YOUR BATTLEFIELD == (empty)")
            lines.append("")

        if board_state.opponent_battlefield:
            lines.append(f"== OPPONENT BATTLEFIELD ({len(board_state.opponent_battlefield)}) ==")
            for card in board_state.opponent_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")

                status_text = f" ({', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                lines.append(f"• {card.name}{power_toughness}{status_text}")
            lines.append("")
        else:
            lines.append("== OPPONENT BATTLEFIELD == (empty)")
            lines.append("")

        # Graveyard (show last 5 cards for recursion opportunities)
        your_gy = [card.name for card in board_state.your_graveyard]
        if your_gy:
            recent_gy = your_gy[-5:] if len(your_gy) > 5 else your_gy
            lines.append(f"== YOUR GRAVEYARD ({len(your_gy)} total, recent: {', '.join(recent_gy)}) ==")
            lines.append("")

        opp_gy = [card.name for card in board_state.opponent_graveyard]
        if opp_gy:
            recent_opp_gy = opp_gy[-5:] if len(opp_gy) > 5 else opp_gy
            lines.append(f"== OPPONENT GRAVEYARD ({len(opp_gy)} total, recent: {', '.join(recent_opp_gy)}) ==")
            lines.append("")

        # Exile
        your_exile = [card.name for card in board_state.your_exile]
        if your_exile:
            lines.append(f"== YOUR EXILE == {', '.join(your_exile)}")
            lines.append("")

        opp_exile = [card.name for card in board_state.opponent_exile]
        if opp_exile:
            lines.append(f"== OPPONENT EXILE == {', '.join(opp_exile)}")
            lines.append("")

        # Stack (active spells/abilities)
        if board_state.stack:
            stack_cards = [card.name for card in board_state.stack]
            lines.append(f"== STACK == {', '.join(stack_cards)}")
            lines.append("")

        # Count available mana (UNTAPPED lands + floating mana in pool)
        your_lands = [card for card in board_state.your_battlefield
                      if not card.is_tapped and  # Only untapped lands!
                      ("land" in card.name.lower() or
                       any(land_type in card.name.lower() for land_type in ["swamp", "forest", "plains", "mountain", "island"]))]

        # Count floating mana in pool
        floating_mana = sum(board_state.your_mana_pool.values()) if board_state.your_mana_pool else 0
        total_available_mana = floating_mana  # For use in prompt text

        # Build detailed mana breakdown
        lines.append(f"== AVAILABLE MANA ==")

        # Show floating mana first (this is what's actually available RIGHT NOW)
        if board_state.your_mana_pool and any(v > 0 for v in board_state.your_mana_pool.values()):
            mana_str = ", ".join([f"{count}{color}" for color, count in sorted(board_state.your_mana_pool.items()) if count > 0])
            lines.append(f"In pool (castable NOW): {mana_str} ({floating_mana} total)")
        else:
            lines.append(f"In pool (castable NOW): None (0 total)")

        # Show potential mana from untapped lands
        if your_lands:
            lines.append(f"Untapped lands: {len(your_lands)} (can tap for mana)")

        # Show energy if present
        if board_state.your_energy > 0:
            lines.append(f"Energy: {board_state.your_energy}")

        lines.append("")

        # Combat state (if in combat)
        if board_state.history and "Combat" in board_state.current_phase:
            history = board_state.history
            if history.current_attackers:
                lines.append("== COMBAT STATE ==")

                # Check if attackers are yours or opponent's
                your_attacker_names = []
                opponent_attacker_names = []
                for attacker_id in history.current_attackers:
                    # Check your battlefield
                    attacker = next((c for c in board_state.your_battlefield if c.instance_id == attacker_id), None)
                    if attacker:
                        power_tough = f" [{attacker.power}/{attacker.toughness}]" if attacker.power is not None else ""
                        your_attacker_names.append(f"{attacker.name}{power_tough}")
                    else:
                        # Check opponent's battlefield
                        attacker = next((c for c in board_state.opponent_battlefield if c.instance_id == attacker_id), None)
                        if attacker:
                            power_tough = f" [{attacker.power}/{attacker.toughness}]" if attacker.power is not None else ""
                            opponent_attacker_names.append(f"{attacker.name}{power_tough}")

                if your_attacker_names:
                    lines.append(f"Your attackers: {', '.join(your_attacker_names)}")

                if opponent_attacker_names:
                    total_damage = sum(attacker.power for attacker in board_state.opponent_battlefield
                                     if attacker.instance_id in history.current_attackers and attacker.power is not None)
                    lines.append(f"⚔️ OPPONENT ATTACKING with {len(opponent_attacker_names)} creatures ({total_damage} total damage):")
                    for name in opponent_attacker_names:
                        lines.append(f"  • {name}")

                if history.current_blockers:
                    lines.append("Blockers assigned:")
                    for attacker_id, blocker_id in history.current_blockers.items():
                        attacker = next((c for c in board_state.your_battlefield if c.instance_id == attacker_id), None)
                        blocker = next((c for c in board_state.opponent_battlefield if c.instance_id == blocker_id), None)
                        if attacker and blocker:
                            lines.append(f"  {blocker.name} blocks {attacker.name}")
                lines.append("")

        # Scry information (known top library cards)
        if board_state.scry_info:
            lines.append("== KNOWN CARDS ==")
            lines.append(board_state.scry_info)
            if board_state.library_top_known:
                lines.append(f"Top of library: {', '.join(board_state.library_top_known)}")
            lines.append("")

        # Deck composition and draw probabilities (for planning only)
        if board_state.your_decklist and board_state.your_deck_remaining > 0:
            lines.append("== LIBRARY COMPOSITION (for probability planning) ==")
            lines.append(f"{board_state.your_deck_remaining} cards remaining in library")

            # Calculate what's still in the deck (deck minus seen cards)
            cards_remaining = {}
            for card_name, original_count in board_state.your_decklist.items():
                # Count how many we've seen across all zones
                seen_count = 0
                seen_count += sum(1 for c in board_state.your_hand if c.name == card_name)
                seen_count += sum(1 for c in board_state.your_battlefield if c.name == card_name)
                seen_count += sum(1 for c in board_state.your_graveyard if c.name == card_name)
                seen_count += sum(1 for c in board_state.your_exile if c.name == card_name)

                remaining = original_count - seen_count
                if remaining > 0:
                    cards_remaining[card_name] = remaining

            # Show top 10 most likely draws (by count)
            if cards_remaining:
                sorted_cards = sorted(cards_remaining.items(), key=lambda x: x[1], reverse=True)
                top_draws = sorted_cards[:10]
                draw_str = ", ".join([f"{name} ({count}x)" for name, count in top_draws])
                lines.append(f"Most likely draws: {draw_str}")

            lines.append("")
            lines.append("NOTE: This shows probability of FUTURE draws only. You CANNOT play these cards this turn.")
            lines.append("")

        # Final question with explicit constraints
        lines.append("== QUESTION ==")

        # Check if opponent is attacking (needs blocking advice)
        opponent_attacking = (board_state.history and
                            len(board_state.history.current_attackers) > 0 and
                            any(c.instance_id in board_state.history.current_attackers
                                for c in board_state.opponent_battlefield))

        if opponent_attacking:
            # Blocking advice
            lines.append("⚔️ BLOCKING DECISION REQUIRED!")
            lines.append(f"The opponent is attacking with {len([c for c in board_state.opponent_battlefield if c.instance_id in board_state.history.current_attackers])} creatures.")
            lines.append("")
            lines.append("Analyze the combat situation:")
            lines.append("1. Which creatures should you block with? Consider:")
            lines.append("   - Favorable trades (killing their creature while saving yours)")
            lines.append("   - Preventing lethal damage")
            lines.append("   - Saving valuable creatures")
            lines.append("   - Abilities that trigger on blocking/damage")
            lines.append("2. Should you take some damage to preserve blockers?")
            lines.append("3. Can you use any instant-speed spells or abilities to improve combat?")
            lines.append("")
            lines.append(f"You have {total_available_mana} mana available for instant-speed responses.")
            lines.append("")
            lines.append("Provide specific blocking recommendations with reasoning for each choice.")
        else:
            # Normal turn advice
            lines.append("Using ONLY the cards in YOUR HAND and YOUR BATTLEFIELD listed above, ")
            lines.append(f"and considering you have {total_available_mana} mana available, ")
            lines.append("what is the optimal tactical play right now?")

        lines.append("")
        lines.append("REMINDER: You can ONLY cast spells from YOUR HAND. Do not reference any cards not explicitly listed above.")

        return "\n".join(lines)

# ----------------------------------------------------------------------------------
# Part 7: Text-to-Speech Output
# ----------------------------------------------------------------------------------

class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0, force_engine: str = None):
        """
        Initialize TTS with Kokoro as primary, BarkTTS as fallback.

        Args:
            voice: Voice name
            volume: Volume (0.0-1.0)
            force_engine: Force specific engine ("kokoro" or "bark"), or None for auto-fallback
        """
        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))  # Clamp volume to 0.0-1.0
        self.tts_engine = None  # Will be "kokoro" or "bark"
        self.tts = None
        self.bark_processor = None
        self.bark_model = None

        if force_engine == "bark":
            # Force BarkTTS
            logging.info(f"Forcing BarkTTS engine")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return
            logging.error("❌ Failed to initialize BarkTTS")
        elif force_engine == "kokoro":
            # Force Kokoro
            logging.info(f"Forcing Kokoro engine with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return
            logging.error("❌ Failed to initialize Kokoro TTS")
        else:
            # Try Kokoro first (primary), then fall back
            logging.info(f"Attempting to initialize Kokoro TTS (primary) with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return

            # Fall back to BarkTTS
            logging.warning("Kokoro TTS failed, falling back to BarkTTS (secondary)")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return

            # No TTS available
            logging.error("❌ Failed to initialize any TTS engine (Kokoro and Bark both failed)")

    def _init_kokoro(self) -> bool:
        """Try to initialize Kokoro TTS. Returns True on success."""
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
            self.tts_engine = "kokoro"
            return True
        except Exception as e:
            logging.debug(f"Kokoro initialization failed: {e}")
            return False

    def _init_bark(self) -> bool:
        """Try to initialize BarkTTS. Returns True on success."""
        try:
            from transformers import AutoProcessor, BarkModel
            import numpy as np
            import torch

            self.np = np
            self.torch = torch

            # Load Bark model and processor
            logging.info("Loading BarkTTS model (this may take a moment)...")
            self.bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.bark_model = BarkModel.from_pretrained("suno/bark-small")

            # Move to GPU if available
            if torch.cuda.is_available():
                self.bark_model = self.bark_model.to("cuda")
                logging.info("BarkTTS using GPU acceleration")

            self.tts_engine = "bark"
            return True
        except Exception as e:
            logging.debug(f"BarkTTS initialization failed: {e}")
            return False

    def set_voice(self, voice: str):
        """Change voice dynamically"""
        self.voice = voice
        logging.info(f"Voice changed to: {voice}")

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        logging.info(f"Volume changed to: {self.volume}")

    def speak(self, text: str):
        """Speak text using available TTS engine (Kokoro or Bark)"""
        if not text:
            logging.debug("No text provided to speak.")
            return

        if not self.tts_engine:
            logging.error("No TTS engine initialized, cannot speak.")
            return

        # Route to appropriate TTS engine
        if self.tts_engine == "kokoro":
            self._speak_kokoro(text)
        elif self.tts_engine == "bark":
            self._speak_bark(text)

    def _speak_kokoro(self, text: str):
        """Speak using Kokoro TTS"""
        logging.info(f"Speaking with Kokoro ({self.voice}): {text[:100]}...")
        try:
            # Generate audio using Kokoro
            audio_array, sample_rate = self.tts.create(text, voice=self.voice, speed=1.0)

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "Kokoro")
            logging.debug("Successfully spoke text with Kokoro.")
        except Exception as e:
            logging.error(f"Kokoro TTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _speak_bark(self, text: str):
        """Speak using BarkTTS"""
        logging.info(f"Speaking with BarkTTS ({self.voice}): {text[:100]}...")
        try:
            # Process text input
            inputs = self.bark_processor(text, voice_preset=self.voice)

            # Move inputs to same device as model
            if self.torch.cuda.is_available():
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate audio
            with self.torch.no_grad():
                audio_array = self.bark_model.generate(**inputs)

            # Convert to numpy and get sample rate
            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = self.bark_model.generation_config.sample_rate

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "BarkTTS")
            logging.debug("Successfully spoke text with BarkTTS.")
        except Exception as e:
            logging.error(f"BarkTTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _save_and_play_audio(self, audio_array, sample_rate: int, engine_name: str):
        """Save audio to temp file and play it"""
        import scipy.io.wavfile as wavfile

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, sample_rate, (audio_array * 32767).astype(self.np.int16))

        logging.info(f"Generated audio saved to {tmp_path}, playing...")

        # Try different audio players
        played = False
        players = [
            (["aplay", tmp_path], "aplay"),
            (["paplay", tmp_path], "paplay"),
            (["ffplay", "-nodisp", "-autoexit", tmp_path], "ffplay")
        ]

        for cmd, player_name in players:
            try:
                subprocess.run(cmd, check=True, timeout=120,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                played = True
                logging.info(f"Audio played with {player_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.debug(f"{player_name} error: {e}")
                continue

        if not played:
            logging.error("No audio player found (aplay, paplay, or ffplay). Cannot play audio.")

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

# ----------------------------------------------------------------------------------
# Part 8: TUI (Text User Interface)
# ----------------------------------------------------------------------------------

class AdvisorTUI:
    """
    Text User Interface for MTGA Voice Advisor using curses.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Status Bar: Turn 5 | Model: llama3.2 | Voice: am_adam     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ Board State Window (scrollable)                            │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ Messages Window (scrollable)                               │
    │ - Advisor responses                                        │
    │ - Game events                                              │
    │ - Command feedback                                         │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │ You: _                                                      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.board_state_lines = []
        self.messages = deque(maxlen=100)  # Keep last 100 messages
        self.input_buffer = ""
        self.input_callback = None

        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Green
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Cyan
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Yellow
        curses.init_pair(4, curses.COLOR_RED, -1)     # Red
        curses.init_pair(5, curses.COLOR_BLUE, -1)    # Blue
        curses.init_pair(6, curses.COLOR_WHITE, -1)   # White

        # Color name to pair ID mapping
        self.color_map = {
            "green": 1,
            "cyan": 2,
            "yellow": 3,
            "red": 4,
            "blue": 5,
            "white": 6,
        }

        # Configure stdscr
        self.stdscr.keypad(True)
        curses.curs_set(1)  # Show cursor

        # Create windows
        self._create_windows()

    def _create_windows(self):
        """Create and layout windows"""
        height, width = self.stdscr.getmaxyx()

        # Status bar: 1 line at top
        self.status_win = curses.newwin(1, width, 0, 0)

        # Board state: 70% of available height (more space for full board display)
        available_height = height - 3  # Minus status, separator, input
        board_height = max(10, int(available_height * 0.7))
        self.board_win = curses.newwin(board_height, width, 1, 0)
        self.board_win.scrollok(True)

        # Messages: remaining 30% of space
        msg_y = 1 + board_height
        msg_height = height - msg_y - 2
        self.msg_win = curses.newwin(msg_height, width, msg_y, 0)
        self.msg_win.scrollok(True)

        # Input prompt: 1 line at bottom
        self.input_win = curses.newwin(1, width, height - 1, 0)

        # Scroll positions
        self.board_scroll = 0
        self.msg_scroll = 0

    def resize(self):
        """Handle terminal resize"""
        try:
            # Update curses internal tracking of terminal size
            curses.update_lines_cols()

            # Clear and refresh main screen
            self.stdscr.clear()
            self.stdscr.refresh()

            # Recreate windows with new dimensions
            self._create_windows()

            # Redraw everything
            self.refresh_all()
        except Exception as e:
            # Silently handle resize errors
            pass

    def set_status(self, text: str):
        """Update status bar"""
        try:
            self.status_win.clear()
            self.status_win.addstr(0, 0, text[:self.status_win.getmaxyx()[1]-1],
                                  curses.color_pair(1) | curses.A_BOLD)
            self.status_win.refresh()
        except curses.error:
            pass

    def set_board_state(self, lines: List[str]):
        """Update board state display"""
        self.board_state_lines = lines
        self._refresh_board()

    def add_message(self, msg: str, color = 0):
        """Add message to message log. Color can be string name or int pair ID."""
        timestamp = time.strftime("%H:%M:%S")
        # Convert color string to int if needed
        if isinstance(color, str):
            color = self.color_map.get(color, 6)  # Default to white
        self.messages.append((timestamp, msg, color))
        # Auto-scroll to bottom
        self.msg_scroll = max(0, len(self.messages) - self.msg_win.getmaxyx()[0])
        self._refresh_messages()

    def _refresh_board(self):
        """Redraw board state window"""
        try:
            self.board_win.clear()
            height, width = self.board_win.getmaxyx()

            # Draw border
            try:
                self.board_win.addstr(0, 0, "═" * (width-1), curses.color_pair(2))
                self.board_win.addstr(0, 2, " BOARD STATE ", curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass

            # Draw visible lines
            visible_lines = self.board_state_lines[self.board_scroll:self.board_scroll + height - 1]
            for i, line in enumerate(visible_lines):
                try:
                    # Truncate to fit width
                    display_line = line[:width-1]
                    self.board_win.addstr(i + 1, 0, display_line)
                except curses.error:
                    pass

            self.board_win.refresh()
        except curses.error:
            pass

    def _refresh_messages(self):
        """Redraw messages window"""
        try:
            self.msg_win.clear()
            height, width = self.msg_win.getmaxyx()

            # Draw border
            try:
                self.msg_win.addstr(0, 0, "═" * (width-1), curses.color_pair(2))
                self.msg_win.addstr(0, 2, " MESSAGES ", curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass

            # Draw visible messages
            visible_msgs = list(self.messages)[self.msg_scroll:self.msg_scroll + height - 1]
            for i, (timestamp, msg, color) in enumerate(visible_msgs):
                try:
                    # Format: [HH:MM:SS] message
                    display_line = f"[{timestamp}] {msg}"[:width-1]
                    attr = curses.color_pair(color) if color else 0
                    self.msg_win.addstr(i + 1, 0, display_line, attr)
                except curses.error:
                    pass

            self.msg_win.refresh()
        except curses.error:
            pass

    def _refresh_input(self):
        """Redraw input prompt"""
        try:
            self.input_win.clear()
            width = self.input_win.getmaxyx()[1]

            # Show prompt and input buffer
            prompt = "You: "
            display = prompt + self.input_buffer

            # Truncate if too long
            if len(display) >= width:
                display = prompt + "..." + self.input_buffer[-(width-len(prompt)-4):]

            self.input_win.addstr(0, 0, display)
            self.input_win.refresh()
        except curses.error:
            pass

    def refresh_all(self):
        """Refresh all windows"""
        try:
            self.status_win.refresh()
        except:
            pass
        self._refresh_board()
        self._refresh_messages()
        self._refresh_input()

    def get_input(self, callback: Callable[[str], None]):
        """
        Get user input (non-blocking with callback).
        Call this in a loop to handle input.
        """
        self.input_callback = callback
        self._refresh_input()

        try:
            # Non-blocking input
            self.stdscr.timeout(100)  # 100ms timeout
            ch = self.stdscr.getch()

            if ch == -1:  # No input
                return True

            if ch == curses.KEY_RESIZE:
                self.resize()
            elif ch == ord('\n') or ch == curses.KEY_ENTER or ch == 10:
                # Enter key - submit input
                if self.input_buffer.strip():
                    user_input = self.input_buffer
                    self.input_buffer = ""
                    self._refresh_input()
                    if self.input_callback:
                        self.input_callback(user_input)
            elif ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
                # Backspace
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                    self._refresh_input()
            elif ch == curses.KEY_UP:
                # Scroll board up
                self.board_scroll = max(0, self.board_scroll - 1)
                self._refresh_board()
            elif ch == curses.KEY_DOWN:
                # Scroll board down
                max_scroll = max(0, len(self.board_state_lines) - self.board_win.getmaxyx()[0] + 1)
                self.board_scroll = min(max_scroll, self.board_scroll + 1)
                self._refresh_board()
            elif ch == curses.KEY_PPAGE:  # Page Up
                # Scroll messages up
                self.msg_scroll = max(0, self.msg_scroll - 5)
                self._refresh_messages()
            elif ch == curses.KEY_NPAGE:  # Page Down
                # Scroll messages down
                max_scroll = max(0, len(self.messages) - self.msg_win.getmaxyx()[0] + 1)
                self.msg_scroll = min(max_scroll, self.msg_scroll + 5)
                self._refresh_messages()
            elif 32 <= ch <= 126:  # Printable ASCII
                self.input_buffer += chr(ch)
                self._refresh_input()

            return self.running

        except KeyboardInterrupt:
            self.running = False
            return False

    def show_popup(self, lines: List[str], title: str = ""):
        """Show a temporary popup overlay (press any key to dismiss)"""
        try:
            if not lines:
                return

            height, width = self.stdscr.getmaxyx()

            # Calculate popup dimensions (80% of screen)
            popup_height = min(len(lines) + 4, int(height * 0.8))
            max_line_len = max((len(line) for line in lines), default=20)
            popup_width = min(max_line_len + 4, int(width * 0.8))

            # Center the popup
            y = (height - popup_height) // 2
            x = (width - popup_width) // 2

            # Create popup window with border
            popup = curses.newwin(popup_height, popup_width, y, x)
            popup.box()

            # Add title if provided
            if title:
                popup.addstr(0, 2, f" {title} ", curses.color_pair(2) | curses.A_BOLD)

            # Add content (scrollable if needed)
            max_content_lines = popup_height - 3
            for i, line in enumerate(lines[:max_content_lines]):
                try:
                    popup.addstr(i + 1, 2, line[:popup_width - 4])
                except curses.error:
                    pass

            # Add footer
            footer = "Press any key to close"
            popup.addstr(popup_height - 1, (popup_width - len(footer)) // 2,
                        footer, curses.color_pair(3))

            popup.refresh()

            # Wait for keypress (blocking)
            self.stdscr.timeout(-1)  # Blocking mode
            self.stdscr.getch()
            self.stdscr.timeout(100)  # Back to non-blocking

            # Clear popup and refresh screen
            del popup
            self.stdscr.touchwin()
            self.refresh_all()

        except Exception as e:
            pass

    def show_settings_menu(self, settings_callback):
        """
        Show interactive settings menu.

        Args:
            settings_callback: Function to call with (setting_name, new_value)

        Returns tuple of (models_list, kokoro_voices_list, bark_voices_list, current_model, current_voice, current_volume, current_tts)
        """
        try:
            height, width = self.stdscr.getmaxyx()

            # Create popup (60% of screen)
            popup_height = min(20, int(height * 0.6))
            popup_width = min(70, int(width * 0.7))
            y = (height - popup_height) // 2
            x = (width - popup_width) // 2

            popup = curses.newwin(popup_height, popup_width, y, x)
            popup.keypad(True)

            # Get initial values from callback
            result = settings_callback("get_values", None)
            models, kokoro_voices, bark_voices, current_model, current_voice, current_volume, current_tts = result

            selected_idx = 0
            settings_items = ["AI Model", "Voice", "Volume", "TTS Engine"]

            while True:
                popup.clear()
                popup.box()
                popup.addstr(0, 2, " Settings ", curses.color_pair(2) | curses.A_BOLD)

                # Display settings with selection
                for i, item in enumerate(settings_items):
                    line_y = i + 2

                    # Highlight selected item
                    attr = curses.A_REVERSE if i == selected_idx else 0

                    if item == "AI Model":
                        value = current_model
                        hint = " (Enter to cycle)"
                    elif item == "Voice":
                        value = current_voice
                        hint = " (Enter to cycle)"
                    elif item == "Volume":
                        value = f"{current_volume}%"
                        hint = " (+/- to adjust)"
                    elif item == "TTS Engine":
                        value = "Kokoro" if current_tts == "kokoro" else "BarkTTS"
                        hint = " (Enter to toggle)"

                    display = f"  {item:15} {value}{hint}"
                    try:
                        popup.addstr(line_y, 2, display[:popup_width-4], attr)
                    except curses.error:
                        pass

                # Footer
                footer_y = popup_height - 2
                popup.addstr(footer_y, 2, "↑↓: Navigate  Enter: Change  ESC/Q: Close", curses.color_pair(3))

                popup.refresh()

                # Handle input
                ch = popup.getch()

                if ch == 27 or ch == ord('q') or ch == ord('Q'):  # ESC or Q
                    break
                elif ch == curses.KEY_UP:
                    selected_idx = (selected_idx - 1) % len(settings_items)
                elif ch == curses.KEY_DOWN:
                    selected_idx = (selected_idx + 1) % len(settings_items)
                elif ch == ord('\n') or ch == ord(' '):  # Enter or Space
                    setting = settings_items[selected_idx]

                    if setting == "AI Model":
                        # Cycle to next model
                        current_idx = models.index(current_model) if current_model in models else 0
                        new_idx = (current_idx + 1) % len(models)
                        current_model = models[new_idx]
                        settings_callback("model", current_model)

                    elif setting == "Voice":
                        # Cycle to next voice (use appropriate voice list)
                        voice_list = kokoro_voices if current_tts == "kokoro" else bark_voices
                        current_idx = voice_list.index(current_voice) if current_voice in voice_list else 0
                        new_idx = (current_idx + 1) % len(voice_list)
                        current_voice = voice_list[new_idx]
                        settings_callback("voice", current_voice)

                    elif setting == "TTS Engine":
                        # Toggle TTS engine
                        new_tts = "bark" if current_tts == "kokoro" else "kokoro"
                        current_tts = new_tts
                        settings_callback("tts_engine", new_tts)

                elif ch == ord('+') or ch == ord('='):  # Volume up
                    if settings_items[selected_idx] == "Volume":
                        current_volume = min(100, current_volume + 10)
                        settings_callback("volume", current_volume)

                elif ch == ord('-') or ch == ord('_'):  # Volume down
                    if settings_items[selected_idx] == "Volume":
                        current_volume = max(0, current_volume - 10)
                        settings_callback("volume", current_volume)

            # Cleanup
            del popup
            self.stdscr.touchwin()
            self.refresh_all()

        except Exception as e:
            pass

    def cleanup(self):
        """Cleanup curses"""
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

# ----------------------------------------------------------------------------------
# Part 9: Tkinter GUI
# ----------------------------------------------------------------------------------

class AdvisorGUI:
    def __init__(self, root, advisor_ref):
        self.root = root
        self.advisor = advisor_ref

        # Load user preferences for GUI settings persistence
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()
            logging.debug("User preferences loaded for GUI mode")

        # Configure root window
        self.root.title("MTGA Voice Advisor")

        # Apply window geometry from prefs (or use default)
        if self.prefs:
            geometry = self.prefs.window_geometry
        else:
            geometry = "900x700"
        self.root.geometry(geometry)

        # Apply always_on_top setting (default True)
        always_on_top = self.prefs.always_on_top if self.prefs else True
        self.root.attributes('-topmost', always_on_top)

        self.root.configure(bg='#2b2b2b')

        # Color scheme
        self.bg_color = '#2b2b2b'
        self.fg_color = '#ffffff'
        self.accent_color = '#00ff88'
        self.warning_color = '#ff5555'
        self.info_color = '#55aaff'

        self._create_widgets()

        # Message queue for thread-safe updates
        self.message_queue = deque(maxlen=100)
        self.board_state_lines = []
        self.rag_panel_expanded = False  # Track RAG panel expansion state

        # Bind F12 for bug reports
        self.root.bind('<F12>', lambda e: self._capture_bug_report())

        # Bind window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start update loop
        self.running = True
        self._update_loop()

    def _create_widgets(self):
        """Create all GUI widgets"""

        # Top status bar
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=30)
        status_frame.pack(side=tk.TOP, fill=tk.X)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Initializing...",
            bg='#1a1a1a',
            fg=self.accent_color,
            font=('Consolas', 10, 'bold'),
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Settings panel (left side)
        settings_frame = tk.Frame(self.root, bg=self.bg_color, width=250)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        settings_frame.pack_propagate(False)

        tk.Label(
            settings_frame,
            text="⚙ SETTINGS",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 12, 'bold')
        ).pack(pady=(0, 10))

        # Model selection
        tk.Label(settings_frame, text="AI Model:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(settings_frame, textvariable=self.model_var, width=25)
        self.model_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.model_dropdown.bind('<<ComboboxSelected>>', self._on_model_change)
        self.model_dropdown.bind('<Return>', self._on_model_change)  # Allow Enter key to confirm

        # Voice selection
        tk.Label(settings_frame, text="Voice:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.voice_var = tk.StringVar()
        self.voice_dropdown = ttk.Combobox(settings_frame, textvariable=self.voice_var, state='readonly', width=25)
        self.voice_dropdown.pack(pady=(0, 10), fill=tk.X)
        self.voice_dropdown.bind('<<ComboboxSelected>>', self._on_voice_change)

        # TTS Engine toggle
        tk.Label(settings_frame, text="TTS Engine:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        self.tts_engine_var = tk.StringVar(value="Kokoro")
        tts_frame = tk.Frame(settings_frame, bg=self.bg_color)
        tts_frame.pack(pady=(0, 10), fill=tk.X)

        tk.Radiobutton(
            tts_frame,
            text="Kokoro",
            variable=self.tts_engine_var,
            value="Kokoro",
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            command=self._on_tts_engine_change
        ).pack(side=tk.LEFT)

        tk.Radiobutton(
            tts_frame,
            text="BarkTTS",
            variable=self.tts_engine_var,
            value="BarkTTS",
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            command=self._on_tts_engine_change
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Volume slider
        tk.Label(settings_frame, text="Volume:", bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        volume_frame = tk.Frame(settings_frame, bg=self.bg_color)
        volume_frame.pack(pady=(0, 10), fill=tk.X)

        self.volume_var = tk.IntVar(value=100)
        self.volume_slider = tk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=self._on_volume_change,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            troughcolor='#1a1a1a'
        )
        self.volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.volume_label = tk.Label(volume_frame, text="100%", bg=self.bg_color, fg=self.fg_color, width=5)
        self.volume_label.pack(side=tk.RIGHT)

        # Opponent Turn Alerts checkbox (renamed from "Continuous Monitoring")
        opponent_alerts_default = self.prefs.opponent_turn_alerts if self.prefs else True
        self.continuous_var = tk.BooleanVar(value=opponent_alerts_default)
        tk.Checkbutton(
            settings_frame,
            text="Opponent Turn Alerts",
            variable=self.continuous_var,
            command=self._on_continuous_toggle,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Show thinking checkbox
        show_thinking_default = self.prefs.show_thinking if self.prefs else True
        self.show_thinking_var = tk.BooleanVar(value=show_thinking_default)
        tk.Checkbutton(
            settings_frame,
            text="Show AI Thinking",
            variable=self.show_thinking_var,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Show Spider-Man Reskins checkbox
        reskin_default = self.prefs.reskin_names if self.prefs else False
        self.reskin_var = tk.BooleanVar(value=reskin_default)
        tk.Checkbutton(
            settings_frame,
            text="Show Spider-Man Reskins",
            variable=self.reskin_var,
            command=self._on_reskin_toggle,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Always on top checkbox
        always_on_top_default = self.prefs.always_on_top if self.prefs else True
        self.always_on_top_var = tk.BooleanVar(value=always_on_top_default)
        tk.Checkbutton(
            settings_frame,
            text="Always on Top",
            variable=self.always_on_top_var,
            command=self._on_always_on_top_toggle,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Pick Two Draft checkbox
        self.pick_two_draft_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            settings_frame,
            text="Pick Two Draft",
            variable=self.pick_two_draft_var,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor='#1a1a1a',
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        ).pack(anchor=tk.W, pady=5)

        # Buttons
        tk.Button(
            settings_frame,
            text="Clear Messages",
            command=self._clear_messages,
            bg='#3a3a3a',
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=(20, 5), fill=tk.X)

        tk.Button(
            settings_frame,
            text="🐛 Bug Report (F12)",
            command=self._capture_bug_report,
            bg='#5555ff',
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=5, fill=tk.X)

        tk.Button(
            settings_frame,
            text="🏗️ Make Deck Suggestion",
            command=self._manual_deck_suggestion,
            bg='#3a3a3a',
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=5, fill=tk.X)

        tk.Button(
            settings_frame,
            text="Exit",
            command=self._on_exit,
            bg=self.warning_color,
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(pady=5, fill=tk.X)

        # Chat/Prompt input area
        chat_label = tk.Label(
            settings_frame,
            text="📝 SEND PROMPT",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        chat_label.pack(pady=(15, 5), anchor=tk.W)

        self.prompt_text = tk.Text(
            settings_frame,
            height=4,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=5,
            pady=5,
            wrap=tk.WORD
        )
        self.prompt_text.pack(pady=(0, 5), fill=tk.BOTH, expand=False)
        self.prompt_text.bind('<Control-Return>', self._on_prompt_send)

        send_btn = tk.Button(
            settings_frame,
            text="Send [Ctrl+Enter]",
            command=self._on_prompt_send,
            bg=self.info_color,
            fg='#1a1a1a',
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        send_btn.pack(pady=5, fill=tk.X)

        # Main content area (right side)
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Board state area (top)
        board_header_frame = tk.Frame(content_frame, bg=self.bg_color)
        board_header_frame.pack(pady=(0, 5), fill=tk.X)

        self.board_label = tk.Label(
            board_header_frame,
            text="═══ BOARD STATE ═══",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        self.board_label.pack(side=tk.LEFT, expand=True)

        # Draft card counter (hidden by default)
        self.draft_counter_label = tk.Label(
            board_header_frame,
            text="",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 9, 'bold')
        )
        self.draft_counter_label.pack(side=tk.RIGHT, padx=10)

        self.board_text = scrolledtext.ScrolledText(
            content_frame,
            height=15,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.board_text.pack(fill=tk.BOTH, expand=True)
        self.board_text.config(state=tk.DISABLED)

        # Configure color tags for draft pack display (matching messages_text colors)
        self.board_text.tag_config('color_w', foreground='#ffff99')   # White - pale yellow
        self.board_text.tag_config('color_u', foreground='#55aaff')   # Blue
        self.board_text.tag_config('color_b', foreground='#aaaaaa')   # Black - gray
        self.board_text.tag_config('color_r', foreground='#ff5555')   # Red
        self.board_text.tag_config('color_g', foreground='#00ff88')   # Green
        self.board_text.tag_config('color_c', foreground='#cccccc')   # Colorless - light gray
        self.board_text.tag_config('color_multi', foreground='#ffdd44') # Multicolor - orange-yellow

        # Advisor messages area (bottom)
        self.advisor_label = tk.Label(
            content_frame,
            text="═══ ADVISOR ═══",
            bg=self.bg_color,
            fg=self.accent_color,
            font=('Consolas', 10, 'bold')
        )
        self.advisor_label.pack(pady=(10, 5))

        self.messages_text = scrolledtext.ScrolledText(
            content_frame,
            height=15,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        self.messages_text.config(state=tk.DISABLED)

        # Configure text tags for colors
        self.messages_text.tag_config('green', foreground='#00ff88')
        self.messages_text.tag_config('blue', foreground='#55aaff')
        self.messages_text.tag_config('cyan', foreground='#00ffff')
        self.messages_text.tag_config('yellow', foreground='#ffff00')
        self.messages_text.tag_config('red', foreground='#ff5555')
        self.messages_text.tag_config('white', foreground='#ffffff')

        # RAG References panel (bottom)
        rag_header_frame = tk.Frame(content_frame, bg=self.bg_color)
        rag_header_frame.pack(pady=(10, 5), fill=tk.X)

        self.rag_label = tk.Label(
            rag_header_frame,
            text="═══ RAG REFERENCES ═══",
            bg=self.bg_color,
            fg=self.info_color,
            font=('Consolas', 10, 'bold')
        )
        self.rag_label.pack(side=tk.LEFT, expand=True)

        # Toggle button for RAG references
        self.rag_toggle_btn = tk.Button(
            rag_header_frame,
            text="[Expand]",
            bg='#1a1a1a',
            fg=self.info_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            command=self._toggle_rag_panel
        )
        self.rag_toggle_btn.pack(side=tk.RIGHT, padx=5)

        self.rag_text = scrolledtext.ScrolledText(
            content_frame,
            height=6,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 8),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.rag_text.pack(fill=tk.BOTH, expand=False)
        self.rag_text.config(state=tk.DISABLED)

        # Tag configurations for RAG panel
        self.rag_text.tag_config('rule', foreground='#ffff00', font=('Consolas', 8, 'bold'))
        self.rag_text.tag_config('card', foreground='#00ff88', font=('Consolas', 8, 'bold'))
        self.rag_text.tag_config('query', foreground='#55aaff', font=('Consolas', 8, 'italic'))
        self.rag_text.tag_config('stats', foreground='#ff88ff')

    def _update_loop(self):
        """Process queued updates"""
        if not self.running:
            return

        # Update board state if changed
        if hasattr(self, '_pending_board_update'):
            self._update_board_display()
            delattr(self, '_pending_board_update')

        # Update messages
        while self.message_queue:
            msg, color = self.message_queue.popleft()
            self._add_message_to_display(msg, color)

        # Update RAG references (every 500ms to reduce overhead)
        if not hasattr(self, '_rag_update_counter'):
            self._rag_update_counter = 0
        self._rag_update_counter += 1
        if self._rag_update_counter >= 5:  # Update every 500ms (100ms * 5)
            self._update_rag_references()
            self._rag_update_counter = 0

        # Schedule next update
        self.root.after(100, self._update_loop)

    def _add_message_to_display(self, message, color):
        """Add message to display area"""
        self.messages_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] ", 'white')
        self.messages_text.insert(tk.END, f"{message}\n", color)
        self.messages_text.see(tk.END)
        self.messages_text.config(state=tk.DISABLED)

    def _update_board_display(self):
        """Update board state display"""
        self.board_text.config(state=tk.NORMAL)
        self.board_text.delete(1.0, tk.END)

        # Insert lines with color tag support for draft pack
        for i, line in enumerate(self.board_state_lines):
            if i > 0:
                self.board_text.insert(tk.END, "\n")

            # Check if this is a draft pack line (has color codes in columns 4-5)
            if len(line) > 5 and line[0].isdigit():
                # Parse line to apply color tags
                # Format: "#  C  R Card Name..."
                # Colors at positions 4 and 6 (after the # and two spaces)
                self._insert_line_with_colors(line)
            else:
                # Regular line, no color formatting
                self.board_text.insert(tk.END, line)

        self.board_text.config(state=tk.DISABLED)

    def _insert_line_with_colors(self, line: str):
        """Insert a draft pack line with color tags applied to color codes"""
        import re

        # Pattern to find color codes: W, U, B, R, G, C, or multicolor like WU, BR, etc.
        # Position 4-5 is the color field (after "#  ")
        # Position 7-8 is the rarity field (after "   ")

        pos = 0

        # Insert #
        if len(line) > pos:
            self.board_text.insert(tk.END, line[pos])
            pos += 1

        # Insert spaces (3 spaces after #)
        while pos < len(line) and line[pos] == ' ' and pos < 3:
            self.board_text.insert(tk.END, line[pos])
            pos += 1

        # Extract and apply color code (1-2 chars)
        color_start = pos
        while pos < len(line) and line[pos] != ' ':
            pos += 1
        color_code = line[color_start:pos]

        # Apply appropriate tag based on color
        if color_code:
            if len(color_code) > 1:
                # Multicolor
                tag = 'color_multi'
            elif color_code in ['W', 'U', 'B', 'R', 'G', 'C']:
                tag = f'color_{color_code.lower()}'
            else:
                tag = None

            if tag:
                self.board_text.insert(tk.END, color_code, tag)
            else:
                self.board_text.insert(tk.END, color_code)

        # Insert remaining text as-is
        if pos < len(line):
            self.board_text.insert(tk.END, line[pos:])

    def _update_rag_references(self):
        """Update RAG references display from advisor's last references"""
        if not hasattr(self.advisor, 'get_last_rag_references'):
            return

        references = self.advisor.get_last_rag_references()
        if not references:
            self.rag_text.config(state=tk.NORMAL)
            self.rag_text.delete(1.0, tk.END)
            self.rag_text.insert(1.0, "No RAG references available. Click [Expand] to see when references are used.", 'white')
            self.rag_text.config(state=tk.DISABLED)
            return

        self.rag_text.config(state=tk.NORMAL)
        self.rag_text.delete(1.0, tk.END)

        # Store references for click handling
        self._current_rag_references = references

        # Display queries
        if references.get('queries'):
            self.rag_text.insert(tk.END, "Queries: ", 'white')
            for i, query in enumerate(references['queries'][:2]):
                if i > 0:
                    self.rag_text.insert(tk.END, " | ")
                self.rag_text.insert(tk.END, f"[{query}]", f'query_{i}')
            self.rag_text.insert(tk.END, "\n")

        # Display rules
        rules = references.get('rules', [])
        if rules:
            self.rag_text.insert(tk.END, "Rules: ", 'white')
            for i, rule in enumerate(rules[:3]):
                if i > 0:
                    self.rag_text.insert(tk.END, " | ")
                rule_id = rule.get('id', 'Unknown')
                self.rag_text.insert(tk.END, f"[{rule_id}]", f'rule_{i}')
            self.rag_text.insert(tk.END, "\n")

        # Display card stats
        cards = references.get('cards', {})
        if cards:
            self.rag_text.insert(tk.END, "Cards: ", 'white')
            card_names = list(cards.keys())[:3]
            for i, card_name in enumerate(card_names):
                if i > 0:
                    self.rag_text.insert(tk.END, " | ")
                card_info = cards[card_name]
                win_rate = card_info.get('gih_win_rate', 0)
                self.rag_text.insert(tk.END, f"[{card_name} {win_rate:.1%}]", f'card_{i}')

        # Configure underline for clickable items
        for tag_name in self.rag_text.tag_names():
            if tag_name.startswith(('query_', 'rule_', 'card_')):
                self.rag_text.tag_configure(tag_name, underline=True, foreground='#00ccff')
                self.rag_text.tag_bind(tag_name, '<Button-1>', lambda e, tag=tag_name: self._on_rag_reference_click(e, tag))
                self.rag_text.tag_bind(tag_name, '<Enter>', lambda e, tag=tag_name: self._on_rag_reference_hover(e, tag, True))
                self.rag_text.tag_bind(tag_name, '<Leave>', lambda e, tag=tag_name: self._on_rag_reference_hover(e, tag, False))

        self.rag_text.config(state=tk.DISABLED)

    def _on_rag_reference_hover(self, event, tag, entering):
        """Handle hover over RAG references"""
        if entering:
            self.rag_text.config(cursor='hand2')
        else:
            self.rag_text.config(cursor='arrow')

    def _on_rag_reference_click(self, event, tag):
        """Handle click on RAG references to show details"""
        if not hasattr(self, '_current_rag_references'):
            return

        details = ""

        # Parse tag to determine type
        if tag.startswith('query_'):
            idx = int(tag.split('_')[1])
            queries = self._current_rag_references.get('queries', [])
            if idx < len(queries):
                details = f"Query used to retrieve rules:\n{queries[idx]}"

        elif tag.startswith('rule_'):
            idx = int(tag.split('_')[1])
            rules = self._current_rag_references.get('rules', [])
            if idx < len(rules):
                rule = rules[idx]
                rule_id = rule.get('id', 'Unknown')
                rule_text = rule.get('text', 'No details available')
                rule_section = rule.get('section', 'Unknown')
                details = f"Rule {rule_id} ({rule_section}):\n{rule_text}"

        elif tag.startswith('card_'):
            idx = int(tag.split('_')[1])
            cards = list(self._current_rag_references.get('cards', {}).items())
            if idx < len(cards):
                card_name, card_info = cards[idx]
                details = f"Card Statistics for {card_name}:\n"
                details += f"  GIH Win Rate: {card_info.get('gih_win_rate', 0):.1%}\n"
                details += f"  Games Played: {card_info.get('games_played', 0)}\n"
                details += f"  IWD (In-hand win delta): {card_info.get('iwd', 'N/A')}\n"
                details += f"  Format: {card_info.get('format', 'Unknown')}"

        if details:
            self._show_rag_detail_popup(details)

    def _show_rag_detail_popup(self, details):
        """Show a popup window with detailed RAG reference information"""
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("RAG Reference Details")
        popup.geometry("600x400")
        popup.configure(bg=self.bg_color)

        # Header
        header = tk.Label(
            popup,
            text="RAG Reference Details",
            bg=self.bg_color,
            fg=self.info_color,
            font=('Consolas', 12, 'bold')
        )
        header.pack(pady=(10, 5), fill=tk.X)

        # Details text
        details_text = scrolledtext.ScrolledText(
            popup,
            bg='#1a1a1a',
            fg=self.fg_color,
            font=('Consolas', 10),
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        details_text.insert(1.0, details)
        details_text.config(state=tk.DISABLED)

        # Close button
        close_btn = tk.Button(
            popup,
            text="Close",
            bg='#1a1a1a',
            fg=self.accent_color,
            font=('Consolas', 10),
            relief=tk.FLAT,
            command=popup.destroy
        )
        close_btn.pack(pady=10)

        # Center the popup on the main window
        popup.transient(self.root)
        popup.grab_set()

    def _toggle_rag_panel(self):
        """Toggle RAG panel expanded/collapsed state"""
        if self.rag_panel_expanded:
            # Collapse
            self.rag_text.config(height=3)
            self.rag_toggle_btn.config(text="[Expand]")
            self.rag_panel_expanded = False
        else:
            # Expand
            self.rag_text.config(height=8)
            self.rag_toggle_btn.config(text="[Collapse]")
            self.rag_panel_expanded = True

    def add_message(self, message, color='white'):
        """Thread-safe message adding"""
        self.message_queue.append((message, color))

    def set_board_state(self, lines):
        """Thread-safe board state update"""
        self.board_state_lines = lines
        self._pending_board_update = True

    def set_draft_panes(self, pack_lines, picked_lines, picked_count=0, total_needed=45):
        """Set draft pack in board pane and picked cards in messages pane"""
        # Update board pane with draft pack
        self.board_state_lines = pack_lines
        self._pending_board_update = True

        # Update messages pane with picked cards
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete(1.0, tk.END)
        if picked_lines:
            self.messages_text.insert(1.0, "\n".join(picked_lines))
        self.messages_text.config(state=tk.DISABLED)

        # Update pane labels for draft mode
        self.board_label.config(text="═══ DRAFT POOL ═══")
        self.advisor_label.config(text="═══ DRAFTED ═══")

        # Update draft counter
        if picked_count > 0:
            counter_color = "green" if picked_count >= total_needed else "yellow"
            self.draft_counter_label.config(
                text=f"📦 {picked_count}/{total_needed}",
                fg=counter_color
            )
        else:
            self.draft_counter_label.config(text="")

    def reset_pane_labels(self):
        """Reset pane labels to default (non-draft mode)"""
        self.board_label.config(text="═══ BOARD STATE ═══")
        self.advisor_label.config(text="═══ ADVISOR ═══")
        self.draft_counter_label.config(text="")  # Hide counter

    def set_status(self, status_text):
        """Update status bar"""
        self.status_label.config(text=status_text)

    def update_settings(self, models, voices, bark_voices, current_model, current_voice, volume, tts_engine):
        """Update settings dropdowns"""
        self.model_dropdown['values'] = models
        self.model_var.set(current_model)

        if tts_engine == 'kokoro':
            self.voice_dropdown['values'] = voices
            self.tts_engine_var.set("Kokoro")
        else:
            self.voice_dropdown['values'] = bark_voices
            self.tts_engine_var.set("BarkTTS")

        self.voice_var.set(current_voice)
        self.volume_var.set(volume)
        self.volume_label.config(text=f"{volume}%")

    def _on_model_change(self, event=None):
        """Handle model selection change"""
        if hasattr(self.advisor, '_on_gui_model_change'):
            self.advisor._on_gui_model_change(self.model_var.get())

    def _on_voice_change(self, event=None):
        """Handle voice selection change"""
        if hasattr(self.advisor, '_on_gui_voice_change'):
            self.advisor._on_gui_voice_change(self.voice_var.get())

    def _on_tts_engine_change(self):
        """Handle TTS engine change"""
        engine = "bark" if self.tts_engine_var.get() == "BarkTTS" else "kokoro"
        if hasattr(self.advisor, '_on_gui_tts_engine_change'):
            self.advisor._on_gui_tts_engine_change(engine)

    def _on_volume_change(self, value):
        """Handle volume slider change"""
        vol = int(float(value))
        self.volume_label.config(text=f"{vol}%")
        if hasattr(self.advisor, '_on_gui_volume_change'):
            self.advisor._on_gui_volume_change(vol)

    def _on_continuous_toggle(self):
        """Handle continuous monitoring toggle"""
        if hasattr(self.advisor, 'continuous_monitoring'):
            self.advisor.continuous_monitoring = self.continuous_var.get()
            # Save to preferences
            if self.prefs:
                self.prefs.opponent_turn_alerts = self.continuous_var.get()
            status = "ENABLED" if self.continuous_var.get() else "DISABLED"
            self.add_message(f"Opponent Turn Alerts {status}", "cyan")

    def _on_reskin_toggle(self):
        """Handle reskin names toggle"""
        if hasattr(self.advisor, 'game_state_mgr') and hasattr(self.advisor.game_state_mgr, 'card_lookup'):
            self.advisor.game_state_mgr.card_lookup.show_reskin_names = self.reskin_var.get()
            # Save to preferences
            if self.prefs:
                self.prefs.reskin_names = self.reskin_var.get()
            status = "ENABLED" if self.reskin_var.get() else "DISABLED"
            self.add_message(f"Spider-Man Reskins {status}", "cyan")

    def _on_always_on_top_toggle(self):
        """Handle always on top toggle"""
        state = self.always_on_top_var.get()
        self.root.attributes('-topmost', state)
        # Save immediately
        if self.prefs:
            self.prefs.always_on_top = state
            self.prefs.save()

    def on_closing(self):
        """Handle window close event - save preferences before exiting"""
        # Save all settings to preferences
        if self.prefs:
            self.prefs.opponent_turn_alerts = self.continuous_var.get()
            self.prefs.show_thinking = self.show_thinking_var.get()
            self.prefs.reskin_names = self.reskin_var.get()
            self.prefs.tts_volume = self.volume_var.get()

            # Save window state
            try:
                self.prefs.window_geometry = self.root.geometry()
                # always_on_top is read from window state (already set)
            except Exception as e:
                logging.warning(f"Could not save window geometry: {e}")

            self.prefs.save()
            logging.info("GUI preferences saved on close")

        self.running = False
        self.root.destroy()

    def _clear_messages(self):
        """Clear message area"""
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete(1.0, tk.END)
        self.messages_text.config(state=tk.DISABLED)

    def _on_prompt_send(self, event=None):
        """Handle user-sent prompt"""
        prompt_text = self.prompt_text.get("1.0", tk.END).strip()

        if not prompt_text:
            return

        # Display user prompt in advisor area
        self.add_message(f"You: {prompt_text}", 'cyan')

        # Clear input field
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.config(state=tk.NORMAL)

        # Send to advisor in background thread
        def send_prompt():
            try:
                # If advisor has client, use it directly for free-form queries
                if hasattr(self.advisor, 'client') and self.advisor.client:
                    response = self.advisor.client.generate(prompt_text)
                    if response:
                        self.add_message(f"Advisor: {response}", 'green')
                    else:
                        self.add_message("Advisor: (No response from AI model)", 'yellow')
                else:
                    self.add_message("Advisor: AI client not available", 'red')
            except Exception as e:
                logging.error(f"Error sending prompt: {e}")
                self.add_message(f"Advisor: Error - {str(e)}", 'red')

        import threading
        thread = threading.Thread(target=send_prompt, daemon=True)
        thread.start()

    def _on_exit(self):
        """Handle exit button"""
        self.running = False
        if hasattr(self.advisor, 'running'):
            self.advisor.running = False
        self.root.quit()

    def _capture_bug_report(self):
        """Capture bug report with screenshot, logs, and board state"""
        import threading
        import subprocess
        import os

        # Prompt for GitHub and ImgBB tokens BEFORE starting background thread
        self.add_message("📸 Preparing bug report...", "cyan")
        github_token = None
        imgbb_token = None
        self.add_message("📤 Submit to GitHub? (requires GitHub token + ImgBB API key)", "cyan")

        # Check for cached tokens first
        github_cache_file = Path("logs/.github_token_cache")
        imgbb_cache_file = Path("logs/.imgbb_api_key_cache")

        if github_cache_file.exists():
            try:
                with open(github_cache_file, 'r') as f:
                    cached_token = f.read().strip()
                    if cached_token and len(cached_token) > 20:  # Basic validation
                        github_token = cached_token
                        self.add_message("✓ Using cached GitHub token", "green")
            except Exception as e:
                logging.debug(f"Could not read GitHub token cache: {e}")

        if imgbb_cache_file.exists():
            try:
                with open(imgbb_cache_file, 'r') as f:
                    cached_key = f.read().strip()
                    if cached_key and len(cached_key) > 20:  # Basic validation
                        imgbb_token = cached_key
                        self.add_message("✓ Using cached ImgBB API key", "green")
            except Exception as e:
                logging.debug(f"Could not read ImgBB cache: {e}")

        # If tokens not cached, prompt user
        if not github_token or not imgbb_token:
            try:
                from tkinter import simpledialog
                if hasattr(self, 'root'):
                    # Prompt for GitHub token if not cached
                    if not github_token:
                        token = simpledialog.askstring(
                            "GitHub Token",
                            "Enter your GitHub Personal Access Token:\n(Leave blank to save locally only)\n\nGet token: https://github.com/settings/tokens\n\nToken format: ghp_...",
                            parent=self.root,
                            show="*"
                        )
                        if token:
                            github_token = token.strip()
                            logging.debug(f"GitHub token received: {len(github_token)} chars")

                            # Validate token format
                            if len(github_token) > 20 and '\n' not in github_token and '\r' not in github_token:
                                # Cache the token
                                try:
                                    github_cache_file.parent.mkdir(parents=True, exist_ok=True)
                                    with open(github_cache_file, 'w') as f:
                                        f.write(github_token)
                                    github_cache_file.chmod(0o600)
                                    self.add_message("✓ GitHub token cached", "green")
                                except Exception as e:
                                    logging.debug(f"Could not cache GitHub token: {e}")
                            else:
                                self.add_message("⚠️ Invalid GitHub token format", "yellow")
                                github_token = None

                    # Prompt for ImgBB API key if not cached
                    if not imgbb_token:
                        key = simpledialog.askstring(
                            "ImgBB API Key",
                            "Enter your ImgBB API key for screenshot hosting:\n(Leave blank to save locally only)\n\nGet key: https://imgbb.com/api\n\nKey format: 32 character hex string",
                            parent=self.root,
                            show="*"
                        )
                        if key:
                            imgbb_token = key.strip()
                            logging.debug(f"ImgBB key received: {len(imgbb_token)} chars")

                            # Validate key format (should be 32 hex chars)
                            if len(imgbb_token) == 32 and '\n' not in imgbb_token and '\r' not in imgbb_token:
                                # Cache the key
                                try:
                                    imgbb_cache_file.parent.mkdir(parents=True, exist_ok=True)
                                    with open(imgbb_cache_file, 'w') as f:
                                        f.write(imgbb_token)
                                    imgbb_cache_file.chmod(0o600)
                                    self.add_message("✓ ImgBB API key cached", "green")
                                except Exception as e:
                                    logging.debug(f"Could not cache ImgBB key: {e}")
                            else:
                                self.add_message("⚠️ Invalid ImgBB key format (should be 32 chars)", "yellow")
                                imgbb_token = None

                    # Check if we have both tokens for GitHub upload
                    if github_token and imgbb_token:
                        self.add_message("✓ Both tokens received - will upload to GitHub", "green")
                    elif github_token or imgbb_token:
                        self.add_message("⚠️ Need both GitHub token AND ImgBB key to upload", "yellow")
                        github_token = None
                        imgbb_token = None

            except Exception as e:
                logging.debug(f"Token prompt not available: {e}")

        def capture_in_background():
            try:
                # Create bug_reports directory if it doesn't exist
                bug_dir = "/home/joshu/logparser/bug_reports"
                os.makedirs(bug_dir, exist_ok=True)

                # Generate timestamp for this report
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                report_file = f"{bug_dir}/bug_report_{timestamp}.txt"

                # Ask user if they want to add a title and description
                add_details = False
                issue_title = f"Bug Report: {timestamp}"
                user_description = "No description provided."

                try:
                    from tkinter import messagebox, simpledialog
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        add_details = messagebox.askyesno(
                            "Bug Report Details",
                            "Do you want to add a title and description to this bug report?",
                            parent=self.root
                        )
                        if add_details:
                            # Prompt for title
                            title_prompt = simpledialog.askstring(
                                "Bug Report Title",
                                "Enter a title for the bug report (or leave blank for default):",
                                parent=self.root
                            )
                            if title_prompt and title_prompt.strip():
                                issue_title = title_prompt.strip()

                            # Prompt for description
                            desc_prompt = simpledialog.askstring(
                                "Bug Report Description",
                                "Please describe the bug:",
                                parent=self.root
                            )
                            if desc_prompt and desc_prompt.strip():
                                user_description = desc_prompt.strip()
                except (ImportError, tk.TclError) as e:
                    logging.debug(f"GUI not available for bug report details: {e}")

                # Take screenshot using intelligent multi-tool detection
                screenshot_file = "Screenshot not available"
                if CONFIG_MANAGER_AVAILABLE:
                    try:
                        from screenshot_util import ScreenshotCapture
                        capturer = ScreenshotCapture()
                        if capturer.available_tool:
                            # Try to capture both MTGA and Advisor windows as a composite
                            screenshot = None
                            mtga_names = ["MTGA", "Magic: The Gathering Arena", "MTG Arena"]
                            advisor_names = ["Advisor", "Voice Advisor"]

                            # Try all combinations to capture both windows
                            for mtga_name in mtga_names:
                                for advisor_name in advisor_names:
                                    screenshot = capturer.capture_multiple_windows([mtga_name, advisor_name])
                                    if screenshot:
                                        logging.debug(f"Captured composite: {mtga_name} + {advisor_name}")
                                        break
                                if screenshot:
                                    break

                            # If composite failed, try single windows
                            if not screenshot:
                                for window_name in mtga_names + advisor_names:
                                    screenshot = capturer.capture_window_by_name(window_name)
                                    if screenshot:
                                        logging.debug(f"Captured {window_name} window")
                                        break

                            # If no specific window found, capture full screen
                            if not screenshot:
                                screenshot = capturer.capture_screen()

                            # Optimize the screenshot
                            if screenshot:
                                screenshot = capturer.resize_for_claude(screenshot)

                            if screenshot:
                                # Copy optimized screenshot to bug_reports directory
                                import shutil
                                dest_screenshot = f"{bug_dir}/screenshot_{timestamp}.png"
                                shutil.copy(str(screenshot), dest_screenshot)
                                screenshot_file = dest_screenshot
                                logging.debug(f"Screenshot captured and optimized: {dest_screenshot}")
                        else:
                            screenshot_file = "Screenshot tool not available (install gnome-screenshot, scrot, or imagemagick)"
                    except Exception as e:
                        screenshot_file = f"Screenshot capture failed: {e}"
                        logging.debug(f"Screenshot error: {e}")

                # Collect current state
                board_state_text = "\n".join(self.board_state_lines) if self.board_state_lines else "No board state"

                # Read recent logs (last 50 lines for GitHub - keep it concise)
                recent_logs = ""
                try:
                    with open("/home/joshu/logparser/logs/advisor.log", "r") as f:
                        lines = f.readlines()
                        recent_logs = "".join(lines[-50:])
                except Exception as e:
                    recent_logs = f"Failed to read logs: {e}"

                # Extract relevant MTGA log snippets
                mtga_log_snippets = ""
                try:
                    # Try to find MTGA log file
                    mtga_log_path = Path.home() / ".local/share/Steam/steamapps/compatdata/2141910/pfx/drive_c/users/steamuser/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
                    if not mtga_log_path.exists():
                        # Try macOS path
                        mtga_log_path = Path.home() / "Library/Logs/Wizards Of The Coast/MTGA/Player.log"
                    if not mtga_log_path.exists():
                        # Try Windows path
                        mtga_log_path = Path("C:/Users") / Path.home().name / "AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"

                    if mtga_log_path.exists():
                        with open(mtga_log_path, "r", errors="ignore") as f:
                            # Read last 50 lines and extract relevant events
                            lines = f.readlines()
                            mtga_lines = lines[-100:] if len(lines) > 100 else lines

                            # Filter for interesting events
                            interesting_keywords = [
                                "GreToClientEvent",
                                "GameState",
                                "ClientToGre",
                                "ERROR",
                                "Exception",
                                "Warning",
                            ]

                            relevant_lines = []
                            for line in mtga_lines:
                                if any(keyword in line for keyword in interesting_keywords):
                                    relevant_lines.append(line.strip())

                            if relevant_lines:
                                mtga_log_snippets = "\n".join(relevant_lines[-20:])  # Last 20 relevant lines
                            else:
                                mtga_log_snippets = "(No relevant MTGA events found in recent logs)"
                    else:
                        mtga_log_snippets = "(MTGA log file not found)"
                except Exception as e:
                    mtga_log_snippets = f"Failed to extract MTGA logs: {e}"

                # Get current settings
                settings = f"""Model: {self.model_var.get()}
Voice: {self.voice_var.get()}
TTS Engine: {self.tts_engine_var.get()}
Volume: {self.volume_var.get()}%
Opponent Turn Alerts: {self.continuous_var.get()}
Show AI Thinking: {self.show_thinking_var.get()}
Show Spider-Man Reskins: {self.reskin_var.get()}
"""

                # Write bug report
                with open(report_file, "w") as f:
                    f.write("="*70 + "\n")
                    f.write(f"BUG REPORT: {issue_title}\n")
                    f.write("="*70 + "\n\n")

                    f.write("USER DESCRIPTION:\n")
                    f.write(f"{user_description}\n\n")

                    f.write("SCREENSHOT:\n")
                    f.write(f"{screenshot_file}\n\n")

                    f.write("="*70 + "\n")
                    f.write("CURRENT SETTINGS:\n")
                    f.write("="*70 + "\n")
                    f.write(settings + "\n")

                    f.write("="*70 + "\n")
                    f.write("CURRENT BOARD STATE:\n")
                    f.write("="*70 + "\n")
                    f.write(board_state_text + "\n\n")

                    f.write("="*70 + "\n")
                    f.write("RECENT LOGS (last 300 lines):\n")
                    f.write("="*70 + "\n")
                    f.write(recent_logs + "\n")

                # Create symlinks to latest versions for easy reference
                try:
                    latest_report_link = f"{bug_dir}/LATEST_BUG_REPORT.txt"
                    latest_screenshot_link = f"{bug_dir}/LATEST_SCREENSHOT.jpg"

                    # Remove old symlinks if they exist
                    for link_path in [latest_report_link, latest_screenshot_link]:
                        if os.path.islink(link_path) or os.path.exists(link_path):
                            try:
                                os.remove(link_path)
                            except:
                                pass

                    # Create new symlinks to latest versions
                    os.symlink(os.path.basename(report_file), latest_report_link)
                    if screenshot_file and screenshot_file.endswith(".jpg"):
                        os.symlink(os.path.basename(screenshot_file), latest_screenshot_link)

                    logging.debug(f"Created symlinks to latest bug report")
                except Exception as e:
                    logging.debug(f"Could not create symlinks: {e}")

                # Show success message
                self.add_message(f"✓ Bug report saved: {report_file}", "green")
                logging.info(f"Bug report captured: {report_file}")

                # Attempt GitHub upload if token was provided
                if github_token:
                    # Create GitHub issue
                    # Build issue body
                    issue_body_parts = []

                    # Note about screenshot (saved locally, not embedded due to GitHub security restrictions)
                    if screenshot_file and os.path.exists(screenshot_file):
                        issue_body_parts.append(f"📸 **Screenshot:** `{os.path.basename(screenshot_file)}`")
                        issue_body_parts.append("")

                    issue_body_parts.extend([
                        "**Description:**",
                        user_description,
                        "",
                        "**Settings:**",
                        "```",
                        settings,
                        "```",
                        "",
                        "**Board State:**",
                        "```",
                        board_state_text,
                        "```",
                        "",
                        "**MTGA Log Snippets:**",
                        "```",
                        mtga_log_snippets,
                        "```",
                        "",
                        "**Advisor Logs:**",
                        "```",
                        recent_logs,
                        "```",
                    ])

                    issue_body = "\n".join(issue_body_parts)

                    # Check size and truncate if necessary (GitHub limit is 65536 chars)
                    max_body_size = 60000  # Leave some buffer
                    if len(issue_body) > max_body_size:
                        # Truncate logs if needed
                        logging.warning(f"Issue body too large ({len(issue_body)} chars), truncating logs")
                        # Remove MTGA logs first if they're too long
                        if "**MTGA Log Snippets:**" in issue_body:
                            issue_body_parts = [p for p in issue_body_parts if "MTGA Log" not in p]
                            issue_body = "\n".join(issue_body_parts[:10])  # Keep just settings and board state
                            if len(issue_body) > max_body_size:
                                # If still too big, truncate advisor logs
                                issue_body_parts = [p for p in issue_body_parts if "Advisor Logs" not in p]
                                issue_body = "\n".join(issue_body_parts[:8])

                    try:
                        # Create issue
                        url = "https://api.github.com/repos/josharmour/mtga-voice-assistant/issues"
                        headers = {
                            "Authorization": f"token {github_token}",
                            "Accept": "application/vnd.github.v3+json",
                        }
                        issue_data = {"title": issue_title, "body": issue_body}
                        response = requests.post(url, json=issue_data, headers=headers, timeout=10)

                        if response.status_code == 201:
                            issue_url = response.json()["html_url"]
                            issue_number = response.json()["number"]
                            self.add_message(f"✓ Bug report uploaded to GitHub: {issue_url}", "green")
                            logging.info(f"Bug report uploaded to GitHub: {issue_url}")

                            # Now attach screenshot as a comment if available
                            if screenshot_file and os.path.exists(screenshot_file):
                                try:
                                    filename = os.path.basename(screenshot_file)

                                    # Read screenshot binary
                                    with open(screenshot_file, 'rb') as f:
                                        screenshot_data = f.read()

                                    file_size_kb = len(screenshot_data) / 1024
                                    logging.info(f"Screenshot size: {file_size_kb:.1f}KB")

                                    # Upload to ImgBB if token is available
                                    imgbb_url = None

                                    if imgbb_token:
                                        try:
                                            # Upload screenshot file directly to ImgBB
                                            logging.info(f"Uploading screenshot to ImgBB ({file_size_kb:.1f}KB, API key: {imgbb_token[:8]}...)...")
                                            imgbb_upload_url = "https://api.imgbb.com/1/upload"

                                            # Use multipart form data with binary file
                                            imgbb_files = {
                                                "image": (filename, screenshot_data, "image/png")
                                            }
                                            imgbb_data = {
                                                "key": imgbb_token  # ImgBB API key
                                            }

                                            imgbb_response = requests.post(
                                                imgbb_upload_url,
                                                files=imgbb_files,
                                                data=imgbb_data,
                                                timeout=60
                                            )

                                            if imgbb_response.status_code == 200:
                                                imgbb_json = imgbb_response.json()
                                                if imgbb_json.get("success"):
                                                    imgbb_url = imgbb_json["data"]["url"]
                                                    logging.info(f"Screenshot uploaded to ImgBB: {imgbb_url}")
                                                else:
                                                    logging.warning(f"ImgBB upload failed (API error): {imgbb_json}")
                                            else:
                                                try:
                                                    error_text = imgbb_response.text[:500]
                                                except:
                                                    error_text = "Unable to read response"
                                                logging.warning(f"ImgBB upload failed: HTTP {imgbb_response.status_code} - {error_text}")
                                        except Exception as e:
                                            logging.warning(f"ImgBB upload error: {e}")
                                    else:
                                        logging.info("No ImgBB token provided, skipping upload")

                                    # Add comment to issue with screenshot
                                    comment_url = f"https://api.github.com/repos/josharmour/mtga-voice-assistant/issues/{issue_number}/comments"
                                    comment_headers = {
                                        "Authorization": f"token {github_token}",
                                        "Accept": "application/vnd.github.v3+json",
                                    }

                                    if imgbb_url:
                                        # Use ImgBB URL (external hosting)
                                        comment_body = f"📸 **Screenshot** ({file_size_kb:.1f}KB)\n\n![Screenshot]({imgbb_url})"
                                    else:
                                        # Fallback: If no ImgBB URL, just mention file was saved locally
                                        comment_body = f"📸 **Screenshot** ({file_size_kb:.1f}KB - saved locally as `{filename}`)"

                                    comment_response = requests.post(
                                        comment_url,
                                        json={"body": comment_body},
                                        headers=comment_headers,
                                        timeout=30
                                    )

                                    if comment_response.status_code == 201:
                                        logging.info(f"Screenshot attached to issue #{issue_number}")
                                        self.add_message("✓ Screenshot attached", "green")
                                    else:
                                        logging.warning(f"Comment creation failed: {comment_response.status_code}")
                                        self.add_message("⚠️ Screenshot comment failed (file saved locally)", "yellow")

                                except Exception as e:
                                    logging.warning(f"Screenshot attachment error: {e}")
                                    self.add_message("⚠️ Screenshot attachment failed (file saved locally)", "yellow")
                        else:
                            self.add_message(f"✗ GitHub issue creation failed: {response.status_code}", "red")
                            logging.error(f"GitHub issue creation failed: {response.text}")
                    except Exception as e:
                        self.add_message(f"✗ GitHub upload failed: {e}", "red")
                        logging.error(f"GitHub upload failed: {e}")
                else:
                    self.add_message("ℹ️ Bug report saved locally only", "yellow")

            except Exception as e:
                self.add_message(f"✗ Bug report failed: {e}", "red")
                logging.error(f"Failed to capture bug report: {e}")

        # Run in background thread so it doesn't freeze the UI
        threading.Thread(target=capture_in_background, daemon=True).start()
        self.add_message("📸 Capturing bug report...", "cyan")

    def _prompt_github_token(self):
        """Prompt user for GitHub token via dialog or return None to skip"""
        try:
            # Show instructions
            self.add_message("🔐 GitHub Authentication Required", "cyan")
            self.add_message("Create a Personal Access Token with 'public_repo' scope:", "yellow")
            self.add_message("  1. Visit: https://github.com/settings/tokens", "cyan")
            self.add_message("  2. Click 'Generate new token (classic)'", "cyan")
            self.add_message("  3. Check only 'public_repo' under scopes", "cyan")
            self.add_message("  4. Generate and copy the token", "cyan")
            self.add_message("  5. Copy token and paste it below when the dialog appears", "cyan")
            self.add_message("", "white")

            # Check if we have root (GUI mode)
            if not hasattr(self, 'root'):
                logging.warning("Not in GUI mode - cannot show token dialog")
                return None, "GUI not available"

            try:
                from tkinter import simpledialog

                # Show the token prompt dialog
                token = simpledialog.askstring(
                    "GitHub Token",
                    "Paste your GitHub Personal Access Token:\n(ghp_... format)\n\nWill be used only for this bug report",
                    parent=self.root,
                    show="*"
                )

                if token:
                    self.add_message("✓ Token received", "green")
                    return token, None
                else:
                    return None, "Token entry cancelled"
            except Exception as e:
                logging.error(f"Dialog error: {e}")
                return None, str(e)

        except Exception as e:
            logging.error(f"Token prompt error: {e}")
            return None, str(e)

    def _manual_deck_suggestion(self):
        """Manually trigger deck suggestions using the current card pool"""
        # Access the parent advisor's draft_advisor and deck_builder
        if not self.advisor.draft_advisor or not self.advisor.deck_builder:
            self.add_message("✗ Draft advisor not available", "red")
            return

        try:
            # First, check if we have picked_cards from live draft tracking
            existing_picks = self.advisor.draft_advisor.picked_cards if self.advisor.draft_advisor else []

            # Check if we have a stored card pool from EventGetCoursesV2
            has_card_pool = hasattr(self.advisor, '_last_card_pool') and self.advisor._last_card_pool

            if not has_card_pool and not existing_picks:
                self.add_message("✗ No card pool found. Options:", "red")
                self.add_message("  1. Complete a draft while the advisor is running, OR", "yellow")
                self.add_message("  2. Go to Play → Events in MTGA to load draft data", "yellow")
                logging.warning("No card pool or picked cards available for deck suggestions")
                return

            # Use card pool if available, otherwise use existing picks
            if has_card_pool:
                event_name = getattr(self.advisor, '_last_draft_event_name', '')
                card_pool_ids = self.advisor._last_card_pool
                current_module = getattr(self.advisor, '_last_draft_module', '')

                logging.info(f"Manual deck suggestion triggered: {len(card_pool_ids)} cards from CardPool, event={event_name}, module={current_module}")

                # Convert card IDs to card names
                card_names = []
                for card_id in card_pool_ids:
                    try:
                        card_name = self.advisor.game_state_mgr.card_lookup.get_card_name(int(card_id))
                        if card_name:
                            card_names.append(card_name)
                    except Exception as e:
                        logging.debug(f"Error converting card ID {card_id}: {e}")

                if not card_names:
                    self.add_message("✗ Could not resolve card names from card pool", "red")
                    return

                # Update picked_cards with the full card pool
                self.advisor.draft_advisor.picked_cards = card_names
                logging.info(f"Updated picked_cards with {len(card_names)} cards from CardPool")
            else:
                # Use existing picked cards from live draft tracking
                card_names = existing_picks
                event_name = getattr(self.advisor, '_last_draft_event_name', 'Unknown_Draft')
                logging.info(f"Manual deck suggestion using {len(card_names)} tracked picks")

            if not card_names or len(card_names) < 20:
                self.add_message(f"✗ Not enough cards for deck building: {len(card_names)} (need at least 20)", "red")
                return

            self.add_message(f"🏗️ Generating deck suggestions from {len(card_names)} cards...", "cyan")

            # Reset the generated flag so we can generate suggestions again
            self.advisor._deck_suggestions_generated = False

            # Call the existing deck suggestion method
            self.advisor._generate_deck_suggestions(event_name)

        except Exception as e:
            logging.error(f"Error in manual deck suggestion: {e}")
            self.add_message(f"✗ Error generating suggestions: {e}", "red")

    def cleanup(self):
        """Cleanup GUI"""
        self.running = False

# ----------------------------------------------------------------------------------
# Part 10: Main CLI Loop
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

    # Available BarkTTS voices (English speakers)
    BARK_VOICES = ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
                   "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
                   "v2/en_speaker_8", "v2/en_speaker_9"]

    def __init__(self, use_tui: bool = False, use_gui: bool = False):
        self.use_tui = use_tui
        self.use_gui = use_gui
        self.tui = None
        self.gui = None
        self.tk_root = None
        self.previous_board_state = None  # Track previous state for importance detection

        # Load user preferences for persistent settings across sessions
        self.prefs = None
        if CONFIG_MANAGER_AVAILABLE:
            self.prefs = UserPreferences.load()
            if not use_tui:
                logging.debug("User preferences loaded successfully")

        # Set continuous monitoring from preferences or use default
        if self.prefs:
            self.continuous_monitoring = self.prefs.opponent_turn_alerts
        else:
            self.continuous_monitoring = True  # Enable continuous advisory mode

        self.last_alert_time = 0  # Timestamp of last critical alert (for rate limiting)

        self.log_path = detect_player_log_path()
        if not self.log_path:
            if not use_tui:
                print("ERROR: Could not find Arena Player.log. Please ensure the game is installed and you have run it at least once.")
            exit(1)

        # Initialize card database and show status
        if not use_tui:
            print("Loading card database...")

        # Pass reskin preference to card database if available
        show_reskin_names = self.prefs.reskin_names if self.prefs else False
        card_db = ArenaCardDatabase(show_reskin_names=show_reskin_names)
        if not use_tui:
            if card_db.conn:
                cursor = card_db.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cards")
                total = cursor.fetchone()[0]
                print(f"✓ Loaded unified card database ({total:,} cards)")
            else:
                print(f"⚠ Card database not found - cards will show as Unknown")
                print("  Run: python build_unified_card_database.py")

        # Check if Ollama is running before initializing AI advisor
        ollama_test = OllamaClient()
        if not ollama_test.is_running():
            if not use_tui:
                print("\n⚠ Ollama service is not running!")
                print("Attempting to start Ollama...")

            if ollama_test.start_ollama():
                if not use_tui:
                    print("✓ Ollama service started successfully")
            else:
                if not use_tui:
                    print("\n❌ Failed to start Ollama automatically.")
                    print("Please start Ollama manually with: ollama serve")
                    print("Then restart this advisor.\n")
                exit(1)
        elif not use_tui:
            print("✓ Ollama service is running")

        self.game_state_mgr = GameStateManager(card_db)
        self.ai_advisor = AIAdvisor(card_db=card_db)
        self.tts = TextToSpeech(voice="am_adam", volume=1.0)
        self.log_follower = LogFollower(self.log_path)

        # Initialize draft advisor if available
        self.draft_advisor = None
        self.deck_builder = None
        if DRAFT_ADVISOR_AVAILABLE:
            try:
                rag_system = self.ai_advisor.rag_system if hasattr(self.ai_advisor, 'rag_system') else None
                ollama_client = self.ai_advisor.client if hasattr(self.ai_advisor, 'client') else None
                self.draft_advisor = DraftAdvisor(card_db, rag_system, ollama_client)
                self.deck_builder = DeckBuilder()

                # Register draft event callbacks with GameStateManager
                self.game_state_mgr.register_draft_callback("EventGetCoursesV2", self._on_draft_pool)
                self.game_state_mgr.register_draft_callback("LogBusinessEvents", self._on_premier_draft_pick)
                self.game_state_mgr.register_draft_callback("Draft.Notify", self._on_draft_notify)
                self.game_state_mgr.register_draft_callback("BotDraftDraftStatus", self._on_quick_draft_status)
                self.game_state_mgr.register_draft_callback("BotDraftDraftPick", self._on_quick_draft_pick)

                if not use_tui:
                    print("✓ Draft advisor enabled")
                    print("✓ Deck builder enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize draft advisor: {e}")
                self.draft_advisor = None
                self.deck_builder = None

        self.last_turn_advised = -1
        self.advice_thread = None
        self.first_turn_detected = False
        self.cli_thread = None
        self.running = True
        self._deck_suggestions_generated = False  # Track if deck suggestions shown
        self._last_announced_pick = None  # Deduplication: track (pack_num, pick_num) of last TTS announcement

        # Fetch available Ollama models
        self.available_models = self._fetch_ollama_models()

    def _output(self, message: str, color: str = "white"):
        """Output message to either CLI, TUI, or GUI"""
        if self.use_gui and self.gui:
            self.gui.add_message(message, color)
        elif self.use_tui and self.tui:
            self.tui.add_message(message, color)
        else:
            print(message)

    def _update_status(self, board_state: BoardState = None):
        """Update status bar for TUI or GUI"""
        if board_state:
            status = f"Turn {board_state.current_turn} | Model: {self.ai_advisor.client.model} | Voice: {self.tts.voice} | Vol: {int(self.tts.volume * 100)}%"
        else:
            status = f"Model: {self.ai_advisor.client.model} | Voice: {self.tts.voice} | Vol: {int(self.tts.volume * 100)}%"

        if self.use_gui and self.gui:
            self.gui.set_status(status)
        elif self.use_tui and self.tui:
            self.tui.set_status(status)

    def get_last_rag_references(self) -> Optional[Dict]:
        """Proxy method to get RAG references from AI advisor"""
        if hasattr(self, 'ai_advisor') and hasattr(self.ai_advisor, 'get_last_rag_references'):
            return self.ai_advisor.get_last_rag_references()
        return None

    # GUI Callback Methods
    def _on_gui_model_change(self, model):
        """Handle model change from GUI with validation and auto-pull"""
        model = model.strip()
        if not model:
            return

        # Check if model is already available locally
        if model in self.available_models:
            self.ai_advisor.client.model = model
            self._update_status()
            logging.info(f"Model changed to: {model}")
            self._output(f"✓ Model changed to: {model}", "green")
            return

        # Model not found locally - try to pull it
        self._output(f"⏳ Model '{model}' not found locally. Attempting to pull from Ollama...", "yellow")
        logging.info(f"Attempting to pull model: {model}")

        try:
            # Use Ollama API to pull the model
            import subprocess
            result = subprocess.run(
                ['ollama', 'pull', model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Refresh available models
                self.available_models = self._fetch_ollama_models()

                # Update GUI dropdown
                if self.gui:
                    self.gui.model_dropdown['values'] = self.available_models

                # Set the new model
                self.ai_advisor.client.model = model
                self._update_status()
                logging.info(f"Successfully pulled and switched to model: {model}")
                self._output(f"✓ Successfully pulled and loaded: {model}", "green")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logging.error(f"Failed to pull model {model}: {error_msg}")
                self._output(f"✗ Failed to pull model '{model}': {error_msg}", "red")
                self._output(f"Available models: {', '.join(self.available_models)}", "cyan")
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout while pulling model: {model}")
            self._output(f"✗ Timeout while pulling model '{model}' (>5 minutes)", "red")
        except FileNotFoundError:
            logging.error("Ollama CLI not found in PATH")
            self._output(f"✗ Ollama CLI not found. Please install Ollama first.", "red")
        except Exception as e:
            logging.error(f"Error pulling model {model}: {e}")
            self._output(f"✗ Error pulling model '{model}': {str(e)}", "red")

    def _on_gui_voice_change(self, voice):
        """Handle voice change from GUI"""
        self.tts.voice = voice
        self._update_status()
        logging.info(f"Voice changed to: {voice}")

    def _on_gui_tts_engine_change(self, engine):
        """Handle TTS engine change from GUI"""
        old_volume = self.tts.volume
        if engine == "bark":
            new_voice = self.BARK_VOICES[0]
        else:
            new_voice = "am_adam"
        self.tts = TextToSpeech(voice=new_voice, volume=old_volume, force_engine=engine)
        self._update_status()
        # Update GUI with new voice list
        if self.gui:
            self.gui.update_settings(
                self.available_models,
                self.AVAILABLE_VOICES,
                self.BARK_VOICES,
                self.ai_advisor.client.model,
                self.tts.voice,
                int(self.tts.volume * 100),
                self.tts.tts_engine
            )
        logging.info(f"TTS engine changed to: {engine}")

    def _on_gui_volume_change(self, volume):
        """Handle volume change from GUI"""
        self.tts.volume = volume / 100.0
        logging.debug(f"Volume set to: {volume}%")

    def _fetch_ollama_models(self) -> list:
        """Query Ollama API for locally installed models"""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = data.get('models', [])
                # Extract model names and sort them
                model_names = sorted([m['name'] for m in models])
                logging.info(f"Found {len(model_names)} Ollama models: {model_names}")
                return model_names
        except Exception as e:
            logging.warning(f"Could not fetch Ollama models: {e}")
            # Return default model as fallback
            return ["llama3.2"]

    # Draft event callbacks
    def _on_draft_pool(self, data: dict):
        """Handle EventGetCoursesV2 - sealed/draft pool event"""
        if not self.draft_advisor:
            return

        try:
            courses = data.get("Courses", [])
            if not courses:
                return

            for course in courses:
                event_name = course.get("InternalEventName", "")
                card_pool = course.get("CardPool", [])
                current_module = course.get("CurrentModule", "")

                # Check if this is a limited event
                if any(fmt in event_name for fmt in ["Sealed", "Draft"]) and card_pool:
                    logging.info(f"Draft pool detected: {event_name} with {len(card_pool)} cards, CurrentModule={current_module}")

                    # Store the most recent card pool and event info for manual deck suggestions
                    self._last_card_pool = card_pool
                    self._last_draft_event_name = event_name
                    self._last_draft_module = current_module

                    # Suppress output - this event fires for all available drafts (including completed ones)
                    # Only show if we actually add Sealed deck building feature
                    # self._output(f"\n📦 Draft Pool Detected: {event_name}", "cyan")
                    # self._output(f"   {len(card_pool)} cards in pool", "cyan")

        except Exception as e:
            logging.error(f"Error handling draft pool event: {e}")

    def _on_draft_notify(self, data: dict):
        """Handle Draft.Notify - Premier Draft pack display"""
        if not self.draft_advisor:
            return

        try:
            # Extract pack information (already 1-indexed from parsing)
            pack_num = data.get("PackNumber", 1)
            pick_num = data.get("PickNumber", 1)
            pack_arena_ids = data.get("PackCards", [])
            draft_id = data.get("DraftId", "")

            if not pack_arena_ids:
                return

            logging.info(f"Draft.Notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")
            print(f"[DEBUG] Processing Draft.Notify: Pack {pack_num}, Pick {pick_num}")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Track previous pack to infer picked card
            # If we have a previous pack and we're now on a new pick, infer what was picked
            if hasattr(self, '_last_draft_pack') and hasattr(self, '_last_draft_recommendation'):
                last_pack_num = getattr(self, '_last_draft_pack_num', 0)
                last_pick_num = getattr(self, '_last_draft_pick_num', 0)

                # Check if we advanced to next pick (same pack, next pick OR next pack, pick 1)
                pick_advanced = (pack_num == last_pack_num and pick_num == last_pick_num + 1) or \
                               (pack_num == last_pack_num + 1 and pick_num == 1)

                if pick_advanced and self._last_draft_recommendation:
                    # Record the recommended card as picked (user likely followed recommendation)
                    # Note: This is an inference - we don't have explicit pick data for Premier Draft
                    picked_card = self._last_draft_recommendation
                    if picked_card and picked_card not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(picked_card)
                        logging.info(f"Inferred pick: {picked_card} (Pack {last_pack_num}, Pick {last_pick_num})")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, draft_id
            )

            # Store current pack info for next iteration
            self._last_draft_pack = pack_arena_ids
            self._last_draft_pack_num = pack_num
            self._last_draft_pick_num = pick_num
            self._last_draft_recommendation = pack_cards[0].name if pack_cards else None

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                from draft_advisor import format_draft_pack_for_gui
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            elif self.use_tui and self.tui:
                # TUI display
                from draft_advisor import format_draft_pack_for_gui
                lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db
                )
                self.tui.set_board_state(lines)
            else:
                # Terminal display
                from draft_advisor import display_draft_pack
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Check if draft is complete
            # Standard draft: Pack 3, Pick 15 or >= 45 cards
            # Pick Two draft: >= 21 cards (7 picks × 2 cards × ~1.5 packs)
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if pack_num == 3 and pick_num == 15 and not pick_two_mode:
                # Standard draft final pick detected
                logging.info("Premier Draft final pick detected - waiting for completion")
            elif len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                # We've tracked enough picks - draft complete!
                logging.info(f"Draft complete! {len(self.draft_advisor.picked_cards)} cards picked (min: {min_cards_for_completion})")
                self._generate_deck_suggestions(draft_id)

        except Exception as e:
            logging.error(f"Error handling Draft.Notify: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _on_premier_draft_pick(self, data: dict):
        """Handle LogBusinessEvents - Premier Draft pick"""
        if not self.draft_advisor:
            return

        try:
            # Extract pack information
            event_id = data.get("EventId", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            cards_in_pack = data.get("CardsInPack", [])

            if not cards_in_pack:
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in cards_in_pack]

            logging.info(f"Premier Draft Pick: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Track previous pack to infer picked card (similar to Draft.Notify)
            if hasattr(self, '_last_premier_pack') and hasattr(self, '_last_premier_recommendation'):
                last_pack_num = getattr(self, '_last_premier_pack_num', 0)
                last_pick_num = getattr(self, '_last_premier_pick_num', 0)

                # Check if we advanced to next pick
                pick_advanced = (pack_num == last_pack_num and pick_num == last_pick_num + 1) or \
                               (pack_num == last_pack_num + 1 and pick_num == 1)

                if pick_advanced and self._last_premier_recommendation:
                    picked_card = self._last_premier_recommendation
                    if picked_card and picked_card not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(picked_card)
                        logging.info(f"Inferred pick: {picked_card} (Pack {last_pack_num}, Pick {last_pick_num})")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_id
            )

            # Store current pack info for next iteration
            self._last_premier_pack = pack_arena_ids
            self._last_premier_pack_num = pack_num
            self._last_premier_pick_num = pick_num
            self._last_premier_recommendation = pack_cards[0].name if pack_cards else None

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Check if draft is complete
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                logging.info(f"Draft complete! {len(self.draft_advisor.picked_cards)} cards picked")
                self._generate_deck_suggestions(event_id)

        except Exception as e:
            logging.error(f"Error handling Premier Draft pick: {e}")

    def _on_quick_draft_status(self, data: dict):
        """Handle BotDraftDraftStatus - Quick Draft status (pack + pool)"""
        print(f"[DEBUG] _on_quick_draft_status called!")
        if not self.draft_advisor:
            print(f"[DEBUG] No draft advisor available!")
            return

        try:
            event_name = data.get("EventName", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            draft_pack = data.get("DraftPack", [])
            picked_cards = data.get("PickedCards", [])

            if not draft_pack:
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in draft_pack]

            logging.info(f"Quick Draft Status: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Update draft advisor's picked cards list
            for card_id in picked_cards:
                try:
                    card_name = self.game_state_mgr.card_lookup.get_card_name(int(card_id))
                    if card_name and card_name not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(card_name)
                except Exception as e:
                    logging.debug(f"Error recording picked card: {e}")

            # Check if draft is complete
            pick_two_mode = False
            if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                pick_two_mode = self.gui.pick_two_draft_var.get()

            min_cards_for_completion = 21 if pick_two_mode else 40

            if len(self.draft_advisor.picked_cards) >= min_cards_for_completion:
                self._generate_deck_suggestions(event_name)

        except Exception as e:
            logging.error(f"Error handling Quick Draft status: {e}")

    def _on_quick_draft_pick(self, data: dict):
        """Handle BotDraftDraftPick - Quick Draft pick confirmation AND next pack"""
        print(f"[DEBUG] _on_quick_draft_pick called! Data keys: {list(data.keys())}")

        # BotDraftDraftPick response includes the NEXT pack to pick from!
        # So we need to process it here, not wait for BotDraftDraftStatus
        if not self.draft_advisor:
            print(f"[DEBUG] No draft advisor available!")
            return

        try:
            event_name = data.get("EventName", "")
            pack_num = data.get("PackNumber", 0) + 1  # Convert to 1-indexed
            pick_num = data.get("PickNumber", 0) + 1  # Convert to 1-indexed
            draft_pack = data.get("DraftPack", [])
            picked_cards = data.get("PickedCards", [])

            print(f"[DEBUG] Processing: Pack {pack_num}, Pick {pick_num}, {len(draft_pack)} cards in pack")

            if not draft_pack:
                print(f"[DEBUG] No draft pack in this response")
                return

            # Convert string IDs to integers
            pack_arena_ids = [int(card_id) for card_id in draft_pack]

            logging.info(f"Quick Draft Pick: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Reset state for new draft
            if pack_num == 1 and pick_num == 1:
                self._deck_suggestions_generated = False
                self._last_announced_pick = None  # Reset deduplication tracking for new draft
                self.draft_advisor.picked_cards = []
                logging.info("New draft detected - reset state")

            # Generate pick recommendation
            pack_cards, recommendation = self.draft_advisor.recommend_pick(
                pack_arena_ids, pack_num, pick_num, event_name
            )

            print(f"[DEBUG] Got {len(pack_cards)} cards, recommendation: {recommendation}")

            # Display based on mode
            # Get metadata DB for showing picked cards by color
            metadata_db = None
            if self.draft_advisor and self.draft_advisor.rag and hasattr(self.draft_advisor.rag, 'card_metadata'):
                metadata_db = self.draft_advisor.rag.card_metadata

            if self.use_gui and self.gui:
                # Format for GUI with split panes (draft pool in board, picked cards in advisor)
                pack_lines, picked_lines = format_draft_pack_for_gui(
                    pack_cards, pack_num, pick_num, recommendation,
                    picked_cards=self.draft_advisor.picked_cards,
                    card_metadata_db=metadata_db,
                    split_panes=True
                )
                # Determine total cards needed based on Pick Two checkbox
                pick_two_mode = self.gui.pick_two_draft_var.get() if hasattr(self.gui, 'pick_two_draft_var') else False
                total_needed = 21 if pick_two_mode else 45

                self.gui.set_draft_panes(
                    pack_lines, picked_lines,
                    picked_count=len(self.draft_advisor.picked_cards),
                    total_needed=total_needed
                )
                self.gui.add_message(f"Pack {pack_num}, Pick {pick_num}: {recommendation}", "cyan")
                print(f"[DEBUG] Sent to GUI!")
            else:
                # Terminal display
                display_draft_pack(pack_cards, pack_num, pick_num, recommendation)

            # Speak recommendation if TTS enabled
            # Deduplication: only speak if this is a new pick (not already announced)
            if self.tts and pack_cards:
                current_pick = (pack_num, pick_num)
                if current_pick != self._last_announced_pick:
                    self._last_announced_pick = current_pick

                    # Check if Pick Two Draft mode is enabled
                    pick_two_mode = False
                    if self.use_gui and self.gui and hasattr(self.gui, 'pick_two_draft_var'):
                        pick_two_mode = self.gui.pick_two_draft_var.get()

                    if pick_two_mode and len(pack_cards) >= 2:
                        # Read both top picks
                        top_pick = pack_cards[0].name
                        second_pick = pack_cards[1].name
                        self.tts.speak(f"Pick {top_pick} and {second_pick}")
                    else:
                        # Read only top pick
                        top_pick = pack_cards[0].name
                        self.tts.speak(f"Pick {top_pick}")

            # Update draft advisor's picked cards list
            for card_id in picked_cards:
                try:
                    card_name = self.game_state_mgr.card_lookup.get_card_name(int(card_id))
                    if card_name and card_name not in self.draft_advisor.picked_cards:
                        self.draft_advisor.record_pick(card_name)
                except Exception as e:
                    logging.debug(f"Error recording picked card: {e}")

            # Check if draft is complete (use lower thresholds since pick inference isn't perfect)
            # Standard draft: 40+ cards (out of 45), Pick Two: 18+ cards (out of 21)
            num_cards_to_pick = data.get("NumCardsToPick", 1)
            total_picks_needed = 18 if num_cards_to_pick == 2 else 40

            if len(self.draft_advisor.picked_cards) >= total_picks_needed:
                self._generate_deck_suggestions(event_name)

        except Exception as e:
            logging.error(f"Error handling Quick Draft pick: {e}")
            import traceback
            traceback.print_exc()

    def _generate_deck_suggestions(self, event_name: str):
        """Generate and display deck building suggestions"""
        if not self.deck_builder or not self.draft_advisor:
            return

        # Prevent duplicate suggestions - only generate once per draft
        if hasattr(self, '_deck_suggestions_generated') and self._deck_suggestions_generated:
            logging.debug("Deck suggestions already generated for this draft")
            return

        try:
            # Extract set code from event name (e.g., "QuickDraft_BLB_20250815" -> "BLB")
            set_code = None
            if event_name and "_" in event_name:
                parts = event_name.split("_")
                if len(parts) >= 2:
                    set_code = parts[1].upper()

            if not set_code:
                logging.warning("Could not determine set code from event name")
                return

            drafted_cards = self.draft_advisor.picked_cards

            # Minimum card check: PickTwoDraft has 21 picks, standard draft has 45
            # We need at least 20 cards to build a reasonable deck
            if not drafted_cards or len(drafted_cards) < 20:
                logging.warning(f"Not enough cards drafted: {len(drafted_cards)}")
                return

            logging.info(f"🏗️  Generating deck suggestions for {len(drafted_cards)} drafted cards from {set_code}...")

            # Generate suggestions (top 3 color pairs)
            suggestions = self.deck_builder.suggest_deck(
                drafted_cards,
                set_code,
                top_n=3
            )

            if not suggestions:
                self._output("⚠️  No deck suggestions available (17lands data may be missing)", "yellow")
                return

            # Display based on mode
            if self.use_gui and self.gui:
                # Format all suggestions for GUI
                all_lines = []
                all_lines.append("="*80)
                all_lines.append("DRAFT COMPLETE - Deck Suggestions")
                all_lines.append("="*80)
                all_lines.append("")

                for i, suggestion in enumerate(suggestions, 1):
                    if i == 1:
                        all_lines.append(f"🏆 BEST MATCH: {suggestion.color_pair_name}")
                    else:
                        all_lines.append(f"📊 ALTERNATIVE #{i-1}: {suggestion.color_pair_name}")
                    all_lines.append("")

                    # Add formatted deck suggestion
                    deck_lines = format_deck_suggestion_for_gui(suggestion)
                    all_lines.extend(deck_lines)

                all_lines.append("ℹ️  Copy the suggested deck into MTGA, then return here for gameplay advice!")
                all_lines.append("="*80)

                # Display in GUI
                self.gui.reset_pane_labels()  # Reset to "BOARD STATE" and "ADVISOR"
                self.gui.set_board_state(all_lines)
                self.gui.add_message(f"Draft complete! Suggested: {suggestions[0].color_pair_name}", "green")

                # Speak the top suggestion
                if self.tts:
                    self.tts.speak(f"Suggested deck: {suggestions[0].color_pair_name}")

            else:
                # Terminal display
                self._output("\n" + "="*80, "cyan")
                self._output("DRAFT COMPLETE - Building Deck Suggestions...", "cyan")
                self._output("="*80 + "\n", "cyan")

                for i, suggestion in enumerate(suggestions, 1):
                    if i == 1:
                        self._output(f"\n🏆 BEST MATCH: {suggestion.color_pair_name}", "green")
                    else:
                        self._output(f"\n📊 ALTERNATIVE #{i-1}: {suggestion.color_pair_name}", "cyan")

                    display_deck_suggestion(suggestion)

                    # Speak the top suggestion
                    if i == 1 and self.tts:
                        self.tts.speak(f"Suggested deck: {suggestion.color_pair_name}")

                self._output("\nℹ️  Copy the suggested deck into MTGA, then return here for gameplay advice!", "cyan")
                self._output("="*80 + "\n", "cyan")

            # Mark suggestions as generated to prevent duplicates
            self._deck_suggestions_generated = True

        except Exception as e:
            logging.error(f"Error generating deck suggestions: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Start the advisor with background log monitoring and interactive CLI"""
        if self.use_gui:
            self.run_gui()
            return
        if self.use_tui:
            self.run_tui()
            return

        print("\n" + "="*60)
        print("MTGA Voice Advisor Started")
        print("="*60)
        print(f"Log: {self.log_path}")
        print(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)")
        print(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%")
        print("\nWaiting for a match... (Enable Detailed Logs in MTGA settings)")
        print("Type /help for commands\n")

        # Speak startup message so user knows audio is working
        startup_msg = "All systems online. Waiting for match."
        self.tts.speak(startup_msg)

        # Start log follower in background thread
        log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
        log_thread.start()

        # Interactive CLI loop
        self._run_cli_loop()

    def run_tui(self):
        """Start the advisor with TUI interface"""
        def _tui_main(stdscr):
            # Initialize TUI
            self.tui = AdvisorTUI(stdscr)

            # Display startup message
            self._output(f"✓ MTGA Voice Advisor Started", "green")
            self._output(f"Log: {self.log_path}", "blue")
            self._output(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)", "blue")
            self._output(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%", "blue")
            self._output(f"✓ Ollama service connected", "green")
            self._output("", "white")
            self._output("Waiting for a match... (Enable Detailed Logs in MTGA settings)", "cyan")
            self._output("Type /help for commands", "cyan")

            # Update status bar
            self._update_status()

            # Speak startup message so user knows audio is working
            startup_msg = "All systems online. Waiting for match."
            self.tts.speak(startup_msg)

            # Start log follower in background thread
            log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
            log_thread.start()

            # Set up input callback
            def on_input(user_input: str):
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_query(user_input)

            # TUI event loop
            try:
                while self.running:
                    # Handle input (non-blocking)
                    self.tui.get_input(on_input)
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.05)
            except KeyboardInterrupt:
                self._output("\nShutting down...", "yellow")
                self.running = False
                self.log_follower.close()
            finally:
                self.tui.cleanup()

                # Save preferences before exiting
                if self.prefs and CONFIG_MANAGER_AVAILABLE:
                    self.prefs.save()
                    logging.info("User preferences saved")

                # Clean up database connection
                if hasattr(self.game_state_mgr.card_lookup, 'close'):
                    self.game_state_mgr.card_lookup.close()
                    logging.info("Card database connection closed")

        curses.wrapper(_tui_main)

    def run_gui(self):
        """Start the advisor with Tkinter GUI interface"""
        if not TKINTER_AVAILABLE:
            print("ERROR: Tkinter is not available. Install with: sudo apt-get install python3-tk")
            exit(1)

        # Create Tk root
        self.tk_root = tk.Tk()
        self.gui = AdvisorGUI(self.tk_root, self)

        # Initialize settings
        self.gui.update_settings(
            self.available_models,
            self.AVAILABLE_VOICES,
            self.BARK_VOICES,
            self.ai_advisor.client.model,
            self.tts.voice,
            int(self.tts.volume * 100),
            self.tts.tts_engine
        )

        # Display startup messages
        self._output(f"✓ MTGA Voice Advisor Started", "green")
        self._output(f"Log: {self.log_path}", "cyan")
        self._output(f"AI Model: {self.ai_advisor.client.model} ({len(self.available_models)} available)", "cyan")
        self._output(f"Voice: {self.tts.voice} | Volume: {int(self.tts.volume * 100)}%", "cyan")
        self._output(f"✓ Ollama service connected", "green")
        self._output("", "white")
        self._output("Waiting for a match... (Enable Detailed Logs in MTGA settings)", "cyan")

        # Update status bar
        self._update_status()

        # Speak startup message so user knows audio is working
        startup_msg = "All systems online. Waiting for match."
        self.tts.speak(startup_msg)

        # Start log follower in background thread
        log_thread = threading.Thread(target=self._run_log_monitor, daemon=True)
        log_thread.start()

        # Run Tk main loop
        try:
            self.tk_root.mainloop()
        except KeyboardInterrupt:
            self._output("\nShutting down...", "yellow")
        finally:
            self.running = False
            if hasattr(self, 'log_follower'):
                self.log_follower.close()
            if hasattr(self.game_state_mgr.card_lookup, 'close'):
                self.game_state_mgr.card_lookup.close()
                logging.info("Card database connection closed")
            if self.gui:
                self.gui.cleanup()

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
                except EOFError:
                    # Running in background without stdin - just sleep and continue
                    logging.info("Running in background mode (no stdin available)")
                    while self.running:
                        time.sleep(1)
                    break
                except KeyboardInterrupt:
                    print("\n\nShutting down...")
                    self.running = False
                    self.log_follower.close()
                    break
        except Exception as e:
            logging.error(f"CLI error: {e}")
        finally:
            # Save preferences before exiting
            if self.prefs and CONFIG_MANAGER_AVAILABLE:
                self.prefs.save()
                logging.info("User preferences saved")

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
        elif cmd == "/quit" or cmd == "/exit":
            self._output("Exiting...", "yellow")
            self.running = False
            if self.use_tui and self.tui:
                self.tui.running = False
        elif cmd == "/clear":
            if self.use_tui and self.tui:
                self.tui.messages.clear()
                self.tui._refresh_messages()
                logging.info("Message history cleared")
            else:
                print("\n" * 50)  # Clear console
        elif cmd == "/settings":
            if self.use_tui and self.tui:
                self._show_settings_interactive()
            else:
                self._show_settings()
        elif cmd == "/voice":
            if len(parts) > 1:
                voice = parts[1].lower()
                if voice in self.AVAILABLE_VOICES:
                    self.tts.set_voice(voice)
                    self._output(f"✓ Voice changed to: {voice}", "green")
                    self._update_status()
                else:
                    self._output(f"✗ Unknown voice. Available: {', '.join(self.AVAILABLE_VOICES[:5])}...", "red")
            else:
                self._output(f"✓ Current voice: {self.tts.voice}", "green")
        elif cmd == "/volume":
            if len(parts) > 1:
                try:
                    vol = float(parts[1]) / 100.0
                    self.tts.set_volume(vol)
                    self._output(f"✓ Volume set to: {int(vol * 100)}%", "green")
                    self._update_status()
                except ValueError:
                    self._output("✗ Volume must be a number (0-100)", "red")
            else:
                self._output(f"✓ Current volume: {int(self.tts.volume * 100)}%", "green")
        elif cmd == "/status":
            board_state = self.game_state_mgr.get_current_board_state()
            if board_state:
                self._output(f"Turn: {board_state.current_turn} | Your Turn: {board_state.is_your_turn} | Has Priority: {board_state.has_priority}", "cyan")
                self._output(f"Your Hand: {len(board_state.your_hand)} cards", "blue")
                self._output(f"Your Battlefield: {len(board_state.your_battlefield)} permanents", "blue")
                self._output(f"Opponent Battlefield: {len(board_state.opponent_battlefield)} permanents", "blue")
            else:
                self._output("No match in progress", "yellow")
        elif cmd == "/tts":
            if self.tts.tts_engine:
                engine_name = "Kokoro" if self.tts.tts_engine == "kokoro" else "BarkTTS"
                self._output(f"✓ TTS Engine: {engine_name} ({self.tts.tts_engine})", "green")
                if self.tts.tts_engine == "kokoro":
                    self._output(f"  Voice: {self.tts.voice}", "blue")
                else:
                    self._output(f"  Using Bark's built-in voice presets", "blue")
            else:
                self._output("✗ No TTS engine initialized", "red")
        elif cmd == "/models":
            model_lines = [
                f"Available Ollama models ({len(self.available_models)}):",
                ""
            ]
            for i, model in enumerate(self.available_models, 1):
                marker = " (current)" if model == self.ai_advisor.client.model else ""
                model_lines.append(f"  {i}. {model}{marker}")
            model_lines.append("")
            model_lines.append("Use '/model <name>' to switch models")

            if self.use_tui and self.tui:
                self.tui.show_popup(model_lines, "Available Models")
            else:
                for line in model_lines:
                    print(line)
        elif cmd == "/model":
            if len(parts) > 1:
                model_name = parts[1]
                # Check if model exists in available models (support partial match)
                matching_models = [m for m in self.available_models if model_name in m]
                if matching_models:
                    if len(matching_models) == 1:
                        selected_model = matching_models[0]
                        self.ai_advisor.client.model = selected_model
                        self._output(f"✓ Model changed to: {selected_model}", "green")
                        self._update_status()
                        logging.info(f"Ollama model switched to: {selected_model}")
                    else:
                        self._output(f"✗ Ambiguous model name. Matching models: {', '.join(matching_models)}", "red")
                        self._output(f"  Please be more specific.", "red")
                else:
                    self._output(f"✗ Model not found: {model_name}", "red")
                    self._output(f"  Available models: {', '.join(self.available_models[:5])}{'...' if len(self.available_models) > 5 else ''}", "yellow")
            else:
                self._output(f"✓ Current model: {self.ai_advisor.client.model}", "green")
        elif cmd == "/continuous":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.continuous_monitoring = True
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = True
                    self._output("✓ Continuous monitoring ENABLED - AI will alert you of critical changes anytime", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.continuous_monitoring = False
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = False
                    self._output("✓ Continuous monitoring DISABLED - advice only when you have priority", "yellow")
                else:
                    self._output("✗ Use '/continuous on' or '/continuous off'", "red")
            else:
                status = "ENABLED" if self.continuous_monitoring else "DISABLED"
                self._output(f"✓ Continuous monitoring: {status}", "green")
        elif cmd == "/opponent-alerts":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.continuous_monitoring = True
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = True
                    self._output("✓ Opponent Turn Alerts: ON", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.continuous_monitoring = False
                    if self.prefs:
                        self.prefs.opponent_turn_alerts = False
                    self._output("✓ Opponent Turn Alerts: OFF", "yellow")
                else:
                    self._output("✗ Use '/opponent-alerts on' or '/opponent-alerts off'", "red")
            else:
                status = "ON" if self.continuous_monitoring else "OFF"
                self._output(f"Opponent Turn Alerts: {status}", "green")
        elif cmd == "/reskins":
            if len(parts) > 1:
                setting = parts[1].lower()
                if setting in ["on", "true", "1", "yes"]:
                    self.game_state_mgr.card_lookup.show_reskin_names = True
                    if self.prefs:
                        self.prefs.reskin_names = True
                    self._output("✓ Reskin names: ON (Spider-Man variants enabled)", "green")
                elif setting in ["off", "false", "0", "no"]:
                    self.game_state_mgr.card_lookup.show_reskin_names = False
                    if self.prefs:
                        self.prefs.reskin_names = False
                    self._output("✓ Reskin names: OFF (canonical names)", "yellow")
                else:
                    self._output("✗ Use '/reskins on' or '/reskins off'", "red")
            else:
                status = "ON" if self.game_state_mgr.card_lookup.show_reskin_names else "OFF"
                self._output(f"Reskin names: {status}", "green")
        else:
            self._output(f"Unknown command: {cmd}. Type /help for commands.", "red")

    def _handle_query(self, query: str):
        """Handle free-form queries to the AI about current board state"""
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state or not board_state.current_turn:
            self._output("No match in progress. Start a game first.", "yellow")
            return

        self._output("\nThinking...", "cyan")
        # Use the AI to answer the query in context of current board state
        prompt = f"""
The user is asking about their current board state in Magic: The Gathering Arena.

Current Board State:
- Current Turn: {board_state.current_turn}
- Your Hand: {', '.join([c.name for c in board_state.your_hand]) if board_state.your_hand else 'Empty'}
- Your Battlefield: {', '.join([c.name for c in board_state.your_battlefield]) if board_state.your_battlefield else 'Empty'}
- Opponent Battlefield: {', '.join([c.name for c in board_state.opponent_battlefield]) if board_state.opponent_battlefield else 'Empty'}

User's Question: {query}

Provide a concise answer (1-2 sentences) based on the board state.
"""
        try:
            response = self.ai_advisor.client.generate(prompt)
            if response:
                # Parse thinking from answer
                thinking, answer = self._parse_reasoning_response(response)

                # Display thinking (not spoken)
                if thinking:
                    self._output("💭 Thinking:", "blue")
                    for line in thinking.split("\n"):
                        if line.strip():
                            self._output(f"   {line.strip()}", "blue")
                    self._output("", "white")

                # Display and speak answer
                final_answer = answer if answer else response
                self._output(f"Advisor: {final_answer}\n", "green")
                clean_response = self._strip_markdown(final_answer)
                self.tts.speak(clean_response)
            else:
                self._output("No response from AI.\n", "yellow")
        except Exception as e:
            self._output(f"Error getting response: {e}\n", "red")
            logging.error(f"Query error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _show_help(self):
        """Display help menu"""
        help_lines = [
            "",
            "Commands:",
            "  /help                    - Show this help menu",
            "  /settings                - Interactive settings menu (TUI: ↑↓ Enter ESC)",
            "  /clear                   - Clear message history",
            "  /quit or /exit           - Exit the advisor",
            "  /opponent-alerts [on/off] - Toggle opponent turn alerts (alerts for critical changes anytime)",
            "  /reskins [on/off]        - Show Spider-Man reskin names (e.g., /reskins on)",
            "  /tts                     - Show active TTS engine (Kokoro or BarkTTS)",
            "  /voice [name]            - Change voice (e.g., /voice bella, /voice v2/en_speaker_3)",
            "  /volume [0-100]          - Set volume (e.g., /volume 80)",
            "  /models                  - List available Ollama models",
            "  /model [name]            - Change AI model (e.g., /model llama3.2, /model qwen)",
            "  /status                  - Show current board state",
            "",
            "Interactive Settings (TUI mode):",
            "  ↑/↓ arrows      - Navigate settings",
            "  Enter/Space     - Cycle through options (Model, Voice, TTS Engine)",
            "  +/-             - Adjust volume",
            "  ESC or Q        - Close settings menu",
            "",
            "Free-form queries:",
            "  Type any question about your board state and the advisor will answer.",
            "",
            "Model Selection:",
            "  Smaller models = faster but less smart (e.g., llama3.2:1b)",
            "  Larger models = slower but smarter (e.g., llama3.2:70b)",
            "",
            "TTS Engines:",
            "  Kokoro: Fast, high-quality, 30+ voices (primary)",
            "  BarkTTS: Fallback engine with built-in voices",
            "  Toggle between them in /settings menu",
            "",
        ]

        if self.use_tui and self.tui:
            self.tui.show_popup(help_lines, "Help")
        else:
            for line in help_lines:
                print(line)

    def _show_settings(self):
        """Show current settings (non-interactive)"""
        self._output("", "white")
        self._output("Current Settings:", "cyan")
        self._output(f"  AI Model: {self.ai_advisor.client.model}", "white")

        # TTS Engine info
        if self.tts.tts_engine:
            engine_name = "Kokoro" if self.tts.tts_engine == "kokoro" else "BarkTTS"
            self._output(f"  TTS:      {engine_name}", "white")
            if self.tts.tts_engine == "kokoro":
                self._output(f"  Voice:    {self.tts.voice}", "white")
        else:
            self._output(f"  TTS:      None (initialization failed)", "white")

        self._output(f"  Volume:   {int(self.tts.volume * 100)}%", "white")
        self._output(f"  Log:      {self.log_path}", "white")
        self._output("", "white")

    def _show_settings_interactive(self):
        """Show interactive settings menu (TUI only)"""
        def settings_callback(action, value):
            if action == "get_values":
                return (
                    self.available_models,
                    self.AVAILABLE_VOICES,
                    self.BARK_VOICES,
                    self.ai_advisor.client.model,
                    self.tts.voice,
                    int(self.tts.volume * 100),
                    self.tts.tts_engine
                )
            elif action == "model":
                self.ai_advisor.client.model = value
                self._update_status()
                # Don't spam message log - just update status bar
                logging.info(f"Model changed to: {value}")
            elif action == "voice":
                self.tts.voice = value
                self._update_status()
                # Don't spam message log - just update status bar
                logging.info(f"Voice changed to: {value}")
            elif action == "volume":
                self.tts.volume = value / 100.0
                self._update_status()
                # Don't spam message log - just update status bar
                logging.debug(f"Volume set to: {value}%")
            elif action == "tts_engine":
                # Reinitialize TTS with new engine and appropriate default voice
                old_volume = self.tts.volume
                # Set default voice for the new engine
                if value == "bark":
                    new_voice = self.BARK_VOICES[0]  # Default to first bark voice
                else:
                    new_voice = "am_adam"  # Default kokoro voice
                self.tts = TextToSpeech(voice=new_voice, volume=old_volume, force_engine=value)
                self._update_status()
                engine_name = "Kokoro" if value == "kokoro" else "BarkTTS"
                # Don't spam message log - just update status bar
                logging.info(f"TTS engine changed to: {value} with voice {new_voice}")

        self.tui.show_settings_menu(settings_callback)

    def on_line(self, line: str):
        """Parse log line and update game state"""
        logging.debug(f"Received line in on_line: {line[:100]}...")
        state_changed = self.game_state_mgr.parse_log_line(line)
        if state_changed:
            logging.debug("Game state changed. Checking for decision point.")
            self._check_for_decision_point()

    def _check_for_decision_point(self):
        """Check if we should give automatic advice or important updates"""
        logging.debug("Checking for decision point...")
        board_state = self.game_state_mgr.get_current_board_state()
        if not board_state:
            logging.debug("No board state available yet.")
            return

        # Check for mulligan phase
        logging.debug(f"Board state in_mulligan_phase: {board_state.in_mulligan_phase}, game_stage: {board_state.game_stage}")

        if board_state.in_mulligan_phase:
            logging.info("🎴 IN MULLIGAN PHASE - will display hand and give advice")
            # Display board state with opening hand
            self._display_board_state(board_state)

            # Only generate mulligan advice once
            if not hasattr(self, 'mulligan_advice_given') or not self.mulligan_advice_given:
                logging.info("🎴 Mulligan phase detected - generating mulligan advice")
                self.mulligan_advice_given = True

                if self.advice_thread and self.advice_thread.is_alive():
                    logging.info("Still processing previous advice request.")
                    return

                self.advice_thread = threading.Thread(target=self._generate_mulligan_advice, args=(board_state,))
                self.advice_thread.start()
            return
        else:
            # Reset mulligan flag when we're out of mulligan phase
            if hasattr(self, 'mulligan_advice_given') and self.mulligan_advice_given:
                logging.info("Resetting mulligan advice flag")
            self.mulligan_advice_given = False

        if board_state.current_turn is None:
            logging.debug("Current turn not yet determined.")
            return

        # On first turn detection, sync to current turn so we only advise FUTURE turns
        if not self.first_turn_detected:
            self.last_turn_advised = board_state.current_turn - 1
            self.first_turn_detected = True
            self.previous_board_state = board_state
            logging.info(f"First turn detected (Turn {board_state.current_turn}). Will advise starting from Turn {board_state.current_turn + 1}")
            return

        # Check for critical updates even during opponent's turn (continuous monitoring)
        if self.continuous_monitoring and self.previous_board_state:
            if self.advice_thread and self.advice_thread.is_alive():
                logging.debug("Advice thread still running, skipping continuous check")
            else:
                # Rate limit: Only check for critical updates every 10 seconds minimum
                current_time = time.time()
                if current_time - self.last_alert_time >= 10:
                    # Check if something critical happened
                    critical_advice = self.ai_advisor.check_important_updates(board_state, self.previous_board_state)
                    if critical_advice:
                        logging.info("Critical update detected - speaking immediately")
                        self._speak_advice(f"Alert: {critical_advice}")
                        self.last_alert_time = current_time
                else:
                    logging.debug(f"Rate limit active: {int(10 - (current_time - self.last_alert_time))}s until next alert check")

        # Update previous state
        self.previous_board_state = board_state

        # Standard priority-based advice for your turn
        is_new_turn_for_player = board_state.is_your_turn and board_state.current_turn > self.last_turn_advised

        # Check if opponent is attacking and you need blocking advice
        opponent_is_attacking = board_state.history and len(board_state.history.current_attackers) > 0
        is_blocking_step = opponent_is_attacking and board_state.has_priority and not board_state.is_your_turn

        # Track if we've given blocking advice for this combat
        if not hasattr(self, '_last_combat_advised'):
            self._last_combat_advised = None

        # Create a unique identifier for this combat (turn + attackers)
        if opponent_is_attacking:
            combat_id = (board_state.current_turn, tuple(sorted(board_state.history.current_attackers)))
        else:
            combat_id = None

        need_blocking_advice = is_blocking_step and combat_id and combat_id != self._last_combat_advised

        if board_state.has_priority and (is_new_turn_for_player or need_blocking_advice):
            if self.advice_thread and self.advice_thread.is_alive():
                logging.info("Still processing previous advice request.")
                return

            if is_new_turn_for_player:
                self.last_turn_advised = board_state.current_turn
                logging.info(f"Generating advice for new turn: {board_state.current_turn}")

            if need_blocking_advice:
                self._last_combat_advised = combat_id
                logging.info(f"⚔️ Opponent attacking with {len(board_state.history.current_attackers)} creatures - generating blocking advice")

            self.advice_thread = threading.Thread(target=self._generate_and_speak_advice, args=(board_state,))
            self.advice_thread.start()

    def _format_card_display(self, card: GameObject) -> str:
        """Format card display with name, type, P/T, and status indicators"""
        # Check if card is unknown
        if "Unknown" in card.name:
            return f"{card.name} ⚠️"

        # Build status indicators
        status_parts = []

        # Add tapped status
        if card.is_tapped:
            status_parts.append("🔄")

        # Add power/toughness for creatures
        if card.power is not None and card.toughness is not None:
            # Handle dict format {'value': int} just in case
            power_val = card.power.get("value") if isinstance(card.power, dict) else card.power
            tough_val = card.toughness.get("value") if isinstance(card.toughness, dict) else card.toughness
            status_parts.append(f"{power_val}/{tough_val}")

        # Add summoning sickness indicator
        if card.summoning_sick:
            status_parts.append("😴")

        # Add attacking status
        if card.is_attacking:
            status_parts.append("⚡")

        # Add any counters
        if card.counters:
            counter_str = ", ".join([f"{count}x {ctype}" for ctype, count in card.counters.items()])
            status_parts.append(f"[{counter_str}]")

        # Combine all parts
        if status_parts:
            return f"{card.name} ({', '.join(status_parts)})"
        else:
            return card.name

    def _display_board_state(self, board_state: BoardState):
        """Display a comprehensive visual representation of the current board state"""
        # Build board state lines
        lines = []
        lines.append("")
        lines.append("="*70)
        if board_state.in_mulligan_phase:
            lines.append("🎴 MULLIGAN PHASE - Opening Hand")
        else:
            lines.append(f"TURN {board_state.current_turn} - {board_state.current_phase}")
        lines.append("="*70)

        # Game History - what happened this turn
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn or history.died_this_turn or history.lands_played_this_turn:
                lines.append("")
                lines.append("📜 THIS TURN:")
                if history.cards_played_this_turn:
                    played_names = [c.name for c in history.cards_played_this_turn]
                    lines.append(f"   ⚡ Played: {', '.join(played_names)}")
                if history.lands_played_this_turn > 0:
                    lines.append(f"   🌍 Lands: {history.lands_played_this_turn}")
                if history.died_this_turn:
                    lines.append(f"   💀 Died: {', '.join(history.died_this_turn)}")

        # Opponent info
        lines.append("")
        lines.append("─"*70)
        opponent_lib = board_state.opponent_library_count if board_state.opponent_library_count > 0 else "?"
        lines.append(f"OPPONENT: ❤️  {board_state.opponent_life} life | 🃏 {board_state.opponent_hand_count} cards | 📖 {opponent_lib} library")

        lines.append("")
        lines.append(f"  ⚔️  Battlefield ({len(board_state.opponent_battlefield)}):")
        if board_state.opponent_battlefield:
            for card in board_state.opponent_battlefield:
                card_info = self._format_card_display(card)
                lines.append(f"      • {card_info}")
        else:
            lines.append("      (empty)")

        if board_state.opponent_graveyard:
            recent = board_state.opponent_graveyard[-5:]
            lines.append(f"  ⚰️ Graveyard ({len(board_state.opponent_graveyard)}): {', '.join([c.name for c in recent])}")

        if board_state.opponent_exile:
            lines.append(f"  🚫 Exile ({len(board_state.opponent_exile)}): {', '.join([c.name for c in board_state.opponent_exile])}")

        # Stack (shared)
        if board_state.stack:
            lines.append("")
            lines.append("─"*70)
            lines.append(f"📋 STACK ({len(board_state.stack)}):")
            for card in board_state.stack:
                lines.append(f"   ⚡ {card.name}")

        # Your info
        lines.append("")
        lines.append("─"*70)
        your_lib = board_state.your_library_count if board_state.your_library_count > 0 else "?"
        lines.append(f"YOU: ❤️  {board_state.your_life} life | 🃏 {board_state.your_hand_count} cards | 📖 {your_lib} library")

        lines.append("")
        lines.append(f"  🃏 Hand ({len(board_state.your_hand)}):")
        if board_state.your_hand:
            for card in board_state.your_hand:
                card_info = self._format_card_display(card)
                lines.append(f"      • {card_info}")
        else:
            lines.append("      (empty)")

        lines.append("")
        lines.append(f"  ⚔️  Battlefield ({len(board_state.your_battlefield)}):")
        if board_state.your_battlefield:
            for card in board_state.your_battlefield:
                card_info = self._format_card_display(card)
                lines.append(f"      • {card_info}")
        else:
            lines.append("      (empty)")

        if board_state.your_graveyard:
            recent = board_state.your_graveyard[-5:]
            lines.append(f"  ⚰️ Graveyard ({len(board_state.your_graveyard)}): {', '.join([c.name for c in recent])}")

        if board_state.your_exile:
            lines.append(f"  🚫 Exile ({len(board_state.your_exile)}): {', '.join([c.name for c in board_state.your_exile])}")

        lines.append("")
        lines.append("="*70)

        # Output board state
        if self.use_gui and self.gui:
            self.gui.set_board_state(lines)
        elif self.use_tui and self.tui:
            self.tui.set_board_state(lines)
        else:
            for line in lines:
                print(line)

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting for TTS (asterisks, hashtags, etc.)"""
        # Remove bold/italic asterisks
        text = re.sub(r'\*+', '', text)
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove bullet points
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
        return text

    def _parse_reasoning_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to separate thinking/reasoning from final answer.
        Returns (thinking, answer) tuple.
        """
        # DeepSeek-R1 and similar models use <think> tags
        if "<think>" in response and "</think>" in response:
            think_start = response.find("<think>")
            think_end = response.find("</think>") + len("</think>")
            thinking = response[think_start+7:think_end-8].strip()  # Extract content between tags
            answer = response[think_end:].strip()
            return (thinking, answer)

        # Some models use "Thought:" or "Reasoning:" prefixes
        if response.startswith("Thought:") or response.startswith("Reasoning:"):
            lines = response.split("\n", 1)
            if len(lines) > 1:
                return (lines[0], lines[1].strip())

        # No thinking detected, return empty thinking and full response as answer
        return ("", response)

    def _speak_advice(self, advice_text: str):
        """Speak advice text (helper for critical updates)"""
        if advice_text:
            # Parse thinking from advice for critical alerts too
            thinking, answer = self._parse_reasoning_response(advice_text)

            # Display thinking (not spoken)
            if thinking:
                self._output("🔴 Critical Alert - Thinking:", "red")
                for line in thinking.split("\n")[:3]:  # Limit to 3 lines for critical alerts
                    if line.strip():
                        self._output(f"   {line.strip()}", "blue")

            # Speak and display the answer
            final_text = answer if answer else advice_text
            clean_advice = self._strip_markdown(final_text)
            self.tts.speak(clean_advice)
            self._output(f"🔴 Alert: {final_text}", "red")

    def _generate_mulligan_advice(self, board_state: BoardState):
        """Generate and speak advice for mulligan decision, enhanced with 17lands data."""
        self._output("\n🎴 MULLIGAN DECISION", "cyan")
        self._output("Analyzing your opening hand with 17lands data...\n", "cyan")

        if not board_state.your_hand:
            self._output("⚠ No hand visible yet - waiting for cards...", "yellow")
            return

        hand_cards_with_stats = []
        total_oh_wr = 0
        cards_with_stats_count = 0

        for card in board_state.your_hand:
            stats = self.ai_advisor.rag_system.card_stats.get_card_stats(card.name) if self.ai_advisor.rag_system else None
            oh_wr = stats.get('opening_hand_win_rate', 0.0) if stats else 0.0
            if oh_wr > 0:
                total_oh_wr += oh_wr
                cards_with_stats_count += 1
            hand_cards_with_stats.append(f"{card.name} (OH WR: {oh_wr:.1%})")
        
        avg_oh_wr = total_oh_wr / cards_with_stats_count if cards_with_stats_count > 0 else 0.0

        deck_strategy = ""
        if board_state.your_decklist:
            total_lands = sum(count for name, count in board_state.your_decklist.items() if "Land" in name or any(lt in name for lt in ["Forest", "Plains", "Island", "Mountain", "Swamp"]))
            total_spells = sum(board_state.your_decklist.values()) - total_lands
            key_cards = [f"{name} (x{count})" for name, count in list(board_state.your_decklist.items())[:5] if "Land" not in name]
            deck_strategy = f"Your deck has {total_lands} lands and {total_spells} spells. Key cards: {', '.join(key_cards)}."

        mulligan_prompt = f"""You are analyzing an opening hand in Magic: The Gathering.

OPENING HAND ({len(board_state.your_hand)} cards):
{', '.join(hand_cards_with_stats)}

HAND ANALYSIS:
- Average Opening Hand Win Rate (OH WR): {avg_oh_wr:.1%} (based on 17lands data)
- A good hand is typically >55% OH WR. A hand below 50% is often a mulligan.

DECK INFORMATION:
{deck_strategy or "Deck information not available."}

MULLIGAN DECISION CRITERIA:
1. Land count: Is it between 2-4 for a 7-card hand?
2. Data-driven advice: Is the average OH WR acceptable?
3. Curve & Synergy: Can you make plays in the early turns?

Based on all this information, should the player mulligan? Respond with:
- KEEP: [Explain why in 1-2 sentences]
- MULLIGAN: [Explain why in 1-2 sentences]"""

        try:
            advice = self.ai_advisor.client.generate(mulligan_prompt)

            if advice:
                # Parse thinking vs answer
                thinking, answer = self._parse_reasoning_response(advice)

                # Display thinking in dim color (not spoken)
                if thinking:
                    self._output("💭 Thinking:", "blue")
                    self._output(thinking, "blue")
                    self._output("", "reset")

                # Speak and display the final answer
                final_text = answer if answer else advice
                self._output(f"Advisor: {final_text}", "green")
                clean_advice = self._strip_markdown(final_text)
                self.tts.speak(clean_advice)
            else:
                self._output("Failed to get mulligan advice.", "red")

        except Exception as e:
            logging.error(f"Error generating mulligan advice: {e}")
            self._output(f"Error getting mulligan advice: {e}", "red")

    def _generate_and_speak_advice(self, board_state: BoardState):
        """Generate and speak advice for the current turn"""
        # Validate before sending to LLM
        if not self.game_state_mgr.validate_board_state(board_state):
            logging.warning("Skipping advice generation due to invalid board state")
            self._output(f"\n>>> Turn {board_state.current_turn}: Waiting for complete board state...", "yellow")
            return

        # Display board state
        self._display_board_state(board_state)

        # Update status bar
        self._update_status(board_state)

        self._output(f"\n>>> Turn {board_state.current_turn}: Your move!", "cyan")
        self._output("Getting advice from the master...\n", "cyan")

        advice = self.ai_advisor.get_tactical_advice(board_state)

        if advice:
            # Parse thinking vs answer
            thinking, answer = self._parse_reasoning_response(advice)

            # Display thinking in dim color (not spoken)
            if thinking:
                self._output("💭 Thinking:", "blue")
                # Split thinking into lines for readability
                for line in thinking.split("\n"):
                    if line.strip():
                        self._output(f"   {line.strip()}", "blue")
                self._output("", "white")  # Blank line

            # Display and speak the answer
            if answer:
                self._output(f"Advisor: {answer}\n", "green")
                logging.info(f"ADVICE:\n{answer}")
                # Strip markdown before speaking
                clean_advice = self._strip_markdown(answer)
                self.tts.speak(clean_advice)
            else:
                # If no clear answer, speak the full advice
                self._output(f"Advisor: {advice}\n", "green")
                logging.info(f"ADVICE:\n{advice}")
                clean_advice = self._strip_markdown(advice)
                self.tts.speak(clean_advice)
        else:
            logging.warning("No advice was generated.")

if __name__ == "__main__":
    import argparse
    import dataclasses

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MTGA Voice Advisor - Real-time tactical advice for Magic: The Gathering Arena")
    parser.add_argument("--tui", action="store_true", help="Use Text User Interface (TUI) mode with curses")
    parser.add_argument("--cli", action="store_true", help="Use basic Command Line Interface (CLI) mode")
    args = parser.parse_args()

    # Check and update card database before starting
    if not check_and_update_card_database():
        print("Cannot start without card database.")
        sys.exit(1)

    # Default to GUI unless TUI or CLI specified
    use_gui = not args.tui and not args.cli and TKINTER_AVAILABLE
    use_tui = args.tui

    # Create and run advisor
    advisor = CLIVoiceAdvisor(use_tui=use_tui, use_gui=use_gui)
    advisor.run()
