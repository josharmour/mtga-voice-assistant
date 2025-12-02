
import logging
import os
import time
from typing import Callable, Any
import dataclasses
import json
import re
from typing import Dict, List, Optional, Callable
from enum import Enum, auto

# Event system for decoupled communication (future integration)
# from .events import get_event_bus, EventType

# Performance monitoring
from .monitoring import get_monitor

# Zone Type Enum for fast comparisons
class ZoneType(Enum):
    """Enum representing different zone types in MTG Arena."""
    UNKNOWN = auto()
    HAND = auto()
    BATTLEFIELD = auto()
    GRAVEYARD = auto()
    EXILE = auto()
    LIBRARY = auto()
    STACK = auto()
    COMMAND = auto()
    LIMBO = auto()  # For cards in transition
    REVEALED = auto()  # For revealed cards

# Map Arena zone type strings to our enum
ZONE_TYPE_MAP = {
    "ZoneType_Hand": ZoneType.HAND,
    "ZoneType_Battlefield": ZoneType.BATTLEFIELD,
    "ZoneType_Graveyard": ZoneType.GRAVEYARD,
    "ZoneType_Exile": ZoneType.EXILE,
    "ZoneType_Library": ZoneType.LIBRARY,
    "ZoneType_Stack": ZoneType.STACK,
    "ZoneType_Command": ZoneType.COMMAND,
    "ZoneType_Limbo": ZoneType.LIMBO,
    "ZoneType_Revealed": ZoneType.REVEALED,
}

def get_zone_type(zone_type_str: str) -> ZoneType:
    """Convert Arena zone type string to ZoneType enum."""
    return ZONE_TYPE_MAP.get(zone_type_str, ZoneType.UNKNOWN)

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

    def _find_current_session_start(self) -> int:
        """
        Find the byte offset where the current/most recent MTGA session begins.

        This prevents processing stale data from old game sessions that are
        still in the log file. We look for recent timestamps and session markers.

        Returns the byte offset to start reading from.
        """
        import datetime
        import re

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                # Get file size
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                if file_size == 0:
                    return 0

                # Strategy: Read backwards in chunks to find session markers
                # Look for patterns that indicate a new MTGA session:
                # - "Client.Connected" or "Connecting to"
                # - Recent timestamps (within last few hours)
                # - "Auth: " login messages

                chunk_size = 100000  # 100KB chunks
                current_time = datetime.datetime.now()
                max_age_hours = 6  # Only process logs from last 6 hours

                best_offset = 0
                last_good_match_offset = 0

                # Timestamp pattern: [UnityCrossThreadLogger]MM/DD/YYYY HH:MM:SS AM/PM
                # or just look for date patterns
                timestamp_pattern = re.compile(r'\[UnityCrossThreadLogger\](\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M)')
                session_start_pattern = re.compile(r'Client\.Connected|Connecting to matchmaker|"authenticateResponse"')
                # Match start includes both gameplay matches AND draft events
                match_start_pattern = re.compile(r'MatchGameRoomStateChangedEvent|GREMessageType_ConnectResp|Draft\.Notify|BotDraft|DraftStatus|"draftId"')

                # Read from end backwards in chunks
                pos = max(0, file_size - chunk_size)

                while pos >= 0:
                    f.seek(pos)
                    chunk = f.read(min(chunk_size, file_size - pos))
                    lines = chunk.split('\n')

                    for i, line in enumerate(lines):
                        # Check for session start markers
                        if session_start_pattern.search(line):
                            # Calculate approximate offset
                            line_offset = pos + sum(len(l) + 1 for l in lines[:i])
                            best_offset = line_offset

                        # Check for match start (more granular)
                        if match_start_pattern.search(line):
                            line_offset = pos + sum(len(l) + 1 for l in lines[:i])
                            last_good_match_offset = line_offset

                        # Check timestamp freshness
                        ts_match = timestamp_pattern.search(line)
                        if ts_match:
                            try:
                                ts_str = ts_match.group(1)
                                log_time = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
                                age_hours = (current_time - log_time).total_seconds() / 3600

                                if age_hours <= max_age_hours:
                                    # This is recent enough, use the earlier of session or match start
                                    if best_offset > 0:
                                        logging.info(f"Found recent session start at offset {best_offset} (age: {age_hours:.1f}h)")
                                        return best_offset
                                    elif last_good_match_offset > 0:
                                        logging.info(f"Found recent match start at offset {last_good_match_offset} (age: {age_hours:.1f}h)")
                                        return last_good_match_offset
                            except ValueError:
                                pass

                    if pos == 0:
                        break
                    pos = max(0, pos - chunk_size + 1000)  # Overlap to avoid splitting lines

                # Fallback: if file is small or we found nothing, check timestamps from start
                # If the log is very old, just start from end (live data only)
                f.seek(0)
                first_chunk = f.read(10000)
                ts_match = timestamp_pattern.search(first_chunk)
                if ts_match:
                    try:
                        ts_str = ts_match.group(1)
                        log_time = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
                        age_hours = (current_time - log_time).total_seconds() / 3600

                        if age_hours > max_age_hours:
                            # Log is old, skip to near end to only get live data
                            skip_to = max(0, file_size - 50000)  # Last 50KB
                            logging.info(f"Log file is {age_hours:.1f}h old, skipping to offset {skip_to}")
                            return skip_to
                    except ValueError:
                        pass

                logging.info("No session markers found, starting from beginning")
                return 0

        except Exception as e:
            logging.error(f"Error finding session start: {e}")
            return 0

    def _log_draft_context_near_offset(self, offset: int):
        """Log any Draft.Notify events near the start offset for debugging.

        NOTE: Disabled for performance - this adds ~100ms+ of file I/O on startup.
        Enable only for debugging draft detection issues.
        """
        # Disabled for performance - uncomment to debug draft detection
        # try:
        #     with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
        #         search_start = max(0, offset - 50000)
        #         f.seek(search_start)
        #         chunk = f.read(100000)
        #         import re
        #         draft_matches = re.findall(r'Draft\.Notify.*?PackCards[^}]+', chunk)
        #         if draft_matches:
        #             logging.info(f"Found {len(draft_matches)} Draft.Notify event(s) near offset {offset}")
        # except Exception as e:
        #     logging.debug(f"Error checking draft context: {e}")
        pass

    def find_last_draft_notify(self) -> Optional[str]:
        """
        Scan the log file backwards to find the most recent Draft.Notify event.
        Returns the full line if found, None otherwise.
        Used to recover draft state when app is started mid-draft.

        Performance: Only scans last 50KB to minimize I/O impact on startup.
        """
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                # Only scan last 50KB - draft events are recent if user is mid-draft
                # This is a significant performance optimization vs scanning entire log
                chunk_size = 50000
                pos = max(0, file_size - chunk_size)

                f.seek(pos)
                chunk = f.read(file_size - pos)

                # Look for Draft.Notify (search from end of chunk)
                lines = chunk.split('\n')
                for line in reversed(lines):
                    if 'Draft.Notify' in line and 'PackCards' in line:
                        logging.debug(f"Found last Draft.Notify in last {chunk_size // 1000}KB")
                        return line

                return None
        except Exception as e:
            logging.error(f"Error finding last Draft.Notify: {e}")
            return None

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
                        # BUG FIX: Find the current session start to avoid processing stale data
                        # from old games that are still in the log file
                        start_offset = self._find_current_session_start()

                        # CRITICAL: Check if offset is valid (file may have been truncated/rotated)
                        self.file.seek(0, 2)  # Seek to end
                        file_size = self.file.tell()
                        if start_offset >= file_size:
                            logging.warning(f"Calculated offset {start_offset} exceeds file size {file_size}, starting from beginning")
                            start_offset = 0

                        self.file.seek(start_offset)
                        self.offset = start_offset
                        self.first_open = False
                        logging.info(f"Log file opened - starting from offset {start_offset} (file size: {file_size}) to process current session only.")

                        # Scan backwards from start_offset to find any Draft.Notify we might process
                        self._log_draft_context_near_offset(start_offset)
                    else:
                        # Log rotation - start from beginning of new file
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning of new file.")

                # BUG FIX #26: Detect file truncation (e.g., MTGA restart)
                # On Windows, inode doesn't change when file is truncated, so we must
                # check if our stored offset exceeds the current file size
                self.file.seek(0, 2)  # Seek to end
                current_file_size = self.file.tell()
                if self.offset > current_file_size:
                    logging.warning(f"🔄 File truncated detected: offset {self.offset} > file size {current_file_size}. Resetting to beginning.")
                    self.offset = 0
                    self.is_caught_up = False  # Re-process from start

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

@dataclasses.dataclass
class DraftEvent:
    """Represents a parsed draft event."""
    event_type: str  # Event type name (e.g., 'Draft.Notify', 'LogBusinessEvents')
    data: Dict[str, Any]  # Parsed event data
    raw_line: str  # Original log line

class DraftEventParser:
    """
    Consolidated parser for all draft-related events in MTGA logs.

    This class handles all draft event detection and parsing, consolidating
    the scattered regex patterns and conditional logic from parse_log_line.

    Supported event types:
    - Draft.Notify: Premier draft pack notifications
    - LogBusinessEvents: Premier draft picks
    - BotDraftDraftStatus: Quick draft status updates
    - BotDraftDraftPick: Quick draft picks
    - EventGetCoursesV2: Draft pool updates
    - Generic <== events: Various draft-related events
    """

    # Compiled regex patterns for efficiency
    PATTERNS = {
        'draft_notify': re.compile(r'\[UnityCrossThreadLogger\]Draft\.Notify (.+)'),
        'start_event': re.compile(r'\[UnityCrossThreadLogger\]==> (\w+) (.*)'),
        'end_event': re.compile(r'<== (\w+)\(([a-f0-9-]+)\)'),
    }

    # Quick string checks before expensive regex (performance optimization)
    DRAFT_KEYWORDS = ('Draft', 'Pick', 'Pack', 'EventGetCourses', 'Event_GetCourses', 'BotDraft', 'toSceneName', 'DeckBuilder')

    def __init__(self, callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize with optional callbacks for each event type.

        Args:
            callbacks: Dict mapping event type to callback function.
                      Event types: 'Draft.Notify', 'LogBusinessEvents', 'BotDraftDraftStatus',
                                  'BotDraftDraftPick', 'EventGetCoursesV2', or any <== event name
        """
        self.callbacks = callbacks or {}
        self._next_line_event: Optional[str] = None  # For <== events with JSON on next line

    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for a specific draft event type."""
        self.callbacks[event_type] = callback
        logging.info(f"Registered draft callback for event type: {event_type}")

    def parse(self, line: str) -> Optional[DraftEvent]:
        """
        Parse a log line for draft events.

        Returns DraftEvent if line contains a draft event, None otherwise.
        Also triggers registered callbacks.

        Args:
            line: Log line to parse

        Returns:
            DraftEvent if a draft event was detected and parsed, None otherwise
        """
        # Quick pre-check to avoid regex on non-draft lines
        if not self._quick_draft_check(line):
            return None

        event = None

        # PATTERN 1: JSON line after <== event marker
        if self._next_line_event:
            event = self._parse_deferred_event(line)
            return event

        # PATTERN 2: Draft.Notify messages (Premier Draft)
        # Format: [UnityCrossThreadLogger]Draft.Notify {"draftId":"...","SelfPick":5,"SelfPack":1,"PackCards":"..."}
        if "[UnityCrossThreadLogger]Draft.Notify" in line:
            event = self._parse_draft_notify(line)
            if event:
                self._trigger_callback(event)
            return event

        # PATTERN 3: Start events with JSON in same line
        # Format: [UnityCrossThreadLogger]==> EventName {...}
        if line.startswith("[UnityCrossThreadLogger]==>"):
            event = self._parse_start_event(line)
            if event:
                self._trigger_callback(event)
            return event

        # PATTERN 4: End event markers (JSON on next line)
        # Format: <== EventName(uuid)
        if line.startswith("<=="):
            self._parse_end_event_marker(line)
            return None  # Event will be parsed when we see the next line

        # PATTERN 5: Scene change (Draft -> DeckBuilder triggers deck building)
        # Format: {"fromSceneName":"Draft","toSceneName":"DeckBuilder",...}
        if '"toSceneName"' in line and '"DeckBuilder"' in line:
            event = self._parse_scene_change(line)
            if event:
                self._trigger_callback(event)
            return event

        return None

    def _quick_draft_check(self, line: str) -> bool:
        """Quick string check before expensive regex."""
        return any(kw in line for kw in self.DRAFT_KEYWORDS)

    def _parse_draft_notify(self, line: str) -> Optional[DraftEvent]:
        """
        Parse Draft.Notify event (Premier Draft).

        Format: [UnityCrossThreadLogger]Draft.Notify {"draftId":"...","SelfPick":5,"SelfPack":1,"PackCards":"..."}
        """
        match = self.PATTERNS['draft_notify'].search(line)
        if not match:
            return None

        json_str = match.group(1)
        try:
            draft_data = json.loads(json_str)

            # Extract pack and pick numbers (1-indexed)
            pack_num = draft_data.get("SelfPack", 1)
            pick_num = draft_data.get("SelfPick", 1)

            # Parse PackCards string (comma-separated card IDs)
            pack_cards_str = draft_data.get("PackCards", "")
            pack_arena_ids = []
            if pack_cards_str:
                pack_arena_ids = [int(card_id) for card_id in pack_cards_str.split(",")]

            logging.info(f"Draft.Notify: Pack {pack_num}, Pick {pick_num}, {len(pack_arena_ids)} cards")

            # Normalize data format
            data = {
                "PackNumber": pack_num,
                "PickNumber": pick_num,
                "PackCards": pack_arena_ids,
                "DraftId": draft_data.get("draftId", "")
            }

            return DraftEvent(
                event_type="Draft.Notify",
                data=data,
                raw_line=line
            )

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse Draft.Notify JSON: {e}")
            return None

    def _parse_start_event(self, line: str) -> Optional[DraftEvent]:
        """
        Parse start event with JSON in same line.

        Format: [UnityCrossThreadLogger]==> EventName {"request": "{...}"}
        """
        match = self.PATTERNS['start_event'].search(line)
        if not match:
            return None

        event_type = match.group(1)
        outer_json_str = match.group(2)

        try:
            outer_json = json.loads(outer_json_str)
            if "request" in outer_json:
                inner_json = json.loads(outer_json["request"])

                logging.info(f"Detected start event: {event_type}")

                # Handle LogBusinessEvents (Premier Draft picks)
                if event_type == "LogBusinessEvents" and "DraftId" in inner_json:
                    logging.info("Premier Draft pick detected")
                    return DraftEvent(
                        event_type=event_type,
                        data=inner_json,
                        raw_line=line
                    )

                # Handle EventPlayerDraftMakePick (user made a pick - contains picked card IDs)
                if event_type == "EventPlayerDraftMakePick" and "GrpIds" in inner_json:
                    logging.info(f"Draft pick made: Pack {inner_json.get('Pack')}, Pick {inner_json.get('Pick')}, Cards: {inner_json.get('GrpIds')}")
                    return DraftEvent(
                        event_type=event_type,
                        data=inner_json,
                        raw_line=line
                    )

                # Generic start event
                return DraftEvent(
                    event_type=event_type,
                    data=inner_json,
                    raw_line=line
                )

        except (json.JSONDecodeError, ValueError) as e:
            logging.debug(f"Failed to parse start event JSON: {e}")

        return None

    def _parse_scene_change(self, line: str) -> Optional[DraftEvent]:
        """
        Parse scene change event, particularly Draft -> DeckBuilder transition.

        Format: {"fromSceneName":"Draft","toSceneName":"DeckBuilder","initiator":"System","context":"deck builder"}
        """
        try:
            json_start = line.find("{")
            if json_start == -1:
                return None

            parsed_data = json.loads(line[json_start:])

            from_scene = parsed_data.get("fromSceneName", "")
            to_scene = parsed_data.get("toSceneName", "")

            if from_scene == "Draft" and to_scene == "DeckBuilder":
                logging.info(f"Scene change detected: {from_scene} -> {to_scene} (deck building phase)")
                return DraftEvent(
                    event_type="SceneChange_DraftToDeckBuilder",
                    data=parsed_data,
                    raw_line=line
                )

        except (json.JSONDecodeError, ValueError) as e:
            logging.debug(f"Failed to parse scene change JSON: {e}")

        return None

    def _parse_end_event_marker(self, line: str):
        """
        Parse end event marker and flag that next line contains JSON.

        Format: <== EventName(uuid)
        """
        match = self.PATTERNS['end_event'].search(line)
        if match:
            event_type = match.group(1)
            # Mark that the next line contains the JSON for this event
            self._next_line_event = event_type
            logging.debug(f"Detected end event marker: {event_type}, expecting JSON on next line")

    def _parse_deferred_event(self, line: str) -> Optional[DraftEvent]:
        """
        Parse the JSON line that follows a <== event marker.
        """
        event_type = self._next_line_event
        self._next_line_event = None  # Clear the flag

        # Try to parse the JSON on this line
        try:
            json_start = line.find("{")
            if json_start == -1:
                # Some events like LogBusinessEvents return a status string, not JSON
                logging.debug(f"No JSON found in line after {event_type}, might be status string: {line[:100]}")
                return None

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

            event = DraftEvent(
                event_type=event_type,
                data=parsed_data,
                raw_line=line
            )

            # Trigger callback
            self._trigger_callback(event)

            return event

        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON for draft event {event_type}: {e}")
            return None

    def _trigger_callback(self, event: DraftEvent):
        """Trigger registered callback for an event."""
        if event.event_type in self.callbacks:
            try:
                self.callbacks[event.event_type](event.data)
            except Exception as e:
                logging.error(f"Error in draft callback for {event.event_type}: {e}")
        else:
            logging.debug(f"No callback registered for draft event: {event.event_type}")

    def reset(self):
        """Reset parser state."""
        self._next_line_event = None

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
        self.zone_id_to_enum: Dict[int, ZoneType] = {}  # P1: Enum-based zone mapping
        self.game_history: GameHistory = GameHistory()
        self.last_event_timestamp: Optional[datetime.datetime] = None # Track when the last event happened
        self._last_timestamp_str: str = "" # P1 Performance: Cache for timestamp string to avoid redundant parsing

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

        # Game event callbacks (for match start, etc.)
        self._game_callbacks: Dict[str, Callable] = {}

    def register_game_callback(self, event_type: str, callback: Callable):
        """Register a callback for a game event (e.g., 'match_started')."""
        self._game_callbacks[event_type] = callback
        logging.info(f"Registered game callback for event type: {event_type}")

    def _trigger_game_callback(self, event_type: str, data: dict = None):
        """Trigger a registered game callback."""
        if event_type in self._game_callbacks:
            try:
                self._game_callbacks[event_type](data or {})
            except Exception as e:
                logging.error(f"Error in game callback for {event_type}: {e}")

    def parse_timestamp(self, line: str):
        """
        Extract timestamp from log line if present.

        P1 Performance Optimization:
        - Caches timestamp string to avoid redundant strptime() calls
        - Only parses when timestamp string actually changes
        - Reduces CPU usage on every log line
        """
        # Format: [UnityCrossThreadLogger]11/22/2025 10:20:17 PM: ...
        if "[UnityCrossThreadLogger]" not in line:
            return

        try:
            # Find the timestamp part (fast string operations)
            start = line.find("]") + 1
            end = line.find(": ", start)  # Simplified: just find next ": "

            if start <= 0 or end <= start:
                return

            ts_str = line[start:end].strip()

            # P1 Performance: Only parse if timestamp string changed
            # This avoids expensive strptime() calls for duplicate timestamps
            if ts_str != self._last_timestamp_str:
                self._last_timestamp_str = ts_str
                try:
                    # Parse "11/22/2025 10:20:17 PM"
                    self.last_event_timestamp = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
                except ValueError:
                    # Invalid timestamp format, ignore
                    pass
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
        # Trigger match_started callback before clearing state
        self._trigger_game_callback("match_started")
        self.game_objects.clear()
        self.players.clear()
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat = None
        self.priority_player_seat = None
        # BUG FIX: Reset local_player_seat_id on new match
        # Seat IDs can change between matches (sometimes you're seat 1, sometimes seat 2)
        # It will be re-set when we receive the systemSeatIds message for the new match
        self.local_player_seat_id = None
        self.zone_type_to_ids.clear()
        self.observed_zone_ids.clear()
        self.zone_id_to_type.clear()
        self.zone_id_to_enum.clear()  # P1: Clear enum mapping
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
        monitor = get_monitor()
        with monitor.measure("mtga.parse_game_state_message"):
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
        monitor = get_monitor()
        with monitor.measure("mtga.parse_game_objects"):
            state_changed = False
            logging.info(f"Parsing {len(game_objects)} game objects")
            return self._parse_game_objects_impl(game_objects)

    def _parse_game_objects_impl(self, game_objects: list) -> bool:
        state_changed = False
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

            # BUG FIX #11: Infer local_player_seat_id from visible hand cards
            # If we can see a card with a real grpId in a hand zone, it must be our hand
            # (opponent's hand cards are hidden/face-down with grpId=0)
            if not self.local_player_seat_id and owner_seat_id and grp_id and grp_id != 0:
                # Check if this card is in a hand zone
                zone_type = self.zone_id_to_type.get(zone_id, "")
                if "Hand" in zone_type:
                    self.local_player_seat_id = owner_seat_id
                    logging.info(f"BUG FIX #11: Inferred local_player_seat_id={owner_seat_id} from visible hand card (grpId={grp_id})")

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
                self.zone_id_to_enum[zone_id] = get_zone_type(zone_type_str)  # P1: Populate enum mapping

                # P1: Use enum for visibility logic (faster than string comparisons)
                zone_enum = get_zone_type(zone_type_str)
                is_private_zone = zone_enum in (ZoneType.HAND, ZoneType.LIBRARY)

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
                            visibility="private" if is_private_zone else "public"
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
            if "handCardCount" in player_data:
                new_count = player_data["handCardCount"]
                if player.hand_count != new_count:
                    logging.info(f"Player {seat_id} hand count changed: {player.hand_count} -> {new_count}")
                    player.hand_count = new_count
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

                            # P1: Get zone enums for fast comparisons
                            zone_src_enum = self.zone_id_to_enum.get(zone_src, ZoneType.UNKNOWN)
                            zone_dest_enum = self.zone_id_to_enum.get(zone_dest, ZoneType.UNKNOWN)

                            # Get zone names for logging (still use strings for readability)
                            zone_src_name = self.zone_id_to_type.get(zone_src, f"Zone{zone_src}")
                            zone_dest_name = self.zone_id_to_type.get(zone_dest, f"Zone{zone_dest}")

                            # P1: Track game history events using enum comparisons (faster)
                            if zone_dest_enum == ZoneType.BATTLEFIELD:
                                # Card entered battlefield (played/put into play)
                                self.game_history.cards_played_this_turn.append(obj)
                                if hasattr(obj, 'card_types') and "CardType_Land" in str(obj.card_types):
                                    self.game_history.lands_played_this_turn += 1

                            if zone_dest_enum == ZoneType.GRAVEYARD and zone_src_enum == ZoneType.BATTLEFIELD:
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

class JsonStreamParser:
    """
    Robust JSON stream parser that correctly handles braces within strings.

    Uses a simple state machine to track:
    - Whether we're inside a string
    - Escape sequences within strings
    - Actual JSON depth (ignoring braces in strings)
    """

    def __init__(self):
        self.buffer: str = ""
        self.depth: int = 0

    def feed(self, line: str) -> Optional[str]:
        """
        Feed a line of text to the parser.

        Returns:
            Complete JSON string if a full object was detected, None otherwise.
        """
        # If we're not in a JSON object, look for the start
        if self.depth == 0:
            json_start = line.find('{')
            if json_start == -1:
                return None
            line = line[json_start:]
            self.buffer = ""

        # Add line to buffer
        self.buffer += line

        # Update depth by parsing character by character
        self.depth = self._calculate_depth(self.buffer)

        # Detect corruption (more closing than opening)
        if self.depth < 0:
            logging.warning(f"JSON depth corruption detected (depth={self.depth}). Resetting parser.")
            self.reset()
            return None

        # If we have a complete object, return it
        if self.depth == 0 and self.buffer:
            result = self.buffer
            self.reset()
            return result

        return None

    def _calculate_depth(self, text: str) -> int:
        """
        Calculate JSON depth while properly handling strings.

        This state machine tracks:
        - in_string: Whether we're inside a quoted string
        - escaped: Whether the previous character was a backslash
        - depth: Current brace nesting level

        Args:
            text: The text to analyze

        Returns:
            Current JSON depth (0 = complete object)
        """
        depth = 0
        in_string = False
        escaped = False

        for char in text:
            if escaped:
                # Previous char was backslash, this char is escaped
                escaped = False
                continue

            if char == '\\' and in_string:
                # Start escape sequence
                escaped = True
                continue

            if char == '"':
                # Toggle string state
                in_string = not in_string
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1

        return depth

    def reset(self):
        """Reset parser state."""
        self.buffer = ""
        self.depth = 0

    def is_valid_json(self, text: str) -> bool:
        """
        Quick validation to check if text looks like valid JSON.

        This provides early validation before attempting to parse.

        Args:
            text: Text to validate

        Returns:
            True if text appears to be valid JSON structure
        """
        if not text or not text.strip():
            return False

        trimmed = text.strip()

        # Must start with { and end with }
        if not (trimmed.startswith('{') and trimmed.endswith('}')):
            return False

        # Depth should be exactly 0 for complete object
        return self._calculate_depth(trimmed) == 0


class GameStateManager:
    def __init__(self, card_lookup: "ArenaCardDatabase"):
        self.scanner = MatchScanner(card_lookup=card_lookup)
        self.card_lookup = card_lookup

        # P3: Robust JSON stream parser (replaces simple brace counting)
        self._json_parser = JsonStreamParser()

        # P2: Consolidated draft event parser
        self.draft_parser = DraftEventParser()

        # P2 Performance: BoardState caching
        self._cached_board_state: Optional[BoardState] = None
        self._board_state_dirty: bool = True
        self._last_board_state_hash: int = 0

    def register_draft_callback(self, event_type: str, callback: Callable):
        """Register a callback for a specific draft event type."""
        self.draft_parser.register_callback(event_type, callback)

    def register_game_callback(self, event_type: str, callback: Callable):
        """Register a callback for a game event (e.g., 'match_started')."""
        self.scanner.register_game_callback(event_type, callback)

    def recover_draft_state(self, log_follower: "LogFollower"):
        """
        Attempt to recover draft state by finding the last Draft.Notify in the log.
        Call this after registering draft callbacks to catch mid-draft restarts.
        """
        last_notify = log_follower.find_last_draft_notify()
        if last_notify:
            logging.info("Attempting to recover draft state from last Draft.Notify...")
            # Process the line through the draft parser
            event = self.draft_parser.parse(last_notify)
            if event:
                logging.info(f"Successfully recovered draft state: {event.event_type}")
            else:
                logging.warning("Found Draft.Notify but failed to parse it")

    def _mark_board_state_dirty(self):
        """
        Mark board state as needing recalculation.

        P2 Performance: This flag prevents redundant board state construction
        when no game state changes have occurred.
        """
        self._board_state_dirty = True

    def _compute_state_hash(self, board_state: BoardState) -> int:
        """
        Quick hash to detect meaningful changes in board state.

        P2 Performance: This hash allows us to skip updates when the state
        hasn't actually changed, even if marked dirty.

        Args:
            board_state: The board state to hash

        Returns:
            Integer hash representing the current state
        """
        return hash((
            board_state.your_life,
            board_state.opponent_life,
            board_state.current_turn,
            board_state.current_phase,
            len(board_state.your_battlefield),
            len(board_state.opponent_battlefield),
            board_state.your_hand_count,
            board_state.has_priority,
        ))

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

                # P2 Performance: Mark board state dirty on match reset
                self._mark_board_state_dirty()

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

        # P1 Performance: Only parse timestamp for lines that trigger game state changes
        # This reduces unnecessary datetime.strptime() calls from ~1000s/game to ~100s/game
        # Timestamp is only used for freshness checks in app.py (30-second stale event detection)
        if "greToClientEvent" in line or "GREMessageType" in line or "gameStateMessage" in line:
            self.scanner.parse_timestamp(line)

        # P2: Consolidated draft event detection
        # Use DraftEventParser to handle all draft-related log lines
        draft_event = self.draft_parser.parse(line)
        if draft_event is not None:
            # Draft event was detected and callbacks were triggered
            return False  # Draft events don't change game state

        # P3: Use robust JSON stream parser
        json_to_parse = self._json_parser.feed(line)

        # If no complete JSON object yet, return
        if json_to_parse is None:
            return False

        # P3: Early validation before attempting to parse
        if not self._json_parser.is_valid_json(json_to_parse):
            logging.warning(f"Invalid JSON structure detected. Skipping malformed content: {json_to_parse[:200]}...")
            return False

        # Try to parse the complete JSON object
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
                state_changed = self.scanner.parse_gre_to_client_event(gre_event_data)
                # P2 Performance: Mark board state dirty when GRE event changes state
                if state_changed:
                    self._mark_board_state_dirty()
                return state_changed
            else:
                logging.debug("Parsed JSON but 'greToClientEvent' not found within the object.")
                return False
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed despite validation. Error: {e}. Content: {json_to_parse[:200]}...")
            return False

        return False

    def get_current_board_state(self) -> Optional[BoardState]:
        """
        Get current board state with caching.

        P2 Performance: Uses dirty flag to avoid rebuilding board state when nothing changed.
        If the state hasn't changed since last call, returns cached copy.
        """
        monitor = get_monitor()
        with monitor.measure("mtga.get_current_board_state"):
            # If not dirty and we have a cached state, return it
            if not self._board_state_dirty and self._cached_board_state is not None:
                logging.debug("PERFORMANCE: Returning cached board state (no changes detected)")
                return self._cached_board_state

            # Build new board state
            board_state = self._build_board_state()

            if board_state is None:
                # Can't build state yet (missing player data, etc.)
                return None

            # Compute hash to detect if state actually changed
            new_hash = self._compute_state_hash(board_state)

            if new_hash == self._last_board_state_hash and self._cached_board_state is not None:
                # State hash hasn't changed - return cached version
                logging.debug("PERFORMANCE: State hash unchanged, returning cached board state")
                self._board_state_dirty = False
                return self._cached_board_state

            # State actually changed - cache it
            self._cached_board_state = board_state
            self._board_state_dirty = False
            self._last_board_state_hash = new_hash

            return board_state

    def _build_board_state(self) -> Optional[BoardState]:
        """
        Actually build the board state (existing logic).

        P2 Performance: This method contains the heavy lifting that was previously
        in get_current_board_state(). It's now only called when state is dirty.
        """
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

        # P1: Helper function to process objects in a zone using enum comparisons (faster)
        def process_zone(zone_type_enum: ZoneType, owner_filter: Optional[int] = None):
            """Get all objects in a specific zone type, optionally filtered by owner."""
            # P1: Find all zone IDs that match this zone type using enum equality (faster than string substring)
            zone_ids = [zid for zid, zt in self.scanner.zone_id_to_enum.items() if zt == zone_type_enum]

            objects = []
            for zone_id in zone_ids:
                if zone_id in self.scanner._zone_objects:
                    for instance_id in self.scanner._zone_objects[zone_id]:
                        obj = self.scanner.game_objects.get(instance_id)
                        if obj:
                            # Filter by owner if specified
                            if owner_filter is None or obj.owner_seat_id == owner_filter:
                                objects.append(obj)
                                zone_name = self.scanner.zone_id_to_type.get(zone_id, f"Zone{zone_id}")
                                logging.debug(f"Card {obj.grp_id} ({obj.name}), color={obj.color_identity}, zone={zone_id} ({zone_name}), owner={obj.owner_seat_id}")
            return objects

        # P1: Process each zone type efficiently using enum comparisons
        board_state.your_hand = process_zone(ZoneType.HAND, your_seat_id)
        board_state.your_battlefield = process_zone(ZoneType.BATTLEFIELD, your_seat_id)
        board_state.your_graveyard = process_zone(ZoneType.GRAVEYARD, your_seat_id)
        board_state.your_exile = process_zone(ZoneType.EXILE, your_seat_id)
        board_state.opponent_battlefield = process_zone(ZoneType.BATTLEFIELD, opponent_seat_id)
        board_state.opponent_graveyard = process_zone(ZoneType.GRAVEYARD, opponent_seat_id)
        board_state.opponent_exile = process_zone(ZoneType.EXILE, opponent_seat_id)
        board_state.stack = process_zone(ZoneType.STACK)  # Stack is shared, no owner filter

        # Count library cards (we don't need the full list, just the count)
        your_library_objects = process_zone(ZoneType.LIBRARY, your_seat_id)
        board_state.your_library_count = len(your_library_objects)

        opponent_library_objects = process_zone(ZoneType.LIBRARY, opponent_seat_id)
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

        # CRITICAL FIX: Count opponent's hand from zone objects
        # Opponent hand cards are hidden (grpId=0) but still tracked as objects in the hand zone
        opponent_hand_objects = process_zone(ZoneType.HAND, opponent_seat_id)
        real_opp_hand_count = len(opponent_hand_objects)
        if real_opp_hand_count != board_state.opponent_hand_count:
            logging.info(f"Correcting opponent hand count from {board_state.opponent_hand_count} to {real_opp_hand_count} based on zone objects.")
            board_state.opponent_hand_count = real_opp_hand_count

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

        # Future: Emit board state changed event for decoupled communication
        # get_event_bus().emit_simple(EventType.BOARD_STATE_CHANGED, board_state, source="GameStateManager")

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
