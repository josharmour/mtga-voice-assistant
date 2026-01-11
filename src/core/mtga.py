import dataclasses
import json
import logging
import os
import re
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from .monitoring import get_monitor
from src.core.domain.game_state import GameState
from src.core.domain.adapters import BoardStateAdapter
from .zone_manager import ZoneManager
from .gre_parser import GREParser

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
    def __init__(self, log_path: str, resume_offset: Optional[int] = None):
        self.log_path = log_path
        self.file = None
        self.inode = None
        self.offset = 0
        self.first_open = True  # Track if this is first time opening
        self.is_caught_up = False # Track if we have processed the backlog
        self.resume_offset = resume_offset  # Optional offset to resume from (for mid-match restart)

    def _check_timestamp_freshness(self, line: bytes, current_time, max_age_hours) -> bool:
        """Helper to check if a log line has a recent timestamp."""
        import datetime
        import re
        # Timestamp pattern (bytes): [UnityCrossThreadLogger]MM/DD/YYYY HH:MM:SS AM/PM
        timestamp_pattern = re.compile(rb'\[UnityCrossThreadLogger\](\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M)')
        ts_match = timestamp_pattern.search(line)
        if ts_match:
            try:
                ts_str = ts_match.group(1).decode('utf-8', errors='ignore')
                log_time = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
                age_hours = (current_time - log_time).total_seconds() / 3600
                return age_hours <= max_age_hours
            except ValueError:
                pass
        return False

    def _find_current_session_start(self) -> int:
        """
        Find the byte offset where the current/most recent MTGA session begins.
        
        Strategy: Scan backward from EOF to find session or draft markers.
        Returns an offset that ensures we capture recent events without
        replaying the entire log.
        """
        import datetime
        import re
        try:
            with open(self.log_path, 'rb') as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                if file_size == 0:
                    return 0

                # Patterns to look for
                session_start_pattern = re.compile(rb'Client\.Connected|Connecting to matchmaker|\"authenticateResponse\"')
                match_start_pattern = re.compile(rb'MatchGameRoomStateChangedEvent|GREMessageType_ConnectResp|Draft\.Notify|BotDraft|DraftStatus|\"draftId\"')
                timestamp_pattern = re.compile(rb'\[UnityCrossThreadLogger\](\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M)')

                # Read the last 500KB of the file
                scan_size = min(500000, file_size)
                scan_start = max(0, file_size - scan_size)
                
                f.seek(scan_start)
                chunk = f.read(scan_size)
                
                # Find all timestamps and their offsets to determine freshness context
                current_time = datetime.datetime.now()
                max_age_hours = 0.5  # 30 minutes - for mid-match restarts
                
                # Find the latest match/session/draft marker in this chunk
                # Search backward (from end of chunk)
                lines = chunk.split(b'\n')
                
                # Track the most recent timestamp we've seen (for context)
                last_seen_timestamp_fresh = None
                
                # First pass: Scan forward to find timestamps and track freshness
                for line in lines:
                    ts_match = timestamp_pattern.search(line)
                    if ts_match:
                        try:
                            ts_str = ts_match.group(1).decode('utf-8', errors='ignore')
                            log_time = datetime.datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")
                            age_hours = (current_time - log_time).total_seconds() / 3600
                            last_seen_timestamp_fresh = age_hours <= max_age_hours
                        except ValueError:
                            pass
                
                # If the most recent timestamp is fresh, start from scan_start
                # This is a simple heuristic: if recent logs are fresh, process them
                if last_seen_timestamp_fresh:
                    logging.info(f"Recent logs are fresh, starting from offset {scan_start}")
                    return scan_start
                
                # If log is stale, skip most of it but keep last 50KB for safety
                skip_to = max(0, file_size - 50000)
                logging.info(f"Log appears stale, skipping to offset {skip_to}")
                return skip_to

        except Exception as e:
            logging.error(f"Error finding session start: {e}")
            return 0

    def _log_draft_context_near_offset(self, offset: int):
        pass

    def find_last_draft_notify(self) -> Optional[str]:
        """Scan backwards for Draft.Notify event (Binary Mode)."""
        try:
            with open(self.log_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                chunk_size = 50000
                pos = max(0, file_size - chunk_size)

                f.seek(pos)
                chunk = f.read(file_size - pos)

                lines = chunk.split(b'\n')
                for line in reversed(lines):
                    if b'Draft.Notify' in line and b'PackCards' in line:
                        return line.decode('utf-8', errors='replace')
                return None
        except Exception as e:
            logging.error(f"Error finding last Draft.Notify: {e}")
            return None

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
        logging.info(f"LogFollower.follow() started! Watching: {self.log_path}")
        while True:
            try:
                current_inode = None
                try:
                    current_inode = os.stat(self.log_path).st_ino
                except FileNotFoundError:
                    logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                    time.sleep(5)
                    continue
                except Exception as e:
                    logging.error(f"Error getting inode: {e}")
                    time.sleep(1)
                    continue

                if self.inode is None or self.inode != current_inode:
                    if self.file:
                        self.file.close()
                    # BINARY MODE OPEN
                    self.file = open(self.log_path, 'rb')
                    self.inode = current_inode
                    
                    if self.first_open:
                        # SMART STARTUP: Use saved match offset if available, otherwise start 500KB before EOF
                        self.file.seek(0, 2)  # Seek to end
                        file_size = self.file.tell()

                        if self.resume_offset is not None and 0 <= self.resume_offset <= file_size:
                            # Resume from saved match offset
                            start_offset = self.resume_offset
                            logging.info(f"📍 Resuming from saved match offset: {start_offset}")
                        else:
                            # SMART STARTUP: Start 500KB before EOF to capture current match context
                            # This balances two needs:
                            # 1. Don't replay entire old matches (stale advice problem)
                            # 2. Recover current game state if app restarts mid-match
                            # MTGA uses diff updates, so we need enough history to reconstruct full state

                            # Start 500KB before EOF (or from start if file is smaller)
                            # 500KB is typically enough for ~1-2 games worth of state
                            context_size = 500000  # 500KB of context for mid-game recovery
                            start_offset = max(0, file_size - context_size)
                            logging.info(f"Log file opened - starting at offset {start_offset} ({context_size/1000:.0f}KB from EOF). Processing recent context...")

                        self.offset = start_offset
                        self.first_open = False
                        self.is_caught_up = False  # Need to catch up on recent context

                    else:
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning.")


                self.file.seek(0, 2)
                current_file_size = self.file.tell()
                
                # CRITICAL FIX: Ensure types are valid before comparison
                if self.offset is None:
                    logging.warning("LogFollower.offset was None. Resetting to 0.")
                    self.offset = 0
                    
                if current_file_size is None:
                     logging.warning("current_file_size was None (unexpected). Defaulting to 0.")
                     current_file_size = 0

                # Check for file truncation (offset > size)
                # This comparison was the source of TypeError if either was None
                if self.offset > current_file_size:
                    logging.warning(f"🔄 File truncated detected (Offset {self.offset} > Size {current_file_size}). Resetting to beginning.")
                    self.offset = 0
                    self.is_caught_up = False

                self.file.seek(self.offset)
                line_count = 0
                while True:
                    line_bytes = self.file.readline()
                    if not line_bytes:
                        self.is_caught_up = True
                        break
                    
                    line_count += 1
                    self.offset = self.file.tell()
                    
                    # Decode line for processing
                    try:
                        line_str = line_bytes.decode('utf-8', errors='replace')
                        stripped_line = line_str.strip()
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(f"Read line: {stripped_line[:100]}...")
                        callback(stripped_line)
                    except Exception as e:
                        logging.error(f"Error processing line: {e}")

                    
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
                import traceback
                logging.error(f"Error following log file: {e}")
                logging.debug(traceback.format_exc())
                time.sleep(1)

    def close(self):
        if self.file:
            self.file.close()

from .legacy_models import BoardState, GameObject, PlayerState, GameHistory

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
    A container for the raw game state data, such as game objects and player states.
    It no longer contains any parsing logic itself.
    """

    def __init__(self, card_lookup=None, zone_manager: ZoneManager = None):
        self.game_objects: Dict[int, GameObject] = {}
        self.players: Dict[int, PlayerState] = {}
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat: Optional[int] = None
        self.priority_player_seat: Optional[int] = None
        self.local_player_seat_id: Optional[int] = None
        self.zone_manager = zone_manager
        self.game_history: GameHistory = GameHistory()
        self.last_event_timestamp: Optional[datetime.datetime] = None
        self._last_timestamp_str: str = ""

        # Phase 3: Deck tracking
        self.submitted_decklist: Dict[int, int] = {}
        self.cards_seen: Dict[int, int] = {}

        # BUG FIX: Load last known deck from persistent storage
        # This allows deck data to survive mid-match restarts
        self._load_last_deck()

        # Phase 3: Known library cards
        self.library_top_known: List[str] = []
        self.scry_info: Optional[str] = None

        # Match state tracking for efficient mid-match resume
        self.match_id: Optional[str] = None
        self.match_start_offset: Optional[int] = None

        # Mulligan tracking
        self.game_stage: str = ""
        self.in_mulligan_phase: bool = False

        # Decision prompt tracking (for triggering advice at key moments)
        self._pending_decision: Optional[str] = None  # Type of decision awaiting input
        self._decision_context: Dict[str, Any] = {}  # Context for the decision

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

    def _save_last_deck(self):
        """Save the current deck to persistent storage."""
        try:
            from pathlib import Path
            deck_file = Path("logs") / "last_deck.json"
            deck_file.parent.mkdir(exist_ok=True)

            # Save deck with timestamp
            deck_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "decklist": self.submitted_decklist,  # {grpId: count}
                "total_cards": sum(self.submitted_decklist.values()),
                "unique_cards": len(self.submitted_decklist)
            }

            with open(deck_file, 'w') as f:
                json.dump(deck_data, f, indent=2)

            logging.debug(f"Saved deck to {deck_file}: {deck_data['total_cards']} cards, {deck_data['unique_cards']} unique")
        except Exception as e:
            logging.warning(f"Failed to save deck to persistent storage: {e}")

    def _load_last_deck(self):
        """Load the last known deck from persistent storage."""
        try:
            from pathlib import Path
            deck_file = Path("logs") / "last_deck.json"

            if not deck_file.exists():
                logging.debug("No saved deck found - this is expected on first run")
                return

            with open(deck_file, 'r') as f:
                deck_data = json.load(f)

            # Restore the decklist (convert string keys back to int if needed)
            decklist = {}
            for key, value in deck_data.get("decklist", {}).items():
                decklist[int(key)] = value

            self.submitted_decklist = decklist

            timestamp = deck_data.get("timestamp", "unknown")
            total_cards = deck_data.get("total_cards", 0)
            unique_cards = deck_data.get("unique_cards", 0)

            logging.info(f"📋 Loaded saved deck from {timestamp}: {total_cards} cards, {unique_cards} unique")
        except Exception as e:
            logging.warning(f"Failed to load saved deck: {e}")

    def _save_match_state(self, log_offset: int, status: str = "active"):
        """Save current match state including log offset for efficient resume."""
        try:
            from pathlib import Path
            import hashlib

            state_file = Path("logs") / "match_state.json"
            state_file.parent.mkdir(exist_ok=True)

            # Generate match ID if we don't have one
            if not self.match_id:
                # Use timestamp + player seats as match identifier
                timestamp = datetime.datetime.now().isoformat()
                match_str = f"{timestamp}_{self.local_player_seat_id}"
                self.match_id = hashlib.md5(match_str.encode()).hexdigest()[:16]

            match_state = {
                "match_id": self.match_id,
                "match_start_time": datetime.datetime.now().isoformat(),
                "match_start_offset": self.match_start_offset or log_offset,
                "last_update_offset": log_offset,
                "status": status,  # "active" or "ended"
                "player_seat_id": self.local_player_seat_id,
                "current_turn": self.current_turn,
                "current_phase": self.current_phase
            }

            with open(state_file, 'w') as f:
                json.dump(match_state, f, indent=2)

            logging.debug(f"Saved match state: {status} at offset {log_offset}")
        except Exception as e:
            logging.warning(f"Failed to save match state: {e}")

    def _load_match_state(self) -> Optional[dict]:
        """Load last match state if available."""
        try:
            from pathlib import Path
            state_file = Path("logs") / "match_state.json"

            if not state_file.exists():
                logging.debug("No saved match state found")
                return None

            with open(state_file, 'r') as f:
                match_state = json.load(f)

            # Only return if match was active (not ended)
            if match_state.get("status") == "active":
                logging.info(f"📍 Found active match state from offset {match_state.get('last_update_offset')}")
                return match_state
            else:
                logging.debug("Last match was already ended")
                return None
        except Exception as e:
            logging.warning(f"Failed to load match state: {e}")
            return None

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

    def reset_match_state(self, log_offset: int = None):
        """Clear all game state when a new match starts"""
        logging.info("🔄 NEW MATCH DETECTED - Clearing all previous match state")
        self._trigger_game_callback("match_started")
        self.game_objects.clear()
        self.players.clear()
        self.current_turn = 0
        self.current_phase = ""
        self.active_player_seat = None
        self.priority_player_seat = None
        self.local_player_seat_id = None

        self.match_id = None
        self.match_start_offset = log_offset
        if log_offset is not None:
            self._save_match_state(log_offset, status="active")

        self.game_history = GameHistory()
        self.cards_seen.clear()
        self.library_top_known.clear()
        self.scry_info = None
        self.game_stage = ""
        self.in_mulligan_phase = False
        if self.zone_manager:
            self.zone_manager.clear()

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

class GameStateManager:
    def __init__(self, card_lookup: "ArenaCardDatabase"):
        self.zone_manager = ZoneManager()
        self.scanner = MatchScanner(card_lookup=card_lookup, zone_manager=self.zone_manager)
        self.card_lookup = card_lookup
        self.gre_parser = GREParser(self.scanner, self.zone_manager, self.card_lookup)

        self._json_parser = JsonStreamParser()
        self.draft_parser = DraftEventParser()
        self._cached_game_state: Optional[GameState] = None
        self._board_state_dirty: bool = True
        self._last_game_state_hash: int = 0

    def register_draft_callback(self, event_type: str, callback: Callable):
        """Register a callback for a specific draft event type."""
        self.draft_parser.register_callback(event_type, callback)

    def register_game_callback(self, event_type: str, callback: Callable):
        """Register a callback for a game event (e.g., 'match_started')."""
        self.scanner.register_game_callback(event_type, callback)

    def get_resume_offset(self, current_file_size: int) -> Optional[int]:
        """
        Get recommended log offset for resuming based on saved match state.
        Returns None if no saved match found, otherwise returns offset to resume from.
        """
        match_state = self.scanner._load_match_state()
        if not match_state:
            return None

        last_offset = match_state.get("last_update_offset", 0)

        # Validate offset is reasonable (within current file size)
        if last_offset > 0 and last_offset <= current_file_size:
            logging.info(f"📍 Resuming from saved match offset: {last_offset} (saved turn {match_state.get('current_turn')})")
            # Restore match ID so we continue the same match
            self.scanner.match_id = match_state.get("match_id")
            self.scanner.match_start_offset = match_state.get("match_start_offset")
            return last_offset
        else:
            logging.warning(f"Saved offset {last_offset} is invalid (file size: {current_file_size})")
            return None

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

    def _compute_state_hash(self, game_state: GameState) -> int:
        """
        Quick hash to detect meaningful changes in game state.
        """
        return hash((
            game_state.local_player.life_total,
            game_state.opponent.life_total,
            game_state.turn_number,
            game_state.phase,
            len(game_state.local_battlefield),
            len(game_state.opponent_battlefield),
            game_state.local_player.hand_size,
            game_state.local_player.has_priority,
        ))

    def get_pending_decision(self) -> Optional[tuple]:
        """
        Get the current pending decision that requires player input.

        Returns:
            Tuple of (decision_type, context) or None if no decision pending.
            decision_type is one of: 'prompt', 'select_targets', 'select_n',
            'declare_attackers', 'declare_blockers', 'mulligan', 'order_damage',
            'actions_available'
        """
        if self.scanner._pending_decision:
            return (self.scanner._pending_decision, self.scanner._decision_context)
        return None

    def clear_pending_decision(self):
        """Clear the pending decision after it has been handled."""
        self.scanner._pending_decision = None
        self.scanner._decision_context = {}

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

                # BUG FIX: Save deck to persistent storage
                # This allows deck data to survive mid-match restarts
                self.scanner._save_last_deck()

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
        if "greToClientEvent" in line or "GREMessageType" in line or "gameStateMessage" in line:
            self.scanner.parse_timestamp(line)

        draft_event = self.draft_parser.parse(line)
        if draft_event is not None:
            return False

        json_to_parse = self._json_parser.feed(line)
        if json_to_parse is None:
            return False

        if not self._json_parser.is_valid_json(json_to_parse):
            logging.warning(f"Invalid JSON structure detected: {json_to_parse[:200]}...")
            return False

        try:
            parsed_data = json.loads(json_to_parse)
            self._parse_deck_submission(parsed_data)
            self._parse_ux_event(parsed_data)
            gre_event_data = self._find_gre_event(parsed_data)
            if gre_event_data:
                state_changed = self.gre_parser.parse_gre_to_client_event(gre_event_data)
                if state_changed:
                    self._mark_board_state_dirty()
                return state_changed
            return False
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed: {e}. Content: {json_to_parse[:200]}...")
            return False

    def get_current_game_state(self) -> Optional[GameState]:
        """
        Get current game state with caching.
        """
        monitor = get_monitor()
        with monitor.measure("mtga.get_current_game_state"):
            if not self._board_state_dirty and self._cached_game_state is not None:
                return self._cached_game_state

            board_state = self._build_board_state()

            if board_state is None:
                return None

            game_state = BoardStateAdapter.to_game_state(board_state)

            if game_state is None:
                return None

            new_hash = self._compute_state_hash(game_state)

            if new_hash == self._last_game_state_hash and self._cached_game_state is not None:
                self._board_state_dirty = False
                return self._cached_game_state

            self._cached_game_state = game_state
            self._board_state_dirty = False
            self._last_game_state_hash = new_hash

            return game_state

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
            game_stage=self.scanner.game_stage,
            pending_decision=self.scanner._pending_decision,
            decision_context=self.scanner._decision_context.copy() if self.scanner._decision_context else {}
        )

        for obj_id, obj in self.scanner.game_objects.items():
            logging.debug(f"  Object {obj_id}: grpId={obj.grp_id}, zoneId={obj.zone_id}, owner={obj.owner_seat_id}")

        # P0 Performance: Card names/colors are now resolved once on object creation.
        # This minimal fallback handles only edge cases (e.g., objects created before card_lookup was available).
        fallback_count = 0
        for obj in self.scanner.game_objects.values():
            # Fallback: Only resolve if still missing (should be rare)
            if obj.grp_id != 0 and (not obj.name or obj.name.startswith("Unknown Card")):
                new_name = self.card_lookup.get_card_name(obj.grp_id)
                # Only count as resolved if we got a real name back (not the default "Unknown Card...")
                if new_name and not new_name.startswith("Unknown Card"):
                    obj.name = new_name
                    fallback_count += 1
                    logging.debug(f"Fallback resolution success for {obj.grp_id}: {obj.name}")

            if obj.grp_id != 0 and not obj.color_identity:
                try:
                    card_data = self.card_lookup.get_card_data(obj.grp_id)
                    if card_data:
                        obj.color_identity = card_data.get("color_identity", "")
                        # We don't count color updates as "fallback resolution needed" warning worthy
                        # as they are less critical than names
                except Exception as e:
                    logging.debug(f"Could not fetch color for {obj.grp_id}: {e}")

        if fallback_count > 0:
            logging.info(f"⚠️ Fallback resolution needed for {fallback_count} objects (should be rare)")

        # P0 Performance: Use zone-based lookups instead of iterating all objects
        # This changes from O(n) to O(zones * objects_per_zone), which is much faster

        # P1: Helper function to process objects in a zone using enum comparisons (faster)
        def process_zone(zone_type_enum: ZoneType, owner_filter: Optional[int] = None):
            """Get all objects in a specific zone type, optionally filtered by owner."""
            zone_ids = [zid for zid, zt in self.zone_manager.zone_id_to_enum.items() if zt == zone_type_enum]
            objects = []
            for zone_id in zone_ids:
                for instance_id in self.zone_manager.get_zone_contents(zone_id):
                    obj = self.scanner.game_objects.get(instance_id)
                    if obj and (owner_filter is None or obj.owner_seat_id == owner_filter):
                        objects.append(obj)
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
            # PERSISTENCE FIX: Update the source of truth to prevent repeated logging
            if your_player:
                your_player.hand_count = real_hand_count

        # CRITICAL FIX: Count opponent's hand from zone objects
        # Opponent hand cards are hidden (grpId=0) but still tracked as objects in the hand zone
        opponent_hand_objects = process_zone(ZoneType.HAND, opponent_seat_id)
        real_opp_hand_count = len(opponent_hand_objects)
        if real_opp_hand_count != board_state.opponent_hand_count:
            logging.info(f"Correcting opponent hand count from {board_state.opponent_hand_count} to {real_opp_hand_count} based on zone objects.")
            board_state.opponent_hand_count = real_opp_hand_count
            # PERSISTENCE FIX: Update the source of truth to prevent repeated logging
            if opponent_player:
                opponent_player.hand_count = real_opp_hand_count

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
