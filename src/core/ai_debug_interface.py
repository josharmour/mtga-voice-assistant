"""
AI Debugging Interface - Real-time state export and command interface.

Allows AI assistants to:
1. Get current match/UI state instantly
2. Query logs with structured filters
3. Execute debugging commands
4. Export state snapshots for analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import dataclasses
from src.core.version import get_version


class AIDebugInterface:
    """
    Interface for AI agents to debug the MTGA Advisor in real-time.

    Usage:
        debug = AIDebugInterface(app_instance)

        # Get current state
        state = debug.get_current_state()

        # Query logs
        errors = debug.query_logs(level="ERROR", last_minutes=5)

        # Get match timeline
        timeline = debug.get_match_timeline()
    """

    def __init__(self, app=None, game_state_mgr=None, ui=None):
        self.app = app
        self.game_state_mgr = game_state_mgr
        self.ui = ui
        self.logger = logging.getLogger("AI_DEBUG_INTERFACE")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get complete current state snapshot for AI analysis.

        Returns everything an AI needs to understand what's happening:
        - Match state (turn, phase, life, cards)
        - UI state (what's displayed)
        - System state (advisor status, connections)
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "match": self._get_match_state(),
            "ui": self._get_ui_state(),
            "system": self._get_system_state(),
            "advisor": self._get_advisor_state()
        }
        return state

    def _get_match_state(self) -> Dict[str, Any]:
        """Get current match state."""
        if not self.game_state_mgr:
            return {"status": "no_game_state_manager"}

        try:
            board_state = self.game_state_mgr.get_current_board_state()
            if not board_state:
                return {"status": "no_active_match"}

            return {
                "status": "active_match",
                "turn": board_state.current_turn,
                "phase": board_state.current_phase,
                "is_your_turn": board_state.is_your_turn,
                "has_priority": board_state.has_priority,
                "life": {
                    "yours": board_state.your_life,
                    "opponent": board_state.opponent_life
                },
                "zones": {
                    "hand": {
                        "count": board_state.your_hand_count,
                        "cards": [{"name": c.name, "grp_id": c.grp_id} for c in board_state.your_hand[:10]]  # First 10
                    },
                    "battlefield": {
                        "count": len(board_state.your_battlefield),
                        "cards": [{"name": c.name, "tapped": c.is_tapped} for c in board_state.your_battlefield[:15]]
                    },
                    "graveyard": {
                        "count": len(board_state.your_graveyard),
                        "cards": [c.name for c in board_state.your_graveyard[:10]]
                    },
                    "library": {
                        "count": board_state.your_library_count
                    }
                },
                "opponent_zones": {
                    "hand_count": board_state.opponent_hand_count,
                    "battlefield_count": len(board_state.opponent_battlefield),
                    "battlefield_cards": [{"name": c.name, "tapped": c.is_tapped} for c in board_state.opponent_battlefield[:15]]
                },
                "deck": {
                    "has_decklist": bool(board_state.your_decklist),
                    "total_cards": sum(board_state.your_decklist.values()) if board_state.your_decklist else 0,
                    "unique_cards": len(board_state.your_decklist) if board_state.your_decklist else 0
                },
                "mana_pool": board_state.your_mana_pool if board_state.your_mana_pool else {}
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _get_ui_state(self) -> Dict[str, Any]:
        """Get current UI state (what user sees)."""
        if not self.ui:
            return {"status": "no_ui"}

        try:
            # Check for TTS on the app object (not the UI object)
            # TTS is stored in app.tts, not gui.tts
            tts_enabled = False
            if self.app and hasattr(self.app, 'tts'):
                tts_enabled = self.app.tts is not None

            return {
                "mode": "gui" if self.ui else "cli",
                "preferences": {
                    "ai_provider": self.ui.prefs.model_provider if hasattr(self.ui, 'prefs') and self.ui.prefs else None,
                    "ai_model": self.ui.prefs.current_model if hasattr(self.ui, 'prefs') and self.ui.prefs else None,
                    "voice_enabled": tts_enabled
                },
                "recent_messages": self._get_recent_ui_messages() if hasattr(self.ui, 'messages_text') else [],
                "settings_visible": getattr(self.ui, '_settings_visible', False) if self.ui else False
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _get_recent_ui_messages(self, count: int = 5) -> List[str]:
        """Get recent messages from UI."""
        try:
            if hasattr(self.ui, 'messages_text'):
                all_text = self.ui.messages_text.get("1.0", "end-1c")
                lines = all_text.split('\n')
                return [line for line in lines[-count:] if line.strip()]
            return []
        except:
            return []

    def _get_system_state(self) -> Dict[str, Any]:
        """Get system/connection state."""
        return {
            "log_follower": {
                "running": True,  # Assume running if we can query
                "caught_up": getattr(self.app.log_follower, 'is_caught_up', False) if hasattr(self.app, 'log_follower') else None
            },
            "files": {
                "advisor_log": str(Path("logs/advisor.log").absolute()),
                "mtga_log": str(Path(self.app.log_path).absolute()) if hasattr(self.app, 'log_path') else None,
                "deck_state": str(Path("logs/last_deck.json").absolute()),
                "match_state": str(Path("logs/match_state.json").absolute())
            }
        }

    def _get_advisor_state(self) -> Dict[str, Any]:
        """Get AI advisor state."""
        advisor = getattr(self.app, 'ai_advisor', None)
        return {
            "initialized": advisor is not None,
            "last_advice_time": advisor.last_advice_time.isoformat() if advisor and advisor.last_advice_time else None,
            "advice_count": getattr(advisor, 'advice_count', 0) if advisor else 0
        }

    def query_logs(self,
                   level: str = None,
                   component: str = None,
                   last_minutes: int = None,
                   pattern: str = None,
                   max_lines: int = 100) -> List[Dict[str, str]]:
        """
        Query advisor logs with filters.

        Args:
            level: Filter by log level (ERROR, WARNING, INFO, DEBUG)
            component: Filter by component tag [COMPONENT]
            last_minutes: Only show logs from last N minutes. If None, uses app start time.
            pattern: Grep pattern to match
            max_lines: Maximum number of lines to return

        Returns:
            List of log entries with metadata
        """
        log_file = Path("logs/advisor.log")
        if not log_file.exists():
            return []

        results = []
        
        # Determine cutoff time: either explicitly N minutes ago, or app start time
        cutoff_time = None
        if last_minutes is not None:
            cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
        elif hasattr(self.app, 'start_time'):
            cutoff_time = self.app.start_time

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Parse timestamp
                    if not line.strip():
                        continue

                    # Simple time filter (assumes YYYY-MM-DD HH:MM:SS format)
                    try:
                        timestamp_str = line[:19]
                        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        if cutoff_time and log_time < cutoff_time:
                            continue
                    except:
                        pass

                    # Apply filters
                    if level and f" - {level} - " not in line:
                        continue
                    if component and f"[{component}]" not in line:
                        continue
                    if pattern and pattern not in line:
                        continue

                    results.append({
                        "timestamp": timestamp_str if timestamp_str else "unknown",
                        "line": line.strip()
                    })

                    # If we have a max_lines limit, we should ideally collect from the end
                    # But for stream processing, we collect all matches then slice
                    # Optimization: If file is huge, this might be slow. 
                    # For now, simplistic approach is fine.

            return results[-max_lines:]  # Most recent first
        except Exception as e:
            self.logger.error(f"Error querying logs: {e}")
            return []

    def get_match_timeline(self, max_events: int = 50) -> List[Dict[str, Any]]:
        """
        Get timeline of current match events.

        Returns:
            List of events in chronological order
        """
        events = []

        # Query for match events
        log_entries = self.query_logs(
            pattern="⚡",
            last_minutes=60,
            max_lines=max_events
        )

        for entry in log_entries:
            events.append({
                "timestamp": entry["timestamp"],
                "event": entry["line"]
            })

        return events

    def get_recent_errors(self, count: int = 10) -> List[Dict[str, str]]:
        """Get recent errors with context."""
        return self.query_logs(level="ERROR", max_lines=count)

    def get_recent_warnings(self, count: int = 10) -> List[Dict[str, str]]:
        """Get recent warnings."""
        return self.query_logs(level="WARNING", max_lines=count)

    def export_debug_bundle(self, output_path: str = None, overwrite: bool = False) -> str:
        """
        Export complete debug bundle for AI analysis.

        Creates a JSON file with:
        - Current state
        - Recent logs (last 5 min)
        - Match timeline
        - System info
        - Screenshot (focused on relevant windows)

        Args:
            output_path: Custom path for the JSON bundle
            overwrite: If True, uses a fixed filename 'logs/debug_bundle.json' and overwrites it.

        Returns:
            Path to exported bundle
        """
        if output_path is None:
            if overwrite:
                output_path = "logs/debug_bundle.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"logs/debug_bundle_{timestamp}.json"

        # Capture screenshot (Targeted)
        screenshot_path = None
        try:
            import pyautogui
            import pygetwindow as gw
            from PIL import Image
            
            screenshot_file = Path("logs/debug_screenshot.png")
            screenshot_file.parent.mkdir(exist_ok=True)
            
            # Find relevant windows
            windows = []
            try:
                for win in gw.getAllWindows():
                    title = win.title.lower()
                    if "mtga" in title or "mtg arena" in title:
                        # Avoid duplicates/empty titles
                        if win.width > 0 and win.height > 0 and win.visible:
                            windows.append(win)
            except Exception as e:
                self.logger.warning(f"Window enumeration failed: {e}")

            if windows:
                self.logger.info(f"Found {len(windows)} relevant windows for screenshot.")
                screenshots = []
                total_width = 0
                max_height = 0
                
                # Capture each window
                for win in windows:
                    try:
                        # Activate window to bring to front (optional, might be intrusive)
                        # win.activate() 
                        region = (win.left, win.top, win.width, win.height)
                        shot = pyautogui.screenshot(region=region)
                        screenshots.append(shot)
                        total_width += shot.width
                        max_height = max(max_height, shot.height)
                    except Exception as e:
                        self.logger.warning(f"Failed to capture window '{win.title}': {e}")

                if screenshots:
                    # Stitch images side-by-side
                    combined_image = Image.new('RGB', (total_width, max_height))
                    x_offset = 0
                    for shot in screenshots:
                        combined_image.paste(shot, (x_offset, 0))
                        x_offset += shot.width
                    
                    combined_image.save(screenshot_file)
                    screenshot_path = str(screenshot_file.absolute())
                    self.logger.info(f"Targeted screenshot saved to: {screenshot_path}")
                else:
                    raise Exception("No valid window screenshots captured")
            else:
                # No windows found, fallback to fullscreen
                self.logger.info("No relevant windows found. Taking full screen screenshot.")
                screenshot = pyautogui.screenshot()
                screenshot.save(screenshot_file)
                screenshot_path = str(screenshot_file.absolute())

        except ImportError as e:
            self.logger.warning(f"Screenshot dependencies missing ({e}) - screenshot skipped")
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            # Try fallback to full screen if window logic failed
            try:
                import pyautogui
                screenshot_file = Path("logs/debug_screenshot.png")
                screenshot = pyautogui.screenshot()
                screenshot.save(screenshot_file)
                screenshot_path = str(screenshot_file.absolute())
                self.logger.info(f"Fallback full screenshot saved to: {screenshot_path}")
            except Exception as f:
                self.logger.error(f"Fallback screenshot also failed: {f}")

        bundle = {
            "export_time": datetime.now().isoformat(),
            "app_version": get_version(),
            "screenshot": screenshot_path,
            "current_state": self.get_current_state(),
            "recent_logs": self.query_logs(max_lines=1000), # No last_minutes = use session start
            "recent_errors": self.get_recent_errors(count=20),
            "recent_warnings": self.get_recent_warnings(count=20),
            "match_timeline": self.get_match_timeline(max_events=100)
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2, default=str)

        self.logger.info(f"Debug bundle exported to: {output_file}")
        return str(output_file.absolute())

    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a debugging command.

        Available commands:
        - "get_state": Get current state
        - "query_logs": Query logs with filters
        - "export_bundle": Export debug bundle
        - "check_health": Check system health

        Returns:
            Command result
        """
        commands = {
            "get_state": lambda: self.get_current_state(),
            "query_logs": lambda: self.query_logs(**kwargs),
            "export_bundle": lambda: {"bundle_path": self.export_debug_bundle()},
            "check_health": lambda: self._check_health(),
            "get_timeline": lambda: self.get_match_timeline(**kwargs),
            "get_errors": lambda: self.get_recent_errors(**kwargs),
        }

        if command not in commands:
            return {"error": f"Unknown command: {command}", "available": list(commands.keys())}

        try:
            result = commands[command]()
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_health(self) -> Dict[str, Any]:
        """Check system health."""
        health = {
            "overall": "healthy",
            "issues": []
        }

        # Check log file
        log_file = Path("logs/advisor.log")
        if not log_file.exists():
            health["issues"].append("advisor.log not found")
            health["overall"] = "degraded"

        # Check for recent errors
        errors = self.get_recent_errors(count=5)
        if len(errors) > 3:
            health["issues"].append(f"{len(errors)} recent errors")
            health["overall"] = "degraded"

        # Check game state
        match_state = self._get_match_state()
        if match_state.get("status") == "error":
            health["issues"].append("game state error")
            health["overall"] = "unhealthy"

        return health


# Convenience function for AI agents
def create_debug_interface(app=None) -> AIDebugInterface:
    """Create a debug interface for AI agents."""
    game_state_mgr = getattr(app, 'game_state_mgr', None) if app else None
    ui = getattr(app, 'gui', None) if app else None
    return AIDebugInterface(app=app, game_state_mgr=game_state_mgr, ui=ui)
