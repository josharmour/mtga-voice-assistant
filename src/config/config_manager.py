#!/usr/bin/env python3
"""
User Preferences Manager for MTGA Voice Advisor

Persists user settings like:
- Window geometry (size/position)
- UI preferences (always on top, theme)
- Voice settings (volume, engine)
- Model preferences (which LLM to use)
- etc.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".mtga_advisor"
PREFS_FILE = CONFIG_DIR / "preferences.json"


@dataclass
class UserPreferences:
    """User preferences for the advisor application."""

    # Window settings
    window_geometry: str = "900x700"  # Format: "WIDTHxHEIGHT"
    window_x: int = 100
    window_y: int = 100
    always_on_top: bool = True

    # Secondary window geometries
    deck_window_geometry: str = "350x600+1000+100"
    board_window_geometry: str = "600x800+50+50"
    log_window_geometry: str = "800x400+50+600"

    # UI preferences
    theme: str = "dark"  # dark/light
    show_board_state: bool = True
    show_messages: bool = True

    # Voice preferences
    voice_enabled: bool = True
    volume: int = 80  # 0-100
    voice_engine: str = "kokoro"  # kokoro/pyttsx3/system

    # Model preferences
    current_model: str = "gemini-3-pro-preview"
    current_voice: str = "am_adam"

    # Game preferences
    auto_detect_draft: bool = True
    show_win_rates: bool = True
    show_rules_context: bool = True
    opponent_turn_alerts: bool = True
    show_thinking: bool = True

    # API Keys (stored securely in user preferences)
    github_token: str = ""
    github_owner: str = ""
    github_repo: str = ""
    imgbb_api_key: str = ""

    @classmethod
    def load(cls) -> "UserPreferences":
        """Load preferences from file or create defaults."""
        if PREFS_FILE.exists():
            try:
                with open(PREFS_FILE, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded preferences from {PREFS_FILE}")
                    return cls(**data)
            except Exception as e:
                logger.warning(f"Failed to load preferences: {e}. Using defaults.")
                return cls()
        else:
            logger.info("No preferences file found. Using defaults.")
            return cls()

    def save(self):
        """Save preferences to file."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            with open(PREFS_FILE, 'w') as f:
                json.dump(asdict(self), f, indent=2)
                logger.debug(f"Saved preferences to {PREFS_FILE}")

        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def set_window_geometry(self, width: int, height: int, x: int = None, y: int = None):
        """Update window geometry."""
        self.window_geometry = f"{width}x{height}"
        if x is not None:
            self.window_x = x
        if y is not None:
            self.window_y = y
        self.save()

    def set_voice_settings(self, enabled: bool = None, volume: int = None, engine: str = None):
        """Update voice settings."""
        if enabled is not None:
            self.voice_enabled = enabled
        if volume is not None:
            self.volume = max(0, min(100, volume))
        if engine is not None:
            self.voice_engine = engine
        self.save()

    def set_model(self, model: str):
        """Set the current LLM model."""
        self.current_model = model
        self.save()

    def set_voice_name(self, voice: str):
        """Set the current TTS voice."""
        self.current_voice = voice
        self.save()

    def set_game_preferences(self, opponent_alerts: bool = None, show_thinking: bool = None):
        """Update game preferences."""
        if opponent_alerts is not None:
            self.opponent_turn_alerts = opponent_alerts
        if show_thinking is not None:
            self.show_thinking = show_thinking
        self.save()

    def set_api_keys(self, github_token: str = None, github_owner: str = None,
                     github_repo: str = None, imgbb_api_key: str = None):
        """Update API keys for bug reporting."""
        if github_token is not None:
            self.github_token = github_token
        if github_owner is not None:
            self.github_owner = github_owner
        if github_repo is not None:
            self.github_repo = github_repo
        if imgbb_api_key is not None:
            self.imgbb_api_key = imgbb_api_key
        self.save()

    def has_github_credentials(self) -> bool:
        """Check if GitHub credentials are configured."""
        return bool(self.github_token and self.github_owner and self.github_repo)

    def has_imgbb_credentials(self) -> bool:
        """Check if ImgBB API key is configured."""
        return bool(self.imgbb_api_key)

    def __repr__(self) -> str:
        """String representation of preferences."""
        return (
            f"UserPreferences("
            f"geometry={self.window_geometry}, "
            f"theme={self.theme}, "
            f"voice={self.voice_engine}, "
            f"model={self.current_model}"
            f")"
        )


class PreferencesManager:
    """Centralized manager for user preferences."""

    _instance: Optional["PreferencesManager"] = None
    _prefs: Optional[UserPreferences] = None

    def __new__(cls) -> "PreferencesManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize preferences manager."""
        if self._prefs is None:
            self._prefs = UserPreferences.load()

    @property
    def prefs(self) -> UserPreferences:
        """Get current preferences."""
        return self._prefs

    def load_fresh(self) -> UserPreferences:
        """Reload preferences from disk."""
        self._prefs = UserPreferences.load()
        return self._prefs

    def save(self):
        """Save current preferences to disk."""
        self._prefs.save()

    def update(self, **kwargs):
        """Update multiple preferences at once."""
        for key, value in kwargs.items():
            if hasattr(self._prefs, key):
                setattr(self._prefs, key, value)
            else:
                logger.warning(f"Unknown preference: {key}")
        self.save()


# Global instance
manager = PreferencesManager()


if __name__ == "__main__":
    # Test the preferences system
    logging.basicConfig(level=logging.DEBUG)

    # Load preferences
    prefs = UserPreferences.load()
    print(f"Current preferences: {prefs}")

    # Modify and save
    prefs.window_geometry = "1200x800"
    prefs.volume = 75
    prefs.current_model = "mistral"
    prefs.save()

    print(f"Updated preferences: {prefs}")

    # Load again to verify
    prefs2 = UserPreferences.load()
    print(f"Reloaded preferences: {prefs2}")

    # Show config location
    print(f"\nConfig location: {PREFS_FILE}")
