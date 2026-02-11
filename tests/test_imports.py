"""
Basic import tests for MTGA Voice Advisor.

These tests verify that all core modules can be imported without error.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_imports():
    """Test that core modules can be imported."""
    from src.core.mtga import LogFollower, GameStateManager
    from src.core.ai import AIAdvisor
    from src.core.formatters import BoardStateFormatter


def test_data_imports():
    """Test that data modules can be imported."""
    from src.data.arena_cards import ArenaCardDatabase
    from src.data.data_management import CardStatsDB


def test_llm_imports():
    """Test that LLM advisor modules can be imported."""
    from src.core.llm.base import BaseMTGAdvisor, LLMConfig
    from src.core.llm.prompt_builder import MTGPromptBuilder
    from src.core.llm.google_advisor import GeminiAdvisor, GEMINI_MODELS


def test_config_imports():
    """Test that config modules can be imported."""
    from src.config.config_manager import UserPreferences
    from src.config.constants import ALL_SETS


def test_gemini_models_list():
    """Verify Gemini models list is populated."""
    from src.core.llm.google_advisor import GEMINI_MODELS
    assert len(GEMINI_MODELS) >= 5, "Expected at least 5 Gemini models"
    assert "gemini-2.5-flash" in GEMINI_MODELS


if __name__ == "__main__":
    test_core_imports()
    print("[OK] Core imports")

    test_data_imports()
    print("[OK] Data imports")

    test_llm_imports()
    print("[OK] LLM imports")

    test_config_imports()
    print("[OK] Config imports")

    test_gemini_models_list()
    print("[OK] Gemini models list")

    print("\nAll import tests passed!")
