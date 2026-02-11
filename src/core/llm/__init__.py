"""
LLM Advisor Module

This module provides a unified interface for multiple LLM providers
used in the MTGA Voice Advisor application.

Supported Providers:
- Google (Gemini)
- OpenAI (GPT)
- Anthropic (Claude)
- Ollama (Local models)

Basic Usage:
    >>> from src.core.llm import LLMConfig, create_advisor
    >>> config = LLMConfig(provider="google", model="gemini-3-pro-preview")
    >>> advisor = create_advisor(config)
    >>> advice = advisor.get_tactical_advice(board_state)

Using with Preferences:
    >>> from src.config.config_manager import UserPreferences
    >>> from src.core.llm import create_advisor_from_preferences
    >>> prefs = UserPreferences.load()
    >>> advisor = create_advisor_from_preferences(prefs)
"""

# Base configuration and protocols
from .base import (
    LLMConfig,
    LLMAdapter,
    BaseMTGAdvisor,
    create_advisor,
    create_advisor_from_preferences,
)

# Individual advisor implementations
from .google_advisor import GeminiAdvisor
from .openai_advisor import OpenAIAdvisor
from .anthropic_advisor import AnthropicAdvisor
from .ollama_advisor import OllamaAdvisor

# Shared prompt building
from .prompt_builder import MTGPromptBuilder

__all__ = [
    # Configuration and protocols
    "LLMConfig",
    "LLMAdapter",
    "BaseMTGAdvisor",
    "create_advisor",
    "create_advisor_from_preferences",

    # Concrete advisors
    "GeminiAdvisor",
    "OpenAIAdvisor",
    "AnthropicAdvisor",
    "OllamaAdvisor",

    # Utilities
    "MTGPromptBuilder",
]

# Version info
__version__ = "0.1.0"
