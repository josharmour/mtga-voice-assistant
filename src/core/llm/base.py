"""
Base configuration and protocol for LLM advisors.

This module provides:
- LLMConfig: Configuration dataclass for all LLM providers
- LLMAdapter: Protocol defining the common interface for advisors
- BaseMTGAdvisor: Base class with shared functionality
- create_advisor: Factory function for instantiating advisors
"""

import logging
import os
from typing import Protocol, Dict, List, Optional, runtime_checkable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM providers.

    This dataclass encapsulates all settings needed to initialize
    any LLM advisor, providing a unified configuration interface.

    Attributes:
        provider: Provider name ("google", "ollama", "openai", "anthropic")
        model: Model identifier (e.g., "gemini-3-pro-preview", "gpt-4-turbo")
        max_tokens: Maximum tokens for generation (default: 500)
        temperature: Sampling temperature 0.0-1.0 (default: 0.7)
        api_key: API key for cloud providers (optional for local models)
        base_url: Base URL for API (primarily for Ollama)
        timeout: Request timeout in seconds (default: 30)
        extra_params: Additional provider-specific parameters
    """
    provider: str
    model: str
    max_tokens: int = 500
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    extra_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Normalize provider name to lowercase
        self.provider = self.provider.lower()

        # Validate provider
        valid_providers = {"google", "ollama", "openai", "anthropic"}
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. "
                f"Must be one of {valid_providers}"
            )

        # Validate numeric ranges
        if not 0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be 0-2.0, got {self.temperature}")

        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    @classmethod
    def from_preferences(cls, prefs) -> "LLMConfig":
        """
        Create LLMConfig from UserPreferences object.

        Args:
            prefs: UserPreferences instance with model settings

        Returns:
            LLMConfig instance configured from preferences
        """
        provider = prefs.model_provider.lower()
        model = prefs.current_model

        # Map provider to API key
        api_key_map = {
            "google": prefs.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "openai": prefs.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": prefs.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "ollama": None,  # No API key needed for local Ollama
        }

        api_key = api_key_map.get(provider)

        # Ollama-specific base URL
        base_url = None
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Protocol defining the common interface for all LLM advisors.

    All advisor implementations should conform to this protocol
    to ensure consistent behavior across different providers.
    """

    def get_tactical_advice(
        self,
        board_state: Dict,
        game_history: Optional[List[str]] = None
    ) -> str:
        """
        Generate tactical advice for the current board state.

        Args:
            board_state: Dictionary containing current game state
            game_history: Optional list of recent game events

        Returns:
            Tactical advice as a string
        """
        ...

    def get_draft_pick(
        self,
        pack_cards: List[str],
        current_pool: List[str]
    ) -> str:
        """
        Analyze a draft pack and suggest a pick.

        Args:
            pack_cards: List of card names in the current pack
            current_pool: List of cards already picked

        Returns:
            Draft pick recommendation as a string
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is ready, False otherwise
        """
        ...


class BaseMTGAdvisor:
    """
    Base class for MTG advisors with common functionality.

    This class provides shared initialization logic and helper methods
    that can be inherited by all concrete advisor implementations.

    Child classes should:
    1. Call super().__init__(config) in their __init__
    2. Implement get_tactical_advice() and get_draft_pick()
    3. Optionally override is_available() for provider-specific checks
    """

    def __init__(self, config: LLMConfig, card_db=None, scryfall_client=None):
        """
        Initialize base advisor with configuration.

        Args:
            config: LLMConfig instance with provider settings
            card_db: Optional card database instance
            scryfall_client: Optional Scryfall client instance
        """
        self.config = config
        self.card_db = card_db
        self.scryfall_client = scryfall_client

        logger.info(
            f"Initializing {config.provider.capitalize()} advisor "
            f"with model: {config.model}"
        )

    def get_tactical_advice(
        self,
        board_state: Dict,
        game_history: Optional[List[str]] = None
    ) -> str:
        """
        Get tactical advice for current board state.

        Must be implemented by child classes.

        Args:
            board_state: Dictionary containing current game state
            game_history: Optional list of recent game events

        Returns:
            Tactical advice as a string
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_tactical_advice()"
        )

    def get_draft_pick(
        self,
        pack_cards: List[str],
        current_pool: List[str]
    ) -> str:
        """
        Get draft pick recommendation.

        Must be implemented by child classes.

        Args:
            pack_cards: List of card names in the current pack
            current_pool: List of cards already picked

        Returns:
            Draft pick recommendation as a string
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_draft_pick()"
        )

    def is_available(self) -> bool:
        """
        Check if the advisor is available.

        Base implementation returns True. Override for provider-specific checks.

        Returns:
            True if advisor is ready to use
        """
        return True

    def _format_error_message(self, error: Exception) -> str:
        """
        Format a user-friendly error message.

        Args:
            error: Exception that occurred

        Returns:
            Formatted error message
        """
        provider = self.config.provider.capitalize()
        return (
            f"I'm having trouble connecting to the {provider} service. "
            f"Error: {str(error)}"
        )


def create_advisor(
    config: LLMConfig,
    card_db=None,
    scryfall_client=None
) -> BaseMTGAdvisor:
    """
    Factory function to create the appropriate advisor based on configuration.

    This function handles dynamic importing and instantiation of the correct
    advisor class based on the provider specified in the config.

    Args:
        config: LLMConfig instance specifying provider and model
        card_db: Optional card database instance
        scryfall_client: Optional Scryfall client instance

    Returns:
        Initialized advisor instance

    Raises:
        ValueError: If provider is unknown or advisor initialization fails

    Example:
        >>> config = LLMConfig(provider="google", model="gemini-3-pro-preview")
        >>> advisor = create_advisor(config)
        >>> advice = advisor.get_tactical_advice(board_state)
    """
    # Import advisors dynamically to avoid circular imports
    # and to only load dependencies for the selected provider
    provider = config.provider.lower()

    try:
        if provider == "google":
            from .google_advisor import GeminiAdvisor
            return GeminiAdvisor(
                model_name=config.model,
                scryfall_client=scryfall_client,
                card_db=card_db
            )

        elif provider == "openai":
            from .openai_advisor import OpenAIAdvisor
            return OpenAIAdvisor(
                model_name=config.model,
                api_key=config.api_key,
                card_db=card_db,
                scryfall_client=scryfall_client
            )

        elif provider == "anthropic":
            from .anthropic_advisor import AnthropicAdvisor
            return AnthropicAdvisor(
                model_name=config.model,
                api_key=config.api_key,
                card_db=card_db,
                scryfall_client=scryfall_client
            )

        elif provider == "ollama":
            from .ollama_advisor import OllamaAdvisor
            return OllamaAdvisor(
                model_name=config.model,
                card_db=card_db,
                scryfall_client=scryfall_client
            )

        else:
            # This should never happen due to validation in LLMConfig.__post_init__
            raise ValueError(f"Unknown provider: {provider}")

    except ImportError as e:
        logger.error(f"Failed to import advisor for {provider}: {e}")
        raise ValueError(
            f"Could not load {provider} advisor. "
            f"Make sure the required dependencies are installed."
        ) from e

    except Exception as e:
        logger.error(f"Failed to initialize {provider} advisor: {e}")
        raise ValueError(
            f"Failed to initialize {provider} advisor: {e}"
        ) from e


# Convenience function for common use case
def create_advisor_from_preferences(prefs, card_db=None, scryfall_client=None) -> BaseMTGAdvisor:
    """
    Create an advisor directly from UserPreferences.

    This is a convenience wrapper around create_advisor() that handles
    the conversion from UserPreferences to LLMConfig.

    Args:
        prefs: UserPreferences instance with model settings
        card_db: Optional card database instance
        scryfall_client: Optional Scryfall client instance

    Returns:
        Initialized advisor instance

    Example:
        >>> from src.config.config_manager import UserPreferences
        >>> prefs = UserPreferences.load()
        >>> advisor = create_advisor_from_preferences(prefs)
    """
    config = LLMConfig.from_preferences(prefs)
    return create_advisor(config, card_db=card_db, scryfall_client=scryfall_client)
