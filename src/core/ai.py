import logging
from typing import Dict, List

from ..config.config_manager import UserPreferences
from .llm.google_advisor import GeminiAdvisor
from .llm.openai_advisor import OpenAIAdvisor
from .llm.anthropic_advisor import AnthropicAdvisor
from .llm.ollama_advisor import OllamaAdvisor

logger = logging.getLogger(__name__)

class AIAdvisor:
    """
    Main AI Advisor interface.
    Dynamically delegates to a selected provider's advisor.
    """
    def __init__(self, card_db=None, prefs: UserPreferences = None):
        if not prefs:
            logger.warning("AIAdvisor initialized without preferences. Loading defaults.")
            prefs = UserPreferences.load()

        self.card_db = card_db
        self.advisor = None

        provider = prefs.model_provider.lower()
        model = prefs.current_model

        advisor_map = {
            "google": (GeminiAdvisor, {}),
            "openai": (OpenAIAdvisor, {"api_key": prefs.openai_api_key}),
            "anthropic": (AnthropicAdvisor, {"api_key": prefs.anthropic_api_key}),
            "ollama": (OllamaAdvisor, {}),
        }

        if provider in advisor_map:
            AdvisorClass, kwargs = advisor_map[provider]
            try:
                self.advisor = AdvisorClass(
                    model_name=model,
                    card_db=card_db,
                    **kwargs
                )
                logger.info(f"{provider.capitalize()} Advisor initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider.capitalize()} Advisor: {e}")
                self.advisor = None
        else:
            logger.error(f"Unknown model provider: {prefs.model_provider}. AI Advisor not initialized.")

    def set_model(self, provider: str, model_name: str, api_key: str = None):
        """
        Hot-swap the AI model/provider.
        """
        provider = provider.lower()
        
        # Pass empty card_db if we don't have one stored (TODO: Store card_db in __init__)
        # Ideally, we should store card_db in self to pass it here
        card_db = getattr(self, 'card_db', None)

        advisor_map = {
            "google": (GeminiAdvisor, {"api_key": api_key}),
            "openai": (OpenAIAdvisor, {"api_key": api_key}),
            "anthropic": (AnthropicAdvisor, {"api_key": api_key}),
            "ollama": (OllamaAdvisor, {}),
        }

        if provider in advisor_map:
            AdvisorClass, kwargs = advisor_map[provider]
            
            # Clean kwargs (remove None values)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            try:
                self.advisor = AdvisorClass(
                    model_name=model_name,
                    card_db=card_db,
                    **kwargs
                )
                logger.info(f"Hot-swapped to {provider.capitalize()} Advisor with model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to hot-swap to {provider.capitalize()} Advisor: {e}")
                return False
        else:
            logger.error(f"Unknown model provider: {provider}")
            return False

    def get_tactical_advice(self, board_state: Dict, user_query: str = "") -> str:
        """
        Get tactical advice from the AI.
        """
        if not self.advisor:
            return "AI Advisor not initialized. Please configure a provider in the settings."
        return self.advisor.get_tactical_advice(board_state)

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        if not self.advisor:
            return "AI Advisor not initialized. Please configure a provider in the settings."
        return self.advisor.get_draft_pick(pack_cards, current_pool)
