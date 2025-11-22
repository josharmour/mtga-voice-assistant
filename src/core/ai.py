import logging
from typing import Dict, List

from ..config.config_manager import UserPreferences
from ..data.data_management import ScryfallClient
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

        self.scryfall = ScryfallClient()
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
                    scryfall_client=self.scryfall,
                    **kwargs
                )
                logger.info(f"{provider.capitalize()} Advisor initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider.capitalize()} Advisor: {e}")
                self.advisor = None
        else:
            logger.error(f"Unknown model provider: {prefs.model_provider}. AI Advisor not initialized.")

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
