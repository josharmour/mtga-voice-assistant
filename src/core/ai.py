import logging
import datetime
from typing import Dict, List, Optional

from ..config.config_manager import UserPreferences

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
        self.advice_count = 0
        self.last_advice_time: Optional[datetime.datetime] = None

        provider = prefs.model_provider.lower()
        model = prefs.current_model

        self._init_advisor(provider, model, prefs)

    def _init_advisor(self, provider: str, model: str, prefs: UserPreferences):
        """Lazy load and initialize the specific advisor class."""
        try:
            if provider == "google":
                from .llm.google_advisor import GeminiAdvisor
                self.advisor = GeminiAdvisor(model_name=model, card_db=self.card_db, api_key=prefs.google_api_key, max_tokens=prefs.max_prompt_tokens)
            elif provider == "openai":
                from .llm.openai_advisor import OpenAIAdvisor
                self.advisor = OpenAIAdvisor(model_name=model, card_db=self.card_db, api_key=prefs.openai_api_key, max_tokens=prefs.max_prompt_tokens)
            elif provider == "anthropic":
                from .llm.anthropic_advisor import AnthropicAdvisor
                self.advisor = AnthropicAdvisor(model_name=model, card_db=self.card_db, api_key=prefs.anthropic_api_key, max_tokens=prefs.max_prompt_tokens)
            elif provider == "ollama":
                from .llm.ollama_advisor import OllamaAdvisor
                self.advisor = OllamaAdvisor(model_name=model, card_db=self.card_db, max_tokens=prefs.max_prompt_tokens)
            elif provider in ("llamacpp", "llama.cpp"):
                from .llm.llamacpp_advisor import LlamaCppAdvisor
                self.advisor = LlamaCppAdvisor(model_name=model, card_db=self.card_db, server_url=prefs.llamacpp_server_url, max_tokens=prefs.max_prompt_tokens)
            elif provider == "cli proxy":
                from .llm.proxy_advisor import CLIProxyAdvisor
                self.advisor = CLIProxyAdvisor(
                    model_name=model, 
                    card_db=self.card_db,
                    api_key=prefs.proxy_api_key,
                    base_url=prefs.proxy_url,
                    max_tokens=prefs.max_prompt_tokens
                )
            elif provider == "planeswalker":
                from .llm.planeswalker_advisor import PlaneswalkerAdvisor
                self.advisor = PlaneswalkerAdvisor(model_name=model, card_db=self.card_db)
            else:
                logger.error(f"Unknown model provider: {provider}")
                self.advisor = None
                
            if self.advisor:
                logger.info(f"{provider.capitalize()} Advisor initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider.capitalize()} Advisor: {e}")
            self.advisor = None

    def set_model(self, provider: str, model_name: str, api_key: str = None, **extra_config):
        """
        Hot-swap the AI model/provider.
        """
        provider = provider.lower()
        
        # Pass empty card_db if we don't have one stored
        card_db = getattr(self, 'card_db', None)

        try:
            if provider == "google":
                from .llm.google_advisor import GeminiAdvisor
                self.advisor = GeminiAdvisor(model_name=model_name, card_db=card_db, api_key=api_key, **extra_config)
            elif provider == "openai":
                from .llm.openai_advisor import OpenAIAdvisor
                self.advisor = OpenAIAdvisor(model_name=model_name, card_db=card_db, api_key=api_key, **extra_config)
            elif provider == "anthropic":
                from .llm.anthropic_advisor import AnthropicAdvisor
                self.advisor = AnthropicAdvisor(model_name=model_name, card_db=card_db, api_key=api_key, **extra_config)
            elif provider == "ollama":
                from .llm.ollama_advisor import OllamaAdvisor
                self.advisor = OllamaAdvisor(model_name=model_name, card_db=card_db, **extra_config)
            elif provider in ("llamacpp", "llama.cpp"):
                from .llm.llamacpp_advisor import LlamaCppAdvisor
                self.advisor = LlamaCppAdvisor(model_name=model_name, card_db=card_db, **extra_config)
            elif provider == "cli proxy":
                from .llm.proxy_advisor import CLIProxyAdvisor
                self.advisor = CLIProxyAdvisor(model_name=model_name, card_db=card_db, api_key=api_key, **extra_config)
            elif provider == "planeswalker":
                from .llm.planeswalker_advisor import PlaneswalkerAdvisor
                self.advisor = PlaneswalkerAdvisor(model_name=model_name, card_db=card_db, **extra_config)
            else:
                logger.error(f"Unknown model provider: {provider}")
                return False
                
            logger.info(f"Hot-swapped to {provider.capitalize()} Advisor with model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to hot-swap to {provider.capitalize()} Advisor: {e}")
            return False

    def get_tactical_advice(self, board_state: Dict, user_query: str = "") -> str:
        """
        Get tactical advice from the AI.
        """
        if not self.advisor:
            return "AI Advisor not initialized. Please configure a provider in the settings."
        
        advice = self.advisor.get_tactical_advice(board_state)
        
        # Track usage stats
        self.advice_count += 1
        self.last_advice_time = datetime.datetime.now()
        
        return advice

    def get_tactical_advice_stream(self, board_state: Dict, user_query: str = ""):
        """
        Get streaming tactical advice from the AI.
        Returns a generator.
        """
        if not self.advisor:
            yield "AI Advisor not initialized. Please configure a provider in the settings."
            return
        
        # Track usage stats
        self.advice_count += 1
        self.last_advice_time = datetime.datetime.now()

        # Check if the underlying advisor supports streaming (it should via BaseMTGAdvisor)
        if hasattr(self.advisor, 'get_tactical_advice_stream'):
            yield from self.advisor.get_tactical_advice_stream(board_state)
        else:
            # Fallback for advisors that don't implement stream (though Base does)
            yield self.advisor.get_tactical_advice(board_state)

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        if not self.advisor:
            return "AI Advisor not initialized. Please configure a provider in the settings."
        return self.advisor.get_draft_pick(pack_cards, current_pool)
