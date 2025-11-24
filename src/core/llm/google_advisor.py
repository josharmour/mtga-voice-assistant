import os
import logging
from google import genai
from google.genai import types
from typing import Dict, List, Optional
from src.data.data_management import ScryfallClient
from .prompt_builder import MTGPromptBuilder

logger = logging.getLogger(__name__)

class GeminiAdvisor:
    """
    AI Advisor powered by Google's Gemini 3 models.
    Provides tactical advice and draft recommendations.
    """
    def __init__(self, model_name: str = "gemini-3-pro-preview", scryfall_client: ScryfallClient = None, card_db=None):
        self.model_name = model_name
        self.scryfall = scryfall_client or ScryfallClient()
        self.card_db = card_db
        self.prompt_builder = MTGPromptBuilder(self.scryfall)
        self._setup_api()

    def _setup_api(self):
        # Check for GOOGLE_API_KEY first (preferred by google-genai SDK)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Fall back to GEMINI_API_KEY for backwards compatibility
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                # Set GOOGLE_API_KEY for the SDK
                os.environ["GOOGLE_API_KEY"] = api_key
        
        if not api_key:
            logger.warning("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables. AI features will be disabled.")
            self.client = None
            return
        
        # Configure with API key
        self.client = genai.Client(api_key=api_key)

    def get_tactical_advice(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Generate tactical advice based on the current board state.
        """
        if not self.client:
            return "AI Advisor is not configured (missing API key)."

        # Use shared prompt builder for consistent context
        prompt = self.prompt_builder.build_tactical_prompt(board_state, game_history)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "I'm having trouble connecting to the strategy network."

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Analyze a draft pack and suggest a pick.
        """
        if not self.client:
            return "AI Advisor not configured."

        # Use shared prompt builder for consistent draft prompts
        prompt = self.prompt_builder.build_draft_prompt(pack_cards, current_pool)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Unable to analyze draft pack."

