"""Google Gemini LLM advisor."""
import logging
import os

from google import genai
from google.genai import types

from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)

# Available Gemini models (updated December 2025)
# See: https://ai.google.dev/gemini-api/docs/models
GEMINI_MODELS = [
    # Gemini 3.0 (Latest - November 2025)
    "gemini-3-pro-preview",

    # Gemini 2.5 (Current production)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",

    # Gemini 2.0
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",

    # Legacy (will be deprecated)
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


class GeminiAdvisor(BaseMTGAdvisor):
    """
    AI Advisor powered by Google's Gemini models.
    Provides tactical advice and draft recommendations.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", card_db=None, **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        self._setup_api()

    def _setup_api(self):
        """Configure the Gemini API client."""
        # Check for GOOGLE_API_KEY first (preferred by google-genai SDK)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Fall back to GEMINI_API_KEY for backwards compatibility
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                # Set GOOGLE_API_KEY for the SDK
                os.environ["GOOGLE_API_KEY"] = api_key

        if not api_key:
            logger.warning("GOOGLE_API_KEY or GEMINI_API_KEY not found. AI features will be disabled.")
            self.client = None
            return

        self.client = genai.Client(api_key=api_key)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Google Gemini."""
        if not self.client:
            return "AI Advisor is not configured (missing API key)."

        # Combine system prompt with user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return response.text.strip()

    def is_available(self) -> bool:
        """Check if the Gemini API is configured."""
        return self.client is not None
