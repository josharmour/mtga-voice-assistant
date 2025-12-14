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
        self._setup_api(kwargs.get("api_key"))

    def _setup_api(self, api_key: str = None):
        """Configure the Gemini API client."""
        # Use provided key, otherwise check environment
        if not api_key:
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

        # Explicitly configure generation to avoid accidental "thinking" defaults
        # on models that don't support it.
        config = types.GenerateContentConfig(
            temperature=0.7,
            candidate_count=1
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=config
        )
        if not response.text:
            return ""
        return response.text.strip()

    def _call_api_stream(self, system_prompt: str, user_prompt: str):
        """Make streaming API call to Google Gemini."""
        if not self.client:
            yield "AI Advisor is not configured (missing API key)."
            return

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        config = types.GenerateContentConfig(
            temperature=0.7,
            candidate_count=1
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield f" [Error: {str(e)}]"

    def is_available(self) -> bool:
        """Check if the Gemini API is configured."""
        return self.client is not None
