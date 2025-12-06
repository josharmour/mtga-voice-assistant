"""Ollama LLM advisor for local model inference."""
import logging

import ollama

from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)


class OllamaAdvisor(BaseMTGAdvisor):
    """Advisor powered by a local Ollama instance."""

    def __init__(self, model_name: str = "llama2", card_db=None, **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        self.client = ollama.Client()

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to local Ollama instance."""
        if not self.model_name:
            logger.warning("No model name provided for Ollama. Defaulting to 'llama3'.")
            self.model_name = "llama3"

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            if "validation error" in str(e) and "string_too_short" in str(e):
                return "Error: Ollama model name is empty. Please select a model in Settings."
            return f"Error getting tactical advice from Ollama: {e}"
