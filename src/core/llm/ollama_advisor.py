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
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']
