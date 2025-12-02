"""OpenAI LLM advisor for GPT models."""
import logging

import openai

from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)


class OpenAIAdvisor(BaseMTGAdvisor):
    """Advisor powered by OpenAI's models."""

    def __init__(self, model_name: str = "gpt-4-turbo", api_key: str = None, card_db=None, **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
