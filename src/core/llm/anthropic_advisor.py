"""Anthropic LLM advisor for Claude models."""
import logging

import anthropic

from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)


class AnthropicAdvisor(BaseMTGAdvisor):
    """Advisor powered by Anthropic's Claude models."""

    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None, card_db=None, **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Anthropic."""
        # Anthropic uses system parameter separately, not in messages
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.content[0].text
