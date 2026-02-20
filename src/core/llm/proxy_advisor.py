"""CLI Proxy LLM advisor (OpenAI-compatible)."""
import logging
import os
import openai
from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)

class CLIProxyAdvisor(BaseMTGAdvisor):
    """Advisor powered by a local CLI Proxy (OpenAI-compatible)."""

    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", api_key: str = "proxy", card_db=None, **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        
        # Use provided URL or default to local proxy
        base_url = kwargs.get("base_url") or os.environ.get("PROXY_BASE_URL") or "http://127.0.0.1:8080/v1"
        api_key = api_key or os.environ.get("PROXY_API_KEY") or "proxy"
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Proxy."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
