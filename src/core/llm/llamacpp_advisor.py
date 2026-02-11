"""Llama.cpp Server advisor."""
import logging
import requests
from typing import Optional

from .base import BaseMTGAdvisor

logger = logging.getLogger(__name__)


class LlamaCppAdvisor(BaseMTGAdvisor):
    """
    Advisor powered by a local llama.cpp server.
    See: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
    """

    def __init__(self, model_name: str = "default", card_db=None, server_url: str = "http://localhost:8080", **kwargs):
        super().__init__(model_name, card_db, **kwargs)
        self.server_url = server_url.rstrip('/')
        self.client = True  # Mark as initialized (no client object needed for requests)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to local llama.cpp server."""
        try:
            # Llama.cpp server completion endpoint
            url = f"{self.server_url}/completion"
            
            # Construct prompt based on standard chat template format (or simple concatenation)
            # Most llama.cpp server setups handle raw text better unless using /chat/completions
            # Here we use a simple format, but /chat/completions is preferred if available
            
            # Try /v1/chat/completions (OpenAI compatible) first as it handles templates better
            chat_url = f"{self.server_url}/v1/chat/completions"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            try:
                response = requests.post(chat_url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
            except requests.exceptions.RequestException:
                # Fallback to legacy /completion endpoint if /v1/chat/completions fails
                pass

            # Fallback: Legacy /completion
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
            payload = {
                "prompt": full_prompt,
                "temperature": 0.7,
                "n_predict": 500
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('content', '')

        except Exception as e:
            logger.error(f"Llama.cpp API Error: {e}")
            return f"Error getting advice from Llama.cpp: {e}"

    def is_available(self) -> bool:
        """Check if the Llama.cpp server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
