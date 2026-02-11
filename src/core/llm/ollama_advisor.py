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

    def _get_available_models(self):
        """Fetch list of locally available Ollama models."""
        try:
            response = self.client.list()
            return [m['name'] for m in response.get('models', [])]
        except Exception:
            return []

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to local Ollama instance."""
        if not self.model_name:
            # Try to auto-detect an available model instead of hardcoding
            available = self._get_available_models()
            if available:
                self.model_name = available[0]
                logger.info(f"No model specified. Auto-selected '{self.model_name}' from available Ollama models.")
            else:
                logger.error("No model name provided and no Ollama models found locally.")
                return ("Error: No Ollama model selected and none found locally. "
                        "Please run 'ollama pull <model>' (e.g. 'ollama pull gemma3n') "
                        "then select it in Settings.")

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
            error_str = str(e).lower()
            logger.error(f"Ollama API Error: {e}")
            if "validation error" in error_str and "string_too_short" in error_str:
                return "Error: Ollama model name is empty. Please select a model in Settings."
            if "not found" in error_str:
                available = self._get_available_models()
                if available:
                    models_str = ", ".join(available[:5])
                    return (f"Error: Model '{self.model_name}' not found. "
                            f"Available models: {models_str}. "
                            f"Select one in Settings or run 'ollama pull {self.model_name}'.")
                return (f"Error: Model '{self.model_name}' not found. "
                        f"Run 'ollama pull {self.model_name}' in your terminal to download it.")
            return f"Error getting tactical advice from Ollama: {e}"
