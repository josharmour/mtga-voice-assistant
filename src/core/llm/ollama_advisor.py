import logging
from typing import Dict, List, Optional
import ollama
from .prompt_builder import MTGPromptBuilder

logger = logging.getLogger(__name__)

class OllamaAdvisor:
    """
    Advisor powered by a local Ollama instance.
    """
    def __init__(self, model_name: str = "llama2", card_db=None, **kwargs):
        self.model_name = model_name
        self.client = ollama.Client()
        self.card_db = card_db
        self.prompt_builder = MTGPromptBuilder(arena_db=card_db)
        logger.info(f"Ollama Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Get tactical advice from the AI with rich context.
        """
        try:
            # Use shared prompt builder for rich context with card text
            prompt = self.prompt_builder.build_tactical_prompt(board_state, game_history)

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering tactical advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error getting tactical advice from Ollama: {e}")
            return "Error getting tactical advice from Ollama."

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        try:
            # Use shared prompt builder for consistent draft prompts
            prompt = self.prompt_builder.build_draft_prompt(pack_cards, current_pool)

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering draft advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error getting draft pick from Ollama: {e}")
            return "Error getting draft pick from Ollama."
