import logging
from typing import Dict, List, Optional
import openai
from src.data.data_management import ScryfallClient
from .prompt_builder import MTGPromptBuilder

logger = logging.getLogger(__name__)

class OpenAIAdvisor:
    """
    Advisor powered by OpenAI's models.
    """
    def __init__(self, model_name: str = "gpt-4-turbo", api_key: str = None, card_db=None, scryfall_client=None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.card_db = card_db
        self.scryfall_client = scryfall_client or ScryfallClient()
        self.prompt_builder = MTGPromptBuilder(self.scryfall_client)
        logger.info(f"OpenAI Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Get tactical advice from the AI with rich context.
        """
        try:
            # Use shared prompt builder for rich context with card text
            prompt = self.prompt_builder.build_tactical_prompt(board_state, game_history)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering tactical advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting tactical advice from OpenAI: {e}")
            return "Error getting tactical advice from OpenAI."

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        try:
            # Use shared prompt builder for consistent draft prompts
            prompt = self.prompt_builder.build_draft_prompt(pack_cards, current_pool)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering draft advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting draft pick from OpenAI: {e}")
            return "Error getting draft pick from OpenAI."
