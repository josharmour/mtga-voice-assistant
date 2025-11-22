import logging
from typing import Dict, List, Optional
import openai

logger = logging.getLogger(__name__)

class OpenAIAdvisor:
    """
    Advisor powered by OpenAI's models.
    """
    def __init__(self, model_name: str = "gpt-4-turbo", api_key: str = None, card_db=None, scryfall_client=None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.card_db = card_db
        self.scryfall_client = scryfall_client
        logger.info(f"OpenAI Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict) -> str:
        """
        Get tactical advice from the AI.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering tactical advisor."},
                    {"role": "user", "content": f"Here is the board state:\n{board_state}\n\nWhat is the best course of action?"}
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering draft advisor."},
                    {"role": "user", "content": f"Here is the current pack:\n{pack_cards}\n\nHere is my current card pool:\n{current_pool}\n\nWhich card should I pick?"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting draft pick from OpenAI: {e}")
            return "Error getting draft pick from OpenAI."
