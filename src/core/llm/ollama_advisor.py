import logging
from typing import Dict, List, Optional
import ollama

logger = logging.getLogger(__name__)

class OllamaAdvisor:
    """
    Advisor powered by a local Ollama instance.
    """
    def __init__(self, model_name: str = "llama2", card_db=None, scryfall_client=None):
        self.model_name = model_name
        self.client = ollama.Client()
        self.card_db = card_db
        self.scryfall_client = scryfall_client
        logger.info(f"Ollama Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict) -> str:
        """
        Get tactical advice from the AI.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering tactical advisor."},
                    {"role": "user", "content": f"Here is the board state:\n{board_state}\n\nWhat is the best course of action?"}
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
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Magic: The Gathering draft advisor."},
                    {"role": "user", "content": f"Here is the current pack:\n{pack_cards}\n\nHere is my current card pool:\n{current_pool}\n\nWhich card should I pick?"}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error getting draft pick from Ollama: {e}")
            return "Error getting draft pick from Ollama."
