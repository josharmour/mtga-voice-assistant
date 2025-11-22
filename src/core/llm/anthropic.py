import logging
from typing import Dict, List, Optional
import anthropic

logger = logging.getLogger(__name__)

class AnthropicAdvisor:
    """
    Advisor powered by Anthropic's models.
    """
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None, card_db=None, scryfall_client=None):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.card_db = card_db
        self.scryfall_client = scryfall_client
        logger.info(f"Anthropic Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict) -> str:
        """
        Get tactical advice from the AI.
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a Magic: The Gathering tactical advisor. Here is the board state:\n{board_state}\n\nWhat is the best course of action?"
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error getting tactical advice from Anthropic: {e}")
            return "Error getting tactical advice from Anthropic."

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a Magic: The Gathering draft advisor. Here is the current pack:\n{pack_cards}\n\nHere is my current card pool:\n{current_pool}\n\nWhich card should I pick?"
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error getting draft pick from Anthropic: {e}")
            return "Error getting draft pick from Anthropic."
