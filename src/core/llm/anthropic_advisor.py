import logging
from typing import Dict, List, Optional
import anthropic
from src.data.data_management import ScryfallClient
from .prompt_builder import MTGPromptBuilder

logger = logging.getLogger(__name__)

class AnthropicAdvisor:
    """
    Advisor powered by Anthropic's models.
    """
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None, card_db=None, scryfall_client=None):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.card_db = card_db
        self.scryfall_client = scryfall_client or ScryfallClient()
        self.prompt_builder = MTGPromptBuilder(self.scryfall_client, arena_db=card_db)
        logger.info(f"Anthropic Advisor initialized with model: {self.model_name}")

    def get_tactical_advice(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Get tactical advice from the AI with rich context.
        """
        try:
            # Use shared prompt builder for rich context with card text
            prompt = self.prompt_builder.build_tactical_prompt(board_state, game_history)

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
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
            # Use shared prompt builder for consistent draft prompts
            prompt = self.prompt_builder.build_draft_prompt(pack_cards, current_pool)

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error getting draft pick from Anthropic: {e}")
            return "Error getting draft pick from Anthropic."
