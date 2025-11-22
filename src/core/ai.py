import logging
from typing import Dict, List, Optional
from .gemini_advisor import GeminiAdvisor
from ..data.data_management import ScryfallClient

logger = logging.getLogger(__name__)

class AIAdvisor:
    """
    Main AI Advisor interface.
    Delegates to GeminiAdvisor for intelligence.
    Maintains compatibility with existing app structure.
    """
    def __init__(self, card_db=None, model: str = "gemini-1.5-flash"):
        # card_db is passed from app.py (ArenaCardDatabase).
        # We pass it to GeminiAdvisor to use as a robust fallback for card data.
        self.scryfall = ScryfallClient()
        self.advisor = GeminiAdvisor(model_name=model, scryfall_client=self.scryfall, card_db=card_db)
        logger.info(f"AI Advisor initialized with model: {model}")

    def get_tactical_advice(self, board_state: Dict, user_query: str = "") -> str:
        """
        Get tactical advice from the AI.
        """
        # We can pass user_query if we want to support custom questions later
        # For now, we just pass the board state
        return self.advisor.get_tactical_advice(board_state)

    def get_draft_pick(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Get draft pick recommendation.
        """
        return self.advisor.get_draft_pick(pack_cards, current_pool)
