
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_management import ScryfallClient, CardStatsDB
from src.core.gemini_advisor import GeminiAdvisor
from src.core.draft_advisor import DraftAdvisor
from src.core.ai import AIAdvisor

logging.basicConfig(level=logging.INFO)

def test_initialization():
    print("Testing ScryfallClient...")
    scryfall = ScryfallClient()
    # Test compatibility methods
    assert hasattr(scryfall, 'get_card_name')
    assert hasattr(scryfall, 'get_card_data')
    print("✓ ScryfallClient initialized and has compatibility methods")

    print("Testing CardStatsDB...")
    stats_db = CardStatsDB()
    print("✓ CardStatsDB initialized")

    print("Testing GeminiAdvisor...")
    # Mock API key if not present to avoid error during init check (though init usually just sets it up)
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "dummy_key"
    
    gemini = GeminiAdvisor(scryfall_client=scryfall)
    print("✓ GeminiAdvisor initialized")

    print("Testing AIAdvisor wrapper...")
    ai_advisor = AIAdvisor(model="gemini-1.5-flash")
    print("✓ AIAdvisor initialized")

    print("Testing DraftAdvisor...")
    draft_advisor = DraftAdvisor(scryfall, stats_db, ai_advisor)
    print("✓ DraftAdvisor initialized")

    print("\nAll components initialized successfully!")

if __name__ == "__main__":
    test_initialization()
