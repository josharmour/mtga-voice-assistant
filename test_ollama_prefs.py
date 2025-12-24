import sys
import os
import logging
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.getcwd())

# Load env
load_dotenv()

from src.config.config_manager import UserPreferences
from src.core.ai import AIAdvisor

def log(msg):
    print(msg)
    with open("ollama_diag.log", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def test_ollama_config():
    # Clear log
    with open("ollama_diag.log", "w", encoding="utf-8") as f:
        f.write("Starting diagnostics...\n")
        
    logging.basicConfig(level=logging.INFO)
    log("--- Diagnostics: User Preferences ---")
    
    try:
        prefs = UserPreferences.load()
        log(f"Provider: {prefs.model_provider}")
        log(f"Model: {prefs.current_model}")
        log(f"Ollama URL: {getattr(prefs, 'llamacpp_server_url', 'N/A')}")
    except Exception as e:
        log(f"Failed to load preferences: {e}")
        return

    log("\n--- Testing AI Advisor Initialization ---")
    try:
        advisor = AIAdvisor(prefs=prefs)
        if not advisor.advisor:
            log("AI Advisor failed to initialize internal advisor object.")
            return
        
        log(f"Internal Advisor Type: {type(advisor.advisor).__name__}")
        log(f"Internal Model Name: {getattr(advisor.advisor, 'model_name', 'Unknown')}")
        
        # Test generation
        log("\n--- Testing Simple Generation ---")
        
        # We need a valid dummy board state to avoid KeyError in prompt builder if strict
        board_state = {
            'your_life': 20,
            'opponent_life': 20,
            'current_phase': 'Phase_Main1',
            'current_turn': 1,
            'is_your_turn': True,
            'your_hand_count': 7,
            'opponent_hand_count': 7,
            'your_battlefield': [],
            'your_hand': [],
            'your_decklist': {}, # added to prevent get(your_decklist) from returning None/empty logic issues?
            'your_mana_pool': {'W': 1},
        }

        stream = advisor.get_tactical_advice_stream(board_state)
        log("Stream started...")
        count = 0
        for chunk in stream:
            count += 1
            log(f"Chunk {count}: {chunk}")
        log(f"\nStream finished. Chunks: {count}")
        
    except Exception as e:
        log(f"\nCRITICAL ERROR during test: {e}")
        import traceback
        with open("ollama_diag.log", "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)

if __name__ == "__main__":
    test_ollama_config()
