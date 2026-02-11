import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure root is in path
sys.path.append(os.getcwd())

try:
    from src.core.llm.planeswalker_advisor import PlaneswalkerAdvisor
except ImportError as e:
    print(f"ImportError: {e}")
    # Try adding src to path explicitly if it fails
    sys.path.append(os.path.join(os.getcwd(), "src"))
    from src.core.llm.planeswalker_advisor import PlaneswalkerAdvisor

if __name__ == "__main__":
    print("Initializing Advisor in Main Thread...")
    advisor = PlaneswalkerAdvisor()
    
    import threading
    print("Running Query in Worker Thread...")
    # We define a wrapper because we need to call the method on the ALREADY INITIALIZED instance
    def worker():
        # Simulate structured card objects (like what the app sends)
        board_state = {
            'current_turn': 1,
            'current_phase': 'Phase_Beginning',
            'your_hand': [{'name': 'Island', 'id': 1}, {'name': 'Storm Crow', 'id': 2}],
            'your_lands': [{'name': 'Mountain', 'id': 3}],
            'your_creatures': [{'name': 'Grizzly Bears', 'id': 4}],
            'opponent_creatures': [{'name': 'Llanowar Elves', 'id': 5}],
            'opponent_life': 20,
            'your_life': 20,
            'user_query': "What should I do?",
        }
        try:
            for chunk in advisor.get_tactical_advice_stream(board_state):
                print(f"Chunk: {chunk}")
        except Exception as e:
            print(f"Caught Exception: {e}")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
