import sys
import os
sys.path.append(os.getcwd())
import logging
from src.core.mtga import GameStateManager
from src.data.arena_cards import ArenaCardDatabase

# Mock the Card Database
class MockCardDB:
    def get_card_name(self, grp_id): return "Mock Card"
    def get_card_data(self, grp_id): return {}

def test_missing_turn_number_crash():
    """
    Reproduction test for Bug #52:
    Simulates a log line where 'turn_info' is present but 'turnNumber' is missing or None.
    This causes current_turn to become None, leading to crashes in downstream logic
    that expects an integer (e.g. > 0 comparisons).
    """
    print("Running reproduction test for Bug #52...")
    
    # Setup
    gsm = GameStateManager(MockCardDB())
    
    # 1. Establish valid state (Turn 1)
    gsm.scanner.current_turn = 1
    print(f"Initial Turn: {gsm.scanner.current_turn}")
    
    # 2. Simulate bad log message (TurnInfo without turnNumber)
    # This matches the hypothesis that some logs have incomplete data
    bad_log_line = '{"greToClientEvent": {"greToClientMessages": [{"type": "GREMessageType_GameStateMessage", "gameStateMessage": {"turnInfo": {"phase": "Phase_Main1"}}}]}}'
    
    try:
        gsm.parse_log_line(bad_log_line)
        print(f"Turn after bad log: {gsm.scanner.current_turn}")
        
        # 3. Simulate access that would crash if None
        # Logic in advice_triggers often checks: if turn > 0:
        if gsm.scanner.current_turn is None:
            print("FAILURE: current_turn is None!")
        elif gsm.scanner.current_turn > 0:
            print("SUCCESS: current_turn is acceptable integer")
        else:
            print(f"WARNING: current_turn is {gsm.scanner.current_turn}")
            
    except Exception as e:
        print(f"CRASH DETECTED: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    test_missing_turn_number_crash()
