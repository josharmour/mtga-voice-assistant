import sys
import os
import logging
import datetime
from src.core.mtga import GameStateManager
from src.core.advice_triggers import AdviceTriggerManager, TriggerEvent, TriggerType
from src.core.app import CLIVoiceAdvisor

# Mock classes to simulate environment without GUI or full game
class MockAIAdvisor:
    def __init__(self):
        # Create a proper instance (not just a class) with model_name attribute
        self.advisor = type('MockAdvisorInner', (object,), {'model_name': 'MockModel'})()

    def get_tactical_advice_stream(self, board_state_dict):
        print(f"DEBUG: get_tactical_advice_stream called with board state keys: {list(board_state_dict.keys())}")
        yield "This is a test advice sentence. "
        yield "It simulates the AI response."

    def set_model(self, *args, **kwargs):
        pass

class MockPreferences:
    volume = 100
    current_voice = "test_voice"

def investigate_advice_flow():
    logging.basicConfig(level=logging.DEBUG)
    
    print("Setting up advice investigation...")
    
    # 1. Initialize Advisor
    try:
        advisor = CLIVoiceAdvisor(use_gui=False)
        advisor.ai_advisor = MockAIAdvisor() # Inject mock AI
        advisor.prefs = MockPreferences()
        advisor.tts = None # Disable TTS for this test
    except Exception as e:
        print(f"Failed to init advisor: {e}")
        return

    # 2. Simulate a Board State
    from src.core.mtga import BoardState
    board_state = BoardState(
        your_seat_id=1,
        opponent_seat_id=2,
        your_life=20,
        opponent_life=20,
        current_turn=3,
        current_phase="Phase_Main1",
        is_your_turn=True,
        has_priority=True,
        your_hand_count=5, # Ensure valid integers
        opponent_hand_count=5,
        your_hand=[], 
        your_battlefield=[],
        your_graveyard=[],
        your_exile=[],
        opponent_battlefield=[],
        opponent_graveyard=[],
        opponent_exile=[],
        stack=[]
    )

    # 3. Trigger Advice manually
    print("\n--- Triggering Manual Advice ---")
    trigger_event = TriggerEvent(
        trigger_type=TriggerType.TURN_START,
        turn=3,
        phase="Phase_Main1",
        is_your_turn=True
    )
    
    advisor._get_advice_for_trigger(board_state, trigger_event)
    
    # Wait for thread to complete (since _get_advice_for_trigger starts a thread)
    import time
    time.sleep(2)
    print("\n--- Investigation Complete ---")

if __name__ == "__main__":
    investigate_advice_flow()
