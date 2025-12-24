import sys
import os
import json
import logging
import traceback

# Setup clean environment
sys.path.append(os.getcwd())

# Mock Card DB
class MockCardDB:
    def get_card_name(self, grp_id): return f"Card_{grp_id}"
    def get_card_data(self, grp_id): return {"color_identity": ["W"]}

from src.core.mtga import GameStateManager, LogFollower
from src.core.advice_triggers import AdviceTriggerManager

def run_reproduction():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Initializing components...")
    card_db = MockCardDB()
    gsm = GameStateManager(card_db)
    trigger_mgr = AdviceTriggerManager()
    
    # Callback to simulate app.py behavior
    def process_log_line(line):
        try:
            # 1. Parse line
            state_changed = gsm.parse_log_line(line)
            
            # 2. Get board state
            # This triggers _build_board_state which might be where the error is
            board_state = gsm.get_current_board_state()
            
            if board_state:
                # 3. Check triggers
                # This triggers check_triggers which does comparisons
                trigger_mgr.check_triggers(board_state)
                
        except Exception:
            # Print full traceback on error
            traceback.print_exc()
            raise

    # Log lines from bug report (with slight formatting fixes if needed)
    # I'm pasteing line 128 which is the huge GameStateMessage.
    # Note: I need to handle the prefix [UnityCrossThreadLogger]...
    
    log_lines = [
        # Line 128 content (cleaned up prefix in the list)
        r'{ "transactionId": "bbceb7f6-f557-4c0f-b753-47d51bc9c174", "requestId": 346, "timestamp": "1765524690835", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_GameStateMessage", "systemSeatIds": [ 1 ], "msgId": 447, "gameStateId": 308, "gameStateMessage": { "type": "GameStateType_Diff", "gameStateId": 308, "gameObjects": [ { "instanceId": 231, "grpId": 97496, "type": "GameObjectType_Card", "zoneId": 28, "visibility": "Visibility_Public", "ownerSeatId": 1, "controllerSeatId": 1, "cardTypes": [ "CardType_Creature" ], "subtypes": [ "SubType_Dragon", "SubType_Insect" ], "color": [ "CardColor_Blue", "CardColor_Red" ], "power": { "value": 2 }, "toughness": { "value": 3 }, "isTapped": true, "attackState": "AttackState_Attacking", "blockState": "BlockState_Unblocked", "attackInfo": { "targetId": 2 }, "name": 1071068, "abilities": [ 8, 143868, 192814, 192815 ], "overlayGrpId": 97496 }, { "instanceId": 244, "grpId": 97315, "type": "GameObjectType_Card", "zoneId": 28, "visibility": "Visibility_Public", "ownerSeatId": 1, "controllerSeatId": 1, "cardTypes": [ "CardType_Creature" ], "subtypes": [ "SubType_Human", "SubType_Soldier", "SubType_Ally" ], "color": [ "CardColor_White" ], "power": { "value": 3 }, "toughness": { "value": 3 }, "isTapped": true, "attackState": "AttackState_Attacking", "blockState": "BlockState_Unblocked", "attackInfo": { "targetId": 2 }, "name": 1070429, "abilities": [ 153171 ], "overlayGrpId": 97315 } ], "prevGameStateId": 307, "timers": [ { "timerId": 11, "type": "TimerType_NonActivePlayer", "durationSec": 58, "behavior": "TimerBehavior_TakeControl", "warningThresholdSec": 30, "elapsedMs": 649 }, { "timerId": 13, "type": "TimerType_Inactivity", "durationSec": 150, "behavior": "TimerBehavior_Timeout", "warningThresholdSec": 30 }, { "timerId": 14, "type": "TimerType_MatchClock", "durationSec": 1800, "elapsedSec": 717, "behavior": "TimerBehavior_Timeout", "warningThresholdSec": 30, "elapsedMs": 717654 } ], "update": "GameStateUpdate_SendHiFi", "actions": [] } } ] } }'
    ]

    print("Setting up game state...")
    # Manually populate players to ensure BoardState is built
    from src.core.mtga import PlayerState
    gsm.scanner.players[1] = PlayerState(seat_id=1, life_total=20, hand_count=5)
    gsm.scanner.players[2] = PlayerState(seat_id=2, life_total=20, hand_count=5)
    gsm.scanner.local_player_seat_id = 1
    gsm.scanner.current_turn = 1
    gsm.scanner.current_phase = "Phase_Main1"

    print("Feeding lines...")
    for line in log_lines:
        print(f"Processing line: {line[:50]}...")
        process_log_line(line)
    print("Done feeding lines.")

if __name__ == "__main__":
    run_reproduction()
