
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.advice_triggers import AdviceTriggerManager

def test_trigger_patterns():
    manager = AdviceTriggerManager()
    
    test_cases = [
        ("Step_DeclareAttack", "attackers", True),
        ("Phase_Combat_Declare_Attackers", "attackers", True),
        ("Step_DeclareBlock", "blockers", True),
        ("Phase_Combat_Declare_Blockers", "blockers", True),
        ("Step_BeginCombat", "combat_begin", True),
        ("Phase_Combat_Begin", "combat_begin", True),
        ("Phase_Main1", "main1", True),
        ("Phase_Main2", "main2", True),
        ("Phase_End", "end", True),
        ("Phase_Upkeep", "upkeep", True),
        ("Phase_Draw", "draw", True),
        ("Step_CombatDamage", "damage", True),
        ("Phase_Combat_Damage", "damage", True),
        ("Phase_Combat_End", "combat_end", True),
        ("Step_Cleanup", "cleanup", True),
        ("Phase_Unknown", "main1", False)
    ]
    
    print("Verifying Phase Patterns:")
    print("-" * 60)
    print(f"{'Log Phase':<35} | {'Key':<12} | {'Expected':<8} | {'Result':<8}")
    print("-" * 60)
    
    all_passed = True
    for log_phase, key, expected in test_cases:
        result = manager._is_phase(log_phase, key)
        status = "PASS" if result == expected else "FAIL"
        if not result == expected:
            all_passed = False
            
        print(f"{log_phase:<35} | {key:<12} | {str(expected):<8} | {str(result):<8} [{status}]")
        
    print("-" * 60)
    if all_passed:
        print("✅ All patterns verified successfully!")
    else:
        print("❌ Some patterns failed verification.")

if __name__ == "__main__":
    test_trigger_patterns()
