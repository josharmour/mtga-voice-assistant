#!/usr/bin/env python3
"""
CQL Red Test Runner

This script demonstrates that the Conservative Q-Learning tests initially fail
(Red phase of Red-Green-Refactor) and will drive implementation improvements.

Usage:
    python3 run_cql_red_tests.py

This will show which tests fail and why, guiding the implementation work needed.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_red_test(test_name, test_func):
    """Run a single red test and report results."""
    print(f"\n{'='*60}")
    print(f"🔴 RED TEST: {test_name}")
    print(f"{'='*60}")

    try:
        test_func()
        print(f"❌ UNEXPECTED SUCCESS: {test_name} passed (should have failed)")
        return False
    except Exception as e:
        print(f"✅ EXPECTED FAILURE: {test_name}")
        print(f"   Error: {e}")
        return True

def test_config_custom_parameters():
    """Test that config should accept custom parameters."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig

    config = ConservativeQLearningConfig(
        state_dim=282,
        action_dim=16,
        cql_alpha=10.0,
        learning_rate=5e-4
    )

    # These assertions should drive implementation
    assert config.state_dim == 282
    assert config.action_dim == 16
    assert config.cql_alpha == 10.0

def test_config_validation():
    """Test that config should validate constitutional requirements."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig

    # Should raise ValueError for constitutional violations
    config = ConservativeQLearningConfig(max_inference_time_ms=150.0)
    assert config.max_inference_time_ms <= 100.0

def test_cql_prepare_batch_method():
    """Test that CQL agent should have _prepare_batch method."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig, ConservativeQLearning, RLState, RLAction, RLTransition
    import torch

    config = ConservativeQLearningConfig()
    agent = ConservativeQLearning(state_dim=282, action_dim=16, config=config)

    # Create test transitions
    state = RLState(
        state_vector=torch.randn(282),
        state_metadata={},
        timestamp=time.time()
    )
    action = RLAction(
        action_id=0,
        action_type="test",
        action_parameters={},
        confidence=0.8
    )
    transition = RLTransition(
        state=state,
        action=action,
        reward=1.0,
        next_state=state,
        done=False
    )

    # This method should exist and work
    batch = agent._prepare_batch([transition])
    assert 'states' in batch
    assert 'actions' in batch
    assert 'rewards' in batch

def test_cql_conservative_loss_method():
    """Test that CQL agent should have _calculate_cql_loss method."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig, ConservativeQLearning
    import torch

    config = ConservativeQLearningConfig()
    agent = ConservativeQLearning(state_dim=282, action_dim=16, config=config)

    states = torch.randn(16, 282)
    q_values = torch.randn(16, 16)

    # This method should exist and compute conservative loss
    cql_loss = agent._calculate_cql_loss(states, q_values)
    assert isinstance(cql_loss, torch.Tensor)
    assert cql_loss >= 0

def test_cql_constitutional_compliance():
    """Test that CQL agent should validate constitutional compliance."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig, ConservativeQLearning
    import torch

    config = ConservativeQLearningConfig(
        max_inference_time_ms=150.0,  # Violates constitutional requirement
        enable_explainability=False,  # Violates requirement
        enable_performance_monitoring=False  # Violates requirement
    )

    # Should raise error for constitutional violations
    agent = ConservativeQLearning(state_dim=282, action_dim=16, config=config)

    # Should detect constitutional violations
    compliance = agent.validate_constitutional_compliance()
    assert compliance['compliant'] == False
    assert len(compliance['violations']) > 0

def test_cql_performance_monitoring():
    """Test that CQL agent should monitor performance requirements."""
    from src.rl.algorithms.cql import ConservativeQLearningConfig, ConservativeQLearning, RLState
    import torch
    import time

    config = ConservativeQLearningConfig(
        max_inference_time_ms=100.0,  # Constitutional requirement
        enable_performance_monitoring=True
    )

    agent = ConservativeQLearning(state_dim=282, action_dim=16, config=config)

    # Perform action selection to trigger performance monitoring
    state = RLState(
        state_vector=torch.randn(282),
        state_metadata={},
        timestamp=time.time()
    )

    action = agent.select_action(state)

    # Should track performance metrics
    stats = agent.get_performance_stats()
    assert 'avg_inference_time_ms' in stats
    assert 'constitutional_compliance' in stats

def main():
    """Run all red tests to demonstrate initial failures."""
    print("🔴 CONSERVATIVE Q-LEARNING RED TEST SUITE")
    print("=" * 80)
    print("This suite demonstrates tests that initially FAIL (Red phase)")
    print("These failures will drive the implementation improvements needed.")
    print("=" * 80)

    # List of tests that should fail initially
    red_tests = [
        ("Config should accept custom parameters", test_config_custom_parameters),
        ("Config should validate constitutional requirements", test_config_validation),
        ("CQL should have _prepare_batch method", test_cql_prepare_batch_method),
        ("CQL should have _calculate_cql_loss method", test_cql_conservative_loss_method),
        ("CQL should validate constitutional compliance", test_cql_constitutional_compliance),
        ("CQL should monitor performance requirements", test_cql_performance_monitoring),
    ]

    failed_tests = 0
    passed_tests = 0

    for test_name, test_func in red_tests:
        try:
            if run_red_test(test_name, test_func):
                failed_tests += 1  # Expected failure (good for red phase)
            else:
                passed_tests += 1  # Unexpected success
        except Exception as e:
            print(f"❌ TEST ERROR in {test_name}: {e}")
            traceback.print_exc()
            failed_tests += 1

    print(f"\n{'='*80}")
    print("📊 RED TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Expected failures (RED phase): {failed_tests}")
    print(f"Unexpected successes: {passed_tests}")

    if failed_tests > 0:
        print("\n✅ GOOD: Tests are failing as expected (Red phase)")
        print("   These failures will drive implementation improvements.")
        print("\n📋 IMPLEMENTATION TASKS NEEDED:")
        print("   1. Add __init__ parameters to ConservativeQLearningConfig")
        print("   2. Add constitutional validation to config")
        print("   3. Implement _prepare_batch method")
        print("   4. Implement _calculate_cql_loss method")
        print("   5. Implement constitutional compliance validation")
        print("   6. Implement performance monitoring")
        print("   7. Add proper error handling and graceful degradation")
    else:
        print("\n⚠️  WARNING: All tests passed - check if implementation already exists")

    print(f"\n🎯 NEXT STEPS:")
    print("   1. Fix failing tests one by one (Green phase)")
    print("   2. Refactor code for better structure (Refactor phase)")
    print("   3. Add comprehensive integration tests")
    print("   4. Validate with real 17Lands data")

if __name__ == "__main__":
    main()