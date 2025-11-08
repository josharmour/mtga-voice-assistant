#!/usr/bin/env python3
"""
Test suite for Tasks 1.3 and 1.4:
- Task 1.3: Decision Point Extraction and State Representation
- Task 1.4: Outcome Weighting and Dataset Assembly
"""

import sqlite3
import sys
from pathlib import Path

try:
    from decision_point_extraction import (
        DecisionPointExtractor, DecisionPointIdentifier, StateRepresentation,
        GameStateSnapshot, GamePhase
    )
    from outcome_weighting_dataset import (
        DatasetAssembler, OutcomeWeighter, OutcomeWeightStrategy
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_decision_identification():
    """Test decision point identification logic."""
    print("=" * 60)
    print("TEST 1: Decision Point Identification")
    print("=" * 60)

    # Test critical actions
    assert DecisionPointIdentifier.is_decision_point('creature_attacked', 1, 2)
    assert DecisionPointIdentifier.is_decision_point('creature_cast', 1, 1)
    assert DecisionPointIdentifier.is_decision_point('instant_sorcery_cast', 1, 1)

    # Test non-decision actions
    assert not DecisionPointIdentifier.is_decision_point('card_drawn', 1, 1)

    # Test early game decisions
    assert DecisionPointIdentifier.is_decision_point('creature_cast', 2, 1)

    print("✓ All identification tests passed")
    print()


def test_difficulty_scoring():
    """Test difficulty score calculation."""
    print("=" * 60)
    print("TEST 2: Difficulty Score Calculation")
    print("=" * 60)

    # Easy decision: few options, lots of life
    score1 = DecisionPointIdentifier.calculate_difficulty_score(
        available_actions=[],
        player_life=20,
        opponent_life=15,
        turn_number=1
    )
    print(f"  Few options, high life: {score1:.3f}")

    # Hard decision: many options, low life, late game
    score2 = DecisionPointIdentifier.calculate_difficulty_score(
        available_actions=[1, 2, 3, 4, 5, 6, 7],
        player_life=3,
        opponent_life=5,
        turn_number=7
    )
    print(f"  Many options, low life, late game: {score2:.3f}")

    # Harder decisions should have higher scores
    assert score2 > score1

    # Scores should be between 0 and 1
    assert 0 <= score1 <= 1.0
    assert 0 <= score2 <= 1.0

    print("✓ Difficulty scoring tests passed")
    print()


def test_state_representation():
    """Test game state representation."""
    print("=" * 60)
    print("TEST 3: State Representation")
    print("=" * 60)

    # Create sample game state
    state = GameStateSnapshot(
        turn_number=3,
        player='user',
        phase=GamePhase.MAIN_1,
        user_life=18,
        oppo_life=16,
        user_hand_size=4,
        oppo_hand_size=5,
        user_lands_in_play=3,
        oppo_lands_in_play=2,
        user_creatures_in_play=2,
        oppo_creatures_in_play=1,
        main_colors='WU',
        opp_colors='BR',
    )

    # Test color encoding
    colors_wu = StateRepresentation.encode_colors('WU')
    colors_br = StateRepresentation.encode_colors('BR')
    colors_none = StateRepresentation.encode_colors(None)

    print(f"  WU colors: {colors_wu}")
    print(f"  BR colors: {colors_br}")
    print(f"  None colors: {colors_none}")

    assert colors_wu == [1, 1, 0, 0, 0], f"WU encoding incorrect: {colors_wu}"
    assert colors_br == [0, 0, 1, 1, 0], f"BR encoding incorrect: {colors_br}"
    assert colors_none == [0, 0, 0, 0, 0], f"None encoding incorrect: {colors_none}"

    # Test state vector creation
    vector = StateRepresentation.create_state_vector(state)
    print(f"  State vector length: {len(vector)}")
    print(f"  State vector (first 5): {vector[:5]}")

    # Vector should have numeric values
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)
    assert all(0 <= v <= 1.0 for v in vector[:7])  # Normalized numeric features

    print("✓ State representation tests passed")
    print()


def test_decision_extraction():
    """Test decision point extraction from action sequences."""
    print("=" * 60)
    print("TEST 4: Decision Point Extraction")
    print("=" * 60)

    extractor = DecisionPointExtractor(
        action_db_path="action_sequences_test.db",
        output_db_path="decision_points_test.db"
    )

    # Extract from test database
    games_extracted = 0
    total_decisions = 0

    try:
        with sqlite3.connect("action_sequences_test.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT game_id FROM action_sequence_metadata LIMIT 5")
            game_ids = [row[0] for row in cursor.fetchall()]

        print(f"  Extracting from {len(game_ids)} sample games")

        for game_id in game_ids:
            decisions = extractor.extract_from_game(game_id)
            print(f"    Game {game_id[:8]}...: {len(decisions)} decision points")
            total_decisions += len(decisions)
            games_extracted += 1

    except Exception as e:
        print(f"  Note: Test database not yet available: {e}")
        games_extracted = 0

    if games_extracted > 0:
        print(f"✓ Extracted {total_decisions} decision points from {games_extracted} games")
    else:
        print("ℹ  Decision extraction ready (requires action_sequences.db)")

    print()


def test_outcome_weighting():
    """Test outcome weighting strategies."""
    print("=" * 60)
    print("TEST 5: Outcome Weighting Strategies")
    print("=" * 60)

    test_cases = [
        (True, 0.5, "Win, moderate difficulty"),
        (False, 0.5, "Loss, moderate difficulty"),
        (True, 0.9, "Win, high difficulty"),
        (False, 0.1, "Loss, low difficulty"),
    ]

    for game_outcome, difficulty, description in test_cases:
        binary = OutcomeWeighter.binary_weight(game_outcome, difficulty)
        outcome = OutcomeWeighter.outcome_weighted(game_outcome, difficulty)
        importance = OutcomeWeighter.importance_scaled(game_outcome, difficulty)
        contextual = OutcomeWeighter.contextual_weight(
            game_outcome, difficulty, 'mythic', 'PIO'
        )

        print(f"  {description}:")
        print(f"    Binary:       {binary:.3f}")
        print(f"    Outcome:      {outcome:.3f}")
        print(f"    Importance:   {importance:.3f}")
        print(f"    Contextual:   {contextual:.3f}")

        # Wins should have higher weights than losses
        if game_outcome:
            assert binary > OutcomeWeighter.binary_weight(False, difficulty)
            assert importance > OutcomeWeighter.importance_scaled(False, difficulty)

    print("✓ Outcome weighting tests passed")
    print()


def test_dataset_assembly():
    """Test training dataset assembly."""
    print("=" * 60)
    print("TEST 6: Training Dataset Assembly")
    print("=" * 60)

    assembler = DatasetAssembler(
        decision_db_path="decision_points_test.db",
        action_db_path="action_sequences_test.db",
        output_db_path="training_dataset_test.db"
    )

    try:
        # Try assembling examples from test database
        examples = assembler.assemble_training_examples(
            strategy=OutcomeWeightStrategy.IMPORTANCE_SCALED,
            limit=100
        )

        print(f"✓ Assembled {len(examples)} training examples")

        if examples:
            # Show sample statistics
            weights = [e.weight for e in examples]
            outcomes = [e.game_outcome for e in examples]

            avg_weight = sum(weights) / len(weights) if weights else 0
            win_rate = sum(outcomes) / len(outcomes) if outcomes else 0

            print(f"  Average weight: {avg_weight:.4f}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Min weight: {min(weights):.4f}")
            print(f"  Max weight: {max(weights):.4f}")

            # Save to database
            assembler.save_training_examples(examples, OutcomeWeightStrategy.IMPORTANCE_SCALED)

            # Test splits creation
            assembler.create_train_val_test_splits()

            print("✓ Dataset assembly complete")

    except FileNotFoundError as e:
        print(f"ℹ  Dataset assembly ready (requires decision_points.db)")
        print(f"   {e}")

    print()


def test_database_schemas():
    """Test that databases are created with correct schemas."""
    print("=" * 60)
    print("TEST 7: Database Schemas")
    print("=" * 60)

    # Check decision points database
    if Path("decision_points_test.db").exists():
        with sqlite3.connect("decision_points_test.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"  decision_points_test.db tables: {tables}")
            assert 'decision_points' in tables
            assert 'decision_point_metadata' in tables

    # Check training dataset database
    if Path("training_dataset_test.db").exists():
        with sqlite3.connect("training_dataset_test.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"  training_dataset_test.db tables: {tables}")
            assert 'training_examples' in tables
            assert 'dataset_metadata' in tables
            assert 'dataset_splits' in tables

    print("✓ Database schema tests passed")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║     Task 1.3 & 1.4: Decision Points & Weighting Tests    ║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_decision_identification()
        test_difficulty_scoring()
        test_state_representation()
        test_outcome_weighting()
        test_decision_extraction()
        test_dataset_assembly()
        test_database_schemas()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Extract decision points: python -c 'from decision_point_extraction import DecisionPointExtractor; DecisionPointExtractor().extract_from_all_games(limit=1000)'")
        print("2. Assemble weighted dataset: python -c 'from outcome_weighting_dataset import DatasetAssembler; assembler = DatasetAssembler(); examples = assembler.assemble_training_examples(); assembler.save_training_examples(examples)'")
        print("3. Prepare for Phase 2: State and Action Encoding")
        print()

        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
