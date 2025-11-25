"""
Unit tests for domain models.

These tests demonstrate the value of separating domain logic from presentation:
- Pure domain logic can be tested without UI dependencies
- Tests are fast (no I/O, no mocking needed)
- Tests are clear and focused on business rules
- Tests serve as documentation of domain behavior

Run with: pytest src/core/domain/test_domain_models.py
"""

import pytest
from .game_state import (
    Phase,
    CardIdentity,
    Permanent,
    ZoneCollection,
    PlayerGameState,
    GameState,
    CombatState,
    TurnHistory,
)


class TestPhase:
    """Test Phase enum and its domain logic."""

    def test_phase_enum_values(self):
        """Test that all phases are defined."""
        assert Phase.UNTAP is not None
        assert Phase.MAIN_1 is not None
        assert Phase.COMBAT_ATTACKERS is not None

    def test_from_arena_string(self):
        """Test conversion from Arena phase strings."""
        assert Phase.from_arena_string("Phase_Main1") == Phase.MAIN_1
        assert Phase.from_arena_string("Phase_Combat_Attackers") == Phase.COMBAT_ATTACKERS
        assert Phase.from_arena_string("Invalid") == Phase.UNKNOWN

    def test_is_combat_phase(self):
        """Test combat phase detection."""
        assert Phase.COMBAT_BEGIN.is_combat_phase is True
        assert Phase.COMBAT_ATTACKERS.is_combat_phase is True
        assert Phase.COMBAT_DAMAGE.is_combat_phase is True
        assert Phase.MAIN_1.is_combat_phase is False
        assert Phase.UPKEEP.is_combat_phase is False

    def test_is_main_phase(self):
        """Test main phase detection."""
        assert Phase.MAIN_1.is_main_phase is True
        assert Phase.MAIN_2.is_main_phase is True
        assert Phase.COMBAT_ATTACKERS.is_main_phase is False
        assert Phase.UPKEEP.is_main_phase is False


class TestCardIdentity:
    """Test CardIdentity value object."""

    def test_card_identity_creation(self):
        """Test basic card identity creation."""
        card = CardIdentity(grp_id=123, instance_id=456, name="Lightning Bolt")
        assert card.grp_id == 123
        assert card.instance_id == 456
        assert card.name == "Lightning Bolt"

    def test_card_identity_immutable(self):
        """Test that CardIdentity is immutable."""
        card = CardIdentity(grp_id=123, instance_id=456, name="Lightning Bolt")
        with pytest.raises(Exception):  # FrozenInstanceError
            card.grp_id = 999

    def test_invalid_instance_id(self):
        """Test validation of instance ID."""
        with pytest.raises(ValueError):
            CardIdentity(grp_id=123, instance_id=-1, name="Invalid")


class TestPermanent:
    """Test Permanent domain model."""

    def test_permanent_creation(self):
        """Test basic permanent creation."""
        identity = CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears")
        perm = Permanent(
            identity=identity,
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
        )
        assert perm.identity.name == "Grizzly Bears"
        assert perm.power == 2
        assert perm.toughness == 2

    def test_is_creature(self):
        """Test creature detection."""
        # Creature has power/toughness
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
        )
        assert creature.is_creature is True

        # Non-creature (enchantment, artifact, etc.)
        enchantment = Permanent(
            identity=CardIdentity(grp_id=2, instance_id=2, name="Glorious Anthem"),
            controller_id=1,
            owner_id=1,
        )
        assert enchantment.is_creature is False

    def test_can_attack_untapped(self):
        """Test attack eligibility for untapped creature."""
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=False,
            summoning_sick=False,
        )
        assert creature.can_attack is True

    def test_cannot_attack_when_tapped(self):
        """Test that tapped creatures can't attack."""
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=True,
            summoning_sick=False,
        )
        assert creature.can_attack is False

    def test_cannot_attack_when_summoning_sick(self):
        """Test that summoning sick creatures can't attack."""
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=False,
            summoning_sick=True,
        )
        assert creature.can_attack is False

    def test_can_block_untapped(self):
        """Test block eligibility for untapped creature."""
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=False,
        )
        assert creature.can_block is True

    def test_cannot_block_when_tapped(self):
        """Test that tapped creatures can't block."""
        creature = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=True,
        )
        assert creature.can_block is False


class TestZoneCollection:
    """Test ZoneCollection domain model."""

    def test_zone_creation(self):
        """Test basic zone creation."""
        zone = ZoneCollection(zone_name="Hand", is_hidden=True)
        assert zone.zone_name == "Hand"
        assert zone.is_hidden is True
        assert len(zone) == 0

    def test_add_card(self):
        """Test adding cards to zone."""
        zone = ZoneCollection(zone_name="Hand")
        card = CardIdentity(grp_id=1, instance_id=1, name="Lightning Bolt")
        zone.add_card(card)
        assert len(zone) == 1
        assert zone.cards[0] == card

    def test_remove_card(self):
        """Test removing cards from zone."""
        zone = ZoneCollection(zone_name="Hand")
        card = CardIdentity(grp_id=1, instance_id=1, name="Lightning Bolt")
        zone.add_card(card)

        removed = zone.remove_card(instance_id=1)
        assert removed == card
        assert len(zone) == 0

    def test_remove_nonexistent_card(self):
        """Test removing card that doesn't exist."""
        zone = ZoneCollection(zone_name="Hand")
        removed = zone.remove_card(instance_id=999)
        assert removed is None

    def test_top_cards_ordered_zone(self):
        """Test getting top cards from ordered zone."""
        zone = ZoneCollection(zone_name="Library", is_ordered=True)
        card1 = CardIdentity(grp_id=1, instance_id=1, name="Card 1")
        card2 = CardIdentity(grp_id=2, instance_id=2, name="Card 2")
        card3 = CardIdentity(grp_id=3, instance_id=3, name="Card 3")

        zone.add_card(card1)
        zone.add_card(card2)
        zone.add_card(card3)

        top_2 = zone.top_cards(2)
        assert len(top_2) == 2
        assert top_2[0] == card1
        assert top_2[1] == card2

    def test_top_cards_unordered_zone(self):
        """Test that unordered zones don't return top cards."""
        zone = ZoneCollection(zone_name="Exile", is_ordered=False)
        card = CardIdentity(grp_id=1, instance_id=1, name="Card 1")
        zone.add_card(card)

        top = zone.top_cards(1)
        assert len(top) == 0  # Unordered zones don't have "top"


class TestPlayerGameState:
    """Test PlayerGameState domain model."""

    def test_player_creation(self):
        """Test basic player state creation."""
        player = PlayerGameState(player_id=1)
        assert player.player_id == 1
        assert player.life_total == 20  # Default
        assert player.hand_size == 0

    def test_hand_size_property(self):
        """Test hand size calculation."""
        player = PlayerGameState(player_id=1)
        assert player.hand_size == 0

        card = CardIdentity(grp_id=1, instance_id=1, name="Card")
        player.hand.add_card(card)
        assert player.hand_size == 1

    def test_total_mana_available(self):
        """Test total mana calculation."""
        player = PlayerGameState(player_id=1)
        player.mana_pool = {"W": 2, "U": 3, "R": 1}
        assert player.total_mana_available == 6


class TestCombatState:
    """Test CombatState domain model."""

    def test_combat_state_creation(self):
        """Test basic combat state creation."""
        combat = CombatState()
        assert len(combat.attackers) == 0
        assert combat.is_in_combat() is False

    def test_add_attacker(self):
        """Test adding attackers."""
        combat = CombatState()
        combat.attackers.add(1)
        combat.attackers.add(2)
        assert len(combat.attackers) == 2
        assert combat.is_in_combat() is True

    def test_add_blocker(self):
        """Test assigning blockers."""
        combat = CombatState()
        combat.attackers.add(1)  # Attacker instance_id = 1
        combat.blockers[1] = [2]  # Blocker instance_id = 2

        assert combat.is_blocked(1) is True
        assert combat.get_blockers_for(1) == [2]

    def test_unblocked_attacker(self):
        """Test checking unblocked attackers."""
        combat = CombatState()
        combat.attackers.add(1)

        assert combat.is_blocked(1) is False
        assert combat.get_blockers_for(1) == []

    def test_clear_combat(self):
        """Test clearing combat state."""
        combat = CombatState()
        combat.attackers.add(1)
        combat.blockers[1] = [2]
        combat.damage_assignments[1] = 5

        combat.clear()

        assert len(combat.attackers) == 0
        assert len(combat.blockers) == 0
        assert len(combat.damage_assignments) == 0
        assert combat.is_in_combat() is False


class TestGameState:
    """Test complete GameState domain model."""

    def test_game_state_creation(self):
        """Test basic game state creation."""
        game = GameState()
        assert game.turn_number == 0
        assert game.phase == Phase.UNKNOWN

    def test_is_local_player_turn(self):
        """Test turn detection."""
        game = GameState()
        game.local_player.is_active_player = True
        assert game.is_local_player_turn is True

        game.local_player.is_active_player = False
        assert game.is_local_player_turn is False

    def test_local_has_priority(self):
        """Test priority detection."""
        game = GameState()
        game.local_player.has_priority = True
        assert game.local_has_priority is True

        game.local_player.has_priority = False
        assert game.local_has_priority is False

    def test_active_player_property(self):
        """Test active player property."""
        game = GameState()
        game.local_player.is_active_player = True
        assert game.active_player == game.local_player

        game.local_player.is_active_player = False
        game.opponent.is_active_player = True
        assert game.active_player == game.opponent

    def test_priority_player_property(self):
        """Test priority player property."""
        game = GameState()
        game.local_player.has_priority = True
        assert game.priority_player == game.local_player

        game.local_player.has_priority = False
        game.opponent.has_priority = True
        assert game.priority_player == game.opponent

    def test_is_combat_phase(self):
        """Test combat phase detection."""
        game = GameState()
        game.phase = Phase.COMBAT_ATTACKERS
        assert game.is_combat_phase is True

        game.phase = Phase.MAIN_1
        assert game.is_combat_phase is False

    def test_is_main_phase(self):
        """Test main phase detection."""
        game = GameState()
        game.phase = Phase.MAIN_1
        assert game.is_main_phase is True

        game.phase = Phase.MAIN_2
        assert game.is_main_phase is True

        game.phase = Phase.COMBAT_ATTACKERS
        assert game.is_main_phase is False

    def test_get_permanent_by_id(self):
        """Test finding permanent by instance ID."""
        game = GameState()

        perm = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=100, name="Test"),
            controller_id=1,
            owner_id=1,
        )
        game.local_battlefield.append(perm)

        found = game.get_permanent_by_id(100)
        assert found == perm

        not_found = game.get_permanent_by_id(999)
        assert not_found is None

    def test_get_creatures_that_can_attack(self):
        """Test getting attackable creatures."""
        game = GameState()
        game.local_player.is_active_player = True

        # Add attackable creature
        creature1 = Permanent(
            identity=CardIdentity(grp_id=1, instance_id=1, name="Grizzly Bears"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=False,
            summoning_sick=False,
        )
        game.local_battlefield.append(creature1)

        # Add tapped creature (can't attack)
        creature2 = Permanent(
            identity=CardIdentity(grp_id=2, instance_id=2, name="Tapped Bear"),
            controller_id=1,
            owner_id=1,
            power=2,
            toughness=2,
            tapped=True,
        )
        game.local_battlefield.append(creature2)

        attackers = game.get_creatures_that_can_attack()
        assert len(attackers) == 1
        assert attackers[0] == creature1

    def test_reset_for_new_turn(self):
        """Test resetting state for new turn."""
        game = GameState()
        game.turn_number = 5
        game.history.cards_played.append(
            CardIdentity(grp_id=1, instance_id=1, name="Test")
        )
        game.combat.attackers.add(1)

        game.reset_for_new_turn(6)

        assert game.turn_number == 6
        assert len(game.history.cards_played) == 0
        assert len(game.combat.attackers) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
