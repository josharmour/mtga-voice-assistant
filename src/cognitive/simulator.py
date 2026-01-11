"""Monte Carlo simulation engine for deck testing and analysis."""

import random
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import statistics
from src.cognitive.mana_utils import parse_mana_cost, ManaSource, can_pay_cost


class Deck:
    """Represents a Magic: The Gathering deck for simulation."""

    def __init__(self, cards: List[Dict[str, Any]]):
        """
        Initialize a deck.

        Args:
            cards: List of card dictionaries with 'name', 'cmc', 'type_line', etc.
        """
        self.cards = cards.copy()
        self.library = cards.copy()
        self.hand: List[Dict[str, Any]] = []
        self.battlefield: List[Dict[str, Any]] = []
        self.graveyard: List[Dict[str, Any]] = []
        self.mana_pool: Dict[str, int] = {}
        # Parsed costs cache
        self._parsed_costs: Dict[str, Dict[str, Any]] = {}

    def get_parsed_cost(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Get or compute parsed mana cost for a card."""
        name = card.get("name", "Unknown")
        cost_str = card.get("mana_cost", "")

        # Use simple cache key. Collisions unlikely for same name in one deck unless different printings.
        # But we use object id if available or just name+cost.
        key = f"{name}_{cost_str}"

        if key not in self._parsed_costs:
            self._parsed_costs[key] = parse_mana_cost(cost_str)

        return self._parsed_costs[key]

    def shuffle(self) -> None:
        """Shuffle the library."""
        random.shuffle(self.library)

    def draw(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Draw n cards from the library.

        Args:
            n: Number of cards to draw

        Returns:
            List of drawn cards
        """
        drawn = []
        for _ in range(n):
            if self.library:
                card = self.library.pop(0)
                self.hand.append(card)
                drawn.append(card)
        return drawn

    def mulligan(self, cards_to_keep: int) -> None:
        """
        Mulligan to n cards.

        Args:
            cards_to_keep: Number of cards to keep (typically hand_size - 1)
        """
        # Put hand back
        self.library.extend(self.hand)
        self.hand = []

        # Shuffle
        self.shuffle()

        # Draw new hand
        self.draw(cards_to_keep)

    def reset(self) -> None:
        """Reset deck to initial state."""
        self.library = self.cards.copy()
        self.hand = []
        self.battlefield = []
        self.graveyard = []
        self.mana_pool = {}
        self._parsed_costs = {}

    def count_lands_in_hand(self) -> int:
        """Count number of lands in current hand."""
        count = 0
        for card in self.hand:
            type_line = card.get("type_line", "").lower()
            if "land" in type_line:
                count += 1
        return count

    def count_spells_in_hand(self) -> int:
        """Count number of non-land cards in hand."""
        return len(self.hand) - self.count_lands_in_hand()

    def get_castable_spells(self, available_mana: int) -> List[Dict[str, Any]]:
        """
        Get spells in hand that can be cast with available mana.
        DEPRECATED: Use check_castability instead for complex mana.
        Keeping for backward compatibility if needed, but updated logic uses complex checks.

        Args:
            available_mana: Amount of generic mana available (heuristic)

        Returns:
            List of castable cards
        """
        castable = []
        for card in self.hand:
            type_line = card.get("type_line", "").lower()
            if "land" not in type_line:  # Not a land
                cmc = card.get("cmc", 0)
                if cmc <= available_mana:
                    castable.append(card)
        return castable


class ManaCurveAnalyzer:
    """Analyzes mana curve statistics for a deck."""

    @staticmethod
    def analyze_curve(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the mana curve of a deck.

        Args:
            cards: List of card dictionaries

        Returns:
            Dictionary with curve statistics
        """
        cmcs = []
        lands = 0
        spells = 0

        for card in cards:
            type_line = card.get("type_line", "").lower()
            cmc = card.get("cmc", 0)

            if "land" in type_line:
                lands += 1
            else:
                spells += 1
                cmcs.append(cmc)

        # Calculate statistics
        if cmcs:
            avg_cmc = statistics.mean(cmcs)
            median_cmc = statistics.median(cmcs)
            try:
                mode_cmc = statistics.mode(cmcs)
            except statistics.StatisticsError:
                mode_cmc = None
        else:
            avg_cmc = 0
            median_cmc = 0
            mode_cmc = None

        # CMC distribution
        cmc_distribution = Counter(cmcs)

        return {
            "total_cards": len(cards),
            "lands": lands,
            "spells": spells,
            "land_ratio": lands / len(cards) if cards else 0,
            "avg_cmc": avg_cmc,
            "median_cmc": median_cmc,
            "mode_cmc": mode_cmc,
            "cmc_distribution": dict(cmc_distribution)
        }


class GoldfishSimulator:
    """Simulates playing a deck without an opponent (goldfishing)."""

    def __init__(self, deck: Deck):
        """
        Initialize the goldfish simulator.

        Args:
            deck: Deck to simulate
        """
        self.deck = deck

    def simulate_opening_hand(self, hand_size: int = 7) -> Dict[str, Any]:
        """
        Simulate drawing an opening hand.

        Args:
            hand_size: Size of opening hand

        Returns:
            Statistics about the opening hand
        """
        self.deck.reset()
        self.deck.shuffle()
        self.deck.draw(hand_size)

        lands = self.deck.count_lands_in_hand()
        spells = self.deck.count_spells_in_hand()

        # Simple mulligan decision: keep if 2-5 lands
        keep = 2 <= lands <= 5

        return {
            "hand_size": hand_size,
            "lands": lands,
            "spells": spells,
            "keep": keep,
            "hand": [card.get("name", "Unknown") for card in self.deck.hand]
        }

    def simulate_turns(self, num_turns: int = 5, starting_hand_size: int = 7) -> Dict[str, Any]:
        """
        Simulate playing out the first N turns.

        Args:
            num_turns: Number of turns to simulate
            starting_hand_size: Size of opening hand

        Returns:
            Statistics about the game simulation
        """
        self.deck.reset()
        self.deck.shuffle()
        self.deck.draw(starting_hand_size)

        turn_data = []
        lands_played_total = 0 # Cumulative lands played over game
        spells_cast_total = 0
        cards_drawn_total = starting_hand_size

        # Track battlefield state objects
        # List of ManaSource objects (which wrap cards)
        battlefield_sources: List[ManaSource] = []

        for turn in range(1, num_turns + 1):
            # Untap Step
            for source in battlefield_sources:
                source.tapped = False
                source.current_turn = turn

            # Draw Step (except turn 1 on the play)
            if turn > 1:
                self.deck.draw(1)
                cards_drawn_total += 1

            # Main Phase

            # 1. Play Land
            land_played = False
            for card in self.deck.hand[:]:
                type_line = card.get("type_line", "").lower()
                if "land" in type_line:
                    self.deck.hand.remove(card)
                    self.deck.battlefield.append(card)
                    lands_played_total += 1
                    land_played = True

                    # Check if enters tapped
                    tapped = "enters the battlefield tapped" in card.get("oracle_text", "").lower()

                    # Add to sources
                    source = ManaSource(card, tapped=tapped, entered_turn=turn, current_turn=turn)
                    battlefield_sources.append(source)
                    break

            # 2. Cast Spells
            spells_cast_this_turn = 0

            while True:
                # Identify castable spells
                castable_candidates = []

                # Filter out lands
                spells_in_hand = [c for c in self.deck.hand if "land" not in c.get("type_line", "").lower()]

                if not spells_in_hand:
                    break

                # Check each spell
                for spell in spells_in_hand:
                    cost = self.deck.get_parsed_cost(spell)
                    can_cast, used_sources = can_pay_cost(cost, battlefield_sources)
                    if can_cast:
                        castable_candidates.append((spell, used_sources))

                if not castable_candidates:
                    break

                # Greedy: Pick highest CMC
                # We need to sort by CMC. spell is tuple[0]
                castable_candidates.sort(key=lambda x: x[0].get("cmc", 0), reverse=True)

                best_spell, sources_to_use = castable_candidates[0]

                # Cast it
                self.deck.hand.remove(best_spell)
                self.deck.battlefield.append(best_spell)

                # Tap sources
                for src in sources_to_use:
                    src.tapped = True

                # Add spell to battlefield sources (if it's a permanent)
                # We add everything for now, but only things that produce mana will be useful sources.
                # However, ManaSource constructor checks capabilities.
                # Check if enters tapped
                tapped = "enters the battlefield tapped" in best_spell.get("oracle_text", "").lower()
                new_source = ManaSource(best_spell, tapped=tapped, entered_turn=turn, current_turn=turn)
                battlefield_sources.append(new_source)

                spells_cast_this_turn += 1
                spells_cast_total += 1

            # Calculate available mana for stats (heuristic: count untapped sources)
            # This is a bit simplified for the stats, as we might have just tapped out.
            # But the requirement is likely to show mana capacity or remaining.
            # The original metric was "lands_played_total".
            # Let's keep "lands_played_total" as "Lands in Play".
            # And add "available_mana" as potentially producible mana at start of post-combat main phase (or end of turn)?
            # The original code logged "available_mana" = lands_played_total.
            # I'll log total potential mana generation of the board (untapped).

            available_mana_count = 0
            for src in battlefield_sources:
                 if not src.tapped and src.is_usable():
                     # Estimate mana count (max 1 for now or sum of options?)
                     # Let's use max production length
                     if src.production_options:
                         # Assume max production is what matters
                         available_mana_count += max(sum(opt.values()) for opt in src.production_options)

            turn_data.append({
                "turn": turn,
                "land_played": land_played,
                "lands_in_play": lands_played_total,
                "spells_cast": spells_cast_this_turn,
                "cards_in_hand": len(self.deck.hand),
                "available_mana": available_mana_count # Now reflects actual unused potential
            })

        return {
            "num_turns": num_turns,
            "lands_played": lands_played_total,
            "spells_cast": spells_cast_total,
            "cards_drawn": cards_drawn_total,
            "final_hand_size": len(self.deck.hand),
            "final_battlefield_size": len(self.deck.battlefield),
            "turns": turn_data
        }


class MonteCarloSimulator:
    """Runs Monte Carlo simulations for statistical deck analysis."""

    def __init__(self, cards: List[Dict[str, Any]]):
        """
        Initialize the Monte Carlo simulator.

        Args:
            cards: List of card dictionaries representing the deck
        """
        self.cards = cards
        self.deck = Deck(cards)

    def run_opening_hand_analysis(self, iterations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo analysis of opening hands.

        Args:
            iterations: Number of hands to simulate

        Returns:
            Aggregated statistics
        """
        print(f"Running {iterations} opening hand simulations...")

        simulator = GoldfishSimulator(self.deck)
        results = []

        for _ in range(iterations):
            result = simulator.simulate_opening_hand()
            results.append(result)

        # Aggregate statistics
        lands_counts = [r["lands"] for r in results]
        keep_rate = sum(1 for r in results if r["keep"]) / len(results)

        return {
            "iterations": iterations,
            "avg_lands": statistics.mean(lands_counts),
            "median_lands": statistics.median(lands_counts),
            "keep_rate": keep_rate,
            "land_distribution": dict(Counter(lands_counts))
        }

    def run_goldfish_analysis(self, iterations: int = 100, num_turns: int = 5) -> Dict[str, Any]:
        """
        Run Monte Carlo goldfish simulations.

        Args:
            iterations: Number of games to simulate
            num_turns: Number of turns per game

        Returns:
            Aggregated statistics
        """
        print(f"Running {iterations} goldfish simulations ({num_turns} turns each)...")

        simulator = GoldfishSimulator(self.deck)
        results = []

        for _ in range(iterations):
            result = simulator.simulate_turns(num_turns=num_turns)
            results.append(result)

        # Aggregate statistics
        lands_played = [r["lands_played"] for r in results]
        spells_cast = [r["spells_cast"] for r in results]

        # Per-turn statistics
        turn_stats = {}
        for turn_num in range(1, num_turns + 1):
            turn_lands = [r["turns"][turn_num - 1]["lands_in_play"] for r in results]
            turn_spells = [r["turns"][turn_num - 1]["spells_cast"] for r in results]

            turn_stats[f"turn_{turn_num}"] = {
                "avg_lands": statistics.mean(turn_lands),
                "avg_spells_cast": statistics.mean(turn_spells)
            }

        return {
            "iterations": iterations,
            "num_turns": num_turns,
            "avg_lands_played": statistics.mean(lands_played),
            "avg_spells_cast": statistics.mean(spells_cast),
            "median_spells_cast": statistics.median(spells_cast),
            "turn_stats": turn_stats
        }

    def full_analysis(self, hand_iterations: int = 1000, goldfish_iterations: int = 100) -> Dict[str, Any]:
        """
        Run complete deck analysis.

        Args:
            hand_iterations: Number of opening hands to simulate
            goldfish_iterations: Number of goldfish games to simulate

        Returns:
            Complete analysis results
        """
        print("\n" + "=" * 60)
        print("MONTE CARLO DECK ANALYSIS")
        print("=" * 60)
        print()

        # Mana curve analysis
        print("Analyzing mana curve...")
        curve_analysis = ManaCurveAnalyzer.analyze_curve(self.cards)

        # Opening hand analysis
        hand_analysis = self.run_opening_hand_analysis(iterations=hand_iterations)

        # Goldfish analysis
        goldfish_analysis = self.run_goldfish_analysis(iterations=goldfish_iterations)

        return {
            "deck_size": len(self.cards),
            "mana_curve": curve_analysis,
            "opening_hands": hand_analysis,
            "goldfish": goldfish_analysis
        }
