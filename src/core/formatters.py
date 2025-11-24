"""
Formatters for displaying game state information.

This module provides dedicated formatters for converting game state objects
into display-ready strings, separating presentation logic from business logic.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class BoardStateFormatter:
    """
    Formats BoardState objects for display in UI.
    Separates presentation logic from business logic.

    This formatter handles the conversion of internal BoardState representations
    into human-readable text suitable for display in GUI, TUI, or CLI modes.
    """

    def __init__(self, card_db=None):
        """
        Initialize formatter with optional card database for name lookups.

        Args:
            card_db: Card database (ArenaCardDatabase) for looking up card names from grpIds
        """
        self.card_db = card_db

    def format_for_display(self, board_state) -> List[str]:
        """
        Format complete board state as lines for display.

        This is the main entry point for formatting a complete board state,
        including all zones, life totals, turn information, etc.

        Args:
            board_state: BoardState object from mtga.py

        Returns:
            List of formatted strings ready for display
        """
        lines = []

        # Header with turn and phase
        lines.append("=" * 70)
        lines.append(f"TURN {board_state.current_turn} - {board_state.current_phase}")
        lines.append("=" * 70)
        lines.append("")

        # Life totals
        lines.append(f"Your Life: {board_state.your_life}  |  Opponent Life: {board_state.opponent_life}")
        lines.append("")

        # Your hand
        hand_count = len(board_state.your_hand) if board_state.your_hand else board_state.your_hand_count
        lines.append(f"YOUR HAND ({hand_count} cards):")
        if board_state.your_hand:
            for card in board_state.your_hand:
                lines.append(self._format_card(card))
        else:
            lines.append("  (Hidden or empty)")
        lines.append("")

        # Opponent hand count
        lines.append(f"OPPONENT'S HAND ({board_state.opponent_hand_count} cards)")
        lines.append("")

        # Your battlefield
        lines.extend(self._format_battlefield(board_state.your_battlefield, "YOUR"))
        lines.append("")

        # Opponent battlefield
        lines.extend(self._format_battlefield(board_state.opponent_battlefield, "OPPONENT'S"))
        lines.append("")

        # Graveyards
        lines.append(f"YOUR GRAVEYARD ({len(board_state.your_graveyard)} cards):")
        if board_state.your_graveyard:
            for card in board_state.your_graveyard[-5:]:  # Last 5
                lines.append(f"  • {card.name}")
        lines.append("")

        lines.append(f"OPPONENT'S GRAVEYARD ({len(board_state.opponent_graveyard)} cards):")
        if board_state.opponent_graveyard:
            for card in board_state.opponent_graveyard[-5:]:  # Last 5
                lines.append(f"  • {card.name}")
        lines.append("")

        # Exile Zones
        lines.append(f"YOUR EXILE ({len(board_state.your_exile)} cards):")
        if board_state.your_exile:
            for card in board_state.your_exile[-5:]:
                lines.append(f"  • {card.name}")
        lines.append("")

        lines.append(f"OPPONENT'S EXILE ({len(board_state.opponent_exile)} cards):")
        if board_state.opponent_exile:
            for card in board_state.opponent_exile[-5:]:
                lines.append(f"  • {card.name}")
        lines.append("")

        return lines

    def _format_battlefield(self, permanents: List, owner_prefix: str) -> List[str]:
        """
        Format battlefield permanents with separation of lands and non-lands.

        Args:
            permanents: List of permanent objects
            owner_prefix: Prefix like "YOUR" or "OPPONENT'S"

        Returns:
            List of formatted strings for the battlefield
        """
        lines = []

        # Separate lands from non-lands
        lands = []
        nonlands = []

        for card in permanents:
            if self._is_land(card):
                lands.append(card)
            else:
                nonlands.append(card)

        # Format non-lands
        lines.append(f"{owner_prefix} BATTLEFIELD ({len(nonlands)} non-lands):")
        for card in nonlands:
            lines.append(self._format_card(card))

        # Group and format lands
        if lands:
            land_counts = Counter([c.name for c in lands])
            lines.append(f"{owner_prefix} LANDS ({len(lands)}):")
            for name, count in land_counts.items():
                lines.append(f"  • {count}x {name}")

        return lines

    def _is_land(self, card) -> bool:
        """
        Determine if a card is a land using heuristics.

        Args:
            card: Card object to check

        Returns:
            True if the card appears to be a land
        """
        # Check for basic lands
        if card.name in ["Plains", "Island", "Swamp", "Mountain", "Forest"]:
            return True

        # Check if "Land" is in the card name
        if "Land" in card.name:
            return True

        # If it has power/toughness, it's likely not a land (probably a creature)
        if card.power is not None:
            return False

        return False

    def _format_card(self, card) -> str:
        """
        Format a single card for display with all relevant information.

        Args:
            card: Card object to format

        Returns:
            Formatted string representation of the card
        """
        # Get power/toughness if present
        pt = ""
        if card.power is not None and card.toughness is not None:
            pt = f" ({card.power}/{card.toughness})"

        # Format counters if present
        counters = ""
        if card.counters:
            # Format counters like [+1/+1, -1/-1]
            c_list = []
            for c_type, c_count in card.counters.items():
                if c_count > 0:
                    c_list.append(f"{c_count} {c_type}")
            if c_list:
                counters = f" [{', '.join(c_list)}]"

        return f"  • {card.name}{pt}{counters}"

    def format_header(self, board_state) -> str:
        """
        Format a simple header with life totals.

        Args:
            board_state: BoardState object

        Returns:
            Formatted header string
        """
        your_life = board_state.your_life or 20
        opp_life = board_state.opponent_life or 20
        return f"Life: You {your_life} | Opponent {opp_life}"

    def format_turn_info(self, board_state) -> str:
        """
        Format turn and phase information.

        Args:
            board_state: BoardState object

        Returns:
            Formatted turn info string
        """
        turn = board_state.current_turn or "?"
        phase = board_state.current_phase or "Unknown"
        whose = "YOUR" if board_state.is_your_turn else "OPPONENT'S"
        return f"Turn {turn} - {whose} {phase}"

    def format_compact(self, board_state) -> str:
        """
        Format a compact one-line summary of board state.

        Useful for status bars or minimal displays.

        Args:
            board_state: BoardState object

        Returns:
            Compact one-line summary
        """
        your_life = board_state.your_life or 20
        opp_life = board_state.opponent_life or 20

        # Count creatures (cards with power/toughness)
        your_creatures = sum(1 for p in board_state.your_battlefield
                           if p.power is not None and p.toughness is not None)
        opp_creatures = sum(1 for p in board_state.opponent_battlefield
                          if p.power is not None and p.toughness is not None)

        return f"You: {your_life}HP {your_creatures}C | Opp: {opp_life}HP {opp_creatures}C"

    def format_hand(self, hand: List, hand_count: int) -> List[str]:
        """
        Format hand contents.

        Args:
            hand: List of card objects in hand (may be empty if hidden)
            hand_count: Number of cards in hand

        Returns:
            List of formatted strings for hand display
        """
        lines = []
        if hand:
            lines.append(f"Hand ({len(hand)} cards):")
            for card in hand:
                name = self._get_card_name(card)
                lines.append(f"  • {name}")
        else:
            lines.append(f"Hand: {hand_count} cards")
        return lines

    def _get_card_name(self, card) -> str:
        """
        Get the display name for a card, using database lookup if needed.

        Args:
            card: Card object

        Returns:
            Card name string
        """
        name = getattr(card, 'name', 'Unknown')

        # If name is Unknown and we have a card database, try to look it up
        if name.startswith("Unknown") and self.card_db:
            grp_id = getattr(card, 'grp_id', None)
            if grp_id:
                looked_up = self.card_db.get_card_name(grp_id)
                if looked_up and not looked_up.startswith("Unknown"):
                    name = looked_up

        return name

    def format_mana_pools(self, board_state) -> List[str]:
        """
        Format mana pool information.

        Args:
            board_state: BoardState object

        Returns:
            List of formatted strings for mana pools
        """
        lines = []

        your_mana = getattr(board_state, 'your_mana_pool', {})
        opp_mana = getattr(board_state, 'opponent_mana_pool', {})

        if your_mana or opp_mana:
            lines.append("--- MANA ---")
            if your_mana:
                mana_str = " ".join(f"{k}:{v}" for k, v in your_mana.items() if v > 0)
                lines.append(f"  You: {mana_str or 'empty'}")
            if opp_mana:
                mana_str = " ".join(f"{k}:{v}" for k, v in opp_mana.items() if v > 0)
                lines.append(f"  Opp: {mana_str or 'empty'}")

        return lines

    def format_graveyards(self, board_state, max_cards: int = 5) -> List[str]:
        """
        Format graveyard summary.

        Args:
            board_state: BoardState object
            max_cards: Maximum number of recent cards to show (default 5)

        Returns:
            List of formatted strings for graveyards
        """
        lines = []

        your_gy = getattr(board_state, 'your_graveyard', [])
        opp_gy = getattr(board_state, 'opponent_graveyard', [])

        if your_gy or opp_gy:
            lines.append("--- GRAVEYARDS ---")
            lines.append(f"  You: {len(your_gy)} cards")
            if your_gy and max_cards > 0:
                for card in your_gy[-max_cards:]:
                    lines.append(f"    • {card.name}")

            lines.append(f"  Opp: {len(opp_gy)} cards")
            if opp_gy and max_cards > 0:
                for card in opp_gy[-max_cards:]:
                    lines.append(f"    • {card.name}")

        return lines
