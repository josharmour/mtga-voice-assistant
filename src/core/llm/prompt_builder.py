import logging
from typing import Dict, List, Optional
from src.data.data_management import ScryfallClient

logger = logging.getLogger(__name__)


class MTGPromptBuilder:
    """
    Shared prompt building for all LLM advisors.
    Provides rich context with card text, mana availability, and game state.
    """

    def __init__(self, scryfall: ScryfallClient = None):
        self.scryfall = scryfall or ScryfallClient()

    def build_tactical_prompt(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Build a complete tactical advice prompt with rich context.

        Args:
            board_state: Dictionary containing current game state
            game_history: Optional list of recent game events

        Returns:
            Formatted prompt string ready for LLM consumption
        """
        context = self._build_context(board_state)

        prompt = f"""You are a Magic: The Gathering expert advisor.
Analyze the current board state and provide a concise, tactical recommendation for the current phase.
Focus on winning lines, potential blocks, and hidden information (opponent's open mana).

Current Phase: {board_state.get('current_phase', 'Unknown')}
Turn: {board_state.get('current_turn', '?')}
My Life: {board_state.get('your_life', '?')} | Opponent Life: {board_state.get('opponent_life', '?')}
Opponent has {board_state.get('opponent_hand_count', 0)} cards in hand

{context}

Recent Events:
{chr(10).join(game_history[-5:] if game_history else ['None'])}

Advice (Keep it short, 1-2 sentences, spoken style):"""

        return prompt

    def build_draft_prompt(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Build a draft pick recommendation prompt.

        Args:
            pack_cards: List of card names in the current pack
            current_pool: List of cards already picked

        Returns:
            Formatted prompt string for draft recommendations
        """
        prompt = f"""I am drafting a deck in Magic Arena.

Current Pack:
{', '.join(pack_cards)}

My Current Pool:
{', '.join(current_pool)}

Which card should I pick? Briefly explain why (synergy, power level, curve)."""

        return prompt

    def _build_context(self, board_state: Dict) -> str:
        """
        Fetch card text for cards on the battlefield and hand to provide context to the LLM.
        This method enriches the board state with full card details including oracle text,
        mana costs, power/toughness, and other relevant information.

        Args:
            board_state: Dictionary containing game state information

        Returns:
            Multi-line string with formatted battlefield, hand, and deck information
        """
        context_lines = ["**Battlefield:**"]

        # Helper to format card info
        def format_card(card_obj):
            is_dict = isinstance(card_obj, dict)
            name = card_obj.get('name') if is_dict else getattr(card_obj, 'name', 'Unknown')
            grp_id = card_obj.get('grp_id') if is_dict else getattr(card_obj, 'grp_id', None)

            # Get fallback stats from object itself (if Scryfall fails)
            obj_power = card_obj.get('power') if is_dict else getattr(card_obj, 'power', None)
            obj_toughness = card_obj.get('toughness') if is_dict else getattr(card_obj, 'toughness', None)

            if grp_id:
                card_data = self.scryfall.get_card_by_arena_id(grp_id)
                if card_data:
                    # Prefer name from Scryfall if local name is unknown
                    if name.startswith("Unknown") and card_data.get("name"):
                        name = card_data.get("name")

                    power = card_data.get('power', '?')
                    toughness = card_data.get('toughness', '?')
                    mana = card_data.get('mana_cost', '')
                    type_line = card_data.get('type_line', '')
                    oracle = card_data.get('oracle_text', '')

                    details = []
                    if mana: details.append(f"Cost: {mana}")
                    if type_line: details.append(f"Type: {type_line}")
                    if power and toughness and 'Creature' in type_line: details.append(f"Stats: {power}/{toughness}")
                    if oracle: details.append(f"Text: {oracle}")

                    return f"- {name} | {' | '.join(details)}"

            # Fallback: Use whatever data we parsed from the logs
            details = []
            if obj_power is not None and obj_toughness is not None:
                details.append(f"Stats: {obj_power}/{obj_toughness}")

            if details:
                return f"- {name} | {' | '.join(details)} (Card text unknown)"

            return f"- {name} (Card text unknown)"

        # Process My Battlefield
        my_battlefield = board_state.get('your_battlefield', [])
        if my_battlefield:
            for card in my_battlefield:
                context_lines.append(format_card(card))
        else:
            context_lines.append("  (empty)")

        context_lines.append("\n**Opponent Battlefield:**")
        opp_battlefield = board_state.get('opponent_battlefield', [])
        if opp_battlefield:
            for card in opp_battlefield:
                context_lines.append(format_card(card))
        else:
            context_lines.append("  (empty)")

        context_lines.append("\n**My Hand:**")
        my_hand = board_state.get('your_hand', [])
        if my_hand:
            for card in my_hand:
                context_lines.append(format_card(card))
        else:
            context_lines.append(f"  {board_state.get('your_hand_count', 0)} cards (contents unknown)")


        # Process My Deck (for probabilistic advice)
        context_lines.append("\n**My Deck (Remaining/Known):**")
        my_deck = board_state.get('your_decklist', {})
        if my_deck:
            # Sort by count descending
            sorted_deck = sorted(my_deck.items(), key=lambda x: x[1], reverse=True)

            # Summarize lands/non-lands for brevity if deck is large
            lands = []
            spells = []
            for name, count in sorted_deck:
                if count <= 0: continue
                if any(l in name for l in ["Plains", "Island", "Swamp", "Mountain", "Forest", "Land"]):
                    lands.append(f"{count}x {name}")
                else:
                    spells.append(f"{count}x {name}")

            if spells:
                context_lines.append("  Spells: " + ", ".join(spells))
            if lands:
                context_lines.append("  Lands: " + ", ".join(lands))
        else:
            context_lines.append("  (Deck list unknown - app started mid-game?)")

        # Add Opponent Resources (Crucial for tactical advice)
        context_lines.append("\n**Opponent Resources:**")
        opp_hand_count = board_state.get('opponent_hand_count', 0)
        context_lines.append(f"  Hand: {opp_hand_count} cards")

        opp_mana = board_state.get('opponent_mana_pool', {})
        if opp_mana:
            mana_str = ", ".join([f"{k}:{v}" for k, v in opp_mana.items() if v > 0])
            context_lines.append(f"  Mana Pool: {mana_str if mana_str else 'Empty (Tapped Out)'}")
        else:
            context_lines.append("  Mana Pool: Unknown (assume open)")

        opp_energy = board_state.get('opponent_energy', 0)
        if opp_energy > 0:
            context_lines.append(f"  Energy: {opp_energy}")

        return "\n".join(context_lines)
