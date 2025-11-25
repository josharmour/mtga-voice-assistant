import logging
import hashlib
import json
from typing import Dict, List, Optional
from src.data.data_management import ScryfallClient
from src.data.arena_cards import ArenaCardDatabase
from src.core.monitoring import get_monitor

logger = logging.getLogger(__name__)


class MTGPromptBuilder:
    """
    Shared prompt building for all LLM advisors.
    Provides rich context with card text, mana availability, and game state.
    Includes intelligent prompt caching to avoid rebuilding identical prompts.
    Includes token budget management to prevent exceeding context limits.
    """

    # Class constants for token budget management
    MAX_PROMPT_TOKENS = 4000  # Default budget, configurable
    CHARS_PER_TOKEN = 4  # Rough estimate for English text

    def __init__(self, scryfall: ScryfallClient = None, arena_db: ArenaCardDatabase = None, max_tokens: int = None):
        self.scryfall = scryfall or ScryfallClient()
        self.arena_db = arena_db  # Local arena card database (fast, always has latest cards)
        self.max_tokens = max_tokens or self.MAX_PROMPT_TOKENS

        # Prompt caching
        self._last_tactical_hash: str = ""
        self._cached_tactical_prompt: str = ""
        self._last_draft_hash: str = ""
        self._cached_draft_prompt: str = ""
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _compute_board_hash(self, board_state: Dict) -> str:
        """
        Compute a hash of the relevant board state fields.
        Only include fields that would affect the prompt.

        Args:
            board_state: Dictionary containing game state information

        Returns:
            MD5 hash string of the board state
        """
        # Extract relevant fields for hashing
        hash_data = {
            'your_life': board_state.get('your_life'),
            'opponent_life': board_state.get('opponent_life'),
            'current_phase': board_state.get('current_phase'),
            'current_turn': board_state.get('current_turn'),
            'is_your_turn': board_state.get('is_your_turn'),
            'opponent_hand_count': board_state.get('opponent_hand_count'),
            'your_hand_count': board_state.get('your_hand_count'),
        }

        # Hash battlefield objects by their instance_ids and key properties
        your_bf = board_state.get('your_battlefield', [])
        hash_data['your_battlefield'] = self._hash_permanents(your_bf)

        opp_bf = board_state.get('opponent_battlefield', [])
        hash_data['opponent_battlefield'] = self._hash_permanents(opp_bf)

        # Hash hand contents
        your_hand = board_state.get('your_hand', [])
        hash_data['your_hand'] = self._hash_cards(your_hand)

        # Hash mana pools
        hash_data['your_mana'] = str(sorted(board_state.get('your_mana_pool', {}).items()))
        hash_data['opponent_mana'] = str(sorted(board_state.get('opponent_mana_pool', {}).items()))

        # Create hash
        hash_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _hash_permanents(self, permanents: List) -> str:
        """
        Create a hash representation of permanents.

        Args:
            permanents: List of permanent objects/dicts

        Returns:
            String representation for hashing
        """
        perm_data = []
        for p in permanents:
            is_dict = isinstance(p, dict)
            perm_data.append({
                'id': p.get('instance_id') if is_dict else getattr(p, 'instance_id', 0),
                'grp': p.get('grp_id') if is_dict else getattr(p, 'grp_id', 0),
                'tapped': p.get('is_tapped') if is_dict else getattr(p, 'is_tapped', False),
                'power': p.get('power') if is_dict else getattr(p, 'power', None),
                'toughness': p.get('toughness') if is_dict else getattr(p, 'toughness', None),
            })
        return str(sorted([str(p) for p in perm_data]))

    def _hash_cards(self, cards: List) -> str:
        """
        Create a hash representation of cards (hand, etc.).

        Args:
            cards: List of card objects/dicts

        Returns:
            String representation for hashing
        """
        card_data = []
        for c in cards:
            is_dict = isinstance(c, dict)
            card_data.append(c.get('grp_id') if is_dict else getattr(c, 'grp_id', 0))
        return str(sorted(card_data))

    def _compute_draft_hash(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Compute hash for draft state.

        Args:
            pack_cards: List of card names in the current pack
            current_pool: List of cards already picked

        Returns:
            MD5 hash string of the draft state
        """
        hash_data = {
            'pack': sorted(pack_cards),
            'pool': sorted(current_pool),
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def build_tactical_prompt(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Build a complete tactical advice prompt with rich context and caching.
        Uses caching to avoid rebuilding identical prompts.

        Args:
            board_state: Dictionary containing current game state
            game_history: Optional list of recent game events

        Returns:
            Formatted prompt string ready for LLM consumption
        """
        monitor = get_monitor()
        with monitor.measure("prompt.build_tactical_prompt"):
            # Compute hash of current state
            current_hash = self._compute_board_hash(board_state)

            # Check cache
            if current_hash == self._last_tactical_hash and self._cached_tactical_prompt:
                self._cache_hits += 1
                logger.debug(f"Prompt cache HIT (total hits: {self._cache_hits})")
                return self._cached_tactical_prompt

            self._cache_misses += 1
            logger.debug(f"Prompt cache MISS (total misses: {self._cache_misses})")

            # Build new prompt
            prompt = self._build_tactical_prompt_impl(board_state, game_history)

            # Update cache
            self._last_tactical_hash = current_hash
            self._cached_tactical_prompt = prompt

            return prompt

    def _build_tactical_prompt_impl(self, board_state: Dict, game_history: List[str] = None) -> str:
        """
        Actual implementation of tactical prompt building with token budget management.

        Args:
            board_state: Dictionary containing current game state
            game_history: Optional list of recent game events

        Returns:
            Formatted prompt string ready for LLM consumption
        """
        # Build base prompt (fixed part)
        base_prompt = self._build_base_prompt(board_state, game_history)

        # Build full context first
        context = self._build_context(board_state)

        # Construct full prompt
        full_prompt = base_prompt + context + "\n\nAdvice (Keep it short, 1-2 sentences, spoken style):"

        # Check token budget
        estimated_tokens = self._estimate_tokens(full_prompt)

        if estimated_tokens > self.max_tokens:
            # Context exceeds budget, use compressed version
            logger.warning(f"Prompt exceeds token budget ({estimated_tokens} > {self.max_tokens}). Using compressed context.")
            available_tokens = self.max_tokens - self._estimate_tokens(base_prompt) - 20  # Reserve for suffix
            context = self._compress_context(board_state, available_tokens)
            full_prompt = base_prompt + context + "\n\nAdvice (Keep it short, 1-2 sentences, spoken style):"

        return full_prompt

    def build_draft_prompt(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Build a draft pick recommendation prompt with caching.

        Args:
            pack_cards: List of card names in the current pack
            current_pool: List of cards already picked

        Returns:
            Formatted prompt string for draft recommendations
        """
        monitor = get_monitor()
        with monitor.measure("prompt.build_draft_prompt"):
            # Compute hash of current draft state
            current_hash = self._compute_draft_hash(pack_cards, current_pool)

            # Check cache
            if current_hash == self._last_draft_hash and self._cached_draft_prompt:
                self._cache_hits += 1
                logger.debug(f"Draft prompt cache HIT (total hits: {self._cache_hits})")
                return self._cached_draft_prompt

            self._cache_misses += 1
            logger.debug(f"Draft prompt cache MISS (total misses: {self._cache_misses})")

            # Build new prompt
            prompt = self._build_draft_prompt_impl(pack_cards, current_pool)

            # Update cache
            self._last_draft_hash = current_hash
            self._cached_draft_prompt = prompt

            return prompt

    def _build_draft_prompt_impl(self, pack_cards: List[str], current_pool: List[str]) -> str:
        """
        Actual implementation of draft prompt building.

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

    def clear_cache(self):
        """Clear the prompt cache (call when game ends or state resets)."""
        self._last_tactical_hash = ""
        self._cached_tactical_prompt = ""
        self._last_draft_hash = ""
        self._cached_draft_prompt = ""
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Prompt cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache hit/miss statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate_percent': round(hit_rate, 1)
        }

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
                # Try local arena database first (fast, always has latest cards)
                card_data = None
                if self.arena_db:
                    local_data = self.arena_db.get_card_data(grp_id)
                    if local_data:
                        card_data = {
                            "name": local_data.get("name", ""),
                            "power": local_data.get("power", ""),
                            "toughness": local_data.get("toughness", ""),
                            "mana_cost": local_data.get("mana_cost", ""),
                            "type_line": local_data.get("type_line", ""),
                            "oracle_text": local_data.get("oracle_text", ""),
                        }

                # Fallback to Scryfall API if not in local DB
                if not card_data:
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

        # Add YOUR mana availability (crucial for play recommendations)
        context_lines.append("\n**My Mana Available:**")
        your_mana = board_state.get('your_mana_pool', {})
        if your_mana:
            mana_str = ", ".join([f"{k}:{v}" for k, v in your_mana.items() if v > 0])
            context_lines.append(f"  Current Mana Pool: {mana_str if mana_str else 'Empty (all tapped)'}")
        else:
            # Count untapped lands on battlefield
            untapped_lands = 0
            for card in my_battlefield:
                is_dict = isinstance(card, dict)
                is_tapped = card.get('is_tapped', False) if is_dict else getattr(card, 'is_tapped', False)
                card_types = card.get('card_types', []) if is_dict else getattr(card, 'card_types', [])
                type_line = card.get('type_line', '') if is_dict else getattr(card, 'type_line', '')
                if not is_tapped and ('Land' in str(card_types) or 'Land' in str(type_line)):
                    untapped_lands += 1
            total_lands = sum(1 for card in my_battlefield if 'Land' in str(card.get('card_types', []) if isinstance(card, dict) else getattr(card, 'card_types', [])) or 'Land' in str(card.get('type_line', '') if isinstance(card, dict) else getattr(card, 'type_line', '')))
            context_lines.append(f"  Lands: {total_lands} total, ~{untapped_lands} untapped (mana available)")

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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // self.CHARS_PER_TOKEN

    def _build_base_prompt(self, board_state: Dict, game_history: List[str] = None) -> str:
        """Build the fixed part of the prompt (game state header)."""
        history_text = chr(10).join(game_history[-5:] if game_history else ['None'])
        return f"""You are a Magic: The Gathering expert advisor.
Analyze the current board state and provide a concise, tactical recommendation for the current phase.
Focus on winning lines, potential blocks, and hidden information (opponent's open mana).
IMPORTANT: Only recommend plays that the player can afford with their current mana. Check card mana costs carefully before suggesting a play.

Current Phase: {board_state.get('current_phase', 'Unknown')}
Turn: {board_state.get('current_turn', '?')}
My Life: {board_state.get('your_life', '?')} | Opponent Life: {board_state.get('opponent_life', '?')}
Opponent has {board_state.get('opponent_hand_count', 0)} cards in hand

Recent Events:
{history_text}

"""

    def _compress_context(self, board_state: Dict, token_budget: int) -> str:
        """Build compressed context that fits within token budget."""
        sections = []
        remaining_budget = token_budget

        battlefield_text = self._format_battlefield_compressed(board_state)
        bf_tokens = self._estimate_tokens(battlefield_text)
        if bf_tokens < remaining_budget * 0.5:
            sections.append(battlefield_text)
            remaining_budget -= bf_tokens
        else:
            minimal_bf = self._format_battlefield_minimal(board_state)
            sections.append(minimal_bf)
            remaining_budget -= self._estimate_tokens(minimal_bf)

        hand_text = self._format_hand_compressed(board_state)
        hand_tokens = self._estimate_tokens(hand_text)
        if hand_tokens < remaining_budget * 0.3:
            sections.append(hand_text)
            remaining_budget -= hand_tokens

        opp_text = self._format_opponent_resources(board_state)
        opp_tokens = self._estimate_tokens(opp_text)
        if opp_tokens < remaining_budget:
            sections.append(opp_text)
            remaining_budget -= opp_tokens

        if remaining_budget > 100:
            deck_text = self._format_deck_minimal(board_state)
            if self._estimate_tokens(deck_text) < remaining_budget:
                sections.append(deck_text)

        return "\n".join(sections)

    def _format_battlefield_compressed(self, board_state: Dict) -> str:
        """Format battlefield with shortened card descriptions (no oracle text)."""
        lines = ["**Battlefield:**"]
        my_bf = board_state.get('your_battlefield', [])
        if my_bf:
            lines.append("My side:")
            for card in my_bf:
                lines.append(f"  - {self._format_card_short(card)}")
        else:
            lines.append("  (empty)")
        opp_bf = board_state.get('opponent_battlefield', [])
        if opp_bf:
            lines.append("Opponent's side:")
            for card in opp_bf:
                lines.append(f"  - {self._format_card_short(card)}")
        else:
            lines.append("  (empty)")
        return "\n".join(lines)

    def _format_battlefield_minimal(self, board_state: Dict) -> str:
        """Ultra-minimal battlefield format - just names and stats."""
        my_bf = board_state.get('your_battlefield', [])
        opp_bf = board_state.get('opponent_battlefield', [])
        my_cards = [self._get_card_name_only(c) for c in my_bf]
        opp_cards = [self._get_card_name_only(c) for c in opp_bf]
        return f"**Battlefield:** You: {', '.join(my_cards) or 'empty'} | Opp: {', '.join(opp_cards) or 'empty'}"

    def _format_hand_compressed(self, board_state: Dict) -> str:
        """Format hand with just card names (no oracle text)."""
        hand = board_state.get('your_hand', [])
        if hand:
            names = [self._get_card_name_only(c) for c in hand]
            return f"**My Hand:** {', '.join(names)}"
        return f"**My Hand:** {board_state.get('your_hand_count', 0)} cards (unknown)"

    def _format_deck_minimal(self, board_state: Dict) -> str:
        """Minimal deck summary - just card count."""
        deck = board_state.get('your_decklist', {})
        if not deck:
            return ""
        total = sum(deck.values())
        return f"**Deck:** {total} cards remaining"

    def _format_opponent_resources(self, board_state: Dict) -> str:
        """Format opponent resource information."""
        lines = ["**Opponent Resources:**"]
        lines.append(f"  Hand: {board_state.get('opponent_hand_count', 0)} cards")
        opp_mana = board_state.get('opponent_mana_pool', {})
        if opp_mana:
            mana_str = ", ".join([f"{k}:{v}" for k, v in opp_mana.items() if v > 0])
            lines.append(f"  Mana: {mana_str or 'Tapped out'}")
        return "\n".join(lines)

    def _format_card_short(self, card) -> str:
        """Format card with name and key stats only (no oracle text)."""
        name = self._get_card_name(card)
        is_dict = isinstance(card, dict)
        power = card.get('power') if is_dict else getattr(card, 'power', None)
        toughness = card.get('toughness') if is_dict else getattr(card, 'toughness', None)
        if power is not None and toughness is not None:
            return f"{name} ({power}/{toughness})"
        return name

    def _get_card_name_only(self, card) -> str:
        """Get just the card name."""
        return self._get_card_name(card)

    def _get_card_name(self, card) -> str:
        """Extract card name from card object or dict."""
        is_dict = isinstance(card, dict)
        name = card.get('name') if is_dict else getattr(card, 'name', 'Unknown')
        if name.startswith("Unknown"):
            grp_id = card.get('grp_id') if is_dict else getattr(card, 'grp_id', None)
            if grp_id:
                # Try local arena database first (fast)
                if self.arena_db:
                    local_data = self.arena_db.get_card_data(grp_id)
                    if local_data and local_data.get('name'):
                        return local_data['name']
                # Fallback to Scryfall API
                if self.scryfall:
                    card_data = self.scryfall.get_card_by_arena_id(grp_id)
                    if card_data and card_data.get('name'):
                        return card_data['name']
        return name
