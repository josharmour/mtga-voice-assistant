import logging
import sys
import threading
from typing import Dict, Generator

# Ensure root (where mtg_agent.py resides) is in path
if "." not in sys.path:
    sys.path.append(".")

try:
    from mtg_agent import create_agent, run_query
except ImportError:
    logging.error("Failed to import mtg_agent. Ensure it is in the root directory.")
    create_agent = None
    run_query = None

logger = logging.getLogger(__name__)

class PlaneswalkerAdvisor:
    """
    Adapter for The Planeswalker Agent (LangGraph-based).
    """
    _agent_instance = None
    _agent_lock = threading.Lock()

    def __init__(self, model_name: str = "planeswalker", card_db=None, **kwargs):
        self.model_name = model_name
        self.card_db = card_db
        
        # Singleton pattern for the heavy agent
        with self._agent_lock:
            if not PlaneswalkerAdvisor._agent_instance:
                if create_agent:
                    logger.info("Initializing Planeswalker Agent (this may take a moment)...")
                    PlaneswalkerAdvisor._agent_instance = create_agent()
                    logger.info("Planeswalker Agent initialized.")
                else:
                    logger.error("Cannot initialize Planeswalker Agent: module not found.")
        
        self.agent = PlaneswalkerAdvisor._agent_instance

    def get_tactical_advice(self, board_state: Dict) -> str:
        """Non-streaming advice generation."""
        # Consume the stream and join
        return "".join(list(self.get_tactical_advice_stream(board_state)))

    def get_tactical_advice_stream(self, board_state: Dict) -> Generator[str, None, None]:
        """
        Generates tactical advice using the Planeswalker Agent.
        The agent returns a final response string, so we yield it as a single chunk
        (or split by sentence if needed, but app.py handles splitting).
        """
        if not self.agent:
            yield "Planeswalker Agent is not available. Check logs for import errors."
            return

        query = self._build_prompt(board_state)
        logger.info(f"Querying Planeswalker Agent: {query[:100]}...")

        try:
            # Prepare result container for thread safety
            result_container = {}
            
            def safe_query_runner():
                """Run query in a separate thread with sufficient stack."""
                try:
                    if run_query:
                        result_container["data"] = run_query(self.agent, query)
                    else:
                        result_container["error"] = "Agent run_query function invalid."
                except Exception as ex:
                    logger.error(f"Error in Planeswalker agent thread: {ex}")
                    result_container["error"] = str(ex)

            # Use a fresh thread with increased stack size
            # Windows default is 1MB, which might be tight for deep recursion in LangGraph/Torch
            # We request 4MB to be safe.
            try:
                threading.stack_size(4 * 1024 * 1024)  # 4MB
            except ValueError:
                pass  # Ignore if not supported or already set differently

            t = threading.Thread(target=safe_query_runner)
            t.start()
            t.join(timeout=60.0) # 60s timeout to prevent hanging forever

            if t.is_alive():
                yield "Error: Agent timed out."
                return

            if "error" in result_container:
                yield f"Error running agent: {result_container['error']}"
                return

            # Process result
            result = result_container.get("data", {})
            response = result.get("final_response", "I could not generate a response.")
            
            # Ensure response is a string
            if isinstance(response, dict):
                import json
                response = json.dumps(response, indent=2)
            elif not isinstance(response, str):
                response = str(response)

            # Yield the full response
            yield response

        except Exception as e:
            logger.error(f"Error wrapping Planeswalker Agent: {e}")
            yield f"Error running agent: {e}"

    def _build_prompt(self, board_state: Dict) -> str:
        """
        Convert structured board state into a natural language query for the agent.
        Harden against structured data (dicts) in lists instead of strings.
        """
        # Extract key info
        turn = board_state.get('current_turn', 0)
        phase = board_state.get('current_phase', 'Unknown Phase')
        
        # Helper to stringify list items safely
        def format_list(items):
            if not items:
                return 'Empty'
            
            # If item is a dict, try to get 'name', 'card_name', or just str() it
            formatted = []
            for item in items:
                if isinstance(item, dict):
                    # Try common name keys
                    name = item.get('name') or item.get('card_name') or str(item)
                    formatted.append(str(name))
                elif hasattr(item, 'name'):
                    # Handle objects/dataclasses if not converted to dict
                    formatted.append(str(item.name))
                else:
                    # Strings or other primitives
                    formatted.append(str(item))
            
            return ', '.join(formatted)

        my_hand = board_state.get('your_hand', [])
        my_lands = board_state.get('your_lands', [])
        my_creatures = board_state.get('your_creatures', [])
        opp_creatures = board_state.get('opponent_creatures', [])
        opp_life = board_state.get('opponent_life', 20)
        my_life = board_state.get('your_life', 20)
        
        # User query from voice command context
        user_question = board_state.get('user_query', "")
        
        # Construct context
        context_lines = [
            f"It is Turn {turn}, {phase}.",
            f"My Life: {my_life} (Opponent: {opp_life})",
            f"My Hand: {format_list(my_hand)}",
            f"My Board: {format_list(my_creatures + my_lands)}",
            f"Opponent Board: {format_list(opp_creatures)}",
        ]
        
        context_str = "\n".join(context_lines)
        
        if user_question:
            prompt = f"{context_str}\n\nUser Question: {user_question}"
        else:
            # Context-aware prompting based on game phase
            # If we are in Draft or Deck Building, we encourage external knowledge.
            # If in Gameplay, we restrict it to avoid slow/irrelevant hallucinations unless necessary.
            
            trigger_type = board_state.get('trigger_type', 'UNKNOWN')
            
            if trigger_type in ['DRAFT', 'DECK_BUILDING', 'MANUAL']:
                # Full capability mode
                prompt = f"{context_str}\n\nAnalyze the board state and recommend the best detailed play or strategic advice. You may use metagame knowledge."
            else:
                # Gameplay / Tactical Mode
                prompt = f"{context_str}\n\nAnalyze the board state and recommend the best tactical play. Focus on the visible board and rules. Do not search for metagame statistics unless specific cards are unknown to you."
            
        return prompt

    def analyze_draft_pick(self, pack_cards: list, picked_cards: list, pack_num: int, pick_num: int) -> str:
        """
        Analyze a draft pick and return raw synergy/metagame context.
        
        This method is designed to be used by DraftAdvisor to enrich its prompt
        for the main LLM. It returns structured text suitable for injection into
        another prompt, NOT a conversational response.
        
        Args:
            pack_cards: List of card names in the current pack
            picked_cards: List of card names already picked
            pack_num: Current pack number (1-3)
            pick_num: Current pick number (1-14)
            
        Returns:
            Structured text with synergy and metagame insights
        """
        if not self.agent:
            return ""
        
        # Build a focused query for the agent
        pack_str = ", ".join(pack_cards[:5])  # Top 5 candidates
        picked_str = ", ".join(picked_cards[-10:]) if picked_cards else "None yet"
        
        query = f"""Draft Analysis Request (Pack {pack_num}, Pick {pick_num}):
Cards in Pack: {pack_str}
Already Picked: {picked_str}

Provide a brief synergy analysis:
1. Which cards synergize best with the current pool?
2. Are any cards part of a strong archetype in this format?
3. Any metagame considerations (e.g., is a color being cut)?

Keep response under 100 words. Focus on card interactions and format trends."""

        try:
            # Run the agent query
            result_container = {}
            
            def safe_query_runner():
                try:
                    if run_query:
                        result_container["data"] = run_query(self.agent, query)
                except Exception as ex:
                    logger.error(f"Error in draft analysis thread: {ex}")
                    result_container["error"] = str(ex)

            try:
                threading.stack_size(4 * 1024 * 1024)
            except ValueError:
                pass

            t = threading.Thread(target=safe_query_runner)
            t.start()
            t.join(timeout=30.0)  # Shorter timeout for draft context

            if t.is_alive() or "error" in result_container:
                return ""

            result = result_container.get("data", {})
            response = result.get("final_response", "")
            
            if isinstance(response, dict):
                import json
                response = json.dumps(response, indent=2)
            elif not isinstance(response, str):
                response = str(response)
            
            return response.strip()

        except Exception as e:
            logger.error(f"Error in analyze_draft_pick: {e}")
            return ""
