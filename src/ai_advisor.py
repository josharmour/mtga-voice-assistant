import logging
import requests
from typing import Dict, Optional

# Import RAG system (optional - will gracefully degrade if not available)
try:
    from rag_advisor import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available. Install dependencies with: pip install chromadb sentence-transformers torch")

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "mistral:7b"):
        self.host = host
        self.model = model

    def is_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_ollama(self) -> bool:
        """Try to start Ollama service"""
        try:
            import subprocess
            # Try to start ollama serve in the background
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait a moment for it to start
            import time
            time.sleep(2)
            return self.is_running()
        except Exception as e:
            logging.error(f"Failed to start Ollama: {e}")
            return False

    def generate(self, prompt: str) -> Optional[str]:
        logging.debug(f"Ollama prompt: {prompt[:500]}...")
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            logging.debug(f"Ollama raw response: {result}")
            return result.get('response', '').strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama error: {e}")
            return None

class AIAdvisor:
    SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor.

CRITICAL RULES:
1. ONLY reference cards explicitly listed in "YOUR HAND" or "YOUR BATTLEFIELD"
2. You CANNOT destroy lands (Forest, Plains, Swamp, Mountain, Island) - lands are permanent
3. You can only cast spells from YOUR HAND during your main phase
4. Creatures can attack if they've been on battlefield since your last turn
5. If you see "Unknown" cards, say "Wait for card identification"

FORBIDDEN ACTIONS:
- Do NOT mention cards not listed in the board state
- Do NOT suggest destroying/removing lands
- Do NOT invent card names

Give ONLY tactical advice in 1-2 short sentences. Start directly with your recommendation."""

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "mistral:7b", use_rag: bool = True, card_db: Optional["ArenaCardDatabase"] = None):
        self.client = OllamaClient(host=ollama_host, model=model)
        self.use_rag = use_rag and RAG_AVAILABLE
        self.rag_system = None
        self.card_db = card_db  # Store card database for oracle text lookups
        self.last_rag_references = None  # Track most recent RAG references used

        # Initialize RAG system if enabled
        if self.use_rag:
            try:
                logging.info("Initializing RAG system...")
                self.rag_system = RAGSystem()
                # Only initialize rules if ChromaDB and embeddings are available
                if hasattr(self.rag_system.rules_db, 'client') and self.rag_system.rules_db.client:
                    logging.info("RAG system initialized with rules database")
                else:
                    logging.info("RAG system initialized (rules search disabled - install chromadb and sentence-transformers)")
            except Exception as e:
                logging.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
                self.use_rag = False
        else:
            if not RAG_AVAILABLE:
                logging.info("RAG system disabled (dependencies not installed)")
            else:
                logging.info("RAG system disabled by configuration")

    def _build_prompt(self, board_state: "BoardState") -> str:
        """Build comprehensive prompt with all zones and game history"""
        lines = [
            f"== GAME STATE: Turn {board_state.current_turn}, {board_state.current_phase} Phase ==",
            f"Your life: {board_state.your_life} | Opponent life: {board_state.opponent_life}",
            f"Your library: {board_state.your_library_count} cards | Opponent library: {board_state.opponent_library_count} cards",
            "",
        ]

        # Game History - what happened this turn
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn or history.died_this_turn or history.lands_played_this_turn:
                lines.append("== THIS TURN ==")
                if history.cards_played_this_turn:
                    played_names = [c.name for c in history.cards_played_this_turn]
                    lines.append(f"Cards played: {', '.join(played_names)}")
                if history.lands_played_this_turn > 0:
                    lines.append(f"Lands played: {history.lands_played_this_turn}")
                if history.died_this_turn:
                    lines.append(f"Creatures died: {', '.join(history.died_this_turn)}")
                lines.append("")

        # Hand - with oracle text
        if board_state.your_hand:
            lines.append(f"== YOUR HAND ({len(board_state.your_hand)}) ==")
            for card in board_state.your_hand:
                card_info = f"• {card.name}"
                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_info += f"\n  ({oracle_text})"
                lines.append(card_info)
            lines.append("")
        else:
            lines.append("== YOUR HAND == (empty)")
            lines.append("")

        # Battlefield - with tapped/untapped status and oracle text
        if board_state.your_battlefield:
            lines.append(f"== YOUR BATTLEFIELD ({len(board_state.your_battlefield)}) ==")
            for card in board_state.your_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.summoning_sick:
                    status_flags.append("summoning sick")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")
                if card.attached_to:
                    status_flags.append(f"attached to instance {card.attached_to}")

                status_text = f" ({', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                card_line = f"• {card.name}{power_toughness}{status_text}"

                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_line += f"\n  ({oracle_text})"

                lines.append(card_line)
            lines.append("")
        else:
            lines.append("== YOUR BATTLEFIELD == (empty)")
            lines.append("")

        # Opponent's battlefield
        if board_state.opponent_battlefield:
            lines.append(f"== OPPONENT BATTLEFIELD ({len(board_state.opponent_battlefield)}) ==")
            for card in board_state.opponent_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")

                status_text = f" ({', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                card_line = f"• {card.name}{power_toughness}{status_text}"

                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_line += f"\n  ({oracle_text})"

                lines.append(card_line)
            lines.append("")
        else:
            lines.append("== OPPONENT BATTLEFIELD == (empty)")
            lines.append("")

        # Graveyards
        if board_state.your_graveyard:
            graveyard_names = [c.name for c in board_state.your_graveyard]
            lines.append(f"== YOUR GRAVEYARD ({len(graveyard_names)}) ==")
            lines.append(", ".join(graveyard_names))
            lines.append("")

        if board_state.opponent_graveyard:
            opp_graveyard_names = [c.name for c in board_state.opponent_graveyard]
            lines.append(f"== OPPONENT GRAVEYARD ({len(opp_graveyard_names)}) ==")
            lines.append(", ".join(opp_graveyard_names))
            lines.append("")

        return "\n".join(lines)

    def get_tactical_advice(self, board_state: "BoardState") -> Optional[str]:
        prompt = self._build_prompt(board_state)

        # Enhance prompt with RAG context if available
        if self.use_rag and self.rag_system:
            try:
                # Convert BoardState to dict format for RAG system
                board_dict = self._board_state_to_dict(board_state)

                # Use the enhanced method that returns references
                if hasattr(self.rag_system, 'enhance_prompt_with_references'):
                    prompt, self.last_rag_references = self.rag_system.enhance_prompt_with_references(board_dict, prompt)
                    logging.debug(f"Prompt enhanced with RAG context. References: {self.last_rag_references}")
                else:
                    # Fallback to old method if the new method isn't available
                    prompt = self.rag_system.enhance_prompt(board_dict, prompt)
                    logging.debug("Prompt enhanced with RAG context (references not tracked)")
            except Exception as e:
                logging.warning(f"Failed to enhance prompt with RAG: {e}")
                self.last_rag_references = None

        advice = self.client.generate(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        if advice:
            logging.debug(f"AI generated advice: {advice[:500]}...")
        else:
            logging.debug("AI did not generate any advice.")
        return advice

    def _board_state_to_dict(self, board_state: "BoardState") -> dict:
        """Converts BoardState dataclass to a dictionary for RAG processing."""
        import dataclasses
        if not board_state:
            return {}
        return dataclasses.asdict(board_state)

    def get_last_rag_references(self) -> Optional[Dict]:
        """Get the RAG references from the last tactical advice generation."""
        return self.last_rag_references

    def check_important_updates(self, board_state: "BoardState", previous_board_state: Optional["BoardState"]) -> Optional[str]:
        """
        Check if there are important changes that warrant immediate notification.
        Returns advice if important, None if not worth speaking.
        """
        if not previous_board_state:
            return None

        # Build a prompt asking the model to evaluate importance
        evaluation_prompt = f"""You are a tactical advisor monitoring a Magic: The Gathering game in progress.

PREVIOUS STATE (just before):
- Turn {previous_board_state.current_turn}, {previous_board_state.current_phase}
- Your life: {previous_board_state.your_life} | Opponent: {previous_board_state.opponent_life}
- Your battlefield: {len(previous_board_state.your_battlefield)} | Opponent: {len(previous_board_state.opponent_battlefield)}

CURRENT STATE (right now):
- Turn {board_state.current_turn}, {board_state.current_phase}
- Your life: {board_state.your_life} | Opponent: {board_state.opponent_life}
- Your battlefield: {len(board_state.your_battlefield)} | Opponent: {len(board_state.opponent_battlefield)}

WHAT JUST HAPPENED:"""

        changes_detected = False
        # Add detected changes
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn:
                evaluation_prompt += f"\n- Cards played: {', '.join([c.name for c in history.cards_played_this_turn])}"
                changes_detected = True
            if history.died_this_turn:
                evaluation_prompt += f"\n- Creatures died: {', '.join(history.died_this_turn)}"
                changes_detected = True

        # Check life total changes (only significant ones)
        life_change = board_state.your_life - previous_board_state.your_life
        if life_change < -5:  # Only care about losing 5+ life
            evaluation_prompt += f"\n- Your life dropped by {abs(life_change)}"
            changes_detected = True
        elif life_change != 0:
            # Track it but don't necessarily alert
            evaluation_prompt += f"\n- Your life: {life_change:+d}"
            changes_detected = True

        opponent_life_change = board_state.opponent_life - previous_board_state.opponent_life
        if opponent_life_change < -5:  # Only care if opponent losing significant life
            evaluation_prompt += f"\n- Opponent life dropped by {abs(opponent_life_change)}"
            changes_detected = True

        # If no significant changes, don't even query
        if not changes_detected:
            return None

        evaluation_prompt += """

IMPORTANT: Most game events are NOT critical. Only alert if this is URGENT and the player must act NOW.

Is this TRULY critical? (Answer NO for 95% of changes)
- YES = Immediate lethal threat, opponent about to win, must counter/respond this instant
- NO = Everything else (normal plays, small damage, regular creatures, incremental advantage)

Examples of NOT critical:
- Opponent played a creature (unless it's lethal next turn)
- Lost 3-4 life (that's normal)
- Opponent gained some life
- A single creature died

Examples of CRITICAL:
- Opponent has exact lethal damage on board ready to attack
- Opponent played a game-ending combo piece
- You're at 2 life and they have burn spell

Respond in EXACTLY this format:
- If critical: "ALERT: [one sentence warning]"
- If not critical: "NO"

Your response:"""

        response = self.client.generate(evaluation_prompt)
        if response:
            response = response.strip()
            logging.debug(f"Importance check response: {response}")

            if response.startswith("ALERT:"):
                # Extract the advice part after "ALERT:"
                advice = response[6:].strip()
                logging.info(f"Critical update detected: {advice}")
                return advice
            elif "ALERT:" in response:
                # Handle case where model adds extra text before ALERT:
                alert_start = response.find("ALERT:")
                advice = response[alert_start + 6:].strip()
                # Remove any trailing quotes or punctuation artifacts
                advice = advice.strip('"\'.,!?')
                logging.info(f"Critical update detected (extracted): {advice}")
                return advice

        return None
