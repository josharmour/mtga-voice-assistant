import unittest
from unittest.mock import MagicMock, patch
from advisor import AIAdvisor, BoardState, GameObject, GameHistory, RAG_AVAILABLE

class TestAIAdvisor(unittest.TestCase):
    def setUp(self):
        """Set up a mock card database for testing."""
        self.mock_card_db = MagicMock()
        self.mock_card_db.get_mana_cost.return_value = "{1}{R}"
        self.mock_card_db.get_type_line.return_value = "Instant"
        self.mock_card_db.get_oracle_text.return_value = "Deal 3 damage to any target."

    @patch('advisor.OllamaClient')
    def test_ai_advisor_initialization(self, MockOllamaClient):
        """Test that AIAdvisor initializes correctly."""
        advisor = AIAdvisor(card_db=self.mock_card_db)
        self.assertIsNotNone(advisor.client)
        if RAG_AVAILABLE:
            self.assertTrue(advisor.use_rag)
            self.assertIsNotNone(advisor.rag_system)
        else:
            self.assertFalse(advisor.use_rag)
            self.assertIsNone(advisor.rag_system)

    def test_build_prompt_with_empty_board(self):
        """Test _build_prompt with a minimal board state."""
        advisor = AIAdvisor(card_db=self.mock_card_db)
        board_state = BoardState(
            your_seat_id=1,
            opponent_seat_id=2,
            current_turn=1,
            current_phase="Main"
        )
        prompt = advisor._build_prompt(board_state)
        self.assertIn("== GAME STATE: Turn 1, Main Phase ==", prompt)
        self.assertIn("== YOUR HAND == (empty)", prompt)
        self.assertIn("== YOUR BATTLEFIELD == (empty)", prompt)

    def test_build_prompt_with_cards_in_play(self):
        """Test _build_prompt with a more complex board state."""
        advisor = AIAdvisor(card_db=self.mock_card_db)
        board_state = BoardState(
            your_seat_id=1,
            opponent_seat_id=2,
            current_turn=3,
            current_phase="Combat",
            your_hand=[
                GameObject(instance_id=101, grp_id=1, zone_id=1, owner_seat_id=1, name="Lightning Bolt")
            ],
            your_battlefield=[
                GameObject(instance_id=102, grp_id=2, zone_id=2, owner_seat_id=1, name="Goblin Guide", power=2, toughness=2)
            ],
            opponent_battlefield=[
                GameObject(instance_id=201, grp_id=3, zone_id=2, owner_seat_id=2, name="Tarmogoyf", power=4, toughness=5)
            ],
            history=GameHistory(turn_number=3)
        )

        prompt = advisor._build_prompt(board_state)
        self.assertIn("== YOUR HAND (1) ==", prompt)
        self.assertIn("• Lightning Bolt {1}{R} (Instant)", prompt)
        self.assertIn("Rules: Deal 3 damage to any target.", prompt)
        self.assertIn("== YOUR BATTLEFIELD (1) ==", prompt)
        self.assertIn("• Goblin Guide [2/2]", prompt)
        self.assertIn("== OPPONENT BATTLEFIELD (1) ==", prompt)
        self.assertIn("• Tarmogoyf [4/5]", prompt)

    @patch('advisor.OllamaClient')
    @patch('advisor.RAGSystem' if RAG_AVAILABLE else 'unittest.mock.MagicMock')
    def test_get_tactical_advice(self, MockRAGSystem, MockOllamaClient):
        """Test that get_tactical_advice returns advice from the Ollama client."""
        mock_ollama = MockOllamaClient.return_value
        mock_ollama.generate.return_value = "Attack with Goblin Guide."

        if RAG_AVAILABLE:
            mock_rag = MockRAGSystem.return_value
            mock_rag.enhance_prompt_with_references.return_value = ("Enhanced Prompt", {})

        advisor = AIAdvisor(card_db=self.mock_card_db)
        board_state = BoardState(your_seat_id=1, opponent_seat_id=2)

        advice = advisor.get_tactical_advice(board_state)

        self.assertEqual(advice, "Attack with Goblin Guide.")
        mock_ollama.generate.assert_called_once()
        if RAG_AVAILABLE:
            mock_rag.enhance_prompt_with_references.assert_called_once()

if __name__ == "__main__":
    unittest.main()
