#!/usr/bin/env python3
"""
Test script for the RAG (Retrieval-Augmented Generation) system

This script initializes the RAG system, loads sample data, and runs queries
to verify that the rules search and card stats integration are working.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil

# Add current directory to path
sys.path.insert(0, '.')

from rag_advisor import RAGSystem, RulesParser, CardStatsDB


class TestRAGSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a sample rules file for testing."""
        if not os.path.exists("data"):
            os.makedirs("data")
        with open("data/MagicCompRules_sample.txt", "w") as f:
            f.write("100.1. These are the Comprehensive Rules of Magic: The Gathering.\n")
            f.write("100.1a. A sub-rule of 100.1.\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up created files."""
        if os.path.exists("data"):
            shutil.rmtree("data")

    def setUp(self):
        """Set up the test environment"""
        # Ensure a clean database for each test
        db_path = "data/card_stats.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        # Mock SentenceTransformer to avoid downloading the model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        # Patch the SentenceTransformer import
        self.sentence_transformer_patch = patch('rag_advisor.SentenceTransformer', return_value=self.mock_model)
        self.sentence_transformer_patch.start()

        self.rag = RAGSystem()
        # Prevent CardMetadataDB from being initialized in tests
        self.rag.card_metadata = None
        self.load_sample_data()

    def tearDown(self):
        """Tear down the test environment"""
        self.sentence_transformer_patch.stop()

    def load_sample_data(self):
        """Load sample 17lands data for testing."""
        sample_data = [
            {'card_name': 'Lightning Bolt', 'set_code': 'M21', 'color': 'R', 'rarity': 'common', 'games_played': 50000, 'win_rate': 0.58, 'gih_win_rate': 0.62, 'iwd': 0.04, 'last_updated': '2025-01-15'},
            {'card_name': 'Counterspell', 'set_code': 'M21', 'color': 'U', 'rarity': 'common', 'games_played': 45000, 'win_rate': 0.59, 'gih_win_rate': 0.61, 'iwd': 0.03, 'last_updated': '2025-01-15'},
        ]
        self.rag.card_stats.insert_card_stats(sample_data)

    def test_rules_parsing(self):
        """Test the parsing of MTG Comprehensive Rules"""
        parser = RulesParser("data/MagicCompRules_sample.txt")
        rules = parser.parse()
        self.assertGreater(len(rules), 0)

    def test_card_stats_retrieval(self):
        """Test the retrieval of card statistics"""
        stats = self.rag.get_card_stats("Lightning Bolt")
        self.assertIsNotNone(stats)
        self.assertEqual(stats['card_name'], "Lightning Bolt")
        self.assertEqual(stats['win_rate'], 0.58)

    def test_prompt_enhancement(self):
        """Test the enhancement of a prompt with RAG data"""
        mock_board_state = {
            'phase': 'combat',
            'battlefield': {
                'player': [{'name': 'Lightning Bolt'}],
                'opponent': [{'name': 'Counterspell'}]
            }
        }
        base_prompt = "Analyze the current board state."
        enhanced_prompt = self.rag.enhance_prompt(mock_board_state, base_prompt)
        self.assertIn("Lightning Bolt", enhanced_prompt)
        self.assertIn("Counterspell", enhanced_prompt)
        self.assertIn("## Tactical Analysis of Key Cards", enhanced_prompt)


if __name__ == "__main__":
    unittest.main()
