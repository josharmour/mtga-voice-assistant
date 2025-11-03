import unittest
import sqlite3
import requests
from unittest.mock import patch, MagicMock
from scryfall_db import ScryfallDB

class TestScryfallDB(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory SQLite database for testing."""
        self.db = ScryfallDB(db_path=":memory:")

    def tearDown(self):
        """Close the database connection after each test."""
        self.db.close()

    def test_create_table(self):
        """Test that the cards table is created on initialization."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cards';")
        self.assertIsNotNone(cursor.fetchone())

    @patch('requests.get')
    def test_get_card_by_grpId_fetches_from_scryfall(self, mock_get):
        """Test that get_card_by_grpId fetches from Scryfall when the card is not in the database."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "arena_id": 12345,
            "name": "Test Card",
            "set": "test",
            "rarity": "common",
            "type_line": "Creature — Test",
            "oracle_text": "This is a test card.",
            "mana_cost": "{1}{W}",
            "power": "1",
            "toughness": "1"
        }
        mock_get.return_value = mock_response

        card = self.db.get_card_by_grpId(12345)
        self.assertIsNotNone(card)
        self.assertEqual(card['name'], "Test Card")
        mock_get.assert_called_once_with("https://api.scryfall.com/cards/arena/12345", timeout=5)

    @patch('requests.get')
    def test_get_card_by_grpId_uses_cache(self, mock_get):
        """Test that get_card_by_grpId uses the local database cache on subsequent calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "arena_id": 12345,
            "name": "Test Card",
            "set": "test",
            "rarity": "common",
            "type_line": "Creature — Test",
            "oracle_text": "This is a test card.",
            "mana_cost": "{1}{W}",
            "power": "1",
            "toughness": "1"
        }
        mock_get.return_value = mock_response

        # First call should fetch from Scryfall
        self.db.get_card_by_grpId(12345)
        mock_get.assert_called_once()

        # Second call should not fetch from Scryfall
        self.db.get_card_by_grpId(12345)
        mock_get.assert_called_once() # Still called only once

    @patch('requests.get')
    def test_fetch_and_cache_card_handles_api_error(self, mock_get):
        """Test that fetch_and_cache_card returns None when the Scryfall API returns an error."""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        card = self.db.fetch_and_cache_card(grpId=54321)
        self.assertIsNone(card)

if __name__ == "__main__":
    unittest.main()
