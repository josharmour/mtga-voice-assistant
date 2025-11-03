import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestGameState(unittest.TestCase):
    def test_import(self):
        """Test that the GameStateManager class can be imported."""
        try:
            from game_state import GameStateManager
        except ImportError:
            self.fail("Failed to import GameStateManager from game_state")

if __name__ == '__main__':
    unittest.main()
