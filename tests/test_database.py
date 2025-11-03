import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestDatabase(unittest.TestCase):
    def test_import(self):
        """Test that the ArenaCardDatabase class can be imported."""
        try:
            from database import ArenaCardDatabase
        except ImportError:
            self.fail("Failed to import ArenaCardDatabase from database")

if __name__ == '__main__':
    unittest.main()
