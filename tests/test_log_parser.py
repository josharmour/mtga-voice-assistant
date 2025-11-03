import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestLogParser(unittest.TestCase):
    def test_import(self):
        """Test that the LogFollower class can be imported."""
        try:
            from log_parser import LogFollower
        except ImportError:
            self.fail("Failed to import LogFollower from log_parser")

if __name__ == '__main__':
    unittest.main()
