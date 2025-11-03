import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestAIAdvisor(unittest.TestCase):
    def test_import(self):
        """Test that the AIAdvisor class can be imported."""
        try:
            from ai_advisor import AIAdvisor
        except ImportError:
            self.fail("Failed to import AIAdvisor from ai_advisor")

if __name__ == '__main__':
    unittest.main()
