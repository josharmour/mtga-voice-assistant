import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestUI(unittest.TestCase):
    def test_import(self):
        """Test that the AdvisorTUI and AdvisorGUI classes can be imported."""
        try:
            from ui import AdvisorTUI, AdvisorGUI
        except ImportError:
            self.fail("Failed to import AdvisorTUI and AdvisorGUI from ui")

if __name__ == '__main__':
    unittest.main()
