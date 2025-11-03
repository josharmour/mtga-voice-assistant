import unittest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestTTS(unittest.TestCase):
    def test_import(self):
        """Test that the TextToSpeech class can be imported."""
        try:
            from tts import TextToSpeech
        except ImportError:
            self.fail("Failed to import TextToSpeech from tts")

if __name__ == '__main__':
    unittest.main()
