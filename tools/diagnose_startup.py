import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Checking imports...")
try:
    from src.core.app import detect_player_log_path
    print("✅ src.core.app imported")
except ImportError as e:
    print(f"❌ src.core.app failed to import: {e}")
    sys.exit(1)

try:
    from src.core.gemini_advisor import GeminiAdvisor
    print("✅ src.core.gemini_advisor imported")
except ImportError as e:
    print(f"❌ src.core.gemini_advisor failed to import: {e}")

print("\nChecking Environment...")
if os.getenv("GEMINI_API_KEY"):
    print("✅ GEMINI_API_KEY found")
else:
    print("❌ GEMINI_API_KEY NOT found")

print("\nChecking Player.log...")
log_path = detect_player_log_path()
if log_path:
    print(f"✅ Player.log found at: {log_path}")
else:
    print("❌ Player.log NOT found")

print("\nDiagnostic complete.")
