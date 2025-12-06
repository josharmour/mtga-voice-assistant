import subprocess
import logging

FALLBACK_VERSION = "0.2.2-dev"

def get_version() -> str:
    """
    Get the current application version.
    
    Tries to get the git short commit hash.
    Falls back to FALLBACK_VERSION if git is unavailable.
    """
    try:
        # Try to get the git short hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1
        )
        git_hash = result.stdout.strip()
        return f"{FALLBACK_VERSION}+{git_hash}"
    except Exception as e:
        logging.debug(f"Could not get git version: {e}")
        return FALLBACK_VERSION
