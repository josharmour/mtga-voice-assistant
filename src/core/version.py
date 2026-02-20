import subprocess
import logging
import os

FALLBACK_VERSION = "0.3.0"

def get_version() -> str:
    """
    Get the current application version.
    
    Tries to get the git short commit hash.
    Falls back to FALLBACK_VERSION if git is unavailable.
    """
    # Quick exit if we're on a slow network share or git isn't needed
    if os.environ.get("SKIP_GIT_VERSION") == "1":
        return FALLBACK_VERSION

    try:
        # Try to get the git short hash with a very short timeout
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            timeout=0.5   # 500ms max
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
            return f"{FALLBACK_VERSION}+{git_hash}"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    
    return FALLBACK_VERSION
