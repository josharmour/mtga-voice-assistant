# Windows Compatibility Guide

This document outlines the necessary steps to make the MTGA Voice Advisor fully compatible with the Windows operating system. The application was originally developed on Linux, and several parts of the codebase rely on Linux-specific file paths, system commands, and environment setups.

The following sections detail the required modifications. No code changes should be implemented; this guide is for documentation purposes only.

## 1. File Path Detection (`advisor.py`)

The current implementation for detecting the MTGA Player.log file and the card database is hardcoded for Linux environments. These paths need to be updated to support Windows installations.

### `detect_player_log_path()` function:

The existing Windows path is a good starting point, but it should be made more robust.

- **Current Code:**
  ```python
  if os.name == 'nt':
      username = os.getenv('USERNAME')
      drive = os.getenv('USERPROFILE')[0]
      windows_path = f"{drive}:/Users/{username}/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
      if os.path.exists(windows_path):
          return windows_path
  ```

- **Required Changes:**
  - Use `Path.home()` or `os.path.expanduser('~')` instead of constructing the path from environment variables like `USERNAME` and `USERPROFILE`.
  - The `AppData` folder can be reliably found using `os.getenv('APPDATA')` (which points to `.../Roaming`) and then navigating up one level to get to the parent `AppData` directory, or by using `os.getenv('LOCALAPPDATA')` which directly points to `.../Local`. The latter is preferable.
  - The final path should be constructed using `os.path.join` or `pathlib.Path` to ensure correct path separators (`\`).

- **Example of a more robust implementation:**
  ```python
  if os.name == 'nt':
      # AppData/LocalLow is the correct path on Windows
      log_path = Path.home() / "AppData" / "LocalLow" / "Wizards Of The Coast" / "MTGA" / "Player.log"
      if log_path.exists():
          return str(log_path)
  ```

### `detect_card_database_path()` function:

The current function only checks a single hardcoded path on Windows. This needs to be expanded to account for different installation directories.

- **Current Code:**
  ```python
  if os.name == 'nt':
      arena_data = Path("C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/")
      if arena_data.exists():
          db_files = list(arena_data.glob("Raw_CardDatabase_*.mtga"))
          if db_files:
              return str(db_files[0])
  ```
- **Required Changes:**
  - Check both `C:\Program Files` and `C:\Program Files (x86)`.
  - The MTGA installer allows for custom installation locations. The installation path is stored in the Windows Registry. The script should be updated to query the registry to find the correct path.
    - Registry Key: `HKEY_LOCAL_MACHINE\SOFTWARE\Wizards of the Coast\MTGArena`
    - Value: Look for a value that contains the installation path (e.g., `Path`).

## 2. Platform-Specific Dependencies (`advisor.py`)

The application uses external commands for text-to-speech (TTS) audio playback, which are specific to Linux.

### `TextToSpeech._save_and_play_audio()` method:

This method relies on `aplay`, `paplay`, and `ffplay`, which are not typically available on Windows.

- **Current Code:**
  ```python
  players = [
      (["aplay", tmp_path], "aplay"),
      (["paplay", tmp_path], "paplay"),
      (["ffplay", "-nodisp", "-autoexit", tmp_path], "ffplay")
  ]
  ```

- **Required Changes:**
  - Add a cross-platform audio playback library to `requirements.txt`. Recommended options include `playsound`, `simpleaudio`, or `pydub`.
  - Replace the `subprocess.run` calls with the appropriate function from the chosen library.
  - For example, with `playsound`:
    ```python
    try:
        from playsound import playsound
        playsound(tmp_path)
        played = True
    except Exception as e:
        logging.error(f"Audio playback failed with playsound: {e}")
    ```

## 3. Windows Launch Script

The `launch_advisor.sh` script is a bash script and will not run on Windows. A batch script (`.bat`) equivalent is needed.

### `launch_advisor.sh` content:
```bash
#!/bin/bash
export PATH="$HOME/ollama/bin:$PATH"
cd /home/joshu/logparser
source venv/bin/activate
python3 advisor.py
```

### Required `launch_advisor.bat` content:
- The script should handle the activation of a virtual environment and then run the Python script.
- It should not rely on hardcoded absolute paths. Paths should be relative to the script's location.

- **Example `launch_advisor.bat`:**
  ```batch
  @echo off
  REM This script launches the MTGA Voice Advisor on Windows.

  REM Set the path to Ollama if it's not in the system PATH
  REM set PATH=%USERPROFILE%\ollama;%PATH%

  REM Activate the virtual environment
  call venv\Scripts\activate.bat

  REM Launch the advisor
  echo Starting MTGA Voice Advisor...
  python advisor.py

  REM Deactivate the virtual environment
  call venv\Scripts\deactivate.bat

  pause
  ```

## 4. Bug Report Feature (`advisor.py`)

The bug report feature in the GUI has hardcoded Linux paths and screenshot commands.

### `AdvisorGUI._capture_bug_report()` method:

- **Current Code:**
  ```python
  bug_dir = "/home/joshu/logparser/bug_reports"
  ...
  subprocess.run(['gnome-screenshot', '-f', screenshot_file], ...)
  subprocess.run(['scrot', screenshot_file], ...)
  ...
  log_file = "/home/joshu/logparser/logs/advisor.log"
  ```
- **Required Changes:**
  - Use relative paths for the `bug_reports` directory and the log file, e.g., `Path("bug_reports")` and `Path("logs/advisor.log")`.
  - For taking screenshots on Windows, the `Pillow` library's `ImageGrab` module is a good cross-platform choice. It should be added to `requirements.txt`.
  - **Example Screenshot Implementation:**
    ```python
    try:
        from PIL import ImageGrab
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_file, "PNG")
    except ImportError:
        screenshot_file = "Screenshot failed (Pillow not installed)"
    except Exception as e:
        screenshot_file = f"Screenshot failed: {e}"
    ```

## 5. Summary of Recommendations

1.  **Modify `advisor.py`:**
    -   Update `detect_player_log_path()` and `detect_card_database_path()` to use robust, cross-platform methods for finding files on Windows.
    -   Replace Linux-specific audio playback commands in `TextToSpeech` with a cross-platform library like `playsound`.
    -   Update the `_capture_bug_report` method to use relative paths and a cross-platform screenshot library like `Pillow`.

2.  **Create `launch_advisor.bat`:**
    -   Create a new batch file to activate the virtual environment and run the main Python script on Windows.

3.  **Update `requirements.txt`:**
    -   Add any new dependencies required for Windows compatibility (e.g., `playsound`, `Pillow`).
