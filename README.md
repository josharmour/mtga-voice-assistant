# MTGA Voice Advisor

![MTGA Voice Advisor Screenshot](https://github.com/josharmour/mtga-voice-assistant/raw/main/mtga-voice-assistant.png)

A real-time tactical advisor for Magic: The Gathering Arena (MTGA) that analyzes game logs and provides strategic recommendations via voice (Text-to-Speech).

## Features

*   **Real-time Advice:** Monitors `Player.log` to understand the board state and suggests optimal plays using local or cloud AI models.
*   **Draft Assistance:** Provides draft pick recommendations overlaid on the screen.
*   **Voice Output:** Speaks advice using Kokoro (local, high quality) or other TTS engines.
*   **Modular AI:** Supports Google Gemini, OpenAI GPT-4, Anthropic Claude, and local Ollama models.
*   **Local & Secure:** Designed to run locally. Your logs and game data stay on your machine (unless you choose a cloud AI provider).

## Installation

1.  **Prerequisites:**
    *   Python 3.10+ installed.
    *   MTG Arena installed with **Detailed Logs** enabled (`Options` -> `Account` -> check `Detailed Logs`).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/mtga-voice-advisor.git
    cd mtga-voice-advisor
    ```

3.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install additional system dependencies for audio (e.g., `espeak` or `ffmpeg`) depending on your OS.*

## Usage

1.  **Start the Advisor:**
    *   **Windows:** Double-click `start_advisor.bat`.
    *   **Command Line:**
        ```bash
        python main.py
        ```

2.  **Configuration:**
    *   The application will launch a GUI.
    *   Select your **AI Provider** (e.g., Google, OpenAI, Ollama).
    *   Enter your **API Key** (if using cloud providers).
    *   Configure **Voice** and **Volume**.
    *   Ensure MTGA is running. The advisor will automatically detect the log file.

## Project Structure

*   `main.py`: Entry point.
*   `src/core/`: Core application logic (App, UI, AI integration).
*   `src/data/`: Data management (Card databases, Scryfall cache).
*   `logs/`: Application logs.
*   `bug_reports/`: Generated bug reports (screenshots/logs).

## Development

*   **Tests:** Run `pytest` in the root directory.
*   **Linting:** Use `flake8` or `black`.

## License

[MIT License](LICENSE)