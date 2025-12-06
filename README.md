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

2.  **Install Ollama (Required for Local AI):**
    *   Download and install [Ollama](https://ollama.com/).
    *   Run the following command in your terminal to pull a model (e.g., Llama 3.2):
        ```bash
        ollama pull llama3.2
        ```
    *   *Note: If you have a slow computer, we recommend using Google Gemini (Flash) or OpenAI instead of local models for better performance.*

3.  **Clone the Repository:**
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

## Troubleshooting & Performance

*   **"Advisor is slow / appears late":**
    *   If you are running a local model (Ollama) on an older PC, it may be too slow to provide advice in real-time.
    *   **Solution:** Switch to **Google Gemini (Flash)** in the settings. It has a free tier, is extremely fast, and provides high-quality advice without using your PC's resources.

*   **"Ollama found, but no models installed":**
    *   Open your terminal (cmd/powershell) and run `ollama pull llama3.2` to download the default model.

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