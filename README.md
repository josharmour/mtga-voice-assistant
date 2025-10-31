# mtga-voice-assistant

## Project Overview

This project is a command-line, voice-enabled tactical advisor for the game *Magic: The Gathering Arena* (MTGA). It monitors the game's log file in real-time, analyzes the current game state, queries a local Large Language Model (LLM) for tactical advice, and speaks the advice to the user using a Text-to-Speech (TTS) engine.

**Core Technologies:**

*   **Language:** Python 3
*   **AI/LLM:** Ollama (running a local model like `llama3.2`)
*   **Data Source:** MTGA's `Player.log` file
*   **Card Information:** [Scryfall API](https://scryfall.com/docs/api)
*   **Voice Output:** TTS libraries like `kokoro`, `edge-tts`, or `pyttsx3`

## Building and Running

### Prerequisites

1.  **Python 3:** Ensure Python 3 and `pip` are installed.
2.  **Ollama:** Install and run a local Ollama instance. Pull a model suitable for tactical advice (e.g., `ollama pull llama3.2`).
3.  **MTGA Detailed Logs:** In MTGA, go to `Options -> Account` and enable **Detailed Logs (Plugin Support)**. This is essential for the application to receive game state data.

### Installation

Install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the Application

The main entry point is `advisor.py`. To run the advisor:

```bash
python advisor.py
```

The application will automatically attempt to locate the `Player.log` file and will start monitoring it. When a game begins and you have a decision to make, the advisor will automatically provide spoken advice.

## Data

The application uses data from [17lands.com](https://www.17lands.com/) and [Scryfall](https://scryfall.com/) to provide card analysis and information. To download and manage this data, use the `manage_data.py` script:

```bash
# To update 17lands data for all standard sets
python manage_data.py --update-17lands

# To update Scryfall card data
python manage_data.py --update-scryfall

# To see all options
python manage_data.py --help
```
