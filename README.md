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

The main entry point is the `main()` function within the `CLIVoiceAdvisor` script. To run the advisor:

```bash
python <your_main_script>.py
```

The application will automatically attempt to locate the `Player.log` file and will start monitoring it. When a game begins and you have a decision to make, the advisor will automatically provide spoken advice.

## Data

The application uses data from [17lands.com](https://www.17lands.com/) to provide card analysis. Due to the size of the data files, they are not included in this repository. You will need to download the data files yourself and place them in the `data/` directory.

You can find the data files at [https://www.17lands.com/public_datasets](https://www.17lands.com/public_datasets).
