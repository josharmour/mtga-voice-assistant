# Project: MTGA Voice Advisor

## Project Overview

This project is a command-line, voice-enabled tactical advisor for the game *Magic: The Gathering Arena* (MTGA). It monitors the game's log file in real-time, analyzes the current game state, queries a local Large Language Model (LLM) for tactical advice, and speaks the advice to the user using a Text-to-Speech (TTS) engine.

**Core Technologies:**

*   **Language:** Python 3
*   **AI/LLM:** Ollama (running a local model like `llama3.2`)
*   **Data Source:** MTGA's `Player.log` file
*   **Card Information:** [Scryfall API](https://scryfall.com/docs/api)
*   **Voice Output:** TTS libraries like `kokoro`, `edge-tts`, or `pyttsx3`

**Architecture:**

The application is designed around a modular architecture:

1.  **Log Follower (`LogFollower`):** Tails the `Player.log` file to get a real-time stream of game events.
2.  **Log Parser (`MatchScanner`):** Parses the `GreToClientEvent` JSON messages from the log to extract detailed game state information.
3.  **Card ID Resolution (`ScryfallClient`):** Converts Arena's internal card identifiers (`grpId`) into full card data (name, type, abilities) by querying the Scryfall API. It includes a caching mechanism to minimize API calls.
4.  **Game State Manager (`GameStateManager`):** Builds a comprehensive and AI-friendly representation of the current board state.
5.  **AI Advisor (`AIAdvisor`):** Formats the board state into a prompt, sends it to a local Ollama instance, and receives tactical advice.
6.  **TTS Engine (`TextToSpeech`):** Converts the AI's text advice into spoken words.
7.  **Main Application (`CLIVoiceAdvisor`):** Orchestrates all the components in a main loop, triggering advice generation at key decision points in the game (e.g., when the player gains priority).

## Building and Running

### Prerequisites

1.  **Python 3:** Ensure Python 3 and `pip` are installed.
2.  **Ollama:** Install and run a local Ollama instance. Pull a model suitable for tactical advice (e.g., `ollama pull llama3.2`).
3.  **MTGA Detailed Logs:** In MTGA, go to `Options -> Account` and enable **Detailed Logs (Plugin Support)**. This is essential for the application to receive game state data.

### Installation

Create a `requirements.txt` file with the following content and install the dependencies:

```
requests
pyttsx3
# Choose one or more TTS libraries:
# kokoro
# edge-tts
```

Install the packages:
```bash
pip install -r requirements.txt
```

### Running the Application

The main entry point is the `main()` function within the `CLIVoiceAdvisor` script. To run the advisor:

```bash
python <your_main_script>.py
```

The application will automatically attempt to locate the `Player.log` file and will start monitoring it. When a game begins and you have a decision to make, the advisor will automatically provide spoken advice.

## Development Conventions

*   **Modularity:** The code is organized into distinct classes, each with a single responsibility (log parsing, API client, AI interaction, etc.).
*   **Typing:** The codebase uses Python's type hints for clarity and robustness.
*   **Configuration:** Key settings like the Ollama model and host are configurable within their respective classes.
*   **Caching:** API responses from Scryfall are cached locally in `card_cache.json` to improve performance and reduce network requests.
*   **Error Handling:** The application uses `try...except` blocks and `logging` to handle potential errors gracefully (e.g., network issues, parsing errors).
*   **Concurrency:** The AI advice generation, which can be slow, should be run in a separate thread to avoid blocking the real-time log parsing.
*   **Key Trigger:** The primary trigger for generating advice is when the `priorityPlayer` in the game state matches the local player's `systemSeatId`, indicating it's their turn to act.
