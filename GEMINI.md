# Project: MTGA Voice Advisor

## Project Overview

This project is a command-line, voice-enabled tactical advisor for the game *Magic: The Gathering Arena* (MTGA). It monitors the game's log file in real-time, analyzes the current game state, queries a local Large Language Model (LLM) for tactical advice, and speaks the advice to the user using a Text-to-Speech (TTS) engine.

**Core Technologies:**

*   **Language:** Python 3
*   **AI/LLM:** Ollama (running a local model like `llama3.2`)
*   **Data Source:** MTGA's `Player.log` file
*   **Card Information:** Local SQLite Database (`data/unified_cards.db`)
*   **Voice Output:** TTS libraries like `kokoro`, `edge-tts`, or `pyttsx3`

**Architecture:**

The application is designed around a modular architecture:

1.  **Log Follower (`src/core/mtga.py`):** Tails the `Player.log` file to get a real-time stream of game events.
2.  **Log Parser (`src/core/mtga.py` - `MatchScanner`):** Parses the `GreToClientEvent` JSON messages from the log to extract detailed game state information.
3.  **Card ID Resolution (`src/data/arena_cards.py` - `ArenaCardDatabase`):** Converts Arena's internal card identifiers (`grpId`) into full card data (name, type, abilities) by querying a local SQLite database (`unified_cards.db`).
4.  **Game State Manager (`src/core/mtga.py` - `GameStateManager`):** Builds a comprehensive and AI-friendly representation of the current board state.
5.  **AI Advisor (`src/core/ai.py`):** Formats the board state into a prompt, sends it to a local Ollama instance, and receives tactical advice.
6.  **TTS Engine (`src/core/ui.py`):** Converts the AI's text advice into spoken words.
7.  **Main Application (`src/core/app.py`):** Orchestrates all the components in a main loop, triggering advice generation at key decision points in the game (e.g., when the player gains priority).

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

The main entry point is the `main.py` script (which calls `src/core/app.py`). To run the advisor:

```bash
python main.py
```

The application will automatically attempt to locate the `Player.log` file and will start monitoring it. When a game begins and you have a decision to make, the advisor will automatically provide spoken advice.

## Development Conventions

*   **Modularity:** The code is organized into distinct classes, each with a single responsibility.
*   **Typing:** The codebase uses Python's type hints for clarity and robustness.
*   **Configuration:** Key settings like the Ollama model and host are configurable.
*   **Data Persistence:** Card data is stored in a local SQLite database to allow for offline access and consistent ID resolution.
*   **Error Handling:** The application uses `try...except` blocks and `logging` to handle potential errors gracefully.
*   **Concurrency:** AI advice generation and TTS are run in separate threads to avoid blocking the real-time log parsing.
*   **Key Trigger:** The primary trigger for generating advice is when the `priorityPlayer` in the game state matches the local player's `systemSeatId`.



## Next-Generation AI Development

The project is currently undergoing a major evolution in its AI capabilities, moving from a prompt-based LLM approach to a state-of-the-art, data-driven model for providing tactical advice.

**Core Principles:**

*   **Data-Driven Decisions:** The new AI will be trained on massive datasets from [17Lands](https://www.17lands.com/).
*   **Transformer Architecture:** Using Transformer-based neural networks to understand complex game states.
*   **Outcome-Weighted Training:** Training the model to maximize win probability.
*   **Explainability:** Providing clear reasoning for every piece of advice.