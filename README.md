# MTGA Voice Advisor

## Overview

This project is a voice-enabled tactical advisor for *Magic: The Gathering Arena* (MTGA). It analyzes the game's log file in real-time to provide strategic advice, draft pick recommendations, and deck-building suggestions. The advisor uses a local Large Language Model (LLM) for analysis and a Text-to-Speech (TTS) engine for voice output, offering a hands-free experience.

It can be run with a full Graphical User Interface (GUI), a Text-based User Interface (TUI) for terminals, or a basic Command-Line Interface (CLI).

## Features

- **Real-Time Tactical Advice**: Get turn-by-turn suggestions during gameplay, including blocking, attacking, and spell-casting advice.
- **Draft Pick Recommendations**: The Draft Advisor analyzes packs using 17lands data to suggest the best picks and help you build a strong deck.
- **Automated Deck Builder**: After a draft, the Deck Builder suggests an optimal 40-card deck from your card pool based on successful archetypes.
- **Mulligan Guidance**: Uses 17lands opening hand win rate data to help you decide whether to keep or mulligan your starting hand.
- **Voice-Enabled**: Hear advice spoken aloud, allowing you to focus on the game.
- **Multiple Interfaces**: Choose between a feature-rich GUI, a terminal-based TUI, or a simple CLI.
- **Local First**: Runs entirely on your machine, using a local LLM (via Ollama) for privacy and offline capability.

---

## Architecture

The application is composed of several key modules working together:

- **`advisor.py`**: The main application entry point. It orchestrates all other components, handles the user interface (GUI, TUI, or CLI), and manages the main event loop.
- **Log Follower (`LogFollower`)**: Continuously monitors the MTGA `Player.log` file for new entries, handling log rotation automatically.
- **Game State Manager (`GameStateManager` & `MatchScanner`)**: Parses the raw log data, reconstructs the game state (players, zones, cards), and tracks game events in real-time.
- **AI Advisor (`AIAdvisor`)**: Constructs detailed prompts from the current `BoardState`, queries the LLM, and processes the response to provide tactical advice.
- **RAG System (`rag_advisor.py`)**: The Retrieval-Augmented Generation system enhances prompts with contextual data from two sources:
    1.  **MTG Comprehensive Rules**: A vector database allows for semantic searches over the official rules.
    2.  **17lands Card Statistics**: A SQLite database provides card performance data (win rates, average-taken-at, etc.).
- **Draft Advisor (`draft_advisor.py`)**: Provides pick recommendations during drafts by scoring cards based on 17lands data.
- **Deck Builder (`deck_builder.py`)**: Suggests an optimized decklist after a draft by comparing the player's card pool against a database of successful deck archetypes.
- **Text-to-Speech (`TextToSpeech`)**: Converts the advisor's text responses into spoken audio using local TTS engines.

---

## First-Time Setup

Follow these steps to get the MTGA Voice Advisor running on your system.

### Step 1: Prerequisites

1.  **Python 3**: Ensure you have Python 3 and `pip` installed.
2.  **Ollama**: Install and run a local Ollama instance. You will need a model suitable for tactical advice. We recommend `llama3.2`:
    ```bash
    ollama pull llama3.2
    ```
3.  **Enable MTGA Detailed Logs**: This is **essential**. In MTGA, go to `Options -> Account` and check the box for **Detailed Logs (Plugin Support)**.

### Step 2: Install Dependencies

It is highly recommended to use a Python virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```

### Step 3: Download and Generate Local Data

The advisor relies on several local databases. Run the following scripts in order to download and generate them.

1.  **Download MTG Rules**:
    ```bash
    python download_rules.py
    ```

2.  **Download 17lands & Scryfall Data**: This script downloads card statistics from 17lands and populates the local Scryfall card database.
    ```bash
    python manage_data.py --update-17lands
    python manage_data.py --update-scryfall
    ```

3.  **Initialize RAG Database**: This processes the rules text into a vector database for semantic search.
    ```bash
    python initialize_rules.py
    ```

---

## Usage

You can run the application in one of three modes: GUI, TUI, or CLI.

### GUI Mode (Default)

The graphical interface provides the best experience with full access to all settings.

```bash
python advisor.py --gui
```
*If you run `python advisor.py` without flags, it will attempt to launch in GUI mode.*

### TUI Mode

The Text-based User Interface runs in your terminal and is controlled by keyboard commands.

```bash
python advisor.py --tui
```
Inside the TUI, type `/help` for a list of available commands.

### CLI Mode

The Command-Line Interface provides basic output and is suitable for environments where a more complex UI is not needed.

```bash
python advisor.py --cli
```

---

## Data Management

The `manage_data.py` script helps keep your local data up to date.

-   **Update 17lands Data**:
    ```bash
    python manage_data.py --update-17lands
    ```
-   **Update for All Sets (not just Standard)**:
    ```bash
    python manage_data.py --update-17lands --all-sets
    ```
-   **Update Scryfall Cache**:
    ```bash
    python manage_data.py --update-scryfall
    ```
-   **See all options**:
    ```bash
    python manage_data.py --help
    ```
