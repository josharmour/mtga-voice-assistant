# MTGA Voice Advisor

## Project Overview

This project is a command-line, voice-enabled tactical advisor for the game *Magic: The Gathering Arena* (MTGA). It monitors the game's log file in real-time, analyzes the current game state, queries a local Large Language Model (LLM) for tactical advice, and speaks the advice to the user using a Text-to-Speech (TTS) engine.

**Core Technologies:**

*   **Language:** Python 3
*   **AI/LLM:** Ollama (running a local model like `llama3.2`)
*   **Data Source:** MTGA's `Player.log` file
*   **Card Information:** [Scryfall API](https://scryfall.com/docs/api)
*   **Voice Output:** TTS libraries like `kokoro`, `edge-tts`, or `pyttsx3`

---

## First-Time Setup: A Step-by-Step Guide

This guide will walk you through setting up the MTGA Voice Advisor for the first time. Because the application relies on local data files that are not checked into Git, you must run a few scripts to download and generate the necessary databases.

### Step 1: Prerequisites

Before you begin, make sure you have the following installed and configured:

1.  **Python 3:** Ensure Python 3 and `pip` are installed on your system.
2.  **Ollama:** Install and run a local Ollama instance. You will need to pull a model suitable for tactical advice. We recommend `llama3.2`:
    ```bash
    ollama pull llama3.2
    ```
3.  **MTGA Detailed Logs:** In MTGA, go to `Options -> Account` and enable **Detailed Logs (Plugin Support)**. This is **essential** for the application to receive game state data from the `Player.log` file.

### Step 2: Install Dependencies

Install the required Python packages using `pip` and the `requirements.txt` file. It is highly recommended to do this within a Python virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all dependencies
pip install -r requirements.txt
```

The `requirements.txt` file is split into core and optional dependencies. The command above installs everything. If you do not need the RAG (Retrieval-Augmented Generation) system, you can skip the optional dependencies.

### Step 3: Download and Generate Local Data

The application uses several local databases for card information, statistics, and rules. The following scripts must be run in order to download and prepare this data.

1.  **Download the MTG Comprehensive Rules:**
    This script downloads the official rules text, which is used by the RAG system for context-aware advice.

    ```bash
    python download_rules.py
    ```
    This will create a `data/MagicCompRules.txt` file.

2.  **Download 17lands Card Statistics:**
    This script downloads draft and gameplay statistics from [17lands.com](https://www.17lands.com/). This data is used for card evaluation and mulligan advice.

    ```bash
    python manage_data.py --update-17lands
    ```
    This will create and populate the `data/card_stats.db` database. This may take a few minutes as it downloads data for all current Standard sets.

3.  **Populate the Scryfall Card Cache:**
    This script pre-populates the local Scryfall database with card information for all the cards found in the 17lands database. This ensures that the application has all the necessary card data cached locally.

    ```bash
    python manage_data.py --update-scryfall
    ```
    This will create and populate the `data/scryfall_cache.db` database. This can take a significant amount of time as it makes many requests to the Scryfall API.

4.  **Initialize the RAG Rules Database:**
    This final script processes the downloaded rules text and creates a vector database, which allows the RAG system to perform semantic searches over the rules.

    ```bash
    python initialize_rules.py
    ```
    This will create the `data/chromadb/` directory and the vector database within it.

### Step 4: Run the Application

Once you have completed the setup, you can run the advisor:

```bash
python advisor.py
```

The application will automatically attempt to locate the `Player.log` file and will start monitoring it. When a game begins and you have a decision to make, the advisor will automatically provide spoken advice.

---

## Data Management

The `manage_data.py` script can be used to keep your local data up to date.

*   **Update 17lands Data:**
    ```bash
    python manage_data.py --update-17lands
    ```
*   **Update All Sets (not just Standard):**
    ```bash
    python manage_data.py --update-17lands --all-sets
    ```
*   **Update Scryfall Cache:**
    ```bash
    python manage_data.py --update-scryfall
    ```
*   **See all options:**
    ```bash
    python manage_data.py --help
    ```
