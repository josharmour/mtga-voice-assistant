# MTGA Voice Assistant (Unified) - Revision 0.3.0

![MTGA Voice Advisor Screenshot](https://github.com/josharmour/mtga-voice-assistant/raw/main/mtga-voice-assistant.png)

**MTGA Voice Assistant** is a professional-grade, real-time tactical advisor for *Magic: The Gathering Arena* (MTGA). It combines a high-fidelity rules-based logic engine (from the ArenaMCP project) with a polished, interactive GUI and high-quality local Text-to-Speech (TTS).

The "Unified Project" (Major Revision 0.3.0) represents a "Brain Transplant" where the logic of `ArenaMCP` was merged into the `mtga-voice-assistant` to provide pro-level strategic coaching without sacrificing user experience.

## Key Features

*   **🧠 High-Fidelity Logic Engine:** Powered by the `ArenaMCP` core, providing deterministic game state tracking, a full rules engine, and advanced strategic analysis.
*   **🎙️ Professional TTS:** Speaks advice using **Kokoro (local, ONNX)** for near-human quality, with support for multiple voices (e.g., Bella, Adam, Sarah).
*   **🧩 Modular AI Backends:**
    *   **Cloud:** Google Gemini 3.1, OpenAI GPT-4o, Anthropic Claude 3.5.
    *   **Subscription:** **Claude Code CLI** support (uses your existing Claude subscription, **no API key required**).
    *   **Local:** Ollama (Llama 3.2, etc.) and Llama.cpp.
*   **📡 MCP Server Integration:** Acts as a **Model Context Protocol (MCP)** server, allowing you to connect external AI agents (like Claude Desktop) directly to your live MTGA game state.
*   **📊 Draft & Sealed Assistance:** Real-time pick recommendations with 17Lands win-rate statistics and AI-generated reasoning.
*   **🛠️ Proactive Triggers:** Automatically detects key decision points (Combat, End of Turn, Complex Stacks) and provides advice without being asked.

## Installation

1.  **Prerequisites:**
    *   Python 3.10+ installed.
    *   MTG Arena installed with **Detailed Logs** enabled (`Options` -> `Account` -> check `Detailed Logs`).

2.  **Clone & Set up:**
    ```bash
    git clone https://github.com/josharmour/mtga-voice-assistant.git
    cd mtga-voice-assistant
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Optional: Claude Code (Recommended):**
    If you have a Claude Pro subscription, install the `claude` CLI:
    ```bash
    npm install -g @anthropic-ai/claude-code
    claude auth
    ```
    The advisor will automatically detect and use the `claude` CLI for "Free" (subscription-based) pro-level advice.

## Usage

1.  **Launch:**
    *   **Windows:** Run `start_advisor.bat` or `python main.py`.
    *   **Linux:** `python main.py`.

2.  **Configuration:**
    *   Select your **AI Provider** in the settings.
    *   Choose a **Voice** (Kokoro is recommended for local quality).
    *   Enable **Proactive Coaching** for hands-free advice.

3.  **MCP Integration (Advanced):**
    To connect Claude Desktop to your game:
    Add the following to your `claude_desktop_config.json`:
    ```json
    {
      "mcpServers": {
        "mtga": {
          "command": "python",
          "args": ["-m", "src.core.engine.server"]
        }
      }
    }
    ```

## Project Structure (Unified)

*   `main.py`: Unified entry point.
*   `src/core/engine/`: **The Brain** — Game state tracking, rules engine, and Coach logic (from `ArenaMCP`).
*   `src/core/ui.py`: **The Interface** — Polished Tkinter dashboard and voice controls.
*   `src/core/mtga.py`: **The Eyes** — Log following and legacy state management.
*   `src/data/`: **The Memory** — Card databases (Scryfall, Arena SQLite) and 17Lands stats.
*   `logs/`: Application and advisor logs.

## Development

*   **Tests:** Run `pytest` to verify engine logic and log parsing.
*   **Engine:** The core logic is maintained in `src/core/engine/`, designed for portability.

## License

[MIT License](LICENSE)
