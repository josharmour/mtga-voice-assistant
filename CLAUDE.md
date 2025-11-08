# MTG AI Development Project - Claude Code Guide

## Project Overview

This is a comprehensive Magic: The Gathering AI development project with two main components:

1. **MTGA Voice Advisor** - A real-time tactical advisor for MTG Arena (production-ready)
2. **MTG Gameplay AI** - A neural network for gameplay decision-making (research/development)

**Status**: Phase 1-3 Complete (Data Processing, State Encoding, Model Architecture), Ready for Phase 4 (Training)

## Quick Start

### Development Setup
```bash
# Navigate to project
cd /home/joshu/logparser

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install optional packages
pip install kokoro-onnx chromadb sentence-transformers numpy scipy

# Download initial data
python3 manage_data.py --update-17lands
python3 manage_data.py --update-scryfall
```

### Running the Application
```bash
# GUI mode (recommended)
python3 app.py

# TUI mode (terminal)
python3 app.py --tui

# CLI mode (simple output)
python3 app.py --cli

# Test AI components
python3 demo_training_pipeline.py
python3 test_mtg_transformer.py
```

## Architecture Overview

### Core System Components

#### 1. MTGA Voice Advisor (Production System)
- **app.py** (1,963 lines) - Main orchestrator
- **mtga.py** (1,187 lines) - Log parsing and game state management
- **ai.py** (1,537 lines) - LLM integration with RAG
- **ui.py** (1,831 lines) - GUI/TUI/CLI interfaces
- **data_management.py** (1,141 lines) - Thread-safe database operations

#### 2. MTG Gameplay AI (Research System)
- **mtg_transformer_encoder.py** - Neural network for game state encoding
- **mtg_action_space.py** - Action representation and scoring
- **mtg_decision_head.py** - Actor-critic decision making
- **mtg_training_pipeline.py** - Complete training infrastructure
- **mtg_evaluation_metrics.py** - Performance evaluation

### Data Pipeline Architecture

#### Phase 1: Data Processing (✅ Complete)
1. **Data Ingestion** - 17Lands replay data download and parsing
2. **Action Normalization** - Raw actions → normalized sequences (21 types)
3. **Decision Extraction** - Strategic decision points with context
4. **Outcome Weighting** - Training examples with importance weights

#### Phase 2: State Encoding (✅ Complete)
1. **Board Tokenization** - 47-token vocabulary for permanents
2. **Hand/Mana Encoding** - 170-dimensional hand/mana vectors
3. **Phase/Priority Encoding** - 48-dimensional turn structure
4. **Complete State Tensors** - 282-dimension fused representations

#### Phase 3: Model Architecture (✅ Complete)
1. **Transformer Encoder** - Multi-modal state processing
2. **Action Space** - 16 action types with validity checking
3. **Decision Head** - Actor-critic architecture with attention
4. **Training Setup** - Outcome-weighted loss functions

#### Phase 4: Training (⏳ Next)
1. **Training Loop** - Implementation pending
2. **Evaluation Metrics** - Performance validation
3. **Hyperparameter Tuning** - Systematic optimization
4. **Final Training** - Scaling to full dataset

### Key Design Patterns

#### Thread Safety
- All SQLite operations use thread-local storage
- Database connections are thread-specific
- `check_same_thread=False` for concurrent access

#### Data Flow
```
MTGA Game → Player.log → LogFollower → MatchScanner → BoardState
          ↓
    GameStateManager → app.py._on_new_line()
          ↓
    GroundedPromptBuilder + RAGSystem
          ↓
    OllamaClient (LLM)
          ↓
    AIAdvisor (analysis)
          ↓
    TextToSpeech + UI
          ↓
    User (advice + voice)
```

#### Module Dependencies
```
app.py (main)
├── mtga.py (log parsing)
├── ai.py (LLM)
├── ui.py (output)
├── data_management.py (databases)
├── config_manager.py (preferences)
└── draft_advisor.py (picks)

ai.py
├── data_management.py (card info)
├── card_rag.py (formatting)
└── constants.py (sets)

ui.py (standalone, no deps)

data_management.py (sqlite3)

config_manager.py (json)
```

## Development Commands and Workflows

### Common Tasks

#### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mtga.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run verbose
pytest -v

# Run specific test
pytest tests/test_mtga.py::test_log_follower
```

#### Testing Components
```bash
# Test card database
python3 -c "from data_management import ArenaCardDatabase; db = ArenaCardDatabase(); print(db.get_card_name(1))"

# Test Ollama connection
python3 -c "from ai import OllamaClient; c = OllamaClient(); print(c.is_running())"

# Test log parser
python3 -c "from mtga import LogFollower; print('LogFollower OK')"

# Test AI components
python3 -c "from mtg_transformer_encoder import MTGTransformerEncoder; print('Transformer OK')"
```

#### Updating Data
```bash
# Update 17lands statistics
python3 manage_data.py --update-17lands

# Update all sets
python3 manage_data.py --update-17lands --all-sets

# Update Scryfall cache
python3 manage_data.py --update-scryfall

# Show status
python3 manage_data.py --status
```

### Code Quality

#### Style Checking
```bash
# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py

# All together
black . && flake8 . && mypy .
```

### Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Log File Analysis
```bash
# View logs
tail -f logs/advisor.log

# Search for errors
grep -i error logs/advisor.log

# Count log levels
grep ERROR logs/advisor.log | wc -l
grep WARNING logs/advisor.log | wc -l
```

## Key Components and Relationships

### Data Management System

#### Thread-Safe Database Classes
- **ArenaCardDatabase** - Unified card database (22,700+ cards)
- **CardStatsDB** - 17lands statistics (thread-local)
- **ScryfallDB** - Scryfall API cache
- **Database Pattern**: All classes use `threading.local()` for per-thread connections

#### Database Schema
```
unified_cards.db (~3.7 MB) - Arena cards with reskin support
card_stats.db (~100 MB) - 17lands performance statistics
scryfall_cache.db - Scryfall API responses
chromadb/ - Vector embeddings for MTG rules
```

### AI System Architecture

#### Multi-Modal Transformer Encoder
- **Input**: 282-dimension state tensor
  - Board tokens (64-dim)
  - Hand/mana (128-dim)
  - Phase/priority (64-dim)
  - Additional features (26-dim)
- **Architecture**: Multi-head attention (8 heads), 6 layers
- **Output**: 128-dimensional state representations

#### Action Space Representation
- **16 Action Types**: PLAY_LAND, CAST_CREATURE, DECLARE_ATTACKERS, etc.
- **82-dim Action Encodings**: Neural network processing
- **Validity Checking**: Mana, timing, targeting restrictions
- **Integration**: Direct compatibility with transformer outputs

#### Decision Head Architecture
- **Actor-Critic Design**: Separate policy and value networks
- **Attention-Based Explainability**: Attention weights for reasoning
- **Adaptive Scoring**: Game-context-aware action ranking
- **Real-time Inference**: ~0.20ms per decision

### Training Pipeline

#### Data Preparation
```json
{
  "tensor_data": [282-dimension state vector],
  "action_label": [16-dimension one-hot vector],
  "outcome_weight": [decision importance weight],
  "game_outcome": [boolean result],
  "decision_type": [strategic context],
  "strategic_context": [additional game state]
}
```

#### Training Configuration
- **Batch Size**: 32 (development), 128 (production)
- **Learning Rate**: 1e-4 with warmup
- **Loss Functions**: 
  - Action classification (cross-entropy)
  - Value estimation (MSE)
  - Outcome weighting (importance-scaled)
- **Curriculum Learning**: 4-stage progressive training

## Configuration and Deployment

### Environment Variables
```bash
# Ollama settings
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODEL=mistral:7b

# Application settings
export MTGA_ADVISOR_MODE=gui  # gui, tui, cli
export MTGA_LOG_PATH=/custom/path/to/Player.log

# Performance
export MTGA_ADVISOR_WORKERS=4
```

### User Preferences
Location: `~/.mtga_advisor/preferences.json`
```json
{
  "window_geometry": "1200x800",
  "ui_theme": "dark",
  "voice_engine": "kokoro",
  "voice_volume": 100,
  "ollama_model": "mistral:7b",
  "use_rag": true
}
```

### Dependencies

#### Core Dependencies
```
requests>=2.31.0
urllib3>=2.0.0
tabulate>=0.9.0
termcolor>=2.3.0
numpy>=1.24.0
```

#### Optional Dependencies
```
# TTS Engines
kokoro-onnx>=0.1.0
scipy>=1.11.0
transformers>=4.30.0
torch>=2.0.0

# RAG System
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Development
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
```

## Project Structure

### Directory Layout
```
logparser/
├── app.py                    # Main orchestrator
├── mtga.py                   # Log parsing & game state
├── ai.py                     # LLM & RAG system
├── ui.py                     # GUI/TUI/CLI interfaces
├── draft_advisor.py          # Draft pick recommendations
├── data_management.py        # Database management
├── config_manager.py         # User preferences
├── card_rag.py               # Card formatting for RAG
├── constants.py              # MTG set constants
├── advisor.py                # Launcher wrapper
├── launch_advisor.sh         # Shell launcher
├── requirements.txt          # Python dependencies
├── CLAUDE.md                 # This file
├── README.md                 # General documentation
├── data/                     # SQLite DBs, ChromaDB, CSVs
│   ├── unified_cards.db      # Arena cards
│   ├── card_stats.db         # 17lands stats
│   ├── scryfall_cache.db     # Scryfall cache
│   ├── chromadb/             # Vector embeddings
│   └── 17lands_*.csv         # Raw statistics
├── tests/                    # Test suite
├── tools/                    # Development utilities
└── logs/                     # Runtime logs

# MTG AI Specific
├── mtg_*.py                  # AI model components
├── action_*.py               # Data processing
├── decision_*.py             # Decision extraction
├── training_*.py             # Training pipeline
├── test_*.py                 # AI component tests
└── data/*.json              # Training datasets
```

## Important Architectural Patterns

### Graceful Degradation
- **Ollama optional**: Shows warnings if not running, continues without advice
- **RAG optional**: Works without ChromaDB/sentence-transformers
- **TTS optional**: Kokoro falls back to BarkTTS, then disables
- **Draft advisor optional**: Works without numpy/scipy

### Producer-Consumer Pipeline
- **LogFollower** (producer) yields lines
- **GameStateManager** (consumer) parses them
- **AIAdvisor** (consumer) analyzes state
- **UI** (consumer) displays results

### Data Grounding
- **All LLM context is cited** with sources
- **Rules** from official MTG Comprehensive Rules
- **Statistics** from 17lands
- **Reduces hallucinations** through verifiable sources

## Development Guidelines

### Adding New Features
1. **Plan**: Understand requirements, design solution, check dependencies
2. **Implement**: Write code, add logging, handle errors
3. **Test**: Unit tests, integration tests, manual testing
4. **Document**: Code comments, docstrings, update README
5. **Review**: Code quality, performance, thread safety

### Code Style
- **Follow PEP 8**: Consistent formatting, meaningful variable names
- **Type Hints**: Use `typing` module for better code documentation
- **Docstrings**: Comprehensive docstrings for all public methods
- **Error Handling**: Graceful degradation with meaningful error messages

### Performance Considerations
- **Database Optimization**: Use thread-local storage for SQLite
- **Memory Management**: Efficient data structures and caching
- **Async Operations**: Consider async for I/O operations
- **Profiling**: Use cProfile for identifying bottlenecks

## Current Status and Next Steps

### Completed (Phases 1-3)
- ✅ **Phase 1**: Data processing pipeline complete
- ✅ **Phase 2**: State encoding with 282-dim tensors
- ✅ **Phase 3**: Model architecture with transformer, action space, decision head

### In Progress (Phase 4)
- ⏳ **Training Loop**: Implementation pending
- ⏳ **Evaluation Metrics**: Performance validation
- ⏳ **Hyperparameter Tuning**: Systematic optimization
- ⏳ **Final Training**: Scaling to full dataset

### Future Work (Phase 5)
- 🔄 **Inference Engine**: Fast decision pipeline
- 🔄 **Bot Integration**: MTG Arena integration
- 🔄 **Deployment**: Production-ready packaging

### Key Metrics
- **Dataset**: 50 games (development), 450K games (production target)
- **Model Size**: Tiny (50K params) to Large (3M params)
- **Training Time**: ~1 hour on 50 games, ~2 days on 450K games
- **Inference Speed**: ~0.20ms per decision

## Resources and References

### External Data Sources
- **17lands.com** - Draft statistics and gameplay data
- **Scryfall.com** - Card metadata and images
- **MTG Comprehensive Rules** - Official rules text

### Key Documentation
- **README.md**: General user documentation
- **DEVELOPMENT.md**: Developer guide and workflow
- **SETUP.md**: Installation and configuration guide
- **tasks.md**: Detailed task tracking and status

### Dependencies and Tools
- **Ollama** - Local LLM backend
- **ChromaDB** - Vector database for RAG
- **Kokoro/BarkTTS** - Text-to-speech engines
- **TensorBoard** - Training monitoring
- **SQLite** - Local data storage

## Troubleshooting

### Common Issues
1. **"Module not found"**: Ensure virtual environment is activated
2. **"Ollama not running"**: Start Ollama service: `ollama serve`
3. **"Card database not found"**: Run `python3 manage_data.py --update-17lands`
4. **"Can't find Player.log"**: Enable detailed logs in MTGA settings
5. **Thread safety errors**: Verify `check_same_thread=False` in SQLite connections

### Performance Optimization
- **Profile bottlenecks** with cProfile
- **Cache frequently accessed** data
- **Batch operations** when possible
- **Use appropriate data structures**

---

**Last Updated**: November 8, 2025
**Status**: Phases 1-3 Complete, Ready for Phase 4 Training
**Next Focus**: Model training and evaluation
