# Implementation Plan: Reinforcement Learning Integration

**Branch**: `001-rl-integration` | **Date**: 2025-11-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-rl-integration/spec.md`

**Note**: This plan implements RL integration for MTG AI system following Conservative Q-Learning approach with 17Lands replay data

## Summary

Transform current supervised learning MTG AI system into reinforcement learning-capable agent using Conservative Q-Learning on 17Lands replay data. The implementation expands state representation from 23 to 380+ dimensions, processes 450K+ game episodes, and achieves 25-40% win rate improvement while maintaining sub-100ms inference latency for real-time gameplay advisory.

## Technical Architecture

### RL Integration Strategy
The RL system integrates with existing MTGA Voice Advisor through a modular architecture:

**Core Components:**
- **State Encoder**: Transformer-based encoder processing 380-dimensional game state vectors
- **Decision Engine**: Conservative Q-Learning (CQL) with dueling DQN architecture
  *Reference: Functional Requirement FR-002 in spec.md for detailed CQL specifications*
- **Inference Pipeline**: Sub-100ms real-time inference with CPU/GPU fallback
- **Explainability Module**: Attention weight visualization and decision rationale

**Data Flow:**
```
MTGA Game State → State Encoder → Q-Network → Action Selection → Explainability → UI Display
```

**Integration Points:**
- `src/core/rl_integration.py`: Main integration orchestrator
- `src/rl/`: New RL-specific modules
- `src/config/rl_config.py`: RL configuration management

**Language/Version**: Python 3.9+ (existing codebase)
**Primary Dependencies**: torch, gymnasium, stable-baselines3, numpy, pandas, SQLite
**Storage**: SQLite databases (unified_cards.db, card_stats.db, model_registry.db), local filesystem for models
**Testing**: pytest (unit tests), integration tests, performance benchmarks, model validation
**Target Platform**: Linux/Windows/macOS with optional CUDA GPU acceleration
**Project Type**: Single project with modular structure (RL components integrate into existing MTGA Voice Advisor)
**Performance Goals**: <100ms inference latency, >1000 state evaluations/second batch processing, <16GB RAM usage
**Constraints**: Real-time gameplay advisory, graceful degradation without GPU, local processing only
**Scale/Scope**: 450K+ game episodes, 380+ dimensional state space, 64+ action types, 10-50 model parameters

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Constitutional Compliance Validation

**I. Data-Driven AI Development**:
- ✅ All RL models require quantitative validation using 17Lands datasets (450K+ games)
- ✅ Performance metrics clearly defined (25-40% win rate improvement)
- ✅ Measurable validation against supervised baseline

**II. Real-Time Responsiveness (NON-NEGOTIABLE)**:
- ✅ Sub-100ms inference latency explicitly required (FR-005, SC-003)
- ✅ Performance benchmarks for real-time gameplay advisory
- ✅ GPU/CPU fallback for performance requirements

**III. Verifiable Testing Requirements**:
- ✅ Comprehensive testing methodology defined (unit, integration, performance tests)
- ✅ 80%+ code coverage requirement
- ✅ Red-Green-Refactor cycle for ML model development
- ✅ Statistical validation with >95% confidence (FR-009)

**IV. Graceful Degradation Architecture**:
- ✅ Fallback to supervised model when RL confidence low (FR-007)
- ✅ CPU-only inference when GPU unavailable
- ✅ Local processing without external dependencies

**V. Explainable AI (XAI) First**:
- ✅ Attention weight visualization required (FR-006)
- ✅ Decision rationale for user understanding
- ✅ Human-understandable quality rating target >8/10 (SC-006)

**🎉 ALL CONSTITUTIONAL REQUIREMENTS SATISFIED**

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Single Project Structure (existing MTGA Voice Advisor codebase)
src/
├── core/                              # Core MTGA Voice Advisor functionality
│   ├── app.py                         # Main orchestrator (existing)
│   ├── mtga.py                        # Log parsing and game state (existing)
│   ├── ai.py                          # LLM integration and RAG (existing)
│   ├── ui.py                          # GUI/TUI/CLI interfaces (existing)
│   ├── draft_advisor.py               # Draft recommendations (existing)
│   ├── deck_builder.py                # Deck building utilities (existing)
│   └── rl_integration.py              # RL integration layer (NEW)
│
├── rl/                                # NEW: Reinforcement Learning components
│   ├── algorithms/                    # RL algorithm implementations
│   │   ├── cql.py                     # Conservative Q-Learning (NEW)
│   │   └── base.py                    # Base RL algorithm classes (NEW)
│   ├── models/                        # Neural network models
│   │   ├── dueling_dqn.py             # Main RL model architecture (NEW)
│   │   └── components/                # Model components (NEW)
│   ├── training/                      # Training pipeline components
│   │   ├── trainer.py                 # Main training pipeline (NEW)
│   │   └── curriculum.py              # Curriculum learning (NEW)
│   ├── inference/                     # Inference and deployment
│   │   ├── engine.py                  # Real-time inference engine (NEW)
│   │   └── explainability.py          # Decision reasoning (NEW)
│   ├── data/                          # Data processing components
│   │   ├── replay_buffer.py           # Prioritized experience replay (NEW)
│   │   ├── state_extractor.py         # Enhanced state representation (NEW)
│   │   └── reward_function.py         # Multi-dimensional rewards (NEW)
│   └── utils/                         # RL utility functions
│       ├── device_manager.py          # GPU/CPU management (NEW)
│       └── model_registry.py          # Model versioning system (NEW)
│
├── mtg_ai/                            # Research/Development MTG AI models (existing)
│   ├── mtg_transformer_encoder.py     # Transformer state encoder (existing)
│   ├── mtg_action_space.py            # Action representation (existing)
│   ├── mtg_decision_head.py           # Actor-critic decision making (existing)
│   └── mtg_training_pipeline.py       # Training infrastructure (existing)
│
├── data/                              # Data management systems (existing)
│   ├── data_management.py             # Thread-safe database operations
│   ├── card_rag.py                    # Card RAG database system
│   └── rl_validator.py                # RL data validation (NEW)
│
└── config/                            # Configuration management (existing)
    ├── config_manager.py              # User preferences and settings
    ├── constants.py                   # Application constants
    └── rl_config.py                   # RL-specific configuration (NEW)

tests/
├── unit/                             # Unit tests (existing)
├── integration/                      # Integration tests (existing)
├── performance/                     # Performance and benchmark tests (existing)
└── rl/                              # RL-specific tests (NEW)
    ├── test_algorithms.py            # RL algorithm tests (NEW)
    ├── test_models.py                # Model architecture tests (NEW)
    └── test_inference.py             # Inference engine tests (NEW)
```

**Structure Decision**: Single project structure integrating new RL components (`src/rl/`) into existing MTGA Voice Advisor codebase while maintaining modular separation and clear integration points through `src/core/rl_integration.py`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
