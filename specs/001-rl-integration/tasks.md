---

description: "Task list for Reinforcement Learning Integration feature implementation"
---

# Tasks: Reinforcement Learning Integration

**Input**: Design documents from `/specs/001-rl-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)
**Feature Branch**: `001-rl-integration`
**Tests**: Test tasks included for comprehensive validation per constitutional requirements

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths follow plan.md structure for MTGA Voice Advisor integration

## User Stories 2 & 3 Status Update

**Current Implementation Status**: User Stories 2 and 3 are **future work** and not part of current implementation scope.

**Rationale**: Current implementation focuses on User Story 1 (Enhanced AI Decision Quality) with complete Conservative Q-Learning integration. Tasks T044-T073 for User Stories 2 and 3 remain pending and should be moved to separate specification for future development phases.

**Action**: Consider creating separate specification document for "RL Integration - Phase 2" covering adaptive opponent recognition and multi-turn strategic planning.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure for RL integration

- [X] T001 Create RL directory structure per implementation plan in src/rl/
- [X] T002 [P] Install RL-specific dependencies (torch, gymnasium, stable-baselines3)
- [X] T003 [P] Configure RL environment and device detection in src/rl/utils/device_manager.py
- [X] T004 Create RL-specific configuration system in src/config/rl_config.py
- [X] T005 [P] Setup model registry database schema in data/model_registry.db

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core RL infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Implement base RL algorithm classes in src/rl/algorithms/base.py
- [X] T007 [P] Create enhanced state representation system in src/rl/data/state_extractor.py (380+ dimensions)
- [X] T008 [P] Implement multi-dimensional reward function in src/rl/data/reward_function.py
- [X] T009 Create prioritized experience replay buffer in src/rl/data/replay_buffer.py
- [X] T010 [P] Implement Conservative Q-Learning algorithm in src/rl/algorithms/cql.py
- [X] T011 Create dueling DQN model architecture in src/rl/models/dueling_dqn.py
- [X] T012 [P] Implement model components in src/rl/models/components/
- [X] T013 Create RL training pipeline in src/rl/training/trainer.py
- [X] T014 [P] Implement curriculum learning system in src/rl/training/curriculum.py
- [X] T015 Create real-time inference engine in src/rl/inference/engine.py
- [X] T016 [P] Implement explainability system in src/rl/inference/explainability.py
- [X] T017 Create model versioning system in src/rl/utils/model_registry.py
- [X] T018 Implement RL integration layer in src/core/rl_integration.py
- [X] T019 [P] Add RL data validation in src/data/rl_validator.py
- [X] T020 Configure RL performance monitoring and logging
- [X] T021 [P] Implement elastic weight consolidation (EWC) for catastrophic forgetting prevention in src/rl/training/continual_learning.py
- [X] T022 [P] Create experience replay buffer with episodic memory segregation in src/rl/data/continual_replay_buffer.py
- [X] T023 Implement progressive neural networks for knowledge preservation in src/rl/models/progressive_networks.py
- [X] T024 [P] Implement domain adaptation system for new MTG sets in src/rl/training/domain_adaptation.py
- [X] T025 [P] Create generalization testing framework for unseen cards/mechanics in tests/rl/test_generalization.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Enhanced AI Decision Quality (Priority: P1) 🎯 MVP

**Goal**: RL agent provides significantly better tactical advice through Conservative Q-Learning on 17Lands replay data

**Independent Test**: Compare RL agent recommendations against supervised baseline on held-out 17Lands games, measuring 25-40% win rate improvement

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T026 [P] [US1] Unit test for Conservative Q-Learning algorithm in tests/rl/test_algorithms.py
- [X] T027 [P] [US1] Unit test for enhanced state representation in tests/rl/test_models.py
- [X] T028 [P] [US1] Unit test for multi-dimensional reward function in tests/rl/test_inference.py
- [X] T029 [P] [US1] Integration test for RL training pipeline in tests/integration/test_rl_training.py
- [X] T030 [P] [US1] Performance benchmark test for sub-100ms inference latency in tests/performance/test_rl_latency.py
- [X] T031 [P] [US1] Statistical validation test for >95% confidence win rate improvement in tests/rl/test_validation.py
- [X] T032 [P] [US1] Unit test for continual learning without catastrophic forgetting in tests/rl/test_continual_learning.py
- [X] T033 [P] [US1] Integration test for knowledge preservation across training episodes in tests/integration/test_continual_integration.py

### Implementation for User Story 1

- [X] T034 [P] [US1] Implement 380+ dimensional state representation extraction in src/rl/data/state_extractor.py
- [X] T035 [P] [US1] Create multi-component reward function in src/rl/data/reward_function.py (game outcome, life/card/board advantage, tempo, strategic progress)
- [X] T036 [US1] Implement Conservative Q-Learning training logic in src/rl/algorithms/cql.py (depends on T034, T035)
- [X] T037 [P] [US1] Build dueling DQN architecture for 380-dim state processing in src/rl/models/dueling_dqn.py
- [X] T038 [US1] Create offline RL training pipeline for 17Lands data in src/rl/training/trainer.py
- [X] T039 [P] [US1] Implement real-time inference engine with sub-100ms latency in src/rl/inference/engine.py
- [X] T040 [US1] Integrate RL agent with existing MTGA Voice Advisor in src/core/rl_integration.py
- [X] T041 [US1] Add graceful degradation to supervised baseline in src/core/rl_integration.py
- [X] T042 [P] [US1] Implement performance monitoring and optimization for real-time usage
- [X] T043 [US1] Add comprehensive logging for RL decision quality metrics

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Adaptive Opponent Strategy Recognition (Priority: P2)

**Goal**: AI system adapts recommendations based on opponent playing style, deck archetype, and observed patterns

**Independent Test**: Simulate games against different opponent archetypes and measure adaptation quality through win rate improvement

### Tests for User Story 2 ⚠️

- [ ] T044 [P] [US2] Unit test for opponent pattern recognition in tests/rl/test_adaptive.py
- [ ] T045 [P] [US2] Integration test for archetype-based strategy adaptation in tests/integration/test_adaptive_rl.py
- [ ] T046 [P] [US2] Performance test for real-time opponent adaptation in tests/performance/test_adaptive_latency.py

### Implementation for User Story 2

- [ ] T047 [P] [US2] Create opponent pattern detection system in src/rl/data/opponent_analyzer.py
- [ ] T048 [P] [US2] Implement deck archetype classification in src/rl/data/archetype_classifier.py
- [ ] T049 [US2] Build adaptive strategy selection system in src/rl/algorithms/adaptive_cql.py
- [ ] T050 [US2] Integrate opponent recognition with RL decision making in src/core/rl_integration.py
- [ ] T051 [P] [US2] Add real-time opponent model updates during gameplay
- [ ] T052 [US2] Implement metagame-aware strategy adaptation

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Multi-Turn Strategic Planning (Priority: P3)

**Goal**: AI system provides recommendations considering 2-3 turn future game states and strategic planning

**Independent Test**: Evaluate recommendations across multi-turn scenarios and compare outcomes against short-term optimized decisions

### Tests for User Story 3 ⚠️

- [ ] T053 [P] [US3] Unit test for multi-turn state prediction in tests/rl/test_planning.py
- [ ] T054 [P] [US3] Integration test for strategic planning horizon in tests/integration/test_strategic_planning.py
- [ ] T055 [P] [US3] Performance test for multi-turn inference speed in tests/performance/test_planning_latency.py

### Implementation for User Story 3

- [ ] T056 [P] [US3] Create multi-turn state prediction system in src/rl/data/state_predictor.py
- [ ] T057 [P] [US3] Implement strategic planning algorithm in src/rl/algorithms/strategic_planner.py
- [ ] T058 [US3] Build 2-3 turn horizon evaluation system in src/rl/inference/planning_engine.py
- [ ] T059 [US3] Integrate strategic planning with RL recommendations in src/core/rl_integration.py
- [ ] T060 [P] [US3] Add trade-off explanation system for long-term vs short-term decisions
- [ ] T061 [US3] Implement resource allocation optimization across multiple turns

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and system integration

- [ ] T062 [P] Add comprehensive documentation for RL system in docs/rl_integration.md
- [ ] T063 [P] Code cleanup and refactoring across all RL components
- [ ] T064 Performance optimization across all user stories for sub-100ms latency guarantee
- [ ] T065 [P] Additional unit tests for edge cases and error conditions in tests/rl/
- [ ] T066 [P] Integration tests for full system with all user stories in tests/integration/test_full_rl_system.py
- [ ] T067 Implement comprehensive error handling and graceful degradation
- [ ] T068 [P] Add configuration validation and system health checks
- [ ] T069 Performance profiling and optimization for production deployment
- [ ] T070 [P] Final validation of all constitutional requirements (testing, explainability, performance)
- [ ] T071 [P] Update main application to integrate RL capabilities in src/core/app.py
- [ ] T072 [P] Performance degradation test for continuous learning in tests/performance/test_continual_degradation.py
- [ ] T073 Run comprehensive system validation and quickstart scenarios

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Data processing components before algorithms
- Algorithm implementation before inference systems
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Data processing tasks within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for Conservative Q-Learning algorithm in tests/rl/test_algorithms.py"
Task: "Unit test for enhanced state representation in tests/rl/test_models.py"
Task: "Unit test for multi-dimensional reward function in tests/rl/test_inference.py"
Task: "Integration test for RL training pipeline in tests/integration/test_rl_training.py"
Task: "Performance benchmark test for sub-100ms inference latency in tests/performance/test_rl_latency.py"
Task: "Statistical validation test for >95% confidence win rate improvement in tests/rl/test_validation.py"

# Launch all data processing for User Story 1 together:
Task: "Implement 380+ dimensional state representation extraction in src/rl/data/state_extractor.py"
Task: "Create multi-component reward function in src/rl/data/reward_function.py"
Task: "Build dueling DQN architecture for 380-dim state processing in src/rl/models/dueling_dqn.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently with 17Lands data
5. Deploy/demo enhanced AI decision quality

### Incremental Delivery

1. Complete Setup + Foundational → RL foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP with 25-40% win rate improvement)
3. Add User Story 2 → Test independently → Deploy/Demo (adaptive opponent recognition)
4. Add User Story 3 → Test independently → Deploy/Demo (multi-turn strategic planning)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Enhanced AI Decision Quality)
   - Developer B: User Story 2 (Adaptive Opponent Recognition)
   - Developer C: User Story 3 (Multi-Turn Strategic Planning)
3. Stories complete and integrate independently

---

## Constitutional Requirements Validation

### Data-Driven AI Development
- All RL models include quantitative validation using 17Lands datasets (T024, T026)
- Performance metrics clearly defined (25-40% win rate improvement in T026)
- Measurable validation against supervised baseline (T026)

### Real-Time Responsiveness (NON-NEGOTIABLE)
- Sub-100ms inference latency explicitly required (T025, T032, T044, T057)
- Performance benchmarks for real-time gameplay advisory (T025)
- GPU/CPU fallback for performance requirements (T003, T020)

### Verifiable Testing Requirements
- Comprehensive testing methodology defined (T021-026, T037-039, T046-048)
- 80%+ code coverage requirement (T058)
- Red-Green-Refactor cycle for ML model development (Tests before implementation)
- Statistical validation with >95% confidence (T026)

### Graceful Degradation Architecture
- Fallback to supervised model when RL confidence low (T034)
- CPU-only inference when GPU unavailable (T003)
- Local processing without external dependencies (all components)

### Explainable AI (XAI) First
- Attention weight visualization required (T016)
- Decision rationale for user understanding (T016, T053)
- Human-understandable quality rating target >8/10 (validated in T026)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (Red-Green-Refactor)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- All performance requirements are constitutionally mandated and must be validated
- RL integration must maintain existing MTGA Voice Advisor functionality