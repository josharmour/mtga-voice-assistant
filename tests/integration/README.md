# Continual Learning Integration Tests

This directory contains comprehensive integration tests for the MTG AI continual learning system, specifically designed to validate knowledge preservation across training episodes.

## Purpose

These tests validate the integration between:
- Elastic Weight Consolidation (EWC)
- Progressive Networks
- Domain Adaptation
- 17Lands Data Integration
- Model Versioning and Rollback
- Catastrophic Forgetting Detection
- Performance Monitoring

## Test Structure

### Main Test Classes

#### `TestContinualLearningIntegration`
Core integration tests that validate the complete continual learning system:

1. **T033-001: System Initialization** - Tests proper initialization of all continual learning components
2. **T033-002: Knowledge Preservation** - Validates knowledge retention across multiple MTG sets
3. **T033-003: EWC + Progressive Networks** - Tests integration between memory consolidation mechanisms
4. **T033-004: Domain Adaptation** - Validates 17Lands data integration and metagame adaptation
5. **T033-005: Model Versioning** - Tests versioning and rollback capabilities
6. **T033-006: Forgetting Detection** - Tests catastrophic forgetting detection and mitigation
7. **T033-007: Graceful Degradation** - Tests performance under resource constraints
8. **T033-008: End-to-End Workflow** - Complete pipeline validation

#### `TestContinualLearningRealData`
Tests with actual 17Lands data when available:
- **T033-009: Real Data Integration** - Tests with real-world data

#### `TestContinualLearningPerformance`
Performance and scalability tests:
- **T033-010: Training Scalability** - Tests scaling with dataset size
- **T033-011: Memory Efficiency** - Tests memory usage optimization

## Red-Green-Refactor Approach

These tests are designed to **FAIL INITIALLY** (Red phase) and drive the implementation of the continual learning components:

### Phase 1: Red (Current)
- Tests import non-existent modules
- All tests fail with import errors
- Validates test structure and logic

### Phase 2: Green (Next)
- Implement minimal continual learning components
- Make tests pass with basic functionality
- Focus on integration rather than optimization

### Phase 3: Refactor (Future)
- Optimize performance
- Improve code quality
- Add additional features

## Running Tests

### With pytest (Recommended)
```bash
# Run all integration tests
pytest tests/integration/test_continual_integration.py -v

# Run specific test
pytest tests/integration/test_continual_integration.py::TestContinualLearningIntegration::test_initialization_continual_learning_system -v

# Run with coverage
pytest tests/integration/test_continual_integration.py --cov=src.rl --cov-report=html
```

### Without pytest (Validation)
```bash
# Validate test structure without running
python3 tests/integration/test_runner.py
```

## Dependencies

The tests expect the following modules to exist (will be implemented):

### Core RL Components
- `src.rl.continual_learning_manager.ContinualLearningManager`
- `src.rl.elastic_weight_consolidation.ElasticWeightConsolidation`
- `src.rl.progressive_networks.ProgressiveNetworks`
- `src.rl.domain_adaptation.DomainAdaptation`
- `src.rl.evaluation_metrics.EvaluationMetrics`
- `src.rl.model_versioning.ModelVersioning`
- `src.rl.training_monitor.TrainingMonitor`

### Data Pipeline
- `src.rl.data_pipeline.SeventeenLandsDataPipeline`

### Existing Components
- `src.core.mtga.GameStateManager`
- `src.mtg_ai.mtg_transformer_encoder.MTGTransformerEncoder`
- `src.mtg_ai.mtg_decision_head.MTGDecisionHead`

## Test Data

### Mock Datasets
Tests automatically generate synthetic datasets for multiple MTG sets:
- DMU (Dominaria United)
- NEO (Kamigawa: Neon Dynasty)
- SNC (Streets of New Capenna)
- ONE (One Piece)

Each dataset includes:
- 282-dimensional state tensors
- Action labels (16-dimensional)
- Outcome weights
- Strategic context

### Real Data (Optional)
When available, tests can use real 17Lands data:
- PremierDraft statistics
- TraditionalDraft data
- Sealed performance metrics

## Expected Test Failures

During the Red phase, expect these import errors:
```
ImportError: No module named 'src.rl.continual_learning_manager'
ImportError: No module named 'src.rl.elastic_weight_consolidation'
ImportError: No module named 'src.rl.progressive_networks'
ImportError: No module named 'src.rl.domain_adaptation'
ImportError: No module named 'src.rl.evaluation_metrics'
ImportError: No module named 'src.rl.model_versioning'
ImportError: No module named 'src.rl.data_pipeline'
```

These failures are **INTENTIONAL** and indicate the Red-Green-Refactor approach is working correctly.

## Implementation Priority

Based on test dependencies, implement components in this order:

1. **High Priority** (Required for most tests)
   - `ContinualLearningManager`
   - `EvaluationMetrics`
   - `ModelVersioning`

2. **Medium Priority** (Core continual learning)
   - `ElasticWeightConsolidation`
   - `ProgressiveNetworks`
   - `DomainAdaptation`

3. **Lower Priority** (Enhancements)
   - `TrainingMonitor`
   - `SeventeenLandsDataPipeline`

## Test Coverage

### Integration Aspects Covered
- ✅ EWC constraint application
- ✅ Progressive network column creation
- ✅ Domain adaptation mechanisms
- ✅ Model versioning and rollback
- ✅ Catastrophic forgetting detection
- ✅ Graceful degradation handling

### Performance Metrics
- Knowledge preservation across sets
- Performance stability metrics
- Memory usage optimization
- Training time scalability
- Cross-domain transfer ratios

### Edge Cases
- Resource constraint handling
- Performance degradation scenarios
- Missing data fallbacks
- System recovery mechanisms

## Configuration

Tests use mock configurations that can be easily modified:

```python
mock_config = {
    "workspace": "/tmp/test_workspace",
    "model_config": {
        "state_dim": 282,
        "action_dim": 16,
        "hidden_dim": 128
    },
    "continual_learning": {
        "ewc_lambda": 1000.0,
        "progressive_columns": 3,
        "forgetting_threshold": 0.15,
        "performance_threshold": 0.05
    }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Expected during Red phase
2. **Missing Test Data**: Tests generate synthetic data automatically
3. **Resource Constraints**: Tests use minimal data for performance

### Debug Mode
Set environment variable for verbose output:
```bash
export DEBUG_CONTINUAL_TESTS=1
python3 tests/integration/test_runner.py
```

## Future Enhancements

### Additional Test Scenarios
- Multi-task learning across different game formats
- Long-term knowledge retention studies
- Real-time adaptation during gameplay
- Distributed training scenarios

### Performance Benchmarks
- Baseline performance metrics
- Scaling studies with larger datasets
- Memory optimization validation
- Real-time inference performance

## Integration with CI/CD

These tests are designed to integrate with continuous integration:
- Fast feedback during Red phase (import failures)
- Comprehensive validation during Green phase
- Performance monitoring during Refactor phase

---

**Status**: Red phase complete - tests created and structured for implementation
**Next**: Implement core continual learning components to achieve Green phase
**Dependencies**: None (tests are self-contained with synthetic data)