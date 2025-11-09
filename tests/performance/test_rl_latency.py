"""
Performance benchmark tests for RL latency requirements

Tests sub-100ms inference latency as required by constitutional
compliance for real-time responsiveness.

This is Task T030 for User Story 1 - Enhanced AI Decision Quality.
Following Red-Green-Refactor approach - tests should FAIL initially.
"""

import pytest
import numpy as np
import time
import torch
import threading
import psutil
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

from src.rl.inference.engine import InferenceEngine, InferenceResult, PerformanceMetrics
from src.rl.data.state_extractor import StateExtractor, StateFeatureConfig, MTGGameState


class TestRLLatencyRequirements:
    """
    Test RL system meets constitutional latency requirements.

    Tests the InferenceEngine class with comprehensive validation of:
    - Sub-100ms inference latency (constitutional requirement)
    - GPU/CPU fallback performance testing
    - Performance under load and stress conditions
    - Memory usage and resource efficiency
    - Performance regression detection
    - Real-time inference engine compliance
    """

    @pytest.fixture
    def mock_inference_engine(self):
        """Create mock inference engine for latency testing."""
        with patch('src.rl.inference.engine.DuelingDQN') as mock_model_class:
            mock_model = Mock()

            # Mock forward pass with realistic timing that will FAIL constitutional requirements
            def mock_forward(x):
                # Simulate computation time that exceeds 100ms requirement
                time.sleep(0.150)  # 150ms - exceeds constitutional limit
                return torch.randn(x.shape[0], 64)

            mock_model.forward = mock_forward
            mock_model.eval = Mock()
            mock_model.parameters = Mock(return_value=[])
            mock_model.to = Mock(return_value=mock_model)

            mock_model_class.return_value = mock_model

            engine = InferenceEngine(max_latency_ms=90.0)
            engine.model = mock_model
            engine.model_loaded = True

            return engine

    @pytest.fixture
    def sample_game_states(self):
        """Create sample game states for comprehensive testing."""
        return {
            'simple_state': MTGGameState(
                life=20, mana_pool={'white': 2, 'blue': 2},
                hand=[{'name': 'Serra Angel', 'types': ['creature'], 'mana_cost': {'cmc': 4}}],
                library_count=50, graveyard_count=0, exile_count=0,
                battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=5, phase='main', step='main1', priority_player='player',
                active_player='player', storm_count=0,
                known_info={}, statistics={}, opponent_info={},
                timestamp=time.time(), format='standard', game_id='test_001'
            ),
            'complex_state': MTGGameState(
                life=15, mana_pool={'white': 5, 'blue': 3, 'black': 2, 'red': 1, 'green': 1, 'colorless': 4},
                hand=[
                    {'name': 'Serra Angel', 'types': ['creature'], 'mana_cost': {'cmc': 4}},
                    {'name': 'Lightning Bolt', 'types': ['instant'], 'mana_cost': {'cmc': 1}},
                    {'name': 'Counterspell', 'types': ['instant'], 'mana_cost': {'cmc': 2}},
                    {'name': 'Dark Ritual', 'types': ['instant'], 'mana_cost': {'cmc': 1}},
                    {'name': 'Giant Growth', 'types': ['instant'], 'mana_cost': {'cmc': 1}},
                    {'name': 'Forest', 'types': ['land'], 'mana_cost': {'cmc': 0}},
                    {'name': 'Island', 'types': ['land'], 'mana_cost': {'cmc': 0}}
                ],
                library_count=35, graveyard_count=8, exile_count=2,
                battlefield=[
                    {'name': 'Serra Angel', 'types': ['creature'], 'mana_cost': {'cmc': 4}, 'power': 4, 'toughness': 4, 'tapped': False},
                    {'name': 'Llanowar Elves', 'types': ['creature'], 'mana_cost': {'cmc': 1}, 'power': 1, 'toughness': 1, 'tapped': True},
                    {'name': 'Forest', 'types': ['land'], 'mana_cost': {'cmc': 0}, 'tapped': False}
                ],
                lands=[
                    {'name': 'Forest', 'types': ['land'], 'mana_cost': {'cmc': 0}, 'tapped': False},
                    {'name': 'Island', 'types': ['land'], 'mana_cost': {'cmc': 0}, 'tapped': True}
                ],
                creatures=[
                    {'name': 'Serra Angel', 'types': ['creature'], 'mana_cost': {'cmc': 4}, 'power': 4, 'toughness': 4, 'tapped': False},
                    {'name': 'Llanowar Elves', 'types': ['creature'], 'mana_cost': {'cmc': 1}, 'power': 1, 'toughness': 1, 'tapped': True}
                ],
                artifacts_enchantments=[
                    {'name': 'Holy Armor', 'types': ['enchantment'], 'mana_cost': {'cmc': 1}}
                ],
                turn_number=12, phase='combat', step='declare_attackers', priority_player='player',
                active_player='player', storm_count=1,
                known_info={'recent_actions': [{'type': 'cast_spell'}, {'type': 'attack'}]},
                statistics={'avg_cmc_cast': 2.5, 'land_drops': 8, 'spells_cast': 5},
                opponent_info={'life': 18, 'hand_size': 4, 'battlefield_count': 3, 'statistics': {}},
                timestamp=time.time(), format='standard', game_id='test_complex_001'
            ),
            'stress_test_state': MTGGameState(
                life=7, mana_pool={'white': 10, 'blue': 8, 'black': 7, 'red': 6, 'green': 6, 'colorless': 15},
                hand=[{'name': f'Card_{i}', 'types': ['creature' if i % 2 == 0 else 'instant'],
                       'mana_cost': {'cmc': i % 7}} for i in range(7)],
                library_count=10, graveyard_count=25, exile_count=5,
                battlefield=[{'name': f'Permanent_{i}', 'types': ['creature'], 'mana_cost': {'cmc': i},
                            'power': i+1, 'toughness': i+2, 'tapped': i % 2 == 0} for i in range(10)],
                lands=[{'name': f'Land_{i}', 'types': ['land'], 'mana_cost': {'cmc': 0}, 'tapped': i % 3 == 0} for i in range(8)],
                creatures=[{'name': f'Creature_{i}', 'types': ['creature'], 'mana_cost': {'cmc': i},
                           'power': i+1, 'toughness': i+2, 'tapped': i % 2 == 0} for i in range(6)],
                artifacts_enchantments=[{'name': f'Enchantment_{i}', 'types': ['enchantment'], 'mana_cost': {'cmc': i}} for i in range(4)],
                turn_number=20, phase='postcombat_main', step='main2', priority_player='opponent',
                active_player='player', storm_count=5,
                known_info={'recent_actions': [{'type': 'cast_spell'} for _ in range(20)]},
                statistics={'avg_cmc_cast': 3.2, 'land_drops': 15, 'spells_cast': 25, 'creatures_cast': 12},
                opponent_info={'life': 22, 'hand_size': 8, 'battlefield_count': 8, 'statistics': {}},
                timestamp=time.time(), format='modern', game_id='stress_test_001'
            )
        }

    def test_sub_100ms_inference_requirement(self, mock_inference_engine, sample_game_states):
        """Test constitutional requirement: sub-100ms inference latency."""
        simple_state = sample_game_states['simple_state']

        # Test many predictions to get reliable statistics
        num_predictions = 100
        latencies = []

        for _ in range(num_predictions):
            start_time = time.time()
            result = mock_inference_engine.predict(simple_state)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Each prediction must meet constitutional requirement - THIS WILL FAIL
            assert latency_ms < 100.0, f"❌ CONSTITUTIONAL VIOLATION: Latency {latency_ms:.2f}ms >= 100ms requirement"

        # Statistical validation - THESE WILL FAIL due to mock timing
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # Constitutional compliance checks
        assert avg_latency < 50.0, f"❌ AVERAGE LATENCY VIOLATION: {avg_latency:.2f}ms >= 50ms target"
        assert p95_latency < 80.0, f"❌ P95 LATENCY VIOLATION: {p95_latency:.2f}ms >= 80ms target"
        assert p99_latency < 95.0, f"❌ P99 LATENCY VIOLATION: {p99_latency:.2f}ms >= 95ms target"

        # Additional constitutional requirements
        assert len(latencies) == num_predictions, "All predictions should complete"
        assert mock_inference_engine.model_loaded == True, "Model should be loaded"

    def test_batch_inference_performance(self, mock_inference_engine, sample_game_states):
        """Test batch inference performance requirements."""
        batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            # Create diverse game states for batch
            game_states = []
            for i in range(batch_size):
                if i % 3 == 0:
                    game_states.append(sample_game_states['simple_state'])
                elif i % 3 == 1:
                    game_states.append(sample_game_states['complex_state'])
                else:
                    game_states.append(sample_game_states['stress_test_state'])

            start_time = time.time()
            results = mock_inference_engine.predict_batch(game_states)
            end_time = time.time()

            total_latency_ms = (end_time - start_time) * 1000
            avg_latency_per_sample = total_latency_ms / batch_size

            # Each sample in batch should meet constitutional requirement - THIS WILL FAIL
            assert avg_latency_per_sample < 100.0, f"❌ BATCH INFERENCE VIOLATION: Batch size {batch_size}: {avg_latency_per_sample:.2f}ms per sample >= 100ms"
            assert len(results) == batch_size, f"❌ BATCH SIZE MISMATCH: Expected {batch_size} results, got {len(results)}"

            # Each result should be valid
            for i, result in enumerate(results):
                assert isinstance(result, InferenceResult), f"❌ INVALID RESULT TYPE: Batch {batch_size}, result {i}"
                assert result.processing_time < 100.0, f"❌ RESULT LATENCY VIOLATION: Batch {batch_size}, result {i}: {result.processing_time:.2f}ms"

    def test_state_extraction_performance(self, sample_game_states):
        """Test state extraction performance requirements."""
        state_extractor = StateExtractor()

        # Test various state complexities
        test_cases = [
            ('simple', sample_game_states['simple_state']),
            ('complex', sample_game_states['complex_state']),
            ('stress', sample_game_states['stress_test_state'])
        ]

        for case_name, test_state in test_cases:
            num_extractions = 1000
            extraction_times = []

            start_time = time.time()
            for _ in range(num_extractions):
                extraction_start = time.time()
                state_vector = state_extractor.extract_state(test_state)
                extraction_end = time.time()
                extraction_times.append((extraction_end - extraction_start) * 1000)
            end_time = time.time()

            avg_time = (end_time - start_time) / num_extractions * 1000
            p95_time = np.percentile(extraction_times, 95)

            # State extraction should be very fast - THESE MAY FAIL due to implementation
            assert avg_time < 5.0, f"❌ STATE EXTRACTION VIOLATION ({case_name}): Average {avg_time:.3f}ms >= 5.0ms target"
            assert p95_time < 10.0, f"❌ STATE EXTRACTION P95 VIOLATION ({case_name}): P95 {p95_time:.3f}ms >= 10.0ms target"

            # Validate state vector dimensions
            expected_dim = state_extractor.config.total_dimensions
            assert len(state_vector) == expected_dim, f"❌ STATE DIMENSION MISMATCH ({case_name}): Got {len(state_vector)}, expected {expected_dim}"

    def test_inference_engine_initialization(self):
        """Test inference engine initialization with constitutional requirements."""
        # Test with various configurations
        configurations = [
            {'max_latency_ms': 50.0},  # Aggressive
            {'max_latency_ms': 90.0},  # Default
            {'max_latency_ms': 100.0}, # Constitutional limit
            {'max_latency_ms': 150.0, 'enable_explanation': True, 'cache_size': 2000}
        ]

        for config in configurations:
            engine = InferenceEngine(**config)

            # Verify constitutional requirements
            assert engine.max_latency_ms <= 100.0, f"❌ INITIALIZATION VIOLATION: max_latency_ms {engine.max_latency_ms}ms > 100ms constitutional limit"
            assert hasattr(engine, 'state_extractor'), "❌ INITIALIZATION ERROR: Missing state_extractor"
            assert hasattr(engine, 'metrics'), "❌ INITIALIZATION ERROR: Missing metrics tracking"
            assert isinstance(engine.metrics, PerformanceMetrics), "❌ INITIALIZATION ERROR: Invalid metrics type"

            # Verify performance tracking is ready
            assert engine.metrics.total_inferences == 0, "❌ INITIALIZATION ERROR: Metrics not reset"
            assert engine.metrics.avg_latency == 0.0, "❌ INITIALIZATION ERROR: Metrics not initialized"

    def test_caching_performance_improvement(self, mock_inference_engine, sample_game_states):
        """Test that caching provides performance improvement."""
        game_state = sample_game_states['complex_state']

        # Time without caching
        times_without_cache = []
        for _ in range(20):
            start_time = time.time()
            result = mock_inference_engine.predict(game_state, use_cache=False)
            end_time = time.time()
            times_without_cache.append((end_time - start_time) * 1000)

        # Time with caching (after first hit, cache should be faster)
        times_with_cache = []
        for i in range(20):
            start_time = time.time()
            result = mock_inference_engine.predict(game_state, use_cache=True)
            end_time = time.time()
            times_with_cache.append((end_time - start_time) * 1000)

        avg_without_cache = np.mean(times_without_cache)
        # Skip first few to allow cache to warm up
        avg_with_cache = np.mean(times_with_cache[5:]) if len(times_with_cache) > 5 else np.mean(times_with_cache)

        # Caching should improve performance after first hit - THIS MAY FAIL
        cache_improvement = avg_without_cache - avg_with_cache
        assert cache_improvement > 10.0, f"❌ CACHING PERFORMANCE VIOLATION: Cache improvement {cache_improvement:.3f}ms <= 10ms target"

        # Verify cache hit rate improves
        cache_hit_rate = mock_inference_engine.metrics.cache_hit_rate
        assert cache_hit_rate > 0.5, f"❌ CACHE HIT RATE VIOLATION: Hit rate {cache_hit_rate:.3f} <= 50% target"

    def test_gpu_cpu_fallback_performance(self, sample_game_states):
        """Test GPU/CPU fallback performance testing."""
        # Test CPU-only configuration
        cpu_engine = InferenceEngine(max_latency_ms=90.0, enable_explanation=False)

        # Mock device manager to force CPU
        with patch('src.rl.inference.engine.DeviceManager') as mock_device_manager:
            mock_device_manager.return_value.get_optimal_device.return_value = torch.device('cpu')

            cpu_engine = InferenceEngine(max_latency_ms=90.0)

            # Test with simple state
            simple_state = sample_game_states['simple_state']

            # Mock a fast CPU model
            def fast_cpu_forward(x):
                time.sleep(0.080)  # 80ms - still exceeds requirement but faster
                return torch.randn(x.shape[0], 64)

            cpu_engine.model = Mock()
            cpu_engine.model.forward = fast_cpu_forward
            cpu_engine.model.eval = Mock()
            cpu_engine.model_loaded = True

            latencies = []
            for _ in range(50):
                start_time = time.time()
                result = cpu_engine.predict(simple_state)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)

            avg_latency = np.mean(latencies)

            # CPU should still try to meet constitutional requirement - THIS WILL FAIL
            assert avg_latency < 100.0, f"❌ CPU FALLBACK VIOLATION: CPU average latency {avg_latency:.2f}ms >= 100ms"
            assert result.device_used == 'cpu', f"❌ DEVICE FALLBACK ERROR: Expected CPU, got {result.device_used}"

    def test_performance_under_load(self, mock_inference_engine, sample_game_states):
        """Test performance under sustained load conditions."""
        # Simulate real gameplay scenario with rapid state changes
        states = list(sample_game_states.values())

        total_inferences = 1000
        latencies = []
        timeout_count = 0
        error_count = 0

        for i in range(total_inferences):
            state = states[i % len(states)]

            start_time = time.time()
            try:
                result = mock_inference_engine.predict(state)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Check for timeout indications
                if result.device_used == 'fallback':
                    timeout_count += 1

            except Exception as e:
                error_count += 1
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)

        # Performance under load analysis
        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0

        # Load performance requirements - THESE WILL FAIL
        assert avg_latency < 100.0, f"❌ LOAD PERFORMANCE VIOLATION: Load avg latency {avg_latency:.2f}ms >= 100ms"
        assert p95_latency < 120.0, f"❌ LOAD P95 VIOLATION: Load P95 latency {p95_latency:.2f}ms >= 120ms"
        assert p99_latency < 150.0, f"❌ LOAD P99 VIOLATION: Load P99 latency {p99_latency:.2f}ms >= 150ms"

        # Error and timeout rates should be minimal
        timeout_rate = timeout_count / total_inferences
        error_rate = error_count / total_inferences

        assert timeout_rate < 0.01, f"❌ TIMEOUT RATE VIOLATION: Timeout rate {timeout_rate:.3f} >= 1%"
        assert error_rate < 0.001, f"❌ ERROR RATE VIOLATION: Error rate {error_rate:.3f} >= 0.1%"

    def test_state_complexity_performance_scaling(self, mock_inference_engine, sample_game_states):
        """Test inference speed across different state complexities."""
        complexity_tests = [
            ('simple', sample_game_states['simple_state']),
            ('complex', sample_game_states['complex_state']),
            ('stress', sample_game_states['stress_test_state'])
        ]

        complexity_latencies = {}

        for complexity_name, state in complexity_tests:
            latencies = []
            for _ in range(100):
                start_time = time.time()
                result = mock_inference_engine.predict(state)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)

            complexity_latencies[complexity_name] = {
                'avg': np.mean(latencies),
                'p95': np.percentile(latencies, 95),
                'max': np.max(latencies)
            }

        # Performance scaling requirements
        simple_avg = complexity_latencies['simple']['avg']
        complex_avg = complexity_latencies['complex']['avg']
        stress_avg = complexity_latencies['stress']['avg']

        # More complex states shouldn't be dramatically slower
        complexity_ratio = complex_avg / simple_avg if simple_avg > 0 else float('inf')
        stress_ratio = stress_avg / simple_avg if simple_avg > 0 else float('inf')

        # All should meet constitutional requirement - THESE WILL FAIL
        for complexity_name, metrics in complexity_latencies.items():
            assert metrics['avg'] < 100.0, f"❌ COMPLEXITY VIOLATION ({complexity_name}): Avg latency {metrics['avg']:.2f}ms >= 100ms"
            assert metrics['p95'] < 100.0, f"❌ COMPLEXITY P95 VIOLATION ({complexity_name}): P95 latency {metrics['p95']:.2f}ms >= 100ms"

        # Complexity scaling should be reasonable
        assert complexity_ratio < 3.0, f"❌ COMPLEXITY SCALING VIOLATION: Complex/simple ratio {complexity_ratio:.2f} >= 3.0x"
        assert stress_ratio < 5.0, f"❌ STRESS SCALING VIOLATION: Stress/simple ratio {stress_ratio:.2f} >= 5.0x"

    def test_performance_regression_detection(self, mock_inference_engine, sample_game_states):
        """Test performance regression detection capabilities."""
        # Baseline performance (would be stored from previous runs)
        baseline_avg_latency = 25.0  # ms - target from good implementation
        baseline_p95_latency = 45.0   # ms
        baseline_p99_latency = 65.0   # ms

        # Current performance measurement with failing mock
        current_latencies = []
        for _ in range(200):
            start_time = time.time()
            result = mock_inference_engine.predict(sample_game_states['complex_state'])
            end_time = time.time()
            current_latencies.append((end_time - start_time) * 1000)

        current_avg = np.mean(current_latencies)
        current_p95 = np.percentile(current_latencies, 95)
        current_p99 = np.percentile(current_latencies, 99)

        # Regression detection (allow 20% degradation, but current will fail)
        avg_regression_threshold = baseline_avg_latency * 1.2
        p95_regression_threshold = baseline_p95_latency * 1.2
        p99_regression_threshold = baseline_p99_latency * 1.2

        # These should all fail due to the slow mock
        assert current_avg < avg_regression_threshold, f"❌ PERFORMANCE REGRESSION: Average {current_avg:.2f}ms vs baseline {avg_regression_threshold:.2f}ms"
        assert current_p95 < p95_regression_threshold, f"❌ P95 REGRESSION: P95 {current_p95:.2f}ms vs baseline {p95_regression_threshold:.2f}ms"
        assert current_p99 < p99_regression_threshold, f"❌ P99 REGRESSION: P99 {current_p99:.2f}ms vs baseline {p99_regression_threshold:.2f}ms"

        # Additional regression detection metrics
        regression_percent = ((current_avg - baseline_avg_latency) / baseline_avg_latency) * 100
        assert regression_percent < 50.0, f"❌ SEVERE REGRESSION: Performance degraded by {regression_percent:.1f}% >= 50%"

    def test_real_time_inference_engine_compliance(self, mock_inference_engine, sample_game_states):
        """Test comprehensive real-time inference engine constitutional compliance."""
        # Test all constitutional requirements together
        constitutional_requirements = {
            'max_latency_ms': 100.0,
            'avg_latency_ms': 50.0,
            'p95_latency_ms': 80.0,
            'p99_latency_ms': 95.0,
            'timeout_rate': 0.01,  # 1%
            'error_rate': 0.001,   # 0.1%
            'memory_increase_mb': 50.0,
            'cache_hit_rate': 0.3   # 30%
        }

        # Comprehensive test suite
        total_inferences = 500
        latencies = []
        timeouts = 0
        errors = 0

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        for i in range(total_inferences):
            state = list(sample_game_states.values())[i % len(sample_game_states)]

            start_time = time.time()
            try:
                result = mock_inference_engine.predict(state)
                end_time = time.time()

                latency = (end_time - start_time) * 1000
                latencies.append(latency)

                if result.device_used == 'fallback':
                    timeouts += 1

            except Exception:
                errors += 1

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Calculate metrics
        avg_latency = np.mean(latencies) if latencies else float('inf')
        p95_latency = np.percentile(latencies, 95) if latencies else float('inf')
        p99_latency = np.percentile(latencies, 99) if latencies else float('inf')
        timeout_rate = timeouts / total_inferences
        error_rate = errors / total_inferences
        memory_increase = final_memory - initial_memory

        # Constitutional compliance validation - ALL WILL FAIL
        assert avg_latency < constitutional_requirements['avg_latency_ms'], f"❌ CONSTITUTIONAL VIOLATION: Average latency {avg_latency:.2f}ms >= {constitutional_requirements['avg_latency_ms']}ms"
        assert p95_latency < constitutional_requirements['p95_latency_ms'], f"❌ CONSTITUTIONAL VIOLATION: P95 latency {p95_latency:.2f}ms >= {constitutional_requirements['p95_latency_ms']}ms"
        assert p99_latency < constitutional_requirements['p99_latency_ms'], f"❌ CONSTITUTIONAL VIOLATION: P99 latency {p99_latency:.2f}ms >= {constitutional_requirements['p99_latency_ms']}ms"
        assert timeout_rate < constitutional_requirements['timeout_rate'], f"❌ CONSTITUTIONAL VIOLATION: Timeout rate {timeout_rate:.3f} >= {constitutional_requirements['timeout_rate']}"
        assert error_rate < constitutional_requirements['error_rate'], f"❌ CONSTITUTIONAL VIOLATION: Error rate {error_rate:.3f} >= {constitutional_requirements['error_rate']}"
        assert memory_increase < constitutional_requirements['memory_increase_mb'], f"❌ CONSTITUTIONAL VIOLATION: Memory increase {memory_increase:.2f}MB >= {constitutional_requirements['memory_increase_mb']}MB"

        # Check engine metrics tracking
        assert mock_inference_engine.metrics.total_inferences >= total_inferences * 0.9, "❌ METRICS TRACKING ERROR: Inference count mismatch"
        assert mock_inference_engine.metrics.cache_hit_rate >= 0, "❌ CACHE METRICS ERROR: Invalid cache hit rate"

    def test_memory_usage_stability(self, mock_inference_engine, sample_game_states):
        """Test that memory usage remains stable during repeated inference."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        game_state = sample_game_states['complex_state']

        # Perform many inferences
        for _ in range(1000):
            result = mock_inference_engine.predict(game_state)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal - THIS MAY FAIL
        assert memory_increase < 50.0, f"❌ MEMORY USAGE VIOLATION: Memory increased {memory_increase:.2f}MB >= 50MB limit"

        # Check for memory leaks in metrics tracking
        assert len(mock_inference_engine.latency_history) <= 10000, "❌ MEMORY LEAK: Latency history growing without bound"

    def test_concurrent_inference_performance(self, mock_inference_engine, sample_game_states):
        """Test concurrent inference performance."""
        import queue

        results_queue = queue.Queue()
        num_threads = 8
        inferences_per_thread = 50

        def worker(thread_id):
            for i in range(inferences_per_thread):
                state_name = ['simple', 'complex', 'stress'][i % 3]
                game_state = sample_game_states[f'{state_name}_state']

                start_time = time.time()
                try:
                    result = mock_inference_engine.predict(game_state)
                    end_time = time.time()
                    results_queue.put({
                        'thread_id': thread_id,
                        'inference_id': i,
                        'latency_ms': (end_time - start_time) * 1000,
                        'success': True,
                        'device_used': result.device_used
                    })
                except Exception as e:
                    end_time = time.time()
                    results_queue.put({
                        'thread_id': thread_id,
                        'inference_id': i,
                        'latency_ms': (end_time - start_time) * 1000,
                        'success': False,
                        'error': str(e)
                    })

        # Start concurrent inferences
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify concurrent performance
        assert len(results) == num_threads * inferences_per_thread, f"❌ CONCURRENT INFERENCE ERROR: Expected {num_threads * inferences_per_thread} results, got {len(results)}"

        successful_results = [r for r in results if r['success']]
        latencies = [r['latency_ms'] for r in successful_results]

        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            # Concurrent inferences should still meet requirements - THESE WILL FAIL
            assert avg_latency < 100.0, f"❌ CONCURRENT AVG LATENCY VIOLATION: {avg_latency:.2f}ms >= 100ms"
            assert p95_latency < 100.0, f"❌ CONCURRENT P95 LATENCY VIOLATION: {p95_latency:.2f}ms >= 100ms"

        # Check thread safety - no exceptions should occur
        failed_results = [r for r in results if not r['success']]
        assert len(failed_results) == 0, f"❌ THREAD SAFETY VIOLATION: {len(failed_results)} failed inferences"

    def test_inference_engine_performance_report(self, mock_inference_engine):
        """Test inference engine performance report generation."""
        # Generate some inference activity
        simple_state = {
            'life': 20, 'hand_size': 5, 'battlefield_size': 3,
            'lands_in_play': 4, 'creatures_in_play': 2, 'library_size': 40,
            'graveyard_size': 5, 'available_mana': 4, 'turn_number': 8,
            'phase': 'main', 'step': 'main1', 'priority_player': 'player',
            'active_player': 'player', 'storm_count': 0, 'board_power': 5,
            'board_toughness': 6, 'opponent_life': 18, 'opponent_hand_size': 4,
            'opponent_battlefield_size': 3, 'timestamp': time.time(),
            'format': 'standard', 'game_id': 'test_report'
        }

        for _ in range(50):
            result = mock_inference_engine.predict(simple_state)

        # Get performance report
        report = mock_inference_engine.get_performance_report()

        # Validate report structure and constitutional compliance
        assert isinstance(report, dict), "❌ REPORT ERROR: Performance report should be dictionary"
        assert 'metrics' in report, "❌ REPORT ERROR: Missing metrics section"
        assert 'device_info' in report, "❌ REPORT ERROR: Missing device info section"
        assert 'cache_info' in report, "❌ REPORT ERROR: Missing cache info section"
        assert 'latency_target' in report, "❌ REPORT ERROR: Missing latency target section"

        # Validate metrics structure
        metrics = report['metrics']
        required_metrics = [
            'total_inferences', 'avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
            'gpu_usage_count', 'cpu_usage_count', 'timeout_count', 'cache_hit_rate'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"❌ METRICS ERROR: Missing required metric: {metric}"

        # Validate constitutional compliance indicators
        latency_target = report['latency_target']
        assert 'target_ms' in latency_target, "❌ LATENCY TARGET ERROR: Missing target_ms"
        assert 'meeting_target' in latency_target, "❌ LATENCY TARGET ERROR: Missing meeting_target"

        # This will fail due to our slow mock
        assert latency_target['meeting_target'] == True, "❌ CONSTITUTIONAL VIOLATION: Inference engine not meeting latency target"
        assert latency_target['target_ms'] <= 100.0, f"❌ CONFIGURATION ERROR: Target latency {latency_target['target_ms']}ms > 100ms"


if __name__ == "__main__":
    pytest.main([__file__])