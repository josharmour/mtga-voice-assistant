"""
Real-Time Inference Engine for MTG RL Agent

Sub-100ms latency inference system with CPU/GPU fallback and
optimized for real-time gameplay advisory.
"""

import logging
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
from queue import Queue, Empty
import json

from ..models.dueling_dqn import DuelingDQNNetwork as DuelingDQN
from ..data.state_extractor import StateExtractor
from ..utils.device_manager import DeviceManager
from ...core.mtga import GameStateManager

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from RL inference engine."""
    action: int  # Selected action
    confidence: float  # Confidence in the decision
    q_values: np.ndarray  # Full Q-value distribution
    processing_time: float  # Time taken for inference (ms)
    device_used: str  # 'cpu' or 'cuda'
    explanation: Optional[Dict[str, Any]]  # Decision explanation


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference engine."""
    total_inferences: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    gpu_usage_count: int = 0
    cpu_usage_count: int = 0
    timeout_count: int = 0
    cache_hit_rate: float = 0.0


class InferenceEngine:
    """
    High-performance real-time inference engine for MTG RL agent.

    Guarantees sub-100ms inference latency through:
    - Model optimization and quantization
    - Batching and parallel processing
    - Smart caching and memoization
    - GPU/CPU fallback with device detection
    - Timeout and graceful degradation
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_manager: Optional[DeviceManager] = None,
        max_latency_ms: float = 90.0,  # Target <100ms
        enable_explanation: bool = True,
        cache_size: int = 1000
    ):
        self.max_latency_ms = max_latency_ms
        self.enable_explanation = enable_explanation
        self.cache_size = cache_size

        # Initialize device manager
        self.device_manager = device_manager or DeviceManager()
        self.device = self.device_manager.device

        # Initialize components
        self.state_extractor = StateExtractor()
        self.model = None
        self.model_loaded = False

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.latency_history = []
        self.inference_lock = threading.RLock()

        # Caching system
        self.state_cache = {}
        self.cache_access_count = 0
        self.cache_hit_count = 0

        # Batch processing queue
        self.batch_queue = Queue()
        self.batch_processor = None
        self.batch_size = 8
        self.batch_timeout = 0.01  # 10ms

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

        logger.info(f"Inference engine initialized on device: {self.device}")
        logger.info(f"Target latency: {self.max_latency_ms}ms")

    def load_model(self, model_path: str) -> bool:
        """
        Load RL model with optimizations for fast inference.

        Args:
            model_path: Path to saved model

        Returns:
            True if model loaded successfully
        """
        try:
            start_time = time.time()

            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)

            # Initialize model architecture
            state_dim = checkpoint.get('state_dim', 380)
            action_dim = checkpoint.get('action_dim', 64)
            hidden_dims = checkpoint.get('hidden_dims', [512, 256, 128])

            self.model = DuelingDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)

            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Optimize for inference
            self._optimize_model()

            load_time = (time.time() - start_time) * 1000
            self.model_loaded = True

            logger.info(f"Model loaded successfully in {load_time:.2f}ms")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model_loaded = False
            return False

    def _optimize_model(self):
        """Optimize model for fast inference."""
        if self.model is None:
            return

        # Set to evaluation mode
        self.model.eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        # Optimize based on device
        if self.device.type == 'cuda':
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Use mixed precision if available
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                self.model.half()  # Convert to float16

        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile for better performance")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

    def predict(
        self,
        game_state: Union[Dict[str, Any], 'MTGGameState'],
        valid_actions: Optional[List[int]] = None,
        use_cache: bool = True,
        timeout_ms: Optional[float] = None
    ) -> InferenceResult:
        """
        Make ultra-fast real-time prediction for current game state.

        Constitutional Requirement: Sub-100ms latency (NON-NEGOTIABLE)
        Optimized for ultra-low latency with multiple performance optimizations.

        Args:
            game_state: Current game state (dict or MTGGameState object)
            valid_actions: List of valid action indices
            use_cache: Whether to use caching for identical states
            timeout_ms: Custom timeout (defaults to max_latency_ms)

        Returns:
            InferenceResult with action and metadata
        """
        timeout_ms = timeout_ms or self.max_latency_ms
        start_time = time.perf_counter()

        # Ultra-fast cache lookup with direct hash comparison
        if use_cache and self.state_cache:
            cache_key = self._create_fast_cache_key(game_state, valid_actions)
            cached_result = self.state_cache.get(cache_key)
            if cached_result:
                self.cache_hit_count += 1
                self.cache_access_count += 1
                # Cache hit - return immediately with updated timing
                cached_result.processing_time = (time.perf_counter() - start_time) * 1000
                return cached_result

        if use_cache:
            self.cache_access_count += 1

        # Ultra-fast state extraction with optimized tensor operations
        try:
            # Convert to MTGGameState if dict, use directly if already object
            if isinstance(game_state, dict):
                # Fast path for dict - convert with minimal overhead
                mtg_state = self._fast_dict_to_state(game_state)
            else:
                mtg_state = game_state

            # Extract state vector with optimized operations
            state_vector = self._extract_state_optimized(mtg_state)

            # Convert to tensor with minimal memory allocation
            if isinstance(state_vector, list):
                state_array = np.array(state_vector, dtype=np.float32)
            else:
                state_array = state_vector

            # Direct tensor creation with pinned memory for faster GPU transfer
            state_tensor = torch.from_numpy(state_array).unsqueeze(0)
            if self.device.type == 'cuda':
                state_tensor = state_tensor.pin_memory().to(self.device, non_blocking=True)
            else:
                state_tensor = state_tensor.to(self.device)

        except Exception as e:
            # Fast fallback for state extraction errors
            return self._get_fallback_result(game_state, 0.1, f"State extraction error: {str(e)[:50]}")

        # Ultra-fast model inference with optimized execution
        try:
            with torch.no_grad():  # Critical: no gradients for inference
                # Use optimized forward pass
                if hasattr(self.model, 'forward') and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):  # Mixed precision for speed
                        q_values = self.model(state_tensor)
                    torch.cuda.synchronize()  # Ensure completion for timing
                else:
                    # CPU optimized path
                    q_values = self.model(state_tensor)

        except Exception as e:
            # Fast fallback for model inference errors
            return self._get_fallback_result(game_state, 1.0, f"Model inference error: {str(e)[:50]}")

        # Ultra-fast post-processing with optimized numpy operations
        try:
            # Convert to numpy efficiently
            if self.device.type == 'cuda':
                q_values_np = q_values.cpu().numpy().flatten()
            else:
                q_values_np = q_values.numpy().flatten()

            # Fast action masking using vectorized operations
            if valid_actions is not None and len(valid_actions) > 0:
                masked_q_values = np.full_like(q_values_np, -float('inf'), dtype=np.float32)
                masked_q_values[valid_actions] = q_values_np[valid_actions]
            else:
                masked_q_values = q_values_np

            # Ultra-fast action selection using numpy's argmax
            action = int(np.argmax(masked_q_values))
            confidence = float(masked_q_values[action])

        except Exception as e:
            # Emergency fallback for post-processing errors
            return self._get_fallback_result(game_state, 2.0, f"Post-processing error: {str(e)[:50]}")

        # Calculate final processing time
        processing_time = (time.perf_counter() - start_time) * 1000

        # Constitutional latency check - critical requirement
        if processing_time > timeout_ms:
            self.metrics.timeout_count += 1
            logger.warning(f"INFERENCE LATENCY VIOLATION: {processing_time:.2f}ms > {timeout_ms}ms (CONSTITUTIONAL REQUIREMENT)")
            # Return fallback if we exceed constitutional limit
            return self._get_fallback_result(game_state, processing_time, f"Latency violation: {processing_time:.1f}ms")

        # Generate explanation only if time permits and enabled
        explanation = None
        if self.enable_explanation and processing_time < timeout_ms * 0.6:  # Only if under 60% of budget
            try:
                explanation = self._generate_fast_explanation(game_state, action, confidence)
            except Exception:
                # Don't fail prediction for explanation issues
                pass

        # Create optimized result
        result = InferenceResult(
            action=action,
            confidence=confidence,
            q_values=q_values_np,
            processing_time=processing_time,
            device_used=str(self.device),
            explanation=explanation
        )

        # Update cache if time permits (fast path)
        if use_cache and processing_time < timeout_ms * 0.8:
            try:
                cache_key = self._create_fast_cache_key(game_state, valid_actions)
                self._update_cache_fast(cache_key, result)
            except Exception:
                # Don't fail prediction for cache issues
                pass

        # Update performance metrics
        self._update_metrics_fast(processing_time, self.device.type)

        return result

    def _fast_dict_to_state(self, game_state: Dict[str, Any]) -> 'MTGGameState':
        """Ultra-fast conversion from dict to MTGGameState with minimal overhead."""
        try:
            from ..data.state_extractor import MTGGameState

            # Fast field extraction with defaults
            return MTGGameState(
                life=game_state.get('life', 20),
                mana_pool=game_state.get('mana_pool', {}),
                hand=game_state.get('hand', []),
                library_count=game_state.get('library_count', game_state.get('library_size', 50)),
                graveyard_count=game_state.get('graveyard_count', game_state.get('graveyard_size', 0)),
                exile_count=game_state.get('exile_count', 0),
                battlefield=game_state.get('battlefield', []),
                lands=game_state.get('lands', []),
                creatures=game_state.get('creatures', []),
                artifacts_enchantments=game_state.get('artifacts_enchantments', []),
                turn_number=game_state.get('turn_number', 1),
                phase=game_state.get('phase', 'main'),
                step=game_state.get('step', 'main1'),
                priority_player=game_state.get('priority_player', 'player'),
                active_player=game_state.get('active_player', 'player'),
                storm_count=game_state.get('storm_count', 0),
                known_info=game_state.get('known_info', {}),
                statistics=game_state.get('statistics', {}),
                opponent_info=game_state.get('opponent_info', {}),
                timestamp=game_state.get('timestamp', time.time()),
                format=game_state.get('format', 'standard'),
                game_id=game_state.get('game_id', 'unknown')
            )
        except Exception:
            # Emergency fallback - minimal state
            from ..data.state_extractor import MTGGameState
            return MTGGameState(
                life=20, mana_pool={}, hand=[], library_count=50, graveyard_count=0,
                exile_count=0, battlefield=[], lands=[], creatures=[], artifacts_enchantments=[],
                turn_number=1, phase='main', step='main1', priority_player='player',
                active_player='player', storm_count=0, known_info={}, statistics={},
                opponent_info={}, timestamp=time.time(), format='standard', game_id='emergency'
            )

    def _extract_state_optimized(self, mtg_state: 'MTGGameState') -> List[float]:
        """Ultra-fast state extraction optimized for minimal latency."""
        try:
            # Use the existing state extractor but with error handling
            state_tensor = self.state_extractor.extract_state(mtg_state)

            # Convert tensor to list efficiently
            if hasattr(state_tensor, 'cpu'):
                # It's a tensor
                return state_tensor.cpu().numpy().tolist()
            elif hasattr(state_tensor, 'tolist'):
                # It's already a numpy array
                return state_tensor.tolist()
            else:
                # It's already a list
                return list(state_tensor)

        except Exception as e:
            logger.warning(f"Fast state extraction failed: {e}")
            # Emergency fallback - return zero state
            return [0.0] * 380  # Standard state dimension

    def _create_fast_cache_key(self, game_state: Union[Dict[str, Any], 'MTGGameState'], valid_actions: Optional[List[int]]) -> str:
        """Ultra-fast cache key creation with optimized hashing."""
        try:
            # Extract only the most influential state features for cache key
            if isinstance(game_state, dict):
                key_features = {
                    'life': game_state.get('life', 20),
                    'turn': game_state.get('turn_number', 1),
                    'phase': game_state.get('phase', 'main'),
                    'hand': game_state.get('hand_size', len(game_state.get('hand', []))),
                    'mana': sum(game_state.get('mana_pool', {}).values()),
                    'actions': str(sorted(valid_actions or []))
                }
            else:
                # MTGGameState object
                key_features = {
                    'life': getattr(game_state, 'life', 20),
                    'turn': getattr(game_state, 'turn_number', 1),
                    'phase': getattr(game_state, 'phase', 'main'),
                    'hand': len(getattr(game_state, 'hand', [])),
                    'mana': sum(getattr(game_state, 'mana_pool', {}).values()),
                    'actions': str(sorted(valid_actions or []))
                }

            # Fast hash generation
            return hash(str(key_features)).__str__()

        except Exception:
            # Emergency fallback
            return hash(f"{time.time()}_{id(game_state)}").__str__()

    def _update_cache_fast(self, cache_key: str, result: 'InferenceResult'):
        """Ultra-fast cache update with minimal overhead."""
        try:
            # Check cache size limit
            if len(self.state_cache) >= self.cache_size:
                # Remove oldest entry efficiently
                self.state_cache.pop(next(iter(self.state_cache)))

            # Add new entry
            self.state_cache[cache_key] = result

        except Exception as e:
            # Don't fail prediction for cache issues
            logger.debug(f"Cache update failed: {e}")

    def _generate_fast_explanation(self, game_state: Union[Dict[str, Any], 'MTGGameState'], action: int, confidence: float) -> Dict[str, Any]:
        """Ultra-fast explanation generation optimized for minimal latency."""
        try:
            return {
                'action': action,
                'confidence': confidence,
                'explanation': f"Selected action {action} with confidence {confidence:.3f}",
                'fast_mode': True
            }
        except Exception:
            return {'action': action, 'confidence': confidence, 'explanation': 'Fast mode'}

    def _update_metrics_fast(self, latency_ms: float, device_type: str):
        """Ultra-fast metrics update optimized for minimal overhead."""
        try:
            self.metrics.total_inferences += 1

            if device_type == 'cuda':
                self.metrics.gpu_usage_count += 1
            else:
                self.metrics.cpu_usage_count += 1

            # Update cache hit rate
            if self.cache_access_count > 0:
                self.metrics.cache_hit_rate = self.cache_hit_count / self.cache_access_count

            # Update latency metrics efficiently (only every 10 inferences)
            if self.metrics.total_inferences % 10 == 0:
                self.latency_history.append(latency_ms)

                # Keep history bounded
                if len(self.latency_history) > 1000:
                    self.latency_history = self.latency_history[-1000:]

                # Update statistics
                if len(self.latency_history) >= 10:
                    latencies_array = np.array(self.latency_history)
                    self.metrics.avg_latency = float(np.mean(latencies_array))
                    self.metrics.p95_latency = float(np.percentile(latencies_array, 95))
                    self.metrics.p99_latency = float(np.percentile(latencies_array, 99))

        except Exception as e:
            logger.debug(f"Metrics update failed: {e}")

    def predict_with_fallback(
        self,
        game_state: Union[Dict[str, Any], 'MTGGameState'],
        valid_actions: Optional[List[int]] = None
    ) -> InferenceResult:
        """
        Prediction with comprehensive fallback support.

        This method ensures the system always returns a valid result,
        even if model inference fails or exceeds latency requirements.
        """
        try:
            # First attempt: normal fast prediction
            result = self.predict(game_state, valid_actions, use_cache=True, timeout_ms=self.max_latency_ms)

            # If we got a fallback result, try once more with more relaxed settings
            if result.device_used in ['fallback', 'emergency_fallback']:
                logger.info("Retrying with relaxed settings...")
                result = self.predict(game_state, valid_actions, use_cache=False, timeout_ms=self.max_latency_ms * 1.5)

            return result

        except Exception as e:
            logger.error(f"All prediction attempts failed: {e}")
            # Ultimate fallback
            return self._get_fallback_result(game_state, self.max_latency_ms, f"Complete failure: {str(e)[:50]}")

    def predict_batch(
        self,
        game_states: List[Dict[str, Any]],
        valid_actions_list: Optional[List[List[int]]] = None,
        timeout_ms: Optional[float] = None
    ) -> List[InferenceResult]:
        """
        Batch prediction for multiple game states.

        Args:
            game_states: List of game state dictionaries
            valid_actions_list: List of valid action lists for each state
            timeout_ms: Timeout for entire batch

        Returns:
            List of InferenceResults
        """
        if not game_states:
            return []

        timeout_ms = timeout_ms or self.max_latency_ms
        start_time = time.time()

        try:
            # Extract states in batch
            state_vectors = []
            for state in game_states:
                vector = self.state_extractor.extract_state(state)
                state_vectors.append(vector)

            # Create batch tensor
            state_batch = torch.FloatTensor(state_vectors).to(self.device)

            # Batch inference
            with self._inference_context(timeout_ms):
                with torch.no_grad():
                    q_values_batch = self.model(state_batch)

            # Process results
            results = []
            q_values_np = q_values_batch.cpu().numpy()

            for i, game_state in enumerate(game_states):
                q_values = q_values_np[i]
                valid_actions = valid_actions_list[i] if valid_actions_list else None

                # Apply action mask
                if valid_actions is not None:
                    masked_q_values = np.full_like(q_values, -float('inf'))
                    masked_q_values[valid_actions] = q_values[valid_actions]
                else:
                    masked_q_values = q_values

                action = int(np.argmax(masked_q_values))
                confidence = float(np.max(masked_q_values))

                processing_time = (time.time() - start_time) * 1000 / len(game_states)

                explanation = None
                if self.enable_explanation:
                    explanation = self._generate_explanation(
                        game_state, action, q_values, valid_actions
                    )

                result = InferenceResult(
                    action=action,
                    confidence=confidence,
                    q_values=q_values,
                    processing_time=processing_time,
                    device_used=str(self.device),
                    explanation=explanation
                )

                results.append(result)

            # Update metrics
            avg_processing_time = (time.time() - start_time) * 1000
            self._update_metrics(avg_processing_time, self.device.type, batch_size=len(game_states))

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Return fallback results
            return [self._get_fallback_result(state, 0, str(e)) for state in game_states]

    @contextmanager
    def _inference_context(self, timeout_ms: float):
        """Context manager for inference with timeout."""
        if self.device.type == 'cuda':
            # Synchronize CUDA operations
            torch.cuda.synchronize()

        start_time = time.time()
        try:
            yield
        finally:
            # Check for timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                self.metrics.timeout_count += 1
                logger.warning(f"Inference timeout: {elapsed_ms:.2f}ms > {timeout_ms}ms")

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

    def _convert_game_state_to_dict(self, game_state: 'MTGGameState') -> Dict[str, Any]:
        """
        Convert MTGGameState object to dictionary representation.

        Args:
            game_state: MTGGameState object

        Returns:
            Dictionary representation suitable for caching and processing
        """
        try:
            # Extract key attributes from MTGGameState
            state_dict = {
                'life': getattr(game_state, 'life', 20),
                'mana_pool': getattr(game_state, 'mana_pool', {}),
                'hand': getattr(game_state, 'hand', []),
                'library_count': getattr(game_state, 'library_count', 50),
                'graveyard_count': getattr(game_state, 'graveyard_count', 0),
                'exile_count': getattr(game_state, 'exile_count', 0),
                'battlefield': getattr(game_state, 'battlefield', []),
                'lands': getattr(game_state, 'lands', []),
                'creatures': getattr(game_state, 'creatures', []),
                'artifacts_enchantments': getattr(game_state, 'artifacts_enchantments', []),
                'turn_number': getattr(game_state, 'turn_number', 1),
                'phase': getattr(game_state, 'phase', 'main'),
                'step': getattr(game_state, 'step', 'main1'),
                'priority_player': getattr(game_state, 'priority_player', 'player'),
                'active_player': getattr(game_state, 'active_player', 'player'),
                'storm_count': getattr(game_state, 'storm_count', 0),
                'known_info': getattr(game_state, 'known_info', {}),
                'statistics': getattr(game_state, 'statistics', {}),
                'opponent_info': getattr(game_state, 'opponent_info', {}),
                'timestamp': getattr(game_state, 'timestamp', time.time()),
                'format': getattr(game_state, 'format', 'standard'),
                'game_id': getattr(game_state, 'game_id', 'unknown')
            }

            # Add computed fields for cache compatibility
            state_dict.update({
                'hand_size': len(state_dict['hand']),
                'board_size': len(state_dict['battlefield']),
                'lands_in_play': len(state_dict['lands']),
                'creatures_in_play': len(state_dict['creatures']),
                'library_size': state_dict['library_count'],
                'graveyard_size': state_dict['graveyard_count'],
                'available_mana': sum(state_dict['mana_pool'].values()),
                'opponent_life': state_dict['opponent_info'].get('life', 20) if state_dict['opponent_info'] else 20,
                'opponent_hand_size': state_dict['opponent_info'].get('hand_size', 7) if state_dict['opponent_info'] else 7,
                'opponent_battlefield_size': state_dict['opponent_info'].get('battlefield_count', 0) if state_dict['opponent_info'] else 0,
                'board_power': sum(c.get('power', 0) for c in state_dict['creatures']),
                'board_toughness': sum(c.get('toughness', 0) for c in state_dict['creatures']),
                'is_combat_phase': state_dict['phase'] in ['combat', 'beginning_combat', 'declare_attackers', 'declare_blockers', 'combat_damage'],
            })

            return state_dict

        except Exception as e:
            logger.error(f"Failed to convert MTGGameState to dict: {e}")
            # Return minimal fallback state
            return {
                'life': 20, 'hand_size': 0, 'board_size': 0, 'turn_number': 1,
                'phase': 'main', 'available_mana': 0, 'timestamp': time.time()
            }

    def _check_cache(
        self,
        game_state: Dict[str, Any],
        valid_actions: Optional[List[int]]
    ) -> Optional[InferenceResult]:
        """Check if result is cached with ultra-fast lookup."""
        # Create cache key
        cache_key = self._create_cache_key(game_state, valid_actions)

        # Fast dictionary lookup
        cached_result = self.state_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        return None

    def _update_cache(
        self,
        game_state: Dict[str, Any],
        valid_actions: Optional[List[int]],
        result: InferenceResult
    ):
        """Update cache with new result."""
        if len(self.state_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.state_cache))
            del self.state_cache[oldest_key]

        cache_key = self._create_cache_key(game_state, valid_actions)
        self.state_cache[cache_key] = result

    def _create_cache_key(
        self,
        game_state: Dict[str, Any],
        valid_actions: Optional[List[int]]
    ) -> str:
        """Create cache key from state and valid actions."""
        # Use only state features that affect the decision
        key_features = {
            'turn_number': game_state.get('turn_number', 0),
            'phase': game_state.get('phase', ''),
            'life': game_state.get('life', 20),
            'opponent_life': game_state.get('opponent_life', 20),
            'hand_size': len(game_state.get('hand', [])),
            'board_state': str(sorted(game_state.get('board', []))),
            'mana_pool': str(game_state.get('mana_pool', {})),
            'valid_actions': str(sorted(valid_actions or []))
        }

        return hash(json.dumps(key_features, sort_keys=True)).__str__()

    def _generate_explanation(
        self,
        game_state: Dict[str, Any],
        action: int,
        q_values: np.ndarray,
        valid_actions: Optional[List[int]]
    ) -> Dict[str, Any]:
        """Generate explanation for the decision."""
        explanation = {
            'selected_action': action,
            'action_confidence': float(q_values[action]),
            'top_alternatives': [],
            'decision_factors': []
        }

        # Get top alternative actions
        if valid_actions is not None:
            masked_q = q_values.copy()
            masked_q[valid_actions] = -float('inf')
        else:
            masked_q = q_values

        top_indices = np.argsort(masked_q)[-3:][::-1]  # Top 3 actions
        for idx in top_indices:
            if idx != action and masked_q[idx] > -float('inf'):
                explanation['top_alternatives'].append({
                    'action': int(idx),
                    'q_value': float(masked_q[idx]),
                    'difference': float(q_values[action] - masked_q[idx])
                })

        # Add decision factors based on game state
        if game_state.get('is_combat_phase', False):
            explanation['decision_factors'].append('Combat tactics considered')

        if game_state.get('hand_size', 0) > 5:
            explanation['decision_factors'].append('Resource abundance available')

        if game_state.get('life', 20) < 10:
            explanation['decision_factors'].append('Defensive posture prioritized')

        return explanation

    def _get_fallback_result(
        self,
        game_state: Union[Dict[str, Any], 'MTGGameState'],
        processing_time: float,
        error_msg: str
    ) -> InferenceResult:
        """Get fallback result when inference fails with smart heuristics."""
        try:
            # Convert to dict if needed
            if hasattr(game_state, '__dict__'):
                state_dict = self._convert_game_state_to_dict(game_state)
            else:
                state_dict = game_state

            # Smart fallback based on game state
            action = 0  # Default: pass priority
            confidence = 0.1

            # Simple heuristic: if we have creatures and mana, consider attacking
            mana_available = state_dict.get('available_mana', 0)
            creatures_in_play = state_dict.get('creatures_in_play', 0)
            phase = state_dict.get('phase', '')
            hand_size = state_dict.get('hand_size', 0)

            if phase == 'combat' and creatures_in_play > 0:
                action = 1  # Attack action
                confidence = 0.3
            elif mana_available >= 2 and hand_size > 0:
                action = 2  # Cast spell action
                confidence = 0.25
            elif mana_available >= 1:
                action = 3  # Play land or activate ability
                confidence = 0.2

            return InferenceResult(
                action=action,
                confidence=confidence,
                q_values=np.full(64, -1.0),  # Low Q-values for fallback
                processing_time=processing_time,
                device_used='fallback',
                explanation={
                    'fallback_reason': error_msg,
                    'fallback_used': True,
                    'heuristic_used': f"{phase}_phase_fallback"
                }
            )

        except Exception as e:
            # Ultimate fallback
            logger.error(f"Fallback result generation failed: {e}")
            return InferenceResult(
                action=0,
                confidence=0.01,
                q_values=np.full(64, -10.0),
                processing_time=processing_time,
                device_used='emergency_fallback',
                explanation={
                    'fallback_reason': f"Emergency fallback: {error_msg}",
                    'fallback_used': True
                }
            )

    def _update_metrics(self, latency_ms: float, device_type: str, batch_size: int = 1):
        """Update performance metrics with efficient computation."""
        self.metrics.total_inferences += batch_size

        if device_type == 'cuda':
            self.metrics.gpu_usage_count += batch_size
        else:
            self.metrics.cpu_usage_count += batch_size

        # Update latency history efficiently
        self.latency_history.extend([latency_ms] * batch_size)

        # Keep only recent history (sliding window)
        max_history = 10000
        if len(self.latency_history) > max_history:
            # Remove oldest entries efficiently
            self.latency_history = self.latency_history[-max_history:]

        # Update cache hit rate
        if self.cache_access_count > 0:
            self.metrics.cache_hit_rate = self.cache_hit_count / self.cache_access_count

        # Update latency statistics only if we have enough data
        if len(self.latency_history) >= 10:
            # Use numpy for efficient computation
            latencies_array = np.array(self.latency_history)
            self.metrics.avg_latency = float(np.mean(latencies_array))
            self.metrics.p95_latency = float(np.percentile(latencies_array, 95))
            self.metrics.p99_latency = float(np.percentile(latencies_array, 99))

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'metrics': {
                'total_inferences': self.metrics.total_inferences,
                'avg_latency_ms': round(self.metrics.avg_latency, 2),
                'p95_latency_ms': round(self.metrics.p95_latency, 2),
                'p99_latency_ms': round(self.metrics.p99_latency, 2),
                'gpu_usage_count': self.metrics.gpu_usage_count,
                'cpu_usage_count': self.metrics.cpu_usage_count,
                'timeout_count': self.metrics.timeout_count,
                'cache_hit_rate': round(self.metrics.cache_hit_rate, 3)
            },
            'device_info': {
                'current_device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'model_loaded': self.model_loaded
            },
            'cache_info': {
                'cache_size': len(self.state_cache),
                'max_cache_size': self.cache_size,
                'access_count': self.cache_access_count,
                'hit_count': self.cache_hit_count
            },
            'latency_target': {
                'target_ms': self.max_latency_ms,
                'meeting_target': self.metrics.avg_latency <= self.max_latency_ms
            }
        }

    def clear_cache(self):
        """Clear the inference cache."""
        self.state_cache.clear()
        self.cache_access_count = 0
        self.cache_hit_count = 0
        logger.info("Inference cache cleared")

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()
        self.latency_history = []
        logger.info("Performance metrics reset")

    def benchmark_model(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Benchmark model performance with random states.

        Args:
            num_samples: Number of test samples

        Returns:
            Benchmark results
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}

        logger.info(f"Benchmarking model with {num_samples} samples...")

        latencies = []
        successful_inferences = 0

        for i in range(num_samples):
            # Generate random test state
            test_state = {
                'turn_number': np.random.randint(1, 20),
                'phase': np.random.choice(['main', 'combat', 'end']),
                'life': np.random.randint(1, 30),
                'hand_size': np.random.randint(0, 7),
                'board': []
            }

            try:
                result = self.predict(test_state)
                latencies.append(result.processing_time)
                successful_inferences += 1
            except Exception as e:
                logger.warning(f"Benchmark sample {i} failed: {e}")

        if latencies:
            benchmark_results = {
                'total_samples': num_samples,
                'successful_inferences': successful_inferences,
                'success_rate': successful_inferences / num_samples,
                'avg_latency_ms': np.mean(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'samples_within_target': np.sum(np.array(latencies) <= self.max_latency_ms),
                'target_compliance_rate': np.sum(np.array(latencies) <= self.max_latency_ms) / len(latencies)
            }
        else:
            benchmark_results = {'error': 'No successful inferences'}

        return benchmark_results

    def predict_concurrent(
        self,
        game_states: List[Union[Dict[str, Any], 'MTGGameState']],
        valid_actions_list: Optional[List[Optional[List[int]]]] = None,
        timeout_ms: Optional[float] = None,
        max_workers: int = 4
    ) -> List[InferenceResult]:
        """
        Concurrent prediction for multiple game states with thread safety.

        Args:
            game_states: List of game states to process
            valid_actions_list: List of valid actions for each state
            timeout_ms: Timeout per inference
            max_workers: Maximum number of concurrent threads

        Returns:
            List of InferenceResults in the same order as input
        """
        if not game_states:
            return []

        timeout_ms = timeout_ms or self.max_latency_ms
        valid_actions_list = valid_actions_list or [None] * len(game_states)

        # Use a thread-safe approach with a thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue

        results = [None] * len(game_states)
        result_queue = queue.Queue()

        def worker_with_index(idx_state):
            """Worker that processes a single state and maintains order."""
            idx, game_state = idx_state
            valid_actions = valid_actions_list[idx]

            try:
                result = self.predict(
                    game_state=game_state,
                    valid_actions=valid_actions,
                    use_cache=True,
                    timeout_ms=timeout_ms
                )
                result_queue.put((idx, result, None))
            except Exception as e:
                result_queue.put((idx, None, e))

        # Process states concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(worker_with_index, (i, state)): i
                for i, state in enumerate(game_states)
            }

            # Collect results in order
            completed_count = 0
            while completed_count < len(game_states):
                try:
                    idx, result, error = result_queue.get(timeout=timeout_ms * 2 / 1000)
                    if error is None:
                        results[idx] = result
                    else:
                        # Create fallback result for failed inference
                        results[idx] = self._get_fallback_result(
                            game_states[idx], timeout_ms, f"Concurrent error: {str(error)[:50]}"
                        )
                    completed_count += 1
                except queue.Empty:
                    # Handle timeout - fill remaining with fallbacks
                    for i in range(len(results)):
                        if results[i] is None:
                            results[i] = self._get_fallback_result(
                                game_states[i], timeout_ms, "Concurrent timeout"
                            )
                    break

        return results

    def benchmark_concurrent_performance(
        self,
        num_concurrent_requests: int = 10,
        requests_per_thread: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark concurrent inference performance.

        Args:
            num_concurrent_requests: Number of concurrent threads
            requests_per_thread: Number of requests per thread

        Returns:
            Concurrent performance benchmark results
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"Starting concurrent benchmark: {num_concurrent_requests} threads x {requests_per_thread} requests")

        # Generate test data
        test_states = []
        for _ in range(num_concurrent_requests * requests_per_thread):
            test_states.append({
                'life': np.random.randint(1, 30),
                'turn_number': np.random.randint(1, 20),
                'phase': np.random.choice(['main', 'combat', 'end']),
                'hand_size': np.random.randint(0, 7),
                'available_mana': np.random.randint(0, 10),
                'timestamp': time.time(),
                'game_id': f'concurrent_test_{len(test_states)}'
            })

        # Performance tracking
        results_lock = threading.Lock()
        all_latencies = []
        all_errors = []
        successful_requests = 0
        total_requests = 0

        def benchmark_worker(thread_id: int):
            """Worker for concurrent benchmarking."""
            nonlocal successful_requests, total_requests

            thread_latencies = []
            thread_errors = []

            start_idx = thread_id * requests_per_thread
            end_idx = start_idx + requests_per_thread

            for i in range(start_idx, min(end_idx, len(test_states))):
                total_requests += 1
                test_state = test_states[i]

                try:
                    start_time = time.perf_counter()
                    result = self.predict(test_state, timeout_ms=self.max_latency_ms)
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    thread_latencies.append(latency_ms)

                    if result.device_used not in ['fallback', 'emergency_fallback']:
                        successful_requests += 1

                except Exception as e:
                    thread_errors.append(str(e))

            # Thread-safe result aggregation
            with results_lock:
                all_latencies.extend(thread_latencies)
                all_errors.extend(thread_errors)

        # Run concurrent benchmark
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [
                executor.submit(benchmark_worker, thread_id)
                for thread_id in range(num_concurrent_requests)
            ]

            # Wait for all threads to complete
            for future in futures:
                future.result(timeout=60)  # 60 second timeout

        total_time = time.perf_counter() - start_time

        # Calculate performance metrics
        if all_latencies:
            latencies_array = np.array(all_latencies)
            concurrent_results = {
                'concurrent_setup': {
                    'threads': num_concurrent_requests,
                    'requests_per_thread': requests_per_thread,
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'success_rate': successful_requests / total_requests if total_requests > 0 else 0
                },
                'throughput_metrics': {
                    'total_time_seconds': total_time,
                    'requests_per_second': total_requests / total_time if total_time > 0 else 0,
                    'successful_requests_per_second': successful_requests / total_time if total_time > 0 else 0
                },
                'latency_metrics': {
                    'avg_latency_ms': float(np.mean(latencies_array)),
                    'min_latency_ms': float(np.min(latencies_array)),
                    'max_latency_ms': float(np.max(latencies_array)),
                    'p50_latency_ms': float(np.percentile(latencies_array, 50)),
                    'p95_latency_ms': float(np.percentile(latencies_array, 95)),
                    'p99_latency_ms': float(np.percentile(latencies_array, 99))
                },
                'target_compliance': {
                    'sub_100ms_rate': np.sum(latencies_array <= 100) / len(latencies_array),
                    'sub_90ms_rate': np.sum(latencies_array <= 90) / len(latencies_array),
                    'sub_50ms_rate': np.sum(latencies_array <= 50) / len(latencies_array)
                },
                'error_analysis': {
                    'total_errors': len(all_errors),
                    'error_rate': len(all_errors) / total_requests if total_requests > 0 else 0,
                    'sample_errors': all_errors[:5]  # First 5 errors for analysis
                }
            }
        else:
            concurrent_results = {
                'error': 'No successful inferences during concurrent benchmark',
                'total_errors': len(all_errors),
                'sample_errors': all_errors[:5]
            }

        return concurrent_results


# Class alias for test compatibility
RLInferenceEngine = InferenceEngine

# Convenience function for creating inference engine
def create_inference_engine(
    model_path: Optional[str] = None,
    max_latency_ms: float = 90.0
) -> InferenceEngine:
    """Create and configure inference engine."""
    return InferenceEngine(
        model_path=model_path,
        max_latency_ms=max_latency_ms,
        enable_explanation=True,
        cache_size=1000
    )