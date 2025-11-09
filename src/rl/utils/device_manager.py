"""
Device Manager for RL Operations

Handles GPU/CPU detection and provides fallback mechanisms to ensure
real-time inference latency requirements are met.

Constitutional Requirements:
- Real-Time Responsiveness: Sub-100ms inference latency (NON-NEGOTIABLE)
- Graceful Degradation: CPU-only inference when GPU unavailable
"""

import logging
import os
import warnings
from typing import Optional, Tuple
import torch
import psutil

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device detection and provides optimal device selection for RL operations.

    Ensures sub-100ms inference latency through intelligent device selection
    and graceful degradation strategies.
    """

    def __init__(self, prefer_gpu: bool = True, fallback_to_cpu: bool = True):
        """
        Initialize device manager with detection preferences.

        Args:
            prefer_gpu: Whether to prefer GPU when available
            fallback_to_cpu: Whether to fallback to CPU if GPU unavailable
        """
        self.prefer_gpu = prefer_gpu
        self.fallback_to_cpu = fallback_to_cpu
        self._device = None
        self._device_info = {}

        self._detect_devices()

    def _detect_devices(self) -> None:
        """Detect available devices and their capabilities."""
        logger.info("Detecting available devices for RL operations...")

        # Check CPU capabilities
        self._detect_cpu_info()

        # Check GPU capabilities (CUDA/MPS)
        self._detect_gpu_info()

        # Select optimal device
        self._select_device()

        logger.info(f"Device detection complete. Using: {self._device}")
        logger.info(f"Device info: {self._device_info}")

    def _detect_cpu_info(self) -> None:
        """Detect CPU capabilities and memory."""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()

        self._device_info['cpu'] = {
            'available': True,
            'cores_logical': cpu_count,
            'cores_physical': cpu_count_physical,
            'frequency_mhz': cpu_freq.current if cpu_freq else None,
            'memory_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3)
        }

        logger.debug(f"CPU detected: {cpu_count} cores, {memory.total/(1024**3):.1f}GB RAM")

    def _detect_gpu_info(self) -> None:
        """Detect GPU capabilities (CUDA/Metal)."""
        self._device_info['cuda'] = {'available': False}
        self._device_info['mps'] = {'available': False}

        # Check CUDA availability
        if torch.cuda.is_available():
            self._device_info['cuda'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'current_device': torch.cuda.current_device(),
                'memory_total': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
            }
            logger.info(f"CUDA detected: {torch.cuda.device_count()} devices")

        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device_info['mps'] = {
                'available': True,
                'name': 'Apple Silicon GPU (MPS)'
            }
            logger.info("Apple Silicon GPU (MPS) detected")

        else:
            logger.info("No GPU detected, will use CPU-only inference")

    def _select_device(self) -> None:
        """Select optimal device based on capabilities and preferences."""
        device_selected = False

        # Prefer GPU if requested and available
        if self.prefer_gpu:
            # Try CUDA first
            if self._device_info['cuda']['available']:
                self._device = torch.device('cuda')
                device_selected = True
                logger.info("Selected CUDA device for RL operations")

            # Try MPS (Apple Silicon)
            elif self._device_info['mps']['available']:
                self._device = torch.device('mps')
                device_selected = True
                logger.info("Selected MPS device for RL operations")

        # Fallback to CPU if needed
        if not device_selected:
            if self.fallback_to_cpu:
                self._device = torch.device('cpu')
                logger.info("Selected CPU device for RL operations")
            else:
                raise RuntimeError("No suitable device found and CPU fallback disabled")

    @property
    def device(self) -> torch.device:
        """Get the selected device."""
        return self._device

    @property
    def device_type(self) -> str:
        """Get device type as string."""
        return str(self._device)

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self._device.type in ['cuda', 'mps']

    @property
    def device_info(self) -> dict:
        """Get device information."""
        return self._device_info.copy()

    def get_memory_info(self) -> dict:
        """Get memory information for the current device."""
        if self._device.type == 'cuda':
            device_idx = self._device.index if self._device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
            total = self._device_info['cuda']['memory_total'][device_idx] / (1024**3)

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated,
                'utilization_percent': (allocated / total) * 100
            }

        elif self._device.type == 'cpu':
            memory = psutil.virtual_memory()
            return {
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'utilization_percent': memory.percent
            }

        else:  # MPS
            return {'info': 'MPS memory usage not directly available'}

    def set_device(self) -> None:
        """Set the default torch device for RL operations."""
        torch.set_default_device(self._device)

        # Suppress warning messages for better user experience
        if self._device.type == 'cuda':
            warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

    def warm_up_device(self, warmup_iterations: int = 10) -> None:
        """
        Warm up the device to ensure optimal performance.

        Args:
            warmup_iterations: Number of warmup iterations to perform
        """
        logger.info(f"Warming up {self._device.type} device...")

        try:
            # Create dummy tensors based on expected RL input size (380-dimensional)
            dummy_state = torch.randn(1, 380, device=self._device)
            dummy_weights = torch.randn(380, 64, device=self._device)  # Linear layer weights

            # Perform dummy forward passes
            with torch.no_grad():
                for i in range(warmup_iterations):
                    # Simulate neural network computation
                    result = torch.mm(dummy_state, dummy_weights)
                    if i == 0:
                        logger.debug(f"First warmup iteration completed on {self._device}")

            # Synchronize for CUDA devices
            if self._device.type == 'cuda':
                torch.cuda.synchronize()

            logger.info(f"Device warmup complete: {warmup_iterations} iterations")

        except Exception as e:
            logger.warning(f"Device warmup failed: {e}")
            logger.info("Proceeding without warmup (may affect first inference latency)")

    def validate_performance_requirements(self, max_latency_ms: float = 100.0) -> bool:
        """
        Validate that the selected device meets performance requirements.

        Args:
            max_latency_ms: Maximum allowed inference latency in milliseconds

        Returns:
            True if device meets requirements, False otherwise
        """
        import time

        logger.info(f"Validating device performance against {max_latency_ms}ms latency requirement...")

        try:
            # Simulate typical RL inference workload
            batch_size = 1
            state_dim = 380
            action_dim = 64

            # Create test tensors
            state = torch.randn(batch_size, state_dim, device=self._device)
            weights = torch.randn(state_dim, action_dim, device=self._device)

            # Warm up
            for _ in range(5):
                _ = torch.mm(state, weights)
                if self._device.type == 'cuda':
                    torch.cuda.synchronize()

            # Measure inference latency
            latencies = []
            for _ in range(50):  # Multiple measurements
                start_time = time.perf_counter()

                # Simulate RL forward pass
                q_values = torch.mm(state, weights)
                _ = torch.argmax(q_values, dim=1)

                if self._device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

            meets_requirement = avg_latency <= max_latency_ms and p95_latency <= max_latency_ms * 2

            logger.info(f"Performance validation results:")
            logger.info(f"  Average latency: {avg_latency:.2f}ms (target: {max_latency_ms}ms)")
            logger.info(f"  P95 latency: {p95_latency:.2f}ms")
            logger.info(f"  Meets requirement: {'✅ YES' if meets_requirement else '❌ NO'}")

            if not meets_requirement:
                logger.warning(f"Device {self._device} may not meet sub-{max_latency_ms}ms latency requirement")
                if self.fallback_to_cpu and self._device.type != 'cpu':
                    logger.info("Consider using CPU with optimizations for better latency")

            return meets_requirement

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return False

    def get_optimization_suggestions(self) -> list:
        """
        Get optimization suggestions for the current device configuration.

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        if self._device.type == 'cpu':
            suggestions.append("Consider enabling GPU acceleration for better performance")
            if self._device_info['cpu']['cores_logical'] >= 8:
                suggestions.append("CPU has adequate cores - ensure PyTorch is using optimized BLAS libraries")
            if self._device_info['cpu']['memory_available_gb'] < 4:
                suggestions.append("Limited RAM available - monitor memory usage during training")

        elif self._device.type == 'cuda':
            memory_info = self.get_memory_info()
            if memory_info['utilization_percent'] > 80:
                suggestions.append("GPU memory usage high - consider smaller batch sizes or model pruning")
            suggestions.append("Enable mixed precision training (torch.cuda.amp) for faster training")

        return suggestions


# Global device manager instance
_device_manager = None


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> torch.device:
    """Get the optimal device for RL operations."""
    return get_device_manager().device


def setup_rl_device(prefer_gpu: bool = True, validate_performance: bool = True) -> DeviceManager:
    """
    Setup and configure device for RL operations.

    Args:
        prefer_gpu: Whether to prefer GPU when available
        validate_performance: Whether to validate performance requirements

    Returns:
        Configured device manager
    """
    manager = DeviceManager(prefer_gpu=prefer_gpu)
    manager.set_device()
    manager.warm_up_device()

    if validate_performance:
        manager.validate_performance_requirements()

    return manager