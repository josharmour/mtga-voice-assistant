"""
Performance Monitoring System for MTGA Voice Advisor

Provides thread-safe performance monitoring with minimal overhead when disabled.
Instruments key operations to identify bottlenecks and track performance over time.

Usage:
    from src.core.monitoring import PerformanceMonitor

    monitor = PerformanceMonitor.get()

    # Measure a code block
    with monitor.measure("operation_name"):
        # Your code here
        pass

    # Get performance report
    report = monitor.report()

    # Set warning threshold (in milliseconds)
    monitor.set_threshold("slow_operation", 100.0)

    # Clear all metrics
    monitor.clear()
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Thread-safe singleton for performance monitoring.

    Tracks execution times for named operations, computing statistics
    (count, avg, max, min, total) with minimal overhead. Can be disabled
    for production use.
    """

    _instance: Optional['PerformanceMonitor'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the performance monitor."""
        # Metric storage: {operation_name: [duration1, duration2, ...]}
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._metrics_lock = threading.Lock()

        # Enable/disable flag
        self.enabled = True

        # Warning thresholds (in milliseconds)
        self._thresholds: Dict[str, float] = {}

        # Operation counts (faster than len(list) for large datasets)
        self._counts: Dict[str, int] = defaultdict(int)

        # Running totals (avoid recalculating sum for each call)
        self._totals: Dict[str, float] = defaultdict(float)

    @classmethod
    def get(cls) -> 'PerformanceMonitor':
        """
        Get the singleton instance of PerformanceMonitor.

        Returns:
            The shared PerformanceMonitor instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = PerformanceMonitor()
        return cls._instance

    @contextmanager
    def measure(self, name: str):
        """
        Context manager for timing code blocks.

        Usage:
            with monitor.measure("parse_game_state"):
                parse_game_state()

        Args:
            name: Operation name for tracking
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            elapsed_ms = elapsed * 1000.0  # Convert to milliseconds

            # Record metric
            with self._metrics_lock:
                self._metrics[name].append(elapsed_ms)
                self._counts[name] += 1
                self._totals[name] += elapsed_ms

            # Check threshold warning
            threshold = self._thresholds.get(name)
            if threshold is not None and elapsed_ms > threshold:
                logger.warning(
                    f"PERFORMANCE: '{name}' exceeded threshold: "
                    f"{elapsed_ms:.2f}ms > {threshold:.2f}ms"
                )

    def report(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a performance report for all tracked operations.

        Returns:
            Dictionary mapping operation names to statistics:
            {
                'operation_name': {
                    'count': number of executions,
                    'total_ms': total time spent (ms),
                    'avg_ms': average time (ms),
                    'min_ms': minimum time (ms),
                    'max_ms': maximum time (ms),
                    'threshold_ms': warning threshold if set
                }
            }
        """
        report = {}

        with self._metrics_lock:
            for name, durations in self._metrics.items():
                if not durations:
                    continue

                stats = {
                    'count': self._counts[name],
                    'total_ms': self._totals[name],
                    'avg_ms': self._totals[name] / self._counts[name],
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                }

                # Add threshold if set
                if name in self._thresholds:
                    stats['threshold_ms'] = self._thresholds[name]

                report[name] = stats

        return report

    def report_sorted(self, sort_by: str = 'total_ms', limit: int = None) -> List[tuple]:
        """
        Generate a sorted performance report.

        Args:
            sort_by: Metric to sort by ('total_ms', 'avg_ms', 'max_ms', 'count')
            limit: Maximum number of entries to return (None = all)

        Returns:
            List of (operation_name, stats_dict) tuples, sorted by specified metric
        """
        report = self.report()
        sorted_items = sorted(
            report.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )

        if limit:
            sorted_items = sorted_items[:limit]

        return sorted_items

    def set_threshold(self, name: str, threshold_ms: float):
        """
        Set a warning threshold for an operation.

        Args:
            name: Operation name
            threshold_ms: Threshold in milliseconds
        """
        self._thresholds[name] = threshold_ms
        logger.debug(f"Set performance threshold for '{name}': {threshold_ms}ms")

    def clear(self):
        """Clear all recorded metrics."""
        with self._metrics_lock:
            self._metrics.clear()
            self._counts.clear()
            self._totals.clear()
        logger.debug("Performance metrics cleared")

    def clear_operation(self, name: str):
        """
        Clear metrics for a specific operation.

        Args:
            name: Operation name to clear
        """
        with self._metrics_lock:
            if name in self._metrics:
                del self._metrics[name]
            if name in self._counts:
                del self._counts[name]
            if name in self._totals:
                del self._totals[name]
        logger.debug(f"Cleared metrics for '{name}'")

    def enable(self):
        """Enable performance monitoring."""
        self.enabled = True
        logger.info("Performance monitoring enabled")

    def disable(self):
        """Disable performance monitoring (zero overhead)."""
        self.enabled = False
        logger.info("Performance monitoring disabled")

    def log_report(self, sort_by: str = 'total_ms', limit: int = 10):
        """
        Log a formatted performance report.

        Args:
            sort_by: Metric to sort by
            limit: Maximum number of entries to show
        """
        sorted_report = self.report_sorted(sort_by=sort_by, limit=limit)

        if not sorted_report:
            logger.info("No performance metrics recorded")
            return

        logger.info("=" * 80)
        logger.info(f"Performance Report (Top {limit} by {sort_by})")
        logger.info("=" * 80)

        for name, stats in sorted_report:
            logger.info(
                f"{name:40s} | "
                f"Count: {stats['count']:5d} | "
                f"Total: {stats['total_ms']:8.2f}ms | "
                f"Avg: {stats['avg_ms']:7.2f}ms | "
                f"Min: {stats['min_ms']:7.2f}ms | "
                f"Max: {stats['max_ms']:7.2f}ms"
            )

            # Show threshold warning if exceeded
            if 'threshold_ms' in stats and stats['max_ms'] > stats['threshold_ms']:
                logger.info(
                    f"  WARNING: Max time ({stats['max_ms']:.2f}ms) "
                    f"exceeds threshold ({stats['threshold_ms']:.2f}ms)"
                )

        logger.info("=" * 80)


# Convenience function for getting the monitor
def get_monitor() -> PerformanceMonitor:
    """Get the global PerformanceMonitor instance."""
    return PerformanceMonitor.get()
