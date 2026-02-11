"""
Enhanced logging system optimized for AI-assisted debugging.

Features:
- Structured log entries with consistent formatting
- Component/module tagging for easy filtering
- Operation boundaries (START/END markers)
- State snapshots at key points
- Correlation IDs to track related operations
- Easy grep patterns for common searches
"""

import logging
import time
import threading
import json
from typing import Any, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# Thread-local storage for correlation IDs
_thread_local = threading.local()


class StructuredLogger:
    """
    Enhanced logger with structured output for AI-assisted debugging.

    Example usage:
        logger = StructuredLogger("MTGA_PARSER")
        logger.operation_start("parse_game_state", turn=5, phase="Main1")
        logger.state_snapshot("hand", hand_cards)
        logger.operation_end("parse_game_state", success=True)
    """

    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(component)

    def _format_message(self, level: str, msg_type: str, message: str, **context) -> str:
        """Format a structured log message."""
        # Get correlation ID if set
        corr_id = getattr(_thread_local, 'correlation_id', None)
        corr_str = f" [CORR:{corr_id}]" if corr_id else ""

        # Build context string
        ctx_parts = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                # Compact JSON for complex types
                ctx_parts.append(f"{key}={json.dumps(value)}")
            else:
                ctx_parts.append(f"{key}={value}")

        ctx_str = f" | {' '.join(ctx_parts)}" if ctx_parts else ""

        return f"[{self.component}]{corr_str} {msg_type}: {message}{ctx_str}"

    def operation_start(self, operation: str, **context):
        """Mark the start of an operation for easy tracking."""
        msg = self._format_message("INFO", "▶ START", operation, **context)
        self.logger.info(msg)

    def operation_end(self, operation: str, success: bool = True, duration_ms: float = None, **context):
        """Mark the end of an operation."""
        status = "✓ SUCCESS" if success else "✗ FAILED"
        if duration_ms is not None:
            context['duration_ms'] = f"{duration_ms:.2f}"
        msg = self._format_message("INFO", f"◀ END ({status})", operation, **context)
        self.logger.info(msg)

    def state_snapshot(self, state_name: str, state_data: Any, **metadata):
        """Log a snapshot of important state for debugging."""
        # Truncate large data structures
        if isinstance(state_data, (list, dict)):
            preview = str(state_data)[:200]
            if len(str(state_data)) > 200:
                preview += "..."
            msg = self._format_message("INFO", "📸 STATE", state_name,
                                       preview=preview,
                                       size=len(state_data) if hasattr(state_data, '__len__') else 'N/A',
                                       **metadata)
        else:
            msg = self._format_message("INFO", "📸 STATE", state_name, value=state_data, **metadata)
        self.logger.info(msg)

    def event(self, event_name: str, **details):
        """Log a significant event."""
        msg = self._format_message("INFO", "⚡ EVENT", event_name, **details)
        self.logger.info(msg)

    def error(self, error_msg: str, exception: Exception = None, **context):
        """Log an error with full context."""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_msg'] = str(exception)
        msg = self._format_message("ERROR", "❌ ERROR", error_msg, **context)
        self.logger.error(msg, exc_info=exception is not None)

    def warning(self, warning_msg: str, **context):
        """Log a warning."""
        msg = self._format_message("WARNING", "⚠ WARNING", warning_msg, **context)
        self.logger.warning(msg)

    def debug_data(self, data_name: str, data: Any, **metadata):
        """Log debug data (only shown when debug logging enabled)."""
        msg = self._format_message("DEBUG", "🔍 DEBUG", data_name, data=str(data)[:500], **metadata)
        self.logger.debug(msg)

    def metric(self, metric_name: str, value: float, unit: str = "", **tags):
        """Log a performance metric."""
        msg = self._format_message("INFO", "📊 METRIC", metric_name,
                                   value=value, unit=unit, **tags)
        self.logger.info(msg)

    @contextmanager
    def operation(self, operation_name: str, **context):
        """Context manager for operations - automatically logs start/end."""
        start_time = time.time()
        self.operation_start(operation_name, **context)
        success = False
        try:
            yield
            success = True
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.operation_end(operation_name, success=success, duration_ms=duration_ms)


def set_correlation_id(corr_id: str):
    """Set correlation ID for current thread (to track related operations)."""
    _thread_local.correlation_id = corr_id


def clear_correlation_id():
    """Clear correlation ID for current thread."""
    if hasattr(_thread_local, 'correlation_id'):
        delattr(_thread_local, 'correlation_id')


def get_correlation_id() -> Optional[str]:
    """Get current thread's correlation ID."""
    return getattr(_thread_local, 'correlation_id', None)


# Convenience function to create loggers
def get_logger(component: str) -> StructuredLogger:
    """Get a structured logger for a component."""
    return StructuredLogger(component)


# Example grep patterns for common debugging tasks:
"""
Common search patterns for AI-assisted debugging:

1. Find all operations:
   grep "▶ START\|◀ END" advisor.log

2. Find failed operations:
   grep "✗ FAILED" advisor.log

3. Find all events:
   grep "⚡ EVENT" advisor.log

4. Find all errors:
   grep "❌ ERROR" advisor.log

5. Find operations for specific component:
   grep "\[MTGA_PARSER\]" advisor.log

6. Track a specific correlation ID:
   grep "CORR:abc123" advisor.log

7. Find state snapshots:
   grep "📸 STATE" advisor.log

8. Find performance metrics:
   grep "📊 METRIC" advisor.log

9. Find slow operations (> 100ms):
   grep "duration_ms" advisor.log | awk -F'duration_ms=' '{if ($2+0 > 100) print}'

10. Get timeline of a match:
    grep "turn=" advisor.log | grep "▶ START\|◀ END"
"""
