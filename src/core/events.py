"""
Event bus for decoupled communication between components.

This module provides a publish-subscribe event system to decouple components
in the MTGA Voice Advisor application. Components can emit events without
knowing who will consume them, and subscribe to events without knowing
who produced them.

Thread-safe for use across UI and background threads.
"""
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import threading

logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types in the application."""
    # Game state events
    BOARD_STATE_CHANGED = auto()
    PRIORITY_GAINED = auto()
    PRIORITY_LOST = auto()
    TURN_CHANGED = auto()
    PHASE_CHANGED = auto()

    # Game lifecycle events
    GAME_STARTED = auto()
    GAME_ENDED = auto()
    MATCH_FOUND = auto()

    # Draft events
    DRAFT_STARTED = auto()
    DRAFT_PACK_OPENED = auto()
    DRAFT_PICK_MADE = auto()
    DRAFT_ENDED = auto()

    # Card events
    CARD_PLAYED = auto()
    CARD_DRAWN = auto()
    CREATURE_DIED = auto()

    # UI events
    ADVICE_READY = auto()
    STATUS_UPDATE = auto()
    ERROR_OCCURRED = auto()


@dataclass
class Event:
    """
    Base event class with type and optional data.

    Attributes:
        event_type: The type of event being emitted
        data: Optional data payload for the event
        source: Optional source identifier (e.g., "GameStateManager", "DraftAdvisor")
    """
    event_type: EventType
    data: Any = None
    source: str = ""


class EventBus:
    """
    Central event bus for publish-subscribe communication.

    This class implements a thread-safe event bus that allows components to
    communicate without direct coupling. Handlers are executed synchronously
    in the order they were registered.

    Thread-safe for use across UI and background threads.

    Example usage:
        >>> bus = EventBus.get_instance()
        >>>
        >>> # Subscribe to specific events
        >>> def on_game_started(event: Event):
        ...     print(f"Game started: {event.data}")
        >>> bus.subscribe(EventType.GAME_STARTED, on_game_started)
        >>>
        >>> # Emit events
        >>> bus.emit_simple(EventType.GAME_STARTED, {"match_id": "12345"}, source="GameStateManager")
    """
    _instance: Optional['EventBus'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._global_handlers: List[Callable[[Event], None]] = []
        self._handler_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'EventBus':
        """
        Get singleton instance of EventBus.

        Returns:
            The global EventBus instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern for thread safety
                if cls._instance is None:
                    cls._instance = EventBus()
        return cls._instance

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Subscribe to a specific event type.

        The handler will be called whenever an event of the specified type
        is emitted. Handlers are called synchronously in the order they
        were registered.

        Args:
            event_type: The type of event to subscribe to
            handler: Callable that accepts an Event object
        """
        with self._handler_lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.name}")

    def subscribe_all(self, handler: Callable[[Event], None]):
        """
        Subscribe to all events (for logging/debugging).

        The handler will be called for every event emitted, regardless
        of event type. Useful for debugging, logging, or monitoring.

        Args:
            handler: Callable that accepts an Event object
        """
        with self._handler_lock:
            self._global_handlers.append(handler)
        logger.debug("Subscribed global event handler")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Unsubscribe from a specific event type.

        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
        """
        with self._handler_lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.name}")

    def emit(self, event: Event):
        """
        Emit an event to all subscribed handlers.

        Handlers are called synchronously. If a handler raises an exception,
        it is logged but does not prevent other handlers from executing.

        Args:
            event: The Event object to emit
        """
        # Create a snapshot of handlers to avoid issues if handlers modify subscriptions
        with self._handler_lock:
            specific_handlers = list(self._handlers.get(event.event_type, []))
            global_handlers = list(self._global_handlers)

        # Call specific handlers
        for handler in specific_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type.name}: {e}", exc_info=True)

        # Call global handlers
        for handler in global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}", exc_info=True)

    def emit_simple(self, event_type: EventType, data: Any = None, source: str = ""):
        """
        Convenience method to emit an event without creating Event object.

        Args:
            event_type: The type of event to emit
            data: Optional data payload
            source: Optional source identifier
        """
        self.emit(Event(event_type=event_type, data=data, source=source))

    def clear(self):
        """
        Clear all handlers (useful for testing).

        This removes all registered handlers from the event bus.
        """
        with self._handler_lock:
            self._handlers.clear()
            self._global_handlers.clear()
        logger.debug("Cleared all event handlers")


# Convenience function for global access
def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.

    Returns:
        The global EventBus singleton
    """
    return EventBus.get_instance()
