"""Centralized Event Bus.

Provides a publish/subscribe event bus for decoupled communication
between framework components. Supports async event handling,
prioritized subscriptions, and event filtering.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event handlers."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Event:
    """Event data container."""

    event_type: str
    payload: Any
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    metadata: Dict[str, Any] = field(default_factory=dict)
    seq_no: int = 0  # assigned by EventBus.publish, per bus
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    utc_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "seq_no": self.seq_no,
            "monotonic_ns": self.monotonic_ns,
            "utc_ns": self.utc_ns,
            "metadata": self.metadata,
        }


@dataclass
class EventSubscription:
    """Subscription to an event type."""

    event_type: str
    handler: Callable[[Event], Any]
    priority: EventPriority = EventPriority.NORMAL
    filter_fn: Optional[Callable[[Event], bool]] = None
    once: bool = False
    subscription_id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex}")


class EventBus:
    """Centralized Event Bus for decoupled component communication.

    Features:
    - Publish/subscribe pattern
    - Async event handling
    - Priority-based handler execution
    - Event filtering
    - One-time subscriptions
    - Event replay capability
    """

    def __init__(self) -> None:
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._event_history: List[Event] = []
        self._history_limit = 1000
        self._next_seq = 1
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        logger.info("EventBus stopped")

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        once: bool = False,
    ) -> EventSubscription:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function for event handling
            priority: Handler priority (lower = higher priority)
            filter_fn: Optional filter function
            once: If True, unsubscribe after first event

        Returns:
            EventSubscription object
        """
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
            once=once,
        )

        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []

        self._subscriptions[event_type].append(subscription)

        # Sort by priority
        self._subscriptions[event_type].sort(key=lambda s: s.priority.value)

        logger.debug(f"Subscribed to {event_type}: {subscription.subscription_id}")
        return subscription

    def unsubscribe(self, subscription: EventSubscription) -> bool:
        """Unsubscribe from an event type.

        Args:
            subscription: Subscription to remove

        Returns:
            True if subscription was found and removed
        """
        if subscription.event_type not in self._subscriptions:
            return False

        subs = self._subscriptions[subscription.event_type]
        if subscription in subs:
            subs.remove(subscription)
            logger.debug(f"Unsubscribed from {subscription.event_type}")
            return True

        return False

    async def publish(self, event: Event) -> List[Any]:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish

        Returns:
            List of handler results
        """
        if not self._running:
            logger.warning("EventBus not running, event not published")
            return []

        event.seq_no = self._next_seq
        self._next_seq += 1

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history.pop(0)

        results = []
        subscriptions = self._subscriptions.get(event.event_type, [])

        # Collect one-time subscriptions to remove
        to_remove = []

        for subscription in subscriptions:
            # Check filter
            if subscription.filter_fn and not subscription.filter_fn(event):
                continue

            try:
                # Handle both async and sync handlers
                if asyncio.iscoroutinefunction(subscription.handler):
                    result = await subscription.handler(event)
                else:
                    result = subscription.handler(event)
                results.append(result)

                if subscription.once:
                    to_remove.append(subscription)

            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")
                results.append(None)

        # Remove one-time subscriptions
        for sub in to_remove:
            self.unsubscribe(sub)

        return results

    async def publish_event(
        self,
        event_type: str,
        payload: Any,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Convenience method to create and publish an event.

        Args:
            event_type: Type of event
            payload: Event data
            source: Event source identifier
            metadata: Additional metadata

        Returns:
            List of handler results
        """
        event = Event(
            event_type=event_type,
            payload=payload,
            source=source,
            metadata=metadata or {},
        )
        return await self.publish(event)

    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get recent event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events

        Returns:
            List of events
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_subscribers(self, event_type: Optional[str] = None) -> Dict[str, int]:
        """Get subscriber counts.

        Args:
            event_type: Filter by event type (optional)

        Returns:
            Dictionary of event_type -> subscriber_count
        """
        if event_type:
            return {event_type: len(self._subscriptions.get(event_type, []))}
        return {k: len(v) for k, v in self._subscriptions.items()}


class DomainEventBus:
    """Domain-specific event bus wrapper with predefined event types.

    Provides type-safe event publishing for specific domains.
    """

    # Predefined event types
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_COMPLETED = "session.completed"
    DATA_PROCESSED = "data.processed"
    ANALYSIS_COMPLETED = "analysis.completed"
    ERROR_OCCURRED = "error.occurred"
    CONFIG_CHANGED = "config.changed"
    COMPLIANCE_CHECK = "compliance.check"

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus

    async def publish_session_created(
        self, session_id: str, config: dict, source: str = ""
    ) -> None:
        """Publish session created event."""
        await self._bus.publish_event(
            self.SESSION_CREATED,
            {"session_id": session_id, "config": config},
            source,
        )

    async def publish_session_completed(
        self, session_id: str, results: dict, source: str = ""
    ) -> None:
        """Publish session completed event."""
        await self._bus.publish_event(
            self.SESSION_COMPLETED,
            {"session_id": session_id, "results": results},
            source,
        )

    async def publish_data_processed(
        self, data_id: str, processing_type: str, metrics: dict, source: str = ""
    ) -> None:
        """Publish data processed event."""
        await self._bus.publish_event(
            self.DATA_PROCESSED,
            {"data_id": data_id, "processing_type": processing_type, "metrics": metrics},
            source,
        )

    async def publish_analysis_completed(
        self, analysis_id: str, analysis_type: str, results: dict, source: str = ""
    ) -> None:
        """Publish analysis completed event."""
        await self._bus.publish_event(
            self.ANALYSIS_COMPLETED,
            {"analysis_id": analysis_id, "analysis_type": analysis_type, "results": results},
            source,
        )

    async def publish_error(
        self, error_type: str, message: str, context: dict, source: str = ""
    ) -> None:
        """Publish error event."""
        await self._bus.publish_event(
            self.ERROR_OCCURRED,
            {"error_type": error_type, "message": message, "context": context},
            source,
        )

    def subscribe_session_created(
        self, handler: Callable[[Event], Any], priority: EventPriority = EventPriority.NORMAL
    ) -> EventSubscription:
        """Subscribe to session created events."""
        return self._bus.subscribe(self.SESSION_CREATED, handler, priority)

    def subscribe_session_completed(
        self, handler: Callable[[Event], Any], priority: EventPriority = EventPriority.NORMAL
    ) -> EventSubscription:
        """Subscribe to session completed events."""
        return self._bus.subscribe(self.SESSION_COMPLETED, handler, priority)

    def subscribe_data_processed(
        self, handler: Callable[[Event], Any], priority: EventPriority = EventPriority.NORMAL
    ) -> EventSubscription:
        """Subscribe to data processed events."""
        return self._bus.subscribe(self.DATA_PROCESSED, handler, priority)


# Global event bus instance
_event_bus: Optional[EventBus] = None
_domain_event_bus: Optional[DomainEventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def get_domain_event_bus() -> DomainEventBus:
    """Get or create global domain event bus."""
    global _domain_event_bus
    if _domain_event_bus is None:
        _domain_event_bus = DomainEventBus(get_event_bus())
    return _domain_event_bus
