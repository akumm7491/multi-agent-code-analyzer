from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class ContextEventType(Enum):
    CONTEXT_CREATED = "context_created"
    CONTEXT_UPDATED = "context_updated"
    CONTEXT_DELETED = "context_deleted"
    RELATIONSHIP_ADDED = "relationship_added"
    RELATIONSHIP_REMOVED = "relationship_removed"
    EMBEDDING_UPDATED = "embedding_updated"


@dataclass
class ContextEvent:
    """Domain event for context changes"""
    event_type: ContextEventType
    context_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ContextEventPublisher:
    """Publishes context events to subscribers"""

    def __init__(self):
        self.subscribers: Dict[ContextEventType, List[callable]] = {
            event_type: [] for event_type in ContextEventType
        }

    async def publish(self, event: ContextEvent):
        """Publish an event to all subscribers"""
        for subscriber in self.subscribers[event.event_type]:
            await subscriber(event)

    def subscribe(self, event_type: ContextEventType, handler: callable):
        """Subscribe to a specific event type"""
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: ContextEventType, handler: callable):
        """Unsubscribe from a specific event type"""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
