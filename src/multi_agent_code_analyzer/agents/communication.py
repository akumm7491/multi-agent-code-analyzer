import asyncio
import json
from typing import Dict, Any, Callable, Awaitable, Optional
import aioredis
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    CODE_ANALYSIS = "code_analysis"
    CONTEXT_UPDATE = "context_update"
    TASK_COMPLETION = "task_completion"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


@dataclass
class AgentMessage:
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        data['message_type'] = MessageType(data['message_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }


class AgentCommunicationService:
    def __init__(self, redis_uri: str):
        self.redis_uri = redis_uri
        self.redis: Optional[aioredis.Redis] = None
        self.message_handlers: Dict[MessageType, list[Callable[[AgentMessage], Awaitable[None]]]] = {
            message_type: [] for message_type in MessageType
        }
        self.subscriptions: Dict[str, aioredis.client.PubSub] = {}

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_uri)
        logger.info("Connected to Redis for agent communication")

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    def register_handler(self, message_type: MessageType,
                         handler: Callable[[AgentMessage], Awaitable[None]]):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for message type: {message_type}")

    async def publish_message(self, message: AgentMessage):
        """Publish a message to Redis"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")

        channel = (f"agent:{message.recipient_id}"
                   if message.recipient_id
                   else "agent:broadcast")

        await self.redis.publish(
            channel,
            json.dumps(message.to_dict())
        )
        logger.debug(f"Published message {
                     message.message_id} to channel {channel}")

    async def subscribe(self, agent_id: str):
        """Subscribe to agent-specific and broadcast channels"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")

        # Create a new PubSub instance
        pubsub = self.redis.pubsub()

        # Subscribe to agent-specific and broadcast channels
        await pubsub.subscribe(f"agent:{agent_id}", "agent:broadcast")

        self.subscriptions[agent_id] = pubsub

        # Start message processing
        asyncio.create_task(self._process_messages(agent_id, pubsub))
        logger.info(f"Agent {agent_id} subscribed to messages")

    async def unsubscribe(self, agent_id: str):
        """Unsubscribe from channels"""
        if agent_id in self.subscriptions:
            pubsub = self.subscriptions[agent_id]
            await pubsub.unsubscribe()
            await pubsub.close()
            del self.subscriptions[agent_id]
            logger.info(f"Agent {agent_id} unsubscribed from messages")

    async def _process_messages(self, agent_id: str, pubsub: aioredis.client.PubSub):
        """Process incoming messages for an agent"""
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        agent_message = AgentMessage.from_dict(data)

                        # Process message with registered handlers
                        handlers = self.message_handlers[agent_message.message_type]
                        await asyncio.gather(
                            *[handler(agent_message) for handler in handlers]
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error in message processing loop: {e}")
        finally:
            await self.unsubscribe(agent_id)

    @staticmethod
    def create_message(
        message_type: MessageType,
        sender_id: str,
        content: Dict[str, Any],
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Create a new agent message"""
        return AgentMessage(
            message_id=str(uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id
        )
