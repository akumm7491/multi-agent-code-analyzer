from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Dict, Any, List
from redis import asyncio as aioredis
from ..config import settings

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageBus:
    def __init__(self):
        self.redis = aioredis.from_url(settings.REDIS_URL)
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}

    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic."""
        await self.redis.publish(topic, str(message))
        if topic in self.subscribers:
            for queue in self.subscribers[topic]:
                await queue.put(message)

    async def subscribe(self, topic: str) -> asyncio.Queue:
        """Subscribe to a topic."""
        queue = asyncio.Queue()
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue):
        """Unsubscribe from a topic."""
        if topic in self.subscribers and queue in self.subscribers[topic]:
            self.subscribers[topic].remove(queue)


message_bus = MessageBus()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        await message_bus.redis.ping()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Message bus unhealthy: {str(e)}"
        )


@app.post("/publish/{topic}")
async def publish_message(topic: str, message: Dict[str, Any]):
    """Publish a message to a topic."""
    try:
        await message_bus.publish(topic, message)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish message: {str(e)}"
        )


@app.websocket("/subscribe/{topic}")
async def subscribe_to_topic(websocket, topic: str):
    """Subscribe to a topic via WebSocket."""
    queue = await message_bus.subscribe(topic)
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    finally:
        await message_bus.unsubscribe(topic, queue)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "message_bus:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.DEBUG
    )
