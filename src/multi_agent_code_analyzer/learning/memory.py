from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
from uuid import UUID
import json
import numpy as np
from pydantic import BaseModel


class MemoryEntry(BaseModel):
    id: UUID
    context_id: UUID
    importance: float
    last_accessed: datetime
    access_count: int
    decay_rate: float = 0.1
    metadata: Dict[str, Any] = {}


class MemoryManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or "redis://redis:6379"
        self.redis = redis.from_url(self.redis_url)
        self.memory_buffer_size = int(
            os.getenv("MCP_MEMORY_BUFFER_SIZE", "50000"))
        self._cleanup_task = None

    async def initialize(self):
        """Initialize memory manager and start cleanup task"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a new memory entry"""
        try:
            # Store memory data
            await self.redis.hset(
                f"memory:{str(memory.id)}",
                mapping={
                    "context_id": str(memory.context_id),
                    "importance": str(memory.importance),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "access_count": str(memory.access_count),
                    "decay_rate": str(memory.decay_rate),
                    "metadata": json.dumps(memory.metadata)
                }
            )

            # Add to sorted set for importance-based retrieval
            await self.redis.zadd(
                "memory_importance",
                {str(memory.id): memory.importance}
            )

            return True
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def update_access(self, context_id: UUID) -> None:
        """Update memory access patterns"""
        try:
            memory_key = f"memory:{str(context_id)}"
            if not await self.redis.exists(memory_key):
                return

            # Update access count and timestamp
            pipe = self.redis.pipeline()
            pipe.hincrby(memory_key, "access_count", 1)
            pipe.hset(memory_key, "last_accessed",
                      datetime.utcnow().isoformat())

            # Recalculate importance
            memory_data = await self.redis.hgetall(memory_key)
            access_count = int(memory_data[b"access_count"]) + 1
            last_accessed = datetime.fromisoformat(
                memory_data[b"last_accessed"].decode())
            decay_rate = float(memory_data[b"decay_rate"])

            importance = self._calculate_importance(
                access_count,
                last_accessed,
                decay_rate
            )

            pipe.hset(memory_key, "importance", str(importance))
            pipe.zadd("memory_importance", {str(context_id): importance})

            await pipe.execute()
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")

    async def get_important_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most important memories"""
        try:
            # Get top memories by importance
            memory_ids = await self.redis.zrevrange(
                "memory_importance",
                0,
                limit - 1,
                withscores=True
            )

            memories = []
            for memory_id, importance in memory_ids:
                memory_data = await self.redis.hgetall(f"memory:{memory_id.decode()}")
                if memory_data:
                    memories.append(MemoryEntry(
                        id=UUID(memory_id.decode()),
                        context_id=UUID(memory_data[b"context_id"].decode()),
                        importance=float(memory_data[b"importance"]),
                        last_accessed=datetime.fromisoformat(
                            memory_data[b"last_accessed"].decode()),
                        access_count=int(memory_data[b"access_count"]),
                        decay_rate=float(memory_data[b"decay_rate"]),
                        metadata=json.loads(memory_data[b"metadata"])
                    ))

            return memories
        except Exception as e:
            logger.error(f"Failed to get important memories: {e}")
            return []

    def _calculate_importance(
        self,
        access_count: int,
        last_accessed: datetime,
        decay_rate: float
    ) -> float:
        """Calculate memory importance based on access patterns and time decay"""
        time_factor = np.exp(
            -decay_rate * (datetime.utcnow() -
                           last_accessed).total_seconds() / 86400
        )
        frequency_factor = np.log1p(access_count)
        return time_factor * frequency_factor

    async def _periodic_cleanup(self):
        """Periodically clean up old or less important memories"""
        while True:
            try:
                # Get total memory count
                total_memories = await self.redis.zcard("memory_importance")

                if total_memories > self.memory_buffer_size:
                    # Remove excess memories with lowest importance
                    to_remove = total_memories - self.memory_buffer_size
                    memory_ids = await self.redis.zrange(
                        "memory_importance",
                        0,
                        to_remove - 1
                    )

                    pipe = self.redis.pipeline()
                    for memory_id in memory_ids:
                        pipe.delete(f"memory:{memory_id.decode()}")
                        pipe.zrem("memory_importance", memory_id)
                    await pipe.execute()

                    logger.info(f"Cleaned up {to_remove} memories")

            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")

            await asyncio.sleep(3600)  # Run every hour

    async def close(self):
        """Clean up resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.redis.close()
