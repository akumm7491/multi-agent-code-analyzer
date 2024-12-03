from typing import Dict, Any, List, Optional
import aiohttp
import json
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime, timedelta
from functools import wraps
from .events import ContextEvent, ContextEventType, ContextEventPublisher


@dataclass
class MCPContext:
    """Model Context Protocol context structure"""
    content: str
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    embeddings: Optional[List[float]] = None
    last_updated: Optional[datetime] = None


class CacheEntry:
    def __init__(self, context: MCPContext, ttl_seconds: int = 300):
        self.context = context
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


def with_retry(max_retries: int = 3, delay_seconds: float = 1.0):
    """Decorator for implementing retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay_seconds * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator


class MCPAdapter:
    """Enhanced adapter for Model Context Protocol integration"""

    def __init__(self, mcp_server_url: str = "http://localhost:8080", cache_ttl: int = 300):
        self.server_url = mcp_server_url
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl = cache_ttl
        self.event_publisher = ContextEventPublisher()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    @with_retry(max_retries=3)
    async def store_context(self, context_id: str, context: MCPContext) -> bool:
        """Store context using MCP server with retry logic"""
        context.last_updated = datetime.now()
        session = self._get_session()

        try:
            async with session.post(
                f"{self.server_url}/contexts/{context_id}",
                json=asdict(context)
            ) as response:
                success = response.status == 200
                if success:
                    self.cache[context_id] = CacheEntry(
                        context, self.cache_ttl)
                    await self.event_publisher.publish(
                        ContextEvent(
                            event_type=ContextEventType.CONTEXT_CREATED,
                            context_id=context_id,
                            timestamp=datetime.now(),
                            data=asdict(context)
                        )
                    )
                return success
        except Exception as e:
            print(f"Error storing context: {e}")
            raise

    async def retrieve_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve context with caching"""
        # Check cache first
        if context_id in self.cache and not self.cache[context_id].is_expired():
            return self.cache[context_id].context

        session = self._get_session()
        try:
            async with session.get(
                f"{self.server_url}/contexts/{context_id}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    context = MCPContext(**data)
                    self.cache[context_id] = CacheEntry(
                        context, self.cache_ttl)
                    return context
                return None
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return None

    @with_retry(max_retries=2)
    async def search_similar_contexts(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Enhanced similarity search with minimum similarity threshold"""
        session = self._get_session()
        try:
            async with session.post(
                f"{self.server_url}/search",
                json={
                    "embedding": query_embedding,
                    "limit": limit,
                    "min_similarity": min_similarity
                }
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    # Cache the retrieved contexts
                    for result in results:
                        if "context" in result:
                            context_id = result["context_id"]
                            self.cache[context_id] = CacheEntry(
                                MCPContext(**result["context"]),
                                self.cache_ttl
                            )
                    return results
                return []
        except Exception as e:
            print(f"Error searching contexts: {e}")
            raise

    @with_retry(max_retries=3)
    async def update_relationships(
        self,
        context_id: str,
        relationships: List[Dict[str, Any]]
    ) -> bool:
        """Update relationships with event publishing"""
        session = self._get_session()
        try:
            async with session.put(
                f"{self.server_url}/contexts/{context_id}/relationships",
                json={"relationships": relationships}
            ) as response:
                success = response.status == 200
                if success:
                    # Invalidate cache for this context
                    self.cache.pop(context_id, None)
                    await self.event_publisher.publish(
                        ContextEvent(
                            event_type=ContextEventType.RELATIONSHIP_ADDED,
                            context_id=context_id,
                            timestamp=datetime.now(),
                            data={"relationships": relationships}
                        )
                    )
                return success
        except Exception as e:
            print(f"Error updating relationships: {e}")
            raise

    async def invalidate_cache(self, context_id: Optional[str] = None):
        """Invalidate cache entries"""
        if context_id:
            self.cache.pop(context_id, None)
        else:
            self.cache.clear()

    def subscribe_to_events(self, event_type: ContextEventType, handler: callable):
        """Subscribe to context events"""
        self.event_publisher.subscribe(event_type, handler)

    def unsubscribe_from_events(self, event_type: ContextEventType, handler: callable):
        """Unsubscribe from context events"""
        self.event_publisher.unsubscribe(event_type, handler)
