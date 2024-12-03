from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import networkx as nx
import hashlib
import asyncio
from datetime import datetime
from .mcp_adapter import MCPAdapter, MCPContext
from .events import ContextEventType, ContextEvent, ContextEventPublisher


@dataclass
class CodeContext:
    file_path: str
    content: str
    dependencies: List[str]
    imports: List[str]
    references: List[str]
    metadata: Dict[str, Any]
    last_updated: datetime = datetime.now()


class ContextManager:
    """Manages code context and relationships across the codebase with enhanced MCP integration."""

    def __init__(self, mcp_server_url: str = "http://localhost:8080"):
        self.context_graph = nx.DiGraph()
        self.file_contexts: Dict[str, CodeContext] = {}
        self.mcp = MCPAdapter(mcp_server_url)
        self.event_publisher = ContextEventPublisher()

    async def __aenter__(self):
        """Context manager entry"""
        await self.mcp.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.mcp.__aexit__(exc_type, exc_val, exc_tb)

    def _generate_context_id(self, file_path: str) -> str:
        """Generate a unique context ID for MCP storage"""
        return hashlib.sha256(file_path.encode()).hexdigest()

    async def add_file_context(self, file_path: str, content: str, metadata: Dict[str, Any]):
        """Add or update context for a file with MCP integration."""
        # Create local context
        context = CodeContext(
            file_path=file_path,
            content=content,
            dependencies=[],
            imports=[],
            references=[],
            metadata=metadata,
            last_updated=datetime.now()
        )

        self.file_contexts[file_path] = context
        self.context_graph.add_node(file_path)

        # Store in MCP with retry
        context_id = self._generate_context_id(file_path)
        mcp_context = MCPContext(
            content=content,
            metadata={
                **metadata,
                "file_path": file_path,
                "dependencies": [],
                "imports": [],
                "references": []
            },
            relationships=[],
            embeddings=None
        )

        success = await self.mcp.store_context(context_id, mcp_context)
        if success:
            await self.event_publisher.publish(
                ContextEvent(
                    event_type=ContextEventType.CONTEXT_CREATED,
                    context_id=context_id,
                    timestamp=datetime.now(),
                    data={"file_path": file_path, "metadata": metadata}
                )
            )

    async def add_relationship(self, from_file: str, to_file: str, relationship_type: str):
        """Add a relationship between files with MCP integration."""
        if from_file in self.file_contexts and to_file in self.file_contexts:
            # Update local graph
            self.context_graph.add_edge(
                from_file, to_file, type=relationship_type)

            # Update MCP relationships with retry
            from_id = self._generate_context_id(from_file)
            to_id = self._generate_context_id(to_file)

            relationship = {
                "source_id": from_id,
                "target_id": to_id,
                "type": relationship_type,
                "timestamp": datetime.now().isoformat()
            }

            success = await self.mcp.update_relationships(from_id, [relationship])
            if success:
                await self.event_publisher.publish(
                    ContextEvent(
                        event_type=ContextEventType.RELATIONSHIP_ADDED,
                        context_id=from_id,
                        timestamp=datetime.now(),
                        data=relationship
                    )
                )

    async def get_related_files(
        self,
        file_path: str,
        max_depth: int = 2,
        min_similarity: float = 0.7
    ) -> List[str]:
        """Get related files using both graph traversal and semantic similarity."""
        if file_path not in self.context_graph:
            return []

        # Get local relationships through graph traversal
        related = set()
        current_depth = 0
        current_files = {file_path}

        while current_depth < max_depth and current_files:
            next_files = set()
            for file in current_files:
                neighbors = set(self.context_graph.predecessors(file)) | \
                    set(self.context_graph.successors(file))
                next_files.update(neighbors - related - {file})

            related.update(current_files)
            current_files = next_files
            current_depth += 1

        # Get semantically similar files from MCP
        context_id = self._generate_context_id(file_path)
        mcp_context = await self.mcp.retrieve_context(context_id)

        if mcp_context and mcp_context.embeddings:
            similar_contexts = await self.mcp.search_similar_contexts(
                mcp_context.embeddings,
                limit=10,
                min_similarity=min_similarity
            )

            for result in similar_contexts:
                if "metadata" in result and "file_path" in result["metadata"]:
                    related.add(result["metadata"]["file_path"])

        return list(related)

    async def get_context(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive context with both local and MCP data."""
        if file_path not in self.file_contexts:
            return None

        context = self.file_contexts[file_path]
        related_files = await self.get_related_files(file_path)

        # Get MCP context with semantic information
        context_id = self._generate_context_id(file_path)
        mcp_context = await self.mcp.retrieve_context(context_id)

        return {
            "file": context,
            "related_files": [
                self.file_contexts[f] for f in related_files
                if f in self.file_contexts
            ],
            "relationships": [
                {
                    "from": u,
                    "to": v,
                    "type": d["type"]
                }
                for u, v, d in self.context_graph.edges(data=True)
                if u == file_path or v == file_path
            ],
            "mcp_context": mcp_context._asdict() if mcp_context else None,
            "last_updated": context.last_updated.isoformat()
        }

    async def search_similar_contexts(
        self,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar contexts using MCP's vector similarity."""
        context = MCPContext(
            content=content,
            metadata={},
            relationships=[],
            embeddings=None
        )

        context_id = "temp_query"
        await self.mcp.store_context(context_id, context)

        query_context = await self.mcp.retrieve_context(context_id)
        if query_context and query_context.embeddings:
            return await self.mcp.search_similar_contexts(
                query_context.embeddings,
                limit,
                min_similarity
            )
        return []

    def subscribe_to_events(self, event_type: ContextEventType, handler: callable):
        """Subscribe to context events"""
        self.event_publisher.subscribe(event_type, handler)
        self.mcp.subscribe_to_events(event_type, handler)

    def unsubscribe_from_events(self, event_type: ContextEventType, handler: callable):
        """Unsubscribe from context events"""
        self.event_publisher.unsubscribe(event_type, handler)
        self.mcp.unsubscribe_from_events(event_type, handler)

    async def invalidate_cache(self, file_path: Optional[str] = None):
        """Invalidate cache entries"""
        if file_path:
            context_id = self._generate_context_id(file_path)
            await self.mcp.invalidate_cache(context_id)
        else:
            await self.mcp.invalidate_cache()
