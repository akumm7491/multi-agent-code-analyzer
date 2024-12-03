from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from fastmcp import Context
from multi_agent_code_analyzer.config.settings import get_settings
from multi_agent_code_analyzer.storage.mcp_storage import MCPStorageService, MCPContext

logger = logging.getLogger(__name__)


class FastMCPAdapter:
    def __init__(self, server_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.settings = get_settings()
        self.server_url = server_url
        self.api_key = api_key
        self.timeout = timeout
        self.storage = MCPStorageService()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.storage.close()

    async def store_context(self, content: str, metadata: Dict[str, Any], project_id: str) -> str:
        """Store context with metadata"""
        try:
            # Create context object with metadata
            context = Context(
                content=content,
                metadata=metadata
            )

            # Get embedding from FastMCP
            embedding = await context.get_embedding()

            # Create MCP context
            mcp_context = MCPContext(
                context_id=str(uuid.uuid4()),
                content=content,
                embedding=embedding,
                metadata=metadata,
                timestamp=datetime.utcnow(),
                project_id=project_id
            )

            # Store in persistent storage
            success = await self.storage.store_context(mcp_context)
            if not success:
                raise Exception("Failed to store context")

            return mcp_context.context_id

        except Exception as e:
            logger.error(f"Error storing context: {e}")
            raise

    async def search_similar(self, query: str, project_id: str, limit: int = 5) -> List[MCPContext]:
        """Search for similar contexts"""
        try:
            # Create context object for query
            context = Context(content=query)

            # Get query embedding
            embedding = await context.get_embedding()

            # Search in storage
            similar_contexts = await self.storage.search_similar_contexts(
                embedding=embedding,
                project_id=project_id,
                limit=limit
            )

            return similar_contexts

        except Exception as e:
            logger.error(f"Error searching contexts: {e}")
            return []

    async def get_project_timeline(self, project_id: str) -> List[MCPContext]:
        """Get chronological timeline of contexts for a project"""
        try:
            return await self.storage.get_project_timeline(project_id)
        except Exception as e:
            logger.error(f"Error getting project timeline: {e}")
            return []

    async def cleanup_old_contexts(self, project_id: str):
        """Clean up old contexts while preserving important ones"""
        try:
            await self.storage.cleanup_old_contexts(
                project_id=project_id,
                days_to_keep=self.settings.mcp.CONTEXT_RETENTION_DAYS
            )
        except Exception as e:
            logger.error(f"Error cleaning up contexts: {e}")


# Create FastAPI application

app = FastAPI(title="MCP Server")


class ContextRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
    project_id: str


class SearchRequest(BaseModel):
    query: str
    project_id: str
    limit: Optional[int] = 5


@app.post("/context")
async def create_context(request: ContextRequest):
    """Create a new context"""
    settings = get_settings()
    adapter = FastMCPAdapter(
        settings.mcp.MCP_SERVER_URL, settings.mcp.MCP_API_KEY)

    try:
        context_id = await adapter.store_context(
            content=request.content,
            metadata=request.metadata,
            project_id=request.project_id
        )
        return {"context_id": context_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_contexts(request: SearchRequest):
    """Search for similar contexts"""
    settings = get_settings()
    adapter = FastMCPAdapter(
        settings.mcp.MCP_SERVER_URL, settings.mcp.MCP_API_KEY)

    try:
        contexts = await adapter.search_similar(
            query=request.query,
            project_id=request.project_id,
            limit=request.limit
        )
        return {"contexts": [context.__dict__ for context in contexts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/timeline/{project_id}")
async def get_timeline(project_id: str):
    """Get project timeline"""
    settings = get_settings()
    adapter = FastMCPAdapter(
        settings.mcp.MCP_SERVER_URL, settings.mcp.MCP_API_KEY)

    try:
        contexts = await adapter.get_project_timeline(project_id)
        return {"contexts": [context.__dict__ for context in contexts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup/{project_id}")
async def cleanup_contexts(project_id: str):
    """Clean up old contexts"""
    settings = get_settings()
    adapter = FastMCPAdapter(
        settings.mcp.MCP_SERVER_URL, settings.mcp.MCP_API_KEY)

    try:
        await adapter.cleanup_old_contexts(project_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
