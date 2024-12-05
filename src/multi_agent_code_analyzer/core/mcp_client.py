from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
from ..config import settings
import os
from dotenv import load_dotenv
import argparse
from fastapi import FastAPI
import uvicorn

# Load environment variables
load_dotenv()


class FastMCPClient:
    def __init__(self):
        self.base_url = settings.FASTMCP_URL
        self.client = httpx.AsyncClient()
        self.store_type = settings.FASTMCP_STORE_TYPE
        self.embedding_model = settings.FASTMCP_EMBEDDING_MODEL
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.prometheus_host = os.getenv('PROMETHEUS_HOST', 'prometheus')
        self.prometheus_port = int(os.getenv('PROMETHEUS_PORT', 9090))

    async def store_context(self, context: Dict[str, Any]) -> str:
        """Store context in FastMCP."""
        response = await self.client.post(
            f"{self.base_url}/context",
            json={
                "context": context,
                "store_type": self.store_type,
                "embedding_model": self.embedding_model,
                "timestamp": datetime.now().isoformat()
            }
        )
        response.raise_for_status()
        return response.json()["context_id"]

    async def get_context(self, context_id: str) -> Dict[str, Any]:
        """Retrieve context from FastMCP."""
        response = await self.client.get(
            f"{self.base_url}/context/{context_id}"
        )
        response.raise_for_status()
        return response.json()["context"]

    async def update_context(self, context: Dict[str, Any]) -> None:
        """Update existing context in FastMCP."""
        response = await self.client.put(
            f"{self.base_url}/context/{context['context_id']}",
            json={
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        )
        response.raise_for_status()

    async def get_context_for_file(
        self,
        file_path: str,
        line_number: int
    ) -> Dict[str, Any]:
        """Get context for a specific file and line number."""
        response = await self.client.get(
            f"{self.base_url}/context/file",
            params={
                "file_path": file_path,
                "line_number": line_number
            }
        )
        response.raise_for_status()
        return response.json()["context"]

    async def search_similar_context(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar context using semantic similarity."""
        response = await self.client.post(
            f"{self.base_url}/context/search",
            json={
                "query": query,
                "limit": limit,
                "embedding_model": self.embedding_model
            }
        )
        response.raise_for_status()
        return response.json()["results"]

    async def get_context_by_pr(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get context associated with a pull request."""
        response = await self.client.get(
            f"{self.base_url}/context/pr/{pr_number}"
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()["context"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-concurrent-rpcs", type=int, default=1000)
    args = parser.parse_args()

    # Start the server
    uvicorn.run("src.multi_agent_code_analyzer.core.mcp_client:app",
                host=args.host,
                port=args.port,
                workers=args.max_workers)
