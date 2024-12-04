from typing import Dict, Any, Optional, List
import aiohttp
import logging
from datetime import datetime


class MCPClient:
    """Client for interacting with Model Context Protocol server."""

    def __init__(self, host: str, port: int, api_key: str):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def create_context(self, model_id: str, task_type: str,
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new MCP context."""
        try:
            async with self.session.post(
                f"{self.base_url}/v1/context",
                json={
                    "model_id": model_id,
                    "task_type": task_type,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                }
            ) as response:
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to create MCP context: {str(e)}")
            return {}

    async def verify_content(self, context: Dict[str, Any], content: str,
                             verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify content through MCP."""
        try:
            async with self.session.post(
                f"{self.base_url}/v1/verify",
                json={
                    "context": context,
                    "content": content,
                    "verification_data": verification_data,
                    "timestamp": datetime.now().isoformat()
                }
            ) as response:
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to verify content: {str(e)}")
            return {}

    async def get_embeddings(self, content: str, content_type: str) -> Dict[str, Any]:
        """Get embeddings from MCP."""
        try:
            async with self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "content": content,
                    "content_type": content_type,
                    "timestamp": datetime.now().isoformat()
                }
            ) as response:
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get embeddings: {str(e)}")
            return {}

    async def get_patterns(self) -> Dict[str, Any]:
        """Get patterns from MCP."""
        try:
            async with self.session.get(
                f"{self.base_url}/v1/patterns"
            ) as response:
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get patterns: {str(e)}")
            return {}

    async def get_component_criticality(self, component_id: str,
                                        component_type: str) -> float:
        """Get component criticality score from MCP."""
        try:
            async with self.session.get(
                f"{self.base_url}/v1/components/{component_id}/criticality",
                params={"type": component_type}
            ) as response:
                result = await response.json()
                return result.get("criticality", 0.0)

        except Exception as e:
            self.logger.error(f"Failed to get component criticality: {str(e)}")
            return 0.0

    async def get_impact_history(self, component_id: str) -> Dict[str, Any]:
        """Get impact history for a component from MCP."""
        try:
            async with self.session.get(
                f"{self.base_url}/v1/components/{component_id}/impact-history"
            ) as response:
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get impact history: {str(e)}")
            return {"impacts": []}

    async def submit_feedback(self, context: Dict[str, Any],
                              feedback: Dict[str, Any]) -> bool:
        """Submit feedback to MCP."""
        try:
            async with self.session.post(
                f"{self.base_url}/v1/feedback",
                json={
                    "context": context,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat()
                }
            ) as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {str(e)}")
            return False

    async def get_verification_strategies(self) -> List[Dict[str, Any]]:
        """Get available verification strategies from MCP."""
        try:
            async with self.session.get(
                f"{self.base_url}/v1/verification/strategies"
            ) as response:
                result = await response.json()
                return result.get("strategies", [])

        except Exception as e:
            self.logger.error(
                f"Failed to get verification strategies: {str(e)}")
            return []
