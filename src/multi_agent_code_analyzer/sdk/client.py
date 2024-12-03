from __future__ import annotations
import aiohttp
import asyncio
from typing import Optional, Dict, List, Union, TypeVar, cast
import logging
from enum import Enum

# Define JSON value type
JsonValue = Union[str, int, float, bool, None,
                  Dict[str, 'JsonValue'], List['JsonValue']]
JsonDict = Dict[str, JsonValue]


class AgentType(str, Enum):
    CODE_ANALYZER = "code_analyzer"
    DEVELOPER = "developer"
    ORCHESTRATOR = "orchestrator"


class Client:
    """Client for interacting with the Multi-Agent Code Analyzer API"""

    def __init__(self, base_url: str = "http://localhost:8000", github_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.github_token = github_token
        self.logger = logging.getLogger(__name__)

    async def analyze_repository(
        self,
        repository_url: str,
        branch: str = "main",
        analysis_type: str = "full",
        wait_for_completion: bool = False,
        timeout: int = 300
    ) -> JsonDict:
        """Analyze a repository using the multi-agent system"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "repository_url": repository_url,
                    "branch": branch,
                    "analysis_type": analysis_type,
                    "wait_for_completion": wait_for_completion,
                    "timeout": timeout
                }

                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.post(
                    f"{self.base_url}/analyze/repository",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to analyze repository: {str(e)}")
            raise

    async def implement_feature(
        self,
        repo_url: str,
        description: str,
        branch: Optional[str] = None,
        target_files: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        timeout: int = 600
    ) -> JsonDict:
        """Implement a feature or fix"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "repo_url": repo_url,
                    "description": description,
                    "branch": branch,
                    "target_files": target_files,
                    "wait_for_completion": wait_for_completion,
                    "timeout": timeout
                }

                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.post(
                    f"{self.base_url}/implement/feature",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to implement feature: {str(e)}")
            raise

    async def get_task_status(self, task_id: str) -> JsonDict:
        """Get the status of a task"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.get(
                    f"{self.base_url}/analysis/{task_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to get task status: {str(e)}")
            raise

    async def get_agent_memory(self, agent_id: str) -> JsonDict:
        """Get an agent's memory"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.get(
                    f"{self.base_url}/agent/{agent_id}/memory",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to get agent memory: {str(e)}")
            raise

    async def get_agent_learnings(self, agent_id: str) -> JsonDict:
        """Get an agent's learning points"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.get(
                    f"{self.base_url}/agent/{agent_id}/learnings",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to get agent learnings: {str(e)}")
            raise

    async def create_custom_task(
        self,
        agent_type: AgentType,
        description: str,
        context: JsonDict,
        dependencies: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        timeout: int = 300
    ) -> JsonDict:
        """Create a custom task"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "agent_type": agent_type,
                    "description": description,
                    "context": context,
                    "dependencies": dependencies,
                    "wait_for_completion": wait_for_completion,
                    "timeout": timeout
                }

                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"Bearer {self.github_token}"

                async with session.post(
                    f"{self.base_url}/task/custom",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return cast(JsonDict, await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to create custom task: {str(e)}")
            raise

# Example usage


async def main():
    async with Client(github_token="your_token") as client:
        # Analyze repository
        analysis = await client.analyze_repository(
            "https://github.com/username/repo",
            wait_for_completion=True
        )
        print("Analysis:", json.dumps(analysis, indent=2))

        # Implement feature
        result = await client.implement_feature(
            "https://github.com/username/repo",
            "Add user authentication with OAuth2",
            wait_for_completion=True
        )
        print("Implementation:", json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
