from typing import Dict, Any, List, Optional
import asyncio
import logging
from .base import BaseAgent
from .code_analyzer import CodeAnalyzerAgent
from .developer import DeveloperAgent


class AgentManager:
    """Manager for coordinating multiple agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def register_agent(self, agent_type: str, agent: BaseAgent):
        """Register an agent with the manager"""
        self.agents[agent_type] = agent
        self.logger.info(
            f"Registered agent {agent.agent_id} of type {agent_type}")

    async def start_analysis(self, repo_path: str, analysis_type: str = "full") -> str:
        """Start repository analysis"""
        try:
            # Get code analyzer agent
            code_analyzer = self.agents.get("code_analyzer")
            if not code_analyzer:
                raise Exception("Code analyzer agent not registered")

            # Create task
            task_id = code_analyzer.agent_id
            self.tasks[task_id] = {
                "status": "in_progress",
                "result": None
            }

            # Execute analysis
            result = await code_analyzer.execute({
                "type": "analyze",
                "repo_path": repo_path,
                "analysis_type": analysis_type
            })

            # Update task status
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result

            return task_id

        except Exception as e:
            self.logger.error(f"Failed to start analysis: {str(e)}")
            raise

    async def start_implementation(
        self,
        repo_url: str,
        description: str,
        branch: Optional[str] = None,
        target_files: Optional[List[str]] = None
    ) -> str:
        """Start feature implementation"""
        try:
            # Get developer agent
            developer = self.agents.get("developer")
            if not developer:
                raise Exception("Developer agent not registered")

            # Create task
            task_id = developer.agent_id
            self.tasks[task_id] = {
                "status": "in_progress",
                "result": None
            }

            # Execute implementation
            result = await developer.execute({
                "type": "implement",
                "repo_url": repo_url,
                "description": description,
                "branch": branch,
                "target_files": target_files
            })

            # Update task status
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result

            return task_id

        except Exception as e:
            self.logger.error(f"Failed to start implementation: {str(e)}")
            raise

    async def start_custom_task(
        self,
        agent_type: str,
        description: str,
        context: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Start a custom task"""
        try:
            # Get agent
            agent = self.agents.get(agent_type)
            if not agent:
                raise Exception(f"Agent of type {agent_type} not registered")

            # Create task
            task_id = agent.agent_id
            self.tasks[task_id] = {
                "status": "in_progress",
                "result": None
            }

            # Execute task
            result = await agent.execute({
                "type": "custom",
                "description": description,
                "context": context,
                "dependencies": dependencies or []
            })

            # Update task status
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result

            return task_id

        except Exception as e:
            self.logger.error(f"Failed to start custom task: {str(e)}")
            raise

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task"""
        task = self.tasks.get(task_id)
        if not task:
            raise Exception(f"Task {task_id} not found")
        return task

    async def wait_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for a task to complete and return its result"""
        while True:
            task = await self.get_task_status(task_id)
            if task["status"] == "completed":
                return task["result"]
            elif task["status"] == "failed":
                raise Exception(f"Task {task_id} failed: {task.get('error')}")
            await asyncio.sleep(1)
