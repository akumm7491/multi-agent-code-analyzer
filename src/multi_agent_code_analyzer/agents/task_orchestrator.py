from typing import Dict, List, Set
import asyncio
from .agent_manager import AgentTask, AgentManager
import logging


class TaskOrchestrator:
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.completed_tasks: Set[str] = set()
        self.logger = logging.getLogger(__name__)

    async def schedule_task(self, task: AgentTask):
        """Schedule a task and its dependencies for execution"""
        self.task_dependencies[task.task_id] = set(task.dependencies)
        await self._check_and_execute_tasks()

    async def _check_and_execute_tasks(self):
        """Check for tasks that can be executed and start them"""
        while True:
            executable_tasks = self._find_executable_tasks()
            if not executable_tasks:
                break

            # Execute tasks concurrently
            await asyncio.gather(*[
                self._execute_task(task_id)
                for task_id in executable_tasks
            ])

    def _find_executable_tasks(self) -> List[str]:
        """Find tasks whose dependencies are all completed"""
        executable = []
        for task_id, dependencies in self.task_dependencies.items():
            if task_id not in self.completed_tasks and dependencies.issubset(self.completed_tasks):
                executable.append(task_id)
        return executable

    async def _execute_task(self, task_id: str):
        """Execute a single task"""
        try:
            task = self.agent_manager.tasks.get(task_id)
            if not task:
                self.logger.error(f"Task {task_id} not found")
                return

            # Assign task to an agent
            await self.agent_manager.assign_task(task)

            # Wait for task completion
            while task.status not in ["completed", "failed"]:
                await asyncio.sleep(1)

            if task.status == "completed":
                self.completed_tasks.add(task_id)
                self.logger.info(f"Task {task_id} completed successfully")
            else:
                self.logger.error(
                    f"Task {task_id} failed: {task.result.get('error')}")

        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}")

    async def get_task_status(self, task_id: str) -> Dict:
        """Get detailed status of a task and its dependencies"""
        task = self.agent_manager.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}

        dependencies_status = {
            dep: "completed" if dep in self.completed_tasks else "pending"
            for dep in task.dependencies
        }

        return {
            "task_id": task_id,
            "status": task.status,
            "dependencies": dependencies_status,
            "result": task.result
        }

    def is_task_chain_complete(self, task_ids: List[str]) -> bool:
        """Check if all tasks in a chain are completed"""
        return all(task_id in self.completed_tasks for task_id in task_ids)
