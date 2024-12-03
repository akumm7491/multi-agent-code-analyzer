from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio
from ..tools.manager import ToolManager
from ..context.fastmcp_adapter import FastMCPAdapter, FastMCPContext


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeAnalysisResult:
    """Result of code analysis"""
    quality_score: float
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    dependencies: List[str]
    security_concerns: List[Dict[str, Any]]
    test_coverage: Optional[float] = None


@dataclass
class TaskResult:
    """Result of task execution"""
    success: bool
    status: TaskStatus
    artifacts: Dict[str, Any]
    analysis: Optional[CodeAnalysisResult] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TaskOrchestrator:
    """Orchestrates autonomous task execution"""

    def __init__(
        self,
        tool_manager: ToolManager,
        context_adapter: FastMCPAdapter,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tool_manager = tool_manager
        self.context_adapter = context_adapter
        self.config = config or {}
        self.task_queue: List[Dict[str, Any]] = []
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []

    async def submit_task(
        self,
        task_type: str,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task for execution"""
        task_id = f"task_{len(self.task_queue) + 1}"
        task = {
            "id": task_id,
            "type": task_type,
            "description": description,
            "priority": priority,
            "status": TaskStatus.PENDING,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        self.task_queue.append(task)
        await self._store_task_context(task)
        return task_id

    async def _store_task_context(self, task: Dict[str, Any]):
        """Store task context in FastMCP"""
        context = FastMCPContext(
            content=task["description"],
            metadata={
                "task_id": task["id"],
                "type": task["type"],
                "priority": task["priority"].value,
                "status": task["status"].value,
                **task["metadata"]
            },
            relationships=[]
        )

        await self.context_adapter.store_context(
            f"task_{task['id']}",
            context
        )

    async def _analyze_code_quality(
        self,
        code: str,
        language: str
    ) -> CodeAnalysisResult:
        """Analyze code quality using various tools"""
        # Get best practices for the language
        practices = await self.tool_manager.get_best_practices(language)

        # Use GitHub tool for code analysis
        github_result = await self.tool_manager.execute_tool(
            "GitHubTool",
            "analyze_code",
            code=code,
            language=language
        )

        # Calculate quality metrics
        quality_score = 0.0
        issues = []
        suggestions = []
        security_concerns = []

        if github_result.success:
            analysis = github_result.data
            quality_score = analysis.get("quality_score", 0.0)
            issues = analysis.get("issues", [])
            suggestions = analysis.get("suggestions", [])
            security_concerns = analysis.get("security_concerns", [])

        return CodeAnalysisResult(
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            dependencies=[],  # To be implemented
            security_concerns=security_concerns
        )

    async def _execute_coding_task(
        self,
        task: Dict[str, Any]
    ) -> TaskResult:
        """Execute a coding task"""
        try:
            # Get relevant context
            similar_contexts = await self.context_adapter.search_similar_contexts(
                await self.context_adapter.get_embeddings(task["description"]),
                limit=5,
                min_similarity=0.7
            )

            # Get best practices
            language = task["metadata"].get("language", "python")
            practices = await self.tool_manager.get_best_practices(language)

            # Create GitHub issue
            issue_result = await self.tool_manager.execute_tool(
                "GitHubTool",
                "create_issue",
                title=f"[{task['type']}] {task['description'][:50]}...",
                body=task["description"],
                labels=[task["type"], task["priority"].value]
            )

            if not issue_result.success:
                return TaskResult(
                    success=False,
                    status=TaskStatus.FAILED,
                    artifacts={},
                    error="Failed to create GitHub issue"
                )

            # Create JIRA ticket
            jira_result = await self.tool_manager.execute_tool(
                "JiraTool",
                "create_issue",
                summary=f"[{task['type']}] {task['description'][:50]}...",
                description=task["description"],
                issue_type="Task"
            )

            # Implement the solution
            # This is a placeholder for actual code implementation
            code = "def example(): pass"  # Replace with actual implementation

            # Analyze code quality
            analysis = await self._analyze_code_quality(code, language)

            # Create pull request if quality is good
            if analysis.quality_score >= 0.8:
                pr_result = await self.tool_manager.execute_tool(
                    "GitHubTool",
                    "create_pr",
                    title=f"Implement {task['type']}",
                    body=f"Closes #{issue_result.data['number']}",
                    head="feature/implementation",
                    base="main"
                )

                return TaskResult(
                    success=True,
                    status=TaskStatus.REVIEWING,
                    artifacts={
                        "github_issue": issue_result.data,
                        "jira_ticket": jira_result.data if jira_result.success else None,
                        "pull_request": pr_result.data if pr_result.success else None,
                        "code": code
                    },
                    analysis=analysis,
                    metrics={
                        "similar_contexts": len(similar_contexts),
                        "quality_score": analysis.quality_score
                    }
                )
            else:
                return TaskResult(
                    success=False,
                    status=TaskStatus.FAILED,
                    artifacts={
                        "github_issue": issue_result.data,
                        "jira_ticket": jira_result.data if jira_result.success else None,
                        "code": code
                    },
                    analysis=analysis,
                    error="Code quality below threshold"
                )

        except Exception as e:
            return TaskResult(
                success=False,
                status=TaskStatus.FAILED,
                artifacts={},
                error=str(e)
            )

    async def process_tasks(self):
        """Process tasks in the queue"""
        while True:
            if not self.task_queue:
                await asyncio.sleep(1)
                continue

            # Sort tasks by priority
            self.task_queue.sort(
                key=lambda x: TaskPriority[x["priority"].name].value,
                reverse=True
            )

            task = self.task_queue.pop(0)
            self.active_tasks[task["id"]] = task

            # Update task status
            task["status"] = TaskStatus.IN_PROGRESS
            task["updated_at"] = datetime.now()
            await self._store_task_context(task)

            # Execute task
            result = await self._execute_coding_task(task)

            # Update task status
            task["status"] = result.status
            task["updated_at"] = datetime.now()
            await self._store_task_context(task)

            # Store result in history
            self.task_history.append({
                **task,
                "result": result
            })

            # Remove from active tasks
            self.active_tasks.pop(task["id"])

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        # Check history
        for task in self.task_history:
            if task["id"] == task_id:
                return task

        return None

    async def get_task_metrics(self) -> Dict[str, Any]:
        """Get metrics about task execution"""
        total_tasks = len(self.task_history)
        successful_tasks = len([
            t for t in self.task_history
            if t["result"].success
        ])

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_quality_score": sum(
                t["result"].analysis.quality_score
                for t in self.task_history
                if t["result"].analysis
            ) / total_tasks if total_tasks > 0 else 0,
            "tasks_by_priority": {
                priority.value: len([
                    t for t in self.task_history
                    if t["priority"] == priority
                ])
                for priority in TaskPriority
            }
        }
