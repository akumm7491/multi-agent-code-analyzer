from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
from .task_orchestrator import TaskOrchestrator, TaskStatus, TaskPriority
from ..generation.code_generator import CodeGenerator
from ..tools.manager import ToolManager
from ..context.fastmcp_adapter import FastMCPAdapter


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowType(Enum):
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"


@dataclass
class WorkflowStep:
    """Step in a workflow"""
    name: str
    handler: Callable[..., Awaitable[Any]]
    dependencies: List[str]
    retry_count: int = 3
    timeout_seconds: int = 300
    required: bool = True


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    success: bool
    status: WorkflowStatus
    artifacts: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None


class WorkflowOrchestrator:
    """Orchestrates development workflows"""

    def __init__(
        self,
        task_orchestrator: TaskOrchestrator,
        code_generator: CodeGenerator,
        tool_manager: ToolManager,
        context_adapter: FastMCPAdapter,
        config: Optional[Dict[str, Any]] = None
    ):
        self.task_orchestrator = task_orchestrator
        self.code_generator = code_generator
        self.tool_manager = tool_manager
        self.context_adapter = context_adapter
        self.config = config or {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []

    async def create_workflow(
        self,
        name: str,
        workflow_type: WorkflowType,
        description: str,
        steps: List[WorkflowStep],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow"""
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        workflow = {
            "id": workflow_id,
            "name": name,
            "type": workflow_type,
            "description": description,
            "steps": steps,
            "status": WorkflowStatus.PENDING,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        self.workflows[workflow_id] = workflow
        await self._store_workflow_context(workflow)
        return workflow_id

    async def _store_workflow_context(self, workflow: Dict[str, Any]):
        """Store workflow context in FastMCP"""
        context = FastMCPContext(
            content=workflow["description"],
            metadata={
                "workflow_id": workflow["id"],
                "name": workflow["name"],
                "type": workflow["type"].value,
                "status": workflow["status"].value,
                **workflow["metadata"]
            },
            relationships=[]
        )

        await self.context_adapter.store_context(
            f"workflow_{workflow['id']}",
            context
        )

    async def _execute_step(
        self,
        step: WorkflowStep,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a workflow step with retry logic"""
        for attempt in range(step.retry_count):
            try:
                result = await asyncio.wait_for(
                    step.handler(workflow, context),
                    timeout=step.timeout_seconds
                )
                return {
                    "success": True,
                    "data": result,
                    "attempt": attempt + 1
                }
            except asyncio.TimeoutError:
                if attempt == step.retry_count - 1:
                    raise
            except Exception as e:
                if attempt == step.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _create_feature_branch(self, workflow: Dict[str, Any]) -> bool:
        """Create a feature branch for the workflow"""
        branch_name = f"feature/{workflow['name'].lower().replace(' ', '-')}"
        result = await self.tool_manager.execute_tool(
            "GitHubTool",
            "create_branch",
            name=branch_name,
            base="main"
        )
        return result.success

    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        self.active_workflows[workflow_id] = workflow

        try:
            # Create feature branch
            if workflow["type"] in (WorkflowType.FEATURE, WorkflowType.ENHANCEMENT):
                await self._create_feature_branch(workflow)

            # Execute steps
            context = {}
            results = {}

            for step in workflow["steps"]:
                # Check dependencies
                if not all(dep in results for dep in step.dependencies):
                    if step.required:
                        raise ValueError(
                            f"Dependencies not met for step {step.name}")
                    continue

                # Execute step
                result = await self._execute_step(step, workflow, context)
                results[step.name] = result
                context.update(result.get("data", {}))

                # Update workflow status
                workflow["status"] = WorkflowStatus.IN_PROGRESS
                workflow["updated_at"] = datetime.now()
                await self._store_workflow_context(workflow)

            # Create pull request
            if workflow["type"] != WorkflowType.DOCUMENTATION:
                pr_result = await self.tool_manager.execute_tool(
                    "GitHubTool",
                    "create_pr",
                    title=f"{workflow['type'].value}: {workflow['name']}",
                    body=workflow["description"],
                    head=f"feature/{workflow['name'].lower().replace(' ', '-')}",
                    base="main"
                )

                if pr_result.success:
                    workflow["status"] = WorkflowStatus.REVIEWING
                else:
                    workflow["status"] = WorkflowStatus.FAILED

            else:
                workflow["status"] = WorkflowStatus.COMPLETED

            # Calculate metrics
            metrics = {
                "total_steps": len(workflow["steps"]),
                "completed_steps": len(results),
                "success_rate": sum(
                    1 for r in results.values() if r["success"]
                ) / len(results) if results else 0,
                "total_attempts": sum(
                    r["attempt"] for r in results.values()
                ),
                "execution_time": (
                    datetime.now() - workflow["created_at"]
                ).total_seconds()
            }

            return WorkflowResult(
                success=all(r["success"] for r in results.values()),
                status=workflow["status"],
                artifacts=results,
                metrics=metrics
            )

        except Exception as e:
            workflow["status"] = WorkflowStatus.FAILED
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                artifacts={},
                metrics={},
                error=str(e)
            )

        finally:
            workflow["updated_at"] = datetime.now()
            await self._store_workflow_context(workflow)
            self.workflow_history.append(workflow)
            self.active_workflows.pop(workflow_id)

    async def get_workflow_status(
        self,
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        # Check active workflows
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Check history
        for workflow in self.workflow_history:
            if workflow["id"] == workflow_id:
                return workflow

        return None

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get metrics about workflow execution"""
        total_workflows = len(self.workflow_history)
        successful_workflows = len([
            w for w in self.workflow_history
            if w["status"] == WorkflowStatus.COMPLETED
        ])

        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": successful_workflows / total_workflows if total_workflows > 0 else 0,
            "workflows_by_type": {
                wtype.value: len([
                    w for w in self.workflow_history
                    if w["type"] == wtype
                ])
                for wtype in WorkflowType
            },
            "average_execution_time": sum(
                (w["updated_at"] - w["created_at"]).total_seconds()
                for w in self.workflow_history
            ) / total_workflows if total_workflows > 0 else 0
        }

    def create_feature_workflow(
        self,
        name: str,
        description: str,
        language: str,
        framework: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a feature development workflow"""
        steps = [
            WorkflowStep(
                name="requirements_analysis",
                handler=self._analyze_requirements,
                dependencies=[],
                required=True
            ),
            WorkflowStep(
                name="code_generation",
                handler=self._generate_code,
                dependencies=["requirements_analysis"],
                required=True
            ),
            WorkflowStep(
                name="test_generation",
                handler=self._generate_tests,
                dependencies=["code_generation"],
                required=True
            ),
            WorkflowStep(
                name="documentation",
                handler=self._generate_documentation,
                dependencies=["code_generation"],
                required=True
            ),
            WorkflowStep(
                name="code_review",
                handler=self._review_code,
                dependencies=["code_generation", "test_generation"],
                required=True
            )
        ]

        return self.create_workflow(
            name=name,
            workflow_type=WorkflowType.FEATURE,
            description=description,
            steps=steps,
            metadata={
                "language": language,
                "framework": framework,
                **(metadata or {})
            }
        )

    async def _analyze_requirements(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze requirements for a feature"""
        similar_contexts = await self.context_adapter.search_similar_contexts(
            await self.context_adapter.get_embeddings(workflow["description"]),
            limit=5,
            min_similarity=0.7
        )

        return {
            "requirements": {
                "functional": [
                    # Extract from description
                ],
                "non_functional": [
                    # Extract from best practices
                ],
                "similar_features": [
                    ctx["content"] for ctx in similar_contexts
                ]
            }
        }

    async def _generate_code(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code for the feature"""
        requirements = context.get("requirements", {})

        code = await self.code_generator.generate_code(
            description=workflow["description"],
            language=workflow["metadata"]["language"],
            framework=workflow["metadata"].get("framework"),
            metadata={
                "workflow_id": workflow["id"],
                "requirements": requirements
            }
        )

        return {
            "code": code.code,
            "tests": code.tests,
            "documentation": code.documentation,
            "quality_score": code.quality_score
        }

    async def _generate_tests(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate tests for the feature"""
        code = context.get("code", {})

        return {
            "unit_tests": code.get("tests", ""),
            "integration_tests": "",  # TODO: Implement
            "coverage": code.get("coverage", 0.0)
        }

    async def _generate_documentation(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate documentation for the feature"""
        code = context.get("code", {})

        return {
            "api_docs": code.get("documentation", ""),
            "usage_examples": "",  # TODO: Implement
            "architecture_diagram": ""  # TODO: Implement
        }

    async def _review_code(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review generated code"""
        code = context.get("code", {})

        validation_results = await self.code_generator.validate_code(
            code=code.get("code", ""),
            language=workflow["metadata"]["language"],
            requirements=context.get("requirements", {}).get("functional", [])
        )

        return {
            "validation_results": validation_results,
            "review_comments": [],  # TODO: Implement
            "approved": validation_results.get("quality_score", 0) >= 0.8
        }
