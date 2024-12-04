from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
import json
from datetime import datetime
from ..generation.code_generator import CodeGenerator
from ..tools.manager import ToolManager
from ..context.fastmcp_adapter import FastMCPAdapter
from ..knowledge.graph import KnowledgeGraph
from ..verification.verifier import VerificationService
from ..mcp.client import MCPClient
from prometheus_client import Counter, Gauge, Histogram

# Metrics
WORKFLOW_COUNTER = Counter(
    'workflow_total', 'Total workflows', ['type', 'status'])
WORKFLOW_DURATION = Histogram('workflow_duration_seconds', 'Workflow duration')
WORKFLOW_STEPS = Gauge(
    'workflow_steps', 'Number of workflow steps', ['workflow_id'])
WORKFLOW_SUCCESS_RATE = Gauge(
    'workflow_success_rate', 'Workflow success rate', ['type'])


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ADAPTING = "adapting"


class WorkflowType(Enum):
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"
    AUTONOMOUS_IMPROVEMENT = "autonomous_improvement"
    CONTINUOUS_LEARNING = "continuous_learning"


@dataclass
class WorkflowStep:
    """Step in a workflow"""
    name: str
    handler: Any  # Callable[..., Awaitable[Any]]
    dependencies: List[str]
    retry_count: int = 3
    timeout_seconds: int = 300
    required: bool = True
    verification_required: bool = True
    # Optional[Callable[..., Awaitable[Any]]]
    rollback_handler: Optional[Any] = None
    adaptation_rules: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    success: bool
    status: WorkflowStatus
    artifacts: Dict[str, Any]
    metrics: Dict[str, Any]
    learnings: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]
    error: Optional[str] = None


class WorkflowOrchestrator:
    """Orchestrates development workflows with autonomous capabilities"""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        code_generator: CodeGenerator,
        tool_manager: ToolManager,
        context_adapter: FastMCPAdapter,
        verifier: VerificationService,
        mcp_client: MCPClient,
        config: Optional[Dict[str, Any]] = None
    ):
        self.knowledge_graph = knowledge_graph
        self.code_generator = code_generator
        self.tool_manager = tool_manager
        self.context_adapter = context_adapter
        self.verifier = verifier
        self.mcp_client = mcp_client
        self.config = config or {}

        # Workflow tracking
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.improvement_suggestions: List[Dict[str, Any]] = []
        self.learning_points: Set[str] = set()

        # Metrics and monitoring
        self.logger = logging.getLogger(__name__)

        # Start autonomous processes
        self._start_autonomous_processes()

    def _start_autonomous_processes(self):
        """Start autonomous background processes"""
        asyncio.create_task(self._continuous_improvement_loop())
        asyncio.create_task(self._learning_aggregation_loop())
        asyncio.create_task(self._workflow_monitoring_loop())

    async def create_workflow(
        self,
        name: str,
        workflow_type: WorkflowType,
        description: str,
        steps: List[WorkflowStep],
        metadata: Optional[Dict[str, Any]] = None,
        autonomous: bool = False
    ) -> str:
        """Create a new workflow with autonomous capabilities"""
        try:
            workflow_id = f"workflow_{len(self.workflows) + 1}"

            # Create MCP context
            context_metadata = {
                "workflow_id": workflow_id,
                "name": name,
                "description": description,
                "autonomous": autonomous,
                **(metadata or {})
            }

            mcp_context = await self.mcp_client.create_context(
                model_id=self.config["model_id"],
                task_type=workflow_type.value,
                metadata=context_metadata
            )

            workflow = {
                "id": workflow_id,
                "name": name,
                "type": workflow_type,
                "description": description,
                "steps": steps,
                "status": WorkflowStatus.PENDING,
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "mcp_context": mcp_context,
                "autonomous": autonomous,
                "learnings": [],
                "improvements": [],
                "metrics": {}
            }

            # Store in knowledge graph
            await self.knowledge_graph.add_node(
                workflow_id,
                {
                    "type": "workflow",
                    "content": json.dumps(workflow),
                    "metadata": {
                        "workflow_type": workflow_type.value,
                        "autonomous": autonomous
                    }
                },
                "Workflow"
            )

            self.workflows[workflow_id] = workflow
            WORKFLOW_COUNTER.labels(
                type=workflow_type.value, status="created").inc()

            # Start autonomous monitoring if enabled
            if autonomous:
                asyncio.create_task(
                    self._monitor_autonomous_workflow(workflow_id))

            return workflow_id

        except Exception as e:
            self.logger.error(f"Failed to create workflow: {str(e)}")
            WORKFLOW_COUNTER.labels(
                type=workflow_type.value, status="failed").inc()
            raise

    async def _monitor_autonomous_workflow(self, workflow_id: str):
        """Monitor and adapt autonomous workflow"""
        try:
            workflow = self.workflows[workflow_id]

            while workflow["status"] != WorkflowStatus.COMPLETED:
                # Get workflow metrics
                metrics = await self._calculate_workflow_metrics(workflow_id)

                # Check for improvements
                improvements = await self._identify_improvements(workflow_id, metrics)

                if improvements:
                    # Apply improvements
                    await self._apply_workflow_improvements(workflow_id, improvements)

                # Update learning points
                learnings = await self._extract_workflow_learnings(workflow_id)
                workflow["learnings"].extend(learnings)

                # Sleep before next check
                await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(
                f"Failed to monitor autonomous workflow: {str(e)}")

    async def _continuous_improvement_loop(self):
        """Continuous improvement process"""
        try:
            while True:
                # Analyze all workflows
                for workflow_id, workflow in self.workflows.items():
                    if workflow["status"] != WorkflowStatus.COMPLETED:
                        continue

                    # Analyze workflow results
                    analysis = await self._analyze_workflow_results(workflow_id)

                    # Generate improvement suggestions
                    suggestions = await self._generate_improvements(analysis)

                    if suggestions:
                        # Create autonomous improvement workflow
                        await self.create_workflow(
                            name=f"Improve {workflow['name']}",
                            workflow_type=WorkflowType.AUTONOMOUS_IMPROVEMENT,
                            description=f"Autonomous improvements for {workflow['name']}",
                            steps=await self._create_improvement_steps(suggestions),
                            metadata={
                                "parent_workflow": workflow_id,
                                "suggestions": suggestions
                            },
                            autonomous=True
                        )

                await asyncio.sleep(3600)  # Check every hour

        except Exception as e:
            self.logger.error(
                f"Error in continuous improvement loop: {str(e)}")

    async def _learning_aggregation_loop(self):
        """Aggregate and apply learnings"""
        try:
            while True:
                # Collect learnings from all workflows
                all_learnings = []
                for workflow in self.workflows.values():
                    all_learnings.extend(workflow.get("learnings", []))

                if all_learnings:
                    # Analyze patterns in learnings
                    patterns = await self._analyze_learning_patterns(all_learnings)

                    # Update knowledge graph
                    for pattern in patterns:
                        await self.knowledge_graph.add_node(
                            f"learning_{len(self.learning_points) + 1}",
                            {
                                "type": "learning",
                                "content": json.dumps(pattern),
                                "metadata": {
                                    "confidence": pattern["confidence"],
                                    "source_workflows": pattern["workflows"]
                                }
                            },
                            "Learning"
                        )
                        self.learning_points.add(pattern["id"])

                await asyncio.sleep(1800)  # Aggregate every 30 minutes

        except Exception as e:
            self.logger.error(f"Error in learning aggregation loop: {str(e)}")

    async def _workflow_monitoring_loop(self):
        """Monitor active workflows"""
        try:
            while True:
                # Check all active workflows
                for workflow_id, workflow in self.active_workflows.items():
                    try:
                        # Update metrics
                        metrics = await self._calculate_workflow_metrics(workflow_id)
                        WORKFLOW_STEPS.labels(workflow_id=workflow_id).set(
                            len(workflow["steps"])
                        )

                        # Check for issues
                        issues = await self._check_workflow_health(workflow_id)
                        if issues:
                            await self._handle_workflow_issues(workflow_id, issues)

                        # Update success rate
                        success_rate = await self._calculate_success_rate(
                            workflow["type"]
                        )
                        WORKFLOW_SUCCESS_RATE.labels(
                            type=workflow["type"].value
                        ).set(success_rate)

                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring workflow {workflow_id}: {str(e)}"
                        )

                await asyncio.sleep(30)  # Check every 30 seconds

        except Exception as e:
            self.logger.error(f"Error in workflow monitoring loop: {str(e)}")

    async def _analyze_workflow_results(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze workflow results for improvements"""
        try:
            workflow = self.workflows[workflow_id]

            # Get workflow data from knowledge graph
            workflow_node = await self.knowledge_graph.get_node(workflow_id)

            # Analyze patterns
            patterns = await self.knowledge_graph.analyze_patterns(
                workflow_node["content"]
            )

            # Get related workflows
            related = await self.knowledge_graph.find_similar_nodes(
                workflow_id,
                node_type="workflow"
            )

            # Analyze success patterns
            success_patterns = await self._analyze_success_patterns(
                workflow_id,
                related
            )

            return {
                "patterns": patterns,
                "related_workflows": related,
                "success_patterns": success_patterns,
                "metrics": workflow.get("metrics", {})
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze workflow results: {str(e)}")
            return {}

    async def _handle_workflow_issues(self, workflow_id: str,
                                      issues: List[Dict[str, Any]]):
        """Handle workflow issues"""
        try:
            workflow = self.workflows[workflow_id]

            for issue in issues:
                if issue["severity"] == "critical":
                    # Pause workflow
                    workflow["status"] = WorkflowStatus.ADAPTING

                    # Create adaptation plan
                    adaptation_plan = await self._create_adaptation_plan(
                        workflow_id,
                        issue
                    )

                    # Apply adaptations
                    await self._apply_adaptations(workflow_id, adaptation_plan)

                elif issue["severity"] == "warning":
                    # Add to improvements
                    self.improvement_suggestions.append({
                        "workflow_id": workflow_id,
                        "issue": issue,
                        "suggested_improvements": await self._suggest_improvements(
                            workflow_id,
                            issue
                        )
                    })

        except Exception as e:
            self.logger.error(f"Failed to handle workflow issues: {str(e)}")
            raise
