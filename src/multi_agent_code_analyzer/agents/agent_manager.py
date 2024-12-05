from typing import Dict, Any, Optional, List
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram

from .base import BaseAgent
from .code_analyzer import CodeAnalyzerAgent
from .developer import DeveloperAgent
from ..core.mcp_client import MCPClient
from ..monitoring.metrics import MetricsCollector
from ..orchestration.workflow import WorkflowOrchestrator

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self.mcp_client = mcp_client or MCPClient()
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_orchestrator = WorkflowOrchestrator(self.mcp_client)
        self.tracer = trace.get_tracer(__name__)
        self.metrics = MetricsCollector("agent_manager")

        # Initialize metrics
        self.active_agents = Gauge(
            'active_agents',
            'Number of active agents',
            ['agent_type']
        )

        self.agent_creation_counter = Counter(
            'agent_creations_total',
            'Total number of agent creations',
            ['agent_type']
        )

        self.agent_task_duration = Histogram(
            'agent_task_duration_seconds',
            'Duration of agent tasks',
            ['agent_type', 'task_type']
        )

    async def initialize(self):
        """Initialize agent manager"""
        await self.mcp_client.connect()
        await self.workflow_orchestrator.initialize()

    async def create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new agent"""
        with self.tracer.start_as_current_span("create_agent") as span:
            span.set_attribute("agent.type", agent_type)

            agent_id = str(uuid4())
            try:
                if agent_type == "code_analyzer":
                    agent = CodeAnalyzerAgent(
                        agent_id=agent_id,
                        mcp_client=self.mcp_client,
                        config=config
                    )
                elif agent_type == "developer":
                    agent = DeveloperAgent(
                        agent_id=agent_id,
                        mcp_client=self.mcp_client,
                        config=config
                    )
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                await agent.initialize()
                self.agents[agent_id] = agent

                # Update metrics
                self.agent_creation_counter.labels(agent_type=agent_type).inc()
                self.active_agents.labels(agent_type=agent_type).inc()

                span.set_attribute("agent.id", agent_id)
                return agent_id

            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                span.record_exception(e)
                raise

    async def execute_task(
        self,
        agent_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task on an agent"""
        with self.tracer.start_as_current_span("execute_task") as span:
            span.set_attribute("agent.id", agent_id)
            span.set_attribute("task.type", task.get("type"))

            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")

            agent = self.agents[agent_id]
            try:
                # Create workflow for task execution
                workflow_id = await self.workflow_orchestrator.create_workflow(
                    workflow_type="agent_task",
                    input_data={
                        "agent_id": agent_id,
                        "task": task
                    }
                )

                # Start workflow execution
                await self.workflow_orchestrator.start_workflow(workflow_id)

                # Monitor workflow status
                while True:
                    status = self.workflow_orchestrator.get_workflow_status(
                        workflow_id)
                    if status["status"] in ["completed", "failed"]:
                        break
                    await asyncio.sleep(1)

                if status["status"] == "failed":
                    raise RuntimeError(
                        f"Workflow failed: {status.get('error')}")

                return status["result"]

            except Exception as e:
                logger.error(f"Failed to execute task: {e}")
                span.record_exception(e)
                raise

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        return {
            "id": agent_id,
            "type": agent.__class__.__name__,
            "status": "active",
            "current_task": agent.current_task,
            "last_updated": datetime.utcnow().isoformat()
        }

    async def cleanup(self):
        """Cleanup resources"""
        tasks = []
        for agent in self.agents.values():
            tasks.append(agent.cleanup())

        await asyncio.gather(*tasks)
        await self.mcp_client.close()
        await self.workflow_orchestrator.cleanup()
