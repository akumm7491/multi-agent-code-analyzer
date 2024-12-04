from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
import docker
import numpy as np
import json
from datetime import datetime, timedelta
from prometheus_client import Counter, Gauge, Histogram
from .base import BaseAgent
from .code_analyzer import CodeAnalyzerAgent
from .developer import DeveloperAgent
from ..knowledge.graph import KnowledgeGraph
from ..mcp.client import MCPClient
from ..verification.verifier import VerificationService

# Metrics
AGENT_COUNT = Gauge('agent_count', 'Number of active agents', ['type'])
AGENT_SCALING = Counter('agent_scaling_operations',
                        'Agent scaling operations', ['type', 'operation'])
AGENT_LOAD = Gauge('agent_load', 'Agent load percentage', ['type'])
TASK_QUEUE_SIZE = Gauge('task_queue_size', 'Size of task queue', ['type'])
AGENT_MEMORY = Gauge('agent_memory_usage', 'Agent memory usage', ['type'])
AGENT_CPU = Gauge('agent_cpu_usage', 'Agent CPU usage', ['type'])
AGENT_EFFICIENCY = Gauge(
    'agent_efficiency', 'Agent efficiency score', ['type'])
AGENT_RESPONSE_TIME = Histogram(
    'agent_response_time', 'Agent response time', ['type'])
SCALING_DECISIONS = Counter(
    'scaling_decisions', 'Scaling decisions', ['type', 'reason'])

# Additional Metrics for Knowledge Sharing
KNOWLEDGE_UPDATES = Counter(
    'knowledge_updates', 'Knowledge update events', ['type', 'source'])
INSIGHT_PROPAGATION = Histogram(
    'insight_propagation_time', 'Time to propagate insights')
KNOWLEDGE_REUSE = Counter(
    'knowledge_reuse', 'Knowledge reuse events', ['type'])
AGENT_COLLABORATION = Counter('agent_collaboration', 'Agent collaboration events', [
                              'initiator', 'contributor'])


class ScalingStrategy:
    """Base class for scaling strategies"""

    def __init__(self, config: Dict):
        self.config = config

    async def should_scale_up(self, metrics: Dict) -> Tuple[bool, str]:
        raise NotImplementedError

    async def should_scale_down(self, metrics: Dict) -> Tuple[bool, str]:
        raise NotImplementedError


class LoadBasedScaling(ScalingStrategy):
    """Scale based on CPU and memory load"""

    async def should_scale_up(self, metrics: Dict) -> Tuple[bool, str]:
        if metrics["cpu_usage"] > self.config["cpu_threshold"]:
            return True, "CPU usage above threshold"
        if metrics["memory_usage"] > self.config["memory_threshold"]:
            return True, "Memory usage above threshold"
        return False, ""

    async def should_scale_down(self, metrics: Dict) -> Tuple[bool, str]:
        if (metrics["cpu_usage"] < self.config["cpu_threshold"] / 2 and
                metrics["memory_usage"] < self.config["memory_threshold"] / 2):
            return True, "Resource usage below thresholds"
        return False, ""


class QueueBasedScaling(ScalingStrategy):
    """Scale based on task queue size"""

    async def should_scale_up(self, metrics: Dict) -> Tuple[bool, str]:
        if metrics["queue_size"] > self.config["queue_threshold"]:
            return True, "Queue size above threshold"
        return False, ""

    async def should_scale_down(self, metrics: Dict) -> Tuple[bool, str]:
        if metrics["queue_size"] == 0:
            return True, "Empty task queue"
        return False, ""


class PredictiveScaling(ScalingStrategy):
    """Scale based on predicted load"""

    async def should_scale_up(self, metrics: Dict) -> Tuple[bool, str]:
        prediction = self._predict_load(metrics["historical_load"])
        if prediction > self.config["load_threshold"]:
            return True, "Predicted high load"
        return False, ""

    async def should_scale_down(self, metrics: Dict) -> Tuple[bool, str]:
        prediction = self._predict_load(metrics["historical_load"])
        if prediction < self.config["load_threshold"] / 2:
            return True, "Predicted low load"
        return False, ""

    def _predict_load(self, historical_load: List[float]) -> float:
        # Simple moving average prediction
        return np.mean(historical_load[-10:]) if len(historical_load) >= 10 else 0


class KnowledgeUpdate:
    """Represents a knowledge update from an agent"""

    def __init__(
        self,
        agent_id: str,
        update_type: str,
        content: Dict,
        confidence: float,
        timestamp: datetime
    ):
        self.agent_id = agent_id
        self.update_type = update_type
        self.content = content
        self.confidence = confidence
        self.timestamp = timestamp
        self.validations = []
        self.applications = []


class CollaborationSession:
    """Represents an active collaboration between agents"""

    def __init__(
        self,
        session_id: str,
        initiator: str,
        topic: str,
        context: Dict
    ):
        self.session_id = session_id
        self.initiator = initiator
        self.topic = topic
        self.context = context
        self.participants = {initiator}
        self.contributions = []
        self.start_time = datetime.now()
        self.status = "active"


class AgentManager:
    """Manages dynamic agent scaling and resource allocation with knowledge sharing"""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        mcp_client: MCPClient,
        verifier: VerificationService,
        config: Optional[Dict] = None
    ):
        self.knowledge_graph = knowledge_graph
        self.mcp_client = mcp_client
        self.verifier = verifier
        self.config = config or {}

        # Docker client for container management
        self.docker = docker.from_env()

        # Agent tracking
        self.active_agents: Dict[str, Dict] = {}
        self.agent_types: Set[str] = {
            "code_analyzer", "developer", "integration"}
        self.agent_loads: Dict[str, Dict] = {}
        self.historical_loads: Dict[str, List[float]] = {
            t: [] for t in self.agent_types}
        self.agent_metrics: Dict[str, Dict] = {}

        # Scaling configuration
        self.min_agents = int(self.config.get("MIN_AGENTS_PER_TYPE", 1))
        self.max_agents = int(self.config.get("MAX_AGENTS_PER_TYPE", 5))
        self.scaling_cooldown = int(
            self.config.get("SCALING_COOLDOWN_SECONDS", 300))
        self.last_scaling: Dict[str, datetime] = {}

        # Scaling strategies
        self.scaling_strategies = [
            LoadBasedScaling(self.config),
            QueueBasedScaling(self.config),
            PredictiveScaling(self.config)
        ]

        self.logger = logging.getLogger(__name__)

        # Start monitoring
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._monitor_resources())
        asyncio.create_task(self._monitor_task_queue())
        asyncio.create_task(self._optimize_resources())

        # Knowledge sharing components
        self.knowledge_updates: Dict[str, KnowledgeUpdate] = {}
        self.active_collaborations: Dict[str, CollaborationSession] = {}
        self.agent_specialties: Dict[str, Set[str]] = {}
        self.knowledge_subscriptions: Dict[str, Set[str]] = {}

        # Start knowledge sharing processes
        asyncio.create_task(self._monitor_knowledge_updates())
        asyncio.create_task(self._facilitate_collaborations())
        asyncio.create_task(self._synchronize_knowledge())

    async def _optimize_resources(self):
        """Continuously optimize resource allocation"""
        try:
            while True:
                for agent_type in self.agent_types:
                    try:
                        # Get agents of this type
                        type_agents = [
                            (aid, a) for aid, a in self.active_agents.items()
                            if a["type"] == agent_type
                        ]

                        if not type_agents:
                            continue

                        # Calculate efficiency scores
                        efficiencies = await self._calculate_efficiencies(type_agents)

                        # Update metrics
                        avg_efficiency = np.mean(list(efficiencies.values()))
                        AGENT_EFFICIENCY.labels(
                            type=agent_type).set(avg_efficiency)

                        # Optimize placement
                        await self._optimize_agent_placement(
                            agent_type,
                            type_agents,
                            efficiencies
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error optimizing resources for {agent_type}: {str(e)}"
                        )

                await asyncio.sleep(300)  # Run optimization every 5 minutes

        except Exception as e:
            self.logger.error(f"Error in resource optimization loop: {str(e)}")

    async def _calculate_efficiencies(
        self,
        agents: List[Tuple[str, Dict]]
    ) -> Dict[str, float]:
        """Calculate efficiency scores for agents"""
        efficiencies = {}

        for agent_id, agent in agents:
            try:
                metrics = self.agent_metrics.get(agent_id, {})

                # Calculate efficiency score based on multiple factors
                cpu_efficiency = 1 - (metrics.get("cpu_usage", 0) / 100)
                memory_efficiency = 1 - (metrics.get("memory_usage", 0) / 100)
                response_time = metrics.get("avg_response_time", 1)
                throughput = metrics.get("tasks_completed", 0)

                # Weighted scoring
                efficiency_score = (
                    0.3 * cpu_efficiency +
                    0.3 * memory_efficiency +
                    0.2 * (1 / response_time if response_time > 0 else 0) +
                    0.2 * (min(throughput / 100, 1))
                )

                efficiencies[agent_id] = efficiency_score

            except Exception as e:
                self.logger.error(f"Error calculating efficiency: {str(e)}")
                efficiencies[agent_id] = 0

        return efficiencies

    async def _optimize_agent_placement(
        self,
        agent_type: str,
        agents: List[Tuple[str, Dict]],
        efficiencies: Dict[str, float]
    ):
        """Optimize agent placement based on efficiency scores"""
        try:
            # Sort agents by efficiency
            sorted_agents = sorted(
                agents,
                key=lambda x: efficiencies.get(x[0], 0),
                reverse=True
            )

            # Keep track of optimizations
            optimizations = []

            for i, (agent_id, agent) in enumerate(sorted_agents):
                if efficiencies.get(agent_id, 0) < 0.5:  # Inefficient agent
                    # Try to optimize
                    optimization = await self._optimize_agent(
                        agent_id,
                        agent,
                        efficiencies[agent_id]
                    )
                    if optimization:
                        optimizations.append(optimization)

            # Apply optimizations
            for opt in optimizations:
                await self._apply_optimization(opt)

        except Exception as e:
            self.logger.error(f"Error optimizing agent placement: {str(e)}")

    async def _optimize_agent(
        self,
        agent_id: str,
        agent: Dict,
        efficiency: float
    ) -> Optional[Dict]:
        """Generate optimization plan for an agent"""
        try:
            container = agent["container"]
            current_resources = container.attrs["HostConfig"]

            if efficiency < 0.3:  # Very inefficient
                # Plan for replacement
                return {
                    "type": "replace",
                    "agent_id": agent_id,
                    "reason": "Very low efficiency"
                }
            elif efficiency < 0.5:  # Somewhat inefficient
                # Plan for resource adjustment
                return {
                    "type": "adjust_resources",
                    "agent_id": agent_id,
                    "new_resources": self._calculate_optimal_resources(agent)
                }

            return None

        except Exception as e:
            self.logger.error(f"Error optimizing agent: {str(e)}")
            return None

    async def _apply_optimization(self, optimization: Dict):
        """Apply an optimization plan"""
        try:
            if optimization["type"] == "replace":
                # Stop old agent
                await self._stop_agent(optimization["agent_id"])
                # Start new agent
                agent_type = self.active_agents[optimization["agent_id"]]["type"]
                await self._start_agent(agent_type)

            elif optimization["type"] == "adjust_resources":
                container = self.active_agents[optimization["agent_id"]]["container"]
                # Update container resources
                container.update(**optimization["new_resources"])

        except Exception as e:
            self.logger.error(f"Error applying optimization: {str(e)}")

    def _calculate_optimal_resources(self, agent: Dict) -> Dict:
        """Calculate optimal resource allocation"""
        try:
            metrics = self.agent_metrics.get(agent["id"], {})

            # Calculate based on actual usage patterns
            cpu_usage = metrics.get("cpu_usage", 0)
            memory_usage = metrics.get("memory_usage", 0)

            # Add headroom
            optimal_cpu = min(max(cpu_usage * 1.2, 0.1), 1.0)
            optimal_memory = min(
                max(memory_usage * 1.2, 256 * 1024 * 1024), 1024 * 1024 * 1024)

            return {
                "cpu_quota": int(optimal_cpu * 100000),
                "memory": int(optimal_memory)
            }

        except Exception as e:
            self.logger.error(f"Error calculating optimal resources: {str(e)}")
            return {}

    async def _check_scaling_needs(self, agent_type: str):
        """Check if scaling is needed using multiple strategies"""
        try:
            # Skip if in cooldown
            if self._is_in_cooldown(agent_type):
                return

            # Gather metrics
            metrics = await self._gather_scaling_metrics(agent_type)

            # Check each strategy
            for strategy in self.scaling_strategies:
                # Check scale up
                should_scale_up, reason = await strategy.should_scale_up(metrics)
                if should_scale_up:
                    await self._scale_up(agent_type)
                    SCALING_DECISIONS.labels(
                        type=agent_type,
                        reason=f"scale_up_{reason}"
                    ).inc()
                    self.last_scaling[agent_type] = datetime.now()
                    return

                # Check scale down
                should_scale_down, reason = await strategy.should_scale_down(metrics)
                if should_scale_down:
                    await self._check_scale_down(agent_type)
                    SCALING_DECISIONS.labels(
                        type=agent_type,
                        reason=f"scale_down_{reason}"
                    ).inc()
                    self.last_scaling[agent_type] = datetime.now()
                    return

        except Exception as e:
            self.logger.error(f"Error checking scaling needs: {str(e)}")

    def _is_in_cooldown(self, agent_type: str) -> bool:
        """Check if agent type is in scaling cooldown"""
        last_scale = self.last_scaling.get(agent_type)
        if not last_scale:
            return False
        return (datetime.now() - last_scale).total_seconds() < self.scaling_cooldown

    async def _gather_scaling_metrics(self, agent_type: str) -> Dict:
        """Gather metrics for scaling decisions"""
        try:
            type_agents = [
                (aid, a) for aid, a in self.active_agents.items()
                if a["type"] == agent_type
            ]

            # Calculate averages
            cpu_loads = [
                self.agent_loads[aid]["cpu"]
                for aid, _ in type_agents
                if aid in self.agent_loads
            ]
            memory_loads = [
                self.agent_loads[aid]["memory"]
                for aid, _ in type_agents
                if aid in self.agent_loads
            ]

            queue_size = await self._get_queue_size(agent_type)

            return {
                "cpu_usage": np.mean(cpu_loads) if cpu_loads else 0,
                "memory_usage": np.mean(memory_loads) if memory_loads else 0,
                "queue_size": queue_size,
                "historical_load": self.historical_loads[agent_type],
                "agent_count": len(type_agents)
            }

        except Exception as e:
            self.logger.error(f"Error gathering scaling metrics: {str(e)}")
            return {}

    async def start(self):
        """Start the agent manager"""
        try:
            # Initialize minimum number of agents
            for agent_type in self.agent_types:
                await self._ensure_minimum_agents(agent_type)

            # Start monitoring loops
            await asyncio.gather(
                self._monitor_agents(),
                self._monitor_resources(),
                self._monitor_task_queue()
            )

        except Exception as e:
            self.logger.error(f"Failed to start agent manager: {str(e)}")
            raise

    async def _ensure_minimum_agents(self, agent_type: str):
        """Ensure minimum number of agents are running"""
        try:
            current_count = len([a for a in self.active_agents.values()
                                 if a["type"] == agent_type])

            if current_count < self.min_agents:
                agents_to_start = self.min_agents - current_count
                for _ in range(agents_to_start):
                    await self._start_agent(agent_type)

        except Exception as e:
            self.logger.error(f"Failed to ensure minimum agents: {str(e)}")

    async def _start_agent(self, agent_type: str) -> str:
        """Start a new agent container"""
        try:
            # Create container
            container = self.docker.containers.run(
                image="multi-agent-code-analyzer",
                environment={
                    "SERVICE_TYPE": "dynamic_agent",
                    "AGENT_TYPE": agent_type,
                    "MCP_HOST": self.config["MCP_HOST"],
                    "MCP_PORT": self.config["MCP_PORT"],
                    "NEO4J_URI": self.config["NEO4J_URI"]
                },
                network="agent_network",
                detach=True,
                remove=True,
                restart_policy={"Name": "on-failure", "MaximumRetryCount": 3}
            )

            agent_id = container.id[:12]
            self.active_agents[agent_id] = {
                "type": agent_type,
                "container": container,
                "started_at": asyncio.get_event_loop().time()
            }

            AGENT_COUNT.labels(type=agent_type).inc()
            AGENT_SCALING.labels(type=agent_type, operation="start").inc()

            return agent_id

        except Exception as e:
            self.logger.error(f"Failed to start agent: {str(e)}")
            raise

    async def _stop_agent(self, agent_id: str):
        """Stop an agent container"""
        try:
            agent = self.active_agents.get(agent_id)
            if agent:
                agent_type = agent["type"]
                container = agent["container"]

                # Stop container
                container.stop(timeout=10)

                # Update tracking
                del self.active_agents[agent_id]
                AGENT_COUNT.labels(type=agent_type).dec()
                AGENT_SCALING.labels(type=agent_type, operation="stop").inc()

        except Exception as e:
            self.logger.error(f"Failed to stop agent: {str(e)}")

    async def _monitor_agents(self):
        """Monitor agent health and status"""
        try:
            while True:
                for agent_id, agent in list(self.active_agents.items()):
                    try:
                        container = agent["container"]

                        # Check container status
                        container.reload()
                        if container.status != "running":
                            await self._handle_agent_failure(agent_id)

                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring agent {agent_id}: {str(e)}")
                        await self._handle_agent_failure(agent_id)

                await asyncio.sleep(30)

        except Exception as e:
            self.logger.error(f"Error in agent monitoring loop: {str(e)}")

    async def _monitor_resources(self):
        """Monitor agent resource usage"""
        try:
            while True:
                for agent_id, agent in self.active_agents.items():
                    try:
                        container = agent["container"]
                        agent_type = agent["type"]

                        # Get stats
                        stats = container.stats(stream=False)

                        # Calculate CPU and memory usage
                        cpu_usage = self._calculate_cpu_usage(stats)
                        memory_usage = self._calculate_memory_usage(stats)

                        # Update metrics
                        AGENT_CPU.labels(type=agent_type).set(cpu_usage)
                        AGENT_MEMORY.labels(type=agent_type).set(memory_usage)

                        # Store for scaling decisions
                        self.agent_loads[agent_id] = {
                            "cpu": cpu_usage,
                            "memory": memory_usage
                        }

                        # Check if scaling is needed
                        await self._check_scaling_needs(agent_type)

                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring resources for agent {agent_id}: {str(e)}"
                        )

                await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(f"Error in resource monitoring loop: {str(e)}")

    async def _monitor_task_queue(self):
        """Monitor task queue size"""
        try:
            while True:
                for agent_type in self.agent_types:
                    try:
                        # Get queue size from MCP
                        queue_size = await self._get_queue_size(agent_type)
                        TASK_QUEUE_SIZE.labels(type=agent_type).set(queue_size)

                        # Check if scaling is needed based on queue size
                        if queue_size > self.queue_threshold:
                            await self._scale_up(agent_type)
                        elif queue_size == 0:
                            await self._check_scale_down(agent_type)

                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring queue for {agent_type}: {str(e)}"
                        )

                await asyncio.sleep(30)

        except Exception as e:
            self.logger.error(f"Error in task queue monitoring loop: {str(e)}")

    async def _scale_up(self, agent_type: str):
        """Scale up agents of a specific type"""
        try:
            current_count = len([a for a in self.active_agents.values()
                                 if a["type"] == agent_type])

            if current_count < self.max_agents:
                await self._start_agent(agent_type)
                self.logger.info(f"Scaled up {agent_type} agents")

        except Exception as e:
            self.logger.error(f"Error scaling up: {str(e)}")

    async def _check_scale_down(self, agent_type: str):
        """Check if we can scale down agents"""
        try:
            type_agents = [
                (aid, a) for aid, a in self.active_agents.items()
                if a["type"] == agent_type
            ]

            current_count = len(type_agents)
            if current_count > self.min_agents:
                # Find least loaded agent
                least_loaded = min(
                    type_agents,
                    key=lambda x: sum(self.agent_loads.get(x[0], {}).values())
                )

                await self._stop_agent(least_loaded[0])
                self.logger.info(f"Scaled down {agent_type} agents")

        except Exception as e:
            self.logger.error(f"Error checking scale down: {str(e)}")

    async def _handle_agent_failure(self, agent_id: str):
        """Handle agent failure"""
        try:
            agent = self.active_agents[agent_id]
            agent_type = agent["type"]

            # Stop failed agent
            await self._stop_agent(agent_id)

            # Start replacement if needed
            current_count = len([a for a in self.active_agents.values()
                                 if a["type"] == agent_type])

            if current_count < self.min_agents:
                await self._start_agent(agent_type)

        except Exception as e:
            self.logger.error(f"Error handling agent failure: {str(e)}")

    async def _get_queue_size(self, agent_type: str) -> int:
        """Get task queue size from MCP"""
        try:
            return await self.mcp_client.get_queue_size(agent_type)
        except Exception as e:
            self.logger.error(f"Error getting queue size: {str(e)}")
            return 0

    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from stats"""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                stats["precpu_stats"]["system_cpu_usage"]

            if system_delta > 0:
                return (cpu_delta / system_delta) * 100
            return 0

        except Exception:
            return 0

    def _calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage percentage from stats"""
        try:
            used_memory = stats["memory_stats"]["usage"]
            max_memory = stats["memory_stats"]["limit"]

            if max_memory > 0:
                return (used_memory / max_memory) * 100
            return 0

        except Exception:
            return 0

    async def register_knowledge_update(
        self,
        agent_id: str,
        update_type: str,
        content: Dict,
        confidence: float
    ) -> str:
        """Register a new knowledge update from an agent"""
        try:
            update_id = f"update_{len(self.knowledge_updates) + 1}"
            update = KnowledgeUpdate(
                agent_id=agent_id,
                update_type=update_type,
                content=content,
                confidence=confidence,
                timestamp=datetime.now()
            )

            # Store update
            self.knowledge_updates[update_id] = update

            # Store in MCP for immediate availability
            await self.mcp_client.store_context(
                context_id=update_id,
                content={
                    "type": "knowledge_update",
                    "agent_id": agent_id,
                    "update_type": update_type,
                    "content": content,
                    "confidence": confidence,
                    "timestamp": update.timestamp.isoformat()
                }
            )

            # Store in knowledge graph
            await self.knowledge_graph.add_node(
                update_id,
                {
                    "type": "knowledge_update",
                    "content": json.dumps(content),
                    "metadata": {
                        "agent_id": agent_id,
                        "update_type": update_type,
                        "confidence": confidence
                    }
                },
                "KnowledgeUpdate"
            )

            # Notify interested agents
            await self._notify_interested_agents(update_id, update)

            KNOWLEDGE_UPDATES.labels(
                type=update_type,
                source=agent_id
            ).inc()

            return update_id

        except Exception as e:
            self.logger.error(f"Error registering knowledge update: {str(e)}")
            raise

    async def subscribe_to_knowledge(
        self,
        agent_id: str,
        topics: List[str]
    ):
        """Subscribe an agent to knowledge update topics"""
        try:
            for topic in topics:
                if topic not in self.knowledge_subscriptions:
                    self.knowledge_subscriptions[topic] = set()
                self.knowledge_subscriptions[topic].add(agent_id)

        except Exception as e:
            self.logger.error(f"Error subscribing to knowledge: {str(e)}")

    async def start_collaboration(
        self,
        initiator: str,
        topic: str,
        context: Dict
    ) -> str:
        """Start a new collaboration session"""
        try:
            session_id = f"collab_{len(self.active_collaborations) + 1}"
            session = CollaborationSession(
                session_id=session_id,
                initiator=initiator,
                topic=topic,
                context=context
            )

            # Store session
            self.active_collaborations[session_id] = session

            # Store in MCP
            await self.mcp_client.store_context(
                context_id=session_id,
                content={
                    "type": "collaboration_session",
                    "initiator": initiator,
                    "topic": topic,
                    "context": context,
                    "status": "active"
                }
            )

            # Find potential collaborators
            await self._invite_collaborators(session)

            return session_id

        except Exception as e:
            self.logger.error(f"Error starting collaboration: {str(e)}")
            raise

    async def contribute_to_collaboration(
        self,
        session_id: str,
        agent_id: str,
        contribution: Dict
    ):
        """Contribute to an active collaboration session"""
        try:
            session = self.active_collaborations.get(session_id)
            if not session:
                raise ValueError(
                    f"Collaboration session {session_id} not found")

            # Add contribution
            session.contributions.append({
                "agent_id": agent_id,
                "content": contribution,
                "timestamp": datetime.now()
            })

            # Update MCP
            await self.mcp_client.update_context(
                context_id=session_id,
                updates={
                    "contributions": session.contributions
                }
            )

            AGENT_COLLABORATION.labels(
                initiator=session.initiator,
                contributor=agent_id
            ).inc()

        except Exception as e:
            self.logger.error(f"Error contributing to collaboration: {str(e)}")

    async def _monitor_knowledge_updates(self):
        """Monitor and validate knowledge updates"""
        try:
            while True:
                for update_id, update in list(self.knowledge_updates.items()):
                    try:
                        # Validate update
                        validation = await self._validate_knowledge_update(update)

                        if validation["is_valid"]:
                            # Propagate to knowledge graph
                            await self._propagate_validated_knowledge(
                                update_id,
                                update,
                                validation
                            )
                        else:
                            # Handle invalid update
                            await self._handle_invalid_knowledge(
                                update_id,
                                update,
                                validation
                            )

                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring update {update_id}: {str(e)}"
                        )

                await asyncio.sleep(30)

        except Exception as e:
            self.logger.error(f"Error in knowledge monitoring loop: {str(e)}")

    async def _facilitate_collaborations(self):
        """Facilitate and monitor active collaborations"""
        try:
            while True:
                for session_id, session in list(self.active_collaborations.items()):
                    try:
                        if session.status == "active":
                            # Check for new insights
                            insights = await self._analyze_collaboration(session)

                            if insights:
                                # Share insights through MCP
                                await self._share_collaboration_insights(
                                    session_id,
                                    insights
                                )

                            # Check if collaboration should end
                            if await self._should_end_collaboration(session):
                                await self._end_collaboration(session_id)

                    except Exception as e:
                        self.logger.error(
                            f"Error facilitating session {session_id}: {str(e)}"
                        )

                await asyncio.sleep(30)

        except Exception as e:
            self.logger.error(
                f"Error in collaboration facilitation loop: {str(e)}")

    async def _synchronize_knowledge(self):
        """Synchronize knowledge across agents"""
        try:
            while True:
                # Get all active agents
                active_agents = set(self.active_agents.keys())

                # Get recent knowledge updates
                recent_updates = await self.mcp_client.get_recent_contexts(
                    context_type="knowledge_update",
                    time_window_seconds=300
                )

                # Synchronize each agent
                for agent_id in active_agents:
                    try:
                        agent_type = self.active_agents[agent_id]["type"]
                        relevant_updates = self._filter_relevant_updates(
                            agent_type,
                            recent_updates
                        )

                        if relevant_updates:
                            await self._sync_agent_knowledge(
                                agent_id,
                                relevant_updates
                            )

                    except Exception as e:
                        self.logger.error(
                            f"Error synchronizing agent {agent_id}: {str(e)}"
                        )

                await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(
                f"Error in knowledge synchronization loop: {str(e)}")

    async def _notify_interested_agents(
        self,
        update_id: str,
        update: KnowledgeUpdate
    ):
        """Notify agents interested in a knowledge update"""
        try:
            # Find interested agents
            interested_agents = set()

            # Add subscribers
            for topic in self.knowledge_subscriptions:
                if topic in update.content.get("topics", []):
                    interested_agents.update(
                        self.knowledge_subscriptions[topic]
                    )

            # Add agents with matching specialties
            for agent_id, specialties in self.agent_specialties.items():
                if any(s in update.content.get("topics", []) for s in specialties):
                    interested_agents.add(agent_id)

            # Notify each interested agent
            for agent_id in interested_agents:
                if agent_id in self.active_agents:
                    await self._notify_agent(agent_id, update_id, update)

        except Exception as e:
            self.logger.error(f"Error notifying interested agents: {str(e)}")

    async def _notify_agent(
        self,
        agent_id: str,
        update_id: str,
        update: KnowledgeUpdate
    ):
        """Notify an agent about a knowledge update"""
        try:
            container = self.active_agents[agent_id]["container"]

            # Send notification through MCP
            await self.mcp_client.send_message(
                target_id=agent_id,
                message_type="knowledge_update",
                content={
                    "update_id": update_id,
                    "type": update.update_type,
                    "content": update.content,
                    "source": update.agent_id,
                    "timestamp": update.timestamp.isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Error notifying agent {agent_id}: {str(e)}")

    def _filter_relevant_updates(
        self,
        agent_type: str,
        updates: List[Dict]
    ) -> List[Dict]:
        """Filter updates relevant to an agent type"""
        try:
            relevant_updates = []
            agent_specialties = self.agent_specialties.get(agent_type, set())

            for update in updates:
                # Check relevance based on topics and specialties
                update_topics = set(update.get(
                    "content", {}).get("topics", []))
                if (update.get("type") == agent_type or
                        update_topics & agent_specialties):
                    relevant_updates.append(update)

            return relevant_updates

        except Exception as e:
            self.logger.error(f"Error filtering updates: {str(e)}")
            return []

    async def _sync_agent_knowledge(
        self,
        agent_id: str,
        updates: List[Dict]
    ):
        """Synchronize an agent's knowledge with recent updates"""
        try:
            container = self.active_agents[agent_id]["container"]

            # Send updates through MCP
            await self.mcp_client.send_message(
                target_id=agent_id,
                message_type="knowledge_sync",
                content={
                    "updates": updates
                }
            )

            KNOWLEDGE_REUSE.labels(
                type=self.active_agents[agent_id]["type"]
            ).inc()

        except Exception as e:
            self.logger.error(f"Error syncing agent {agent_id}: {str(e)}")
