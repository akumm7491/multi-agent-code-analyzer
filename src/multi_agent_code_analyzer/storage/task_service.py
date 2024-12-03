from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .graph_service import GraphService
from .models import (
    TaskNode,
    CodeNode,
    ContextNode,
    DependencyRelationship,
    TaskMetrics,
    TaskAnalysis,
    TaskStatus,
    TaskPriority,
    RelationType
)


class TaskGraphService:
    """Service for managing task relationships in the graph database"""

    def __init__(self, graph_service: GraphService):
        self.graph = graph_service
        self.logger = logging.getLogger(__name__)

    async def create_task(
        self,
        title: str,
        description: str,
        priority: TaskPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a new task node"""
        task_id = f"task_{int(datetime.now().timestamp())}"
        task = TaskNode(
            task_id=task_id,
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            metadata=metadata or {}
        )

        success = await self.graph.create_node(task)
        return task_id if success else None

    async def link_dependent_task(
        self,
        task_id: str,
        dependent_task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a dependency relationship between tasks"""
        relationship = DependencyRelationship(
            source_id=task_id,
            target_id=dependent_task_id,
            dependency_type=RelationType.DEPENDS_ON,
            metadata=metadata or {}
        )
        return await self.graph.create_relationship(relationship)

    async def link_task_to_code(
        self,
        task_id: str,
        code_id: str,
        relationship_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between task and code"""
        relationship = DependencyRelationship(
            source_id=task_id,
            target_id=code_id,
            dependency_type=relationship_type,
            metadata=metadata or {}
        )
        return await self.graph.create_relationship(relationship)

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update task status"""
        properties = {
            "status": status.value,
            **(metadata or {})
        }
        return await self.graph.update_node(task_id, properties)

    async def get_task_dependencies(
        self,
        task_id: str,
        recursive: bool = False,
        max_depth: int = 3
    ) -> List[str]:
        """Get task dependencies"""
        if not recursive:
            relationships = await self.graph.get_relationships(
                task_id,
                RelationType.DEPENDS_ON.value,
                direction="outgoing"
            )
            return [rel.target_id for rel in relationships]

        # Use Cypher for recursive query
        query = """
        MATCH (t:Task {id: $task_id})
        CALL apoc.path.subgraphNodes(t, {
            relationshipFilter: "DEPENDS_ON>",
            maxLevel: $max_depth
        })
        YIELD node
        WHERE node:Task AND node.id <> $task_id
        RETURN node.id as dependent_id
        """

        try:
            async with self.graph.driver.session() as session:
                result = await session.run(
                    query,
                    {"task_id": task_id, "max_depth": max_depth}
                )
                return [record["dependent_id"] async for record in result]
        except Exception as e:
            self.logger.error(f"Error getting recursive dependencies: {e}")
            return []

    async def get_blocking_tasks(self, task_id: str) -> List[str]:
        """Get tasks blocking this task"""
        relationships = await self.graph.get_relationships(
            task_id,
            RelationType.BLOCKS.value,
            direction="incoming"
        )
        return [rel.source_id for rel in relationships]

    async def get_related_code(
        self,
        task_id: str,
        relationship_types: Optional[List[RelationType]] = None
    ) -> List[str]:
        """Get code related to this task"""
        if not relationship_types:
            relationship_types = [
                RelationType.IMPLEMENTS,
                RelationType.TESTS,
                RelationType.DOCUMENTS
            ]

        code_ids = []
        for rel_type in relationship_types:
            relationships = await self.graph.get_relationships(
                task_id,
                rel_type.value,
                direction="outgoing"
            )
            code_ids.extend([rel.target_id for rel in relationships])

        return code_ids

    async def analyze_task(self, task_id: str) -> Optional[TaskAnalysis]:
        """Analyze a task's relationships and metrics"""
        try:
            # Get task node
            task_node = await self.graph.get_node(task_id)
            if not task_node:
                return None

            # Get dependencies
            dependencies = await self.get_task_dependencies(task_id)

            # Get blockers
            blockers = await self.get_blocking_tasks(task_id)

            # Get related code
            related_code = await self.get_related_code(task_id)

            # Get related contexts
            context_relationships = await self.graph.get_relationships(
                task_id,
                RelationType.RELATED_TO.value
            )
            contexts = [
                rel.target_id for rel in context_relationships
                if "Context" in await self._get_node_labels(rel.target_id)
            ]

            # Calculate metrics
            metrics = TaskMetrics(
                dependency_count=len(dependencies),
                blocking_issues=len(blockers),
                complexity_score=self._calculate_complexity(
                    len(dependencies),
                    len(related_code)
                )
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                task_id,
                metrics,
                len(related_code)
            )

            return TaskAnalysis(
                task_id=task_id,
                metrics=metrics,
                dependencies=dependencies,
                blockers=blockers,
                related_code=related_code,
                contexts=contexts,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error analyzing task: {e}")
            return None

    async def _get_node_labels(self, node_id: str) -> List[str]:
        """Get labels for a node"""
        node = await self.graph.get_node(node_id)
        return node.labels if node else []

    def _calculate_complexity(
        self,
        dependency_count: int,
        code_count: int
    ) -> float:
        """Calculate task complexity score"""
        # Simple complexity calculation
        base_score = 0.5
        dependency_weight = 0.2
        code_weight = 0.3

        return min(
            1.0,
            base_score +
            (dependency_count * dependency_weight) +
            (code_count * code_weight)
        )

    async def _generate_recommendations(
        self,
        task_id: str,
        metrics: TaskMetrics,
        code_count: int
    ) -> List[str]:
        """Generate recommendations for task improvement"""
        recommendations = []

        if metrics.blocking_issues > 0:
            recommendations.append(
                "Address blocking issues before proceeding"
            )

        if metrics.dependency_count > 5:
            recommendations.append(
                "Consider breaking task into smaller subtasks"
            )

        if code_count == 0:
            recommendations.append(
                "No code implementation found - start implementation"
            )

        if metrics.complexity_score > 0.8:
            recommendations.append(
                "High complexity - consider simplifying or refactoring"
            )

        return recommendations
