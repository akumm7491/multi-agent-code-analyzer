from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import hashlib
from .graph_service import GraphService
from .models import (
    CodeNode,
    CodeMetrics,
    CodeAnalysis,
    DependencyRelationship,
    CodeType,
    RelationType
)


class CodeGraphService:
    """Service for managing code relationships in the graph database"""

    def __init__(self, graph_service: GraphService):
        self.graph = graph_service
        self.logger = logging.getLogger(__name__)

    def _generate_code_id(self, file_path: str, content: str) -> str:
        """Generate a unique code ID based on file path and content"""
        hash_input = f"{file_path}:{content}"
        return f"code_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    async def store_code(
        self,
        content: str,
        file_path: str,
        code_type: CodeType,
        language: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store code in the graph database"""
        code_id = self._generate_code_id(file_path, content)
        code = CodeNode(
            code_id=code_id,
            content=content,
            file_path=file_path,
            code_type=code_type,
            language=language,
            metadata=metadata or {}
        )

        success = await self.graph.create_node(code)
        return code_id if success else None

    async def link_code_dependency(
        self,
        source_code_id: str,
        target_code_id: str,
        dependency_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a dependency relationship between code nodes"""
        relationship = DependencyRelationship(
            source_id=source_code_id,
            target_id=target_code_id,
            dependency_type=dependency_type,
            metadata=metadata or {}
        )
        return await self.graph.create_relationship(relationship)

    async def get_code_dependencies(
        self,
        code_id: str,
        recursive: bool = False,
        max_depth: int = 3
    ) -> List[str]:
        """Get code dependencies"""
        if not recursive:
            relationships = await self.graph.get_relationships(
                code_id,
                RelationType.DEPENDS_ON.value,
                direction="outgoing"
            )
            return [rel.target_id for rel in relationships]

        # Use Cypher for recursive query
        query = """
        MATCH (c:Code {id: $code_id})
        CALL apoc.path.subgraphNodes(c, {
            relationshipFilter: "DEPENDS_ON>",
            maxLevel: $max_depth
        })
        YIELD node
        WHERE node:Code AND node.id <> $code_id
        RETURN node.id as dependent_id
        """

        try:
            async with self.graph.driver.session() as session:
                result = await session.run(
                    query,
                    {"code_id": code_id, "max_depth": max_depth}
                )
                return [record["dependent_id"] async for record in result]
        except Exception as e:
            self.logger.error(f"Error getting recursive dependencies: {e}")
            return []

    async def get_related_tasks(
        self,
        code_id: str,
        relationship_types: Optional[List[RelationType]] = None
    ) -> List[str]:
        """Get tasks related to this code"""
        if not relationship_types:
            relationship_types = [
                RelationType.IMPLEMENTS,
                RelationType.TESTS,
                RelationType.DOCUMENTS
            ]

        task_ids = []
        for rel_type in relationship_types:
            relationships = await self.graph.get_relationships(
                code_id,
                rel_type.value,
                direction="incoming"
            )
            task_ids.extend([rel.source_id for rel in relationships])

        return task_ids

    async def analyze_code(self, code_id: str) -> Optional[CodeAnalysis]:
        """Analyze code relationships and metrics"""
        try:
            # Get code node
            code_node = await self.graph.get_node(code_id)
            if not code_node:
                return None

            # Get dependencies
            dependencies = await self.get_code_dependencies(code_id)

            # Get related tasks
            related_tasks = await self.get_related_tasks(code_id)

            # Get related contexts
            context_relationships = await self.graph.get_relationships(
                code_id,
                RelationType.RELATED_TO.value
            )
            contexts = [
                rel.target_id for rel in context_relationships
                if "Context" in await self._get_node_labels(rel.target_id)
            ]

            # Calculate metrics
            metrics = await self._calculate_metrics(code_node)

            # Generate suggestions
            issues, suggestions = await self._analyze_code_quality(
                code_node,
                metrics
            )

            return CodeAnalysis(
                code_id=code_id,
                metrics=metrics,
                dependencies=dependencies,
                related_tasks=related_tasks,
                contexts=contexts,
                issues=issues,
                suggestions=suggestions
            )

        except Exception as e:
            self.logger.error(f"Error analyzing code: {e}")
            return None

    async def _get_node_labels(self, node_id: str) -> List[str]:
        """Get labels for a node"""
        node = await self.graph.get_node(node_id)
        return node.labels if node else []

    async def _calculate_metrics(self, code_node: CodeNode) -> CodeMetrics:
        """Calculate code metrics"""
        content = code_node.properties.get("content", "")

        # Calculate basic metrics
        lines = content.split("\n")
        loc = len([line for line in lines if line.strip()])

        # Calculate complexity (placeholder for actual complexity calculation)
        complexity = min(1.0, loc / 1000)

        # Get test coverage if available
        test_coverage = code_node.properties.get("test_coverage", 0.0)

        # Calculate security score based on patterns (placeholder)
        security_score = 0.8  # Default good score

        # Get dependency count
        dependencies = await self.get_code_dependencies(code_node.id)

        return CodeMetrics(
            lines_of_code=loc,
            complexity=complexity,
            test_coverage=test_coverage,
            quality_score=self._calculate_quality_score(
                complexity,
                test_coverage,
                security_score
            ),
            dependency_count=len(dependencies),
            security_score=security_score
        )

    def _calculate_quality_score(
        self,
        complexity: float,
        test_coverage: float,
        security_score: float
    ) -> float:
        """Calculate overall code quality score"""
        weights = {
            "complexity": 0.3,
            "test_coverage": 0.4,
            "security": 0.3
        }

        # Normalize complexity (lower is better)
        complexity_score = 1.0 - complexity

        return (
            (complexity_score * weights["complexity"]) +
            (test_coverage * weights["test_coverage"]) +
            (security_score * weights["security"])
        )

    async def _analyze_code_quality(
        self,
        code_node: CodeNode,
        metrics: CodeMetrics
    ) -> tuple[List[str], List[str]]:
        """Analyze code quality and generate issues and suggestions"""
        issues = []
        suggestions = []

        # Check complexity
        if metrics.complexity > 0.7:
            issues.append("High code complexity")
            suggestions.append("Consider refactoring complex methods")

        # Check test coverage
        if metrics.test_coverage < 0.8:
            issues.append("Low test coverage")
            suggestions.append("Add more unit tests to improve coverage")

        # Check security score
        if metrics.security_score < 0.7:
            issues.append("Security concerns detected")
            suggestions.append("Review and address security vulnerabilities")

        # Check dependencies
        if metrics.dependency_count > 10:
            issues.append("High number of dependencies")
            suggestions.append("Consider reducing dependencies")

        # Check documentation
        if "documentation" not in code_node.properties:
            suggestions.append("Add documentation to improve maintainability")

        return issues, suggestions
