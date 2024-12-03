from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from .graph_service import Node, Relationship


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


class CodeType(Enum):
    SOURCE = "source"
    TEST = "test"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"


class RelationType(Enum):
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    TESTS = "TESTS"
    DOCUMENTS = "DOCUMENTS"
    RELATED_TO = "RELATED_TO"
    BLOCKS = "BLOCKS"
    REQUIRES = "REQUIRES"


@dataclass
class TaskNode(Node):
    """Task node in the graph"""

    def __init__(
        self,
        task_id: str,
        title: str,
        description: str,
        status: TaskStatus,
        priority: TaskPriority,
        metadata: Dict[str, Any]
    ):
        super().__init__(
            id=task_id,
            labels=["Task"],
            properties={
                "title": title,
                "description": description,
                "status": status.value,
                "priority": priority.value,
                **metadata
            }
        )


@dataclass
class CodeNode(Node):
    """Code node in the graph"""

    def __init__(
        self,
        code_id: str,
        content: str,
        file_path: str,
        code_type: CodeType,
        language: str,
        metadata: Dict[str, Any]
    ):
        super().__init__(
            id=code_id,
            labels=["Code", code_type.value],
            properties={
                "content": content,
                "file_path": file_path,
                "language": language,
                **metadata
            }
        )


@dataclass
class ContextNode(Node):
    """Context node in the graph"""

    def __init__(
        self,
        context_id: str,
        content: str,
        context_type: str,
        metadata: Dict[str, Any],
        embeddings: Optional[List[float]] = None
    ):
        super().__init__(
            id=context_id,
            labels=["Context", context_type],
            properties={
                "content": content,
                "embeddings": embeddings,
                **metadata
            }
        )


@dataclass
class DependencyRelationship(Relationship):
    """Dependency relationship between nodes"""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        dependency_type: RelationType,
        metadata: Dict[str, Any]
    ):
        super().__init__(
            source_id=source_id,
            target_id=target_id,
            type=dependency_type.value,
            properties=metadata
        )


@dataclass
class TaskMetrics:
    """Metrics for a task"""
    completion_time: Optional[float] = None
    success_rate: float = 0.0
    complexity_score: float = 0.0
    dependency_count: int = 0
    blocking_issues: int = 0
    quality_score: float = 0.0


@dataclass
class CodeMetrics:
    """Metrics for code"""
    lines_of_code: int = 0
    complexity: float = 0.0
    test_coverage: float = 0.0
    quality_score: float = 0.0
    dependency_count: int = 0
    security_score: float = 0.0


@dataclass
class GraphMetrics:
    """Metrics for the entire graph"""
    total_nodes: int = 0
    total_relationships: int = 0
    average_node_degree: float = 0.0
    clustering_coefficient: float = 0.0
    connected_components: int = 0
    density: float = 0.0


@dataclass
class TaskAnalysis:
    """Analysis results for a task"""
    task_id: str
    metrics: TaskMetrics
    dependencies: List[str]
    blockers: List[str]
    related_code: List[str]
    contexts: List[str]
    recommendations: List[str]


@dataclass
class CodeAnalysis:
    """Analysis results for code"""
    code_id: str
    metrics: CodeMetrics
    dependencies: List[str]
    related_tasks: List[str]
    contexts: List[str]
    issues: List[str]
    suggestions: List[str]
