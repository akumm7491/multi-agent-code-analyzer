import pytest
import os
from datetime import datetime
from multi_agent_code_analyzer.storage.graph_service import GraphService
from multi_agent_code_analyzer.storage.task_service import TaskGraphService
from multi_agent_code_analyzer.storage.code_service import CodeGraphService
from multi_agent_code_analyzer.storage.models import (
    TaskPriority,
    CodeType,
    RelationType
)


@pytest.fixture
async def graph_service():
    """Initialize graph service"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    service = GraphService(uri, user, password)
    async with service:
        yield service


@pytest.fixture
async def task_service(graph_service):
    """Initialize task service"""
    return TaskGraphService(graph_service)


@pytest.fixture
async def code_service(graph_service):
    """Initialize code service"""
    return CodeGraphService(graph_service)


@pytest.mark.asyncio
async def test_full_workflow(task_service, code_service):
    """Test complete workflow with tasks and code"""

    # 1. Create a task
    task_id = await task_service.create_task(
        title="Implement Authentication",
        description="Create OAuth2 authentication system",
        priority=TaskPriority.HIGH,
        metadata={
            "assignee": "test_user",
            "due_date": datetime.now().isoformat()
        }
    )
    assert task_id is not None

    # 2. Create code implementations
    code_impl = """
    def authenticate(token: str) -> bool:
        # Implementation
        return True
    """

    impl_id = await code_service.store_code(
        content=code_impl,
        file_path="auth.py",
        code_type=CodeType.SOURCE,
        language="python",
        metadata={"framework": "fastapi"}
    )
    assert impl_id is not None

    # Create test code
    test_code = """
    def test_authenticate():
        assert authenticate("test_token") is True
    """

    test_id = await code_service.store_code(
        content=test_code,
        file_path="test_auth.py",
        code_type=CodeType.TEST,
        language="python",
        metadata={"framework": "pytest"}
    )
    assert test_id is not None

    # 3. Link code to task
    success = await task_service.link_task_to_code(
        task_id,
        impl_id,
        RelationType.IMPLEMENTS
    )
    assert success is True

    success = await task_service.link_task_to_code(
        task_id,
        test_id,
        RelationType.TESTS
    )
    assert success is True

    # 4. Create code dependencies
    success = await code_service.link_code_dependency(
        test_id,
        impl_id,
        RelationType.DEPENDS_ON
    )
    assert success is True

    # 5. Analyze task
    task_analysis = await task_service.analyze_task(task_id)
    assert task_analysis is not None
    assert len(task_analysis.related_code) == 2
    assert task_analysis.metrics.dependency_count > 0

    # 6. Analyze code
    code_analysis = await code_service.analyze_code(impl_id)
    assert code_analysis is not None
    assert len(code_analysis.related_tasks) == 1
    assert code_analysis.metrics.lines_of_code > 0

    # 7. Verify relationships
    related_code = await task_service.get_related_code(task_id)
    assert len(related_code) == 2
    assert impl_id in related_code
    assert test_id in related_code

    related_tasks = await code_service.get_related_tasks(impl_id)
    assert len(related_tasks) == 1
    assert task_id in related_tasks

    # 8. Check code dependencies
    dependencies = await code_service.get_code_dependencies(test_id)
    assert len(dependencies) == 1
    assert impl_id in dependencies


@pytest.mark.asyncio
async def test_complex_task_relationships(task_service):
    """Test complex task relationships and analysis"""

    # Create main task
    main_task_id = await task_service.create_task(
        title="Implement API Gateway",
        description="Create API Gateway with auth and routing",
        priority=TaskPriority.HIGH
    )

    # Create subtasks
    subtask_ids = []
    for i in range(3):
        subtask_id = await task_service.create_task(
            title=f"Subtask {i+1}",
            description=f"Implementation subtask {i+1}",
            priority=TaskPriority.MEDIUM
        )
        subtask_ids.append(subtask_id)

        # Link to main task
        await task_service.link_dependent_task(
            main_task_id,
            subtask_id
        )

    # Analyze main task
    analysis = await task_service.analyze_task(main_task_id)
    assert analysis is not None
    assert len(analysis.dependencies) == 3
    assert analysis.metrics.complexity_score > 0.5


@pytest.mark.asyncio
async def test_code_quality_analysis(code_service):
    """Test code quality analysis features"""

    # Create code with potential issues
    complex_code = """
    def complex_function(a, b, c, d):
        if a > 0:
            if b > 0:
                if c > 0:
                    if d > 0:
                        return a + b + c + d
                    return a + b + c
                return a + b
            return a
        return 0
    """

    code_id = await code_service.store_code(
        content=complex_code,
        file_path="complex.py",
        code_type=CodeType.SOURCE,
        language="python"
    )

    # Analyze code
    analysis = await code_service.analyze_code(code_id)
    assert analysis is not None
    assert analysis.metrics.complexity > 0.7
    assert len(analysis.issues) > 0
    assert len(analysis.suggestions) > 0
    assert "High code complexity" in analysis.issues
