import pytest
import asyncio
from unittest.mock import Mock, patch
from ..agents.base_agent import BaseAgent, Memory, AgentState
from ..agents.code_analyzer import CodeAnalyzerAgent
from ..agents.developer import DeveloperAgent
from ..agents.agent_manager import AgentManager, AgentType, AgentTask


@pytest.fixture
def agent_manager():
    return AgentManager()


@pytest.fixture
def mock_github_service():
    with patch("multi_agent_code_analyzer.tools.github.GithubService") as mock:
        yield mock


@pytest.mark.asyncio
async def test_code_analyzer_agent():
    """Test CodeAnalyzerAgent functionality"""
    agent = CodeAnalyzerAgent("test_analyzer")

    # Test repository analysis
    result = await agent.analyze_repository(
        "https://github.com/test/repo",
        "main",
        "test_token"
    )

    assert "analysis" in result
    assert "reflection" in result
    assert "understanding" in result

    # Verify memory storage
    assert len(agent.memories) > 0
    assert agent.memories[0].action == "Analyze repository structure and architecture"


@pytest.mark.asyncio
async def test_developer_agent():
    """Test DeveloperAgent functionality"""
    agent = DeveloperAgent("test_developer")

    # Test feature implementation
    result = await agent.implement_feature(
        "Add user authentication",
        {
            "repo_url": "https://github.com/test/repo",
            "branch": "main",
            "access_token": "test_token"
        }
    )

    assert "changes" in result
    assert "reflection" in result
    assert "tests" in result

    # Verify test generation
    assert len(agent.test_cases) > 0


@pytest.mark.asyncio
async def test_agent_manager(agent_manager):
    """Test AgentManager functionality"""
    # Test agent spawning
    agent = await agent_manager.spawn_agent(AgentType.CODE_ANALYZER)
    assert agent.agent_id in agent_manager.agents

    # Test task assignment
    task = AgentTask(
        task_id="test_task",
        agent_type=AgentType.CODE_ANALYZER,
        description="Analyze test repository",
        context={
            "repo_url": "https://github.com/test/repo",
            "branch": "main",
            "access_token": "test_token"
        }
    )

    success = await agent_manager.assign_task(task)
    assert success
    assert task.task_id in agent_manager.tasks


@pytest.mark.asyncio
async def test_agent_learning():
    """Test agent learning capabilities"""
    agent = CodeAnalyzerAgent("test_learner")

    # Simulate some experiences
    memory1 = Memory(
        timestamp="2024-01-01T00:00:00",
        context='{"type": "analysis"}',
        action="Analyze code patterns",
        result='{"patterns": ["singleton"]}',
        reflection="Pattern detection successful"
    )

    memory2 = Memory(
        timestamp="2024-01-01T00:01:00",
        context='{"type": "analysis"}',
        action="Analyze dependencies",
        result='{"error": "Missing package.json"}',
        reflection="Need to check file existence first"
    )

    agent.memories.extend([memory1, memory2])

    # Test learning process
    await agent.learn()

    assert "success_patterns" in agent.learning_points
    assert "failure_patterns" in agent.learning_points


@pytest.mark.asyncio
async def test_end_to_end_workflow(agent_manager, mock_github_service):
    """Test complete workflow from analysis to implementation"""
    # 1. Analyze repository
    analysis_task = AgentTask(
        task_id="analysis_1",
        agent_type=AgentType.CODE_ANALYZER,
        description="Analyze test repository",
        context={
            "repo_url": "https://github.com/test/repo",
            "branch": "main",
            "access_token": "test_token"
        }
    )

    await agent_manager.assign_task(analysis_task)
    analysis_result = agent_manager.tasks[analysis_task.task_id].result

    # 2. Implement feature based on analysis
    dev_task = AgentTask(
        task_id="dev_1",
        agent_type=AgentType.DEVELOPER,
        description="Implement user authentication",
        context={
            "repo_url": "https://github.com/test/repo",
            "branch": "feature/auth",
            "access_token": "test_token",
            "analysis_result": analysis_result
        }
    )

    await agent_manager.assign_task(dev_task)
    dev_result = agent_manager.tasks[dev_task.task_id].result

    # Verify workflow results
    assert analysis_task.status == "completed"
    assert dev_task.status == "completed"
    assert "pull_request_url" in dev_result


@pytest.mark.asyncio
async def test_error_handling(agent_manager):
    """Test system error handling"""
    # Test with invalid agent type
    with pytest.raises(ValueError):
        await agent_manager.spawn_agent("invalid_type")

    # Test with invalid task
    task = AgentTask(
        task_id="invalid_task",
        agent_type=AgentType.ARCHITECT,  # Not implemented yet
        description="Invalid task",
        context={}
    )

    success = await agent_manager.assign_task(task)
    assert not success
    assert task.status == "failed"


@pytest.mark.asyncio
async def test_concurrent_tasks(agent_manager):
    """Test handling of concurrent tasks"""
    tasks = []
    for i in range(5):
        task = AgentTask(
            task_id=f"task_{i}",
            agent_type=AgentType.CODE_ANALYZER,
            description=f"Analysis task {i}",
            context={
                "repo_url": "https://github.com/test/repo",
                "branch": "main",
                "access_token": "test_token"
            }
        )
        tasks.append(task)

    # Execute tasks concurrently
    results = await asyncio.gather(*[
        agent_manager.assign_task(task)
        for task in tasks
    ])

    # Verify all tasks completed
    assert all(results)
    assert len(agent_manager.agents) > 0  # Should have spawned multiple agents
