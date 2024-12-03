import asyncio
import pytest
from multi_agent_code_analyzer.tools.manager import ToolManager
from multi_agent_code_analyzer.context.fastmcp_adapter import FastMCPAdapter
from multi_agent_code_analyzer.generation.code_generator import CodeGenerator
from multi_agent_code_analyzer.orchestration.task_orchestrator import TaskOrchestrator, TaskPriority
from multi_agent_code_analyzer.orchestration.workflow import WorkflowOrchestrator, WorkflowType
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")


@pytest.fixture
async def mcp_server():
    """Verify MCP server is running"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{MCP_SERVER_URL}/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "healthy"
        except Exception as e:
            pytest.fail(f"MCP server is not running: {e}")


@pytest.fixture
async def tool_manager():
    """Initialize tool manager with configurations"""
    manager = ToolManager()

    # Configure GitHub tool
    manager.configure_tool("GitHubTool", {
        "token": GITHUB_TOKEN,
        "owner": GITHUB_OWNER,
        "repo": GITHUB_REPO
    })

    # Configure JIRA tool
    manager.configure_tool("JiraTool", {
        "domain": JIRA_DOMAIN,
        "email": JIRA_EMAIL,
        "api_token": JIRA_API_TOKEN,
        "project_key": JIRA_PROJECT_KEY
    })

    return manager


@pytest.fixture
async def context_adapter():
    """Initialize FastMCP adapter"""
    adapter = FastMCPAdapter(MCP_SERVER_URL)
    async with adapter:
        yield adapter


@pytest.fixture
async def code_generator(tool_manager, context_adapter):
    """Initialize code generator"""
    return CodeGenerator(tool_manager, context_adapter)


@pytest.fixture
async def task_orchestrator(tool_manager, context_adapter):
    """Initialize task orchestrator"""
    return TaskOrchestrator(tool_manager, context_adapter)


@pytest.fixture
async def workflow_orchestrator(task_orchestrator, code_generator, tool_manager, context_adapter):
    """Initialize workflow orchestrator"""
    return WorkflowOrchestrator(
        task_orchestrator,
        code_generator,
        tool_manager,
        context_adapter
    )


@pytest.mark.asyncio
async def test_mcp_context_operations(context_adapter, mcp_server):
    """Test basic MCP context operations"""
    # Store context
    context_id = "test_context"
    content = "def test_function(): pass"
    metadata = {"language": "python", "type": "function"}

    success = await context_adapter.store_context(
        context_id,
        FastMCPContext(
            content=content,
            metadata=metadata,
            relationships=[]
        )
    )
    assert success

    # Retrieve context
    retrieved = await context_adapter.retrieve_context(context_id)
    assert retrieved is not None
    assert retrieved.content == content
    assert retrieved.metadata == metadata


@pytest.mark.asyncio
async def test_code_generation(code_generator):
    """Test code generation capabilities"""
    code = await code_generator.generate_code(
        description="Create a REST API endpoint for user authentication",
        language="python",
        framework="fastapi"
    )

    assert code.code is not None
    assert code.tests is not None
    assert code.documentation is not None
    assert code.quality_score >= 0.8


@pytest.mark.asyncio
async def test_task_execution(task_orchestrator):
    """Test task execution flow"""
    task_id = await task_orchestrator.submit_task(
        task_type="feature",
        description="Implement user authentication endpoint",
        priority=TaskPriority.HIGH,
        metadata={"language": "python", "framework": "fastapi"}
    )

    assert task_id is not None

    # Start task processing
    asyncio.create_task(task_orchestrator.process_tasks())

    # Wait for task completion
    for _ in range(30):  # Wait up to 30 seconds
        status = await task_orchestrator.get_task_status(task_id)
        if status and status["status"] not in ("pending", "in_progress"):
            break
        await asyncio.sleep(1)

    status = await task_orchestrator.get_task_status(task_id)
    assert status is not None
    assert status["status"] in ("completed", "reviewing")


@pytest.mark.asyncio
async def test_workflow_execution(workflow_orchestrator):
    """Test workflow execution"""
    workflow_id = workflow_orchestrator.create_feature_workflow(
        name="User Authentication",
        description="Implement OAuth2 authentication with JWT tokens",
        language="python",
        framework="fastapi",
        metadata={
            "priority": "high",
            "security_requirements": ["OWASP Top 10"]
        }
    )

    assert workflow_id is not None

    result = await workflow_orchestrator.execute_workflow(workflow_id)
    assert result.success
    assert result.status in ("completed", "reviewing")

    # Check metrics
    metrics = await workflow_orchestrator.get_workflow_metrics()
    assert metrics["total_workflows"] > 0
    assert metrics["success_rate"] > 0


@pytest.mark.asyncio
async def test_end_to_end_feature_development(
    workflow_orchestrator,
    task_orchestrator,
    context_adapter
):
    """Test end-to-end feature development flow"""
    # Create a feature workflow
    workflow_id = workflow_orchestrator.create_feature_workflow(
        name="User Profile API",
        description="""
        Create a REST API endpoint for user profile management with the following requirements:
        - CRUD operations for user profiles
        - Input validation
        - Error handling
        - Authentication middleware
        - Unit tests
        - API documentation
        """,
        language="python",
        framework="fastapi"
    )

    # Execute workflow
    result = await workflow_orchestrator.execute_workflow(workflow_id)
    assert result.success

    # Verify artifacts
    assert "code" in result.artifacts
    assert "tests" in result.artifacts
    assert "documentation" in result.artifacts

    # Check code quality
    code_quality = result.artifacts.get(
        "code_review", {}).get("validation_results", {})
    assert code_quality.get("quality_score", 0) >= 0.8

    # Verify context storage
    workflow_context = await context_adapter.retrieve_context(f"workflow_{workflow_id}")
    assert workflow_context is not None

    # Check metrics
    metrics = await workflow_orchestrator.get_workflow_metrics()
    assert metrics["success_rate"] > 0
