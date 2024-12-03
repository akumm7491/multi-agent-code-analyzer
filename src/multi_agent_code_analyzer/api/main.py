from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import uuid
from prometheus_client import Counter, generate_latest
from ..tools.github import GithubService
from ..repository.connector import RepositoryConnector
from ..agents.agent_manager import AgentManager, AgentType, AgentTask

app = FastAPI(
    title="Multi-Agent Code Analyzer API",
    description="A service for automated code analysis and development tasks using multiple AI agents",
    version="1.0.0"
)

# Metrics
analysis_requests = Counter(
    'code_analysis_requests_total', 'Total number of code analysis requests')
development_tasks = Counter(
    'development_tasks_total', 'Total number of development task requests')

# Initialize agent manager
agent_manager = AgentManager()


class RepositoryRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = "main"
    access_token: Optional[str] = None


class DevelopmentTask(BaseModel):
    repo_url: str
    task_description: str
    branch: Optional[str] = "main"
    access_token: Optional[str] = None
    target_files: Optional[List[str]] = None


class AgentTaskRequest(BaseModel):
    agent_type: AgentType
    description: str
    context: Dict
    dependencies: List[str] = []


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return generate_latest()


@app.post("/agents/tasks")
async def create_agent_task(task_request: AgentTaskRequest):
    """
    Create and assign a new task to an appropriate agent
    """
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        agent_type=task_request.agent_type,
        description=task_request.description,
        context=task_request.context,
        dependencies=task_request.dependencies
    )

    success = await agent_manager.assign_task(task)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to assign task")

    return {"task_id": task.task_id, "status": "assigned"}


@app.get("/agents/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a specific task
    """
    task = await agent_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/agents/active")
async def get_active_agents():
    """
    Get list of all active agents
    """
    return await agent_manager.get_active_agents()


@app.post("/analyze/repository")
async def analyze_repository(request: RepositoryRequest):
    """
    Analyze an entire repository using multiple specialized agents
    """
    analysis_requests.inc()
    try:
        # Create analysis tasks for different aspects
        tasks = []
        for agent_type in [AgentType.CODE_ANALYZER, AgentType.ARCHITECT]:
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=agent_type,
                description=f"Analyze repository: {request.repo_url}",
                context={
                    "repo_url": request.repo_url,
                    "branch": request.branch,
                    "access_token": request.access_token
                }
            )
            await agent_manager.assign_task(task)
            tasks.append(task.task_id)

        return {
            "status": "analysis_started",
            "task_ids": tasks
        }
    except Exception as e:
        logging.error(f"Repository analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/develop/task")
async def execute_development_task(task: DevelopmentTask):
    """
    Execute a development task using multiple specialized agents
    """
    development_tasks.inc()
    try:
        # Create development task chain
        tasks = []

        # 1. Architect review
        architect_task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.ARCHITECT,
            description=f"Review architecture for task: {task.task_description}",
            context={
                "repo_url": task.repo_url,
                "branch": task.branch,
                "access_token": task.access_token
            }
        )
        await agent_manager.assign_task(architect_task)
        tasks.append(architect_task.task_id)

        # 2. Developer implementation
        dev_task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.DEVELOPER,
            description=task.task_description,
            context={
                "repo_url": task.repo_url,
                "branch": task.branch,
                "access_token": task.access_token,
                "target_files": task.target_files
            },
            dependencies=[architect_task.task_id]
        )
        await agent_manager.assign_task(dev_task)
        tasks.append(dev_task.task_id)

        # 3. Code review
        review_task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.REVIEWER,
            description=f"Review changes for: {task.task_description}",
            context={
                "repo_url": task.repo_url,
                "branch": task.branch,
                "access_token": task.access_token
            },
            dependencies=[dev_task.task_id]
        )
        await agent_manager.assign_task(review_task)
        tasks.append(review_task.task_id)

        return {
            "status": "development_started",
            "task_ids": tasks
        }
    except Exception as e:
        logging.error(f"Development task failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
