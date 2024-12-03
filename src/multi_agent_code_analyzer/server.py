import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, Dict, List, Union, Any
import os
from enum import Enum
import json
import uuid
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from multi_agent_code_analyzer.agents.agent_manager import AgentManager
from multi_agent_code_analyzer.agents.code_analyzer import CodeAnalyzerAgent
from multi_agent_code_analyzer.agents.developer import DeveloperAgent
from multi_agent_code_analyzer.config import Settings
from multi_agent_code_analyzer.tools.github import GithubService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Code Analyzer",
    description="A system for analyzing and improving codebases using multiple specialized agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings
settings = Settings()

# Initialize services
github_service = GithubService()
agent_manager = AgentManager()


class AgentType(str, Enum):
    CODE_ANALYZER = "code_analyzer"
    DEVELOPER = "developer"
    ORCHESTRATOR = "orchestrator"


class AnalysisRequest(BaseModel):
    repository_url: str
    branch: str = "main"
    analysis_type: str = "full"
    wait_for_completion: bool = False
    timeout: int = 300


class ImplementRequest(BaseModel):
    repo_url: str
    description: str
    branch: Optional[str] = None
    target_files: Optional[List[str]] = None
    wait_for_completion: bool = True
    timeout: int = 600


class CustomTaskRequest(BaseModel):
    agent_type: AgentType
    description: str
    context: Dict[str, Any]
    dependencies: Optional[List[str]] = None
    wait_for_completion: bool = True
    timeout: int = 300


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            # Add custom JSON encoders if needed
        }


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Verify the GitHub token"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization scheme"
            )
        return token
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/analyze/repository", response_model=TaskResponse)
async def analyze_repository(
    request: AnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyze a repository using the multi-agent system"""
    try:
        # Clone repository
        repo_path = await github_service.clone_repository(
            request.repository_url,
            request.branch
        )

        # Initialize agents with unique IDs
        code_analyzer = CodeAnalyzerAgent(agent_id=str(uuid.uuid4()))
        developer = DeveloperAgent(agent_id=str(uuid.uuid4()))

        # Register agents with manager
        agent_manager.register_agent("code_analyzer", code_analyzer)
        agent_manager.register_agent("developer", developer)

        # Start analysis
        task_id = await agent_manager.start_analysis(
            repo_path,
            analysis_type=request.analysis_type
        )

        if request.wait_for_completion:
            # Wait for analysis to complete
            result = await agent_manager.wait_for_completion(task_id)
            return TaskResponse(
                task_id=task_id,
                status="completed",
                result=result
            )
        else:
            return TaskResponse(
                task_id=task_id,
                status="in_progress"
            )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/implement/feature", response_model=TaskResponse)
async def implement_feature(
    request: ImplementRequest,
    token: str = Depends(verify_token)
):
    """Implement a feature or fix"""
    try:
        # Initialize developer agent with unique ID
        developer = DeveloperAgent(agent_id=str(uuid.uuid4()))

        # Register agent with manager
        agent_manager.register_agent("developer", developer)

        # Start implementation
        task_id = await agent_manager.start_implementation(
            request.repo_url,
            request.description,
            branch=request.branch,
            target_files=request.target_files
        )

        if request.wait_for_completion:
            # Wait for implementation to complete
            result = await agent_manager.wait_for_completion(task_id)
            return TaskResponse(
                task_id=task_id,
                status="completed",
                result=result
            )
        else:
            return TaskResponse(
                task_id=task_id,
                status="in_progress"
            )

    except Exception as e:
        logger.error(f"Implementation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Implementation failed: {str(e)}"
        )


@app.get("/analysis/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    token: str = Depends(verify_token)
):
    """Get the status of a task"""
    try:
        status = await agent_manager.get_task_status(task_id)
        if status["status"] == "completed":
            return TaskResponse(
                task_id=task_id,
                status="completed",
                result=status["result"]
            )
        else:
            return TaskResponse(
                task_id=task_id,
                status=status["status"]
            )

    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@app.get("/agent/{agent_id}/memory", response_model=Dict[str, Any])
async def get_agent_memory(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Get an agent's memory"""
    try:
        agent = agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        return agent.memory

    except Exception as e:
        logger.error(f"Failed to get agent memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent memory: {str(e)}"
        )


@app.get("/agent/{agent_id}/learnings", response_model=Dict[str, Any])
async def get_agent_learnings(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Get an agent's learning points"""
    try:
        agent = agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        return {
            "patterns": agent.pattern_learner.patterns,
            "confidence": agent.pattern_learner.pattern_confidence
        }

    except Exception as e:
        logger.error(f"Failed to get agent learnings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent learnings: {str(e)}"
        )


@app.post("/task/custom", response_model=TaskResponse)
async def create_custom_task(
    request: CustomTaskRequest,
    token: str = Depends(verify_token)
):
    """Create a custom task"""
    try:
        # Initialize appropriate agent with unique ID
        agent_id = str(uuid.uuid4())
        if request.agent_type == AgentType.CODE_ANALYZER:
            agent = CodeAnalyzerAgent(agent_id=agent_id)
        elif request.agent_type == AgentType.DEVELOPER:
            agent = DeveloperAgent(agent_id=agent_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type: {request.agent_type}"
            )

        # Register agent with manager
        agent_manager.register_agent(request.agent_type, agent)

        # Start task
        task_id = await agent_manager.start_custom_task(
            request.agent_type,
            request.description,
            request.context,
            dependencies=request.dependencies
        )

        if request.wait_for_completion:
            # Wait for task to complete
            result = await agent_manager.wait_for_completion(task_id)
            return TaskResponse(
                task_id=task_id,
                status="completed",
                result=result
            )
        else:
            return TaskResponse(
                task_id=task_id,
                status="in_progress"
            )

    except Exception as e:
        logger.error(f"Failed to create custom task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create custom task: {str(e)}"
        )


def start():
    """Start the FastAPI server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("SERVICE_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    start()
