from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from .config import settings
from .models.agent import Agent, AgentCreate, AgentUpdate, AgentState
from .services.agent_service import AgentService
import uuid
import re
from datetime import datetime

class Project(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    repo_url: str = Field(..., min_length=1)

    @field_validator('name')
    @classmethod
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty or whitespace')
        return v.strip()

    @field_validator('repo_url')
    @classmethod
    def repo_url_must_be_github(cls, v):
        pattern = r'^https://github\.com/[\w-]+/[\w-]+$'
        if not re.match(pattern, v):
            raise ValueError('repository URL must be a valid GitHub URL (e.g., https://github.com/username/repo)')
        return v

class ProjectResponse(Project):
    id: str
    status: str

app = FastAPI(
    title="Multi-Agent Code Analyzer",
    description="A system for analyzing code using multiple agents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
projects = {}
agent_service = AgentService()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/projects", response_model=ProjectResponse)
async def create_project(project: Project):
    # Check for duplicate project name
    if any(p.name == project.name for p in projects.values()):
        raise HTTPException(status_code=409, detail="Project with this name already exists")
    
    project_id = str(uuid.uuid4())
    project_data = ProjectResponse(
        id=project_id,
        status="Active",
        **project.model_dump()
    )
    projects[project_id] = project_data
    return project_data

@app.get("/projects", response_model=List[ProjectResponse])
async def list_projects():
    return list(projects.values())

@app.post("/agents", response_model=Agent)
async def create_agent(agent: AgentCreate):
    try:
        return agent_service.create_agent(agent)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/agents", response_model=List[Agent])
async def list_agents():
    return agent_service.list_agents()

@app.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str):
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.patch("/agents/{agent_id}", response_model=Agent)
async def update_agent(agent_id: str, agent_update: AgentUpdate):
    agent = agent_service.update_agent(agent_id, agent_update)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if not agent_service.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "success", "message": "Agent deleted"}
