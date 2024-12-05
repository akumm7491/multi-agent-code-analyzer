from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..config import settings
from ..core.mcp_client import FastMCPClient

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Agent:
    def __init__(self, agent_type: str, agent_id: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.status = "idle"
        self.current_task: Optional[Dict] = None
        self.last_heartbeat = datetime.now()


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.mcp = FastMCPClient()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.max_agents = settings.MAX_AGENTS

    async def register_agent(self, agent_type: str) -> str:
        """Register a new agent."""
        if len(self.agents) >= self.max_agents:
            raise HTTPException(
                status_code=400,
                detail="Maximum number of agents reached"
            )

        agent_id = f"{agent_type}_{len(self.agents)}"
        self.agents[agent_id] = Agent(agent_type, agent_id)
        return agent_id

    async def assign_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Assign a task to an available agent."""
        for agent_id, agent in self.agents.items():
            if agent.status == "idle":
                agent.status = "busy"
                agent.current_task = task
                return agent_id

        await self.task_queue.put(task)
        return None

    async def complete_task(self, agent_id: str, result: Dict[str, Any]):
        """Mark a task as completed."""
        if agent_id not in self.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        agent = self.agents[agent_id]
        agent.status = "idle"
        agent.current_task = None

        # Check for queued tasks
        if not self.task_queue.empty():
            next_task = await self.task_queue.get()
            agent.status = "busy"
            agent.current_task = next_task

    async def heartbeat(self, agent_id: str):
        """Update agent heartbeat."""
        if agent_id not in self.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        self.agents[agent_id].last_heartbeat = datetime.now()


agent_manager = AgentManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/agents/register/{agent_type}")
async def register_agent(agent_type: str):
    """Register a new agent."""
    try:
        agent_id = await agent_manager.register_agent(agent_type)
        return {"agent_id": agent_id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register agent: {str(e)}"
        )


@app.post("/tasks/assign")
async def assign_task(task: Dict[str, Any]):
    """Assign a task to an agent."""
    try:
        agent_id = await agent_manager.assign_task(task)
        return {
            "status": "queued" if agent_id is None else "assigned",
            "agent_id": agent_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign task: {str(e)}"
        )


@app.post("/tasks/complete/{agent_id}")
async def complete_task(agent_id: str, result: Dict[str, Any]):
    """Mark a task as completed."""
    try:
        await agent_manager.complete_task(agent_id, result)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete task: {str(e)}"
        )


@app.post("/agents/heartbeat/{agent_id}")
async def agent_heartbeat(agent_id: str):
    """Update agent heartbeat."""
    try:
        await agent_manager.heartbeat(agent_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update heartbeat: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent_manager:app",
        host="0.0.0.0",
        port=8081,
        reload=settings.DEBUG
    )
