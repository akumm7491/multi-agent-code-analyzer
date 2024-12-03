from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
from ..network import AgentNetwork
from ..config import ConfigManager

app = FastAPI(title="Multi-Agent Code Analyzer API")
config_manager = ConfigManager()
network = AgentNetwork()

@app.post("/analyze/query")
async def process_query(query: str, context: Optional[Dict[str, Any]] = None):
    """Process a query using the agent network."""
    try:
        result = await network.process_query(query, context or {})
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/codebase")
async def analyze_codebase(
    path: str,
    background_tasks: BackgroundTasks,
    full_analysis: bool = False
):
    """Analyze an entire codebase."""
    try:
        # Start analysis in background for large codebases
        if full_analysis:
            task_id = f"analysis_{path.replace('/', '_')}"
            background_tasks.add_task(network.analyze_codebase, path)
            return {
                "status": "initiated",
                "task_id": task_id,
                "message": "Analysis started in background"
            }
        
        # Synchronous analysis for quick overview
        result = await network.analyze_codebase(path)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all available agents and their status."""
    return {
        "agents": [
            {
                "name": name,
                "config": config
            }
            for name, config in config_manager.agent_configs.items()
        ]
    }

@app.patch("/agents/{agent_name}/config")
async def update_agent_config(agent_name: str, updates: Dict[str, Any]):
    """Update configuration for a specific agent."""
    try:
        config_manager.update_agent_config(agent_name, **updates)
        return {
            "status": "success",
            "message": f"Configuration updated for agent {agent_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health status of the system."""
    return {
        "status": "healthy",
        "agents": sum(1 for config in config_manager.agent_configs.values() if config.enabled),
        "version": "0.1.0"
    }