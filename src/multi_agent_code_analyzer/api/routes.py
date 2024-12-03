from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from ..repository.connector import RepositoryConnector
from ..network import AgentNetwork

router = APIRouter()

# Models for request/response


class RepositoryRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = "main"


class AnalysisRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    task_id: Optional[str] = None


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mcp",
        "version": "1.0.0"
    }


@router.post("/repository/connect", response_model=Dict[str, Any])
async def connect_repository(request: RepositoryRequest):
    """Connect to a git repository and initialize analysis capabilities."""
    try:
        connector = RepositoryConnector(request.repo_url)
        success = await connector.initialize()

        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to connect to repository")

        structure = await connector.get_file_structure()
        return {
            "status": "success",
            "repository": request.repo_url,
            "structure": structure
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest, background_tasks: BackgroundTasks, req: Request):
    """Analyze code based on a specific query or request."""
    try:
        # Get tool manager from app state
        tool_manager = req.app.state.tool_manager

        # If context specifies a tool, use it directly
        if request.context and "tool" in request.context:
            tool_name = request.context["tool"].title() + "Tool"
            if tool_name == "GithubTool":
                tool_name = "GitHubTool"
            context = request.context.copy()
            action = context.pop("action", "execute")

            # Execute the tool
            result = await tool_manager.execute_tool(
                tool_name=tool_name,
                action=action,
                **context
            )

            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool execution failed: {result.error}"
                )

            return {
                "status": "success",
                "result": result.__dict__
            }

        # If no specific tool is requested, use the agent network
        network = AgentNetwork()
        result = await network.process_query(request.query, request.context or {})

        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repository/files/{path:path}", response_model=Dict[str, Any])
async def get_file_content(path: str):
    """Get content and metadata for a specific file in the repository."""
    try:
        connector = RepositoryConnector(".")  # Use current connection
        content = await connector.get_file_content(path)
        history = await connector.get_file_history(path)

        if content is None:
            raise HTTPException(status_code=404, detail="File not found")

        return {
            "path": path,
            "content": content,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/codebase")
async def analyze_codebase(
    path: str,
    background_tasks: BackgroundTasks,
    full_analysis: bool = False
):
    """Analyze an entire codebase."""
    try:
        network = AgentNetwork()
        if full_analysis:
            task_id = f"analysis_{path.replace('/', '_')}"
            background_tasks.add_task(network.analyze_codebase, path)
            return {
                "status": "initiated",
                "task_id": task_id,
                "message": "Analysis started in background"
            }

        result = await network.analyze_codebase(path)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
