from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from ..repository.connector import RepositoryConnector
from ..network import AgentNetwork

app = FastAPI(
    title="Multi-Agent Code Analyzer API",
    description="API for analyzing and understanding complex codebases using multiple specialized AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/repository/connect", response_model=Dict[str, Any])
async def connect_repository(request: RepositoryRequest):
    """
    Connect to a git repository and initialize analysis capabilities.
    """
    try:
        connector = RepositoryConnector(request.repo_url)
        success = await connector.initialize()
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to connect to repository")
            
        structure = await connector.get_file_structure()
        return {
            "status": "success",
            "repository": request.repo_url,
            "structure": structure
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze code based on a specific query or request.
    """
    try:
        network = AgentNetwork()
        result = await network.process_query(request.query, request.context or {})
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/repository/files/{path:path}", response_model=Dict[str, Any])
async def get_file_content(path: str):
    """
    Get content and metadata for a specific file in the repository.
    """
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