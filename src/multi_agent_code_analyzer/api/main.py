from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from ..tools.manager import ToolManager
from ..config.settings import get_settings

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

# Initialize tool manager with configurations
settings = get_settings()
tool_manager = ToolManager()

# Configure GitHub tool
if settings.integrations.GITHUB_TOKEN:
    tool_manager.configure_tool("GitHubTool", {
        "token": settings.integrations.GITHUB_TOKEN,
        "owner": settings.integrations.GITHUB_OWNER,
        "repo": settings.integrations.GITHUB_REPO
    })

# Configure JIRA tool
if settings.integrations.JIRA_API_TOKEN:
    tool_manager.configure_tool("JiraTool", {
        "domain": settings.integrations.JIRA_DOMAIN,
        "email": settings.integrations.JIRA_EMAIL,
        "token": settings.integrations.JIRA_API_TOKEN,
        "project_key": settings.integrations.JIRA_PROJECT_KEY
    })

# Make tool manager available to routes
app.state.tool_manager = tool_manager


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy"})


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Include routes
app.include_router(router, prefix="/api/v1")
