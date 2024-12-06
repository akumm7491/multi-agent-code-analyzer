from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import structlog
import git
import os
import tempfile
from typing import Optional, List, Dict
import asyncio
from neo4j import GraphDatabase, AsyncGraphDatabase
from redis import Redis
from pymilvus import connections, Collection, utility
import ast
import glob
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from .domain_analysis import DomainAnalyzer
import hashlib

logger = structlog.get_logger()

app = FastAPI(
    title="DDD Code Analyzer",
    description="API for analyzing code repositories using Domain-Driven Design principles",
    version="1.0.0"
)

# Initialize Redis for status tracking
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", "your_secure_redis_password")
)

# Initialize Neo4j client
neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "your_secure_password")

neo4j_client = AsyncGraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password)
)

# Initialize Milvus
connections.connect(
    host=os.getenv("MILVUS_HOST", "standalone"),
    port=int(os.getenv("MILVUS_PORT", "19530"))
)

# Get GitHub token from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


class AnalysisRequest(BaseModel):
    repo_url: str
    analysis_type: str = "full"
    include_patterns: Optional[List[str]] = ["*.py", "*.java", "*.cs", "*.ts"]
    exclude_patterns: Optional[List[str]] = [
        "*test*", "*vendor*", "*node_modules*"]


class AnalysisStatus(BaseModel):
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


def generate_tracking_id(repo_url: str) -> str:
    """Generate a consistent tracking ID for a repository."""
    return f"analysis:{hashlib.sha256(repo_url.encode()).hexdigest()}"


async def clone_repository(repo_url: str, target_path: str) -> bool:
    """Clone a repository to a target path."""
    try:
        # Add token to URL if available
        if GITHUB_TOKEN and "github.com" in repo_url:
            # Extract the repo path after github.com
            repo_path = repo_url.split("github.com/")[-1]
            auth_url = f"https://{GITHUB_TOKEN}@github.com/{repo_path}"
        else:
            auth_url = repo_url

        logger.info(f"Cloning repository from {repo_url} to {target_path}")
        git.Repo.clone_from(auth_url, target_path)
        return True
    except Exception as e:
        logger.error(f"Failed to clone repository: {str(e)}")
        return False


async def _analyze_repository(repo_url: str, temp_dir: str, patterns: List[str]) -> Dict:
    """Analyze a repository directory."""
    try:
        # Initialize analyzer
        analyzer = DomainAnalyzer()
        analyzer.neo4j_client = neo4j_client  # Pass Neo4j client to analyzer

        # Run analysis
        result = await analyzer.analyze_repository(repo_url=repo_url, repo_path=temp_dir, patterns=patterns)
        return result
    except Exception as e:
        logger.error(f"Error in repository analysis: {str(e)}")
        raise


async def run_analysis(request: AnalysisRequest, tracking_id: str):
    """Run the analysis asynchronously and store results in Redis."""
    try:
        # Create temporary directory for repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repository
            if not await clone_repository(request.repo_url, temp_dir):
                raise Exception("Failed to clone repository")

            # Run analysis
            result = await _analyze_repository(request.repo_url, temp_dir, request.include_patterns)

            # Store result in Redis
            redis_client.hset(
                tracking_id,
                mapping={
                    "status": "completed",
                    "result": json.dumps(result)
                }
            )
            redis_client.expire(tracking_id, 3600)  # Expire after 1 hour

    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        redis_client.hset(
            tracking_id,
            mapping={
                "status": "failed",
                "error": str(e)
            }
        )
        redis_client.expire(tracking_id, 3600)


@app.post("/analyze")
async def analyze_repository(request: AnalysisRequest):
    """Start repository analysis."""
    try:
        # Generate tracking ID
        tracking_id = generate_tracking_id(request.repo_url)

        # Initialize status in Redis
        redis_client.hset(
            tracking_id,
            mapping={
                "status": "running",
                "repo": request.repo_url
            }
        )

        # Start analysis in background
        asyncio.create_task(run_analysis(request, tracking_id))

        return {
            "status": "Analysis started",
            "repo": request.repo_url,
            "tracking_id": tracking_id
        }

    except Exception as e:
        logger.error("Failed to start analysis", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{tracking_id}")
async def get_analysis_status(tracking_id: str) -> AnalysisStatus:
    """Get the status of an analysis."""
    try:
        # Get status from Redis
        status_data = redis_client.hgetall(tracking_id)

        if not status_data:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Convert bytes to strings
        status_data = {k.decode(): v.decode() for k, v in status_data.items()}

        if status_data["status"] == "completed":
            return AnalysisStatus(
                status="completed",
                result=json.loads(status_data["result"])
            )
        elif status_data["status"] == "failed":
            return AnalysisStatus(
                status="failed",
                error=status_data.get("error", "Unknown error")
            )
        else:
            return AnalysisStatus(status="running")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error checking analysis status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health of all system components."""
    try:
        # Check Redis
        redis_client.ping()

        # Check Neo4j
        async with neo4j_client.session() as session:
            await session.run("RETURN 1")

        # Check Milvus
        utility.get_server_version()

        return {"status": "healthy"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}
