from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import structlog
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import neo4j
import redis
from pymilvus import connections

# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger()

# Configure metrics
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', [
                   'method', 'endpoint', 'status'])
LATENCY = Histogram('http_request_duration_seconds',
                    'HTTP request latency', ['method', 'endpoint'])

# Initialize FastAPI app
app = FastAPI(
    title="Code Analyzer",
    description="A service for analyzing code repositories using DDD and Event-Driven principles",
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

# Initialize OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


class AnalysisRequest(BaseModel):
    repository_url: str
    analysis_type: str = "full"
    include_dependencies: bool = True
    branch: Optional[str] = "main"


class DatabaseConnections:
    def __init__(self):
        self.neo4j_driver = None
        self.redis_client = None
        self.milvus_connection = None

    async def connect(self):
        # Neo4j connection
        try:
            self.neo4j_driver = neo4j.GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise

        # Redis connection
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URI"),
                password=os.getenv("REDIS_PASSWORD")
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

        # Milvus connection
        try:
            connections.connect(
                alias="default",
                host=os.getenv("MILVUS_HOST"),
                port=int(os.getenv("MILVUS_PORT"))
            )
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error("Failed to connect to Milvus", error=str(e))
            raise

    async def disconnect(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
        if self.redis_client:
            self.redis_client.close()
        connections.disconnect("default")


db = DatabaseConnections()


@app.on_event("startup")
async def startup_event():
    await db.connect()
    logger.info("Application started")


@app.on_event("shutdown")
async def shutdown_event():
    await db.disconnect()
    logger.info("Application shutdown")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path

    with LATENCY.labels(method=method, endpoint=path).time():
        response = await call_next(request)

    REQUESTS.labels(
        method=method,
        endpoint=path,
        status=response.status_code
    ).inc()

    return response


@app.get("/health")
async def health_check():
    try:
        # Check database connections
        assert db.neo4j_driver is not None
        assert db.redis_client is not None
        assert db.redis_client.ping()

        return {
            "status": "healthy",
            "services": {
                "neo4j": "connected",
                "redis": "connected",
                "milvus": "connected"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/api/analyze")
async def analyze_repository(request: AnalysisRequest):
    try:
        logger.info(
            "Starting repository analysis",
            repository=request.repository_url,
            analysis_type=request.analysis_type
        )

        # TODO: Implement actual analysis logic

        return {
            "status": "success",
            "message": f"Analysis started for {request.repository_url}",
            "type": request.analysis_type,
            "include_dependencies": request.include_dependencies,
            "branch": request.branch
        }
    except Exception as e:
        logger.error(
            "Analysis failed",
            repository=request.repository_url,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
