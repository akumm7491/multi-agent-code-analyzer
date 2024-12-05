"""Settings module for the application."""

from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Service settings
    SERVICE_NAME: str = "multi-agent-code-analyzer"
    SERVICE_PORT: int = 8080
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Database settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "secure_password_123")

    # Redis settings
    REDIS_URL: str = os.getenv(
        "REDIS_URL", "redis://:secure_redis_123@redis:6379")

    # FastMCP settings
    FASTMCP_URL: str = os.getenv("FASTMCP_URL", "http://fastmcp:8000")
    FASTMCP_STORE_TYPE: str = os.getenv("FASTMCP_STORE_TYPE", "milvus")
    FASTMCP_EMBEDDING_MODEL: str = os.getenv(
        "FASTMCP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CONTEXT_RETENTION_DAYS: int = int(
        os.getenv("CONTEXT_RETENTION_DAYS", "90"))
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "384"))

    # Message Bus settings
    MESSAGE_BUS_TYPE: str = os.getenv("MESSAGE_BUS_TYPE", "redis")
    MESSAGE_BUS_TOPICS: List[str] = [
        "agent.tasks", "agent.results", "system.events"]
    MESSAGE_BUS_MAX_RETRY: int = int(os.getenv("MESSAGE_BUS_MAX_RETRY", "3"))
    MESSAGE_BUS_RETRY_DELAY: int = int(
        os.getenv("MESSAGE_BUS_RETRY_DELAY", "1"))

    # Vector Database settings
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "milvus")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "mcp_contexts")

    # MinIO settings
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "secure_minio_123")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "mcp-storage")

    # Agent settings
    MAX_AGENTS: int = int(os.getenv("MAX_AGENTS", "10"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    AGENT_TYPES: List[str] = [
        "code_analyzer",
        "pattern_detector",
        "security_scanner",
        "dependency_analyzer"
    ]

    # Monitoring settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    GRAFANA_PORT: int = int(os.getenv("GRAFANA_PORT", "3000"))
    METRICS_ENABLED: bool = os.getenv(
        "METRICS_ENABLED", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Security settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "secure_jwt_key_123")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
