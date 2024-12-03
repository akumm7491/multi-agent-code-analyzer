from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""

    # Service settings
    SERVICE_NAME: str = "multi-agent-code-analyzer"
    SERVICE_PORT: int = 8000
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    METRICS_ENABLED: bool = True
    LOG_LEVEL: str = "DEBUG"

    # Database settings
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "development_password"

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "development_password"

    # Milvus settings
    MILVUS_HOST: str = "standalone"
    MILVUS_PORT: int = 19530

    # GitHub settings
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_OWNER: Optional[str] = None
    GITHUB_REPO: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True
