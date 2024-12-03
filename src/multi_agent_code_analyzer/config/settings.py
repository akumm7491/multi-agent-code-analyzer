from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import os
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")
    REDIS_URI: str = Field(default="redis://localhost:6379")
    REDIS_PASSWORD: str = Field(default="your_redis_password")

    class Config:
        env_file = ".env"
        extra = "allow"


class VectorDBSettings(BaseSettings):
    MILVUS_HOST: str = Field(default="localhost")
    MILVUS_PORT: int = Field(default=19530)
    COLLECTION_NAME: str = Field(default="mcp_contexts")
    VECTOR_DIM: int = Field(default=384)

    class Config:
        env_file = ".env"
        extra = "allow"


class MinIOSettings(BaseSettings):
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin")
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000")
    MINIO_BUCKET_NAME: str = Field(default="mcp-storage")

    class Config:
        env_file = ".env"
        extra = "allow"


class ETCDSettings(BaseSettings):
    ETCD_HOST: str = Field(default="localhost")
    ETCD_PORT: int = Field(default=2379)
    ETCD_PREFIX: str = Field(default="/mcp/")

    class Config:
        env_file = ".env"
        extra = "allow"


class MCPSettings(BaseSettings):
    MCP_SERVER_URL: str = Field(default="http://localhost:8000")
    MCP_API_KEY: Optional[str] = Field(default=None)
    MCP_TIMEOUT: int = Field(default=30)
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    CONTEXT_RETENTION_DAYS: int = Field(default=90)
    FASTMCP_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    FASTMCP_STORE_TYPE: str = Field(default="milvus")

    class Config:
        env_file = ".env"
        extra = "allow"


class MonitoringSettings(BaseSettings):
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3000)
    METRICS_ENABLED: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        extra = "allow"


class ServiceSettings(BaseSettings):
    SERVICE_NAME: str = Field(default="multi-agent-code-analyzer")
    SERVICE_PORT: int = Field(default=8080)
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    DEBUG: bool = Field(default=False)
    CORS_ORIGINS: list[str] = Field(default=["*"])

    @validator("DEBUG", pre=True)
    def set_debug_based_on_env(cls, v: bool, values: Dict[str, Any]) -> bool:
        if "ENVIRONMENT" in values:
            return values["ENVIRONMENT"] == Environment.DEVELOPMENT
        return v

    class Config:
        env_file = ".env"
        extra = "allow"


class IntegrationSettings(BaseSettings):
    GITHUB_TOKEN: Optional[str] = Field(default=None)
    GITHUB_OWNER: Optional[str] = Field(default=None)
    GITHUB_REPO: Optional[str] = Field(default=None)
    JIRA_DOMAIN: Optional[str] = Field(default=None)
    JIRA_EMAIL: Optional[str] = Field(default=None)
    JIRA_API_TOKEN: Optional[str] = Field(default=None)
    JIRA_PROJECT_KEY: Optional[str] = Field(default=None)

    class Config:
        env_file = ".env"
        extra = "allow"


class SecuritySettings(BaseSettings):
    JWT_SECRET_KEY: str = Field(default="your_jwt_secret_key")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)

    class Config:
        env_file = ".env"
        extra = "allow"


class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    minio: MinIOSettings = MinIOSettings()
    etcd: ETCDSettings = ETCDSettings()
    mcp: MCPSettings = MCPSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    service: ServiceSettings = ServiceSettings()
    integrations: IntegrationSettings = IntegrationSettings()
    security: SecuritySettings = SecuritySettings()

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def initialize_settings() -> None:
    """Initialize settings and validate environment"""
    settings = get_settings()

    # Validate environment variables
    required_vars = [
        "NEO4J_PASSWORD",
        "MCP_API_KEY",
        "REDIS_PASSWORD"
    ]

    if settings.service.ENVIRONMENT == Environment.PRODUCTION:
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables in production: {', '.join(missing)}")

        # Additional production requirements
        if not settings.security.JWT_SECRET_KEY or settings.security.JWT_SECRET_KEY == "your_jwt_secret_key":
            raise ValueError("Production requires a secure JWT_SECRET_KEY")

    # Set up logging based on environment
    import logging
    logging.basicConfig(
        level=getattr(logging, settings.monitoring.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Additional environment-specific initialization
    if settings.service.ENVIRONMENT == Environment.DEVELOPMENT:
        logging.info("Running in development mode")
    elif settings.service.ENVIRONMENT == Environment.TESTING:
        logging.info("Running in testing mode")
    elif settings.service.ENVIRONMENT == Environment.PRODUCTION:
        logging.info("Running in production mode")

        # Additional production checks
        if settings.service.DEBUG:
            raise ValueError("Debug mode should not be enabled in production")

        if "*" in settings.service.CORS_ORIGINS:
            raise ValueError("Wildcard CORS origin not allowed in production")


def get_environment_settings() -> Dict[str, Any]:
    """Get environment-specific settings"""
    settings = get_settings()

    return {
        "environment": settings.service.ENVIRONMENT,
        "debug": settings.service.DEBUG,
        "database_uri": settings.database.NEO4J_URI,
        "mcp_server": settings.mcp.MCP_SERVER_URL,
        "metrics_enabled": settings.monitoring.METRICS_ENABLED,
        "service_port": settings.service.SERVICE_PORT,
        "vector_db": {
            "host": settings.vector_db.MILVUS_HOST,
            "port": settings.vector_db.MILVUS_PORT
        },
        "storage": {
            "minio_endpoint": settings.minio.MINIO_ENDPOINT,
            "etcd_host": settings.etcd.ETCD_HOST
        }
    }
