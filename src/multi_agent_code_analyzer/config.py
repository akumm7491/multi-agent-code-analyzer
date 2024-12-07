from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    # Service settings
    SERVICE_NAME: str = "multi-agent-code-analyzer"
    SERVICE_PORT: int = 8080
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    class Config:
        env_file = ".env"

settings = Settings()
