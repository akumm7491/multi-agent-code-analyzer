FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    aiohttp \
    structlog \
    pydantic \
    sentence-transformers \
    neo4j \
    redis \
    pymilvus \
    python-dotenv \
    gitpython

# Copy source code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["uvicorn", "src.multi_agent_code_analyzer.core.analyzer:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 