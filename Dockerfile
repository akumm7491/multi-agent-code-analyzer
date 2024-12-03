FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Create log directory
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Copy only the requirements and setup files first
COPY requirements.txt setup.py ./

# Install dependencies and the package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Copy the source code
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY entrypoint.sh .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=multi-agent-code-analyzer
ENV SERVICE_PORT=8000
ENV ENVIRONMENT=development
ENV DEBUG=true
ENV METRICS_ENABLED=true
ENV LOG_LEVEL=DEBUG

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Add a healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"] 