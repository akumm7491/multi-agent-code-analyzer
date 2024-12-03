#!/bin/bash
set -e

# Function to check if a service is ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    
    echo "Waiting for $service to be ready..."
    while ! nc -z $host $port; do
        echo "$service is not ready yet..."
        sleep 2
    done
    echo "$service is ready!"
}

echo "=== Checking Dependencies ==="
# Wait for Neo4j
wait_for_service neo4j 7687 "Neo4j"
# Wait for Redis
wait_for_service redis 6379 "Redis"
# Wait for Milvus
wait_for_service standalone 19530 "Milvus"

echo "=== Environment Validation ==="
# Required environment variables
required_vars=(
    "PYTHONPATH"
    "SERVICE_NAME"
    "SERVICE_PORT"
    "ENVIRONMENT"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

echo "=== Environment Setup ==="
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la
echo "Python path: $PYTHONPATH"
echo "Python version: $(python --version)"

echo "=== Python Package Structure ==="
echo "Contents of src directory:"
ls -la src/
echo "Contents of multi_agent_code_analyzer:"
ls -la src/multi_agent_code_analyzer/

echo "=== Python Package Test ==="
echo "Attempting to import package..."
python -c "import multi_agent_code_analyzer; print('Package imported successfully')"
python -c "from multi_agent_code_analyzer.api.main import app; print('API imported successfully')"

echo "=== Starting Application ==="
echo "Starting FastAPI application with debug logging..."
exec python -m uvicorn multi_agent_code_analyzer.api.main:app --host 0.0.0.0 --port 8000 --log-level debug 