#!/bin/bash
set -e

# Function to wait for a service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    echo "Waiting for $service to be ready..."
    while ! nc -z "$host" "$port"; do
        echo "Waiting for $service..."
        sleep 1
    done
    echo "$service is ready!"
}

# Wait for Redis
wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"

# Wait for Neo4j if needed
if [ -n "$NEO4J_URI" ]; then
    NEO4J_HOST=$(echo "$NEO4J_URI" | sed 's|.*://\([^:]*\).*|\1|')
    wait_for_service "$NEO4J_HOST" 7687 "Neo4j"
fi

# Wait for Milvus if needed
if [ -n "$MILVUS_HOST" ]; then
    wait_for_service "$MILVUS_HOST" "$MILVUS_PORT" "Milvus"
fi

# Create necessary directories
mkdir -p /app/logs /app/data

# Run setup scripts
echo "Running setup scripts..."
python -m src.multi_agent_code_analyzer.core.setup
python -m src.multi_agent_code_analyzer.core.init_neo4j

# Start the application
echo "Starting the application..."
exec uvicorn src.multi_agent_code_analyzer.core.analyzer:app \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --workers 1 \
    --log-level info