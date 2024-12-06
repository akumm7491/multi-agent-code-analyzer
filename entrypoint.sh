#!/bin/bash
set -e

# Function to wait for a service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local max_retries=30
    local retry_count=0
    
    echo "Waiting for $service to be ready..."
    while ! nc -z "$host" "$port"; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "Error: $service not ready after $max_retries attempts"
            exit 1
        fi
        echo "Waiting for $service... (attempt $retry_count/$max_retries)"
        sleep 2
    done
    echo "$service is ready!"
}

# Function to check Neo4j connection
check_neo4j() {
    local max_retries=30
    local retry_count=0
    local neo4j_host=$(echo "$NEO4J_URI" | sed 's|.*://\([^:]*\).*|\1|')
    
    echo "Checking Neo4j connection..."
    while ! curl -s "http://$neo4j_host:7474/browser/" > /dev/null; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "Error: Neo4j not responding after $max_retries attempts"
            exit 1
        fi
        echo "Waiting for Neo4j... (attempt $retry_count/$max_retries)"
        sleep 2
    done
    echo "Neo4j is responding!"
}

# Function to check Milvus connection
check_milvus() {
    local max_retries=30
    local retry_count=0
    
    echo "Checking Milvus connection..."
    while ! curl -s "http://$MILVUS_HOST:9091/healthz" > /dev/null; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "Error: Milvus not healthy after $max_retries attempts"
            exit 1
        fi
        echo "Waiting for Milvus... (attempt $retry_count/$max_retries)"
        sleep 2
    done
    echo "Milvus is healthy!"
}

echo "Starting service initialization..."

# Wait for Redis
wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"

# Wait for Neo4j
if [ -n "$NEO4J_URI" ]; then
    NEO4J_HOST=$(echo "$NEO4J_URI" | sed 's|.*://\([^:]*\).*|\1|')
    wait_for_service "$NEO4J_HOST" 7687 "Neo4j"
    check_neo4j
fi

# Wait for Milvus
if [ -n "$MILVUS_HOST" ]; then
    wait_for_service "$MILVUS_HOST" "$MILVUS_PORT" "Milvus"
    check_milvus
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