#!/bin/bash

set -e

# Function to check environment variables
check_environment() {
    local required_vars=("$@")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo "Error: Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
}

# Function to wait for Neo4j using Python
wait_for_neo4j() {
    local host="$1"
    local port="$2"
    local max_attempts="${3:-30}"
    local wait_time="${4:-2}"
    
    echo "Waiting for Neo4j at $host:$port..."
    for i in $(seq 1 $max_attempts); do
        echo "Attempt $i - Testing Neo4j connection..."
        python3 -c "
from neo4j import GraphDatabase
import sys
import traceback

try:
    uri = 'bolt://$host:$port'
    print(f'Debug: Connecting to Neo4j at {uri}...')
    driver = GraphDatabase.driver(uri, connection_acquisition_timeout=5)
    print('Debug: Driver created, testing session...')
    with driver.session(database='neo4j') as session:
        print('Debug: Session created, running test query...')
        result = session.run('MATCH (n) RETURN count(n) as count')
        value = result.single()['count']
        print(f'Debug: Query successful, count: {value}')
    driver.close()
    sys.exit(0)
except Exception as e:
    print(f'Debug: Connection failed with error: {str(e)}')
    print('Debug: Full traceback:')
    traceback.print_exc()
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            echo "Neo4j is ready!"
            return 0
        fi
        sleep $wait_time
    done
    echo "Timeout waiting for Neo4j"
    return 1
}

# Function to wait for Redis
wait_for_redis() {
    local host="$1"
    local port="$2"
    local max_attempts="${3:-30}"
    local wait_time="${4:-2}"
    
    echo "Waiting for Redis at $host:$port..."
    for i in $(seq 1 $max_attempts); do
        if nc -z -w1 "$host" "$port" >/dev/null 2>&1; then
            echo "Redis is ready!"
            return 0
        fi
        echo "Waiting for Redis... $i/$max_attempts"
        sleep $wait_time
    done
    echo "Timeout waiting for Redis"
    return 1
}

# Function to setup metrics
setup_metrics() {
    if [ "$METRICS_ENABLED" = "true" ]; then
        echo "Setting up metrics..."
        mkdir -p /app/metrics
        touch /app/metrics/metrics.prom
        export PROMETHEUS_MULTIPROC_DIR=/app/metrics
    fi
}

# Function to setup directories and logging
setup_directories() {
    echo "Setting up directories and logging..."
    mkdir -p /app/logs /app/data /app/cache /app/temp
    export LOG_FILE="/app/logs/mcp.log"
    touch "$LOG_FILE"
}

# Main execution
echo "Starting service initialization..."

# Setup directories and logging
setup_directories

# Check required environment variables
check_environment "SERVICE_PORT" "NEO4J_URI" "REDIS_HOST" "REDIS_PORT"

echo "Initializing mcp service..."

# Extract Neo4j host and port from URI
NEO4J_HOST=$(echo "$NEO4J_URI" | sed -n 's/.*\/\/\([^:]*\).*/\1/p')
NEO4J_PORT=$(echo "$NEO4J_URI" | sed -n 's/.*:\([0-9]*\).*/\1/p')
if [ -z "$NEO4J_PORT" ]; then
    NEO4J_PORT=7687  # Default Neo4j Bolt port
fi

echo "Neo4j connection details:"
echo "Host: $NEO4J_HOST"
echo "Port: $NEO4J_PORT"
echo "URI: $NEO4J_URI"

# Wait for required services
if ! wait_for_neo4j "$NEO4J_HOST" "$NEO4J_PORT"; then
    echo "ERROR: Failed to establish Neo4j connection"
    exit 1
fi

if ! wait_for_redis "$REDIS_HOST" "$REDIS_PORT"; then
    echo "ERROR: Failed to establish Redis connection"
    exit 1
fi

# Setup metrics if enabled
setup_metrics

# Start the MCP service
echo "Starting MCP service..."
exec python -m multi_agent_code_analyzer.mcp.server \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --max-workers "${MCP_WORKERS:-10}" \
    --max-concurrent-rpcs "${MCP_CONCURRENCY:-1000}"