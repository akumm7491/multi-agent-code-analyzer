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

# Enhanced Neo4j connection testing
wait_for_neo4j() {
    local host="$1"
    local port="$2"
    local max_attempts="${NEO4J_CONNECTION_RETRY_COUNT:-30}"
    local retry_delay="${NEO4J_CONNECTION_RETRY_DELAY:-10}"
    
    echo "=== Neo4j Connection Debug Info ==="
    echo "Connection details:"
    echo "URI: $NEO4J_URI"
    echo "User: $NEO4J_USER"
    echo "Password hash: $(echo -n "$NEO4J_PASSWORD" | sha256sum)"
    echo "Max attempts: $max_attempts"
    echo "Retry delay: $retry_delay seconds"
    
    # Initial delay to let Neo4j fully initialize
    echo "Waiting 30 seconds for Neo4j to initialize..."
    sleep 30
    
    # DNS resolution check
    echo "=== DNS Resolution Check ==="
    if ! getent hosts neo4j; then
        echo "Failed to resolve neo4j hostname"
        return 1
    fi
    
    # Network connectivity check
    echo "=== Network Connectivity Check ==="
    if ! nc -zv neo4j 7687; then
        echo "Failed to connect to neo4j:7687"
        return 1
    fi
    
    # Test with Python driver
    for i in $(seq 1 $max_attempts); do
        echo "=== Neo4j Connection Test (attempt $i/$max_attempts) ==="
        python3 -c "
import os, sys, socket, logging, time
from neo4j import GraphDatabase

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('neo4j').setLevel(logging.DEBUG)

def test_connection(uri, user, password, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f'Connection attempt {attempt + 1}/{max_retries}')
            print(f'Connecting to {uri}...')
            
            driver = GraphDatabase.driver(
                uri, 
                auth=(user, password),
                max_connection_lifetime=5,
                connection_timeout=10
            )
            
            print('Driver created, verifying connectivity...')
            driver.verify_connectivity()
            
            # Test a simple query
            with driver.session() as session:
                result = session.run('RETURN 1 as num')
                print(f'Test query result: {result.single()[0]}')
            
            print('Success!')
            driver.close()
            return True
        except Exception as e:
            print(f'Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}')
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return False

try:
    uri = os.environ['NEO4J_URI']
    user = os.environ['NEO4J_USER']
    password = os.environ['NEO4J_PASSWORD']
    
    if test_connection(uri, user, password):
        sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f'Error: {type(e).__name__}: {str(e)}')
    print('Full error details:', str(e), file=sys.stderr)
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            echo "Neo4j connection established"
            return 0
        fi
        echo "Waiting $retry_delay seconds before next attempt..."
        sleep $retry_delay
    done
    
    echo "Timeout waiting for Neo4j after $max_attempts attempts"
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
    if [ "$PROMETHEUS_MULTIPROC_DIR" = "" ]; then
        mkdir -p /app/metrics
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

# Check required environment variables
check_environment NEO4J_URI NEO4J_USER NEO4J_PASSWORD REDIS_HOST REDIS_PORT

echo "Checking system resources..."
echo "Initializing mcp service..."

# Setup directories and logging
setup_directories

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

# Wait for Neo4j
echo "Environment variables:"
env | grep -E 'NEO4J|REDIS'

echo "Waiting for Neo4j to be ready..."
if ! wait_for_neo4j "$NEO4J_HOST" "$NEO4J_PORT"; then
    echo "Failed to connect to Neo4j"
    exit 1
fi

# Wait for Redis
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
