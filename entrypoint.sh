#!/bin/bash
set -e

# Function to wait for a service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local timeout="${4:-30}"

    echo "Waiting for $service to be ready..."
    for i in $(seq 1 $timeout); do
        if (echo > /dev/tcp/$host/$port) >/dev/null 2>&1; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... $i/$timeout"
        sleep 1
    done
    echo "Timeout waiting for $service"
    return 1
}

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

# Function to initialize service
initialize_service() {
    local service_type="$1"
    
    echo "Initializing $service_type service..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/cache /app/temp
    
    # Set up logging
    export LOG_FILE="/app/logs/${service_type}.log"
    touch "$LOG_FILE"
    
    # Service-specific initialization
    case "$service_type" in
        "api")
            check_environment "SERVICE_PORT" "MCP_HOST" "MCP_PORT" "NEO4J_URI" "REDIS_HOST"
            wait_for_service "$MCP_HOST" "$MCP_PORT" "MCP"
            wait_for_service "${NEO4J_URI#*//}" "7687" "Neo4j"
            wait_for_service "$REDIS_HOST" "6379" "Redis"
            ;;
        "mcp")
            check_environment "SERVICE_PORT" "NEO4J_URI" "REDIS_HOST"
            wait_for_service "${NEO4J_URI#*//}" "7687" "Neo4j"
            wait_for_service "$REDIS_HOST" "6379" "Redis"
            ;;
        "agent_manager")
            check_environment "MCP_HOST" "MCP_PORT" "NEO4J_URI" "REDIS_HOST"
            wait_for_service "$MCP_HOST" "$MCP_PORT" "MCP"
            wait_for_service "${NEO4J_URI#*//}" "7687" "Neo4j"
            wait_for_service "$REDIS_HOST" "6379" "Redis"
            # Wait for Docker socket
            if [ ! -S "/var/run/docker.sock" ]; then
                echo "Error: Docker socket not found"
                exit 1
            fi
            ;;
        "dynamic_agent")
            check_environment "MCP_HOST" "MCP_PORT" "NEO4J_URI" "AGENT_TYPE" "REDIS_HOST"
            wait_for_service "$MCP_HOST" "$MCP_PORT" "MCP"
            wait_for_service "${NEO4J_URI#*//}" "7687" "Neo4j"
            wait_for_service "$REDIS_HOST" "6379" "Redis"
            ;;
        "code_analyzer"|"developer"|"integration"|"orchestrator")
            # Keep direct agent execution for development/testing
            check_environment "MCP_HOST" "MCP_PORT" "NEO4J_URI" "REDIS_HOST"
            wait_for_service "$MCP_HOST" "$MCP_PORT" "MCP"
            wait_for_service "${NEO4J_URI#*//}" "7687" "Neo4j"
            wait_for_service "$REDIS_HOST" "6379" "Redis"
            ;;
        *)
            echo "Unknown service type: $service_type"
            exit 1
            ;;
    esac
}

# Function to run database operations
run_database_ops() {
    # Initialize database if needed
    if [ "$INITIALIZE_DB" = "true" ]; then
        echo "Initializing database..."
        python -m multi_agent_code_analyzer.db.init
    fi

    # Run migrations if needed
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "Running database migrations..."
        python -m multi_agent_code_analyzer.db.migrate
    fi
}

# Function to setup metrics
setup_metrics() {
    if [ "$METRICS_ENABLED" = "true" ]; then
        echo "Setting up metrics..."
        # Create metrics directory if it doesn't exist
        mkdir -p /app/metrics
        # Initialize metrics file
        touch /app/metrics/metrics.prom
        # Setup Prometheus client
        export PROMETHEUS_MULTIPROC_DIR=/app/metrics
        # Initialize metrics collectors
        python -m multi_agent_code_analyzer.metrics.init
    fi
}

# Function to setup caching
setup_caching() {
    echo "Setting up caching..."
    mkdir -p /app/cache
    # Initialize cache
    python -m multi_agent_code_analyzer.cache.init
}

# Function to check system resources
check_system_resources() {
    echo "Checking system resources..."
    # Check available memory
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$available_memory" -lt 512 ]; then
        echo "Warning: Low memory available: ${available_memory}MB"
    fi
    
    # Check disk space
    local available_disk=$(df -m /app | awk 'NR==2 {print $4}')
    if [ "$available_disk" -lt 1024 ]; then
        echo "Warning: Low disk space available: ${available_disk}MB"
    fi
}

# Main execution
echo "Starting service initialization..."

# Check if service type is set
if [ -z "$SERVICE_TYPE" ]; then
    echo "Error: SERVICE_TYPE environment variable is not set"
    exit 1
fi

# Check system resources
check_system_resources

# Initialize service
initialize_service "$SERVICE_TYPE"

# Run database operations if needed
run_database_ops

# Setup metrics if enabled
setup_metrics

# Setup caching
setup_caching

# Start the application based on service type
echo "Starting $SERVICE_TYPE service..."
case "$SERVICE_TYPE" in
    "api")
        exec python -m uvicorn multi_agent_code_analyzer.server:app \
            --host 0.0.0.0 \
            --port "$SERVICE_PORT" \
            --log-level "${LOG_LEVEL:-info}" \
            --workers "${API_WORKERS:-4}" \
            --limit-concurrency "${API_CONCURRENCY:-1000}" \
            --timeout-keep-alive "${KEEP_ALIVE:-120}"
        ;;
    "mcp")
        exec python -m multi_agent_code_analyzer.mcp.server \
            --host 0.0.0.0 \
            --port "$SERVICE_PORT" \
            --max-workers "${MCP_WORKERS:-10}" \
            --max-concurrent-rpcs "${MCP_CONCURRENCY:-1000}"
        ;;
    "agent_manager")
        exec python -m multi_agent_code_analyzer.agents.agent_manager \
            --log-level "${LOG_LEVEL:-info}" \
            --metrics-port "${METRICS_PORT:-8000}"
        ;;
    "dynamic_agent")
        case "$AGENT_TYPE" in
            "code_analyzer")
                exec python -m multi_agent_code_analyzer.agents.code_analyzer \
                    --log-level "${LOG_LEVEL:-info}" \
                    --metrics-port "${METRICS_PORT:-8000}"
                ;;
            "developer")
                exec python -m multi_agent_code_analyzer.agents.developer \
                    --log-level "${LOG_LEVEL:-info}" \
                    --metrics-port "${METRICS_PORT:-8000}"
                ;;
            "integration")
                exec python -m multi_agent_code_analyzer.agents.integration \
                    --log-level "${LOG_LEVEL:-info}" \
                    --metrics-port "${METRICS_PORT:-8000}"
                ;;
            "orchestrator")
                exec python -m multi_agent_code_analyzer.agents.orchestrator \
                    --log-level "${LOG_LEVEL:-info}" \
                    --metrics-port "${METRICS_PORT:-8000}"
                ;;
            *)
                echo "Unknown agent type: $AGENT_TYPE"
                exit 1
                ;;
        esac
        ;;
    # Keep direct agent execution for development/testing
    "code_analyzer")
        exec python -m multi_agent_code_analyzer.agents.code_analyzer \
            --log-level "${LOG_LEVEL:-info}" \
            --metrics-port "${METRICS_PORT:-8000}"
        ;;
    "developer")
        exec python -m multi_agent_code_analyzer.agents.developer \
            --log-level "${LOG_LEVEL:-info}" \
            --metrics-port "${METRICS_PORT:-8000}"
        ;;
    "integration")
        exec python -m multi_agent_code_analyzer.agents.integration \
            --log-level "${LOG_LEVEL:-info}" \
            --metrics-port "${METRICS_PORT:-8000}"
        ;;
    "orchestrator")
        exec python -m multi_agent_code_analyzer.agents.orchestrator \
            --log-level "${LOG_LEVEL:-info}" \
            --metrics-port "${METRICS_PORT:-8000}"
        ;;
    *)
        echo "Unknown service type: $SERVICE_TYPE"
        exit 1
        ;;
esac