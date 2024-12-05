#!/bin/bash

# Exit on error
set -e

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    docker-compose down
    deactivate 2>/dev/null || true
}

# Trap cleanup function
trap cleanup ERR EXIT

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

echo "Setting up Multi-Agent Code Analyzer..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    check_status "Virtual environment creation"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
check_status "Virtual environment activation"

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .
check_status "Package installation"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example"
    cp .env.example .env
    check_status "Environment file creation"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs/{neo4j,mcp}
mkdir -p prometheus
mkdir -p grafana/{dashboards,provisioning}
mkdir -p scripts
check_status "Directory creation"

# Initialize the system
echo "Initializing the system..."
python -m src.multi_agent_code_analyzer.scripts.init_system
check_status "System initialization"

# Start the services using docker-compose
echo "Starting services..."
docker-compose down --remove-orphans
docker-compose up -d
check_status "Service startup"

# Wait for services to be ready
echo "Checking service health..."
./scripts/check_services.sh
check_status "Service health check"

echo "System is ready!"
echo "FastMCP UI: http://localhost:8000"
echo "Neo4j Browser: http://localhost:7474"
echo "Milvus Console: http://localhost:19530"
echo "MinIO Console: http://localhost:9001"
echo "Grafana: http://localhost:3000" 