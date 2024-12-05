#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=$3
    local attempt=1
    local delay=5

    echo -e "${YELLOW}Checking ${service} health...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ ${service} is healthy${NC}"
            return 0
        fi
        echo -e "${YELLOW}Attempt $attempt/$max_attempts: ${service} not ready yet...${NC}"
        attempt=$((attempt + 1))
        sleep $delay
    done
    echo -e "${RED}✗ ${service} health check failed after $max_attempts attempts${NC}"
    return 1
}

# Function to check docker container status
check_container_status() {
    local container=$1
    local status=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)
    
    if [ "$status" != "running" ]; then
        echo -e "${RED}✗ Container $container is not running (status: $status)${NC}"
        echo "Container logs:"
        docker logs "$container"
        return 1
    fi
    return 0
}

# Check all services
services=(
    "FastMCP|http://localhost:8000/health"
    "Message Bus|http://localhost:8080/health"
    "Agent Manager|http://localhost:8081/health"
    "Neo4j|http://localhost:7474"
    "Milvus|http://localhost:19530/health"
    "MinIO|http://localhost:9000/minio/health/live"
    "Prometheus|http://localhost:9090/-/healthy"
    "Grafana|http://localhost:3000/api/health"
)

# Maximum number of attempts per service
max_attempts=12
failed=0

echo -e "${YELLOW}Starting service health checks...${NC}"

# First check if all containers are running
for container in $(docker-compose ps -q); do
    container_name=$(docker inspect -f '{{.Name}}' "$container" | sed 's/\///')
    if ! check_container_status "$container_name"; then
        failed=1
    fi
done

# Then check service endpoints
for service in "${services[@]}"; do
    IFS="|" read -r name url <<< "$service"
    if ! check_service "$name" "$url" "$max_attempts"; then
        failed=1
    fi
done

if [ $failed -eq 1 ]; then
    echo -e "${RED}Some services failed health check${NC}"
    exit 1
else
    echo -e "${GREEN}All services are healthy!${NC}"
    exit 0
fi 