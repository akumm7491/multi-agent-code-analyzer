#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Verifying Docker setup...${NC}"

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed!${NC}"
    exit 1
fi

# Check Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed!${NC}"
    exit 1
fi

# Check if required ports are available
required_ports=(8000 8080 8081 8082 6379 7474 7687 9090 3000 16686)
for port in "${required_ports[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Port $port is already in use!${NC}"
        exit 1
    fi
done

# Check if required directories exist
required_dirs=("data" "volumes" "logs")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}Creating directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}.env file is missing!${NC}"
    echo -e "${YELLOW}Creating from .env.example...${NC}"
    cp .env.example .env
fi

# Check if docker-compose.override.yml exists for development
if [ ! -f "docker-compose.override.yml" ]; then
    echo -e "${YELLOW}Creating docker-compose.override.yml for development...${NC}"
    cp docker-compose.override.yml.example docker-compose.override.yml
fi

# Verify docker-compose configuration
echo -e "${YELLOW}Verifying docker-compose configuration...${NC}"
docker-compose config --quiet

echo -e "${GREEN}Docker setup verification completed successfully!${NC}" 