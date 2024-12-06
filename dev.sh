#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function start_dev() {
    echo -e "${GREEN}Starting development environment...${NC}"
    docker-compose -f docker-compose.dev.yml up -d
    echo -e "${GREEN}Development environment started!${NC}"
    echo -e "API is available at http://localhost:8000"
    echo -e "Neo4j browser is available at http://localhost:7474"
    echo -e "Use 'docker-compose -f docker-compose.dev.yml logs -f app' to view app logs"
}

function stop_dev() {
    echo -e "${RED}Stopping development environment...${NC}"
    docker-compose -f docker-compose.dev.yml down
    echo -e "${RED}Development environment stopped!${NC}"
}

function restart_app() {
    echo -e "${GREEN}Restarting app service...${NC}"
    docker-compose -f docker-compose.dev.yml restart app
    echo -e "${GREEN}App service restarted!${NC}"
}

function view_logs() {
    echo -e "${GREEN}Viewing app logs...${NC}"
    docker-compose -f docker-compose.dev.yml logs -f app
}

case "$1" in
    "start")
        start_dev
        ;;
    "stop")
        stop_dev
        ;;
    "restart")
        restart_app
        ;;
    "logs")
        view_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs}"
        exit 1
        ;;
esac 