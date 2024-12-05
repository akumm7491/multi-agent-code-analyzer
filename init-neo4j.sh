#!/bin/bash
set -e

echo "Initializing Neo4j..."

# Wait for Neo4j to be available
until cypher-shell -u neo4j -p neo4j \
    "CALL dbms.security.changePassword('${NEO4J_PASSWORD}');" > /dev/null 2>&1; do
    echo "Waiting for Neo4j to be ready..."
    sleep 2
done

echo "Neo4j initialized successfully!" 