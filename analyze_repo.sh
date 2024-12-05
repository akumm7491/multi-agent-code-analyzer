#!/bin/bash
set -e

# Clone the repository
REPO_URL="https://github.com/akumm7491/shared-ddd-ed-microservices-layer"
REPO_PATH="./repo"

echo "Cloning repository..."
if [ ! -d "$REPO_PATH" ]; then
    git clone "$REPO_URL" "$REPO_PATH"
else
    echo "Repository already exists, pulling latest changes..."
    cd "$REPO_PATH"
    git pull
    cd ..
fi

# Export the repo path for docker-compose
export REPO_PATH="$PWD/repo"

# Start the services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Start the analysis
echo "Starting analysis..."
curl -X POST "http://localhost:8080/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"repo_path\": \"/app/repo\", \"analysis_type\": \"full\"}"

echo "Analysis started. You can check the progress at http://localhost:8080/status" 