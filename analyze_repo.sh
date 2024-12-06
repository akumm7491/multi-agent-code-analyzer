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

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting services..."
    docker-compose up -d
    echo "Waiting for services to be ready (60s)..."
    sleep 60
else
    echo "Services are already running"
fi

# Check app health
echo "Checking application health..."
MAX_RETRIES=10
RETRY_COUNT=0

while true; do
    if curl -s -f http://localhost:8000/health > /dev/null; then
        echo "Application is healthy!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Application not healthy after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "Waiting for application to be healthy... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
done

# Start the analysis
echo "Starting analysis..."
ANALYSIS_RESPONSE=$(curl -s -X POST "http://localhost:8000/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"repo_path\": \"/app/repo\", \"analysis_type\": \"full\"}")

echo "Analysis started. Response: $ANALYSIS_RESPONSE"
echo "You can check the progress at http://localhost:8000/status"

# Monitor analysis status
echo "Monitoring analysis progress..."
while true; do
    STATUS=$(curl -s "http://localhost:8000/status?repo_path=/app/repo")
    echo "Current status: $STATUS"
    if [[ $STATUS == *"completed"* ]] || [[ $STATUS == *"failed"* ]]; then
        break
    fi
    sleep 10
done 