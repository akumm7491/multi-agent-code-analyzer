#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FastMCP
pip install fastmcp

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOL
# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repo_name

# JIRA Configuration
JIRA_DOMAIN=your_domain.atlassian.net
JIRA_EMAIL=your_email
JIRA_API_TOKEN=your_jira_token
JIRA_PROJECT_KEY=your_project_key

# MCP Configuration
MCP_SERVER_URL=http://localhost:8000
FASTMCP_EMBEDDING_MODEL=all-MiniLM-L6-v2
FASTMCP_STORE_TYPE=memory
EOL
    echo "Created .env file. Please update it with your credentials."
fi

# Create test directory if it doesn't exist
mkdir -p tests

# Start MCP server in the background
echo "Starting MCP server..."
python -m uvicorn fastmcp.server:app --host 0.0.0.0 --port 8000 &
MCP_PID=$!

# Wait for server to start
sleep 5

# Run tests
echo "Running tests..."
pytest tests/test_system.py -v

# Cleanup
kill $MCP_PID 