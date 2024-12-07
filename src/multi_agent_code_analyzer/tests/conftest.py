import pytest
from fastapi.testclient import TestClient
from ..main import app, projects, agent_service

@pytest.fixture(autouse=True)
def clear_data():
    """Clear all data before each test"""
    projects.clear()
    agent_service.agents.clear()

@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)
