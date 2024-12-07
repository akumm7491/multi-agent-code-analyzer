import pytest
from fastapi.testclient import TestClient
from ..models.agent import AgentType, AgentState

@pytest.fixture
def client():
    from ..main import app
    return TestClient(app)

def test_create_agent_success(client: TestClient):
    """Test successful agent creation"""
    agent_data = {
        "name": "Test Agent",
        "type": AgentType.CODE_ANALYZER,
        "description": "A test agent"
    }
    response = client.post("/agents", json=agent_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == agent_data["name"]
    assert data["type"] == agent_data["type"]
    assert data["description"] == agent_data["description"]
    assert "id" in data
    assert "state" in data
    assert data["state"] == AgentState.IDLE
    assert "created_at" in data
    assert "last_active" in data
    assert data["tasks_completed"] == 0

def test_create_agent_invalid_name(client: TestClient):
    """Test agent creation with invalid name"""
    agent_data = {
        "name": "",  # Empty name
        "type": AgentType.CODE_ANALYZER
    }
    response = client.post("/agents", json=agent_data)
    assert response.status_code == 422
    error = response.json()
    assert any(e["loc"] == ["body", "name"] for e in error["detail"])

def test_create_agent_invalid_type(client: TestClient):
    """Test agent creation with invalid type"""
    agent_data = {
        "name": "Test Agent",
        "type": "invalid_type"
    }
    response = client.post("/agents", json=agent_data)
    assert response.status_code == 422
    error = response.json()
    assert any(e["loc"] == ["body", "type"] for e in error["detail"])

def test_list_agents_empty(client: TestClient):
    """Test listing agents when no agents exist"""
    response = client.get("/agents")
    assert response.status_code == 200
    assert response.json() == []

def test_list_agents_with_data(client: TestClient):
    """Test listing agents after creating some"""
    # Create an agent first
    agent_data = {
        "name": "Test Agent",
        "type": AgentType.CODE_ANALYZER
    }
    client.post("/agents", json=agent_data)
    
    # List agents
    response = client.get("/agents")
    assert response.status_code == 200
    agents = response.json()
    assert len(agents) == 1
    assert agents[0]["name"] == agent_data["name"]

def test_get_agent(client: TestClient):
    """Test getting a specific agent"""
    # Create an agent first
    agent_data = {
        "name": "Test Agent",
        "type": AgentType.CODE_ANALYZER
    }
    create_response = client.post("/agents", json=agent_data)
    agent_id = create_response.json()["id"]
    
    # Get the agent
    response = client.get(f"/agents/{agent_id}")
    assert response.status_code == 200
    agent = response.json()
    assert agent["id"] == agent_id
    assert agent["name"] == agent_data["name"]

def test_get_agent_not_found(client: TestClient):
    """Test getting a non-existent agent"""
    response = client.get("/agents/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_update_agent(client: TestClient):
    """Test updating an agent"""
    # Create an agent first
    agent_data = {
        "name": "Test Agent",
        "type": AgentType.CODE_ANALYZER
    }
    create_response = client.post("/agents", json=agent_data)
    agent_id = create_response.json()["id"]
    
    # Update the agent
    update_data = {
        "name": "Updated Agent",
        "description": "Updated description"
    }
    response = client.patch(f"/agents/{agent_id}", json=update_data)
    assert response.status_code == 200
    updated_agent = response.json()
    assert updated_agent["name"] == update_data["name"]
    assert updated_agent["description"] == update_data["description"]

def test_update_agent_not_found(client: TestClient):
    """Test updating a non-existent agent"""
    update_data = {"name": "Updated Agent"}
    response = client.patch("/agents/nonexistent-id", json=update_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_delete_agent(client: TestClient):
    """Test deleting an agent"""
    # Create an agent first
    agent_data = {
        "name": "Test Agent",
        "type": AgentType.CODE_ANALYZER
    }
    create_response = client.post("/agents", json=agent_data)
    agent_id = create_response.json()["id"]
    
    # Delete the agent
    response = client.delete(f"/agents/{agent_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify the agent is gone
    get_response = client.get(f"/agents/{agent_id}")
    assert get_response.status_code == 404

def test_delete_agent_not_found(client: TestClient):
    """Test deleting a non-existent agent"""
    response = client.delete("/agents/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
