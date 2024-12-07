from typing import Dict, List, Optional
from datetime import datetime
import uuid
from ..models.agent import Agent, AgentCreate, AgentUpdate, AgentState

class AgentService:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def create_agent(self, agent_create: AgentCreate) -> Agent:
        """Create a new agent"""
        agent_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        agent = Agent(
            id=agent_id,
            created_at=current_time,
            last_active=current_time,
            **agent_create.model_dump()
        )
        
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        """List all agents"""
        return list(self.agents.values())

    def update_agent(self, agent_id: str, agent_update: AgentUpdate) -> Optional[Agent]:
        """Update an agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        update_data = agent_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(agent, field, value)
        
        if agent_update.state:
            agent.last_active = datetime.utcnow()
        
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        return True

    def update_agent_state(self, agent_id: str, state: AgentState) -> Optional[Agent]:
        """Update an agent's state"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        agent.state = state
        agent.last_active = datetime.utcnow()
        
        if state == AgentState.IDLE:
            agent.tasks_completed += 1
        
        return agent
