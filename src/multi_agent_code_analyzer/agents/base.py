from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.knowledge_base = {}

    @abstractmethod
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query within the agent's specialty."""
        pass

    @abstractmethod
    async def update_knowledge(self, new_information: Dict[str, Any]):
        """Update the agent's knowledge base."""
        pass

    @abstractmethod
    async def collaborate(self, other_agent: 'BaseAgent', query: str) -> Dict[str, Any]:
        """Collaborate with another agent on a query."""
        pass