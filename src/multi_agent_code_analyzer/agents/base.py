from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAgent(ABC):
    """Base class for all specialized agents in the system."""

    def __init__(self, name: str, specialty: str):
        """
        Initialize a new agent.

        Args:
            name (str): Unique identifier for the agent
            specialty (str): The agent's area of expertise
        """
        self.name = name
        self.specialty = specialty
        self.knowledge_base: Dict[str, Any] = {}
        self.confidence_threshold = 0.7

    @abstractmethod
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query within the agent's specialty.

        Args:
            query (str): The query to process
            context (Dict[str, Any]): Relevant context for the query

        Returns:
            Dict[str, Any]: The agent's response including analysis and confidence
        """
        pass

    @abstractmethod
    async def update_knowledge(self, new_information: Dict[str, Any]) -> None:
        """
        Update the agent's knowledge base with new information.

        Args:
            new_information (Dict[str, Any]): New information to incorporate
        """
        pass

    @abstractmethod
    async def collaborate(self, other_agent: 'BaseAgent', query: str) -> Dict[str, Any]:
        """
        Collaborate with another agent on a query.

        Args:
            other_agent (BaseAgent): The agent to collaborate with
            query (str): The query to collaborate on

        Returns:
            Dict[str, Any]: Combined insights from both agents
        """
        pass

    async def evaluate_confidence(self, analysis: Dict[str, Any]) -> float:
        """
        Evaluate the confidence level of an analysis.

        Args:
            analysis (Dict[str, Any]): The analysis to evaluate

        Returns:
            float: Confidence score between 0 and 1
        """
        # Default implementation - should be overridden by specific agents
        if not analysis:
            return 0.0
        return 0.5

    async def get_knowledge(self, key: str) -> Optional[Any]:
        """
        Retrieve information from the agent's knowledge base.

        Args:
            key (str): The key to lookup

        Returns:
            Optional[Any]: The stored information or None if not found
        """
        return self.knowledge_base.get(key)

    async def can_handle(self, query: str) -> bool:
        """
        Determine if the agent can handle a specific query.

        Args:
            query (str): The query to evaluate

        Returns:
            bool: True if the agent can handle the query, False otherwise
        """
        # Default implementation - should be overridden by specific agents
        return False

    @property
    def capabilities(self) -> List[str]:
        """
        List the agent's capabilities.

        Returns:
            List[str]: List of capabilities
        """
        return []
