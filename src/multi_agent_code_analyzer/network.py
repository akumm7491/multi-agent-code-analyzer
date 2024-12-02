from typing import Dict, Any, List, Optional
from .agents import (
    ArchitectAgent,
    CodeAgent,
    DomainAgent,
    SecurityAgent,
    TestingAgent,
    DocumentationAgent,
    DependencyAgent,
    OrchestratorAgent
)
from .knowledge.graph import KnowledgeGraph

class AgentNetwork:
    """Coordinates the network of specialized agents for code analysis."""
    
    def __init__(self):
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize agents
        self.agents: Dict[str, Any] = {
            "architect": ArchitectAgent("architect"),
            "code": CodeAgent("code"),
            "domain": DomainAgent("domain"),
            "security": SecurityAgent("security"),
            "testing": TestingAgent("testing"),
            "documentation": DocumentationAgent("documentation"),
            "dependency": DependencyAgent("dependency")
        }
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent("orchestrator")
        
        # Register agents with orchestrator
        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query using the agent network."""
        if context is None:
            context = {}
            
        # Let orchestrator determine relevant agents
        relevant_agents = await self.orchestrator.route_query(query)
        
        # Collect responses from relevant agents
        responses = []
        for agent in relevant_agents:
            agent_context = await self._get_agent_context(agent, context)
            response = await agent.process(query, agent_context)
            responses.append(response)
        
        # Synthesize responses
        final_response = await self.orchestrator.synthesize_responses(responses)
        
        # Update knowledge graph
        await self._update_knowledge(query, final_response)
        
        return final_response
    
    async def analyze_codebase(self, path: str) -> Dict[str, Any]:
        """Analyze an entire codebase."""
        analysis = {
            "architecture": {},
            "code_quality": {},
            "security": {},
            "testing": {},
            "documentation": {},
            "dependencies": {}
        }
        
        # Perform initial analysis with each agent
        for agent_type, agent in self.agents.items():
            result = await agent.process("analyze_all", {"path": path})
            analysis[agent_type] = result
        
        # Update knowledge graph with comprehensive analysis
        await self._update_knowledge("full_analysis", analysis)
        
        return analysis