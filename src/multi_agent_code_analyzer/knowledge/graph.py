import networkx as nx
from typing import Dict, Any, List, Optional

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.metadata = {}

    async def add_node(self, node_id: str, metadata: Dict[str, Any]):
        """Add a node to the knowledge graph."""
        self.graph.add_node(node_id)
        self.metadata[node_id] = metadata

    async def add_relationship(self, from_node: str, to_node: str, relationship_type: str):
        """Add a relationship between nodes."""
        self.graph.add_edge(from_node, to_node, type=relationship_type)

    async def get_related_nodes(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """Get nodes related to the given node."""
        if relationship_type:
            return [n for n in self.graph.neighbors(node_id) 
                   if self.graph[node_id][n]['type'] == relationship_type]
        return list(self.graph.neighbors(node_id))