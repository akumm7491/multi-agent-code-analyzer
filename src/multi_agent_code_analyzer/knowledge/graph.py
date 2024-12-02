import networkx as nx
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class Relationship:
    """Represents a relationship between nodes."""
    type: str
    metadata: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime

class KnowledgeGraph:
    """Maintains relationships between code components and their understanding."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.relationship_metadata: Dict[tuple, Relationship] = {}
        
    async def add_node(self, node_id: str, node_type: str, metadata: Dict[str, Any]) -> None:
        """Add a node to the knowledge graph."""
        now = datetime.now()
        node = Node(
            id=node_id,
            type=node_type,
            metadata=metadata,
            created_at=now,
            updated_at=now
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id)

    async def add_relationship(
        self, 
        from_node: str, 
        to_node: str, 
        relationship_type: str,
        metadata: Dict[str, Any] = None,
        confidence: float = 1.0
    ) -> None:
        """Add a relationship between nodes."""
        if metadata is None:
            metadata = {}

        if not (from_node in self.nodes and to_node in self.nodes):
            raise ValueError("Both nodes must exist in the graph")

        now = datetime.now()
        relationship = Relationship(
            type=relationship_type,
            metadata=metadata,
            confidence=confidence,
            created_at=now,
            updated_at=now
        )
        
        self.relationship_metadata[(from_node, to_node)] = relationship
        self.graph.add_edge(from_node, to_node, 
                           type=relationship_type,
                           metadata=metadata,
                           confidence=confidence)

    async def get_related_nodes(
        self, 
        node_id: str, 
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[str]:
        """Get nodes related to the given node."""
        if node_id not in self.nodes:
            return []

        related_nodes = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.edges[node_id, neighbor]
            if edge_data['confidence'] >= min_confidence:
                if relationship_type is None or edge_data['type'] == relationship_type:
                    related_nodes.append(neighbor)
        
        return related_nodes

    async def get_node_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific node."""
        node = self.nodes.get(node_id)
        return node.metadata if node else None

    async def update_node(
        self, 
        node_id: str, 
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> None:
        """Update a node's metadata."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")

        node = self.nodes[node_id]
        if merge:
            node.metadata.update(metadata)
        else:
            node.metadata = metadata
        node.updated_at = datetime.now()

    async def find_paths(
        self, 
        start_node: str, 
        end_node: str,
        min_confidence: float = 0.0
    ) -> List[List[str]]:
        """Find all paths between two nodes above a confidence threshold."""
        if not (start_node in self.nodes and end_node in self.nodes):
            return []

        def confidence_filter(u, v, d):
            return d['confidence'] >= min_confidence

        try:
            paths = list(nx.all_simple_paths(self.graph, start_node, end_node))
            return [path for path in paths if self._path_meets_confidence(path, min_confidence)]
        except nx.NetworkXNoPath:
            return []

    def _path_meets_confidence(self, path: List[str], min_confidence: float) -> bool:
        """Check if all edges in a path meet the minimum confidence threshold."""
        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i + 1]]
            if edge_data['confidence'] < min_confidence:
                return False
        return True

    async def get_subgraph(
        self, 
        node_ids: Set[str],
        include_neighbors: bool = False,
        min_confidence: float = 0.0
    ) -> 'KnowledgeGraph':
        """Extract a subgraph containing specified nodes and their relationships."""
        if include_neighbors:
            expanded_nodes = set(node_ids)
            for node_id in node_ids:
                neighbors = await self.get_related_nodes(node_id, min_confidence=min_confidence)
                expanded_nodes.update(neighbors)
            node_ids = expanded_nodes

        subgraph = KnowledgeGraph()
        
        # Copy nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                await subgraph.add_node(node.id, node.type, node.metadata.copy())

        # Copy relationships
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            if from_node in node_ids and to_node in node_ids:
                if edge_data['confidence'] >= min_confidence:
                    await subgraph.add_relationship(
                        from_node,
                        to_node,
                        edge_data['type'],
                        edge_data['metadata'].copy(),
                        edge_data['confidence']
                    )

        return subgraph