from typing import Dict, Any, List, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class Node:
    """Base class for graph nodes"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


@dataclass
class Relationship:
    """Base class for graph relationships"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    created_at: datetime = datetime.now()


class GraphService:
    """Service for managing graph database operations"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        )
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        await self.driver.verify_connectivity()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.driver.close()

    async def create_node(self, node: Node) -> bool:
        """Create a node in the graph"""
        query = """
        CREATE (n:{labels} $properties)
        RETURN n
        """.format(labels=":".join(node.labels))

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    properties={
                        **node.properties,
                        "id": node.id,
                        "created_at": node.created_at.isoformat(),
                        "updated_at": node.updated_at.isoformat()
                    }
                )
                return await result.single() is not None
        except Exception as e:
            self.logger.error(f"Error creating node: {e}")
            return False

    async def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between nodes"""
        query = """
        MATCH (source), (target)
        WHERE source.id = $source_id AND target.id = $target_id
        CREATE (source)-[r:{rel_type} $properties]->(target)
        RETURN r
        """.format(rel_type=relationship.type)

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    {
                        "source_id": relationship.source_id,
                        "target_id": relationship.target_id,
                        "properties": {
                            **relationship.properties,
                            "created_at": relationship.created_at.isoformat()
                        }
                    }
                )
                return await result.single() is not None
        except Exception as e:
            self.logger.error(f"Error creating relationship: {e}")
            return False

    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(query, {"node_id": node_id})
                record = await result.single()
                if record:
                    node = record["n"]
                    return Node(
                        id=node["id"],
                        labels=record["labels"],
                        properties={
                            k: v for k, v in node.items()
                            if k not in ["id", "created_at", "updated_at"]
                        },
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node["updated_at"])
                    )
                return None
        except Exception as e:
            self.logger.error(f"Error getting node: {e}")
            return None

    async def get_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Relationship]:
        """Get relationships for a node"""
        if direction == "outgoing":
            match_clause = f"MATCH (n {{id: $node_id}})-[r{
                ':' + relationship_type if relationship_type else ''}]->(target)"
        elif direction == "incoming":
            match_clause = f"MATCH (n {{id: $node_id}})<-[r{
                ':' + relationship_type if relationship_type else ''}]-(source)"
        else:
            match_clause = f"MATCH (n {{id: $node_id}})-[r{
                ':' + relationship_type if relationship_type else ''}]-(other)"

        query = f"""
        {match_clause}
        RETURN r, startNode(r).id as source_id, endNode(r).id as target_id, type(r) as type
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(query, {"node_id": node_id})
                relationships = []
                async for record in result:
                    rel = record["r"]
                    relationships.append(
                        Relationship(
                            source_id=record["source_id"],
                            target_id=record["target_id"],
                            type=record["type"],
                            properties={
                                k: v for k, v in rel.items()
                                if k != "created_at"
                            },
                            created_at=datetime.fromisoformat(
                                rel["created_at"])
                        )
                    )
                return relationships
        except Exception as e:
            self.logger.error(f"Error getting relationships: {e}")
            return []

    async def search_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Node]:
        """Search nodes by labels and properties"""
        label_clause = f":{':'.join(labels)}" if labels else ""
        property_conditions = " AND ".join(
            f"n.{k} = ${k}" for k in (properties or {}).keys()
        )
        where_clause = f"WHERE {
            property_conditions}" if property_conditions else ""

        query = f"""
        MATCH (n{label_clause})
        {where_clause}
        RETURN n, labels(n) as labels
        LIMIT $limit
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    {**(properties or {}), "limit": limit}
                )
                nodes = []
                async for record in result:
                    node = record["n"]
                    nodes.append(
                        Node(
                            id=node["id"],
                            labels=record["labels"],
                            properties={
                                k: v for k, v in node.items()
                                if k not in ["id", "created_at", "updated_at"]
                            },
                            created_at=datetime.fromisoformat(
                                node["created_at"]),
                            updated_at=datetime.fromisoformat(
                                node["updated_at"])
                        )
                    )
                return nodes
        except Exception as e:
            self.logger.error(f"Error searching nodes: {e}")
            return []

    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update node properties"""
        query = """
        MATCH (n {id: $node_id})
        SET n += $properties, n.updated_at = $updated_at
        RETURN n
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    {
                        "node_id": node_id,
                        "properties": properties,
                        "updated_at": datetime.now().isoformat()
                    }
                )
                return await result.single() is not None
        except Exception as e:
            self.logger.error(f"Error updating node: {e}")
            return False
