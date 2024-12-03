from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from neo4j import GraphDatabase
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import numpy as np
from dataclasses import dataclass
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MCPContext:
    context_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    project_id: str
    agent_id: Optional[str] = None


class MCPStorageService:
    def __init__(self):
        self.settings = get_settings()

        # Initialize Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            self.settings.database.NEO4J_URI,
            auth=(self.settings.database.NEO4J_USER,
                  self.settings.database.NEO4J_PASSWORD)
        )

        # Initialize Milvus connection
        connections.connect(
            alias="default",
            host=self.settings.vector_db.MILVUS_HOST,
            port=self.settings.vector_db.MILVUS_PORT
        )

        # Initialize collections
        self._init_milvus_collection()
        self._init_neo4j_schema()

    def _init_milvus_collection(self):
        """Initialize Milvus collection for storing embeddings"""
        collection_name = "mcp_contexts"

        if utility.exists_collection(collection_name):
            self.collection = Collection(collection_name)
            return

        # Define fields for the collection
        fields = [
            FieldSchema(name="context_id", dtype=DataType.VARCHAR,
                        max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=384),  # Dimension matches the model
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="project_id",
                        dtype=DataType.VARCHAR, max_length=100)
        ]

        schema = CollectionSchema(fields)
        self.collection = Collection(collection_name, schema)

        # Create index for vector similarity search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(
            field_name="embedding", index_params=index_params)

    def _init_neo4j_schema(self):
        """Initialize Neo4j schema for storing context metadata"""
        with self.neo4j_driver.session() as session:
            session.run("""
                CREATE CONSTRAINT context_id IF NOT EXISTS
                FOR (c:Context) REQUIRE c.context_id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT project_id IF NOT EXISTS
                FOR (p:Project) REQUIRE p.project_id IS UNIQUE
            """)

    async def store_context(self, context: MCPContext) -> bool:
        """Store context in both Neo4j and Milvus"""
        try:
            # Store in Milvus
            self.collection.insert([
                [context.context_id],
                [context.embedding],
                [int(context.timestamp.timestamp())],
                [context.project_id]
            ])

            # Store in Neo4j
            with self.neo4j_driver.session() as session:
                session.run("""
                    MERGE (p:Project {project_id: $project_id})
                    CREATE (c:Context {
                        context_id: $context_id,
                        content: $content,
                        timestamp: datetime($timestamp),
                        metadata: $metadata
                    })
                    CREATE (c)-[:BELONGS_TO]->(p)
                    WITH c
                    MATCH (prev:Context)
                    WHERE prev.context_id <> c.context_id
                        AND prev.timestamp < c.timestamp
                    WITH c, prev
                    ORDER BY prev.timestamp DESC
                    LIMIT 1
                    CREATE (prev)-[:NEXT]->(c)
                """, {
                    "context_id": context.context_id,
                    "content": context.content,
                    "timestamp": context.timestamp.isoformat(),
                    "metadata": context.metadata,
                    "project_id": context.project_id
                })

            return True

        except Exception as e:
            logger.error(f"Error storing context: {e}")
            return False

    async def search_similar_contexts(
        self,
        embedding: List[float],
        project_id: str,
        limit: int = 5
    ) -> List[MCPContext]:
        """Search for similar contexts within a project"""
        try:
            # Search in Milvus
            self.collection.load()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=f'project_id == "{project_id}"'
            )

            context_ids = [hit.id for hit in results[0]]

            # Get full context data from Neo4j
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (c:Context)
                    WHERE c.context_id IN $context_ids
                    RETURN c
                    ORDER BY c.timestamp DESC
                """, {"context_ids": context_ids})

                contexts = []
                for record in result:
                    node = record["c"]
                    contexts.append(MCPContext(
                        context_id=node["context_id"],
                        content=node["content"],
                        embedding=self._get_embedding(node["context_id"]),
                        metadata=node["metadata"],
                        timestamp=node["timestamp"],
                        project_id=project_id
                    ))

                return contexts

        except Exception as e:
            logger.error(f"Error searching contexts: {e}")
            return []

    def _get_embedding(self, context_id: str) -> List[float]:
        """Get embedding from Milvus by context_id"""
        results = self.collection.query(
            expr=f'context_id == "{context_id}"',
            output_fields=["embedding"]
        )
        return results[0]["embedding"] if results else []

    async def get_project_timeline(self, project_id: str) -> List[MCPContext]:
        """Get chronological timeline of contexts for a project"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Context)-[:BELONGS_TO]->(p:Project {project_id: $project_id})
                RETURN c
                ORDER BY c.timestamp
            """, {"project_id": project_id})

            contexts = []
            for record in result:
                node = record["c"]
                contexts.append(MCPContext(
                    context_id=node["context_id"],
                    content=node["content"],
                    embedding=self._get_embedding(node["context_id"]),
                    metadata=node["metadata"],
                    timestamp=node["timestamp"],
                    project_id=project_id
                ))

            return contexts

    async def cleanup_old_contexts(self, project_id: str, days_to_keep: int = 90):
        """Clean up contexts older than specified days while maintaining key milestones"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

        # Find contexts to delete
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Context)-[:BELONGS_TO]->(p:Project {project_id: $project_id})
                WHERE c.timestamp < datetime($cutoff)
                    AND NOT EXISTS((c)<-[:REFERENCES]-())  // Keep referenced contexts
                    AND NOT c.metadata.milestone = true    // Keep milestones
                RETURN c.context_id as context_id
            """, {
                "project_id": project_id,
                "cutoff": datetime.fromtimestamp(cutoff_date).isoformat()
            })

            context_ids = [record["context_id"] for record in result]

            if not context_ids:
                return

            # Delete from Milvus
            context_ids_str = '","'.join(context_ids)
            expr = f'context_id in ["{context_ids_str}"]'
            self.collection.delete(expr)

            # Delete from Neo4j
            session.run("""
                MATCH (c:Context)
                WHERE c.context_id IN $context_ids
                DETACH DELETE c
            """, {"context_ids": context_ids})

    def close(self):
        """Close connections"""
        self.neo4j_driver.close()
        connections.disconnect("default")

    async def delete_contexts(self, context_ids: List[str]) -> bool:
        """Delete contexts by their IDs"""
        try:
            # Create expression for deleting multiple contexts
            context_ids_str = '","'.join(context_ids)
            expr = f'context_id in ["{context_ids_str}"]'

            # Delete from Milvus
            await self.collection.delete(expr)
            await self.collection.flush()

            return True
        except Exception as e:
            logger.error(f"Error deleting contexts: {e}")
            return False
