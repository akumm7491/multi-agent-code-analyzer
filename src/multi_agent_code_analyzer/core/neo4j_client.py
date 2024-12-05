from typing import Dict, Any, List
from neo4j import AsyncGraphDatabase
from ..config import settings


class Neo4jClient:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    async def store_patterns(self, patterns: List[Dict[str, Any]]):
        """Store code patterns in Neo4j."""
        async with self.driver.session() as session:
            for pattern in patterns:
                await session.run("""
                    MERGE (p:Pattern {id: $id})
                    SET p += $properties
                    """,
                                  id=pattern['id'],
                                  properties=pattern
                                  )

    async def get_fix_patterns(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Get fix patterns for a specific type."""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (p:Pattern)
                WHERE p.type = $type
                RETURN p
                """,
                                       type=pattern_type
                                       )
            return [record["p"] async for record in result]

    async def store_code_structure(self, structure: Dict[str, Any]):
        """Store code structure in Neo4j."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (c:CodeStructure {id: $id})
                SET c += $properties
                """,
                              id=structure['id'],
                              properties=structure
                              )

    async def analyze_change_impact(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of a code change."""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (c:CodeStructure {id: $id})
                MATCH (c)-[r:DEPENDS_ON*]->(d)
                RETURN d
                """,
                                       id=change['file_id']
                                       )
            impacted_nodes = [record["d"] async for record in result]
            return {
                "is_safe": len(impacted_nodes) < 5,  # Example threshold
                "impacted_files": [node["path"] for node in impacted_nodes]
            }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.driver.close()
