#!/usr/bin/env python3

import asyncio
from neo4j import AsyncGraphDatabase
from redis import asyncio as aioredis
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import os
import logging
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_neo4j():
    """Initialize Neo4j database with required schemas and constraints."""
    try:
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        async with driver.session() as session:
            # Create constraints
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE
            """)
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE
            """)
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Context) REQUIRE c.id IS UNIQUE
            """)
        logger.info("Neo4j initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {str(e)}")
        raise


async def init_redis():
    """Initialize Redis with required keys and configurations."""
    try:
        redis = aioredis.from_url(settings.REDIS_URL)
        await redis.ping()
        logger.info("Redis initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        raise


def init_milvus():
    """Initialize Milvus with required collections."""
    try:
        connections.connect(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        # Create collection schema
        schema = CollectionSchema(
            fields=fields,
            description="Context embeddings collection"
        )

        # Create collection if it doesn't exist
        collection_name = settings.COLLECTION_NAME
        if not Collection.list_collections() or collection_name not in Collection.list_collections():
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default',
                shards_num=2
            )

            # Create index on the embedding field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            logger.info(f"Created Milvus collection: {collection_name}")
        else:
            logger.info(f"Milvus collection {collection_name} already exists")

        logger.info("Milvus initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {str(e)}")
        raise


async def main():
    """Main initialization function."""
    try:
        # Create required directories
        os.makedirs("logs/neo4j", exist_ok=True)
        os.makedirs("logs/mcp", exist_ok=True)

        # Initialize databases
        await init_neo4j()
        await init_redis()
        init_milvus()

        logger.info("System initialization completed successfully")
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
