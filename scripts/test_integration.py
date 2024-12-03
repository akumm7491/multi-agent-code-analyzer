import asyncio
import logging
from datetime import datetime
import uuid
from multi_agent_code_analyzer.storage.mcp_storage import MCPStorageService, MCPContext
from multi_agent_code_analyzer.agents.communication import AgentCommunicationService, MessageType
from multi_agent_code_analyzer.config.settings import get_settings, initialize_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_storage_service():
    """Test MCP storage service"""
    storage = MCPStorageService()
    project_id = str(uuid.uuid4())

    try:
        # Create test context
        test_context = MCPContext(
            context_id=str(uuid.uuid4()),
            content="Test context content",
            embedding=[0.1] * 384,  # Dummy embedding
            metadata={"type": "test", "milestone": True},
            timestamp=datetime.utcnow(),
            project_id=project_id
        )

        # Store context
        success = await storage.store_context(test_context)
        assert success, "Failed to store context"
        logger.info("✓ Context storage test passed")

        # Search similar contexts
        similar = await storage.search_similar_contexts(
            embedding=test_context.embedding,
            project_id=project_id
        )
        assert len(similar) > 0, "Failed to find similar contexts"
        logger.info("✓ Context search test passed")

        # Get project timeline
        timeline = await storage.get_project_timeline(project_id)
        assert len(timeline) > 0, "Failed to get project timeline"
        logger.info("✓ Project timeline test passed")

    finally:
        storage.close()


async def test_agent_communication():
    """Test agent communication service"""
    settings = get_settings()
    comm_service = AgentCommunicationService(settings.database.REDIS_URI)

    try:
        await comm_service.connect()

        # Test message handler
        async def handle_test_message(message):
            logger.info(f"Received test message: {message.content}")
            assert message.content["test"] == "data"

        # Register handler
        comm_service.register_handler(
            MessageType.TASK_ASSIGNMENT,
            handle_test_message
        )

        # Subscribe test agent
        await comm_service.subscribe("test_agent")

        # Send test message
        test_message = comm_service.create_message(
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id="test_sender",
            recipient_id="test_agent",
            content={"test": "data"}
        )
        await comm_service.publish_message(test_message)

        # Wait for message processing
        await asyncio.sleep(1)
        logger.info("✓ Agent communication test passed")

    finally:
        await comm_service.disconnect()


async def verify_neo4j_connection():
    """Verify Neo4j connection"""
    from neo4j import GraphDatabase
    settings = get_settings()

    driver = GraphDatabase.driver(
        settings.database.NEO4J_URI,
        auth=(settings.database.NEO4J_USER, settings.database.NEO4J_PASSWORD)
    )

    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
            logger.info("✓ Neo4j connection test passed")
    finally:
        driver.close()


async def verify_redis_connection():
    """Verify Redis connection"""
    import aioredis
    settings = get_settings()

    redis = await aioredis.from_url(
        settings.database.REDIS_URI,
        password=settings.database.REDIS_PASSWORD
    )

    try:
        await redis.ping()
        logger.info("✓ Redis connection test passed")
    finally:
        await redis.close()


async def verify_milvus_connection():
    """Verify Milvus connection"""
    from pymilvus import connections
    settings = get_settings()

    try:
        connections.connect(
            alias="default",
            host=settings.vector_db.MILVUS_HOST,
            port=settings.vector_db.MILVUS_PORT
        )
        logger.info("✓ Milvus connection test passed")
    finally:
        connections.disconnect("default")


async def main():
    """Run all integration tests"""
    logger.info("Starting integration tests...")
    initialize_settings()

    try:
        # Test basic connectivity
        await verify_neo4j_connection()
        await verify_redis_connection()
        await verify_milvus_connection()

        # Test core services
        await test_storage_service()
        await test_agent_communication()

        logger.info("All integration tests passed successfully!")

    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
