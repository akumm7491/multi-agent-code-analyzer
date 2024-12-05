from neo4j import GraphDatabase
import sys
import time
import os
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_port_open(host, port, timeout=2):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking port: {str(e)}")
        return False


def test_connection():
    try:
        # Get connection details from environment
        uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
        user = os.environ.get('NEO4J_USER', 'neo4j')
        password = os.environ.get('NEO4J_PASSWORD', 'password123')

        if not all([uri, user, password]):
            logger.error(f"Missing required environment variables:")
            logger.error(f"NEO4J_URI: {uri}")
            logger.error(f"NEO4J_USER: {user}")
            logger.error(
                f"NEO4J_PASSWORD: {'*' * len(password) if password else None}")
            return False

        # Parse host and port from URI
        host = uri.split('://')[1].split(':')[0]
        port = int(uri.split(':')[-1])

        # First check if the port is open
        logger.info(f"Checking if Neo4j port is open on {host}:{port}")
        if not check_port_open(host, port):
            logger.error(f"Neo4j port {port} is not open on {host}")
            return False

        logger.info(f"Attempting to connect to Neo4j at {uri}")
        driver = GraphDatabase.driver(uri, auth=(user, password))

        logger.info("Verifying connectivity...")
        driver.verify_connectivity()

        logger.info("Testing query...")
        with driver.session() as session:
            result = session.run('RETURN 1 as num')
            assert result.single()['num'] == 1

        driver.close()
        logger.info("Neo4j connection test successful!")
        return True

    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False


def main():
    max_attempts = 30
    wait_time = 5  # Increased wait time between attempts

    for i in range(max_attempts):
        logger.info(f"Attempt {i + 1}/{max_attempts} to connect to Neo4j")
        if test_connection():
            sys.exit(0)
        if i < max_attempts - 1:
            logger.info(f"Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)

    logger.error("Failed to connect to Neo4j after maximum attempts")
    sys.exit(1)


if __name__ == '__main__':
    main()
