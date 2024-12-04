import os
import requests
from neo4j import GraphDatabase
import redis
from pymilvus import connections


def test_github_connection():
    repo_url = "https://github.com/akumm7491/shared-ddd-ed-microservices-layer"
    headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
    response = requests.get(
        f"https://api.github.com/repos/akumm7491/shared-ddd-ed-microservices-layer", headers=headers)
    assert response.status_code == 200, "GitHub connection failed"
    return True


def test_neo4j_connection():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1")
            assert result.single()[0] == 1
        driver.close()
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {str(e)}")
        return False


def test_redis_connection():
    try:
        r = redis.Redis.from_url(
            os.getenv("REDIS_URI"),
            password=os.getenv("REDIS_PASSWORD")
        )
        assert r.ping()
        return True
    except Exception as e:
        print(f"Redis connection failed: {str(e)}")
        return False


def test_milvus_connection():
    try:
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST"),
            port=int(os.getenv("MILVUS_PORT"))
        )
        return True
    except Exception as e:
        print(f"Milvus connection failed: {str(e)}")
        return False


def run_all_tests():
    print("Testing all connections...")

    results = {
        "GitHub": test_github_connection(),
        "Neo4j": test_neo4j_connection(),
        "Redis": test_redis_connection(),
        "Milvus": test_milvus_connection()
    }

    for service, status in results.items():
        print(f"{service}: {'✅ Connected' if status else '❌ Failed'}")

    return all(results.values())


if __name__ == "__main__":
    run_all_tests()
