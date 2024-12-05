from neo4j import GraphDatabase
import os


def init_neo4j():
    """Initialize Neo4j schema and constraints."""
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    auth = (
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "your_secure_password")
    )

    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT domain_concept_unique IF NOT EXISTS
                    FOR (c:DomainConcept)
                    REQUIRE (c.name, c.repo_path) IS UNIQUE
                """)

                session.run("""
                    CREATE CONSTRAINT bounded_context_unique IF NOT EXISTS
                    FOR (bc:BoundedContext)
                    REQUIRE (bc.name, bc.repo_path) IS UNIQUE
                """)

                # Create indexes
                session.run("""
                    CREATE INDEX domain_concept_type IF NOT EXISTS
                    FOR (c:DomainConcept)
                    ON (c.type)
                """)

                session.run("""
                    CREATE INDEX domain_concept_repo IF NOT EXISTS
                    FOR (c:DomainConcept)
                    ON (c.repo_path)
                """)

        return True
    except Exception as e:
        print(f"Error initializing Neo4j: {str(e)}")
        return False


if __name__ == "__main__":
    init_neo4j()
