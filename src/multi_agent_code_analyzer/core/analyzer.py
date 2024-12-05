from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import structlog
import git
import os
from typing import Optional, List, Dict
import asyncio
from neo4j import GraphDatabase
from redis import Redis
from pymilvus import connections, Collection, utility
import ast
import glob
from pathlib import Path
import json
from .domain_analysis import (
    DomainAnalyzer,
    identify_bounded_context,
    detect_ddd_patterns,
    calculate_metrics
)

logger = structlog.get_logger()
app = FastAPI(
    title="DDD Code Analyzer",
    description="API for analyzing code repositories using Domain-Driven Design principles",
    version="1.0.0"
)

# Initialize connections
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", "your_secure_redis_password")
)

neo4j_client = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "your_secure_password")
    )
)

# Connect to Milvus
connections.connect(
    alias="default",
    host=os.getenv("MILVUS_HOST", "milvus"),
    port=os.getenv("MILVUS_PORT", 19530)
)


class AnalysisRequest(BaseModel):
    repo_path: str
    analysis_type: str = "full"
    branch: Optional[str] = None
    include_patterns: Optional[List[str]] = ["*.py", "*.java", "*.cs", "*.ts"]
    exclude_patterns: Optional[List[str]] = [
        "*test*", "*vendor*", "*node_modules*"]


class DomainConcept(BaseModel):
    name: str
    type: str  # aggregate, entity, value_object, service, etc.
    file_path: str
    line_number: int
    relationships: List[Dict[str, str]]
    properties: Dict[str, str]


class AnalysisResult(BaseModel):
    repo_path: str
    domain_concepts: List[DomainConcept]
    bounded_contexts: List[Dict[str, any]]
    patterns_found: List[Dict[str, any]]
    metrics: Dict[str, float]


def store_analysis_results(repo_path: str, domain_concepts: List[Dict],
                           bounded_contexts: List[Dict], patterns: List[Dict]):
    """Store analysis results in Neo4j."""
    with neo4j_client.session() as session:
        # Clear previous results for this repo
        session.run("""
            MATCH (n)-[r]-() WHERE n.repo_path = $repo_path
            DELETE n, r
        """, repo_path=repo_path)

        # Store domain concepts
        for concept in domain_concepts:
            session.run("""
                CREATE (c:DomainConcept {
                    name: $name,
                    type: $type,
                    file_path: $file_path,
                    line_number: $line_number,
                    repo_path: $repo_path
                })
            """, **concept, repo_path=repo_path)

        # Store relationships between concepts
        for concept in domain_concepts:
            for rel in concept["relationships"]:
                session.run("""
                    MATCH (a:DomainConcept {name: $source_name, repo_path: $repo_path})
                    MATCH (b:DomainConcept {name: $target_name, repo_path: $repo_path})
                    CREATE (a)-[:RELATES_TO {type: $rel_type}]->(b)
                """, source_name=concept["name"], target_name=rel["target"],
                            rel_type=rel["type"], repo_path=repo_path)

        # Store bounded contexts
        for context in bounded_contexts:
            session.run("""
                CREATE (bc:BoundedContext {
                    name: $name,
                    path: $path,
                    repo_path: $repo_path
                })
            """, **context, repo_path=repo_path)

            # Connect concepts to their bounded context
            for concept_name in context["concepts"]:
                session.run("""
                    MATCH (bc:BoundedContext {name: $context_name, repo_path: $repo_path})
                    MATCH (c:DomainConcept {name: $concept_name, repo_path: $repo_path})
                    CREATE (bc)-[:CONTAINS]->(c)
                """, context_name=context["name"], concept_name=concept_name,
                            repo_path=repo_path)


def store_embeddings(domain_concepts: List[Dict]):
    """Store concept embeddings in Milvus for similarity search."""
    try:
        analyzer = DomainAnalyzer()
        collection = Collection("domain_concepts")

        embeddings = []
        concept_names = []

        for concept in domain_concepts:
            # Create text representation of the concept
            concept_text = f"{concept['name']} {concept['type']} {concept['properties']['documentation']}"
            embedding = analyzer.model.encode(concept_text)
            embeddings.append(embedding)
            concept_names.append(concept['name'])

        # Insert into Milvus
        collection.insert([
            concept_names,  # Primary keys
            embeddings     # Vector data
        ])
        collection.flush()
    except Exception as e:
        logger.error("Error storing embeddings", error=str(e))


@app.post("/analyze", response_model=Dict[str, str])
async def analyze_repo(request: AnalysisRequest):
    """
    Start a new analysis of a code repository.
    The analysis will identify DDD patterns, bounded contexts, and domain concepts.
    """
    try:
        # Validate repository path
        if not os.path.exists(request.repo_path):
            raise HTTPException(
                status_code=404, detail="Repository path not found")

        # Initialize Git repository
        repo = git.Repo(request.repo_path)
        if request.branch:
            repo.git.checkout(request.branch)

        # Start analysis in background
        asyncio.create_task(perform_analysis(request))

        return {
            "status": "Analysis started",
            "repo": request.repo_path,
            "tracking_id": f"analysis:{request.repo_path}"
        }
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def perform_analysis(request: AnalysisRequest):
    try:
        redis_client.set(f"analysis:{request.repo_path}:status", "running")

        # Initialize analyzer
        analyzer = DomainAnalyzer()

        # Get files to analyze
        files = []
        for pattern in request.include_patterns:
            files.extend(
                glob.glob(f"{request.repo_path}/**/{pattern}", recursive=True))

        # Apply exclude patterns
        for pattern in request.exclude_patterns:
            files = [f for f in files if not glob.fnmatch(f, pattern)]

        # Analyze each file
        domain_concepts = []
        bounded_contexts = []
        patterns = []

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())

                # Extract domain concepts
                file_concepts = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        concept = analyzer.extract_domain_concept(
                            node, file_path)
                        if concept:
                            file_concepts.append(concept)

                domain_concepts.extend(file_concepts)

                # Identify bounded context
                context = identify_bounded_context(file_path, file_concepts)
                if context:
                    bounded_contexts.append(context)

                # Detect patterns
                file_patterns = detect_ddd_patterns(file_path, file_concepts)
                patterns.extend(file_patterns)

            except Exception as e:
                logger.error(f"Error analyzing file {file_path}", error=str(e))

        # Calculate metrics
        metrics = calculate_metrics(
            domain_concepts, bounded_contexts, patterns)

        # Store results
        store_analysis_results(
            request.repo_path, domain_concepts, bounded_contexts, patterns)
        store_embeddings(domain_concepts)

        # Cache the results
        result = AnalysisResult(
            repo_path=request.repo_path,
            domain_concepts=domain_concepts,
            bounded_contexts=bounded_contexts,
            patterns_found=patterns,
            metrics=metrics
        )

        redis_client.set(
            f"analysis:{request.repo_path}:result",
            result.json(),
            ex=3600  # Cache for 1 hour
        )

        redis_client.set(f"analysis:{request.repo_path}:status", "completed")

    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        redis_client.set(
            f"analysis:{request.repo_path}:status", f"failed: {str(e)}")


@app.get("/status", response_model=Dict[str, str])
async def get_status(repo_path: str):
    """Get the current status of an analysis."""
    status = redis_client.get(f"analysis:{repo_path}:status")
    if not status:
        raise HTTPException(
            status_code=404, detail="No analysis found for this repository")
    return {"status": status.decode()}


@app.get("/results/{repo_path}", response_model=AnalysisResult)
async def get_results(
    repo_path: str,
    include_concepts: bool = Query(
        True, description="Include domain concepts"),
    include_contexts: bool = Query(
        True, description="Include bounded contexts"),
    include_patterns: bool = Query(
        True, description="Include detected patterns")
):
    """Get the results of a completed analysis."""
    result_json = redis_client.get(f"analysis:{repo_path}:result")
    if not result_json:
        raise HTTPException(
            status_code=404, detail="No results found for this repository")

    result = json.loads(result_json)
    if not include_concepts:
        result.pop("domain_concepts", None)
    if not include_contexts:
        result.pop("bounded_contexts", None)
    if not include_patterns:
        result.pop("patterns_found", None)

    return result


@app.get("/similar-concepts/{concept_name}")
async def find_similar_concepts(
    concept_name: str,
    limit: int = Query(
        10, description="Maximum number of similar concepts to return")
):
    """Find similar domain concepts using vector similarity search."""
    try:
        collection = Collection("domain_concepts")
        results = collection.search(
            [concept_name],
            "embeddings",
            limit=limit,
            param={"metric_type": "L2"}
        )
        return {"similar_concepts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health of all system components."""
    try:
        # Check Redis
        redis_client.ping()

        # Check Neo4j
        with neo4j_client.session() as session:
            session.run("RETURN 1")

        # Check Milvus
        utility.get_server_version()

        return {"status": "healthy"}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))
