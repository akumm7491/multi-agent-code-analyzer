import ast
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import structlog
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from neo4j import AsyncGraphDatabase
from pymilvus import Collection, utility

logger = structlog.get_logger()

__all__ = ['DomainAnalyzer', 'identify_bounded_context',
           'detect_ddd_patterns', 'calculate_metrics']


@dataclass
class DDDPattern:
    name: str
    description: str
    indicators: List[str]
    code_patterns: List[str]
    ts_patterns: List[str] = None  # TypeScript specific patterns


# Define common DDD patterns
DDD_PATTERNS = {
    "aggregate_root": DDDPattern(
        name="Aggregate Root",
        description="An entity that ensures consistency for a group of objects",
        indicators=["Repository dependency",
                    "Collection management", "Invariant enforcement"],
        code_patterns=[
            r"class.*\(.*AggregateRoot\)",
            r"@aggregate_root",
            r"def validate\(",
            r"_entities",
            r"_children"
        ],
        ts_patterns=[
            r"extends\s+AggregateRoot",
            r"@AggregateRoot",
            r"private\s+_events:\s*DomainEvent\[\]",
            r"validate\(",
            r"applyEvent\("
        ]
    ),
    "entity": DDDPattern(
        name="Entity",
        description="An object with a unique identity",
        indicators=["Identity field", "Equality by ID", "Mutable state"],
        code_patterns=[
            r"class.*\(.*Entity\)",
            r"@entity",
            r"id: UUID",
            r"def __eq__",
            r"@property.*id"
        ],
        ts_patterns=[
            r"extends\s+Entity",
            r"@Entity",
            r"private\s+_id:\s*(?:string|UUID)",
            r"getId\(\):\s*(?:string|UUID)",
            r"equals\(other:\s*\w+\):\s*boolean"
        ]
    ),
    "value_object": DDDPattern(
        name="Value Object",
        description="An immutable object without identity",
        indicators=["Immutability", "Equality by attributes", "No identity"],
        code_patterns=[
            r"@dataclass\(frozen=True\)",
            r"class.*\(.*ValueObject\)",
            r"@value_object",
            r"@property",
            r"__slots__"
        ],
        ts_patterns=[
            r"extends\s+ValueObject",
            r"@ValueObject",
            r"readonly\s+\w+:",
            r"private\s+constructor",
            r"static\s+create"
        ]
    ),
    "repository": DDDPattern(
        name="Repository",
        description="Persistence abstraction for aggregates",
        indicators=["CRUD operations",
                    "Collection management", "Persistence logic"],
        code_patterns=[
            r"class.*Repository",
            r"class.*\(.*Repository\)",
            r"def find_by",
            r"def save\(",
            r"def delete\("
        ],
        ts_patterns=[
            r"implements\s+Repository",
            r"@Repository",
            r"findBy\w+",
            r"save\(",
            r"delete\(",
            r"findAll\("
        ]
    ),
    "domain_service": DDDPattern(
        name="Domain Service",
        description="Stateless operations that don't belong to entities",
        indicators=["Stateless operations",
                    "Multiple entity coordination", "Complex business rules"],
        code_patterns=[
            r"class.*Service",
            r"@service",
            r"def process\(",
            r"def calculate\(",
            r"def validate\("
        ],
        ts_patterns=[
            r"implements\s+\w+Service",
            r"@Service",
            r"@Injectable",
            r"process\w+",
            r"handle\w+"
        ]
    ),
    "factory": DDDPattern(
        name="Factory",
        description="Encapsulates complex object creation",
        indicators=["Object creation",
                    "Complex initialization", "Creation strategy"],
        code_patterns=[
            r"class.*Factory",
            r"@factory",
            r"def create\(",
            r"@classmethod.*create",
            r"return cls\("
        ],
        ts_patterns=[
            r"implements\s+\w+Factory",
            r"@Factory",
            r"static\s+create\w+",
            r"private\s+constructor",
            r"new\s+\w+\("
        ]
    ),
    "event": DDDPattern(
        name="Domain Event",
        description="Represents something that happened in the domain",
        indicators=["Event data", "Timestamp", "Event metadata"],
        code_patterns=[
            r"class.*Event",
            r"@event",
            r"timestamp",
            r"event_type",
            r"metadata"
        ],
        ts_patterns=[
            r"implements\s+DomainEvent",
            r"@DomainEvent",
            r"readonly\s+timestamp:",
            r"eventType\(",
            r"toJSON\("
        ]
    ),
    "specification": DDDPattern(
        name="Specification",
        description="Encapsulates business rules and validation logic",
        indicators=["Rule evaluation", "Composable logic", "Validation rules"],
        code_patterns=[
            r"class.*Specification",
            r"@specification",
            r"def is_satisfied",
            r"def and_",
            r"def or_"
        ],
        ts_patterns=[
            r"implements\s+Specification",
            r"@Specification",
            r"isSatisfiedBy\(",
            r"and\(",
            r"or\("
        ]
    )
}


class DomainAnalyzer:
    def __init__(self):
        try:
            self.patterns = DDD_PATTERNS
            self.vectorizer = TfidfVectorizer()
            self.neo4j_client = None  # Will be set by analyzer.py
            logger.info("DomainAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DomainAnalyzer: {str(e)}")
            raise

    def extract_domain_concept(self, file_path: str, content: str) -> List[Dict]:
        """Extract domain concepts from a file."""
        try:
            concepts = []

            # For TypeScript files, use regex patterns
            if file_path.endswith('.ts'):
                for pattern_name, pattern in self.patterns.items():
                    if not pattern.ts_patterns:
                        continue

                    for ts_pattern in pattern.ts_patterns:
                        matches = re.finditer(
                            ts_pattern, content, re.MULTILINE)
                        for match in matches:
                            # Extract class name if it's a class definition
                            class_name = None
                            class_match = re.search(
                                r'class\s+(\w+)', content[max(0, match.start()-50):match.end()+50])
                            if class_match:
                                class_name = class_match.group(1)
                            else:
                                # Try to find interface name
                                interface_match = re.search(
                                    r'interface\s+(\w+)', content[max(0, match.start()-50):match.end()+50])
                                if interface_match:
                                    class_name = interface_match.group(1)

                            if class_name:
                                concept = {
                                    "name": class_name,
                                    "type": pattern_name,
                                    "file_path": file_path,
                                    "line_number": content[:match.start()].count('\n') + 1,
                                    "properties": {
                                        "pattern_matched": ts_pattern,
                                        "code_snippet": content[max(0, match.start()-50):match.end()+50].strip()
                                    }
                                }
                                concepts.append(concept)

            return concepts

        except Exception as e:
            logger.error(f"Error extracting domain concepts: {str(e)}")
            return []

    def identify_bounded_context(self, file_path: str, concepts: List[Dict]) -> Optional[Dict]:
        """Identify bounded context for a file based on its concepts."""
        try:
            if not concepts:
                return None

            # Use directory structure as initial context boundary
            path = Path(file_path)

            # Look for explicit bounded context markers
            context_markers = [
                "domain",
                "bounded-context",
                "context",
                "module"
            ]

            # Find the most specific context directory
            context_path = None
            for part in path.parts:
                if any(marker in part.lower() for marker in context_markers):
                    context_path = part

            if not context_path:
                context_path = path.parent.name

            # Analyze concept relationships
            concept_names = [c["name"] for c in concepts]

            # Look for related concepts in the same directory
            related_files = list(path.parent.glob("*.ts"))
            related_concepts = []
            for related_file in related_files:
                if related_file.name != path.name:  # Skip the current file
                    try:
                        with open(related_file, 'r') as f:
                            content = f.read()
                            related_concepts.extend(
                                self.extract_domain_concept(str(related_file), content))
                    except Exception:
                        continue

            return {
                "name": context_path,
                "path": str(path.parent),
                "concepts": concept_names,
                "related_concepts": [c["name"] for c in related_concepts]
            }
        except Exception as e:
            logger.error(f"Error identifying bounded context: {str(e)}")
            return None

    def detect_ddd_patterns(self, file_path: str, concepts: List[Dict]) -> List[Dict]:
        """Detect DDD patterns in the file."""
        patterns_found = []
        try:
            for concept in concepts:
                for pattern_name, pattern in self.patterns.items():
                    if concept["type"] == pattern_name:
                        patterns_found.append({
                            "name": pattern_name,
                            "file_path": file_path,
                            "concept": concept["name"],
                            "description": pattern.description
                        })
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
        return patterns_found

    def calculate_metrics(self, domain_concepts: List[Dict], bounded_contexts: List[Dict], patterns: List[Dict]) -> Dict[str, float]:
        """Calculate metrics about the analyzed codebase."""
        try:
            total_concepts = len(domain_concepts)
            total_contexts = len(bounded_contexts)
            patterns_by_type = {}

            for pattern in patterns:
                pattern_type = pattern["name"]
                patterns_by_type[pattern_type] = patterns_by_type.get(
                    pattern_type, 0) + 1

            return {
                "total_concepts": total_concepts,
                "total_contexts": total_contexts,
                "patterns_distribution": patterns_by_type,
                "concepts_per_context": total_concepts / total_contexts if total_contexts > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    async def store_analysis_results(self, repo_path: str, domain_concepts: List[Dict], bounded_contexts: List[Dict], patterns: List[Dict]):
        """Store analysis results in Neo4j."""
        try:
            async with self.neo4j_client.session() as session:
                # Clear previous results for this repo
                await session.run("""
                    MATCH (n)-[r]-() WHERE n.repo_path = $repo_path
                    DELETE n, r
                """, repo_path=repo_path)

                # Store domain concepts
                for concept in domain_concepts:
                    await session.run("""
                        CREATE (c:DomainConcept {
                            name: $name,
                            type: $type,
                            file_path: $file_path,
                            line_number: $line_number,
                            repo_path: $repo_path
                        })
                    """, **concept, repo_path=repo_path)

                # Store bounded contexts
                for context in bounded_contexts:
                    await session.run("""
                        CREATE (bc:BoundedContext {
                            name: $name,
                            path: $path,
                            repo_path: $repo_path
                        })
                    """, **context, repo_path=repo_path)

                    # Connect concepts to their bounded context
                    for concept_name in context["concepts"]:
                        await session.run("""
                            MATCH (bc:BoundedContext {name: $context_name, repo_path: $repo_path})
                            MATCH (c:DomainConcept {name: $concept_name, repo_path: $repo_path})
                            CREATE (bc)-[:CONTAINS]->(c)
                        """, context_name=context["name"], concept_name=concept_name, repo_path=repo_path)

        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            raise

    async def store_embeddings(self, domain_concepts: List[Dict]):
        """Store concept embeddings in Milvus for similarity search."""
        try:
            if not utility.has_collection("domain_concepts"):
                # Create collection if it doesn't exist
                from pymilvus import FieldSchema, CollectionSchema, DataType
                fields = [
                    FieldSchema(name="concept_id", dtype=DataType.VARCHAR,
                                is_primary=True, max_length=100),
                    FieldSchema(name="embedding",
                                dtype=DataType.FLOAT_VECTOR, dim=100)
                ]
                schema = CollectionSchema(
                    fields=fields, description="Domain concepts embeddings")
                collection = Collection(name="domain_concepts", schema=schema)
            else:
                collection = Collection("domain_concepts")

            # Create text representations
            texts = []
            concept_ids = []
            for concept in domain_concepts:
                # Get documentation from properties if it exists, otherwise use name and type
                doc = concept['properties'].get(
                    'documentation', f"{concept['name']} {concept['type']}")
                concept_text = f"{concept['name']} {concept['type']} {doc}"
                texts.append(concept_text)
                concept_ids.append(concept['name'])

            if not texts:  # Skip if no concepts
                return

            # Create embeddings using TF-IDF
            from sklearn.decomposition import TruncatedSVD

            # First get TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Get number of features
            n_features = tfidf_matrix.shape[1]

            if n_features >= 100:
                # If we have enough features, reduce to 100 dimensions
                svd = TruncatedSVD(n_components=100)
                embeddings = svd.fit_transform(tfidf_matrix)
            else:
                # If we have fewer features, pad with zeros
                embeddings = np.zeros((len(texts), 100))
                tfidf_array = tfidf_matrix.toarray()
                embeddings[:, :n_features] = tfidf_array

            # Insert into Milvus
            collection.insert([
                concept_ids,  # Primary keys
                embeddings.tolist()  # Vector data
            ])
            collection.flush()

        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise

    async def analyze_repository(self, repo_url: str, repo_path: str, patterns: List[str]) -> Dict:
        """Analyze a repository directory."""
        try:
            domain_concepts = []
            bounded_contexts = []
            patterns_found = []

            # Walk through repository looking for TypeScript files
            for file_path in Path(repo_path).rglob("*"):
                if not any(file_path.match(pattern) for pattern in patterns):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # Extract concepts from file
                        file_concepts = self.extract_domain_concept(
                            str(file_path), content)
                        if file_concepts:
                            domain_concepts.extend(file_concepts)

                            # Identify bounded context
                            context = self.identify_bounded_context(
                                str(file_path), file_concepts)
                            if context:
                                # Check if we already have this context
                                existing_context = next(
                                    (c for c in bounded_contexts if c["name"] == context["name"]),
                                    None
                                )
                                if existing_context:
                                    # Merge concepts
                                    existing_context["concepts"].extend(
                                        context["concepts"])
                                    existing_context["concepts"] = list(
                                        set(existing_context["concepts"]))
                                    # Merge related concepts
                                    if "related_concepts" in context:
                                        existing_context.setdefault("related_concepts", []).extend(
                                            context["related_concepts"])
                                        existing_context["related_concepts"] = list(
                                            set(existing_context["related_concepts"]))
                                else:
                                    bounded_contexts.append(context)

                        # Detect patterns
                        file_patterns = self.detect_ddd_patterns(
                            str(file_path), file_concepts)
                        patterns_found.extend(file_patterns)

                except Exception as e:
                    logger.warning(
                        f"Error analyzing file {file_path}: {str(e)}")
                    continue

            # Calculate metrics
            metrics = self.calculate_metrics(
                domain_concepts, bounded_contexts, patterns_found)

            # Store results
            await self.store_analysis_results(repo_url, domain_concepts, bounded_contexts, patterns_found)
            await self.store_embeddings(domain_concepts)

            # Analyze relationships between bounded contexts
            context_relationships = []
            for context in bounded_contexts:
                for other_context in bounded_contexts:
                    if context["name"] != other_context["name"]:
                        # Check for shared concepts
                        shared_concepts = set(context.get("concepts", [])) & set(
                            other_context.get("concepts", []))
                        if shared_concepts:
                            context_relationships.append({
                                "source": context["name"],
                                "target": other_context["name"],
                                "type": "shared_concepts",
                                "concepts": list(shared_concepts)
                            })

                        # Check for related concepts
                        source_related = set(
                            context.get("related_concepts", []))
                        target_concepts = set(
                            other_context.get("concepts", []))
                        related_concepts = source_related & target_concepts
                        if related_concepts:
                            context_relationships.append({
                                "source": context["name"],
                                "target": other_context["name"],
                                "type": "related_concepts",
                                "concepts": list(related_concepts)
                            })

            return {
                "domain_concepts": domain_concepts,
                "bounded_contexts": bounded_contexts,
                "patterns_found": patterns_found,
                "context_relationships": context_relationships,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Error analyzing repository: {str(e)}")
            raise
