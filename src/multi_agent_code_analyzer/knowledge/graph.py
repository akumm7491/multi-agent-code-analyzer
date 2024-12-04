import networkx as nx
from neo4j import AsyncGraphDatabase
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime
import logging
import os
from prometheus_client import Counter, Gauge
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from pathlib import Path
import re
import gherkin_parser
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import git
from difflib import unified_diff
from itertools import chain

# Define Prometheus metrics
GRAPH_NODES = Gauge('knowledge_graph_nodes_total',
                    'Total number of nodes in the knowledge graph')
GRAPH_RELATIONSHIPS = Gauge('knowledge_graph_relationships_total',
                            'Total number of relationships in the knowledge graph')
GRAPH_OPERATIONS = Counter('knowledge_graph_operations_total',
                           'Total number of graph operations', ['operation_type'])
GRAPH_INSIGHTS = Counter('knowledge_graph_insights_total',
                         'Total number of insights generated', ['insight_type'])
PATTERN_CONFIDENCE = Gauge(
    'pattern_confidence', 'Confidence in pattern detection', ['pattern_type'])


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph"""
    REPOSITORY = "Repository"
    COMPONENT = "Component"
    FILE = "File"
    CLASS = "Class"
    METHOD = "Method"
    FUNCTION = "Function"
    VARIABLE = "Variable"
    PATTERN = "Pattern"
    ENTITY = "Entity"
    SERVICE = "Service"
    TEST = "Test"
    DEPENDENCY = "Dependency"
    AGENT = "Agent"
    INSIGHT = "Insight"
    CHANGE = "Change"
    IMPACT = "Impact"
    SEMANTIC_GROUP = "SemanticGroup"
    CODE_BLOCK = "CodeBlock"
    PATTERN_INSTANCE = "PatternInstance"


class RelationType(str, Enum):
    """Types of relationships in the knowledge graph"""
    BELONGS_TO = "BELONGS_TO"
    DEPENDS_ON = "DEPENDS_ON"
    CALLS = "CALLS"
    IMPLEMENTS = "IMPLEMENTS"
    INHERITS_FROM = "INHERITS_FROM"
    TESTS = "TESTS"
    IMPACTS = "IMPACTS"
    ANALYZED_BY = "ANALYZED_BY"
    RELATED_TO = "RELATED_TO"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    DETECTED_IN = "DETECTED_IN"
    MODIFIED_BY = "MODIFIED_BY"
    SEMANTICALLY_SIMILAR = "SEMANTICALLY_SIMILAR"
    FOLLOWS_PATTERN = "FOLLOWS_PATTERN"
    PRECEDES = "PRECEDES"
    USES = "USES"


@dataclass
class CodeInsight:
    """Represents an insight discovered about the code"""
    type: str
    description: str
    confidence: float
    source_nodes: List[str]
    timestamp: datetime
    agent_id: str
    metadata: Dict[str, Any]


@dataclass
class PatternInstance:
    """Represents a detected pattern instance in the code"""
    pattern_type: str
    components: List[str]
    confidence: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class SemanticGroup:
    """Represents a group of semantically related code elements"""
    elements: List[str]
    similarity_score: float
    common_theme: str
    metadata: Dict[str, Any]


@dataclass
class GraphRAGResult:
    """Represents a result from graph-based retrieval"""
    node_id: str
    content: str
    relevance_score: float
    context: Dict[str, Any]
    path_to_root: List[str]


@dataclass
class KnowledgeFragment:
    """Represents a fragment of knowledge in the graph"""
    id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    relationships: List[Dict[str, str]]
    source_type: str
    timestamp: datetime


@dataclass
class RequirementValidation:
    """Represents a validation of a requirement"""
    requirement_id: str
    validation_type: str  # static, dynamic, semantic
    confidence_score: float
    evidence: List[Dict[str, Any]]
    timestamp: datetime
    validation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class CodeStateAssumption:
    """Represents an assumption about code state"""
    assumption_id: str
    description: str
    confidence_score: float
    affected_components: List[str]
    validation_status: str
    last_validated: datetime
    dependencies: List[str]
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Represents the result of a validation check"""
    success: bool
    confidence: float
    evidence: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]
    suggested_actions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ValidationStrategy(str, Enum):
    """Types of validation strategies"""
    STATIC_ANALYSIS = "static_analysis"
    DYNAMIC_TESTING = "dynamic_testing"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    REQUIREMENT_TRACING = "requirement_tracing"
    ASSUMPTION_CHECKING = "assumption_checking"
    IMPACT_ANALYSIS = "impact_analysis"


@dataclass
class BehaviorScenario:
    """Represents a Gherkin behavior scenario"""
    scenario_id: str
    feature: str
    scenario: str
    given: List[str]
    when: List[str]
    then: List[str]
    examples: List[Dict[str, Any]]
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class CodeUnderstanding:
    """Represents deep understanding of code components"""
    component_id: str
    behavioral_patterns: List[Dict[str, Any]]
    data_flows: List[Dict[str, Any]]
    control_flows: List[Dict[str, Any]]
    state_transitions: List[Dict[str, Any]]
    invariants: List[Dict[str, Any]]
    cross_cutting_concerns: List[Dict[str, Any]]
    architectural_roles: List[str]
    domain_concepts: List[Dict[str, Any]]
    confidence: float


class ComprehensionStrategy(str, Enum):
    """Types of code comprehension strategies"""
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    DOMAIN = "domain"


@dataclass
class NLUContext:
    """Natural Language Understanding context"""
    intent: str
    entities: List[Dict[str, Any]]
    sentiment: float
    confidence: float
    context_window: List[str]
    metadata: Dict[str, Any]


@dataclass
class CodeChangeVerification:
    """Verification result for code changes"""
    is_verified: bool
    confidence: float
    evidence: List[Dict[str, Any]]
    git_verification: Dict[str, Any]
    test_results: Dict[str, Any]
    static_analysis: Dict[str, Any]
    semantic_validation: Dict[str, Any]
    hallucination_score: float


class VerificationStrategy(str, Enum):
    """Types of verification strategies"""
    GIT_HISTORY = "git_history"
    TEST_COVERAGE = "test_coverage"
    STATIC_ANALYSIS = "static_analysis"
    SEMANTIC_DIFF = "semantic_diff"
    PATTERN_MATCHING = "pattern_matching"
    HALLUCINATION_CHECK = "hallucination_check"


@dataclass
class LearningContext:
    """Context for continuous learning"""
    patterns: Dict[str, Any]
    adaptations: List[Dict[str, Any]]
    feedback: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    confidence_thresholds: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class AdaptationStrategy:
    """Strategy for system adaptation"""
    trigger_conditions: List[Dict[str, Any]]
    adaptation_rules: List[Dict[str, Any]]
    rollback_conditions: List[Dict[str, Any]]
    success_metrics: Dict[str, Any]
    confidence: float


@dataclass
class MCPContext:
    """Model Context Protocol context"""
    model_id: str
    task_id: str
    session_id: str
    conversation_id: str
    metadata: Dict[str, Any]
    confidence: float
    verification_status: str


@dataclass
class KnowledgeNode:
    """Base class for knowledge graph nodes"""
    node_id: str
    node_type: str
    content: str
    metadata: Dict[str, Any]
    confidence: float
    verification_status: str
    mcp_context: MCPContext


class KnowledgeGraph:
    """Advanced knowledge graph implementation with RAG and JSON agent capabilities."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        # Initialize base components
        self.graph = nx.DiGraph()
        self.metadata = {}
        self.neo4j_enabled = all([uri, user, password])
        if self.neo4j_enabled:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        else:
            self.driver = None

        # Initialize advanced components
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._insight_cache = {}
        self._change_history = []
        self._pattern_instances = defaultdict(list)
        self._semantic_embeddings = {}
        self._vectorizer = TfidfVectorizer(max_features=1000)
        self._semantic_groups = []

        # Initialize pattern learning
        self._pattern_templates = self._load_pattern_templates()
        self._pattern_scores = defaultdict(float)

        # Initialize RAG components
        self._knowledge_fragments = {}
        self._embedding_cache = {}
        self._relevance_threshold = 0.75
        self._max_context_items = 5

        # Initialize JSON agent components
        self._agent_schemas = {}
        self._agent_capabilities = defaultdict(set)
        self._interaction_history = []

        # Load agent schemas
        self._load_agent_schemas()

        # Initialize validation components
        self._requirements = {}
        self._assumptions = {}
        self._validation_history = []
        self._confidence_scores = defaultdict(float)
        self._validation_strategies = self._load_validation_strategies()
        self._assumption_dependencies = nx.DiGraph()

        # Initialize continuous analysis
        self._analysis_queue = asyncio.Queue()
        self._validation_tasks = set()
        self._confidence_threshold = 0.8
        self._min_validations_required = 3

        # Initialize NLU components
        self.nlu_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base")
        self.nlu_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.intent_classifier = pipeline(
            "text-classification", model="facebook/bart-large-mnli")
        self.entity_extractor = pipeline(
            "ner", model="microsoft/codebert-base-mlm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

        # Initialize verification components
        self.repo = git.Repo(self.workspace_path)
        self.verification_history = []
        self.confidence_threshold = 0.85
        self.hallucination_threshold = 0.3

    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load pattern templates from configuration"""
        try:
            pattern_path = os.path.join(
                os.path.dirname(__file__), 'patterns.json')
            with open(pattern_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load pattern templates: {str(e)}")
            return {}

    def _load_agent_schemas(self):
        """Load JSON schemas defining agent capabilities"""
        try:
            schema_dir = Path(__file__).parent / "schemas"
            if schema_dir.exists():
                for schema_file in schema_dir.glob("*.json"):
                    with open(schema_file) as f:
                        schema = json.load(f)
                        agent_type = schema_file.stem
                        self._agent_schemas[agent_type] = schema
                        self._agent_capabilities[agent_type].update(
                            schema.get("capabilities", [])
                        )
        except Exception as e:
            self.logger.error(f"Failed to load agent schemas: {str(e)}")

    async def close(self):
        """Close the Neo4j database connection if enabled"""
        if self.neo4j_enabled:
            await self.driver.close()

    async def add_node(self, node_id: str, metadata: Dict[str, Any], node_type: str = "Generic"):
        """Add a node to both NetworkX and Neo4j graphs."""
        try:
            # Add to NetworkX
            self.graph.add_node(node_id)
            self.metadata[node_id] = metadata

            # Add to Neo4j if enabled
            if self.neo4j_enabled:
                async with self.driver.session() as session:
                    # Convert metadata to string properties for Neo4j
                    props = {k: str(v) if isinstance(v, (dict, list)) else v
                             for k, v in metadata.items()}
                    props['id'] = node_id

                    await session.run(
                        f"""
                        CREATE (n:{node_type} $props)
                        """,
                        props=props
                    )

            # Update metrics
            GRAPH_NODES.inc()
            GRAPH_OPERATIONS.labels(operation_type="add_node").inc()

        except Exception as e:
            self.logger.error(f"Failed to add node to graph: {str(e)}")
            raise

    async def add_relationship(self, from_node: str, to_node: str, relationship_type: str):
        """Add a relationship between nodes in both graphs."""
        try:
            # Add to NetworkX
            self.graph.add_edge(from_node, to_node, type=relationship_type)

            # Add to Neo4j if enabled
            if self.neo4j_enabled:
                async with self.driver.session() as session:
                    await session.run(
                        f"""
                        MATCH (a), (b)
                        WHERE a.id = $from_id AND b.id = $to_id
                        CREATE (a)-[r:{relationship_type}]->(b)
                        """,
                        from_id=from_node,
                        to_id=to_node
                    )

            # Update metrics
            GRAPH_RELATIONSHIPS.inc()
            GRAPH_OPERATIONS.labels(operation_type="add_relationship").inc()

        except Exception as e:
            self.logger.error(f"Failed to add relationship to graph: {str(e)}")
            raise

    async def get_related_nodes(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """Get nodes related to the given node from NetworkX graph."""
        try:
            if relationship_type:
                return [n for n in self.graph.neighbors(node_id)
                        if self.graph[node_id][n]['type'] == relationship_type]
            return list(self.graph.neighbors(node_id))
        except Exception as e:
            self.logger.error(f"Failed to get related nodes: {str(e)}")
            return []

    async def store_analysis_results(self, repo_name: str, analysis_data: Dict[str, Any]):
        """Store repository analysis results in the graph."""
        try:
            # Clear existing data
            await self.clear_repository_data(repo_name)

            # Add repository node
            repo_node_id = f"repo:{repo_name}"
            await self.add_node(
                repo_node_id,
                {
                    "name": repo_name,
                    "last_analyzed": datetime.now().isoformat(),
                    "architecture_style": analysis_data.get("architecture", {}).get("summary", {}).get("architectural_style", "unknown")
                },
                NodeType.REPOSITORY
            )

            # Store architecture components
            for component in analysis_data.get("architecture", {}).get("components", []):
                comp_node_id = f"component:{repo_name}:{component['name']}"
                await self.add_node(
                    comp_node_id,
                    component,
                    NodeType.COMPONENT
                )
                await self.add_relationship(comp_node_id, repo_node_id, RelationType.BELONGS_TO)

            # Store patterns
            for pattern in analysis_data.get("patterns", {}).get("detected", []):
                pattern_node_id = f"pattern:{repo_name}:{pattern['name']}"
                await self.add_node(
                    pattern_node_id,
                    pattern,
                    NodeType.PATTERN
                )
                await self.add_relationship(pattern_node_id, repo_node_id, RelationType.DETECTED_IN)

            # Store data models
            for entity in analysis_data.get("data_models", {}).get("entities", []):
                entity_node_id = f"entity:{repo_name}:{entity['name']}"
                await self.add_node(
                    entity_node_id,
                    entity,
                    NodeType.ENTITY
                )
                await self.add_relationship(entity_node_id, repo_node_id, RelationType.DEFINED_IN)

            # Store business logic
            for service in analysis_data.get("business_logic", {}).get("services", []):
                service_node_id = f"service:{repo_name}:{service['name']}"
                await self.add_node(
                    service_node_id,
                    service,
                    NodeType.SERVICE
                )
                await self.add_relationship(service_node_id, repo_node_id, RelationType.IMPLEMENTS)

            GRAPH_OPERATIONS.labels(operation_type="store_analysis").inc()

        except Exception as e:
            self.logger.error(f"Failed to store analysis results: {str(e)}")
            raise

    async def clear_repository_data(self, repo_name: str):
        """Clear existing data for a repository from both graphs."""
        try:
            # Clear from NetworkX
            nodes_to_remove = [node for node, data in self.metadata.items()
                               if data.get('repository') == repo_name]
            self.graph.remove_nodes_from(nodes_to_remove)
            for node in nodes_to_remove:
                self.metadata.pop(node, None)

            # Clear from Neo4j if enabled
            if self.neo4j_enabled:
                async with self.driver.session() as session:
                    await session.run(
                        """
                        MATCH (n {repository: $repo_name})-[r*0..]->(m)
                        DETACH DELETE n, m
                        """,
                        repo_name=repo_name
                    )

            # Update metrics
            GRAPH_NODES.set(len(self.graph.nodes))
            GRAPH_RELATIONSHIPS.set(len(self.graph.edges))
            GRAPH_OPERATIONS.labels(operation_type="clear_data").inc()

        except Exception as e:
            self.logger.error(f"Failed to clear repository data: {str(e)}")
            raise

    async def analyze_code_semantics(self, code_blocks: List[Tuple[str, str]]) -> List[SemanticGroup]:
        """Analyze semantic relationships between code blocks."""
        try:
            # Extract text and create vectors
            texts = [block[1] for block in code_blocks]
            vectors = self._vectorizer.fit_transform(texts)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(vectors)

            # Group similar code blocks
            groups = []
            processed = set()

            for i in range(len(code_blocks)):
                if i in processed:
                    continue

                similar_indices = np.where(similarity_matrix[i] > 0.7)[0]
                if len(similar_indices) > 1:
                    group = SemanticGroup(
                        elements=[code_blocks[j][0] for j in similar_indices],
                        similarity_score=float(
                            np.mean(similarity_matrix[i][similar_indices])),
                        common_theme=self._extract_common_theme(
                            [code_blocks[j][1] for j in similar_indices]
                        ),
                        metadata={"vector_centroid": vectors[similar_indices].mean(
                            axis=0).tolist()}
                    )
                    groups.append(group)
                    processed.update(similar_indices)

            self._semantic_groups.extend(groups)
            return groups

        except Exception as e:
            self.logger.error(f"Failed to analyze code semantics: {str(e)}")
            return []

    async def detect_patterns(self, code_block: str, context: Dict[str, Any]) -> List[PatternInstance]:
        """Detect patterns in a code block using machine learning and template matching."""
        try:
            patterns = []

            # Vectorize code block
            code_vector = self._vectorizer.fit_transform(
                [code_block]).toarray()
            )

            # Match against pattern templates
            for pattern_type, template in self._pattern_templates.items():
                confidence = self._calculate_pattern_confidence(
                    code_block, template)

                if confidence > 0.6:  # Confidence threshold
                    pattern = PatternInstance(
                        pattern_type = pattern_type,
                        components = self._extract_pattern_components(
                            code_block, template),
                        confidence = confidence,
                        context = context,
                        metadata = {"vector": code_vector.tolist()}
                    )
                    patterns.append(pattern)

                    # Update pattern statistics
                    self._pattern_scores[pattern_type] = (
                        self._pattern_scores[pattern_type] *
                        0.9 + confidence * 0.1
                    )
                    PATTERN_CONFIDENCE.labels(
                        pattern_type = pattern_type).set(confidence)

            return patterns

        except Exception as e:
            self.logger.error(f"Failed to detect patterns: {str(e)}")
            return []

    def _calculate_pattern_confidence(self, code_block: str, template: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern match."""
        try:
            # Implement pattern matching logic
            required_elements = template.get("required_elements", [])
            optional_elements = template.get("optional_elements", [])

            required_count = sum(
                1 for elem in required_elements if elem in code_block)
            optional_count = sum(
                1 for elem in optional_elements if elem in code_block)

            if not required_elements:
                return 0.0

            base_score = required_count / len(required_elements)
            bonus_score = optional_count / \
                (len(optional_elements) if optional_elements else 1)

            return min(1.0, base_score * 0.8 + bonus_score * 0.2)

        except Exception as e:
            self.logger.error(
                f"Failed to calculate pattern confidence: {str(e)}")
            return 0.0

    def _extract_pattern_components(self, code_block: str, template: Dict[str, Any]) -> List[str]:
        """Extract components that match the pattern template."""
        try:
            components = []
            for component in template.get("components", []):
                if component["pattern"] in code_block:
                    components.append(component["name"])
            return components
        except Exception as e:
            self.logger.error(
                f"Failed to extract pattern components: {str(e)}")
            return []

    def _extract_common_theme(self, code_blocks: List[str]) -> str:
        """Extract common theme from a group of code blocks."""
        try:
            # Use TF-IDF to find most common meaningful terms
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = vectorizer.fit_transform(code_blocks)

            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-3:][::-1]

            return ", ".join(feature_names[i] for i in top_indices)

        except Exception as e:
            self.logger.error(f"Failed to extract common theme: {str(e)}")
            return "Unknown Theme"

    async def add_semantic_relationship(self, node1_id: str, node2_id: str, similarity_score: float):
        """Add a semantic relationship between nodes based on their similarity."""
        try:
            if similarity_score > 0.7:  # Threshold for semantic similarity
                await self.add_relationship(
                    node1_id,
                    node2_id,
                    RelationType.SEMANTICALLY_SIMILAR
                )
                self.metadata[(node1_id, node2_id)] = {
                    "similarity_score": similarity_score}

                # Create or update semantic group
                await self._update_semantic_groups(node1_id, node2_id, similarity_score)

        except Exception as e:
            self.logger.error(f"Failed to add semantic relationship: {str(e)}")
            raise

    async def _update_semantic_groups(self, node1_id: str, node2_id: str, similarity_score: float):
        """Update semantic groups based on new relationship."""
        try:
            # Find existing groups containing either node
            existing_groups = [
                group for group in self._semantic_groups
                if node1_id in group.elements or node2_id in group.elements
            ]

            if existing_groups:
                # Update existing group
                group = existing_groups[0]
                if node1_id not in group.elements:
                    group.elements.append(node1_id)
                if node2_id not in group.elements:
                    group.elements.append(node2_id)
                group.similarity_score = (
                    group.similarity_score + similarity_score) / 2
            else:
                # Create new group
                group = SemanticGroup(
                    elements=[node1_id, node2_id],
                    similarity_score=similarity_score,
                    common_theme="",  # Will be updated by analyze_code_semantics
                    metadata={}
                )
                self._semantic_groups.append(group)

        except Exception as e:
            self.logger.error(f"Failed to update semantic groups: {str(e)}")

    async def query_graph(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query on Neo4j graph."""
        if not self.neo4j_enabled:
            raise RuntimeError(
                "Neo4j is not enabled. Provide connection details during initialization.")

        try:
            async with self.driver.session() as session:
                result = await session.run(cypher_query, parameters=params or {})
                return [record.data() for record in await result.fetch()]
        except Exception as e:
            self.logger.error(f"Failed to execute graph query: {str(e)}")
            raise

    def get_networkx_graph(self) -> nx.DiGraph:
        """Get the NetworkX graph for advanced graph algorithms and analysis."""
        return self.graph

    async def sync_graphs(self):
        """Synchronize NetworkX graph with Neo4j (useful after Neo4j operations)."""
        if not self.neo4j_enabled:
            return

        try:
            async with self.driver.session() as session:
                # Get all nodes
                nodes_result = await session.run(
                    """
                    MATCH (n) 
                    RETURN n
                    """
                )

                # Clear existing NetworkX data
                self.graph.clear()
                self.metadata.clear()

                # Add nodes to NetworkX
                async for record in nodes_result:
                    node = record['n']
                    node_id = node.get('id')
                    if node_id:
                        self.graph.add_node(node_id)
                        self.metadata[node_id] = dict(node)

                # Get all relationships
                rels_result = await session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN a.id as from_id, b.id as to_id, type(r) as rel_type
                    """
                )

                # Add relationships to NetworkX
                async for record in rels_result:
                    self.graph.add_edge(
                        record['from_id'],
                        record['to_id'],
                        type=record['rel_type']
                    )

                # Update metrics
                GRAPH_NODES.set(len(self.graph.nodes))
                GRAPH_RELATIONSHIPS.set(len(self.graph.edges))
                GRAPH_OPERATIONS.labels(operation_type="sync_graphs").inc()

        except Exception as e:
            self.logger.error(f"Failed to sync graphs: {str(e)}")
            raise

    async def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Register an agent in the knowledge graph."""
        try:
            agent_node_id = f"agent:{agent_id}"
            await self.add_node(
                agent_node_id,
                {
                    "type": agent_type,
                    "capabilities": capabilities,
                    "last_active": datetime.now().isoformat()
                },
                NodeType.AGENT
            )
            GRAPH_OPERATIONS.labels(operation_type="register_agent").inc()
        except Exception as e:
            self.logger.error(f"Failed to register agent: {str(e)}")
            raise

    async def record_code_change(self, file_path: str, change_type: str, description: str,
                                 agent_id: str, affected_components: List[str]):
        """Record a code change and its impacts."""
        try:
            change_id = f"change:{file_path}:{datetime.now().isoformat()}"

            # Record the change
            await self.add_node(
                change_id,
                {
                    "file_path": file_path,
                    "type": change_type,
                    "description": description,
                    "timestamp": datetime.now().isoformat()
                },
                NodeType.CHANGE
            )

            # Link change to agent
            await self.add_relationship(change_id, f"agent:{agent_id}", RelationType.MODIFIED_BY)

            # Analyze and record impacts
            impacts = await self._analyze_change_impact(file_path, affected_components)
            for impact in impacts:
                impact_id = f"impact:{impact['component']}:{datetime.now().isoformat()}"
                await self.add_node(
                    impact_id,
                    {
                        "component": impact["component"],
                        "severity": impact["severity"],
                        "description": impact["description"]
                    },
                    NodeType.IMPACT
                )
                await self.add_relationship(change_id, impact_id, RelationType.IMPACTS)

            self._change_history.append({
                "id": change_id,
                "file_path": file_path,
                "type": change_type,
                "timestamp": datetime.now().isoformat(),
                "impacts": impacts
            })

            GRAPH_OPERATIONS.labels(operation_type="record_change").inc()
            return change_id, impacts

        except Exception as e:
            self.logger.error(f"Failed to record code change: {str(e)}")
            raise

    async def add_insight(self, insight: CodeInsight):
        """Add a code insight to the graph."""
        try:
            insight_id = f"insight:{insight.type}:{datetime.now().isoformat()}"

            # Add insight node
            await self.add_node(
                insight_id,
                {
                    "type": insight.type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "timestamp": insight.timestamp.isoformat(),
                    "agent_id": insight.agent_id,
                    **insight.metadata
                },
                NodeType.INSIGHT
            )

            # Link insight to source nodes
            for source_node in insight.source_nodes:
                await self.add_relationship(insight_id, source_node, RelationType.RELATED_TO)

            # Cache the insight
            self._insight_cache[insight_id] = insight

            GRAPH_INSIGHTS.labels(insight_type=insight.type).inc()
            return insight_id

        except Exception as e:
            self.logger.error(f"Failed to add insight: {str(e)}")
            raise

    async def get_component_dependencies(self, component_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get dependencies for a component with specified depth."""
        try:
            deps = {
                "direct": [],
                "indirect": [],
                "reverse": [],
                "test_coverage": []
            }

            # Use NetworkX for efficient graph traversal
            for d in range(1, depth + 1):
                paths = nx.single_source_shortest_path(
                    self.graph, component_id, cutoff=d)
                for target, path in paths.items():
                    if target != component_id:
                        if d == 1:
                            deps["direct"].append({
                                "id": target,
                                "type": self.metadata[target].get("type"),
                                "relationship": self.graph[path[-2]][target]["type"]
                            })
                        else:
                            deps["indirect"].append({
                                "id": target,
                                "path": path,
                                "type": self.metadata[target].get("type")
                            })

            # Get reverse dependencies
            for node in self.graph.predecessors(component_id):
                deps["reverse"].append({
                    "id": node,
                    "type": self.metadata[node].get("type"),
                    "relationship": self.graph[node][component_id]["type"]
                })

            # Get associated tests
            if self.neo4j_enabled:
                async with self.driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (c {id: $component_id})<-[:TESTS]-(t:Test)
                        RETURN t
                        """,
                        component_id=component_id
                    )
                    deps["test_coverage"] = [record["t"] for record in await result.fetch()]

            return deps

        except Exception as e:
            self.logger.error(
                f"Failed to get component dependencies: {str(e)}")
            raise

    async def predict_change_impact(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict the impact of changing specified files."""
        try:
            impacts = []
            components_seen = set()
            
            for file_path in file_paths:
                # Get file node
                file_node = self._get_file_node(file_path)
                if not file_node:
                    continue
                    
                # Get direct dependencies
                deps = await self.get_component_dependencies(file_node["id"])
                
                # Analyze each dependency
                for dep in deps["direct"]:
                    if dep["id"] in components_seen:
                        continue
                        
                    components_seen.add(dep["id"])
                    
                    # Get historical impacts
                    historical_impacts = await self._get_historical_impacts(dep["id"])
                    
                    if historical_impacts:
                        # Calculate impact probability based on historical data
                        historical_impact_count = sum(
                            1 for change in historical_impacts
                            if any(impact["component"] == dep["id"] 
                                  for impact in change["impacts"])
                        )
                        impact_probability = historical_impact_count / len(historical_impacts)
                        
                        impacts.append({
                            "component": dep["id"],
                            "probability": impact_probability,
                            "type": dep["type"],
                            "severity": await self._calculate_impact_severity(dep)
                        })
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Failed to predict change impact: {str(e)}")
            return []

    async def _calculate_impact_severity(self, component: Dict[str, Any]) -> str:
        """Calculate potential impact severity of changes to a component."""
        try:
            # Get component criticality from MCP
            criticality = await self.mcp_client.get_component_criticality(
                component_id=component["id"],
                component_type=component["type"]
            )
            
            if criticality > 0.8:
                return "HIGH"
            elif criticality > 0.5:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            self.logger.error(f"Failed to calculate impact severity: {str(e)}")
            return "UNKNOWN"

    async def _get_historical_impacts(self, component_id: str) -> List[Dict[str, Any]]:
        """Get historical impact data for a component."""
        try:
            # Get impact history from MCP
            history = await self.mcp_client.get_impact_history(
                component_id=component_id
            )
            
            return history.get("impacts", [])
            
        except Exception as e:
            self.logger.error(f"Failed to get historical impacts: {str(e)}")
            return []

    async def verify_node(self, node_id: str, verification_data: Dict[str, Any]) -> bool:
        """Verify a node through MCP."""
        try:
            node = self._knowledge_fragments.get(node_id)
            if not node:
                return False
                
            # Submit verification to MCP
            verification_result = await self.mcp_client.verify_content(
                context=node.mcp_context,
                content=node.content,
                verification_data=verification_data
            )
            
            # Update node verification status
            node.verification_status = verification_result["status"]
            node.confidence = verification_result["confidence"]
            
            # Update graph
            self.graph.nodes[node_id]["data"]["status"] = verification_result["status"]
            self.graph.nodes[node_id]["data"]["confidence"] = verification_result["confidence"]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify node: {str(e)}")
            return False

    async def add_relationship(self, source_id: str, target_id: str, 
                             relationship_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a relationship between nodes."""
        try:
            # Verify relationship through MCP
            verification_result = await self.mcp_client.verify_content(
                context={
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relationship_type
                },
                content=metadata or {},
                verification_data={
                    "type": "relationship_verification",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if verification_result.get("verified", False):
                # Add to graph
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=relationship_type,
                    metadata=metadata or {},
                    verification=verification_result
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add relationship: {str(e)}")
            return False

    async def get_component_dependencies(self, component_id: str) -> Dict[str, Any]:
        """Get component dependencies with MCP verification."""
        try:
            # Get direct dependencies from graph
            direct_deps = list(self.graph.successors(component_id))
            
            # Get indirect dependencies
            indirect_deps = []
            for dep in direct_deps:
                indirect = list(self.graph.successors(dep))
                indirect_deps.extend(indirect)
            
            # Verify dependencies through MCP
            verification_result = await self.mcp_client.verify_content(
                context={"component_id": component_id},
                content={
                    "direct": direct_deps,
                    "indirect": indirect_deps
                },
                verification_data={
                    "type": "dependency_verification",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "direct": [
                    {
                        "id": dep,
                        "type": self.graph.nodes[dep].get("type", "unknown"),
                        "verification": verification_result.get("direct", {}).get(dep, {})
                    }
                    for dep in direct_deps
                ],
                "indirect": [
                    {
                        "id": dep,
                        "type": self.graph.nodes[dep].get("type", "unknown"),
                        "verification": verification_result.get("indirect", {}).get(dep, {})
                    }
                    for dep in indirect_deps
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get component dependencies: {str(e)}")
            return {"direct": [], "indirect": []}

    async def _analyze_change_impact(self, file_path: str, affected_components: List[str]) -> List[Dict[str, Any]]:
        """Analyze the impact of a code change."""
        impacts = []
        try:
            # Get component dependencies
            for component in affected_components:
                deps = await self.get_component_dependencies(component)

                # Analyze impact on each dependent component
                for dep in deps["direct"] + deps["indirect"] + deps["reverse"]:
                    impact_severity = "HIGH" if dep["type"] in [
                        NodeType.SERVICE, NodeType.ENTITY] else "MEDIUM"

                    impacts.append({
                        "component": dep["id"],
                        "severity": impact_severity,
                        "description": f"Change in {file_path} affects dependent {dep['type'].lower()}"
                    })

            return impacts

        except Exception as e:
            self.logger.error(f"Failed to analyze change impact: {str(e)}")
            return []

    async def add_knowledge_fragment(self, content: str, source_type: str,
                                     metadata: Dict[str, Any], relationships: List[Dict[str, str]]):
        """Add a new knowledge fragment to the graph."""
        try:
            fragment_id = f"fragment:{uuid.uuid4()}"

            # Create embedding for the content
            embedding = await self._create_embedding(content)

            fragment = KnowledgeFragment(
                id=fragment_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
                relationships=relationships,
                source_type=source_type,
                timestamp=datetime.now()
            )

            # Add to graph
            await self.add_node(
                fragment_id,
                {
                    "content": content,
                    "source_type": source_type,
                    "metadata": metadata,
                    "timestamp": fragment.timestamp.isoformat()
                },
                "KnowledgeFragment"
            )

            # Add relationships
            for rel in relationships:
                await self.add_relationship(
                    fragment_id,
                    rel["target_id"],
                    rel["type"]
                )

            # Cache fragment
            self._knowledge_fragments[fragment_id] = fragment
            if embedding is not None:
                self._embedding_cache[fragment_id] = embedding

            return fragment_id

        except Exception as e:
            self.logger.error(f"Failed to add knowledge fragment: {str(e)}")
            raise

    async def query_knowledge_graph(self, query: str, k: int = 5, strategy: str = "hybrid") -> List[GraphRAGResult]:
        """Query the knowledge graph using advanced RAG techniques.

        Strategies:
        - semantic: Pure semantic similarity search
        - structural: Graph structure-based search
        - hybrid: Combines semantic and structural relevance
        - contextual: Uses surrounding context for relevance
        """
        try:
            # Create query embedding
            query_embedding = await self._create_embedding(query)
            if query_embedding is None:
                return []

            # Get initial candidates using semantic search
            candidates = []
            for fragment_id, embedding in self._embedding_cache.items():
                similarity = cosine_similarity(
                    [query_embedding],
                    [embedding]
                )[0][0]

                if similarity >= self._relevance_threshold:
                    candidates.append((fragment_id, similarity))

            # Apply selected search strategy
            if strategy == "semantic":
                relevant_fragments = await self._semantic_search(candidates)
            elif strategy == "structural":
                relevant_fragments = await self._structural_search(candidates)
            elif strategy == "contextual":
                relevant_fragments = await self._contextual_search(candidates, query)
            else:  # hybrid
                relevant_fragments = await self._hybrid_search(candidates, query)

            # Sort by relevance and return top k
            relevant_fragments.sort(
                key=lambda x: x.relevance_score, reverse=True)
            return relevant_fragments[:k]

        except Exception as e:
            self.logger.error(f"Failed to query knowledge graph: {str(e)}")
            return []

    async def _semantic_search(self, candidates: List[Tuple[str, float]]) -> List[GraphRAGResult]:
        """Perform semantic similarity-based search."""
        results = []
        for fragment_id, similarity in candidates:
            fragment = self._knowledge_fragments[fragment_id]
            path = await self._find_path_to_root(fragment_id)

            results.append(GraphRAGResult(
                node_id=fragment_id,
                content=fragment.content,
                relevance_score=similarity,
                context=await self._get_fragment_context(fragment_id),
                path_to_root=path
            ))
        return results

    async def _structural_search(self, candidates: List[Tuple[str, float]]) -> List[GraphRAGResult]:
        """Perform graph structure-based search using centrality and connectivity."""
        results = []
        try:
            # Calculate centrality scores for candidate nodes
            subgraph = self.graph.subgraph([c[0] for c in candidates])
            centrality = nx.pagerank(subgraph)

            for fragment_id, semantic_score in candidates:
                # Combine semantic similarity with structural importance
                structural_score = centrality.get(fragment_id, 0)
                combined_score = 0.7 * semantic_score + 0.3 * structural_score

                fragment = self._knowledge_fragments[fragment_id]
                path = await self._find_path_to_root(fragment_id)

                results.append(GraphRAGResult(
                    node_id=fragment_id,
                    content=fragment.content,
                    relevance_score=combined_score,
                    context=await self._get_fragment_context(fragment_id),
                    path_to_root=path
                ))
        except Exception as e:
            self.logger.error(f"Failed in structural search: {str(e)}")
        return results

    async def _contextual_search(self, candidates: List[Tuple[str, float]], query: str) -> List[GraphRAGResult]:
        """Perform context-aware search considering neighborhood information."""
        results = []
        try:
            query_embedding = await self._create_embedding(query)

            for fragment_id, base_similarity in candidates:
                # Get neighborhood context
                neighbors = list(self.graph.neighbors(fragment_id))
                neighbor_fragments = [
                    self._knowledge_fragments[n] for n in neighbors
                    if n in self._knowledge_fragments
                ]

                # Create context embedding
                context_text = " ".join(
                    [f.content for f in neighbor_fragments])
                context_embedding = await self._create_embedding(context_text)

                if context_embedding is not None:
                    # Calculate context similarity
                    context_similarity = cosine_similarity(
                        [query_embedding],
                        [context_embedding]
                    )[0][0]

                    # Combine base similarity with context relevance
                    combined_score = 0.6 * base_similarity + 0.4 * context_similarity

                    fragment = self._knowledge_fragments[fragment_id]
                    path = await self._find_path_to_root(fragment_id)

                    results.append(GraphRAGResult(
                        node_id=fragment_id,
                        content=fragment.content,
                        relevance_score=combined_score,
                        context=await self._get_enhanced_context(fragment_id, neighbors),
                        path_to_root=path
                    ))

        except Exception as e:
            self.logger.error(f"Failed in contextual search: {str(e)}")
        return results

    async def _hybrid_search(self, candidates: List[Tuple[str, float]], query: str) -> List[GraphRAGResult]:
        """Perform hybrid search combining semantic, structural, and contextual signals."""
        results = []
        try:
            # Get results from each strategy
            semantic_results = await self._semantic_search(candidates)
            structural_results = await self._structural_search(candidates)
            contextual_results = await self._contextual_search(candidates, query)

            # Combine results with weighted scoring
            result_map = {}

            # Process semantic results
            for result in semantic_results:
                result_map[result.node_id] = {
                    'result': result,
                    'semantic_score': result.relevance_score
                }

            # Process structural results
            for result in structural_results:
                if result.node_id in result_map:
                    result_map[result.node_id]['structural_score'] = result.relevance_score
                else:
                    result_map[result.node_id] = {
                        'result': result,
                        'structural_score': result.relevance_score,
                        'semantic_score': 0
                    }

            # Process contextual results
            for result in contextual_results:
                if result.node_id in result_map:
                    result_map[result.node_id]['contextual_score'] = result.relevance_score
                else:
                    result_map[result.node_id] = {
                        'result': result,
                        'contextual_score': result.relevance_score,
                        'semantic_score': 0,
                        'structural_score': 0
                    }

            # Calculate final scores
            for node_id, scores in result_map.items():
                semantic_score = scores.get('semantic_score', 0)
                structural_score = scores.get('structural_score', 0)
                contextual_score = scores.get('contextual_score', 0)

                # Weighted combination of scores
                final_score = (
                    0.4 * semantic_score +
                    0.3 * structural_score +
                    0.3 * contextual_score
                )

                result = scores['result']
                result.relevance_score = final_score
                results.append(result)

        except Exception as e:
            self.logger.error(f"Failed in hybrid search: {str(e)}")
        return results

    async def _get_enhanced_context(self, fragment_id: str, neighbors: List[str]) -> Dict[str, Any]:
        """Get enhanced contextual information including neighborhood analysis."""
        try:
            context = await self._get_fragment_context(fragment_id)

            # Add neighborhood analysis
            neighborhood_stats = {
                "total_neighbors": len(neighbors),
                "neighbor_types": defaultdict(int),
                "relationship_types": defaultdict(int),
                "common_patterns": [],
                "semantic_clusters": []
            }

            # Analyze neighborhood
            for neighbor in neighbors:
                if neighbor in self._knowledge_fragments:
                    fragment = self._knowledge_fragments[neighbor]
                    neighborhood_stats["neighbor_types"][fragment.source_type] += 1

                    # Get relationship type
                    rel_type = self.graph[fragment_id][neighbor].get(
                        "type", "unknown")
                    neighborhood_stats["relationship_types"][rel_type] += 1

            # Find common patterns in neighborhood
            pattern_nodes = [n for n in neighbors if self.metadata.get(
                n, {}).get("type") == NodeType.PATTERN]
            if pattern_nodes:
                patterns = [self.metadata[n].get(
                    "name") for n in pattern_nodes]
                neighborhood_stats["common_patterns"] = list(set(patterns))

            # Find semantic clusters in neighborhood
            if len(neighbors) > 1:
                neighbor_fragments = [
                    self._knowledge_fragments[n] for n in neighbors
                    if n in self._knowledge_fragments
                ]
                semantic_groups = await self.analyze_code_semantics([
                    (f.id, f.content) for f in neighbor_fragments
                ])
                neighborhood_stats["semantic_clusters"] = [
                    {
                        "theme": group.common_theme,
                        "similarity": group.similarity_score,
                        "members": group.elements
                    }
                    for group in semantic_groups
                ]

            context["neighborhood_analysis"] = neighborhood_stats
            return context

        except Exception as e:
            self.logger.error(f"Failed to get enhanced context: {str(e)}")
            return await self._get_fragment_context(fragment_id)

    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text using configured model."""
        try:
            # For now, using TF-IDF as a simple embedding
            # In production, you'd want to use a more sophisticated embedding model
            vector = self._vectorizer.fit_transform([text]).toarray()[0]
            return vector.tolist()
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {str(e)}")
            return None

    async def _get_fragment_context(self, fragment_id: str) -> Dict[str, Any]:
        """Get contextual information for a knowledge fragment."""
        try:
            context = {
                "related_fragments": [],
                "agent_insights": [],
                "dependencies": []
            }

            # Get related fragments
            related = await self.get_related_nodes(fragment_id)
            for node_id in related[:self._max_context_items]:
                if node_id in self._knowledge_fragments:
                    context["related_fragments"].append({
                        "id": node_id,
                        "content": self._knowledge_fragments[node_id].content,
                        "type": self._knowledge_fragments[node_id].source_type
                    })

            # Get agent insights
            insights = await self.get_insights_for_component(fragment_id)
            context["agent_insights"] = [
                {
                    "type": insight.type,
                    "description": insight.description,
                    "confidence": insight.confidence
                }
                for insight in insights[:self._max_context_items]
            ]

            # Get dependencies if it's a code fragment
            if self._knowledge_fragments[fragment_id].source_type == "code":
                deps = await self.get_component_dependencies(fragment_id)
                context["dependencies"] = deps["direct"]

            return context

        except Exception as e:
            self.logger.error(f"Failed to get fragment context: {str(e)}")
            return {}

    async def _find_path_to_root(self, node_id: str) -> List[str]:
        """Find path from node to repository root."""
        try:
            paths = []
            current = node_id

            while current:
                paths.append(current)
                parents = [n for n in self.graph.predecessors(current)
                           if self.metadata[n].get("type") in
                           [NodeType.REPOSITORY, NodeType.COMPONENT, NodeType.FILE]]
                if not parents:
                    break
                current = parents[0]

            return list(reversed(paths))

        except Exception as e:
            self.logger.error(f"Failed to find path to root: {str(e)}")
            return [node_id]

    async def register_json_agent(self, agent_type: str, capabilities: List[str], schema: Dict[str, Any]):
        """Register a JSON agent with its schema and capabilities."""
        try:
            # Validate schema
            if not all(key in schema for key in ["properties", "required"]):
                raise ValueError("Invalid agent schema")

            agent_id = f"agent:{agent_type}:{uuid.uuid4()}"

            # Register agent
            await self.register_agent(agent_id, agent_type, capabilities)

            # Store schema
            self._agent_schemas[agent_type] = schema
            self._agent_capabilities[agent_type].update(capabilities)

            return agent_id

        except Exception as e:
            self.logger.error(f"Failed to register JSON agent: {str(e)}")
            raise

    async def process_agent_interaction(self, agent_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction from a JSON agent."""
        try:
            # Validate interaction against schema
            agent_type = agent_id.split(":")[1]
            schema = self._agent_schemas.get(agent_type)
            if not schema:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Record interaction
            interaction_id = str(uuid.uuid4())
            self._interaction_history.append({
                "id": interaction_id,
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "data": interaction_data
            })

            # Process interaction based on type
            if "query" in interaction_data:
                # Knowledge query
                results = await self.query_knowledge_graph(
                    interaction_data["query"],
                    k=interaction_data.get("max_results", 5)
                )
                return {
                    "interaction_id": interaction_id,
                    "results": [
                        {
                            "content": r.content,
                            "relevance": r.relevance_score,
                            "context": r.context
                        }
                        for r in results
                    ]
                }
            elif "insight" in interaction_data:
                # Add insight
                insight = CodeInsight(
                    type=interaction_data["insight"]["type"],
                    description=interaction_data["insight"]["description"],
                    confidence=interaction_data["insight"]["confidence"],
                    source_nodes=interaction_data["insight"]["source_nodes"],
                    timestamp=datetime.now(),
                    agent_id=agent_id,
                    metadata=interaction_data["insight"].get("metadata", {})
                )
                insight_id = await self.add_insight(insight)
                return {
                    "interaction_id": interaction_id,
                    "insight_id": insight_id
                }
            else:
                raise ValueError("Unknown interaction type")

        except Exception as e:
            self.logger.error(f"Failed to process agent interaction: {str(e)}")
            raise

    async def add_requirement(self, description: str, metadata: Dict[str, Any]) -> str:
        """Add a new requirement to track and validate."""
        try:
            requirement_id = f"req:{uuid.uuid4()}"

            # Add requirement node
            await self.add_node(
                requirement_id,
                {
                    "type": "requirement",
                    "description": description,
                    "status": "pending_validation",
                    "confidence": 0.0,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                },
                "Requirement"
            )

            # Initialize validation tracking
            self._requirements[requirement_id] = {
                "description": description,
                "validations": [],
                "confidence_history": [],
                "affected_components": [],
                "metadata": metadata
            }

            # Schedule initial validation
            await self._schedule_validation(requirement_id)

            return requirement_id

        except Exception as e:
            self.logger.error(f"Failed to add requirement: {str(e)}")
            raise

    async def add_code_assumption(self, description: str, components: List[str],
                                  dependencies: List[str]) -> str:
        """Add a new code state assumption to track."""
        try:
            assumption_id = f"assumption:{uuid.uuid4()}"

            # Add assumption node
            await self.add_node(
                assumption_id,
                {
                    "type": "assumption",
                    "description": description,
                    "status": "unverified",
                    "confidence": 0.0,
                    "components": components,
                    "timestamp": datetime.now().isoformat()
                },
                "Assumption"
            )

            # Link to affected components
            for component in components:
                await self.add_relationship(
                    assumption_id,
                    component,
                    "AFFECTS"
                )

            # Track dependencies
            self._assumption_dependencies.add_node(assumption_id)
            for dep in dependencies:
                self._assumption_dependencies.add_edge(assumption_id, dep)

            # Initialize assumption tracking
            self._assumptions[assumption_id] = CodeStateAssumption(
                assumption_id=assumption_id,
                description=description,
                confidence_score=0.0,
                affected_components=components,
                validation_status="pending",
                last_validated=datetime.now(),
                dependencies=dependencies,
                metadata={}
            )

            # Schedule validation
            await self._schedule_validation(assumption_id)

            return assumption_id

        except Exception as e:
            self.logger.error(f"Failed to add assumption: {str(e)}")
            raise

    async def validate_requirement(self, requirement_id: str,
                                   strategies: Optional[List[ValidationStrategy]] = None) -> ValidationResult:
        """Validate a requirement using specified strategies."""
        try:
            if not strategies:
                strategies = list(ValidationStrategy)

            results = []
            total_confidence = 0.0

            for strategy in strategies:
                result = await self._apply_validation_strategy(
                    requirement_id,
                    strategy
                )
                results.append(result)
                total_confidence += result.confidence

            # Aggregate results
            overall_confidence = total_confidence / len(results)
            success = overall_confidence >= self._confidence_threshold

            # Collect evidence and impacts
            evidence = []
            impacts = defaultdict(list)
            actions = []

            for result in results:
                evidence.extend(result.evidence)
                for component, impact in result.impact_assessment.items():
                    impacts[component].append(impact)
                actions.extend(result.suggested_actions)

            # Update requirement status
            await self._update_requirement_status(
                requirement_id,
                success,
                overall_confidence,
                evidence
            )

            return ValidationResult(
                success=success,
                confidence=overall_confidence,
                evidence=evidence,
                impact_assessment=dict(impacts),
                suggested_actions=actions,
                metadata={
                    "strategies_used": [s.value for s in strategies],
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to validate requirement: {str(e)}")
            raise

    async def validate_assumption(self, assumption_id: str) -> ValidationResult:
        """Validate a code state assumption."""
        try:
            assumption = self._assumptions[assumption_id]

            # Check dependencies first
            for dep_id in assumption.dependencies:
                dep_result = await self.validate_assumption(dep_id)
                if not dep_result.success:
                    return ValidationResult(
                        success=False,
                        confidence=0.0,
                        evidence=[{
                            "type": "dependency_failure",
                            "dependency": dep_id,
                            "details": dep_result.evidence
                        }],
                        impact_assessment={},
                        suggested_actions=[{
                            "action": "validate_dependency",
                            "dependency": dep_id
                        }],
                        metadata={"failed_dependency": dep_id}
                    )

            # Perform validation
            validations = []

            # Static analysis
            static_result = await self._static_assumption_check(assumption)
            validations.append(static_result)

            # Semantic analysis
            semantic_result = await self._semantic_assumption_check(assumption)
            validations.append(semantic_result)

            # Dynamic analysis if possible
            if self._can_perform_dynamic_validation(assumption):
                dynamic_result = await self._dynamic_assumption_check(assumption)
                validations.append(dynamic_result)

            # Calculate overall confidence
            total_confidence = sum(v.confidence for v in validations)
            overall_confidence = total_confidence / len(validations)

            # Determine success
            success = (
                overall_confidence >= self._confidence_threshold and
                len(validations) >= self._min_validations_required
            )

            # Collect evidence and impacts
            evidence = []
            impacts = defaultdict(list)
            actions = []

            for validation in validations:
                evidence.extend(validation.evidence)
                for component, impact in validation.impact_assessment.items():
                    impacts[component].append(impact)
                actions.extend(validation.suggested_actions)

            # Update assumption status
            assumption.confidence_score = overall_confidence
            assumption.validation_status = "valid" if success else "invalid"
            assumption.last_validated = datetime.now()

            return ValidationResult(
                success=success,
                confidence=overall_confidence,
                evidence=evidence,
                impact_assessment=dict(impacts),
                suggested_actions=actions,
                metadata={
                    "validation_count": len(validations),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to validate assumption: {str(e)}")
            raise

    async def _apply_validation_strategy(self, requirement_id: str,
                                         strategy: ValidationStrategy) -> ValidationResult:
        """Apply a specific validation strategy."""
        try:
            requirement = self._requirements[requirement_id]

            if strategy == ValidationStrategy.STATIC_ANALYSIS:
                return await self._static_requirement_check(requirement)
            elif strategy == ValidationStrategy.DYNAMIC_TESTING:
                return await self._dynamic_requirement_check(requirement)
            elif strategy == ValidationStrategy.SEMANTIC_ANALYSIS:
                return await self._semantic_requirement_check(requirement)
            elif strategy == ValidationStrategy.REQUIREMENT_TRACING:
                return await self._trace_requirement_implementation(requirement)
            elif strategy == ValidationStrategy.ASSUMPTION_CHECKING:
                return await self._validate_requirement_assumptions(requirement)
            elif strategy == ValidationStrategy.IMPACT_ANALYSIS:
                return await self._analyze_requirement_impact(requirement)
            else:
                raise ValueError(f"Unknown validation strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"Failed to apply validation strategy: {str(e)}")
            raise

    async def _static_requirement_check(self, requirement: Dict[str, Any]) -> ValidationResult:
        """Perform static analysis validation of a requirement."""
        try:
            # Analyze code patterns and structure
            affected_components = await self._find_affected_components(requirement)
            pattern_matches = await self._analyze_code_patterns(affected_components)
            structural_validity = await self._check_structural_validity(affected_components)

            # Calculate confidence based on static analysis
            confidence = (
                pattern_matches["confidence"] * 0.6 +
                structural_validity["confidence"] * 0.4
            )

            return ValidationResult(
                success=confidence >= self._confidence_threshold,
                confidence=confidence,
                evidence=[pattern_matches, structural_validity],
                impact_assessment=pattern_matches["impacts"],
                suggested_actions=structural_validity["suggestions"],
                metadata={"validation_type": "static"}
            )

        except Exception as e:
            self.logger.error(f"Failed in static requirement check: {str(e)}")
            raise

    async def _semantic_requirement_check(self, requirement: Dict[str, Any]) -> ValidationResult:
        """Perform semantic analysis validation of a requirement."""
        try:
            # Analyze semantic relationships and consistency
            semantic_groups = await self.analyze_code_semantics([
                (c["id"], c["content"])
                for c in requirement.get("affected_components", [])
            ])

            consistency_score = await self._analyze_semantic_consistency(
                requirement["description"],
                semantic_groups
            )

            return ValidationResult(
                success=consistency_score >= self._confidence_threshold,
                confidence=consistency_score,
                evidence=[{
                    "type": "semantic_analysis",
                    "semantic_groups": semantic_groups,
                    "consistency_score": consistency_score
                }],
                impact_assessment={
                    "semantic_impact": await self._assess_semantic_impact(semantic_groups)
                },
                suggested_actions=await self._generate_semantic_suggestions(
                    semantic_groups,
                    consistency_score
                ),
                metadata={"validation_type": "semantic"}
            )

        except Exception as e:
            self.logger.error(
                f"Failed in semantic requirement check: {str(e)}")
            raise

    async def _update_requirement_status(self, requirement_id: str, success: bool,
                                         confidence: float, evidence: List[Dict[str, Any]]):
        """Update requirement status based on validation results."""
        try:
            requirement = self._requirements[requirement_id]

            # Update validation history
            requirement["validations"].append({
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "confidence": confidence,
                "evidence": evidence
            })

            # Update confidence history
            requirement["confidence_history"].append({
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            })

            # Update node in graph
            await self.add_node(
                requirement_id,
                {
                    **requirement,
                    "status": "validated" if success else "failed_validation",
                    "confidence": confidence,
                    "last_validated": datetime.now().isoformat()
                },
                "Requirement"
            )

            # Schedule next validation if needed
            if not success or confidence < self._confidence_threshold:
                await self._schedule_validation(requirement_id)

        except Exception as e:
            self.logger.error(f"Failed to update requirement status: {str(e)}")
            raise

    async def _schedule_validation(self, target_id: str):
        """Schedule validation for a requirement or assumption."""
        try:
            await self._analysis_queue.put({
                "id": target_id,
                "type": "requirement" if target_id.startswith("req:") else "assumption",
                "priority": self._calculate_validation_priority(target_id),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Failed to schedule validation: {str(e)}")

    def _calculate_validation_priority(self, target_id: str) -> float:
        """Calculate validation priority based on various factors."""
        try:
            if target_id.startswith("req:"):
                requirement = self._requirements[target_id]
                return self._calculate_requirement_priority(requirement)
            else:
                assumption = self._assumptions[target_id]
                return self._calculate_assumption_priority(assumption)
        except Exception as e:
            self.logger.error(
                f"Failed to calculate validation priority: {str(e)}")
            return 0.0

    async def start_continuous_validation(self):
        """Start continuous validation process."""
        try:
            while True:
                validation_task = await self._analysis_queue.get()

                if validation_task["type"] == "requirement":
                    result = await self.validate_requirement(validation_task["id"])
                else:
                    result = await self.validate_assumption(validation_task["id"])

                self._validation_history.append({
                    **validation_task,
                    "result": result,
                    "completed_at": datetime.now().isoformat()
                })

                self._analysis_queue.task_done()

        except Exception as e:
            self.logger.error(f"Error in continuous validation: {str(e)}")

    async def _analyze_semantic_consistency(self, description: str, semantic_groups: List[SemanticGroup]) -> float:
        """Analyze semantic consistency between requirement and implementation."""
        try:
            # Create embeddings
            req_embedding = await self._create_embedding(description)

            # Analyze each semantic group
            consistency_scores = []
            for group in semantic_groups:
                # Create group embedding from content
                group_content = " ".join([
                    self._knowledge_fragments[elem].content
                    for elem in group.elements
                    if elem in self._knowledge_fragments
                ])
                group_embedding = await self._create_embedding(group_content)

                if group_embedding is not None:
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        [req_embedding],
                        [group_embedding]
                    )[0][0]

                    # Weight by group's internal consistency
                    weighted_score = similarity * group.similarity_score
                    consistency_scores.append(weighted_score)

            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

        except Exception as e:
            self.logger.error(
                f"Failed to analyze semantic consistency: {str(e)}")
            return 0.0

    async def _analyze_code_patterns(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze code patterns in affected components."""
        try:
            pattern_analysis = {
                "patterns": [],
                "confidence": 0.0,
                "impacts": {},
                "suggestions": []
            }

            for component in components:
                # Detect patterns
                patterns = await self.detect_patterns(
                    component.get("content", ""),
                    {"file": component.get("id")}
                )

                for pattern in patterns:
                    pattern_info = {
                        "pattern": pattern.pattern_type,
                        "confidence": pattern.confidence,
                        "components": pattern.components,
                        "location": component.get("id")
                    }
                    pattern_analysis["patterns"].append(pattern_info)

                    # Assess impact on requirements
                    impact = await self._assess_pattern_impact(
                        pattern,
                        component.get("id")
                    )
                    pattern_analysis["impacts"][component.get("id")] = impact

                    # Generate suggestions if needed
                    if pattern.confidence < self._confidence_threshold:
                        suggestions = await self._generate_pattern_suggestions(
                            pattern,
                            component.get("id")
                        )
                        pattern_analysis["suggestions"].extend(suggestions)

            # Calculate overall confidence
            if pattern_analysis["patterns"]:
                pattern_analysis["confidence"] = sum(
                    p["confidence"] for p in pattern_analysis["patterns"]
                ) / len(pattern_analysis["patterns"])

            return pattern_analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze code patterns: {str(e)}")
            return {"patterns": [], "confidence": 0.0, "impacts": {}, "suggestions": []}

    async def _check_structural_validity(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check structural validity of components."""
        try:
            validity_analysis = {
                "valid": True,
                "confidence": 0.0,
                "violations": [],
                "suggestions": []
            }

            # Build dependency graph
            dep_graph = nx.DiGraph()
            for component in components:
                deps = await self.get_component_dependencies(component.get("id"))
                for dep in deps["direct"]:
                    dep_graph.add_edge(component.get("id"), dep["id"])

            # Check for cycles
            try:
                cycles = list(nx.simple_cycles(dep_graph))
                if cycles:
                    validity_analysis["valid"] = False
                    validity_analysis["violations"].append({
                        "type": "circular_dependency",
                        "components": cycles
                    })
                    validity_analysis["suggestions"].append({
                        "type": "break_cycle",
                        "description": "Break circular dependencies",
                        "cycles": cycles
                    })
            except Exception:
                pass

            # Check layering violations
            layers = self._analyze_architectural_layers(components)
            violations = self._check_layer_violations(layers)
            if violations:
                validity_analysis["valid"] = False
                validity_analysis["violations"].extend(violations)
                validity_analysis["suggestions"].extend(
                    self._generate_layer_suggestions(violations)
                )

            # Calculate confidence based on various factors
            factors = [
                not bool(cycles),  # No cycles
                len(violations) == 0,  # No layer violations
                self._check_naming_consistency(
                    components),  # Consistent naming
                self._check_interface_consistency(
                    components)  # Consistent interfaces
            ]
            validity_analysis["confidence"] = sum(
                1 for f in factors if f) / len(factors)

            return validity_analysis

        except Exception as e:
            self.logger.error(f"Failed to check structural validity: {str(e)}")
            return {"valid": False, "confidence": 0.0, "violations": [], "suggestions": []}

    async def _assess_pattern_impact(self, pattern: PatternInstance, component_id: str) -> Dict[str, Any]:
        """Assess the impact of a pattern on requirements and assumptions."""
        try:
            impact = {
                "requirements": [],
                "assumptions": [],
                "confidence": pattern.confidence,
                "risk_level": "LOW"
            }

            # Find affected requirements
            for req_id, req_data in self._requirements.items():
                if component_id in req_data.get("affected_components", []):
                    # Check if pattern supports or violates requirement
                    consistency = await self._check_pattern_requirement_consistency(
                        pattern,
                        req_data
                    )
                    impact["requirements"].append({
                        "requirement_id": req_id,
                        "consistency": consistency,
                        "confidence": pattern.confidence
                    })

            # Find affected assumptions
            for assumption_id, assumption in self._assumptions.items():
                if component_id in assumption.affected_components:
                    # Check if pattern validates or invalidates assumption
                    validation = await self._check_pattern_assumption_consistency(
                        pattern,
                        assumption
                    )
                    impact["assumptions"].append({
                        "assumption_id": assumption_id,
                        "validation": validation,
                        "confidence": pattern.confidence
                    })

            # Calculate risk level
            if any(r["consistency"] < 0.5 for r in impact["requirements"]):
                impact["risk_level"] = "HIGH"
            elif any(a["validation"] < 0.5 for a in impact["assumptions"]):
                impact["risk_level"] = "MEDIUM"

            return impact

        except Exception as e:
            self.logger.error(f"Failed to assess pattern impact: {str(e)}")
            return {"requirements": [], "assumptions": [], "confidence": 0.0, "risk_level": "UNKNOWN"}

    async def _generate_pattern_suggestions(self, pattern: PatternInstance,
                                            component_id: str) -> List[Dict[str, Any]]:
        """Generate suggestions for improving pattern implementation."""
        try:
            suggestions = []

            # Check pattern completeness
            missing_components = await self._check_pattern_completeness(
                pattern,
                component_id
            )
            if missing_components:
                suggestions.append({
                    "type": "missing_components",
                    "pattern": pattern.pattern_type,
                    "missing": missing_components,
                    "priority": "HIGH"
                })

            # Check pattern best practices
            violations = await self._check_pattern_best_practices(
                pattern,
                component_id
            )
            for violation in violations:
                suggestions.append({
                    "type": "best_practice_violation",
                    "pattern": pattern.pattern_type,
                    "violation": violation,
                    "priority": "MEDIUM"
                })

            # Check pattern consistency
            inconsistencies = await self._check_pattern_consistency(
                pattern,
                component_id
            )
            for inconsistency in inconsistencies:
                suggestions.append({
                    "type": "pattern_inconsistency",
                    "pattern": pattern.pattern_type,
                    "inconsistency": inconsistency,
                    "priority": "HIGH"
                })

            return suggestions

        except Exception as e:
            self.logger.error(
                f"Failed to generate pattern suggestions: {str(e)}")
            return []

    def _analyze_architectural_layers(self, components: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze architectural layering of components."""
        try:
            layers = defaultdict(list)

            # Define layer patterns
            layer_patterns = {
                "presentation": ["controller", "view", "ui"],
                "application": ["service", "application", "usecase"],
                "domain": ["domain", "model", "entity"],
                "infrastructure": ["repository", "dao", "infrastructure"]
            }

            # Categorize components into layers
            for component in components:
                component_path = component.get("id", "").lower()
                assigned = False

                for layer, patterns in layer_patterns.items():
                    if any(pattern in component_path for pattern in patterns):
                        layers[layer].append(component.get("id"))
                        assigned = True
                        break

                if not assigned:
                    layers["unknown"].append(component.get("id"))

            return dict(layers)

        except Exception as e:
            self.logger.error(
                f"Failed to analyze architectural layers: {str(e)}")
            return {}

    def _check_layer_violations(self, layers: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Check for architectural layer violations."""
        try:
            violations = []

            # Define layer dependencies (lower layers can't depend on upper layers)
            layer_order = ["infrastructure", "domain",
                           "application", "presentation"]

            for higher_idx, higher_layer in enumerate(layer_order):
                higher_components = set(layers.get(higher_layer, []))

                # Check dependencies on higher layers
                for lower_layer in layer_order[higher_idx+1:]:
                    lower_components = set(layers.get(lower_layer, []))

                    for component in higher_components:
                        deps = self.graph.successors(component)
                        violations.extend([
                            {
                                "type": "layer_violation",
                                "from_layer": higher_layer,
                                "to_layer": lower_layer,
                                "from_component": component,
                                "to_component": dep
                            }
                            for dep in deps if dep in lower_components
                        ])

            return violations

        except Exception as e:
            self.logger.error(f"Failed to check layer violations: {str(e)}")
            return []

    def _check_naming_consistency(self, components: List[Dict[str, Any]]) -> bool:
        """Check naming consistency across components."""
        try:
            # Extract component names
            names = [c.get("id", "").split("/")[-1] for c in components]

            # Check naming patterns
            patterns = {
                "service": r".*Service$",
                "repository": r".*Repository$",
                "controller": r".*Controller$",
                "entity": r".*Entity$"
            }

            consistent = True
            for pattern_name, pattern in patterns.items():
                matches = [n for n in names if re.search(pattern, n)]
                if matches and len(matches) != len([n for n in names if pattern_name.lower() in n.lower()]):
                    consistent = False
                    break

            return consistent

        except Exception as e:
            self.logger.error(f"Failed to check naming consistency: {str(e)}")
            return False

    def _check_interface_consistency(self, components: List[Dict[str, Any]]) -> bool:
        """Check interface consistency across similar components."""
        try:
            # Group similar components
            component_groups = defaultdict(list)
            for component in components:
                name = component.get("id", "").split("/")[-1]
                for type_name in ["Service", "Repository", "Controller"]:
                    if type_name in name:
                        component_groups[type_name].append(component)
                        break

            # Check interface consistency within groups
            consistent = True
            for group_type, group_components in component_groups.items():
                if len(group_components) <= 1:
                    continue

                # Compare method signatures
                base_methods = set(self._extract_methods(group_components[0]))
                for component in group_components[1:]:
                    methods = set(self._extract_methods(component))
                    if not (base_methods & methods):  # No common methods
                        consistent = False
                        break

            return consistent

        except Exception as e:
            self.logger.error(
                f"Failed to check interface consistency: {str(e)}")
            return False

    def _extract_methods(self, component: Dict[str, Any]) -> Set[str]:
        """Extract method signatures from a component."""
        try:
            methods = set()
            content = component.get("content", "")

            # Simple method extraction (can be enhanced)
            method_pattern = r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)"
            matches = re.finditer(method_pattern, content)
            methods.update(m.group(1) for m in matches)

            return methods

        except Exception as e:
            self.logger.error(f"Failed to extract methods: {str(e)}")
            return set()

    async def parse_gherkin_requirements(self, gherkin_text: str) -> List[BehaviorScenario]:
        """Parse Gherkin requirements into structured scenarios."""
        try:
            # Parse Gherkin text
            feature = gherkin_parser.parse(gherkin_text)
            scenarios = []

            for scenario in feature.get("scenarios", []):
                # Extract steps
                given_steps = [step["text"]
                               for step in scenario.get("given", [])]
                when_steps = [step["text"]
                              for step in scenario.get("when", [])]
                then_steps = [step["text"]
                              for step in scenario.get("then", [])]

                # Create scenario object
                behavior_scenario = BehaviorScenario(
                    scenario_id=f"scenario:{uuid.uuid4()}",
                    feature=feature.get("name", ""),
                    scenario=scenario.get("name", ""),
                    given=given_steps,
                    when=when_steps,
                    then=then_steps,
                    examples=scenario.get("examples", []),
                    tags=scenario.get("tags", []),
                    metadata={
                        "description": scenario.get("description", ""),
                        "line": scenario.get("line", 0),
                        "type": scenario.get("type", "scenario")
                    }
                )
                scenarios.append(behavior_scenario)

                # Add to knowledge graph
                await self._add_scenario_to_graph(behavior_scenario)

            return scenarios

        except Exception as e:
            self.logger.error(
                f"Failed to parse Gherkin requirements: {str(e)}")
            return []

    async def _add_scenario_to_graph(self, scenario: BehaviorScenario):
        """Add behavior scenario to knowledge graph."""
        try:
            # Add scenario node
            await self.add_node(
                scenario.scenario_id,
                {
                    "type": "behavior_scenario",
                    "feature": scenario.feature,
                    "scenario": scenario.scenario,
                    "steps": {
                        "given": scenario.given,
                        "when": scenario.when,
                        "then": scenario.then
                    },
                    "examples": scenario.examples,
                    "tags": scenario.tags,
                    "metadata": scenario.metadata
                },
                "BehaviorScenario"
            )

            # Link to related components
            affected_components = await self._identify_affected_components(scenario)
            for component_id in affected_components:
                await self.add_relationship(
                    scenario.scenario_id,
                    component_id,
                    "IMPLEMENTS"
                )

        except Exception as e:
            self.logger.error(f"Failed to add scenario to graph: {str(e)}")

    async def deep_code_comprehension(self, component_id: str,
                                      strategies: Optional[List[ComprehensionStrategy]] = None) -> CodeUnderstanding:
        """Perform deep code comprehension of a component."""
        try:
            if not strategies:
                strategies = list(ComprehensionStrategy)

            understanding = CodeUnderstanding(
                component_id=component_id,
                behavioral_patterns=[],
                data_flows=[],
                control_flows=[],
                state_transitions=[],
                invariants=[],
                cross_cutting_concerns=[],
                architectural_roles=[],
                domain_concepts=[],
                confidence=0.0
            )

            for strategy in strategies:
                if strategy == ComprehensionStrategy.BEHAVIORAL:
                    patterns = await self._analyze_behavioral_patterns(component_id)
                    understanding.behavioral_patterns.extend(patterns)

                elif strategy == ComprehensionStrategy.STRUCTURAL:
                    flows = await self._analyze_flows(component_id)
                    understanding.data_flows.extend(flows["data_flows"])
                    understanding.control_flows.extend(flows["control_flows"])

                elif strategy == ComprehensionStrategy.SEMANTIC:
                    concepts = await self._analyze_domain_concepts(component_id)
                    understanding.domain_concepts.extend(concepts)

                elif strategy == ComprehensionStrategy.TEMPORAL:
                    transitions = await self._analyze_state_transitions(component_id)
                    understanding.state_transitions.extend(transitions)

                elif strategy == ComprehensionStrategy.CAUSAL:
                    invariants = await self._analyze_invariants(component_id)
                    understanding.invariants.extend(invariants)

                elif strategy == ComprehensionStrategy.DOMAIN:
                    roles = await self._analyze_architectural_roles(component_id)
                    understanding.architectural_roles.extend(roles)

                # Analyze cross-cutting concerns
                concerns = await self._analyze_cross_cutting_concerns(component_id)
                understanding.cross_cutting_concerns.extend(concerns)

            # Calculate overall confidence
            confidence_scores = []
            if understanding.behavioral_patterns:
                confidence_scores.append(
                    sum(p["confidence"] for p in understanding.behavioral_patterns) /
                    len(understanding.behavioral_patterns)
                )
            if understanding.data_flows:
                confidence_scores.append(
                    sum(f["confidence"] for f in understanding.data_flows) /
                    len(understanding.data_flows)
                )
            if understanding.control_flows:
                confidence_scores.append(
                    sum(f["confidence"] for f in understanding.control_flows) /
                    len(understanding.control_flows)
                )
            if understanding.state_transitions:
                confidence_scores.append(
                    sum(t["confidence"] for t in understanding.state_transitions) /
                    len(understanding.state_transitions)
                )
            if understanding.invariants:
                confidence_scores.append(
                    sum(i["confidence"] for i in understanding.invariants) /
                    len(understanding.invariants)
                )
            if understanding.domain_concepts:
                confidence_scores.append(
                    sum(c["confidence"] for c in understanding.domain_concepts) /
                    len(understanding.domain_concepts)
                )
            if understanding.cross_cutting_concerns:
                confidence_scores.append(
                    sum(c["confidence"] for c in understanding.cross_cutting_concerns) /
                    len(understanding.cross_cutting_concerns)
                )

            understanding.confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores else 0.0
            )

            return understanding

        except Exception as e:
            self.logger.error(f"Failed in deep code comprehension: {str(e)}")
            return None

    async def _analyze_cross_cutting_concerns(self, component_id: str) -> List[Dict[str, Any]]:
        """Analyze cross-cutting concerns in component."""
        try:
            concerns = []
            component = self._knowledge_fragments.get(component_id)
            if not component:
                return concerns

            # Analyze logging patterns
            logging_concerns = await self._analyze_logging_patterns(component)
            concerns.extend([
                {
                    "type": "logging",
                    "pattern": concern["pattern"],
                    "locations": concern["locations"],
                    "level": concern["level"],
                    "confidence": concern["confidence"]
                }
                for concern in logging_concerns
            ])

            # Analyze security patterns
            security_concerns = await self._analyze_security_patterns(component)
            concerns.extend([
                {
                    "type": "security",
                    "pattern": concern["pattern"],
                    "mechanism": concern["mechanism"],
                    "scope": concern["scope"],
                    "confidence": concern["confidence"]
                }
                for concern in security_concerns
            ])

            # Analyze error handling
            error_concerns = await self._analyze_error_handling(component)
            concerns.extend([
                {
                    "type": "error_handling",
                    "pattern": concern["pattern"],
                    "strategy": concern["strategy"],
                    "recovery": concern["recovery"],
                    "confidence": concern["confidence"]
                }
                for concern in error_concerns
            ])

            # Analyze performance aspects
            performance_concerns = await self._analyze_performance_patterns(component)
            concerns.extend([
                {
                    "type": "performance",
                    "pattern": concern["pattern"],
                    "optimization": concern["optimization"],
                    "impact": concern["impact"],
                    "confidence": concern["confidence"]
                }
                for concern in performance_concerns
            ])

            # Analyze transaction management
            transaction_concerns = await self._analyze_transaction_patterns(component)
            concerns.extend([
                {
                    "type": "transaction",
                    "pattern": concern["pattern"],
                    "scope": concern["scope"],
                    "isolation": concern["isolation"],
                    "confidence": concern["confidence"]
                }
                for concern in transaction_concerns
            ])

            return concerns

        except Exception as e:
            self.logger.error(
                f"Failed to analyze cross-cutting concerns: {str(e)}")
            return []

    async def _analyze_logging_patterns(self, component: Any) -> List[Dict[str, Any]]:
        """Analyze logging patterns in component."""
        try:
            patterns = []
            content = component.content

            # Look for logging statements
            log_patterns = {
                r'log\.(debug|info|warning|error|critical)': 'python_logging',
                r'console\.(log|info|warn|error)': 'javascript_console',
                r'logger\.(debug|info|warn|error)': 'logger_framework'
            }

            for pattern, log_type in log_patterns.items():
                matches = re.finditer(pattern, content)
                for match in matches:
                    patterns.append({
                        "pattern": log_type,
                        "locations": [match.start()],
                        "level": match.group(1),
                        "confidence": 0.9
                    })

            return patterns

        except Exception as e:
            self.logger.error(f"Failed to analyze logging patterns: {str(e)}")
            return []

    async def _analyze_security_patterns(self, component: Any) -> List[Dict[str, Any]]:
        """Analyze security patterns in component."""
        try:
            patterns = []
            content = component.content

            # Look for security mechanisms
            security_patterns = {
                r'@authenticate': ('authentication', 'decorator'),
                r'@authorize': ('authorization', 'decorator'),
                r'encrypt\(': ('encryption', 'function'),
                r'decrypt\(': ('decryption', 'function'),
                r'validate\(': ('input_validation', 'function'),
                r'sanitize\(': ('sanitization', 'function')
            }

            for pattern, (mechanism, pattern_type) in security_patterns.items():
                if re.search(pattern, content):
                    patterns.append({
                        "pattern": pattern_type,
                        "mechanism": mechanism,
                        "scope": "method" if pattern_type == "decorator" else "data",
                        "confidence": 0.85
                    })

            return patterns

        except Exception as e:
            self.logger.error(f"Failed to analyze security patterns: {str(e)}")
            return []

    async def _analyze_error_handling(self, component: Any) -> List[Dict[str, Any]]:
        """Analyze error handling patterns in component."""
        try:
            patterns = []
            content = component.content

            # Look for error handling patterns
            try_blocks = re.finditer(r'try\s*:', content)
            for match in try_blocks:
                # Find corresponding except blocks
                except_blocks = re.finditer(
                    r'except\s+(\w+)?\s*:', content[match.end():])
                recovery_actions = []

                for except_match in except_blocks:
                    exception_type = except_match.group(1) or "Exception"
                    # Look for recovery actions
                    block_content = content[except_match.end():].split('\n')[0]
                    if 'retry' in block_content.lower():
                        recovery_actions.append('retry')
                    elif 'fallback' in block_content.lower():
                        recovery_actions.append('fallback')
                    elif 'log' in block_content.lower():
                        recovery_actions.append('logging')

                patterns.append({
                    "pattern": "try_except",
                    "strategy": "defensive",
                    "recovery": recovery_actions or ['unknown'],
                    "confidence": 0.8
                })

            return patterns

        except Exception as e:
            self.logger.error(f"Failed to analyze error handling: {str(e)}")
            return []

    async def _analyze_performance_patterns(self, component: Any) -> List[Dict[str, Any]]:
        """Analyze performance patterns in component."""
        try:
            patterns = []
            content = component.content

            # Look for caching
            if re.search(r'@cache', content):
                patterns.append({
                    "pattern": "caching",
                    "optimization": "method_result",
                    "impact": "response_time",
                    "confidence": 0.9
                })

            # Look for bulk operations
            if re.search(r'bulk_create|bulk_update|executemany', content):
                patterns.append({
                    "pattern": "bulk_operation",
                    "optimization": "database_access",
                    "impact": "throughput",
                    "confidence": 0.85
                })

            # Look for lazy loading
            if re.search(r'lazy=True|defer\(|select_related', content):
                patterns.append({
                    "pattern": "lazy_loading",
                    "optimization": "data_loading",
                    "impact": "memory_usage",
                    "confidence": 0.8
                })

            return patterns

        except Exception as e:
            self.logger.error(
                f"Failed to analyze performance patterns: {str(e)}")
            return []

    async def _analyze_transaction_patterns(self, component: Any) -> List[Dict[str, Any]]:
        """Analyze transaction patterns in component."""
        try:
            patterns = []
            content = component.content

            # Look for transaction decorators/context managers
            transaction_patterns = {
                r'@transaction': ('decorator', 'method'),
                r'with\s+transaction': ('context_manager', 'block'),
                r'begin_transaction': ('explicit', 'method'),
                r'commit\(': ('explicit', 'operation'),
                r'rollback\(': ('explicit', 'operation')
            }

            for pattern, (type_, scope) in transaction_patterns.items():
                if re.search(pattern, content):
                    patterns.append({
                        "pattern": type_,
                        "scope": scope,
                        "isolation": "default",  # Could be enhanced to detect isolation levels
                        "confidence": 0.85
                    })

            return patterns

        except Exception as e:
            self.logger.error(
                f"Failed to analyze transaction patterns: {str(e)}")
            return []

    async def initialize_nlu(self):
        """Initialize NLU components."""
        try:
            self.nlu_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.nlu_model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
            self.entity_extractor = pipeline("ner", model="microsoft/codebert-base-mlm")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            
            # Initialize verification components
            self.repo = git.Repo(self.workspace_path)
            self.verification_history = []
            self.confidence_threshold = 0.85
            self.hallucination_threshold = 0.3
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLU components: {str(e)}")

    async def understand_natural_language(self, text: str, context: Optional[List[str]] = None) -> NLUContext:
        """Understand natural language input with context."""
        try:
            # Classify intent
            intent_result = await self._classify_intent(text)
            
            # Extract entities
            entities = await self._extract_entities(text)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(text)
            
            # Build context window
            context_window = context[-5:] if context else []
            
            # Calculate confidence
            confidence = (
                intent_result["confidence"] * 0.4 +
                sum(e["confidence"] for e in entities) / len(entities) * 0.3 +
                sentiment["confidence"] * 0.3
                if entities else
                intent_result["confidence"] * 0.6 +
                sentiment["confidence"] * 0.4
            )
            
            return NLUContext(
                intent=intent_result["intent"],
                entities=entities,
                sentiment=sentiment["score"],
                confidence=confidence,
                context_window=context_window,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "context_size": len(context_window) if context_window else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to understand natural language: {str(e)}")
            return None

    async def verify_code_changes(self, changes: List[Dict[str, Any]], 
                                strategies: Optional[List[VerificationStrategy]] = None) -> CodeChangeVerification:
        """Verify code changes using multiple strategies."""
        try:
            if not strategies:
                strategies = list(VerificationStrategy)
                
            verification = CodeChangeVerification(
                is_verified=False,
                confidence=0.0,
                evidence=[],
                git_verification={},
                test_results={},
                static_analysis={},
                semantic_validation={},
                hallucination_score=0.0
            )
            
            for strategy in strategies:
                if strategy == VerificationStrategy.GIT_HISTORY:
                    git_result = await self._verify_against_git_history(changes)
                    verification.git_verification = git_result
                    verification.evidence.extend(git_result["evidence"])
                    
                elif strategy == VerificationStrategy.TEST_COVERAGE:
                    test_result = await self._verify_test_coverage(changes)
                    verification.test_results = test_result
                    verification.evidence.extend(test_result["evidence"])
                    
                elif strategy == VerificationStrategy.STATIC_ANALYSIS:
                    static_result = await self._perform_static_analysis(changes)
                    verification.static_analysis = static_result
                    verification.evidence.extend(static_result["evidence"])
                    
                elif strategy == VerificationStrategy.SEMANTIC_DIFF:
                    semantic_result = await self._verify_semantic_consistency(changes)
                    verification.semantic_validation = semantic_result
                    verification.evidence.extend(semantic_result["evidence"])
                    
                elif strategy == VerificationStrategy.HALLUCINATION_CHECK:
                    hallucination_result = await self._check_hallucinations(changes)
                    verification.hallucination_score = hallucination_result["score"]
                    verification.evidence.extend(hallucination_result["evidence"])
            
            # Calculate overall confidence
            confidence_scores = [
                verification.git_verification.get("confidence", 0),
                verification.test_results.get("confidence", 0),
                verification.static_analysis.get("confidence", 0),
                verification.semantic_validation.get("confidence", 0),
                1 - verification.hallucination_score
            ]
            
            verification.confidence = sum(confidence_scores) / len(confidence_scores)
            verification.is_verified = (
                verification.confidence >= self.confidence_threshold and
                verification.hallucination_score <= self.hallucination_threshold
            )
            
            # Record verification
            self.verification_history.append({
                "timestamp": datetime.now().isoformat(),
                "changes": changes,
                "verification": verification,
                "strategies": [s.value for s in strategies]
            })
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to verify code changes: {str(e)}")
            return None

    async def _verify_against_git_history(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify changes against Git history."""
        try:
            verification = {
                "confidence": 0.0,
                "evidence": [],
                "similar_changes": []
            }
            
            for change in changes:
                file_path = change["file"]
                content = change["content"]
                
                # Get file history
                commits = list(self.repo.iter_commits(paths=file_path))
                
                for commit in commits[:10]:  # Check last 10 commits
                    # Get file version at commit
                    old_content = self.repo.git.show(f"{commit.hexsha}:{file_path}")
                    
                    # Compare changes
                    diff = list(unified_diff(
                        old_content.splitlines(),
                        content.splitlines()
                    ))
                    
                    if diff:
                        similarity = await self._calculate_change_similarity(
                            diff,
                            change["description"]
                        )
                        
                        if similarity > 0.7:  # High similarity threshold
                            verification["similar_changes"].append({
                                "commit": commit.hexsha,
                                "similarity": similarity,
                                "message": commit.message,
                                "date": commit.committed_datetime.isoformat()
                            })
            
            # Calculate confidence based on similar changes
            if verification["similar_changes"]:
                verification["confidence"] = max(
                    change["similarity"] 
                    for change in verification["similar_changes"]
                )
                verification["evidence"].append({
                    "type": "git_history",
                    "similar_changes": len(verification["similar_changes"]),
                    "max_similarity": verification["confidence"]
                })
            else:
                verification["confidence"] = 0.5  # Neutral confidence for new changes
                verification["evidence"].append({
                    "type": "git_history",
                    "message": "No similar changes found in history"
                })
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to verify against Git history: {str(e)}")
            return {"confidence": 0.0, "evidence": [], "similar_changes": []}

    async def _verify_test_coverage(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify test coverage for changes."""
        try:
            verification = {
                "confidence": 0.0,
                "evidence": [],
                "coverage": {},
                "test_results": []
            }
            
            for change in changes:
                # Find related tests
                tests = await self._find_related_tests(change["file"])
                
                if tests:
                    # Run tests and collect coverage
                    test_results = await self._run_tests(tests)
                    coverage = await self._analyze_test_coverage(
                        change["file"],
                        test_results
                    )
                    
                    verification["coverage"][change["file"]] = coverage
                    verification["test_results"].extend(test_results)
                    
                    # Calculate confidence based on coverage and test results
                    coverage_score = coverage["line_rate"]
                    test_score = sum(
                        1 for t in test_results if t["status"] == "passed"
                    ) / len(test_results)
                    
                    change_confidence = (coverage_score * 0.6 + test_score * 0.4)
                    verification["evidence"].append({
                        "type": "test_coverage",
                        "file": change["file"],
                        "coverage_rate": coverage_score,
                        "test_pass_rate": test_score
                    })
                else:
                    verification["evidence"].append({
                        "type": "test_coverage",
                        "file": change["file"],
                        "message": "No related tests found"
                    })
            
            # Calculate overall confidence
            if verification["coverage"]:
                verification["confidence"] = sum(
                    c["line_rate"] 
                    for c in verification["coverage"].values()
                ) / len(verification["coverage"])
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to verify test coverage: {str(e)}")
            return {"confidence": 0.0, "evidence": [], "coverage": {}, "test_results": []}

    async def _perform_static_analysis(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform static analysis on changes."""
        try:
            verification = {
                "confidence": 0.0,
                "evidence": [],
                "issues": []
            }
            
            for change in changes:
                # Run static analyzers
                lint_results = await self._run_linters(change["file"], change["content"])
                type_check_results = await self._run_type_checker(
                    change["file"],
                    change["content"]
                )
                security_results = await self._run_security_checker(
                    change["file"],
                    change["content"]
                )
                
                # Collect issues
                issues = []
                issues.extend(lint_results)
                issues.extend(type_check_results)
                issues.extend(security_results)
                
                if issues:
                    verification["issues"].extend(issues)
                    verification["evidence"].append({
                        "type": "static_analysis",
                        "file": change["file"],
                        "issue_count": len(issues),
                        "severity_counts": self._count_issue_severities(issues)
                    })
                
                # Calculate confidence based on issues
                severity_weights = {
                    "error": 0.4,
                    "warning": 0.2,
                    "info": 0.1
                }
                
                if issues:
                    weighted_issues = sum(
                        severity_weights.get(issue["severity"], 0)
                        for issue in issues
                    )
                    change_confidence = max(0, 1 - weighted_issues)
                else:
                    change_confidence = 1.0
                
                verification["confidence"] = change_confidence
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to perform static analysis: {str(e)}")
            return {"confidence": 0.0, "evidence": [], "issues": []}

    async def _verify_semantic_consistency(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify semantic consistency of changes."""
        try:
            verification = {
                "confidence": 0.0,
                "evidence": [],
                "semantic_diffs": []
            }
            
            for change in changes:
                # Get semantic embeddings
                old_embedding = await self._create_embedding(change.get("old_content", ""))
                new_embedding = await self._create_embedding(change["content"])
                
                if old_embedding is not None and new_embedding is not None:
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        [old_embedding],
                        [new_embedding]
                    )[0][0]
                    
                    # Analyze semantic impact
                    impact = await self._analyze_semantic_impact(
                        change["file"],
                        old_embedding,
                        new_embedding
                    )
                    
                    verification["semantic_diffs"].append({
                        "file": change["file"],
                        "similarity": similarity,
                        "impact": impact
                    })
                    
                    verification["evidence"].append({
                        "type": "semantic_diff",
                        "file": change["file"],
                        "similarity": similarity,
                        "impact_score": impact["score"]
                    })
            
            # Calculate overall confidence
            if verification["semantic_diffs"]:
                verification["confidence"] = sum(
                    diff["similarity"] 
                    for diff in verification["semantic_diffs"]
                ) / len(verification["semantic_diffs"])
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to verify semantic consistency: {str(e)}")
            return {"confidence": 0.0, "evidence": [], "semantic_diffs": []}

    async def _check_hallucinations(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for potential hallucinations in changes."""
        try:
            verification = {
                "score": 0.0,
                "evidence": [],
                "hallucination_risks": []
            }
            
            for change in changes:
                # Check for common hallucination patterns
                risks = []
                
                # Check for inconsistent references
                ref_check = await self._check_reference_consistency(change)
                if ref_check["inconsistencies"]:
                    risks.extend(ref_check["inconsistencies"])
                
                # Check for unrealistic patterns
                pattern_check = await self._check_unrealistic_patterns(change)
                if pattern_check["risks"]:
                    risks.extend(pattern_check["risks"])
                
                # Check for semantic drift
                drift_check = await self._check_semantic_drift(change)
                if drift_check["drift_score"] > 0.3:  # Significant drift
                    risks.append({
                        "type": "semantic_drift",
                        "score": drift_check["drift_score"],
                        "details": drift_check["details"]
                    })
                
                if risks:
                    verification["hallucination_risks"].append({
                        "file": change["file"],
                        "risks": risks,
                        "risk_score": sum(r.get("score", 0.1) for r in risks) / len(risks)
                    })
                    
                    verification["evidence"].append({
                        "type": "hallucination_check",
                        "file": change["file"],
                        "risk_count": len(risks),
                        "risk_types": [r["type"] for r in risks]
                    })
            
            # Calculate overall hallucination score
            if verification["hallucination_risks"]:
                verification["score"] = sum(
                    risk["risk_score"] 
                    for risk in verification["hallucination_risks"]
                ) / len(verification["hallucination_risks"])
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Failed to check hallucinations: {str(e)}")
            return {"score": 1.0, "evidence": [], "hallucination_risks": []}

    async def _check_reference_consistency(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of code references."""
        try:
            result = {
                "inconsistencies": []
            }
            
            # Extract references (imports, function calls, etc.)
            refs = await self._extract_code_references(change["content"])
            
            for ref in refs:
                # Check if reference exists in codebase
                exists = await self._verify_reference_exists(ref)
                if not exists:
                    result["inconsistencies"].append({
                        "type": "missing_reference",
                        "reference": ref,
                        "score": 0.8
                    })
                
                # Check if reference usage is consistent
                usage_check = await self._verify_reference_usage(ref, change["content"])
                if not usage_check["is_consistent"]:
                    result["inconsistencies"].append({
                        "type": "inconsistent_usage",
                        "reference": ref,
                        "details": usage_check["details"],
                        "score": 0.6
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to check reference consistency: {str(e)}")
            return {"inconsistencies": []}

    async def _check_unrealistic_patterns(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Check for unrealistic code patterns."""
        try:
            result = {
                "risks": []
            }
            
            # Check for impossible combinations
            impossible = await self._check_impossible_combinations(change["content"])
            result["risks"].extend(impossible)
            
            # Check for unrealistic performance claims
            perf_risks = await self._check_performance_claims(change["content"])
            result["risks"].extend(perf_risks)
            
            # Check for security vulnerabilities
            sec_risks = await self._check_security_patterns(change["content"])
            result["risks"].extend(sec_risks)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to check unrealistic patterns: {str(e)}")
            return {"risks": []}

    async def _check_semantic_drift(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Check for semantic drift in changes."""
        try:
            result = {
                "drift_score": 0.0,
                "details": []
            }
            
            # Get semantic representation of original code
            original_semantics = await self._extract_code_semantics(
                change.get("old_content", "")
            )
            
            # Get semantic representation of new code
            new_semantics = await self._extract_code_semantics(change["content"])
            
            # Calculate semantic drift
            drift_metrics = await self._calculate_semantic_drift(
                original_semantics,
                new_semantics
            )
            
            result["drift_score"] = drift_metrics["score"]
            result["details"] = drift_metrics["details"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to check semantic drift: {str(e)}")
            return {"drift_score": 1.0, "details": []}

    async def initialize_learning(self):
        """Initialize learning components."""
        try:
            self.learning_context = LearningContext(
                patterns={},
                adaptations=[],
                feedback=[],
                performance_metrics={},
                confidence_thresholds={
                    "verification": 0.85,
                    "hallucination": 0.3,
                    "semantic": 0.75,
                    "adaptation": 0.8
                },
                metadata={}
            )
            
            # Initialize adaptation strategies
            self.adaptation_strategies = []
            await self._load_adaptation_strategies()
            
            # Start learning loops
            self._start_continuous_learning()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning: {str(e)}")

    async def _load_adaptation_strategies(self):
        """Load and initialize adaptation strategies."""
        try:
            # Strategy for verification threshold adaptation
            self.adaptation_strategies.append(
                AdaptationStrategy(
                    trigger_conditions=[
                        {
                            "metric": "false_positive_rate",
                            "threshold": 0.2,
                            "window": "1d"
                        },
                        {
                            "metric": "false_negative_rate",
                            "threshold": 0.1,
                            "window": "1d"
                        }
                    ],
                    adaptation_rules=[
                        {
                            "target": "confidence_threshold",
                            "adjustment": "dynamic",
                            "factors": ["error_rate", "impact_severity"]
                        }
                    ],
                    rollback_conditions=[
                        {
                            "metric": "verification_accuracy",
                            "threshold": 0.9,
                            "window": "1h"
                        }
                    ],
                    success_metrics={
                        "accuracy": 0.95,
                        "precision": 0.9,
                        "recall": 0.9
                    },
                    confidence=0.8
                )
            )
            
            # Strategy for hallucination detection improvement
            self.adaptation_strategies.append(
                AdaptationStrategy(
                    trigger_conditions=[
                        {
                            "metric": "hallucination_rate",
                            "threshold": 0.1,
                            "window": "6h"
                        }
                    ],
                    adaptation_rules=[
                        {
                            "target": "hallucination_patterns",
                            "action": "update",
                            "source": "verified_changes"
                        },
                        {
                            "target": "semantic_thresholds",
                            "action": "adjust",
                            "factors": ["drift_rate", "change_complexity"]
                        }
                    ],
                    rollback_conditions=[
                        {
                            "metric": "false_hallucination_rate",
                            "threshold": 0.05,
                            "window": "1h"
                        }
                    ],
                    success_metrics={
                        "hallucination_detection_rate": 0.95,
                        "false_positive_rate": 0.05
                    },
                    confidence=0.85
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load adaptation strategies: {str(e)}")

    def _start_continuous_learning(self):
        """Start continuous learning loops."""
        try:
            asyncio.create_task(self._pattern_learning_loop())
            asyncio.create_task(self._threshold_adaptation_loop())
            asyncio.create_task(self._feedback_processing_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start continuous learning: {str(e)}")

    async def _pattern_learning_loop(self):
        """Continuous pattern learning loop."""
        try:
            while True:
                # Analyze verification history
                patterns = await self._extract_verification_patterns(
                    self.verification_history[-1000:]  # Last 1000 verifications
                )
                
                # Update learned patterns
                self.learning_context.patterns.update(patterns)
                
                # Adjust pattern confidence
                await self._adjust_pattern_confidence()
                
                # Sleep for interval
                await asyncio.sleep(3600)  # Every hour
                
        except Exception as e:
            self.logger.error(f"Error in pattern learning loop: {str(e)}")

    async def _threshold_adaptation_loop(self):
        """Continuous threshold adaptation loop."""
        try:
            while True:
                # Calculate performance metrics
                metrics = await self._calculate_performance_metrics()
                
                # Check adaptation triggers
                for strategy in self.adaptation_strategies:
                    if await self._should_adapt(strategy, metrics):
                        # Apply adaptation rules
                        await self._apply_adaptation_rules(strategy)
                        
                        # Monitor for rollback conditions
                        asyncio.create_task(
                            self._monitor_adaptation(strategy)
                        )
                
                # Update performance metrics
                self.learning_context.performance_metrics.update(metrics)
                
                # Sleep for interval
                await asyncio.sleep(1800)  # Every 30 minutes
                
        except Exception as e:
            self.logger.error(f"Error in threshold adaptation loop: {str(e)}")

    async def _feedback_processing_loop(self):
        """Continuous feedback processing loop."""
        try:
            while True:
                # Process new feedback
                feedback = await self._collect_feedback()
                
                # Update learning context
                self.learning_context.feedback.extend(feedback)
                
                # Adjust strategies based on feedback
                await self._adjust_strategies(feedback)
                
                # Sleep for interval
                await asyncio.sleep(300)  # Every 5 minutes
                
        except Exception as e:
            self.logger.error(f"Error in feedback processing loop: {str(e)}")

    async def _extract_verification_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from verification history."""
        try:
            patterns = defaultdict(list)
            
            for verification in history:
                # Extract change patterns
                change_patterns = await self._analyze_change_patterns(
                    verification["changes"]
                )
                
                # Extract verification patterns
                verification_patterns = await self._analyze_verification_patterns(
                    verification["verification"]
                )
                
                # Extract outcome patterns
                outcome_patterns = await self._analyze_outcome_patterns(
                    verification["verification"],
                    verification.get("feedback", {})
                )
                
                # Group patterns by confidence
                for pattern in chain(change_patterns, verification_patterns, outcome_patterns):
                    patterns[pattern["type"]].append({
                        "pattern": pattern["pattern"],
                        "confidence": pattern["confidence"],
                        "context": pattern["context"],
                        "timestamp": verification["timestamp"]
                    })
            
            # Aggregate pattern confidence
            aggregated = {}
            for pattern_type, instances in patterns.items():
                aggregated[pattern_type] = {
                    "patterns": instances,
                    "confidence": sum(p["confidence"] for p in instances) / len(instances),
                    "frequency": len(instances) / len(history)
                }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Failed to extract verification patterns: {str(e)}")
            return {}

    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate system performance metrics."""
        try:
            metrics = {}
            
            # Calculate verification metrics
            verification_metrics = await self._calculate_verification_metrics()
            metrics.update(verification_metrics)
            
            # Calculate adaptation metrics
            adaptation_metrics = await self._calculate_adaptation_metrics()
            metrics.update(adaptation_metrics)
            
            # Calculate learning metrics
            learning_metrics = await self._calculate_learning_metrics()
            metrics.update(learning_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {str(e)}")
            return {}

    async def _should_adapt(self, strategy: AdaptationStrategy, 
                           metrics: Dict[str, float]) -> bool:
        """Check if adaptation should be triggered."""
        try:
            # Check all trigger conditions
            for condition in strategy.trigger_conditions:
                metric_value = metrics.get(condition["metric"])
                if metric_value is None:
                    continue
                    
                if condition.get("window"):
                    # Calculate metric over time window
                    window_value = await self._calculate_metric_window(
                        condition["metric"],
                        condition["window"]
                    )
                    if window_value > condition["threshold"]:
                        return True
                else:
                    # Check instant metric value
                    if metric_value > condition["threshold"]:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check adaptation triggers: {str(e)}")
            return False

    async def _apply_adaptation_rules(self, strategy: AdaptationStrategy):
        """Apply adaptation rules."""
        try:
            adaptations = []
            
            for rule in strategy.adaptation_rules:
                if rule["target"] == "confidence_threshold":
                    # Adjust confidence threshold
                    new_threshold = await self._calculate_dynamic_threshold(
                        rule["factors"]
                    )
                    self.confidence_threshold = new_threshold
                    adaptations.append({
                        "type": "threshold_adjustment",
                        "target": "confidence",
                        "old_value": self.confidence_threshold,
                        "new_value": new_threshold,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif rule["target"] == "hallucination_patterns":
                    # Update hallucination patterns
                    new_patterns = await self._update_hallucination_patterns(
                        rule["source"]
                    )
                    adaptations.append({
                        "type": "pattern_update",
                        "target": "hallucination",
                        "patterns": new_patterns,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif rule["target"] == "semantic_thresholds":
                    # Adjust semantic thresholds
                    new_thresholds = await self._adjust_semantic_thresholds(
                        rule["factors"]
                    )
                    adaptations.append({
                        "type": "threshold_adjustment",
                        "target": "semantic",
                        "thresholds": new_thresholds,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Record adaptations
            self.learning_context.adaptations.extend(adaptations)
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptation rules: {str(e)}")

    async def _monitor_adaptation(self, strategy: AdaptationStrategy):
        """Monitor adaptation for rollback conditions."""
        try:
            while True:
                # Check rollback conditions
                for condition in strategy.rollback_conditions:
                    metric_value = await self._calculate_metric_window(
                        condition["metric"],
                        condition["window"]
                    )
                    
                    if metric_value < condition["threshold"]:
                        # Rollback adaptation
                        await self._rollback_adaptation(strategy)
                        return
                
                # Check success metrics
                success = await self._check_success_metrics(
                    strategy.success_metrics
                )
                if success:
                    # Adaptation successful
                    return
                
                # Sleep before next check
                await asyncio.sleep(300)  # Every 5 minutes
                
        except Exception as e:
            self.logger.error(f"Failed to monitor adaptation: {str(e)}")

    async def _rollback_adaptation(self, strategy: AdaptationStrategy):
        """Rollback adaptation changes."""
        try:
            # Get last adaptation for this strategy
            adaptations = [
                a for a in self.learning_context.adaptations
                if a["type"] in [r["target"] for r in strategy.adaptation_rules]
            ]
            
            if not adaptations:
                return
            
            last_adaptation = adaptations[-1]
            
            # Rollback changes
            if last_adaptation["type"] == "threshold_adjustment":
                if last_adaptation["target"] == "confidence":
                    self.confidence_threshold = last_adaptation["old_value"]
                elif last_adaptation["target"] == "semantic":
                    await self._restore_semantic_thresholds(
                        last_adaptation["thresholds"]
                    )
                    
            elif last_adaptation["type"] == "pattern_update":
                await self._restore_patterns(
                    last_adaptation["target"],
                    last_adaptation.get("previous_patterns", {})
                )
            
            # Record rollback
            self.learning_context.adaptations.append({
                "type": "rollback",
                "target": last_adaptation["type"],
                "timestamp": datetime.now().isoformat(),
                "reason": "rollback_condition_triggered"
            })
            
        except Exception as e:
            self.logger.error(f"Failed to rollback adaptation: {str(e)}")

    async def _vectorize_code(self, code_block: str) -> Optional[List[float]]:
        """Vectorize code using MCP embeddings."""
        try:
            # Get embeddings from MCP
            embedding = await self.mcp_client.get_embeddings(
                content=code_block,
                content_type="code"
            )
            
            return embedding.get("vector")
            
        except Exception as e:
            self.logger.error(f"Failed to vectorize code: {str(e)}")
            return None
