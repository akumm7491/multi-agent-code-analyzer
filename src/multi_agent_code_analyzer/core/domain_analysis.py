import ast
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

__all__ = ['DomainAnalyzer', 'identify_bounded_context',
           'detect_ddd_patterns', 'calculate_metrics']


@dataclass
class DDDPattern:
    name: str
    description: str
    indicators: List[str]
    code_patterns: List[str]


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
        ]
    )
}


class DomainAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_domain_concept(self, node: ast.ClassDef, file_path: str) -> Optional[Dict]:
        """Extract domain concept from a class definition."""
        try:
            # Get class documentation and annotations
            doc = ast.get_docstring(node) or ""
            decorators = [
                d.id for d in node.decorator_list if isinstance(d, ast.Name)]

            # Analyze class structure
            methods = []
            properties = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    properties.append(item.target.id)

            # Detect DDD pattern
            pattern_type = self.detect_pattern(
                node, doc, decorators, methods, properties)

            # Extract relationships
            relationships = self.extract_relationships(node)

            return {
                "name": node.name,
                "type": pattern_type,
                "file_path": file_path,
                "line_number": node.lineno,
                "relationships": relationships,
                "properties": {
                    "methods": methods,
                    "attributes": properties,
                    "documentation": doc
                }
            }
        except Exception as e:
            print(f"Error extracting domain concept: {str(e)}")
            return None

    def detect_pattern(self, node: ast.ClassDef, doc: str, decorators: List[str],
                       methods: List[str], properties: List[str]) -> str:
        """Detect which DDD pattern the class implements."""
        class_text = f"{node.name} {doc} {' '.join(decorators)} {' '.join(methods)}"

        # Check each pattern's indicators
        pattern_scores = {}
        for pattern_name, pattern in DDD_PATTERNS.items():
            score = 0
            # Check code patterns
            for cp in pattern.code_patterns:
                if re.search(cp, class_text, re.IGNORECASE):
                    score += 1

            # Check semantic similarity with pattern indicators
            text_embedding = self.model.encode(class_text)
            for indicator in pattern.indicators:
                indicator_embedding = self.model.encode(indicator)
                similarity = np.dot(text_embedding, indicator_embedding)
                score += similarity

            pattern_scores[pattern_name] = score

        # Return the pattern with highest score
        if pattern_scores:
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        return "unknown"

    def extract_relationships(self, node: ast.ClassDef) -> List[Dict[str, str]]:
        """Extract relationships with other domain concepts."""
        relationships = []

        for item in ast.walk(node):
            # Check for inheritance
            if isinstance(item, ast.ClassDef) and item.bases:
                for base in item.bases:
                    if isinstance(base, ast.Name):
                        relationships.append({
                            "type": "inherits_from",
                            "target": base.id
                        })

            # Check for composition/aggregation
            elif isinstance(item, ast.AnnAssign) and isinstance(item.annotation, ast.Name):
                relationships.append({
                    "type": "contains",
                    "target": item.annotation.id
                })

        return relationships


def identify_bounded_context(file_path: str, concepts: List[Dict]) -> Optional[Dict]:
    """Identify bounded context based on file location and concepts."""
    try:
        # Use directory structure as initial context boundary
        context_path = str(Path(file_path).parent)

        # Analyze concept relationships within the directory
        internal_concepts = [c["name"] for c in concepts]
        external_dependencies = []

        return {
            "name": Path(context_path).name,
            "path": context_path,
            "concepts": internal_concepts,
            "external_dependencies": external_dependencies
        }
    except Exception as e:
        print(f"Error identifying bounded context: {str(e)}")
        return None


def detect_ddd_patterns(file_path: str, concepts: List[Dict]) -> List[Dict]:
    """Detect DDD patterns in the file."""
    patterns = []
    analyzer = DomainAnalyzer()

    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                pattern = analyzer.detect_pattern(
                    node,
                    ast.get_docstring(node) or "",
                    [d.id for d in node.decorator_list if isinstance(
                        d, ast.Name)],
                    [m.name for m in node.body if isinstance(
                        m, ast.FunctionDef)],
                    [p.target.id for p in node.body if isinstance(
                        p, ast.AnnAssign) and isinstance(p.target, ast.Name)]
                )
                if pattern != "unknown":
                    patterns.append({
                        "pattern": pattern,
                        "location": {
                            "file": file_path,
                            "line": node.lineno
                        },
                        "class_name": node.name
                    })
    except Exception as e:
        print(f"Error detecting patterns: {str(e)}")

    return patterns


def calculate_metrics(concepts: List[Dict], contexts: List[Dict], patterns: List[Dict]) -> Dict[str, float]:
    """Calculate various metrics about the codebase."""
    try:
        total_concepts = len(concepts)
        total_contexts = len(contexts)
        pattern_distribution = {}

        for pattern in patterns:
            pattern_type = pattern["pattern"]
            pattern_distribution[pattern_type] = pattern_distribution.get(
                pattern_type, 0) + 1

        # Calculate coupling metrics
        coupling_score = 0
        for context in contexts:
            coupling_score += len(context.get("external_dependencies", []))

        if total_contexts > 0:
            coupling_score /= total_contexts

        return {
            "total_concepts": total_concepts,
            "total_contexts": total_contexts,
            "patterns_found": len(patterns),
            "patterns_distribution": pattern_distribution,
            "coupling_score": coupling_score
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {}
