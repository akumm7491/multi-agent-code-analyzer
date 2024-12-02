from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    ANALYZE = "analyze"
    MODIFY = "modify"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    REVIEW = "review"

@dataclass
class ParsedQuery:
    query_type: QueryType
    target: str
    context: Dict[str, Any]
    constraints: List[str]
    scope: str

class QueryParser:
    """Parses and understands user queries."""
    
    def __init__(self):
        self.keywords = {
            QueryType.ANALYZE: ["analyze", "understand", "explain", "describe"],
            QueryType.MODIFY: ["change", "update", "modify", "implement"],
            QueryType.EXPLAIN: ["how", "why", "what", "explain"],
            QueryType.REFACTOR: ["refactor", "improve", "optimize", "clean"],
            QueryType.REVIEW: ["review", "check", "assess", "evaluate"]
        }
        
    async def parse(self, query: str) -> ParsedQuery:
        """Parse a user query into structured format."""
        query = query.lower()
        
        # Determine query type
        query_type = await self._determine_type(query)
        
        # Extract target
        target = await self._extract_target(query)
        
        # Extract context
        context = await self._extract_context(query)
        
        # Extract constraints
        constraints = await self._extract_constraints(query)
        
        # Determine scope
        scope = await self._determine_scope(query)
        
        return ParsedQuery(
            query_type=query_type,
            target=target,
            context=context,
            constraints=constraints,
            scope=scope
        )