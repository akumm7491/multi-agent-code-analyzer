from typing import Dict, Any, List, Optional
from .base import BaseAgent

class CodeAgent(BaseAgent):
    """Agent specialized in analyzing implementation details and code patterns."""
    
    def __init__(self, name: str):
        super().__init__(name, specialty="code")
        self.code_patterns = {}
        self.implementation_details = {}
        self.known_antipatterns = {}
        
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        response = {
            "agent": self.name,
            "specialty": self.specialty,
            "analysis": {},
            "concerns": [],
            "confidence": 0.0
        }
        
        # Analyze code patterns
        patterns = await self._analyze_patterns(query, context)
        if patterns:
            response["analysis"]["patterns"] = patterns
        
        # Identify potential issues
        issues = await self._identify_issues(patterns)
        if issues:
            response["concerns"] = issues
        
        # Add implementation insights
        insights = await self._gather_implementation_insights(query, context)
        if insights:
            response["analysis"]["implementation"] = insights
        
        response["confidence"] = await self._calculate_confidence(response["analysis"])
        return response

    async def update_knowledge(self, new_information: Dict[str, Any]):
        """Update code-related knowledge."""
        if "patterns" in new_information:
            self.code_patterns.update(new_information["patterns"])
        if "implementations" in new_information:
            self.implementation_details.update(new_information["implementations"])
        if "antipatterns" in new_information:
            self.known_antipatterns.update(new_information["antipatterns"])

    async def _analyze_patterns(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code patterns in the query context."""
        patterns = {}
        for pattern, info in self.code_patterns.items():
            if await self._is_pattern_relevant(pattern, info, query, context):
                patterns[pattern] = {
                    "description": info.get("description", ""),
                    "confidence": await self._calculate_pattern_confidence(info, context)
                }
        return patterns

    async def _identify_issues(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential code issues and antipatterns."""
        issues = []
        for pattern_name, pattern_info in patterns.items():
            if pattern_name in self.known_antipatterns:
                issues.append({
                    "type": "antipattern",
                    "pattern": pattern_name,
                    "description": self.known_antipatterns[pattern_name],
                    "severity": "high"
                })
        return issues

    async def _gather_implementation_insights(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather insights about implementation details."""
        insights = {}
        for detail, info in self.implementation_details.items():
            if detail.lower() in query.lower() or await self._is_detail_relevant(detail, info, context):
                insights[detail] = info
        return insights

    async def _is_pattern_relevant(self, pattern: str, info: Dict[str, Any], query: str, context: Dict[str, Any]) -> bool:
        """Determine if a code pattern is relevant to the current query."""
        # Simple relevance check - could be enhanced with more sophisticated matching
        return pattern.lower() in query.lower() or any(keyword in query.lower() for keyword in info.get("keywords", []))

    async def _calculate_pattern_confidence(self, pattern_info: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence score for a pattern match."""
        # Basic confidence calculation - could be enhanced with more sophisticated metrics
        confidence = 0.5  # Base confidence
        if "frequency" in pattern_info:
            confidence += 0.3 * min(1.0, pattern_info["frequency"] / 100)
        if "complexity" in pattern_info:
            confidence -= 0.2 * min(1.0, pattern_info["complexity"] / 10)
        return max(0.0, min(1.0, confidence))

    async def _is_detail_relevant(self, detail: str, info: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if an implementation detail is relevant to the current context."""
        # Simple relevance check - could be enhanced
        return any(keyword in str(context).lower() for keyword in info.get("keywords", []))