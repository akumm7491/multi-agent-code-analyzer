from typing import Dict, Any, List
from .base import BaseAgent

class ArchitectAgent(BaseAgent):
    """Agent specialized in understanding system architecture and component relationships."""
    
    def __init__(self, name: str):
        super().__init__(name, specialty="architecture")
        self.design_patterns = {}
        self.component_relationships = {}
        self.system_boundaries = {}
        
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        response = {
            "agent": self.name,
            "specialty": self.specialty,
            "analysis": {},
            "confidence": 0.0
        }
        
        # Analyze architectural patterns
        patterns = await self._identify_patterns(query)
        if patterns:
            response["analysis"]["patterns"] = patterns
            
        # Identify relevant components
        components = await self._identify_components(query, context)
        if components:
            response["analysis"]["components"] = components
            
        # Calculate confidence
        response["confidence"] = await self._calculate_confidence(response["analysis"])
        
        return response

    async def update_knowledge(self, new_information: Dict[str, Any]):
        """Update architectural knowledge."""
        if "patterns" in new_information:
            self.design_patterns.update(new_information["patterns"])
        if "components" in new_information:
            self.component_relationships.update(new_information["components"])
        if "boundaries" in new_information:
            self.system_boundaries.update(new_information["boundaries"])

    async def _identify_patterns(self, query: str) -> Dict[str, Any]:
        """Identify architectural patterns relevant to the query."""
        patterns = {}
        for pattern, info in self.design_patterns.items():
            if any(keyword in query.lower() for keyword in info.get("keywords", [])):
                patterns[pattern] = info
        return patterns

    async def _identify_components(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify system components relevant to the query."""
        components = {}
        for component, relations in self.component_relationships.items():
            if component.lower() in query.lower():
                components[component] = {
                    "relations": relations,
                    "boundaries": self.system_boundaries.get(component, [])
                }
        return components

    async def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        if not analysis:
            return 0.0
            
        confidence = 0.0
        if "patterns" in analysis:
            confidence += 0.5 * len(analysis["patterns"]) / max(1, len(self.design_patterns))
        if "components" in analysis:
            confidence += 0.5 * len(analysis["components"]) / max(1, len(self.component_relationships))
            
        return min(confidence, 1.0)