from typing import Dict, Any, List
from .base import BaseAgent

class DomainAgent(BaseAgent):
    """Agent specialized in understanding business logic and domain concepts."""
    
    def __init__(self, name: str):
        super().__init__(name, specialty="domain")
        self.domain_concepts = {}
        self.business_rules = {}
        self.domain_workflows = {}
        self.terminology_mappings = {}
        
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        response = {
            "agent": self.name,
            "specialty": self.specialty,
            "analysis": {},
            "domain_insights": [],
            "confidence": 0.0
        }
        
        # Analyze domain concepts
        concepts = await self._analyze_domain_concepts(query)
        if concepts:
            response["analysis"]["concepts"] = concepts
        
        # Identify business rules
        rules = await self._identify_business_rules(query, context)
        if rules:
            response["analysis"]["rules"] = rules
        
        # Map domain workflows
        workflows = await self._map_workflows(query, context)
        if workflows:
            response["analysis"]["workflows"] = workflows
            
        # Generate domain insights
        insights = await self._generate_domain_insights(response["analysis"])
        if insights:
            response["domain_insights"] = insights
        
        response["confidence"] = await self._calculate_confidence(response["analysis"])
        return response

    async def update_knowledge(self, new_information: Dict[str, Any]):
        """Update domain knowledge."""
        if "concepts" in new_information:
            self.domain_concepts.update(new_information["concepts"])
        if "rules" in new_information:
            self.business_rules.update(new_information["rules"])
        if "workflows" in new_information:
            self.domain_workflows.update(new_information["workflows"])
        if "terminology" in new_information:
            self.terminology_mappings.update(new_information["terminology"])

    async def _analyze_domain_concepts(self, query: str) -> Dict[str, Any]:
        """Analyze domain concepts in the query."""
        relevant_concepts = {}
        for concept, info in self.domain_concepts.items():
            if await self._is_concept_relevant(concept, info, query):
                relevant_concepts[concept] = {
                    "description": info.get("description", ""),
                    "relationships": info.get("relationships", []),
                    "importance": await self._calculate_concept_importance(concept, info)
                }
        return relevant_concepts

    async def _identify_business_rules(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevant business rules."""
        rules = []
        for rule_id, rule_info in self.business_rules.items():
            if await self._is_rule_relevant(rule_info, query, context):
                rules.append({
                    "id": rule_id,
                    "description": rule_info["description"],
                    "constraints": rule_info.get("constraints", []),
                    "priority": rule_info.get("priority", "medium")
                })
        return rules

    async def _map_workflows(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Map domain workflows related to the query."""
        workflows = {}
        for workflow, steps in self.domain_workflows.items():
            if await self._is_workflow_relevant(workflow, steps, query):
                workflows[workflow] = {
                    "steps": steps,
                    "entry_points": await self._identify_entry_points(workflow, steps),
                    "dependencies": await self._identify_workflow_dependencies(workflow)
                }
        return workflows

    async def _generate_domain_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on domain analysis."""
        insights = []
        
        # Analyze concept relationships
        if "concepts" in analysis:
            for concept, info in analysis["concepts"].items():
                if info["importance"] > 0.7:  # High importance concepts
                    insights.append({
                        "type": "key_concept",
                        "concept": concept,
                        "impact": "high",
                        "recommendation": f"Consider documenting {concept} thoroughly"
                    })
        
        # Analyze rule interactions
        if "rules" in analysis:
            rule_conflicts = await self._identify_rule_conflicts(analysis["rules"])
            insights.extend(rule_conflicts)
        
        return insights

    async def _is_concept_relevant(self, concept: str, info: Dict[str, Any], query: str) -> bool:
        """Determine if a domain concept is relevant to the query."""
        return concept.lower() in query.lower() or \
               any(keyword in query.lower() for keyword in info.get("keywords", []))

    async def _calculate_concept_importance(self, concept: str, info: Dict[str, Any]) -> float:
        """Calculate importance score for a domain concept."""
        importance = 0.5  # Base importance
        if "usage_frequency" in info:
            importance += 0.3 * min(1.0, info["usage_frequency"] / 100)
        if "relationships" in info:
            importance += 0.2 * min(1.0, len(info["relationships"]) / 10)
        return min(1.0, importance)

    async def _is_rule_relevant(self, rule_info: Dict[str, Any], query: str, context: Dict[str, Any]) -> bool:
        """Determine if a business rule is relevant."""
        return any(keyword in query.lower() for keyword in rule_info.get("keywords", []))

    async def _is_workflow_relevant(self, workflow: str, steps: List[Dict[str, Any]], query: str) -> bool:
        """Determine if a workflow is relevant to the query."""
        return workflow.lower() in query.lower() or \
               any(step["name"].lower() in query.lower() for step in steps)