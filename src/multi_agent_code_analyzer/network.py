    async def _get_agent_context(self, agent: Any, global_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context for a specific agent."""
        context = global_context.copy()
        
        # Add agent-specific knowledge from knowledge graph
        graph_context = await self.knowledge_graph.get_context_for_specialty(agent.specialty)
        if graph_context:
            context.update(graph_context)
        
        # Add relevant relationships
        relationships = await self.knowledge_graph.get_relationships_for_specialty(agent.specialty)
        if relationships:
            context["relationships"] = relationships
        
        return context
    
    async def _update_knowledge(self, query: str, response: Dict[str, Any]):
        """Update knowledge graph with new information."""
        # Extract entities and relationships from response
        entities = await self._extract_entities(response)
        relationships = await self._extract_relationships(response)
        
        # Update knowledge graph
        for entity in entities:
            await self.knowledge_graph.add_node(
                entity["id"],
                entity["type"],
                entity["metadata"]
            )
            
        for rel in relationships:
            await self.knowledge_graph.add_relationship(
                rel["from"],
                rel["to"],
                rel["type"],
                rel["metadata"]
            )
    
    async def _extract_entities(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from agent response."""
        entities = []
        
        # Extract based on response structure
        if "analysis" in response:
            for key, value in response["analysis"].items():
                if isinstance(value, dict) and "type" in value:
                    entities.append({
                        "id": f"{key}_{len(entities)}",
                        "type": value["type"],
                        "metadata": value
                    })
        
        return entities
    
    async def _extract_relationships(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships from agent response."""
        relationships = []
        
        # Extract based on response structure
        if "relationships" in response:
            for rel in response["relationships"]:
                if "from" in rel and "to" in rel:
                    relationships.append({
                        "from": rel["from"],
                        "to": rel["to"],
                        "type": rel.get("type", "related_to"),
                        "metadata": rel.get("metadata", {})
                    })
        
        return relationships