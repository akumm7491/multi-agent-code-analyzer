    async def _determine_type(self, query: str) -> QueryType:
        """Determine the type of query based on keywords."""
        for query_type, keywords in self.keywords.items():
            if any(keyword in query for keyword in keywords):
                return query_type
        return QueryType.ANALYZE  # Default type
    
    async def _extract_target(self, query: str) -> str:
        """Extract the target of the query (file, function, class, etc.)."""
        # Common target indicators
        indicators = [
            "in", "the", "file", "function", "class", "method",
            "module", "component", "service"
        ]
        
        words = query.split()
        for i, word in enumerate(words):
            if word in indicators and i + 1 < len(words):
                return words[i + 1]
        
        return ""  # No specific target found
    
    async def _extract_context(self, query: str) -> Dict[str, Any]:
        """Extract contextual information from the query."""
        context = {}
        
        # Extract file path patterns
        if "path:" in query:
            path_start = query.index("path:") + 5
            path_end = query.find(" ", path_start)
            if path_end == -1:
                path_end = len(query)
            context["path"] = query[path_start:path_end]
        
        # Extract language if specified
        if "language:" in query:
            lang_start = query.index("language:") + 9
            lang_end = query.find(" ", lang_start)
            if lang_end == -1:
                lang_end = len(query)
            context["language"] = query[lang_start:lang_end]
        
        return context
    
    async def _extract_constraints(self, query: str) -> List[str]:
        """Extract any constraints or requirements from the query."""
        constraints = []
        
        # Look for constraint indicators
        indicators = [
            "must", "should", "needs to", "required",
            "ensure", "maintain", "keep"
        ]
        
        words = query.split()
        for i, word in enumerate(words):
            if word in indicators:
                # Collect words until next punctuation or indicator
                constraint = []
                j = i + 1
                while j < len(words) and words[j] not in indicators and words[j] not in [".", ","]:
                    constraint.append(words[j])
                    j += 1
                if constraint:
                    constraints.append(" ".join(constraint))
        
        return constraints
    
    async def _determine_scope(self, query: str) -> str:
        """Determine the scope of the query (single file, module, project-wide)."""
        if any(word in query for word in ["all", "entire", "project", "codebase"]):
            return "project"
        elif any(word in query for word in ["module", "package", "directory"]):
            return "module"
        return "file"  # Default to file scope