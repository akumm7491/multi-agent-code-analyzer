from typing import Dict, Any, List, Optional
import aiohttp
from .base import Tool, ToolResult, ToolCategory


class WebSearchTool(Tool):
    """Web search tool for finding documentation and solutions"""

    def __init__(self):
        super().__init__()
        self.category = ToolCategory.SEARCH
        self.name = "WebSearchTool"
        self.description = "Search the web for documentation and solutions"

    async def validate(self, **kwargs) -> bool:
        required = {"api_key", "search_engine_id"}
        return all(key in self.config for key in required)

    async def execute(self, **kwargs) -> ToolResult:
        if not await self.validate(**kwargs):
            return ToolResult(
                success=False,
                data=None,
                error="Missing required configuration"
            )

        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(**kwargs)
        elif action == "get_best_practices":
            return await self._get_best_practices(**kwargs)

        return ToolResult(
            success=False,
            data=None,
            error=f"Unknown action: {action}"
        )

    async def _search(self, **kwargs) -> ToolResult:
        """Perform a web search"""
        query = kwargs.get("query")
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="Missing search query"
            )

        async with aiohttp.ClientSession() as session:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.config["api_key"],
                "cx": self.config["search_engine_id"],
                "q": query,
                "num": kwargs.get("num_results", 10)
            }

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Search API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_best_practices(self, **kwargs) -> ToolResult:
        """Get curated best practices for different domains"""
        domain = kwargs.get("domain", "").lower()

        best_practices = {
            "api_design": {
                "rest": {
                    "principles": [
                        "Use nouns for resources",
                        "Use HTTP methods correctly",
                        "Use proper status codes",
                        "Version your API",
                        "Use pagination for lists",
                        "Support filtering and sorting",
                        "Use HATEOAS for discoverability"
                    ],
                    "security": [
                        "Use HTTPS",
                        "Implement rate limiting",
                        "Use proper authentication",
                        "Validate input",
                        "Sanitize output"
                    ],
                    "documentation": [
                        "Use OpenAPI/Swagger",
                        "Document error responses",
                        "Provide examples",
                        "Include authentication details"
                    ]
                },
                "graphql": {
                    "principles": [
                        "Design schema first",
                        "Use proper naming conventions",
                        "Implement pagination",
                        "Handle errors properly",
                        "Use fragments for reusability"
                    ],
                    "security": [
                        "Implement query complexity analysis",
                        "Use proper authentication",
                        "Validate input",
                        "Rate limit operations"
                    ]
                }
            },
            "microservices": {
                "principles": [
                    "Single Responsibility",
                    "Database per Service",
                    "API Gateway pattern",
                    "Event-Driven Architecture",
                    "Circuit Breaker pattern"
                ],
                "communication": [
                    "Use async messaging",
                    "Implement retry logic",
                    "Handle partial failures",
                    "Use correlation IDs"
                ],
                "monitoring": [
                    "Implement distributed tracing",
                    "Use health checks",
                    "Monitor service metrics",
                    "Implement proper logging"
                ]
            },
            "testing": {
                "unit_testing": [
                    "Test one thing at a time",
                    "Use meaningful names",
                    "Follow AAA pattern",
                    "Mock external dependencies",
                    "Aim for high coverage"
                ],
                "integration_testing": [
                    "Test service interactions",
                    "Use test containers",
                    "Clean up test data",
                    "Test error scenarios"
                ],
                "e2e_testing": [
                    "Test critical paths",
                    "Use realistic data",
                    "Test in production-like environment",
                    "Implement proper cleanup"
                ]
            },
            "security": {
                "authentication": [
                    "Use strong password policies",
                    "Implement MFA",
                    "Use secure session management",
                    "Implement proper logout"
                ],
                "authorization": [
                    "Use Role-Based Access Control",
                    "Implement principle of least privilege",
                    "Validate permissions properly",
                    "Audit access logs"
                ],
                "data_protection": [
                    "Encrypt sensitive data",
                    "Use secure protocols",
                    "Implement proper key management",
                    "Regular security audits"
                ]
            }
        }

        if domain in best_practices:
            return ToolResult(
                success=True,
                data=best_practices[domain],
                metadata={"domain": domain}
            )

        return ToolResult(
            success=True,
            data=best_practices,
            metadata={"available_domains": list(best_practices.keys())}
        )
