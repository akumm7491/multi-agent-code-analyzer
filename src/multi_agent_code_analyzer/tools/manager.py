from typing import Dict, Any, List, Optional, Type
from .base import Tool, ToolResult, ToolCategory, ToolRegistry
from .github import GitHubTool
from .jira import JiraTool
from .search import WebSearchTool
import asyncio
from datetime import datetime


class ToolManager:
    """Manages tool execution and integration with MCP"""

    def __init__(self):
        self.registry = ToolRegistry()
        self._register_default_tools()
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def _register_default_tools(self):
        """Register default tools"""
        self.registry.register(GitHubTool)
        self.registry.register(JiraTool)
        self.registry.register(WebSearchTool)

    def configure_tool(self, tool_name: str, config: Dict[str, Any]):
        """Configure a specific tool"""
        self.tool_configs[tool_name] = config

    async def execute_tool(
        self,
        tool_name: str,
        action: str,
        **kwargs
    ) -> ToolResult:
        """Execute a tool with given parameters"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {tool_name}"
            )

        # Configure tool
        if tool_name in self.tool_configs:
            tool.configure(self.tool_configs[tool_name])

        # Execute tool
        try:
            result = await tool.execute(action=action, **kwargs)

            # Record execution
            self.execution_history.append({
                "tool": tool_name,
                "action": action,
                "params": kwargs,
                "timestamp": datetime.now().isoformat(),
                "success": result.success
            })

            return result
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution error: {str(e)}"
            )

    async def get_best_practices(
        self,
        domain: str,
        tool_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get best practices from multiple tools"""
        if not tool_names:
            tool_names = list(self.registry.tools.keys())

        results = {}
        tasks = []

        for tool_name in tool_names:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue

            if tool_name in self.tool_configs:
                tool.configure(self.tool_configs[tool_name])

            tasks.append(
                tool.execute(
                    action="get_best_practices",
                    domain=domain
                )
            )

        tool_results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_name, result in zip(tool_names, tool_results):
            if isinstance(result, Exception):
                continue
            if isinstance(result, ToolResult) and result.success:
                results[tool_name] = result.data

        return results

    def get_available_tools(self) -> Dict[ToolCategory, List[str]]:
        """Get all available tools by category"""
        tools_by_category = {}
        for category in ToolCategory:
            tools = self.registry.get_tools_by_category(category)
            if tools:
                tools_by_category[category] = tools
        return tools_by_category

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "category": tool.category.value,
            "description": tool.description,
            "version": tool.version
        }

    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get tool execution history"""
        history = self.execution_history
        if tool_name:
            history = [h for h in history if h["tool"] == tool_name]
        return sorted(
            history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
