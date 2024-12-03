from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from enum import Enum


class ToolCategory(Enum):
    VERSION_CONTROL = "version_control"
    PROJECT_MANAGEMENT = "project_management"
    DEVELOPMENT = "development"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SEARCH = "search"


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Base class for all tools"""

    def __init__(self):
        self.category: ToolCategory = ToolCategory.DEVELOPMENT
        self.name: str = self.__class__.__name__
        self.description: str = ""
        self.version: str = "1.0.0"
        self.config: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    @abstractmethod
    async def validate(self, **kwargs) -> bool:
        """Validate tool parameters"""
        pass

    def configure(self, config: Dict[str, Any]):
        """Configure the tool"""
        self.config.update(config)


class ToolRegistry:
    """Registry for managing available tools"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tools: Dict[str, Type[Tool]] = {}
            cls._instance.categories: Dict[ToolCategory, List[str]] = {
                category: [] for category in ToolCategory
            }
        return cls._instance

    def register(self, tool_class: Type[Tool]):
        """Register a new tool"""
        tool = tool_class()
        self.tools[tool.name] = tool_class
        self.categories[tool.category].append(tool.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool instance by name"""
        tool_class = self.tools.get(name)
        return tool_class() if tool_class else None

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tool names in a category"""
        return self.categories.get(category, [])
