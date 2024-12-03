from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
from ..tools.manager import ToolManager
from ..context.fastmcp_adapter import FastMCPAdapter, FastMCPContext


@dataclass
class CodeTemplate:
    """Template for code generation"""
    language: str
    framework: str
    patterns: List[str]
    best_practices: Dict[str, Any]
    dependencies: List[str]
    structure: Dict[str, Any]


@dataclass
class GeneratedCode:
    """Result of code generation"""
    code: str
    tests: str
    documentation: str
    quality_score: float
    coverage: float
    dependencies: List[str]
    metadata: Dict[str, Any]


class CodeGenerator:
    """Service for generating production-grade code"""

    def __init__(
        self,
        tool_manager: ToolManager,
        context_adapter: FastMCPAdapter,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tool_manager = tool_manager
        self.context_adapter = context_adapter
        self.config = config or {}
        self.templates: Dict[str, CodeTemplate] = {}

    async def _load_templates(self):
        """Load code templates from tools"""
        # Get templates from GitHub
        github_result = await self.tool_manager.execute_tool(
            "GitHubTool",
            "get_best_practices",
            language="all"
        )

        if github_result.success:
            for language, practices in github_result.data.items():
                self.templates[language] = CodeTemplate(
                    language=language,
                    framework=practices.get("framework", ""),
                    patterns=practices.get("patterns", []),
                    best_practices=practices.get("best_practices", {}),
                    dependencies=practices.get("dependencies", []),
                    structure=practices.get("structure", {})
                )

    async def _get_similar_code(
        self,
        description: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Find similar code examples"""
        embeddings = await self.context_adapter.get_embeddings(description)
        return await self.context_adapter.search_similar_contexts(
            embeddings,
            limit=5,
            min_similarity=0.7
        )

    async def _generate_tests(
        self,
        code: str,
        language: str
    ) -> str:
        """Generate unit tests for the code"""
        template = self.templates.get(language)
        if not template:
            return ""

        # Use test patterns from template
        test_framework = template.best_practices.get(
            "testing", {}).get("framework")
        if not test_framework:
            return ""

        # Generate tests based on code analysis
        test_code = f"""
        import {test_framework}

        def test_functionality():
            # TODO: Implement tests
            pass
        """

        return test_code

    async def _generate_documentation(
        self,
        code: str,
        language: str
    ) -> str:
        """Generate documentation for the code"""
        template = self.templates.get(language)
        if not template:
            return ""

        doc_style = template.best_practices.get(
            "documentation", {}).get("style")
        if not doc_style:
            return ""

        # Generate documentation based on code analysis
        documentation = f"""
        # Module Documentation

        ## Overview
        This module implements...

        ## Usage
        ```{language}
        # Example usage
        ```

        ## API Reference
        ...
        """

        return documentation

    async def generate_code(
        self,
        description: str,
        language: str,
        framework: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GeneratedCode:
        """Generate production-grade code"""
        # Load templates if not loaded
        if not self.templates:
            await self._load_templates()

        # Get template for language
        template = self.templates.get(language)
        if not template:
            raise ValueError(f"No template available for {language}")

        # Get similar code examples
        similar_code = await self._get_similar_code(description, language)

        # Generate code structure
        code_structure = template.structure.copy()
        if framework:
            code_structure["framework"] = framework
        if patterns:
            code_structure["patterns"] = patterns

        # Generate main code
        # This is a placeholder for actual code generation
        code = f"""
        def main():
            # TODO: Implement based on description and patterns
            pass
        """

        # Generate tests
        tests = await self._generate_tests(code, language)

        # Generate documentation
        documentation = await self._generate_documentation(code, language)

        # Store in context
        context = FastMCPContext(
            content=code,
            metadata={
                "language": language,
                "framework": framework,
                "patterns": patterns,
                "description": description,
                **(metadata or {})
            },
            relationships=[
                {
                    "type": "similar_to",
                    "target_id": result["context_id"]
                }
                for result in similar_code
            ]
        )

        await self.context_adapter.store_context(
            f"code_{hash(description)}",
            context
        )

        return GeneratedCode(
            code=code,
            tests=tests,
            documentation=documentation,
            quality_score=0.9,  # Placeholder
            coverage=0.8,  # Placeholder
            dependencies=template.dependencies,
            metadata={
                "language": language,
                "framework": framework,
                "patterns": patterns,
                "similar_examples": len(similar_code)
            }
        )

    async def improve_code(
        self,
        code: str,
        language: str,
        suggestions: List[str]
    ) -> GeneratedCode:
        """Improve existing code based on suggestions"""
        # Get template for language
        template = self.templates.get(language)
        if not template:
            raise ValueError(f"No template available for {language}")

        # Analyze current code
        # This is a placeholder for actual code analysis
        current_quality = 0.7

        # Apply improvements
        improved_code = code  # Placeholder for actual improvement

        # Generate new tests
        tests = await self._generate_tests(improved_code, language)

        # Generate updated documentation
        documentation = await self._generate_documentation(improved_code, language)

        return GeneratedCode(
            code=improved_code,
            tests=tests,
            documentation=documentation,
            quality_score=0.9,  # Should be higher than current_quality
            coverage=0.85,
            dependencies=template.dependencies,
            metadata={
                "language": language,
                "original_quality": current_quality,
                "improvements": suggestions
            }
        )

    async def validate_code(
        self,
        code: str,
        language: str,
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate code against requirements and best practices"""
        template = self.templates.get(language)
        if not template:
            raise ValueError(f"No template available for {language}")

        # Validate against best practices
        practices_result = await self.tool_manager.execute_tool(
            "GitHubTool",
            "validate_code",
            code=code,
            language=language,
            practices=template.best_practices
        )

        validation_results = {
            "passes_best_practices": practices_result.success,
            "issues": practices_result.data.get("issues", []),
            "suggestions": practices_result.data.get("suggestions", []),
            "quality_score": practices_result.data.get("quality_score", 0.0)
        }

        if requirements:
            # Validate against specific requirements
            validation_results["requirements_met"] = all(
                req in code for req in requirements
            )

        return validation_results
