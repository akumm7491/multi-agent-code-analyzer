from typing import Dict, Any, List, Optional
import aiohttp
from .base import Tool, ToolResult, ToolCategory
from datetime import datetime


class GitHubTool(Tool):
    """GitHub integration tool"""

    def __init__(self):
        super().__init__()
        self.category = ToolCategory.VERSION_CONTROL
        self.name = "GitHubTool"
        self.description = "Interact with GitHub repositories and APIs"

    async def validate(self, **kwargs) -> bool:
        required = {"token", "owner", "repo"}
        return all(key in self.config for key in required)

    async def execute(self, **kwargs) -> ToolResult:
        if not await self.validate(**kwargs):
            return ToolResult(
                success=False,
                data=None,
                error="Missing required configuration"
            )

        action = kwargs.get("action")
        if not action:
            return ToolResult(
                success=False,
                data=None,
                error="No action specified"
            )

        handlers = {
            "get_repo": self._get_repo,
            "create_pr": self._create_pull_request,
            "get_issues": self._get_issues,
            "create_issue": self._create_issue,
            "get_workflows": self._get_workflows,
            "get_best_practices": self._get_best_practices
        }

        handler = handlers.get(action)
        if not handler:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown action: {action}"
            )

        return await handler(**kwargs)

    async def _get_repo(self, **kwargs) -> ToolResult:
        """Get repository information"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.config['token']}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-Agent"
            }

            url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}"

            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    elif response.status == 401:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="GitHub API authentication failed. Please check your token."
                        )
                    elif response.status == 404:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Repository {self.config['owner']}/{self.config['repo']} not found"
                        )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"GitHub API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _create_pull_request(self, **kwargs) -> ToolResult:
        """Create a pull request"""
        title = kwargs.get("title")
        body = kwargs.get("body")
        head = kwargs.get("head")
        base = kwargs.get("base", "main")

        if not all([title, body, head]):
            return ToolResult(
                success=False,
                data=None,
                error="Missing required parameters"
            )

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.config['token']}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-Agent"
            }

            url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}/pulls"

            data = {
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }

            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status in (200, 201):
                        result = await response.json()
                        return ToolResult(success=True, data=result)
                    elif response.status == 401:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="GitHub API authentication failed. Please check your token."
                        )
                    elif response.status == 404:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Repository {self.config['owner']}/{self.config['repo']} not found"
                        )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"GitHub API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_best_practices(self, **kwargs) -> ToolResult:
        """Get repository best practices and templates"""
        language = kwargs.get("language", "").lower()

        best_practices = {
            "python": {
                "project_structure": [
                    "src/",
                    "tests/",
                    "docs/",
                    "requirements/",
                    "scripts/"
                ],
                "code_style": {
                    "formatter": "black",
                    "linter": "flake8",
                    "type_checker": "mypy"
                },
                "testing": {
                    "framework": "pytest",
                    "coverage": "pytest-cov"
                },
                "documentation": {
                    "docstring_style": "Google",
                    "documentation_generator": "Sphinx"
                },
                "ci_cd": {
                    "github_actions": [
                        "lint.yml",
                        "test.yml",
                        "build.yml",
                        "deploy.yml"
                    ]
                }
            },
            "typescript": {
                "project_structure": [
                    "src/",
                    "test/",
                    "docs/",
                    "scripts/"
                ],
                "code_style": {
                    "formatter": "prettier",
                    "linter": "eslint",
                    "type_checker": "tsc"
                },
                "testing": {
                    "framework": "jest",
                    "coverage": "jest --coverage"
                },
                "documentation": {
                    "generator": "TypeDoc"
                },
                "ci_cd": {
                    "github_actions": [
                        "lint.yml",
                        "test.yml",
                        "build.yml",
                        "deploy.yml"
                    ]
                }
            }
        }

        if language in best_practices:
            return ToolResult(
                success=True,
                data=best_practices[language],
                metadata={"language": language}
            )

        return ToolResult(
            success=True,
            data=best_practices,
            metadata={"available_languages": list(best_practices.keys())}
        )

    async def _get_issues(self, **kwargs) -> ToolResult:
        """Get repository issues"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.config['token']}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-Agent"
            }

            url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}/issues"

            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    elif response.status == 401:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="GitHub API authentication failed. Please check your token."
                        )
                    elif response.status == 404:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Repository {self.config['owner']}/{self.config['repo']} not found"
                        )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"GitHub API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _create_issue(self, **kwargs) -> ToolResult:
        """Create a repository issue"""
        title = kwargs.get("title")
        body = kwargs.get("body")
        labels = kwargs.get("labels", [])

        if not all([title, body]):
            return ToolResult(
                success=False,
                data=None,
                error="Missing required parameters"
            )

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.config['token']}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-Agent"
            }

            url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}/issues"

            data = {
                "title": title,
                "body": body,
                "labels": labels
            }

            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status in (200, 201):
                        result = await response.json()
                        return ToolResult(success=True, data=result)
                    elif response.status == 401:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="GitHub API authentication failed. Please check your token."
                        )
                    elif response.status == 404:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Repository {self.config['owner']}/{self.config['repo']} not found"
                        )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"GitHub API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_workflows(self, **kwargs) -> ToolResult:
        """Get repository workflows"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.config['token']}",
                "Accept": "application/vnd.github.v3+json"
            }

            url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}/actions/workflows"

            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"GitHub API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))
