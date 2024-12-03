from typing import Dict, Any, List, Optional
import aiohttp
from .base import Tool, ToolResult, ToolCategory
from datetime import datetime
import asyncio
import base64
import logging
import os
from pathlib import Path
import tempfile
import git


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


class GithubService:
    """Service for interacting with GitHub"""

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or os.getenv("GITHUB_TOKEN")
        self.api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.logger = logging.getLogger(__name__)

    async def clone_repository(self, repo_url: str, branch: str = "main") -> str:
        """Clone a repository to a temporary directory"""
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()

            # Add token to URL for authentication
            if self.access_token:
                repo_url = repo_url.replace(
                    "https://",
                    f"https://{self.access_token}@"
                )

            # Clone repository
            repo = git.Repo.clone_from(
                repo_url,
                temp_dir,
                branch=branch
            )

            self.logger.info(f"Cloned repository to {temp_dir}")
            return temp_dir

        except Exception as e:
            self.logger.error(f"Failed to clone repository: {str(e)}")
            raise

    async def create_branch(self, repo_url: str, branch_name: str) -> bool:
        """Create a new branch in the repository"""
        try:
            owner, repo = self._parse_repo_url(repo_url)
            default_branch = await self._get_default_branch(owner, repo)

            # Get the SHA of the default branch
            sha = await self._get_ref_sha(owner, repo, f"heads/{default_branch}")

            # Create new branch
            async with aiohttp.ClientSession(headers=self.headers) as session:
                url = f"{self.api_base}/repos/{owner}/{repo}/git/refs"
                data = {
                    "ref": f"refs/heads/{branch_name}",
                    "sha": sha
                }

                async with session.post(url, json=data) as response:
                    if response.status == 201:
                        self.logger.info(f"Created branch {branch_name}")
                        return True
                    else:
                        error = await response.text()
                        self.logger.error(f"Failed to create branch: {error}")
                        return False

        except Exception as e:
            self.logger.error(f"Failed to create branch: {str(e)}")
            return False

    async def create_pull_request(self, repo_url: str, branch: str,
                                  title: str, body: str) -> str:
        """Create a pull request"""
        try:
            owner, repo = self._parse_repo_url(repo_url)
            default_branch = await self._get_default_branch(owner, repo)

            async with aiohttp.ClientSession(headers=self.headers) as session:
                url = f"{self.api_base}/repos/{owner}/{repo}/pulls"
                data = {
                    "title": title,
                    "body": body,
                    "head": branch,
                    "base": default_branch
                }

                async with session.post(url, json=data) as response:
                    if response.status == 201:
                        result = await response.json()
                        self.logger.info(
                            f"Created pull request: {result['html_url']}")
                        return result['html_url']
                    else:
                        error = await response.text()
                        self.logger.error(
                            f"Failed to create pull request: {error}")
                        raise Exception(
                            f"Failed to create pull request: {error}")

        except Exception as e:
            self.logger.error(f"Failed to create pull request: {str(e)}")
            raise

    async def commit_changes(self, repo_url: str, branch: str,
                             files: Dict[str, str], message: str) -> bool:
        """Commit changes to files"""
        try:
            owner, repo = self._parse_repo_url(repo_url)

            # Get the current commit SHA
            sha = await self._get_ref_sha(owner, repo, f"heads/{branch}")

            # Create blobs for each file
            blobs = []
            for path, content in files.items():
                blob = await self._create_blob(owner, repo, content)
                blobs.append((path, blob))

            # Create tree
            tree = await self._create_tree(owner, repo, sha, blobs)

            # Create commit
            commit = await self._create_commit(owner, repo, message, tree, [sha])

            # Update reference
            return await self._update_ref(owner, repo, f"heads/{branch}", commit)

        except Exception as e:
            self.logger.error(f"Failed to commit changes: {str(e)}")
            return False

    async def _get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch of a repository"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['default_branch']
                else:
                    raise Exception(f"Failed to get repository info: {await response.text()}")

    async def _get_ref_sha(self, owner: str, repo: str, ref: str) -> str:
        """Get the SHA for a reference"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}/git/refs/{ref}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['object']['sha']
                else:
                    raise Exception(f"Failed to get ref: {await response.text()}")

    async def _create_blob(self, owner: str, repo: str, content: str) -> str:
        """Create a blob for file content"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}/git/blobs"
            data = {
                "content": base64.b64encode(content.encode()).decode(),
                "encoding": "base64"
            }

            async with session.post(url, json=data) as response:
                if response.status == 201:
                    result = await response.json()
                    return result['sha']
                else:
                    raise Exception(f"Failed to create blob: {await response.text()}")

    async def _create_tree(self, owner: str, repo: str,
                           base_tree: str, blobs: List[tuple]) -> str:
        """Create a tree with the given blobs"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}/git/trees"
            data = {
                "base_tree": base_tree,
                "tree": [
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha
                    }
                    for path, blob_sha in blobs
                ]
            }

            async with session.post(url, json=data) as response:
                if response.status == 201:
                    result = await response.json()
                    return result['sha']
                else:
                    raise Exception(f"Failed to create tree: {await response.text()}")

    async def _create_commit(self, owner: str, repo: str,
                             message: str, tree: str, parents: List[str]) -> str:
        """Create a commit"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}/git/commits"
            data = {
                "message": message,
                "tree": tree,
                "parents": parents
            }

            async with session.post(url, json=data) as response:
                if response.status == 201:
                    result = await response.json()
                    return result['sha']
                else:
                    raise Exception(f"Failed to create commit: {await response.text()}")

    async def _update_ref(self, owner: str, repo: str, ref: str, sha: str) -> bool:
        """Update a reference to point to a commit"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.api_base}/repos/{owner}/{repo}/git/refs/{ref}"
            data = {"sha": sha}

            async with session.patch(url, json=data) as response:
                return response.status == 200

    def _parse_repo_url(self, repo_url: str) -> tuple:
        """Parse owner and repo from GitHub URL"""
        parts = repo_url.rstrip("/").split("/")
        return parts[-2], parts[-1]
