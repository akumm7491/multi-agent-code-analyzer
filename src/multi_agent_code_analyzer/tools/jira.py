from typing import Dict, Any, List, Optional
import aiohttp
from .base import Tool, ToolResult, ToolCategory
import base64


class JiraTool(Tool):
    """JIRA integration tool"""

    def __init__(self):
        super().__init__()
        self.category = ToolCategory.PROJECT_MANAGEMENT
        self.name = "JiraTool"
        self.description = "Interact with JIRA for project management"

    async def validate(self, **kwargs) -> bool:
        required = {"domain", "email", "api_token", "project_key"}
        return all(key in self.config for key in required)

    def _get_auth_header(self) -> Dict[str, str]:
        """Get basic auth header for JIRA API"""
        auth_str = f"{self.config['email']}:{self.config['api_token']}"
        auth_bytes = auth_str.encode('ascii')
        base64_auth = base64.b64encode(auth_bytes).decode('ascii')
        return {
            "Authorization": f"Basic {base64_auth}",
            "Accept": "application/json"
        }

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
            "create_issue": self._create_issue,
            "get_issue": self._get_issue,
            "search_issues": self._search_issues,
            "get_sprint": self._get_sprint,
            "get_project_templates": self._get_project_templates,
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

    async def _create_issue(self, **kwargs) -> ToolResult:
        """Create a JIRA issue"""
        summary = kwargs.get("summary")
        description = kwargs.get("description")
        issue_type = kwargs.get("issue_type", "Task")

        if not all([summary, description]):
            return ToolResult(
                success=False,
                data=None,
                error="Missing required parameters"
            )

        async with aiohttp.ClientSession() as session:
            url = f"https://{self.config['domain']}/rest/api/3/issue"

            data = {
                "fields": {
                    "project": {
                        "key": self.config["project_key"]
                    },
                    "summary": summary,
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": description
                                    }
                                ]
                            }
                        ]
                    },
                    "issuetype": {
                        "name": issue_type
                    }
                }
            }

            try:
                async with session.post(
                    url,
                    headers=self._get_auth_header(),
                    json=data
                ) as response:
                    if response.status in (200, 201):
                        result = await response.json()
                        return ToolResult(success=True, data=result)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"JIRA API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_issue(self, **kwargs) -> ToolResult:
        """Get a JIRA issue"""
        issue_key = kwargs.get("issue_key")
        if not issue_key:
            return ToolResult(
                success=False,
                data=None,
                error="Missing issue key"
            )

        async with aiohttp.ClientSession() as session:
            url = f"https://{self.config['domain']}/rest/api/3/issue/{issue_key}"

            try:
                async with session.get(
                    url,
                    headers=self._get_auth_header()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"JIRA API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _search_issues(self, **kwargs) -> ToolResult:
        """Search JIRA issues"""
        jql = kwargs.get("jql", f"project = {self.config['project_key']}")

        async with aiohttp.ClientSession() as session:
            url = f"https://{self.config['domain']}/rest/api/3/search"

            params = {
                "jql": jql,
                "maxResults": kwargs.get("max_results", 50)
            }

            try:
                async with session.get(
                    url,
                    headers=self._get_auth_header(),
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"JIRA API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_sprint(self, **kwargs) -> ToolResult:
        """Get sprint information"""
        board_id = kwargs.get("board_id")
        if not board_id:
            return ToolResult(
                success=False,
                data=None,
                error="Missing board ID"
            )

        async with aiohttp.ClientSession() as session:
            url = f"https://{self.config['domain']}/rest/agile/1.0/board/{board_id}/sprint"

            try:
                async with session.get(
                    url,
                    headers=self._get_auth_header()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ToolResult(success=True, data=data)
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"JIRA API error: {response.status}"
                    )
            except Exception as e:
                return ToolResult(success=False, data=None, error=str(e))

    async def _get_project_templates(self, **kwargs) -> ToolResult:
        """Get project templates and best practices"""
        templates = {
            "agile": {
                "issue_types": [
                    "Epic",
                    "Story",
                    "Task",
                    "Bug",
                    "Sub-task"
                ],
                "workflows": {
                    "default": [
                        "To Do",
                        "In Progress",
                        "In Review",
                        "Done"
                    ],
                    "bug": [
                        "Open",
                        "Investigating",
                        "In Progress",
                        "Testing",
                        "Closed"
                    ]
                },
                "fields": {
                    "story_points": "customfield_10026",
                    "sprint": "customfield_10020",
                    "epic_link": "customfield_10014"
                }
            },
            "kanban": {
                "issue_types": [
                    "Task",
                    "Bug",
                    "Improvement"
                ],
                "workflows": {
                    "default": [
                        "Backlog",
                        "Selected for Development",
                        "In Progress",
                        "Done"
                    ]
                }
            }
        }

        template_type = kwargs.get("type", "agile")
        if template_type in templates:
            return ToolResult(
                success=True,
                data=templates[template_type],
                metadata={"type": template_type}
            )

        return ToolResult(
            success=True,
            data=templates,
            metadata={"available_types": list(templates.keys())}
        )

    async def _get_best_practices(self, **kwargs) -> ToolResult:
        """Get JIRA best practices"""
        best_practices = {
            "issue_management": {
                "naming_conventions": {
                    "epics": "Domain: Feature Area",
                    "stories": "[Feature] Action Description",
                    "tasks": "Specific Implementation Task",
                    "bugs": "[Bug] Issue Description"
                },
                "description_template": """
                    *Objective*
                    What needs to be done
                    
                    *Acceptance Criteria*
                    - Criterion 1
                    - Criterion 2
                    
                    *Technical Notes*
                    Implementation details
                    
                    *Dependencies*
                    - Dependency 1
                    - Dependency 2
                """,
                "story_points": {
                    "1": "Trivial change",
                    "2": "Simple change",
                    "3": "Medium complexity",
                    "5": "Complex change",
                    "8": "Very complex change",
                    "13": "Major feature"
                }
            },
            "workflow": {
                "code_review": {
                    "checklist": [
                        "Code follows style guide",
                        "Tests are included",
                        "Documentation is updated",
                        "No security issues"
                    ]
                },
                "testing": {
                    "types": [
                        "Unit tests",
                        "Integration tests",
                        "End-to-end tests"
                    ]
                }
            }
        }

        return ToolResult(
            success=True,
            data=best_practices
        )
