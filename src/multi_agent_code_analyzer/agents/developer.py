from typing import Dict, List, Optional
import json
from .base_agent import BaseAgent
from ..tools.github import GithubService
import logging
import ast
from pathlib import Path


class DeveloperAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.current_changes: Dict[str, str] = {}
        self.test_cases: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(f"agent.developer.{agent_id}")

    async def implement_feature(self, task_description: str, context: Dict) -> Dict:
        """Implement a new feature"""
        # Think about implementation strategy
        plan = await self.think(f"Implement feature: {task_description}", context)

        # Execute implementation
        result = await self.execute(plan, context)

        # Reflect on implementation
        reflection = await self.reflect(result)

        # Learn from implementation
        await self.learn()

        return {
            "changes": result,
            "reflection": reflection,
            "tests": self.test_cases
        }

    async def _execute_plan(self, plan: str, context: Dict) -> Dict:
        """Execute implementation plan"""
        github_service = GithubService(context["access_token"])

        try:
            # Create feature branch
            branch_name = f"feature/{context.get('task_id', 'new-feature')}"
            await github_service.create_branch(context["repo_url"], branch_name)

            # Implement changes
            changes = await self._implement_changes(plan, context)

            # Write tests
            tests = await self._write_tests(changes)

            # Create pull request
            pr_url = await github_service.create_pull_request(
                context["repo_url"],
                branch_name,
                f"Feature: {context.get('task_description', 'New Feature')}",
                self._generate_pr_description(changes, tests)
            )

            return {
                "changes": changes,
                "tests": tests,
                "pull_request_url": pr_url
            }

        except Exception as e:
            self.logger.error(f"Implementation failed: {str(e)}")
            return {"error": str(e)}

    async def _implement_changes(self, plan: str, context: Dict) -> Dict[str, Dict]:
        """Implement code changes based on plan"""
        changes = {}

        # Parse implementation plan
        implementation_steps = await self._parse_implementation_plan(plan)

        for step in implementation_steps:
            file_path = step["file"]
            change_type = step["type"]

            if change_type == "create":
                content = await self._generate_file_content(
                    step["description"],
                    step.get("template"),
                    context
                )
                changes[file_path] = {
                    "type": "create",
                    "content": content
                }

            elif change_type == "modify":
                original_content = await self._read_file_content(file_path)
                modified_content = await self._modify_file_content(
                    original_content,
                    step["changes"],
                    context
                )
                changes[file_path] = {
                    "type": "modify",
                    "content": modified_content
                }

        self.current_changes = changes
        return changes

    async def _write_tests(self, changes: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Write tests for implemented changes"""
        tests = {}

        for file_path, change in changes.items():
            if change["type"] == "create":
                tests[file_path] = await self._generate_tests(
                    file_path,
                    change["content"],
                    "new_file"
                )
            else:
                tests[file_path] = await self._generate_tests(
                    file_path,
                    change["content"],
                    "modified_file"
                )

        self.test_cases = tests
        return tests

    async def _generate_tests(self, file_path: str, content: str, change_type: str) -> List[str]:
        """Generate test cases for a file"""
        prompt = f"""Generate test cases for:
File: {file_path}
Content:
{content}

Consider:
1. Unit tests
2. Integration tests
3. Edge cases
4. Error scenarios
5. Performance scenarios

Format test cases as Python code."""

        test_code = await self._get_completion(prompt)
        return [test_code]

    async def _parse_implementation_plan(self, plan: str) -> List[Dict]:
        """Parse implementation plan into structured steps"""
        prompt = f"""Parse this implementation plan into structured steps:
{plan}

Format as JSON array with fields:
- file: file path
- type: create/modify
- description: what to implement
- template: optional template to use
- changes: list of changes for modifications

Each step should be atomic and independently implementable."""

        steps = await self._get_completion(prompt)
        return json.loads(steps)

    async def _generate_file_content(self, description: str,
                                     template: Optional[str],
                                     context: Dict) -> str:
        """Generate content for a new file"""
        prompt = f"""Generate code for:
Description: {description}
Template: {template or 'None'}
Context: {json.dumps(context, indent=2)}

Consider:
1. Best practices
2. Design patterns
3. Error handling
4. Documentation
5. Type hints
6. Performance

Generate complete, production-ready code."""

        return await self._get_completion(prompt)

    async def _modify_file_content(self, original_content: str,
                                   changes: List[Dict],
                                   context: Dict) -> str:
        """Modify existing file content"""
        prompt = f"""Modify this code according to the changes:
Original Content:
{original_content}

Changes to make:
{json.dumps(changes, indent=2)}

Context:
{json.dumps(context, indent=2)}

Provide the complete modified code while maintaining:
1. Code style consistency
2. Existing patterns
3. Error handling
4. Documentation
5. Performance"""

        return await self._get_completion(prompt)

    def _generate_pr_description(self, changes: Dict[str, Dict],
                                 tests: Dict[str, List[str]]) -> str:
        """Generate pull request description"""
        description = ["# Changes Implemented\n"]

        for file_path, change in changes.items():
            description.append(f"## {file_path}")
            if change["type"] == "create":
                description.append("- Created new file")
            else:
                description.append("- Modified existing file")

            if file_path in tests:
                description.append("\nTests added:")
                for test in tests[file_path]:
                    description.append(f"- {test.splitlines()[0]}")

            description.append("")

        return "\n".join(description)

    async def review_code(self, file_path: str, content: str) -> Dict:
        """Review code for quality and improvements"""
        context = {
            "file_path": file_path,
            "content": content
        }

        prompt = f"""Review this code for quality and improvements:
File: {file_path}
Content:
{content}

Provide detailed review considering:
1. Code quality
2. Performance
3. Security
4. Maintainability
5. Test coverage
6. Documentation

Format as JSON with 'issues' and 'suggestions' fields."""

        review = await self._get_completion(prompt)
        return json.loads(review)

    async def _read_file_content(self, file_path: str) -> str:
        """Read content of a file"""
        # Implement file reading logic
        return ""
