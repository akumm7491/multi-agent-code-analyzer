from __future__ import annotations
import click
import asyncio
import json
import os
from typing import Optional, Dict, Any, List, Union
from ..sdk.client import Client, AgentType


def print_json(data):
    """Print formatted JSON"""
    click.echo(json.dumps(data, indent=2))


@click.group()
def cli():
    """Multi-Agent Code Analyzer CLI"""
    pass


@cli.command()
@click.argument("repository_url")
@click.option("--branch", default="main", help="Branch to analyze")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--timeout", default=300, help="Timeout in seconds")
def analyze(repository_url: str, branch: str, wait: bool, timeout: int):
    """Analyze a GitHub repository"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    async def run():
        client = Client(github_token=token)
        result = await client.analyze_repository(
            repository_url=repository_url,
            branch=branch,
            wait_for_completion=wait,
            timeout=timeout
        )
        print_json(result)

    asyncio.run(run())


@cli.command()
@click.argument("repo_url")
@click.argument("description")
@click.option("--branch", help="Target branch (optional)")
@click.option("--files", help="Comma-separated list of files to modify")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--timeout", default=600, help="Timeout in seconds")
def implement(repo_url: str, description: str, branch: Optional[str],
              files: Optional[str], wait: bool, timeout: int):
    """Implement a feature or fix"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    target_files = files.split(",") if files else None

    async def run():
        client = Client(github_token=token)
        result = await client.implement_feature(
            repo_url,
            description,
            branch=branch,
            target_files=target_files,
            wait_for_completion=wait,
            timeout=timeout
        )
        print_json(result)

    asyncio.run(run())


@cli.command()
@click.argument("task_id")
def status(task_id: str):
    """Get task status"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    async def run():
        client = Client(github_token=token)
        result = await client.get_task_status(task_id)
        print_json(result)

    asyncio.run(run())


@cli.command()
@click.argument("agent_id")
def memory(agent_id: str):
    """Get agent's memory"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    async def run():
        client = Client(github_token=token)
        result = await client.get_agent_memory(agent_id)
        print_json(result)

    asyncio.run(run())


@cli.command()
@click.argument("agent_id")
def learnings(agent_id: str):
    """Get agent's learning points"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    async def run():
        client = Client(github_token=token)
        result = await client.get_agent_learnings(agent_id)
        print_json(result)

    asyncio.run(run())


@cli.command()
@click.argument("agent_type")
@click.argument("description")
@click.option("--context", help="JSON string of task context")
@click.option("--dependencies", help="Comma-separated list of task IDs")
@click.option("--wait/--no-wait", default=True, help="Wait for completion")
@click.option("--timeout", default=300, help="Timeout in seconds")
def custom_task(agent_type: str, description: str, context: Optional[str],
                dependencies: Optional[str], wait: bool, timeout: int):
    """Create a custom task"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        click.echo("Error: GITHUB_TOKEN environment variable not set")
        return

    try:
        agent_type_enum = AgentType(agent_type)
    except ValueError:
        click.echo(
            f"Error: Invalid agent type. Valid types: {[t.value for t in AgentType]}")
        return

    try:
        context_dict = json.loads(context) if context else {}
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON context")
        return

    deps = dependencies.split(",") if dependencies else None

    async def run():
        client = Client(github_token=token)
        result = await client.create_custom_task(
            agent_type_enum,
            description,
            context_dict,
            dependencies=deps,
            wait_for_completion=wait,
            timeout=timeout
        )
        print_json(result)

    asyncio.run(run())


if __name__ == "__main__":
    cli()
