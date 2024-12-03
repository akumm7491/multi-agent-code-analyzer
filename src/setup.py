from setuptools import setup, find_packages

setup(
    name="multi_agent_code_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "fastmcp>=0.3.4",
        "pydantic>=2.0.0",
        "neo4j>=4.4.0",
        "redis>=4.0.0",
        "prometheus-client>=0.11.0",
        "pymilvus>=2.3.1"
    ],
    python_requires=">=3.8",
)
