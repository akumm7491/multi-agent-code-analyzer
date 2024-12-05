from setuptools import setup, find_packages

setup(
    name="multi-agent-code-analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.21.0",
        "httpx>=0.24.0",
        "neo4j>=4.4.0",
        "redis>=4.0.0",
        "pymilvus>=2.0.0",
        "minio>=7.0.0",
        "sentence-transformers>=2.2.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=0.19.0",
        "prometheus-client>=0.12.0",
        "websockets>=10.0",
        "starlette>=0.39.0",
        "aioredis>=2.0.0",
        "asyncio>=3.4.3",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.0.0"
    ],
    python_requires=">=3.11",
)
