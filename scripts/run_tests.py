import asyncio
import os
import sys
import logging
import subprocess
import time
import requests
from typing import List, Tuple
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemTestRunner:
    """Runner for system integration tests"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.services_healthy = False

    def check_prerequisites(self) -> List[str]:
        """Check if all required services are installed"""
        missing = []

        # Check Docker
        try:
            self.docker_client.ping()
        except Exception:
            missing.append("Docker")

        # Check Python dependencies
        try:
            import pytest
            import neo4j
            import fastapi
            import fastmcp
        except ImportError as e:
            missing.append(str(e))

        return missing

    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        return False

    def start_services(self) -> bool:
        """Start all required services using docker-compose"""
        try:
            logger.info("Starting services with docker-compose...")
            subprocess.run(
                ["docker-compose", "up", "-d"],
                check=True
            )

            # Wait for services to be ready
            services = [
                ("Neo4j", "http://localhost:7474"),
                ("MCP Server", "http://localhost:8000/health"),
                ("Redis", "http://localhost:6379"),
                ("Grafana", "http://localhost:3000"),
                ("Prometheus", "http://localhost:9090")
            ]

            for service_name, url in services:
                logger.info(f"Waiting for {service_name}...")
                if not self.wait_for_service(url):
                    logger.error(f"{service_name} failed to start")
                    return False
                logger.info(f"{service_name} is ready")

            self.services_healthy = True
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting services: {e}")
            return False

    def run_integration_tests(self) -> Tuple[bool, str]:
        """Run the integration tests"""
        try:
            if not self.services_healthy:
                return False, "Services are not healthy"

            logger.info("Running integration tests...")
            result = subprocess.run(
                ["pytest", "tests/test_integration.py", "-v"],
                capture_output=True,
                text=True
            )

            success = result.returncode == 0
            return success, result.stdout

        except subprocess.CalledProcessError as e:
            return False, str(e)

    def run_system_tests(self) -> Tuple[bool, str]:
        """Run the system tests"""
        try:
            if not self.services_healthy:
                return False, "Services are not healthy"

            logger.info("Running system tests...")
            result = subprocess.run(
                ["pytest", "tests/test_system.py", "-v"],
                capture_output=True,
                text=True
            )

            success = result.returncode == 0
            return success, result.stdout

        except subprocess.CalledProcessError as e:
            return False, str(e)

    def verify_metrics(self) -> bool:
        """Verify system metrics are being collected"""
        try:
            response = requests.get("http://localhost:8080/metrics")
            if response.status_code != 200:
                return False

            metrics = response.json()
            return all(
                key in metrics
                for key in ["task_metrics", "code_metrics", "status"]
            )
        except requests.RequestException:
            return False

    def cleanup(self):
        """Clean up all services"""
        try:
            logger.info("Cleaning up services...")
            subprocess.run(
                ["docker-compose", "down", "-v"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    runner = SystemTestRunner()

    # Check prerequisites
    missing = runner.check_prerequisites()
    if missing:
        logger.error(f"Missing prerequisites: {', '.join(missing)}")
        sys.exit(1)

    try:
        # Start services
        if not runner.start_services():
            logger.error("Failed to start services")
            sys.exit(1)

        # Run integration tests
        success, output = runner.run_integration_tests()
        logger.info("Integration Test Results:")
        print(output)

        if not success:
            logger.error("Integration tests failed")
            sys.exit(1)

        # Run system tests
        success, output = runner.run_system_tests()
        logger.info("System Test Results:")
        print(output)

        if not success:
            logger.error("System tests failed")
            sys.exit(1)

        # Verify metrics
        if not runner.verify_metrics():
            logger.error("Metrics verification failed")
            sys.exit(1)

        logger.info("All tests passed successfully!")

    except KeyboardInterrupt:
        logger.info("Test run interrupted")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
