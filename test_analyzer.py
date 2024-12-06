import os
import sys
import asyncio
import aiohttp
import structlog
from pydantic import BaseModel
from urllib.parse import quote

logger = structlog.get_logger()


class AnalysisRequest(BaseModel):
    repo_url: str
    analysis_type: str = "full"
    include_patterns: list[str] = ["*.py", "*.java", "*.cs", "*.ts"]
    exclude_patterns: list[str] = ["*test*", "*vendor*", "*node_modules*"]


async def poll_analysis_status(session: aiohttp.ClientSession, tracking_id: str, max_retries: int = 30, delay: int = 2):
    """Poll the analysis status endpoint until completion or timeout."""
    encoded_id = quote(tracking_id)
    for _ in range(max_retries):
        try:
            async with session.get(f"http://app:8000/status/{encoded_id}") as response:
                result = await response.json()
                logger.info("Status check", status=result.get("status"))

                if result.get("status") == "completed":
                    return result.get("result")
                elif result.get("status") == "failed":
                    logger.error("Analysis failed", error=result.get("error"))
                    return None

                await asyncio.sleep(delay)
        except Exception as e:
            logger.error("Error checking status", error=str(e))
            await asyncio.sleep(delay)

    logger.error("Analysis timed out")
    return None


async def test_analyzer():
    try:
        # Create test request
        request = AnalysisRequest(
            repo_url="https://github.com/akumm7491/shared-ddd-ed-microservices-layer",
            analysis_type="full",
            include_patterns=["*.py", "*.java", "*.cs", "*.ts"],
            exclude_patterns=["*test*", "*vendor*", "*node_modules*"]
        )
        logger.info("Created analysis request", repo_url=request.repo_url)

        try:
            # Call the API
            async with aiohttp.ClientSession() as session:
                # Start analysis
                async with session.post(
                    "http://app:8000/analyze",
                    json=request.model_dump()
                ) as response:
                    result = await response.json()
                    logger.info("Analysis started", response=result)

                    if response.status == 200:
                        tracking_id = result.get("tracking_id")
                        if not tracking_id:
                            logger.error("No tracking ID received")
                            return

                        # Poll for results
                        analysis_result = await poll_analysis_status(session, tracking_id)

                        if analysis_result:
                            logger.info("Analysis completed successfully!")
                            logger.info("Results",
                                        total_concepts=len(
                                            analysis_result.get("domain_concepts", [])),
                                        total_contexts=len(
                                            analysis_result.get("bounded_contexts", [])),
                                        total_patterns=len(
                                            analysis_result.get("patterns_found", [])),
                                        metrics=analysis_result.get("metrics", {}))

                            # Print detailed results
                            logger.info("Domain Concepts", concepts=analysis_result.get(
                                "domain_concepts", []))
                            logger.info("Bounded Contexts", contexts=analysis_result.get(
                                "bounded_contexts", []))
                            logger.info("Patterns Found", patterns=analysis_result.get(
                                "patterns_found", []))
                        else:
                            logger.error("Analysis failed or timed out")
                    else:
                        logger.error("Failed to start analysis",
                                     status=response.status, error=result)

        except Exception as e:
            logger.error("Error during analysis", error=str(e))
            import traceback
            traceback.print_exc()
    except Exception as e:
        logger.error("Error in test setup", error=str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ]
    )

    # Run the async test
    asyncio.run(test_analyzer())
