import logging
import uvicorn
import sys
from multi_agent_code_analyzer.api.main import app
from multi_agent_code_analyzer.config.settings import get_settings, initialize_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize settings
initialize_settings()
settings = get_settings()

# Add startup event handler


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    try:
        logger.info("Successfully imported settings")
        logger.info(f"Environment: {settings.service.ENVIRONMENT}")
        logger.info(f"Debug mode: {settings.service.DEBUG}")
        logger.info(f"Service port: {settings.service.SERVICE_PORT}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting the server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(settings.service.SERVICE_PORT),
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
