import asyncio
import logging
from uuid import uuid4
from multi_agent_code_analyzer.agents.communication import (
    AgentCommunicationService,
    MessageType,
    AgentMessage
)
from multi_agent_code_analyzer.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeAnalyzerAgent:
    def __init__(self, agent_id: str, communication_service: AgentCommunicationService):
        self.agent_id = agent_id
        self.comm = communication_service

    async def start(self):
        """Start the agent"""
        # Register message handlers
        self.comm.register_handler(
            MessageType.TASK_ASSIGNMENT,
            self.handle_task_assignment
        )
        self.comm.register_handler(
            MessageType.CODE_ANALYSIS,
            self.handle_code_analysis
        )

        # Subscribe to messages
        await self.comm.subscribe(self.agent_id)
        logger.info(f"Agent {self.agent_id} started")

    async def handle_task_assignment(self, message: AgentMessage):
        """Handle task assignment messages"""
        logger.info(f"Agent {self.agent_id} received task: {message.content}")

        # Simulate processing
        await asyncio.sleep(2)

        # Send analysis request to another agent
        analysis_message = self.comm.create_message(
            message_type=MessageType.CODE_ANALYSIS,
            sender_id=self.agent_id,
            recipient_id="analyzer_2",  # Send to specific agent
            content={
                "task_id": message.content["task_id"],
                "code_snippet": message.content["code"],
                "analysis_type": "security_review"
            },
            correlation_id=message.message_id
        )
        await self.comm.publish_message(analysis_message)

    async def handle_code_analysis(self, message: AgentMessage):
        """Handle code analysis messages"""
        logger.info(f"Agent {self.agent_id} analyzing code: {message.content}")

        # Simulate analysis
        await asyncio.sleep(1)

        # Send completion message
        completion_message = self.comm.create_message(
            message_type=MessageType.TASK_COMPLETION,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={
                "task_id": message.content["task_id"],
                "result": "Code analysis completed successfully",
                "findings": ["No security issues found"]
            },
            correlation_id=message.correlation_id
        )
        await self.comm.publish_message(completion_message)


async def run_demo():
    """Run a demonstration of agent communication"""
    settings = get_settings()

    # Create communication service
    comm_service = AgentCommunicationService(settings.database.REDIS_URI)
    await comm_service.connect()

    try:
        # Create agents
        agent1 = CodeAnalyzerAgent("analyzer_1", comm_service)
        agent2 = CodeAnalyzerAgent("analyzer_2", comm_service)

        # Start agents
        await asyncio.gather(
            agent1.start(),
            agent2.start()
        )

        # Simulate task assignment
        task_message = comm_service.create_message(
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id="orchestrator",
            recipient_id="analyzer_1",
            content={
                "task_id": str(uuid4()),
                "code": "def example(): pass",
                "priority": "high"
            }
        )
        await comm_service.publish_message(task_message)

        # Let the demonstration run for a while
        await asyncio.sleep(10)

    finally:
        # Cleanup
        await comm_service.disconnect()


def main():
    """Main entry point"""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
