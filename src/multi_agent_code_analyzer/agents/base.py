from typing import Dict, Any, Optional, List
import asyncio
from uuid import UUID, uuid4
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from ..learning.memory import MemoryManager
from ..mcp.models import Context, ContextType
from ..core.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: Optional[str] = None,
        mcp_client: Optional[MCPClient] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.agent_id = agent_id or str(uuid4())
        self.mcp_client = mcp_client or MCPClient()
        self.memory_manager = memory_manager or MemoryManager()
        self.current_task: Optional[Dict[str, Any]] = None
        self.context_history: List[UUID] = []
        self.max_context_length = int(
            os.getenv("MCP_MAX_CONTEXT_LENGTH", "100000"))

    async def initialize(self):
        """Initialize agent components"""
        await self.memory_manager.initialize()
        await self.mcp_client.connect()

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return the result"""
        try:
            self.current_task = task

            # Create task context
            context = await self._create_task_context(task)
            self.context_history.append(context.id)

            # Get relevant memories
            relevant_memories = await self._get_relevant_memories(context)

            # Process task with context and memories
            result = await self._execute_task(task, context, relevant_memories)

            # Store task result in memory
            await self._store_task_result(task, result, context)

            self.current_task = None
            return result

        except Exception as e:
            logger.error(f"Task processing failed: {e}", exc_info=True)
            raise

    async def _create_task_context(self, task: Dict[str, Any]) -> Context:
        """Create context for the current task"""
        context = Context(
            id=uuid4(),
            type=ContextType.SYSTEM,
            content=self._format_task_context(task),
            metadata={
                "agent_id": self.agent_id,
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        await self.mcp_client.store_context(context)
        return context

    def _format_task_context(self, task: Dict[str, Any]) -> str:
        """Format task data into context string"""
        return f"""Task ID: {task.get('id')}
Type: {task.get('type')}
Description: {task.get('description')}
Input Data: {task.get('input_data')}
Timestamp: {datetime.utcnow().isoformat()}"""

    async def _get_relevant_memories(self, context: Context) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for the current context"""
        query_result = await self.mcp_client.query_contexts(
            text=context.content,
            context_type=context.type,
            top_k=5,
            threshold=0.7
        )
        return query_result

    @abstractmethod
    async def _execute_task(
        self,
        task: Dict[str, Any],
        context: Context,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute the task using context and memories"""
        pass

    async def _store_task_result(
        self,
        task: Dict[str, Any],
        result: Dict[str, Any],
        context: Context
    ):
        """Store task result in memory"""
        result_context = Context(
            id=uuid4(),
            type=ContextType.SYSTEM,
            content=self._format_result_context(task, result),
            metadata={
                "agent_id": self.agent_id,
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "result_type": result.get("type"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        await self.mcp_client.store_context(result_context)
        self.context_history.append(result_context.id)

    def _format_result_context(self, task: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Format task result into context string"""
        return f"""Task ID: {task.get('id')}
Type: {task.get('type')}
Result Type: {result.get('type')}
Status: {result.get('status')}
Output: {result.get('output')}
Timestamp: {datetime.utcnow().isoformat()}"""

    async def cleanup(self):
        """Cleanup agent resources"""
        await self.memory_manager.close()
        await self.mcp_client.close()
