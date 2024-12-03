from typing import Dict, Any, List, Optional, Union
import uuid


class PatternLearner:
    """A class for learning patterns from agent interactions"""

    def __init__(self):
        self.patterns = {}
        self.pattern_confidence = {}

    async def learn_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], success: bool):
        """Learn a pattern from an interaction"""
        if pattern_type not in self.patterns:
            self.patterns[pattern_type] = []
            self.pattern_confidence[pattern_type] = 1.0

        self.patterns[pattern_type].append(pattern_data)

        # Adjust confidence based on success/failure
        if success:
            self.pattern_confidence[pattern_type] *= 1.1
        else:
            self.pattern_confidence[pattern_type] *= 0.9

        # Keep confidence in reasonable bounds
        self.pattern_confidence[pattern_type] = max(
            0.1, min(2.0, self.pattern_confidence[pattern_type]))

    async def get_pattern_confidence(self, pattern_type: str) -> float:
        return self.pattern_confidence.get(pattern_type, 1.0)

    async def get_similar_patterns(self, pattern_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        similar_patterns = []
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if await self._calculate_similarity(pattern, pattern_data) > 0.7:
                    similar_patterns.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "confidence": self.pattern_confidence[pattern_type]
                    })
        return similar_patterns

    async def _calculate_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns"""
        # Simple implementation - can be enhanced with more sophisticated metrics
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0

        similarity = 0.0
        for key in common_keys:
            if pattern1[key] == pattern2[key]:
                similarity += 1.0

        return similarity / len(common_keys)


class BaseAgent:
    """Base class for all agents in the system"""

    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.pattern_learner = PatternLearner()
        self.memory = {}
        self.capabilities = []

    async def reflect(self, action: str, result: Dict[str, Any], success: bool):
        """Reflect on an action and its result"""
        await self.pattern_learner.learn_pattern("action", {
            "action": action,
            "result": result
        }, success)

    async def learn(self, data: Dict[str, Any]):
        """Learn from new data"""
        pass

    async def adapt(self, feedback: Dict[str, Any]):
        """Adapt behavior based on feedback"""
        pass

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the given context"""
        try:
            # Get task type
            task_type = context.get("type", "unknown")

            # Execute task based on type
            if task_type == "analyze":
                result = await self._analyze(context)
            elif task_type == "implement":
                result = await self._implement(context)
            elif task_type == "custom":
                result = await self._custom_task(context)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Reflect on the execution
            await self.reflect(task_type, result, True)

            return result

        except Exception as e:
            # Reflect on the failure
            await self.reflect(context.get("type", "unknown"), {"error": str(e)}, False)
            raise

    async def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a repository"""
        raise NotImplementedError("Subclasses must implement _analyze()")

    async def _implement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement a feature or fix"""
        raise NotImplementedError("Subclasses must implement _implement()")

    async def _custom_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom task"""
        raise NotImplementedError("Subclasses must implement _custom_task()")
