from typing import Dict, List, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from datetime import datetime
import json
import logging
from enum import Enum


class Memory(BaseModel):
    timestamp: datetime
    context: str
    action: str
    result: str
    reflection: Optional[str] = None


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    LEARNING = "learning"


class BaseAgent:
    def __init__(self, agent_id: str, model: str = "gpt-4-turbo-preview"):
        self.agent_id = agent_id
        self.model = model
        self.memories: List[Memory] = []
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.context_window: List[Dict] = []
        self.learning_points: Dict[str, List[str]] = {}
        self.client = AsyncOpenAI()

    async def think(self, task_description: str, context: Dict) -> str:
        """Strategic thinking phase before taking action"""
        self.state = AgentState.THINKING

        # Construct prompt with memory and context
        relevant_memories = self._get_relevant_memories(task_description)
        reflection_insights = self._get_reflection_insights()

        prompt = self._construct_thinking_prompt(
            task_description,
            context,
            relevant_memories,
            reflection_insights
        )

        response = await self._get_completion(prompt)
        self.context_window.append({
            "role": "assistant",
            "content": response
        })

        return response

    async def execute(self, plan: str, context: Dict) -> Dict:
        """Execute the planned action"""
        self.state = AgentState.EXECUTING
        try:
            # Record the action in memory
            memory = Memory(
                timestamp=datetime.now(),
                context=json.dumps(context),
                action=plan,
                result="pending"
            )

            # Execute the plan
            result = await self._execute_plan(plan, context)

            # Update memory with result
            memory.result = json.dumps(result)
            self.memories.append(memory)

            return result
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return {"error": str(e)}

    async def reflect(self, action_result: Dict) -> str:
        """Reflect on the action and its results"""
        self.state = AgentState.REFLECTING

        reflection_prompt = self._construct_reflection_prompt(action_result)
        reflection = await self._get_completion(reflection_prompt)

        # Update the last memory with reflection
        if self.memories:
            self.memories[-1].reflection = reflection

        # Extract and store learning points
        await self._extract_learning_points(reflection)

        return reflection

    async def learn(self) -> None:
        """Consolidate learnings from reflections"""
        self.state = AgentState.LEARNING

        # Analyze patterns in successful and failed actions
        success_patterns = self._analyze_success_patterns()
        failure_patterns = self._analyze_failure_patterns()

        # Update learning points
        self.learning_points["success_patterns"] = success_patterns
        self.learning_points["failure_patterns"] = failure_patterns

        # Adjust future behavior based on learnings
        await self._update_behavior_model()

    def _get_relevant_memories(self, task_description: str) -> List[Memory]:
        """Retrieve relevant memories for the current task"""
        # Implement semantic search over memories
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:5]

    def _get_reflection_insights(self) -> List[str]:
        """Get insights from past reflections"""
        return [m.reflection for m in self.memories if m.reflection][-5:]

    async def _get_completion(self, prompt: str) -> str:
        """Get completion from OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    *self.context_window,
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise

    def _construct_thinking_prompt(self, task: str, context: Dict,
                                   memories: List[Memory],
                                   insights: List[str]) -> str:
        """Construct prompt for the thinking phase"""
        return f"""Task: {task}

Context: {json.dumps(context, indent=2)}

Relevant Past Experiences:
{self._format_memories(memories)}

Past Insights:
{self._format_insights(insights)}

Based on the above information, analyze the task and create a detailed plan.
Consider:
1. Past successes and failures
2. Potential risks and mitigation strategies
3. Best practices and patterns
4. Dependencies and requirements

Provide your analysis and plan in a structured format."""

    def _construct_reflection_prompt(self, result: Dict) -> str:
        """Construct prompt for reflection"""
        return f"""Result of Recent Action:
{json.dumps(result, indent=2)}

Reflect on:
1. What worked well?
2. What could be improved?
3. Any unexpected challenges?
4. Lessons learned
5. How to apply these lessons in future tasks

Provide a detailed reflection with specific insights and actionable improvements."""

    async def _extract_learning_points(self, reflection: str) -> None:
        """Extract and categorize learning points from reflection"""
        prompt = f"""Extract key learning points from this reflection:
{reflection}

Categorize them into:
1. Technical insights
2. Process improvements
3. Best practices
4. Common pitfalls

Format as JSON."""

        learning_points = await self._get_completion(prompt)
        try:
            parsed_points = json.loads(learning_points)
            for category, points in parsed_points.items():
                if category not in self.learning_points:
                    self.learning_points[category] = []
                self.learning_points[category].extend(points)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse learning points")

    def _analyze_success_patterns(self) -> List[str]:
        """Analyze patterns in successful actions"""
        successful_memories = [
            m for m in self.memories
            if isinstance(m.result, str) and "error" not in m.result.lower()
        ]
        # Implement pattern analysis logic
        return ["Success pattern 1", "Success pattern 2"]

    def _analyze_failure_patterns(self) -> List[str]:
        """Analyze patterns in failed actions"""
        failed_memories = [
            m for m in self.memories
            if isinstance(m.result, str) and "error" in m.result.lower()
        ]
        # Implement pattern analysis logic
        return ["Failure pattern 1", "Failure pattern 2"]

    async def _update_behavior_model(self) -> None:
        """Update behavior model based on learnings"""
        # Implement behavior model updating logic
        pass

    async def _execute_plan(self, plan: str, context: Dict) -> Dict:
        """Execute the plan - to be implemented by specific agents"""
        raise NotImplementedError

    def _format_memories(self, memories: List[Memory]) -> str:
        """Format memories for prompt inclusion"""
        return "\n".join([
            f"- {m.timestamp}: {m.action} -> {m.result}"
            for m in memories
        ])

    def _format_insights(self, insights: List[str]) -> str:
        """Format insights for prompt inclusion"""
        return "\n".join([f"- {insight}" for insight in insights])
