from typing import Dict, Any, List, Optional
from datetime import datetime
import json

class AgentMemory:
    """Manages persistent learning and memory for agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.patterns = {}
        self.success_cases = {}
        self.failure_cases = {}
        self.confidence_adjustments = {}
        
    async def record_success(self, pattern: str, context: Dict[str, Any]):
        """Record a successful pattern recognition."""
        if pattern not in self.success_cases:
            self.success_cases[pattern] = []
            
        self.success_cases[pattern].append({
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        await self._update_confidence(pattern, True)
        
    async def record_failure(self, pattern: str, context: Dict[str, Any]):
        """Record a failed pattern recognition."""
        if pattern not in self.failure_cases:
            self.failure_cases[pattern] = []
            
        self.failure_cases[pattern].append({
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        await self._update_confidence(pattern, False)
    
    async def _update_confidence(self, pattern: str, success: bool):
        """Update confidence scores based on success/failure."""
        if pattern not in self.confidence_adjustments:
            self.confidence_adjustments[pattern] = 1.0
            
        if success:
            self.confidence_adjustments[pattern] *= 1.1  # Increase confidence
        else:
            self.confidence_adjustments[pattern] *= 0.9  # Decrease confidence
            
        # Keep confidence between 0.1 and 2.0
        self.confidence_adjustments[pattern] = max(0.1, min(2.0, self.confidence_adjustments[pattern]))