from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AdaptiveConfig:
    load_threshold: float = 0.8
    scale_factor: float = 1.0
    min_confidence: float = 0.6
    max_retries: int = 3

class DynamicConfigurator:
    def __init__(self):
        self.config = AdaptiveConfig()
        self.performance_metrics = {}
        
    async def adjust_for_load(self, current_load: float):
        if current_load > self.config.load_threshold:
            self.config.scale_factor *= 1.2
        else:
            self.config.scale_factor = max(1.0, self.config.scale_factor * 0.9)
            
    async def update_confidence_threshold(self, success_rate: float):
        if success_rate > 0.8:
            self.config.min_confidence = min(0.8, self.config.min_confidence + 0.05)
        else:
            self.config.min_confidence = max(0.4, self.config.min_confidence - 0.05)