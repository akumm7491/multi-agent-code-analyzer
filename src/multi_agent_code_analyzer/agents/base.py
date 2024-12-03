class PatternLearner:
    def __init__(self):
        self.patterns = {}
        self.pattern_confidence = {}
        
    async def learn_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], success: bool):
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
        self.pattern_confidence[pattern_type] = max(0.1, min(2.0, self.pattern_confidence[pattern_type]))
        
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