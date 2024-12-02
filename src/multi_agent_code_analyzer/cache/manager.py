from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class CacheManager:
    """Manages caching of analysis results."""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key in self.cache:
            item = self.cache[key]
            if datetime.now() - item['timestamp'] < self.ttl:
                return item['data']
            else:
                del self.cache[key]
        return None
        
    async def set(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }