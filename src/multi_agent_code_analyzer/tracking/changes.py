from typing import Dict, Any, List
from datetime import datetime

class ChangeTracker:
    """Tracks and manages code changes made by agents."""
    
    def __init__(self):
        self.changes = []
        self.pending_changes = {}
        
    async def record_change(self, file_path: str, change_type: str, details: Dict[str, Any]):
        """Record a code change."""
        change = {
            "file": file_path,
            "type": change_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        self.changes.append(change)
        self.pending_changes[file_path] = change
        
    async def commit_change(self, file_path: str):
        """Mark a change as committed."""
        if file_path in self.pending_changes:
            self.pending_changes[file_path]["status"] = "committed"
            del self.pending_changes[file_path]