from typing import Dict, Any, List
from pathlib import Path
import shutil
import json
from datetime import datetime

class RollbackManager:
    """Manages rollback operations for code changes."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / ".backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.change_log = []
        
    async def create_backup(self, files: List[str]) -> str:
        """Create a backup of files before modification."""
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir()
        
        backup_info = {
            "id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "files": {}
        }
        
        for file_path in files:
            src_path = self.project_root / file_path
            if src_path.exists():
                # Copy file
                dest_path = backup_path / file_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                
                # Record metadata
                backup_info["files"][file_path] = {
                    "size": src_path.stat().st_size,
                    "mtime": src_path.stat().st_mtime
                }
        
        # Save backup info
        with open(backup_path / "backup_info.json", "w") as f:
            json.dump(backup_info, f, indent=2)
            
        self.change_log.append(backup_info)
        return backup_id