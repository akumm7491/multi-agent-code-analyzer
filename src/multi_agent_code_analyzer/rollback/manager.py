    async def rollback(self, backup_id: str) -> Dict[str, Any]:
        """Rollback files to a previous backup."""
        backup_path = self.backup_dir / backup_id
        if not backup_path.exists():
            return {
                "success": False,
                "error": f"Backup {backup_id} not found"
            }
            
        try:
            # Load backup info
            with open(backup_path / "backup_info.json", "r") as f:
                backup_info = json.load(f)
                
            # Create a new backup of current state
            current_backup_id = await self.create_backup(
                list(backup_info["files"].keys())
            )
            
            # Restore files
            for file_path, metadata in backup_info["files"].items():
                src_path = backup_path / file_path
                dest_path = self.project_root / file_path
                
                if src_path.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_path)
                    
            return {
                "success": True,
                "backup_id": backup_id,
                "current_backup": current_backup_id,
                "files_restored": list(backup_info["files"].keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                info_file = backup_path / "backup_info.json"
                if info_file.exists():
                    with open(info_file, "r") as f:
                        backup_info = json.load(f)
                        backups.append(backup_info)
                        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)