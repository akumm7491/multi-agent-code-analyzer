    async def _find_affected_files(self, changed_files: List[str]) -> Set[str]:
        """Find all files affected by changes."""
        affected = set(changed_files)
        to_check = set(changed_files)
        checked = set()
        
        while to_check:
            current = to_check.pop()
            checked.add(current)
            
            # Find files that depend on the current file
            for file, deps in self.file_dependencies.items():
                if current in deps['imports'] or current in deps['from_imports']:
                    affected.add(file)
                    if file not in checked:
                        to_check.add(file)
                        
        return affected
        
    async def _analyze_api_impact(self, changed_files: List[str]) -> Set[str]:
        """Analyze impact on APIs."""
        affected_apis = set()
        
        for file in changed_files:
            # Check if file contains APIs
            if file in self.api_usage:
                affected_apis.update(self.api_usage[file])
                
            # Check if file uses APIs
            if file in self.file_dependencies:
                deps = self.file_dependencies[file]
                for dep_file, apis in self.api_usage.items():
                    if dep_file in deps['imports'] or dep_file in deps['from_imports']:
                        affected_apis.update(apis)
                        
        return affected_apis
        
    async def _calculate_impact_level(self, 
                                    changed_files: List[str],
                                    affected_files: Set[str],
                                    affected_apis: Set[str]) -> ImpactLevel:
        """Calculate the overall impact level of changes."""
        # Start with lowest impact
        impact = ImpactLevel.LOW
        
        # Check number of affected files
        if len(affected_files) > 10:
            impact = max(impact, ImpactLevel.MEDIUM)
        if len(affected_files) > 25:
            impact = max(impact, ImpactLevel.HIGH)
            
        # Check API impact
        if affected_apis:
            impact = max(impact, ImpactLevel.MEDIUM)
        if len(affected_apis) > 5:
            impact = max(impact, ImpactLevel.HIGH)
            
        # Check critical path impact
        if any(file in self.critical_paths for file in affected_files):
            impact = max(impact, ImpactLevel.CRITICAL)
            
        return impact
        
    def mark_critical_path(self, file_path: str):
        """Mark a file as being on the critical path."""
        self.critical_paths.add(file_path)
        
    def is_critical_path(self, file_path: str) -> bool:
        """Check if a file is on the critical path."""
        return file_path in self.critical_paths