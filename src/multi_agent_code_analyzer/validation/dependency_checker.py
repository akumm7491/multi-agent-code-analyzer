from typing import Dict, Any, List, Set
import ast
from pathlib import Path

class DependencyChecker:
    """Checks dependencies and their impacts in code changes."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = {}
        self.import_cache = {}
        
    async def check_dependencies(self, modified_file: str) -> Dict[str, Any]:
        """Check dependencies for a modified file."""
        results = {
            "safe": True,
            "affected_files": [],
            "breaking_changes": [],
            "suggested_updates": []
        }
        
        # Build dependency graph if not cached
        if not self.dependency_graph:
            await self._build_dependency_graph()
        
        # Find affected files
        affected = await self._find_affected_files(modified_file)
        results["affected_files"] = list(affected)
        
        # Check for breaking changes
        breaking = await self._check_breaking_changes(modified_file, affected)
        if breaking:
            results["safe"] = False
            results["breaking_changes"] = breaking
            
        # Generate update suggestions
        results["suggested_updates"] = await self._generate_update_suggestions(breaking)
        
        return results
        
    async def _build_dependency_graph(self):
        """Build a graph of project dependencies."""
        for file_path in self.project_root.rglob("*.py"):
            imports = await self._extract_imports(file_path)
            self.dependency_graph[str(file_path)] = imports
            
    async def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract imports from a Python file."""
        if str(file_path) in self.import_cache:
            return self.import_cache[str(file_path)]
            
        imports = set()
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        
            self.import_cache[str(file_path)] = imports
            return imports
            
        except Exception:
            return set()