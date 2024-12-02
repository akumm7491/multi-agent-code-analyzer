from typing import Dict, List, Set
import ast

class DependencyAnalyzer:
    """Analyzes code dependencies and import relationships."""
    
    def __init__(self):
        self.dependencies = {}
        self.import_graph = {}
        
    async def analyze_file(self, file_path: str, content: str):
        """Analyze dependencies in a file."""
        try:
            tree = ast.parse(content)
            imports = await self._extract_imports(tree)
            self.dependencies[file_path] = imports
            
            # Update import graph
            self.import_graph[file_path] = list(imports)
            
        except SyntaxError:
            self.dependencies[file_path] = set()
            
    async def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract import statements from AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports