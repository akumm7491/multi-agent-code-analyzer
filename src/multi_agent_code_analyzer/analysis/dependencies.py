from typing import Dict, Any, List, Set
from pathlib import Path
import ast

class DependencyAnalyzer:
    """Analyzes code dependencies and import relationships."""
    
    def __init__(self):
        self.import_graph = {}
        self.function_calls = {}
        self.class_usage = {}
        
    async def analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Analyze dependencies in a single file."""
        try:
            tree = ast.parse(content)
            imports = await self._extract_imports(tree)
            calls = await self._extract_function_calls(tree)
            classes = await self._extract_class_usage(tree)
            
            return {
                "imports": imports,
                "function_calls": calls,
                "class_usage": classes
            }
        except SyntaxError:
            return {"error": "Invalid syntax"}
            
    async def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract all imports from the file."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "module": name.name,
                        "alias": name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports.append({
                        "module": f"{node.module}.{name.name}",
                        "alias": name.asname
                    })
        return imports