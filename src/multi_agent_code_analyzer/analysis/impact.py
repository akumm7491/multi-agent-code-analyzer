    async def _ensure_dependencies(self):
        """Build dependency information if not already built."""
        if not self.file_dependencies:
            for file_path in self.project_root.rglob("*.py"):
                relative_path = file_path.relative_to(self.project_root)
                deps = await self._analyze_file_dependencies(file_path)
                self.file_dependencies[str(relative_path)] = deps
                
                # Track API usage
                apis = await self._analyze_api_usage(file_path)
                self.api_usage[str(relative_path)] = apis
                
    async def _analyze_file_dependencies(self, file_path: Path) -> Dict[str, Set[str]]:
        """Analyze dependencies for a single file."""
        deps = {
            'imports': set(),
            'from_imports': set(),
            'calls': set(),
            'attributes': set()
        }
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        deps['imports'].add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        deps['from_imports'].add(node.module)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        deps['calls'].add(node.func.id)
                elif isinstance(node, ast.Attribute):
                    deps['attributes'].add(node.attr)
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return deps
        
    async def _analyze_api_usage(self, file_path: Path) -> Set[str]:
        """Analyze API usage in a file."""
        apis = set()
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            # Look for class and function definitions that might be APIs
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if any(d.id == 'api' for d in node.decorator_list 
                          if isinstance(d, ast.Name)):
                        apis.add(node.name)
                        
        except Exception as e:
            print(f"Error analyzing API usage in {file_path}: {e}")
            
        return apis