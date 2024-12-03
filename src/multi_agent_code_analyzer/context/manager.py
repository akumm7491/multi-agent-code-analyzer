from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import networkx as nx

@dataclass
class CodeContext:
    file_path: str
    content: str
    dependencies: List[str]
    imports: List[str]
    references: List[str]
    metadata: Dict[str, Any]

class ContextManager:
    """Manages code context and relationships across the codebase."""
    
    def __init__(self):
        self.context_graph = nx.DiGraph()
        self.file_contexts: Dict[str, CodeContext] = {}
        
    async def add_file_context(self, file_path: str, content: str, metadata: Dict[str, Any]):
        """Add or update context for a file."""
        context = CodeContext(
            file_path=file_path,
            content=content,
            dependencies=[],
            imports=[],
            references=[],
            metadata=metadata
        )
        
        self.file_contexts[file_path] = context
        self.context_graph.add_node(file_path)
        
    async def add_relationship(self, from_file: str, to_file: str, relationship_type: str):
        """Add a relationship between files."""
        if from_file in self.file_contexts and to_file in self.file_contexts:
            self.context_graph.add_edge(from_file, to_file, type=relationship_type)
            
    async def get_related_files(self, file_path: str, max_depth: int = 2) -> List[str]:
        """Get related files up to a certain depth."""
        if file_path not in self.context_graph:
            return []
            
        related = set()
        current_depth = 0
        current_files = {file_path}
        
        while current_depth < max_depth and current_files:
            next_files = set()
            for file in current_files:
                # Get immediate neighbors
                neighbors = set(self.context_graph.predecessors(file)) | \
                           set(self.context_graph.successors(file))
                next_files.update(neighbors - related - {file})
            
            related.update(current_files)
            current_files = next_files
            current_depth += 1
            
        return list(related)
        
    async def get_context(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive context for a file."""
        if file_path not in self.file_contexts:
            return None
            
        context = self.file_contexts[file_path]
        related_files = await self.get_related_files(file_path)
        
        return {
            "file": context,
            "related_files": [
                self.file_contexts[f] for f in related_files
            ],
            "relationships": [
                {
                    "from": u,
                    "to": v,
                    "type": d["type"]
                }
                for u, v, d in self.context_graph.edges(data=True)
                if u == file_path or v == file_path
            ]
        }