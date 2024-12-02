from typing import Dict, Any, List
from .modification import ModificationType, ModificationStep

class StepPlanner:
    """Plans individual modification steps."""
    
    async def plan_steps(self, modification_type: ModificationType, context: Dict[str, Any]) -> List[ModificationStep]:
        """Plan steps for a specific modification type."""
        if modification_type == ModificationType.ADD:
            return await self._plan_addition(context)
        elif modification_type == ModificationType.UPDATE:
            return await self._plan_update(context)
        elif modification_type == ModificationType.REFACTOR:
            return await self._plan_refactor(context)
            
        return []
        
    async def _plan_addition(self, context: Dict[str, Any]) -> List[ModificationStep]:
        """Plan steps for adding new code."""
        steps = []
        target = context.get('target', '')
        content = context.get('content', {})
        
        # Check for necessary imports
        if 'imports' in content:
            steps.append(ModificationStep(
                type=ModificationType.ADD,
                target=f"{target}_imports",
                content=content['imports'],
                dependencies=[],
                validation_rules=["import_syntax", "no_conflicts"]
            ))
            
        # Main addition step
        steps.append(ModificationStep(
            type=ModificationType.ADD,
            target=target,
            content=content.get('main_content', ''),
            dependencies=[f"{target}_imports"] if 'imports' in content else [],
            validation_rules=["syntax", "style", "tests"]
        ))
        
        return steps