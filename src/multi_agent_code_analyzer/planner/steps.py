    async def _plan_refactor(self, context: Dict[str, Any]) -> List[ModificationStep]:
        """Plan steps for refactoring code."""
        steps = []
        target = context.get('target', '')
        refactor_type = context.get('refactor_type', '')
        
        # Create backup
        steps.append(ModificationStep(
            type=ModificationType.ADD,
            target=f"{target}.backup",
            content=context.get('original_content', ''),
            dependencies=[],
            validation_rules=[]
        ))
        
        if refactor_type == 'extract_method':
            steps.extend(await self._plan_extract_method(context))
        elif refactor_type == 'rename':
            steps.extend(await self._plan_rename(context))
        elif refactor_type == 'move':
            steps.extend(await self._plan_move(context))
            
        return steps
        
    async def _plan_extract_method(self, context: Dict[str, Any]) -> List[ModificationStep]:
        """Plan steps for extracting a method."""
        target = context.get('target', '')
        new_method = context.get('new_method', {})
        
        steps = [
            ModificationStep(
                type=ModificationType.ADD,
                target=f"{target}_new_method",
                content=new_method.get('implementation', ''),
                dependencies=[f"{target}.backup"],
                validation_rules=["syntax", "style", "tests"]
            ),
            ModificationStep(
                type=ModificationType.UPDATE,
                target=target,
                content=new_method.get('caller_changes', ''),
                dependencies=[f"{target}_new_method"],
                validation_rules=["syntax", "style", "tests", "backward_compatibility"]
            )
        ]
        
        return steps