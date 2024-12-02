    async def _plan_update(self, context: Dict[str, Any]) -> List[ModificationStep]:
        """Plan steps for updating existing code."""
        steps = []
        target = context.get('target', '')
        changes = context.get('changes', {})
        
        # Backup step
        steps.append(ModificationStep(
            type=ModificationType.ADD,
            target=f"{target}.backup",
            content=context.get('original_content', ''),
            dependencies=[],
            validation_rules=[]
        ))
        
        # Handle import changes
        if 'import_changes' in changes:
            steps.append(ModificationStep(
                type=ModificationType.UPDATE,
                target=f"{target}_imports",
                content=changes['import_changes'],
                dependencies=[f"{target}.backup"],
                validation_rules=["import_syntax", "no_conflicts"]
            ))
        
        # Main update step
        steps.append(ModificationStep(
            type=ModificationType.UPDATE,
            target=target,
            content=changes.get('main_changes', ''),
            dependencies=[
                f"{target}.backup",
                f"{target}_imports" if 'import_changes' in changes else None
            ],
            validation_rules=["syntax", "style", "tests", "backward_compatibility"]
        ))
        
        return steps