from typing import Dict, Any, List
from enum import Enum

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationResult:
    def __init__(self, success: bool, level: ValidationLevel, message: str):
        self.success = success
        self.level = level
        self.message = message

class ValidationChecker:
    """Validates planned code modifications."""
    
    async def validate_step(self, step: Any, context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate a single modification step."""
        results = []
        
        for rule in step.validation_rules:
            if rule == "syntax":
                result = await self._check_syntax(step.content)
                results.append(result)
            elif rule == "style":
                result = await self._check_style(step.content)
                results.append(result)
            elif rule == "tests":
                result = await self._check_tests(step.content, context)
                results.append(result)
                
        return results
        
    async def _check_syntax(self, content: str) -> ValidationResult:
        """Check code syntax."""
        try:
            compile(content, '<string>', 'exec')
            return ValidationResult(True, ValidationLevel.INFO, "Syntax check passed")
        except SyntaxError as e:
            return ValidationResult(False, ValidationLevel.ERROR, f"Syntax error: {str(e)}")
            
    async def _check_style(self, content: str) -> ValidationResult:
        """Check code style."""
        # Simple style checks for now
        issues = []
        
        # Check line length
        for line in content.split('\n'):
            if len(line) > 100:
                issues.append("Line too long")
                
        if issues:
            return ValidationResult(False, ValidationLevel.WARNING, f"Style issues: {', '.join(issues)}")
        return ValidationResult(True, ValidationLevel.INFO, "Style check passed")