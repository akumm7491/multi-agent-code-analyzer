from typing import Dict, Any, List, Optional
import ast
import logging
import re
from pathlib import Path
import asyncio
from uuid import uuid4

from .base import BaseAgent
from ..mcp.models import Context, ContextType
from ..tools.github import GithubService
from .pattern_learning import PatternLearner

logger = logging.getLogger(__name__)


class CodeAnalyzerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.github_service = GithubService()
        self.pattern_learner = PatternLearner(self.memory_manager.redis)
        self.analysis_patterns = {
            "security": self._analyze_security,
            "performance": self._analyze_performance,
            "maintainability": self._analyze_maintainability,
            "architecture": self._analyze_architecture
        }
        self.code_metrics = {
            "lines_of_code": 0,
            "comment_lines": 0,
            "complexity_sum": 0,
            "function_count": 0,
            "class_count": 0,
            "test_coverage": 0.0
        }

    async def _execute_task(
        self,
        task: Dict[str, Any],
        context: Context,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute code analysis task"""
        analysis_type = task.get("analysis_type", "full")
        repo_path = task.get("repo_path")

        if not repo_path:
            raise ValueError("Repository path is required")

        results = {
            "type": "code_analysis",
            "status": "completed",
            "findings": [],
            "metrics": {},
            "recommendations": []
        }

        # Reset metrics for new analysis
        self._reset_metrics()

        # Analyze code files
        for file_path in Path(repo_path).rglob("*.py"):
            try:
                with open(file_path, 'r') as f:
                    code = f.read()

                # Create context for the file
                file_context = await self._create_file_context(file_path, code)

                # Update basic metrics
                self._update_basic_metrics(code)

                # Perform requested analysis
                if analysis_type == "full" or analysis_type == "security":
                    security_issues = await self._analyze_security(code, file_context)
                    results["findings"].extend(security_issues)

                if analysis_type == "full" or analysis_type == "performance":
                    perf_issues = await self._analyze_performance(code, file_context)
                    results["findings"].extend(perf_issues)

                if analysis_type == "full" or analysis_type == "maintainability":
                    maint_issues = await self._analyze_maintainability(code, file_context)
                    results["findings"].extend(maint_issues)

                if analysis_type == "full" or analysis_type == "architecture":
                    arch_issues = await self._analyze_architecture(code, file_context)
                    results["findings"].extend(arch_issues)

                # Learn from the analysis
                for finding in results["findings"]:
                    await self._learn_from_finding(finding, file_context)

            except Exception as e:
                logger.error(
                    f"Error analyzing {file_path}: {e}", exc_info=True)
                results["findings"].append({
                    "type": "error",
                    "file": str(file_path),
                    "message": f"Analysis failed: {str(e)}"
                })

        # Add metrics to results
        results["metrics"].update(self.code_metrics)

        # Generate recommendations
        results["recommendations"] = await self._generate_recommendations(
            results["findings"],
            memories
        )

        return results

    async def _create_file_context(self, file_path: Path, code: str) -> Context:
        """Create context for a code file"""
        return Context(
            id=uuid4(),
            type=ContextType.CODE_SNIPPET,
            content=code,
            metadata={
                "file_path": str(file_path),
                "language": "python",
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _analyze_security(self, code: str, context: Context) -> List[Dict[str, Any]]:
        """Analyze code for security issues"""
        issues = []
        try:
            tree = ast.parse(code)

            # Check for potential security issues
            for node in ast.walk(tree):
                # Check for hardcoded secrets
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if any(secret in target.id.lower() for secret in ["password", "secret", "key", "token"]):
                                issues.append({
                                    "type": "security",
                                    "severity": "high",
                                    "file": context.metadata["file_path"],
                                    "line": node.lineno,
                                    "message": f"Potential hardcoded secret in variable '{target.id}'"
                                })

                # Check for unsafe eval usage
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "eval":
                        issues.append({
                            "type": "security",
                            "severity": "high",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": "Unsafe use of eval()"
                        })

                # Check for SQL injection vulnerabilities
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['execute', 'executemany']:
                            issues.append({
                                "type": "security",
                                "severity": "high",
                                "file": context.metadata["file_path"],
                                "line": node.lineno,
                                "message": "Potential SQL injection vulnerability"
                            })

                # Get similar security patterns
                similar_patterns = await self.pattern_learner.get_similar_patterns(
                    "security",
                    {"file_path": context.metadata["file_path"]},
                    threshold=0.7
                )

                # Apply learned patterns
                for pattern in similar_patterns:
                    if pattern.confidence > 0.8:  # High confidence threshold
                        # Apply pattern-specific checks
                        if await self._check_security_pattern(node, pattern):
                            issues.append(
                                self._create_issue_from_pattern(pattern, context))

        except Exception as e:
            logger.error(f"Security analysis failed: {e}", exc_info=True)

        return issues

    async def _analyze_performance(self, code: str, context: Context) -> List[Dict[str, Any]]:
        """Analyze code for performance issues"""
        issues = []
        try:
            tree = ast.parse(code)

            # Get similar performance patterns
            similar_patterns = await self.pattern_learner.get_similar_patterns(
                "performance",
                {"file_path": context.metadata["file_path"]},
                threshold=0.7
            )

            for node in ast.walk(tree):
                # Check for nested loops
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.For, ast.While)) and child is not node:
                            issues.append({
                                "type": "performance",
                                "severity": "medium",
                                "file": context.metadata["file_path"],
                                "line": node.lineno,
                                "message": "Nested loop detected - potential performance issue"
                            })
                            break

                # Check for large list comprehensions
                if isinstance(node, ast.ListComp):
                    if len(list(ast.walk(node))) > 10:
                        issues.append({
                            "type": "performance",
                            "severity": "low",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": "Complex list comprehension - consider breaking down"
                        })

                # Check for inefficient string concatenation
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str):
                        issues.append({
                            "type": "performance",
                            "severity": "low",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": "Inefficient string concatenation - consider using join() or f-strings"
                        })

                # Apply learned patterns
                for pattern in similar_patterns:
                    if pattern.confidence > 0.8:
                        if await self._check_performance_pattern(node, pattern):
                            issues.append(
                                self._create_issue_from_pattern(pattern, context))

            # Check file size
            self._check_file_size(context, issues)

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}", exc_info=True)

        return issues

    async def _analyze_maintainability(self, code: str, context: Context) -> List[Dict[str, Any]]:
        """Analyze code for maintainability issues"""
        issues = []
        try:
            tree = ast.parse(code)

            # Get similar maintainability patterns
            similar_patterns = await self.pattern_learner.get_similar_patterns(
                "maintainability",
                {"file_path": context.metadata["file_path"]},
                threshold=0.7
            )

            for node in ast.walk(tree):
                # Check function complexity
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    if complexity > 10:
                        issues.append({
                            "type": "maintainability",
                            "severity": "medium",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": f"Function '{node.name}' has high cyclomatic complexity ({complexity})"
                        })

                    # Check function length
                    if len(node.body) > 50:
                        issues.append({
                            "type": "maintainability",
                            "severity": "medium",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": f"Function '{node.name}' is too long ({len(node.body)} lines)"
                        })

                    # Check for too many parameters
                    if len(node.args.args) > 5:
                        issues.append({
                            "type": "maintainability",
                            "severity": "medium",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})"
                        })

                    # Check for lack of docstring
                    if not ast.get_docstring(node):
                        issues.append({
                            "type": "maintainability",
                            "severity": "low",
                            "file": context.metadata["file_path"],
                            "line": node.lineno,
                            "message": f"Function '{node.name}' lacks a docstring"
                        })

                # Apply learned patterns
                for pattern in similar_patterns:
                    if pattern.confidence > 0.8:
                        if await self._check_maintainability_pattern(node, pattern):
                            issues.append(
                                self._create_issue_from_pattern(pattern, context))

            # Check code duplication
            await self._check_code_duplication(context, issues)

        except Exception as e:
            logger.error(
                f"Maintainability analysis failed: {e}", exc_info=True)

        return issues

    async def _analyze_architecture(self, code: str, context: Context) -> List[Dict[str, Any]]:
        """Analyze code for architectural issues"""
        issues = []
        try:
            tree = ast.parse(code)

            # Get similar architecture patterns
            similar_patterns = await self.pattern_learner.get_similar_patterns(
                "architecture",
                {"file_path": context.metadata["file_path"]},
                threshold=0.7
            )

            # Check for circular imports
            imports = self._extract_imports(tree)
            if self._detect_circular_imports(context.metadata["file_path"], imports):
                issues.append({
                    "type": "architecture",
                    "severity": "high",
                    "file": context.metadata["file_path"],
                    "line": 1,
                    "message": "Potential circular import detected"
                })

            # Check for proper layering
            if not self._check_layering(context.metadata["file_path"]):
                issues.append({
                    "type": "architecture",
                    "severity": "medium",
                    "file": context.metadata["file_path"],
                    "line": 1,
                    "message": "File may violate layering principles"
                })

            # Apply learned patterns
            for pattern in similar_patterns:
                if pattern.confidence > 0.8:
                    if await self._check_architecture_pattern(context, pattern):
                        issues.append(
                            self._create_issue_from_pattern(pattern, context))

        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}", exc_info=True)

        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    async def _generate_recommendations(
        self,
        findings: List[Dict[str, Any]],
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on findings and past experiences"""
        recommendations = []

        # Get relevant patterns for recommendations
        relevant_patterns = []
        for finding_type in set(f["type"] for f in findings):
            patterns = await self.pattern_learner.get_similar_patterns(finding_type, {}, threshold=0.8)
            relevant_patterns.extend(patterns)

        # Group findings by type
        findings_by_type = {}
        for finding in findings:
            findings_by_type.setdefault(finding["type"], []).append(finding)

        # Generate recommendations for each type
        for finding_type, type_findings in findings_by_type.items():
            if finding_type == "security":
                recommendations.extend(await self._generate_security_recommendations(type_findings))
            elif finding_type == "performance":
                recommendations.extend(await self._generate_performance_recommendations(type_findings))
            elif finding_type == "maintainability":
                recommendations.extend(await self._generate_maintainability_recommendations(type_findings))
            elif finding_type == "architecture":
                recommendations.extend(await self._generate_architecture_recommendations(type_findings))

        # Add pattern-based recommendations
        for pattern in relevant_patterns:
            if pattern.success_rate > 0.8:  # Only use highly successful patterns
                recommendations.append({
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "message": f"Based on past experience: {pattern.pattern_data.get('recommendation', '')}"
                })

        return recommendations

    async def _calculate_metrics(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics based on findings"""
        return {
            "total_issues": len(findings),
            "issues_by_type": {
                issue_type: len(
                    [f for f in findings if f["type"] == issue_type])
                for issue_type in set(f["type"] for f in findings)
            },
            "issues_by_severity": {
                severity: len(
                    [f for f in findings if f["severity"] == severity])
                for severity in ["high", "medium", "low"]
            },
            "code_quality_score": self._calculate_code_quality_score(findings),
            "maintainability_index": self._calculate_maintainability_index(),
            "technical_debt_ratio": self._calculate_technical_debt_ratio(findings),
            "code_metrics": {
                "avg_complexity": self.code_metrics["complexity_sum"] / max(self.code_metrics["function_count"], 1),
                "comment_ratio": self.code_metrics["comment_lines"] / max(self.code_metrics["lines_of_code"], 1),
                "test_coverage": self.code_metrics["test_coverage"]
            }
        }

    def _reset_metrics(self):
        """Reset code metrics for new analysis"""
        self.code_metrics = {
            "lines_of_code": 0,
            "comment_lines": 0,
            "complexity_sum": 0,
            "function_count": 0,
            "class_count": 0,
            "test_coverage": 0.0
        }

    def _update_basic_metrics(self, code: str):
        """Update basic code metrics"""
        lines = code.split('\n')
        self.code_metrics["lines_of_code"] += len(lines)

        # Count comment lines
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        self.code_metrics["comment_lines"] += comment_lines

        try:
            tree = ast.parse(code)

            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.code_metrics["function_count"] += 1
                    self.code_metrics["complexity_sum"] += self._calculate_complexity(
                        node)
                elif isinstance(node, ast.ClassDef):
                    self.code_metrics["class_count"] += 1

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _learn_from_finding(self, finding: Dict[str, Any], context: Context):
        """Learn from analysis findings"""
        try:
            pattern_data = {
                "type": finding["type"],
                "severity": finding["severity"],
                "message": finding["message"],
                "context": context.metadata
            }

            await self.pattern_learner.learn_pattern(finding["type"], pattern_data, True)
        except Exception as e:
            logger.error(f"Error learning from finding: {e}")

    def _check_file_size(self, context: Context, issues: List[Dict[str, Any]]):
        """Check if file is too large"""
        file_size = len(context.content.split('\n'))
        if file_size > 500:  # Configurable threshold
            issues.append({
                "type": "maintainability",
                "severity": "medium",
                "file": context.metadata["file_path"],
                "line": 1,
                "message": f"File is too large ({file_size} lines)"
            })

    async def _check_code_duplication(self, context: Context, issues: List[Dict[str, Any]]):
        """Check for code duplication"""
        # Simple implementation - can be enhanced with more sophisticated algorithms
        lines = context.content.split('\n')
        chunks = {}

        for i in range(len(lines) - 5):  # Look for duplicates of 6+ lines
            chunk = '\n'.join(lines[i:i+6])
            if len(chunk.strip()) > 0:
                if chunk in chunks:
                    chunks[chunk].append(i+1)
                else:
                    chunks[chunk] = [i+1]

        for chunk, locations in chunks.items():
            if len(locations) > 1:
                issues.append({
                    "type": "maintainability",
                    "severity": "medium",
                    "file": context.metadata["file_path"],
                    "line": locations[0],
                    "message": f"Duplicate code found at lines: {', '.join(map(str, locations))}"
                })

    async def _check_security_pattern(self, node: ast.AST, pattern: Pattern) -> bool:
        """Check if node matches a security pattern"""
        try:
            if "variable_pattern" in pattern.pattern_data:
                if isinstance(node, ast.Name):
                    return bool(re.match(
                        pattern.pattern_data["variable_pattern"],
                        node.id
                    ))
            if "function_pattern" in pattern.pattern_data:
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        return bool(re.match(
                            pattern.pattern_data["function_pattern"],
                            node.func.id
                        ))
        except Exception as e:
            logger.error(f"Error checking security pattern: {e}")
        return False

    async def _check_performance_pattern(self, node: ast.AST, pattern: Pattern) -> bool:
        """Check if node matches a performance pattern"""
        try:
            if "node_type" in pattern.pattern_data:
                if pattern.pattern_data["node_type"] == node.__class__.__name__:
                    return True
            if "complexity_threshold" in pattern.pattern_data:
                if isinstance(node, ast.FunctionDef):
                    return self._calculate_complexity(node) > pattern.pattern_data["complexity_threshold"]
        except Exception as e:
            logger.error(f"Error checking performance pattern: {e}")
        return False

    async def _check_maintainability_pattern(self, node: ast.AST, pattern: Pattern) -> bool:
        """Check if node matches a maintainability pattern"""
        try:
            if "max_length" in pattern.pattern_data:
                if isinstance(node, ast.FunctionDef):
                    return len(node.body) > pattern.pattern_data["max_length"]
            if "naming_pattern" in pattern.pattern_data:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    return not bool(re.match(
                        pattern.pattern_data["naming_pattern"],
                        node.name
                    ))
        except Exception as e:
            logger.error(f"Error checking maintainability pattern: {e}")
        return False

    async def _check_architecture_pattern(self, context: Context, pattern: Pattern) -> bool:
        """Check if context matches an architecture pattern"""
        try:
            if "file_pattern" in pattern.pattern_data:
                return bool(re.match(
                    pattern.pattern_data["file_pattern"],
                    context.metadata["file_path"]
                ))
            if "import_pattern" in pattern.pattern_data:
                tree = ast.parse(context.content)
                imports = self._extract_imports(tree)
                return any(re.match(pattern.pattern_data["import_pattern"], imp) for imp in imports)
        except Exception as e:
            logger.error(f"Error checking architecture pattern: {e}")
        return False
